"""Selected native roots preserve pending decisions and produce legal rollouts."""

import json

import numpy
import pytest

from log_dataset import _native_collection
from model import actions
from training import self_play


def greedy(request):
    rows = numpy.frombuffer(request["features"], self_play.FEATURE_DTYPE)
    legal = numpy.frombuffer(request["legal"], numpy.bool_).reshape(-1, 150)
    scores = numpy.zeros_like(legal, numpy.float32)
    scores[:, :37] = (
        -100 * rows["discard_action_all_shanten"].astype(numpy.float32)
        + rows["discard_action_all_ukeire_count"]
    )
    scores[numpy.asarray(request["head"]) != 0] = numpy.arange(150)
    return numpy.where(legal, scores, -1e9).argmax(1).tolist()


def finish(collector):
    while (request := collector.next()) is not None:
        collector.submit(greedy(request))
    return collector.finish()[0]


def test_selected_roots_keep_prefix_and_repeat_seeded_continuations():
    collector = _native_collection.NativeCollector(1, 941)
    roots = []
    reached = False
    while (request := collector.next()) is not None:
        rows = numpy.frombuffer(request["features"], self_play.FEATURE_DTYPE)
        if len(roots) < 3 and (
            not roots or rows["live_wall_count"].min() <= 16
        ):
            roots.append((collector.snapshot(0), request, collector.logs()))
        if not reached and (rows["riichi_state"] == 2).any():
            roots.append((collector.snapshot(0), request, collector.logs()))
            reached = True
        collector.submit(greedy(request))
    assert reached
    assert len(roots) == 4
    for root, original, prefix in roots:
        clone = root.fork(1, 42, False)
        copied = clone.next()
        for name in ("features", "head", "legal", "seat", "event", "hand"):
            assert copied[name] == original[name]
        assert clone.logs() == prefix
        assert root.decision == 0
        clone.submit(greedy(copied))
        finish(clone)
        branch = root.fork(2, 314)
        first = finish(branch)
        assert first == finish(root.fork(2, 314))
        hands = riichi_turn_sum = 0
        start = max(
            index
            for index, event in enumerate(json.loads(prefix[0]))
            if event["type"] == "start_kyoku"
        )
        for log in first:
            events = json.loads(log)
            assert events[: len(json.loads(prefix[0]))] == json.loads(prefix[0])
            assert events[-1]["type"] == "end_game"
            for event in events[start:]:
                if event["type"] == "start_kyoku":
                    cuts = [0] * 4
                elif event["type"] == "dahai":
                    cuts[event["actor"]] += 1
                elif event["type"] == "reach_accepted":
                    riichi_turn_sum += cuts[event["actor"]]
                elif event["type"] == "end_kyoku":
                    hands += 1
        statistics = branch.finish()[1]
        assert statistics["hands"] == hands
        assert statistics["riichi_turn_sum"] == riichi_turn_sum
        assert len(set(first)) == 2
    with pytest.raises(ValueError, match="pending"):
        collector.snapshot(0)
    with pytest.raises(ValueError, match="positive"):
        roots[0][0].fork(0, 42)


def test_training_rows_from_roots_exclude_prefix_and_keep_behavior():
    roots = []

    def capture(collector, batch):
        if not roots and batch["features"]["live_wall_count"].min() <= 16:
            roots.append(collector.snapshot(0))

    def predict(features):
        return {
            name: numpy.zeros(
                (len(features), actions.ACTOR_OUTPUTS[name][0]), numpy.float32
            )
            for name in actions.TRAIN_POLICY_HEADS
        }

    self_play.collect(predict, 1, 941, root_callback=capture)
    assert roots
    branch = roots[0].fork(2, 123)
    batch, statistics = self_play.collect(
        predict,
        2,
        123,
        collector=branch,
    )
    assert statistics["games"] == 2
    assert numpy.all(
        batch["legal"][numpy.arange(len(batch["chosen"])), batch["chosen"]]
    )
    numpy.testing.assert_allclose(
        numpy.exp(batch["behavior"]).sum(1), 1, atol=1e-5
    )
    assert batch["features"]["live_wall_count"][0] <= 16
    assert numpy.isfinite(batch["payment_points"]).all()


def test_root_scoring_keeps_legal_values_across_padded_chunks(monkeypatch):
    import types

    import tensorflow as tf

    from training.corrections import collection

    def actor(features, training):
        count = tf.shape(features["decision_phase"])[0]
        return {
            "selected": tf.stack(
                [tf.repeat(tf.range(count), 2), tf.tile([0, 36], [count])],
                axis=1,
            )
        }

    def values(*, network, features, selected, state, chunk_size):
        assert chunk_size == 32
        rows = tf.cast(features["live_wall_count"], tf.float32)
        return tf.scatter_nd(
            selected,
            tf.reshape(tf.stack([rows, -rows], axis=1), [-1]),
            [tf.shape(rows)[0], 295],
        )

    monkeypatch.setattr(collection.trainer, "action_values", values)
    monkeypatch.setattr(
        collection._native_collection,
        "INPUTS",
        {
            **{
                name: {"shape": [], "dtype": "int8"}
                for name in (
                    "decision_phase",
                    "live_wall_count",
                    "trigger_tile_id",
                )
            },
            "discard_action_legal": {"shape": [37], "dtype": "int8"},
        },
    )
    evaluate = collection.root_scorer(
        types.SimpleNamespace(
            actor=actor,
            candidate_context=lambda features, outputs: (
                outputs,
                tf.zeros([tf.shape(outputs["selected"])[0], 1]),
            ),
        )
    )
    for count in (2, 129):
        features = {
            "decision_phase": numpy.zeros(count, numpy.int8),
            "live_wall_count": numpy.arange(count, dtype=numpy.int8) % 2,
            "trigger_tile_id": numpy.zeros(count, numpy.int8),
            "discard_action_legal": numpy.broadcast_to(
                numpy.isin(numpy.arange(37), [0, 36]).astype(numpy.int8),
                (count, 37),
            ),
        }
        values = evaluate(features)
        assert values.shape == (count, 37)
        numpy.testing.assert_array_equal(
            values[:, 0], features["live_wall_count"]
        )
        numpy.testing.assert_array_equal(
            values[:, 36], -features["live_wall_count"]
        )
        assert numpy.isneginf(values[:, 1:36]).all()


def test_action_pairs_repeat_each_choice_on_matched_walls():
    collector = _native_collection.NativeCollector(1, 941)
    request = collector.next()
    root = collector.snapshot(0)
    codes = numpy.flatnonzero(
        numpy.frombuffer(request["legal"], numpy.bool_).reshape(-1, 150)[0]
    )[[0, -1]]
    assert codes[0] != codes[1]

    def predict(features):
        return {
            name: numpy.zeros(
                (len(features), actions.ACTOR_OUTPUTS[name][0]), numpy.float32
            )
            for name in actions.TRAIN_POLICY_HEADS
        }

    streams = [
        self_play.collection_steps(
            2,
            123,
            collector=root.fork(2, 123),
            forced_root=(root.decision, int(action)),
        )
        for action in codes
    ]
    batches = self_play.collect_streams(predict, streams)
    for action, (batch, statistics) in zip(codes, batches, strict=True):
        assert statistics["games"] == 2
        assert batch["paired_root"].sum() == 2
        numpy.testing.assert_array_equal(
            batch["chosen"][batch["paired_root"]], numpy.full(2, action)
        )
        numpy.testing.assert_array_equal(
            batch["behavior"][batch["paired_root"], action], numpy.zeros(2)
        )
        assert batch["legal"][
            numpy.arange(len(batch["chosen"])), batch["chosen"]
        ].all()
    for name in batches[0][0]["features"]:
        numpy.testing.assert_array_equal(
            batches[0][0]["features"][name][batches[0][0]["paired_root"]],
            batches[1][0]["features"][name][batches[1][0]["paired_root"]],
        )


def test_improving_calls_use_unclipped_shanten():
    features = numpy.zeros((), self_play.FEATURE_DTYPE)
    features["self_efficiency"][0] = 4 / 8
    features["chi_discard_legal"][0, :2] = 1
    features["chi_discard_shanten"][0, :2] = [4, 5]
    improving, _ = self_play.improving_call_choices(features)
    assert improving[1]
    assert not improving[2]
