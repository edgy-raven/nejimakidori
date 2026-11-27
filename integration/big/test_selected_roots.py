"""Native roots preserve pending decisions and produce legal rollouts."""

import json

import numpy
import pytest
import tensorflow as tf

from integration import reference_policy
from log_dataset import _native_collection, call_states, training_targets
from training import self_play


def test_shared_root_affinity_preserves_all_continuation_labels():
    collector = _native_collection.NativeCollector(1, 941)
    request = collector.next()
    root = collector.snapshot(0)
    features = call_states.from_request(request)
    affinity = training_targets.candidate_yaku_affinity(
        {name: value[0] for name, value in call_states.base(features).items()}
    )
    action = int(
        numpy.flatnonzero(
            numpy.frombuffer(request["legal"], numpy.bool_).reshape(-1, 150)[0]
        )[0]
    )
    batches = self_play.collect_streams(
        reference_policy.uniform_logits,
        (
            self_play.collection_steps(
                games=8,
                seed=123,
                collector=root.fork(8, 123, until_hand_end=True),
                forced_root=(root.decision, action),
                with_labels=True,
                root_only=True,
                root_affinity=value,
            )
            for value in (None, affinity)
        ),
    )
    tf.nest.map_structure(
        numpy.testing.assert_array_equal, batches[0], batches[1]
    )


def test_selected_roots_keep_prefix_and_repeat_seeded_continuations():
    collector = _native_collection.NativeCollector(1, 941)
    roots = []
    reached = False
    while (request := collector.next()) is not None:
        rows = numpy.frombuffer(request["features"], call_states.ROW_DTYPE)
        if len(roots) < 3 and (
            not roots or rows["live_wall_count"].min() <= 16
        ):
            roots.append((collector.snapshot(0), request, collector.logs()))
        if not reached and (rows["riichi_state"] == 2).any():
            roots.append((collector.snapshot(0), request, collector.logs()))
            reached = True
        collector.submit(reference_policy.greedy(request))
    assert reached
    assert len(roots) == 4
    for root, original, prefix in roots:
        clone = root.fork(1, 42, False)
        copied = clone.next()
        for name in ("features", "head", "legal", "seat", "event", "hand"):
            assert copied[name] == original[name]
        assert clone.logs() == prefix
        assert root.decision == 0
        clone.submit(reference_policy.greedy(copied))
        reference_policy.finish(clone)
        branch = root.fork(2, 314)
        first = reference_policy.finish(branch)
        assert first == reference_policy.finish(
            _native_collection.NativeRoot.loads(root.dumps()).fork(2, 314)
        )
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
    collector = _native_collection.NativeCollector(1, 941)
    while (request := collector.next()) is not None:
        rows = numpy.frombuffer(request["features"], call_states.ROW_DTYPE)
        if rows["live_wall_count"].min() <= 16:
            root = collector.snapshot(0)
            break
        collector.submit(reference_policy.greedy(request))

    branch = root.fork(2, 123)
    batch, statistics = self_play.collect(
        reference_policy.uniform_logits,
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
            candidate_context=lambda features, outputs: tf.zeros(
                [tf.shape(outputs["selected"])[0], 1]
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
        features = call_states.combine(
            features,
            {name: value[:0] for name, value in features.items()},
            numpy.empty(0, dtype="int32"),
            numpy.empty(0, dtype="int32"),
        )
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

    streams = [
        self_play.collection_steps(
            2,
            123,
            collector=root.fork(2, 123),
            forced_root=(root.decision, int(action)),
        )
        for action in codes
    ]
    batches = self_play.collect_streams(
        reference_policy.uniform_logits, streams
    )
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
    root_batches = self_play.collect_streams(
        reference_policy.uniform_logits,
        (
            self_play.collection_steps(
                2,
                123,
                collector=root.fork(2, 123),
                forced_root=(root.decision, int(action)),
                root_only=True,
            )
            for action in codes
        ),
        max_active=1,
    )
    for (full, statistics), (retained, root_statistics) in zip(
        batches, root_batches, strict=True
    ):
        selected = numpy.flatnonzero(full["paired_root"])
        expected = {
            "features": call_states.select(full["features"], selected),
            **{
                name: value[selected]
                for name, value in full.items()
                if name != "features"
            },
        }
        tf.nest.map_structure(
            numpy.testing.assert_array_equal, retained, expected
        )
        assert root_statistics == statistics
    paired = [
        call_states.select(
            batch["features"], numpy.flatnonzero(batch["paired_root"])
        )
        for batch, _ in batches
    ]
    for name in paired[0]:
        numpy.testing.assert_array_equal(paired[0][name], paired[1][name])


def test_improving_calls_use_unclipped_shanten():
    features = numpy.zeros((), call_states.ROW_DTYPE)
    features["self_efficiency"][0] = 4 / 8
    features["call_discard_legal"][:, :2] = 1
    features["call_discard_shanten"][:, :2] = [4, 5]
    improving, _ = self_play.improving_call_choices(features)
    numpy.testing.assert_array_equal(
        numpy.flatnonzero(improving), [1, 38, 75, 112]
    )
