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
    assert numpy.isfinite(batch["settlement"]).all()


@pytest.mark.parametrize(
    "focus,workers",
    [
        ("keiten", 1),
        ("tenpai", 1),
        ("calls", 2),
        ("disagreement", 1),
        ("disagreement", 2),
        ("targeted", 2),
    ],
)
def test_focused_worker_collection_keeps_natural_rows(workers, focus):
    scored_rows = []

    def predict(features):
        outputs = {
            name: numpy.broadcast_to(
                numpy.arange(
                    actions.ACTOR_OUTPUTS[name][0], dtype=numpy.float32
                ),
                (
                    len(features["decision_phase"]),
                    actions.ACTOR_OUTPUTS[name][0],
                ),
            ).copy()
            for name in actions.TRAIN_POLICY_HEADS
        }
        outputs[actions.TRAIN_POLICY_HEADS[0]] = (
            -1000 * features["discard_action_all_shanten"].astype(numpy.float32)
            + features["discard_action_all_ukeire_count"]
        )
        return outputs

    def score_roots(features):
        scored_rows.append(len(features["decision_phase"]))
        if focus not in ("disagreement", "targeted"):
            pytest.fail("Q/V pairing must only score disagreement roots")
        pairs = numpy.full(
            (len(features["decision_phase"]), 2), -1, numpy.int32
        )
        legal = actions.legal_choices(features)
        for row, phase in enumerate(features["decision_phase"]):
            name = actions.TRAIN_POLICY_HEADS[[0, 2, 3, 1, 4][phase]]
            codes = numpy.flatnonzero(legal[name][row])
            if len(codes) > 1:
                pairs[row] = codes[[0, -1]]
        return {
            "priority": numpy.full(len(pairs), 4, numpy.float32),
            "actions": pairs,
        }

    with self_play.Collector(
        predict,
        workers,
        focus=focus,
        root_evaluate=score_roots,
    ) as collector:
        batch = collector.collect(2, 941)
        assert 0 < collector.metrics["selected_roots"] <= 2
        assert collector.metrics["root_continuations"] == (
            2 * collector.metrics["selected_roots"]
        )
        assert collector.metrics["games"] == (
            2 + collector.metrics["root_continuations"]
        )
        assert collector.metrics["root_evaluation_rows"] == sum(scored_rows)
        assert (collector.metrics["root_evaluation_seconds"] > 0) == bool(
            scored_rows
        )
    assert len(batch["features"]["decision_phase"]) == len(batch["chosen"])
    numpy.testing.assert_allclose(
        numpy.exp(batch["behavior"]).sum(1), 1, atol=1e-5
    )
    assert batch["features"]["live_wall_count"].max() > 16
    numpy.testing.assert_array_equal(
        batch["root_path"] > 0, batch["paired_root"]
    )
    if focus in ("keiten", "tenpai"):
        rows = numpy.flatnonzero(batch["paired_root"])
        distance = batch["features"]["discard_action_all_shanten"][rows]
        best = numpy.where(
            batch["legal"][rows], distance[:, numpy.arange(150) % 37], 100
        ).min(axis=1)
        selected_distance = distance[
            numpy.arange(len(rows)), batch["chosen"][rows] % 37
        ]
        assert numpy.count_nonzero(selected_distance == best) * 2 == len(rows)
    paths = batch["root_path"][batch["paired_root"]]
    if focus != "targeted":
        numpy.testing.assert_array_equal(
            paths,
            numpy.full(
                len(paths),
                {
                    "keiten": 1,
                    "tenpai": 2,
                    "calls": 4,
                    "disagreement": 8,
                }[focus],
            ),
        )


def test_disagreement_priorities_compare_actions_and_value_in_st3(monkeypatch):
    import types

    import tensorflow as tf

    from training import online_training

    outputs = {
        "state_value": tf.Variable([0.01, 0.15]),
        "critic_projected_st3": tf.Variable([10.0, 0.0, 5.0, 5.0]),
        actions.TRAIN_POLICY_HEADS[0]: tf.constant(
            [[0.0, numpy.log(9.0)], [0.0, 0.0]]
        ),
    }

    def score(**inputs):
        rows = tf.cast(inputs["live_wall_count"], tf.int32)
        return {
            "state_value": tf.gather(outputs["state_value"], rows),
            "critic_action_indices": tf.stack(
                [
                    tf.repeat(tf.range(tf.shape(rows)[0]), 2),
                    tf.tile(tf.constant([0, 1]), [tf.shape(rows)[0]]),
                ],
                axis=1,
            ),
            "critic_projected_st3": tf.reshape(
                tf.gather(
                    tf.reshape(outputs["critic_projected_st3"], [2, 2]), rows
                ),
                [-1],
            ),
            actions.TRAIN_POLICY_HEADS[0]: tf.gather(
                outputs[actions.TRAIN_POLICY_HEADS[0]], rows
            ),
        }

    monkeypatch.setattr(
        online_training.export,
        "ServingModel",
        lambda network: types.SimpleNamespace(
            root_evaluation=types.SimpleNamespace(
                get_concrete_function=lambda **kwargs: score,
            )
        ),
    )

    def repeated_actor(*args, **kwargs):
        pytest.fail("Root scoring must reuse V from its actor forward pass")

    monkeypatch.setattr(
        online_training._native_collection,
        "INPUTS",
        {
            name: {"shape": [], "dtype": "int8"}
            for name in ("decision_phase", "live_wall_count")
        },
    )
    network = types.SimpleNamespace(actor=repeated_actor)
    evaluate = online_training.disagreement_evaluator(network)
    for count in (2, 9):
        priority = evaluate(
            {
                "decision_phase": numpy.zeros(count, numpy.int8),
                "live_wall_count": numpy.arange(count, dtype=numpy.int8) % 2,
            }
        )
        numpy.testing.assert_array_equal(
            priority["priority"], numpy.arange(count) % 2 == 0
        )
        numpy.testing.assert_array_equal(priority["actions"][0], [0, 1])
        numpy.testing.assert_array_equal(priority["actions"][1], [-1, -1])

    # A max-Q/closest-V gap below 0.25 does not qualify, unless the
    # closest action is still at least eight ST3 from V.
    outputs["critic_projected_st3"].assign([0.1, 0.0, 0.1, 0.0])
    outputs["state_value"].assign([-0.079, -0.08])
    features = {
        "decision_phase": numpy.zeros(2, numpy.int8),
        "live_wall_count": numpy.arange(2, dtype=numpy.int8),
    }
    numpy.testing.assert_array_equal(evaluate(features)["priority"], [0, 1])
    outputs["state_value"].assign([0.0, 0.0])
    outputs["critic_projected_st3"].assign([0.249, 0.0, 0.25, 0.0])
    numpy.testing.assert_array_equal(evaluate(features)["priority"], [0, 1])
    rows = numpy.zeros(2, self_play.FEATURE_DTYPE)
    for name in (
        "normal_self_shanten",
        "chiitoitsu_self_shanten",
        "kokushi_self_shanten",
    ):
        rows[name] = 3
    numpy.testing.assert_array_equal(
        self_play.target_categories(rows, numpy.zeros(2), numpy.array([0, 1])),
        [0, 8],
    )


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
            8,
            123,
            collector=root.fork(8, 123),
            forced_root=(root.decision, int(action), 8),
        )
        for action in codes
    ]
    batches = self_play.collect_streams(predict, streams)
    for action, (batch, statistics) in zip(codes, batches, strict=True):
        assert statistics["games"] == 8
        assert batch["paired_root"].sum() == 8
        numpy.testing.assert_array_equal(
            batch["chosen"][batch["paired_root"]], numpy.full(8, action)
        )
        numpy.testing.assert_array_equal(
            batch["behavior"][batch["paired_root"], action], numpy.zeros(8)
        )
        assert batch["legal"][
            numpy.arange(len(batch["chosen"])), batch["chosen"]
        ].all()
    for name in batches[0][0]["features"]:
        numpy.testing.assert_array_equal(
            batches[0][0]["features"][name][batches[0][0]["paired_root"]],
            batches[1][0]["features"][name][batches[1][0]["paired_root"]],
        )


def test_call_and_hand_pairs_use_policy_choices_within_separate_groups():
    features = {"discard_action_all_shanten": numpy.full(37, 2)}
    features["discard_action_all_shanten"][[1, 2]] = 1
    for head, codes, scores, expected in (
        (0, [1, 2, 3], [1, 2, 9], [2, 3]),
        (1, [1, 38, 3], [2, 3, 9], [38, 3]),
        (2, [0, 1, 112, 149], [10, 2, 3, 4], [149, 0]),
        (3, [0, 2, 9], [10, 2, 3], [9, 0]),
    ):
        legal = numpy.zeros(150, numpy.bool_)
        policy = numpy.full(150, -1e9)
        legal[codes] = True
        policy[codes] = scores
        numpy.testing.assert_array_equal(
            self_play.focused_action_pair(features, head, legal, policy),
            expected,
        )
    legal = numpy.zeros(150, numpy.bool_)
    legal[[1, 2]] = True
    assert self_play.focused_action_pair(features, 0, legal, policy) is None
    legal[:] = False
    legal[0] = True
    assert self_play.focused_action_pair(features, 2, legal, policy) is None
