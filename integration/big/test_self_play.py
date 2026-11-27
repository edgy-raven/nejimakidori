"""Native collection preserves payments and deterministic batched streams."""

import json
import pathlib
import types

import numpy
import pytest
import tensorflow as tf

from integration import reference_policy
from log_dataset import _native_collection, call_states
from model import actions
from model import actor as actor_layers
from model import joint_settlement, outcomes
from training import self_play


def test_native_nagashi_targets_charge_deposits_once(monkeypatch):
    recorded = json.loads(
        pathlib.Path(
            "integration/fixtures/model/nagashi_deposit.json"
        ).read_text()
    )
    starts = [
        index
        for index, event in enumerate(recorded["events"])
        if event["type"] == "start_kyoku"
    ]
    events = recorded["events"][: starts[1]] + [{"type": "end_game"}]
    request = iter(
        [
            {
                "features": numpy.zeros(4, call_states.ROW_DTYPE).tobytes(),
                "post_call_features": b"",
                "post_call_owner": [],
                "post_call_option": b"",
                "head": [0] * 4,
                "legal": numpy.tile(numpy.arange(150) == 0, (4, 1)).tobytes(),
                "game": [0] * 4,
                "hand": [0] * 4,
                "seat": list(range(4)),
                "event": [starts[0] + 1] * 4,
            },
            None,
        ]
    )
    monkeypatch.setattr(
        _native_collection,
        "NativeCollector",
        lambda games, seed: types.SimpleNamespace(
            next=lambda: next(request),
            submit=lambda choices: None,
            finish=lambda: ([json.dumps(events)], {}),
        ),
    )
    batch, _ = self_play.collect(reference_policy.uniform_logits, 1, 0)
    labels = joint_settlement.payment_labels(
        batch["payment_points"], {"pot": tf.zeros([4])}
    )
    numpy.testing.assert_array_equal(
        tf.gather(
            joint_settlement.payment_values(labels),
            labels["payment"],
            axis=1,
            batch_dims=1,
        ),
        batch["payment_points"],
    )
    assert "settlement" not in batch
    payments = numpy.zeros((4, 4, 4), numpy.int32)
    payments[2, :, 2] = [2000, 2000, 0, 4000]
    payments[3, 3, 3] = -1000
    numpy.testing.assert_array_equal(
        batch["payment_points"][0], payments[outcomes.PAYMENT_CELLS]
    )
    # Stacked yakuman remain in the declared monetary support.
    for count in range(1, 5):
        outcomes.payment_classes(
            numpy.array([8000, 16000, 32000, 48000]) * count
        )
    with pytest.raises(tf.errors.InvalidArgumentError, match="Unsupported"):
        joint_settlement.payment_labels(
            tf.constant([[5000]]), {"pot": tf.zeros([1])}
        )


def test_batched_streams_preserve_serial_features_and_targets():
    def streams():
        return (self_play.collection_steps(1, seed) for seed in (941, 944, 947))

    actual = self_play.collect_streams(
        reference_policy.uniform_logits, streams(), max_active=2
    )
    expected = [
        self_play.collect_streams(reference_policy.uniform_logits, [stream])[0]
        for stream in streams()
    ]
    tf.nest.map_structure(numpy.testing.assert_array_equal, actual, expected)
    for rows, statistics in actual:
        assert statistics["simulated_decisions"] == len(rows["chosen"])


def test_greedy_reference_selects_marginal_branches_before_continuations():
    logits = numpy.full((5, 150), -1e9, numpy.float32)
    logits[0, 5] = 0
    logits[1, [0, 37, 38]] = numpy.log([0.45, 0.28, 0.27])
    logits[2, [0, 1, 2, 38, 112]] = numpy.log([0.4, 0.23, 0.22, 0.05, 0.1])
    logits[3, 2] = 0
    logits[4, 1] = 0
    numpy.testing.assert_array_equal(
        actor_layers.greedy_choices(tf.constant(logits), tf.range(5)),
        [5, 37, 1, 2, 1],
    )


def test_native_self_play_records_legal_sampled_actions():
    class Actor:
        def __init__(self):
            self.scale = tf.Variable(0.0)
            self.traces = 0

        def __call__(self, inputs, training=False):
            self.traces += 1
            count = tf.shape(inputs["decision_phase"])[0]
            return {
                **{
                    name: tf.zeros((count, actions.ACTOR_OUTPUTS[name][0]))
                    + self.scale
                    * tf.range(
                        actions.ACTOR_OUTPUTS[name][0],
                        dtype=tf.float32,
                    )
                    for name in actions.TRAIN_POLICY_HEADS
                },
                "board": tf.zeros((count, 1)),
            }

    tf.keras.utils.set_random_seed(12)
    actor = Actor()
    predict = self_play.predictor(actor)

    collector = _native_collection.NativeCollector(1, 12)
    batch, statistics = self_play.collect(predict, 1, 12, collector=collector)
    events = json.loads(collector.finish()[0][0])
    tf.keras.utils.set_random_seed(12)
    tf.nest.map_structure(
        numpy.testing.assert_array_equal,
        (batch, statistics),
        self_play.collect(predict, 1, 12),
    )
    assert statistics["games"] == 1
    assert statistics["hands"] >= 1
    for name, event_type in (
        ("hands", "start_kyoku"),
        ("wins", "hora"),
        ("draws", "ryukyoku"),
        ("games", "end_game"),
        ("riichi_hands", "reach_accepted"),
    ):
        assert statistics[name] == sum(
            event["type"] == event_type for event in events
        )
    assert statistics["win_points"] == sum(
        event["deltas"][event["actor"]]
        for event in events
        if event["type"] == "hora"
    )
    assert len(batch["chosen"]) > 100
    probability = numpy.exp(batch["behavior"])
    numpy.testing.assert_allclose(probability.sum(-1), 1, atol=1e-6)
    assert numpy.all(probability[~batch["legal"]] == 0)
    assert numpy.all(
        batch["legal"][numpy.arange(len(batch["chosen"])), batch["chosen"]]
    )
    assert len(numpy.unique(batch["payment_points"], axis=0)) > 1
    assert "advantages" not in batch
    assert "value" not in batch
    current = actor_layers.distribution(
        Actor()(batch["features"]), batch["head"], batch["legal"]
    )
    numpy.testing.assert_allclose(current, batch["behavior"], atol=1e-6)
    actor.scale.assign(1.0)
    for count in (1, 2):
        outputs = predict(
            call_states.select(batch["features"], numpy.arange(count))
        )
        numpy.testing.assert_array_equal(
            outputs["discard_policy_logits"][:, 1], numpy.ones(count)
        )
    assert actor.traces == 1


def test_fresh_critic_targets_preserve_deposits_and_preempted_calls(
    monkeypatch,
):
    events = [
        {
            "type": "start_kyoku",
            "scores": [25000] * 4,
            "oya": 0,
            "kyoku": 1,
            "bakaze": "E",
            "honba": 1,
            "kyotaku": 1,
        },
        {"type": "reach", "actor": 0},
        {"type": "reach_accepted", "actor": 0},
        {"type": "dahai", "actor": 0},
        {"type": "pon", "actor": 2},
        {"type": "dahai", "actor": 2},
        {"type": "reach", "actor": 3},
        {"type": "reach_accepted", "actor": 3},
        {"type": "dahai", "actor": 3},
        {
            "type": "hora",
            "actor": 1,
            "target": 3,
            "deltas": [0, 11300, 0, -8300],
        },
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    features = numpy.zeros(4, call_states.ROW_DTYPE)
    features["riichi_state"][1, 3] = 2  # Absolute seat 0 already paid.
    features["riichi_state"][2, 2] = 2
    features["riichi_state"][3, 1] = 2
    features["riichi_state"][3, 0] = 1  # Declaration is not acceptance.
    legal = numpy.zeros((4, 150), bool)
    legal[numpy.arange(4), [37, 112, 112, 0]] = True

    class Collector:
        submitted = False

        def __init__(self, games, seed):
            pass

        def next(self):
            if self.submitted:
                return None
            return {
                "features": features.tobytes(),
                "post_call_features": b"",
                "post_call_owner": [],
                "post_call_option": b"",
                "head": [1, 2, 2, 0],
                "game": [0] * 4,
                "hand": [0] * 4,
                "seat": [0, 1, 2, 3],
                "event": [1, 4, 4, 7],
                "legal": legal.tobytes(),
            }

        def submit(self, chosen):
            assert chosen == [37, 112, 112, 0]
            self.submitted = True

        def finish(self):
            return [json.dumps(events)], {"games": 1}

    monkeypatch.setattr(_native_collection, "NativeCollector", Collector)
    batch, _ = self_play.collect(
        lambda features: {
            name: numpy.zeros((4, actions.ACTOR_OUTPUTS[name][0]))
            for name in actions.TRAIN_POLICY_HEADS
        },
        1,
        12,
    )
    numpy.testing.assert_array_equal(batch["chosen"], [37, 112, 112, 0])
    # Honba is removed; carried pots and only future deposits remain.
    expected = numpy.zeros((4, 4, 4), numpy.int32)
    expected[0, 3, 1] = 8000
    expected[3] = numpy.diag([-1000, 3000, 0, -1000])
    numpy.testing.assert_array_equal(
        batch["payment_points"][0], expected[outcomes.PAYMENT_CELLS]
    )
    expected[3, 0, 0] = 0
    for seat in (1, 2, 3):
        order = (seat + numpy.arange(4)) % 4
        numpy.testing.assert_array_equal(
            batch["payment_points"][seat],
            expected[:, order][:, :, order][outcomes.PAYMENT_CELLS],
        )
