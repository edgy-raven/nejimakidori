"""AWR replay preserves hand targets, gradient boundaries and a fixed policy."""

import json
import pathlib
import types

import numpy
import pytest
import tensorflow as tf

from log_dataset import afk, replay, training_data
from model import actions
from model import actor as actor_layers
from model import network as joint_network
from training import self_play
from training import trainer as training


@pytest.mark.parametrize(
    "head, gap, tail",
    [("response_policy_logits", 0, 147), ("riichi_policy_logits", 36, 35)],
)
def test_entropy_explores_followups_without_biasing_the_parent(head, gap, tail):
    parent = tf.Variable(
        [numpy.log(0.25), 0.0, numpy.log(4.0), -85.0, -90.0, -1e9],
        dtype=tf.float32,
    )
    followup = tf.Variable(
        [[numpy.log(0.8), numpy.log(0.2)]] * parent.shape[0], dtype=tf.float32
    )
    with tf.GradientTape(persistent=True) as tape:
        logits = tf.concat(
            [
                tf.zeros((parent.shape[0], 1)),
                tf.fill((parent.shape[0], gap), -1e9),
                parent[:, None] + tf.nn.log_softmax(followup),
                tf.fill((parent.shape[0], tail), -1e9),
            ],
            axis=1,
        )
        log_probability = tf.nn.log_softmax(logits)
        entropy = actor_layers.action_entropy(log_probability, head)
        total = tf.reduce_sum(entropy)
        offline = actor_layers.policy_entropy(
            {
                name: tf.fill((parent.shape[0],), 0 if name == head else -1)
                for name in actions.TRAIN_POLICY_HEADS
            },
            {
                name: (
                    logits if name == head else tf.zeros((parent.shape[0], 150))
                )
                for name in actions.TRAIN_POLICY_HEADS
            },
        )
    # Parent exploration is exactly binary entropy, independent of the
    # continuation's entropy. Rare and impossible branches stay finite.
    expected = [0.16 * numpy.log(4), 0, -0.16 * numpy.log(4), 0, 0, 0]
    numpy.testing.assert_allclose(
        tape.gradient(total, parent), expected, atol=1e-6
    )
    numpy.testing.assert_allclose(
        tape.gradient(offline, parent),
        numpy.array(expected) / parent.shape[0],
        atol=1e-6,
    )
    gradient = tape.gradient(total, followup).numpy()
    assert (gradient[:3, 0] < 0).all()
    assert (gradient[:3, 1] > 0).all()
    assert numpy.isfinite(gradient).all()
    numpy.testing.assert_array_equal(gradient[-1], 0)
    numpy.testing.assert_allclose(
        entropy,
        -tf.reduce_sum(tf.exp(log_probability) * log_probability, axis=-1),
        atol=1e-6,
    )


def test_magnet_uses_current_to_fixed_reference_kl():
    reference = tf.Variable(tf.math.log([[0.1, 0.9], [0.5, 0.5]]))
    logits = tf.Variable(tf.math.log([[0.8, 0.2], [0.2, 0.8]]))
    with tf.GradientTape(persistent=True) as tape:
        current = tf.nn.log_softmax(logits)
        kl = actor_layers.policy_kl(reference, current, reverse=True)
    numpy.testing.assert_allclose(
        kl,
        tf.reduce_mean(
            tf.reduce_sum(tf.exp(current) * (current - reference), axis=-1)
        ),
    )
    assert tape.gradient(kl, reference) is None
    assert numpy.linalg.norm(tape.gradient(kl, logits)) > 0
    numpy.testing.assert_allclose(
        actor_layers.policy_kl(reference, reference, reverse=True), 0
    )


def test_native_self_play_records_legal_sampled_actions(monkeypatch):
    game_logs = []
    continuation = self_play.outcomes.hand_continuation_states

    def record_log(events):
        game_logs.append(events)
        return continuation(events)

    monkeypatch.setattr(
        self_play.outcomes, "hand_continuation_states", record_log
    )

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
    collector = self_play.Collector(predict, 1)
    batch, statistics = self_play.collect(collector.predict, 1, 12)
    tf.keras.utils.set_random_seed(12)
    tf.nest.map_structure(
        numpy.testing.assert_array_equal,
        (batch, statistics),
        self_play.collect(collector.predict, 1, 12),
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
            event["type"] == event_type for event in game_logs[0]
        )
    assert statistics["win_points"] == sum(
        event["deltas"][event["actor"]]
        for event in game_logs[0]
        if event["type"] == "hora"
    )
    assert len(batch["chosen"]) > 100
    probability = numpy.exp(batch["behavior"])
    numpy.testing.assert_allclose(probability.sum(-1), 1, atol=1e-6)
    assert numpy.all(probability[~batch["legal"]] == 0)
    assert numpy.all(
        batch["legal"][numpy.arange(len(batch["chosen"])), batch["chosen"]]
    )
    assert len(numpy.unique(batch["settlement"], axis=0)) > 1
    assert "advantages" not in batch
    assert "value" not in batch
    current = actor_layers.distribution(
        Actor()(batch["features"]), batch["head"], batch["legal"]
    )
    numpy.testing.assert_allclose(current, batch["behavior"], atol=1e-6)
    actor.scale.assign(1.0)
    for count in (1, 2):
        outputs = predict(
            {name: value[:count] for name, value in batch["features"].items()}
        )
        numpy.testing.assert_array_equal(
            outputs["discard_policy_logits"][:, 1], numpy.ones(count)
        )
    assert actor.traces == 1


def test_fresh_critic_targets_preserve_deposits_and_preempted_calls(
    monkeypatch,
):
    from log_dataset import _native_collection
    from model import outcomes

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
    features = numpy.zeros(4, self_play.FEATURE_DTYPE)
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
    numpy.testing.assert_array_equal(
        batch["settlement"][0, :4], [24000, 36300, 25000, 15700]
    )


@pytest.mark.parametrize("awr_q_std", [None, 1.0])
def test_awr_updates_shared_model_with_fixed_reference(awr_q_std):
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    targets = training_data.game_targets(events + [{"type": "end_game"}])
    observations = list(
        replay.Replay(
            "training-probe",
            afk.retained_hands(events)[0],
        ).observations()
    )[:2]
    inputs = training_data.parse_batch(
        tf.constant(
            [
                training_data.serialize(row, targets[row.kyoku_id])
                for row in observations
            ]
        )
    )
    network = joint_network.JointPolicyPayment()
    outputs = network(inputs)
    network.compile_training()
    reference = joint_network.JointPolicyPayment()
    reference(inputs)
    reference.actor.set_weights(network.actor.get_weights())
    reference_weights = reference.actor.get_weights()
    trainer = training.Trainer(
        network=network,
        features=inputs[0],
        mirror=reference.actor,
        mirror_coefficient=0.3,
        awr_q_std=awr_q_std,
        args=types.SimpleNamespace(
            awr_learning_rate=1e-5,
            awr_entropy=0.01,
            awr_temperature=10.0,
        ),
    )
    heads = tf.stack(
        [inputs[1][name] >= 0 for name in actions.TRAIN_POLICY_HEADS],
        axis=1,
    )
    head = tf.argmax(tf.cast(heads, tf.int32), axis=1, output_type=tf.int32)
    chosen = tf.reduce_max(
        tf.stack(
            [
                tf.cast(inputs[1][name], tf.int32)
                for name in actions.TRAIN_POLICY_HEADS
            ],
            axis=1,
        ),
        axis=1,
    )
    legal = tf.gather(
        tf.stack(
            [
                tf.pad(
                    outputs[name] > -1e3,
                    [[0, 0], [0, 150 - outputs[name].shape[-1]]],
                )
                for name in actions.TRAIN_POLICY_HEADS
            ],
            axis=1,
        ),
        head,
        batch_dims=1,
    )
    batch = dict(
        features=trainer.feature_packing.pack(inputs[0]),
        head=head,
        legal=legal,
        chosen=chosen,
        payment_points=tf.gather(
            training.joint_settlement.payment_values(inputs[2]),
            inputs[2]["payment"],
            axis=1,
            batch_dims=1,
        ),
        returns=inputs[2]["returns"],
    )
    # Actor regression must not backpropagate through its value baseline.
    with tf.GradientTape(persistent=True) as tape:
        awr_metrics = trainer.policy_update.python_function(
            batch, tf.constant(2), train=False
        )
    value_variables = network.actor.get_layer("state_value").trainable_variables
    if awr_q_std is not None:
        assert float(awr_metrics["q_std_st3"]) > 0
        numpy.testing.assert_allclose(
            awr_metrics["q_risk_st3"],
            awr_metrics["q_mean_st3"] - awr_metrics["q_std_st3"],
            atol=1e-4,
        )
        assert (
            float(
                tf.linalg.global_norm(
                    tape.gradient(
                        awr_metrics["policy"],
                        network.critic.trainable_variables,
                    )
                )
            )
            == 0
        )
    assert (
        float(
            tf.linalg.global_norm(
                tape.gradient(awr_metrics["policy"], value_variables)
            )
        )
        == 0
    )
    assert (
        float(
            tf.linalg.global_norm(
                tape.gradient(awr_metrics["value_loss"], value_variables)
            )
        )
        > 0
    )
    before = {
        "policy": network.actor.get_layer("final_pass").kernel.numpy().copy(),
        "encoder": network.actor.get_layer("board_block_3_contract")
        .kernel.numpy()
        .copy(),
        "critic": network.critic.event.kernel.numpy().copy(),
        "value": network.actor.get_layer("state_value").kernel.numpy().copy(),
    }
    for _ in range(2):
        metrics = trainer.policy_update(batch, tf.constant(2))
        assert all(numpy.isfinite(float(v)) for v in metrics.values())
        numpy.testing.assert_allclose(
            metrics["weighted_mirror_loss"],
            0.3 * metrics["mirror_kl"],
            atol=1e-7,
        )
    assert int(trainer.awr_optimizer.iterations) == 2
    assert float(metrics["mirror_kl"]) > 0
    assert float(metrics["payment_nll"]) > 0
    assert float(metrics["settlement_supported_fraction"]) == 1
    for name, value in {
        "policy": network.actor.get_layer("final_pass").kernel,
        "encoder": network.actor.get_layer("board_block_3_contract").kernel,
        "critic": network.critic.event.kernel,
        "value": network.actor.get_layer("state_value").kernel,
    }.items():
        assert numpy.any(before[name] != value.numpy()), name
    for before, after in zip(reference_weights, reference.actor.get_weights()):
        numpy.testing.assert_array_equal(before, after)
