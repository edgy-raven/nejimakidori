"""Replay preserves hand targets, gradient boundaries and fixed anchors."""

import json
import pathlib
import types

import numpy
import pytest
import tensorflow as tf

from log_dataset import afk, replay, training_data
from model import actions
from model import actor as actor_layers
from model import beliefs
from model import network as joint_network
from training import self_play
from training import trainer as training
from training.candidate import replay as candidate_replay


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


def test_greedy_reference_selects_marginal_branches_before_continuations():
    logits = numpy.full((5, 150), -1e9, numpy.float32)
    logits[0, 5] = 0
    logits[1, [0, 37, 38]] = numpy.log([0.45, 0.28, 0.27])
    logits[2, [0, 1, 2, 38, 112]] = numpy.log([0.4, 0.17, 0.15, 0.18, 0.1])
    logits[3, 2] = 0
    logits[4, 1] = 0
    numpy.testing.assert_array_equal(
        actor_layers.greedy_choices(tf.constant(logits), tf.range(5)),
        [5, 37, 1, 2, 1],
    )


def test_magnet_uses_current_to_fixed_reference_kl():
    reference = tf.Variable(tf.math.log([[0.1, 0.9], [0.5, 0.5]]))
    logits = tf.Variable(tf.math.log([[0.8, 0.2], [0.2, 0.8]]))
    with tf.GradientTape(persistent=True) as tape:
        current = tf.nn.log_softmax(logits)
        kl = actor_layers.policy_kl(
            reference=reference,
            current=current,
            reverse=True,
        )
    numpy.testing.assert_allclose(
        kl,
        tf.reduce_mean(
            tf.reduce_sum(tf.exp(current) * (current - reference), axis=-1)
        ),
    )
    assert tape.gradient(kl, reference) is None
    assert numpy.linalg.norm(tape.gradient(kl, logits)) > 0
    numpy.testing.assert_allclose(
        actor_layers.policy_kl(
            reference=reference,
            current=reference,
            reverse=True,
        ),
        0,
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
                "opponent_shanten": numpy.zeros((4, 3), numpy.int8),
                "opponent_ukeire": numpy.zeros((4, 3, 34), numpy.int8),
                "opponent_hand_counts": numpy.zeros((4, 3, 37), numpy.int8),
                "ron_legal": numpy.full((4, 3, 37), -1, numpy.int8),
                "ron_payments": numpy.full((4, 3, 37), -1, numpy.int32),
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


def test_mixed_policy_updates_shared_model_with_fixed_anchors():
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/payment_dealer_90fu.json")
        .read_text()
    )
    targets = training_data.game_targets(events + [{"type": "end_game"}])
    observations = [
        row
        for row in replay.Replay(
            "training-probe", afk.retained_hands(events)[0]
        ).observations()
        if numpy.any(
            row.features["action_win_value_range"]
            != row.features["magnet_win_value_range"].astype(numpy.float32)
            * 100
        )
    ][:2]
    raw = tf.constant(
        [
            training_data.serialize(row, targets[row.kyoku_id])
            for row in observations
        ]
    )
    inputs = training_data.parse_batch(raw)
    network = joint_network.JointPolicyPayment()
    network(inputs)
    network.compile_training()
    reference = joint_network.JointPolicyPayment()
    reference(inputs)
    reference.actor.set_weights(network.actor.get_weights())
    reference_weights = reference.actor.get_weights()
    trainer = training.Trainer(
        network=network,
        features=inputs[0],
        mirror=reference.actor,
        magnet=reference.actor,
        args=types.SimpleNamespace(
            learning_rate=1e-5,
            beta=16.0,
            expert_prior=0.2,
            mirror_coefficient=0.3,
            magnet_coefficient=0.3,
            risk_coefficient=0.1,
        ),
    )
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
    batch = candidate_replay.decode_records(raw, chosen)
    batch["features"] = trainer.feature_packing.pack(batch["features"])
    batch["recorded"] = tf.constant([True, False])
    batch["distill"] = tf.where(
        batch["recorded"],
        tf.argmax(
            tf.cast(batch["legal"], tf.int32), axis=1, output_type=tf.int32
        ),
        -1,
    )
    # Both actor objectives must leave critic forecasts detached; the
    # observed-outcome loss still trains the critic through its shared inputs.
    with tf.GradientTape(persistent=True) as tape:
        metrics = trainer.policy_update.python_function(
            batch, tf.constant(2), train=False
        )
        # Compare the same compiled training forward: bfloat16 kernels can
        # round differently when no gradient recording is requested.
        predictions = network.actor(inputs[0], training=False)
        expected, _ = beliefs.belief_loss_terms(
            inputs[0], inputs[1], predictions
        )
        current = actor_layers.distribution(
            predictions, batch["head"], batch["legal"]
        )
    assert (
        tf.linalg.global_norm(
            tape.gradient(metrics["policy"], network.critic.trainable_variables)
        )
        == 0
    )
    assert (
        tf.linalg.global_norm(
            tape.gradient(
                metrics["payment_nll"], network.critic.trainable_variables
            )
        )
        > 0
    )
    fixed_inputs = {
        **inputs[0],
        "action_win_value_range": tf.cast(
            inputs[0]["magnet_win_value_range"], tf.float32
        )
        * 100,
        "action_win_value_valid": inputs[0]["magnet_win_value_valid"],
    }
    fixed = actor_layers.distribution(
        reference.actor(fixed_inputs, training=False),
        batch["head"],
        batch["legal"],
    )
    numpy.testing.assert_allclose(
        metrics["magnet_kl"],
        actor_layers.policy_kl(fixed, current, reverse=True),
        atol=1e-5,
    )
    assert float(metrics["risk_penalty_st3"]) >= 0
    for name, value in expected.items():
        numpy.testing.assert_allclose(metrics[name], value, rtol=1e-5)
    numpy.testing.assert_allclose(
        metrics["belief_loss"], sum(expected.values()) / 3, rtol=1e-5
    )
    for head in network.beliefs.heads.values():
        assert (
            tf.linalg.global_norm(
                tape.gradient(metrics["belief_loss"], head.trainable_variables)
            )
            > 0
        )
    assert (
        tf.linalg.global_norm(
            tape.gradient(
                metrics["belief_loss"],
                [network.actor.get_layer("board_block_3_contract").kernel],
            )
        )
        > 0
    )
    numpy.testing.assert_allclose(
        metrics["joint_loss"],
        metrics["policy"]
        + metrics["weighted_mirror_loss"]
        + metrics["weighted_magnet_loss"]
        + 0.2 * metrics["expert_prior_loss"]
        + 0.1 * (metrics["payment_nll"] + metrics["critic_constraint_loss"])
        + 0.01 * metrics["constraint_loss"]
        + metrics["belief_loss"],
        rtol=1e-6,
    )
    padded = tf.nest.map_structure(
        lambda value: tf.concat([value, value[-1:]], axis=0), batch
    )
    padded_metrics = trainer.policy_update.python_function(
        padded, tf.constant(2), train=False
    )
    padded["beliefs"] = tf.nest.map_structure(
        lambda value: tf.concat([value[:2], tf.zeros_like(value[-1:])], axis=0),
        padded["beliefs"],
    )
    changed_padding = trainer.policy_update.python_function(
        padded, tf.constant(2), train=False
    )
    for name in (*expected, "belief_loss", "joint_wait_brier"):
        numpy.testing.assert_allclose(
            changed_padding[name], padded_metrics[name], rtol=1e-5, atol=1e-7
        )
    numpy.testing.assert_allclose(metrics["record_fraction"], 0.5)
    numpy.testing.assert_allclose(
        metrics["policy"],
        0.5 * (metrics["expert_loss"] + metrics["self_play_loss"]),
        rtol=1e-6,
    )
    numpy.testing.assert_allclose(
        metrics["self_play_loss"],
        -metrics["self_play_gain_st3"],
        rtol=1e-6,
    )
    numpy.testing.assert_allclose(
        metrics["expert_loss"],
        0.5 * (metrics["expert_weighted_ce"] + metrics["distill_weighted_ce"]),
        rtol=1e-6,
    )
    before = {
        "policy": network.actor.get_layer("final_pass").kernel.numpy().copy(),
        "encoder": network.actor.get_layer("board_block_3_contract")
        .kernel.numpy()
        .copy(),
        "critic": network.critic.event.kernel.numpy().copy(),
    }
    distributed = training.DistributedTrainer(
        trainer, tf.distribute.get_strategy()
    )
    for _ in range(2):
        metrics = distributed.policy_update(batch, tf.constant(2))
        assert all(numpy.isfinite(float(v)) for v in metrics.values())
        numpy.testing.assert_allclose(
            metrics["weighted_mirror_loss"],
            0.3 * metrics["mirror_kl"],
            atol=1e-7,
        )
        numpy.testing.assert_allclose(
            metrics["weighted_magnet_loss"],
            0.3 * metrics["magnet_kl"],
            atol=1e-7,
        )
    assert int(trainer.optimizer.iterations) == 2
    assert float(metrics["mirror_kl"]) > 0
    assert float(metrics["payment_nll"]) > 0
    assert float(metrics["settlement_supported_fraction"]) == 1
    for name, value in {
        "policy": network.actor.get_layer("final_pass").kernel,
        "encoder": network.actor.get_layer("board_block_3_contract").kernel,
        "critic": network.critic.event.kernel,
    }.items():
        assert numpy.any(before[name] != value.numpy()), name
    for before, after in zip(reference_weights, reference.actor.get_weights()):
        numpy.testing.assert_array_equal(before, after)
