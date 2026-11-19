"""Paired outcomes correct either action without treating winners as labels."""

import numpy
import tensorflow as tf

from model import actions
from model import network as joint_network
from training import self_play
from training.corrections import trainer


def test_paired_outcomes_reverse_gradients_and_ignore_padding():
    logits = tf.Variable([[0.0, 0.0, 0.0]] * 4)
    magnet = tf.Variable(tf.nn.log_softmax(logits))
    # Equal placement, different terminal points: +0.1, -0.1, then a tie.
    settlement = numpy.array(
        [[[25000] * 4 + [0, 1, 2, 3, 0, 8, 1, 0]] * 2] * 4,
        numpy.int32,
    )
    settlement[0, 1, :2] += [10000, -10000]
    settlement[1, 0, :2] += [10000, -10000]
    settlement[3, 1, :2] += [1000000, -1000000]
    with tf.GradientTape(persistent=True) as tape:
        metrics = trainer.paired_terms(
            tf.nn.log_softmax(logits),
            magnet,
            {"settlement": settlement, "actions": tf.constant([[0, 1]] * 4)},
            tf.constant(3),
        )
    gradient = tape.gradient(metrics["pair_loss"], logits).numpy()
    assert gradient[0, 0] > 0 > gradient[0, 1]
    numpy.testing.assert_allclose(gradient[0], -gradient[1], atol=1e-7)
    numpy.testing.assert_allclose(gradient[2:], 0, atol=1e-7)
    numpy.testing.assert_allclose(gradient[:, 2], 0, atol=1e-7)
    numpy.testing.assert_allclose(metrics["pair_gap_objective"], 0, atol=1e-7)
    numpy.testing.assert_allclose(metrics["magnet_kl"], 0, atol=1e-7)
    assert tape.gradient(metrics["magnet_kl"], magnet) is None


def test_correction_updates_policy_and_critic_from_native_paired_roots():
    roots = []

    def predict(features):
        return {
            name: numpy.zeros(
                (len(features), actions.ACTOR_OUTPUTS[name][0]), numpy.float32
            )
            for name in actions.TRAIN_POLICY_HEADS
        }

    def capture(collector, batch):
        if not roots and batch["head"][0] == 0:
            roots.append(
                (
                    collector.snapshot(0),
                    numpy.flatnonzero(batch["legal"][0])[:2],
                )
            )

    self_play.collect(predict, 1, 941, root_callback=capture)
    branches = []
    for action in roots[0][1]:
        batch, _ = self_play.collect_streams(
            predict,
            [
                self_play.collection_steps(
                    1,
                    123,
                    collector=roots[0][0].fork(1, 123),
                    forced_root=(roots[0][0].decision, int(action), 8),
                )
            ],
        )[0]
        branches.append(
            tf.nest.map_structure(
                lambda value: value[batch["paired_root"]], batch
            )
        )
    tf.nest.map_structure(
        numpy.testing.assert_array_equal,
        branches[0]["features"],
        branches[1]["features"],
    )
    batch = {
        "features": branches[0]["features"],
        "head": branches[0]["head"].astype(numpy.int32),
        "legal": branches[0]["legal"],
        "actions": numpy.stack(
            [row["chosen"] for row in branches], axis=1
        ).astype(numpy.int32),
        "settlement": numpy.stack(
            [row["settlement"] for row in branches], axis=1
        ),
        "payment_points": numpy.stack(
            [row["payment_points"] for row in branches], axis=1
        ),
    }
    network = joint_network.JointPolicyPayment()
    outputs, state = network.candidate_context(
        batch["features"], network.actor(batch["features"], training=False)
    )
    network.critic(
        network.chosen_inputs(
            batch["features"], outputs, state, batch["actions"][:, 0]
        )
    )
    fixed = {name: tf.identity(value) for name, value in outputs.items()}
    before_actor = network.actor.get_weights()
    before_critic = network.critic.get_weights()
    learner = trainer.CorrectionTrainer(
        network=network,
        magnet=lambda **features: fixed,
        learning_rate=1e-5,
    )
    metrics = learner.update(batch, tf.constant(1))
    assert all(numpy.isfinite(value).all() for value in metrics.values())
    assert metrics["gradient/local_norm"] > 0
    assert metrics["settlement_supported_fraction"] == 1
    assert any(
        not numpy.array_equal(before, after)
        for before, after in zip(before_actor, network.actor.get_weights())
    )
    assert any(
        not numpy.array_equal(before, after)
        for before, after in zip(before_critic, network.critic.get_weights())
    )
    numpy.testing.assert_allclose(
        metrics["joint_loss"],
        metrics["pair_loss"]
        + 0.3 * metrics["magnet_kl"]
        + 0.1 * (metrics["payment_nll"] + metrics["critic_constraint_loss"])
        + 0.01 * metrics["constraint_loss"],
        rtol=1e-6,
    )
