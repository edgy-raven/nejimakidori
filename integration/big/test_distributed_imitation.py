"""Masked imitation has the same gradient across replica layouts."""

import os
import pathlib
import subprocess
import sys


def test_imitation_keeps_its_weight_with_uneven_replica_labels():
    # Labeled rows can all land on the last replica. Their policy
    # gradient must match a single-device mean over the same valid labels.
    subprocess.run(
        args=[
            sys.executable,
            "-c",
            """
import tensorflow as tf
cpu = tf.config.list_physical_devices("CPU")[0]
tf.config.set_logical_device_configuration(cpu, [
    tf.config.LogicalDeviceConfiguration(),
    tf.config.LogicalDeviceConfiguration(),
])
from model import feature_vector
from model import objectives
for replicas in (1, 2):
    strategy = tf.distribute.MirroredStrategy(
        devices=[f"/cpu:{index}" for index in range(replicas)])
    for packed in (False, True):
        with strategy.scope():
            value = tf.Variable(0.0)

        def update():
            active = (
                tf.distribute.get_replica_context().replica_id_in_sync_group
                == replicas - 1)
            labels = {
                name: tf.constant([-1])
                for name in feature_vector.TRAIN_POLICY_HEADS
            }
            labels["discard_policy_logits"] = tf.reshape(
                tf.where(active, 0, -1), (1,))
            with tf.GradientTape(persistent=True) as tape:
                outputs = {
                    name: tf.reshape(tf.stack([value, 0.0]), (1, 2))
                    for name in feature_vector.TRAIN_POLICY_HEADS
                }
                def terms(mean):
                    return (
                        objectives.policy_loss(labels, outputs, mean=mean),
                        objectives.awr_policy_terms(
                            objectives.policy_nll(labels, outputs)[0],
                            tf.ones((1,)), tf.reshape(active, (1,)),
                            mean=mean)["loss"],
                    )
                if packed:
                    means = objectives.MaskedMeans()
                    terms(means.collect)
                    means.reduce()
                    imitation, loss = terms(means.read)
                else:
                    imitation, loss = terms(objectives.mean_valid)
                scaled_imitation = imitation / replicas
                scaled = loss / replicas
            return (loss, tape.gradient(scaled, value),
                    tape.gradient(scaled_imitation, value))

        losses, gradients, imitation_gradients = strategy.run(update)
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, losses, None)
        gradient = strategy.reduce(tf.distribute.ReduceOp.SUM, gradients, None)
        tf.debugging.assert_near(loss, tf.math.log(2.0))
        tf.debugging.assert_near(gradient, -0.5)
        tf.debugging.assert_near(gradient, strategy.reduce(
            tf.distribute.ReduceOp.SUM, imitation_gradients, None))
""",
        ],
        cwd=pathlib.Path(__file__).parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
        timeout=60,
    )


def test_streamed_payment_backward_matches_one_and_two_replicas():
    subprocess.run(
        args=[
            sys.executable,
            "-c",
            """
import numpy
import tensorflow as tf

cpu = tf.config.list_physical_devices("CPU")[0]
tf.config.set_logical_device_configuration(
    cpu,
    [
        tf.config.LogicalDeviceConfiguration(),
        tf.config.LogicalDeviceConfiguration(),
    ],
)
from model import critic, joint_settlement, objectives, payment_scoring
from log_dataset import critic_data, payment_bonus
from integration.small.test_joint_settlement import inputs

public = inputs()
public.update(
    {
        "state": tf.ones((1, 64)),
        "immediate_wait": tf.ones((1, 3)) * 0.3,
        "immediate_safe": tf.zeros((1, 3), tf.bool),
        "immediate_allowed": tf.ones(1, tf.bool),
        "immediate_ankan": tf.zeros(1, tf.bool),
        "has_meld": tf.zeros((1, 4), tf.bool),
        "opened": tf.zeros((1, 4), tf.bool),
        "riichi_required": tf.zeros((1, 4), tf.bool),
        "ippatsu_allowed": tf.ones((1, 4), tf.bool),
        "immediate_ippatsu": tf.ones((1, 4), tf.bool),
        "public_fu": tf.zeros((1, 4), tf.int32),
        "immediate_draw_kind": tf.zeros(1, tf.int32),
        "immediate_tenpai": tf.fill((1, 4), -1),
    }
)
for name in [
    "immediate_probability",
    "immediate_score_probability",
    "score_probability",
    "event_logits",
    "deposit_logits",
]:
    del public[name]
labels = {
    "payment": tf.ones((1, 40), tf.int32),
    "round_placement": tf.constant([[0, 1, 2, 3]]),
    "pot": public["pot"],
    "points": tf.zeros(1),
    "score": tf.fill(
        (1, 24, len(payment_bonus.SCENARIOS)),
        payment_scoring.SCORES.index((0, 1)),
    ),
}
labels.update(critic_data.score_targets(labels.pop("score")))
public = {name: tf.repeat(value, 2, axis=0) for name, value in public.items()}
public["state"] *= tf.constant([[1.0], [2.0]])
public["immediate_wait"] *= tf.constant([[1.0], [2.0]])
labels = {name: tf.repeat(value, 2, axis=0) for name, value in labels.items()}
tf.keras.utils.set_random_seed(941)
for replicas in (1, 2):
    strategy = tf.distribute.MirroredStrategy(
        devices=[f"/cpu:{i}" for i in range(replicas)]
    )
    with strategy.scope():
        model = critic.PaymentCritic()
        model(public)
        if replicas == 1:
            weights = model.get_weights()
        else:
            model.set_weights(weights)
        model.distributions = tf.function(model.distributions, jit_compile=True)
        model.joint_distributions = tf.function(
            model.joint_distributions, jit_compile=True
        )
        model.chunk_statistics = tf.function(
            model.chunk_statistics, jit_compile=True
        )
        model.chunk_gradients = tf.function(
            model.chunk_gradients, jit_compile=True
        )
        model.supervised_statistics = tf.function(model.supervised_statistics)

    def update():
        begin = (
            tf.distribute.get_replica_context().replica_id_in_sync_group
            * (2 // replicas)
        )
        with tf.GradientTape() as tape:
            stats = model.supervised_statistics(
                {name: value[begin:begin + 2 // replicas]
                 for name, value in public.items()},
                {name: value[begin:begin + 2 // replicas]
                 for name, value in labels.items()},
            )
            means = objectives.packed_means(stats)
            loss = (
                0.1 * means[critic.CRITIC_METRICS.index("payment_nll")]
                + 0.01 * means[critic.CRITIC_METRICS.index("score_nll")]
            ) / replicas
        return loss, tape.gradient(loss, model.trainable_variables)

    @tf.function
    def distributed():
        loss, gradient = strategy.run(update)
        return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, None), [
            strategy.reduce(tf.distribute.ReduceOp.SUM, g, None)
            for g in gradient
        ]

    loss, gradients = distributed()
    if replicas == 1:
        # Recomputed training gradients must equal the direct objective;
        # detaching reported diagnostics cannot change an optimizer update.
        with tf.GradientTape() as tape:
            outputs = model.joint_distributions(public)
            from integration.small.test_joint_settlement import expanded_terms
            terms = expanded_terms(labels, outputs)
            direct = (
                0.1 * terms["payment_nll"]
                + 0.01 * critic.score_cross_entropy(
                    labels["score"], labels["score_weight"],
                    outputs["group_score_logits"]
                )
            )
        numpy.testing.assert_allclose(loss, direct, atol=1e-5, rtol=1e-4)
        for actual, expected in zip(
            gradients, tape.gradient(direct, model.trainable_variables)
        ):
            numpy.testing.assert_allclose(
                actual, expected, atol=1e-5, rtol=1e-4
            )
        reference = [g.numpy() for g in gradients]
    else:
        for first, second in zip(reference, gradients):
            numpy.testing.assert_allclose(first, second, atol=1e-5, rtol=1e-4)
""",
        ],
        cwd=pathlib.Path(__file__).parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
        timeout=180,
    )
