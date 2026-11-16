"""One- and two-replica updates preserve masked losses and gradients."""

import os
import pathlib
import subprocess
import sys


def test_distributed_updates_match_single_device():
    subprocess.run(
        args=[
            sys.executable,
            "-m",
            "integration.big.test_distributed_imitation",
        ],
        cwd=pathlib.Path(__file__).parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
        timeout=240,
    )


def check_streamed_payment_backward_matches_one_and_two_replicas():
    from integration.small.test_joint_settlement import inputs
    from log_dataset import scoring
    from model import critic, losses, payment_scoring

    public = inputs()
    public.update(
        {
            "state": tf.concat(
                [
                    tf.fill((1, 12), 0.25),
                    tf.fill((1, 408), 0.5),
                    tf.ones((1, 92)),
                ],
                -1,
            ),
            "action_tile": tf.zeros(1, tf.int32),
            "has_meld": tf.zeros((1, 4), tf.bool),
            "opened": tf.zeros((1, 4), tf.bool),
            "riichi_required": tf.zeros((1, 4), tf.bool),
            "ippatsu_allowed": tf.ones((1, 4), tf.bool),
            "public_fu": tf.zeros((1, 4), tf.int32),
            "public_bonus": tf.zeros((1, 4), tf.int32),
            "yakuman_allowed": tf.ones((1, 4), tf.bool),
        }
    )
    for name in [
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
        "bonus": tf.fill((1, 24), 2),
        "score": tf.fill(
            (1, 24, len(scoring.BONUS_HAN)),
            payment_scoring.SCORES.index((0, 1)),
        ),
    }
    public = {
        name: tf.repeat(value, 2, axis=0) for name, value in public.items()
    }
    public["state"] = tf.concat(
        [
            public["state"][:, :420],
            public["state"][:, 420:] * tf.constant([[1.0], [2.0]]),
        ],
        -1,
    )
    labels = {
        name: tf.repeat(value, 2, axis=0) for name, value in labels.items()
    }
    tf.keras.utils.set_random_seed(941)
    for replicas in (1, 2):
        strategy = strategies[replicas]
        with strategy.scope():
            model = critic.PaymentCritic(belief_offset=0)
            model(public)
            if replicas == 1:
                weights = model.get_weights()
            else:
                model.set_weights(weights)
            model.distributions = tf.function(
                model.distributions, jit_compile=True
            )
            model.joint_distributions = tf.function(
                model.joint_distributions, jit_compile=True
            )
            model.chunk_statistics = tf.function(
                model.chunk_statistics, jit_compile=True
            )
            model.chunk_gradients = tf.function(
                model.chunk_gradients, jit_compile=True
            )
            model.supervised_statistics = tf.function(
                model.supervised_statistics
            )

        def update():
            begin = (
                tf.distribute.get_replica_context().replica_id_in_sync_group
                * (2 // replicas)
            )
            with tf.GradientTape() as tape:
                stats = model.supervised_statistics(
                    {
                        name: value[begin : begin + 2 // replicas]
                        for name, value in public.items()
                    },
                    {
                        name: value[begin : begin + 2 // replicas]
                        for name, value in labels.items()
                    },
                )
                means = losses.packed_means(stats)
                loss = (
                    0.1 * means[critic.CRITIC_METRICS.index("payment_nll")]
                    + 0.01 * means[critic.CRITIC_METRICS.index("score_nll")]
                    + 0.01 * means[critic.CRITIC_METRICS.index("bonus_han_nll")]
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
                from integration.small.test_joint_settlement import (
                    expanded_terms,
                )

                terms = expanded_terms(labels, outputs)
                direct = (
                    0.1 * terms["payment_nll"]
                    + 0.01
                    * critic.score_cross_entropy(
                        labels["score"], outputs["score_logits"]
                    )
                    + 0.01
                    * losses.mean_valid(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=labels["bonus"],
                            logits=outputs["bonus_logits"],
                        ),
                        labels["bonus"] >= 0,
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
                numpy.testing.assert_allclose(
                    first, second, atol=1e-5, rtol=1e-4
                )


def check_training_adamw_and_masked_losses_match_reference():
    from log_dataset import _native_collection
    from model import actions
    from model import actor as actor_layers
    from model import losses
    from training import trainer as training

    # Exercise real reductions and AdamW without a mock model architecture.
    for replicas in (1, 2):
        strategy = strategies[replicas]
        with strategy.scope():
            logits = tf.Variable([0.1, -0.1])
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=0.01, weight_decay=1e-4, clipnorm=1.0
            )
            optimizer.build([logits])

        def step(count):
            begin = (
                tf.distribute.get_replica_context().replica_id_in_sync_group
                * (4 // replicas)
            )
            indices = tf.range(begin, begin + 4 // replicas)
            valid = indices < count
            recorded = (indices >= 2) & valid
            generated = (indices < 2) & valid
            chosen = indices % 2
            with tf.GradientTape() as tape:
                current = tf.nn.log_softmax(
                    logits[None]
                    * tf.cast(indices[:, None] + 1, tf.float32)
                    / 10
                )

                def terms(mean):
                    expert = actor_layers.weighted_imitation_terms(
                        nll=-tf.gather(current, chosen, batch_dims=1),
                        advantages=tf.cast(indices, tf.float32) / 10,
                        eligible=recorded,
                        mean=mean,
                    )
                    return {
                        "expert": expert["loss"],
                        "generated": mean(
                            -tf.reduce_sum(tf.exp(current) * [0.1, -0.1], -1),
                            generated,
                        ),
                        "fraction": mean(tf.cast(recorded, tf.float32), valid),
                        "mirror": mean(
                            tf.reduce_sum(
                                tf.exp(current)
                                * (current - tf.math.log([0.6, 0.4])),
                                -1,
                            ),
                            valid,
                        ),
                        "magnet": mean(
                            tf.reduce_sum(
                                tf.exp(current)
                                * (current - tf.math.log([0.4, 0.6])),
                                -1,
                            ),
                            valid,
                        ),
                        "ess": expert["effective_sample_fraction"],
                    }

                means = losses.MaskedMeans()
                terms(means.collect)
                means.reduce()
                metrics = terms(means.read)
                loss = (
                    metrics["fraction"] * metrics["expert"]
                    + (1 - metrics["fraction"]) * metrics["generated"]
                    + 0.3 * (metrics["mirror"] + metrics["magnet"])
                )
                scaled = loss / replicas
            training.apply_gradients(optimizer, tape, scaled, [logits])
            return tf.stack([loss, *metrics.values()])

        @tf.function
        def update(count):
            return strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                strategy.run(step, args=(count,)),
                None,
            )

        result = []
        # Includes unequal source labels and an entirely padded second replica.
        for count in (4, 3, 1):
            result.append(update(tf.constant(count)).numpy())
        result.extend(value.numpy() for value in [logits, *optimizer.variables])
        if replicas == 1:
            expected = result
        else:
            for left, right in zip(expected, result):
                numpy.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
    from training import self_play

    class Actor(tf.Module):
        def __init__(self):
            super().__init__()
            self.scale = tf.Variable(1.0)

        def __call__(self, features, training=False):
            return {
                name: tf.cast(features["decision_phase"][:, None], tf.float32)
                * self.scale
                * tf.ones((1, actions.ACTOR_OUTPUTS[name][0]))
                for name in actions.TRAIN_POLICY_HEADS
            }

    for replicas in (1, 2):
        with strategies[replicas].scope():
            actor = Actor()
            predict = self_play.predictor(actor)
        for scale in (1.0, 2.0):
            actor.scale.assign(scale)
            for count in (1, 3, 8):
                features = {
                    name: numpy.zeros((count, *spec["shape"]), spec["dtype"])
                    for name, spec in _native_collection.INPUTS.items()
                }
                features["decision_phase"][:] = numpy.arange(count)
                for values in predict(features).values():
                    numpy.testing.assert_array_equal(
                        values[:, 0], numpy.arange(count) * scale
                    )
                    assert len(values) == count


if __name__ == "__main__":
    import numpy
    import tensorflow as tf

    cpu = tf.config.list_physical_devices("CPU")[0]
    tf.config.set_logical_device_configuration(
        cpu, [tf.config.LogicalDeviceConfiguration()] * 2
    )
    strategies = {
        replicas: tf.distribute.MirroredStrategy(
            devices=[f"/cpu:{i}" for i in range(replicas)]
        )
        for replicas in (1, 2)
    }
    check_streamed_payment_backward_matches_one_and_two_replicas()
    check_training_adamw_and_masked_losses_match_reference()
