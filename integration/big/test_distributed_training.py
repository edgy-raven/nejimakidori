"""One- and two-replica updates preserve masked losses and gradients."""

import os
import pathlib
import subprocess
import sys


def test_count_collectives_accept_control_and_collection_vectors():
    import numpy
    import tensorflow as tf

    from training.corrections import distributed

    group = distributed.LearnerGroup()
    for values in ([0, 1280], [4096, 4889, 174], [1, 0]):
        numpy.testing.assert_array_equal(
            group.sum_counts(tf.constant(values, tf.int32)).numpy(), values
        )
    group.barrier()


def test_distributed_updates_match_single_device():
    subprocess.run(
        args=[
            sys.executable,
            "-m",
            "integration.big.test_distributed_training",
        ],
        cwd=pathlib.Path(__file__).parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
        timeout=240,
    )


def check_streamed_payment_backward_matches_one_and_two_replicas():
    from integration import settlement_reference
    from model import critic, losses

    public = settlement_reference.inputs()
    public.update(
        {
            "state": tf.concat(
                [
                    tf.fill((1, 12), 0.25),
                    tf.fill((1, 408), 0.5),
                    tf.fill((1, 555), 0.2),
                    tf.ones((1, 60)),
                    tf.zeros((1, critic.PREDICTED_TAIL_WIDTH)),
                    tf.ones((1, 32)),
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
            model = critic.PaymentCritic(
                belief_offset=0,
                shanten_offset=critic.BELIEF_WIDTH + 4,
                player_width=8,
            )
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
                    critic.settlement_objective(
                        dict(zip(critic.CRITIC_METRICS, tf.unstack(means)))
                    )
                    / replicas
                )
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
                terms = settlement_reference.expanded_terms(labels, outputs)
                direct = terms["payment_nll"] + tf.reduce_mean(
                    outputs["critic_constraint_loss"]
                    + tf.reduce_sum(model.direction_losses(public), axis=-1)
                    + model.shanten_direction_loss(public)
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
    from training import trainer as training
    from training.corrections import trainer as corrections

    # Exercise real reductions and AdamW without a mock model architecture.
    for replicas in (1, 2):
        strategy = strategies[replicas]
        with strategy.scope():
            logits = tf.Variable([0.1, -0.1])
            optimizer = training.AdamW(
                learning_rate=0.01, weight_decay=1e-4, clipnorm=1.0
            )
            optimizer.build([logits])

        def step(count):
            begin = (
                tf.distribute.get_replica_context().replica_id_in_sync_group
                * (4 // replicas)
            )
            indices = tf.range(begin, begin + 4 // replicas)
            with tf.GradientTape() as tape:
                current = tf.nn.log_softmax(
                    logits[None]
                    * tf.cast(indices[:, None] + 1, tf.float32)
                    / 10
                )

                metrics = corrections.mmd_terms(
                    features={
                        "decision_phase": tf.zeros_like(indices),
                        "trigger_tile_id": tf.zeros_like(indices),
                    },
                    current=tf.pad(
                        current, [[0, 0], [0, 148]], constant_values=-1e9
                    ),
                    mirror=tf.tile(
                        tf.pad(
                            tf.math.log([[0.6, 0.4]]),
                            [[0, 0], [0, 148]],
                            constant_values=-1e9,
                        ),
                        [4 // replicas, 1],
                    ),
                    magnet=tf.tile(
                        tf.pad(
                            tf.math.log([[0.4, 0.6]]),
                            [[0, 0], [0, 148]],
                            constant_values=-1e9,
                        ),
                        [4 // replicas, 1],
                    ),
                    values=tf.pad(
                        tf.cast(indices[:, None] + 1, tf.float32)
                        * [[0.1, -0.1]],
                        [[0, 0], [0, 293]],
                    ),
                    valid=indices < count,
                )
                loss = metrics["mmd_loss"]
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
        # Includes a partial batch and an entirely padded second replica.
        for count in (4, 3, 1):
            result.append(update(tf.constant(count)).numpy())
        result.extend(value.numpy() for value in [logits, *optimizer.variables])
        if replicas == 1:
            expected = result
        else:
            for left, right in zip(expected, result):
                numpy.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
    from log_dataset import call_states
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
                features = call_states.combine(
                    features,
                    {name: value[:0] for name, value in features.items()},
                    numpy.zeros(0, dtype="int32"),
                    numpy.zeros(0, dtype="int32"),
                )
                for values in predict(features).values():
                    numpy.testing.assert_array_equal(
                        values[:, 0], numpy.arange(count) * scale
                    )
                    assert len(values) == count


def check_imitation_weights_with_empty_replica():
    from model import losses
    from training import imitation

    for gaps in ([0.001, 0.01, 0.1, 0.5], [-0.3, -0.1, 0.1, 0.3]):
        beta, fraction, _ = imitation.calibrate_beta(tf.constant(gaps))
        numpy.testing.assert_allclose(fraction, 0.75, atol=1e-6)
        assert 0.0001 <= float(beta) <= 1.28
    beta, fraction, _ = imitation.calibrate_beta(tf.fill([4], 0.2))
    numpy.testing.assert_allclose([beta, fraction], [1.28, 1.0], atol=1e-6)

    for replicas in (1, 2):
        strategy = strategies[replicas]
        with strategy.scope():
            scale = tf.Variable(1.0)

        def step():
            rank = tf.distribute.get_replica_context().replica_id_in_sync_group
            indices = rank * (4 // replicas) + tf.range(4 // replicas)
            valid = indices < 2
            # Both valid gaps are far below zero with a tiny spread. The
            # second replica is padding, so its zero gaps must never overflow.
            gaps = tf.gather([-1.0, -1.0005, 0.0, 0.0], indices)
            with tf.GradientTape() as tape:
                tape.watch(gaps)
                weights, metrics = imitation.imitation_weights(gaps, valid)
                loss = (
                    losses.mean_valid(
                        scale * weights * tf.cast(indices + 1, tf.float32),
                        valid,
                    )
                    / replicas
                )
            gradient, q_gradient = tape.gradient(loss, [scale, gaps])
            assert q_gradient is None
            return weights, metrics, gradient

        @tf.function
        def update():
            weights, metrics, gradient = strategy.run(step)
            return (
                strategy.gather(weights, axis=0),
                strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, metrics["ess_fraction"], None
                ),
                strategy.reduce(tf.distribute.ReduceOp.SUM, gradient, None),
            )

        result = [value.numpy() for value in update()]
        assert all(numpy.isfinite(value).all() for value in result)
        numpy.testing.assert_array_equal(result[0][2:], 0.0)
        numpy.testing.assert_allclose(result[0][:2].mean(), 1.0, atol=1e-6)
        assert result[0][0] > result[0][1]
        assert result[2] > 0
        if replicas == 1:
            expected = result
        else:
            for left, right in zip(expected, result):
                numpy.testing.assert_allclose(left, right, atol=1e-6)


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

    check_imitation_weights_with_empty_replica()
