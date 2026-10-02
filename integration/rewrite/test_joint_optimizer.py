"""Joint training preserves AdamW math for dense and sparse parameters."""

import numpy
import tensorflow as tf

from model import critic_full, objectives, train


def test_fused_adam_matches_clipped_dense_and_duplicate_sparse_updates():
    tf.keras.utils.set_random_seed(941)
    initial = tf.random.normal((37, 64))
    variables = [[tf.Variable(initial), tf.Variable(initial)] for _ in range(2)]
    optimizers = [
        optimizer(learning_rate=1e-4, weight_decay=1e-4, clipnorm=1.0)
        for optimizer in (tf.keras.optimizers.AdamW, critic_full.FusedAdamW)
    ]
    for _ in range(100):
        gradients = [
            tf.random.normal((37, 64)),
            tf.IndexedSlices(
                tf.random.normal((6, 64)), [1, 1, 2, 4, 4, 6], [37, 64]
            ),
        ]
        for optimizer, values in zip(optimizers, variables):
            optimizer.apply_gradients(zip(gradients, values))
    for actual, expected in zip(
        [*variables[1], *optimizers[1].variables],
        [*variables[0], *optimizers[0].variables],
    ):
        numpy.testing.assert_allclose(
            actual.numpy(), expected.numpy(), atol=3e-7, rtol=2e-5
        )


def test_full_schedule_halves_production_exposure_with_focus_proportions():
    phases = critic_full.phase_schedule(
        "full",
        {
            "partition_counts": [[51200, 25600]] * 7 + [[999999, 999999]] * 3,
        },
    )
    assert phases == (
        ("natural_start", "natural", 22285),
        ("focused", "focused", 89142),
        ("natural_finish", "natural", 22285),
    )
    total = sum(steps for _, _, steps in phases)
    assert (
        total * critic_full.BATCH_SIZE == critic_full.PRODUCTION_EXPOSURES // 2
    )
    assert (
        sum(
            steps
            for _, _, steps in critic_full.phase_schedule(
                "full", {"partition_counts": [[51200, 25600]] * 10}, 10240
            )
        )
        * 10240
        == critic_full.PRODUCTION_EXPOSURES // 2
    )
    schedule = train.ReferenceSchedule(total, critic_full.BATCH_SIZE)
    numpy.testing.assert_allclose(
        [float(schedule(step)) for step in (0, total // 2, total)],
        [2e-5, 2e-4, 2e-5],
        rtol=1e-6,
    )
    numpy.testing.assert_allclose(
        objectives.payoff_weight(
            tf.constant([0, 0.1, 0.35, 0.6, 1.0]) * total, total
        ),
        [0, 0, 1, 2, 2],
        atol=1e-6,
    )


def test_specialist_routing_keeps_global_auxiliary_gradient():
    layer = tf.keras.layers.Dense(
        1, use_bias=False, name="attack_decisions", kernel_initializer="ones"
    )
    with tf.GradientTape(persistent=True) as tape:
        prediction = layer(tf.constant([[1.0], [3.0]]))[:, 0]
        imitation = tf.reduce_mean(prediction)
        auxiliary = tf.reduce_sum(prediction * 2)
        guidance = 10 * tf.reduce_mean(prediction)
        loss = imitation + auxiliary + guidance
        routed = {"attack": loss - guidance - imitation + prediction[0]}
    gradient = objectives.specialist_gradients(
        tape, loss, routed, layer.trainable_variables
    )
    numpy.testing.assert_allclose(gradient[0], [[9.0]])
    del tape
