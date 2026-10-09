"""Joint training preserves AdamW math for dense and sparse parameters."""

import numpy
import tensorflow as tf

from model import critic_full, objectives


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


def test_full_schedule_halves_production_exposure_with_continuous_mixture():
    phases = critic_full.phase_schedule("full")
    assert phases == (("mixed", "mixed", 133712),)
    total = sum(steps for _, _, steps in phases)
    assert (
        total * critic_full.BATCH_SIZE == critic_full.PRODUCTION_EXPOSURES // 2
    )
    assert (
        sum(steps for _, _, steps in critic_full.phase_schedule("full", 10240))
        * 10240
        == critic_full.PRODUCTION_EXPOSURES // 2
    )
    schedule = critic_full.ReferenceSchedule(total, critic_full.BATCH_SIZE)
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


def test_awr_ess_matches_effective_weights_and_survives_tiny_advantages():
    for values in (
        numpy.log([1.0, 2.0, 100.0, 1e10]),
        numpy.array([-1000.0, -1001.0, -1002.0, 0.0]),
    ):
        nll = tf.Variable([1.0, 2.0, 4.0, 100.0])
        advantages = tf.Variable(values, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            terms = objectives.awr_policy_terms(
                nll, advantages, tf.constant([True, True, True, False])
            )
        weights = numpy.exp(
            numpy.minimum(values[:3], numpy.log(20))
            - numpy.minimum(values[:3], numpy.log(20)).max()
        )
        weights /= weights.sum()
        numpy.testing.assert_allclose(
            terms["loss"], weights @ [1, 2, 4], rtol=1e-6
        )
        numpy.testing.assert_allclose(
            terms["effective_sample_fraction"],
            1 / (3 * numpy.square(weights).sum()),
            rtol=1e-6,
        )
        numpy.testing.assert_allclose(
            tape.gradient(terms["loss"], nll), [*weights, 0], rtol=1e-6
        )
        assert tape.gradient(terms["loss"], advantages) is None
