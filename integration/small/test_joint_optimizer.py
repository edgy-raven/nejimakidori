"""Joint training uses dense AdamW updates and detached AWR weights."""

import numpy
import pytest
import tensorflow as tf

from model import actor as actor_layers
from training import optimization


def test_repeated_tiles_match_dense_adamw_updates():
    table = tf.Variable(tf.reshape(tf.range(544, dtype=tf.float32), (34, 16)))
    reference = tf.Variable(table)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.01, weight_decay=0.1, clipnorm=1.0
    )
    reference_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.01, weight_decay=0.1, clipnorm=1.0
    )
    for tiles, weights in (([0, 0, 33], [1, 2, -1]), ([5, 5, 33], [2, -1, 3])):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(
                tf.gather(table, tiles)
                * tf.constant(weights, tf.float32)[:, None]
            )
        metrics = optimization.apply_gradients(optimizer, tape, loss, [table])
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(
                tf.matmul(tf.one_hot(tiles, 34), reference)
                * tf.constant(weights, tf.float32)[:, None]
            )
        gradient = tape.gradient(loss, reference)
        numpy.testing.assert_allclose(
            metrics["gradient/local_norm"], tf.linalg.norm(gradient)
        )
        reference_optimizer.apply_gradients([(gradient, reference)])
        for actual, expected in zip(
            [table, *optimizer.variables],
            [reference, *reference_optimizer.variables],
        ):
            numpy.testing.assert_allclose(
                actual, expected, rtol=1e-6, atol=1e-7
            )


def test_awr_ess_matches_effective_weights_and_survives_tiny_advantages():
    for values in (
        numpy.log([1.0, 2.0, 100.0, 1e10]),
        numpy.array([-1000.0, -1001.0, -1002.0, 0.0]),
    ):
        nll = tf.Variable([1.0, 2.0, 4.0, 100.0])
        advantages = tf.Variable(values, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            terms = actor_layers.awr_policy_terms(
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


def test_nonfinite_gradients_fail_before_optimizer_mutation():
    value = tf.Variable(1.0)
    optimizer = tf.keras.optimizers.AdamW(0.01, clipnorm=1.0)
    optimizer.build([value])

    @tf.function
    def update():
        with tf.GradientTape() as tape:
            loss = value * tf.constant(float("nan"))
        return optimization.apply_gradients(optimizer, tape, loss, [value])

    with pytest.raises(
        tf.errors.InvalidArgumentError, match="Non-finite local gradient norm"
    ):
        update()
    assert float(value) == 1.0
    assert int(optimizer.iterations) == 0
