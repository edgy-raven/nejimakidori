"""Joint training uses dense AdamW updates and rejects nonfinite gradients."""

import numpy
import pytest
import tensorflow as tf

from training import trainer


def test_repeated_tiles_match_dense_adamw_updates():
    table = tf.Variable(tf.reshape(tf.range(544, dtype=tf.float32), (34, 16)))
    reference = tf.Variable(table)
    optimizer = trainer.AdamW(
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
        metrics = trainer.apply_gradients(optimizer, tape, loss, [table])
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(
                tf.matmul(tf.one_hot(tiles, 34), reference)
                * tf.constant(weights, tf.float32)[:, None]
            )
        gradient = tape.gradient(loss, reference)
        numpy.testing.assert_allclose(
            metrics["gradient/global_norm"], tf.linalg.norm(gradient)
        )
        reference_optimizer.apply_gradients([(gradient, reference)])
        for actual, expected in zip(
            [table, *optimizer.variables],
            [reference, *reference_optimizer.variables],
        ):
            numpy.testing.assert_allclose(
                actual, expected, rtol=1e-6, atol=1e-7
            )


def test_nonfinite_gradients_fail_before_optimizer_mutation():
    value = tf.Variable(1.0)
    optimizer = trainer.AdamW(0.01, clipnorm=1.0)
    optimizer.build([value])

    @tf.function
    def update():
        with tf.GradientTape() as tape:
            loss = value * tf.constant(float("nan"))
        return trainer.apply_gradients(optimizer, tape, loss, [value])

    with pytest.raises(
        tf.errors.InvalidArgumentError, match="Non-finite global gradient norm"
    ):
        update()
    assert float(value) == 1.0
    assert int(optimizer.iterations) == 0
