"""Opponent counts obey public hand sizes and physical tile limits."""

import numpy
import tensorflow as tf

from model import beliefs


def test_count_projection_respects_melds_reds_and_shared_ankou_blockers():
    features = {
        "meld_tile_ids": numpy.zeros((1, 4, 4, 4), numpy.int8),
        "meld_tile_red": numpy.zeros((1, 4, 4, 4), numpy.int8),
        "meld_tile_valid": numpy.zeros((1, 4, 4, 4), numpy.int8),
        "meld_valid": numpy.zeros((1, 4, 4), numpy.int8),
        "known_unavailable_counts": numpy.zeros((1, 34), numpy.int8),
        "candidate_hand_red": numpy.zeros((1, 96, 3), numpy.int8),
        "public_visible_red_counts": numpy.array([[1, 0, 0]], numpy.int8),
        "seat_wind": numpy.array([[0, 1, 2, 3]], numpy.int8),
        "prevailing_wind": numpy.array([0], numpy.int8),
        "dora_multiplicity": numpy.eye(34, dtype=numpy.int8)[[0]],
    }
    # Our 111m and ten other tiles; opponent 1 has melded 550m.
    features["known_unavailable_counts"][0, [0, 4]] = 3
    features["known_unavailable_counts"][0, 18:28] = 1
    features["meld_tile_ids"][0, 1, 0, :3] = 4
    features["meld_tile_red"][0, 1, 0, 2] = 1
    features["meld_tile_valid"][0, 1, 0, :3] = 1
    features["meld_valid"][0, 1, 0] = 1
    features = {key: tf.constant(value) for key, value in features.items()}
    counts = numpy.zeros((1, 3, 37), numpy.int32)
    counts[0, :, [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]] = 1
    counts[0, 1:, [15, 16, 17]] = 1
    counts[0, 0, [4, 34]] = [2, 1]
    logits = tf.Variable(numpy.zeros((1, 3, 37, 5), numpy.float32))
    lower, available, size = beliefs.held_count_context(features)
    # Projection must resolve the shared last copy while preserving the
    # exposed red meld and each concealed hand size. Gradients stay usable.
    with tf.GradientTape() as tape:
        projected = beliefs.CountProjection()(
            [beliefs.HeldCountMask()([logits, features]), features]
        )
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=counts, logits=projected
            )
        )
    gradient = tape.gradient(loss, logits).numpy()
    assert numpy.isfinite(gradient).all()
    assert numpy.any(gradient != 0)
    probability = tf.nn.softmax(projected, -1).numpy()
    concealed = (probability * numpy.arange(5)).sum(-1) - lower.numpy()
    numpy.testing.assert_allclose(concealed.sum(-1), size.numpy(), atol=1e-5)
    assert numpy.all(concealed.sum(1) <= available.numpy() + 1e-5)
    numpy.testing.assert_array_equal(probability[0, :, 0, 2:], 0)
    assert probability[0, :, 0, 1].sum() <= 1 + 1e-5
    numpy.testing.assert_array_equal(probability[0, 0, 34], [0, 1, 0, 0, 0])
    numpy.testing.assert_allclose(
        tf.nn.softmax(beliefs.CountProjection()([projected, features]), -1),
        probability,
        atol=1e-5,
    )

    # Check the constrained derivative against the forward solve, independent
    # of the backward implementation and its handling of active copy limits.
    direction = tf.random.stateless_normal(logits.shape, [941, 3])
    difference = (
        sum(
            sign
            * tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=counts,
                    logits=beliefs.CountProjection()(
                        [
                            beliefs.HeldCountMask()(
                                [logits + sign * 0.01 * direction, features]
                            ),
                            features,
                        ]
                    ),
                )
            )
            for sign in (-1, 1)
        )
        / 0.02
    )
    numpy.testing.assert_allclose(
        tf.reduce_sum(gradient * direction), difference, atol=2e-3, rtol=2e-3
    )

    # Saturated count heads previously exhausted the solver's 80 steps and
    # returned physically inconsistent means. Keep their gradients finite.
    logits.assign(tf.random.stateless_normal(logits.shape, [941, 2]) * 32)
    with tf.GradientTape() as tape:
        projected = beliefs.CountProjection()(
            [beliefs.HeldCountMask()([logits, features]), features]
        )
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=counts, logits=projected
            )
        )
    gradient = tape.gradient(loss, logits).numpy()
    assert numpy.isfinite(gradient).all()
    assert numpy.any(gradient != 0)
    concealed = tf.reduce_sum(
        tf.nn.softmax(projected)
        * (tf.range(5, dtype=tf.float32) - lower[..., None]),
        -1,
    ).numpy()
    numpy.testing.assert_allclose(concealed.sum(-1), size, atol=1e-4)
    assert numpy.all(concealed.sum(1) <= available.numpy() + 1e-4)
