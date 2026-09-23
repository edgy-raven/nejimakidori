"""Ukeire targets become less regularized as the live wall shrinks."""

import numpy
import tensorflow as tf

from model import objectives


def test_ukeire_progression_preserves_masks_metrics_and_tenpai_emphasis():
    logits = tf.Variable(tf.zeros((1, 3, 4, 34)))
    outputs = {
        "opponent_ukeire_by_shanten_logits": logits,
        "opponent_wait_logits": tf.zeros((1, 3, 34)),
        "opponent_ukeire_logits": tf.zeros((1, 3, 34)),
    }
    gradients = {}
    for wall in (70, 35, 0):
        for label in (0, 1):
            labels = {
                "opponent_shanten": tf.constant([[0, 1, -1]]),
                "opponent_ukeire": tf.constant(
                    [[[label] * 34, [label] * 34, [-1] * 34]]
                ),
            }
            with tf.GradientTape() as tape:
                loss, metrics = objectives.ukeire_loss(
                    labels, outputs, {"live_wall_count": tf.constant([wall])}
                )
            gradients[wall, label] = tape.gradient(loss, logits).numpy()
            numpy.testing.assert_allclose(loss, numpy.log(2), rtol=1e-6)
            # Calibration metrics still use actual labels, not soft targets.
            numpy.testing.assert_allclose(metrics["tenpai_wait_brier"], 0.25)
            numpy.testing.assert_array_equal(gradients[wall, label][:, 2], 0)
            numpy.testing.assert_array_equal(
                gradients[wall, label][:, 0, 1:], 0
            )
    for wall, retained in ((70, 0.5), (35, 0.75), (0, 1.0)):
        delta = gradients[wall, 1] - gradients[wall, 0]
        numpy.testing.assert_allclose(
            delta, retained * (gradients[0, 1] - gradients[0, 0]), atol=1e-7
        )
        numpy.testing.assert_allclose(
            delta[0, 0, 0], 9 * delta[0, 1, 1], atol=1e-7
        )
    # Broad prior respects suit/reflection symmetry, not uniform tile odds.
    early = gradients[70, 0][0, 0, 0]
    numpy.testing.assert_allclose(early[:9], early[9:18])
    numpy.testing.assert_allclose(early[:9], early[:9][::-1])
    assert early[4] < early[0] < early[27]
