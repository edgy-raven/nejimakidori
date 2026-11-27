"""Supervise learned predictions on eligible decisions and known targets."""

import tensorflow as tf

from model import beliefs, candidate_predictions, losses

LABELS = (
    "opponent_shanten",
    "opponent_ukeire",
    "opponent_hand_counts",
    "round_completion_yaku",
    "round_completion_score",
    "candidate_yaku_affinity",
    "ron_legal",
)


def reduce_terms(*, features, labels, outputs, valid, terms):
    """Reduce row metrics and the seven supervised heads in one collective."""
    means = losses.MaskedMeans()

    def evaluate(reduce_mean):
        def mean(value, mask, weight=None):
            row_mask = tf.reshape(
                tf.cast(valid, tf.float32), [-1] + [1] * (mask.shape.rank - 1)
            )
            return reduce_mean(
                value, row_mask * tf.cast(mask, tf.float32), weight
            )

        metrics = {
            name: mean(value, mask) for name, (value, mask) in terms.items()
        }
        supervised, diagnostics = beliefs.belief_loss_terms(
            features, labels, outputs, mean=mean
        )
        supervised.update(beliefs.completion_loss_terms(labels, outputs, mean))
        supervised.update(
            candidate_predictions.loss_terms(features, labels, outputs, mean)
        )
        return metrics, supervised, diagnostics

    evaluate(means.collect)
    means.reduce()
    return evaluate(means.read)
