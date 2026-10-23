"""Masked loss reductions shared across components and replicas."""

import tensorflow as tf


def masked_totals(loss, mask, weight=None):
    mask = tf.cast(mask, tf.float32)
    if weight is not None:
        mask *= tf.cast(weight, tf.float32)
    return tf.stack(
        [
            tf.reduce_sum(tf.cast(loss, tf.float32) * mask),
            tf.reduce_sum(mask),
        ]
    )


def mean_valid(loss, mask, weight=None):
    numerator, denominator = tf.unstack(masked_totals(loss, mask, weight))
    replica = tf.distribute.get_replica_context()
    if replica is not None:
        numerator = replica.all_reduce(tf.distribute.ReduceOp.SUM, numerator)
        denominator = replica.all_reduce(
            tf.distribute.ReduceOp.SUM, denominator
        )
    return tf.math.divide_no_nan(numerator, denominator)


def candidate_mean(loss, mask, rows, mean=mean_valid):
    """Equal decision mass over valid candidate/head entries.

    Rows are the sorted decision IDs returned by DecisionCandidates. Unknown
    candidates contribute neither loss nor normalization mass.
    """
    mask = tf.cast(mask, tf.float32)
    totals = tf.math.segment_sum(
        tf.stack(
            [
                tf.reduce_sum(tf.cast(loss, tf.float32) * mask, axis=-1),
                tf.reduce_sum(mask, axis=-1),
            ],
            axis=-1,
        ),
        rows,
    )
    return mean(
        tf.math.divide_no_nan(totals[:, 0], totals[:, 1]), totals[:, 1] > 0
    )


def packed_means(statistics):
    """Reduce masked statistics using TensorFlow's distributed autodiff."""
    replica = tf.distribute.get_replica_context()
    if replica is not None:
        statistics = replica.all_reduce(tf.distribute.ReduceOp.SUM, statistics)
    return tf.math.divide_no_nan(statistics[:, 0], statistics[:, 1])


class MaskedMeans:
    """Trace independent masked statistics, then assemble globally shared loss.

    Collection results must not feed another mean's loss, mask or weight.
    Both passes build loss arithmetic only; the actor forward runs once.
    The unused second-pass arithmetic is pruned from the TensorFlow graph.
    """

    def __init__(self):
        self.statistics = []

    def collect(self, loss, mask, weight=None):
        self.statistics.append(masked_totals(loss, mask, weight))
        return tf.constant(0.0)

    def reduce(self):
        self.values = iter(tf.unstack(packed_means(tf.stack(self.statistics))))

    def read(self, loss, mask, weight=None):
        return next(self.values)


def classification(labels, logits, weight=None, mean=mean_valid):
    labels = tf.cast(labels, tf.int32)
    return mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.maximum(labels, 0), logits=tf.cast(logits, tf.float32)
        ),
        labels >= 0,
        weight,
    )
