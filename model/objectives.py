"""Masked valid-label means and probability-mass regret objectives."""

import tensorflow as tf

from model import feature_vector

TENPAI_UKEIRE_WEIGHT = 9.0
POLICY_ENTROPY_WEIGHT = 0.01


def payoff_weight(step, total_steps):
    return 2.0 * tf.clip_by_value(
        (tf.cast(step, tf.float32) / total_steps - 0.1) / 0.5, 0.0, 1.0
    )


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


@tf.custom_gradient
def packed_means(statistics):
    """Reduce statistics for an identical scalar objective on every replica.

    Its upstream derivatives are identical, so their backward all-reduce
    equals multiplication by the replica count. The trainer divides its
    objective by that count before the optimizer sums parameter gradients.
    Replica-dependent coefficients after this reduction are unsupported.
    """
    replica = tf.distribute.get_replica_context()
    replicas = replica.num_replicas_in_sync if replica is not None else 1
    totals = (
        replica.all_reduce(tf.distribute.ReduceOp.SUM, statistics)
        if replicas > 1
        else statistics
    )
    means = tf.math.divide_no_nan(totals[:, 0], totals[:, 1])

    def gradient(upstream):
        numerator = tf.math.divide_no_nan(upstream * replicas, totals[:, 1])
        return tf.stack([numerator, -numerator * means], axis=-1)

    return means, gradient


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


def placement_prior(scores, remaining, scale=0.0472):
    """Coherent actor rank probabilities from a Gumbel future-score prior.

    Scores and scale must share units; leading batch dimensions broadcast.
    Enumerate only opponent orders, avoiding the 24-permutation score tensor.
    """
    remaining = tf.clip_by_value(tf.cast(remaining, tf.float32), 0.0, 7.0)
    scores = tf.cast(scores, tf.float32)
    location = (scores - scores[..., :1]) / (
        scale * tf.sqrt(remaining[..., None] + 1)
    )
    total = tf.reduce_logsumexp(location, axis=-1)
    first = tf.exp(-total)
    second = tf.zeros_like(first)
    fourth = tf.zeros_like(first)
    for seat in (1, 2, 3):
        after_first = tf.reduce_logsumexp(
            tf.gather(location, [i for i in range(4) if i != seat], axis=-1),
            axis=-1,
        )
        second += tf.exp(location[..., seat] - total - after_first)
        for other in (i for i in (1, 2, 3) if i != seat):
            last = 6 - seat - other
            fourth += tf.exp(
                tf.reduce_sum(location[..., 1:], axis=-1)
                - total
                - after_first
                - tf.nn.softplus(location[..., last])
            )
    probability = tf.stack(
        [first, second, tf.maximum(1 - first - second - fourth, 0), fourth],
        axis=-1,
    )
    return probability / tf.reduce_sum(probability, axis=-1, keepdims=True)


def signed_log(value):
    """Compress signed values; payment callers use units of 10,000 points."""
    value = tf.cast(value, tf.float32)
    # Keep the derivative equal to one at zero for predicted values.
    return tf.where(
        value >= 0,
        tf.math.log1p(tf.maximum(value, 0)),
        -tf.math.log1p(tf.maximum(-value, 0)),
    )


def policy_nll(labels, logits):
    losses = tf.add_n(
        [
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.maximum(tf.cast(labels[name], tf.int32), 0),
                logits=tf.cast(logits[name], tf.float32),
            )
            * tf.cast(labels[name] >= 0, tf.float32)
            for name in feature_vector.TRAIN_POLICY_HEADS
        ]
    )
    return losses, tf.reduce_any(
        tf.stack(
            [labels[name] >= 0 for name in feature_vector.TRAIN_POLICY_HEADS]
        ),
        axis=0,
    )


def policy_loss(labels, logits, mean=mean_valid):
    return mean(*policy_nll(labels, logits))


def awr_policy_terms(nll, advantages, eligible, mean=mean_valid):
    """Detached, capped positive regression weights on recorded actions."""
    log_weights = tf.minimum(tf.cast(advantages, tf.float32), tf.math.log(20.0))
    offset = tf.reduce_max(tf.where(eligible, log_weights, -1e30))
    replica = tf.distribute.get_replica_context()
    if replica is not None and replica.num_replicas_in_sync > 1:
        offset = tf.reduce_max(replica.all_gather(offset[None], axis=0))
    # A shared scale cancels from weighted CE and ESS. It prevents tiny
    # negative-advantage weights from underflowing, including their squares.
    weights = tf.stop_gradient(
        tf.exp(tf.where(eligible, log_weights - offset, -1e30))
    )
    weight_mean = mean(weights, eligible)
    return {
        "loss": mean(nll, eligible, weights),
        "weight_mean": weight_mean * tf.exp(offset),
        "weight_clipped_fraction": mean(
            tf.cast(
                advantages >= tf.math.log(20.0),
                tf.float32,
            ),
            eligible,
        ),
        "effective_sample_fraction": tf.math.divide_no_nan(
            tf.square(weight_mean), mean(tf.square(weights), eligible)
        ),
    }


def policy_entropy(labels, logits, mean=mean_valid):
    # Each active head is a complete legal-action distribution.
    entropies = []
    for name in feature_vector.TRAIN_POLICY_HEADS:
        log_probability = tf.nn.log_softmax(
            tf.cast(logits[name], tf.float32), axis=-1
        )
        entropy = -tf.reduce_sum(
            tf.exp(log_probability) * log_probability, axis=-1
        )
        entropies.append(tf.where(labels[name] >= 0, entropy, 0.0))
    return mean(
        tf.add_n(entropies),
        tf.reduce_any(
            tf.stack(
                [
                    labels[name] >= 0
                    for name in feature_vector.TRAIN_POLICY_HEADS
                ]
            ),
            axis=0,
        ),
    )


def ukeire_loss(labels, outputs, mean=mean_valid):
    target = tf.cast(labels["opponent_ukeire"], tf.float32)
    distance = tf.cast(labels["opponent_shanten"], tf.int32)
    logits = tf.gather(
        outputs["opponent_ukeire_by_shanten_logits"],
        tf.maximum(distance, 0),
        axis=2,
        batch_dims=2,
    )
    valid = (target >= 0) & (distance[..., None] >= 0)
    tenpai = valid & (distance[..., None] == 0)
    other = valid & (distance[..., None] > 0)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.maximum(target, 0), logits=logits
    )
    tenpai_loss = mean(losses, tenpai)
    other_loss = mean(losses, other)
    loss = tf.math.divide_no_nan(
        TENPAI_UKEIRE_WEIGHT * tenpai_loss + other_loss,
        TENPAI_UKEIRE_WEIGHT * mean(tf.ones_like(target), tenpai)
        + mean(tf.ones_like(target), other),
    )
    probability = tf.math.sigmoid(logits)
    metrics = {
        "tenpai_ukeire_loss": tenpai_loss,
        "other_ukeire_loss": other_loss,
        "tenpai_wait_recall_at_0_5": mean(
            tf.cast(probability >= 0.5, tf.float32), tenpai & (target == 1)
        ),
        "tenpai_wait_precision_at_0_5": mean(
            target, tenpai & (probability >= 0.5)
        ),
        "tenpai_wait_brier": mean(tf.square(probability - target), tenpai),
        "tenpai_wait_positive_brier": mean(
            tf.square(probability - target), tenpai & (target == 1)
        ),
        "tenpai_wait_recall_at_5": mean(
            tf.reduce_sum(
                tf.one_hot(tf.math.top_k(probability, 5).indices, 34), -2
            ),
            tenpai & (target == 1),
        ),
        "joint_wait_brier": mean(
            tf.square(
                tf.math.sigmoid(outputs["opponent_wait_logits"])
                - tf.cast(tenpai & (target == 1), tf.float32)
            ),
            valid,
        ),
        "opponent_ukeire_brier": mean(
            tf.square(
                tf.math.sigmoid(outputs["opponent_ukeire_logits"]) - target
            ),
            valid,
        ),
    }
    return loss, metrics
