"""Masked valid-label means and probability-mass regret objectives."""

import itertools

import tensorflow as tf

import feature_vector

from . import decision_layers, outcomes

TENPAI_UKEIRE_WEIGHT = 9.0
POLICY_ENTROPY_WEIGHT = 0.01
LOSS_GROUP_WEIGHTS = {
    "board": 0.5,
    "specialist_policy": 0.5,
    "specialist_supervision": 0.1,
    "specialist_baseline": 0.05,
    "outcome_critics": 0.1,
    "outcome_baselines": 0.1,
}
PLACEMENT_UTILITY = tuple(
    outcomes.dealership_utility(0, rank, 0) for rank in range(1, 5)
)


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


def placement_prior(scores, remaining):
    """Actor rank probabilities at 4,720 points per sqrt(remaining + 1)."""
    remaining = tf.clip_by_value(tf.cast(remaining, tf.float32), 0.0, 7.0)
    scores = tf.cast(scores, tf.float32)
    orders = tf.constant(list(itertools.permutations(range(4))))
    ordered = tf.gather(
        scores / (0.0472 * tf.sqrt(remaining[:, None] + 1)), orders, axis=1
    )
    probability = tf.exp(
        tf.add_n(
            [
                ordered[:, :, rank]
                - tf.reduce_logsumexp(ordered[:, :, rank:], axis=-1)
                for rank in range(3)
            ]
        )
    )
    return tf.linalg.matmul(probability, tf.cast(orders == 0, tf.float32))


def final_placement_targets(labels, features):
    """Shrink distant rank labels, retaining five percent at East 1."""
    remaining = tf.clip_by_value(
        tf.cast(features["scheduled_hands_remaining"], tf.float32), 0.0, 7.0
    )
    smoothing = 0.95 * remaining[:, None] / 7
    target = (1 - smoothing) * tf.one_hot(
        tf.cast(labels, tf.int32), 4
    ) + smoothing * placement_prior(
        features["seat_context"][:, :, 0], remaining
    )
    return tf.where(labels[:, None] >= 0, target, 0.0)


def dealership_placement_weight(features):
    """Expert placement share; also decodes legacy terminal ron labels."""
    future = tf.range(1, 8)[None, :]
    dealerships = tf.reduce_sum(
        tf.cast(
            (
                future
                <= tf.cast(
                    features["scheduled_hands_remaining"][:, None], tf.int32
                )
            )
            & (
                (
                    tf.cast(features["dealer_relative_seat"][:, None], tf.int32)
                    + future
                )
                % 4
                == 0
            ),
            tf.int32,
        ),
        axis=1,
    )
    return tf.where(dealerships >= 2, 0.0, tf.where(dealerships == 1, 0.5, 1.0))


def placement_value_prior(features):
    """Expected final-placement utility, independent of dealership advice."""
    return tf.stop_gradient(
        tf.reduce_sum(
            placement_prior(
                features["seat_context"][:, :, 0],
                features["scheduled_hands_remaining"],
            )
            * tf.constant(PLACEMENT_UTILITY),
            axis=1,
        )
    )


def ordered_distance(target, logits, values):
    """CDF squared error integrated over actual ordered outcome distances."""
    values = tf.cast(values, tf.float32)
    error = tf.cumsum(
        tf.nn.softmax(tf.cast(logits, tf.float32)) - target, axis=-1
    )[..., :-1]
    return tf.reduce_sum(tf.square(error) * (values[1:] - values[:-1]), -1)


def signed_log(value):
    """Compress signed values; payment callers use units of 10,000 points."""
    value = tf.cast(value, tf.float32)
    return tf.sign(value) * tf.math.log1p(tf.abs(value))


def payment_terms(target, logits):
    """Categorical amounts with signed-log CDF distances and occurrence loss."""
    nonzero = [i for i, points in enumerate(outcomes.PAYMENT_VALUES) if points]
    values = signed_log(tf.constant(outcomes.PAYMENT_VALUES) / 10000)
    conditional_target = tf.gather(target, nonzero, axis=-1)
    mass = tf.reduce_sum(conditional_target, -1)
    conditional_logits = tf.gather(logits, nonzero, axis=-1)
    return {
        "payment_cross_entropy": tf.nn.softmax_cross_entropy_with_logits(
            labels=target, logits=logits
        ),
        "payment_occurrence_loss": (
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=mass,
                logits=tf.reduce_logsumexp(conditional_logits, -1)
                - logits[..., 1],
            )
        ),
        "payment_distance_loss": ordered_distance(target, logits, values),
        "payment_conditional_distance_loss": mass
        * ordered_distance(
            tf.math.divide_no_nan(conditional_target, mass[..., None]),
            conditional_logits,
            tf.gather(values, nonzero),
        ),
    }


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


def expert_policy_nll(scores, legal, chosen):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=chosen,
        logits=tf.where(legal, tf.cast(scores, tf.float32), -1e4),
    )


def awr_policy_terms(nll, advantages, eligible, mean=mean_valid):
    """Detached, capped positive regression weights on recorded actions."""
    weights = tf.stop_gradient(
        tf.exp(
            tf.minimum(
                tf.cast(advantages, tf.float32),
                tf.math.log(20.0),
            )
        )
    )
    weight_mean = mean(weights, eligible)
    return {
        "loss": mean(nll, eligible, weights),
        "weight_mean": weight_mean,
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


def tenpai_quality_reward(reached, quality):
    """Joint reach-and-quality target for an unconditional state baseline."""
    return tf.where(
        reached == 0,
        0.0,
        tf.where(
            (reached > 0) & (quality >= 0),
            tf.math.divide_no_nan(quality, 1 + quality),
            -1.0,
        ),
    )


def expert_reward_terms(
    rewards, estimates, baseline, eligible, mean=mean_valid
):
    """Critic advantages; baselines regress only observed known components."""
    known = rewards >= 0
    error = tf.where(known[:, None, :], estimates - baseline[:, None, :], 0.0)
    advantages = {
        "attack": tf.stop_gradient(
            error[:, 0, 0] + error[:, 0, 1] + 0.1 * error[:, 0, 3]
        ),
        "defense": tf.stop_gradient(
            error[:, 1, 0] - error[:, 1, 2] - 0.1 * error[:, 1, 4]
        ),
    }
    return (
        advantages,
        mean(tf.square(rewards - baseline), known & eligible[:, None]),
    )


def categorical_entropy(logits):
    log_probability = tf.nn.log_softmax(tf.cast(logits, tf.float32), axis=-1)
    return -tf.reduce_sum(tf.exp(log_probability) * log_probability, axis=-1)


def policy_entropy(labels, logits, mean=mean_valid):
    # Each active head is a complete legal-action distribution.
    entropies = []
    for name in feature_vector.TRAIN_POLICY_HEADS:
        entropy = categorical_entropy(logits[name])
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


def offline_policy_terms(
    labels, logits, baseline, features, mean=mean_valid, advice_strength=0.25
):
    """Outcome critics: placement residual and signed-log hand-point utility.

    Dealership advice reweights demonstrated actions using point advantages;
    it never changes either critic's regression targets.
    """
    nll, decisions = policy_nll(labels, logits)
    placement_weight = dealership_placement_weight(features)
    prior = placement_value_prior(features)
    payments = tf.gather(
        tf.constant(outcomes.PAYMENT_VALUES, tf.float32),
        tf.maximum(tf.cast(labels["terminal_payment"], tf.int32), 0),
    )
    point_target = (
        tf.reduce_sum(payments[:, :, :, 0], axis=(1, 2))
        - tf.reduce_sum(payments[:, :, 0, :], axis=(1, 2))
        + tf.reduce_sum(payments[:, :, 0, 0], axis=1)
    ) / 10000
    targets = [
        tf.gather(
            tf.constant(PLACEMENT_UTILITY),
            tf.maximum(tf.cast(labels["final_placement"], tf.int32), 0),
        ),
        signed_log(point_target),
    ]
    valid_components = [
        decisions & (labels["final_placement"] >= 0),
        decisions & tf.reduce_all(labels["terminal_payment"] >= 0, (1, 2, 3)),
    ]
    ron_points = (
        -tf.reduce_sum(tf.cast(labels["ron_payments"], tf.float32), axis=1)
        / 10000
    )
    ron_known = tf.reduce_all(labels["ron_payments"] >= 0, axis=1) & (
        ron_points < 0
    )
    # The legacy utility encodes exact terminal placement only when its
    # placement share is nonzero. Never invent continuation rank labels.
    ron_rank_known = (labels["ron_return_valid"] > 0) & (
        placement_weight[:, None] > 0
    )
    ron_rank = tf.math.divide_no_nan(
        labels["ron_return"] - (1 - placement_weight[:, None]) * ron_points,
        placement_weight[:, None],
    )
    ron_points += (
        0.03
        * tf.cast(features["honba"][:, None], tf.float32)
        * tf.reduce_sum(tf.cast(labels["ron_payments"] > 0, tf.float32), axis=1)
    )
    draw = tf.where(
        tf.cast(features["current_draw_is_red"], tf.bool),
        34 + tf.cast(features["current_draw_tile_id"], tf.int32) // 9,
        tf.cast(features["current_draw_tile_id"], tf.int32),
    )
    codes = tf.concat(
        [
            tf.broadcast_to(tf.range(37)[None], (tf.shape(draw)[0], 37)),
            draw[:, None],
        ],
        axis=1,
    )
    masks = decision_layers.legal_choices(features)
    result = {}
    advantages = []
    for component, component_name in enumerate(("placement", "points")):
        q = logits["payoff_action_values"][:, :, component]
        predictions = {
            "discard_policy_logits": q[:, :38],
            "riichi_policy_logits": q[:, 38:114],
            "response_policy_logits": decision_layers.ResponseKan(
                dtype="float32"
            )([q, features]),
            "kan_action_logits": q[:, 263:298],
        }
        component_prior = prior if component == 0 else tf.zeros_like(prior)
        valid = valid_components[component]
        critic_losses, critic_rows, selected_values, counts = [], [], [], []
        for name in feature_vector.TRAIN_POLICY_HEADS:
            action = tf.maximum(tf.cast(labels[name], tf.int32), 0)
            chosen = tf.one_hot(
                action,
                tf.shape(logits[name])[-1],
                on_value=True,
                off_value=False,
            )
            active = labels[name] >= 0
            selected_values.append(
                tf.where(
                    active,
                    tf.gather(predictions[name], action, batch_dims=1),
                    0.0,
                )
            )
            known = chosen & (valid & active)[:, None]
            values = tf.broadcast_to(
                targets[component][:, None], tf.shape(logits[name])
            )
            if name in ("discard_policy_logits", "riichi_policy_logits"):
                tiles = (
                    tf.tile(codes, [1, 2])
                    if name == "riichi_policy_logits"
                    else codes
                )
                alternative = (
                    tf.gather(
                        ron_rank_known if component == 0 else ron_known,
                        tiles,
                        batch_dims=1,
                    )
                    & masks[name]
                    & active[:, None]
                    & ~chosen
                )
                values = tf.where(
                    alternative,
                    tf.gather(
                        ron_rank if component == 0 else signed_log(ron_points),
                        tiles,
                        batch_dims=1,
                    ),
                    values,
                )
                known |= alternative
                if component == 1:
                    abort = tf.gather(
                        labels["abort_points"], codes, axis=2, batch_dims=1
                    )
                    abort = (
                        abort[:, 0]
                        if name == "discard_policy_logits"
                        else tf.reshape(abort, (-1, 76))
                    )
                    abort_known = (
                        (abort != -1) & masks[name] & active[:, None] & ~chosen
                    )
                    values = tf.where(
                        abort_known,
                        signed_log(tf.cast(abort, tf.float32) / 10000),
                        values,
                    )
                    known |= abort_known
                    alternative |= abort_known
                counts.append(
                    tf.reduce_sum(tf.cast(alternative, tf.float32), axis=-1)
                )
            critic_losses.append(
                tf.math.divide_no_nan(
                    tf.reduce_sum(
                        tf.where(
                            known,
                            tf.square(
                                predictions[name]
                                - tf.stop_gradient(
                                    values - component_prior[:, None]
                                )
                            ),
                            0.0,
                        ),
                        axis=-1,
                    ),
                    tf.reduce_sum(tf.cast(known, tf.float32), axis=-1),
                )
            )
            critic_rows.append(tf.reduce_any(known, axis=-1))
        advantage = tf.stop_gradient(
            tf.where(
                valid, tf.add_n(selected_values) - baseline[:, component], 0.0
            )
        )
        advantages.append(advantage)
        result.update(
            {
                component_name
                + "_critic_loss": mean(
                    tf.add_n(critic_losses),
                    tf.reduce_any(tf.stack(critic_rows), axis=0),
                ),
                component_name
                + "_baseline_loss": mean(
                    tf.square(
                        targets[component]
                        - component_prior
                        - baseline[:, component]
                    ),
                    valid,
                ),
                component_name + "_advantage": mean(advantage, valid),
                component_name
                + "_known_alternatives": mean(tf.add_n(counts), decisions),
            }
        )
    awr = awr_policy_terms(nll, advantages[0], decisions, mean=mean)
    advice = awr_policy_terms(
        nll, (1 - placement_weight) * advantages[1], decisions, mean=mean
    )
    result.update(
        {
            "offline_payoff_loss": awr.pop("loss"),
            "dealership_advice_loss": advice.pop("loss"),
            "dealership_advice_strength": tf.cast(advice_strength, tf.float32),
            "payoff_critic_loss": (
                result["placement_critic_loss"] + result["points_critic_loss"]
            )
            / 2,
            "payoff_baseline_loss": (
                result["placement_baseline_loss"]
                + result["points_baseline_loss"]
            )
            / 2,
            **{"payoff_awr_" + name: value for name, value in awr.items()},
            **{"advice_awr_" + name: value for name, value in advice.items()},
        }
    )
    return result


def ukeire_loss(labels, outputs, features, mean=mean_valid):
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
    # Empirical structural-ukeire frequencies, conditioned on shanten.
    # Suits and reflected ranks share rates; winds/dragons remain distinct.
    prior = tf.gather(
        tf.constant(
            [
                [
                    0.03435,
                    0.05227,
                    0.066874,
                    0.076206,
                    0.081084,
                    0.011457,
                    0.015364,
                ],
                [
                    0.09745,
                    0.147443,
                    0.186987,
                    0.208891,
                    0.218084,
                    0.030926,
                    0.035477,
                ],
                [
                    0.183989,
                    0.26689,
                    0.334356,
                    0.343389,
                    0.351488,
                    0.066411,
                    0.079643,
                ],
                [
                    0.390602,
                    0.492233,
                    0.570452,
                    0.561589,
                    0.554813,
                    0.209887,
                    0.280638,
                ],
            ]
        ),
        tf.maximum(distance, 0),
    )
    prior = tf.gather(
        prior,
        [0, 1, 2, 3, 4, 3, 2, 1, 0] * 3 + [5] * 4 + [6] * 3,
        axis=-1,
    )
    smoothing = (
        0.5
        * tf.cast(features["live_wall_count"], tf.float32)[:, None, None]
        / 70
    )
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=(1 - smoothing) * tf.maximum(target, 0) + smoothing * prior,
        logits=logits,
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


def regret_score_terms(labels, logits):
    target = tf.cast(labels["discard_regret"], tf.float32)
    logits = tf.cast(logits, tf.float32)
    minimum = tf.reduce_min(tf.where(target >= 0, target, 1.0), axis=-1)
    valid = tf.reduce_any(target > minimum[:, None], axis=-1)
    probability = tf.nn.softmax(tf.where(target >= 0, logits, -1e9), axis=-1)
    return (
        tf.reduce_sum(
            probability * tf.maximum(target - minimum[:, None], 0), -1
        ),
        valid,
    )


def masked_losses(terms, mean=mean_valid):
    return tf.stack([mean(value, mask) for value, mask in terms])
