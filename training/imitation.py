"""Q-weighted expert imitation with settlement fitting."""

import tensorflow as tf

from model import actor as actor_layers
from model import critic, joint_settlement, losses
from training import auxiliary, restoration, trainer

ESS_TARGET = 0.75


def effective_fraction(gaps, beta):
    log_weights = tf.minimum(gaps / beta, tf.math.log(20.0))
    weights = tf.exp(log_weights - tf.reduce_max(log_weights, axis=0))
    return tf.square(tf.reduce_mean(weights, axis=0)) / tf.reduce_mean(
        tf.square(weights), axis=0
    )


def calibrate_beta(gaps):
    """Choose the least aggressive crossing of 75% ESS within J bounds."""
    grid = tf.exp(tf.linspace(tf.math.log(0.0001), tf.math.log(1.28), 65))
    curve = effective_fraction(gaps[:, None], grid)
    crossings = tf.where(
        (curve[:-1] - ESS_TARGET) * (curve[1:] - ESS_TARGET) <= 0
    )[:, 0]

    def crossing():
        index = crossings[-1]
        lower, upper = grid[index], grid[index + 1]
        lower_value = curve[index] - ESS_TARGET
        for _ in tf.range(20):
            middle = tf.sqrt(lower * upper)
            same_side = (
                effective_fraction(gaps, middle) - ESS_TARGET
            ) * lower_value > 0
            lower = tf.where(same_side, middle, lower)
            upper = tf.where(same_side, upper, middle)
        return upper

    closest = 64 - tf.argmin(
        tf.reverse(tf.abs(curve - ESS_TARGET), [0]), output_type=tf.int32
    )
    beta = tf.cond(tf.size(crossings) > 0, crossing, lambda: grid[closest])
    return beta, effective_fraction(gaps, beta), tf.reduce_min(curve)


def imitation_weights(gaps, valid):
    """Global ESS and mean-one weights; padded replicas carry no mass."""
    gaps = tf.debugging.check_numerics(gaps, "Non-finite Q gap")
    gathered = tf.distribute.get_replica_context().all_gather(
        tf.stack([tf.stop_gradient(gaps), tf.cast(valid, tf.float32)], axis=1),
        axis=0,
    )
    global_gaps = tf.boolean_mask(gathered[:, 0], gathered[:, 1] > 0)
    beta, ess, minimum = calibrate_beta(global_gaps)
    global_logs = tf.minimum(global_gaps / beta, tf.math.log(20.0))
    offset = tf.reduce_max(global_logs)
    weights = tf.exp(
        tf.where(
            valid, tf.minimum(gaps / beta, tf.math.log(20.0)) - offset, 0.0
        )
    )
    weights /= tf.reduce_mean(tf.exp(global_logs - offset))
    return tf.stop_gradient(tf.where(valid, weights, 0.0)), {
        "beta": beta,
        "ess_fraction": ess,
        "minimum_ess_fraction": minimum,
        "ess_target_unattainable": tf.cast(
            tf.abs(ess - ESS_TARGET) > 0.001, tf.float32
        ),
        "q_gap_mean": tf.reduce_mean(global_gaps),
        "q_gap_stddev": tf.math.reduce_std(global_gaps),
        "q_gap_positive_fraction": tf.reduce_mean(
            tf.cast(global_gaps > 0, tf.float32)
        ),
    }


class ImitationTrainer(trainer.CriticTrainer):
    """Fit policy, observed settlements and beliefs with one stage optimizer."""

    @tf.function(reduce_retracing=True)
    def update(self, batch, valid):
        features = batch["features"]
        supported = restoration.policy_target_supported(
            features, batch["head"], batch["expert"]
        )
        with tf.GradientTape() as tape:
            # One live encoding supplies policy, settlement supervision and Q.
            outputs = self.network.actor(features, training=False)
            current = actor_layers.distribution(
                outputs, batch["head"], batch["legal"]
            )
            state = self.network.candidate_context(features, outputs)
            observed = self.network.chosen_inputs(
                features, outputs, state, batch["chosen"]
            )
            statistics = self.network.critic.supervised_statistics(
                observed,
                joint_settlement.payment_labels(
                    batch["payment_points"], observed
                ),
                valid=valid,
            )
            critic_metrics = dict(
                zip(
                    critic.CRITIC_METRICS,
                    tf.unstack(losses.packed_means(statistics)),
                )
            )
            with tape.stop_recording():
                selected = outputs["selected"]
                relevant = (
                    (
                        selected[:, 1]
                        == tf.cast(
                            tf.gather(batch["chosen"], selected[:, 0]),
                            selected.dtype,
                        )
                    )
                    | (
                        selected[:, 1]
                        == tf.cast(
                            tf.gather(
                                batch["reference_action"], selected[:, 0]
                            ),
                            selected.dtype,
                        )
                    )
                ) & tf.gather(valid, selected[:, 0])
                # Identical actions have an exactly zero Q gap. Avoid
                # evaluating their settlement distributions altogether.
                relevant &= tf.gather(
                    batch["chosen"] != batch["reference_action"],
                    selected[:, 0],
                )
                values = trainer.action_values(
                    network=self.network,
                    features=features,
                    selected=tf.boolean_mask(selected, relevant),
                    state=tf.boolean_mask(state, relevant),
                )
                gaps = tf.gather(
                    values, batch["chosen"], batch_dims=1
                ) - tf.gather(values, batch["reference_action"], batch_dims=1)
                weights, weighting_metrics = imitation_weights(gaps, valid)
            cross_entropy = -tf.gather(current, batch["expert"], batch_dims=1)
            terms = {
                "weighted_ce": (cross_entropy * weights, supported),
                "unweighted_ce": (cross_entropy, supported),
                "policy_target_coverage": (
                    tf.cast(supported, tf.float32),
                    tf.ones_like(valid),
                ),
                "entropy": (
                    -tf.reduce_sum(tf.exp(current) * current, axis=-1),
                    tf.ones_like(valid),
                ),
                "weight_mean": (weights, tf.ones_like(valid)),
            }
            metrics, supervised, diagnostics = auxiliary.reduce_terms(
                features=features,
                labels=batch["labels"],
                outputs=outputs,
                valid=valid,
                terms=terms,
            )
            critic_loss = critic.settlement_objective(critic_metrics)
            loss = (
                metrics["weighted_ce"]
                + critic_loss
                + tf.add_n(list(supervised.values()))
            )
            scaled = loss / tf.distribute.get_strategy().num_replicas_in_sync
        return {
            **metrics,
            **critic_metrics,
            "critic_loss": critic_loss,
            **supervised,
            **diagnostics,
            **weighting_metrics,
            **trainer.apply_gradients(
                self.optimizer,
                tape,
                scaled,
                self.network.trainable_variables,
            ),
            "joint_loss": loss,
            "learning_rate": tf.identity(self.optimizer.learning_rate),
        }
