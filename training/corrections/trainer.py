"""Train only selected roots, using both measured continuation outcomes."""

import tensorflow as tf

from model import actions
from model import actor as actor_layers
from model import critic, joint_settlement, losses
from training import trainer


def paired_terms(current, magnet, batch, count):
    """Risk-neutral current-hand improvement, conditional on the tested pair."""
    settlement = tf.reshape(batch["settlement"], (-1, 12))
    st3, probability = joint_settlement.post_hand_forecast(
        scores=tf.cast(settlement[:, :4], tf.float32),
        ranks=tf.cast(settlement[:, 4:8], tf.int64),
        next_dealer=tf.cast(settlement[:, 8], tf.int64),
        next_hand=tf.cast(settlement[:, 9], tf.int32),
        terminal=tf.cast(settlement[:, 10], tf.bool),
        pot=tf.cast(settlement[:, 11], tf.float32),
    )
    objective = tf.reshape(
        joint_settlement.policy_objective(st3, probability), (-1, 2)
    )
    gap = tf.stop_gradient(objective[:, 1] - objective[:, 0])
    pair = tf.nn.softmax(
        tf.gather(current, batch["actions"], batch_dims=1), axis=-1
    )
    valid = tf.range(tf.shape(current)[0]) < count
    return {
        "pair_loss": losses.mean_valid(-pair[:, 1] * gap, valid),
        "pair_gap_objective": losses.mean_valid(gap, valid),
        "pair_absolute_gap_objective": losses.mean_valid(tf.abs(gap), valid),
        "pair_alternative_probability": losses.mean_valid(pair[:, 1], valid),
        "magnet_kl": losses.mean_valid(
            tf.reduce_sum(
                tf.exp(current) * (current - tf.stop_gradient(magnet)), axis=-1
            ),
            valid,
        ),
    }


class CorrectionTrainer:
    """Paired outcome policy updates and payment labels, with no imitation.

    Each row is one root. ``actions``, ``payment_points`` and ``settlement``
    have a two-branch axis. Features are the raw model input dictionary.
    The full legal-head KL is applied once per root, not once per branch.
    """

    def __init__(self, *, network, magnet, learning_rate):
        self.network = network
        self.magnet = magnet
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0
        )
        self.optimizer.build(network.trainable_variables)

    @tf.function(reduce_retracing=True)
    def update(self, batch, count):
        magnet = actor_layers.distribution(
            self.magnet(**batch["features"]), batch["head"], batch["legal"]
        )
        with tf.GradientTape() as tape:
            outputs, state = self.network.candidate_context(
                batch["features"],
                self.network.actor(batch["features"], training=False),
            )
            metrics = paired_terms(
                actor_layers.distribution(
                    outputs, batch["head"], batch["legal"]
                ),
                magnet,
                batch,
                count,
            )
            observed = [
                self.network.chosen_inputs(
                    batch["features"],
                    outputs,
                    state,
                    critic.chosen_actions(
                        batch["features"],
                        {
                            name: tf.where(
                                batch["head"] == head,
                                batch["actions"][:, branch],
                                -1,
                            )
                            for head, name in enumerate(
                                actions.TRAIN_POLICY_HEADS
                            )
                        },
                    ),
                )
                for branch in range(2)
            ]
            observed = tf.nest.map_structure(
                lambda first, second: tf.concat((first, second), axis=0),
                *observed,
            )
            statistics = self.network.critic.supervised_statistics(
                observed,
                joint_settlement.payment_labels(
                    tf.concat(
                        (
                            batch["payment_points"][:, 0],
                            batch["payment_points"][:, 1],
                        ),
                        axis=0,
                    ),
                    observed,
                ),
                valid=tf.tile(
                    tf.range(tf.shape(batch["head"])[0]) < count, (2,)
                ),
            )
            metrics.update(
                zip(
                    critic.CRITIC_METRICS,
                    tf.unstack(losses.packed_means(statistics)),
                )
            )
            metrics["constraint_loss"] = losses.mean_valid(
                outputs["constraint_loss"],
                tf.range(tf.shape(batch["head"])[0]) < count,
            )
            loss = (
                metrics["pair_loss"]
                + 0.3 * metrics["magnet_kl"]
                + 0.1 * critic.settlement_objective(metrics)
                + 0.01 * metrics["constraint_loss"]
            )
            scaled = loss / tf.distribute.get_strategy().num_replicas_in_sync
        return {
            **metrics,
            **trainer.apply_gradients(
                self.optimizer, tape, scaled, self.network.trainable_variables
            ),
            "joint_loss": loss,
            "learning_rate": tf.identity(self.optimizer.learning_rate),
        }
