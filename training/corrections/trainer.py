"""Magnetic mirror descent at every sampled rollout decision."""

import tensorflow as tf

from model import actions
from model import actor as actor_layers
from model import critic, losses
from training import trainer


def mmd_terms(*, features, current, mirror, magnet, values, count):
    """Expected action value with proximal and fixed-magnet KL penalties."""
    values = tf.stop_gradient(values)
    mirror = tf.stop_gradient(mirror)
    magnet = tf.stop_gradient(magnet)
    probabilities = [
        critic.candidate_probabilities(
            features,
            {
                name: distribution[:, : actions.ACTOR_OUTPUTS[name][0]]
                for name in actions.TRAIN_POLICY_HEADS
            },
        )
        for distribution in (current, mirror)
    ]
    valid = tf.range(tf.shape(current)[0]) < count
    metrics = {
        "policy_gain_objective": losses.mean_valid(
            tf.reduce_sum(
                (probabilities[0] - probabilities[1]) * values, axis=-1
            ),
            valid,
        ),
        **{
            name: losses.mean_valid(
                tf.reduce_sum(tf.exp(current) * (current - reference), axis=-1),
                valid,
            )
            for name, reference in (
                ("mirror_kl", mirror),
                ("magnet_kl", magnet),
            )
        },
        "entropy": losses.mean_valid(
            -tf.reduce_sum(tf.exp(current) * current, axis=-1), valid
        ),
    }
    metrics["mmd_loss"] = (
        -metrics["policy_gain_objective"]
        + 0.3 * metrics["mirror_kl"]
        + 0.3 * metrics["magnet_kl"]
    )
    return metrics


class MMDTrainer:
    """One MMD objective for discovery, forced roots and descendants."""

    def __init__(self, *, network, mirror, magnet, learning_rate):
        self.network = network
        self.mirror = mirror
        self.magnet = magnet
        self.optimizer = trainer.AdamW(
            learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0
        )
        self.optimizer.build(network.actor.trainable_variables)

    @tf.function(reduce_retracing=True)
    def update(self, batch, count):
        mirror = actor_layers.distribution(
            self.mirror(batch["features"], training=False),
            batch["head"],
            batch["legal"],
        )
        magnet = actor_layers.distribution(
            self.magnet(**batch["features"]), batch["head"], batch["legal"]
        )
        with tf.GradientTape() as tape:
            outputs, state = self.network.candidate_context(
                batch["features"],
                self.network.actor(batch["features"], training=False),
            )
            selected = critic.active_candidates(
                batch["features"], outputs["selected"]
            ) & (outputs["selected"][:, 0] < count)
            values = trainer.action_values(
                network=self.network,
                features=batch["features"],
                selected=tf.boolean_mask(outputs["selected"], selected),
                state=tf.boolean_mask(state, selected),
            )
            metrics = mmd_terms(
                features=batch["features"],
                current=actor_layers.distribution(
                    outputs, batch["head"], batch["legal"]
                ),
                mirror=mirror,
                magnet=magnet,
                values=values,
                count=count,
            )
            metrics["constraint_loss"] = losses.mean_valid(
                outputs["constraint_loss"],
                tf.range(tf.shape(batch["head"])[0]) < count,
            )
            loss = metrics["mmd_loss"] + 0.01 * metrics["constraint_loss"]
            scaled = loss / tf.distribute.get_strategy().num_replicas_in_sync
        return {
            **metrics,
            **trainer.apply_gradients(
                self.optimizer,
                tape,
                scaled,
                self.network.actor.trainable_variables,
            ),
            "joint_loss": loss,
            "learning_rate": tf.identity(self.optimizer.learning_rate),
        }
