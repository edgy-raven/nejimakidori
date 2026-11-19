"""Shared optimizer, detached action values and observed-payment fitting."""

import tensorflow as tf

from log_dataset import training_data
from model import critic, joint_settlement, losses


class AdamW(tf.keras.optimizers.AdamW):
    """Clip already-reduced gradients; do not sum them a second time."""

    def _all_reduce_sum_gradients(self, grads_and_vars):
        return grads_and_vars


def apply_gradients(optimizer, tape, loss, variables):
    # Sum repeated tile entries locally: each embedding has only 34 rows.
    # Clip and synchronize the actual table gradient, not every occurrence.
    pairs = [
        (tf.convert_to_tensor(gradient), variable)
        for gradient, variable in zip(tape.gradient(loss, variables), variables)
        if gradient is not None
    ]
    gradients = tf.distribute.get_replica_context().all_reduce(
        tf.distribute.ReduceOp.SUM, [gradient for gradient, _ in pairs]
    )
    norm = tf.debugging.check_numerics(
        tf.linalg.global_norm(gradients), "Non-finite global gradient norm"
    )
    with tf.control_dependencies([norm]):
        optimizer.apply_gradients(
            zip(gradients, [variable for _, variable in pairs])
        )
    return {"gradient/global_norm": norm}


def action_values(*, network, features, selected, state, chunk_size=256):
    """Detached settlement forecasts for the MMD actor's legal actions."""
    public = joint_settlement.public_inputs(features)
    inputs = {
        **{
            name: tf.gather(value, selected[:, 0])
            for name, value in public.items()
        },
        **critic.public_inputs(features, selected),
        "state": tf.stop_gradient(state),
    }
    positions, position_indices = tf.unique(selected[:, 0])
    payoffs = joint_settlement.gather_position_payoffs(
        {**public, "dealer": features["dealer_relative_seat"]},
        positions,
    )
    integer_fields = tuple(
        name for name, value in inputs.items() if value.dtype == tf.int32
    )
    inputs = {
        name: (
            tf.bitcast(value, tf.float32) if name in integer_fields else value
        )
        for name, value in inputs.items()
    }
    weights = [
        tf.stop_gradient(value.value.read_value())
        for value in network.critic.trainable_variables
    ]
    chunk_count = (tf.shape(selected)[0] + chunk_size - 1) // chunk_size
    values = tf.TensorArray(
        tf.float32, size=chunk_count, element_shape=[chunk_size]
    )
    for chunk in tf.range(chunk_count):
        indices = tf.minimum(
            chunk * chunk_size + tf.range(chunk_size),
            tf.shape(selected)[0] - 1,
        )
        values = values.write(
            chunk,
            network.critic.candidate_chunk(
                inputs={
                    name: tf.gather(value, indices)
                    for name, value in inputs.items()
                },
                payoff=tf.gather(payoffs, tf.gather(position_indices, indices)),
                weights=weights,
                integer_fields=integer_fields,
            ),
        )
    return tf.stop_gradient(
        tf.scatter_nd(
            selected,
            tf.reshape(values.stack(), [-1])[: tf.shape(selected)[0]],
            [tf.shape(features["decision_phase"])[0], 295],
        )
    )


class CriticTrainer:
    """Observed payment NLL and critic constraints.

    Shared features/beliefs receive indirect gradients. This phase has no
    direct belief-label loss.
    See OBJECTIVE.md for the phase-by-head supervision table.
    """

    def __init__(self, network, features, learning_rate):
        self.network = network
        self.feature_packing = training_data.PackedBatch(features)
        self.optimizer = AdamW(
            learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0
        )
        self.optimizer.build(network.trainable_variables)

    @tf.function(reduce_retracing=True)
    def critic_update(self, batch, count):
        features = self.feature_packing.unpack(batch["features"])
        with tf.GradientTape() as tape:
            encoded, state = self.network.candidate_context(
                features, self.network.encoder(features, training=False)
            )
            observed = self.network.chosen_inputs(
                features, encoded, state, batch["chosen"]
            )
            statistics = self.network.critic.supervised_statistics(
                observed,
                joint_settlement.payment_labels(
                    batch["payment_points"], observed
                ),
                valid=tf.range(tf.shape(batch["chosen"])[0]) < count,
            )
            metrics = dict(
                zip(
                    critic.CRITIC_METRICS,
                    tf.unstack(losses.packed_means(statistics)),
                )
            )
            loss = critic.settlement_objective(metrics)
            scaled = loss / tf.distribute.get_strategy().num_replicas_in_sync
        return {
            **{
                name: value
                for name, value in metrics.items()
                if name not in ("score_nll", "bonus_han_nll")
            },
            **apply_gradients(
                self.optimizer, tape, scaled, self.network.trainable_variables
            ),
            "loss": loss,
            "learning_rate": tf.identity(self.optimizer.learning_rate),
        }
