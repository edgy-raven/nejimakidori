"""Hybrid expert/distill policy imitation and magnetic self-play policy improvement."""

import tensorflow as tf
from log_dataset import training_data
from model import actions
from model import actor as actor_layers
from model import beliefs, critic, joint_settlement, losses
from training import temperature

VALUE_CHUNK = 256


def apply_gradients(optimizer, tape, loss, variables):
    # Sum repeated tile entries locally: each embedding has only 34 rows.
    # Clip and synchronize the actual table gradient, not every occurrence.
    pairs = [
        (tf.convert_to_tensor(gradient), variable)
        for gradient, variable in zip(tape.gradient(loss, variables), variables)
        if gradient is not None
    ]
    norm = tf.debugging.check_numerics(
        tf.linalg.global_norm([gradient for gradient, _ in pairs]),
        "Non-finite local gradient norm",
    )
    with tf.control_dependencies([norm]):
        optimizer.apply_gradients(pairs)
    return {"gradient/local_norm": norm}


class Trainer:
    """Hybrid imitation and magnetic policy improvement on self play."""

    def __init__(self, *, network, args, features, mirror, magnet):
        self.network = network
        self.args = args
        self.temperature = temperature.RollingTemperature(args.beta)
        self.beta = tf.Variable(
            [args.beta, args.beta], trainable=False, dtype=tf.float32
        )
        self.mirror = mirror
        self.magnet = magnet
        self.mirror.trainable = False
        self.magnet.trainable = False
        self.feature_packing = training_data.PackedBatch(features)
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0,
        )
        self.optimizer.build(network.trainable_variables)

    def candidate_values(
        self,
        *,
        features,
        batch,
        outputs,
        state,
        chosen,
        distill,
        reference,
        count,
    ):
        """Detached action values; only played outcomes supervise the critic."""
        selected = outputs["selected"]
        public = joint_settlement.public_inputs(features)
        value_rows = (
            critic.active_candidates(features, selected)
            & (selected[:, 0] < count)
            & (
                ~tf.gather(batch["recorded"], selected[:, 0])
                | (
                    tf.gather(
                        (chosen != reference) | (distill != reference),
                        selected[:, 0],
                    )
                    & (
                        (selected[:, 1] == tf.gather(chosen, selected[:, 0]))
                        | (selected[:, 1] == tf.gather(distill, selected[:, 0]))
                        | (
                            selected[:, 1]
                            == tf.gather(reference, selected[:, 0])
                        )
                    )
                )
            )
        )
        selected = tf.boolean_mask(selected, value_rows)
        inputs = {
            **{
                name: tf.gather(value, selected[:, 0])
                for name, value in public.items()
            },
            **critic.public_inputs(features, selected),
            "state": tf.stop_gradient(tf.boolean_mask(state, value_rows)),
        }
        positions, position_indices = tf.unique(selected[:, 0])
        payoffs = joint_settlement.gather_position_payoffs(
            {**public, "dealer": features["dealer_relative_seat"]}, positions
        )
        integer_fields = tuple(
            name for name, value in inputs.items() if value.dtype == tf.int32
        )
        inputs = {
            name: (
                tf.bitcast(value, tf.float32)
                if name in integer_fields
                else value
            )
            for name, value in inputs.items()
        }
        weights = [
            tf.stop_gradient(value.value.read_value())
            for value in self.network.critic.trainable_variables
        ]
        chunk_count = (tf.shape(selected)[0] + VALUE_CHUNK - 1) // VALUE_CHUNK
        values = tf.TensorArray(
            tf.float32, size=chunk_count, element_shape=[VALUE_CHUNK, 2]
        )
        for chunk in tf.range(chunk_count):
            indices = tf.minimum(
                chunk * VALUE_CHUNK + tf.range(VALUE_CHUNK),
                tf.shape(selected)[0] - 1,
            )
            values = values.write(
                chunk,
                self.network.critic.candidate_chunk(
                    inputs={
                        name: tf.gather(value, indices)
                        for name, value in inputs.items()
                    },
                    payoff=tf.gather(
                        payoffs, tf.gather(position_indices, indices)
                    ),
                    weights=weights,
                    integer_fields=integer_fields,
                ),
            )
        return tf.stop_gradient(
            tf.scatter_nd(
                selected,
                tf.reshape(values.stack(), [-1, 2])[: tf.shape(selected)[0]],
                [tf.shape(chosen)[0], 295, 2],
            )
        )

    @tf.function(reduce_retracing=True)
    def policy_update(self, batch, count, train=True, capture_gaps=False):
        """One joint update; actor targets cannot move the critic forecasts."""
        features = self.feature_packing.unpack(batch["features"])
        mirror = actor_layers.distribution(
            self.mirror(features, training=False), batch["head"], batch["legal"]
        )
        magnet = actor_layers.distribution(
            self.magnet(features, training=False), batch["head"], batch["legal"]
        )
        reference = actor_layers.greedy_choices(magnet, batch["head"])
        chosen_actions = critic.chosen_actions(
            features,
            {
                name: tf.where(batch["head"] == index, batch["chosen"], -1)
                for index, name in enumerate(actions.TRAIN_POLICY_HEADS)
            },
        )
        distill_actions = critic.chosen_actions(
            features,
            {
                name: tf.where(
                    batch["head"] == index,
                    tf.maximum(batch["distill"], 0),
                    -1,
                )
                for index, name in enumerate(actions.TRAIN_POLICY_HEADS)
            },
        )
        reference_actions = critic.chosen_actions(
            features,
            {
                name: tf.where(batch["head"] == index, reference, -1)
                for index, name in enumerate(actions.TRAIN_POLICY_HEADS)
            },
        )
        with tf.GradientTape() as tape:
            outputs, state = self.network.candidate_context(
                features, self.network.actor(features, training=False)
            )
            observed = self.network.chosen_inputs(
                features, outputs, state, chosen_actions
            )
            statistics = self.network.critic.supervised_statistics(
                observed,
                joint_settlement.payment_labels(
                    batch["payment_points"], observed
                ),
                valid=tf.range(tf.shape(batch["chosen"])[0]) < count,
            )
            values = self.candidate_values(
                features=features,
                batch=batch,
                outputs=outputs,
                state=state,
                chosen=chosen_actions,
                distill=distill_actions,
                reference=reference_actions,
                count=count,
            )
            current = actor_layers.distribution(
                outputs, batch["head"], batch["legal"]
            )
            expert_values = tf.gather(values, chosen_actions, batch_dims=1)
            reference_values = tf.gather(
                values, reference_actions, batch_dims=1
            )
            q_gap = expert_values[:, 0] - reference_values[:, 0]
            distill_q_gap = (
                tf.gather(values, distill_actions, batch_dims=1)[:, 0]
                - reference_values[:, 0]
            )
            risk = self.args.risk_coefficient * tf.maximum(
                values[:, :, 1] - reference_values[:, None, 1], 0.0
            )
            score = values[:, :, 0] - reference_values[:, None, 0] - risk
            # Use each row's normalized active head, including the response-kan
            # mapping. WIN has one forecast, so its value term is constant.
            probabilities = critic.candidate_probabilities(
                features,
                {
                    name: current[:, : actions.ACTOR_OUTPUTS[name][0]]
                    for name in actions.TRAIN_POLICY_HEADS
                },
            )
            expected_gain = tf.reduce_sum(probabilities * score, axis=1)[:count]
            expected_risk = tf.reduce_sum(probabilities * risk, axis=1)[:count]
            recorded = batch["recorded"][:count]
            valid = tf.ones(count, tf.bool)

            def terms(mean):
                belief_terms, belief_metrics = beliefs.belief_loss_terms(
                    {name: value[:count] for name, value in features.items()},
                    {
                        name: value[:count]
                        for name, value in batch["beliefs"].items()
                    },
                    {name: value[:count] for name, value in outputs.items()},
                    mean=mean,
                )
                imitation = {
                    name: actor_layers.weighted_imitation_terms(
                        nll=-tf.gather(
                            current[:count],
                            tf.maximum(batch[label][:count], 0),
                            batch_dims=1,
                        ),
                        advantages=gap[:count] / self.beta[source],
                        eligible=recorded,
                        mean=mean,
                    )
                    for source, (name, label, gap) in enumerate(
                        (
                            ("expert", "chosen", q_gap),
                            ("distill", "distill", distill_q_gap),
                        )
                    )
                }
                record_fraction = mean(tf.cast(recorded, tf.float32), valid)
                return {
                    **belief_terms,
                    **belief_metrics,
                    "belief_loss": tf.add_n(list(belief_terms.values())) / 3,
                    "belief_constraint_loss": mean(
                        outputs["belief_constraint_loss"][:count], valid
                    ),
                    "expert_loss": 0.5
                    * (
                        imitation["expert"]["loss"]
                        + imitation["distill"]["loss"]
                    ),
                    "expert_weighted_ce": imitation["expert"]["loss"],
                    "distill_weighted_ce": imitation["distill"]["loss"],
                    **{
                        f"{source}_{name}": value
                        for source, metrics in imitation.items()
                        for name, value in metrics.items()
                        if name != "loss"
                    },
                    "self_play_loss": mean(-expected_gain, ~recorded),
                    "record_fraction": record_fraction,
                    "expert_q_gap_st3": mean(q_gap[:count], recorded),
                    "distill_q_gap_st3": mean(distill_q_gap[:count], recorded),
                    "self_play_gain_st3": mean(expected_gain, ~recorded),
                    "risk_penalty_st3": mean(expected_risk, ~recorded),
                    "mirror_kl": actor_layers.policy_kl(
                        reference=mirror[:count],
                        current=current[:count],
                        mean=mean,
                        reverse=True,
                    ),
                    "magnet_kl": actor_layers.policy_kl(
                        reference=magnet[:count],
                        current=current[:count],
                        mean=mean,
                        reverse=True,
                    ),
                    "expert_prior_loss": actor_layers.expert_prior_loss(
                        current[:count],
                        actor_layers.expert_prior(features, batch["head"])[
                            :count
                        ],
                        batch["head"][:count],
                        mean=mean,
                    ),
                    **actor_layers.decision_regularizers(
                        outputs=outputs,
                        current=current[:count],
                        heads=batch["head"],
                        count=count,
                        mean=mean,
                    ),
                    **{
                        name: value
                        for name, value in critic.settlement_metrics(
                            statistics, mean
                        ).items()
                        if name not in ("score_nll", "bonus_han_nll")
                    },
                }

            means = losses.MaskedMeans()
            terms(means.collect)
            means.reduce()
            metrics = terms(means.read)
            metrics["policy"] = (
                metrics["record_fraction"] * metrics["expert_loss"]
                + (1.0 - metrics["record_fraction"]) * metrics["self_play_loss"]
            )
            metrics["weighted_mirror_loss"] = (
                self.args.mirror_coefficient * metrics["mirror_kl"]
            )
            metrics["weighted_magnet_loss"] = (
                self.args.magnet_coefficient * metrics["magnet_kl"]
            )
            loss = (
                metrics["policy"]
                + metrics["weighted_mirror_loss"]
                + metrics["weighted_magnet_loss"]
                + self.args.expert_prior * metrics["expert_prior_loss"]
                + 0.1 * critic.settlement_objective(metrics)
                + 0.01 * metrics["constraint_loss"]
                + metrics["belief_loss"]
            )
            scaled = loss / tf.distribute.get_strategy().num_replicas_in_sync
        if train:
            metrics.update(
                apply_gradients(
                    self.optimizer,
                    tape,
                    scaled,
                    self.network.trainable_variables,
                )
            )
        matches = (
            tf.math.top_k(current[:count], k=3).indices
            == batch["chosen"][:count, None]
        )
        metrics = {
            **metrics,
            "expert_beta": tf.identity(self.beta[0]),
            "distill_beta": tf.identity(self.beta[1]),
            "joint_loss": loss,
            "learning_rate": tf.identity(self.optimizer.learning_rate),
            "imitation_count": tf.reduce_sum(tf.cast(recorded, tf.float32)),
            "imitation_correct": tf.reduce_sum(
                tf.cast(matches[:, 0] & recorded, tf.float32)
            ),
            "imitation_top3_correct": tf.reduce_sum(
                tf.cast(tf.reduce_any(matches, axis=1) & recorded, tf.float32)
            ),
        }
        if capture_gaps:
            stride = tf.maximum((tf.shape(q_gap)[0] + 63) // 64, 1)
            return metrics, tf.stop_gradient(
                tf.stack(
                    [
                        q_gap,
                        distill_q_gap,
                        tf.cast(
                            batch["recorded"]
                            & (tf.range(tf.shape(q_gap)[0]) < count),
                            tf.float32,
                        ),
                    ],
                    axis=1,
                )[::stride]
            )
        return metrics


class DistributedTrainer:
    """Shard packed CPU replay and reduce globally normalized policy metrics."""

    def __init__(self, trainer, strategy):
        self.trainer = trainer
        self.strategy = strategy
        self.feature_packing = trainer.feature_packing
        self.optimizer = trainer.optimizer

    def distribute(self, batch):
        size = (
            len(tf.nest.flatten(batch)[0]) // self.strategy.num_replicas_in_sync
        )
        with tf.device("/CPU:0"):
            return self.strategy.experimental_distribute_values_from_function(
                lambda ctx: tf.nest.map_structure(
                    lambda value: tf.convert_to_tensor(
                        value[
                            ctx.replica_id_in_sync_group
                            * size : (ctx.replica_id_in_sync_group + 1)
                            * size
                        ]
                    ),
                    batch,
                )
            )

    @tf.function
    def update(self, batch, count):
        metrics, samples = self.strategy.run(
            self.trainer.policy_update.python_function,
            args=(batch, count),
            kwargs={"capture_gaps": True},
        )
        packed = self.strategy.run(
            lambda values: tf.stack(tuple(values.values())), args=(metrics,)
        )
        values = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, packed, None)
        metrics = dict(zip(metrics, tf.unstack(values)))
        count = metrics.pop("imitation_count")
        metrics["imitation_accuracy"] = tf.math.divide_no_nan(
            metrics.pop("imitation_correct"), count
        )
        metrics["imitation_top3_accuracy"] = tf.math.divide_no_nan(
            metrics.pop("imitation_top3_correct"), count
        )
        return metrics, self.strategy.gather(samples, axis=0)

    def policy_update(self, batch, count):
        # Replay owns the batch contract. Preserve its complete nested payload
        # instead of silently dropping new supervision at this boundary.
        return self.step(batch, count)

    def distributed_update(self, batch, count):
        size = len(batch["chosen"]) // self.strategy.num_replicas_in_sync
        valid = self.strategy.experimental_distribute_values_from_function(
            lambda ctx: tf.clip_by_value(
                count - ctx.replica_id_in_sync_group * size, 0, size
            )
        )
        return self.update(self.distribute(batch), valid)

    def step(self, batch, count):
        metrics, samples = self.distributed_update(batch, count)
        if self.trainer.temperature.observe(samples.numpy()):
            self.trainer.beta.assign(self.trainer.temperature.beta)
        for source, name in enumerate(("expert", "distill")):
            metrics[name + "_rolling_ess"] = self.trainer.temperature.ess[
                source
            ]
            metrics[name + "_minimum_rolling_ess"] = (
                self.trainer.temperature.minimum_ess[source]
            )
        return metrics


class CriticTrainer:
    """Observed payment NLL, ST3 MSE and critic constraints.

    Shared features/beliefs receive indirect gradients. This phase has no
    direct belief-label loss.
    See OBJECTIVE.md for the phase-by-head supervision table.
    """

    def __init__(self, network, features, learning_rate):
        self.network = network
        self.feature_packing = training_data.PackedBatch(features)
        self.optimizer = tf.keras.optimizers.AdamW(
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


class DistributedCriticTrainer(DistributedTrainer):
    def step(self, batch, count):
        return self.distributed_update(batch, count)

    @tf.function
    def update(self, batch, count):
        metrics = self.strategy.run(
            self.trainer.critic_update.python_function, args=(batch, count)
        )
        return {
            name: self.strategy.reduce(tf.distribute.ReduceOp.MEAN, value, None)
            for name, value in metrics.items()
        }
