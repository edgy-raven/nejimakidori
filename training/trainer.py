"""AWR updates for mixed recorded and generated replay."""

import tensorflow as tf

from log_dataset import training_data
from model import actions
from model import actor as actor_layers
from model import critic, joint_settlement, losses
from training import optimization


class Trainer:
    def __init__(
        self,
        *,
        network,
        args,
        features,
        mirror=None,
        mirror_coefficient=0.0,
        awr_q_std=None,
    ):
        self.network = network
        self.args = args
        self.awr_q_std = awr_q_std
        self.mirror = mirror
        self.mirror_coefficient = mirror_coefficient
        if mirror is not None:
            mirror.trainable = False
        self.feature_packing = training_data.PackedBatch(features)
        self.awr_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=args.awr_learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0,
        )
        self.awr_optimizer.build(network.trainable_variables)

    @tf.function(reduce_retracing=True)
    def policy_update(self, batch, count, train=True):
        """One actor/value/settlement update for compact packed replay."""
        features = self.feature_packing.unpack(batch["features"])
        chosen = batch["chosen"][:count]
        if self.mirror is not None:
            reference = self.mirror(features, training=False)
        with tf.GradientTape() as tape:
            outputs, state = self.network.candidate_context(
                features, self.network.actor(features, training=False)
            )
            recorded = self.network.chosen_inputs(
                features,
                outputs,
                state,
                critic.chosen_actions(
                    features,
                    {
                        name: tf.where(
                            batch["head"] == index, batch["chosen"], -1
                        )
                        for index, name in enumerate(actions.TRAIN_POLICY_HEADS)
                    },
                ),
            )
            statistics = self.network.critic.supervised_statistics(
                recorded,
                joint_settlement.payment_labels(
                    batch["payment_points"], recorded
                ),
                valid=tf.range(tf.shape(batch["returns"])[0]) < count,
            )
            current = actor_layers.distribution(
                outputs, batch["head"], batch["legal"]
            )[:count]
            if self.mirror is not None:
                reference = actor_layers.distribution(
                    reference, batch["head"], batch["legal"]
                )[:count]
            valid = tf.ones(count, tf.bool)
            if self.awr_q_std is not None:
                # Score only the observed action, without recording gradients
                # or materializing settlements for the entire replay batch.
                with tape.stop_recording():
                    moments = tf.TensorArray(
                        tf.float32,
                        size=0,
                        dynamic_size=True,
                        element_shape=[32, 2],
                    )
                    for begin in tf.range(0, tf.shape(batch["returns"])[0], 32):
                        moments = moments.write(
                            begin // 32,
                            self.network.critic.projected_mean_std(
                                {
                                    name: tf.gather(
                                        value,
                                        tf.minimum(
                                            tf.range(begin, begin + 32),
                                            tf.shape(batch["returns"])[0] - 1,
                                        ),
                                    )
                                    for name, value in recorded.items()
                                }
                            ),
                        )
                    moments = moments.concat()[:count]
                    policy_target = (
                        moments[:, 0] - self.awr_q_std * moments[:, 1]
                    )
            else:
                policy_target = batch["returns"][:count] * 100.0
            advantages = tf.stop_gradient(
                (policy_target - outputs["state_value"][:count] * 100.0)
                / self.args.awr_temperature
            )

            def terms(mean):
                return {
                    **(
                        {
                            "q_mean_st3": mean(moments[:, 0], valid),
                            "q_std_st3": mean(moments[:, 1], valid),
                            "q_risk_st3": mean(policy_target, valid),
                        }
                        if self.awr_q_std is not None
                        else {}
                    ),
                    **(
                        {
                            "mirror_kl": actor_layers.policy_kl(
                                reference, current, mean=mean, reverse=True
                            )
                        }
                        if self.mirror is not None
                        else {}
                    ),
                    **actor_layers.awr_policy_terms(
                        -tf.gather(current, chosen, batch_dims=1),
                        advantages,
                        valid,
                        mean=mean,
                    ),
                    "value_loss": mean(
                        tf.square(
                            outputs["state_value"][:count]
                            - batch["returns"][:count]
                        ),
                        valid,
                    ),
                    "value_bias": mean(
                        outputs["state_value"][:count]
                        - batch["returns"][:count],
                        valid,
                    ),
                    **actor_layers.decision_regularizers(
                        outputs=outputs,
                        current=current,
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
            loss = (
                metrics["loss"]
                + metrics["value_loss"]
                + 0.1
                * (metrics["payment_nll"] + metrics["critic_constraint_loss"])
                + 0.01 * metrics["constraint_loss"]
                - self.args.awr_entropy * metrics["entropy"]
            )
            if self.mirror is not None:
                metrics["weighted_mirror_loss"] = (
                    self.mirror_coefficient * metrics["mirror_kl"]
                )
                loss += metrics["weighted_mirror_loss"]
            scaled = loss / tf.distribute.get_strategy().num_replicas_in_sync
        if train:
            metrics.update(
                optimization.apply_gradients(
                    self.awr_optimizer,
                    tape,
                    scaled,
                    self.network.trainable_variables,
                )
            )
        # Preserve exact counts for distributed dataset reporting.
        matches = tf.math.top_k(current, k=3).indices == chosen[:, None]
        return {
            **metrics,
            "policy": metrics["loss"],
            "joint_loss": loss,
            "learning_rate": tf.identity(self.awr_optimizer.learning_rate),
            "imitation_count": tf.reduce_sum(tf.cast(valid, tf.float32)),
            "imitation_correct": tf.reduce_sum(
                tf.cast(matches[:, 0], tf.float32)
            ),
            "imitation_top3_correct": tf.reduce_sum(
                tf.cast(tf.reduce_any(matches, axis=1), tf.float32)
            ),
        }


class DistributedTrainer:
    """Shard packed CPU replay and reduce globally normalized AWR metrics."""

    def __init__(self, trainer, strategy):
        self.trainer = trainer
        self.strategy = strategy
        self.feature_packing = trainer.feature_packing
        self.awr_optimizer = trainer.awr_optimizer

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
        metrics = self.strategy.run(
            self.trainer.policy_update.python_function, args=(batch, count)
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
        return metrics

    def policy_update(self, batch, count):
        batch = {
            name: batch[name]
            for name in (
                "features",
                "head",
                "legal",
                "chosen",
                "returns",
                "payment_points",
            )
        }
        size = len(batch["chosen"]) // self.strategy.num_replicas_in_sync
        valid = self.strategy.experimental_distribute_values_from_function(
            lambda ctx: tf.clip_by_value(
                count - ctx.replica_id_in_sync_group * size, 0, size
            )
        )
        return self.update(self.distribute(batch), valid)
