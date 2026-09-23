"""Shared supervised learning with explicit valid-label means."""

import json
import os
import pathlib
import time

import numpy
import tensorflow as tf

import feature_vector

from . import decision_layers, objectives, outcomes


@tf.keras.utils.register_keras_serializable(package="nejimakidori")
class ReferenceSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, batch_size, peak=2e-4):
        self.config = dict(
            total_steps=total_steps, batch_size=batch_size, peak=peak
        )
        self.warmup_steps = max(1, total_steps * 2 // 100)
        self.decay_start = total_steps * 3 // 4
        self.decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=peak,
            decay_steps=total_steps - self.decay_start,
            alpha=0.1,
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.where(
            step < self.warmup_steps,
            self.config["peak"] * (0.1 + 0.9 * step / self.warmup_steps),
            self.decay(tf.maximum(step - self.decay_start, 0.0)),
        )

    def get_config(self):
        return self.config


class GroupedModel(tf.keras.Model):
    def __init__(self, actor, advice_strength=0.25):
        super().__init__(autocast=False)
        self.actor = actor
        self.advice_strength = advice_strength
        self.forward = tf.keras.Model(
            actor.input,
            {
                **actor.output,
                "board_state": actor.get_layer("board_state_normalized").output,
                "final_pass": actor.get_layer("final_pass_normalized").output,
                **{
                    name: actor.get_layer(name).output
                    for name in (
                        "attack_predictions",
                        "defense_predictions",
                        "attack_decisions",
                        "defense_decisions",
                    )
                },
                "candidate_indices": actor.get_layer(
                    "decision_candidates"
                ).output[1],
            },
        )
        self.baseline = tf.keras.layers.Dense(
            2,
            dtype="float32",
            name="payoff_baseline",
            kernel_initializer="zeros",
        )
        self.baseline.build(
            actor.get_layer("board_state_normalized").output.shape
        )
        self.expert_baseline = tf.keras.layers.Dense(
            5, dtype="float32", name="expert_baseline"
        )
        self.expert_baseline.build(
            actor.get_layer("board_state_normalized").output.shape
        )

        self.payoff_critic = tf.keras.layers.Dense(
            2,
            dtype="float32",
            name="payoff_action_values",
            kernel_initializer="zeros",
        )
        self.payoff_critic.build(
            actor.get_layer("final_pass_normalized").output.shape
        )

    def call(self, inputs, training=False):
        outputs = self.forward(inputs, training=training)
        outputs["payoff_action_values"] = self.payoff_critic(
            outputs.pop("final_pass")
        )
        for name in (
            "attack_predictions",
            "defense_predictions",
            "attack_decisions",
            "defense_decisions",
            "payoff_action_values",
        ):
            outputs[name] = tf.scatter_nd(
                outputs["candidate_indices"],
                outputs[name],
                (
                    tf.shape(inputs["decision_phase"])[0],
                    298,
                    outputs[name].shape[-1],
                ),
            )
        outputs["expert_baseline"] = self.expert_baseline(
            outputs["board_state"]
        )
        outputs["payoff_baseline"] = self.baseline(outputs.pop("board_state"))
        return outputs

    def specialist_loss(
        self, features, labels, outputs, metrics, mean=objectives.mean_valid
    ):
        expert_terms = {}
        masks = decision_layers.legal_choices(features)
        discard = tf.cast(labels["discard_policy_logits"], tf.int32)
        riichi = tf.cast(labels["riichi_policy_logits"], tf.int32)
        response = tf.cast(labels["response_policy_logits"], tf.int32)
        kan = tf.cast(labels["kan_action_logits"], tf.int32)
        eligible = (discard >= 0) | (riichi >= 0) | (response >= 0) | (kan >= 0)
        chosen = tf.where(
            discard >= 0,
            discard,
            tf.where(
                riichi >= 0,
                38 + riichi,
                tf.where(
                    response >= 0,
                    tf.where(
                        response == 149,
                        264 + tf.cast(features["trigger_tile_id"], tf.int32),
                        114 + response,
                    ),
                    263 + tf.maximum(kan, 0),
                ),
            ),
        )
        attack = tf.gather(outputs["attack_predictions"], chosen, batch_dims=1)
        defense = tf.gather(
            outputs["defense_predictions"], chosen, batch_dims=1
        )
        target = tf.cast(labels["specialist_outcomes"], tf.float32)
        reached = tf.cast(labels["tenpai_reached"], tf.float32)
        tenpai_value = tf.cast(labels["tenpai_value"], tf.float32)
        holding_cost = tf.cast(labels["sakigiri_cost"], tf.float32)
        terms = {
            "attack_tenpai_value_loss": mean(
                tf.square(
                    tf.sigmoid(attack[:, 2])
                    - tf.math.divide_no_nan(tenpai_value, 1 + tenpai_value)
                ),
                eligible & (reached > 0) & (tenpai_value >= 0),
            ),
            "defense_sakigiri_loss": 0.1
            * mean(
                tf.square(
                    tf.sigmoid(defense[:, 2])
                    - tf.math.divide_no_nan(holding_cost, 1 + holding_cost)
                ),
                eligible & (reached > 0) & (holding_cost >= 0),
            ),
        }
        for name, prediction in (
            ("attack", attack[:, 1]),
            ("defense", defense[:, 5]),
        ):
            terms[name + "_tenpai_reached_loss"] = mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.maximum(reached, 0), logits=prediction
                ),
                eligible & (reached >= 0),
            )
        for name in ("attack", "defense"):
            scores = tf.where(
                tf.concat(
                    [
                        masks["discard_policy_logits"],
                        masks["riichi_policy_logits"],
                    ],
                    axis=1,
                ),
                outputs[name + "_decisions"][:, :114, 0],
                -1e4,
            )
            scores = tf.where(
                (features["decision_phase"] == 3)[:, None],
                tf.reduce_logsumexp(
                    tf.reshape(scores[:, 38:114], (-1, 2, 38)), axis=1
                ),
                scores[:, :38],
            )
            value, valid = objectives.regret_score_terms(labels, scores)
            expert_terms[name + "_regret_loss"] = mean(
                value,
                valid
                & (
                    (features["decision_phase"] == 0)
                    | (features["decision_phase"] == 3)
                ),
            )
        draw = tf.where(
            tf.cast(features["current_draw_is_red"], tf.bool),
            34 + tf.cast(features["current_draw_tile_id"], tf.int32) // 9,
            tf.cast(features["current_draw_tile_id"], tf.int32),
        )
        batch = tf.shape(draw)[0]
        cuts = tf.concat(
            [tf.broadcast_to(tf.range(37)[None], (batch, 37)), draw[:, None]],
            axis=1,
        )
        codes = tf.concat(
            [
                cuts,
                cuts,
                cuts,
                tf.fill((batch, 1), 37),
                tf.broadcast_to(tf.tile(tf.range(37), [4])[None], (batch, 148)),
                tf.fill((batch, 35), 37),
            ],
            axis=1,
        )
        legal = tf.concat(
            [
                masks["discard_policy_logits"]
                & (features["decision_phase"] == 0)[:, None],
                masks["riichi_policy_logits"]
                & (features["decision_phase"] == 3)[:, None],
                masks["response_policy_logits"][:, :149]
                & (features["decision_phase"] == 1)[:, None],
                masks["kan_action_logits"]
                & (
                    (features["decision_phase"] == 2)[:, None]
                    | (
                        (features["decision_phase"] == 1)[:, None]
                        & (tf.range(35)[None] > 0)
                    )
                ),
            ],
            axis=1,
        )
        danger = tf.cast(
            tf.gather(
                labels["defense_ron"], tf.minimum(codes, 36), batch_dims=1
            ),
            tf.float32,
        )
        valid = eligible[:, None] & legal & (codes < 37) & (danger >= 0)
        terms["defense_ron_probability_loss"] = mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(danger > 0, tf.float32),
                logits=outputs["defense_predictions"][:, :, 3],
            ),
            valid,
        )
        terms["defense_ron_loss"] = mean(
            tf.square(
                tf.nn.softplus(outputs["defense_predictions"][:, :, 4])
                - objectives.signed_log(danger)
            ),
            valid,
        )
        rewards = tf.stack(
            [
                tf.where(
                    reached >= 0,
                    reached,
                    -1.0,
                ),
                objectives.signed_log(target[:, 0]),
                objectives.signed_log(target[:, 2]),
                objectives.tenpai_quality_reward(reached, tenpai_value),
                objectives.tenpai_quality_reward(reached, holding_cost),
            ],
            axis=-1,
        )
        advantages, baseline_loss = objectives.expert_reward_terms(
            rewards=rewards,
            estimates=tf.stack(
                [
                    tf.stack(
                        [
                            tf.sigmoid(attack[:, 1]),
                            tf.nn.softplus(attack[:, 0]),
                            tf.zeros_like(target[:, 2]),
                            tf.sigmoid(attack[:, 1]) * tf.sigmoid(attack[:, 2]),
                            tf.zeros_like(holding_cost),
                        ],
                        axis=-1,
                    ),
                    tf.stack(
                        [
                            tf.sigmoid(defense[:, 5]),
                            tf.zeros_like(target[:, 0]),
                            tf.nn.softplus(defense[:, 1]),
                            tf.zeros_like(tenpai_value),
                            tf.sigmoid(defense[:, 5])
                            * tf.sigmoid(defense[:, 2]),
                        ],
                        axis=-1,
                    ),
                ],
                axis=1,
            ),
            baseline=outputs["expert_baseline"],
            eligible=eligible,
            mean=mean,
        )
        selected = tf.one_hot(
            chosen, tf.shape(legal)[1], on_value=True, off_value=False
        )
        payments = tf.gather(
            tf.cast(labels["ron_payments"], tf.float32),
            tf.minimum(codes, 36),
            axis=2,
            batch_dims=1,
        )
        ron_loss = tf.reduce_sum(payments, axis=1) / 10000
        known_ron = (
            tf.reduce_all(payments >= 0, axis=1)
            & (ron_loss > 0)
            & legal
            & ~selected
            & eligible[:, None]
            & (
                (features["decision_phase"] == 0)
                | (features["decision_phase"] == 3)
            )[:, None]
        )
        expert_terms["defense_known_alternatives"] = mean(
            tf.reduce_sum(tf.cast(known_ron, tf.float32), axis=-1), eligible
        )
        for name, column, prediction in (
            ("attack_receipts_loss", 0, 0),
            ("defense_dealin_probability_loss", 1, 0),
            ("defense_dealin_loss", 2, 1),
        ):
            known = known_ron | (
                selected & (target[:, column] >= 0)[:, None] & eligible[:, None]
            )
            values = tf.where(
                selected,
                target[:, column, None],
                0.0 if column == 0 else (1.0 if column == 1 else ron_loss),
            )
            prediction_logits = outputs[
                "attack_predictions" if column == 0 else "defense_predictions"
            ][:, :, prediction]
            error = (
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.maximum(values, 0), logits=prediction_logits
                )
                if column == 1
                else tf.square(
                    tf.nn.softplus(prediction_logits)
                    - objectives.signed_log(values)
                )
            )
            terms[name] = mean(
                tf.math.divide_no_nan(
                    tf.reduce_sum(tf.where(known, error, 0.0), axis=-1),
                    tf.reduce_sum(tf.cast(known, tf.float32), axis=-1),
                ),
                tf.reduce_any(known, axis=-1),
            )
        metrics.update(terms)
        expert_losses = []
        for name in ("attack", "defense"):
            nll = objectives.expert_policy_nll(
                scores=outputs[name + "_decisions"][:, :, 0],
                legal=legal,
                chosen=chosen,
            )
            imitation = mean(nll, eligible)
            awr = objectives.awr_policy_terms(
                nll, advantages[name], eligible, mean=mean
            )
            reward_loss = awr.pop("loss")
            expert_terms.update(
                {name + "_awr_" + key: value for key, value in awr.items()}
            )

            expert_terms[name + "_imitation_loss"] = imitation
            expert_terms[name + "_reward_loss"] = reward_loss
            expert_terms[name + "_advantage"] = mean(advantages[name], eligible)
            expert_losses.append(
                0.5 * imitation
                + 0.5 * reward_loss
                + 0.025 * expert_terms[name + "_regret_loss"]
            )
        metrics.update(
            {"expert_" + name: value for name, value in expert_terms.items()}
        )
        metrics["expert_baseline_loss"] = baseline_loss
        metrics["expert_objective_loss"] = tf.add_n(expert_losses) / 2
        metrics["expert_supervision_loss"] = tf.add_n(list(terms.values()))
        return {
            "specialist_policy": metrics["expert_objective_loss"],
            "specialist_supervision": metrics["expert_supervision_loss"],
            "specialist_baseline": baseline_loss,
        }

    def loss_and_metrics(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
        payoff_weight=1.0,
    ):
        means = objectives.MaskedMeans()
        self.loss_terms(
            x=x,
            y=y,
            y_pred=y_pred,
            payoff_weight=payoff_weight,
            mean=means.collect,
        )
        means.reduce()
        return self.loss_terms(
            x=x,
            y=y,
            y_pred=y_pred,
            payoff_weight=payoff_weight,
            mean=means.read,
        )

    def loss_terms(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
        payoff_weight=1.0,
        mean=objectives.mean_valid,
    ):
        payoff = objectives.offline_policy_terms(
            labels=y,
            logits=y_pred,
            baseline=y_pred["payoff_baseline"],
            advice_strength=self.advice_strength,
            features=x,
            mean=mean,
        )
        terms = [objectives.policy_loss(y, y_pred, mean=mean)]
        distance_metrics = {}
        for label, output in (
            ("opponent_shanten", "opponent_shanten_logits"),
            ("round_placement", "round_placement_logits"),
            ("final_placement", "final_placement_logits"),
        ):
            if label == "opponent_shanten":
                value = objectives.classification(
                    y[label], y_pred[output], mean=mean
                )
            else:
                target = (
                    objectives.final_placement_targets(y[label], x)
                    if label == "final_placement"
                    else tf.one_hot(tf.cast(y[label], tf.int32), 4)
                )
                value = mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=target, logits=y_pred[output]
                    ),
                    y[label] >= 0,
                )
                distance = mean(
                    objectives.ordered_distance(
                        target, y_pred[output], tf.range(4)
                    ),
                    y[label] >= 0,
                )
                distance_metrics[label + "_distance_loss"] = distance
                value += distance
            terms.append(value)
        ukeire_loss, ukeire_metrics = objectives.ukeire_loss(
            y, y_pred, x, mean=mean
        )
        terms.append(ukeire_loss)
        target = tf.cast(y["round_completion_yaku"], tf.float32)
        terms.append(
            mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.maximum(target, 0),
                    logits=y_pred["round_completion_yaku_logits"],
                ),
                target >= 0,
            )
        )
        base = tf.constant(list(range(34)) + [4, 13, 22])
        dora = tf.gather(
            tf.cast(x["dora_multiplicity"], tf.float32), base, axis=-1
        )
        red = tf.cast(tf.range(37) >= 34, tf.float32)
        winds = tf.cast(x["seat_wind"][:, 1:], tf.int32) + 27
        prevailing = tf.cast(x["prevailing_wind"], tf.int32) + 27
        yakuhai = (
            (base[None, None, :] >= 31)
            | (base[None, None, :] == winds[..., None])
            | (base[None, None, :] == prevailing[:, None, None])
        )
        weight = (
            1
            + 9 * dora[:, None, :]
            + 9 * red
            + 4 * tf.cast(yakuhai, tf.float32)
        )
        terms.append(
            objectives.classification(
                y["opponent_hand_counts"],
                y_pred["opponent_hand_count_logits"],
                weight,
                mean=mean,
            )
        )
        payment_targets = tf.one_hot(
            tf.cast(y["terminal_payment"], tf.int32),
            y_pred["terminal_payment_logits"].shape[-1],
        )
        payment_metrics = objectives.payment_terms(
            payment_targets, y_pred["terminal_payment_logits"]
        )
        payment_losses = tf.add_n(list(payment_metrics.values()))
        terms.append(
            mean(
                payment_losses,
                (tf.reduce_sum(payment_targets, axis=-1) > 0)
                & tf.constant(outcomes.PAYMENT_CELLS)[None],
            )
        )
        metrics = dict(
            zip(
                (
                    "policy_loss",
                    "opponent_shanten_loss",
                    "round_placement_loss",
                    "final_placement_loss",
                    "opponent_ukeire_loss",
                    "completion_yaku_loss",
                    "opponent_hand_count_loss",
                    "terminal_payment_loss",
                ),
                terms,
            )
        )
        metrics.update(distance_metrics)
        metrics.update(
            {
                name: mean(
                    value,
                    (tf.reduce_sum(payment_targets, axis=-1) > 0)
                    & tf.constant(outcomes.PAYMENT_CELLS)[None],
                )
                for name, value in payment_metrics.items()
            }
        )
        metrics.update(
            {
                "payment_"
                + kind
                + "_loss": mean(
                    payment_losses[:, index],
                    (tf.reduce_sum(payment_targets[:, index], axis=-1) > 0)
                    & tf.constant(outcomes.PAYMENT_CELLS[index])[None],
                )
                for index, kind in enumerate(outcomes.PAYMENT_TYPES)
            }
        )
        metrics.update(ukeire_metrics)
        metrics.update(payoff)
        payoff_mix = payoff_weight / (1 + payoff_weight)
        weighted_payoff = (
            payoff_mix
            * (
                payoff["offline_payoff_loss"]
                + self.advice_strength * payoff["dealership_advice_loss"]
            )
            / (1 + self.advice_strength)
        )
        metrics.update(
            {
                "offline_payoff_weight": tf.cast(payoff_weight, tf.float32),
                "weighted_offline_payoff_loss": weighted_payoff,
                "offline_policy_mix": tf.cast(payoff_mix, tf.float32),
            }
        )
        entropy = objectives.policy_entropy(y, y_pred, mean=mean)
        entropy_loss = -objectives.POLICY_ENTROPY_WEIGHT * entropy
        metrics.update(
            {"policy_entropy": entropy, "policy_entropy_loss": entropy_loss}
        )
        metrics["imitation_correct"] = tf.add_n(
            [
                tf.reduce_sum(
                    tf.cast(
                        (y[name] >= 0)
                        & (
                            tf.argmax(y_pred[name], -1, output_type=tf.int32)
                            == tf.cast(y[name], tf.int32)
                        ),
                        tf.float32,
                    )
                )
                for name in feature_vector.TRAIN_POLICY_HEADS
            ]
        )
        metrics["imitation_count"] = tf.add_n(
            [
                tf.reduce_sum(tf.cast(y[name] >= 0, tf.float32))
                for name in feature_vector.TRAIN_POLICY_HEADS
            ]
        )
        metrics["imitation_accuracy"] = tf.math.divide_no_nan(
            metrics["imitation_correct"],
            metrics["imitation_count"],
        )
        ranking = tf.math.top_k(y_pred["discard_policy_logits"], k=3).indices
        for name, width in (("discard_match", 1), ("discard_top3", 3)):
            metrics[name] = mean(
                tf.cast(
                    tf.reduce_any(
                        ranking[:, :width]
                        == tf.cast(
                            y["discard_policy_logits"][:, None], tf.int32
                        ),
                        axis=1,
                    ),
                    tf.float32,
                ),
                y["discard_policy_logits"] >= 0,
            )
        groups = {
            "actor_imitation": (1 - payoff_mix) * terms[0],
            "actor_payoff": weighted_payoff,
            "outcome_critics": payoff["payoff_critic_loss"],
            "outcome_baselines": payoff["payoff_baseline_loss"],
            "entropy": entropy_loss,
            "board": tf.add_n(terms[1:]) / 7,
            **self.specialist_loss(
                features=x, labels=y, outputs=y_pred, metrics=metrics, mean=mean
            ),
        }
        for name, weight in objectives.LOSS_GROUP_WEIGHTS.items():
            groups[name] *= weight
        metrics.update(
            {"weighted_" + name: value for name, value in groups.items()}
        )
        return tf.add_n(list(groups.values())), metrics


class TopCheckpoints:
    """Save phase records in imitation accuracy, with one global cooldown."""

    def __init__(self, learner, root):
        self.learner = learner
        self.root = pathlib.Path(root)
        self.best = {}
        self.last_saved = -float("inf")

    def consider(self, phase, step, accuracy):
        if accuracy <= self.best.setdefault(phase, -1.0):
            return
        self.best[phase] = accuracy
        if time.monotonic() - self.last_saved < 1800:
            return
        path = self.root / "top" / f"{phase}-{step:09d}"
        path.mkdir(parents=True)
        self.learner.actor.save(path / "model.keras")
        self.learner.save_weights(path / "learner.weights.h5")
        (path / "checkpoint.json").write_text(
            json.dumps(
                {
                    "phase": phase,
                    "step": step,
                    "metric": "imitation_accuracy",
                    "accuracy": accuracy,
                    "window": "up to 100 updates within the phase",
                    "cooldown_seconds": 1800,
                },
                indent=2,
            )
        )
        self.last_saved = time.monotonic()
        print(
            f"top_checkpoint={path} imitation_accuracy={accuracy:.6f}",
            flush=True,
        )


def run(data, output, experiment=False, advice_strength=0.25):
    import csv
    import math

    from . import inference, model, records

    root = pathlib.Path(output)
    root.mkdir(parents=True, exist_ok=True)
    batch_size = 4 if experiment else 5120
    total_batches = 8 if experiment else 1540362240 // batch_size
    phases = [
        {
            "phase": name,
            "sampling": sampling,
            "epochs": 1,
            "steps_per_epoch": steps,
        }
        for name, sampling, steps in (
            ("natural_start", "natural", total_batches // 4),
            ("focused", "focused", total_batches // 2),
            (
                "natural_finish",
                "natural",
                total_batches - total_batches // 4 - total_batches // 2,
            ),
        )
    ]
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.set_logical_device_configuration(
            device,
            [tf.config.LogicalDeviceConfiguration(memory_limit=23552)],
        )
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        actor = model.build_model()
        learner = GroupedModel(actor, advice_strength=advice_strength)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=ReferenceSchedule(total_batches, batch_size),
            weight_decay=1e-4,
            epsilon=1e-6,
            clipnorm=1.0,
        )
        optimizer.build(learner.trainable_variables)
    (root / "run_config.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "batch_size": batch_size,
                "phase_budget": phases,
                "actual_positions": total_batches * batch_size,
                "actor_parameters": actor.count_params(),
                "dtype_policy": "mixed_bfloat16",
                "regret_weight": 0.025,
                "dealership_advice_strength": advice_strength,
                "critic_components": [
                    "placement",
                    "signed_log_honba_free_points",
                ],
                "point_value_transform": "sign(p)*log1p(abs(p)/10000)",
                "payment_distance_transform": "sign(p)*log1p(abs(p)/10000)",
                "specialist_value_transform": "sign(p)*log1p(abs(p)/10000)",
                "policy_entropy_weight": objectives.POLICY_ENTROPY_WEIGHT,
                "loss_group_weights": objectives.LOSS_GROUP_WEIGHTS,
                "structural_priors": False,
                "top_checkpoint_metric": "imitation_accuracy",
                "top_checkpoint_scope": "per_phase",
                "top_checkpoint_window_updates": 100,
                "top_checkpoint_cooldown_seconds": 1800,
                "validation": {
                    "game_fraction": 0 if experiment else 0.05,
                    "game_hash": "strong_bucket_20_key_239_20260923_zero",
                    "batches": 8,
                    "interval_updates": 2000,
                    "smoke_test_only": experiment,
                },
                "optimizer": tf.keras.optimizers.serialize(optimizer),
            },
            indent=2,
        )
    )

    @tf.function
    def step(batch):
        def update(inputs, labels, weights):
            with tf.GradientTape() as tape:
                outputs = learner(inputs, training=True)
                loss, metrics = learner.loss_and_metrics(
                    x=inputs,
                    y=labels,
                    y_pred=outputs,
                    sample_weight=weights,
                    payoff_weight=objectives.payoff_weight(
                        step=optimizer.iterations,
                        total_steps=total_batches,
                    ),
                )
                scaled = loss / strategy.num_replicas_in_sync
            gradients = tape.gradient(scaled, learner.trainable_variables)
            disconnected = [
                variable.path
                for gradient, variable in zip(
                    gradients, learner.trainable_variables
                )
                if gradient is None
            ]
            if disconnected:
                raise ValueError(
                    f"Disconnected trainable variables: {disconnected}"
                )
            learning_rate = tf.identity(optimizer.learning_rate)
            gradient_norm = tf.linalg.global_norm(gradients)
            value_gradients, actor_gradients = [], []
            for variable, gradient in zip(
                learner.trainable_variables, gradients
            ):
                if variable.path.startswith(
                    (
                        "payoff_baseline/",
                        "expert_baseline/",
                        "payoff_action_values/",
                    )
                ):
                    value_gradients.append(gradient)
                else:
                    actor_gradients.append(gradient)
            optimizer.apply_gradients(
                zip(gradients, learner.trainable_variables)
            )
            return {
                "loss": loss,
                **metrics,
                "learning_rate": learning_rate,
                "replica_gradient_norm": gradient_norm,
                "replica_actor_gradient_norm": tf.linalg.global_norm(
                    actor_gradients
                ),
                "replica_value_gradient_norm": tf.linalg.global_norm(
                    value_gradients
                ),
            }

        losses = strategy.run(update, args=batch)
        packed = strategy.run(
            lambda values: tf.stack(tuple(values.values())), args=(losses,)
        )
        reduced = strategy.reduce(tf.distribute.ReduceOp.MEAN, packed, None)
        return {
            name: (
                value * strategy.num_replicas_in_sync
                if name in ("imitation_correct", "imitation_count")
                else value
            )
            for name, value in zip(losses, tf.unstack(reduced))
        }

    @tf.function
    def validate(batch):
        inputs, labels, weights = batch
        _, metrics = learner.loss_and_metrics(
            x=inputs,
            y=labels,
            y_pred=learner(inputs, training=False),
            sample_weight=weights,
            payoff_weight=0.0,
        )
        return {
            name: metrics[name]
            for name in (
                "placement_observed_mse",
                "placement_prior_mse",
                "points_observed_mse",
                "points_prior_mse",
                "policy_loss",
                "imitation_accuracy",
            )
        }

    validation = (
        []
        if experiment
        else list(
            records.dataset(
                data=data,
                batch_size=512,
                training=False,
                sampling="natural",
                partition="validation",
            ).take(8)
        )
    )
    checkpoints = TopCheckpoints(learner, root)
    for phase in phases:
        print(
            f"phase={phase['phase']} epochs={phase['epochs']} steps={phase['steps_per_epoch']}",
            flush=True,
        )
        stream = records.dataset(
            data=data,
            batch_size=batch_size,
            training=True,
            sampling=phase["sampling"],
            partition=None if experiment else "train",
        )
        iterator = iter(strategy.experimental_distribute_dataset(stream))
        with (root / (phase["phase"] + ".csv")).open("w", newline="") as output:
            writer = None
            for epoch in range(phase["epochs"]):
                print(f"Epoch {epoch + 1}/{phase['epochs']}", flush=True)
                start = time.perf_counter()
                totals = {}
                imitation_correct = imitation_count = 0.0
                for index in range(phase["steps_per_epoch"]):
                    batch = next(iterator)
                    result = step(batch)
                    values = dict(
                        zip(
                            result,
                            tf.stack(tuple(result.values())).numpy().tolist(),
                        )
                    )
                    imitation_correct += values["imitation_correct"]
                    imitation_count += values["imitation_count"]
                    loss = values["loss"]
                    if not math.isfinite(loss):
                        raise FloatingPointError("nonfinite training objective")
                    step_count = int(optimizer.iterations.numpy())
                    if validation and (
                        step_count == 1 or step_count % 2000 == 0
                    ):
                        validation_values = [
                            validate(batch) for batch in validation
                        ]
                        validation_metrics = {
                            name: float(
                                numpy.average(
                                    [
                                        float(values[name])
                                        for values in validation_values
                                    ],
                                    weights=[
                                        len(batch[2]) for batch in validation
                                    ],
                                )
                            )
                            for name in validation_values[0]
                        }
                        with (root / "validation.jsonl").open("a") as log:
                            log.write(
                                json.dumps(
                                    {"step": step_count, **validation_metrics}
                                )
                                + "\n"
                            )
                        print(
                            f"validation_step={step_count} "
                            + " ".join(
                                f"{name}={value:.6f}"
                                for name, value in validation_metrics.items()
                            ),
                            flush=True,
                        )
                    for name, value in values.items():
                        totals[name] = totals.get(name, 0) + value
                    if (index + 1) % 100 == 0 or index + 1 == phase[
                        "steps_per_epoch"
                    ]:
                        checkpoints.consider(
                            phase["phase"],
                            int(optimizer.iterations.numpy()),
                            imitation_correct / imitation_count,
                        )
                        imitation_correct = imitation_count = 0.0
                        rate = (
                            (index + 1)
                            * batch_size
                            / (time.perf_counter() - start)
                        )
                        print(
                            f"step={index + 1} examples_per_second={rate:.3f} "
                            + " ".join(
                                f"{name}={value:.6f}"
                                for name, value in values.items()
                            )
                            + " loss_scale=1",
                            flush=True,
                        )
                if writer is None:
                    writer = csv.DictWriter(output, ["epoch", *totals])
                    writer.writeheader()
                writer.writerow(
                    {
                        "epoch": epoch,
                        **{
                            name: value / phase["steps_per_epoch"]
                            for name, value in totals.items()
                        },
                    }
                )
                output.flush()
    actor.save(root / "model.keras")
    learner.save_weights(root / "learner.weights.h5")
    module = tf.Module()
    module.actor = actor

    @tf.function
    def serve(**inputs):
        return actor(inputs, training=False)

    signature = serve.get_concrete_function(
        **{
            name: tf.TensorSpec(
                [None] + definition["shape"], definition["dtype"], name=name
            )
            for name, definition in feature_vector.INPUTS.items()
        }
    )
    saved_model = root / "saved_model"
    tf.saved_model.save(
        module, str(saved_model), signatures={"serving_default": signature}
    )
    inputs, _, _ = next(iter(records.dataset(data, 64)))
    expected = tf.function(
        lambda **values: actor(values, training=False)
    ).get_concrete_function(
        **{
            name: tf.TensorSpec([None, *value.shape[1:]], value.dtype)
            for name, value in inputs.items()
        }
    )(
        **inputs
    )
    actual = inference.Predictor(saved_model).predict(**inputs)
    errors = {}
    probability_errors = {}
    for name, value in expected.items():
        numpy.testing.assert_allclose(
            actual=actual[name],
            desired=value,
            rtol=1e-6,
            atol=1e-6,
            err_msg=name,
        )
        errors[name] = float(numpy.max(numpy.abs(actual[name] - value)))
    for name in feature_vector.TRAIN_POLICY_HEADS:
        reference = tf.nn.softmax(expected[name], -1)
        exported = tf.nn.softmax(actual[name], -1)
        numpy.testing.assert_allclose(exported, reference, rtol=0, atol=1e-6)
        numpy.testing.assert_array_equal(
            tf.argmax(actual[name], -1), tf.argmax(expected[name], -1)
        )
        probability_errors[name] = float(
            tf.reduce_max(tf.abs(exported - reference))
        )
    (root / "export-verification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "rows": 64,
                "max_errors": errors,
                "policy_probability_max_errors": probability_errors,
                "policy_argmax_identical": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument(
        "--dealership-advice-strength", type=float, default=0.25
    )
    arguments = parser.parse_args()
    if arguments.dealership_advice_strength < 0:
        parser.error("dealership advice strength must be nonnegative")
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES", "0" if arguments.experiment else "0,2,3,4"
    )
    run(
        arguments.data,
        arguments.output,
        arguments.experiment,
        arguments.dealership_advice_strength,
    )
