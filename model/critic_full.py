"""Joint imitation and payment-critic training on the selected GPUs."""

import argparse
import json
import pathlib
import time

import numpy
import tensorflow as tf
from log_dataset import critic_data, focus_sampling, records
from model import (
    beliefs,
    critic,
    critic_utility,
    feature_vector,
    objectives,
    rollout_training,
    specialists,
)

PAYMENT_LOSS_WEIGHT = 0.1
AUXILIARY_LOSS_WEIGHT = 0.01
BATCH_SIZE = 5760
PRODUCTION_EXPOSURES = 300852 * 5120
PHASES = {
    "full": None,
    "experiment": 12000,
    "smoke": 4,
}


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


def phase_schedule(mode, batch_size=BATCH_SIZE):
    """A continuous mixture with half the production position exposure."""
    if mode == "full":
        total, remainder = divmod(PRODUCTION_EXPOSURES // 2, batch_size)
        assert remainder == 0, "batch must divide the half-production exposure"
    else:
        total = PHASES[mode]
    return (("mixed", "mixed", total),)


@tf.function(jit_compile=True)
def clip_dense_gradients(gradients, clipnorm):
    return [tf.clip_by_norm(gradient, clipnorm) for gradient in gradients]


class FusedAdamW(tf.keras.optimizers.AdamW):
    """Per-variable clipping and compact, equivalent sparse Adam reduction."""

    def _clip_gradients(self, gradients):
        dense = iter(
            clip_dense_gradients(
                tuple(
                    g for g in gradients if not isinstance(g, tf.IndexedSlices)
                ),
                self.clipnorm,
            )
        )
        return [
            (
                super(FusedAdamW, self)._clip_gradients([gradient])[0]
                if isinstance(gradient, tf.IndexedSlices)
                else next(dense)
            )
            for gradient in gradients
        ]

    def _all_reduce_sum_gradients(self, grads_and_vars):
        # Clipping has already happened. Preserve Keras' sum of per-entry
        # squares for duplicate sparse indices, rather than squaring their sum.
        return super()._all_reduce_sum_gradients(
            [
                (
                    (
                        tf.stack(
                            [
                                tf.math.unsorted_segment_sum(
                                    values, gradient.indices, variable.shape[0]
                                )
                                for values in (
                                    gradient.values,
                                    tf.square(gradient.values),
                                )
                            ]
                        )
                        if isinstance(gradient, tf.IndexedSlices)
                        else gradient
                    ),
                    variable,
                )
                for gradient, variable in grads_and_vars
            ]
        )

    def update_step(self, gradient, variable, learning_rate):
        if gradient.shape.rank == variable.shape.rank + 1:
            index = self._get_variable_index(variable)
            alpha = (
                tf.cast(learning_rate, variable.dtype)
                * tf.sqrt(
                    1
                    - tf.pow(
                        self.beta_2,
                        tf.cast(self.iterations + 1, variable.dtype),
                    )
                )
                / (
                    1
                    - tf.pow(
                        self.beta_1,
                        tf.cast(self.iterations + 1, variable.dtype),
                    )
                )
            )
            self.assign_add(
                self._momentums[index],
                (gradient[0] - self._momentums[index]) * (1 - self.beta_1),
            )
            self.assign_add(
                self._velocities[index],
                (gradient[1] - self._velocities[index]) * (1 - self.beta_2),
            )
            self.assign_sub(
                variable,
                self._momentums[index]
                * alpha
                / (tf.sqrt(self._velocities[index]) + self.epsilon),
            )
            return
        index = self._get_variable_index(variable)
        tf.raw_ops.ResourceApplyAdam(
            var=variable.handle,
            m=self._momentums[index].value.handle,
            v=self._velocities[index].value.handle,
            beta1_power=tf.pow(
                tf.cast(self.beta_1, variable.dtype),
                tf.cast(self.iterations + 1, variable.dtype),
            ),
            beta2_power=tf.pow(
                tf.cast(self.beta_2, variable.dtype),
                tf.cast(self.iterations + 1, variable.dtype),
            ),
            lr=tf.cast(learning_rate, variable.dtype),
            beta1=tf.cast(self.beta_1, variable.dtype),
            beta2=tf.cast(self.beta_2, variable.dtype),
            epsilon=tf.cast(self.epsilon, variable.dtype),
            grad=tf.cast(gradient, variable.dtype),
        )


def objective_metrics(inputs, outputs, utility=None, statistics=None):
    means = objectives.MaskedMeans()
    if statistics is None:
        objective_terms(inputs, outputs, mean=means.collect, utility=utility)
        means.reduce()
    else:
        means.values = iter(
            tf.unstack(
                tf.math.divide_no_nan(statistics[:, 0], statistics[:, 1])
            )
        )
    return objective_terms(inputs, outputs, mean=means.read, utility=utility)


def objective_terms(inputs, outputs, mean=objectives.mean_valid, utility=None):
    loss, metrics = critic.objective_metrics(outputs, inputs[2], mean=mean)
    metrics["critic_loss"] = loss
    loss *= PAYMENT_LOSS_WEIGHT
    metrics["policy_loss"] = objectives.policy_loss(
        inputs[1], outputs, mean=mean
    )
    metrics["completion_yaku_loss"] = mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.maximum(
                tf.cast(inputs[1]["round_completion_yaku"], tf.float32), 0
            ),
            logits=outputs["round_completion_yaku_logits"],
        ),
        inputs[1]["round_completion_yaku"] >= 0,
    )
    metrics["completion_score_loss"] = mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.maximum(
                tf.cast(inputs[1]["round_completion_score"], tf.int32), 0
            ),
            logits=outputs["round_completion_score_logits"],
        ),
        inputs[1]["round_completion_score"] >= 0,
    )
    belief_terms, belief_metrics = beliefs.loss_terms(
        inputs[0], inputs[1], outputs, mean=mean
    )
    metrics.update(belief_terms)
    metrics["belief_loss"] = tf.add_n(list(belief_terms.values())) / 3
    metrics.update(belief_metrics)
    metrics["board_loss"] = AUXILIARY_LOSS_WEIGHT * (
        metrics["completion_yaku_loss"] + metrics["completion_score_loss"]
    )
    metrics["policy_entropy"] = objectives.policy_entropy(
        inputs[1], outputs, mean=mean
    )
    loss += (
        metrics["policy_loss"]
        + metrics["board_loss"]
        + metrics["belief_loss"]
        - objectives.POLICY_ENTROPY_WEIGHT * metrics["policy_entropy"]
    )
    specialist_terms = specialists.loss_terms(inputs, outputs, mean=mean)
    metrics.update(specialist_terms)
    metrics["specialist_loss"] = AUXILIARY_LOSS_WEIGHT * tf.add_n(
        list(specialist_terms.values())
    )
    metrics["scoring_loss"] = AUXILIARY_LOSS_WEIGHT * metrics["score_nll"]
    loss += metrics["specialist_loss"] + metrics["scoring_loss"]
    metrics["policy_guidance_adjustment"] = tf.constant(0.0)
    if utility is not None:
        nll, valid = objectives.policy_nll(inputs[1], outputs)
        guidance = objectives.awr_policy_terms(
            nll, utility["advantages"], valid & utility["enabled"], mean=mean
        )
        metrics.update(
            {"utility_" + name: value for name, value in guidance.items()}
        )
        mix = tf.stop_gradient(tf.cast(utility["mix"], tf.float32))
        metrics["policy_guidance_adjustment"] = mix * (
            guidance["loss"] - metrics["policy_loss"]
        )
        loss += metrics["policy_guidance_adjustment"]
        metrics["utility_policy_mix"] = mix
        # For normalized AWR weights u with E[u]=1, the combined policy
        # weight is (1-m)+m*u. Its second moment is 1+m²*(E[u²]-1).
        metrics["utility_blended_effective_sample_fraction"] = tf.where(
            mix > 0,
            tf.math.divide_no_nan(
                guidance["effective_sample_fraction"],
                (1 - tf.square(mix)) * guidance["effective_sample_fraction"]
                + tf.square(mix),
            ),
            1.0,
        )
    for name in feature_vector.TRAIN_POLICY_HEADS:
        prefix = name.removesuffix("_logits")
        valid = inputs[1][name] >= 0
        ranked_actions = tf.math.top_k(outputs[name], k=3).indices
        recorded_action = tf.cast(inputs[1][name], tf.int32)
        matches = ranked_actions == recorded_action[:, None]
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.maximum(tf.cast(inputs[1][name], tf.int32), 0),
            logits=outputs[name],
        )
        # Counts make validation aggregation exact even when
        # the frequencies of calls and kan differ between batches/replicas.
        metrics[prefix + "_count"] = tf.reduce_sum(tf.cast(valid, tf.float32))
        metrics[prefix + "_correct"] = tf.reduce_sum(
            tf.cast(valid & matches[:, 0], tf.float32)
        )
        metrics[prefix + "_top3_correct"] = tf.reduce_sum(
            tf.cast(
                valid & tf.reduce_any(matches, axis=-1),
                tf.float32,
            )
        )
        metrics[prefix + "_nll_sum"] = tf.reduce_sum(tf.where(valid, nll, 0.0))
        if name == "discard_policy_logits":
            drawn = tf.where(
                inputs[0]["current_draw_is_red"] > 0,
                34 + tf.cast(inputs[0]["current_draw_tile_id"], tf.int32) // 9,
                tf.cast(inputs[0]["current_draw_tile_id"], tf.int32),
            )
            selected_draw = (inputs[0]["current_draw_valid"] > 0) & (
                recorded_action == drawn
            )
            for stratum, selected in {
                "focused": inputs[2]["focused"] > 0,
                "ordinary": inputs[2]["focused"] == 0,
                "drawn_tile": selected_draw,
                "other_tile": ~selected_draw,
            }.items():
                selected &= valid
                metrics[f"discard_{stratum}_count"] = tf.reduce_sum(
                    tf.cast(selected, tf.float32)
                )
                metrics[f"discard_{stratum}_correct"] = tf.reduce_sum(
                    tf.cast(selected & matches[:, 0], tf.float32)
                )
                metrics[f"discard_{stratum}_nll_sum"] = tf.reduce_sum(
                    tf.where(selected, nll, 0.0)
                )
    metrics["joint_loss"] = loss
    return loss, metrics


def validation_statistics(inputs, outputs):
    means = objectives.MaskedMeans()
    metrics = objective_terms(inputs, outputs, mean=means.collect)[1]
    return {
        "statistics": tf.stack(means.statistics),
        "counts": {
            "example_count": tf.cast(
                tf.shape(inputs[2]["points"])[0], tf.float32
            ),
            **{
                name: value
                for name, value in metrics.items()
                if name.endswith(("_count", "_correct", "_nll_sum"))
            },
        },
    }


def policy_rates(metrics):
    correct = count = 0.0
    for name in feature_vector.TRAIN_POLICY_HEADS:
        prefix = name.removesuffix("_logits")
        correct += metrics[prefix + "_correct"]
        count += metrics[prefix + "_count"]
        metrics[prefix + "_accuracy"] = (
            metrics[prefix + "_correct"] / metrics[prefix + "_count"]
            if metrics[prefix + "_count"]
            else 0.0
        )
        metrics[prefix + "_nll"] = (
            metrics[prefix + "_nll_sum"] / metrics[prefix + "_count"]
            if metrics[prefix + "_count"]
            else 0.0
        )
    for stratum in ("focused", "ordinary", "drawn_tile", "other_tile"):
        prefix = "discard_" + stratum
        for suffix, numerator in (("accuracy", "correct"), ("nll", "nll_sum")):
            metrics[prefix + "_" + suffix] = (
                metrics[prefix + "_" + numerator] / metrics[prefix + "_count"]
                if metrics[prefix + "_count"]
                else 0.0
            )
    metrics["imitation_accuracy"] = correct / count
    metrics["imitation_top3_accuracy"] = (
        sum(
            metrics[name.removesuffix("_logits") + "_top3_correct"]
            for name in feature_vector.TRAIN_POLICY_HEADS
        )
        / count
    )
    metrics["policy_loss"] = (
        sum(
            metrics[name.removesuffix("_logits") + "_nll_sum"]
            for name in feature_vector.TRAIN_POLICY_HEADS
        )
        / count
    )
    return metrics


def validation_batch_size(examples, requested, replicas):
    """Keep the last batch nonempty on every replica without dropping rows."""
    assert examples >= replicas
    batch_size = min(requested, examples)
    while 0 < examples % batch_size < replicas:
        batch_size -= 1
    return batch_size


def validate(args, network, strategy, evaluate, writer, step):
    root = pathlib.Path(args.output)
    (root / "status.json").write_text(
        json.dumps({"stage": "validation", "updates": step})
    )
    summary = json.loads(
        (pathlib.Path(args.data) / "dataset_summary.json").read_text()
    )
    # Include the final partial batch, with at least one row per replica.
    batch_size = validation_batch_size(
        summary["validation_examples"],
        args.batch_size,
        strategy.num_replicas_in_sync,
    )
    validation = critic_data.dataset(
        root=args.data,
        sampling="natural",
        partition="validation",
        batch_size=batch_size,
        repeat=False,
    )
    if args.mode == "smoke":
        validation = validation.take(2)
    validation = validation.map(
        critic_data.parse_batch, num_parallel_calls=2
    ).prefetch(2)
    statistics = None
    counts = {}
    for batch in validation:
        count = int(tf.shape(batch[2]["points"])[0])
        # Automatic dataset rebatching uses the nominal batch size even for
        # a short final batch, which can leave later replicas empty.
        raw = strategy.experimental_distribute_values_from_function(
            lambda ctx: tf.nest.map_structure(
                lambda value: value[
                    count
                    * ctx.replica_id_in_sync_group
                    // strategy.num_replicas_in_sync : count
                    * (ctx.replica_id_in_sync_group + 1)
                    // strategy.num_replicas_in_sync
                ],
                batch,
            )
        )
        values = evaluate(raw)
        statistics = (
            values["statistics"]
            if statistics is None
            else statistics + values["statistics"]
        )
        for name, value in values["counts"].items():
            counts[name] = counts.get(name, 0.0) + float(value)
    assert statistics is not None
    # Reassemble compound objectives from corpus-wide valid-label means.
    # One final forward supplies the shape contract; its batch means are
    # replaced by the accumulated statistics, not counted a second time.
    measured = {
        name: float(value) for name, value in evaluate(raw, statistics).items()
    }
    measured.update(counts)
    measured = policy_rates(measured)
    measured["updates"] = step
    measured["examples"] = measured.pop("example_count")
    best = root / "best-policy.json"
    if (
        not best.exists()
        or measured["policy_loss"] < json.loads(best.read_text())["policy_loss"]
    ):
        network.save_weights(root / "best-policy.weights.h5")
        best.write_text(json.dumps(measured, indent=2))
    (root / f"validation-{step}.json").write_text(
        json.dumps(measured, indent=2)
    )
    with writer.as_default():
        for name, value in measured.items():
            tf.summary.scalar("validation/" + name, value, step=step)
    writer.flush()


def run(args):
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(16)
    tf.config.threading.set_inter_op_parallelism_threads(16)
    devices = tf.config.list_physical_devices("GPU")
    assert devices, "Joint training requires CUDA GPUs"
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.keras.utils.set_random_seed(941)
    summary = json.loads(
        (pathlib.Path(args.data) / "dataset_summary.json").read_text()
    )
    records.validate_summary(args.data)
    if not (args.mode == "smoke"):
        assert summary["limit_games"] is None
        assert summary["games"] > 100000
        assert summary["examples"] > 50000000
    assert 0 <= args.awr_mix <= 1
    assert args.awr_temperature > 0
    utility_settings = {
        "mix": args.awr_mix,
        "temperature": args.awr_temperature,
    }
    strategy = tf.distribute.MirroredStrategy()
    assert args.batch_size > 0
    assert args.batch_size % strategy.num_replicas_in_sync == 0
    phases = phase_schedule(args.mode, args.batch_size)
    if args.phase_batches is not None:
        if min(args.phase_batches) < 0 or sum(args.phase_batches) == 0:
            raise ValueError(
                "Phase batches must be nonnegative with a positive total"
            )
        phases = tuple(
            (phase, sampling, batches)
            for (phase, sampling, _), batches in zip(
                phases, args.phase_batches, strict=True
            )
        )
    total_steps = sum(steps for _, _, steps in phases)
    if args.mode == "full":
        assert total_steps * args.batch_size == PRODUCTION_EXPOSURES // 2
    schedule = ReferenceSchedule(total_steps, args.batch_size)
    replay = (
        rollout_training.Replay(
            args.rollouts, args.rollout_overhead, args.rollout_stale_weight
        )
        if args.rollouts is not None
        else None
    )
    plan = {
        "rollouts": (
            {
                "store": args.rollouts,
                "overhead_fraction": args.rollout_overhead,
                "stale_weight": args.rollout_stale_weight,
                "positions": len(replay.positions),
                "sampling": "uniform position, policy, paired screen trial",
                "gradient_scope": (
                    "settlement critic; paired win regret actor/shared encoder"
                ),
                "generation": "external collector; no simulation in trainer",
            }
            if replay is not None
            else None
        ),
        "data": args.data,
        "initialization": (
            "warm_actor_and_critic"
            if args.initial
            else "fresh_actor_and_critic"
        ),
        "initial_weights": args.initial,
        "seed": 941,
        "belief_initialization": args.initial or args.beliefs,
        "count_projection": "weighted_KL_public_hand_sizes_shared_copies",
        "belief_training": (
            "shared encoder; supervised every batch; joint optimizer"
        ),
        "objective": "joint_imitation_and_payment_prediction",
        "payment_loss_weight": PAYMENT_LOSS_WEIGHT,
        "settlement_loss": "categorical_observed_event_score_bank_nll",
        "settlement_support": (
            "49_events_coexisting_ron_shared_tsumo_score_exhaustive_tenpai_"
            "abortive_draw_conditional_deposits_bank"
        ),
        "validation": {
            "sample": summary["validation_sample"],
            "aggregation": "global_valid_label_totals_including_partial_batch",
        },
        "count_projection_convergence": {
            "target_residual": 1e-5,
            "maximum_residual": 1e-4,
            "maximum_iterations": 2048,
            "failure": "raise",
        },
        "immediate_outcomes": (
            "separate_ron_legal_hanfu_amount_and_exhaustive_mask_model"
        ),
        "candidate_constraints": (
            "post_action_meld_fu_open_riichi_ippatsu_deposit_timing"
        ),
        "immediate_supervision": (
            "native_discard_call_discard_chankan_ron_and_current_draw_readiness"
        ),
        "yaku_constraints": "irreversible_public_meld_necessary_conditions",
        "continuation": "fixed_st3_continuation_20261003_seed11",
        "zero_transfer_draw_continuation": (
            "distinct_nobody_all_tenpai_and_abortive_dealer_repeat"
        ),
        "point_loss": None,
        "final_placement_loss": None,
        "policy_accuracy": "tile_action_preserve_red_and_riichi",
        "utility": utility_settings,
        "utility_blend": "linear AWR share ramp; ESS diagnostic only",
        "auxiliary_loss_weight": AUXILIARY_LOSS_WEIGHT,
        "specialist_objective": specialists.CONTRACT,
        "yaku_value_link": (
            "paired_tenpai_takame_yaku_score_and_routes_condition_joint_han_fu"
        ),
        "sampling_mixture": {"natural": 0.75, "focused": 0.25},
        "focus_sampling": focus_sampling.CONTRACT,
        "focus_category_shares": summary["focus_category_shares"],
        "production_position_exposures": PRODUCTION_EXPOSURES,
        "full_exposure_fraction": 0.5,
        "policy": (
            "final AWR policy with directly supervised attack/defense branches"
        ),
        "payment_predictors": 1,
        "replicas": strategy.num_replicas_in_sync,
        "batch_per_replica": args.batch_size // strategy.num_replicas_in_sync,
        "gradient_accumulation": 1,
        "effective_batch": args.batch_size,
        "critic_statistics_chunk": critic.STATISTICS_CHUNK,
        "critic_batch_per_replica": args.batch_size
        // strategy.num_replicas_in_sync,
        "critic_position_exposures": args.batch_size * total_steps,
        "critic_sampling": "all_policy_positions",
        "critic_validation": "all_validation_positions",
        "awr_action_chunk": critic_utility.ACTION_CHUNK,
        "awr_position_chunk": critic_utility.POSITION_CHUNK,
        "awr_payoff_sharing": "identical_public_context_per_replica_batch",
        "score_target_encoding": "CPU_group_class_and_scenario_count",
        "rollout_policy_objective": "paired_win_regret_wilson95",
        "phase_updates": {phase: steps for phase, _, steps in phases},
        "learning_rate_schedule": {
            **schedule.get_config(),
            "kind": "warmup_stable_cosine_decay",
            "warmup_steps": schedule.warmup_steps,
            "decay_start": schedule.decay_start,
            "final_fraction": 0.1,
            "scope": "global; shared encoder and heads",
        },
        "policy_ramp": {"start_fraction": 0.1, "end_fraction": 0.6},
        "position_exposures": args.batch_size * total_steps,
        "reference": "full-20260912-optimized",
        "mode": args.mode,
        "resume_data": "fresh shuffle seeded by restored update; no seeking",
        "corpus_games": summary["games"],
        "corpus_positions": summary["examples"],
        "validation_examples": (
            min(
                summary["validation_examples"],
                2
                * validation_batch_size(
                    summary["validation_examples"],
                    args.batch_size,
                    strategy.num_replicas_in_sync,
                ),
            )
            if args.mode == "smoke"
            else summary["validation_examples"]
        ),
        "loss_normalization": "valid-label means across the full global batch",
    }
    if (root / "plan.json").exists():
        assert json.loads((root / "plan.json").read_text()) == plan
    (root / "status.json").write_text(json.dumps({"stage": "initializing"}))
    (root / "plan.json").write_text(json.dumps(plan, indent=2))
    stream = critic_data.dataset(
        root=args.data,
        sampling="natural",
        partition="train",
        batch_size=args.batch_size,
        repeat=True,
    )
    first = critic_data.parse_batch(next(iter(stream))[:128])
    packing = critic_data.PackedBatch(first)
    with strategy.scope():
        network = critic.JointPolicyPayment()
        network(first)
        if args.initial:
            network.load_weights(args.initial)
        else:
            network.beliefs.load_weights(args.beliefs)
        network.compile_training()
        variables = network.trainable_variables
        optimizer = FusedAdamW(
            learning_rate=schedule, weight_decay=1e-4, clipnorm=1.0
        )
        optimizer.build(variables)
        checkpoint = tf.train.Checkpoint(network=network, optimizer=optimizer)

    manager = tf.train.CheckpointManager(checkpoint, str(root / "recovery"), 2)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).assert_consumed()

    def update(inputs, replay_batch):
        inputs = packing.unpack(inputs)
        replay_inputs = rollout_training.candidate_inputs(network, replay_batch)
        replica = tf.distribute.get_replica_context()
        with tf.GradientTape() as tape:
            outputs = network(inputs)
            utility = None
            if utility_settings["mix"] > 0:
                with tape.stop_recording():
                    mix = (
                        utility_settings["mix"]
                        * objectives.payoff_weight(
                            optimizer.iterations, total_steps
                        )
                        / 2
                    )
                    enabled = mix > 0
                    utility = {
                        "enabled": enabled,
                        "mix": mix,
                        "advantages": tf.cond(
                            enabled,
                            lambda: critic_utility.advantages(
                                network, inputs, outputs, utility_settings
                            ),
                            lambda: tf.zeros(tf.shape(inputs[2]["points"])),
                        ),
                    }
            loss, metrics = objective_metrics(inputs, outputs, utility=utility)
            replay_loss, replay_metrics = rollout_training.loss(
                network.critic, replay_inputs, args.rollout_stale_weight
            )
            loss += PAYMENT_LOSS_WEIGHT * replay_loss
            metrics.update(replay_metrics)
            policy_regret, policy_metrics = rollout_training.policy_loss(
                network.actor, replay_batch, args.rollout_stale_weight
            )
            loss += policy_regret
            metrics.update(policy_metrics)
            loss = loss / replica.num_replicas_in_sync
        gradients = tape.gradient(loss, variables)
        assert all(value is not None for value in gradients), [
            variable.path
            for variable, gradient in zip(variables, gradients)
            if gradient is None
        ]
        tf.debugging.assert_all_finite(
            tf.linalg.global_norm(gradients), "nonfinite gradient norm"
        )
        optimizer.apply_gradients(zip(gradients, variables))
        return metrics

    @tf.function
    def distributed_update(raw, replay_batch=None):
        metrics = strategy.run(update, args=(raw, replay_batch))
        packed = strategy.run(
            lambda values: tf.stack(tuple(values.values())), args=(metrics,)
        )
        reduced = strategy.reduce(tf.distribute.ReduceOp.MEAN, packed, None)
        return {
            name: value
            * (
                strategy.num_replicas_in_sync
                if name.endswith(("_count", "_correct", "_nll_sum"))
                else 1
            )
            for name, value in zip(metrics, tf.unstack(reduced))
        }

    def evaluate(inputs, statistics=None):
        outputs = network(inputs)
        if statistics is not None:
            return objective_metrics(inputs, outputs, statistics=statistics)[1]
        return validation_statistics(inputs, outputs)

    @tf.function
    def distributed_evaluate(raw, statistics=None):
        values = strategy.run(evaluate, args=(raw, statistics))
        return tf.nest.map_structure(
            lambda value: strategy.reduce(
                (
                    tf.distribute.ReduceOp.SUM
                    if statistics is None
                    else tf.distribute.ReduceOp.MEAN
                ),
                value,
                None,
            ),
            values,
        )

    writer = tf.summary.create_file_writer(str(root / "tensorboard"))

    started = time.monotonic()
    completed = int(optimizer.iterations.numpy())
    if completed == 0:
        validate(
            args=args,
            network=network,
            strategy=strategy,
            evaluate=distributed_evaluate,
            writer=writer,
            step=0,
        )
    phase_start = 0
    probes = [
        network.encoder.get_layer("hand_bank").tiles.kernel,
        network.encoder.get_layer("board_projection").kernel,
        network.critic.state.kernel,
        network.beliefs.heads[beliefs.HEADS[0]].kernel,
        network.actor.get_layer("attack_residual").kernel,
        network.actor.get_layer("defense_residual").kernel,
        network.actor.get_layer("final_pass").kernel,
    ]
    before = [value.numpy().copy() for value in probes]
    for phase, sampling, updates in phases:
        if completed >= phase_start + updates:
            phase_start += updates
            continue
        dataset = critic_data.dataset(
            root=args.data,
            sampling=sampling,
            partition="train",
            batch_size=args.batch_size,
            repeat=True,
            seed=941 + max(completed, phase_start),
        )
        # Restore optimizer progress, but start a fresh shuffled data stream.
        dataset = (
            dataset.map(critic_data.parse_batch, num_parallel_calls=2)
            .map(packing.pack, num_parallel_calls=2)
            .prefetch(2)
        )
        iterator = iter(strategy.experimental_distribute_dataset(dataset))
        for step in range(max(completed, phase_start), phase_start + updates):
            learning_rate = float(schedule(step))
            batch_started = time.monotonic()
            replay_batch = (
                replay.distributed_batch(strategy, step)
                if replay is not None
                else None
            )
            metrics = distributed_update(next(iterator), replay_batch)
            # Synchronize before timing so queued GPU work is charged here.
            assert int(optimizer.iterations.numpy()) == step + 1
            if replay is not None:
                replay.budget.record(
                    time.monotonic() - batch_started, replay_batch is not None
                )
                metrics.update(replay.budget.metrics())
            metrics["learning_rate"] = learning_rate
            if step == completed:
                changes = [
                    float(numpy.max(numpy.abs(value.numpy() - old)))
                    for value, old in zip(probes, before)
                ]
                assert all(value > 0 for value in changes)
                (root / "gradient-audit.json").write_text(json.dumps(changes))
            if (step + 1) % 20 == 0 or args.mode == "smoke":
                status = {
                    "stage": "training",
                    "phase": phase,
                    "updates": step + 1,
                    "target_updates": total_steps,
                    "positions": (step + 1) * args.batch_size,
                    "elapsed_seconds": time.monotonic() - started,
                }
                (root / "status.json").write_text(json.dumps(status, indent=2))
                metrics = policy_rates(
                    {name: float(value) for name, value in metrics.items()}
                )
                with writer.as_default():
                    for name, value in metrics.items():
                        tf.summary.scalar("train/" + name, value, step=step + 1)
                writer.flush()
                print(status, flush=True)
            if (step + 1) % 1000 == 0 or step + 1 == phase_start + updates:
                manager.save(checkpoint_number=step + 1)
                network.beliefs.save_weights(root / "beliefs.weights.h5")
                network.save_weights(root / "latest.weights.h5")
            if (
                (step + 1) % 25071 == 0
                or (step + 1) % 5000 == 0
                or step + 1 == 1000
                or step + 1 == phase_start + updates
            ):
                validate(
                    args=args,
                    network=network,
                    strategy=strategy,
                    evaluate=distributed_evaluate,
                    writer=writer,
                    step=step + 1,
                )
        phase_start += updates
    if replay is not None:
        replay.close()
        (root / "rollout-budget.json").write_text(
            json.dumps(replay.budget.metrics(), indent=2)
        )
    network.save_weights(root / "final.weights.h5")
    (root / "status.json").write_text(
        json.dumps(
            {
                "stage": "complete",
                "updates": int(optimizer.iterations.numpy()),
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--initial", help="Joint weights; fresh optimizer and schedule"
    )
    parser.add_argument(
        "--beliefs", required=True, help="Initial persistent belief weights"
    )
    parser.add_argument(
        "--awr-mix",
        type=float,
        default=1.0,
        help="Final AWR share after the 10%%-60%% ramp; 0 disables AWR",
    )
    parser.add_argument("--awr-temperature", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--mode", choices=tuple(PHASES), default="full")
    parser.add_argument(
        "--phase-batches",
        nargs=1,
        type=int,
        metavar="MIXED",
        help="Total mixed-stream batch budget, not additional batches",
    )
    parser.add_argument(
        "--rollouts", help="SQLite one-hand replay store; enables replay"
    )
    parser.add_argument("--rollout-overhead", type=float, default=0.1)
    parser.add_argument("--rollout-stale-weight", type=float, default=0.5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
