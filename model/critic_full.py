"""Joint imitation and payment-critic training on five GPUs."""

import json
import pathlib
import time

import h5py
import numpy
import tensorflow as tf

import feature_vector

from . import critic, critic_data, critic_utility, objectives, records, train

PAYMENT_LOSS_WEIGHT = 0.1
BATCH_SIZE = 5760
PRODUCTION_EXPOSURES = 300852 * 5120
PHASES = {
    "full": (
        ("natural_start", "natural", 0.5),
        ("focused", "focused", 4),
        ("natural_finish", "natural", 0.5),
    ),
    "experiment": (
        ("natural_start", "natural", 3000),
        ("focused", "focused", 6000),
        ("natural_finish", "natural", 3000),
    ),
    "smoke": (
        ("natural_start", "natural", 1),
        ("focused", "focused", 2),
        ("natural_finish", "natural", 1),
    ),
}


def phase_schedule(mode, summary, batch_size=BATCH_SIZE):
    """Half production exposure, retaining the requested pass proportions."""
    if mode != "full":
        return PHASES[mode]
    natural, focused = numpy.sum(summary["partition_counts"][:7], axis=0)
    total, remainder = divmod(PRODUCTION_EXPOSURES // 2, batch_size)
    assert remainder == 0, "batch must divide the half-production exposure"
    positions = numpy.array(
        [
            passes * (focused if sampling == "focused" else natural)
            for _, sampling, passes in PHASES["full"]
        ]
    )
    natural_steps = int(numpy.rint(total * positions[0] / positions.sum()))
    return (
        ("natural_start", "natural", natural_steps),
        ("focused", "focused", total - 2 * natural_steps),
        ("natural_finish", "natural", natural_steps),
    )


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


def objective_metrics(inputs, outputs, utility=None):
    means = objectives.MaskedMeans()
    objective_terms(inputs, outputs, mean=means.collect, utility=utility)
    means.reduce()
    return objective_terms(inputs, outputs, mean=means.read, utility=utility)


def objective_terms(inputs, outputs, mean=objectives.mean_valid, utility=None):
    loss, metrics = critic.objective_metrics(outputs, inputs[2], mean=mean)
    metrics["payment_auxiliary_loss"] = loss
    loss *= PAYMENT_LOSS_WEIGHT
    terms, board_metrics = train.board_loss_terms(
        inputs[0], inputs[1], outputs, mean=mean
    )
    metrics.update(terms)
    metrics.update(board_metrics)
    specialists = train.GroupedModel.specialist_loss(
        features=inputs[0],
        labels=inputs[1],
        outputs=outputs,
        metrics=metrics,
        mean=mean,
    )
    metrics["board_loss"] = (
        tf.add_n(
            [value for name, value in terms.items() if name != "policy_loss"]
        )
        / 7
    )
    metrics["policy_entropy"] = objectives.policy_entropy(
        inputs[1], outputs, mean=mean
    )
    loss += (
        terms["policy_loss"]
        + objectives.LOSS_GROUP_WEIGHTS["board"] * metrics["board_loss"]
        - objectives.POLICY_ENTROPY_WEIGHT * metrics["policy_entropy"]
        + tf.add_n(
            [
                objectives.LOSS_GROUP_WEIGHTS[name] * value
                for name, value in specialists.items()
            ]
        )
    )
    metrics["policy_guidance_adjustment"] = tf.constant(0.0)
    if utility is not None:
        nll, valid = objectives.policy_nll(inputs[1], outputs)
        guidance = objectives.awr_policy_terms(
            nll, utility["advantages"], valid & utility["enabled"], mean=mean
        )
        metrics.update(
            {"utility_" + name: value for name, value in guidance.items()}
        )
        mix = tf.cast(utility["enabled"], tf.float32) * (
            utility["weight"] / (1 + utility["weight"])
        )
        metrics["policy_guidance_adjustment"] = mix * (
            guidance["loss"] - terms["policy_loss"]
        )
        loss += metrics["policy_guidance_adjustment"]
        metrics["utility_weight"] = tf.cast(utility["weight"], tf.float32)
        metrics["utility_policy_mix"] = mix
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
    metrics["joint_loss"] = loss
    return loss, metrics


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


def restore_policy_heads(actor, path):
    restored = {}
    dense_weights = []
    with h5py.File(path) as source:

        def restore(name, group):
            if name.startswith("actor/layers/dense") and name.endswith("/vars"):
                dense_weights.append([group["0"][:], group["1"][:]])
            if name.endswith("/vars") and group.attrs["name"] in (
                "attack_residual",
                "defense_residual",
                "attack_predictions",
                "defense_predictions",
                "attack_decisions",
                "defense_decisions",
                "final_pass",
                "final_scores",
            ):
                layer = actor.get_layer(group.attrs["name"])
                values = [group[str(index)][:] for index in range(len(group))]
                layer.set_weights(values)
                for actual, expected in zip(layer.get_weights(), values):
                    numpy.testing.assert_array_equal(actual, expected)
                restored[layer.name] = [list(value.shape) for value in values]

        source.visititems(restore)
    assert len(restored) == 8
    # Keep the six previous board auxiliaries and their fusion projection.
    # Remove only the old 4*4*4*50 payment-logit columns from that projection;
    # payments are now learned once by the unified critic.
    for shape in (
        (512, 12),
        (512, 408),
        (512, 555),
        (512, 92),
        (512, 16),
        (512, 4),
        (1599, 128),
    ):
        (layer,) = [
            layer
            for layer in actor.layers
            if isinstance(layer, tf.keras.layers.Dense)
            and layer.name != "discard_origin_scores"
            and tuple(layer.kernel.shape) == shape
        ]
        (values,) = [
            values
            for values in dense_weights
            if values[0].shape
            == ((4799, 128) if shape == (1599, 128) else shape)
        ]
        if shape == (1599, 128):
            values = [
                numpy.concatenate([values[0][:1487], values[0][4687:]], axis=0),
                values[1],
            ]
        layer.set_weights(values)
        for actual, expected in zip(layer.get_weights(), values):
            numpy.testing.assert_array_equal(actual, expected)
        restored[layer.name] = [list(value.shape) for value in values]
    assert len(restored) == 15
    return restored


def validate(args, network, strategy, evaluate, writer, step):
    root = pathlib.Path(args.output)
    (root / "status.json").write_text(
        json.dumps({"stage": "validation", "updates": step})
    )
    validation = critic_data.dataset(
        root=args.data,
        sampling="natural",
        partition="validation",
        batch_size=640,
        repeat=False,
    ).take(2 if (args.mode == "smoke") else 1000000 // 640)
    values = []
    validation = validation.map(
        critic_data.parse_batch, num_parallel_calls=2
    ).prefetch(2)
    for raw in strategy.experimental_distribute_dataset(validation):
        values.append(
            {name: float(value) for name, value in evaluate(raw).items()}
        )
    assert values
    measured = {
        name: float(
            numpy.sum([row[name] for row in values])
            if name.endswith(("_count", "_correct", "_nll_sum"))
            else numpy.mean([row[name] for row in values])
        )
        for name in values[0]
    }
    measured = policy_rates(measured)
    measured["updates"] = step
    measured["examples"] = len(values) * 640
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
    assert len(devices) == 5
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
    utility_settings = (
        json.loads(pathlib.Path(args.utility).read_text())
        if args.utility is not None
        else None
    )
    if args.mode == "full":
        assert utility_settings is not None, "full training requires AWR"
    if utility_settings is not None:
        assert utility_settings["every"] == 1, "AWR must run every update"
        assert utility_settings["reference_weight"] >= 0
        assert utility_settings["reference_every"] >= 1
        assert utility_settings["temperature"] > 0
        assert utility_settings["placement_scale"] > 0
        assert utility_settings["variance_scale"] > 0
        correlation = numpy.asarray(utility_settings["correlation"])
        assert correlation.shape == (19, 19)
        numpy.testing.assert_allclose(correlation, correlation.T, atol=1e-6)
        numpy.testing.assert_allclose(numpy.diag(correlation), 1, atol=1e-6)
        assert numpy.linalg.eigvalsh(correlation).min() >= -1e-6
    strategy = tf.distribute.MirroredStrategy()
    assert args.batch_size > 0
    assert args.batch_size % strategy.num_replicas_in_sync == 0
    phases = phase_schedule(args.mode, summary, args.batch_size)
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
    schedule = train.ReferenceSchedule(total_steps, args.batch_size)
    plan = {
        "data": args.data,
        "initialization": args.initialization,
        "policy_initialization": args.policy_initialization,
        "objective": "joint_imitation_and_payment_prediction",
        "payment_loss_weight": PAYMENT_LOSS_WEIGHT,
        "point_loss": "squared_signed_log_error_of_expected_net_payment",
        "point_loss_scale": 10000,
        "policy_accuracy": "tile_action_preserve_red_and_riichi",
        "utility": utility_settings,
        "utility_blend": "reference blend divided by reference cadence",
        "specialist_objective": "allocated_imitation_global_entropy_and_regret",
        "specialist_entropy_weight": objectives.SPECIALIST_ENTROPY_WEIGHT,
        "never_tenpai_regret": objectives.NEVER_TENPAI_REGRET,
        "specialist_final_imitation_routing": True,
        "full_pass_proportions": [0.5, 4, 0.5],
        "production_position_exposures": PRODUCTION_EXPOSURES,
        "full_exposure_fraction": 0.5,
        "policy": "previous attack/defense heads and final pass",
        "payment_predictors": 1,
        "replicas": 5,
        "batch_per_replica": args.batch_size // strategy.num_replicas_in_sync,
        "gradient_accumulation": 1,
        "effective_batch": args.batch_size,
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
        "validation_examples": 1000000,
        "loss_normalization": "valid-label means across the full global batch",
    }
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
        initialization = critic.JointPayment()
        initialization(first)
        initialization.load_weights(args.initialization)
        network = critic.JointPolicyPayment()
        network(first)
        network.encoder.set_weights(initialization.encoder.get_weights())
        network.critic.set_weights(initialization.critic.get_weights())
        del initialization
        network.compile_training()
        (root / "policy-restore.json").write_text(
            json.dumps(
                restore_policy_heads(network.actor, args.policy_initialization),
                indent=2,
            )
        )
        encoder_variables = network.encoder.trainable_variables
        head_variables = [
            value
            for value in network.trainable_variables
            if id(value) not in {id(item) for item in encoder_variables}
        ]

        variables = encoder_variables + head_variables
        optimizer = FusedAdamW(
            learning_rate=schedule, weight_decay=1e-4, clipnorm=1.0
        )
        optimizer.build(variables)
        checkpoint = tf.train.Checkpoint(network=network, optimizer=optimizer)

    manager = tf.train.CheckpointManager(checkpoint, str(root / "recovery"), 2)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).assert_consumed()

    def update(inputs):
        inputs = packing.unpack(inputs)
        with tf.GradientTape(persistent=True) as tape:
            outputs = network(inputs)
            utility = None
            if utility_settings is not None:
                with tape.stop_recording():
                    weight = (
                        utility_settings["reference_weight"]
                        * objectives.payoff_weight(
                            optimizer.iterations, total_steps
                        )
                        / 2
                    )
                    # Preserve the reference's average blend while evaluating
                    # the new utility every update instead of periodically.
                    weight /= (
                        utility_settings["reference_every"]
                        + (utility_settings["reference_every"] - 1) * weight
                    )
                    enabled = weight > 0
                    utility = {
                        "enabled": enabled,
                        "weight": weight,
                        "advantages": tf.cond(
                            enabled,
                            lambda: critic_utility.advantages(
                                network, inputs, outputs, utility_settings
                            ),
                            lambda: tf.zeros(tf.shape(inputs[2]["points"])),
                        ),
                    }
            loss, metrics = objective_metrics(inputs, outputs, utility=utility)
            loss = loss / strategy.num_replicas_in_sync
            routed = {
                name: loss
                + (
                    metrics["expert_" + name + "_final_imitation_adjustment"]
                    - metrics["policy_guidance_adjustment"]
                )
                / strategy.num_replicas_in_sync
                for name in ("attack", "defense")
            }
        gradients = objectives.specialist_gradients(
            tape, loss, routed, variables
        )
        del tape
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
    def distributed_update(raw):
        metrics = strategy.run(update, args=(raw,))
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

    def evaluate(inputs):
        return objective_metrics(inputs, network(inputs))[1]

    @tf.function
    def distributed_evaluate(raw):
        metrics = strategy.run(evaluate, args=(raw,))
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
            metrics = distributed_update(next(iterator))
            metrics["learning_rate"] = learning_rate
            assert int(optimizer.iterations.numpy()) == step + 1
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
