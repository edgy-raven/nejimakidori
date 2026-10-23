"""Foundation, AWR prefix, fresh-policy AWR/PPO, then anchored AWR."""

import argparse
import concurrent.futures
import fractions
import json
import math
import pathlib
import re
import time
import types

import numpy
import tensorboard.plugins.custom_scalar.layout_pb2 as scalar_layout
import tensorboard.plugins.custom_scalar.summary as scalar_summary
import tensorflow as tf

from log_dataset import focus_sampling, training_data
from model import actions
from model import actor as actor_layers
from model import beliefs, critic, joint_settlement, losses
from model import network as joint_network
from model import optimization, self_play

PAYMENT_LOSS_WEIGHT = 0.1
AUXILIARY_LOSS_WEIGHT = 0.01


def objective_metrics(inputs, outputs, advantages=None, statistics=None):
    means = losses.MaskedMeans()
    if statistics is None:
        objective_terms(
            inputs, outputs, mean=means.collect, advantages=advantages
        )
        means.reduce()
    else:
        means.values = iter(
            tf.unstack(
                tf.math.divide_no_nan(statistics[:, 0], statistics[:, 1])
            )
        )
    return objective_terms(
        inputs, outputs, mean=means.read, advantages=advantages
    )


def objective_terms(inputs, outputs, mean=losses.mean_valid, advantages=None):
    loss, metrics = critic.objective_metrics(outputs, mean=mean)
    metrics["critic_loss"] = loss
    loss *= PAYMENT_LOSS_WEIGHT
    metrics.update(
        actor_layers.offline_loss_terms(
            inputs, outputs, mean=mean, advantages=advantages
        )
    )
    belief_terms, belief_metrics = beliefs.belief_loss_terms(
        inputs[0], inputs[1], outputs, mean=mean
    )
    metrics.update(belief_terms)
    metrics["belief_loss"] = tf.add_n(list(belief_terms.values())) / 3
    metrics.update(belief_metrics)
    metrics["board_loss"] = AUXILIARY_LOSS_WEIGHT * (
        metrics["completion_yaku_loss"] + metrics["completion_score_loss"]
    )
    loss += (
        metrics["policy_loss"]
        + metrics["policy_prior_loss"]
        + metrics["board_loss"]
        + metrics["belief_loss"]
        - actor_layers.POLICY_ENTROPY_WEIGHT * metrics["policy_entropy"]
    )
    specialist_terms = actor_layers.auxiliary_loss_terms(
        inputs, outputs, mean=mean
    )
    metrics.update(specialist_terms)
    metrics["specialist_loss"] = AUXILIARY_LOSS_WEIGHT * tf.add_n(
        list(specialist_terms.values())
    )
    metrics["scoring_loss"] = AUXILIARY_LOSS_WEIGHT * (
        metrics["score_nll"] + metrics["bonus_han_nll"]
    )
    loss += metrics["specialist_loss"] + metrics["scoring_loss"]
    loss += metrics["policy_guidance_adjustment"]
    metrics["joint_loss"] = loss
    return loss, metrics


def validation_statistics(inputs, outputs):
    means = losses.MaskedMeans()
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
    for name in actions.TRAIN_POLICY_HEADS:
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
            for name in actions.TRAIN_POLICY_HEADS
        )
        / count
    )
    metrics["policy_loss"] = (
        sum(
            metrics[name.removesuffix("_logits") + "_nll_sum"]
            for name in actions.TRAIN_POLICY_HEADS
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


def validation_evaluator(network, strategy):
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

    return distributed_evaluate


def validate(
    args,
    network,
    strategy,
    evaluate,
    writer,
    step,
):
    root = pathlib.Path(args.output)
    summary = json.loads(
        (pathlib.Path(args.data) / "dataset_summary.json").read_text()
    )
    # Include the final partial batch, with at least one row per replica.
    batch_size = validation_batch_size(
        summary["validation_examples"],
        args.batch_size,
        strategy.num_replicas_in_sync,
    )
    validation = training_data.dataset(
        root=args.data,
        sampling=focus_sampling.Sampling.NATURAL,
        partition="validation",
        batch_size=batch_size,
        repeat=False,
    )
    validation = validation.map(
        training_data.parse_batch, num_parallel_calls=2
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
    # Reassemble compound losses from corpus-wide valid-label means.
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
            if name in {
                "policy_loss",
                "imitation_accuracy",
                "imitation_top3_accuracy",
                "payment_nll",
                "belief_loss",
            } or (
                name.startswith(("discard_", "riichi_", "response_", "kan_"))
                and name.endswith(("_accuracy", "_nll"))
            ):
                tf.summary.scalar("validation/" + name, value, step=step)
    writer.flush()
    return measured


def ppo_exposures(args):
    return int(
        (args.awr_exposures - args.awr_only_exposures)
        * fractions.Fraction(str(args.ppo_exposure_ratio))
    )


def supervised_schedule(args):
    # Include partial updates at every block boundary, in both offline phases.
    return optimization.OfflineSchedule(
        total_steps=sum(
            (budget // args.block_exposures)
            * ((args.block_exposures + args.batch_size - 1) // args.batch_size)
            + (budget % args.block_exposures + args.batch_size - 1)
            // args.batch_size
            for budget in (
                args.foundation_exposures,
                args.awr_only_exposures,
                args.awr_exposures - args.awr_only_exposures,
                args.final_awr_exposures,
            )
        ),
        peak=args.supervised_learning_rate,
    )


class Trainer:
    def __init__(self, *, network, args, first):
        self.network = network
        self.args = args
        self.packing = training_data.PackedBatch(first)
        self.feature_packing = training_data.PackedBatch(first[0])
        self.policy_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=args.ppo_learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0,
        )
        self.auxiliary_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=supervised_schedule(args),
            weight_decay=1e-4,
            clipnorm=1.0,
        )
        self.policy_optimizer.build(network.trainable_variables)
        self.auxiliary_optimizer.build(network.trainable_variables)
        if args.final_awr_exposures:
            reference = joint_network.JointPolicyPayment()
            reference(first)
            reference.compile_training()
            self.anchor = reference.actor
            self.anchor.trainable = False
            self.anchor_ready = tf.Variable(False, trainable=False)

    @tf.function(reduce_retracing=True)
    def predict(self, batch):
        return actor_layers.distribution(
            self.network.actor(
                self.feature_packing.unpack(batch["features"]), training=False
            ),
            batch["head"],
            batch["legal"],
        )

    @tf.function(reduce_retracing=True)
    def baseline(self, features):
        features = self.feature_packing.unpack(features)
        outputs, state = self.network.candidate_context(
            features, self.network.actor(features, training=False)
        )
        return (
            critic.action_values(self.network, features, outputs, state)[2]
            / 100.0
        )

    @tf.function(reduce_retracing=True)
    def policy_update(self, batch, inputs, count):
        inputs = self.packing.unpack(inputs)
        features = self.feature_packing.unpack(batch["features"])
        learning_rate = tf.identity(self.policy_optimizer.learning_rate)
        with tf.GradientTape() as tape:
            outputs, state = self.network.candidate_context(
                features, self.network.actor(features, training=False)
            )
            recorded_inputs = self.network.chosen_inputs(
                features,
                outputs,
                state,
                critic.chosen_actions(
                    features,
                    {
                        name: tf.where(
                            batch["head"] == head, batch["chosen"], -1
                        )
                        for head, name in enumerate(actions.TRAIN_POLICY_HEADS)
                    },
                ),
            )
            statistics = self.network.critic.supervised_statistics(
                recorded_inputs,
                joint_settlement.payment_labels(
                    batch["payment_points"], recorded_inputs
                ),
                valid=batch["critic_valid"]
                & (tf.range(tf.shape(batch["chosen"])[0]) < count),
            )
            current = actor_layers.distribution(
                outputs, batch["head"], batch["legal"]
            )[:count]
            belief_outputs = self.network.actor(inputs[0], training=False)

            def terms(mean):
                belief_terms, _ = beliefs.belief_loss_terms(
                    *tf.nest.map_structure(
                        lambda value: value[:count],
                        (inputs[0], inputs[1], belief_outputs),
                    ),
                    mean=mean,
                )
                return {
                    **{
                        name: value
                        for name, value in critic.settlement_metrics(
                            statistics, mean
                        ).items()
                        if name not in ("score_nll", "bonus_han_nll")
                    },
                    "critic_valid_fraction": mean(
                        tf.cast(batch["critic_valid"][:count], tf.float32),
                        tf.ones(count),
                    ),
                    **actor_layers.online_loss_terms(
                        outputs=outputs,
                        batch=batch,
                        current=current,
                        count=count,
                        clip=self.args.clip,
                        dual_clip=self.args.dual_clip,
                        mean=mean,
                    ),
                    **belief_terms,
                    "belief_loss": tf.add_n(list(belief_terms.values())) / 3,
                }

            means = losses.MaskedMeans()
            terms(means.collect)
            means.reduce()
            metrics = terms(means.read)
            loss = (
                metrics["policy"]
                + self.args.prior_weight * metrics["policy_prior_loss"]
                + PAYMENT_LOSS_WEIGHT * metrics["payment_nll"]
                + metrics["belief_loss"]
                - self.args.entropy * metrics["entropy"]
            ) / tf.distribute.get_strategy().num_replicas_in_sync
        optimization.apply_gradients(
            self.policy_optimizer,
            tape,
            loss,
            self.network.trainable_variables,
        )
        return {**metrics, "learning_rate": learning_rate}

    @tf.function(reduce_retracing=True)
    def auxiliary_update(self, inputs, awr, train=True, anchored=False):
        learning_rate = tf.identity(self.auxiliary_optimizer.learning_rate)
        inputs = self.packing.unpack(inputs)
        with tf.GradientTape() as tape:
            outputs = self.network(inputs)
            advantages = None
            if awr:
                with tape.stop_recording():
                    advantages = critic.advantages(
                        self.network,
                        inputs,
                        outputs,
                        self.args.awr_temperature,
                    )
            loss, metrics = objective_metrics(
                inputs, outputs, advantages=advantages
            )
            if awr:
                loss -= (
                    self.args.awr_entropy - actor_layers.POLICY_ENTROPY_WEIGHT
                ) * metrics["policy_entropy"]
            loss += (self.args.prior_weight - 1.0) * metrics[
                "policy_prior_loss"
            ]
            if anchored:
                with tape.stop_recording():
                    size = tf.shape(inputs[0]["decision_phase"])[0]
                    selected = tf.random.shuffle(tf.range(size))[
                        : tf.maximum(1, size // 4)
                    ]
                    reference = self.anchor(
                        tf.nest.map_structure(
                            lambda value: tf.gather(value, selected), inputs[0]
                        ),
                        training=False,
                    )
                metrics["anchor_kl"] = actor_layers.decision_kl(
                    reference,
                    {
                        name: tf.gather(outputs[name], selected)
                        for name in actions.TRAIN_POLICY_HEADS
                    },
                    tf.nest.map_structure(
                        lambda value: tf.gather(value, selected), inputs[1]
                    ),
                )
                metrics["weighted_anchor_loss"] = (
                    self.args.final_awr_kl * metrics["anchor_kl"]
                )
                loss += metrics["weighted_anchor_loss"]
            metrics["joint_loss"] = loss
            scaled_loss = (
                loss / tf.distribute.get_strategy().num_replicas_in_sync
            )
        if train:
            optimization.apply_gradients(
                self.auxiliary_optimizer,
                tape,
                scaled_loss,
                self.network.trainable_variables,
            )
        return {
            "auxiliary": loss,
            "learning_rate": learning_rate,
            **{
                "imitation_"
                + suffix: tf.add_n(
                    [
                        metrics[name.removesuffix("_logits") + "_" + suffix]
                        for name in actions.TRAIN_POLICY_HEADS
                    ]
                )
                for suffix in ("count", "correct", "top3_correct")
            },
            **{
                name: value
                for name, value in metrics.items()
                if not name.endswith(("_count", "_correct", "_nll_sum"))
            },
        }


class DistributedTrainer:
    """Shard packed CPU buffers; reduce metrics and concatenate predictions."""

    def __init__(self, trainer, strategy):
        self.trainer = trainer
        self.strategy = strategy
        self.packing = trainer.packing
        self.feature_packing = trainer.feature_packing
        self.policy_optimizer = trainer.policy_optimizer
        self.auxiliary_optimizer = trainer.auxiliary_optimizer
        if trainer.args.final_awr_exposures:
            self.anchor = trainer.anchor
            self.anchor_ready = trainer.anchor_ready
        self.evaluate = validation_evaluator(trainer.network, strategy)

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
    def update(self, batch, inputs, count, awr, train=True, anchored=False):
        if awr is None:
            metrics = self.strategy.run(
                self.trainer.policy_update.python_function,
                args=(batch, inputs, count),
            )
        else:
            metrics = self.strategy.run(
                self.trainer.auxiliary_update.python_function,
                args=(inputs, awr, train, anchored),
            )
        packed = self.strategy.run(
            lambda values: tf.stack(tuple(values.values())), args=(metrics,)
        )
        values = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, packed, None)
        metrics = dict(zip(metrics, tf.unstack(values)))
        if awr is not None:
            count = metrics.pop("imitation_count")
            metrics["imitation_accuracy"] = tf.math.divide_no_nan(
                metrics.pop("imitation_correct"), count
            )
            metrics["imitation_top3_accuracy"] = tf.math.divide_no_nan(
                metrics.pop("imitation_top3_correct"), count
            )
        return metrics

    def auxiliary_update(self, inputs, awr, train=True, anchored=False):
        return self.update(
            batch=None,
            inputs=self.distribute(inputs),
            count=None,
            awr=awr,
            train=train,
            anchored=anchored,
        )

    def policy_update(self, batch, inputs, count):
        size = len(batch["chosen"]) // self.strategy.num_replicas_in_sync
        valid = self.strategy.experimental_distribute_values_from_function(
            lambda ctx: tf.clip_by_value(
                count - ctx.replica_id_in_sync_group * size, 0, size
            )
        )
        return self.update(
            self.distribute(batch), self.distribute(inputs), valid, None
        )

    @tf.function
    def prediction(self, batch, baseline):
        values = self.strategy.run(
            (
                self.trainer.baseline.python_function
                if baseline
                else self.trainer.predict.python_function
            ),
            args=(batch,),
        )
        return self.strategy.gather(values, axis=0)

    def baseline(self, features):
        return self.prediction(self.distribute(features), True)

    def predict(self, batch):
        return self.prediction(
            self.distribute(
                {name: batch[name] for name in ("features", "head", "legal")}
            ),
            False,
        )


def offline_stream(args, seed, packing, skip=0):
    dataset = training_data.dataset(
        root=args.data,
        sampling=focus_sampling.Sampling.MIXED,
        partition="train",
        batch_size=args.batch_size,
        repeat=True,
        seed=seed,
    )
    if skip:
        dataset = dataset.unbatch().skip(skip).batch(args.batch_size)
    return iter(
        dataset.map(training_data.parse_batch, num_parallel_calls=2)
        .map(packing.pack, num_parallel_calls=2)
        .prefetch(2)
    )


def auxiliary_updates(
    *, trainer, stream, exposures, batch_size, awr, anchored=False
):
    """Stage one next batch on its GPUs while the current update runs."""
    if exposures <= 0:
        return

    def stage(count):
        with tf.device("/CPU:0"):
            inputs = tf.nest.map_structure(
                lambda value: value[:count], next(stream)
            )
        distributed = trainer.distribute(inputs)

        def copy(ctx):
            with tf.device(
                trainer.strategy.extended.worker_devices[
                    ctx.replica_id_in_sync_group
                ]
            ):
                return tf.nest.map_structure(
                    lambda value: tf.identity(
                        trainer.strategy.experimental_local_results(value)[
                            ctx.replica_id_in_sync_group
                        ]
                    ),
                    distributed,
                )

        return trainer.strategy.experimental_distribute_values_from_function(
            copy
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(stage, min(batch_size, exposures))
        for start in range(0, exposures, batch_size):
            count = min(batch_size, exposures - start)
            inputs = pending.result()
            if start + count < exposures:
                pending = pool.submit(
                    stage, min(batch_size, exposures - start - count)
                )
            yield count, trainer.update(
                batch=None,
                inputs=inputs,
                count=None,
                awr=awr,
                anchored=anchored,
            )


def padded_batch(batch, indices, size):
    """Repeat a valid row for shape stability; callers exclude padded rows."""
    indices = numpy.pad(indices, (0, size - len(indices)), mode="edge")
    return tf.nest.map_structure(lambda value: value[indices], batch)


def cohort_kl(trainer, batch, batch_size):
    count = len(batch["chosen"])
    total = 0.0
    for start in range(0, count, batch_size):
        indices = numpy.arange(start, min(start + batch_size, count))
        part = padded_batch(batch, indices, batch_size)
        total += float(
            actor_layers.forward_kl(
                part["behavior"][: len(indices)],
                trainer.predict(part)[: len(indices)],
            )
        ) * len(indices)
    tf.debugging.assert_all_finite(total, "Nonfinite cohort policy KL")
    return total / count


def ppo_block(trainer, args, stream, batch, remaining):
    """Consume a fresh cohort until its reuse or exposure limit is hit."""
    collected = len(batch["chosen"])
    generator = numpy.random.default_rng(args.seed)
    # Select uniformly before scoring, retaining each decision's hand ending.
    if collected > remaining:
        selected = generator.permutation(collected)[:remaining]
        batch = tf.nest.map_structure(lambda value: value[selected], batch)
    count = len(batch["chosen"])
    batch = {
        **batch,
        "features": trainer.feature_packing.pack_numpy(batch["features"]),
    }
    indices = numpy.arange(count)
    scoring_started = time.monotonic()
    settlements, settlement_rows = numpy.unique(
        batch.pop("settlement"), axis=0, return_inverse=True
    )
    # Project once per distinct hand/seat, without TensorFlow in CPU workers.
    with tf.device("/CPU:0"):
        batch["returns"] = (
            joint_settlement.post_hand_forecast(
                scores=tf.convert_to_tensor(settlements[:, :4], tf.float32),
                ranks=tf.convert_to_tensor(settlements[:, 4:8], tf.int64),
                next_dealer=tf.convert_to_tensor(settlements[:, 8], tf.int64),
                next_hand=tf.convert_to_tensor(settlements[:, 9], tf.int32),
                terminal=tf.convert_to_tensor(settlements[:, 10], tf.bool),
                pot=tf.convert_to_tensor(settlements[:, 11], tf.float32),
            )[0].numpy()[settlement_rows]
            / 100.0
        )
    baseline = numpy.concatenate(
        [
            trainer.baseline(
                padded_batch(batch["features"], part, args.batch_size)
            ).numpy()[: len(part)]
            for part in numpy.split(
                indices, range(args.batch_size, count, args.batch_size)
            )
        ]
    )
    scoring_seconds = time.monotonic() - scoring_started
    advantage = batch["returns"] - baseline
    batch["advantages"] = (advantage - advantage.mean()) / max(
        advantage.std(), 1e-8
    )
    diagnostic_started = time.monotonic()
    diagnostics = {
        "behavior_kl_before": cohort_kl(trainer, batch, args.batch_size)
    }
    diagnostic_seconds = time.monotonic() - diagnostic_started
    used = 0
    update_started = time.monotonic()
    for epoch in range(args.ppo_epochs):
        shuffled = generator.permutation(count)
        for start in range(0, count, args.batch_size):
            selected = shuffled[
                start : start + min(args.batch_size, remaining - used)
            ]
            part = padded_batch(batch, selected, args.batch_size)
            valid = len(selected)
            metrics = trainer.policy_update(
                part, next(stream), tf.constant(valid, tf.int32)
            )
            used += valid
            if used == remaining:
                break
        diagnostic_started = time.monotonic()
        total_kl = cohort_kl(trainer, batch, args.batch_size)
        diagnostics[f"behavior_kl_epoch_{epoch + 1}"] = total_kl
        duration = time.monotonic() - diagnostic_started
        diagnostic_seconds += duration
        update_started += duration
        if used == remaining:
            break
    metrics = {name: float(value) for name, value in metrics.items()}
    metrics.update(diagnostics)
    update_seconds = time.monotonic() - update_started
    metrics["scoring_seconds"] = scoring_seconds
    metrics["update_seconds"] = update_seconds
    metrics["diagnostic_seconds"] = diagnostic_seconds
    metrics["update_positions_per_second"] = used / update_seconds
    metrics["behavior_kl"] = total_kl
    metrics["ppo_epochs"] = epoch + 1
    metrics["collected_decisions"] = collected
    metrics["scored_decisions"] = count
    metrics["trained_exposures"] = used
    metrics["return_mean"] = float(numpy.mean(batch["returns"]))
    metrics["return_std"] = float(numpy.std(batch["returns"]))
    metrics["baseline_mean"] = float(numpy.mean(baseline))
    metrics["advantage_std"] = float(numpy.std(advantage))
    metrics["reuse"] = used / count
    return used, metrics


def collect_cohort(
    *,
    collector,
    args,
    needed,
    seed,
    prior_games,
    prior_rows,
    snapshot_exposures,
):
    started = time.monotonic()
    previous_metrics = dict(collector.metrics)
    batches = []
    collected = games = 0
    while collected < needed:
        count = args.games
        if prior_rows + collected:
            count = min(
                args.games,
                max(
                    1,
                    int(
                        numpy.ceil(
                            1.1
                            * (needed - collected)
                            * (prior_games + games)
                            / (prior_rows + collected)
                        )
                    ),
                ),
            )
        batches.append(collector.collect(count, seed + games))
        collected += len(batches[-1]["chosen"])
        games += count
    batch = (
        batches[0]
        if len(batches) == 1
        else tf.nest.map_structure(
            lambda *values: numpy.concatenate(values), *batches
        )
    )
    seconds = time.monotonic() - started
    counts = {
        name: value - previous_metrics[name]
        for name, value in collector.metrics.items()
    }
    return (
        batch,
        games,
        {
            "collection_seconds": seconds,
            "collection_requests": len(batches),
            "collection_positions_per_second": collected / seconds,
            "collection_games_per_second": games / seconds,
            "collection_seed_start": seed,
            "collection_seed_end": seed + games,
            "collection_snapshot_exposures": snapshot_exposures,
            **{"collection_" + name: value for name, value in counts.items()},
            **{
                "collection_" + name: numerator / denominator
                for name, numerator, denominator in (
                    ("win_pct", 100 * counts["wins"], 4 * counts["hands"]),
                    (
                        "dealin_pct",
                        100 * counts["dealin_hands"],
                        4 * counts["hands"],
                    ),
                    (
                        "riichi_pct",
                        100 * counts["riichi_hands"],
                        4 * counts["hands"],
                    ),
                    (
                        "open_pct",
                        100 * counts["open_hands"],
                        4 * counts["hands"],
                    ),
                    ("draw_pct", 100 * counts["draws"], counts["hands"]),
                    ("tsumo_pct", 100 * counts["tsumo_wins"], counts["wins"]),
                    (
                        "draw_tenpai_pct",
                        100 * counts["exhaustive_tenpai_seats"],
                        4 * counts["exhaustive_draws"],
                    ),
                    (
                        "riichi_win_pct",
                        100 * counts["riichi_wins"],
                        counts["riichi_hands"],
                    ),
                    (
                        "riichi_dealin_pct",
                        100 * counts["riichi_dealins"],
                        counts["riichi_hands"],
                    ),
                    ("mean_win_points", counts["win_points"], counts["wins"]),
                    (
                        "mean_dealin_points",
                        counts["dealin_points"],
                        counts["dealin_hands"],
                    ),
                    (
                        "mean_riichi_turn",
                        counts["riichi_turn_sum"],
                        counts["riichi_hands"],
                    ),
                    (
                        "mean_inference_batch",
                        counts["inference_rows"],
                        counts["inference_calls"],
                    ),
                )
                if denominator
            },
        },
    )


def save_checkpoint(manager):
    manager.save()
    path = pathlib.Path(manager.directory) / "latest.weights.h5"
    manager.checkpoint.network.save_weights(path)
    path.replace(path.parent.parent / "latest.weights.h5")


def offline_phase(args, trainer, progress, manager, report, name, target):
    if int(progress[name]) >= target:
        return
    stream = offline_stream(
        args, args.seed, trainer.packing, int(progress[name])
    )
    while int(progress[name]) < target:
        end = min(target, int(progress[name]) + args.block_exposures)
        for count, metrics in auxiliary_updates(
            trainer=trainer,
            stream=stream,
            exposures=end - int(progress[name]),
            batch_size=args.batch_size,
            awr=name != "foundation",
            anchored=name == "final_awr",
        ):
            progress[name].assign_add(count)
            if int(trainer.auxiliary_optimizer.iterations.numpy()) % 20 == 0:
                report("awr_only" if name == "awr" else name, metrics)
        report("awr_only" if name == "awr" else name, metrics)
        save_checkpoint(manager)


def validate_phase(
    *, args, network, trainer, strategy, progress, writer, report, stage
):
    report("validation", {})
    validate(
        args=types.SimpleNamespace(
            **{**vars(args), "data": args.validation_data or args.data},
        ),
        network=network,
        strategy=strategy,
        evaluate=trainer.evaluate,
        writer=writer,
        step=sum(
            int(progress[name])
            for name in ("foundation", "awr", "ppo", "final_awr")
        ),
    )
    network.save_weights(pathlib.Path(args.output) / (stage + ".weights.h5"))
    report(stage + "_complete", {})


def train_cycles(
    *,
    args,
    trainer,
    collector,
    progress,
    manager,
    ppo_manager,
    report,
):
    while int(progress["awr"]) < args.awr_exposures or int(
        progress["ppo"]
    ) < ppo_exposures(args):
        cycle_started = time.monotonic()
        cycle_exposures = int(progress["awr"]) + int(progress["ppo"])
        target = min(
            args.awr_exposures,
            args.awr_only_exposures
            + (int(progress["cycle"]) + args.cycle_blocks)
            * args.block_exposures,
        )
        ppo_target = int(
            (target - args.awr_only_exposures)
            * fractions.Fraction(str(args.ppo_exposure_ratio))
        )
        if int(progress["awr"]) < target:
            stream = offline_stream(
                args,
                args.seed + 1000 + int(progress["cycle"]),
                trainer.packing,
                int(progress["awr"])
                - args.awr_only_exposures
                - int(progress["cycle"]) * args.block_exposures,
            )
            for count, metrics in auxiliary_updates(
                trainer=trainer,
                stream=stream,
                exposures=target - int(progress["awr"]),
                batch_size=args.batch_size,
                awr=True,
            ):
                progress["awr"].assign_add(count)
                if (
                    int(trainer.auxiliary_optimizer.iterations.numpy()) % 20
                    == 0
                    or int(progress["awr"]) == target
                ):
                    report(
                        "awr",
                        metrics,
                    )
            save_checkpoint(manager)
        while int(progress["ppo"]) < ppo_target:
            remaining = min(
                ppo_target - int(progress["ppo"]), args.cohort_exposures
            )
            seed = args.seed + 100000 + int(progress["games"])
            report("collecting", {})
            batch, games, collection = collect_cohort(
                collector=collector,
                args=args,
                needed=int(numpy.ceil(remaining / args.ppo_epochs)),
                seed=seed,
                prior_games=int(progress["games"]),
                prior_rows=int(progress["collected"]),
                snapshot_exposures=sum(
                    int(progress[name]) for name in ("foundation", "awr", "ppo")
                ),
            )
            stream = offline_stream(args, seed, trainer.packing)
            used, metrics = ppo_block(
                trainer=trainer,
                args=args,
                stream=stream,
                batch=batch,
                remaining=remaining,
            )
            metrics.update(collection)
            progress["ppo"].assign_add(used)
            progress["cohort"].assign_add(1)
            progress["collected"].assign_add(len(batch["chosen"]))
            progress["games"].assign_add(games)
            report("ppo", metrics)
            save_checkpoint(manager)
            if int(progress["ppo"]) == ppo_exposures(args):
                ppo_manager.save()
        progress["cycle"].assign_add(args.cycle_blocks)
        save_checkpoint(manager)
        report(
            "cycle_complete",
            {
                "positions_per_second": (
                    int(progress["awr"])
                    + int(progress["ppo"])
                    - cycle_exposures
                )
                / (time.monotonic() - cycle_started)
            },
        )


def write_plan(root, plan, resume):
    """Require explicit recovery and preserve the optimizer schedule."""
    if (root / "plan.json").exists() and not resume:
        raise ValueError(
            "Existing run requires --resume; warm starts need a new output"
        )
    if resume and not tf.train.latest_checkpoint(str(root / "recovery")):
        raise ValueError("--resume requires an existing recovery checkpoint")
    if (root / "plan.json").exists():
        previous_plan = json.loads((root / "plan.json").read_text())
        if {
            name: value
            for name, value in previous_plan.items()
            if name not in ("games", "cycle_blocks")
        } != {
            name: value
            for name, value in plan.items()
            if name not in ("games", "cycle_blocks")
        }:
            raise ValueError("Changed settings require a new output directory")
        if previous_plan != plan:
            with (root / "plan-changes.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "previous": previous_plan,
                            "current": plan,
                        }
                    )
                    + "\n"
                )
    training_data.write_json(root / "plan.json", plan)


def validate_settings(args):
    """Validate mixed training or an entirely offline AWR budget."""
    if not (
        args.foundation_exposures >= 0
        and args.final_awr_exposures >= 0
        and math.isfinite(args.final_awr_kl)
        and args.final_awr_kl >= 0
        and math.isfinite(args.prior_weight)
        and args.prior_weight >= 0
        and 0 <= args.awr_only_exposures <= args.awr_exposures
        and min(
            args.awr_exposures,
            args.block_exposures,
            args.cycle_blocks,
            args.cohort_exposures,
            args.games,
            args.collection_workers,
            args.batch_size,
            args.ppo_epochs,
        )
        > 0
        and (
            (
                0 < args.ppo_exposure_ratio <= 1
                and ppo_exposures(args) > 0
                and args.awr_only_exposures < args.awr_exposures
            )
            or (
                args.ppo_exposure_ratio == 0
                and args.awr_only_exposures == args.awr_exposures
            )
        )
        and 0 < args.clip < 1
        and math.isfinite(args.dual_clip)
        and args.dual_clip > 1
        and args.awr_temperature > 0
        and args.ppo_learning_rate > 0
        and args.supervised_learning_rate > 0
        and args.entropy >= 0
        and args.awr_entropy >= 0
    ):
        raise ValueError("Invalid training training settings")


def run(args):
    validate_settings(args)
    root = pathlib.Path(args.output)
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise ValueError(
            "Phasic training requires visible GPUs; check CUDA libraries "
            "and select training UUIDs with CUDA_VISIBLE_DEVICES"
        )
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    strategy = tf.distribute.MirroredStrategy()
    if any(
        value % strategy.num_replicas_in_sync
        for value in (
            args.batch_size,
            args.foundation_exposures,
            args.awr_only_exposures,
            args.awr_exposures,
            args.final_awr_exposures,
            args.block_exposures,
            ppo_exposures(args),
        )
    ):
        raise ValueError(
            "Batch and phase budgets must divide evenly across GPUs"
        )
    tf.keras.utils.set_random_seed(args.seed)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        **{
            name: value
            for name, value in vars(args).items()
            if name != "resume"
        },
        "ppo_exposures": ppo_exposures(args),
        "critic": "shared_joint_payment_st3",
        "ppo_target": "observed_hand_settlement_fixed_continuation_st3",
        "ppo_critic": "fresh_chosen_action_payment_likelihood",
        "entropy_objective": "branch_detached_conditional",
        "ppo_ratio_basis": "middle_awr_exposures",
        "final_awr_reference": "end_of_middle_actor",
        "anchor_sample_fraction": 0.25,
        "response_utility": "neutral_awr_weight_conditional_call_forecasts",
        "phase_rule": "initial_foundation_then_fixed_awr_ppo",
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "clipnorm": 1.0,
        "supervised_schedule": supervised_schedule(args).get_config(),
        "replicas": strategy.num_replicas_in_sync,
    }
    write_plan(root, plan, args.resume)
    first = training_data.parse_batch(
        next(
            iter(
                training_data.dataset(
                    root=args.data,
                    sampling=focus_sampling.Sampling.MIXED,
                    partition="train",
                    batch_size=min(
                        args.batch_size // strategy.num_replicas_in_sync, 128
                    ),
                    repeat=True,
                    seed=args.seed,
                )
            )
        )
    )
    with strategy.scope():
        network = joint_network.JointPolicyPayment()
        network(first)
        if args.initial:
            network.load_weights(args.initial)
        network.compile_training()
        trainer = Trainer(
            network=network,
            args=args,
            first=first,
        )
    trainer = DistributedTrainer(trainer, strategy)
    progress = {
        name: tf.Variable(0, dtype=tf.int64, trainable=False)
        for name in (
            "foundation",
            "awr",
            "final_awr",
            "ppo",
            "cycle",
            "cohort",
            "collected",
            "games",
        )
    }
    elapsed = tf.Variable(0.0, dtype=tf.float64, trainable=False)
    checkpoint = tf.train.Checkpoint(
        elapsed_seconds=elapsed,
        network=network,
        policy_optimizer=trainer.policy_optimizer,
        auxiliary_optimizer=trainer.auxiliary_optimizer,
        **(
            dict(anchor=trainer.anchor, anchor_ready=trainer.anchor_ready)
            if args.final_awr_exposures
            else {}
        ),
        **progress,
    )
    manager = tf.train.CheckpointManager(checkpoint, str(root / "recovery"), 2)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).assert_consumed()
    else:
        save_checkpoint(manager)
    writer = tf.summary.create_file_writer(str(root / "tensorboard"))
    started = time.monotonic()
    restored_seconds = float(elapsed)
    budgets = {
        "foundation": args.foundation_exposures,
        "awr": args.awr_exposures,
        "final_awr": args.final_awr_exposures,
        "ppo": plan["ppo_exposures"],
    }
    previous = (started, sum(int(progress[name]) for name in budgets))
    with writer.as_default():
        tf.summary.text("run/plan", json.dumps(plan, indent=2), step=0)
        layout = scalar_layout.Layout()
        for category, charts in {
            "Held-out quality": {
                "Expert agreement": r"validation/imitation_.*accuracy$",
                "Policy NLL": r"validation/policy_loss$",
                "Agreement by decision": (
                    r"validation/(discard_policy|riichi_policy|"
                    r"response_policy|kan_action)_accuracy$"
                ),
                "Focused vs ordinary discards": (
                    r"validation/discard_(focused|ordinary)_accuracy$"
                ),
                "Payment NLL": r"validation/payment_nll$",
                "Belief health": r"validation/belief_loss$",
            },
            "Policy updates": {
                "Sampled expert agreement": (
                    r"(foundation|awr_only|awr|final_awr)/"
                    r"imitation_accuracy$"
                ),
                "Sampled policy NLL": (
                    r"(foundation|awr_only|awr|final_awr)/policy_loss$"
                ),
                "Sampled entropy": r".*/(policy_entropy|entropy)$",
                "AWR effective sample fraction": (
                    r"(awr_only|awr|final_awr)/"
                    r"utility_effective_sample_fraction$"
                ),
                "AWR clipped weight fraction": (
                    r"(awr_only|awr|final_awr)/"
                    r"utility_weight_clipped_fraction$"
                ),
                "PPO cohort KL": r"ppo/behavior_kl$",
                "Final anchor KL": r"final_awr/anchor_kl$",
                "Learning rate": r".*/learning_rate$",
            },
            "Self-play": {
                "Projected return and baseline (ST3 / 100)": (
                    r"ppo/(return_mean|baseline_mean)$"
                ),
                "Hand outcomes (%)": r"ppo/collection_(win|dealin|draw)_pct$",
                "Calling and riichi (%)": r"ppo/collection_(open|riichi)_pct$",
                "Cohort reuse": r"ppo/reuse$",
            },
            "Throughput": {
                "Training window positions/s": (
                    r"runtime/window_positions_per_second$"
                ),
                "Full cycle positions/s": (
                    r"cycle_complete/positions_per_second$"
                ),
                "Collection positions/s": (
                    r"ppo/collection_positions_per_second$"
                ),
                "PPO update positions/s": r"ppo/update_positions_per_second$",
                "Elapsed hours": r"runtime/elapsed_hours$",
            },
            "Progress": {
                "Budget fractions": r"progress/.*_fraction$",
                "Games": r"progress/games$",
            },
        }.items():
            group = layout.category.add(title=category)
            for title, pattern in charts.items():
                group.chart.add(title=title).multiline.tag.append(pattern)
        tf.summary.experimental.write_raw_pb(
            scalar_summary.pb(layout).SerializeToString(), step=0
        )

    scalar_patterns = [
        re.compile(pattern)
        for category in layout.category
        for chart in category.chart
        for pattern in chart.multiline.tag
    ]

    def report(stage, metrics):
        nonlocal previous
        now = time.monotonic()
        elapsed.assign(restored_seconds + now - started)
        exposures = sum(int(progress[name]) for name in budgets)
        status = {
            "stage": stage,
            **{name: int(value.numpy()) for name, value in progress.items()},
            "elapsed_seconds": float(elapsed),
            "total_exposures": exposures,
            "budget_exposures": sum(budgets.values()),
            "window_positions_per_second": (
                (exposures - previous[1]) / (now - previous[0])
            ),
            **{name: float(value) for name, value in metrics.items()},
        }
        training_data.write_json(root / "status.json", status)
        with (root / "progress.jsonl").open("a") as output:
            output.write(json.dumps(status) + "\n")
        with writer.as_default(step=exposures):
            tf.summary.text("run/stage", stage)
            for name, budget in budgets.items():
                if budget:
                    tf.summary.scalar(
                        "progress/" + name + "_fraction",
                        int(progress[name]) / budget,
                    )
            tf.summary.scalar(
                "progress/total_fraction", exposures / sum(budgets.values())
            )
            tf.summary.scalar("progress/games", int(progress["games"]))
            tf.summary.scalar("runtime/elapsed_hours", float(elapsed) / 3600)
            if exposures > previous[1]:
                tf.summary.scalar(
                    "runtime/window_positions_per_second",
                    status["window_positions_per_second"],
                )
            for name, value in metrics.items():
                if any(
                    pattern.fullmatch(stage + "/" + name)
                    for pattern in scalar_patterns
                ):
                    tf.summary.scalar(stage + "/" + name, value)
        previous = (now, exposures)
        writer.flush()

    report("initializing", {})
    if not sum(int(progress[name]) for name in budgets):
        report(
            "initial",
            trainer.auxiliary_update(
                next(offline_stream(args, args.seed, trainer.packing)),
                False,
                train=False,
            ),
        )

    offline_phase(
        args=args,
        trainer=trainer,
        progress=progress,
        manager=manager,
        report=report,
        name="foundation",
        target=args.foundation_exposures,
    )
    if args.foundation_exposures and not int(progress["awr"]):
        validate_phase(
            args=args,
            network=network,
            trainer=trainer,
            strategy=strategy,
            progress=progress,
            writer=writer,
            report=report,
            stage="foundation",
        )
    offline_phase(
        args=args,
        trainer=trainer,
        progress=progress,
        manager=manager,
        report=report,
        name="awr",
        target=args.awr_only_exposures,
    )
    if (
        args.awr_only_exposures
        and int(progress["awr"]) == args.awr_only_exposures
    ):
        validate_phase(
            args=args,
            network=network,
            trainer=trainer,
            strategy=strategy,
            progress=progress,
            writer=writer,
            report=report,
            stage="awr_only",
        )
    if args.ppo_exposure_ratio:
        with strategy.scope():
            predict = self_play.predictor(network.actor)
        with self_play.Collector(predict, args.collection_workers) as collector:
            train_cycles(
                args=args,
                trainer=trainer,
                collector=collector,
                progress=progress,
                manager=manager,
                ppo_manager=tf.train.CheckpointManager(
                    checkpoint, str(root / "ppo-budget"), max_to_keep=1
                ),
                report=report,
            )
    if args.final_awr_exposures and not bool(trainer.anchor_ready.numpy()):
        trainer.anchor.set_weights(network.actor.get_weights())
        trainer.anchor_ready.assign(True)
        network.save_weights(root / "middle.weights.h5")
        save_checkpoint(manager)
    offline_phase(
        args=args,
        trainer=trainer,
        progress=progress,
        manager=manager,
        report=report,
        name="final_awr",
        target=args.final_awr_exposures,
    )
    if args.final_awr_exposures:
        validate_phase(
            args=args,
            network=network,
            trainer=trainer,
            strategy=strategy,
            progress=progress,
            writer=writer,
            report=report,
            stage="final",
        )
    else:
        network.save_weights(root / "final.weights.h5")
    report("complete", {})
    writer.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--initial", help="Optional joint initializer")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Explicitly resume saved state"
    )
    parser.add_argument("--foundation-exposures", type=int, required=True)
    parser.add_argument("--awr-exposures", type=int, required=True)
    parser.add_argument("--awr-only-exposures", type=int, default=0)
    parser.add_argument("--final-awr-exposures", type=int, default=0)
    parser.add_argument("--final-awr-kl", type=float, default=0.1)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--block-exposures", type=int, default=16384)
    parser.add_argument("--cycle-blocks", type=int, default=1)
    parser.add_argument(
        "--cohort-exposures",
        type=int,
        default=1572864,
        help="Maximum PPO exposures before collecting a fresh cohort",
    )
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--collection-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument(
        "--ppo-exposure-ratio",
        type=float,
        default=1.0,
        help="PPO exposures per middle-stage AWR exposure (excludes prefix)",
    )
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=3.0)
    parser.add_argument("--entropy", type=float, default=0.001)
    parser.add_argument("--awr-entropy", type=float, default=0.01)
    parser.add_argument("--validation-data")
    parser.add_argument("--ppo-learning-rate", type=float, required=True)
    parser.add_argument("--supervised-learning-rate", type=float, default=2e-4)
    parser.add_argument("--awr-temperature", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=941)
    return parser


def main():
    run(argument_parser().parse_args())


if __name__ == "__main__":
    main()
