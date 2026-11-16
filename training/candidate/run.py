"""Hybrid expert/distill policy imitation, then five magnetic self-play mixed passes."""

import argparse
import functools
import hashlib
import json
import os
import pathlib
import subprocess
import time

import numpy
import tensorflow as tf
from model import export, joint_settlement
from training import self_play
from training import trainer as trainers
from training.candidate import collection, monitor, q_fit, replay


def settings(args):
    for name in (
        "batch_size",
        "replay_positions",
        "cohort_exposures",
        "fresh_positions",
        "games",
        "collection_workers",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    command = json.loads(pathlib.Path(args.evaluation_command).read_text())
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) for value in command)
        or not any("{weights}" in value for value in command)
        or not any("{output}" in value for value in command)
    ):
        raise ValueError("Evaluation argv must contain {weights} and {output}")
    int(command[command.index("--seed") + 1])
    games = int(command[command.index("--games") + 1])
    if games <= 0 or games % 4:
        raise ValueError("Evaluation needs complete four-game seed groups")
    return args.record_fractions, command


def save_cohort(network, root, exposures):
    """Keep cohorts immutable; latest atomically points to their weights."""
    checkpoint = root / "checkpoints" / f"{exposures:012d}.weights.h5"
    checkpoint.parent.mkdir(exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    network.save_weights(checkpoint)
    pending = root / "latest.weights.h5.pending"
    pending.symlink_to(checkpoint.relative_to(root))
    pending.replace(root / "latest.weights.h5")


def finish_checkpoint(
    stage, *, root, network, pools, report, evaluation_command, fit_critic
):
    weights = root / f"{stage}.weights.h5"
    network.save_weights(weights)
    for pool in pools[1:]:
        pool.flush()
    report(stage + "_evaluation", {})
    command = evaluation_command.copy()
    phase = 0 if stage == "record-imitation" else int(stage.rsplit("-", 1)[1])
    seed_argument = command.index("--seed") + 1
    command[seed_argument] = str(
        int(command[seed_argument])
        + phase * (int(command[command.index("--games") + 1]) // 4)
    )
    subprocess.run(
        [
            value.format(
                weights=str(weights.resolve()),
                output=str((root / (stage + "-evaluation")).resolve()),
            )
            for value in command
        ],
        check=True,
    )
    if stage != "mixed-policy-5":
        fit_critic(stage)
    report(stage + "_complete", {})


def run(args):
    fractions, evaluation_command = settings(args)
    label_plan = json.loads(
        (pathlib.Path(args.distill_labels) / "plan.json").read_text()
    )
    summary = json.loads(
        (pathlib.Path(label_plan["data"]) / "dataset_summary.json").read_text()
    )
    positions = sum(row[0] for row in summary["partition_counts"][:7])
    assert positions > 0, "Training partition is empty"
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) < 2:
        raise ValueError("Separate training and collection GPUs required")
    collection_device_ix = (
        os.environ["CUDA_VISIBLE_DEVICES"]
        .split(",")
        .index(args.collection_device)
    )
    for device_ix, device in enumerate(devices):
        # Leave room for checkpoint evaluation workers on every device.
        tf.config.set_logical_device_configuration(
            device,
            [
                tf.config.LogicalDeviceConfiguration(
                    memory_limit=(
                        8192 if device_ix == collection_device_ix else 18432
                    )
                )
            ],
        )
    strategy = tf.distribute.MirroredStrategy(
        devices=[
            f"/GPU:{device_ix}"
            for device_ix in range(len(devices))
            if device_ix != collection_device_ix
        ]
    )
    if args.batch_size % strategy.num_replicas_in_sync:
        raise ValueError("Batch size must divide evenly across replicas")
    tf.keras.utils.set_random_seed(args.seed)
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    generator = numpy.random.default_rng(args.seed)
    features = self_play.initial_features(args.seed)
    with strategy.scope():
        network = export.load_network(args.initial, features)
        reference = export.load_network(args.reference, features)
        network.save_weights(root / "initial.weights.h5")
        previous = export.load_network(args.initial, features)
        trainer = trainers.DistributedTrainer(
            trainers.Trainer(
                network=network,
                args=args,
                features=features,
                mirror=previous.actor,
                magnet=reference.actor,
            ),
            strategy,
        )
    monitor.write_json(
        root / "plan.json",
        {
            **vars(args),
            "positions_per_pass": positions,
            "record_fractions": fractions,
            "expert_weight": 0.5,
            "distill_weight": 0.5,
            "imitation_weighting": "q_gap_to_fixed_reference",
            "beta_mode": "rolling_ess",
            "ess_target": 0.7,
            "evaluation_argv": evaluation_command,
            "initial_sha256": hashlib.sha256(
                pathlib.Path(args.initial).read_bytes()
            ).hexdigest(),
            "reference_sha256": hashlib.sha256(
                pathlib.Path(args.reference).read_bytes()
            ).hexdigest(),
            "root_source": "generated_games",
            "optimizer_restart": True,
            "reset_policy_weights": [],
            "preserve_shared_state": True,
            "preserve_policy_state": True,
            "q_passes_per_checkpoint": 2,
            "q_updates_encoder": True,
            "final_q_fit": False,
            "collection_policy_lag_cohorts": 1,
            "target_bits": ["keiten", "tenpai", "calls", "disagreement"],
        },
    )
    pools = [None] + [
        replay.ReplayBuffer(
            root / "replay" / name,
            args.replay_positions,
            generator,
            reservoir=index == 0,
        )
        for index, name in enumerate(
            ("record-targeted", "generated-natural", "generated-targeted")
        )
    ]
    with tf.device(strategy.extended.worker_devices[0]):
        evaluate_roots = collection.root_scorer(network)
    progress = monitor.Monitor(root, positions, vars(args))
    exposures = cohort = games = q_exposures = 0

    def report(stage, metrics):
        progress.report(
            stage=stage,
            metrics=metrics,
            exposures=exposures,
            cohort=cohort,
            games=games,
        )

    def fit_critic(stage):
        nonlocal q_exposures
        q_exposures = q_fit.run(
            stage=stage,
            root=root,
            network=network,
            strategy=strategy,
            args=args,
            report=report,
            total_exposures=q_exposures,
        )
        previous.actor.set_weights(network.actor.get_weights())

    finish = functools.partial(
        finish_checkpoint,
        root=root,
        network=network,
        pools=pools,
        report=report,
        evaluation_command=evaluation_command,
        fit_critic=fit_critic,
    )

    stage = "record-imitation"
    report(stage, {})
    seen = 0
    input_started = time.monotonic()
    for tensors in replay.recorded_batches(args):
        batch = tf.nest.map_structure(lambda value: value.numpy(), tensors)
        count = len(batch["chosen"])
        root_started = time.monotonic()
        targets = replay.classify_records(batch, evaluate_roots, generator)
        root_seconds = time.monotonic() - root_started
        packed = replay.pack_batch(batch, trainer.feature_packing)
        pools[1].add(packed, numpy.flatnonzero(targets > 0))
        padded = replay.pad_batch(packed, args.batch_size)
        update_started = time.monotonic()
        metrics = {
            name: float(value)
            for name, value in trainer.policy_update(
                padded, tf.constant(count, tf.int32)
            ).items()
        }
        metrics.update(
            {
                "runtime/input_seconds": (
                    update_started - input_started - root_seconds
                ),
                "runtime/root_seconds": root_seconds,
                "runtime/update_seconds": time.monotonic() - update_started,
            }
        )
        exposures += count
        seen += count
        if (
            seen // args.cohort_exposures
            != (seen - count) // args.cohort_exposures
        ):
            save_cohort(network, root, exposures)
            previous.actor.set_weights(network.actor.get_weights())
        report(stage, metrics)
        input_started = time.monotonic()
    if seen != positions:
        raise ValueError(f"Dataset count mismatch: {seen} != {positions}")
    finish(stage)
    pools[1].reservoir = False
    with collection.BackgroundCollector(
        weights=args.initial,
        device=f"/GPU:{collection_device_ix}",
        games=args.games,
        workers=args.collection_workers,
        root_copies=args.root_copies,
        seed=args.seed,
    ) as collector:
        for phase, fraction in enumerate(fractions, 1):
            stage = f"mixed-policy-{phase}"
            if fraction == 0:
                pools[0] = None
            elif pools[0] is None:
                pools[0] = replay.RecordedStream(
                    args, trainer, pools[1], evaluate_roots, generator
                )
            used = 0
            report(stage, {})
            while used < positions:
                count = min(args.cohort_exposures, positions - used)
                quotas = replay.allocation(
                    used + count, fraction, args.target_fraction, start=used
                )
                previous.actor.set_weights(network.actor.get_weights())
                collection_started = time.monotonic()
                collected = None
                fresh = 0
                fresh_required = (
                    (args.fresh_positions * count + args.cohort_exposures - 1)
                    // args.cohort_exposures
                    if quotas[2:].sum()
                    else 0
                )
                if fresh_required:
                    if collector.pending is None:
                        collector.start(
                            network.get_weights(),
                            fresh_required,
                            args.seed + 100000 + games,
                        )
                    report(stage + "_collecting", {"fresh_positions": 0})
                    collected = collector.finish()
                    games += collected["base_games"]
                    fresh = collected["positions"]
                next_count = min(
                    args.cohort_exposures, positions - used - count
                )
                if replay.allocation(
                    used + count + next_count,
                    fraction,
                    args.target_fraction,
                    start=used + count,
                )[2:].sum():
                    collector.start(
                        network.get_weights(),
                        (
                            args.fresh_positions * next_count
                            + args.cohort_exposures
                            - 1
                        )
                        // args.cohort_exposures,
                        args.seed + 100000 + games,
                    )
                if collected:
                    for batch in collected.pop("batches"):
                        batch["distill"] = numpy.full(
                            len(batch["chosen"]), -1, numpy.int32
                        )
                        batch["recorded"] = numpy.zeros(
                            len(batch["chosen"]), numpy.bool_
                        )
                        batch["projected_st3"] = (
                            joint_settlement.observed_payoffs(
                                batch["settlement"]
                            ).numpy()
                        )
                        packed = replay.pack_batch(
                            batch, trainer.feature_packing
                        )
                        pools[2].add(packed)
                        pools[3].add(
                            packed, numpy.flatnonzero(batch["targets"] > 0)
                        )
                collection_seconds = time.monotonic() - collection_started
                report(stage + "_updating", {})
                for begin in range(0, count, args.batch_size):
                    size = min(args.batch_size, count - begin)
                    counts = replay.allocation(
                        used + begin + size,
                        fraction,
                        args.target_fraction,
                        start=used + begin,
                    )
                    input_started = time.monotonic()
                    batch = replay.mixed_batch(pools, counts, generator)
                    batch = replay.pad_batch(batch, args.batch_size)
                    update_started = time.monotonic()
                    metrics = {
                        name: float(value)
                        for name, value in trainer.policy_update(
                            batch, tf.constant(size, tf.int32)
                        ).items()
                    }
                    metrics.update(
                        {
                            "runtime/input_seconds": (
                                update_started - input_started
                            ),
                            "runtime/update_seconds": (
                                time.monotonic() - update_started
                            ),
                        }
                    )
                    exposures += size
                    report(stage, metrics)
                used += count
                cohort += 1
                save_cohort(network, root, exposures)
                report(
                    stage + "_cohort",
                    {
                        "record_positions": quotas[:2].sum(),
                        "record_target_positions": quotas[1],
                        "generated_positions": quotas[2:].sum(),
                        "generated_target_positions": quotas[3],
                        **(
                            {
                                (
                                    "completed_collection_games"
                                    if name == "games"
                                    else name
                                ): value
                                for name, value in collected["metrics"].items()
                            }
                            if collected
                            else {}
                        ),
                        "positions_per_second": (
                            fresh / collected["seconds"] if collected else 0
                        ),
                        "collection_wait_and_replay_seconds": (
                            collection_seconds
                        ),
                        "fresh_positions": fresh,
                        "fresh_position_target": fresh_required,
                        "replay/record_fraction": quotas[:2].sum() / count,
                        "replay/generated_fraction": quotas[2:].sum() / count,
                        **{
                            f"replay/{name}_target_fraction": quotas[index + 1]
                            / quotas[index : index + 2].sum()
                            for name, index in (("record", 0), ("generated", 2))
                            if quotas[index : index + 2].sum()
                        },
                        **{
                            f"replay/{name}_pool_positions": min(
                                pool.seen, pool.capacity
                            )
                            for name, pool in zip(
                                (
                                    "record_targeted",
                                    "generated_natural",
                                    "generated_targeted",
                                ),
                                pools[1:],
                            )
                        },
                    },
                )
            finish(stage)
    network.save_weights(root / "final.weights.h5")
    report("complete", {})
    progress.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "initial",
        "reference",
        "distill-labels",
        "output",
        "evaluation-command",
        "collection-device",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--batch-size", type=int, default=9216)
    parser.add_argument("--replay-positions", type=int, default=524288)
    parser.add_argument("--cohort-exposures", type=int, default=995328)
    parser.add_argument("--fresh-positions", type=int, default=497664)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--collection-workers", type=int, default=24)
    # Save the selected recipe alongside capacity settings in plan.json.
    # Alternative objectives and schedules belong on experiment branches.
    parser.set_defaults(
        record_fractions=[0.8, 0.6, 0.4, 0.2, 0.2],
        target_fraction=0.5,
        root_copies=1,
        learning_rate=2e-5,
        beta=16.0,
        mirror_coefficient=0.3,
        magnet_coefficient=0.3,
        risk_coefficient=0.1,
        expert_prior=0.2,
        riichi_rate_target=0.20,
    )
    parser.add_argument("--seed", type=int, default=2042)
    return parser
