"""Recorded AWR, then five mixed replay AWR passes from released weights."""

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

from log_dataset import focus_sampling, scoring, training_data
from model import actions, export, joint_settlement
from training import self_play
from training import trainer as training
from training.candidate import collection, monitor, state


def allocation(total, record_fraction, target_fraction, *, start=0):
    """Allocate [start, total) using cumulative rounding at both boundaries."""
    if start:
        return allocation(total, record_fraction, target_fraction) - allocation(
            start, record_fraction, target_fraction
        )
    records = round(total * record_fraction)
    generated = total - records
    record_targets = round(records * target_fraction)
    generated_targets = round(generated * target_fraction)
    return numpy.array(
        [
            records - record_targets,
            record_targets,
            generated - generated_targets,
            generated_targets,
        ],
        numpy.int64,
    )


class Replay:
    """Bounded disk-backed reservoir (records) or recent ring (self play)."""

    def __init__(self, path, capacity, generator, reservoir):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.capacity = capacity
        self.generator = generator
        self.reservoir = reservoir
        self.seen = 0
        self.arrays = None
        self.structure = None

    def add(self, batch, rows=None):
        """Append all rows, or the supplied ordered integer row indices."""
        values = tf.nest.flatten(batch)
        count = len(values[0]) if rows is None else len(rows)
        if not count:
            return
        if self.arrays is None:
            self.structure = tf.nest.map_structure(lambda value: None, batch)
            self.arrays = [
                numpy.lib.format.open_memmap(
                    self.path / f"{index}.npy",
                    mode="w+",
                    dtype=value.dtype,
                    shape=(self.capacity, *value.shape[1:]),
                )
                for index, value in enumerate(values)
            ]
        positions = numpy.arange(self.seen, self.seen + count)
        slots = (
            numpy.where(
                positions < self.capacity,
                positions,
                self.generator.integers(positions + 1),
            )
            if self.reservoir
            else positions % self.capacity
        )
        selected = numpy.flatnonzero(slots < self.capacity)
        # Repeated slots must keep the last replacement in stream order.
        _, last = numpy.unique(slots[selected][::-1], return_index=True)
        selected = selected[::-1][last]
        source = (
            selected
            if self.reservoir
            else self.generator.permutation(count)[selected]
        )
        if rows is not None:
            source = rows[source]
        if self.reservoir:
            for destination, value in zip(self.arrays, values, strict=True):
                destination[slots[selected]] = value[source]
        else:
            destinations = slots[selected]
            boundaries = numpy.r_[
                0,
                numpy.flatnonzero(numpy.diff(destinations) != 1) + 1,
                len(destinations),
            ]
            for destination, value in zip(self.arrays, values, strict=True):
                for start, stop in zip(boundaries[:-1], boundaries[1:]):
                    for begin in range(start, stop, 512):
                        end = min(begin + 512, stop)
                        destination[
                            destinations[begin] : destinations[end - 1] + 1
                        ] = value[source[begin:end]]
        self.seen += count

    def sample(self, count):
        if not self.seen:
            raise ValueError(f"Empty replay stratum: {self.path}")
        selected = self.generator.integers(
            min(self.seen, self.capacity), size=count
        )
        return tf.nest.pack_sequence_as(
            self.structure, [value[selected] for value in self.arrays]
        )

    def flush(self):
        if self.arrays is not None:
            for value in self.arrays:
                value.flush()
            state.save(
                self.path / "manifest.json",
                {
                    "seen": self.seen,
                    "capacity": self.capacity,
                    "reservoir": self.reservoir,
                    "fields": tf.nest.pack_sequence_as(
                        self.structure,
                        [
                            {
                                "file": f"{index}.npy",
                                "shape": list(value.shape),
                                "dtype": value.dtype.name,
                            }
                            for index, value in enumerate(self.arrays)
                        ],
                    ),
                },
            )


@tf.function(reduce_retracing=True)
def record_batch(raw):
    features, labels, auxiliary = training_data.parse_batch(raw)
    selected = tf.stack(
        [
            tf.cast(labels[name], tf.int32)
            for name in actions.TRAIN_POLICY_HEADS
        ],
        axis=1,
    )
    tf.debugging.assert_equal(
        tf.reduce_sum(tf.cast(selected >= 0, tf.int32), axis=1), 1
    )
    heads = tf.argmax(
        tf.cast(selected >= 0, tf.int32), axis=1, output_type=tf.int32
    )
    masks = actions.legal_choices(features)
    legal = tf.gather(
        tf.stack(
            [
                tf.pad(masks[name], [[0, 0], [0, 150 - masks[name].shape[-1]]])
                for name in actions.TRAIN_POLICY_HEADS
            ],
            axis=1,
        ),
        heads,
        batch_dims=1,
    )
    chosen = tf.cast(tf.gather(selected, heads, batch_dims=1), tf.int32)
    tf.debugging.assert_equal(tf.gather(legal, chosen, batch_dims=1), True)
    return {
        "features": features,
        "head": heads,
        "chosen": chosen,
        "legal": legal,
        "returns": auxiliary["returns"],
        "payment_points": tf.gather(
            tf.reshape(labels["terminal_payment"], [-1, 64]),
            numpy.ravel_multi_index(numpy.asarray(scoring.CELLS).T, (4, 4, 4)),
            axis=1,
        ),
    }


def record_stream(args, repeat=False):
    # Finite natural traversal: preserve the tail, do not repeat or oversample.
    return (
        training_data.dataset(
            root=args.data,
            sampling=focus_sampling.Sampling.NATURAL,
            partition="train",
            batch_size=args.batch_size,
            repeat=repeat,
            seed=args.seed,
            drop_remainder=False,
        )
        .map(record_batch, num_parallel_calls=2)
        .prefetch(2)
    )


def prepare(batch, packing):
    """One compact replay contract, with returns in ST3 / 100."""
    return {
        **{
            name: numpy.asarray(batch[name], dtype=dtype)
            for name, dtype in (
                ("head", "int32"),
                ("chosen", "int32"),
                ("legal", "bool"),
                ("returns", "float32"),
                ("payment_points", "int32"),
            )
        },
        "features": packing.pack_numpy(batch["features"]),
    }


def full_batch(batch, batch_size):
    """Keep full batches intact; repeat the final valid row for a short tail."""
    count = len(batch["chosen"])
    if count == batch_size:
        return batch
    return training.padded_batch(batch, numpy.arange(count), batch_size)


def add_replay(pools, batch, targets):
    pools[0].add(batch)
    pools[1].add(batch, numpy.flatnonzero(targets > 0))


def classify_records(batch, evaluate, generator):
    count = len(batch["chosen"])
    priorities = numpy.zeros(count, numpy.float32)
    selected = numpy.flatnonzero(
        (batch["head"] == 0) & (generator.random(count) < 1 / 64)
    )
    if len(selected):
        priorities[selected] = evaluate(
            {name: value[selected] for name, value in batch["features"].items()}
        )["priority"]
    return self_play.target_categories(
        batch["features"], batch["head"], priorities
    )


class RecordStream:
    """Consume ordinary records once in stream order; refresh target replay."""

    def __init__(self, args, trainer, targets, evaluate, generator):
        self.stream = iter(record_stream(args, repeat=True).as_numpy_iterator())
        self.trainer = trainer
        self.targets = targets
        self.evaluate = evaluate
        self.generator = generator
        self.pending = None
        self.offset = 0

    def sample(self, count):
        parts = []
        while count:
            if self.pending is None:
                batch = next(self.stream)
                targets = classify_records(batch, self.evaluate, self.generator)
                self.pending = prepare(batch, self.trainer.feature_packing)
                self.targets.add(self.pending, numpy.flatnonzero(targets > 0))
                self.offset = 0
            size = min(count, len(self.pending["chosen"]) - self.offset)
            parts.append(
                tf.nest.map_structure(
                    lambda value: value[self.offset : self.offset + size],
                    self.pending,
                )
            )
            self.offset += size
            count -= size
            if self.offset == len(self.pending["chosen"]):
                self.pending = None
        return (
            parts[0]
            if len(parts) == 1
            else tf.nest.map_structure(
                lambda *values: numpy.concatenate(values), *parts
            )
        )


def mixed_batch(pools, counts, generator):
    parts = [
        pool.sample(int(count)) for pool, count in zip(pools, counts) if count
    ]
    destinations = numpy.argsort(generator.permutation(sum(counts)))

    def combine(*values):
        result = numpy.empty(
            (len(destinations), *values[0].shape[1:]), dtype=values[0].dtype
        )
        offset = 0
        for value in values:
            result[destinations[offset : offset + len(value)]] = value
            offset += len(value)
        return result

    return tf.nest.map_structure(combine, *parts)


def settings(args):
    fractions = [float(value) for value in args.record_fractions.split(",")]
    if (
        len(fractions) != 5
        or any(not 0 <= value <= 1 for value in fractions)
        or any(a < b for a, b in zip(fractions, fractions[1:]))
    ):
        raise ValueError("Five nonincreasing record shares in [0, 1] required")
    if not 0 < args.target_fraction < 1:
        raise ValueError(
            "Target sampling fraction must be between zero and one"
        )
    for name in (
        "batch_size",
        "replay_positions",
        "cohort_exposures",
        "fresh_positions",
        "games",
        "collection_workers",
        "root_copies",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in (
        "awr_learning_rate",
        "awr_final_learning_rate",
        "awr_temperature",
    ):
        if not numpy.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not numpy.isfinite(args.awr_pull) or args.awr_pull < 0:
        raise ValueError("AWR pull must be finite and nonnegative")
    if args.awr_pull and not args.awr_reference:
        raise ValueError("AWR pull requires fixed --awr-reference weights")
    if args.awr_q_std is not None and (
        not numpy.isfinite(args.awr_q_std) or args.awr_q_std < 0
    ):
        raise ValueError("AWR Q standard-deviation penalty must be nonnegative")
    if not numpy.isfinite(args.awr_entropy) or args.awr_entropy < 0:
        raise ValueError("AWR entropy must be finite and nonnegative")
    command = json.loads(pathlib.Path(args.evaluation_command).read_text())
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) for value in command)
        or not any("{weights}" in value for value in command)
        or not any("{output}" in value for value in command)
    ):
        raise ValueError("Evaluation argv must contain {weights} and {output}")
    return fractions, command


def save_cohort(network, root, exposures):
    """Keep each cohort immutable; latest is an atomic pointer to its weights."""
    checkpoint = root / "checkpoints" / f"{exposures:012d}.weights.h5"
    checkpoint.parent.mkdir(exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    network.save_weights(checkpoint)
    pending = root / "latest.weights.h5.pending"
    pending.symlink_to(checkpoint.relative_to(root))
    pending.replace(root / "latest.weights.h5")


def finish_checkpoint(
    stage, *, root, network, pools, report, evaluation_command
):
    weights = root / f"{stage}.weights.h5"
    network.save_weights(weights)
    for pool in pools[1:]:
        pool.flush()
    report(stage + "_evaluation", {})
    subprocess.run(
        [
            value.format(
                weights=str(weights.resolve()),
                output=str((root / (stage + "-evaluation")).resolve()),
            )
            for value in evaluation_command
        ],
        check=True,
    )
    report(stage + "_complete", {})


def set_awr_learning_rate(trainer, args, exposures, positions):
    """Keep each pass constant; interpolate rates by its self-play share."""
    fractions = ([1.0] if not args.mixed_only else []) + [
        float(value) for value in args.record_fractions.split(",")
    ]
    phase = exposures // positions
    fraction = (
        (fractions[0] - fractions[phase]) / (fractions[0] - fractions[-1])
        if fractions[0] != fractions[-1]
        else 0.0
    )
    trainer.awr_optimizer.learning_rate.assign(
        args.awr_learning_rate
        + fraction * (args.awr_final_learning_rate - args.awr_learning_rate)
    )


def run(args):
    fractions, evaluation_command = settings(args)
    summary = json.loads(
        (pathlib.Path(args.data) / "dataset_summary.json").read_text()
    )
    positions = sum(row[0] for row in summary["partition_counts"][:7])
    assert positions > 0, "Training partition is empty"
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) < 2:
        raise ValueError("Separate training and collection GPUs required")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    collection_device_ix = (
        os.environ["CUDA_VISIBLE_DEVICES"]
        .split(",")
        .index(args.collection_device)
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
        trainer = training.DistributedTrainer(
            training.Trainer(
                network=network,
                args=args,
                features=features,
                mirror=(
                    export.load_network(args.awr_reference, features).actor
                    if args.awr_pull
                    else None
                ),
                mirror_coefficient=args.awr_pull,
                awr_q_std=args.awr_q_std,
            ),
            strategy,
        )
    state.save(
        root / "plan.json",
        {
            **vars(args),
            "positions_per_pass": positions,
            "record_fractions": fractions,
            "evaluation_argv": evaluation_command,
            "initial_sha256": hashlib.sha256(
                pathlib.Path(args.initial).read_bytes()
            ).hexdigest(),
            "awr_reference_sha256": (
                hashlib.sha256(
                    pathlib.Path(args.awr_reference).read_bytes()
                ).hexdigest()
                if args.awr_pull
                else None
            ),
            "root_source": "generated_games",
            "optimizer_restart": True,
            "collection_policy_lag_cohorts": 1,
            "target_bits": ["keiten", "tenpai", "calls", "disagreement"],
        },
    )
    pools = [None] + [
        Replay(
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
        evaluate_roots = collection.disagreement_evaluator(network)
    progress = monitor.Monitor(root, positions, vars(args))
    exposures = cohort = games = 0

    def report(stage, metrics):
        progress.report(
            stage=stage,
            metrics=metrics,
            exposures=exposures,
            cohort=cohort,
            games=games,
        )

    finish = functools.partial(
        finish_checkpoint,
        root=root,
        network=network,
        pools=pools,
        report=report,
        evaluation_command=evaluation_command,
    )

    if not args.mixed_only:
        stage = "record-awr"
        report(stage, {})
        seen = 0
        for tensors in record_stream(args):
            batch = tf.nest.map_structure(lambda value: value.numpy(), tensors)
            count = len(batch["chosen"])
            targets = classify_records(batch, evaluate_roots, generator)
            packed = prepare(batch, trainer.feature_packing)
            pools[1].add(packed, numpy.flatnonzero(targets > 0))
            padded = full_batch(packed, args.batch_size)
            set_awr_learning_rate(trainer, args, exposures, positions)
            metrics = trainer.policy_update(
                padded, tf.constant(count, tf.int32)
            )
            exposures += count
            seen += count
            if (
                seen // args.cohort_exposures
                != (seen - count) // args.cohort_exposures
            ):
                save_cohort(network, root, exposures)
            report(stage, metrics)
        if seen != positions:
            raise ValueError(f"Dataset count mismatch: {seen} != {positions}")
        finish(stage)
    pools[1].reservoir = False
    with collection.Prefetch(
        weights=args.initial,
        device=f"/GPU:{collection_device_ix}",
        games=args.games,
        workers=args.collection_workers,
        root_copies=args.root_copies,
        seed=args.seed,
    ) as collector:
        for phase, fraction in enumerate(fractions, 1):
            stage = f"mixed-awr-{phase}"
            if fraction == 0:
                pools[0] = None
            elif pools[0] is None:
                pools[0] = RecordStream(
                    args, trainer, pools[1], evaluate_roots, generator
                )
            used = 0
            report(stage, {})
            while used < positions:
                count = min(args.cohort_exposures, positions - used)
                quotas = allocation(
                    used + count, fraction, args.target_fraction, start=used
                )
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
                if allocation(
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
                        states, inverse = numpy.unique(
                            batch.pop("settlement"), axis=0, return_inverse=True
                        )
                        with tf.device("/CPU:0"):
                            batch["returns"] = (
                                joint_settlement.observed_payoffs(
                                    tf.convert_to_tensor(states)
                                ).numpy()[inverse]
                                / 100
                            )
                        add_replay(
                            pools[2:],
                            prepare(batch, trainer.feature_packing),
                            batch["targets"],
                        )
                collection_seconds = time.monotonic() - collection_started
                report(stage + "_updating", {})
                for begin in range(0, count, args.batch_size):
                    size = min(args.batch_size, count - begin)
                    counts = allocation(
                        used + begin + size,
                        fraction,
                        args.target_fraction,
                        start=used + begin,
                    )
                    batch = mixed_batch(pools, counts, generator)
                    set_awr_learning_rate(trainer, args, exposures, positions)
                    metrics = trainer.policy_update(
                        full_batch(batch, args.batch_size),
                        tf.constant(size, tf.int32),
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
                        "collection_wait_and_replay_seconds": collection_seconds,
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
        "data",
        "output",
        "evaluation-command",
        "collection-device",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mixed-only", action="store_true")
    parser.add_argument("--record-fractions", default="0.8,0.6,0.4,0.2,0")
    parser.add_argument("--target-fraction", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=9216)
    parser.add_argument("--replay-positions", type=int, default=524288)
    parser.add_argument("--cohort-exposures", type=int, default=995328)
    parser.add_argument("--fresh-positions", type=int, default=497664)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--collection-workers", type=int, default=24)
    parser.add_argument("--root-copies", type=int, default=1)
    parser.add_argument("--awr-learning-rate", type=float, default=1e-5)
    parser.add_argument("--awr-final-learning-rate", type=float, default=1e-5)
    parser.add_argument("--awr-temperature", type=float, default=3)
    parser.add_argument("--awr-entropy", type=float, default=0.01)
    parser.add_argument("--awr-pull", type=float, default=0.0)
    parser.add_argument(
        "--awr-q-std",
        type=float,
        help="Weight actions by predicted Q mean minus this many ST3 stds",
    )
    parser.add_argument(
        "--awr-reference",
        help="Fixed reference-policy weights for the AWR KL magnet",
    )
    parser.add_argument("--seed", type=int, default=2042)
    return parser
