"""Recorded batches, targeted replay pools and mixed-source sampling."""

import os
import pathlib

import numpy
import tensorflow as tf
from log_dataset import scoring, training_data
from model import actions
from training import hybrid, self_play
from training.candidate import monitor


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


class ReplayBuffer:
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
            # A sparse mmap can SIGBUS mid-update when the disk fills.
            # Reserve every field before writing any replay rows.
            for array in self.arrays:
                with open(array.filename, "r+b") as stream:
                    os.posix_fallocate(
                        stream.fileno(), 0, os.fstat(stream.fileno()).st_size
                    )
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
            monitor.write_json(
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
def decode_records(raw, distill):
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
    tf.debugging.assert_greater_equal(distill, 0)
    tf.debugging.assert_less(distill, 150)
    tf.debugging.assert_equal(tf.gather(legal, distill, batch_dims=1), True)
    return {
        "distill": tf.cast(distill, tf.int32),
        "features": features,
        "head": heads,
        "chosen": chosen,
        "legal": legal,
        "recorded": tf.ones_like(heads, dtype=tf.bool),
        "projected_st3": auxiliary["projected_st3"],
        "ron_legal": labels["ron_legal"],
        "ron_payments": labels["ron_payments"],
        "beliefs": {
            name: labels[name]
            for name in (
                "opponent_shanten",
                "opponent_ukeire",
                "opponent_hand_counts",
            )
        },
        "payment_points": tf.gather(
            tf.reshape(labels["terminal_payment"], [-1, 64]),
            numpy.ravel_multi_index(numpy.asarray(scoring.CELLS).T, (4, 4, 4)),
            axis=1,
        ),
    }


def recorded_batches(args, repeat=False):
    stream = hybrid.raw_batches(
        args.distill_labels, args.batch_size, "train", args.seed
    )
    if repeat:
        stream = stream.repeat()
    return stream.map(decode_records, num_parallel_calls=2).prefetch(2)


def pack_batch(batch, packing):
    """Compact replay with explicit actor-objective source labels."""
    return {
        **{
            name: numpy.asarray(batch[name], dtype=dtype)
            for name, dtype in (
                ("head", "int32"),
                ("chosen", "int32"),
                ("distill", "int32"),
                ("legal", "bool"),
                ("recorded", "bool"),
                ("payment_points", "int32"),
                ("projected_st3", "float32"),
                ("ron_legal", "int8"),
                ("ron_payments", "int32"),
            )
        },
        "features": packing.pack_numpy(batch["features"]),
        "beliefs": {
            name: numpy.asarray(value, dtype=numpy.int8)
            for name, value in batch["beliefs"].items()
        },
    }


def pad_batch(batch, batch_size):
    """Keep full batches intact; repeat the final valid row for a short tail."""
    count = len(batch["chosen"])
    if count == batch_size:
        return batch
    indices = numpy.pad(
        numpy.arange(count), (0, batch_size - count), mode="edge"
    )
    return tf.nest.map_structure(lambda value: value[indices], batch)


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
        batch["features"], batch["head"], priorities, chosen=batch["chosen"]
    )


class RecordedStream:
    """Consume ordinary records once in stream order; refresh target replay."""

    def __init__(self, args, trainer, targets, evaluate, generator):
        self.stream = iter(
            recorded_batches(args, repeat=True).as_numpy_iterator()
        )
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
                self.pending = pack_batch(batch, self.trainer.feature_packing)
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
