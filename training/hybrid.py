"""Aligned expert records and greedy distill-policy labels."""

import hashlib
import json
import pathlib

import numpy
import tensorflow as tf
from log_dataset import records
from model import actions


def decode(raw, distill):
    features, labels, _ = records.parse_batch(raw)
    choices = tf.stack(
        [
            tf.cast(labels[name], tf.int32)
            for name in actions.TRAIN_POLICY_HEADS
        ],
        axis=1,
    )
    tf.debugging.assert_equal(
        tf.reduce_sum(tf.cast(choices >= 0, tf.int32), axis=1), 1
    )
    head = tf.argmax(choices >= 0, axis=1, output_type=tf.int32)
    expert = tf.gather(choices, head, batch_dims=1)
    masks = actions.legal_choices(features)
    legal = tf.gather(
        tf.stack(
            [
                tf.pad(masks[name], [[0, 0], [0, 150 - masks[name].shape[-1]]])
                for name in actions.TRAIN_POLICY_HEADS
            ],
            axis=1,
        ),
        head,
        batch_dims=1,
    )
    tf.debugging.assert_equal(tf.gather(legal, expert, batch_dims=1), True)
    tf.debugging.assert_greater_equal(distill, 0)
    tf.debugging.assert_less(distill, 150)
    tf.debugging.assert_equal(tf.gather(legal, distill, batch_dims=1), True)
    return features, head, expert, distill


def raw_batches(labels, batch_size, partition, seed):
    root = pathlib.Path(labels)
    plan = json.loads((root / "plan.json").read_text())
    source = pathlib.Path(plan["data"])
    assert (
        hashlib.sha256(
            (source / "dataset_summary.json").read_bytes()
        ).hexdigest()
        == plan["dataset_sha256"]
    )
    records.validate_summary(source)
    paths = sorted(source.glob("train-*.tfrecord.gz"))
    numpy.random.default_rng(seed).shuffle(paths)
    for path in paths:
        if not (root / (path.name + ".json")).exists():
            raise ValueError(f"Incomplete distill policy labels: {path.name}")

    def shards():
        for path in paths:
            manifest = json.loads((root / (path.name + ".json")).read_text())
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == (
                    manifest["source_sha256"]
                )
            choices = numpy.load(root / (path.name + ".npy"))
            assert (
                hashlib.sha256(
                    (root / (path.name + ".npy")).read_bytes()
                ).hexdigest()
                == manifest["labels_sha256"]
            )
            assert len(choices) == manifest["positions"]
            yield str(path), choices.astype(numpy.int32)

    def select(indices, raw, choices):
        value = tf.io.parse_example(
            raw, {"critic/partition": tf.io.FixedLenFeature([], tf.int64)}
        )["critic/partition"]
        selected = value < 7 if partition == "train" else value == 7
        return (
            tf.boolean_mask(raw, selected),
            tf.gather(choices, tf.boolean_mask(indices, selected)),
        )

    return (
        tf.data.Dataset.from_generator(
            shards,
            output_signature=(
                tf.TensorSpec([], tf.string),
                tf.TensorSpec([None], tf.int32),
            ),
        )
        .flat_map(
            lambda path, choices: tf.data.TFRecordDataset(
                path, compression_type="GZIP"
            )
            .apply(
                tf.data.experimental.assert_cardinality(
                    tf.size(choices, out_type=tf.int64)
                )
            )
            .enumerate()
            .batch(1024)
            .map(lambda indices, raw: select(indices, raw, choices))
            .unbatch()
        )
        .shuffle(8192, seed=seed)
        .batch(batch_size)
    )
