"""Expert records keep their aligned distill-policy labels."""

import hashlib
import json

import numpy
import pytest
import tensorflow as tf
from training import hybrid


def test_shard_labels_stay_aligned_after_partition_filtering(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    labels = tmp_path / "labels"
    source.mkdir()
    labels.mkdir()
    summary = source / "dataset_summary.json"
    summary.write_text("{}")
    (labels / "plan.json").write_text(
        json.dumps(
            {
                "data": str(source),
                "dataset_sha256": hashlib.sha256(
                    summary.read_bytes()
                ).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(hybrid.records, "validate_summary", lambda path: None)

    def decode(raw, distill):
        parsed = tf.io.parse_example(
            raw, {"id": tf.io.FixedLenFeature([], tf.int64)}
        )
        return parsed["id"], distill

    path = source / "train-00000.tfrecord.gz"
    with tf.io.TFRecordWriter(str(path), options="GZIP") as writer:
        for index, partition in enumerate([0, 7, 1, 8]):
            writer.write(
                tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            name: tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[value])
                            )
                            for name, value in [
                                ("id", index),
                                ("critic/partition", partition),
                            ]
                        }
                    )
                ).SerializeToString()
            )
    for choices in ([10, 11, 12, 13], [10, 11, 12], [10, 11, 12, 13, 14]):
        choice_path = labels / (path.name + ".npy")
        numpy.save(choice_path, numpy.array(choices, numpy.int16))
        (labels / (path.name + ".json")).write_text(
            json.dumps(
                {
                    "source_sha256": hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                    "labels_sha256": hashlib.sha256(
                        choice_path.read_bytes()
                    ).hexdigest(),
                    "positions": len(choices),
                }
            )
        )
        stream = hybrid.raw_batches(labels, 2, "train", 941).map(decode)
        if len(choices) == 4:
            rows = [
                (int(row), int(choice))
                for batch in stream
                for row, choice in zip(*batch)
            ]
            assert sorted(rows) == [(0, 10), (2, 12)]
        else:
            with pytest.raises(tf.errors.FailedPreconditionError):
                list(stream)
