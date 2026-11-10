"""Join retained production inputs to current records by decision identity."""

import json
import pathlib
import sys

import numpy
import teacher_feature_worker
import tensorflow as tf

from log_dataset import records
from model import feature_vector


def identity(row):
    return tuple(
        row["meta/" + name].bytes_list.value[0]
        for name in ("game_id", "kyoku_id")
    ) + tuple(
        row["meta/" + name].int64_list.value[0]
        for name in ("decision_index", "actor")
    )


def run(job):
    raw = list(
        tf.data.TFRecordDataset(
            job["input"], compression_type="GZIP"
        ).as_numpy_iterator()
    )
    metadata = [
        tf.train.Example.FromString(value).features.feature for value in raw
    ]
    wanted = {
        identity(row): index
        for index, row in enumerate(metadata)
        if row["critic/partition"].int64_list.value[0] < 7
    }
    results = {}
    for value in tf.data.TFRecordDataset(
        job["record_shards"], compression_type="GZIP"
    ).as_numpy_iterator():
        row = tf.train.Example.FromString(value).features.feature
        key = identity(row)
        if key not in wanted:
            continue
        index = wanted[key]
        assert index not in results, (key, "Duplicate production decision")
        for name in (
            "decision_phase",
            "live_wall_count",
            "candidate_hand_counts",
        ):
            expected, actual = [
                records.decode(
                    "feature/" + name,
                    item["feature/" + name].bytes_list.value[0],
                    feature_vector.INPUTS[name],
                )
                for item in (metadata[index], row)
            ]
            numpy.testing.assert_array_equal(
                expected[0] if name == "candidate_hand_counts" else expected,
                actual[0] if name == "candidate_hand_counts" else actual,
                err_msg=str((key, name)),
            )
        results[index] = {
            name: records.decode(
                "feature/" + name,
                row["feature/" + name].bytes_list.value[0],
                spec,
            ).tobytes()
            for name, spec in job["fields"].items()
        }
    copied = len(results)
    missing = {}
    for key, index in wanted.items():
        if index not in results:
            missing.setdefault(key[0].decode(), {})[
                (key[1].decode(), key[2], key[3])
            ] = index
    if missing:
        from log_dataset import shanten

        shanten.LIBRARY = pathlib.Path(job["native_library"])
        results.update(
            teacher_feature_worker.reconstruct(job, metadata, missing)
        )
    fingerprints = tf.raw_ops.Fingerprint(
        data=tf.constant(raw), method="farmhash64"
    ).numpy()
    output = pathlib.Path(job["output"])
    with tf.io.TFRecordWriter(
        str(output) + ".partial", options="GZIP"
    ) as writer:
        for index in range(len(raw)):
            example = tf.train.Example()
            example.features.feature["fingerprint"].bytes_list.value.append(
                fingerprints[index].tobytes()
            )
            for name in job["fields"]:
                example.features.feature[name].bytes_list.value.append(
                    results[index][name] if index in results else b""
                )
            writer.write(example.SerializeToString())
    pathlib.Path(str(output) + ".partial").replace(output)
    return {
        "records": len(raw),
        "teacher_records": len(results),
        "copied_records": copied,
        "reconstructed_records": len(results) - copied,
    }


if __name__ == "__main__":
    print(json.dumps(run(json.loads(pathlib.Path(sys.argv[1]).read_text()))))
