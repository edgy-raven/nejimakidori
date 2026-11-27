"""Dataset jobs build current records and resume interrupted output."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import zipfile

import numpy
import tensorflow as tf

from integration import archive_fixture
from log_dataset import records, training_data


def test_dataset_job_builds_joint_records_and_resumes(tmp_path):
    archives = tmp_path / "archives"
    archives.mkdir()
    archive_fixture.write_enriched_archive(archives)
    reserve = tmp_path / "reserve.json"
    reserve.write_text(json.dumps({"games": []}))
    command = [
        sys.executable,
        "-m",
        "log_dataset",
        "--archives",
        str(archives),
        "--reserve",
        str(reserve),
        "--output",
        str(tmp_path / "dataset"),
        "--workers",
        "1",
        "--limit-games",
        "2",
    ]
    subprocess.run(
        command, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    )
    data = tmp_path / "dataset/data"
    summary = json.loads((data / "dataset_summary.json").read_text())
    assert summary["games"] == 1
    assert summary["examples"] > 0
    records.validate_summary(data)
    assert summary["critic_auxiliary"] == training_data.AUXILIARY
    assert len(summary["partition_counts"]) == 10
    assert sum(summary["partition_counts"]) == summary["examples"]
    assert summary["sampling_fields"] == list(training_data.SAMPLING_FIELDS)
    assert summary["validation_sample"]["decision_hash_modulus"] == 16
    assert summary["validation_examples"] == sum(
        row["validation_count"] for row in summary["shard_manifest"]
    )
    for row in summary["shard_manifest"]:
        sampling_counts = numpy.zeros((10, 2), numpy.int64)
        for raw in tf.data.TFRecordDataset(
            str(data / row["path"]), compression_type="GZIP"
        ):
            fields = tf.train.Example.FromString(raw.numpy()).features.feature
            sampling_counts[fields["critic/partition"].int64_list.value[0]] += [
                fields[name].int64_list.value[0]
                for name in training_data.SAMPLING_FIELDS
            ]
        numpy.testing.assert_array_equal(
            sampling_counts, row["sampling_counts"]
        )
        assert sampling_counts[:, 0].sum() > 0
        raw_batch = next(
            iter(
                tf.data.TFRecordDataset(
                    str(data / row["path"]), compression_type="GZIP"
                ).batch(8)
            )
        )
        inputs = training_data.parse_batch(raw_batch)
        assert inputs[2]["bonus"].shape[-1] == 24
        assert numpy.isfinite(inputs[2]["score"]).all()
        for raw in raw_batch.numpy():
            fields = tf.train.Example.FromString(raw).features.feature
            assert set(fields) == {
                *records.FIELDS,
                *training_data.SAMPLING_FIELDS,
                "meta/game_id",
                "meta/kyoku_id",
                "meta/decision_index",
                "meta/actor",
                "critic/partition",
                "critic/bonus",
                "critic/score",
            }
        validation_rows = list(
            tf.data.TFRecordDataset(
                str(data / row["validation_path"]), compression_type="GZIP"
            )
        )
        assert len(validation_rows) == row["validation_count"]
        for raw in validation_rows:
            features = tf.train.Example.FromString(raw.numpy()).features.feature
            assert features["critic/partition"].int64_list.value[0] == 7
    command[command.index("--workers") + 1] = "2"
    before = {p.name: p.read_bytes() for p in data.glob("*.tfrecord.gz")}
    subprocess.run(
        command, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    )
    assert {
        p.name: p.read_bytes() for p in data.glob("*.tfrecord.gz")
    } == before

    shard = next(data.glob("train-*.tfrecord.gz"))
    shard.write_bytes(b"interrupted")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Incomplete dataset shard" in result.stderr
    shard.with_suffix(".json").unlink()
    subprocess.run(command, check=True)
    rebuilt = json.loads((data / "dataset_summary.json").read_text())
    assert rebuilt["partition_counts"] == summary["partition_counts"]
    game = json.loads((data / "games.jsonl").read_text())
    partition = (
        int.from_bytes(hashlib.sha256(game["game_id"].encode()).digest()[:4])
        % 10
    )
    for raw in tf.data.TFRecordDataset(str(shard), compression_type="GZIP"):
        example = tf.train.Example.FromString(raw.numpy())
        assert (
            example.features.feature["critic/partition"].int64_list.value[0]
            == partition
        )
    with (tmp_path / "dataset/pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0
        assert "BlockingIOError" in result.stderr
    reserve.write_text(json.dumps({"games": [{"game_id": "changed"}]}))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Dataset inputs changed" in result.stderr

    reserve.write_text('{"games": []}')
    with zipfile.ZipFile(archives / "2024.zip") as archive:
        member = archive.namelist()[0]
        events = [json.loads(row) for row in archive.read(member).splitlines()]
    del events[0]["player_ranks"]
    with zipfile.ZipFile(archives / "2024.zip", "w") as archive:
        archive.writestr(member, "\n".join(map(json.dumps, events)))
    ranks = json.loads((archives / "player_ranks.json").read_text())
    with (archives / "2024.zip").open("rb") as stream:
        ranks["years"]["2024"]["merged_sha256"] = hashlib.file_digest(
            stream, "sha256"
        ).hexdigest()
    (archives / "player_ranks.json").write_text(json.dumps(ranks))
    command[command.index("--output") + 1] = str(tmp_path / "unranked")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "KeyError: 'player_ranks'" in result.stderr
