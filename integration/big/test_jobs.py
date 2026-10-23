"""Dataset jobs build current records and resume interrupted output."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import zipfile

import tensorflow as tf

from integration.big import test_dataset_sampling
from log_dataset import _native_collection


def test_dataset_job_builds_joint_records_and_resumes(tmp_path):
    result = subprocess.run(
        [sys.executable, "dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout
    archives = tmp_path / "archives"
    archives.mkdir()
    test_dataset_sampling.write_enriched_archive(archives)
    reserve = tmp_path / "reserve.json"
    reserve.write_text(json.dumps({"games": []}))
    command = [
        sys.executable,
        "dataset.py",
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
    assert summary["validation_sample"]["decision_hash_modulus"] == 16
    assert summary["validation_examples"] == sum(
        row["validation_count"] for row in summary["shard_manifest"]
    )
    for row in summary["shard_manifest"]:
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
