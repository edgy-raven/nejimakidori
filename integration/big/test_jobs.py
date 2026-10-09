"""The two jobs build training input and hand a frozen candidate to a match."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import zipfile

import pytest
import tensorflow as tf

import train
from integration.big import test_dataset_sampling


def test_dataset_job_builds_joint_records_and_resumes(tmp_path):
    for job in ("dataset.py", "train.py"):
        result = subprocess.run(
            [sys.executable, job, "--help"],
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


def test_training_job_freezes_source_and_resumes_failed_handover(
    tmp_path, monkeypatch
):
    args = argparse.Namespace(
        output=str(tmp_path / "run"),
        data=str(tmp_path / "data"),
        beliefs=str(tmp_path / "beliefs.weights.h5"),
        initial=None,
        rollouts=None,
        opponent=str(tmp_path / "old/saved_model"),
        devices="GPU-training",
        mode="smoke",
        batch_size=8,
        phase_batches=1,
        awr_mix=1.0,
        awr_temperature=3.0,
        rollout_overhead=0.1,
        rollout_stale_weight=0.5,
        games=4,
        seed=14,
        match_workers=1,
        batch_games=4,
    )
    calls = []
    fail_match = True

    def worker(*, args, cwd, env, stdout, stderr, check):
        module = args[args.index("-m") + 1]
        calls.append(module)
        assert cwd == tmp_path / "run/source"
        assert env["PYTHONPATH"] == str(cwd)
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-training"
        assert (cwd / "model/feature_vector.py").is_file()
        assert check and stderr == subprocess.STDOUT
        if module == "model.critic_full":
            candidate = tmp_path / "run/candidate"
            candidate.mkdir(exist_ok=True)
            (candidate / "final.weights.h5").write_text("trained")
            (candidate / "status.json").write_text('{"stage": "complete"}')
        elif module == "model.export":
            assert "--weights" in args and "--data" in args
            policy = tmp_path / "run/policy"
            policy.mkdir()
            (policy / "export-verification.json").write_text('{"passed": true}')
        else:
            assert module == "model.evaluate"
            assert args[args.index("--new") + 1] == str(
                tmp_path / "run/policy/saved_model"
            )
            if fail_match:
                raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(train.subprocess, "run", worker)
    with pytest.raises(subprocess.CalledProcessError):
        train.run(args)
    assert calls == ["model.critic_full", "model.export", "model.evaluate"]
    frozen = tmp_path / "run/source/model/feature_vector.py"
    before = frozen.read_bytes()
    fail_match = False
    train.run(args)
    assert calls == [
        "model.critic_full",
        "model.export",
        "model.evaluate",
        "model.evaluate",
    ]
    assert frozen.read_bytes() == before
    assert json.loads((tmp_path / "run/pipeline_status.json").read_text()) == {
        "stage": "complete"
    }
    args.games = 8
    with pytest.raises(ValueError, match="settings changed"):
        train.run(args)
