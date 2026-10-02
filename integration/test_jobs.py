"""Batch commands and frozen pipelines remain independently runnable."""

import json
import os
import subprocess
import sys
import types

import train


def test_batch_job_commands_expose_help():
    for name in (
        "download_dataset.py",
        "enrich_ranks.py",
        "import_riichilab.py",
        "build_dataset.py",
        "export_dataset.py",
        "build_critic_dataset.py",
        "train_policy.py",
        "train_joint.py",
        "export_policy.py",
        "evaluate.py",
        "train.py",
    ):
        result = subprocess.run(
            [sys.executable, name, "--help"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        )
        assert "usage:" in result.stdout
        assert "Job:" in result.stdout


def test_pipeline_freezes_runnable_job_sources(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    root = tmp_path / "run"
    stages = []

    def check_stage(command, root, log, devices="-1"):
        stages.append(command[0])
        subprocess.run(
            [sys.executable, command[0], "--help"],
            cwd=root,
            check=True,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": ""},
        )

    monkeypatch.setattr(train, "execute", check_stage)
    monkeypatch.setattr(train.records, "validate_summary", lambda _: None)
    train.run(
        types.SimpleNamespace(
            root=str(root),
            reference=str(reference),
            data=str(data),
            archives=None,
            experiment=True,
            training_devices="-1",
            dealership_advice_strength=0.25,
        )
    )
    assert stages == ["train_policy.py", "evaluate.py"]
    plan = json.loads((root / "plan.json").read_text())
    assert "train.py" in plan["source_hashes"]
    assert "train_policy.py" in plan["source_hashes"]
    assert "evaluate.py" in plan["source_hashes"]
    assert (root / "train.py").resolve() == root / "source/train.py"
    subprocess.run(
        [sys.executable, "build_dataset.py", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": ""},
    )
    assert json.loads((root / "pipeline_status.json").read_text()) == {
        "stage": "complete"
    }
