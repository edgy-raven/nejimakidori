"""Reproducible fixed-budget candidate pipeline; no automatic promotion."""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import log_dataset
from model import records, source_snapshot


def execute(command, root, log, devices="-1"):
    with pathlib.Path(log).open("a") as stream:
        try:
            subprocess.run(
                [sys.executable, *command],
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": devices,
                    "TF_CUDNN_USE_AUTOTUNE": "0",
                },
            )
        except subprocess.CalledProcessError as error:
            (pathlib.Path(root) / "pipeline_status.json").write_text(
                json.dumps(
                    {
                        "stage": "failed",
                        "failed_stage": pathlib.Path(log).stem,
                        "exit_code": error.returncode,
                    }
                )
            )
            raise


def run(args):
    root = pathlib.Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    reference = pathlib.Path(args.reference).resolve()
    source = root / "source"
    source.mkdir()
    source_snapshot.freeze(
        pathlib.Path(__file__).parent / "model", source / "model"
    )
    shutil.copy2(pathlib.Path(__file__).parent / "feature_vector.py", source)
    shutil.copy2(pathlib.Path(__file__), source / "pipeline.py")
    shutil.copy2(pathlib.Path(__file__).parent / "MODEL_CARD.md", source)
    source_hashes = source_snapshot.hashes(source)
    (root / "model").symlink_to("source/model", target_is_directory=True)
    (root / "feature_vector.py").symlink_to("source/feature_vector.py")
    log_dataset_hashes = source_snapshot.freeze(
        pathlib.Path(log_dataset.__file__).parent, root / "log_dataset"
    )
    plan = {
        **vars(args),
        "devices": "1" if args.experiment else "1,2,3,4",
        "evaluation_games": 4 if args.experiment else 10000,
        "selection": "fresh_duplicate_games",
        "evaluation_format": "riichienv_duplicate_south",
        "source_hashes": source_hashes,
        "log_dataset_hashes": log_dataset_hashes,
        "reference_hashes": source_snapshot.hashes(reference),
    }
    (root / "plan.json").write_text(json.dumps(plan, indent=2))
    if args.data is None:
        data = root / "data"
        (root / "pipeline_status.json").write_text(
            json.dumps({"stage": "records"})
        )
        command = [
            "-m",
            "log_dataset.rebuild",
            "--archives",
            args.archives,
            "--output",
            str(data),
        ]
        if args.experiment:
            command.append("--experiment")
        execute(command=command, root=root, log=root / "records.log")
    else:
        data = pathlib.Path(args.data).resolve()
    records.validate_summary(data)
    if args.data is not None:
        (root / "data").symlink_to(data, target_is_directory=True)
    commands = [
        (
            "training",
            [
                "-m",
                "model.train",
                "--dealership-advice-strength",
                str(args.dealership_advice_strength),
                "--data",
                str(data),
                "--output",
                str(root / "candidate"),
            ]
            + (["--experiment"] if args.experiment else []),
        ),
        (
            "games",
            [
                "-m",
                "model.head_to_head",
                "--new",
                str(root / "candidate/saved_model"),
                "--old",
                str(reference),
                "--output",
                str(root / "games"),
            ]
            + (["--experiment"] if args.experiment else []),
        ),
    ]
    for stage, command in commands:
        (root / "pipeline_status.json").write_text(json.dumps({"stage": stage}))
        execute(
            command=command,
            root=root,
            log=root / (stage + ".log"),
            devices=plan["devices"] if stage == "training" else "-1",
        )
    (root / "pipeline_status.json").write_text('{"stage":"complete"}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--reference", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data")
    inputs.add_argument("--archives")
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument(
        "--dealership-advice-strength", type=float, default=0.25
    )
    arguments = parser.parse_args()
    if arguments.dealership_advice_strength < 0:
        parser.error("dealership advice strength must be nonnegative")
    run(arguments)
