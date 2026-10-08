"""Train, export and evaluate a frozen joint-model candidate."""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

from model import source_snapshot


def run(args):
    root = pathlib.Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = vars(args).copy()
    for name in (
        "data",
        "beliefs",
        "initial",
        "rollouts",
        "opponent",
        "output",
    ):
        if plan[name] is not None:
            plan[name] = str(pathlib.Path(plan[name]).resolve())
    if (root / "pipeline.json").exists():
        if json.loads((root / "pipeline.json").read_text()) != plan:
            raise ValueError("Pipeline settings changed; use a fresh output")
    else:
        (root / "pipeline.json").write_text(json.dumps(plan, indent=2) + "\n")
    source = root / "source"
    if not source.exists():
        pending = root / "source.pending"
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir()
        for name in ("model", "log_dataset", "docker"):
            source_snapshot.freeze(
                pathlib.Path(__file__).parent / name,
                pending / name,
                include_native=True,
            )
        pending.rename(source)
    training = [
        "--data",
        plan["data"],
        "--output",
        str(root / "candidate"),
        "--beliefs",
        plan["beliefs"],
        "--mode",
        args.mode,
        "--batch-size",
        str(args.batch_size),
        "--awr-mix",
        str(args.awr_mix),
        "--awr-temperature",
        str(args.awr_temperature),
        "--rollout-overhead",
        str(args.rollout_overhead),
        "--rollout-stale-weight",
        str(args.rollout_stale_weight),
    ]
    for name in ("initial", "rollouts"):
        if plan[name] is not None:
            training.extend(["--" + name, plan[name]])
    if args.phase_batches is not None:
        training.extend(["--phase-batches", str(args.phase_batches)])
    stages = [
        ("training", "model.critic_full", training),
        (
            "export",
            "model.export",
            [
                "--weights",
                str(root / "candidate/final.weights.h5"),
                "--data",
                plan["data"],
                "--output",
                str(root / "policy"),
            ],
        ),
        (
            "evaluation",
            "model.evaluate",
            [
                "--new",
                str(root / "policy/saved_model"),
                "--old",
                plan["opponent"],
                "--output",
                str(root / "evaluation"),
                "--games",
                str(args.games),
                "--seed",
                str(args.seed),
                "--workers",
                str(args.match_workers),
                "--devices",
                args.devices,
                "--batch-games",
                str(args.batch_games),
            ],
        ),
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.devices,
        "PYTHONPATH": str(source),
    }
    for stage, module, arguments in stages:
        # Each producer commits completion after its artifacts are written.
        status = root / "candidate/status.json"
        if stage == "training" and status.exists():
            if json.loads(status.read_text())["stage"] == "complete":
                continue
        if (
            stage == "export"
            and (root / "policy/export-verification.json").exists()
        ):
            continue
        (root / "pipeline_status.json").write_text(
            json.dumps({"stage": stage}) + "\n"
        )
        with (root / f"{stage}.log").open("a") as log:
            subprocess.run(
                args=[
                    sys.executable,
                    str(source / "docker/model_entrypoint.py"),
                    sys.executable,
                    "-u",
                    "-m",
                    module,
                    *arguments,
                ],
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    (root / "pipeline_status.json").write_text(
        json.dumps({"stage": "complete"}) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beliefs", required=True)
    parser.add_argument("--initial")
    parser.add_argument("--opponent", required=True, help="Frozen SavedModel")
    parser.add_argument("--devices", required=True, help="Training GPU UUIDs")
    parser.add_argument(
        "--mode", choices=("smoke", "experiment", "full"), default="full"
    )
    parser.add_argument("--batch-size", type=int, default=9216)
    parser.add_argument("--phase-batches", type=int)
    parser.add_argument("--awr-mix", type=float, default=1.0)
    parser.add_argument("--awr-temperature", type=float, default=3.0)
    parser.add_argument("--rollouts")
    parser.add_argument("--rollout-overhead", type=float, default=0.1)
    parser.add_argument("--rollout-stale-weight", type=float, default=0.5)
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match-workers", type=int, default=1)
    parser.add_argument("--batch-games", type=int, default=8)
    args = parser.parse_args()
    if args.games <= 0 or args.games % 4:
        parser.error("games must be a positive multiple of four")
    if min(args.match_workers, args.batch_games, args.batch_size) < 1:
        parser.error("worker and batch sizes must be positive")
    run(args)


if __name__ == "__main__":
    main()
