"""Export one phase and run its fixed-seed comparison against the incumbent."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


def run(args):
    if args.games <= 0 or args.games % 4:
        raise ValueError("Evaluation needs complete four-game seed groups")
    root = pathlib.Path(args.output)
    if (root / "policy").exists():
        verification = json.loads(
            (root / "policy" / "export-verification.json").read_text()
        )
        if (
            verification["weight_sha256"]
            != hashlib.sha256(
                pathlib.Path(args.weights).read_bytes()
            ).hexdigest()
        ):
            raise ValueError(
                "Existing evaluation export uses different weights"
            )
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "model.export",
                "--weights",
                args.weights,
                "--self-play-seed",
                "941",
                "--output",
                str(root / "policy"),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.evaluate",
            "--new",
            str(root / "policy" / "saved_model"),
            "--old",
            args.opponent,
            "--output",
            str(root / "matches"),
            "--games",
            str(args.games),
            "--seed",
            str(args.seed),
            "--workers",
            str(args.workers),
            "--devices",
            args.devices,
            "--batch-games",
            str(args.batch_games),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "opponent", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-games", type=int, default=8)
    parser.add_argument("--devices", default="-1")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
