"""Run match-then-root magnetic mirror descent."""

import argparse
import json
import os
import pathlib
import sys
import time

from training import monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initial", "reference", "output", "evaluation-command"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--root-games", type=int, default=10000)
    parser.add_argument("--critic-batch-size", type=int, default=18432)
    parser.add_argument("--actor-batch-size", type=int, default=16384)
    parser.add_argument("--root-batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gpu-memory-mb", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sync-command")
    args = parser.parse_args()

    from training.corrections import run

    existing_output = pathlib.Path(args.output).exists()
    try:
        run.run(args)
    finally:
        path = pathlib.Path(args.output) / "status.json"
        if (
            (
                "TF_CONFIG" not in os.environ
                or json.loads(os.environ["TF_CONFIG"])["task"]["index"] == 0
            )
            and not existing_output
            and sys.exception() is not None
            and path.exists()
        ):
            status = json.loads(path.read_text())
            status.update(
                stage="failed",
                error=str(sys.exception()),
                updated_at=time.time(),
            )
            monitor.write_json(path, status)


if __name__ == "__main__":
    main()
