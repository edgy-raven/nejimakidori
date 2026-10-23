"""Run correctness checks, plus CPU performance gates in the big suite."""

import argparse
import os
import pathlib
import subprocess
import sys


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", choices=("small", "big"))
    parser.add_argument("--performance-baseline", type=pathlib.Path)
    args = parser.parse_args()
    if args.size == "small" and args.performance_baseline:
        parser.error("--performance-baseline requires big")
    root = pathlib.Path(__file__).resolve().parent
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_NUM_INTRAOP_THREADS": "2",
        "TF_NUM_INTEROP_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }
    if args.size == "big":
        subprocess.run(
            args=[
                sys.executable,
                "-m",
                "integration.performance",
                *(
                    ["--baseline", str(args.performance_baseline.resolve())]
                    if args.performance_baseline
                    else []
                ),
            ],
            cwd=root,
            env=environment,
            check=True,
        )
    for tier in (("small", "big") if args.size == "big" else ("small",)):
        for path in sorted((root / "integration" / tier).glob("test_*.py")):
            print(path.relative_to(root), flush=True)
            # Separate processes release TensorFlow graphs between workflows.
            subprocess.run(
                args=[
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--disable-warnings",
                    str(path),
                ],
                cwd=root,
                env=environment,
                check=True,
            )
    for directory in ("controller",):
        paths = sorted((root / directory).glob("*.test.mjs"))
        if args.size == "small":
            paths = [
                path
                for path in paths
                if path.name
                not in (
                    "browser_transport.test.mjs",
                    "live_prediction.test.mjs",
                )
            ]
        subprocess.run(
            args=["node", "--test", *map(str, paths)],
            cwd=root / directory,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    run()
