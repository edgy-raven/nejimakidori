"""Build the Linux native collector for the current Python interpreter."""

import os
import pathlib
import shutil
import subprocess
import sys


def run():
    root = pathlib.Path(__file__).resolve().parent.parent
    target = pathlib.Path(
        os.environ.get(
            "CARGO_TARGET_DIR", root / "log_dataset/native_collection/target"
        )
    ).resolve()
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "--manifest-path",
            str(root / "log_dataset/native_collection/Cargo.toml"),
        ],
        cwd=root,
        env={
            **os.environ,
            "PYO3_PYTHON": sys.executable,
            "CARGO_TARGET_DIR": str(target),
        },
        check=True,
    )
    destination = root / "log_dataset/_native_collection.so"
    temporary = destination.with_suffix(".so.tmp")
    shutil.copyfile(target / "release/lib_native_collection.so", temporary)
    temporary.replace(destination)


if __name__ == "__main__":
    run()
