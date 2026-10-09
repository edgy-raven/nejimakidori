"""Expose wheel-provided CUDA libraries before starting the model service."""

import os
import pathlib
import sys
import sysconfig

os.environ["LD_LIBRARY_PATH"] = ":".join(
    [
        *filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(":")),
        *map(
            str,
            (pathlib.Path(sysconfig.get_path("purelib")) / "nvidia").glob(
                "*/lib"
            ),
        ),
    ]
)
os.execvp(sys.argv[1], sys.argv[1:])
