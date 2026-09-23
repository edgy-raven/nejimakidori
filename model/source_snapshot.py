"""Immutable source snapshots and stable content hashes."""

import hashlib
import pathlib
import shutil


def hashes(root):
    root = pathlib.Path(root)
    return {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def freeze(source, target, include_native=False):
    source, target = pathlib.Path(source), pathlib.Path(target)
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            ".pytest_cache",
            "*.pyc",
            *(() if include_native else (".native", "*.so"))
        ),
    )
    return hashes(target)
