"""Atomic progress metadata for training and monitoring."""

import json


def write_json(path, value):
    """Replace run metadata atomically so readers never see partial JSON."""
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
