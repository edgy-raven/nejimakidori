"""Read stored one-hand samples for settlement training."""

import hashlib
import io
import json
import pathlib
import sqlite3

import numpy

from model import feature_vector


def eligible(features, disagreement=False):
    """Disagreement, tenpai, or one-shanten with fewer than 16 wall tiles."""
    distance = (
        min(
            int(features[name + "_self_shanten"])
            for name in ("normal", "chiitoitsu", "kokushi")
        )
        - 1
    )
    return (
        disagreement
        or distance == 0
        or (distance == 1 and int(features["live_wall_count"]) < 16)
    )


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class RolloutStore:
    def __init__(self, path):
        self.connection = sqlite3.connect(
            pathlib.Path(path).resolve().as_uri() + "?mode=ro",
            timeout=60,
            uri=True,
        )

    def close(self):
        self.connection.close()

    def root(self, identity):
        state, packed = self.connection.execute(
            "SELECT state, features FROM roots WHERE id=?", (identity,)
        ).fetchone()
        with numpy.load(io.BytesIO(packed)) as arrays:
            features = dict(arrays)
        if {
            name: {"shape": list(value.shape), "dtype": value.dtype.name}
            for name, value in features.items()
        } != feature_vector.INPUTS:
            raise ValueError(
                "Rollout features do not match the current input schema; "
                "regenerate the root feature cache before training"
            )
        return json.loads(state), features
