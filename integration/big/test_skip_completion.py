"""Collection and cached training inputs agree without completion or upgrade search."""

import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import _native_collection, features, replay, shanten
from model import encoder


def test_retired_analysis_is_skipped_and_cached_inputs_are_ignored(monkeypatch):
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    state = replay.Replay("skip-completion", events)
    next(state.policy_decisions())
    actor = state.history[-1]["actor"]

    def rejected_python_analysis(*args, **kwargs):
        raise AssertionError("feature extraction must stay native")

    monkeypatch.setattr(shanten, "native", rejected_python_analysis)
    packed = features.pack(state, actor, "DISCARD")
    assert not any(name.startswith("discard_completion_") for name in packed)
    for name in _native_collection.IGNORED_INPUTS:
        numpy.testing.assert_array_equal(packed[name], 0)
        packed[name].fill(1)
    fresh = features.pack(state, actor, "DISCARD")
    for name in _native_collection.IGNORED_INPUTS:
        numpy.testing.assert_array_equal(fresh[name], 0)
    packed = fresh
    inputs, board, _, _, _ = encoder.encode_inputs()
    board_model = tf.keras.Model(inputs, board)
    batch = {name: value[None] for name, value in packed.items()}
    expected = board_model(batch).numpy()
    # Retired fields in cached records cannot influence the encoder.
    for name in _native_collection.IGNORED_INPUTS:
        batch[name] = numpy.ones_like(batch[name])
    numpy.testing.assert_array_equal(board_model(batch), expected)

    # Exact candidate distances reach the encoder, unlike retired inputs.
    batch["candidate_yaku_shanten"] = batch["candidate_yaku_shanten"] + 1
    assert not numpy.allclose(board_model(batch), expected)
