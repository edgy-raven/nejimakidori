"""Latest hand discards are categorical, public, and actor-relative."""

import json
import pathlib

import numpy

from model import feature_vector, features
from log_dataset import records, replay


def test_last_tedashi_survives_tsumogiri_and_resets_with_the_hand():
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/riichi_push.json")
        .read_text()
    )
    state = replay.Replay.from_start("last-tedashi", events[0])
    packed = feature_vector.FeatureVector().tensors
    features.table_features(packed, state, 3, state.legal_observation(3))
    assert not packed["last_tedashi_tile"].any()
    for event in events[1:]:
        state.apply(event)
    features.table_features(packed, state, 3, state.legal_observation(3))
    # Actor 3: F, 7p, 2m, then seat 2's 3m declaration followed by tsumogiri.
    numpy.testing.assert_array_equal(
        packed["last_tedashi_tile"],
        numpy.eye(37, dtype=numpy.int8)[[32, 15, 1, 2]],
    )
    features.table_features(packed, state, 0, state.legal_observation(0))
    numpy.testing.assert_array_equal(
        packed["last_tedashi_tile"],
        numpy.eye(37, dtype=numpy.int8)[[15, 1, 2, 32]],
    )

    # Red and ordinary fives remain distinct. A drawn discard cannot
    # replace the latest hand-discarded red five.
    state.rivers[0].append(
        {**state.rivers[0][-1], "pai": "5mr", "tsumogiri": False}
    )
    state.rivers[0].append(
        {**state.rivers[0][-1], "pai": "5m", "tsumogiri": True}
    )
    features.table_features(packed, state, 0, state.legal_observation(0))
    numpy.testing.assert_array_equal(
        packed["last_tedashi_tile"][0], numpy.eye(37, dtype=numpy.int8)[34]
    )
    numpy.testing.assert_array_equal(
        records.decode(
            "feature/last_tedashi_tile",
            records.encode(
                "feature/last_tedashi_tile", packed["last_tedashi_tile"]
            ),
            feature_vector.INPUTS["last_tedashi_tile"],
        ),
        packed["last_tedashi_tile"],
    )
    state = replay.Replay.from_start("next-hand", events[0])
    features.table_features(packed, state, 0, state.legal_observation(0))
    assert not packed["last_tedashi_tile"].any()
