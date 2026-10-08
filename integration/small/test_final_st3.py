"""Final ST3 evaluation scoring and target isolation."""

import json
import pathlib

import numpy

from log_dataset import afk, records, replay
from model import feature_vector, outcomes


def test_full_scoring_and_final_rank_record_roundtrip():
    numpy.testing.assert_array_equal(
        outcomes.saint3_jade_south([35100, 25000, 24900, -1000], [1, 2, 3, 4]),
        [136, 60, -5, -281],
    )
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model")
        .joinpath("payment_dealer_90fu.json")
        .read_text()
    )
    hanchan = replay.hanchan_targets(events)
    rows = list(
        replay.Replay(
            "st3-targets", afk.retained_hands(events)[0], hanchan=hanchan
        ).observations()
    )[:8]
    inputs, labels, _ = records.parse_batch(
        [records.serialize_observation(row, True) for row in rows]
    )
    assert "final_score" not in labels
    numpy.testing.assert_array_equal(
        labels["final_placement"],
        [hanchan["ranks"][row.actor] - 1 for row in rows],
    )
    assert {"final_score", "final_placement"}.isdisjoint(inputs)
    assert set(inputs) == set(feature_vector.INPUTS)
