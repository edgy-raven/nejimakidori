"""Passed-tile evidence follows public hand changes, not hidden waits."""

import json
import pathlib

import numpy

import feature_vector
from model import features, records, replay


def test_passed_tile_survives_tsumogiri_and_expires_on_hand_changes():
    # East 3, just before request 188; all opponent hands/draws are masked.
    # https://riichi.dev/games/12756256-1419-4c8c-9c22-92ba235fc707
    events = json.loads(
        (pathlib.Path(__file__).parent / "fixtures/passed_7m.json").read_text()
    )
    state = replay.Replay.from_start("passed-7m", events[0])
    for event in events[1:]:
        state.apply(event)

    def pack(actor=2):
        packed = feature_vector.FeatureVector().__dict__
        features.table_features(
            packed, state, actor, state.legal_observation(actor)
        )
        return packed

    packed = pack()
    assert packed["passed_unchanged_to_seat"][2, 6] == 1
    assert packed["genbutsu_to_seat"][2, 6] == 0
    assert packed["passed_unchanged_to_seat"][2, 1] == 0
    assert pack(actor=0)["passed_unchanged_to_seat"][0, 6] == 1
    numpy.testing.assert_array_equal(
        records.decode(
            "feature/passed_unchanged_to_seat",
            records.encode(
                "feature/passed_unchanged_to_seat",
                packed["passed_unchanged_to_seat"],
            ),
            feature_vector.INPUTS["passed_unchanged_to_seat"],
        ),
        packed["passed_unchanged_to_seat"],
    )

    # The same observed river, before the server resolves the 7m response.
    state.rivers[3][-1]["resolution"] = 0
    assert pack()["passed_unchanged_to_seat"][2, 6] == 0
    state.rivers[3][-1]["resolution"] = 1

    # Momochan's following South was tsumogiri. Tedashi would erase the
    # inference, even when discarding the same publicly visible tile.
    state.rivers[0][-1]["tsumogiri"] = False
    assert pack()["passed_unchanged_to_seat"][2, 6] == 0
    state.rivers[0][-1]["tsumogiri"] = True
    assert pack()["passed_unchanged_to_seat"][2, 6] == 1

    for kind in ("chi", "pon", "daiminkan", "ankan", "kakan"):
        state.meld_events[0].append(
            {"type": kind, "event_index": len(state.history)}
        )
        assert pack()["passed_unchanged_to_seat"][2, 6] == 0
        state.meld_events[0].pop()
    state.meld_events[0][0]["upgrade_index"] = len(state.history)
    assert pack()["passed_unchanged_to_seat"][2, 6] == 0

    state = replay.Replay.from_start("next-hand", events[0])
    assert not pack()["passed_unchanged_to_seat"].any()
