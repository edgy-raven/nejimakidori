"""Pushing is measured against safety known at the discard's timestamp."""

import json
import pathlib

import numpy

from log_dataset import records, replay
from model import feature_vector, features


def test_pushing_against_riichi_tracks_time_target_and_hand_discards():
    # South 2, before request 594, with opponents' private tiles masked.
    # https://riichi.dev/games/95089bb9-f2ff-485a-bb82-b780abcfef0e
    events = json.loads(
        (
            pathlib.Path(__file__).parent.parent
            / "fixtures/model/riichi_push.json"
        ).read_text()
    )
    state = replay.Replay.from_start("riichi-push", events[0])
    for event in events[1:]:
        state.apply(event)

    def pack(actor=3):
        packed = feature_vector.FeatureVector().__dict__
        features.table_features(
            packed, state, actor, state.legal_observation(actor)
        )
        return packed

    packed = pack()
    expected = numpy.zeros((4, 4), numpy.int8)
    # Physical seats 0, 1, 3 pushed N/4p/7p, C, and F respectively.
    expected[:, 3] = [1, 3, 1, 0]
    numpy.testing.assert_array_equal(packed["riichi_push_count"], expected)
    # All three tiles are safe now; that must not erase earlier pushing.
    numpy.testing.assert_array_equal(
        packed["genbutsu_to_seat"][3, [30, 12, 15]], [1, 1, 1]
    )
    numpy.testing.assert_array_equal(
        pack(actor=0)["riichi_push_count"][:, 2], [3, 1, 0, 1]
    )
    numpy.testing.assert_array_equal(
        records.decode(
            "feature/riichi_push_count",
            records.encode("feature/riichi_push_count", expected),
            feature_vector.INPUTS["riichi_push_count"],
        ),
        expected,
    )
    state.rivers[0][-1]["tsumogiri"] = True
    assert pack()["riichi_push_count"][1, 3] == 2
    state.rivers[0][-1]["tsumogiri"] = False

    # A second riichi starts counting from its own acceptance. One new
    # red-five hand discard pushes both threats; its ordinary twin is safe.
    state.accepted[1] = True
    state.history.append({"type": "reach_accepted", "actor": 1})
    assert not pack()["riichi_push_count"][:, 2].any()
    for offset, tile in enumerate(("5mr", "5m")):
        state.rivers[0].append(
            {
                **state.rivers[0][-1],
                "pai": tile,
                "event_index": len(state.history) + offset,
                "resolution": 1,
            }
        )
    numpy.testing.assert_array_equal(
        pack()["riichi_push_count"][1, [2, 3]], [1, 4]
    )

    state = replay.Replay.from_start("next-hand", events[0])
    assert not pack()["riichi_push_count"].any()
