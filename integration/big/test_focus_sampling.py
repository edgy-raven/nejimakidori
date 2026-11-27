"""Focused priorities and retained, unused placement-gap metadata."""

import json
import pathlib
import types

import numpy

from log_dataset import features, focus_sampling, replay
from training.corrections import collection


def observation():
    state = replay.Replay(
        "focus",
        json.loads(
            pathlib.Path(
                "integration/fixtures/model/pass_speed_round.json"
            ).read_text()
        ),
    )
    _, actor, phase, _, _ = next(state.policy_decisions())
    return types.SimpleNamespace(features=features.pack(state, actor, phase))


def test_recorded_decisions_preserve_route_and_threat_priorities():
    # Recorded cases cover route weights, calls, overlapping threats,
    # late-hand placement and gap-only rows.
    expected = {
        "pass_speed_round": {
            (3, 1, "DISCARD"): (2, False),  # Tanyao.
            (5, 0, "DISCARD"): (1, False),  # Outside family.
            (9, 2, "DISCARD"): (2, False),  # Flush family.
            (37, 3, "DISCARD"): (3, False),  # Tanyao + threat.
        },
        "sakigiri_swap": {
            (3, 0, "DISCARD"): (0, True),  # Gap alone is separate.
            (28, 2, "RESPONSE"): (3, False),  # Tanyao + any call.
        },
        "payment_chankan_ippatsu": {
            (1, 3, "DISCARD"): (3, True),  # Tanyao + late.
            (79, 3, "RIICHI"): (2, True),  # Threat + late.
            (87, 2, "KAN"): (5, True),  # Call + tanyao + threat + late.
        },
    }
    for name, decisions in expected.items():
        state = replay.Replay(
            name,
            json.loads(
                pathlib.Path(
                    f"integration/fixtures/model/{name}.json"
                ).read_text()
            ),
        )
        actual = {}
        for _, actor, phase, _, _ in state.policy_decisions():
            key = (state.event_index, actor, phase)
            if key in decisions:
                actual[key] = focus_sampling.metadata(
                    types.SimpleNamespace(
                        features=features.pack(state, actor, phase)
                    )
                )
        assert actual == decisions


def test_placement_thresholds_follow_trailing_seat_and_preserve_ties():
    row = observation()
    row.features["hands_remaining"] = numpy.int8(0)
    row.features["prevailing_wind"] = numpy.int8(1)
    # Placement-gap thresholds are strict and
    # depend on whether the trailing player still has a scheduled dealership.
    for scores, dealer, expected in (
        ([41000, 30000, 20000, 9000], 2, [True, False, False, False]),
        ([41000, 30000, 20000, 9000], 1, [False, True, False, False]),
        ([41000, 30000, 20000, 9000], 3, [True, False, True, True]),
        ([39000, 30000, 20000, 11000], 0, [False, True, True, True]),
        ([40000, 30000, 20000, 10000], 0, [False] * 4),
        ([46000, 30000, 20000, 10000], 1, [False] * 4),
        ([30000, 30000, 20000, 20000], 0, [False, False, True, True]),
    ):
        # Wire fields use E/S/W/N, while the querying actor may be any seat.
        row.features["scores"] = numpy.roll(numpy.array(scores), -dealer)
        row.features["ranks"] = numpy.roll(numpy.arange(1, 5), -dealer)
        actual = []
        for actor in range(4):
            row.features["seat_wind"] = (actor + numpy.arange(4) - dealer) % 4
            actual.append(focus_sampling.metadata(row)[1])
        assert actual == expected
    row.features["hands_remaining"] = numpy.int8(7)
    row.features["prevailing_wind"] = numpy.int8(0)
    assert focus_sampling.metadata(row)[1] is False


def test_public_focus_excludes_unknown_routes_and_ineligible_decisions(
    monkeypatch,
):
    row = observation()
    # A bounded route search may genuinely be incomplete; positive rounded
    # training labels must not invent support for that search.
    monkeypatch.setattr(focus_sampling.shanten, "tenpai_paths", lambda *a: None)
    row.candidate_yaku_affinity = numpy.ones((16, 3))
    assert focus_sampling.metadata(row) == (0, False)
    row.features["riichi_state"][1] = 1
    assert focus_sampling.metadata(row) == (1, False)
    row.features["riichi_state"][1] = 0
    row.features["meld_type"][1, :2] = [1, 1]
    row.features["meld_tiles"][1, :2] = [[0, 1, 2, 37], [3, 4, 5, 37]]
    row.features["opponent_discard_count"][1] = 8
    assert focus_sampling.metadata(row) == (0, False)
    row.features["opponent_discard_count"][1] = 9
    assert focus_sampling.metadata(row) == (1, False)
    row.features["opponent_discard_count"][1] = 8
    row.features["river_pre_tile_id"][0, :2] = [12, 21]
    row.features["river_pre_valid"][0, :2] = 1
    row.features["dora_multiplicity"][0] = 1
    assert focus_sampling.metadata(row) == (1, False)
    row.features["hands_remaining"] = numpy.int8(1)
    assert focus_sampling.metadata(row) == (2, False)
    row.features["discard_action_legal"][:] = 0
    row.features["discard_action_legal"][0] = 1
    assert focus_sampling.metadata(row) == (0, False)
    row.features["decision_phase"] = numpy.int8(4)
    assert focus_sampling.metadata(row) == (0, False)


def test_correction_threat_uses_historical_rule_and_correct_kan_encoding():
    row = observation()
    row.features["riichi_state"][:] = 0
    row.features["meld_type"][:] = 0
    row.features["meld_tiles"][:] = 37
    row.features["opponent_discard_count"][:] = 0
    row.features["dora_multiplicity"][:] = 0
    row.features["meld_type"][1, :3] = [1, 1, 4]
    row.features["meld_tiles"][1, :3] = [
        [0, 1, 2, 37],
        [9, 10, 11, 37],
        [18, 18, 18, 18],
    ]
    row.features["normal_self_shanten"] = 1
    row.features["chiitoitsu_self_shanten"] = 4
    row.features["kokushi_self_shanten"] = 4
    row.features["discard_action_all_shanten"][:2] = [1, 2]
    legal = numpy.zeros(150, bool)
    legal[:2] = True
    # Two early calls plus a closed kan are not three open melds.
    # Changing that kan to open must activate the very same public rule.
    for kind, expected in ((4, None), (3, (1, 0))):
        row.features["meld_type"][1, 2] = kind
        assert focus_sampling.visible_threat(row.features, 1) == (
            expected is not None
        )
        assert (
            collection.action_pair(
                features=row.features,
                head=0,
                legal=legal,
                policy=numpy.zeros(150),
                category="tenpai_threat",
                preferred=1,
            )
            == expected
        )
