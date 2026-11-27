"""Feature extraction preserves native post-discard yaku semantics."""

import json
import pathlib

import numpy

from log_dataset import features, replay
from model import actions


def test_resulting_hand_yaku_accounts_for_the_discarded_public_copy():
    events = json.loads(
        pathlib.Path(
            "integration/fixtures/model/pass_speed_round.json"
        ).read_text()
    )
    row = next(
        row
        for row in replay.Replay("haku-availability", events).observations()
        if row.decision_index == 6 and row.actor == 3
    )
    assert row.features["discard_action_legal"][31]
    assert row.features["candidate_hand_counts"][0, 31] == 1
    assert row.features["known_unavailable_counts"][31] == 2
    assert row.features["candidate_hand_yaku_possibility"][0, 1] == 1
    assert (
        row.features["candidate_hand_yaku_possibility"][
            row.features["discard_candidate_hand"][31], 1
        ]
        == 0
    )


def test_reachable_ranges_follow_each_discard_and_riichi_slot():
    start = dict(
        type="start_kyoku",
        bakaze="E",
        dora_marker="9s",
        honba=0,
        kyoku=1,
        kyotaku=0,
        oya=0,
        scores=[25000] * 4,
        tehais=[["?"] * 13 for _ in range(4)],
    )
    start["tehais"][1] = "1m 2m 3m 4m 5m 6m 7p 8p 9p 6s E 2p 2p".split()
    state = replay.Replay.from_start("ranges", start)
    state.apply(dict(type="tsumo", actor=1, pai="7s"))
    packed = features.pack(state, 1, "DISCARD")
    # East reaches tenpai; cutting 7s instead searches a one-shanten hand.
    numpy.testing.assert_array_equal(
        packed["action_win_value_range"][27], [[1000, 2000], [1500, 2700]]
    )
    numpy.testing.assert_array_equal(
        packed["action_win_value_range"][24], [[1000, 2000], [1100, 2700]]
    )
    for cut in (24, 27):
        assert packed["action_value_eligible"][cut]
    numpy.testing.assert_array_equal(
        packed["action_win_value_range"][37 + 27],
        [[2000, 3900], [2700, 5200]],
    )
    assert not packed["action_value_eligible"][37 + 24]


def test_reported_public_safety_features_reach_native_inputs():
    for filename, actor in (("passed_7m", 2), ("riichi_push", 3)):
        events = json.loads(
            pathlib.Path(
                f"integration/fixtures/model/{filename}.json"
            ).read_text()
        )
        state = replay.Replay.from_start(filename, events[0])
        for event in events[1:]:
            state.apply(event)
        phase = actions.phase_for_actions(
            state.legal_observation(actor).legal_actions()
        )
        packed = features.pack(state, actor, phase)
        if filename == "passed_7m":
            assert packed["passed_unchanged_to_seat"][2, 6] == 1
            assert packed["genbutsu_to_seat"][2, 6] == 0
            assert packed["passed_unchanged_to_seat"][2, 1] == 0
        else:
            expected = numpy.zeros((4, 4), numpy.int8)
            expected[:, 3] = [1, 3, 1, 0]
            numpy.testing.assert_array_equal(
                packed["riichi_push_count"], expected
            )
            # Later safety must not erase earlier pushes.
            numpy.testing.assert_array_equal(
                packed["genbutsu_to_seat"][3, [30, 12, 15]], [1, 1, 1]
            )
            numpy.testing.assert_array_equal(
                packed["last_tedashi_tile_id"],
                [32, 15, 1, 2],
            )


def test_hatsu_pair_dama_cannot_ron_but_riichi_can():
    # East 2, 1 honba: 234345m 123p red-56s FFF, dora 3m.
    start = dict(
        type="start_kyoku",
        bakaze="E",
        dora_marker="2m",
        honba=1,
        kyoku=2,
        kyotaku=0,
        oya=1,
        scores=[25000] * 4,
        tehais=[["?"] * 13 for _ in range(4)],
    )
    start["tehais"][0] = "2m 3m 4m 3m 4m 5m 1p 2p 3p 5sr 6s F F".split()
    state = replay.Replay.from_start("hatsu-dama", start)
    state.apply(dict(type="tsumo", actor=0, pai="F"))
    packed = features.pack(state, 0, "RIICHI")
    assert (
        packed["candidate_hand_counts"][
            packed["discard_candidate_hand"][32], 32
        ]
        == 2
    )
    numpy.testing.assert_array_equal(
        packed["action_win_value_valid"][[32, 69]], [[0, 1], [1, 1]]
    )
    numpy.testing.assert_array_equal(
        packed["action_win_value_range"][[32, 69]],
        [[[0, 0], [7900, 7900]], [[8000, 8000], [8000, 8000]]],
    )
    state.apply(dict(type="reach", actor=0))
    state.apply(dict(type="dahai", actor=0, pai="F", tsumogiri=True))
    state.apply(dict(type="reach_accepted", actor=0))
    state.apply(dict(type="tsumo", actor=0, pai="9m"))
    accepted = features.pack(state, 0, "DISCARD")
    numpy.testing.assert_array_equal(
        accepted["action_win_value_range"][8],
        packed["action_win_value_range"][69],
    )
