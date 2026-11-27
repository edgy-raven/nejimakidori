"""Future defensive dilemmas select only safe earlier equal-shanten cuts."""

import json
import pathlib

import numpy

from log_dataset import features, replay, sakigiri_regret, shanten


def test_calls_allow_all_earlier_alternatives_before_first_dilemma():
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/sakigiri_calls.json")
        .read_text()
    )
    witnesses = sakigiri_regret.round_pairs("calls", events)
    # Seat 2 calls chi at 55. At 56, 2p is genbutsu to the threat;
    # 3s and 5s tie its best shanten, although only 3s can actually be ronned.
    expected = {
        14: [22, 9],
        22: [20, 22, 1],
        30: [20, 22, 26],
        38: [20, 22, 9],
        46: [20, 22, 18],
    }
    assert {
        index: witness
        for index, witness in witnesses.items()
        if events[index]["actor"] == 2
    } == {index: (actions, 0.2) for index, actions in expected.items()}
    assert set(witnesses) == {8, 14, 16, 22, 24, 30, 32, 38, 40, 46, 48}
    assert witnesses[40] == ([0, 1, 11, 18, 21, 25, 30, 7], 1 / 6)
    assert 56 not in witnesses and 58 not in witnesses
    # Later tenpai and further danger cannot add roots after the first endpoint.
    assert sakigiri_regret.round_pairs("calls", events[:57]) == {
        index: (actions, 0.2) for index, actions in expected.items()
    }
    state = replay.Replay.from_start("calls", events[0])
    for index, event in enumerate(events[1:57], 1):
        if index in witnesses:
            actions = witnesses[index][0]
            assert actions[-1] == shanten.mjai_tile_code(event["pai"])
            assert (
                len(
                    {
                        shanten.calculate_shanten(
                            state.concealed_counts(event["actor"])
                            - numpy.eye(34, dtype=int)[shanten.base_id(tile)]
                        )
                        for tile in actions
                    }
                )
                == 1
            )
            assert not any(
                sakigiri_regret.ron_exposure(
                    state, event["actor"], actions
                ).values()
            )
        if index == 56:
            packed = features.pack(state, 2, "DISCARD")
            numpy.testing.assert_array_equal(
                packed["genbutsu_to_seat"][3, [10, 20, 22]], [1, 0, 0]
            )
            numpy.testing.assert_array_equal(
                packed["discard_action_all_shanten"][[10, 20, 22]], [2, 2, 2]
            )
            assert sakigiri_regret.ron_exposure(state, 2, [20, 22]) == {
                20: 1,
                22: 0,
            }
        state.apply(event)


def test_early_shanten_filter_does_not_require_actual_ron_at_endpoint():
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/last_discard_ron_pass.json")
        .read_text()
    )
    witnesses = sakigiri_regret.round_pairs("shanten", events)
    assert set(witnesses) == set(range(6, 103, 8))
    # 5m/7m/6s worsen shanten at the early roots, then become equal-shanten
    # options later. Preserve every qualifying tile, not one arbitrary pair.
    assert witnesses[6] == ([7, 28], 1 / 13)
    assert witnesses[54] == ([7, 23, 19], 1 / 13)
    assert witnesses[62] == ([4, 6, 7, 23, 25], 1 / 13)
    assert 112 not in witnesses
    state = replay.Replay.from_start("shanten", events[0])
    for event in events[1:112]:
        state.apply(event)
    assert sakigiri_regret.ron_exposure(state, 1, [4, 6, 7, 23]) == {
        4: 0,
        6: 0,
        7: 0,
        23: 0,
    }


def test_later_actual_danger_without_a_defensive_dilemma_does_not_qualify():
    for name in ("sakigiri_swap", "recorded_nonred_pon"):
        events = json.loads(
            pathlib.Path(__file__)
            .parent.parent.joinpath(f"fixtures/model/{name}.json")
            .read_text()
        )
        assert sakigiri_regret.round_pairs(name, events) == {}
