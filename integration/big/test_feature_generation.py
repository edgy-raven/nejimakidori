"""Feature extraction preserves native post-discard yaku semantics."""

import json
import pathlib

import numpy

from integration import collection_reference, feature_reference
from log_dataset import features, replay, shanten
from model import actions


def test_discard_yaku_matches_native_transitions_including_reds_and_melds(
    monkeypatch,
):
    checked = set()
    pack = features.pack

    def checked_pack(*args, **kwargs):
        packed = pack(*args, **kwargs)
        for code in numpy.flatnonzero(packed["discard_action_legal"]):
            after = kwargs["state"].native_env.clone()
            after.apply_event(
                {
                    "type": "dahai",
                    "actor": kwargs["actor"],
                    "pai": shanten.TILE_NAMES[code],
                    "tsumogiri": bool(
                        packed["current_draw_valid"]
                        and code == features.drawn_tile_code(packed)
                    ),
                }
            )
            numpy.testing.assert_array_equal(
                packed["discard_action_yaku_possibility"][code],
                feature_reference.yaku_flags(
                    after.get_observation(kwargs["actor"]), kwargs["actor"]
                ),
            )
            checked.add("red" if code >= 34 else "ordinary")
            if kwargs["state"].melds[kwargs["actor"]]:
                checked.add("melds")
        return packed

    def predict(inputs):
        return {
            name: numpy.zeros(
                (
                    len(inputs["decision_phase"]),
                    actions.ACTOR_OUTPUTS[name][0],
                ),
                numpy.float32,
            )
            for name in actions.TRAIN_POLICY_HEADS
        }

    monkeypatch.setattr(features, "pack", checked_pack)
    collection_reference.collect(predict, 2, 12)
    assert checked == {"red", "ordinary", "melds"}
    assert [
        shanten.mjai_tile_code(tile) for tile in shanten.TILE_NAMES
    ] == list(range(37))


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
        packed["action_value_range"][27, 0], [[2000, 3900], [2700, 5200]]
    )
    numpy.testing.assert_array_equal(
        packed["action_value_range"][24, 0], [[1300, 3900], [2000, 5200]]
    )
    for cut in (24, 27):
        assert packed["action_value_eligible"][cut]
        numpy.testing.assert_array_equal(
            packed["action_value_range"][37 + cut],
            packed["action_value_range"][cut],
        )
        assert not packed["action_value_valid"][cut, 1].any()
    numpy.testing.assert_array_equal(
        packed["action_value_range"][74 + 27],
        packed["action_value_range"][27],
    )
    assert not packed["action_value_eligible"][74 + 24]


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
                packed["last_tedashi_tile"],
                numpy.eye(37, dtype=numpy.int8)[[32, 15, 1, 2]],
            )
