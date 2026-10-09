"""Tile policy execution randomizes only interchangeable legal copies."""

import types

import numpy
import riichienv

from model import actions


def test_declining_optional_actions_preserves_drawn_tile():
    discard = riichienv.Action(riichienv.ActionType.DISCARD, tile=18)
    request = {
        "phase": "WIN",
        "drawn_tile": 18,
        "legal_actions": [
            riichienv.Action(riichienv.ActionType.TSUMO),
            riichienv.Action(riichienv.ActionType.ANKAN, tile=0),
            discard,
        ],
        "features": {"win_kind": numpy.array(1, numpy.int8)},
    }
    request = actions.continue_request(request)
    assert request["phase"] == "KAN"
    assert request["drawn_tile"] == 18
    request = actions.continue_request(request)
    assert request["phase"] == "DISCARD"
    assert request["drawn_tile"] == 18
    assert request["legal_actions"] == [discard]
    assert request["features"]["win_kind"] == 0


def test_discard_copies_preserve_tile_red_identity_and_forced_legality(
    monkeypatch,
):
    legal = [
        riichienv.Action(riichienv.ActionType.DISCARD, tile=tile)
        for tile in (16, 17, 18, 20)
    ]
    # Exercise both ends of the physical-copy draw without a flaky sample.
    for offset in (0, -1):
        monkeypatch.setattr(
            actions.numpy.random, "randint", lambda count: offset % count
        )
        assert actions.select_discard(legal, 4, 18).tile == (17, 18)[offset]
        assert actions.select_discard(legal, 34, 18).tile == 16
        assert actions.select_discard(legal, 5, 18).tile == 20
        state = types.SimpleNamespace(
            legal_observation=lambda actor: types.SimpleNamespace(drawn_tile=18)
        )
        assert (
            actions.forced_policy_action(state, 0, legal[1:3]).tile
            == (17, 18)[offset]
        )
        assert actions.forced_policy_action(state, 0, legal[2:3]).tile == 18
