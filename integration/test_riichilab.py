"""RiichiLab wire actions preserve masked histories and request identity."""

import asyncio
import json
import types
import unittest.mock

import pytest
import riichienv

import bots.riichilab


class DrawPolicy:
    def reset(self):
        self.previous_events = []

    def select(self, env, observation):
        assert env.mjai_log[: len(self.previous_events)] == self.previous_events
        self.previous_events = list(env.mjai_log)
        for event in env.mjai_log:
            if event["type"] == "start_kyoku":
                assert all(
                    tile == "?"
                    for seat, hand in enumerate(event["tehais"])
                    if seat != observation.player_id
                    for tile in hand
                )
        return next(
            (
                action
                for action in observation.legal_actions()
                if action.action_type == riichienv.ActionType.DISCARD
                and action.tile == observation.drawn_tile
            ),
            observation.legal_actions()[0],
        )


def test_complete_masked_game_and_request_binding():
    policy = DrawPolicy()
    result = bots.riichilab.local_game(policy, 42)
    assert result["decisions"] > 40
    assert len(result["scores"]) == 4
    assert sum(e["type"] == "start_kyoku" for e in policy.previous_events) >= 4

    policy.reset()
    observation = riichienv.RiichiEnv(seed=42).reset()[0]
    message = {
        "observation": observation.serialize_to_base64(),
        "possible_actions": [
            json.loads(a.to_mjai()) for a in observation.legal_actions()
        ],
        "request_id": 73,
    }
    response = bots.riichilab.choose(policy, [], message)
    assert response["request_id"] == 73
    assert response["tsumogiri"] is True
    assert observation.select_action_from_mjai(json.dumps(response)) is not None

    policy.reset()
    message["possible_actions"] = [{"type": "none"}]
    with pytest.raises(ValueError, match="possible_actions"):
        bots.riichilab.choose(policy, [], message)


@pytest.mark.parametrize("games", [2, 0])
def test_bounded_and_continuous_sessions_close_policy(
    tmp_path, monkeypatch, games
):
    policy = unittest.mock.Mock()
    monkeypatch.setattr(
        bots.riichilab.model.frozen_policy,
        "FrozenPolicy",
        lambda **kwargs: policy,
    )
    monkeypatch.setattr(bots.riichilab, "local_game", lambda *args: {})
    played = []

    async def play(session, selected_policy, args):
        assert selected_policy is policy
        played.append(True)
        if games == 0 and len(played) == 3:
            raise ConnectionError("Simulated disconnect")

    monkeypatch.setattr(bots.riichilab, "play", play)
    args = types.SimpleNamespace(
        output=tmp_path,
        model="release",
        source="src",
        device="gpu",
        memory_mb=2048,
        mode="ranked",
        games=games,
    )
    if games:
        asyncio.run(bots.riichilab.run(args))
        assert len(played) == 2
    else:
        with pytest.raises(ConnectionError, match="Simulated disconnect"):
            asyncio.run(bots.riichilab.run(args))
        assert len(played) == 3
    policy.close.assert_called_once()
