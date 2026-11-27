"""Opponent requests expose only seat-visible incremental game events."""

import io
import json
import types

import riichienv

from training import mortal_policy


def test_mortal_wire_hides_other_hands_and_draws_and_resets():
    policy = mortal_policy.MortalPolicy.__new__(mortal_policy.MortalPolicy)
    policy.process = types.SimpleNamespace(stdin=io.StringIO())
    policy.reset([14])
    env = riichienv.RiichiEnv(game_mode=2, seed=14)
    observations = env.reset()
    seen = set()
    for step in range(12):
        policy.process.stdin.seek(0)
        policy.process.stdin.truncate()
        policy.send_requests(
            [
                (seat, object(), observation)
                for seat, observation in observations.items()
            ]
        )
        request = json.loads(policy.process.stdin.getvalue())
        assert request["reset"] == (step == 0)
        for item in request["items"]:
            for raw in item["events"]:
                event = json.loads(raw)
                if event["type"] == "start_kyoku":
                    for seat, hand in enumerate(event["tehais"]):
                        assert ("?" in hand) == (seat != item["seat"])
                    seen.add("hands")
                elif event["type"] == "tsumo":
                    assert (event["pai"] == "?") == (
                        event["actor"] != item["seat"]
                    )
                    seen.add("draws")
        observations = env.step(
            {
                seat: observation.legal_actions()[0]
                for seat, observation in observations.items()
            }
        )
    assert seen == {"hands", "draws"}
    policy.reset([14])
    assert policy.reset_pending
