"""Opponent requests expose only seat-visible incremental game events."""

import io
import json
import types

from training import distill_policy

import riichienv


def test_distill_wire_hides_other_hands_and_draws_and_resets():
    policy = distill_policy.DistillPolicy.__new__(distill_policy.DistillPolicy)
    policy.process = types.SimpleNamespace(stdin=io.StringIO())
    policy.reset()
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
    policy.reset()
    assert policy.reset_pending
