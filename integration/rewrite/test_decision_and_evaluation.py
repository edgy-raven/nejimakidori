"""Serving choices and duplicate-evaluation result contracts."""

import numpy
import riichienv

import feature_vector
from model import actions, head_to_head, replay
from service import web_adapter


class ZeroPredictor:
    """A serving-compatible predictor whose scores leave legal masking visible."""

    def infer(self, instances):
        outputs = {
            name: numpy.zeros((len(instances), *shape), numpy.float32)
            for name, shape in feature_vector.ACTOR_OUTPUTS.items()
        }
        return type(
            "Result",
            (),
            {
                "latency_ms": 0.0,
                "outputs": outputs,
                "as_dict": lambda self: {"predictions": []},
            },
        )()


def played_game():
    environment = riichienv.RiichiEnv(
        seed=14, rule=riichienv.GameRule.default_tenhou()
    )
    observations = environment.reset()
    while not environment.done():
        observations = environment.step(
            {
                seat: observation.legal_actions()[0]
                for seat, observation in observations.items()
            }
        )
    return environment.mjai_log


def browser_tile(tile):
    if tile in "ESWNPFC":
        return str("ESWNPFC".index(tile) + 1) + "z"
    return "0" + tile[1] if tile.endswith("r") else tile


def test_serving_recommends_only_a_native_legal_discard():
    events = played_game()
    start = events[1]
    draw = events[2]
    round_data = {
        "prevailingWind": 0,
        "kyoku": 1,
        "honba": 0,
        "riichiSticks": 0,
        "scores": start["scores"],
        "doraIndicators": [browser_tile(start["dora_marker"])],
        "hands": [
            [browser_tile(tile) for tile in hand] for hand in start["tehais"]
        ],
        "events": [
            {
                "type": "draw",
                "seat": draw["actor"],
                "tile": browser_tile(draw["pai"]),
            }
        ],
    }

    result = web_adapter.predict_live(
        ZeroPredictor(), round_data, draw["actor"]
    )

    assert result["decision"]["phase"] == "DISCARD"
    assert result["decision"]["action_kind"] == "recommended"
    assert (
        result["decision"]["action"] in result["decision"]["possible_actions"]
    )


def test_replay_masks_call_choices_and_bypasses_win_inference():
    events = played_game()
    state = replay.Replay.from_start("rewrite", events[1])
    saw_call = saw_win = False
    for index, event in enumerate(events[2:], 2):
        state.apply(event)
        following = events[index + 1] if index + 1 < len(events) else None
        if not saw_call and following and following["type"] in {"chi", "pon"}:
            actor = following["actor"]
            phase, request = actions.policy_request(
                state, actor, state.legal_observation(actor)
            )
            selected, _ = actions.select_policy_action(
                request,
                ZeroPredictor().infer([request["features"]]).outputs,
                0,
            )
            assert phase == "RESPONSE"
            assert selected in request["legal_actions"]
            saw_call = True
        if not saw_win and following and following["type"] == "hora":
            actor = following["actor"]
            selected = actions.forced_policy_action(
                state, actor, state.legal_observation(actor).legal_actions()
            )
            assert selected is not None
            assert selected.action_type in {
                riichienv.ActionType.RON,
                riichienv.ActionType.TSUMO,
            }
            saw_win = True
    assert saw_call
    assert saw_win


def test_duplicate_summary_requires_all_four_candidate_rotations():
    rows = [
        {
            "format": "riichienv_duplicate_south",
            "seed": 91,
            "new_seats": [seat],
            "ranks": [1, 2, 3, 4],
            "scores": [34000, 28000, 22000, 16000],
            "rank_points": [125, 60, -5, -255],
        }
        for seat in range(4)
    ]

    summary = head_to_head.summarize(rows, bootstrap_seed=3)

    assert summary["format"] == "riichienv_duplicate_south"
    assert summary["compared_games"] == 4
    assert summary["seed_groups"] == 1
    assert head_to_head.summarize(rows[:3]) is None


def test_duplicate_evaluation_only_queries_the_controlling_policy(
    tmp_path, monkeypatch
):
    class LegalPolicy:
        def __init__(self, candidate):
            self.candidate = candidate
            self.requests = 0

        def reset(self):
            pass

        def select_many(self, requests, batch_size):
            for stream, _, observation in requests:
                game, actor = divmod(stream, 4)
                assert (actor == game) == self.candidate
                assert observation.player_id == actor
            self.requests += len(requests)
            return [
                observation.legal_actions()[0] for _, _, observation in requests
            ]

    policies = {"new": LegalPolicy(True), "old": LegalPolicy(False)}
    monkeypatch.setattr(head_to_head, "WORKER_POLICIES", policies)
    (tmp_path / "replays").mkdir()
    result = head_to_head.play_games(
        [(tmp_path, 14, seat) for seat in range(4)]
    )
    assert len(result) == 4
    assert all(policy.requests > 0 for policy in policies.values())
