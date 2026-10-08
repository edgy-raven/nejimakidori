"""HTTP flows with local game fixtures and a deterministic model boundary."""

import asyncio
import json
import pathlib
import threading

import aiohttp.test_utils
import numpy
import service.serve
from log_dataset import review, shanten
from model import actions, decision_layers, features, inference

import riichienv


class LocalPredictor:
    model_path = pathlib.Path("/fixture/export")
    metadata = {"model_contract": "fixture-payments", "revision": "fixture"}

    def __init__(self):
        self.calls = []

    def infer(self, instances):
        self.calls.extend(instances)
        predictions = []
        for instance in instances:
            if "ticket" in instance:
                predictions.append({"ticket": instance["ticket"]})
                continue
            prediction = {
                head + "_probabilities": [1 / size] * size
                for head, size in {
                    "discard_policy": 37,
                    "riichi_action": 2,
                    "response_action": 4,
                    "chi_pattern": 3,
                    "kan_action": 35,
                    "win_decision": 2,
                    "riichi_policy": 74,
                    "response_policy": 150,
                }.items()
            }
            # Prefer declining a discretionary kan; the adapter must proceed
            # to the current legal discard instead of returning no action.
            prediction["kan_action_probabilities"] = [1.0] + [0.0] * 34
            prediction["action_values"] = [
                {
                    "action": (
                        261 + int(instance["trigger_tile_id"])
                        if head == "response_policy_logits" and action == 149
                        else int(action)
                        + {
                            "discard_policy_logits": 0,
                            "riichi_policy_logits": 37,
                            "response_policy_logits": 111,
                            "kan_action_logits": 260,
                        }[head]
                    ),
                    "expected_points": [0.0] * 4,
                    "projected_st3": 0.0,
                }
                for head, mask in decision_layers.legal_choices(
                    {
                        name: numpy.asarray(value)[None]
                        for name, value in instance.items()
                    }
                ).items()
                if head != "win_decision_logits"
                for action in numpy.flatnonzero(mask.numpy()[0])
            ]
            predictions.append(prediction)
        outputs = {
            name.replace("probabilities", "logits"): numpy.log(
                numpy.maximum([row[name] for row in predictions], 1e-30)
            )
            for name in predictions[0]
            if name.endswith("_probabilities")
        }
        return inference.Inference(predictions, 2.0, self.metadata, outputs)


def test_live_and_review_http_preserve_decisions_and_failures():
    async def run():
        predictor = LocalPredictor()
        async with aiohttp.test_utils.TestClient(
            aiohttp.test_utils.TestServer(
                service.serve.create_application(
                    predictor, ["https://review.example"]
                )
            )
        ) as client:
            response = await client.get("/health")
            assert (await response.json())["model"] == "/fixture/export"
            fixture = pathlib.Path(__file__).parent.parent / "fixtures"
            abort = json.loads(
                (fixture / "live_abortive_draw.json").read_text()
            )
            response = await client.post("/predict-live", json=abort)
            assert response.status == 200
            result = await response.json()
            assert result["decision"]["action"] == {
                "actor": abort["actor"],
                "type": "ryukyoku",
            }
            assert result["inference"] is None
            assert predictor.calls == []
            round_data = json.loads(
                (fixture / "review_reach_round.json").read_text()
            )
            response = await client.post(
                "/replay-deal-ins",
                json={"rounds": [round_data], "seat": 1},
            )
            assert response.status == 200
            danger = (await response.json())["deal_in_tiles"][0]
            assert set(danger[29]) == {"5p", "5pr"}
            assert danger[28] == []  # Before drawing, not a discard choice.
            assert danger[27] == []  # Another player's decision.
            assert predictor.calls == []
            # Passing this player's 5p after riichi makes the opponent
            # furiten; the remaining red five must then be unmarked.
            passed = json.loads(json.dumps(round_data))
            passed["events"] = passed["events"][:38]
            passed["events"][29]["tile"] = "5p"
            passed["events"][29]["tsumogiri"] = False
            response = await client.post(
                "/replay-deal-ins", json={"rounds": [passed], "seat": 1}
            )
            assert response.status == 200
            after_pass = (await response.json())["deal_in_tiles"][0]
            assert after_pass[29] == danger[29]
            assert after_pass[37] == []
            cursor = next(
                index
                for index, event in enumerate(round_data["events"])
                if event["type"] == "discard" and event["riichi"]
            )
            response = await client.post(
                "/predict-replay",
                json={"round": round_data, "event_count": cursor},
            )
            assert response.status == 200, await response.text()
            result = await response.json()
            assert result["position"]["expert_decisions"]["riichi_action"] == 1
            assert list(predictor.calls[-1]["riichi_legal_mask"]) == [1, 1]
            assert sum(result["position"]["actual_settlement"]) == 0
            bank_before_deposit = result["position"]["actual_settlement"][4]
            assert len(result["position"]["discard_options"]) > 1
            for option in result["position"]["discard_options"]:
                code = option["action"]
                assert option["tile"] == shanten.TILE_NAMES[code]
                assert option["tsumogiri"] == bool(
                    predictor.calls[-1]["current_draw_valid"]
                    and code == features.drawn_tile_code(predictor.calls[-1])
                )
                assert (
                    option["all_shanten"]
                    == int(
                        predictor.calls[-1]["discard_action_all_shanten"][code]
                    )
                    - 1
                )
                assert option["all_ukeire_count"] == int(
                    predictor.calls[-1]["discard_action_all_ukeire_count"][code]
                )
                assert option["all_upgrade_count"] == int(
                    predictor.calls[-1][
                        "discard_action_all_upgrade_tile_count"
                    ][code]
                )
            next_cursor = next(
                index
                for index in range(cursor + 1, len(round_data["events"]))
                if round_data["events"][index]["type"] == "discard"
            )
            response = await client.post(
                "/predict-replay",
                json={"round": round_data, "event_count": next_cursor},
            )
            after_deposit = (await response.json())["position"]
            assert after_deposit["actual_settlement"][4] == (
                bank_before_deposit - 1000
            )
            assert (
                after_deposit["current_riichi_sticks"]
                == round_data["riichiSticks"] + 1
            )
            response = await client.post(
                "/predict-replay", json={"round": round_data, "event_count": 0}
            )
            assert response.status == 409
            error = (await response.json())["error"]
            assert error["code"] == "no_decision"
            assert len(error["context"]["actual_shanten"]) == 4
            response = await client.post("/predict", json={"unknown": []})
            assert response.status == 400
            assert (await response.json())["error"]["code"] == "invalid_request"
            response = await client.options(
                "/predict", headers={"Origin": "https://review.example"}
            )
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == (
                "https://review.example"
            )

    asyncio.run(run())


def test_declined_kan_continues_to_a_legal_discard():
    async def run():
        predictor = LocalPredictor()
        fixture = pathlib.Path(__file__).parent.parent / "fixtures"
        async with aiohttp.test_utils.TestClient(
            aiohttp.test_utils.TestServer(
                service.serve.create_application(predictor, [])
            )
        ) as client:
            live = json.loads((fixture / "live_declined_kan.json").read_text())
            response = await client.post("/predict-live", json=live)
            assert response.status == 200, await response.text()
            result = await response.json()
            assert result["decision"]["phase"] == "DISCARD"
            assert result["decision"]["action_kind"] == "recommended"
            assert result["decision"]["action"]["type"] == "dahai"
            assert (
                result["decision"]["action"]
                in result["decision"]["possible_actions"]
            )
            assert result["inference"]["latency_ms"] == 4.0

    asyncio.run(run())


def test_request_pool_keeps_health_responsive_and_results_independent():
    class WaitingPredictor(LocalPredictor):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.started = 0
            self.four = threading.Event()
            self.release = threading.Event()

        def infer(self, instances):
            with self.lock:
                self.started += 1
                if self.started == 4:
                    self.four.set()
            assert self.release.wait(10)
            return super().infer(instances)

    async def run():
        predictor = WaitingPredictor()
        async with aiohttp.test_utils.TestClient(
            aiohttp.test_utils.TestServer(
                service.serve.create_application(predictor, [])
            )
        ) as client:
            requests = [
                asyncio.create_task(
                    client.post(
                        "/predict", json={"instances": [{"ticket": ticket}]}
                    )
                )
                for ticket in range(6)
            ]
            try:
                assert await asyncio.to_thread(predictor.four.wait, 3)
                response = await asyncio.wait_for(client.get("/health"), 1)
                assert (await response.json())["max_concurrent_requests"] == 4
                assert predictor.started == 4
            finally:
                predictor.release.set()
            results = await asyncio.gather(*requests)
            assert [
                (await response.json())["inference"]["predictions"]
                for response in results
            ] == [[{"ticket": ticket}] for ticket in range(6)]

    asyncio.run(run())


def test_calls_and_riichi_choose_the_branch_before_its_discard():
    fixture = pathlib.Path(__file__).parent.parent / "fixtures"
    missed = json.loads((fixture / "live_pon_response.json").read_text())
    state, _ = review.reconstruct(
        missed["round"], len(missed["round"]["events"])
    )
    phase, request = actions.policy_request(
        state, missed["actor"], state.legal_observation(missed["actor"])
    )
    assert phase == "RESPONSE"
    probabilities = numpy.zeros(150)
    probabilities[[0, 140, 120, 113]] = [0.4035, 0.3235, 0.2459, 0.0271]
    outputs = {
        "response_policy_logits": numpy.log(
            numpy.maximum(probabilities[None], 1e-30)
        )
    }
    selected, metadata = actions.select_policy_action(request, outputs, 0)
    assert selected.action_type == riichienv.ActionType.PON
    assert (
        metadata["continuation_discard"] == 28
    )  # South, after red-dragon pon.
    assert abs(metadata["probability"] - 0.3235) < 1e-6
    outputs["response_policy_logits"][0, 0] = 0  # Passing now wins in total.
    selected, _ = actions.select_policy_action(request, outputs, 0)
    assert selected.action_type == riichienv.ActionType.PASS

    # Chi mass can be split across both patterns and follow-up discards.
    chi_low = riichienv.Action(
        riichienv.ActionType.CHI, tile=8, consume_tiles=[12, 17], actor=0
    )
    chi_high = riichienv.Action(
        riichienv.ActionType.CHI, tile=8, consume_tiles=[0, 4], actor=0
    )
    request = {
        "phase": "RESPONSE",
        "legal_actions": [
            riichienv.Action(riichienv.ActionType.PASS),
            chi_low,
            chi_high,
        ],
        "features": {"chi_discard_legal": numpy.ones((3, 37), bool)},
    }
    probabilities = numpy.zeros(150)
    probabilities[[0, 1, 2, 75, 76]] = [0.4, 0.16, 0.15, 0.28, 0.01]
    selected, metadata = actions.select_policy_action(
        request,
        {
            "response_policy_logits": numpy.log(
                numpy.maximum(probabilities[None], 1e-30)
            )
        },
        0,
    )
    assert (
        selected is chi_low
    )  # .31 beats .29; neither individually beats pass.
    assert metadata["continuation_discard"] == 0

    request = {
        "phase": "RIICHI",
        "drawn_tile": None,
        "legal_actions": [
            riichienv.Action(riichienv.ActionType.DISCARD, tile=0),
            riichienv.Action(riichienv.ActionType.DISCARD, tile=4),
            riichienv.Action(riichienv.ActionType.RIICHI),
        ],
        "features": {
            "discard_action_legal": [1, 1] + [0] * 35,
            "riichi_discard_legal_mask": [1, 1] + [0] * 35,
        },
    }
    probabilities = numpy.zeros(74)
    probabilities[[0, 1, 37, 38]] = [0.4, 0, 0.31, 0.29]
    outputs = {
        "riichi_policy_logits": numpy.log(
            numpy.maximum(probabilities[None], 1e-30)
        )
    }
    selected, metadata = actions.select_policy_action(request, outputs, 0)
    assert selected.action_type == riichienv.ActionType.RIICHI
    assert metadata["discard_action"] == 0
    outputs["riichi_policy_logits"][0, 0] = 0
    selected, _ = actions.select_policy_action(request, outputs, 0)
    assert selected.action_type == riichienv.ActionType.DISCARD
