"""HTTP flows with local game fixtures and a deterministic model boundary."""

import asyncio
import json
import pathlib
import threading

import aiohttp.test_utils
import numpy
import werkzeug.test
import werkzeug.wrappers

import service.dashboard
import service.serve
from log_dataset import shanten
from model import features, inference


class LocalPredictor:
    model_path = pathlib.Path("/fixture/export")
    metadata = {"model_contract": "fixture-payments"}

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
                    "discard_policy": 38,
                    "riichi_action": 2,
                    "response_action": 4,
                    "chi_pattern": 3,
                    "kan_action": 35,
                    "win_decision": 2,
                    "riichi_policy": 76,
                    "response_policy": 150,
                }.items()
            }
            # Prefer declining a discretionary kan; the adapter must proceed
            # to the current legal discard instead of returning no action.
            prediction["kan_action_probabilities"] = [1.0] + [0.0] * 34
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
            fixture = pathlib.Path(__file__).parent / "fixtures"
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
                code = features.discard_tile_code(
                    option["action"], predictor.calls[-1]
                )
                assert option["tile"] == shanten.TILE_NAMES[code]
                assert option["tsumogiri"] == (option["action"] == 37)
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
        fixture = pathlib.Path(__file__).parent / "fixtures"
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


def test_dashboard_reads_current_status_and_preserves_chart_navigation(
    tmp_path,
):
    path = tmp_path / "monitor_status.json"
    path.write_text(json.dumps({"training": {"step": 23}}))
    charts = werkzeug.wrappers.Response(
        "<body>local chart application</body>", content_type="text/html"
    )
    client = werkzeug.test.Client(service.dashboard.Dashboard(tmp_path, charts))
    page = client.get("/tensorboard/")
    assert page.status_code == 200
    assert "#custom_scalars&amp;_smoothingWeight=0" in page.text
    assert page.headers["Cache-Control"] == "no-store"
    assert client.get("/tensorboard/status.json").json["training"]["step"] == 23
    path.write_text(json.dumps({"training": {"step": 47}}))
    assert client.get("/tensorboard/status.json").json["training"]["step"] == 47
    chart_page = client.get("/tensorboard/charts/").text
    assert "local chart application" in chart_page
    assert '<script src="/tensorboard/chart-colors.js"></script>' in chart_page
    colors = client.get("/tensorboard/chart-colors.js")
    assert colors.mimetype == "text/javascript"
    assert "metricColors" in colors.text
