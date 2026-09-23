"""Bounded HTTP inference over stateless game requests."""

import asyncio
import concurrent.futures
import functools
import json
import os
import pathlib

import aiohttp.web

import service.web_adapter


class RequestError(ValueError):
    def __init__(self, status, code, message, context=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.context = context


def compute(predictor, path, body):
    if path == "/predict":
        try:
            return {"inference": predictor.infer(body["instances"]).as_dict()}
        except ValueError as error:
            raise RequestError(
                400, "invalid_feature_value", str(error)
            ) from error
    try:
        if path == "/predict-replay":
            return service.web_adapter.predict_replay(
                predictor, body["round"], body["event_count"]
            )
        return service.web_adapter.predict_live(
            predictor, body["round"], body["actor"]
        )
    except service.web_adapter.NoDecision as error:
        raise RequestError(
            409, "no_decision", str(error), error.context
        ) from error
    except ValueError as error:
        code = (
            "invalid_replay_position"
            if path == "/predict-replay"
            else "invalid_live_position"
        )
        raise RequestError(400, code, str(error)) from error


def create_application(predictor, allowed_origins):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    @aiohttp.web.middleware
    async def responses(request, handler):
        try:
            response = await handler(request)
        except RequestError as error:
            body = {"code": error.code, "message": str(error)}
            if error.context is not None:
                body["context"] = error.context
            response = aiohttp.web.json_response(
                {"error": body}, status=error.status
            )
        origin = request.headers.get("Origin")
        if origin and (origin in allowed_origins or "*" in allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = (
                "*" if "*" in allowed_origins else origin
            )
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Vary"] = "Origin"
        return response

    app = aiohttp.web.Application(
        middlewares=[responses], client_max_size=64 * 1024 * 1024
    )

    async def stop(_app):
        await asyncio.to_thread(executor.shutdown, wait=True)

    app.on_cleanup.append(stop)

    async def health(_request):
        return aiohttp.web.json_response(
            {
                "ok": True,
                "service": "nejimakidori-model",
                "model": str(predictor.model_path),
                "model_contract": predictor.metadata["model_contract"],
                **(
                    {"revision": predictor.metadata["revision"]}
                    if "revision" in predictor.metadata
                    else {}
                ),
                "max_concurrent_requests": 4,
            }
        )

    async def describe(_request):
        return aiohttp.web.json_response(
            {
                "service": "nejimakidori-model",
                "metadata": predictor.metadata,
                "max_batch_size": predictor.max_batch_size,
                "max_concurrent_requests": 4,
                "inputs": {
                    name: {"shape": list(shape), "dtype": str(dtype)}
                    for name, (shape, dtype) in predictor.input_specs.items()
                },
                "outputs": {
                    name: {
                        "shape": spec.shape.as_list(),
                        "dtype": spec.dtype.name,
                    }
                    for name, spec in predictor.output_specs.items()
                },
                "requests": {
                    "/predict": ["instances"],
                    "/predict-replay": ["round", "event_count"],
                    "/predict-live": ["round", "actor"],
                },
                "responses": {
                    "/predict": ["inference"],
                    "/predict-replay": ["position", "inference"],
                    "/predict-live": ["decision", "inference"],
                },
            }
        )

    async def predict(request):
        if request.content_type != "application/json":
            raise RequestError(415, "unsupported_media_type", "Use JSON.")
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RequestError(
                400, "invalid_request", "Invalid JSON."
            ) from error
        fields = {
            "/predict": {"instances"},
            "/predict-replay": {"round", "event_count"},
            "/predict-live": {"round", "actor"},
        }[request.path]
        if not isinstance(body, dict) or set(body) != fields:
            raise RequestError(
                400, "invalid_request", "Expected fields: " + ", ".join(fields)
            )
        result = await asyncio.get_running_loop().run_in_executor(
            executor,
            functools.partial(
                compute,
                predictor=predictor,
                path=request.path,
                body=body,
            ),
        )
        return aiohttp.web.json_response(result)

    async def options(_request):
        return aiohttp.web.Response(status=204)

    app.router.add_get("/", describe)
    app.router.add_get("/health", health)
    for path in (
        "/predict",
        "/predict-replay",
        "/predict-live",
    ):
        app.router.add_post(path, predict)
    app.router.add_route("OPTIONS", "/{path:.*}", options)
    return app


def run():
    import tensorflow

    if "GPU_MEMORY_LIMIT_MB" in os.environ:
        for device in tensorflow.config.list_physical_devices("GPU"):
            tensorflow.config.set_logical_device_configuration(
                device,
                [
                    tensorflow.config.LogicalDeviceConfiguration(
                        memory_limit=int(os.environ["GPU_MEMORY_LIMIT_MB"])
                    )
                ],
            )
    from model import inference

    predictor = inference.Predictor(pathlib.Path(os.environ["MODEL_PATH"]))
    aiohttp.web.run_app(
        create_application(
            predictor, os.environ.get("ALLOWED_ORIGINS", "").split(",")
        ),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8792")),
    )


if __name__ == "__main__":
    run()
