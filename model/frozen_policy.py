"""Run each frozen feature producer and actor in its own Python process."""

import json
import os
import pathlib
import subprocess
import sys

WORKER = r"""
import json
import sys
import types
import pathlib

package = types.ModuleType("frozen_model")
package.__path__ = [str(pathlib.Path.cwd())]
sys.modules[package.__name__] = package
from frozen_model import actions, features, replay
import numpy
import riichienv
import tensorflow as tf

for device in tf.config.list_physical_devices("GPU"):
    tf.config.set_logical_device_configuration(
        device,
        [tf.config.LogicalDeviceConfiguration(memory_limit=int(sys.argv[2]))],
    )
predict = tf.saved_model.load(sys.argv[1]).signatures["serving_default"]


def infer(values, batch_size):
    values = values + [values[-1]] * (batch_size - len(values))
    output = predict(
        **{
            key: tf.convert_to_tensor(
                numpy.stack([value[key] for value in values]), dtype=spec.dtype
            )
            for key, spec in predict.structured_input_signature[1].items()
        }
    )
    return {
        name.replace("probabilities", "logits"): (
            numpy.log(numpy.maximum(value.numpy(), 1e-30))
            if name.endswith("probabilities")
            else value.numpy()
        )
        for name, value in output.items()
    }


contexts = {}
for line in sys.stdin:
    request = json.loads(line)
    if request["reset"]:
        contexts.clear()
    results = [None] * len(request["items"])
    decisions = []
    for index, item in enumerate(request["items"]):
        stream = item["stream"]
        if stream not in contexts or (
            item["events"][: contexts[stream]["cursor"]]
            != contexts[stream]["previous"]
        ):
            contexts[stream] = {
                "state": None,
                "cursor": 0,
                "pending": {},
                "previous": [],
            }
        context = contexts[stream]
        for event in item["events"][context["cursor"] :]:
            if event["type"] == "start_kyoku":
                context["state"] = replay.Replay.from_start("frozen", event)
                context["pending"].clear()
            elif context["state"] is not None:
                if event["type"] in ("chi", "pon", "daiminkan"):
                    context["pending"] = {
                        seat: plan
                        for seat, plan in context["pending"].items()
                        if seat == event["actor"] and plan[1] == event["type"]
                    }
                elif event["type"] in ("tsumo", "dahai"):
                    context["pending"].pop(event["actor"], None)
                elif event["type"] in ("hora", "ryukyoku", "end_kyoku"):
                    context["pending"].clear()
                context["state"].apply(event)
                if event["type"] == "tsumo" and "tiles_left" in event:
                    context["state"].live_wall = event["tiles_left"]
        context["cursor"] = len(item["events"])
        context["previous"] = item["events"]
        observation = riichienv.Observation.deserialize_from_base64(
            item["observation"]
        )
        actor = observation.player_id
        legal = observation.legal_actions()
        selected = actions.forced_policy_action(context["state"], actor, legal)
        if actor in context["pending"] and any(
            action.action_type == riichienv.ActionType.DISCARD
            for action in legal
        ):
            cut, kind = context["pending"].pop(actor)
            if selected is None:
                selected = actions.select_discard(
                    legal, cut, observation.drawn_tile
                )
        if selected is not None:
            results[index] = features.action_event(
                selected, observation.drawn_tile
            )
        else:
            phase, value = actions.policy_request(
                context["state"], actor, observation
            )
            decisions.append((index, context, observation, value))
    while decisions:
        output = infer(
            [value["features"] for _, _, _, value in decisions],
            request["batch_size"],
        )
        deferred = []
        for row, (index, context, observation, value) in enumerate(decisions):
            selected, metadata = actions.select_policy_action(
                value, output, row
            )
            if selected is None:
                if hasattr(actions, "continue_request"):
                    value = actions.continue_request(value)
                elif "after_kan" in value:
                    value = value["after_kan"]
                    value["phase"] = actions.phase_for_actions(value["actions"])
                else:
                    value["actions"] = [
                        action
                        for action in observation.legal_actions()
                        if action.action_type not in actions.KAN_ACTIONS
                    ]
                    value["phase"] = actions.phase_for_actions(value["actions"])
                    value["features"] = features.pack(
                        context["state"],
                        observation.player_id,
                        value["phase"],
                        observation,
                    )
                if value["phase"] == "DISCARD":
                    selected = actions.forced_policy_action(
                        context["state"],
                        observation.player_id,
                        [
                            action for action in observation.legal_actions()
                            if action.action_type == riichienv.ActionType.DISCARD
                        ],
                    )
                if selected is None:
                    deferred.append((index, context, observation, value))
                    continue
            results[index] = features.action_event(
                selected, observation.drawn_tile
            )
            if "discard_action" in metadata and selected.action_type in (
                riichienv.ActionType.RIICHI,
                riichienv.ActionType.CHI,
                riichienv.ActionType.PON,
            ):
                context["pending"][observation.player_id] = (
                    metadata["discard_action"],
                    results[index]["type"],
                )
        decisions = deferred
    print(json.dumps(results), flush=True)
"""


def source_directory(model_path):
    """Find the enclosing release/run snapshot for a SavedModel artifact."""
    for parent in pathlib.Path(model_path).resolve().parents:
        for source in (parent / "source/model", parent / "source"):
            if (source / "features.py").is_file():
                return source
    raise FileNotFoundError(f"frozen feature source missing for {model_path}")


class FrozenPolicy:
    def __init__(
        self, model_path, source_path=None, device="-1", memory_mb=2048
    ):
        self.model_path = pathlib.Path(model_path).resolve()
        self.source = (
            pathlib.Path(source_path).resolve()
            if source_path
            else source_directory(self.model_path)
        )
        if not (self.source / "features.py").is_file():
            raise FileNotFoundError(
                "frozen feature source missing: " + str(self.source)
            )
        self.reset_pending = True
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                WORKER,
                str(self.model_path),
                str(memory_mb),
            ],
            cwd=self.source,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    str(path)
                    for path in (
                        self.source,
                        self.source.parent,
                        self.source.parent.parent,
                    )
                ),
                "CUDA_VISIBLE_DEVICES": device,
                "TF_NUM_INTRAOP_THREADS": "2",
                "TF_NUM_INTEROP_THREADS": "2",
                "TF_DETERMINISTIC_OPS": "1",
                "TF_CUDNN_USE_AUTOTUNE": "0",
            },
        )

    def reset(self):
        self.reset_pending = True

    def select(self, env, observation):
        return self.select_many([(0, env, observation)])[0]

    def select_many(self, requests, batch_size=None):
        self.process.stdin.write(
            json.dumps(
                {
                    "reset": self.reset_pending,
                    "batch_size": (
                        len(requests) if batch_size is None else batch_size
                    ),
                    "items": [
                        {
                            "stream": stream,
                            "events": env.mjai_log,
                            "observation": observation.serialize_to_base64(),
                        }
                        for stream, env, observation in requests
                    ],
                }
            )
            + "\n"
        )
        self.process.stdin.flush()
        self.reset_pending = False
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError(f"frozen policy exited: {self.process.poll()}")
        return [
            observation.select_action_from_mjai(json.dumps(action))
            for (_, _, observation), action in zip(
                requests, json.loads(response), strict=True
            )
        ]

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)
