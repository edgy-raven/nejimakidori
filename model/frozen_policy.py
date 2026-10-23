"""Frozen source snapshots and isolated policy execution."""

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

WORKER = r"""
import json
import sys
from log_dataset import features
from model import actions
from log_dataset import replay
import numpy
import riichienv
import tensorflow as tf

for device in tf.config.list_physical_devices("GPU"):
    tf.config.set_logical_device_configuration(
        device,
        [tf.config.LogicalDeviceConfiguration(memory_limit=int(sys.argv[2]))],
    )
signatures = tf.saved_model.load(sys.argv[1]).signatures
predict = signatures["policy"]


def infer(values):
    output = predict(
        **{
            key: tf.convert_to_tensor(
                numpy.stack([value[key] for value in values]), dtype=spec.dtype
            )
            for key, spec in predict.structured_input_signature[1].items()
        }
    )
    return {name: value.numpy() for name, value in output.items()}


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
        )
        deferred = []
        for row, (index, context, observation, value) in enumerate(decisions):
            selected, metadata = actions.select_policy_action(
                value, output, row
            )
            if selected is None:
                value = actions.continue_request(value)
                if value["phase"] == "DISCARD":
                    selected = actions.forced_policy_action(
                        context["state"],
                        observation.player_id,
                        [
                            action for action in observation.legal_actions()
                            if action.action_type
                            == riichienv.ActionType.DISCARD
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


def source_hashes(root):
    root = pathlib.Path(root)
    return {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def freeze_source(source, target, include_native=False):
    source, target = pathlib.Path(source), pathlib.Path(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            ".pytest_cache",
            "*.pyc",
            "target",
            *(() if include_native else (".native", "*.so")),
        ),
    )
    return source_hashes(target)


def source_directory(model_path):
    """Find the enclosing release/run snapshot for a SavedModel artifact."""
    for parent in pathlib.Path(model_path).resolve().parents:
        source = parent / "source/model"
        if (source / "frozen_policy.py").is_file():
            return source
    raise FileNotFoundError(f"frozen policy source missing for {model_path}")


class FrozenPolicy:
    def __init__(self, model_path, device="-1", memory_mb=2048):
        self.model_path = pathlib.Path(model_path).resolve()
        self.source = source_directory(self.model_path)
        self.reset_pending = True
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.pop(0); "
                "from model.frozen_policy import WORKER; exec(WORKER)",
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
                    (str(self.source.parent), str(self.source.parent.parent))
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

    def select_many(self, requests):
        self.process.stdin.write(
            json.dumps(
                {
                    "reset": self.reset_pending,
                    # Saved opponent workers read this wire field.
                    "batch_size": 1,
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
