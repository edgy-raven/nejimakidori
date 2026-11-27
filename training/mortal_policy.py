"""Frozen Mortal opponent using RiichiEnv's private-seat MJAI stream."""

import concurrent.futures
import importlib.util
import io
import json
import os
import pathlib
import queue
import subprocess
import sys
import zipfile

from training import frozen_policy


class MortalPolicy(frozen_policy.FrozenPolicy):
    def __init__(self, root, capacity, device="-1"):
        root = pathlib.Path(root).resolve()
        self.model_path = root / "checkpoint"
        self.reset_pending = True
        self.process = subprocess.Popen(
            args=[
                sys.executable,
                "-m",
                "training.mortal_policy",
                str(root),
                str(capacity),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=root,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": device,
                "PYTHONPATH": str(pathlib.Path(root) / "adapter"),
            },
        )

    def send_requests(self, requests):
        self.process.stdin.write(
            json.dumps(
                {
                    "reset": self.reset_pending,
                    "items": [
                        {
                            "stream": stream,
                            "seat": observation.player_id,
                            "events": observation.new_events(),
                        }
                        for stream, _, observation in requests
                    ],
                }
            )
            + "\n"
        )
        self.process.stdin.flush()
        self.reset_pending = False


class BatchedReactions:
    """Batch neural requests while each upstream Bot keeps its event loop."""

    def __init__(self, opponent, capacity):
        self.infer = opponent.react_batch
        opponent.react_batch = self.inference_request
        self.queue = queue.SimpleQueue()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=capacity)

    def inference_request(self, obs, masks, invisible):
        future = concurrent.futures.Future()
        self.queue.put((obs, masks, future))
        return future.result()

    def react(self, bot, events):
        try:
            reaction = None
            # https://riichi.dev/docs/mortal: every new event, final response.
            for event in events:
                response = bot.react(event)
                # A delayed dora announcement can follow a kakan in the
                # same observation. It updates state, not the response.
                if json.loads(event)["type"] != "dora":
                    reaction = response
            return None if reaction is None else json.loads(reaction)
        finally:
            self.queue.put(None)

    def select_many(self, bots, items):
        futures = [
            self.pool.submit(self.react, bots[item["stream"]], item["events"])
            for item in items
        ]
        return self.collect(futures)

    def collect(self, futures):
        """Resolve inference from tasks that signal completion on the queue."""
        active = len(futures)
        while active:
            pending = []
            # Each active Bot either needs inference or has finished.
            while len(pending) < active:
                request = self.queue.get()
                if request is None:
                    active -= 1
                else:
                    pending.append(request)
            if pending:
                try:
                    output = self.infer(
                        [obs for request in pending for obs in request[0]],
                        [mask for request in pending for mask in request[1]],
                        None,
                    )
                    offset = 0
                    for obs, _, future in pending:
                        future.set_result(
                            tuple(
                                values[offset : offset + len(obs)]
                                for values in output
                            )
                        )
                        offset += len(obs)
                finally:
                    # Release waiting callers if inference fails.
                    for _, _, future in pending:
                        future.cancel()
        return [future.result() for future in futures]


def load_engine(root, device):
    import torch

    sys.path.insert(0, str(root / "source"))
    modules = {}
    for name in ("engine", "model"):
        spec = importlib.util.spec_from_file_location(
            "mortal_" + name, root / "source" / (name + ".py")
        )
        modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modules[name])

    torch.set_num_threads(2)
    checkpoint = root / "checkpoint"
    if zipfile.is_zipfile(checkpoint):
        with zipfile.ZipFile(checkpoint) as archive:
            # A torch checkpoint is itself a ZIP; only unwrap the model bundle.
            if len(archive.namelist()) == 1:
                checkpoint = io.BytesIO(archive.read(archive.namelist()[0]))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    brain = modules["model"].Brain(
        version=state["config"]["control"]["version"],
        **state["config"]["resnet"],
    )
    dqn = modules["model"].DQN(version=state["config"]["control"]["version"])
    # Preserve the upstream checkpoint key and engine API.
    brain.load_state_dict(state["mortal"], strict=True)
    dqn.load_state_dict(state["current_dqn"], strict=True)
    return modules["engine"].MortalEngine(
        brain=brain,
        dqn=dqn,
        is_oracle=False,
        version=state["config"]["control"]["version"],
        device=torch.device(device),
        enable_amp=False,
        enable_rule_based_agari_guard=True,
        name="mortal",
    )


def run(root, capacity):
    device = "cpu" if os.environ["CUDA_VISIBLE_DEVICES"] == "-1" else "cuda:0"
    opponent = load_engine(root, device)
    import libriichi.mjai

    bots = {}
    batcher = BatchedReactions(opponent, capacity)
    for line in sys.stdin:
        request = json.loads(line)
        if request["reset"]:
            bots.clear()
        for item in request["items"]:
            if item["stream"] not in bots:
                bots[item["stream"]] = libriichi.mjai.Bot(
                    opponent, item["seat"]
                )
        print(
            json.dumps(batcher.select_many(bots, request["items"])), flush=True
        )
    batcher.pool.shutdown()


if __name__ == "__main__":
    run(pathlib.Path(sys.argv[1]), int(sys.argv[2]))
