"""Warmed CPU workflow timings with an explicitly recorded regression baseline."""

import argparse
import datetime
import json
import os
import pathlib
import platform
import statistics
import time

from model import actions
from model import network as joint_network


def measure(workload):
    for _ in range(2):
        workload()
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        workload()
        samples.append(time.perf_counter() - started)
    return {"seconds": statistics.median(samples), "samples": samples}


def compare(baseline, measured):
    if baseline["environment"] != measured["environment"]:
        raise ValueError("Performance baseline environment does not match")
    if baseline["workloads"].keys() != measured["workloads"].keys():
        raise ValueError("Performance baseline workloads do not match")
    failures = []
    for name, result in measured["workloads"].items():
        reference = baseline["workloads"][name]["seconds"]
        ratio = result["seconds"] / reference
        print(
            f"{name}: {result['seconds']:.4f}s, "
            f"baseline {reference:.4f}s, {ratio:.2f}x",
            flush=True,
        )
        if ratio > 1.25:
            failures.append(name)
    if failures:
        raise RuntimeError(
            "Performance regression (>25%): " + ", ".join(failures)
        )


def workloads():
    import numpy
    import tensorflow as tf

    from log_dataset import features, replay
    from model import self_play

    tf.keras.utils.set_random_seed(941)
    events = json.loads(
        pathlib.Path(__file__)
        .with_name("fixtures")
        .joinpath("model/pass_speed_round.json")
        .read_text()
    )

    def extract():
        for _ in range(25):
            state = replay.Replay("performance", events)
            for _, actor, phase, _, _ in state.policy_decisions():
                features.pack(state, actor, phase)

    yield "recorded_hand_features_25_replays", extract

    def uniform(inputs):
        return {
            name: numpy.zeros(
                (
                    len(inputs["decision_phase"]),
                    actions.ACTOR_OUTPUTS[name][0],
                ),
                numpy.float32,
            )
            for name in actions.TRAIN_POLICY_HEADS
        }

    # Includes native simulation, feature transport and final replay assembly.
    yield "native_collection_8_games", lambda: self_play.collect(
        uniform, 8, 941
    )
    batch, _ = self_play.collect(uniform, 1, 941)
    inputs = {name: value[:32] for name, value in batch["features"].items()}
    predict = self_play.predictor(joint_network.JointPolicyPayment().actor)

    def infer():
        for _ in range(10):
            # Materialize outputs inside the timer, including asynchronous work.
            for value in predict(inputs).values():
                value.numpy()

    yield "packed_actor_10_batches_32", infer


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("performance_cpu.json"),
    )
    parser.add_argument("--record-baseline", action="store_true")
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES="-1",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="2",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    import numpy
    import tensorflow as tf

    measured = {
        "environment": {
            "cpu": next(
                line.split(":", 1)[1].strip()
                for line in pathlib.Path("/proc/cpuinfo")
                .read_text()
                .splitlines()
                if line.startswith("model name")
            ),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": numpy.__version__,
            "threads": 2,
            "device": "CPU",
        },
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "workloads": {},
    }
    baseline = (
        None if args.record_baseline else json.loads(args.baseline.read_text())
    )
    if (
        baseline is not None
        and baseline["environment"] != measured["environment"]
    ):
        raise ValueError(
            "Performance baseline environment does not match; record a "
            "baseline from a known-good checkout for this environment"
        )
    for name, workload in workloads():
        print(f"Measuring {name}", flush=True)
        measured["workloads"][name] = measure(workload)
    if args.record_baseline:
        args.baseline.write_text(json.dumps(measured, indent=2) + "\n")
        print(f"Recorded baseline: {args.baseline}", flush=True)
    else:
        compare(baseline, measured)


if __name__ == "__main__":
    run()
