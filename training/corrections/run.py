"""Bounded rounds of paired decision correction against a fixed policy anchor."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy
import tensorflow as tf
from model import actor as actor_layers
from model import export
from training import self_play
from training.candidate import collection as candidate_collection
from training.candidate import monitor
from training.corrections import collection, matches, trainer


def evaluate(command, weights, output, round_number):
    command = command.copy()
    seed_argument = command.index("--seed") + 1
    games = int(command[command.index("--games") + 1])
    command[seed_argument] = str(
        int(command[seed_argument]) + round_number * (games // 4)
    )
    subprocess.run(
        [
            value.format(weights=str(weights), output=str(output))
            for value in command
        ],
        check=True,
    )


def match_sources(directories):
    """Completed seeded matches that the simulator can reconstruct exactly."""
    recorded = []
    for directory in directories:
        directory = pathlib.Path(directory).resolve()
        manifest = json.loads((directory / "evaluation.json").read_text())
        if manifest["format"] not in (
            "riichienv_duplicate_south",
            "riichienv_standard_south",
        ):
            raise ValueError("Unsupported match replay format")
        games = [
            json.loads(line)
            for line in (directory / "games.jsonl").read_text().splitlines()
        ]
        if len(games) != manifest["games"]:
            raise ValueError("Match manifest must describe a completed match")
        recorded.extend(
            {
                "path": str(directory / game["replay"]),
                "seed": game["seed"],
                "duplicate": manifest["format"] == "riichienv_duplicate_south",
            }
            for game in games
        )
    if len({game["path"] for game in recorded}) != len(recorded):
        raise ValueError("Match inputs contain duplicate replay paths")
    return recorded


def run(args):
    if not 1 <= args.start_round <= args.rounds:
        raise ValueError("Start round must be within the planned rounds")
    if min(args.rounds, args.games, args.batch_size) <= 0:
        raise ValueError(
            "Rounds, discovery games and batch size must be positive"
        )
    command = json.loads(pathlib.Path(args.evaluation_command).read_text())
    evaluation_games = int(command[command.index("--games") + 1])
    if evaluation_games <= 0 or evaluation_games % 4:
        raise ValueError("Evaluation requires complete four-seat seed groups")
    int(command[command.index("--seed") + 1])
    if not all(
        any(placeholder in value for value in command)
        for placeholder in ("{weights}", "{output}")
    ):
        raise ValueError("Evaluation command needs {weights} and {output}")
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) > 1:
        raise ValueError("Select one training GPU by UUID for this experiment")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(args.seed)
    recorded = (
        matches.expert_sources(args.expert_corpus)
        if args.expert_corpus
        else match_sources(args.matches)
    )
    if args.rounds * (args.games // 2) > len(recorded):
        raise ValueError("Not enough logged games for fresh roots each round")
    root = pathlib.Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    network = export.load_network(
        args.initial, self_play.initial_features(args.seed)
    )
    magnet = tf.saved_model.load(args.reference)
    learner = trainer.CorrectionTrainer(
        network=network,
        magnet=magnet.signatures["policy"],
        learning_rate=5e-6,
    )
    network.save_weights(root / "initial.weights.h5")
    monitor.write_json(
        root / "plan.json",
        {
            **vars(args),
            "method": "paired-current-hand-corrections",
            "evaluation_argv": command,
            "logged_discovery_fraction": 0.5,
            "learning_rate": 5e-6,
            "magnet_coefficient": 0.3,
            "imitation": False,
            "ordinary_position_training": False,
            "continuations_per_action": 1,
            "passes_per_pair": 1,
            "initial_sha256": hashlib.sha256(
                pathlib.Path(args.initial).read_bytes()
            ).hexdigest(),
            "reference_files_sha256": {
                str(path.relative_to(args.reference)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(pathlib.Path(args.reference).rglob("*"))
                if path.is_file()
            },
        },
    )
    packed_predict = self_play.predictor(network.actor)

    def predict(rows):
        return packed_predict({name: rows[name] for name in rows.dtype.names})

    root_evaluate = candidate_collection.root_scorer(network)
    writer = tf.summary.create_file_writer(str(root / "tensorboard"))
    generator = numpy.random.default_rng(args.seed)
    recorded_order = generator.permutation(len(recorded))
    monitor.write_json(root / "match-sources.json", recorded)
    started = time.monotonic()
    exposures = 0

    def report(stage, round_number, metrics):
        status = {
            "stage": stage,
            "round": round_number,
            "phase_name": f"round-{round_number}",
            "rounds": args.rounds,
            "exposures": exposures,
            "elapsed_seconds": time.monotonic() - started,
            "updated_at": time.time(),
            **metrics,
        }
        monitor.write_json(root / "status.json", status)
        with (root / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(status) + "\n")
        with writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=exposures)
        writer.flush()

    report("initialized", 0, {})
    for round_number in range(args.start_round, args.rounds + 1):
        if (root / "STOP").exists():
            break
        report("collecting-pairs", round_number, {})
        # Weights are unchanged throughout discovery and both continuations.
        batches = []
        counts = {}
        for source, source_games in enumerate(
            (args.games // 2, args.games - args.games // 2)
        ):
            for begin in range(0, source_games, 32):
                count = min(32, source_games - begin)
                recorded_begin = (round_number - 1) * (args.games // 2) + begin
                selected_games = recorded_order[
                    recorded_begin : recorded_begin + count
                ]
                batch, collected = collection.collect(
                    predict=predict,
                    root_evaluate=root_evaluate,
                    games=count,
                    seed=(
                        args.seed
                        + (round_number - 1) * args.games
                        + begin
                        + source * (args.games // 2)
                    ),
                    discover=(
                        (
                            lambda capture: matches.discover(
                                predict=predict,
                                games=[
                                    recorded[index] for index in selected_games
                                ],
                                capture=capture,
                            )
                        )
                        if source == 0
                        else None
                    ),
                )
                batch["match_index"] = (
                    selected_games[batch["source"][:, 0]]
                    if source == 0
                    else numpy.full(len(batch["head"]), -1, numpy.int64)
                )
                path = root / (
                    f"round-{round_number}-source-{source}-{begin:04d}-pairs.npz"
                )
                pending = path.with_suffix(".partial.npz")
                numpy.savez_compressed(
                    pending,
                    **{
                        "features/" + name: value
                        for name, value in batch["features"].items()
                    },
                    **{
                        name: value
                        for name, value in batch.items()
                        if name != "features"
                    },
                )
                pending.replace(path)
                batches.append(batch)
                for name, value in collected.items():
                    counts[name] = counts.get(name, 0) + value
                report("collecting-pairs", round_number, counts)
                if (root / "STOP").exists():
                    break
            if (root / "STOP").exists():
                break
        batch = tf.nest.map_structure(
            lambda *parts: numpy.concatenate(parts), *batches
        )
        if not len(batch["head"]):
            if (root / "STOP").exists():
                break
            raise ValueError("Discovery produced no eligible action pairs")
        report("pairs-collected", round_number, counts)
        indices = generator.permutation(len(batch["head"]))
        for begin in range(0, len(indices), args.batch_size):
            if (root / "STOP").exists():
                break
            selected = indices[begin : begin + args.batch_size]
            tensors = tf.nest.map_structure(
                lambda value: tf.convert_to_tensor(value[selected]), batch
            )
            metrics = learner.update(
                tensors, tf.constant(len(selected), tf.int32)
            )
            exposures += len(selected)
            report(
                "correcting",
                round_number,
                {name: float(value) for name, value in metrics.items()},
            )
        weights = root / f"round-{round_number}.weights.h5"
        network.save_weights(weights)
        pending = root / "latest.weights.h5.pending"
        pending.symlink_to(weights.name)
        pending.replace(root / "latest.weights.h5")
        if (root / "STOP").exists():
            break
        choices = []
        for begin in range(0, len(batch["head"]), args.batch_size):
            end = begin + args.batch_size
            outputs = packed_predict(
                {
                    name: value[begin:end]
                    for name, value in batch["features"].items()
                }
            )
            with tf.device("/CPU:0"):
                choices.append(
                    actor_layers.greedy_choices(
                        actor_layers.distribution(
                            outputs,
                            batch["head"][begin:end],
                            batch["legal"][begin:end],
                        ),
                        batch["head"][begin:end],
                    ).numpy()
                )
        choices = numpy.concatenate(choices)
        report(
            "root-policy-audit",
            round_number,
            {
                "policy/greedy_changed_fraction": float(
                    (choices != batch["actions"][:, 0]).mean()
                ),
                "policy/alternative_chosen_fraction": float(
                    (choices == batch["actions"][:, 1]).mean()
                ),
                "policy/untested_chosen_fraction": float(
                    (~(choices[:, None] == batch["actions"]).any(axis=1)).mean()
                ),
            },
        )
        report("evaluation", round_number, {})
        evaluate(
            command,
            weights,
            root / f"round-{round_number}-evaluation",
            round_number - 1,
        )
        report("round-complete", round_number, {})
    report(
        "stopped" if (root / "STOP").exists() else "complete", round_number, {}
    )
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initial", "reference", "output", "evaluation-command"):
        parser.add_argument("--" + name, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--matches", nargs="+")
    sources.add_argument("--expert-corpus", nargs="+")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    existing_output = pathlib.Path(args.output).exists()
    try:
        run(args)
    finally:
        # Preserve the traceback, but do not leave a failed collector shown
        # as running. Preflight errors can occur before status is created.
        path = pathlib.Path(args.output) / "status.json"
        if (
            not existing_output
            and sys.exception() is not None
            and path.exists()
        ):
            status = json.loads(path.read_text())
            status.update(
                stage="failed",
                error=str(sys.exception()),
                updated_at=time.time(),
            )
            monitor.write_json(path, status)
