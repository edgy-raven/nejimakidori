"""Mortal matches, match MMD, then MMD on balanced targeted-root reruns."""

import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy
import tensorflow as tf

from log_dataset import sakigiri_regret
from model import export
from training import (
    auxiliary,
    fixed_reference,
    match_records,
    monitor,
    self_play,
)
from training import trainer as base_trainer
from training.corrections import (
    collection,
    distributed,
    matches,
    rollouts,
    trainer,
)


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


def validate_launch(args):
    if (
        min(
            args.rounds,
            args.root_games,
            args.critic_batch_size,
            args.actor_batch_size,
            args.root_batch_size,
            args.workers,
        )
        <= 0
    ):
        raise ValueError(
            "Round, root-game, batch and worker budgets must be positive"
        )
    if args.root_games > 10000:
        raise ValueError("Root games cannot exceed the 10,000-game match")
    command = json.loads(pathlib.Path(args.evaluation_command).read_text())
    if int(command[command.index("--games") + 1]) != 10000:
        raise ValueError("Each Mortal match must contain 10,000 games")
    if "--mortal-source" not in command:
        raise ValueError("The match command must select a Mortal source")
    int(command[command.index("--seed") + 1])
    if not all(
        any(placeholder in value for value in command)
        for placeholder in ("{weights}", "{output}")
    ):
        raise ValueError("Evaluation command needs {weights} and {output}")
    if args.sync_command:
        sync = json.loads(pathlib.Path(args.sync_command).read_text())
        if not all(
            any(placeholder in value for value in sync)
            for placeholder in ("{root}", "{stage}")
        ):
            raise ValueError("Sync command needs {root} and {stage}")
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) > 1:
        raise ValueError("Select one learner GPU by UUID")
    for device in devices:
        if args.gpu_memory_mb is None:
            tf.config.experimental.set_memory_growth(device, True)
        else:
            tf.config.set_logical_device_configuration(
                device,
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=args.gpu_memory_mb
                    )
                ],
            )
    tf.keras.utils.set_random_seed(args.seed)
    return command


def save_checkpoint(network, root, name):
    weights = root / f"{name}.weights.h5"
    network.save_weights(weights)
    pending = root / "latest.weights.h5.pending"
    pending.symlink_to(weights.name)
    pending.replace(root / "latest.weights.h5")
    return weights


def train_phase(
    *, phase, batches, batch_sizes, learner, root, progress, report, group
):
    """Fit outcomes, then actor rows, with persistent optimizers."""
    learner.mirror.set_weights(learner.network.actor.get_weights())
    for update in ("critic", "actor"):
        group.set_batch_size(batch_sizes[update])
        for batch in batches(batch_sizes[update]):
            metrics = group.update(update, batch, root)
            if metrics is None:
                return
            progress[update + "_exposures"] += metrics["trained_positions"]
            if group.chief:
                report(
                    phase + "-" + update,
                    {name: float(value) for name, value in metrics.items()},
                )


def run(args):
    command = validate_launch(args)
    group = distributed.LearnerGroup()
    root = pathlib.Path(args.output).resolve()
    if group.chief:
        root.mkdir(parents=True, exist_ok=False)
    with group.strategy.scope():
        features = self_play.initial_features(args.seed)
        network = export.load_network(args.initial, features)
        previous = export.load_network(args.initial, features).actor
        previous.trainable = False
        learner = trainer.MMDTrainer(
            network=network,
            mirror=previous,
            learning_rate=1e-5,
        )
        critic_learner = base_trainer.CriticTrainer(network, features, 1e-5)
    group.bind(learner, critic_learner, args.critic_batch_size)
    reference_policy = (
        fixed_reference.Policy(args.reference) if group.chief else None
    )
    if group.chief:
        weights = save_checkpoint(network, root, "initial")
        monitor.write_json(
            root / "plan.json",
            {
                **vars(args),
                "method": "match-then-root-magnetic-mirror-descent",
                "replicas": group.replicas,
                "evaluation_argv": command,
                "learning_rate": 1e-5,
                "mirror_coefficient": 0.3,
                "magnet_coefficient": 0.3,
                "objective_scale": 1.0,
                "match_training_seats": [0, 1, 2, 3],
                "root_selection_seats": [0, 1, 2, 3],
                "targeted_root_fraction": 0.5,
                "categories": list(collection.CATEGORIES),
                "sakigiri": sakigiri_regret.DEFINITION,
                "continuations_per_root": 2,
                "continuation_policy": "sampled-frozen-self-play",
                "continuation_horizon": "current-hand",
                "critic_passes_per_phase": 1,
                "actor_passes_per_phase": 1,
                "supervised_losses_per_update": dict.fromkeys(
                    auxiliary.LABELS, 1.0
                ),
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
                "reference_policy": reference_policy.provenance,
            },
        )
    predict = self_play.predictor(network.actor)

    root_evaluate = collection.root_scorer(network)
    writer = (
        tf.summary.create_file_writer(str(root / "tensorboard"))
        if group.chief
        else None
    )
    generator = numpy.random.default_rng(args.seed)
    progress = {"actor_exposures": 0, "critic_exposures": 0}
    started = time.monotonic()
    round_number = 0

    def report(stage, metrics):
        status = {
            "stage": stage,
            "round": round_number,
            "phase_name": f"round-{round_number}",
            "rounds": args.rounds,
            "exposures": progress["actor_exposures"],
            **progress,
            "elapsed_seconds": time.monotonic() - started,
            "updated_at": time.time(),
            **metrics,
        }
        monitor.write_json(root / "status.json", status)
        with (root / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(status) + "\n")
        with writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(
                    stage + "/" + name,
                    value,
                    step=sum(progress.values()),
                )
        writer.flush()

    if group.chief:
        report("evaluation", {})
        evaluate(command, weights, root / "initial-evaluation", 0)
    match_directory = root / "initial-evaluation" / "matches"
    for round_number in range(1, args.rounds + 1):
        data = root / f"round-{round_number}-match-data"
        if group.chief and not (root / "STOP").exists():
            recorded = matches.sources(match_directory)
            report("match-records", {})
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.match_records",
                    "--matches",
                    str(match_directory),
                    "--output",
                    str(data),
                    "--reference",
                    args.reference,
                    "--workers",
                    str(args.workers),
                ],
                check=True,
            )
        if distributed.publish_stage(
            args, group, root, f"round-{round_number}-match"
        ):
            break
        train_phase(
            phase="match",
            learner=learner,
            root=root,
            progress=progress,
            report=report,
            group=group,
            batch_sizes={
                "critic": args.critic_batch_size,
                "actor": args.actor_batch_size,
            },
            batches=lambda batch_size: match_records.batches(
                data,
                batch_size,
                args.seed + round_number,
                decode=group.decode_records,
            ),
        )
        if group.chief:
            weights = save_checkpoint(
                network, root, f"round-{round_number}-match"
            )
        if group.chief and not (root / "STOP").exists():
            rollouts.collect(
                root=root,
                round_number=round_number,
                match_directory=match_directory,
                recorded=recorded,
                order=generator.permutation(len(recorded))[: args.root_games],
                seed=args.seed + round_number * 100000,
                predict=predict,
                root_evaluate=root_evaluate,
                packing=critic_learner.feature_packing,
                report=report,
                reference_policy=reference_policy,
            )
        if distributed.publish_stage(
            args, group, root, f"round-{round_number}-roots"
        ):
            break
        manifest = json.loads(
            (root / f"round-{round_number}-rollouts.json").read_text()
        )
        rollout_files = [root / name for name in manifest["files"]]
        if not rollout_files:
            raise ValueError("Match pool yielded no eligible targeted roots")
        train_phase(
            phase="roots",
            learner=learner,
            root=root,
            progress=progress,
            report=report,
            group=group,
            batch_sizes={
                "critic": args.root_batch_size,
                "actor": args.root_batch_size,
            },
            batches=lambda batch_size: rollouts.batches(
                paths=rollout_files,
                packing=critic_learner.feature_packing,
                batch_size=batch_size,
                seed=args.seed + round_number,
                localize=group.localize,
            ),
        )
        if group.chief:
            weights = save_checkpoint(network, root, f"round-{round_number}")
            if not (root / "STOP").exists():
                report("evaluation", {})
                evaluate(
                    command,
                    weights,
                    root / f"round-{round_number}-evaluation",
                    round_number,
                )
                report("round-complete", {})
        match_directory = root / f"round-{round_number}-evaluation" / "matches"
    if group.chief:
        report("stopped" if (root / "STOP").exists() else "complete", {})
        writer.close()
    distributed.publish_stage(args, group, root, "complete")
