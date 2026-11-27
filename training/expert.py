"""Three current-Q expert passes fit policy, critic and beliefs."""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import tensorflow as tf

from log_dataset import (
    _native_collection,
    records,
    root_states,
    sakigiri_regret,
    scoring,
    training_data,
)
from model import actions, critic, export
from training import (
    auxiliary,
    fixed_reference,
    imitation,
    imitation_rollouts,
    monitor,
    restoration,
    self_play,
)
from training.corrections import collection, distributed
from training.corrections import trainer as mmd_trainer


def decode_records(raw):
    parsed = tf.io.parse_example(
        raw,
        {
            name: tf.io.FixedLenFeature([], tf.string)
            for name in (*records.FIELDS, "expert/mortal_action")
        },
    )
    features, labels = records.decode_batch(parsed)
    head, expert, legal = restoration.policy_choices(features, labels)
    mortal = tf.cast(
        tf.reshape(
            tf.io.decode_raw(parsed["expert/mortal_action"], tf.int16), [-1]
        ),
        tf.int32,
    )
    return {
        "features": features,
        "head": head,
        "expert": expert,
        "legal": legal,
        "chosen": critic.chosen_actions(features, labels),
        "payment_points": tf.gather(
            tf.reshape(labels["terminal_payment"], [-1, 64]),
            [
                kind * 16 + payer * 4 + winner
                for kind, payer, winner in scoring.CELLS
            ],
            axis=1,
        ),
        "reference_action": critic.chosen_actions(
            features,
            {
                name: tf.where(labels[name] >= 0, mortal, -1)
                for name in actions.TRAIN_POLICY_HEADS
            },
        ),
        "labels": {name: labels[name] for name in auxiliary.LABELS},
    }


def input_batches(*, data, group, batch_size, seed):
    """Equal natural and focused exposures on disjoint learner shards."""
    if batch_size < group.replicas:
        raise ValueError("Expert batch size must cover every replica")
    summary = json.loads(
        (pathlib.Path(data) / "dataset_summary.json").read_text()
    )
    assert "meta/focused_priority" in summary["sampling_fields"]
    counts = [
        sum(
            sum(row["counts"][:7])
            for row in summary["shard_manifest"][rank :: group.replicas]
        )
        for rank in range(group.replicas)
    ]
    if min(counts) == 0:
        raise ValueError("Each replica needs at least one training record")
    for rank in range(group.replicas):
        if not sum(
            sum(partition[0] for partition in row["sampling_counts"][:7])
            for row in summary["shard_manifest"][rank :: group.replicas]
        ):
            raise ValueError("Each replica needs focused records")
    sizes = [
        batch_size // group.replicas + (rank < batch_size % group.replicas)
        for rank in range(group.replicas)
    ]
    steps = max(
        (count + size - 1) // size for count, size in zip(counts, sizes)
    )
    with tf.device("/CPU:0"):
        streams = []
        for category in range(2):
            stream = training_data.dataset(
                root=data,
                partition="train",
                batch_size=1024,
                rank=group.rank,
                replicas=group.replicas,
                seed=seed + category,
            )
            if category:

                def emphasize(raw):
                    metadata = tf.io.parse_example(
                        raw,
                        {
                            "meta/focused_priority": tf.io.FixedLenFeature(
                                [], tf.int64
                            )
                        },
                    )
                    return tf.repeat(raw, metadata["meta/focused_priority"])

                stream = stream.map(emphasize, num_parallel_calls=2)
            streams.append(
                stream.unbatch().shuffle(8192, seed=seed + category).repeat()
            )
        stream = tf.data.Dataset.choose_from_datasets(
            streams,
            tf.data.Dataset.from_tensor_slices(tf.constant([0, 1], tf.int64))
            .repeat()
            .take(counts[group.rank]),
            stop_on_empty_dataset=True,
        )
        stream = stream.apply(
            tf.data.experimental.assert_cardinality(counts[group.rank])
        ).batch(sizes[group.rank])
        # Empty tail replicas repeat a real example with zero valid mass.
        # Rank-local lengths are known from the verified shard manifest.
        stream = stream.concatenate(stream.take(1).repeat()).take(steps)

        def decode(step, raw):
            valid_counts = tf.clip_by_value(
                tf.constant(counts, tf.int64)
                - step * tf.constant(sizes, tf.int64),
                0,
                tf.constant(sizes, tf.int64),
            )
            raw = tf.gather(
                raw, tf.minimum(tf.range(max(sizes)), tf.size(raw) - 1)
            )
            metadata = tf.io.parse_example(
                raw,
                {
                    "meta/" + name: tf.io.FixedLenFeature([], dtype)
                    for name, dtype in (
                        ("game_id", tf.string),
                        ("kyoku_id", tf.string),
                        ("decision_index", tf.int64),
                        ("actor", tf.int64),
                    )
                },
            )
            return {
                **decode_records(raw),
                "metadata": {
                    name.removeprefix("meta/"): value
                    for name, value in metadata.items()
                },
                "natural": (
                    step * tf.cast(sizes[group.rank], tf.int64)
                    + tf.range(tf.size(raw), dtype=tf.int64)
                )
                % 2
                == 0,
                "local_count": tf.cast(valid_counts[group.rank], tf.int32),
                "global_count": tf.reduce_sum(valid_counts),
            }

        return stream.enumerate().map(decode, num_parallel_calls=2).prefetch(2)


def save_weights(network, path):
    pending = path.with_name("pending." + path.name)
    network.save_weights(pending)
    pending.replace(path)


def run(args):  # noqa: C901 — executable outline for initialization
    if args.batch_size <= 0 or args.rollout_batch_size <= 0:
        raise ValueError("Batch size must be positive")
    command = json.loads(pathlib.Path(args.evaluation_command).read_text())
    if int(command[command.index("--games") + 1]) != 10000:
        raise ValueError(
            "The initial main-loop match must contain 10,000 games"
        )
    for placeholder in ("{weights}", "{output}"):
        if not any(placeholder in value for value in command):
            raise ValueError("Evaluation command needs {weights} and {output}")
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) > 1:
        raise ValueError("Select one learner GPU per worker")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(args.seed)
    group = distributed.LearnerGroup()
    root = pathlib.Path(args.output).resolve()
    if group.chief:
        root.mkdir(parents=True, exist_ok=False)
    with group.strategy.scope():
        features = self_play.initial_features(args.seed)
        network = export.load_network(args.initial, features)
        mirror = export.load_network(args.initial, features).actor
        mirror.trainable = False
    reference_policy = fixed_reference.Policy(args.reference)
    summary_path = pathlib.Path(args.data) / "dataset_summary.json"
    summary = json.loads(summary_path.read_text())
    expected = sum(summary["partition_counts"][:7])
    prepared_roots = json.loads(
        (pathlib.Path(args.data) / "root-states/complete.json").read_text()
    )
    assert (
        prepared_roots["summary_sha256"]
        == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    )
    assert prepared_roots["input_schema"] == _native_collection.INPUTS
    assert prepared_roots["limit_games"] is None
    assert prepared_roots["sakigiri"] == sakigiri_regret.DEFINITION
    source = root_states.Store(pathlib.Path(args.data) / "root-states")
    predict = self_play.predictor(network.actor)
    evaluate_roots = collection.root_scorer(network)
    if group.chief:
        save_weights(network, root / "initial.weights.h5")
        monitor.write_json(
            root / "plan.json",
            {
                **vars(args),
                "method": "q-weighted-expert-imitation",
                "passes": 3,
                "sampling": {
                    "natural": 0.5,
                    "focused": 0.5,
                },
                "pass_exposures": expected,
                "imitation_rollouts": {
                    "categories": list(collection.CATEGORIES),
                    "positions": "focused roots paired with natural-source roots",
                    "targeted_root_fraction": 0.5,
                    "sakigiri": sakigiri_regret.DEFINITION,
                    "advancing_calls": "all legal advancing calls plus pass",
                    "continuations_per_action": 1,
                    "position_budget": (
                        "16-batch estimated simulated-decision target "
                        "equals human exposures"
                    ),
                    "budget_unit": "all requested decisions in every continuation",
                    "sampling": (
                        "balanced seeds per batch; "
                        "collected together every 16 batches"
                    ),
                    "overshoot": "finish started hands; no debt carryover",
                    "paired_walls": "one shared target wall; normal branches independent",
                    "policy": "frozen learner at collection window end",
                    "collection_interval_batches": 16,
                    "initial_decisions_per_continuation": 32,
                    "length_estimate": (
                        "previous window measured decisions per continuation"
                    ),
                    "training": "critic then MMD actor after each human window",
                    "mmd": {
                        "mirror_kl": 0.3,
                        "magnet_kl": 0.3,
                        "actor_constraints": 0.01,
                        "learning_rate": 1e-5,
                        "magnet": reference_policy.provenance,
                        "mirror": "learner immediately before MC critic fit",
                    },
                    "imitation_labels": "unchanged human actions",
                },
                "replicas": group.replicas,
                "expected_exposures": expected * 3,
                "evaluation_argv": command,
                "initial_sha256": hashlib.sha256(
                    pathlib.Path(args.initial).read_bytes()
                ).hexdigest(),
                "dataset_sha256": hashlib.sha256(
                    summary_path.read_bytes()
                ).hexdigest(),
                "reference_policy": summary["preparation"]["mortal_policy"],
                "weighting_critic": "current-network-detached",
                "learning_rates": {"imitation": 1e-5},
                "beta_target": imitation.ESS_TARGET,
                "beta_bounds": [0.0001, 1.28],
                "raw_weight_cap": 20,
                "beta_calibration": "global-batch-highest-crossing",
                "loss_coefficients": {
                    "imitation": {
                        "expert_ce": 1,
                        "each_auxiliary": 1,
                        "payment_nll": 1,
                        "critic_constraint_loss": 1,
                    },
                },
            },
        )
    writer = (
        tf.summary.create_file_writer(str(root / "tensorboard"))
        if group.chief
        else None
    )
    started = time.monotonic()
    exposures = actor_exposures = critic_exposures = steps = 0

    mc_human_positions = mc_decisions = mc_actor_exposures = 0

    def report(stage, metrics):
        status = {
            "stage": stage,
            "pass": expert_pass,
            "passes": 3,
            "exposures": exposures,
            "actor_exposures": actor_exposures,
            "critic_exposures": critic_exposures,
            "steps": steps,
            "mc_human_positions": mc_human_positions,
            "mc_simulated_decisions": mc_decisions,
            "mc_actor_exposures": mc_actor_exposures,
            "elapsed_seconds": time.monotonic() - started,
            "updated_at": time.time(),
            **metrics,
        }
        monitor.write_json(root / "status.json", status)
        with (root / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(status) + "\n")
        with writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(stage + "/" + name, value, step=exposures)
        writer.flush()
        if steps and steps % 256 == 0:
            save_weights(network, root / "latest.weights.h5")

    complete = True
    for expert_pass in range(1, 4):
        with group.strategy.scope():
            learner = imitation.ImitationTrainer(
                network=network,
                features=features,
                learning_rate=1e-5,
            )
            rollout_actor = mmd_trainer.MMDTrainer(
                network=network, mirror=mirror, learning_rate=1e-5
            )
        group.bind_expert(
            learner=learner,
            rollout_actor=rollout_actor,
            batch_size=args.batch_size,
        )
        if group.chief:
            report("imitation", {})
        batches = input_batches(
            data=args.data,
            group=group,
            batch_size=args.batch_size,
            seed=args.seed + expert_pass - 1,
        )
        group.barrier()
        decisions_per_continuation = 32.0
        seeds = []
        decision_target = 0
        for batch_number, batch in enumerate(batches):
            seeds.extend(
                imitation_rollouts.select_seeds(
                    source=source,
                    batch=batch,
                    predict=predict,
                    evaluate=evaluate_roots,
                    seed=args.seed + exposures + group.rank * group.local_batch,
                    decisions_per_continuation=decisions_per_continuation,
                    stop=root / "STOP",
                )
            )
            decision_target += int(batch["local_count"])
            metrics = group.update(
                "imitation",
                {
                    name: value
                    for name, value in batch.items()
                    if name not in ("metadata", "natural")
                },
                root,
            )
            if metrics is None:
                complete = False
                break
            exposures += int(batch["global_count"])
            trained = int(metrics["trained_positions"])
            actor_exposures += trained
            critic_exposures += trained
            steps += bool(trained)
            if group.chief:
                report(
                    "imitation",
                    {name: float(value) for name, value in metrics.items()},
                )
            if (batch_number + 1) % 16 and exposures < expected * expert_pass:
                continue
            rollout_rows, rollout_statistics = imitation_rollouts.collect(
                source=source,
                seeds=seeds,
                reference_policy=reference_policy,
                decision_target=decision_target,
                predict=predict,
                packing=group.packing,
                directory=root
                / "imitation-rollouts"
                / f"pass-{expert_pass}"
                / f"window-{batch_number // 16:06d}-rank-{group.rank}",
                stop=root / "STOP",
            )
            credited, spent, continuations = group.sum_counts(
                tf.constant(
                    [
                        decision_target,
                        rollout_statistics["simulated_decisions"],
                        rollout_statistics["continuations"],
                    ],
                    tf.int32,
                )
            ).numpy()
            mc_human_positions += int(credited)
            mc_decisions += int(spent)
            if continuations:
                decisions_per_continuation = float(spent / continuations)
            seeds = []
            decision_target = 0
            mirror.set_weights(network.actor.get_weights())
            for update in ("imitation-rollout", "imitation-mmd"):
                for rollout_batch in imitation_rollouts.batches(
                    rollout_rows, batch, group, args.rollout_batch_size
                ):
                    metrics = group.update(update, rollout_batch, root)
                    if metrics is None:
                        complete = False
                        break
                    trained = int(metrics["trained_positions"])
                    if update == "imitation-rollout":
                        critic_exposures += trained
                    else:
                        actor_exposures += trained
                        mc_actor_exposures += trained
                    steps += bool(trained)
                    if group.chief:
                        report(
                            update,
                            {
                                name: float(value)
                                for name, value in metrics.items()
                            },
                        )
                if not complete:
                    break
            if devices:
                monitor.write_json(
                    root / f"memory-rank-{group.rank}.json",
                    {
                        **tf.config.experimental.get_memory_info("GPU:0"),
                        "human_exposures": exposures,
                        "simulated_decisions": mc_decisions,
                        "updated_at": time.time(),
                    },
                )
            if not complete:
                break
        if not complete:
            break
        assert exposures == expected * expert_pass
        if group.chief:
            save_weights(network, root / f"pass-{expert_pass}.weights.h5")
            save_weights(network, root / "latest.weights.h5")
    group.barrier()
    source.close()
    if group.chief:
        save_weights(network, root / "latest.weights.h5")
        if complete:
            save_weights(network, root / "final.weights.h5")
            monitor.write_json(
                root / "training-complete.json",
                {
                    "weights": str(root / "final.weights.h5"),
                    "exposures": exposures,
                    "steps": steps,
                    "mc_human_positions": mc_human_positions,
                    "mc_simulated_decisions": mc_decisions,
                    "mc_actor_exposures": mc_actor_exposures,
                },
            )
        report("main-loop-ready" if complete else "stopped", {})
    group.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data",
        "initial",
        "reference",
        "output",
        "evaluation-command",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    existing_output = pathlib.Path(args.output).exists()
    try:
        run(args)
    finally:
        path = pathlib.Path(args.output) / "status.json"
        if (
            not existing_output
            and sys.exception() is not None
            and (
                "TF_CONFIG" not in os.environ
                or json.loads(os.environ["TF_CONFIG"])["task"]["index"] == 0
            )
            and path.exists()
        ):
            status = json.loads(path.read_text())
            status.update(
                stage="failed",
                error=str(sys.exception()),
                updated_at=time.time(),
            )
            monitor.write_json(path, status)


if __name__ == "__main__":
    main()
