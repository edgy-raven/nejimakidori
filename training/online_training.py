"""Fresh self-play optimization with no expert dataset or auxiliary replay."""

import argparse
import hashlib
import json
import pathlib
import time

import numpy
import tensorflow as tf

from log_dataset import _native_collection, training_data
from model import export
from training import self_play
from training import trainer as training


def save(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def disagreement_evaluator(network):
    """Compare discard max-Q and closest-to-V in projected current-hand ST3."""
    module = export.ServingModel(network)
    specs = {
        name: tf.TensorSpec([None, *spec["shape"]], spec["dtype"])
        for name, spec in _native_collection.INPUTS.items()
    }
    signature = module.root_evaluation.get_concrete_function(**specs)
    packing = training_data.PackedBatch(specs)

    @tf.function(reduce_retracing=True)
    def predict(inputs):
        return signature(**packing.unpack(inputs))

    def evaluate(features):
        with tf.device("/CPU:0"):
            inputs = {
                name: tf.convert_to_tensor(value)
                for name, value in packing.pack_numpy(features).items()
            }
        outputs = predict(inputs)
        values = outputs["state_value"].numpy() * 100
        indices = outputs["critic_action_indices"].numpy()
        estimates = outputs["critic_projected_st3"].numpy()
        priority = numpy.zeros(len(values), numpy.float32)
        pairs = numpy.full((len(values), 2), -1, numpy.int32)
        # Encoder tf.where order is preserved by the export's active mask.
        boundaries = numpy.searchsorted(
            indices[:, 0], numpy.arange(len(values) + 1)
        )
        for row, (begin, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            codes = indices[begin:end, 1]
            q = estimates[begin:end]
            best = q.argmax()
            closest = numpy.abs(q - values[row]).argmin()
            if best == closest:
                continue
            pairs[row] = codes[[best, closest]]
            priority[row] = (
                q[best] - q[closest] >= 0.25
                or abs(float(q[closest] - values[row])) >= 8
            )
        return {"priority": priority, "actions": pairs}

    return evaluate


def run(args):
    root = pathlib.Path(args.output)
    assert args.exposures > 0 and args.cohort_exposures > 0
    assert args.ppo_epochs > 0 and args.phase_cohorts > 0
    assert args.exposures % args.cohort_exposures == 0
    assert args.cohort_exposures % args.ppo_epochs == 0
    assert all(
        name in ("ppo", "awr", "ppo-awr") for name in args.objectives.split(",")
    )
    assert all(
        name
        in ("natural", "keiten", "tenpai", "calls", "disagreement", "targeted")
        for name in args.focus.split(",")
    )
    assert args.root_copies > 0
    if set(args.focus.split(",")) != {"natural"} and set(
        args.objectives.split(",")
    ) != {"awr"}:
        raise ValueError("Paired forced-root continuations require AWR")
    assert numpy.isfinite(args.awr_temperature) and args.awr_temperature > 0
    assert numpy.isfinite(args.awr_entropy) and args.awr_entropy >= 0
    devices = tf.config.list_physical_devices("GPU")
    assert devices, "Visible training GPUs required"
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    strategy = tf.distribute.MirroredStrategy()
    assert args.batch_size % strategy.num_replicas_in_sync == 0
    tf.keras.utils.set_random_seed(args.seed)
    root.mkdir(parents=True, exist_ok=False)
    # Unused offline optimizer exists for checkpoint structure only.
    args.foundation_exposures = args.awr_exposures = 0
    args.awr_only_exposures = args.final_awr_exposures = 0
    args.block_exposures = 1
    args.supervised_learning_rate = args.ppo_learning_rate
    save(
        root / "plan.json",
        {
            **vars(args),
            "expert_records": False,
            "initial_sha256": hashlib.sha256(
                pathlib.Path(args.initial).read_bytes()
            ).hexdigest(),
        },
    )
    features = self_play.initial_features(args.seed)
    with strategy.scope():
        network = export.load_network(args.initial, features)
        local = training.Trainer(
            network=network, args=args, first=(features, {}, {})
        )
    trainer = training.DistributedTrainer(local, strategy)
    optimizer_values = [
        value.numpy()
        for optimizer in (local.policy_optimizer, local.awr_optimizer)
        for value in optimizer.variables
        if "learning_rate" not in value.name
    ]
    assert all(numpy.all(value == 0) for value in optimizer_values)
    save(
        root / "reset-verification.json",
        {
            "verified": True,
            "exposures": 0,
            "cohort": 0,
            "optimizer_tensors": len(optimizer_values),
            "expert_records": False,
        },
    )
    exposures = games = collected = cohort = 0
    started = time.monotonic()
    writer = tf.summary.create_file_writer(str(root / "tensorboard"))

    def report(stage, metrics):
        status = {
            "stage": stage,
            "total_exposures": exposures,
            "budget_exposures": args.exposures,
            "cohort": cohort,
            "games": games,
            "collected": collected,
            "elapsed_seconds": time.monotonic() - started,
            "expert_records": False,
            **metrics,
        }
        save(root / "status.json", status)
        with (root / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(status) + "\n")
        with writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=exposures)
        writer.flush()

    report("initialized", {})
    with strategy.scope():
        predict = self_play.predictor(network.actor)
    with self_play.Collector(
        predict,
        args.collection_workers,
        root_copies=args.root_copies,
        root_evaluate=(
            disagreement_evaluator(network)
            if set(args.focus.split(",")) != {"natural"}
            else None
        ),
    ) as collector:
        while exposures < args.exposures:
            args.rollout_objective = args.objectives.split(",")[
                (cohort // args.phase_cohorts) % len(args.objectives.split(","))
            ]
            collector.focus = args.focus.split(",")[
                (cohort // args.phase_cohorts) % len(args.focus.split(","))
            ]
            report("collecting", {})
            batch, count, metrics = training.collect_cohort(
                collector=collector,
                args=args,
                needed=args.cohort_exposures // args.ppo_epochs,
                seed=args.seed + 100000 + games,
                prior_games=games,
                prior_rows=collected,
                snapshot_exposures=exposures,
            )
            report("updating", metrics)
            used, updates = training.ppo_block(
                trainer=trainer,
                args=args,
                stream=None,
                batch=batch,
                remaining=args.cohort_exposures,
            )
            exposures += used
            games += count
            collected += len(batch["chosen"])
            cohort += 1
            network.save_weights(root / "latest.weights.h5")
            report("cohort_complete", {**metrics, **updates})
    network.save_weights(root / "final.weights.h5")
    report("complete", {})
    writer.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exposures", type=int, required=True)
    parser.add_argument("--cohort-exposures", type=int, default=995328)
    parser.add_argument("--objectives", default="ppo")
    parser.add_argument("--focus", default="natural")
    parser.add_argument("--root-copies", type=int, default=1)
    parser.add_argument("--phase-cohorts", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-learning-rate", type=float, default=1e-5)
    parser.add_argument("--awr-temperature", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=9216)
    parser.add_argument("--collection-workers", type=int, default=24)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=3)
    parser.add_argument("--entropy", type=float, default=0.001)
    parser.add_argument("--awr-entropy", type=float, default=0.01)
    parser.add_argument("--prior-weight", type=float, default=0)
    return parser


if __name__ == "__main__":
    run(argument_parser().parse_args())
