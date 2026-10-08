"""Independent belief updates and recoverable optimizer state."""

import argparse
import json
import pathlib

import tensorflow as tf

from log_dataset import critic_data, records
from model import beliefs, objectives


def optimizer_checkpoint(network, learning_rate):
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0
    )
    optimizer.build(network.trainable_variables)
    return tf.train.Checkpoint(beliefs=network, optimizer=optimizer)


@tf.function
def update(checkpoint, inputs):
    features, labels, _ = inputs
    with tf.GradientTape() as tape:
        outputs = checkpoint.beliefs(features)
        terms, metrics = beliefs.loss_terms(features, labels, outputs)
        loss = tf.add_n(list(terms.values())) / 3
    gradients = tape.gradient(loss, checkpoint.beliefs.trainable_variables)
    tf.debugging.assert_all_finite(
        tf.linalg.global_norm(gradients), "nonfinite belief gradient"
    )
    checkpoint.optimizer.apply_gradients(
        zip(gradients, checkpoint.beliefs.trainable_variables)
    )
    return {"belief_loss": loss, **terms, **metrics}


def run(args):
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    records.validate_summary(args.data)
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "data": str(pathlib.Path(args.data).resolve()),
        "initialization": str(pathlib.Path(args.initialization).resolve()),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "objective": "shanten_conditional_ukeire_hand_counts",
        "count_projection": "weighted_KL_public_hand_sizes_shared_copies",
        "ukeire_target": "hard_structural_0_1",
        "sampling": "natural_train_partition",
    }
    if (root / "plan.json").exists():
        assert json.loads((root / "plan.json").read_text()) == plan
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    training = iter(
        critic_data.dataset(
            root=args.data,
            sampling="natural",
            partition="train",
            batch_size=args.batch_size,
            repeat=True,
        )
    )
    validation = list(
        critic_data.dataset(
            root=args.data,
            sampling="natural",
            partition="validation",
            batch_size=args.batch_size,
            repeat=False,
        ).take(args.validation_batches)
    )
    assert len(validation) == args.validation_batches
    network = beliefs.OpponentBeliefs()
    network(critic_data.parse_batch(validation[0])[0])
    network.load_weights(args.initialization)
    checkpoint = optimizer_checkpoint(network, args.learning_rate)
    manager = tf.train.CheckpointManager(checkpoint, str(root / "recovery"), 2)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).assert_consumed()
    writer = tf.summary.create_file_writer(str(root / "events"))
    while int(checkpoint.optimizer.iterations) < args.steps:
        metrics = update(checkpoint, critic_data.parse_batch(next(training)))
        step = int(checkpoint.optimizer.iterations)
        if step % 20 == 0:
            with writer.as_default(step=step):
                for name, value in metrics.items():
                    tf.summary.scalar("train/" + name, value)
        if step % args.validate_every == 0 or step == args.steps:
            statistics = []
            for raw in validation:
                features, labels, _ = critic_data.parse_batch(raw)
                outputs = network(features)
                means = objectives.MaskedMeans()
                beliefs.loss_terms(
                    features, labels, outputs, mean=means.collect
                )
                statistics.append(tf.stack(means.statistics))
            means.statistics = list(tf.unstack(tf.reduce_sum(statistics, 0)))
            means.reduce()
            terms, values = beliefs.loss_terms(
                features, labels, outputs, mean=means.read
            )
            result = {
                name: float(value)
                for name, value in {**terms, **values}.items()
            }
            with writer.as_default(step=step):
                for name, value in result.items():
                    tf.summary.scalar("validation/" + name, value)
            (root / f"validation-{step}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            network.save_weights(root / f"beliefs-{step}.weights.h5")
            manager.save(checkpoint_number=step)
            writer.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Total optimizer steps, including recovered steps",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=100)
    args = parser.parse_args()
    assert (
        min(
            args.steps,
            args.batch_size,
            args.learning_rate,
            args.validate_every,
            args.validation_batches,
        )
        > 0
    )
    run(args)


if __name__ == "__main__":
    main()
