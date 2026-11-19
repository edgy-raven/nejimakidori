"""Collect balanced root reruns and read their packed training shards."""

import numpy
import tensorflow as tf

from model import actions, critic
from training import monitor
from training.corrections import collection, matches


def batches(paths, packing, batch_size, seed):
    """Visit every retained decision once, with a bounded in-memory shard."""
    generator = numpy.random.default_rng(seed)
    for file_index in generator.permutation(len(paths)):
        with numpy.load(paths[file_index]) as archive:
            rows = {
                "features": {
                    name.removeprefix("features/"): archive[name]
                    for name in archive.files
                    if name.startswith("features/")
                },
                **{
                    name: archive[name]
                    for name in archive.files
                    if not name.startswith("features/")
                },
            }
        indices = generator.permutation(len(rows["chosen"]))
        for begin in range(0, len(indices), batch_size):
            selected = indices[begin : begin + batch_size]
            with tf.device("/CPU:0"):
                tensors = tf.nest.map_structure(
                    lambda value: tf.convert_to_tensor(value[selected]), rows
                )
                tensors["features"] = packing.unpack(tensors["features"])
            yield tensors


def collect(
    *,
    root,
    round_number,
    match_directory,
    recorded,
    order,
    seed,
    predict,
    root_evaluate,
    packing,
    report,
):
    """Mine a round in bounded chunks, retaining completed shards on stop."""
    rollout_files = []
    counts = {}
    rollout_positions = {"targeted": 0, "normal": 0}

    def retain(batch, source):
        path = root / (
            f"round-{round_number}-{source}-"
            f"{len(rollout_files):05d}-rollouts.npz"
        )
        pending = path.with_suffix(".partial.npz")
        numpy.savez_compressed(
            file=pending,
            **{
                "features/" + name: value
                for name, value in packing.pack_numpy(batch["features"]).items()
            },
            head=batch["head"],
            legal=batch["legal"],
            chosen=critic.chosen_actions(
                batch["features"],
                {
                    name: numpy.where(
                        batch["head"] == head, batch["chosen"], -1
                    )
                    for head, name in enumerate(actions.TRAIN_POLICY_HEADS)
                },
            ).numpy(),
            payment_points=batch["payment_points"],
            paired_root=batch["paired_root"],
        )
        pending.replace(path)
        rollout_files.append(path)
        rollout_positions[source] += len(batch["head"])

    for begin in range(0, len(order), 32):
        selected = order[begin : begin + 32]
        batch, collected = collection.collect(
            predict=predict,
            root_evaluate=root_evaluate,
            games=len(selected),
            seed=seed + begin * 4,
            retain=retain,
            discover=lambda capture: matches.discover(
                predict=predict,
                games=[recorded[index] for index in selected],
                capture=capture,
            ),
        )
        path = root / f"round-{round_number}-{begin:05d}-roots.npz"
        numpy.savez_compressed(
            file=path,
            **{
                "features/" + name: value
                for name, value in batch["features"].items()
            },
            **{
                name: value
                for name, value in batch.items()
                if name != "features"
            },
            match_index=selected[batch["source"][:, 0]],
        )
        for name, value in collected.items():
            counts[name] = counts.get(name, 0) + value
        report(
            "collecting-roots",
            {
                **counts,
                **{
                    "positions/" + name: value
                    for name, value in rollout_positions.items()
                },
            },
        )
        if (root / "STOP").exists():
            break
    monitor.write_json(
        root / f"round-{round_number}-rollouts.json",
        {
            "positions": rollout_positions,
            "files": [path.name for path in rollout_files],
            "match_source": str(match_directory),
            "games": [
                recorded[index] for index in order[: counts["logged_games"]]
            ],
            "features": "PackedBatch",
            "chosen": "critic-action-index",
        },
    )
