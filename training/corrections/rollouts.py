"""Collect balanced root reruns and read their packed training shards."""

import numpy
import tensorflow as tf

from log_dataset import call_states
from model import actions, critic
from training import auxiliary, monitor
from training.corrections import collection, matches


def select(rows, indices):
    return {
        "features": call_states.packed_select(rows["features"], indices),
        **{
            name: tf.nest.map_structure(lambda array: array[indices], value)
            for name, value in rows.items()
            if name != "features"
        },
    }


def concatenate(rows):
    return {
        "features": call_states.packed_concatenate(
            [row["features"] for row in rows]
        ),
        **{
            name: tf.nest.map_structure(
                lambda *arrays: numpy.concatenate(arrays),
                *[row[name] for row in rows],
            )
            for name in rows[0]
            if name != "features"
        },
    }


def batches(paths, packing, batch_size, seed, localize=None):
    generator = numpy.random.default_rng(seed)
    pending = []
    count = 0
    for file_index in generator.permutation(len(paths)):
        with numpy.load(paths[file_index]) as archive:
            rows = {
                "features": call_states.unflatten(
                    {
                        name.removeprefix("features/"): archive[name]
                        for name in archive.files
                        if name.startswith("features/")
                    }
                ),
                "labels": {
                    name: archive["label/" + name] for name in auxiliary.LABELS
                },
                **{
                    name: archive[name]
                    for name in archive.files
                    if not name.startswith(("features/", "label/"))
                },
            }
        indices = generator.permutation(len(rows["chosen"]))
        for begin in range(0, len(indices), batch_size):
            pending.append(select(rows, indices[begin : begin + batch_size]))
            count += len(pending[-1]["chosen"])
            if count >= batch_size:
                combined = concatenate(pending)
                yield unpack(
                    select(combined, numpy.arange(batch_size)),
                    packing,
                    localize,
                )
                pending = [select(combined, numpy.arange(batch_size, count))]
                count -= batch_size
    if count:
        yield unpack(concatenate(pending), packing, localize)


def unpack(rows, packing, localize):
    with tf.device("/CPU:0"):
        batch = tf.nest.map_structure(tf.convert_to_tensor, rows)
        if localize is not None:
            batch = localize(batch)
        batch["features"] = call_states.unpack(batch["features"], packing)
        return batch


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
    reference_policy,
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
                for name, value in call_states.flatten(
                    call_states.pack_numpy(batch["features"], packing)
                ).items()
            },
            **{
                "label/" + name: value
                for name, value in batch["labels"].items()
            },
            head=batch["head"],
            legal=batch["legal"],
            reference_log_probability=batch["reference_log_probability"],
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
            reference_policy=reference_policy,
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
            "features": "sparse-call PackedBatch",
            "auxiliary_labels": list(auxiliary.LABELS),
            "reference_policy": reference_policy.provenance,
            "reference_log_probability": {"dtype": "float32", "shape": [150]},
            "chosen": "critic-action-index",
        },
    )
