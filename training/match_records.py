"""Current-hand critic records from completed evaluation match logs."""

import argparse
import concurrent.futures
import gzip
import json
import multiprocessing
import pathlib

import numpy
import tensorflow as tf
from log_dataset import (
    features,
    records,
    replay,
    scoring,
    training_data,
    training_targets,
)
from model import critic, outcomes


def game_records(path):
    """All recoverable choices, without hindsight rescoring or AFK filtering."""
    path = pathlib.Path(path)
    with gzip.open(path, "rt") as stream:
        events = [json.loads(line) for line in stream]
    outcomes.normalize_draw_payments(events)
    settlements = outcomes.hand_continuation_states(events)
    starts = [
        index
        for index, event in enumerate(events)
        if event["type"] == "start_kyoku"
    ]
    for hand, (start, end) in enumerate(
        zip(starts, starts[1:] + [len(events)])
    ):
        state = replay.Replay(path.stem, events[start:end])
        payments = outcomes.terminal_payments(events[start:end])
        target = {
            "payment": payments,
            "settlement": settlements[hand],
            "bonus": numpy.full((4, 4, 4), -1, numpy.int16),
            "score": numpy.full(
                (4, 4, 4, len(scoring.BONUS_HAN)), -1, numpy.int16
            ),
        }
        for (
            index,
            actor,
            phase,
            next_index,
            win_choices,
        ) in state.policy_decisions():
            observation = replay.Observation(
                game_id=path.stem,
                kyoku_id=state.kyoku_id,
                decision_index=state.decision_index,
                actor=actor,
                features=features.pack(state, actor, phase),
                focus=(0, 0, 0),
                hand_yaku_focus=0,
            )
            for name, definition in records.LABELS.items():
                setattr(
                    observation,
                    name,
                    numpy.full(definition["shape"], -1, definition["dtype"]),
                )
            training_targets.attach_beliefs(observation, state)
            observation.terminal_payment = outcomes.relative_payments(
                payments, actor, state.accepted
            )
            state.recorded_action(
                observation=observation,
                index=index,
                phase=phase,
                next_index=next_index,
                win_choices=win_choices,
            )
            yield training_data.serialize(observation, target)


def write_shard(task):
    tf.config.set_visible_devices([], "GPU")
    path, games = task
    positions = 0
    with tf.io.TFRecordWriter(str(path) + ".partial", options="GZIP") as writer:
        for game in games:
            for raw in game_records(game):
                writer.write(raw)
                positions += 1
    pathlib.Path(str(path) + ".partial").replace(path)
    return {"path": path.name, "games": len(games), "positions": positions}


def decode_records(raw):
    """Chosen critic action and observed payments, with no actor targets."""
    features, labels, auxiliary = training_data.parse_batch(raw)
    return {
        "features": features,
        "chosen": critic.chosen_actions(features, labels),
        "beliefs": {
            name: labels[name]
            for name in (
                "opponent_shanten",
                "opponent_ukeire",
                "opponent_hand_counts",
            )
        },
        "payment_points": tf.gather(
            tf.reshape(labels["terminal_payment"], [-1, 64]),
            numpy.ravel_multi_index(numpy.asarray(scoring.CELLS).T, (4, 4, 4)),
            axis=1,
        ),
        "projected_st3": auxiliary["projected_st3"],
    }


def batches(output, batch_size, seed):
    """One finite shuffled pass, including every row in the final batch."""
    output = pathlib.Path(output)
    manifest = json.loads((output / "dataset.json").read_text())
    return (
        tf.data.Dataset.from_tensor_slices(
            [str(output / row["path"]) for row in manifest["shards"]]
        )
        .shuffle(len(manifest["shards"]), seed=seed)
        .interleave(
            lambda path: tf.data.TFRecordDataset(path, compression_type="GZIP"),
            cycle_length=8,
            block_length=64,
            num_parallel_calls=4,
            deterministic=True,
        )
        .shuffle(8192, seed=seed)
        .batch(batch_size)
        .map(
            lambda raw: {
                name: value
                for name, value in decode_records(raw).items()
                if name != "beliefs"
            },
            num_parallel_calls=2,
        )
        .prefetch(2)
    )


def build(matches, output, workers):
    """Convert the completed evaluator manifest in isolated CPU workers."""
    matches = pathlib.Path(matches)
    output = pathlib.Path(output)
    manifest = json.loads((matches / "evaluation.json").read_text())
    games = [
        json.loads(line)
        for line in (matches / "games.jsonl").read_text().splitlines()
    ]
    assert len(games) == manifest["games"]
    paths = sorted(matches / game["replay"] for game in games)
    assert len(set(paths)) == len(paths)
    output.mkdir(parents=True)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        shards = list(
            pool.map(
                write_shard,
                [
                    (
                        output / f"train-{start // 100:04d}.tfrecord.gz",
                        paths[start : start + 100],
                    )
                    for start in range(0, len(paths), 100)
                ],
            )
        )
    result = {
        "games": len(games),
        "positions": sum(row["positions"] for row in shards),
        "shards": shards,
    }
    (output / "dataset.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    print(
        json.dumps(
            build(arguments.matches, arguments.output, arguments.workers)
        )
    )
