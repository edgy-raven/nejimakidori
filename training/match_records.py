"""Current-hand critic records from completed evaluation match logs."""

import argparse
import concurrent.futures
import functools
import gzip
import json
import multiprocessing
import pathlib

import numpy
import tensorflow as tf

from log_dataset import (
    call_states,
    features,
    records,
    replay,
    scoring,
    training_data,
    training_targets,
)
from model import critic, outcomes
from training import auxiliary, fixed_reference, restoration


def game_records(path, reference_policy):
    """Label all recoverable choices in a completed four-seat match."""
    path = pathlib.Path(path)
    with gzip.open(path, "rt") as stream:
        events = [json.loads(line) for line in stream]
    outcomes.normalize_draw_payments(events)
    starts = [
        index
        for index, event in enumerate(events)
        if event["type"] == "start_kyoku"
    ]
    raw = []
    requests = []
    for start, end in zip(starts, starts[1:] + [len(events)]):
        state = replay.Replay(path.stem, events[start:end])
        target = {
            "payment": outcomes.terminal_payments(state.events),
            "bonus": numpy.full((4, 4, 4), -1, numpy.int16),
            "score": numpy.full(
                (4, 4, 4, len(scoring.BONUS_HAN)), -1, numpy.int16
            ),
        }
        for observation in state.observations():
            raw.append(training_data.serialize(observation, target))
            requests.append(
                (
                    start + state.event_index,
                    observation.actor,
                    observation.features["decision_phase"],
                )
            )
    packed, labels = records.parse_batch(raw)
    heads, _, legal = restoration.policy_choices(packed, labels)
    probabilities = reference_policy.distribution(
        events, requests, heads.numpy(), legal.numpy()
    )
    for value, probability in zip(raw, probabilities, strict=True):
        example = tf.train.Example.FromString(value)
        example.features.feature[
            "reference_log_probability"
        ].bytes_list.value.append(probability.astype(numpy.float32).tobytes())
        yield example.SerializeToString()


def label_rollouts(logs, indices, packed, *, root_affinity=None):
    """Join native decisions to their branch's hidden hands and future tenpai.

    root_affinity may be supplied when all rows retain the same root features.
    """
    labels = {
        name: numpy.empty(
            (len(indices), *records.LABELS[name]["shape"]),
            records.LABELS[name]["dtype"],
        )
        for name in auxiliary.LABELS
    }
    base = call_states.base(packed)
    phases = {code: phase for phase, code in features.PHASES.items()}
    for game, events in enumerate(logs):
        starts = [
            index
            for index, event in enumerate(events)
            if event["type"] == "start_kyoku"
        ] + [len(events)]
        game_rows = numpy.flatnonzero(indices[:, 0] == game)
        for hand in numpy.unique(indices[game_rows, 1]):
            hand_events = events[starts[hand] : starts[hand + 1]]
            state = replay.Replay(str(game), hand_events)
            hand_rows = game_rows[indices[game_rows, 1] == hand]
            state.targets = training_targets.round_targets(
                replay.Replay.from_start(str(game), hand_events[0]),
                hand_events,
                indices=indices[hand_rows, 3] - starts[hand] - 1,
            )
            for row in hand_rows:
                # Native event is the number of events consumed before next().
                # A declined win or kan may yield a new phase at the same event.
                index = indices[row, 3] - starts[hand] - 1
                while state.event_index < index:
                    state.apply(hand_events[state.event_index + 1])
                assert state.event_index == index
                observation = replay.Observation(
                    actor=int(indices[row, 2]),
                    features={
                        name: values[row] for name, values in base.items()
                    },
                )
                training_targets.attach(
                    observation, state, index, affinity=root_affinity
                )
                observation.ron_legal = training_targets.ron_labels(
                    state,
                    observation.actor,
                    phases[base["decision_phase"][row]],
                )["ron_legal"]
                for name in labels:
                    labels[name][row] = getattr(observation, name)
    return labels


@functools.cache
def load_reference(directory):
    tf.config.set_visible_devices([], "GPU")
    return fixed_reference.Policy(directory)


def write_shard(task):
    path, games, reference = task
    reference_policy = load_reference(reference)
    positions = 0
    with tf.io.TFRecordWriter(str(path) + ".partial", options="GZIP") as writer:
        for game in games:
            for raw in game_records(game, reference_policy):
                writer.write(raw)
                positions += 1
    pathlib.Path(str(path) + ".partial").replace(path)
    return {
        "path": path.name,
        "games": len(games),
        "positions": positions,
        "reference_policy": reference_policy.provenance,
    }


def decode_records(raw):
    """Chosen critic action, observed payments and supervised belief labels."""
    parsed = tf.io.parse_example(
        raw,
        {
            name: tf.io.FixedLenFeature([], tf.string)
            for name in (*records.FIELDS, "reference_log_probability")
        },
    )
    features, labels = records.decode_batch(parsed)
    head, _, legal = restoration.policy_choices(features, labels)
    return {
        "head": head,
        "legal": legal,
        "features": features,
        "labels": {name: labels[name] for name in auxiliary.LABELS},
        "reference_log_probability": tf.reshape(
            tf.io.decode_raw(
                parsed["reference_log_probability"],
                tf.float32,
            ),
            [-1, 150],
        ),
        "chosen": critic.chosen_actions(features, labels),
        "payment_points": tf.gather(
            tf.reshape(labels["terminal_payment"], [-1, 64]),
            numpy.ravel_multi_index(numpy.asarray(scoring.CELLS).T, (4, 4, 4)),
            axis=1,
        ),
    }


def batches(output, batch_size, seed, decode=decode_records):
    """One finite shuffled pass, including every row in the final batch."""
    output = pathlib.Path(output)
    manifest = json.loads((output / "dataset.json").read_text())
    if manifest["auxiliary_labels"] != list(auxiliary.LABELS):
        raise ValueError("Rebuild match records with supervised belief labels")
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
        .map(decode, num_parallel_calls=2)
        .prefetch(2)
    )


def build(matches, output, workers, reference):
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
                        reference,
                    )
                    for start in range(0, len(paths), 100)
                ],
            )
        )
    reference_policy = shards[0].pop("reference_policy")
    assert all(
        row.pop("reference_policy") == reference_policy for row in shards[1:]
    )
    result = {
        "games": len(games),
        "positions": sum(row["positions"] for row in shards),
        "auxiliary_labels": list(auxiliary.LABELS),
        "reference_policy": reference_policy,
        "reference_log_probability": {"dtype": "float32", "shape": [150]},
        "shards": shards,
    }
    (output / "dataset.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reference", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            build(
                arguments.matches,
                arguments.output,
                arguments.workers,
                arguments.reference,
            )
        )
    )
