"""Streamable full-corpus critic records with same-hand scoring labels."""

import concurrent.futures
import contextlib
import gzip
import hashlib
import json
import multiprocessing
import pathlib
import shutil
import time
import zipfile

import numpy
import tensorflow as tf

from log_dataset import afk, rebuild, shanten

from . import critic, outcomes, payment_bonus, payment_scoring, records, replay

AUXILIARY = {
    "bonus": [24],
    "scenario": [24, len(payment_bonus.SCENARIOS)],
    "score": [24, len(payment_bonus.SCENARIOS)],
}


def serialize(observation, focused, target):
    example = tf.train.Example.FromString(
        records.serialize_observation(observation, focused)
    )
    example.features.feature["critic/partition"].int64_list.value[:] = [
        int.from_bytes(
            hashlib.sha256(observation.game_id.encode()).digest()[:4]
        )
        % 10
    ]
    order = (observation.actor + numpy.arange(4)) % 4
    cells = numpy.asarray(payment_bonus.CELLS[:24])
    numpy.testing.assert_array_equal(
        outcomes.PAYMENT_VALUES[observation.terminal_payment][:2],
        target["payment"][:2, order][:, :, order],
    )
    for name in AUXILIARY:
        value = target[name][:, order][:, :, order][
            cells[:, 0], cells[:, 1], cells[:, 2]
        ]
        example.features.feature["critic/" + name].bytes_list.value[:] = [
            value.astype("int16").tobytes()
        ]
    return example.SerializeToString()


def parse_batch(raw):
    features, labels, _ = records.parse_batch(raw)
    parsed = tf.io.parse_example(
        raw,
        {
            "critic/" + name: tf.io.FixedLenFeature([], tf.string)
            for name in AUXILIARY
        },
    )
    auxiliary = {
        name: tf.cast(
            tf.reshape(
                tf.io.decode_raw(parsed["critic/" + name], tf.int16),
                [-1, *shape],
            ),
            tf.int32,
        )
        for name, shape in AUXILIARY.items()
    }
    cells = numpy.asarray(payment_bonus.CELLS)
    payments = tf.gather(
        tf.reshape(labels["terminal_payment"], [-1, 64]),
        numpy.ravel_multi_index(cells.T, (4, 4, 4)),
        axis=1,
    )
    auxiliary.update(
        {
            **critic.public_inputs(features),
            "payment": tf.cast(payments, tf.int32),
            "points": tf.reduce_sum(
                tf.gather(
                    tf.constant(outcomes.PAYMENT_VALUES / 10000, tf.float32),
                    payments,
                )
                * tf.constant(
                    (cells[:, 2] == 0).astype(float)
                    - (cells[:, 1] == 0)
                    + ((cells[:, 0] == 3) & (cells[:, 1] == 0)),
                    tf.float32,
                ),
                axis=1,
            ),
        }
    )
    return features, labels, auxiliary


class PackedBatch:
    """Transfer one buffer per dtype, then restore the existing input tree."""

    def __init__(self, inputs):
        self.structure = tf.nest.map_structure(
            lambda value: tf.TensorSpec(value.shape[1:], value.dtype), inputs
        )
        self.specs = tf.nest.flatten(self.structure)
        self.groups = {
            dtype: [
                index
                for index, spec in enumerate(self.specs)
                if spec.dtype.name == dtype
            ]
            for dtype in sorted({spec.dtype.name for spec in self.specs})
        }

    def pack(self, *inputs):
        values = tf.nest.flatten(inputs)
        batch = tf.shape(values[0])[0]
        return {
            dtype: tf.concat(
                [
                    tf.reshape(
                        (
                            tf.bitcast(values[index], tf.float32)
                            if dtype == "int32"
                            else values[index]
                        ),
                        (batch, -1),
                    )
                    for index in indices
                ],
                axis=1,
            )
            for dtype, indices in self.groups.items()
        }

    def unpack(self, inputs):
        values = [None] * len(self.specs)
        for dtype, indices in self.groups.items():
            # int32 SplitV uses host memory. Bitcasting preserves every bit
            # while allowing the split itself to stay on the GPU.
            parts = tf.split(
                inputs[dtype],
                [int(numpy.prod(self.specs[index].shape)) for index in indices],
                axis=1,
            )
            for index, value in zip(indices, parts):
                values[index] = tf.reshape(
                    tf.bitcast(value, tf.int32) if dtype == "int32" else value,
                    (-1, *self.specs[index].shape),
                )
        return tf.nest.pack_sequence_as(self.structure, values)


def process_shard(task):
    index, root, games = task
    path = pathlib.Path(root) / f"train-{index:05d}.tfrecord.gz"
    summary_path = path.with_suffix(".json")
    if summary_path.exists():
        assert path.is_file()
        return json.loads(summary_path.read_text())
    if shutil.disk_usage(root).free < 15 * 1024**3:
        raise OSError("Corpus rebuild requires at least 15 GiB free space")
    counts = numpy.zeros((10, 2), numpy.int64)
    with contextlib.ExitStack() as stack:
        archives = {
            path: stack.enter_context(zipfile.ZipFile(path))
            for path in {game.archive for game in games}
        }
        writer = stack.enter_context(
            tf.io.TFRecordWriter(str(path) + ".partial", options="GZIP")
        )
        for game in games:
            raw = archives[game.archive].read(game.member)
            if raw.startswith(b"\x1f\x8b"):
                raw = gzip.decompress(raw)
            events = [json.loads(line) for line in raw.splitlines()]
            seats = events[0]["player_ranks"]["retained_seats"]
            if not seats:
                continue
            targets = payment_bonus.round_labels(events)
            partition = (
                int.from_bytes(
                    hashlib.sha256(game.game_id.encode()).digest()[:4]
                )
                % 10
            )
            hanchan = replay.hanchan_targets(events)
            for round_events in afk.retained_hands(events):
                state = replay.Replay(
                    game.game_id, round_events, hanchan=hanchan
                )
                for observation in state.observations():
                    if observation.actor not in seats:
                        continue
                    focused = rebuild.retain_focused(observation)
                    writer.write(
                        serialize(
                            observation, focused, targets[observation.kyoku_id]
                        )
                    )
                    counts[partition, 0] += 1
                    counts[partition, 1] += focused
    pathlib.Path(str(path) + ".partial").replace(path)
    summary = {
        "path": path.name,
        "games": len(games),
        "counts": counts.tolist(),
        "bytes": path.stat().st_size,
    }
    summary_path.write_text(json.dumps(summary))
    return summary


def dataset(root, sampling, partition, batch_size, repeat, seed=941):
    summary = json.loads(
        (pathlib.Path(root) / "dataset_summary.json").read_text()
    )
    assert summary["critic_scoring_states"] == [
        list(s) for s in payment_scoring.SCORES
    ]
    assert summary["critic_auxiliary"] == AUXILIARY
    records.validate_summary(root)
    paths = [
        str(pathlib.Path(root) / row["path"])
        for row in summary["shard_manifest"]
    ]
    stream = tf.data.Dataset.from_tensor_slices(paths)
    if repeat:
        stream = stream.shuffle(len(paths), seed=seed).repeat()
    stream = stream.interleave(
        lambda path: tf.data.TFRecordDataset(path, compression_type="GZIP"),
        cycle_length=min(8, len(paths)),
        num_parallel_calls=min(8, len(paths)),
        deterministic=True,
    )

    def select(raw):
        metadata = tf.io.parse_example(
            raw,
            {
                "critic/partition": tf.io.FixedLenFeature([], tf.int64),
                "meta/focused": tf.io.FixedLenFeature([], tf.int64),
            },
        )
        selected = (
            metadata["critic/partition"] < 7
            if partition == "train"
            else metadata["critic/partition"] == 7
        )
        return tf.boolean_mask(
            raw,
            selected
            & (metadata["meta/focused"] > 0 if sampling == "focused" else True),
        )

    stream = stream.batch(1024).map(select, num_parallel_calls=2).unbatch()
    if repeat:
        stream = stream.shuffle(4096, seed=seed)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 12
    return (
        stream.batch(batch_size, drop_remainder=True)
        .with_options(options)
        .prefetch(2)
    )


def run(args):
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    games, ranks = rebuild.archive_games(args.archives, args.extra_archives)
    excluded = {
        row["game_id"]
        for row in json.loads(pathlib.Path(args.reserve).read_text())["games"]
    }
    games = [game for game in games if game.game_id not in excluded]
    if args.limit_games:
        games = [
            games[index]
            for index in numpy.linspace(
                0, len(games) - 1, args.limit_games, dtype=int
            )
        ]
    plan = {
        "archives": str(pathlib.Path(args.archives).resolve()),
        "games": len(games),
        "excluded_reserve_games": sorted(excluded),
        "critic_auxiliary": AUXILIARY,
        "critic_scoring_states": [list(s) for s in payment_scoring.SCORES],
        "limit_games": args.limit_games,
        "player_ranks": ranks,
        **records.DATA_CONTRACT,
    }
    if (root / "build-plan.json").exists():
        assert json.loads((root / "build-plan.json").read_text()) == plan
    (root / "build-plan.json").write_text(json.dumps(plan, indent=2))
    (root / "games.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "game_id": game.game_id,
                    "archive": str(game.archive),
                    "member": game.member,
                }
            )
            + "\n"
            for game in games
        )
    )
    tasks = [
        (index, str(root), tuple(games[start : start + 32]))
        for index, start in enumerate(range(0, len(games), 32))
    ]
    shanten.native()
    writer = tf.summary.create_file_writer(str(root / "tensorboard"))
    started = time.monotonic()
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        for row in pool.map(process_shard, tasks):
            results.append(row)
            status = {
                "stage": "building",
                "shards": len(results),
                "total_shards": len(tasks),
                "rows": sum(sum(c[0] for c in r["counts"]) for r in results),
                "elapsed_seconds": time.monotonic() - started,
            }
            (root / "status.json").write_text(json.dumps(status, indent=2))
            with writer.as_default():
                for name in ("shards", "rows", "elapsed_seconds"):
                    tf.summary.scalar(
                        "rebuild/" + name, status[name], step=len(results)
                    )
            writer.flush()
            print(status, flush=True)
    (root / "dataset_summary.json").write_text(
        json.dumps(
            {
                **plan,
                "shard_manifest": results,
                "partition_counts": numpy.sum(
                    [row["counts"] for row in results], axis=0
                ).tolist(),
                "examples": sum(
                    sum(c[0] for c in row["counts"]) for row in results
                ),
            },
            indent=2,
        )
    )
    (root / "status.json").write_text(
        json.dumps({**status, "stage": "complete"})
    )
