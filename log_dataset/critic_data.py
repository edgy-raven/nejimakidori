"""Streamable full-corpus critic records with same-hand scoring labels."""

import concurrent.futures
import contextlib
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import pathlib
import shutil
import time

import numpy
import tensorflow as tf
from log_dataset import (
    afk,
    archives,
    focus_sampling,
    payment_bonus,
    records,
    replay,
    riichilab,
    shanten,
)
from model import critic, joint_settlement, payment_scoring

AUXILIARY = {
    "bonus": [24],
    "scenario": [24, len(payment_bonus.SCENARIOS)],
    "score": [24, len(payment_bonus.SCENARIOS)],
}


VALIDATION_SAMPLE = {
    "partition": 7,
    "decision_hash_modulus": 16,
    "hash": "sha256:validation-941:game_id:kyoku_id:decision_index:actor",
}


def serialize(observation, target):
    category = focus_sampling.categories(observation.features)
    example = tf.train.Example.FromString(
        records.serialize_observation(observation, category > 0)
    )
    example.features.feature["meta/focus_categories"].int64_list.value[:] = [
        category
    ]
    example.features.feature["critic/partition"].int64_list.value[:] = [
        records.game_partition(observation.game_id)
    ]
    order = (observation.actor + numpy.arange(4)) % 4
    cells = numpy.asarray(payment_bonus.CELLS[:24])
    numpy.testing.assert_array_equal(
        observation.terminal_payment[:2],
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


def score_targets(labels):
    """Compact equivalent scenario labels before transferring a GPU batch.

    Equal added-han/open/yakuman/ura-tail groups score the same physical
    completion identically. Counts retain repeated-scenario loss weights.
    """
    labels = tf.transpose(labels, [2, 0, 1])
    groups = tf.constant(payment_bonus.SCORE_GROUPS)
    return {
        "score": tf.transpose(
            tf.math.unsorted_segment_max(
                labels, groups, len(payment_bonus.SCORE_SCENARIOS)
            ),
            [1, 2, 0],
        ),
        "score_weight": tf.transpose(
            tf.math.unsorted_segment_sum(
                tf.cast(labels >= 0, tf.float32),
                groups,
                len(payment_bonus.SCORE_SCENARIOS),
            ),
            [1, 2, 0],
        ),
    }


def parse_batch(raw):
    features, labels, _ = records.parse_batch(raw)
    parsed = tf.io.parse_example(
        raw,
        {
            "critic/score": tf.io.FixedLenFeature([], tf.string),
            "meta/focused": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    auxiliary = score_targets(
        tf.cast(
            tf.reshape(
                tf.io.decode_raw(parsed["critic/score"], tf.int16),
                [-1, *AUXILIARY["score"]],
            ),
            tf.int32,
        )
    )
    cells = numpy.asarray(payment_bonus.CELLS)
    payments = tf.gather(
        tf.reshape(labels["terminal_payment"], [-1, 64]),
        numpy.ravel_multi_index(cells.T, (4, 4, 4)),
        axis=1,
    )
    public = joint_settlement.public_inputs(features)
    values = joint_settlement.payment_values(public)
    matches = tf.cast(payments[..., None], tf.float32) == values[:, None]
    tf.debugging.assert_equal(tf.reduce_all(tf.reduce_any(matches, -1)), True)
    auxiliary.update(
        {
            **critic.public_inputs(features),
            **joint_settlement.public_inputs(features),
            "focused": tf.cast(parsed["meta/focused"], tf.int32),
            "payment": tf.argmax(
                tf.cast(matches, tf.int32), -1, output_type=tf.int32
            ),
            "round_placement": tf.cast(labels["round_placement"], tf.int32),
            "points": tf.reduce_sum(
                tf.cast(payments, tf.float32)
                / 10000
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
        packed = {}
        for dtype, indices in self.groups.items():
            parts = []
            for index in indices:
                width = int(numpy.prod(self.specs[index].shape))
                value = tf.reshape(values[index], (batch, width))
                if dtype in ("bool", "int8", "int16", "int32"):
                    slots = 4 // self.specs[index].dtype.size
                    if dtype == "bool":
                        value = tf.cast(value, tf.uint8)
                    if slots > 1:
                        value = tf.reshape(
                            tf.pad(value, ((0, 0), (0, (-width) % slots))),
                            (batch, -1, slots),
                        )
                    value = tf.bitcast(value, tf.float32)
                parts.append(value)
            packed[dtype] = tf.concat(parts, axis=1)
        return packed

    def unpack(self, inputs):
        values = [None] * len(self.specs)
        for dtype, indices in self.groups.items():
            # Narrow integer/bool SplitV kernels run on the host. Transfer
            # bit patterns in float32 buffers so splitting stays on device.
            slots = (
                4 // self.specs[indices[0]].dtype.size
                if dtype in ("bool", "int8", "int16", "int32")
                else 1
            )
            parts = tf.split(
                inputs[dtype],
                [
                    (int(numpy.prod(self.specs[index].shape)) + slots - 1)
                    // slots
                    for index in indices
                ],
                axis=1,
            )
            for index, value in zip(indices, parts):
                if dtype in ("bool", "int8", "int16", "int32"):
                    value = tf.bitcast(
                        value, tf.uint8 if dtype == "bool" else dtype
                    )
                    value = tf.reshape(value, (tf.shape(value)[0], -1))
                    value = value[:, : int(numpy.prod(self.specs[index].shape))]
                    if dtype == "bool":
                        value = tf.cast(value, tf.bool)
                values[index] = tf.reshape(
                    value, (-1, *self.specs[index].shape)
                )
        return tf.nest.pack_sequence_as(self.structure, values)


def process_shard(task):
    index, root, games = task
    path = pathlib.Path(root) / f"train-{index:05d}.tfrecord.gz"
    validation_path = pathlib.Path(root) / f"validation-{index:05d}.tfrecord.gz"
    summary_path = path.with_suffix(".json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if (
            path.stat().st_size != summary["bytes"]
            or validation_path.stat().st_size != summary["validation_bytes"]
        ):
            raise ValueError(f"Incomplete dataset shard: {path}")
        return summary
    if shutil.disk_usage(root).free < 15 * 1024**3:
        raise OSError("Corpus rebuild requires at least 15 GiB free space")
    validation_count = 0
    counts = numpy.zeros((10, 2), numpy.int64)
    focus_counts = numpy.zeros((10, 64), numpy.int64)
    with contextlib.ExitStack() as stack:
        source_archives = {
            path: archives.open_archive(path)
            for path in {game.archive for game in games}
        }
        writer = stack.enter_context(
            tf.io.TFRecordWriter(str(path) + ".partial", options="GZIP")
        )
        validation_writer = stack.enter_context(
            tf.io.TFRecordWriter(
                str(validation_path) + ".partial", options="GZIP"
            )
        )
        for game in games:
            raw = source_archives[game.archive].read(game.member)
            if raw.startswith(b"\x1f\x8b"):
                raw = gzip.decompress(raw)
            events = [json.loads(line) for line in raw.splitlines()]
            if "riichilab" in events[0]:
                riichilab.normalize_draw_payments(events)
            seats = events[0]["player_ranks"]["retained_seats"]
            if not seats:
                continue
            targets = payment_bonus.round_labels(events)
            partition = records.game_partition(game.game_id)
            hanchan = replay.hanchan_targets(events)
            for round_events in afk.retained_hands(events):
                state = replay.Replay(
                    game.game_id, round_events, hanchan=hanchan
                )
                for observation in state.observations(actors=seats):
                    raw = serialize(observation, targets[observation.kyoku_id])
                    category = (
                        tf.train.Example.FromString(raw)
                        .features.feature["meta/focus_categories"]
                        .int64_list.value[0]
                    )
                    focus_counts[partition, category] += 1
                    writer.write(raw)
                    if partition == VALIDATION_SAMPLE["partition"] and (
                        int.from_bytes(
                            hashlib.sha256(
                                (
                                    f"validation-941:{observation.game_id}:"
                                    f"{observation.kyoku_id}:"
                                    f"{observation.decision_index}:"
                                    f"{observation.actor}"
                                ).encode()
                            ).digest()[:8]
                        )
                        % VALIDATION_SAMPLE["decision_hash_modulus"]
                        == 0
                    ):
                        validation_writer.write(raw)
                        validation_count += 1
                    counts[partition, 0] += 1
                    counts[partition, 1] += category > 0
    pathlib.Path(str(path) + ".partial").replace(path)
    pathlib.Path(str(validation_path) + ".partial").replace(validation_path)
    summary = {
        "path": path.name,
        "validation_path": validation_path.name,
        "validation_count": validation_count,
        "validation_bytes": validation_path.stat().st_size,
        "games": len(games),
        "counts": counts.tolist(),
        "focus_counts": focus_counts.tolist(),
        "bytes": path.stat().st_size,
    }
    write_json(summary_path, summary)
    return summary


def dataset(root, sampling, partition, batch_size, repeat, seed=941):
    if sampling == "mixed":
        assert batch_size % 4 == 0
        streams = [
            dataset(
                root=root,
                sampling=kind,
                partition=partition,
                batch_size=batch_size * share // 4,
                repeat=repeat,
                seed=seed + index,
            )
            for index, (kind, share) in enumerate(
                (("natural", 3), ("focused", 1))
            )
        ]
        return (
            tf.data.Dataset.zip(tuple(streams))
            .map(
                lambda natural, focused: tf.reshape(
                    tf.concat(
                        [tf.reshape(natural, [-1, 3]), focused[:, None]], axis=1
                    ),
                    [batch_size],
                ),
                num_parallel_calls=1,
            )
            .prefetch(2)
        )
    summary = json.loads(
        (pathlib.Path(root) / "dataset_summary.json").read_text()
    )
    assert summary["critic_scoring_states"] == [
        list(s) for s in payment_scoring.SCORES
    ]
    assert summary["critic_auxiliary"] == AUXILIARY
    records.validate_summary(root)
    assert summary["critic_focus_sampling"] == focus_sampling.CONTRACT
    assert summary["validation_sample"] == VALIDATION_SAMPLE
    paths = [
        str(
            pathlib.Path(root)
            / row["validation_path" if partition == "validation" else "path"]
        )
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
                "meta/focus_categories": tf.io.FixedLenFeature([], tf.int64),
            },
        )
        selected = (
            metadata["critic/partition"] < 7
            if partition == "train"
            else metadata["critic/partition"] == 7
        )
        if sampling == "focused":
            return tf.repeat(
                raw,
                tf.where(
                    selected,
                    tf.cast(
                        tf.gather(
                            tf.constant(focus_sampling.PRIORITIES),
                            metadata["meta/focus_categories"],
                        ),
                        tf.int32,
                    ),
                    0,
                ),
            )
        return tf.boolean_mask(raw, selected)

    stream = stream.batch(1024).map(select, num_parallel_calls=2).unbatch()
    if repeat:
        stream = stream.shuffle(4096, seed=seed)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 12
    return (
        stream.batch(batch_size, drop_remainder=partition != "validation")
        .with_options(options)
        .prefetch(2)
    )


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(args):
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        games, ranks = archives.games(args.archives, args.extra_archives)
        excluded = {
            row["game_id"]
            for row in json.loads(pathlib.Path(args.reserve).read_text())[
                "games"
            ]
        }
        games = [game for game in games if game.game_id not in excluded]
        if not games:
            raise ValueError(
                "No eligible training games after reserve exclusion"
            )
        if args.limit_games and args.limit_games < len(games):
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
            "critic_focus_sampling": focus_sampling.CONTRACT,
            "validation_sample": VALIDATION_SAMPLE,
            "critic_scoring_states": [list(s) for s in payment_scoring.SCORES],
            "limit_games": args.limit_games,
            "player_ranks": ranks,
            **records.DATA_CONTRACT,
        }
        if (root / "build-plan.json").exists():
            if json.loads((root / "build-plan.json").read_text()) != plan:
                raise ValueError("Dataset inputs changed; use a fresh output")
        write_json(root / "build-plan.json", plan)
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
                    "rows": sum(
                        sum(c[0] for c in r["counts"]) for r in results
                    ),
                    "elapsed_seconds": time.monotonic() - started,
                }
                write_json(root / "status.json", status)
                with writer.as_default():
                    for name in ("shards", "rows", "elapsed_seconds"):
                        tf.summary.scalar(
                            "rebuild/" + name, status[name], step=len(results)
                        )
                writer.flush()
                print(status, flush=True)
        focus_counts = numpy.sum(
            [row["focus_counts"] for row in results], axis=0
        )
        category_counts = numpy.array(
            [
                focus_counts[:7, numpy.arange(64) & (1 << bit) > 0].sum()
                for bit in range(6)
            ]
        )
        category_mass = category_counts * focus_sampling.EMPHASIS
        write_json(
            root / "dataset_summary.json",
            {
                **plan,
                "shard_manifest": results,
                "focus_counts": focus_counts.tolist(),
                "focus_category_shares": dict(
                    zip(
                        focus_sampling.NAMES,
                        (category_mass / category_mass.sum()).tolist(),
                    )
                ),
                "partition_counts": numpy.sum(
                    [row["counts"] for row in results], axis=0
                ).tolist(),
                "validation_examples": sum(
                    row["validation_count"] for row in results
                ),
                "examples": sum(
                    sum(c[0] for c in row["counts"]) for row in results
                ),
            },
        )
        write_json(root / "status.json", {**status, "stage": "complete"})
