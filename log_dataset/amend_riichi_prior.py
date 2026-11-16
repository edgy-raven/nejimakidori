"""Add exact expert-prior facts to existing records without replaying games."""

import argparse
import concurrent.futures
import json
import math
import pathlib
import shutil
import time

import numpy
import tensorflow as tf
from log_dataset import _native_collection, focus_sampling, records

FIELDS = (
    "policy_prior_no_threat",
    "riichi_dama_points",
    "riichi_extra_han",
    "call_open_points",
    "call_riichi_points",
)


def prior_values(features, threatened):
    result = {
        name: numpy.zeros(
            _native_collection.INPUTS[name]["shape"],
            _native_collection.INPUTS[name]["dtype"],
        )
        for name in FIELDS
    }
    if int(features["decision_phase"]) not in (1, 3):
        return result
    result["policy_prior_no_threat"][...] = not threatened
    choices = (
        [
            (cut, int(features["discard_candidate_hand"][cut]))
            for cut in numpy.flatnonzero(features["riichi_discard_legal_mask"])
        ]
        if int(features["decision_phase"]) == 3
        else [
            (
                option * 37 + cut + offset,
                int(features[kind + "_candidate_hand"][option, cut]),
            )
            for kind, offset in (("chi", 0), ("pon", 111))
            for option, cut in numpy.argwhere(
                (features[kind + "_discard_legal"] > 0)
                & (features[kind + "_discard_shanten"] == 1)
            )
        ]
    )
    for cut, hand in choices:
        tiles = []
        for base, count in enumerate(features["candidate_hand_counts"][hand]):
            red = (
                base in (4, 13, 22)
                and features["candidate_hand_red"][hand, base // 9] > 0
            )
            first = int(base in (4, 13, 22) and not red)
            tiles.extend(
                base * 4 + copy for copy in range(first, first + count)
            )
        melds = []
        reds = features["candidate_melds"][hand, [4, 13, 22], 5].copy()
        for base, kind in numpy.argwhere(
            features["candidate_melds"][hand, :, :5] > 0
        ):
            for _ in range(int(features["candidate_melds"][hand, base, kind])):
                bases = (
                    range(base, base + 3)
                    if kind == 0
                    else [base] * (3 if kind == 1 else 4)
                )
                physical = []
                for tile in bases:
                    value = int(tile) * 4
                    if tile in (4, 13, 22):
                        if reds[tile // 9]:
                            reds[tile // 9] -= 1
                        else:
                            value += 1
                    while value in physical:
                        value += 1
                    physical.append(value)
                melds.append((int(kind), physical))
        points, extra, riichi = _native_collection.policy_prior_values(
            tiles=tiles,
            melds=melds,
            dora=(
                features["dora_tile_id"][features["dora_valid"] > 0].astype(
                    numpy.int32
                )
                * 4
            ).tolist(),
            unavailable=features["known_unavailable_counts"].tolist(),
            unavailable_red=(
                features["public_visible_red_counts"]
                + features["candidate_hand_red"][0]
            ).tolist(),
            player_wind=int(features["seat_wind"][0]),
            round_wind=int(features["prevailing_wind"]),
        )
        if int(features["decision_phase"]) == 3:
            result["riichi_dama_points"][cut] = points
            result["riichi_extra_han"][cut] = extra
        else:
            result["call_open_points"].reshape(-1)[cut] = points
            result["call_riichi_points"].reshape(-1)[cut] = riichi
    return result


def amend(raw):
    example = tf.train.Example.FromString(raw)
    needed = (
        "decision_phase",
        "riichi_discard_legal_mask",
        "discard_candidate_hand",
        "candidate_hand_counts",
        "candidate_hand_red",
        "candidate_melds",
        "chi_candidate_hand",
        "pon_candidate_hand",
        "chi_discard_legal",
        "pon_discard_legal",
        "chi_discard_shanten",
        "pon_discard_shanten",
        "dora_tile_id",
        "dora_valid",
        "known_unavailable_counts",
        "public_visible_red_counts",
        "seat_wind",
        "prevailing_wind",
    )
    phase = int.from_bytes(
        example.features.feature["feature/decision_phase"].bytes_list.value[0],
        signed=True,
    )
    needed = needed if phase in (1, 3) else ("decision_phase",)
    assert all("feature/" + name in example.features.feature for name in needed)
    assert all(
        "feature/" + name not in example.features.feature for name in FIELDS
    )
    features = {}
    for name in needed:
        spec = _native_collection.INPUTS[name]
        dtype = numpy.dtype(spec["dtype"])
        data = example.features.feature["feature/" + name].bytes_list.value[0]
        features[name] = numpy.frombuffer(
            data.ljust(math.prod(spec["shape"]) * dtype.itemsize, b"\0"), dtype
        ).reshape(spec["shape"])
    values = prior_values(
        features,
        bool(
            example.features.feature["meta/focus_categories"].int64_list.value[
                0
            ]
            & int(focus_sampling.Category.THREAT)
        ),
    )
    for name, value in values.items():
        example.features.feature["feature/" + name].bytes_list.value.append(
            records.encode("feature/" + name, value)
        )
    return example.SerializeToString(), int(features["decision_phase"]) == 3


def amend_shard(task):
    source, destination = map(pathlib.Path, task)
    counts = {"rows": 0, "riichi_rows": 0}
    metadata = json.loads(source.with_suffix(".json").read_text())
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    for prefix in ("train", "validation"):
        name = source.name.replace("train-", prefix + "-")
        temporary = destination / (name + ".tmp")
        with tf.io.TFRecordWriter(str(temporary), options="GZIP") as writer:
            for raw in tf.data.TFRecordDataset(
                str(source.parent / name), compression_type="GZIP"
            ).with_options(options):
                amended, riichi = amend(raw.numpy())
                writer.write(amended)
                counts["rows"] += 1
                counts["riichi_rows"] += riichi
        temporary.replace(destination / name)
        metadata["validation_bytes" if prefix == "validation" else "bytes"] = (
            (destination / name).stat().st_size
        )
    (destination / source.with_suffix(".json").name).write_text(
        json.dumps(metadata, indent=2)
    )
    return counts


def run(args):
    started = time.monotonic()
    assert args.source.resolve() != args.output.resolve()
    args.output.mkdir(exist_ok=args.replace, parents=True)
    shards = sorted(args.source.glob("train-*.tfrecord.gz"))
    selected = shards[: args.limit]
    progress = {
        "source": str(args.source),
        "fields": FIELDS,
        "shards": len(selected),
        "completed_shards": 0,
        "rows": 0,
        "riichi_rows": 0,
        "complete": False,
    }
    (args.output / "amendment.json").write_text(json.dumps(progress, indent=2))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as pool:
        for result in pool.map(
            amend_shard, [(str(path), str(args.output)) for path in selected]
        ):
            progress["completed_shards"] += 1
            progress["rows"] += result["rows"]
            progress["riichi_rows"] += result["riichi_rows"]
            progress["elapsed_seconds"] = time.monotonic() - started
            (args.output / "amendment.json").write_text(
                json.dumps(progress, indent=2)
            )
            print(json.dumps(progress), flush=True)
    if len(selected) == len(shards):
        for name in ("games.jsonl", "build-plan.json"):
            shutil.copy2(args.source / name, args.output / name)
        summary = json.loads((args.source / "dataset_summary.json").read_text())
        summary["input_schema"] = _native_collection.INPUTS
        summary["shard_manifest"] = [
            json.loads(
                (args.output / path.with_suffix(".json").name).read_text()
            )
            for path in selected
        ]
        summary["policy_prior"] = {
            "threat": "existing native visible_threat / focus THREAT bit",
            "value": "minimum live-wait non-dealer dama/open ron, without honba",
            "call": "post-call open and hypothetical closed-riichi values",
            "extra_han": "minimum han beyond riichi, including visible/red dora",
            "source": str(args.source),
        }
        (args.output / "dataset_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
    progress["complete"] = True
    (args.output / "amendment.json").write_text(json.dumps(progress, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--replace", action="store_true")
    run(parser.parse_args())
