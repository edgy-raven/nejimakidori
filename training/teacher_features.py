"""Align production inputs from retained records or original game archives."""

import argparse
import bisect
import concurrent.futures
import gzip
import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys
import time


def build_shard(task):
    job, source = task
    with job.with_suffix(".log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(
                    pathlib.Path(__file__).with_name(
                        "teacher_record_worker.py"
                        if "record_shards" in json.loads(job.read_text())
                        else "teacher_feature_worker.py"
                    )
                ),
                str(job),
            ],
            cwd=source,
            env={
                **os.environ,
                "PYTHONPATH": str(source),
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TF_NUM_INTEROP_THREADS": "1",
                "TF_NUM_INTRAOP_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return json.loads(job.with_suffix(".log").read_text().splitlines()[-1])


def retained_shards(directory, games, workers):
    """Locate shard intervals; workers verify every joined decision explicitly."""
    import tensorflow as tf

    positions = {game.encode(): index for index, game in enumerate(games)}

    def first_retained(path):
        with gzip.open(path, "rb") as stream:
            while header := stream.read(12):
                size = struct.unpack("<Q", header[:8])[0]
                row = tf.train.Example.FromString(
                    stream.read(size)
                ).features.feature
                stream.read(4)
                game = row["meta/game_id"].bytes_list.value[0]
                if game in positions:
                    return positions[game], str(path)
        return None

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        anchors = [
            value
            for value in pool.map(
                first_retained, sorted(directory.glob("train-*.tfrecord.gz"))
            )
            if value is not None
        ]
    starts = [row[0] for row in anchors]
    if not starts or starts[0] != 0 or starts != sorted(set(starts)):
        raise ValueError("Retained shard order does not match current games")
    return starts, [row[1] for row in anchors]


def run(args):
    import tensorflow as tf
    import tensorflow.core.protobuf.saved_model_pb2 as saved_model_pb2

    from log_dataset import _native_collection

    if args.workers < 1:
        raise ValueError("Workers must be positive")
    data = pathlib.Path(args.data).resolve()
    source = pathlib.Path(args.source).resolve()
    teacher = pathlib.Path(args.teacher).resolve()
    model = saved_model_pb2.SavedModel()
    model.ParseFromString((teacher / "saved_model.pb").read_bytes())
    fields = {
        name: {
            "shape": [dim.size for dim in spec.tensor_shape.dim][1:],
            "dtype": tf.as_dtype(spec.dtype).name,
        }
        for name, spec in model.meta_graphs[0]
        .signature_def["policy"]
        .inputs.items()
        if name not in _native_collection.INPUTS
    }
    summary = json.loads((data / "dataset_summary.json").read_text())
    games = {
        row["game_id"]: row
        for row in map(
            json.loads, (data / "games.jsonl").read_text().splitlines()
        )
    }
    root = pathlib.Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "jobs").mkdir()
    (root / "native").mkdir()
    plan = {
        "fields": fields,
        "data": str(data),
        "source": str(source),
        "dataset_sha256": hashlib.sha256(
            (data / "dataset_summary.json").read_bytes()
        ).hexdigest(),
        "teacher_files": {
            str(path.relative_to(teacher)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(teacher.rglob("*"))
            if path.is_file()
        },
        "source_files": {
            str(path.relative_to(source)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(source.rglob("*"))
            if path.suffix in (".py", ".cpp", ".hpp", ".h")
        },
        "complete": False,
    }
    (root / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    # Shards contain ordered groups of 32 games from the existing manifest.
    tasks = []
    counts = [None] * len(summary["shard_manifest"])
    if args.reconstructed:
        previous = json.loads(
            (pathlib.Path(args.reconstructed) / "manifest.json").read_text()
        )
        for name in (
            "fields",
            "dataset_sha256",
            "teacher_files",
            "source_files",
        ):
            if previous[name] != plan[name]:
                raise ValueError(f"Reconstructed companions differ: {name}")
        plan["reconstructed_companions"] = str(
            pathlib.Path(args.reconstructed).resolve()
        )
    offset = 0
    ordered = list(games.values())
    retained = None
    if args.records:
        retained = retained_shards(
            pathlib.Path(args.records), games, args.workers
        )
        plan["retained_records"] = str(pathlib.Path(args.records).resolve())
        plan["retained_dataset_sha256"] = hashlib.sha256(
            (pathlib.Path(args.records) / "dataset_summary.json").read_bytes()
        ).hexdigest()
        (root / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    for index, shard in enumerate(summary["shard_manifest"]):
        job = root / "jobs" / f"{index}.json"
        job.write_text(
            json.dumps(
                {
                    "native_library": str(root / "native" / "analysis.so"),
                    "input": str(data / shard["path"]),
                    "output": str(root / shard["path"]),
                    "fields": fields,
                    **(
                        {
                            "record_shards": retained[1][
                                bisect.bisect_right(retained[0], offset)
                                - 1 : bisect.bisect_left(
                                    retained[0], offset + shard["games"]
                                )
                            ]
                        }
                        if retained is not None
                        else {}
                    ),
                    "games": {
                        row["game_id"]: row
                        for row in ordered[offset : offset + shard["games"]]
                    },
                }
            )
        )
        offset += shard["games"]
        if (
            args.reconstructed
            and (pathlib.Path(args.reconstructed) / shard["path"]).is_file()
        ):
            previous_counts = json.loads(
                (pathlib.Path(args.reconstructed) / "jobs" / f"{index}.log")
                .read_text()
                .splitlines()[-1]
            )
            os.link(
                pathlib.Path(args.reconstructed) / shard["path"],
                root / shard["path"],
            )
            counts[index] = {
                **previous_counts,
                "copied_records": 0,
                "reconstructed_records": previous_counts["teacher_records"],
            }
        else:
            tasks.append((index, (job, source)))
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        pending = {
            pool.submit(build_shard, task): index for index, task in tasks
        }
        started = time.monotonic()
        completed = reused = sum(row is not None for row in counts)
        prepared = reused_records = sum(
            row["records"] for row in counts if row is not None
        )
        teacher_rows = reconstructed = sum(
            row["teacher_records"] for row in counts if row is not None
        )
        copied = 0
        writer = tf.summary.create_file_writer(str(root / "tensorboard"))

        def report():
            elapsed = time.monotonic() - started
            status = {
                "stage": (
                    "complete" if completed == len(counts) else "preparing"
                ),
                "completed_shards": completed,
                "total_shards": len(counts),
                "reused_shards": reused,
                "prepared_records": prepared,
                "total_records": summary["examples"],
                "teacher_records": teacher_rows,
                "copied_records": copied,
                "reconstructed_records": reconstructed,
                "elapsed_seconds": elapsed,
                "updated_at": time.time(),
                "fraction": completed / len(counts),
                "records_per_second": (prepared - reused_records) / elapsed,
            }
            if completed - reused >= args.workers:
                status["eta_seconds"] = (
                    elapsed * (len(counts) - completed) / (completed - reused)
                )
            temporary = root / "status.partial"
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(root / "status.json")
            with writer.as_default(step=completed):
                for name in (
                    "fraction",
                    "prepared_records",
                    "records_per_second",
                ):
                    tf.summary.scalar("preparation/" + name, status[name])
            writer.flush()

        report()
        try:
            for future in concurrent.futures.as_completed(pending):
                counts[pending[future]] = future.result()
                completed += 1
                prepared += counts[pending[future]]["records"]
                teacher_rows += counts[pending[future]]["teacher_records"]
                copied += counts[pending[future]]["copied_records"]
                reconstructed += counts[pending[future]][
                    "reconstructed_records"
                ]
                report()
        finally:
            # Do not continue the entire corpus after one reconstruction fails.
            for future in pending:
                future.cancel()
            writer.close()
    if sum(row["records"] for row in counts) != summary["examples"]:
        raise ValueError("Teacher companion count differs from source corpus")
    plan.update(complete=True, shards=counts)
    (root / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "source", "teacher", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--records", help="Retained production TFRecord corpus")
    parser.add_argument(
        "--reconstructed",
        help="Reuse completed archive-reconstructed companions",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
