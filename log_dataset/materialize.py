"""Stream normalized preprocessing components into training TFRecords."""

import concurrent.futures
import json
import pathlib
import time
import zlib

import tensorflow as tf

from . import cache


def write_shard(task):
    database, output, shard, first_hand, last_hand = task
    path = pathlib.Path(output) / f"train-{shard:05d}.tfrecord.gz"
    # Reuse the protobuf and its field objects throughout the ordered scan.
    example = tf.train.Example()
    fields = example.features.feature
    fields["meta/sample_weight"].float_list.value.append(1)
    for name in ("game_id", "kyoku_id"):
        fields["meta/" + name].bytes_list.value.append(b"")
    for name in ("decision_index", "actor", "hand_yaku_focus", "focused"):
        fields["meta/" + name].int64_list.value.append(0)
    count = 0
    with cache.connect(database) as connection, tf.io.TFRecordWriter(
        str(path) + ".partial", options="GZIP"
    ) as writer:
        for row in connection.execute(
            cache.export_query(), (first_hand, last_hand)
        ):
            fields["meta/game_id"].bytes_list.value[0] = row[1].encode()
            fields["meta/kyoku_id"].bytes_list.value[0] = row[2].encode()
            for name, value in zip(
                ("decision_index", "actor", "hand_yaku_focus", "focused"),
                row[3:7],
                strict=True,
            ):
                fields["meta/" + name].int64_list.value[0] = value
            writer.write(
                b"".join(zlib.decompress(raw) for raw in row[7:])
                + example.SerializeToString()
            )
            count += 1
    pathlib.Path(str(path) + ".partial").replace(path)
    return count


def write(database, output, workers=8):
    started = time.monotonic()
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with cache.connect(database) as connection:
        summary = cache.summary(connection)
        tasks = [
            (str(database), str(output), shard, first_hand, last_hand)
            for shard, first_hand, last_hand in connection.execute(
                "SELECT g.shard, MIN(h.id), MAX(h.id) FROM games AS g "
                "JOIN hands AS h ON h.game_id = g.id GROUP BY g.shard"
            )
        ]
    (output / "dataset_summary.json").unlink(missing_ok=True)
    if workers == 1 or len(tasks) <= 1:
        count = sum(map(write_shard, tasks))
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(tasks)),
        ) as executor:
            count = sum(executor.map(write_shard, tasks))
    if count != summary["examples"]:
        raise ValueError(
            f"export row count {count} != selection count "
            f"{summary['examples']}"
        )
    expected_paths = {f"train-{task[2]:05d}.tfrecord.gz" for task in tasks}
    for path in output.glob("train-*.tfrecord.gz"):
        if path.name not in expected_paths:
            path.unlink()
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    elapsed = time.monotonic() - started
    print(
        f"export examples={count} seconds={elapsed:.3f} "
        f"rows_per_second={count / elapsed:.1f}",
        flush=True,
    )
