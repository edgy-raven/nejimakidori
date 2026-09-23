"""Write training TFRecords from a completed corpus database."""

import pathlib
import shutil
import sqlite3

import tensorflow as tf


def write(database, output):
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    shard = None
    writer = None
    for row_shard, example in connection.execute(
        "SELECT shard, example FROM records ORDER BY shard, game_id, kyoku_id, "
        "decision_index, actor"
    ):
        if row_shard != shard:
            if writer is not None:
                writer.close()
                pathlib.Path(str(path) + ".partial").replace(path)
            shard = row_shard
            path = output / f"train-{shard:05d}.tfrecord.gz"
            writer = tf.io.TFRecordWriter(
                str(path) + ".partial", options="GZIP"
            )
        writer.write(example)
    if writer is not None:
        writer.close()
        pathlib.Path(str(path) + ".partial").replace(path)
    connection.close()
    source_summary = pathlib.Path(database).with_name("dataset_summary.json")
    if source_summary.parent != output:
        shutil.copyfile(source_summary, output / "dataset_summary.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    write(arguments.database, arguments.output)
