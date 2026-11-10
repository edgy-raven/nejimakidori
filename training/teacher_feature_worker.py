"""Run only through teacher_features, with the frozen teacher source on PYTHONPATH."""

import gzip
import json
import pathlib
import sys

import numpy
import tensorflow as tf

from log_dataset import afk, archives, records, replay, shanten
from model import feature_vector, features


def reconstruct(job, metadata, wanted):
    results = {}
    for game_id, positions in wanted.items():
        game = job["games"][game_id]
        contents = archives.open_archive(game["archive"]).read(game["member"])
        if contents.startswith(b"\x1f\x8b"):
            contents = gzip.decompress(contents)
        events = [json.loads(line) for line in contents.splitlines()]
        for hand in afk.retained_hands(events):
            state = replay.Replay(game_id, hand)
            for _, actor, phase, _, _ in state.decisions():
                key = (state.kyoku_id, state.decision_index, actor)
                if key not in positions:
                    continue
                packed = features.pack(
                    state,
                    actor,
                    phase,
                    state.legal_observation(actor),
                    stage="analysis",
                )
                features.candidate_features(
                    packed,
                    state,
                    actor,
                    state.legal_observation(actor),
                    stage="scoring",
                )
                row = positions[key]
                for name in (
                    "decision_phase",
                    "live_wall_count",
                    "candidate_hand_counts",
                ):
                    expected = records.decode(
                        "feature/" + name,
                        metadata[row]["feature/" + name].bytes_list.value[0],
                        feature_vector.INPUTS[name],
                    )
                    numpy.testing.assert_array_equal(
                        (
                            packed[name][0]
                            if name == "candidate_hand_counts"
                            else packed[name]
                        ),
                        (
                            expected[0]
                            if name == "candidate_hand_counts"
                            else expected
                        ),
                        err_msg=str((game_id, key, name)),
                    )
                results[row] = {
                    name: numpy.asarray(
                        packed[name], dtype=spec["dtype"]
                    ).tobytes()
                    for name, spec in job["fields"].items()
                }
        assert all(row in results for row in positions.values()), (
            game_id,
            "Unmatched teacher decisions",
            [key for key, row in positions.items() if row not in results],
        )
    return results


def run():
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    shanten.LIBRARY = pathlib.Path(job["native_library"])
    raw = list(
        tf.data.TFRecordDataset(
            job["input"], compression_type="GZIP"
        ).as_numpy_iterator()
    )
    fingerprints = tf.raw_ops.Fingerprint(
        data=tf.constant(raw), method="farmhash64"
    ).numpy()
    metadata = [
        tf.train.Example.FromString(value).features.feature for value in raw
    ]
    wanted = {}
    for row, value in enumerate(metadata):
        if value["critic/partition"].int64_list.value[0] >= 7:
            continue
        key = (
            value["meta/kyoku_id"].bytes_list.value[0].decode(),
            value["meta/decision_index"].int64_list.value[0],
            value["meta/actor"].int64_list.value[0],
        )
        game = value["meta/game_id"].bytes_list.value[0].decode()
        wanted.setdefault(game, {})[key] = row
    results = reconstruct(job, metadata, wanted)
    output = pathlib.Path(job["output"])
    with tf.io.TFRecordWriter(
        str(output) + ".partial", options="GZIP"
    ) as writer:
        for row in range(len(raw)):
            example = tf.train.Example()
            example.features.feature["fingerprint"].bytes_list.value.append(
                fingerprints[row].tobytes()
            )
            for name in job["fields"]:
                example.features.feature[name].bytes_list.value.append(
                    results[row][name] if row in results else b""
                )
            writer.write(example.SerializeToString())
    pathlib.Path(str(output) + ".partial").replace(output)
    print(
        json.dumps(
            {
                "records": len(raw),
                "teacher_records": len(results),
                "copied_records": 0,
                "reconstructed_records": len(results),
            }
        )
    )


if __name__ == "__main__":
    run()
