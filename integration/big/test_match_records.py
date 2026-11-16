"""Match payments share recorded action labels without hindsight rescoring."""

import gzip
import json
import pathlib

import numpy
import tensorflow as tf
from log_dataset import replay, training_data
from model import outcomes
from training import match_records


def test_match_records_preserve_actions_payments_and_every_position(tmp_path):
    events = json.loads(
        pathlib.Path(__file__)
        .parents[1]
        .joinpath("fixtures/model/pass_speed_round.json")
        .read_text()
    ) + [{"type": "end_game"}]
    outcomes.normalize_draw_payments(events)
    path = tmp_path / "match.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.writelines(json.dumps(event) + "\n" for event in events)
    targets = training_data.game_targets(events)
    expected = match_records.decode_records(
        tf.constant(
            [
                training_data.serialize(
                    observation, targets[observation.kyoku_id]
                )
                for observation in replay.Replay(
                    path.stem, events
                ).observations()
            ]
        )
    )
    actual = match_records.decode_records(
        tf.constant(list(match_records.game_records(path)))
    )
    for value, reference in zip(
        tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True
    ):
        numpy.testing.assert_array_equal(value, reference)

    output = tmp_path / "data"
    output.mkdir()
    shard = match_records.write_shard(
        (output / "train-0000.tfrecord.gz", [path])
    )
    count = len(actual["chosen"])
    assert shard["positions"] == count
    (output / "dataset.json").write_text(json.dumps({"shards": [shard]}))
    batches = list(match_records.batches(output, count - 1, 941))
    assert [len(batch["chosen"]) for batch in batches] == [count - 1, 1]
    numpy.testing.assert_array_equal(
        numpy.sort(
            numpy.concatenate([batch["projected_st3"] for batch in batches])
        ),
        numpy.sort(actual["projected_st3"]),
    )
