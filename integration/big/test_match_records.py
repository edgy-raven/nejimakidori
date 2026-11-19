"""Match payments share recorded action labels without hindsight rescoring."""

import gzip
import json
import pathlib

import numpy
import tensorflow as tf

from log_dataset import replay, scoring, training_data
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
    targets = scoring.round_labels(events)
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
    raw = tf.constant(list(match_records.game_records(path)))
    actual = match_records.decode_records(raw)
    _, labels, _ = training_data.parse_batch(raw)
    for name in ("opponent_shanten", "opponent_ukeire", "opponent_hand_counts"):
        numpy.testing.assert_array_equal(labels[name], -1)
    for value, reference in zip(
        tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True
    ):
        numpy.testing.assert_array_equal(value, reference)

    output = tmp_path / "data"
    output.mkdir()
    shard = match_records.write_shard(
        (
            output / "train-0000.tfrecord.gz",
            [path],
        )
    )
    count = len(actual["chosen"])
    assert shard["positions"] == count
    (output / "dataset.json").write_text(json.dumps({"shards": [shard]}))
    batches = list(match_records.batches(output, count - 1, 941))
    assert [len(batch["chosen"]) for batch in batches] == [count - 1, 1]
    assert set(batches[0]) == {
        "features",
        "head",
        "legal",
        "chosen",
        "payment_points",
    }
    numpy.testing.assert_array_equal(
        sorted(
            tuple(row)
            for batch in batches
            for row in numpy.column_stack(
                (batch["head"], batch["chosen"], batch["payment_points"])
            )
        ),
        sorted(
            tuple(row)
            for row in numpy.column_stack(
                (actual["head"], actual["chosen"], actual["payment_points"])
            )
        ),
    )

    observations = list(replay.Replay(path.stem, events).observations())
    assert {row.actor for row in observations} == {0, 1, 2, 3}
    assert count == len(observations)
    assert actual["legal"].shape == (count, 150)
