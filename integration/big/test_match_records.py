"""Match payments share recorded action labels without hindsight rescoring."""

import gzip
import json
import pathlib

import numpy
import tensorflow as tf

from integration import reference_policy
from log_dataset import call_states, replay, scoring, training_data
from model import critic, outcomes
from training import auxiliary, match_records, restoration
from training.corrections import distributed


def test_match_records_preserve_actions_payments_and_every_position(
    tmp_path, monkeypatch
):
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
    expected_features, expected_labels, _ = training_data.parse_batch(
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
    policy = reference_policy.Policy()
    raw = tf.constant(list(match_records.game_records(path, policy)))
    actual = match_records.decode_records(raw)
    _, labels, _ = training_data.parse_batch(raw)
    for name in auxiliary.LABELS:
        numpy.testing.assert_array_equal(actual["labels"][name], labels[name])
        assert numpy.any(labels[name] >= 0)
    for value, reference in zip(
        tf.nest.flatten(actual["features"]),
        tf.nest.flatten(expected_features),
        strict=True,
    ):
        numpy.testing.assert_array_equal(value, reference)
    numpy.testing.assert_array_equal(
        actual["chosen"],
        critic.chosen_actions(expected_features, expected_labels),
    )
    heads, _, legal = restoration.policy_choices(
        expected_features, expected_labels
    )
    numpy.testing.assert_array_equal(actual["head"], heads)
    numpy.testing.assert_array_equal(actual["legal"], legal)
    state = replay.Replay(path.stem, events)
    assert policy.requests == [
        [state.event_index, row.actor, row.features["decision_phase"]]
        for row in state.observations()
    ]
    numpy.testing.assert_array_equal(
        actual["payment_points"],
        tf.gather(
            tf.reshape(expected_labels["terminal_payment"], [-1, 64]),
            numpy.ravel_multi_index(numpy.asarray(scoring.CELLS).T, (4, 4, 4)),
            axis=1,
        ),
    )
    numpy.testing.assert_array_equal(
        actual["reference_log_probability"],
        reference_policy.Policy().distribution(
            events, policy.requests, heads.numpy(), legal.numpy()
        ),
    )

    group = object.__new__(distributed.LearnerGroup)
    group.local_batch = 2
    for rank in (0, 6):
        group.rank = rank
        local = group.decode_records(raw[:3])
        global_rows = match_records.decode_records(raw[:3])
        packing = training_data.PackedBatch(
            call_states.base(global_rows["features"])
        )
        reference = group.localize(
            {
                **global_rows,
                "features": call_states.pack(global_rows["features"], packing),
            }
        )
        reference["features"] = call_states.unpack(
            reference["features"], packing
        )
        for left, right in zip(
            tf.nest.flatten(local), tf.nest.flatten(reference)
        ):
            numpy.testing.assert_array_equal(left, right)
        assert int(local["global_count"]) == 3
        assert int(local["local_count"]) == (2 if rank == 0 else 0)

    output = tmp_path / "data"
    output.mkdir()
    monkeypatch.setattr(match_records, "load_reference", lambda path: policy)
    shard = match_records.write_shard(
        (
            output / "train-0000.tfrecord.gz",
            [path],
            "fixture",
        )
    )
    count = len(actual["chosen"])
    assert shard["positions"] == count
    (output / "dataset.json").write_text(
        json.dumps(
            {"auxiliary_labels": list(auxiliary.LABELS), "shards": [shard]}
        )
    )
    batches = list(match_records.batches(output, count - 1, 941))
    assert [len(batch["chosen"]) for batch in batches] == [count - 1, 1]
    assert set(batches[0]) == {
        "features",
        "labels",
        "head",
        "legal",
        "chosen",
        "payment_points",
        "reference_log_probability",
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


def test_only_open_decisions_reach_the_optimizer(tmp_path):
    group = distributed.LearnerGroup()
    group._pack = lambda features: features
    weight = tf.Variable(1.0)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.1)

    def fit(batch, valid):
        with tf.GradientTape() as tape:
            # Each decision contributes a distinct gradient.
            loss = tf.reduce_sum(tf.where(valid, weight * batch["value"], 0.0))
        optimizer.apply_gradients([(tape.gradient(loss, weight), weight)])
        return {"loss": loss}

    group._updates = {"fit": group._make_update(fit, actor=False)}
    # Voluntary riichi, its pending and locked cuts, voluntary dama cut,
    # post-riichi WIN, singleton response, ordinary discard, padded row.
    batch = {
        "head": tf.constant([1, 0, 0, 0, 4, 2, 0, 0]),
        "features": {
            "riichi_state": tf.constant(
                [[i, 0, 0, 0] for i in (0, 1, 2, 0, 2, 0, 0, 0)]
            ),
            "riichi_legal_mask": tf.constant(
                [[1, i] for i in (1, 0, 0, 1, 0, 0, 0, 0)]
            ),
        },
        "legal": tf.sequence_mask([3, 2, 1, 2, 2, 1, 2, 2], 3),
        "value": tf.constant([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
        "local_count": tf.constant(7),
        "global_count": tf.constant(7),
    }
    metrics = group.update("fit", batch, tmp_path)
    assert metrics == {"loss": 89.0, "trained_positions": 4}
    snapshot = [v.numpy() for v in (weight, *optimizer.variables)]
    batch["legal"] = tf.sequence_mask([1] * 8, 3)
    assert group.update("fit", batch, tmp_path) == {"trained_positions": 0}
    for old, variable in zip(snapshot, [weight, *optimizer.variables]):
        numpy.testing.assert_array_equal(old, variable)
    (tmp_path / "STOP").touch()
    assert group.update("fit", batch, tmp_path) is None
