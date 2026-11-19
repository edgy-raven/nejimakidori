"""Record sampling, retained-game routing and lossless batch transport."""

import json
import pathlib

import numpy
import riichienv
import tensorflow as tf

from log_dataset import (
    _native_collection,
    afk,
    features,
    records,
    replay,
    training_data,
)
from model import payment_scoring


def test_retained_players_skip_work_without_changing_decisions_or_labels():
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/sakigiri_swap.json")
        .read_text()
    )
    rows = list(replay.Replay("retained", events).observations())
    actor = rows[0].actor

    def retained_features(state, actor, phase, **kwargs):
        assert actor == rows[0].actor
        return features.pack(state=state, actor=actor, phase=phase, **kwargs)

    retained = list(
        replay.Replay("retained", events).observations(
            retained_features, actors=(actor,)
        )
    )
    assert 0 < len(retained) < len(rows)
    assert [records.serialize_observation(row) for row in retained] == [
        records.serialize_observation(row) for row in rows if row.actor == actor
    ]


def test_critic_filter_preserves_partition_order_and_partial_metadata_batch(
    tmp_path,
):
    rows = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    name: tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[value])
                    )
                    for name, value in {
                        "critic/partition": index % 10,
                        "test/index": index,
                    }.items()
                }
            )
        ).SerializeToString()
        for index in range(2053)
    ]
    with tf.io.TFRecordWriter(
        str(tmp_path / "train-00000.tfrecord.gz"), options="GZIP"
    ) as writer:
        for row in rows:
            writer.write(row)
    (tmp_path / "dataset_summary.json").write_text(
        json.dumps(
            {
                "input_schema": _native_collection.INPUTS,
                "label_schema": records.LABELS,
                "critic_auxiliary": training_data.AUXILIARY,
                "critic_bonus_han": list(range(14)),
                "critic_scoring_states": [
                    list(s) for s in payment_scoring.SCORES
                ],
                "validation_sample": training_data.VALIDATION_SAMPLE,
                "shard_manifest": [
                    {
                        "path": "train-00000.tfrecord.gz",
                        "validation_path": "train-00000.tfrecord.gz",
                    }
                ],
            }
        )
    )
    for partition in ("train", "validation"):
        batches = list(training_data.dataset(tmp_path, partition, 4))
        actual = [bytes(row.numpy()) for batch in batches for row in batch]
        expected = [
            row
            for index, row in enumerate(rows)
            if (index % 10 < 7 if partition == "train" else index % 10 == 7)
        ]
        assert actual == expected
        assert len(batches[-1]) == len(expected) % 4


def all_tsumogiri_round(seed):
    environment = riichienv.RiichiEnv(
        seed=seed, rule=riichienv.GameRule.default_tenhou()
    )
    observations = environment.reset()
    while not environment.done():
        choices = {}
        for seat, observation in observations.items():
            legal = observation.legal_actions()
            choices[seat] = next(
                (
                    action
                    for action in legal
                    if action.action_type == riichienv.ActionType.DISCARD
                    and action.tile == observation.drawn_tile
                ),
                next(
                    (
                        action
                        for action in legal
                        if action.action_type == riichienv.ActionType.PASS
                    ),
                    legal[0],
                ),
            )
        observations = environment.step(choices)
    return environment.mjai_log[1:-1]


def test_afk_filter_removes_every_seat_from_two_tsumogiri_hands():
    first = all_tsumogiri_round(1)
    second = all_tsumogiri_round(2)

    retained = afk.retained_hands(
        [{"type": "start_game"}, *first, *second, {"type": "end_game"}]
    )

    assert retained == []


def test_numpy_and_tensor_packing_preserve_bits_and_nested_shapes():
    inputs = (
        {
            "mask": numpy.array([[True, False, True], [False, True, False]]),
            "signed": numpy.array([[-128, -1, 127], [42, 0, -7]], "int8")[
                :, ::-1
            ],
            "other_signed": numpy.array([[[-1]], [[-128]]], "int8"),
            "aligned": numpy.array([[-128, -1, 0, 127], [1, 2, 3, 4]], "int8"),
            "short": numpy.array([[-32768, -1, 32767], [0, 7, -9]], "int16"),
        },
        {
            "integer": numpy.array([-2147483648, 2147483647], "int32"),
            "float": numpy.array([[0.0, -0.0], [numpy.nan, -4.0]], "float32"),
            "half": numpy.array([[0.0, -0.0], [1.25, -4.0]], "float16"),
        },
    )
    packer = training_data.PackedBatch(inputs)
    packed = packer.pack_numpy(inputs)
    tensor_packed = packer.pack(inputs)
    assert {name: value.shape for name, value in packed.items()} == {
        "bool": (2, 1),
        "int8": (2, 3),
        "int16": (2, 2),
        "int32": (2, 1),
        "float32": (2, 2),
        "float16": (2, 2),
    }
    for name, value in packed.items():
        assert value.dtype == tensor_packed[name].numpy().dtype
        assert value.tobytes() == tensor_packed[name].numpy().tobytes()
    restored = tf.function(packer.unpack)(packed)
    tf.nest.assert_same_structure(inputs, restored)
    for expected, actual in zip(
        tf.nest.flatten(inputs), tf.nest.flatten(restored)
    ):
        assert actual.numpy().dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.numpy().tobytes() == expected.tobytes()
