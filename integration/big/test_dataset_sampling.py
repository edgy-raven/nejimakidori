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
    focus_sampling,
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
    assert [records.serialize_observation(row, False) for row in retained] == [
        records.serialize_observation(row, False)
        for row in rows
        if row.actor == actor
    ]


def test_critic_filter_preserves_partition_order_and_partial_metadata_batch(
    tmp_path,
):
    rows = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    "feature/decision_phase": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[
                                bytes(
                                    [
                                        (
                                            1
                                            if index == 0
                                            else 2 if index == 3 else 0
                                        )
                                    ]
                                )
                            ]
                        )
                    ),
                    **{
                        name: tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[value])
                        )
                        for name, value in {
                            "critic/partition": index % 10,
                            "meta/focused": int(index % 3 != 0),
                            "meta/focus_categories": (
                                (1 << (index % 6)) if index % 3 != 0 else 0
                            ),
                            "meta/late_placement": int(
                                index % 6 == 5 and index % 4 == 3
                            ),
                            "meta/placement_gap": int(index % 6 == 5),
                            "test/index": index,
                        }.items()
                    },
                }
            )
        ).SerializeToString()
        for index in range(2051)
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
        for sampling in (
            focus_sampling.Sampling.FOCUSED,
            focus_sampling.Sampling.PLACEMENT_GAP,
            focus_sampling.Sampling.NATURAL,
        ):
            actual = [
                bytes(row.numpy())
                for batch in training_data.dataset(
                    root=tmp_path,
                    sampling=sampling,
                    partition=partition,
                    batch_size=4,
                    repeat=False,
                )
                for row in batch
            ]
            expected = [
                row
                for index, row in enumerate(rows)
                if (index % 10 < 7 if partition == "train" else index % 10 == 7)
                and (
                    index % 3 != 0
                    or (
                        sampling == focus_sampling.Sampling.FOCUSED
                        and index in (0, 3)
                    )
                    if sampling != focus_sampling.Sampling.NATURAL
                    else True
                )
                and (
                    index % 6 == 5
                    if sampling == focus_sampling.Sampling.PLACEMENT_GAP
                    else True
                )
                and (
                    index % 6 != 5 or index % 4 == 3
                    if sampling == focus_sampling.Sampling.FOCUSED
                    else True
                )
                for _ in range(
                    (
                        1
                        if index in (0, 3)
                        else focus_sampling.EMPHASIS[index % 6]
                    )
                    if sampling == focus_sampling.Sampling.FOCUSED
                    else 1
                )
            ]
            assert actual == (
                expected
                if partition == "validation"
                else expected[: len(expected) // 4 * 4]
            )

    # Every quartet has two natural, one original-focus and one gap row, with the
    # original partition preserved and no random quota fluctuation.
    natural = iter(
        training_data.dataset(
            root=tmp_path,
            sampling=focus_sampling.Sampling.NATURAL,
            partition="train",
            batch_size=8,
            repeat=True,
            seed=941,
        )
    )
    focused = iter(
        training_data.dataset(
            root=tmp_path,
            sampling=focus_sampling.Sampling.FOCUSED,
            partition="train",
            batch_size=4,
            repeat=True,
            seed=942,
        )
    )
    placement = iter(
        training_data.dataset(
            root=tmp_path,
            sampling=focus_sampling.Sampling.PLACEMENT_GAP,
            partition="train",
            batch_size=4,
            repeat=True,
            seed=943,
        )
    )
    mixed = iter(
        training_data.dataset(
            root=tmp_path,
            sampling=focus_sampling.Sampling.MIXED,
            partition="train",
            batch_size=16,
            repeat=True,
            seed=941,
        )
    )
    for _ in range(3):
        actual = next(mixed).numpy().reshape(-1, 4)
        numpy.testing.assert_array_equal(
            actual[:, :2].reshape(-1), next(natural).numpy()
        )
        numpy.testing.assert_array_equal(actual[:, 2], next(focused).numpy())
        numpy.testing.assert_array_equal(actual[:, 3], next(placement).numpy())

    # A resumed stream starts immediately with a different reproducible shuffle.
    batches = []
    for seed in (941, 185941, 185941):
        batches.append(
            next(
                iter(
                    training_data.dataset(
                        root=tmp_path,
                        sampling=focus_sampling.Sampling.FOCUSED,
                        partition="train",
                        batch_size=32,
                        repeat=True,
                        seed=seed,
                    )
                )
            )
            .numpy()
            .tolist()
        )
    assert batches[0] != batches[1]
    assert batches[1] == batches[2]
    for row in batches[1]:
        example = tf.train.Example.FromString(row).features.feature
        assert example["critic/partition"].int64_list.value[0] < 7
        assert example["meta/focused"].int64_list.value[0] == 1


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
    packed = packer.pack_numpy(*inputs)
    tensor_packed = packer.pack(*inputs)
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
