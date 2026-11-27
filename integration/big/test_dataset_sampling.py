"""Record sampling, retained-game routing and lossless batch transport."""

import json
import pathlib
import types

import numpy
import pytest
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
from training import expert


def test_retained_players_skip_work_without_changing_decisions_or_labels(
    monkeypatch,
):
    events = json.loads(
        pathlib.Path(__file__)
        .parent.parent.joinpath("fixtures/model/sakigiri_swap.json")
        .read_text()
    )
    rows = list(replay.Replay("retained", events).observations())
    actor = rows[0].actor

    pack = features.pack

    def retained_features(state, actor, phase):
        assert actor == rows[0].actor
        return pack(state=state, actor=actor, phase=phase)

    monkeypatch.setattr(features, "pack", retained_features)
    retained = list(
        replay.Replay("retained", events).observations(actors=(actor,))
    )
    assert 0 < len(retained) < len(rows)
    assert [records.serialize_observation(row) for row in retained] == [
        records.serialize_observation(row) for row in rows if row.actor == actor
    ]


def test_critic_filter_preserves_partition_order_and_partial_metadata_batch(
    tmp_path,
    monkeypatch,
):
    rows = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    **{
                        name: tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[value])
                        )
                        for name, value in {
                            "critic/partition": index % 10,
                            "test/index": index,
                            "meta/focused_priority": 3 * (index % 5 == 1)
                            + (index % 5 == 2),
                            "meta/decision_index": index,
                            "meta/actor": index % 4,
                        }.items()
                    },
                    **{
                        name: tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[value])
                        )
                        for name, value in {
                            "meta/game_id": f"game-{index // 100}".encode(),
                            "meta/kyoku_id": f"hand-{index // 10}".encode(),
                        }.items()
                    },
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
                "reference_schema": records.REFERENCE_INPUTS,
                "label_schema": records.LABELS,
                "critic_auxiliary": training_data.AUXILIARY,
                "critic_bonus_han": list(range(14)),
                "critic_scoring_states": [
                    list(s) for s in payment_scoring.SCORES
                ],
                "validation_sample": training_data.VALIDATION_SAMPLE,
                "sampling_fields": ["meta/focused_priority"],
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

    # Each compressed shard belongs to exactly one learner. Uneven lengths
    # exercise partial and entirely padded tails with repeated focused draws.
    summary = json.loads((tmp_path / "dataset_summary.json").read_text())
    summary["shard_manifest"] = []
    for rank, indices in enumerate((range(3), range(3, 17), range(17, 2053))):
        name = f"train-{rank:05d}.tfrecord.gz"
        with tf.io.TFRecordWriter(
            str(tmp_path / name), options="GZIP"
        ) as writer:
            for index in indices:
                writer.write(rows[index])
        summary["shard_manifest"].append(
            {
                "path": name,
                "counts": [
                    sum(index % 10 == partition for index in indices)
                    for partition in range(10)
                ],
                "sampling_counts": [
                    [
                        sum(
                            3 * (index % 5 == 1) + (index % 5 == 2)
                            for index in indices
                            if index % 10 == partition
                        ),
                    ]
                    for partition in range(10)
                ],
            }
        )
    (tmp_path / "dataset_summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(expert, "decode_records", lambda raw: {"raw": raw})
    consumed_passes = []
    for seed in (941, 942, 941):
        consumed = []
        global_counts = []
        for rank in range(3):
            batches = list(
                expert.input_batches(
                    data=tmp_path,
                    group=types.SimpleNamespace(rank=rank, replicas=3),
                    batch_size=128,
                    seed=seed,
                )
            )
            global_counts.append(
                [int(batch["global_count"]) for batch in batches]
            )
            rank_rows = [
                bytes(raw.numpy())
                for batch in batches
                for raw in batch["raw"][: int(batch["local_count"])]
            ]
            consumed.extend(rank_rows)
            focused = tf.io.parse_example(
                rank_rows[1::2],
                {"meta/focused_priority": tf.io.FixedLenFeature([], tf.int64)},
            )["meta/focused_priority"].numpy()
            # Every second real exposure is focused, even across odd batch
            # boundaries. Natural draws may themselves qualify as focused.
            assert len(focused) == len(rank_rows) // 2
            assert (focused > 0).all()
            numpy.testing.assert_array_equal(
                numpy.concatenate(
                    [
                        batch["natural"].numpy()[: int(batch["local_count"])]
                        for batch in batches
                    ]
                ),
                numpy.arange(len(rank_rows)) % 2 == 0,
            )
            assert all(len(batch["raw"]) == 43 for batch in batches)
            for batch in batches:
                indices = tf.io.parse_example(
                    batch["raw"],
                    {"test/index": tf.io.FixedLenFeature([], tf.int64)},
                )["test/index"].numpy()
                for name, expected in {
                    "decision_index": indices,
                    "actor": indices % 4,
                    "game_id": [
                        f"game-{index // 100}".encode() for index in indices
                    ],
                    "kyoku_id": [
                        f"hand-{index // 10}".encode() for index in indices
                    ],
                }.items():
                    numpy.testing.assert_array_equal(
                        batch["metadata"][name], expected
                    )
        assert global_counts[0] == global_counts[1] == global_counts[2]
        expected = [row for index, row in enumerate(rows) if index % 10 < 7]
        assert len(consumed) == len(expected)
        assert set(consumed) <= set(expected)
        assert len(set(consumed)) < len(consumed)
        indices = tf.io.parse_example(
            consumed, {"test/index": tf.io.FixedLenFeature([], tf.int64)}
        )["test/index"].numpy()
        natural = numpy.array(
            [index for index in range(len(rows)) if index % 10 < 7]
        )
        # Calls/routes gain exposure without leaking validation records or
        # exceeding the corpus-sized budget. No placement metadata is present.
        assert numpy.mean(indices % 5 == 1) > numpy.mean(natural % 5 == 1) + 0.1
        assert sum(global_counts[0]) == len(expected)
        consumed_passes.append(consumed)
    assert consumed_passes[0] != consumed_passes[1]
    assert consumed_passes[0] == consumed_passes[2]
    summary["shard_manifest"][0]["sampling_counts"] = [[0]] * 10
    (tmp_path / "dataset_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="focused records"):
        expert.input_batches(
            data=tmp_path,
            group=types.SimpleNamespace(rank=0, replicas=3),
            batch_size=128,
            seed=941,
        )


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
