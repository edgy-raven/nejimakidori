"""Single numeric TFRecord contract for training and inference."""

import hashlib
import json
import math
import pathlib

import numpy

from log_dataset import _native_collection

LABELS = {
    "ron_payments": {"shape": [3, 37], "dtype": "int32"},
    "ron_legal": {"shape": [3, 37], "dtype": "int8"},
    "win_action": {"shape": [], "dtype": "int8"},
    "sakigiri_pair": {"shape": [2], "dtype": "int8"},
    "sakigiri_weight": {"shape": [], "dtype": "float32"},
    "candidate_yaku_affinity": {"shape": [96, 3], "dtype": "float16"},
    "opponent_shanten": {"shape": [3], "dtype": "int8"},
    "opponent_ukeire": {"shape": [3, 34], "dtype": "int8"},
    "opponent_hand_counts": {"shape": [3, 37], "dtype": "int8"},
    "terminal_payment": {"shape": [4, 4, 4], "dtype": "int32"},
    "round_completion_yaku": {"shape": [4, 2, 23], "dtype": "int8"},
    "round_completion_score": {"shape": [4, 2], "dtype": "int16"},
    "discard_action": {"shape": [], "dtype": "int8"},
    "discard_regret": {"shape": [37], "dtype": "float32"},
    "continuation_discard": {"shape": [], "dtype": "int8"},
    "riichi_action": {"shape": [], "dtype": "int8"},
    "response_action": {"shape": [], "dtype": "int8"},
    "chi_pattern": {"shape": [], "dtype": "int8"},
    "kan_action": {"shape": [], "dtype": "int8"},
}


PACKING = {
    "feature/last_tedashi_tile": (1, 0),
    "feature/river_pre_tsumogiri": (1, 0),
    "feature/river_pre_tedashi_relation": (2, 0),
    "feature/river_pre_tedashi_gap": (5, 0),
    "feature/river_pre_current_dora_multiplicity": (3, 0),
    "feature/river_pre_dora_multiplicity_at_discard": (3, 0),
    "feature/river_post_tsumogiri": (1, 0),
    "feature/river_post_tedashi_relation": (2, 0),
    "feature/river_post_tedashi_gap": (5, 0),
    "feature/river_post_current_dora_multiplicity": (3, 0),
    "feature/river_post_dora_multiplicity_at_discard": (3, 0),
    "feature/self_yaku_possibility": (1, 0),
    "feature/open_opponent_yaku_possibility": (1, 0),
    "feature/opponent_called_tile_count": (5, 0),
    "feature/opponent_discard_count": (6, 0),
    "feature/riichi_push_count": (7, 0),
    "feature/genbutsu_to_seat": (1, 0),
    "feature/passed_unchanged_to_seat": (1, 0),
    "feature/blocker_counts": (2, 0),
    "feature/sotogawa_earliest_turn_to_seat": (3, 0),
    "feature/chi_call_yaku_possibility": (1, 0),
    "feature/pon_call_yaku_possibility": (1, 0),
    "feature/kan_yaku_possibility": (1, 0),
    "feature/discard_action_yaku_possibility": (1, 0),
    "label/opponent_ukeire": (2, 1),
    "label/round_completion_yaku": (2, 1),
}

FIELDS = {
    prefix + "/" + name: definition
    for prefix, fields in (
        ("feature", _native_collection.INPUTS),
        ("label", LABELS),
    )
    for name, definition in fields.items()
}


def game_partition(game_id):
    """Keep original games in the same corpus and rollout holdout bucket."""
    return int.from_bytes(hashlib.sha256(game_id.encode()).digest()[:4]) % 10


def encode(name, value):
    value = numpy.asarray(value)
    if name in PACKING:
        width, offset = PACKING[name]
        values = value.reshape(-1).astype(numpy.int64) + offset
        bits = ((values[:, None] >> numpy.arange(width)) & 1).reshape(-1)
        raw = numpy.packbits(
            bits.astype(numpy.uint8), bitorder="little"
        ).tobytes()
    else:
        raw = value.tobytes()
    return raw.rstrip(b"\0")


def serialize_observation(observation, focused):
    import tensorflow as tf

    example = tf.train.Example()
    for prefix, values, definitions in (
        ("feature", observation.features, _native_collection.INPUTS),
        ("label", vars(observation), LABELS),
    ):
        for name, definition in definitions.items():
            example.features.feature[
                prefix + "/" + name
            ].bytes_list.value.append(
                encode(
                    prefix + "/" + name,
                    numpy.asarray(
                        values[name], dtype=definition["dtype"]
                    ).reshape(definition["shape"]),
                )
            )
    example.features.feature["meta/sample_weight"].float_list.value.append(1)
    example.features.feature["meta/focused"].int64_list.value.append(
        int(focused)
    )
    for name in ("game_id", "kyoku_id"):
        example.features.feature["meta/" + name].bytes_list.value.append(
            getattr(observation, name).encode()
        )
    for name in ("decision_index", "actor", "hand_yaku_focus"):
        example.features.feature["meta/" + name].int64_list.value.append(
            int(getattr(observation, name))
        )
    return example.SerializeToString()


def parse_batch(examples):
    import tensorflow as tf

    parsed = tf.io.parse_example(
        examples,
        {
            **{name: tf.io.FixedLenFeature([], tf.string) for name in FIELDS},
            "meta/sample_weight": tf.io.FixedLenFeature([], tf.float32),
        },
    )
    decoded = {}
    for name, definition in FIELDS.items():
        size = math.prod(definition["shape"])
        dtype = tf.as_dtype(definition["dtype"])
        if name in PACKING:
            width, offset = PACKING[name]
            bit = numpy.arange(size) * width
            raw = tf.io.decode_raw(
                parsed[name], tf.uint8, fixed_length=(size * width + 7) // 8
            )
            if 8 % width == 0:
                words = tf.gather(raw, bit // 8, axis=1)
            else:
                # Read at most two bytes per value, including crossings.
                raw = tf.pad(tf.cast(raw, tf.uint16), [[0, 0], [0, 1]])
                words = tf.bitwise.bitwise_or(
                    tf.gather(raw, bit // 8, axis=1),
                    tf.bitwise.left_shift(
                        tf.gather(raw, bit // 8 + 1, axis=1), 8
                    ),
                )
            values = (
                tf.cast(
                    tf.bitwise.bitwise_and(
                        tf.bitwise.right_shift(
                            words, tf.constant(bit % 8, words.dtype)
                        ),
                        (1 << width) - 1,
                    ),
                    tf.int32,
                )
                - offset
            )
            values = tf.cast(values, dtype)
        else:
            values = tf.io.decode_raw(
                parsed[name],
                tf.uint16 if dtype == tf.float16 else dtype,
                fixed_length=size * dtype.size,
            )
            if dtype == tf.float16:
                values = tf.bitcast(values, tf.float16)
        decoded[name] = tf.reshape(values, [-1, *definition["shape"]])
    inputs = {
        name: decoded["feature/" + name] for name in _native_collection.INPUTS
    }
    labels = {name: decoded["label/" + name] for name in LABELS}
    labels.update(
        {
            "win_decision_logits": labels["win_action"],
            "discard_policy_logits": labels["discard_action"],
            "riichi_policy_logits": tf.where(
                labels["riichi_action"] >= 0,
                tf.cast(labels["riichi_action"], tf.int32) * 37
                + tf.cast(labels["continuation_discard"], tf.int32),
                -1,
            ),
            "response_policy_logits": tf.where(
                labels["response_action"] == 0,
                0,
                tf.where(
                    labels["response_action"] == 1,
                    1
                    + tf.cast(labels["chi_pattern"], tf.int32) * 37
                    + tf.cast(labels["continuation_discard"], tf.int32),
                    tf.where(
                        labels["response_action"] == 2,
                        112 + tf.cast(labels["continuation_discard"], tf.int32),
                        tf.where(labels["response_action"] == 3, 149, -1),
                    ),
                ),
            ),
            "kan_action_logits": labels["kan_action"],
        }
    )
    return inputs, labels, parsed["meta/sample_weight"]


def validate_summary(data):
    summary = json.loads(
        (pathlib.Path(data) / "dataset_summary.json").read_text()
    )
    for name, expected in {
        "input_schema": _native_collection.INPUTS,
        "label_schema": LABELS,
    }.items():
        if name not in summary or summary[name] != expected:
            raise ValueError(f"dataset {name} requires regeneration")
