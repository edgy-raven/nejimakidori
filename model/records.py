"""Single numeric TFRecord contract for training and inference."""

import json
import math
import pathlib

import numpy
import tensorflow as tf

import feature_vector
from log_dataset import player_ranks

from . import completion_yaku, outcomes

LABELS = {
    "abort_points": {"shape": [2, 37], "dtype": "int32"},
    "abort_return_valid": {"shape": [2, 37], "dtype": "int8"},
    "ron_payments": {"shape": [3, 37], "dtype": "int32"},
    "ron_return": {"shape": [37], "dtype": "float32"},
    "ron_return_valid": {"shape": [37], "dtype": "int8"},
    "win_action": {"shape": [], "dtype": "int8"},
    "specialist_outcomes": {"shape": [3], "dtype": "float32"},
    "tenpai_reached": {"shape": [], "dtype": "int8"},
    "tenpai_value": {"shape": [], "dtype": "float32"},
    "defense_ron": {"shape": [37], "dtype": "float16"},
    "sakigiri_cost": {"shape": [], "dtype": "float32"},
    "opponent_shanten": {"shape": [3], "dtype": "int8"},
    "opponent_ukeire": {"shape": [3, 34], "dtype": "int8"},
    "opponent_hand_counts": {"shape": [3, 37], "dtype": "int8"},
    "terminal_payment": {"shape": [4, 4, 4], "dtype": "int16"},
    "round_completion_yaku": {"shape": [4, 23], "dtype": "int8"},
    "discard_action": {"shape": [], "dtype": "int8"},
    "discard_regret": {"shape": [38], "dtype": "float32"},
    "continuation_discard": {"shape": [], "dtype": "int8"},
    "riichi_action": {"shape": [], "dtype": "int8"},
    "response_action": {"shape": [], "dtype": "int8"},
    "chi_pattern": {"shape": [], "dtype": "int8"},
    "kan_action": {"shape": [], "dtype": "int8"},
    "round_placement": {"shape": [4], "dtype": "int8"},
    "final_placement": {"shape": [], "dtype": "int8"},
    "dealership_return": {"shape": [], "dtype": "float32"},
    "dealership_return_valid": {"shape": [], "dtype": "int8"},
}

SPECIALIST_LABELS = [
    "specialist_outcomes",
    "tenpai_reached",
    "tenpai_value",
    "defense_ron",
    "sakigiri_cost",
]

SAKIGIRI_TARGET = {
    "rule": "recorded_future_ron_holding_turns_reaching_tenpai",
    "units": "discounted_opponent_threat_turns",
    "eligibility": "own_structural_tenpai_at_or_after_discard",
    "neededness": "distinct_improving_tenpais_unordered_draw_combinations",
    "copies": "weighted_original_copies_retained_per_starting_copy",
    "max_draws": 3,
    "search_nodes": 512,
    "future_draw_discount": 0.9,
}

FOCUSED_SAMPLING = {
    "rule": "calls_tenpai_threats_late_wall_south_or_yaku_hand_tedashi_dora",
    "dora_participation_below": 0.25,
    "qualifying_retention": 1.0,
    "kan_alone_retention": 0.0,
    "yaku_hand_classes": list(completion_yaku.FOCUSED_NAMES),
    "yaku_hand_eligibility": "any_tenpai_scored_wait_entire_hand_tedashi",
    "ordinary_call_pass_retention": 0.0,
    "ordinary_tsumogiri_retention": 0.0,
}


DATA_CONTRACT = {
    "abort_payoff_target": "safe_four_riichi_or_multi_owner_four_kan_deposit",
    "observation_selection": "nonautomatic_discretionary_decisions_only",
    "player_rank_sampling": player_ranks.SAMPLING,
    "call_red_policy": "always_consume_available_red_five",
    "dealership_return_target": outcomes.DEALERSHIP_TARGET,
    "specialist_outcome_target": "receipts_ron_incidence_ron_loss_only",
    "terminal_payment_encoding": "type_payer_recipient_payment_class_index",
    "payment_types": list(outcomes.PAYMENT_TYPES),
    "terminal_payment_target": "typed_remaining_honba_free_payments",
    "round_placement_target": "terminal_scores_initial_dealer_ties",
    "kyoku_identity": "wind_round_honba",
    "decision_context": "kan_dora_call_origin_public_tiles",
    "input_schema": feature_vector.INPUTS,
    "label_schema": LABELS,
    "win_target": "native_legal_observed_accept_or_decline_censored_winners",
    "scored_wait_summary": "base_types_legal_ranges_live_weights",
    "completion_yaku_target": "scorer_ids_quantified_structural_routes",
    "tenpai_reached_target": "whole_hand_ever_structural_tenpai_or_win",
    "tenpai_value_target": "first_tenpai_live_copy_weighted_legal_ron_per_10000",
    "discard_regret_target": (
        "mean_missed_improving_draws_until_tedashi_or_improving_draw_"
        "only_tenpai_hands"
    ),
    "counterfactual_ron_target": "recipient_points_honba_known_ura_terminal_utility",
    "ron_target": "live_physical_tiles_structural_furiten",
    "opponent_tile_target": "structural_ukeire_by_shanten",
    "specialist_labels": SPECIALIST_LABELS,
    "sakigiri_target": SAKIGIRI_TARGET,
    "focused_sampling": FOCUSED_SAMPLING,
    "opponent_value_target": "live_ron_at_least_7700_within_one_shanten",
    "afk_filter": {"scope": "kyoku"},
    "regret_threat_filter": (
        "riichi_or_live_ron_at_least_7700_within_one_shanten_or_current_ron"
    ),
}


PACKING = {
    "feature/river_tsumogiri": (1, 0),
    "feature/river_tedashi_relation": (2, 0),
    "feature/river_tedashi_gap": (5, 0),
    "feature/river_current_dora_multiplicity": (3, 0),
    "feature/river_dora_multiplicity_at_discard": (3, 0),
    "feature/self_yaku_possibility": (1, 0),
    "feature/open_opponent_yaku_possibility": (1, 0),
    "feature/opponent_called_tile_count": (5, 0),
    "feature/opponent_discard_count": (6, 0),
    "feature/genbutsu_to_seat": (1, 0),
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
        ("feature", feature_vector.INPUTS),
        ("label", LABELS),
    )
    for name, definition in fields.items()
}


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


def decode(name, raw, definition):
    size = math.prod(definition["shape"])
    dtype = numpy.dtype(definition["dtype"])
    if name in PACKING:
        width, offset = PACKING[name]
        packed = numpy.frombuffer(
            raw.ljust((size * width + 7) // 8, b"\0"), dtype=numpy.uint8
        )
        bits = numpy.unpackbits(packed, bitorder="little")[: size * width]
        values = (bits.reshape(size, width) * (1 << numpy.arange(width))).sum(1)
        return (values - offset).astype(dtype).reshape(definition["shape"])
    return numpy.frombuffer(
        raw.ljust(size * dtype.itemsize, b"\0"), dtype=dtype
    ).reshape(definition["shape"])


def parse_batch(examples):
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
            bit = numpy.arange(size)[:, None] * width + numpy.arange(width)
            raw = tf.io.decode_raw(
                parsed[name], tf.uint8, fixed_length=(size * width + 7) // 8
            )
            values = (
                tf.reduce_sum(
                    tf.bitwise.bitwise_and(
                        tf.bitwise.right_shift(
                            tf.cast(tf.gather(raw, bit // 8, axis=1), tf.int32),
                            bit % 8,
                        ),
                        1,
                    )
                    * (1 << numpy.arange(width)),
                    axis=-1,
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
        name: decoded["feature/" + name] for name in feature_vector.INPUTS
    }
    labels = {name: decoded["label/" + name] for name in LABELS}
    labels.update(
        {
            "win_decision_logits": labels["win_action"],
            "discard_policy_logits": labels["discard_action"],
            "riichi_policy_logits": tf.where(
                labels["riichi_action"] >= 0,
                tf.cast(labels["riichi_action"], tf.int32) * 38
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
    for name, expected in DATA_CONTRACT.items():
        if name not in summary or summary[name] != expected:
            raise ValueError(f"dataset {name} requires regeneration")


def dataset(data, batch_size, training=False, sampling="natural"):
    validate_summary(data)
    files = sorted(pathlib.Path(data).glob("*.tfrecord.gz"))
    paths = tf.data.Dataset.from_tensor_slices([str(path) for path in files])
    if training:
        paths = paths.shuffle(len(files), seed=17)
    stream = paths.interleave(
        lambda path: tf.data.TFRecordDataset(path, compression_type="GZIP"),
        cycle_length=min(16, len(files)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    if sampling == "focused":
        stream = (
            stream.batch(1024)
            .map(
                lambda rows: tf.boolean_mask(
                    rows,
                    tf.io.parse_example(
                        rows,
                        {"meta/focused": tf.io.FixedLenFeature([], tf.int64)},
                    )["meta/focused"]
                    > 0,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .unbatch()
        )
    if training:
        stream = stream.shuffle(100000, seed=17).repeat()
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    return (
        stream.batch(batch_size, drop_remainder=training)
        .map(parse_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(2)
        .with_options(options)
    )
