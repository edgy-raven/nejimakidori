"""Single numeric TFRecord contract for training and inference."""

import hashlib
import json
import math
import pathlib

import numpy
import tensorflow as tf

from log_dataset import player_ranks
from model import feature_vector, outcomes

LABELS = {
    "abort_points": {"shape": [2, 37], "dtype": "int32"},
    "abort_return_valid": {"shape": [2, 37], "dtype": "int8"},
    "ron_payments": {"shape": [3, 37], "dtype": "int32"},
    "ron_legal": {"shape": [3, 37], "dtype": "int8"},
    "ron_return": {"shape": [37], "dtype": "float32"},
    "ron_return_valid": {"shape": [37], "dtype": "int8"},
    "win_action": {"shape": [], "dtype": "int8"},
    "specialist_outcomes": {"shape": [3], "dtype": "float32"},
    "tenpai_reached": {"shape": [], "dtype": "int8"},
    "specialist_allocation": {"shape": [2], "dtype": "int8"},
    "tenpai_value": {"shape": [], "dtype": "float32"},
    "defense_ron": {"shape": [37], "dtype": "float16"},
    "sakigiri_pair": {"shape": [2], "dtype": "int8"},
    "sakigiri_weight": {"shape": [], "dtype": "float32"},
    "candidate_yaku_affinity": {"shape": [96, 3], "dtype": "float16"},
    "sakigiri_cost": {"shape": [], "dtype": "float32"},
    "opponent_shanten": {"shape": [3], "dtype": "int8"},
    "opponent_ukeire": {"shape": [3, 34], "dtype": "int8"},
    "opponent_hand_counts": {"shape": [3, 37], "dtype": "int8"},
    "terminal_payment": {"shape": [4, 4, 4], "dtype": "int32"},
    "round_completion_yaku": {"shape": [4, 2, 23], "dtype": "int8"},
    "round_completion_score": {"shape": [4, 2], "dtype": "int16"},
    "discard_origin": {"shape": [], "dtype": "int8"},
    "discard_action": {"shape": [], "dtype": "int8"},
    "discard_regret": {"shape": [37], "dtype": "float32"},
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
    "eligibility": "eventual_tenpai_cost_or_flat_never_tenpai",
    # At most 136 discards, 36 retained tile codes, three opponents;
    # each discounted, participation-weighted opponent contribution is <= 1.
    "maximum_tenpai_cost": 136 * 36 * 3,
    "never_tenpai_cost": 136 * 36 * 3 + 1,
    "neededness": "distinct_improving_tenpais_unordered_draw_combinations",
    "copies": "weighted_original_copies_retained_per_starting_copy",
    "max_draws": 3,
    "search_nodes": 512,
    "future_draw_discount": 0.9,
}

FOCUSED_SAMPLING = {
    "rule": "visible_threat_call_tenpai_critical_context_state_only",
    "dora_participation_below": 0.25,
    "qualifying_retention": 1.0,
    "legal_own_kan_opportunity_retention": 1.0,
    "own_open_yakuhai": "retain_after_confirmed_pon_including_upgraded_kan",
    "yaku_hand_eligibility": {
        "source": "decision_time_shortest_tenpai_paths",
        "fraction_strictly_above": [2, 3],
        "weight": "unordered_available_physical_draw_combinations",
        "path_yaku": "all_structural_winning_tiles_ron_and_tsumo",
        "families": "scorer_completion_yaku_merge_hon_chinitsu_chanta_junchan",
        "max_draws": 3,
        "search_nodes": 512,
        "max_tenpai_shapes": 4096,
        "incomplete_search": "does_not_qualify",
    },
    "call_to_tenpai_value": {
        "statistic": "live_wait_weighted_base_ron",
        "open_ron_strictly_above": 3900,
        "or_valid_closed_riichi_ron_strictly_below": 7700,
        "recorded_call_or_pass_retention": 1.0,
    },
    "opponent_threat": {
        "riichi": True,
        "visible_han_at_least": 3,
        "visible_han": "meld_dora_red_yakuhai_double_wind_counts_twice",
        "honitsu_han": 2,
        "honitsu_evidence": (
            "two_open_melds_one_suit_plus_3_to_7_discards_in_both_other_suits"
        ),
        "open_melds_at_least": 3,
        "late_open_melds_at_least": 2,
        "late_opponent_discards_at_least": 9,
    },
    "broad_context_discard_filter": "either_origin",
    "threat_discard_origin": "all_visible_opponent_threat_or_last_16_tiles",
    "ordinary_call_pass_retention": 0.0,
}


DATA_CONTRACT = {
    "discard_policy": "37_tile_codes_origin_auxiliary_only",
    "discard_origin_target": "recorded_tedashi_0_tsumogiri_1",
    "riichi_push_count": "tedashi_non_genbutsu_at_discard_per_riichi_seat",
    "passed_tile_safety": "resolved_since_last_tedashi_or_meld_change",
    "abort_payoff_target": "safe_four_riichi_or_multi_owner_four_kan_deposit",
    "observation_selection": "nonautomatic_discretionary_decisions_only",
    "player_rank_sampling": player_ranks.SAMPLING,
    "call_red_policy": "always_consume_available_red_five",
    "dealership_return_target": outcomes.DEALERSHIP_TARGET,
    "specialist_outcome_target": "receipts_ron_incidence_ron_loss_only",
    "terminal_payment_encoding": "type_payer_recipient_raw_int32_points",
    "payment_event_contract": (
        "three_ron_claims_tsumo_exhaustive_masks_abort_bank"
    ),
    "payment_types": list(outcomes.PAYMENT_TYPES),
    "terminal_payment_target": "typed_remaining_honba_free_payments",
    "round_placement_target": "terminal_scores_initial_dealer_ties",
    "kyoku_identity": "wind_round_honba",
    "decision_context": "kan_dora_call_origin_public_tiles",
    "river_layout": (
        "three_opponents_pre_including_declaration_" "and_post_padded_32"
    ),
    "input_schema": feature_vector.INPUTS,
    "label_schema": LABELS,
    "win_target": "native_legal_observed_accept_or_decline_censored_winners",
    "scored_wait_summary": "base_types_legal_ranges_live_weights",
    "completion_yaku_target": (
        "next_tenpai_takame_same_witness_as_score_ron_tsumo_unknown_minus_one"
    ),
    "completion_score_target": {
        "selection": "next_observed_tenpai_max_points_per_ron_tsumo",
        "support": "legal_publicly_unexhausted_waits",
        "conditions": "declared_riichi_known_dora_no_ura_ippatsu_special",
        "encoding": "paired_han_fu_limit_tiers_true_yakuman_multiplicity",
        "ties": "native_wait_order_normal_before_red",
        "unknown": -1,
    },
    "tenpai_reached_target": "whole_hand_ever_structural_tenpai_or_win",
    "specialist_allocation_target": (
        "first_tenpai_shared_value_pursuit_position_dora_push_safe_retreat"
    ),
    "tenpai_value_target": (
        "first_tenpai_live_copy_weighted_legal_ron_per_10000"
    ),
    "discard_regret_target": (
        "mean_missed_improving_draws_until_tedashi_or_improving_draw_"
        "only_tenpai_hands"
    ),
    "counterfactual_ron_target": (
        "discard_and_post_call_ron_legal_amount_known_ura_cancel_ippatsu"
    ),
    "ron_target": "live_physical_tiles_structural_furiten",
    "opponent_tile_target": "structural_ukeire_by_shanten",
    "specialist_labels": SPECIALIST_LABELS,
    "sakigiri_target": SAKIGIRI_TARGET,
    "sakigiri_pair_target": (
        "native_verified_safe_early_same_live_completion_binary_ron_order_"
        "first_tenpai_endpoint_normalized"
    ),
    "candidate_yaku_affinity_target": (
        "visible_shortest_route_mass_tanyao_flush_outside_unknown_minus_one"
    ),
    "focused_sampling": FOCUSED_SAMPLING,
    "opponent_value_target": "live_ron_at_least_7700_within_one_shanten",
    "afk_filter": {"scope": "kyoku"},
    "regret_threat_filter": (
        "riichi_or_live_ron_at_least_7700_within_one_shanten_or_current_ron"
    ),
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
        ("feature", feature_vector.INPUTS),
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


def serialize_observation(observation, focused):
    example = tf.train.Example()
    for prefix, values, definitions in (
        ("feature", observation.features, feature_vector.INPUTS),
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
        name: decoded["feature/" + name] for name in feature_vector.INPUTS
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
    for name, expected in DATA_CONTRACT.items():
        if name not in summary or summary[name] != expected:
            raise ValueError(f"dataset {name} requires regeneration")
