"""Named actor-input tensors with a zero-copy dictionary boundary."""

import dataclasses

import numpy


def tensor(shape, dtype):
    return dataclasses.field(metadata={"shape": shape, "dtype": dtype})


@dataclasses.dataclass(init=False, eq=False, repr=False)
class FeatureVector:
    candidate_hand_counts: numpy.ndarray = tensor([96, 34], "int8")
    candidate_hand_red: numpy.ndarray = tensor([96, 3], "int8")
    candidate_melds: numpy.ndarray = tensor([96, 34, 6], "int8")
    candidate_hand_valid: numpy.ndarray = tensor([96], "int8")
    discard_candidate_hand: numpy.ndarray = tensor([37], "int8")
    chi_candidate_hand: numpy.ndarray = tensor([3, 37], "int8")
    pon_candidate_hand: numpy.ndarray = tensor([1, 37], "int8")
    kan_candidate_hand: numpy.ndarray = tensor([34], "int8")
    river_tile_id: numpy.ndarray = tensor([4, 32], "int8")
    river_is_red: numpy.ndarray = tensor([4, 32], "int8")
    river_tsumogiri: numpy.ndarray = tensor([4, 32], "int8")
    river_tedashi_relation: numpy.ndarray = tensor([4, 32], "int8")
    river_tedashi_gap: numpy.ndarray = tensor([4, 32], "int8")
    river_origin: numpy.ndarray = tensor([4, 32], "int8")
    river_riichi: numpy.ndarray = tensor([4, 32], "int8")
    river_call_resolution: numpy.ndarray = tensor([4, 32], "int8")
    river_caller: numpy.ndarray = tensor([4, 32], "int8")
    river_event_age: numpy.ndarray = tensor([4, 32], "float16")
    river_own_gap: numpy.ndarray = tensor([4, 32], "float16")
    river_live_wall: numpy.ndarray = tensor([4, 32], "float16")
    river_current_dora_multiplicity: numpy.ndarray = tensor([4, 32], "int8")
    river_dora_multiplicity_at_discard: numpy.ndarray = tensor([4, 32], "int8")
    river_valid: numpy.ndarray = tensor([4, 32], "int8")
    meld_type: numpy.ndarray = tensor([4, 4], "int8")
    meld_tile_ids: numpy.ndarray = tensor([4, 4, 4], "int8")
    meld_tile_red: numpy.ndarray = tensor([4, 4, 4], "int8")
    meld_tile_valid: numpy.ndarray = tensor([4, 4, 4], "int8")
    meld_source: numpy.ndarray = tensor([4, 4], "int8")
    meld_call_age: numpy.ndarray = tensor([4, 4], "float16")
    meld_upgrade_age: numpy.ndarray = tensor([4, 4], "float16")
    meld_valid: numpy.ndarray = tensor([4, 4], "int8")
    current_draw_tile_id: numpy.ndarray = tensor([], "int8")
    current_draw_is_red: numpy.ndarray = tensor([], "int8")
    current_draw_valid: numpy.ndarray = tensor([], "int8")
    normal_self_shanten: numpy.ndarray = tensor([], "int8")
    chiitoitsu_self_shanten: numpy.ndarray = tensor([], "int8")
    kokushi_self_shanten: numpy.ndarray = tensor([], "int8")
    closed_hand: numpy.ndarray = tensor([], "int8")
    self_efficiency: numpy.ndarray = tensor([3], "float32")
    self_yaku_possibility: numpy.ndarray = tensor([19], "int8")
    dora_tile_id: numpy.ndarray = tensor([5], "int8")
    dora_valid: numpy.ndarray = tensor([5], "int8")
    dora_multiplicity: numpy.ndarray = tensor([34], "int8")
    public_visible_red_counts: numpy.ndarray = tensor([3], "int8")
    known_unavailable_counts: numpy.ndarray = tensor([34], "int8")
    prevailing_wind: numpy.ndarray = tensor([], "int8")
    kyoku_number: numpy.ndarray = tensor([], "int8")
    dealer_relative_seat: numpy.ndarray = tensor([], "int8")
    honba: numpy.ndarray = tensor([], "int16")
    riichi_sticks: numpy.ndarray = tensor([], "int8")
    live_wall_count: numpy.ndarray = tensor([], "int8")
    total_kan_count: numpy.ndarray = tensor([], "int8")
    nominal_all_last: numpy.ndarray = tensor([], "int8")
    dealer_currently_leads: numpy.ndarray = tensor([], "int8")
    any_player_at_or_above_30000: numpy.ndarray = tensor([], "int8")
    scheduled_hands_remaining: numpy.ndarray = tensor([], "int8")
    seat_context: numpy.ndarray = tensor([4, 2], "float32")
    score_extrema: numpy.ndarray = tensor([2], "float32")
    seat_wind: numpy.ndarray = tensor([4], "int8")
    open_opponent_yaku_possibility: numpy.ndarray = tensor([4, 13], "int8")
    opponent_called_tile_count: numpy.ndarray = tensor([4], "int8")
    opponent_discard_count: numpy.ndarray = tensor([4], "int8")
    genbutsu_to_seat: numpy.ndarray = tensor([4, 34], "int8")
    blocker_counts: numpy.ndarray = tensor([34, 5], "int8")
    sotogawa_earliest_turn_to_seat: numpy.ndarray = tensor([4, 34], "int8")
    win_kind: numpy.ndarray = tensor([], "int8")
    decision_phase: numpy.ndarray = tensor([], "int8")
    trigger_tile_id: numpy.ndarray = tensor([], "int8")
    trigger_is_red: numpy.ndarray = tensor([], "int8")
    trigger_valid: numpy.ndarray = tensor([], "int8")
    trigger_source: numpy.ndarray = tensor([], "int8")
    riichi_state: numpy.ndarray = tensor([4], "int8")
    ippatsu_alive: numpy.ndarray = tensor([4], "int8")
    rinshan_state: numpy.ndarray = tensor([], "int8")
    riichi_legal_mask: numpy.ndarray = tensor([2], "int8")
    riichi_discard_legal_mask: numpy.ndarray = tensor([37], "int8")
    response_legal_mask: numpy.ndarray = tensor([4], "int8")
    chi_pattern_legal_mask: numpy.ndarray = tensor([3], "int8")
    chi_discard_legal: numpy.ndarray = tensor([3, 37], "int8")
    chi_discard_shanten: numpy.ndarray = tensor([3, 37], "int8")
    chi_discard_ukeire_count: numpy.ndarray = tensor([3, 37], "int16")
    pon_discard_legal: numpy.ndarray = tensor([1, 37], "int8")
    pon_discard_shanten: numpy.ndarray = tensor([1, 37], "int8")
    pon_discard_ukeire_count: numpy.ndarray = tensor([1, 37], "int16")
    kan_legal_mask: numpy.ndarray = tensor([35], "int8")
    chi_call_efficiency: numpy.ndarray = tensor([3, 3], "float32")
    chi_call_yaku_possibility: numpy.ndarray = tensor([3, 19], "int8")
    pon_call_efficiency: numpy.ndarray = tensor([1, 3], "float32")
    pon_call_yaku_possibility: numpy.ndarray = tensor([1, 19], "int8")
    kan_efficiency: numpy.ndarray = tensor([34, 3], "float32")
    kan_yaku_possibility: numpy.ndarray = tensor([34, 19], "int8")
    discard_action_legal: numpy.ndarray = tensor([37], "int8")
    discard_tedashi_legal: numpy.ndarray = tensor([37], "int8")
    discard_completion_probability: numpy.ndarray = tensor([37], "float32")
    discard_completion_valid: numpy.ndarray = tensor([37], "int8")
    discard_action_normal_shanten: numpy.ndarray = tensor([37], "int8")
    discard_action_normal_ukeire_count: numpy.ndarray = tensor([37], "int16")
    discard_action_normal_upgrade_valid: numpy.ndarray = tensor([37], "int8")
    discard_action_normal_upgrade_type_count: numpy.ndarray = tensor(
        [37], "int8"
    )
    discard_action_normal_upgrade_tile_count: numpy.ndarray = tensor(
        [37], "int16"
    )
    discard_action_normal_upgrade_weighted_gain: numpy.ndarray = tensor(
        [37], "int16"
    )
    discard_action_all_shanten: numpy.ndarray = tensor([37], "int8")
    discard_action_all_ukeire_count: numpy.ndarray = tensor([37], "int16")
    discard_action_all_upgrade_valid: numpy.ndarray = tensor([37], "int8")
    discard_action_all_upgrade_type_count: numpy.ndarray = tensor([37], "int8")
    discard_action_all_upgrade_tile_count: numpy.ndarray = tensor([37], "int16")
    discard_action_all_upgrade_weighted_gain: numpy.ndarray = tensor(
        [37], "int16"
    )
    discard_action_yaku_possibility: numpy.ndarray = tensor([37, 19], "int8")
    discard_tenpai_values: numpy.ndarray = tensor([37, 4, 8], "float32")
    self_tenpai_values: numpy.ndarray = tensor([4, 8], "float32")
    discard_retained_genbutsu: numpy.ndarray = tensor([37, 4], "int8")
    chi_call_tenpai_values: numpy.ndarray = tensor([3, 37, 2, 8], "float32")
    pon_call_tenpai_values: numpy.ndarray = tensor([1, 37, 2, 8], "float32")
    chi_call_closed_riichi_values: numpy.ndarray = tensor(
        [3, 37, 2, 8], "float32"
    )
    pon_call_closed_riichi_values: numpy.ndarray = tensor(
        [1, 37, 2, 8], "float32"
    )

    def __init__(self):
        self.__dict__ = {
            name: numpy.zeros(spec["shape"], spec["dtype"])
            for name, spec in INPUTS.items()
        }

    @property
    def tensors(self):
        """The actual attribute dictionary, not a copy or serialization."""
        return self.__dict__


INPUTS = {
    field.name: dict(field.metadata)
    for field in dataclasses.fields(FeatureVector)
}
"""Actor-output shapes shared by setup, export, and inference."""

TRAIN_POLICY_HEADS = (
    "discard_policy_logits",
    "riichi_policy_logits",
    "response_policy_logits",
    "kan_action_logits",
)

# All actor outputs use float32; only their shapes vary.
ACTOR_OUTPUTS = {
    "win_decision_logits": [2],
    "opponent_shanten_logits": [3, 4],
    "opponent_tenpai_logits": [3],
    "opponent_ukeire_by_shanten_logits": [3, 4, 34],
    "opponent_ukeire_logits": [3, 34],
    "opponent_wait_logits": [3, 34],
    "opponent_hand_count_logits": [3, 37, 5],
    "terminal_payment_logits": [4, 4, 4, 50],
    "round_completion_yaku_logits": [4, 23],
    "discard_policy_logits": [38],
    "round_placement_logits": [4, 4],
    "final_placement_logits": [4],
    "discard_origin_logits": [2, 2],
    "riichi_policy_logits": [76],
    "response_policy_logits": [150],
    "riichi_action_logits": [2],
    "tenpai_decision_logits": [3],
    "response_action_logits": [4],
    "chi_pattern_logits": [3],
    "kan_action_logits": [35],
}
