#[derive(Clone, Copy)]
pub enum Dtype {
    INT8,
    INT16,
    FLOAT16,
    FLOAT32,
}
impl Dtype {
    pub const fn name(self) -> &'static str {
        match self {
            Self::INT8 => "int8",
            Self::INT16 => "int16",
            Self::FLOAT16 => "float16",
            Self::FLOAT32 => "float32",
        }
    }
    const fn width(self) -> usize {
        match self {
            Self::INT8 => 1,
            Self::INT16 | Self::FLOAT16 => 2,
            Self::FLOAT32 => 4,
        }
    }
}
pub struct Field {
    pub name: &'static str,
    pub shape: &'static [usize],
    pub dtype: Dtype,
    pub count: usize,
    pub width: usize,
    pub offset: usize,
    pub size: usize,
}
const fn layout<const N: usize>(mut fields: [Field; N]) -> [Field; N] {
    let mut offset = 0;
    let mut index = 0;
    while index < N {
        fields[index].count = 1;
        let mut dimension = 0;
        while dimension < fields[index].shape.len() {
            fields[index].count *= fields[index].shape[dimension];
            dimension += 1;
        }
        fields[index].width = fields[index].dtype.width();
        fields[index].size = fields[index].count * fields[index].width;
        fields[index].offset = offset;
        offset += fields[index].size;
        index += 1;
    }
    fields
}
macro_rules! fields {
    ($($name:ident: $dtype:ident [$($dimension:literal),*]),* $(,)?) => {
        pub const FIELDS: &[Field] = &layout([$(Field {
            name: stringify!($name), shape: &[$($dimension),*], dtype: Dtype::$dtype,
            count: 0, width: 0, offset: 0, size: 0,
        }),*]);
    };
}
// Keep declaration order stable: Python model projections use this order.
// Names, dimensions and storage types belong to the native output boundary.
fields! {
    candidate_hand_counts: INT8 [96, 34],
    candidate_hand_red: INT8 [96, 3],
    candidate_melds: INT8 [96, 34, 6],
    candidate_hand_valid: INT8 [96],
    discard_candidate_hand: INT8 [37],
    chi_candidate_hand: INT8 [3, 37],
    pon_candidate_hand: INT8 [1, 37],
    kan_candidate_hand: INT8 [34],
    river_pre_tile_id: INT8 [3, 32],
    river_pre_is_red: INT8 [3, 32],
    river_pre_tsumogiri: INT8 [3, 32],
    river_pre_tedashi_relation: INT8 [3, 32],
    river_pre_tedashi_gap: INT8 [3, 32],
    river_pre_origin: INT8 [3, 32],
    river_pre_riichi: INT8 [3, 32],
    river_pre_call_resolution: INT8 [3, 32],
    river_pre_caller: INT8 [3, 32],
    river_pre_event_age: FLOAT16 [3, 32],
    river_pre_own_gap: FLOAT16 [3, 32],
    river_pre_live_wall: FLOAT16 [3, 32],
    river_pre_current_dora_multiplicity: INT8 [3, 32],
    river_pre_dora_multiplicity_at_discard: INT8 [3, 32],
    river_pre_valid: INT8 [3, 32],
    river_post_tile_id: INT8 [3, 32],
    river_post_is_red: INT8 [3, 32],
    river_post_tsumogiri: INT8 [3, 32],
    river_post_tedashi_relation: INT8 [3, 32],
    river_post_tedashi_gap: INT8 [3, 32],
    river_post_origin: INT8 [3, 32],
    river_post_riichi: INT8 [3, 32],
    river_post_call_resolution: INT8 [3, 32],
    river_post_caller: INT8 [3, 32],
    river_post_event_age: FLOAT16 [3, 32],
    river_post_own_gap: FLOAT16 [3, 32],
    river_post_live_wall: FLOAT16 [3, 32],
    river_post_current_dora_multiplicity: INT8 [3, 32],
    river_post_dora_multiplicity_at_discard: INT8 [3, 32],
    river_post_valid: INT8 [3, 32],
    meld_type: INT8 [4, 4],
    meld_tile_ids: INT8 [4, 4, 4],
    meld_tile_red: INT8 [4, 4, 4],
    meld_tile_valid: INT8 [4, 4, 4],
    meld_source: INT8 [4, 4],
    meld_call_age: FLOAT16 [4, 4],
    meld_upgrade_age: FLOAT16 [4, 4],
    meld_valid: INT8 [4, 4],
    current_draw_tile_id: INT8 [],
    current_draw_is_red: INT8 [],
    current_draw_valid: INT8 [],
    normal_self_shanten: INT8 [],
    chiitoitsu_self_shanten: INT8 [],
    kokushi_self_shanten: INT8 [],
    closed_hand: INT8 [],
    self_efficiency: FLOAT32 [3],
    self_yaku_possibility: INT8 [19],
    dora_tile_id: INT8 [5],
    dora_valid: INT8 [5],
    dora_multiplicity: INT8 [34],
    public_visible_red_counts: INT8 [3],
    known_unavailable_counts: INT8 [34],
    prevailing_wind: INT8 [],
    kyoku_number: INT8 [],
    dealer_relative_seat: INT8 [],
    honba: INT16 [],
    riichi_sticks: INT8 [],
    live_wall_count: INT8 [],
    total_kan_count: INT8 [],
    nominal_all_last: INT8 [],
    dealer_currently_leads: INT8 [],
    any_player_at_or_above_30000: INT8 [],
    scheduled_hands_remaining: INT8 [],
    seat_context: FLOAT32 [4, 2],
    score_extrema: FLOAT32 [2],
    seat_wind: INT8 [4],
    open_opponent_yaku_possibility: INT8 [4, 13],
    opponent_called_tile_count: INT8 [4],
    opponent_discard_count: INT8 [4],
    last_tedashi_tile: INT8 [4, 37],
    riichi_push_count: INT8 [4, 4],
    genbutsu_to_seat: INT8 [4, 34],
    passed_unchanged_to_seat: INT8 [4, 34],
    blocker_counts: INT8 [34, 5],
    sotogawa_earliest_turn_to_seat: INT8 [4, 34],
    win_kind: INT8 [],
    decision_phase: INT8 [],
    trigger_tile_id: INT8 [],
    trigger_is_red: INT8 [],
    trigger_valid: INT8 [],
    trigger_source: INT8 [],
    riichi_state: INT8 [4],
    ippatsu_alive: INT8 [4],
    rinshan_state: INT8 [],
    riichi_legal_mask: INT8 [2],
    riichi_discard_legal_mask: INT8 [37],
    response_legal_mask: INT8 [4],
    chi_pattern_legal_mask: INT8 [3],
    chi_discard_legal: INT8 [3, 37],
    chi_discard_shanten: INT8 [3, 37],
    chi_discard_ukeire_count: INT16 [3, 37],
    pon_discard_legal: INT8 [1, 37],
    pon_discard_shanten: INT8 [1, 37],
    pon_discard_ukeire_count: INT16 [1, 37],
    kan_legal_mask: INT8 [35],
    chi_call_efficiency: FLOAT32 [3, 3],
    chi_call_yaku_possibility: INT8 [3, 19],
    pon_call_efficiency: FLOAT32 [1, 3],
    pon_call_yaku_possibility: INT8 [1, 19],
    kan_efficiency: FLOAT32 [34, 3],
    kan_yaku_possibility: INT8 [34, 19],
    discard_abort: INT8 [2, 37],
    discard_action_legal: INT8 [37],
    discard_tedashi_legal: INT8 [37],
    discard_action_normal_shanten: INT8 [37],
    discard_action_normal_ukeire_count: INT16 [37],
    discard_action_normal_upgrade_valid: INT8 [37],
    discard_action_normal_upgrade_type_count: INT8 [37],
    discard_action_normal_upgrade_tile_count: INT16 [37],
    discard_action_normal_upgrade_weighted_gain: INT16 [37],
    discard_action_all_shanten: INT8 [37],
    discard_action_all_ukeire_count: INT16 [37],
    discard_action_all_upgrade_valid: INT8 [37],
    discard_action_all_upgrade_type_count: INT8 [37],
    discard_action_all_upgrade_tile_count: INT16 [37],
    discard_action_all_upgrade_weighted_gain: INT16 [37],
    discard_action_yaku_possibility: INT8 [37, 19],
    discard_retained_genbutsu: INT8 [37, 4],
    action_value_range: FLOAT32 [295, 2, 2, 2],
    action_value_valid: INT8 [295, 2, 2],
    action_value_eligible: INT8 [295],
    action_open_yaku_restricted: INT8 [295, 2],
    candidate_yaku_shanten: INT8 [96, 3],
    public_fu: INT16 [4],
    public_bonus: INT8 [4],
    candidate_public_fu: INT16 [96],
    candidate_public_bonus: INT8 [96],
}
pub const ROW_BYTES: usize = FIELDS[FIELDS.len() - 1].offset + FIELDS[FIELDS.len() - 1].size;
