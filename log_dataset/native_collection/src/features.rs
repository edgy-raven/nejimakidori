//! Typed engine snapshots for the shared C++ feature calculator.
use crate::{candidate, focus, history::PublicHistory, packed::Row, scoring};
use riichienv_core::{
    action::ActionType,
    observation::Observation,
    replay::MjaiEvent,
    types::{Meld, MeldType},
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NativeMeld {
    kind: i32,
    opened: i32,
    size: i32,
    tiles: [i32; 4],
    called: i32,
    event_size: i32,
    event_tiles: [i32; 4],
    target: i32,
    event: i32,
    upgrade: i32,
}
impl NativeMeld {
    pub fn new(meld: &Meld) -> Self {
        let mut out = Self {
            kind: meld.meld_type as i32,
            opened: meld.opened as i32,
            size: meld.tiles.len() as i32,
            called: meld.called_tile.map_or(-1, i32::from),
            ..Self::default()
        };
        for (i, &tile) in meld.tiles.iter().enumerate() {
            out.tiles[i] = tile as i32;
        }
        out
    }
    fn meld(&self) -> Meld {
        Meld {
            meld_type: match self.kind {
                0 => MeldType::Chi,
                1 => MeldType::Pon,
                2 => MeldType::Daiminkan,
                3 => MeldType::Ankan,
                4 => MeldType::Kakan,
                _ => unreachable!(),
            },
            tiles: self.tiles[..self.size as usize]
                .iter()
                .map(|&t| t as u8)
                .collect(),
            opened: self.opened != 0,
            from_who: -1,
            called_tile: (self.called >= 0).then_some(self.called as u8),
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeRiver {
    tile: i32,
    event: i32,
    wall: i32,
    turns: [i32; 4],
    origin: i32,
    riichi: i32,
    resolution: i32,
    caller: i32,
    dora: i32,
    tsumogiri: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NativeHand {
    size: i32,
    tiles: [i32; 14],
    meld_count: i32,
    melds: [NativeMeld; 4],
}
impl NativeHand {
    fn new(obs: &Observation, actor: usize) -> Self {
        let mut out = Self {
            size: obs.hands[actor].len() as i32,
            meld_count: obs.melds[actor].len() as i32,
            ..Self::default()
        };
        for (i, &tile) in obs.hands[actor].iter().enumerate() {
            out.tiles[i] = tile as i32;
        }
        for (i, meld) in obs.melds[actor].iter().enumerate() {
            out.melds[i] = NativeMeld::new(meld);
        }
        out
    }
}
#[repr(C)]
struct State {
    hand: NativeHand,
    melds: [[NativeMeld; 4]; 4],
    meld_count: [i32; 4],
    rivers: [*const NativeRiver; 4],
    river_count: [usize; 4],
    boundary: [i32; 4],
    accepted_event: [i32; 4],
    declared: [i32; 4],
    accepted: [i32; 4],
    ippatsu: [i32; 4],
    draw: i32,
    actor: i32,
    phase: i32,
    east_only: i32,
    tsumo: i32,
    rinshan: i32,
    events: i32,
    wall: i32,
    oya: i32,
    kyoku: i32,
    wind: i32,
    honba: i32,
    sticks: i32,
    scores: [i32; 4],
    dora_size: i32,
    dora: [i32; 5],
    public_counts: [i32; 37],
    exposed: [i32; 34],
    trigger_seat: i32,
    trigger_tile: i32,
}
#[repr(C)]
struct Analysis {
    counts: [u8; 34],
    unavailable: [u8; 34],
    unavailable_red: [i32; 3],
    genbutsu: [[u8; 34]; 4],
    dora_counts: [i32; 34],
    public_fu: [i32; 4],
    public_bonus: [i32; 4],
    distance: i32,
    wind: i32,
    kyoku: i32,
    oya: i32,
    remaining: i32,
    order: [i32; 4],
}
#[repr(C)]
struct Branch {
    hand: NativeHand,
    response: i32,
    option: i32,
    tile: i32,
    cuts: [u8; 37],
    exposed: [i32; 34],
}
#[repr(C)]
struct Legal {
    discards: [u8; 37],
    riichi: [u8; 37],
    calls: [[u8; 37]; 4],
}
#[repr(C)]
struct Focus {
    routes: [u8; 3],
    fast: u8,
    threat: u8,
    phase: u8,
    yakuhai: i32,
}
#[repr(C)]
#[derive(Default)]
pub struct Ranges {
    pub values: [f64; 8],
    pub valid: [u8; 4],
    pub restricted: [u8; 2],
    pub eligible: u8,
}
#[repr(C)]
pub struct RangeContext {
    pub own_discards: u64,
    pub missed_agari: i32,
    pub unavailable: [u8; 34],
    pub unavailable_red: [i32; 3],
}
impl RangeContext {
    pub fn new(ctx: &scoring::Context) -> Self {
        Self {
            own_discards: ctx.own_discards.iter().fold(0, |mask, &t| mask | (1 << t)),
            missed_agari: ctx.missed_agari as i32,
            unavailable: ctx.unavailable,
            unavailable_red: ctx.unavailable_red,
        }
    }
    pub fn available_tiles(&self, base: u8) -> impl Iterator<Item = (u8, bool)> {
        let five = matches!(base, 4 | 13 | 22);
        let reds = if five {
            1 - self.unavailable_red[base as usize / 9]
        } else {
            0
        };
        [
            (
                base * 4 + u8::from(five),
                false,
                4 - self.unavailable[base as usize] as i32 - reds,
            ),
            (base * 4, true, reds),
        ]
        .into_iter()
        .filter(|&(_, _, copies)| copies > 0)
        .map(|(tile, red, _)| (tile, red))
    }
}
type Score = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *const NativeHand,
    *const RangeContext,
    i32,
    u64,
    *mut Ranges,
);
unsafe extern "C" {
    fn feature_pack_base(row: *mut u8, state: *const State, analysis: *mut Analysis);
    fn feature_pack_candidates(
        row: *mut u8,
        state: *const State,
        analysis: *const Analysis,
        branches: *const Branch,
        size: usize,
        legal: *mut Legal,
        focus: *mut Focus,
        availability: *const RangeContext,
        context: *mut std::ffi::c_void,
        score: Score,
    );
    #[cfg(test)]
    fn feature_ranges(
        hand: *const NativeHand,
        ctx: *const RangeContext,
        cut: i32,
        opaque: *mut std::ffi::c_void,
        score: Score,
        out: *mut Ranges,
    );
}
unsafe extern "C" fn score(
    context: *mut std::ffi::c_void,
    hand: *const NativeHand,
    availability: *const RangeContext,
    cut: i32,
    waits: u64,
    out: *mut Ranges,
) {
    let ctx = unsafe { &*(context as *const scoring::Context) };
    let hand = unsafe { &*hand };
    let tiles: Vec<u8> = hand.tiles[..hand.size as usize]
        .iter()
        .map(|&t| t as u8)
        .collect();
    let melds: Vec<Meld> = hand.melds[..hand.meld_count as usize]
        .iter()
        .map(NativeMeld::meld)
        .collect();
    let result = scoring::ready_ranges(
        tiles,
        melds,
        ctx,
        unsafe { &*availability },
        (cut >= 0).then_some(cut as u8),
        waits,
    );
    unsafe {
        *out = Ranges {
            values: result.values,
            valid: result.valid,
            restricted: result.restricted,
            eligible: result.eligible as u8,
        };
    }
}
#[cfg(test)]
pub fn ranges(
    tiles: &[u8],
    melds: &[Meld],
    ctx: &scoring::Context,
    cut: Option<u8>,
) -> scoring::Ranges {
    let mut hand = NativeHand {
        size: tiles.len() as i32,
        meld_count: melds.len() as i32,
        ..NativeHand::default()
    };
    for (i, &t) in tiles.iter().enumerate() {
        hand.tiles[i] = t as i32;
    }
    for (i, m) in melds.iter().enumerate() {
        hand.melds[i] = NativeMeld::new(m);
    }
    let mut out = Ranges::default();
    unsafe {
        feature_ranges(
            &hand,
            &RangeContext::new(ctx),
            cut.map_or(-1, i32::from),
            ctx as *const _ as *mut std::ffi::c_void,
            score,
            &mut out,
        );
    }
    scoring::Ranges {
        values: out.values,
        valid: out.valid,
        restricted: out.restricted,
        eligible: out.eligible != 0,
    }
}
// This is a public observation snapshot, not hidden opponent hand data.
pub fn exposed(obs: &Observation) -> [i32; 34] {
    let mut values = [0; 34];
    for tile in obs
        .discards
        .iter()
        .flatten()
        .chain(obs.dora_indicators.iter())
    {
        values[*tile as usize / 4] += 1;
    }
    for meld in obs.melds.iter().flatten() {
        for &tile in &meld.tiles {
            values[tile as usize / 4] += 1;
        }
        if let Some(tile) = meld.called_tile {
            values[tile as usize / 4] -= 1;
        }
    }
    values
}
pub fn pack(
    row: &mut Row,
    history: &mut PublicHistory,
    obs: &Observation,
    actor: usize,
    phase: u8,
    east_only: bool,
    include_focus: bool,
) -> (candidate::Legal, focus::Focus) {
    let rivers: [Vec<NativeRiver>; 4] = std::array::from_fn(|seat| {
        history.rivers[seat]
            .iter()
            .map(|cut| NativeRiver {
                tile: cut.tile as i32,
                event: cut.event_index as i32,
                wall: cut.live_wall,
                turns: cut.own_turns.map(|x| x as i32),
                origin: cut.origin as i32,
                riichi: cut.riichi as i32,
                resolution: cut.resolution as i32,
                caller: cut.caller.map_or(-1, |x| x as i32),
                dora: cut.dora_multiplicity as i32,
                tsumogiri: cut.tsumogiri as i32,
            })
            .collect()
    });
    let mut state = State {
        hand: NativeHand::new(obs, actor),
        melds: [[NativeMeld::default(); 4]; 4],
        meld_count: std::array::from_fn(|seat| history.replay.players[seat].melds.len() as i32),
        rivers: std::array::from_fn(|seat| rivers[seat].as_ptr()),
        river_count: std::array::from_fn(|seat| rivers[seat].len()),
        boundary: history.riichi_boundary.map(|v| v.map_or(-1, |x| x as i32)),
        accepted_event: history.accepted_event.map(|v| v.map_or(-1, |x| x as i32)),
        declared: history.declared.map(i32::from),
        accepted: history.accepted.map(i32::from),
        ippatsu: history.ippatsu.map(i32::from),
        draw: history.draws[actor].map_or(-1, |x| x as i32),
        actor: actor as i32,
        phase: phase as i32,
        east_only: east_only as i32,
        tsumo: obs
            .legal_actions_method()
            .iter()
            .any(|a| a.action_type == ActionType::Tsumo) as i32,
        rinshan: history.rinshan_actor.map_or(-1, |x| x as i32),
        events: history.events.len() as i32,
        wall: history.live_wall,
        oya: history.start["oya"].as_i64().unwrap() as i32,
        kyoku: history.start["kyoku"].as_i64().unwrap() as i32,
        wind: "ESWN"
            .find(history.start["bakaze"].as_str().unwrap())
            .unwrap() as i32,
        honba: history.start["honba"].as_i64().unwrap() as i32,
        sticks: obs.riichi_sticks as i32,
        scores: obs.scores.clone().try_into().unwrap(),
        dora_size: history.dora_markers.len() as i32,
        dora: [0; 5],
        public_counts: [0; 37],
        exposed: exposed(obs),
        trigger_seat: history.pending_discard.map_or(-1, |(seat, _)| seat as i32),
        trigger_tile: history
            .pending_discard
            .map_or(-1, |(seat, index)| history.rivers[seat][index].tile as i32),
    };
    for seat in 0..4 {
        for (i, meld) in history.replay.players[seat].melds.iter().enumerate() {
            let event = &history.meld_events[seat][i];
            let mut m = NativeMeld::new(meld);
            let tiles: Vec<usize> = event
                .tile
                .iter()
                .chain(event.consumed.iter())
                .chain(event.upgrade_tile.iter())
                .copied()
                .collect();
            m.event_size = tiles.len() as i32;
            for (j, tile) in tiles.into_iter().enumerate() {
                m.event_tiles[j] = tile as i32;
            }
            m.target = event.target.map_or(-1, |x| x as i32);
            m.event = event.event_index as i32;
            m.upgrade = event.upgrade_index.map_or(-1, |x| x as i32);
            state.melds[seat][i] = m;
        }
    }
    for (i, &tile) in history.dora_markers.iter().enumerate() {
        state.dora[i] = tile as i32;
    }
    for tile in history.public_tiles() {
        state.public_counts[tile] += 1;
    }
    let mut analysis: Analysis = unsafe { std::mem::zeroed() };
    unsafe {
        feature_pack_base(row.bytes.as_mut_ptr(), &state, &mut analysis);
    }
    let base_analysis = focus::Analysis {
        counts: analysis.counts,
        unavailable: analysis.unavailable,
        distance: analysis.distance,
        wind: analysis.wind as usize,
        kyoku: analysis.kyoku as usize,
        oya: analysis.oya as usize,
        remaining: analysis.remaining as usize,
        order: analysis.order.map(|x| x as usize),
    };
    let mut sampling = if include_focus {
        focus::Focus::analyze(history, obs, actor, phase, &base_analysis)
    } else {
        focus::Focus::default()
    };
    let mut focus = Focus {
        routes: sampling.routes.map(u8::from),
        fast: sampling.fast as u8,
        threat: sampling.threat as u8,
        phase: sampling.phase,
        yakuhai: focus::yakuhai(history, actor, &base_analysis, true) as i32,
    };
    let mut legal = Legal {
        discards: [0; 37],
        riichi: [0; 37],
        calls: [[0; 37]; 4],
    };
    let actions = obs.legal_actions_method();
    for action in &actions {
        if action.action_type == ActionType::Discard {
            legal.discards[candidate::code(action.tile.unwrap())] = 1;
        }
        if action.action_type == ActionType::Riichi {
            let mut declared = history.replay.clone();
            declared.apply_mjai_event(MjaiEvent::Reach { actor });
            for cut in declared.get_observation(actor as u8).legal_actions_method() {
                if cut.action_type == ActionType::Discard {
                    legal.riichi[candidate::code(cut.tile.unwrap())] = 1;
                }
            }
        }
    }
    let mut branches = Vec::new();
    for action in candidate::prefer_red_calls(actions) {
        let response = match action.action_type {
            ActionType::Chi => 1,
            ActionType::Pon => 2,
            ActionType::Daiminkan => 3,
            ActionType::Ankan | ActionType::Kakan => 0,
            _ => continue,
        };
        let mut event: serde_json::Value = serde_json::from_str(&action.to_mjai()).unwrap();
        event["actor"] = serde_json::json!(actor);
        if response != 0 {
            event["target"] = serde_json::json!(history.pending_discard.unwrap().0);
        }
        let mut env = history.replay.clone();
        env.apply_mjai_event(serde_json::from_value(event).unwrap());
        let next = env.get_observation(actor as u8);
        let tile = action.tile.unwrap() / 4;
        let option = if response == 1 {
            tile - action
                .consume_tiles
                .iter()
                .map(|t| t / 4)
                .chain([tile])
                .min()
                .unwrap()
        } else {
            0
        };
        let mut branch = Branch {
            hand: NativeHand::new(&next, actor),
            response,
            option: option as i32,
            tile: tile as i32,
            cuts: [0; 37],
            exposed: exposed(&next),
        };
        if response == 1 || response == 2 {
            for cut in next.legal_actions_method() {
                if cut.action_type == ActionType::Discard {
                    branch.cuts[candidate::code(cut.tile.unwrap())] = 1;
                }
            }
        }
        branches.push(branch);
    }
    let mut ctx = scoring::Context {
        own_discards: history.rivers[actor]
            .iter()
            .map(|event| candidate::base(event.tile) as u8)
            .collect(),
        missed_agari: history.missed_agari(actor),
        unavailable: analysis.unavailable,
        unavailable_red: analysis.unavailable_red,
        dora: obs.dora_indicators.iter().map(|&t| t as u8).collect(),
        player_wind: ((actor + 4 - analysis.oya as usize) % 4) as u8,
        round_wind: analysis.wind as u8,
    };
    unsafe {
        feature_pack_candidates(
            row.bytes.as_mut_ptr(),
            &state,
            &analysis,
            branches.as_ptr(),
            branches.len(),
            &mut legal,
            &mut focus,
            &RangeContext::new(&ctx),
            &mut ctx as *mut _ as *mut std::ffi::c_void,
            score,
        );
    }
    sampling.fast = focus.fast != 0;
    sampling.threat = focus.threat != 0;
    (
        candidate::Legal {
            discards: legal.discards.map(|v| v != 0),
            riichi: legal.riichi.map(|v| v != 0),
            calls: legal.calls.map(|v| v.map(|x| x != 0)),
        },
        sampling,
    )
}

#[cfg(test)]
#[repr(C)]
pub struct Bank {
    keys: [[u8; 44]; 96],
    size: i32,
}
#[cfg(test)]
impl Default for Bank {
    fn default() -> Self {
        Self {
            keys: [[0; 44]; 96],
            size: 0,
        }
    }
}
#[cfg(test)]
pub fn bank_hand(
    row: &mut Row,
    bank: &mut Bank,
    tiles: &[u8],
    melds: &[Meld],
    values: (u16, u8),
) -> usize {
    let mut hand = NativeHand {
        size: tiles.len() as i32,
        meld_count: melds.len() as i32,
        ..NativeHand::default()
    };
    for (i, &t) in tiles.iter().enumerate() {
        hand.tiles[i] = t as i32;
    }
    for (i, m) in melds.iter().enumerate() {
        hand.melds[i] = NativeMeld::new(m);
    }
    unsafe extern "C" {
        fn feature_bank_hand(
            row: *mut u8,
            bank: *mut Bank,
            hand: *const NativeHand,
            fu: i32,
            bonus: i32,
        ) -> i32;
    }
    unsafe {
        feature_bank_hand(
            row.bytes.as_mut_ptr(),
            bank,
            &hand,
            values.0 as i32,
            values.1 as i32,
        ) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transport_layout_matches_cpp() {
        use std::mem::{offset_of, size_of};
        unsafe extern "C" {
            fn feature_layout(out: *mut usize);
        }
        let mut layout = [0usize; 16];
        unsafe {
            feature_layout(layout.as_mut_ptr());
        }
        assert_eq!(
            layout,
            [
                size_of::<NativeMeld>(),
                size_of::<NativeRiver>(),
                size_of::<NativeHand>(),
                size_of::<State>(),
                size_of::<Analysis>(),
                size_of::<Branch>(),
                size_of::<Legal>(),
                size_of::<Focus>(),
                size_of::<Ranges>(),
                size_of::<RangeContext>(),
                size_of::<Bank>(),
                offset_of!(State, rivers),
                offset_of!(State, exposed),
                offset_of!(Analysis, genbutsu),
                offset_of!(Focus, yakuhai),
                offset_of!(RangeContext, unavailable_red)
            ]
        );
    }
}
