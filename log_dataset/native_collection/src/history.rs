use riichienv_core::{
    action::{Action, ActionType},
    hand_evaluator::HandEvaluator,
    observation::Observation,
    parser::mjai_to_tid,
    replay::MjaiEvent,
    rule::GameRule,
    state::GameState,
};
use serde_json::Value;

pub fn tile_code(tile: u8) -> usize {
    if [16, 52, 88].contains(&tile) {
        34 + tile as usize / 36
    } else {
        tile as usize / 4
    }
}
pub fn base(tile: usize) -> usize {
    if tile < 34 { tile } else { (tile - 34) * 9 + 4 }
}
pub fn code(value: &Value) -> usize {
    tile_code(mjai_to_tid(value.as_str().unwrap()).unwrap())
}

pub fn dora(markers: &[usize]) -> [i32; 34] {
    let mut out = [0; 34];
    for &marker in markers {
        let tile = base(marker);
        let next = if tile < 27 {
            tile / 9 * 9 + (tile + 1) % 9
        } else if tile < 31 {
            27 + (tile - 26) % 4
        } else {
            31 + (tile - 30) % 3
        };
        out[next] += 1;
    }
    out
}
#[derive(Clone)]
pub struct RiverRow {
    pub tile: usize,
    pub event_index: usize,
    pub live_wall: i32,
    pub own_turns: [usize; 4],
    pub origin: u8,
    pub riichi: bool,
    pub resolution: usize,
    pub caller: Option<usize>,
    pub dora_multiplicity: u8,
    pub tsumogiri: bool,
}
#[derive(Clone)]
pub struct MeldEvent {
    pub kind: String,
    pub tile: Option<usize>,
    pub consumed: Vec<usize>,
    pub target: Option<usize>,
    pub event_index: usize,
    pub upgrade_index: Option<usize>,
    pub upgrade_tile: Option<usize>,
}
pub struct PublicHistory {
    pub replay: GameState,
    pub events: Vec<Value>,
    pub start: Value,
    pub rivers: [Vec<RiverRow>; 4],
    pub discards: Vec<(usize, usize)>,
    pub riichi_boundary: [Option<usize>; 4],
    pub accepted_event: [Option<usize>; 4],
    pub meld_events: [Vec<MeldEvent>; 4],
    pub declared: [bool; 4],
    pub accepted: [bool; 4],
    pub ippatsu: [bool; 4],
    pub draws: [Option<usize>; 4],
    pub live_wall: i32,
    pub pending_discard: Option<(usize, usize)>,
    pub rinshan_actor: Option<usize>,
    pub dora_markers: Vec<usize>,
    last_origin: [u8; 4],
}
impl PublicHistory {
    pub fn new() -> Self {
        Self {
            replay: GameState::new(0, true, Some(0), 0, GameRule::default_tenhou()),
            events: Vec::new(),
            start: Value::Null,
            rivers: std::array::from_fn(|_| Vec::new()),
            discards: Vec::new(),
            riichi_boundary: [None; 4],
            accepted_event: [None; 4],
            meld_events: std::array::from_fn(|_| Vec::new()),
            declared: [false; 4],
            accepted: [false; 4],
            ippatsu: [false; 4],
            draws: [None; 4],
            live_wall: 70,
            pending_discard: None,
            rinshan_actor: None,
            dora_markers: Vec::new(),
            last_origin: [0; 4],
        }
    }
    pub fn apply(&mut self, event: &Value) {
        self.events.push(event.clone());
        let kind = event["type"].as_str().unwrap();
        match kind {
            "start_kyoku" => {
                self.replay = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
                self.start = event.clone();
                self.events = vec![event.clone()];
                self.rivers = std::array::from_fn(|_| Vec::new());
                self.discards.clear();
                self.riichi_boundary = [None; 4];
                self.accepted_event = [None; 4];
                self.meld_events = std::array::from_fn(|_| Vec::new());
                self.declared = [false; 4];
                self.accepted = [false; 4];
                self.ippatsu = [false; 4];
                self.draws = [None; 4];
                self.live_wall = 70;
                self.pending_discard = None;
                self.last_origin = [0; 4];
                self.rinshan_actor = None;
                self.dora_markers = vec![code(&event["dora_marker"])];
            }
            "tsumo" => {
                let actor = event["actor"].as_u64().unwrap() as usize;
                self.live_wall -= 1;
                self.draws[actor] = if event["pai"] == "?" {
                    None
                } else {
                    Some(code(&event["pai"]))
                };
                self.last_origin[actor] = 1;
                if let Some((seat, index)) = self.pending_discard {
                    self.rivers[seat][index].resolution = 1;
                }
                self.pending_discard = None;
            }
            "dahai" => {
                let actor = event["actor"].as_u64().unwrap() as usize;
                if let Some((seat, index)) = self.pending_discard {
                    self.rivers[seat][index].resolution = 1;
                }
                let row = RiverRow {
                    tile: code(&event["pai"]),
                    event_index: self.events.len() - 1,
                    live_wall: self.live_wall,
                    own_turns: std::array::from_fn(|seat| self.rivers[seat].len()),
                    origin: self.last_origin[actor],
                    riichi: self.declared[actor] && !self.accepted[actor],
                    resolution: 0,
                    caller: None,
                    dora_multiplicity: dora(&self.dora_markers)[base(code(&event["pai"]))] as u8,
                    tsumogiri: event["tsumogiri"].as_bool().unwrap(),
                };
                if row.riichi {
                    self.riichi_boundary[actor] = Some(self.rivers[actor].len() + 1);
                }
                self.discards.push((actor, self.rivers[actor].len()));
                self.rivers[actor].push(row);
                self.pending_discard = Some((actor, self.rivers[actor].len() - 1));
                self.draws[actor] = None;
                self.rinshan_actor = None;
                if self.accepted[actor] {
                    self.ippatsu[actor] = false;
                }
            }
            "reach" => self.declared[event["actor"].as_u64().unwrap() as usize] = true,
            "reach_accepted" => {
                let actor = event["actor"].as_u64().unwrap() as usize;
                self.accepted[actor] = true;
                self.accepted_event[actor] = Some(self.events.len() - 1);
                self.ippatsu[actor] = true;
            }
            "chi" | "pon" | "daiminkan" | "ankan" | "kakan" => {
                let actor = event["actor"].as_u64().unwrap() as usize;
                self.ippatsu = [false; 4];
                if let Some((seat, index)) = self.pending_discard {
                    if matches!(kind, "chi" | "pon" | "daiminkan") {
                        self.rivers[seat][index].resolution = match kind {
                            "chi" => 2,
                            "pon" => 3,
                            _ => 4,
                        };
                        self.rivers[seat][index].caller = Some(actor);
                    }
                }
                self.pending_discard = None;
                self.draws[actor] = None;
                self.last_origin[actor] = match kind {
                    "chi" => 2,
                    "pon" => 3,
                    _ => 4,
                };
                if matches!(kind, "ankan" | "kakan" | "daiminkan") {
                    self.rinshan_actor = Some(actor);
                }
                if kind == "kakan" {
                    for meld in &mut self.meld_events[actor] {
                        if meld.kind == "pon"
                            && base(meld.tile.unwrap()) == base(code(&event["pai"]))
                        {
                            meld.upgrade_index = Some(self.events.len() - 1);
                            meld.upgrade_tile = Some(code(&event["pai"]));
                            meld.kind = kind.into();
                        }
                    }
                } else {
                    self.meld_events[actor].push(MeldEvent {
                        kind: kind.into(),
                        tile: if kind == "ankan" {
                            None
                        } else {
                            Some(code(&event["pai"]))
                        },
                        consumed: event["consumed"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(code)
                            .collect(),
                        target: event["target"].as_u64().map(|x| x as usize),
                        event_index: self.events.len() - 1,
                        upgrade_index: None,
                        upgrade_tile: None,
                    });
                }
            }
            "dora" => self.dora_markers.push(code(&event["dora_marker"])),
            _ => {}
        }
        self.replay
            .apply_mjai_event(serde_json::from_value::<MjaiEvent>(event.clone()).unwrap());
    }
    pub fn public_tiles(&self) -> impl Iterator<Item = usize> + '_ {
        self.dora_markers
            .iter()
            .copied()
            .chain(self.rivers.iter().flatten().map(|row| row.tile))
            .chain(
                self.meld_events
                    .iter()
                    .flatten()
                    .flat_map(|meld| meld.consumed.iter().copied()),
            )
            .chain(
                self.meld_events
                    .iter()
                    .flatten()
                    .filter_map(|meld| meld.upgrade_tile),
            )
    }
    pub fn missed_agari(&self, actor: usize) -> bool {
        if self.replay.players[actor].missed_agari_doujun {
            return true;
        }
        let waits = HandEvaluator::new(
            self.replay.players[actor].hand.clone(),
            self.replay.players[actor].melds.clone(),
        )
        .get_waits_u8();
        for (index, event) in self.events.iter().enumerate().rev() {
            let kind = event["type"].as_str().unwrap();
            if event["actor"].as_u64() == Some(actor as u64)
                && (kind == "reach_accepted"
                    || (!self.accepted[actor]
                        && matches!(
                            kind,
                            "tsumo" | "chi" | "pon" | "daiminkan" | "ankan" | "kakan"
                        )))
            {
                break;
            }
            if kind == "dahai"
                && event["actor"].as_u64() != Some(actor as u64)
                && self
                    .pending_discard
                    .map(|(seat, row)| self.rivers[seat][row].event_index)
                    != Some(index)
                && waits.contains(&(base(code(&event["pai"])) as u8))
            {
                return true;
            }
        }
        false
    }
    /// Return the drawn copy ID used by the canonical replay observation.
    /// Ordinary copies increase from the base ID, skipping the red-five ID;
    /// the drawn copy is the last matching copy, not the engine's physical ID.
    pub fn canonical_drawn_tile(&self, actor: usize) -> Option<u8> {
        self.draws[actor].map(|code| {
            let tile = (base(code) * 4) as u8;
            if code >= 34 {
                return tile;
            }
            let copies = self.replay.players[actor]
                .hand
                .iter()
                .filter(|&&held| tile_code(held) == code)
                .count() as u8;
            tile + u8::from(matches!(tile, 16 | 52 | 88)) + copies - 1
        })
    }
    pub fn legal_observation(&mut self, actor: usize) -> Observation {
        let mut obs = self.replay.get_observation(actor as u8);
        let mut hand = Vec::new();
        for &tile in &obs.hands[actor] {
            let mut physical = tile as u8 / 4 * 4;
            if ![16, 52, 88].contains(&(tile as u8)) && [16, 52, 88].contains(&physical) {
                physical += 1;
            }
            while hand.contains(&(physical as u32)) {
                physical += 1;
            }
            hand.push(physical as u32);
        }
        let drawn = self.canonical_drawn_tile(actor);
        let mut legal: Vec<Action> = Vec::new();
        for action in obs.legal_actions_method() {
            if action.action_type == ActionType::Discard {
                for &tile in &hand {
                    if tile_code(tile as u8) == tile_code(action.tile.unwrap()) {
                        if self.accepted[actor] && Some(tile as u8) != drawn {
                            continue;
                        }
                        if !legal.iter().any(|item| {
                            item.action_type == action.action_type && item.tile == Some(tile as u8)
                        }) {
                            legal.push(Action::new(
                                action.action_type,
                                Some(tile as u8),
                                Vec::new(),
                                Some(actor as u8),
                            ));
                        }
                    }
                }
            } else {
                legal.push(action);
            }
        }
        obs.hands[actor] = hand;
        obs.drawn_tile = drawn;
        obs.replace_legal_actions(legal);
        obs
    }
}
