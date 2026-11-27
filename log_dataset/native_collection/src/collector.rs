use crate::{
    actions, candidate,
    history::PublicHistory,
    packed::{ROW_BYTES, Row},
};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use riichienv_core::{
    action::{Action, ActionType},
    hand_evaluator::HandEvaluator,
    observation::Observation,
    rule::GameRule,
    state::GameState,
};
use serde_json::Value;
use std::collections::HashMap;
const COUNTERS: &[&str] = &[
    "games",
    "hands",
    "wins",
    "tsumo_wins",
    "win_points",
    "dealin_hands",
    "dealin_points",
    "riichi_hands",
    "riichi_wins",
    "riichi_dealins",
    "riichi_turn_sum",
    "open_hands",
    "draws",
    "exhaustive_draws",
    "exhaustive_tenpai_seats",
];
fn add(counts: &mut HashMap<String, i64>, name: &str, value: i64) {
    *counts.get_mut(name).unwrap() += value;
}
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Game {
    engine: GameState,
    history: PublicHistory,
    logs: Vec<Value>,
    hand: usize,
    pending: [Option<(usize, ActionType)>; 4],
    choices: HashMap<u8, Action>,
    reached: u8,
    called: u8,
    won: u8,
    dealt: u8,
    cuts: [i64; 4],
    riichi_turn_sum: i64,
    recorded: Option<Vec<Value>>,
    human_hand: bool,
}
impl Game {
    fn complete(&self, stop_after_hand: Option<usize>) -> bool {
        self.engine.is_done || stop_after_hand.is_some_and(|hand| self.hand > hand)
    }

    fn new(seed: u64, duplicate: bool) -> Self {
        let mut rule = GameRule::default_tenhou();
        rule.duplicate = duplicate;
        let mut engine = GameState::new(2, true, Some(seed), 0, rule);
        engine.event_buffer = Some(Vec::new());
        engine.reset();
        engine._initialize_round(0, 0, 0, 0, None, Some(vec![25000; 4]));
        Self {
            engine,
            history: PublicHistory::new(),
            logs: Vec::new(),
            hand: 0,
            pending: [None; 4],
            choices: HashMap::new(),
            reached: 0,
            called: 0,
            won: 0,
            dealt: 0,
            cuts: [0; 4],
            riichi_turn_sum: 0,
            recorded: None,
            human_hand: false,
        }
    }

    fn recorded_event(&self, actor: usize) -> Option<&Value> {
        let events = &self.recorded.as_ref().unwrap()[self.logs.len()..];
        let start = events
            .iter()
            .position(|event| !matches!(event["type"].as_str().unwrap(), "reach_accepted" | "dora"))
            .unwrap();
        let event = &events[start];
        if event["type"] == "ryukyoku"
            && matches!(event["reason"].as_str(), Some("kyushu_kyuhai" | "sanchaho"))
        {
            return Some(event);
        }
        if event["type"] == "hora" {
            events[start..]
                .iter()
                .take_while(|event| event["type"] == "hora")
                .find(|event| event["actor"].as_u64() == Some(actor as u64))
        } else {
            (event["actor"].as_u64() == Some(actor as u64)
                && matches!(
                    event["type"].as_str().unwrap(),
                    "dahai"
                        | "reach"
                        | "chi"
                        | "pon"
                        | "daiminkan"
                        | "ankan"
                        | "kakan"
                        | "ryukyoku"
                ))
            .then_some(event)
        }
    }

    fn recorded_action(&self, obs: &Observation, actor: usize) -> Action {
        let event = self.recorded_event(actor);
        obs.legal_actions_method()
            .into_iter()
            .find(|action| {
                if let Some(event) = event {
                    // Triple ron is logged as a draw, without individual hora
                    // events. All three legal ron claims must be reconstructed.
                    if event["type"] == "ryukyoku" && event["reason"] == "sanchaho" {
                        return action.action_type == ActionType::Ron;
                    }
                    let candidate: Value = serde_json::from_str(&action.to_mjai()).unwrap();
                    if candidate["type"] != event["type"] {
                        return false;
                    }
                    if action.action_type != ActionType::Ankan
                        && candidate.get("pai").is_some()
                        && candidate["pai"] != event["pai"]
                    {
                        return false;
                    }
                    if let Some(consumed) = candidate.get("consumed") {
                        let mut actual = consumed
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|tile| tile.as_str().unwrap())
                            .collect::<Vec<_>>();
                        let mut expected = event["consumed"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|tile| tile.as_str().unwrap())
                            .collect::<Vec<_>>();
                        actual.sort();
                        expected.sort();
                        if actual != expected {
                            return false;
                        }
                    }
                    action.action_type != ActionType::Discard
                        || event["tsumogiri"].as_bool().unwrap() == (action.tile == obs.drawn_tile)
                } else {
                    action.action_type == ActionType::Pass
                }
            })
            .unwrap_or_else(|| {
                panic!(
                    "recorded action missing at event {} for seat {actor}: {event:?}",
                    self.logs.len()
                )
            })
    }
    // Counterfactual terminal labels stay separate from observable feature rows.
    fn sync(&mut self, counts: &mut HashMap<String, i64>) {
        for mut event in self.engine.drain_events() {
            if let Some(recorded) = &self.recorded {
                if self.human_hand {
                    let expected = &recorded[self.logs.len()];
                    assert_eq!(event["type"], expected["type"]);
                    // Archive annotations and settlement detail differ from
                    // simulator output. Decision events must still agree.
                    if !matches!(event["type"].as_str().unwrap(), "hora" | "ryukyoku") {
                        for key in [
                            "actor",
                            "pai",
                            "consumed",
                            "target",
                            "tsumogiri",
                            "dora_marker",
                        ] {
                            if let Some(value) = expected.get(key) {
                                if key == "consumed" {
                                    let mut actual: Vec<String> =
                                        serde_json::from_value(event[key].clone()).unwrap();
                                    let mut expected: Vec<String> =
                                        serde_json::from_value(value.clone()).unwrap();
                                    actual.sort();
                                    expected.sort();
                                    assert_eq!(
                                        actual,
                                        expected,
                                        "human consumed tiles at {}",
                                        self.logs.len()
                                    );
                                    // History features preserve archive meld
                                    // order, including the red-five position.
                                    event[key] = value.clone();
                                } else {
                                    assert_eq!(
                                        Some(value),
                                        event.get(key),
                                        "human event {} field {key}",
                                        self.logs.len()
                                    );
                                }
                            }
                        }
                    }
                } else {
                    assert_eq!(
                        &event,
                        &recorded[self.logs.len()],
                        "recorded replay diverged at event {}",
                        self.logs.len()
                    );
                }
            }
            let kind = event["type"].as_str().unwrap();
            match kind {
                "start_kyoku" => {
                    if !self.history.start.is_null() {
                        self.hand += 1;
                    }
                    self.pending = [None; 4];
                    self.reached = 0;
                    self.called = 0;
                    self.won = 0;
                    self.dealt = 0;
                    self.cuts = [0; 4];
                    self.riichi_turn_sum = 0;
                }
                "chi" | "pon" | "daiminkan" => {
                    let actor = event["actor"].as_u64().unwrap() as usize;
                    self.called |= 1 << actor;
                    let action_type = match kind {
                        "chi" => ActionType::Chi,
                        "pon" => ActionType::Pon,
                        _ => ActionType::Daiminkan,
                    };
                    for seat in 0..4 {
                        if seat != actor
                            || self.pending[seat].is_some_and(|(_, kind)| kind != action_type)
                        {
                            self.pending[seat] = None;
                        }
                    }
                }
                "tsumo" | "dahai" => {
                    let actor = event["actor"].as_u64().unwrap() as usize;
                    self.pending[actor] = None;
                    if kind == "dahai" {
                        self.cuts[actor] += 1;
                    }
                }
                "reach_accepted" => {
                    let actor = event["actor"].as_u64().unwrap() as usize;
                    self.reached |= 1 << actor;
                    self.riichi_turn_sum += self.cuts[actor];
                }
                "hora" => {
                    self.pending = [None; 4];
                    let actor = event["actor"].as_u64().unwrap() as usize;
                    let target = event["target"].as_u64().unwrap() as usize;
                    self.won |= 1 << actor;
                    add(
                        counts,
                        "win_points",
                        event["deltas"][actor].as_i64().unwrap(),
                    );
                    if actor == target {
                        add(counts, "tsumo_wins", 1);
                    } else {
                        self.dealt |= 1 << target;
                        add(
                            counts,
                            "dealin_points",
                            -event["deltas"][target].as_i64().unwrap(),
                        );
                    }
                }
                "ryukyoku" => {
                    self.pending = [None; 4];
                    add(counts, "draws", 1);
                    if event["reason"] == "exhaustive_draw" {
                        add(counts, "exhaustive_draws", 1);
                        let tenpai = self
                            .history
                            .replay
                            .players
                            .iter()
                            .filter(|p| {
                                HandEvaluator::new(p.hand.clone(), p.melds.clone()).is_tenpai()
                            })
                            .count();
                        add(counts, "exhaustive_tenpai_seats", tenpai as i64);
                    }
                }
                "end_kyoku" => {
                    self.pending = [None; 4];
                    add(counts, "hands", 1);
                    add(counts, "riichi_turn_sum", self.riichi_turn_sum);
                    for (name, seats) in [
                        ("wins", self.won),
                        ("dealin_hands", self.dealt),
                        ("riichi_hands", self.reached),
                        ("open_hands", self.called),
                        ("riichi_wins", self.reached & self.won),
                        ("riichi_dealins", self.reached & self.dealt),
                    ] {
                        add(counts, name, seats.count_ones() as i64);
                    }
                }
                "end_game" => add(counts, "games", 1),
                _ => {}
            }
            if kind != "start_game" {
                self.history.apply(&event);
                if kind == "tsumo" {
                    if let Some(left) = event["tiles_left"].as_i64() {
                        self.history.live_wall = left as i32;
                    }
                }
            }
            self.logs.push(event);
        }
    }
}
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Request {
    game: usize,
    seat: usize,
    obs: Observation,
    phase: u8,
    row: Row,
    legal: candidate::Legal,
    candidates: Vec<actions::Candidate>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[pyclass]
pub struct NativeRoot {
    game: Game,
    requests: Vec<Request>,
    decision: usize,
}
#[pymethods]
impl NativeRoot {
    /// Persist the complete pending simulator state without replay on load.
    fn dumps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = serde_json::to_vec(self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    fn loads(bytes: &[u8]) -> PyResult<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn decision(&self) -> usize {
        self.decision
    }

    /// Keep the full simulator state; optionally resample only unseen wall tiles.
    #[pyo3(signature = (copies, seed, shuffle_remaining=true, *, until_hand_end=false))]
    fn fork(
        &self,
        copies: usize,
        seed: u64,
        shuffle_remaining: bool,
        until_hand_end: bool,
    ) -> PyResult<NativeCollector> {
        if copies == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "copies must be positive",
            ));
        }
        let mut games = Vec::with_capacity(copies);
        let mut requests = Vec::with_capacity(copies * self.requests.len());
        for index in 0..copies {
            let mut game = self.game.clone();
            game.recorded = None;
            if shuffle_remaining {
                let seed = seed.wrapping_add(index as u64);
                let mut rng = StdRng::seed_from_u64(seed);
                let mut unseen = game
                    .engine
                    .wall
                    .tiles
                    .iter()
                    .copied()
                    .filter(|tile| !game.engine.wall.dora_indicators.contains(tile))
                    .collect::<Vec<_>>();
                unseen.shuffle(&mut rng);
                let mut unseen = unseen.into_iter();
                let mut replacements = [0; 136];
                for tile in &mut game.engine.wall.tiles {
                    let original = *tile;
                    if !game.engine.wall.dora_indicators.contains(tile) {
                        *tile = unseen.next().unwrap();
                    }
                    replacements[original as usize] = *tile;
                }
                if let Some(queues) = &mut game.engine.wall.duplicate_draws {
                    // Queue entries refer to physical wall tiles. Apply the
                    // same permutation without changing ownership or order.
                    for tile in queues.iter_mut().flatten() {
                        *tile = replacements[*tile as usize];
                    }
                }
                game.engine.wall.seed = Some(seed);
            }
            games.push(game);
            requests.extend(self.requests.iter().cloned().map(|mut request| {
                request.game = index;
                request
            }));
        }
        Ok(NativeCollector {
            games,
            requests,
            rng: StdRng::seed_from_u64(seed),
            counts: COUNTERS.iter().map(|name| (name.to_string(), 0)).collect(),
            submitted: true,
            stop_after_hand: until_hand_end.then_some(self.game.hand),
        })
    }
}

#[pyclass]
pub struct NativeCollector {
    games: Vec<Game>,
    requests: Vec<Request>,
    rng: StdRng,
    counts: HashMap<String, i64>,
    submitted: bool,
    stop_after_hand: Option<usize>,
}
impl NativeCollector {
    fn prepare(&mut self) {
        if !self.requests.is_empty() {
            return;
        }
        loop {
            for (game, ctx) in self.games.iter_mut().enumerate() {
                ctx.sync(&mut self.counts);
                if ctx.complete(self.stop_after_hand) {
                    continue;
                }
                if ctx.human_hand && ctx.recorded.is_some() {
                    // The archive feature producer consumes a dora event
                    // following rinshan before exposing the discard choice.
                    while ctx.recorded.as_ref().unwrap()[ctx.logs.len()]["type"] == "dora" {
                        assert!(ctx.engine.wall.pending_kan_dora_count > 0);
                        ctx.engine.wall.pending_kan_dora_count -= 1;
                        ctx.engine._reveal_kan_dora();
                        ctx.sync(&mut self.counts);
                    }
                }
                let mut seats = ctx.engine.active_players.clone();
                seats.sort();
                for seat in seats {
                    let obs = ctx.engine.get_observation(seat);
                    let seat = seat as usize;
                    let legal = obs.legal_actions_method();
                    let mut selected = actions::forced_action(
                        &legal,
                        ctx.history.canonical_drawn_tile(seat),
                        &mut self.rng,
                    );
                    if ctx.pending[seat].is_some()
                        && legal.iter().any(|a| a.action_type == ActionType::Discard)
                    {
                        let (cut, _) = ctx.pending[seat].take().unwrap();
                        // Human records also train the ordinary discard
                        // after a call. Expose it while restoring the log;
                        // counterfactual forks still commit joint choices.
                        if selected.is_none() && !(ctx.human_hand && ctx.recorded.is_some()) {
                            selected = Some(actions::select_discard(
                                &legal,
                                cut,
                                obs.drawn_tile,
                                &mut self.rng,
                            ));
                        }
                    }
                    if let Some(action) = selected {
                        ctx.choices.insert(
                            seat as u8,
                            if ctx.recorded.is_some() {
                                ctx.recorded_action(&obs, seat)
                            } else {
                                action
                            },
                        );
                    } else {
                        let phase = actions::phase_for_actions(&legal);
                        let mut row = Row::new();
                        let legal =
                            crate::features::pack(&mut row, &mut ctx.history, &obs, seat, phase);
                        let candidates = actions::candidates(&obs, phase, &legal, &mut self.rng);
                        self.requests.push(Request {
                            game,
                            seat,
                            obs,
                            phase,
                            row,
                            legal,
                            candidates,
                        });
                    }
                }
            }
            if !self.requests.is_empty()
                || self
                    .games
                    .iter()
                    .all(|ctx| ctx.complete(self.stop_after_hand))
            {
                return;
            }
            self.step_games();
        }
    }
    fn step_games(&mut self) {
        for ctx in &mut self.games {
            if !ctx.complete(self.stop_after_hand) {
                ctx.engine.step(&ctx.choices);
                ctx.choices.clear();
                if let Some(error) = ctx.engine.last_error.take() {
                    panic!("native collection: {error}");
                }
            }
        }
    }
}
#[pymethods]
impl NativeCollector {
    #[new]
    fn new(games: usize, seed: u64) -> PyResult<Self> {
        if games == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "games must be positive",
            ));
        }
        Ok(Self {
            games: (0..games)
                .map(|game| Game::new(seed + game as u64, false))
                .collect(),
            requests: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            counts: COUNTERS.iter().map(|name| (name.to_string(), 0)).collect(),
            submitted: true,
            stop_after_hand: None,
        })
    }
    #[staticmethod]
    fn from_match(events: &str, seed: u64, duplicate: bool) -> PyResult<Self> {
        let mut collector = Self::new(1, seed)?;
        collector.games[0] = Game::new(seed, duplicate);
        collector.games[0].recorded = Some(
            serde_json::from_str(events)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
        );
        Ok(collector)
    }

    /// Reconstruct an arbitrary complete human hand using its observed draws.
    /// Unknown wall slots are filled without replacement. Forks subsequently
    /// shuffle all unseen tiles, including unrevealed dora and ura slots.
    #[staticmethod]
    fn from_hand(events: &str, seed: u64) -> PyResult<Self> {
        let mut events: Vec<Value> = serde_json::from_str(events)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        if events.first().map(|e| e["type"].as_str()) != Some(Some("start_kyoku"))
            || events.last().map(|e| e["type"].as_str()) != Some(Some("end_kyoku"))
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "expected one complete hand",
            ));
        }
        let mut remaining: Vec<u8> = (0..136).collect();
        let mut wall = vec![None; 136];
        let mut assign = |slot: usize, tile: &Value| -> PyResult<()> {
            let code = crate::history::code(tile);
            let position = remaining
                .iter()
                .position(|&tile| crate::candidate::code(tile) == code)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("hand exceeds physical tile supply")
                })?;
            wall[slot] = Some(remaining.remove(position));
            Ok(())
        };
        let oya = events[0]["oya"].as_u64().unwrap() as usize;
        let mut dealt = 0;
        for block in 0..4 {
            for offset in 0..4 {
                let seat = (oya + offset) % 4;
                for tile in (block * 4)..((block * 4 + 4).min(13)) {
                    assign(135 - dealt, &events[0]["tehais"][seat][tile])?;
                    dealt += 1;
                }
            }
        }
        assign(4, &events[0]["dora_marker"])?;
        let mut slots: Vec<usize> = (0..84).collect();
        let mut rinshan = false;
        let mut indicators = 1;
        for event in &events[1..] {
            match event["type"].as_str().unwrap() {
                "ankan" | "kakan" | "daiminkan" => rinshan = true,
                "tsumo" => {
                    let slot = if rinshan {
                        slots.remove(0)
                    } else {
                        slots.pop().unwrap()
                    };
                    assign(slot, &event["pai"])?;
                    rinshan = false;
                }
                "dora" => {
                    assign(4 + 2 * indicators, &event["dora_marker"])?;
                    indicators += 1;
                }
                _ => {}
            }
        }
        remaining.shuffle(&mut StdRng::seed_from_u64(seed));
        let wall = wall
            .into_iter()
            .rev()
            .map(|tile| tile.unwrap_or_else(|| remaining.pop().unwrap()))
            .collect();
        let mut game = Game::new(seed, false);
        game.engine = GameState::new(0, true, Some(seed), 0, GameRule::default_tenhou());
        game.engine.event_buffer = Some(Vec::new());
        game.engine._initialize_round(
            oya as u8,
            "ESWN".find(events[0]["bakaze"].as_str().unwrap()).unwrap() as u8,
            events[0]["honba"].as_u64().unwrap() as u8,
            events[0]["kyotaku"].as_u64().unwrap() as u32,
            Some(wall),
            Some(serde_json::from_value(events[0]["scores"].clone()).unwrap()),
        );
        events.push(serde_json::json!({"type": "end_game"}));
        game.recorded = Some(events);
        game.human_hand = true;
        Ok(Self {
            games: vec![game],
            requests: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            counts: COUNTERS.iter().map(|name| (name.to_string(), 0)).collect(),
            submitted: true,
            stop_after_hand: None,
        })
    }

    fn submit_recorded(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut codes = Vec::with_capacity(self.requests.len());
        for request in &mut self.requests {
            let game = &self.games[request.game];
            let action = game.recorded_action(&request.obs, request.seat);
            let followup = if matches!(
                action.action_type,
                ActionType::Riichi | ActionType::Chi | ActionType::Pon
            ) {
                let event = game.recorded.as_ref().unwrap()[game.logs.len()..]
                    .iter()
                    .find(|event| {
                        event["type"] == "dahai"
                            && event["actor"].as_u64() == Some(request.seat as u64)
                    })
                    .unwrap();
                Some(crate::history::code(&event["pai"]))
            } else {
                None
            };
            // Recorded legal actions need not use the policy's canonical
            // red-call or physical-discard representative. Replay the actual
            // action, retaining staged win/kan declines before it is applied.
            let declined = (request.phase == 4
                && !matches!(action.action_type, ActionType::Ron | ActionType::Tsumo))
                || (request.phase == 2
                    && !matches!(action.action_type, ActionType::Ankan | ActionType::Kakan));
            request.candidates = vec![actions::Candidate {
                head: request.candidates[0].head,
                code: 0,
                action: if declined { None } else { Some(action) },
                discard: followup,
            }];
            codes.push(0);
        }
        self.submit(py, codes)
    }
    fn next<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if !self.submitted {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "submit the previous decisions first",
            ));
        }
        py.detach(|| self.prepare());
        if self.requests.is_empty() {
            return Ok(None);
        }
        self.submitted = false;
        let result = PyDict::new(py);
        let features = PyBytes::new_with(py, ROW_BYTES * self.requests.len(), |bytes| {
            for (destination, request) in bytes.chunks_exact_mut(ROW_BYTES).zip(&self.requests) {
                destination.copy_from_slice(&request.row.bytes);
            }
            Ok(())
        })?;
        result.set_item("features", features)?;
        let mut calls = Vec::new();
        let mut owners = Vec::new();
        let mut options = Vec::new();
        for (owner, request) in self.requests.iter().enumerate() {
            for (option, state) in &request.row.post_calls {
                calls.extend_from_slice(&state.bytes);
                owners.push(owner);
                options.push(*option);
            }
        }
        result.set_item("post_call_features", PyBytes::new(py, &calls))?;
        result.set_item("post_call_owner", owners)?;
        result.set_item("post_call_option", options)?;
        for (name, begin) in [("reference_dora_tile_id", 0), ("reference_dora_valid", 5)] {
            let values: Vec<u8> = self
                .requests
                .iter()
                .flat_map(|request| request.row.reference_dora[begin..begin + 5].iter().copied())
                .collect();
            result.set_item(name, PyBytes::new(py, &values))?;
        }
        result.set_item(
            "head",
            self.requests
                .iter()
                .map(|r| r.candidates[0].head)
                .collect::<Vec<_>>(),
        )?;
        for (name, values) in [
            (
                "game",
                self.requests.iter().map(|r| r.game).collect::<Vec<_>>(),
            ),
            (
                "seat",
                self.requests.iter().map(|r| r.seat).collect::<Vec<_>>(),
            ),
            (
                "event",
                self.requests
                    .iter()
                    .map(|r| self.games[r.game].logs.len())
                    .collect::<Vec<_>>(),
            ),
            (
                "hand",
                self.requests
                    .iter()
                    .map(|r| self.games[r.game].hand)
                    .collect::<Vec<_>>(),
            ),
        ] {
            result.set_item(name, values)?;
        }
        let legal = PyBytes::new_with(py, self.requests.len() * 150, |bytes| {
            bytes.fill(0);
            for (index, request) in self.requests.iter().enumerate() {
                for candidate in &request.candidates {
                    bytes[index * 150 + candidate.code] = 1;
                }
            }
            Ok(())
        })?;
        result.set_item("legal", legal)?;
        Ok(Some(result))
    }
    /// Capture after next(), before submit(), including all simultaneous claims.
    #[pyo3(signature = (decision, *, phase=None))]
    fn snapshot(&self, decision: usize, phase: Option<u8>) -> PyResult<NativeRoot> {
        if self.submitted || decision >= self.requests.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "snapshot requires a pending decision index",
            ));
        }
        let game = self.requests[decision].game;
        let mut root = NativeRoot {
            game: self.games[game].clone(),
            decision: self.requests[..decision]
                .iter()
                .filter(|r| r.game == game)
                .count(),
            requests: self
                .requests
                .iter()
                .filter(|r| r.game == game)
                .cloned()
                .collect(),
        };
        if let Some(phase) = phase {
            let request = &mut root.requests[root.decision];
            request.phase = phase;
            request.row = Row::new();
            request.legal = crate::features::pack(
                &mut request.row,
                &mut root.game.history,
                &request.obs,
                request.seat,
                phase,
            );
            request.candidates = actions::candidates(
                &request.obs,
                phase,
                &request.legal,
                &mut StdRng::seed_from_u64(0),
            );
        }
        Ok(root)
    }

    fn submit(&mut self, py: Python<'_>, chosen: Vec<usize>) -> PyResult<()> {
        if self.submitted || chosen.len() != self.requests.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "one selected action per pending decision is required",
            ));
        }
        if self.requests.iter().zip(&chosen).any(|(request, code)| {
            !request
                .candidates
                .iter()
                .any(|candidate| candidate.code == *code)
        }) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "illegal selected action",
            ));
        }
        py.detach(|| {
            let mut selected_codes = chosen.into_iter();
            // Declining a win or kan continues with the remaining choices.
            self.requests.retain_mut(|request| {
                let selected_code = selected_codes.next().unwrap();
                let candidate = request
                    .candidates
                    .iter_mut()
                    .find(|candidate| candidate.code == selected_code)
                    .unwrap();
                if let Some(action) = candidate.action.take() {
                    if matches!(
                        action.action_type,
                        ActionType::Riichi | ActionType::Chi | ActionType::Pon
                    ) {
                        self.games[request.game].pending[request.seat] =
                            Some((candidate.discard.unwrap(), action.action_type));
                    }
                    self.games[request.game]
                        .choices
                        .insert(request.seat as u8, action);
                    false
                } else {
                    assert!(matches!(request.phase, 2 | 4));
                    let legal = request
                        .obs
                        .legal_actions_method()
                        .into_iter()
                        .filter(|a| {
                            if request.phase == 4 {
                                !matches!(a.action_type, ActionType::Ron | ActionType::Tsumo)
                            } else {
                                !matches!(a.action_type, ActionType::Ankan | ActionType::Kakan)
                            }
                        })
                        .collect::<Vec<_>>();
                    request.phase = actions::phase_for_actions(&legal);
                    request.obs.replace_legal_actions(legal);
                    request.row.set("decision_phase", 0, request.phase as f64);
                    request.row.set("win_kind", 0, 0.0);
                    request.candidates = actions::candidates(
                        &request.obs,
                        request.phase,
                        &request.legal,
                        &mut self.rng,
                    );
                    true
                }
            });
            if self.requests.is_empty() {
                self.step_games();
            }
        });
        self.submitted = true;
        Ok(())
    }
    fn finish(&mut self) -> PyResult<(Vec<String>, HashMap<String, i64>)> {
        if !self.games.iter().all(|g| g.complete(self.stop_after_hand)) || !self.requests.is_empty()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "collection is not complete",
            ));
        }
        for game in &mut self.games {
            game.sync(&mut self.counts);
            if let Some(recorded) = &game.recorded {
                assert_eq!(
                    game.logs.len(),
                    recorded.len(),
                    "recorded match ended early"
                );
            }
        }
        Ok((
            self.games
                .iter()
                .map(|g| {
                    let end = if g.engine.is_done {
                        g.logs.len()
                    } else {
                        // The following start identifies the real continuation;
                        // it is not a terminal game or a completed extra hand.
                        g.logs
                            .iter()
                            .rposition(|event| event["type"] == "start_kyoku")
                            .unwrap()
                            + 1
                    };
                    serde_json::to_string(&g.logs[..end]).unwrap()
                })
                .collect(),
            self.counts.clone(),
        ))
    }
    fn logs(&self) -> Vec<String> {
        self.games
            .iter()
            .map(|g| serde_json::to_string(&g.logs).unwrap())
            .collect()
    }
}

#[cfg(test)]
mod root_tests {
    use super::*;

    #[test]
    fn recorded_triple_ron_restores_all_three_claims() {
        let mut game = Game::new(941, false);
        game.recorded = Some(vec![serde_json::json!({
            "type": "ryukyoku", "reason": "sanchaho", "deltas": [0, 0, 0, 0]
        })]);
        for actor in 1..4 {
            let mut observation = game.engine.get_observation(actor);
            observation.replace_legal_actions(vec![
                Action::new(ActionType::Pass, None, vec![], None),
                Action::new(ActionType::Ron, Some(76), vec![], Some(0)),
            ]);
            assert_eq!(
                game.recorded_action(&observation, actor as usize)
                    .action_type,
                ActionType::Ron
            );
        }
    }

    #[test]
    fn shuffled_duplicate_roots_preserve_queues_and_shared_rinshan() {
        let mut collector = NativeCollector::new(1, 941).unwrap();
        collector.games[0] = Game::new(941, true);
        collector.prepare();
        collector.submitted = false;
        let root = collector.snapshot(0, None).unwrap();
        let mut branches = root.fork(2, 17, true, true).unwrap();
        let paired = root.fork(1, 17, true, true).unwrap();
        assert_eq!(
            branches.games[0].engine.wall.tiles,
            paired.games[0].engine.wall.tiles
        );
        assert_eq!(
            branches.games[0].engine.wall.duplicate_draws,
            paired.games[0].engine.wall.duplicate_draws
        );
        let original = &root.game.engine.wall;
        for branch in &mut branches.games {
            let wall = &mut branch.engine.wall;
            let replacement: HashMap<_, _> = original
                .tiles
                .iter()
                .copied()
                .zip(wall.tiles.iter().copied())
                .collect();
            for (before, after) in original
                .duplicate_draws
                .as_ref()
                .unwrap()
                .iter()
                .zip(wall.duplicate_draws.as_ref().unwrap())
            {
                assert_eq!(
                    after,
                    &before
                        .iter()
                        .map(|tile| replacement[tile])
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(wall.dora_indicators, original.dora_indicators);
            let mut drawn = Vec::new();
            for player in 0..4 {
                let rinshan = wall.tiles[0];
                assert_eq!(wall.draw(player, true), Some(rinshan));
                drawn.push(rinshan);
                drawn.push(wall.draw(player, false).unwrap());
            }
            while wall.drawable_count > 0 {
                // Deplete one queue and exercise borrowing from queue backs.
                drawn.push(wall.draw(0, false).unwrap());
            }
            let mut unique = drawn.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), drawn.len());
            assert_eq!(wall.tiles.len(), 14);
        }
    }

    #[test]
    fn snapshots_preserve_state_and_shuffle_only_unseen_wall() {
        let mut collector = NativeCollector::new(1, 941).unwrap();
        collector.prepare();
        collector.submitted = false;
        let root = collector.snapshot(0, None).unwrap();
        let saved = serde_json::to_vec(&root).unwrap();
        let restored = NativeRoot::loads(&saved).unwrap();
        assert_eq!(
            serde_json::to_value(&root).unwrap(),
            serde_json::to_value(&restored).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&root.fork(8, 17, true, true).unwrap().games).unwrap(),
            serde_json::to_value(&restored.fork(8, 17, true, true).unwrap().games).unwrap()
        );
        let unchanged = restored.fork(1, 17, false, false).unwrap();
        assert_eq!(
            format!("{:?}", collector.games[0].engine),
            format!("{:?}", unchanged.games[0].engine)
        );
        assert_eq!(collector.games[0].logs, unchanged.games[0].logs);
        assert_eq!(
            collector.requests[0].row.bytes,
            unchanged.requests[0].row.bytes
        );
        let branches = root.fork(2, 17, true, false).unwrap();
        for branch in &branches.games {
            let mut before = collector.games[0].engine.wall.tiles.clone();
            let mut after = branch.engine.wall.tiles.clone();
            before.sort();
            after.sort();
            assert_eq!(before, after);
            for (index, tile) in collector.games[0].engine.wall.tiles.iter().enumerate() {
                if collector.games[0]
                    .engine
                    .wall
                    .dora_indicators
                    .contains(tile)
                {
                    assert_eq!(*tile, branch.engine.wall.tiles[index]);
                }
            }
            assert_eq!(
                branch.engine.wall.drawable_count,
                collector.games[0].engine.wall.drawable_count
            );
            assert_eq!(
                format!("{:?}", branch.engine.players),
                format!("{:?}", collector.games[0].engine.players)
            );
        }
        assert_ne!(
            branches.games[0].engine.wall.tiles,
            branches.games[1].engine.wall.tiles
        );
        assert_eq!(
            format!("{:?}", root.game.engine),
            format!("{:?}", collector.games[0].engine)
        );
    }
}
