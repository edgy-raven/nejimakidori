use crate::{
    actions, candidate,
    history::PublicHistory,
    packed::{ROW_BYTES, Row},
};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
};
use rand::{SeedableRng, rngs::StdRng};
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
}
impl Game {
    fn new(seed: u64) -> Self {
        let mut engine = GameState::new(2, true, Some(seed), 0, GameRule::default_tenhou());
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
        }
    }
    fn sync(&mut self, counts: &mut HashMap<String, i64>) {
        for event in self.engine.drain_events() {
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
                    add(counts, "hands", 1);
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
                    add(counts, "riichi_turn_sum", self.cuts[actor]);
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
struct Request {
    game: usize,
    seat: usize,
    obs: Observation,
    phase: u8,
    row: Row,
    legal: candidate::Legal,
    candidates: Vec<actions::Candidate>,
}
#[pyclass]
pub struct NativeCollector {
    games: Vec<Game>,
    requests: Vec<Request>,
    rng: StdRng,
    counts: HashMap<String, i64>,
    submitted: bool,
}
impl NativeCollector {
    fn prepare(&mut self) {
        if !self.requests.is_empty() {
            return;
        }
        loop {
            for (game, ctx) in self.games.iter_mut().enumerate() {
                ctx.sync(&mut self.counts);
                if ctx.engine.is_done {
                    continue;
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
                        if selected.is_none() {
                            selected = Some(actions::select_discard(
                                &legal,
                                cut,
                                obs.drawn_tile,
                                &mut self.rng,
                            ));
                        }
                    }
                    if let Some(action) = selected {
                        ctx.choices.insert(seat as u8, action);
                    } else {
                        let phase = actions::phase_for_actions(&legal);
                        let mut row = Row::new();
                        let (legal, _) = crate::features::pack(
                            &mut row,
                            &mut ctx.history,
                            &obs,
                            seat,
                            phase,
                            false,
                            false,
                        );
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
            if !self.requests.is_empty() || self.games.iter().all(|ctx| ctx.engine.is_done) {
                return;
            }
            self.step_games();
        }
    }
    fn step_games(&mut self) {
        for ctx in &mut self.games {
            if !ctx.engine.is_done {
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
                .map(|game| Game::new(seed + game as u64))
                .collect(),
            requests: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            counts: COUNTERS.iter().map(|name| (name.to_string(), 0)).collect(),
            submitted: true,
        })
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
        if !self.games.iter().all(|g| g.engine.is_done) || !self.requests.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "collection is not complete",
            ));
        }
        for game in &mut self.games {
            game.sync(&mut self.counts);
        }
        Ok((
            self.games
                .iter()
                .map(|g| serde_json::to_string(&g.logs).unwrap())
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
    fn observations(&self) -> Vec<String> {
        self.requests
            .iter()
            .map(|r| serde_json::to_string(&r.obs).unwrap())
            .collect()
    }
}
