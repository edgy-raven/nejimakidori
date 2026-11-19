use crate::{
    actions, candidate,
    history::PublicHistory,
    nyanten,
    packed::{ROW_BYTES, Row},
    scoring,
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
    types::{Conditions, Wind},
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
#[derive(Clone)]
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
    fn ron_labels(&self, request: &Request) -> (Vec<Vec<i8>>, Vec<Vec<i32>>) {
        let mut legal = vec![vec![-1; 37]; 3];
        let mut payments = vec![vec![-1; 37]; 3];
        if !matches!(request.phase, 0 | 2 | 3) {
            return (legal, payments);
        }
        let cuts: Vec<_> = request
            .obs
            .legal_actions_method()
            .into_iter()
            .filter(|action| {
                if request.phase == 2 {
                    matches!(action.action_type, ActionType::Ankan | ActionType::Kakan)
                } else {
                    action.action_type == ActionType::Discard
                }
            })
            .collect();
        let mut dora = self.engine.wall.dora_indicators.clone();
        // Discard and kakan expose deferred indicators before any ron settlement.
        for index in
            dora.len()..(dora.len() + self.engine.wall.pending_kan_dora_count as usize).min(5)
        {
            dora.push(
                self.engine.wall.tiles
                    [(4 + 2 * index).saturating_sub(self.engine.wall.dead_wall_shift())],
            );
        }
        let ura: Vec<_> = (0..dora.len())
            .map(|index| {
                self.engine.wall.tiles
                    [(5 + 2 * index).saturating_sub(self.engine.wall.dead_wall_shift())]
            })
            .collect();
        for offset in 1..4 {
            let seat = (request.seat + offset) % 4;
            let player = &self.engine.players[seat];
            for action in &cuts {
                let code = if request.phase == 2 {
                    (action.tile.unwrap() / 4) as usize
                } else {
                    candidate::code(action.tile.unwrap())
                };
                legal[offset - 1][code] = 0;
                payments[offset - 1][code] = 0;
            }
            if player.hand.len() % 3 != 1
                || player.missed_agari_riichi
                || player.missed_agari_doujun
            {
                continue;
            }
            let evaluator = HandEvaluator::new(player.hand.clone(), player.melds.clone());
            let waits = evaluator.get_waits_u8();
            if player
                .discards
                .iter()
                .any(|tile| waits.contains(&(tile / 4)))
            {
                continue;
            }
            for action in &cuts {
                let tile = action.tile.unwrap();
                if !waits.contains(&(tile / 4)) {
                    continue;
                }
                let scored = scoring::normalize_score(
                    evaluator.calc_with_indicators(
                        tile,
                        &dora,
                        if player.riichi_declared { &ura } else { &[] },
                        Some(Conditions {
                            riichi: player.riichi_declared,
                            double_riichi: player.double_riichi_declared,
                            ippatsu: player.ippatsu_cycle,
                            houtei: self.engine.wall.drawable_count == 0
                                && !self.engine.is_rinshan_flag
                                && request.phase != 2,
                            chankan: request.phase == 2,
                            player_wind: Wind::from((seat as u8 + 4 - self.engine.oya) % 4),
                            round_wind: Wind::from(self.engine.round_wind),
                            ..Conditions::default()
                        }),
                    ),
                    seat as u8 == self.engine.oya,
                    false,
                );
                if !scored.is_win
                    || (action.action_type == ActionType::Ankan
                        && !(scored.yaku.contains(&42) || scored.yaku.contains(&49)))
                {
                    continue;
                }
                let code = if request.phase == 2 {
                    (tile / 4) as usize
                } else {
                    candidate::code(tile)
                };
                legal[offset - 1][code] = 1;
                // Each claimant includes honba; settlement assigns it once by seat priority.
                payments[offset - 1][code] = if player.pao.is_empty() {
                    scored.ron_agari as i32 + 300 * self.engine.honba as i32
                } else {
                    -1
                };
            }
        }
        for code in 0..37 {
            if legal.iter().all(|opponent| opponent[code] == 1) {
                for opponent in &mut payments {
                    opponent[code] = -1;
                }
            }
        }
        (legal, payments)
    }
    fn sync(&mut self, counts: &mut HashMap<String, i64>) {
        for event in self.engine.drain_events() {
            if let Some(recorded) = &self.recorded {
                assert_eq!(
                    &event,
                    &recorded[self.logs.len()],
                    "recorded replay diverged at event {}",
                    self.logs.len()
                );
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
/// Build a valid physical wall using only the current-hand observed prefix.
fn expert_wall(events: &[Value]) -> Vec<u8> {
    let mut assigned = [None; 136];
    let mut available = (0..136u8).collect::<Vec<_>>();
    let mut put = |position: usize, value: &Value| {
        let code = crate::history::code(value);
        let index = available
            .iter()
            .position(|&tile| crate::history::tile_code(tile) == code)
            .expect("expert prefix contains too many copies of a tile");
        assert!(assigned[position].is_none());
        assigned[position] = Some(available.remove(index));
    };
    let start = &events[0];
    assert_eq!(start["type"], "start_kyoku");
    let dealer = start["oya"].as_u64().unwrap() as usize;
    let mut position = 0;
    for block in 0..3 {
        for offset in 0..4 {
            for tile in 0..4 {
                put(
                    position,
                    &start["tehais"][(dealer + offset) % 4][block * 4 + tile],
                );
                position += 1;
            }
        }
    }
    for offset in 0..4 {
        put(position, &start["tehais"][(dealer + offset) % 4][12]);
        position += 1;
    }
    put(131, &start["dora_marker"]);
    let mut rinshan = false;
    let mut replacements = 0;
    let mut markers = 1;
    for event in &events[1..] {
        match event["type"].as_str().unwrap() {
            "ankan" | "kakan" | "daiminkan" => rinshan = true,
            "tsumo" => {
                if rinshan {
                    put(135 - replacements, &event["pai"]);
                    replacements += 1;
                    rinshan = false;
                } else {
                    put(position, &event["pai"]);
                    position += 1;
                }
            }
            "dora" => {
                put(131 - 2 * markers, &event["dora_marker"]);
                markers += 1;
            }
            _ => {}
        }
    }
    // Unobserved positions are placeholders until NativeRoot::fork shuffles
    // their complement once. Do not compose two permutations with one seed.
    let mut remaining = available.into_iter();
    assigned
        .into_iter()
        .map(|tile| tile.unwrap_or_else(|| remaining.next().unwrap()))
        .collect()
}

#[derive(Clone)]
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
pub struct NativeRoot {
    game: Game,
    requests: Vec<Request>,
    decision: usize,
}
#[pymethods]
impl NativeRoot {
    /// Reconstruct simulation flags by executing the observed hand prefix.
    #[staticmethod]
    fn from_expert_prefix(events: &str, actor: usize, phase: u8, seed: u64) -> PyResult<Self> {
        let events: Vec<Value> = serde_json::from_str(events)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let mut collector = NativeCollector::new(1, seed)?;
        let game = &mut collector.games[0];
        game.engine.event_buffer = Some(Vec::new());
        game.engine._initialize_round(
            events[0]["oya"].as_u64().unwrap() as u8,
            match events[0]["bakaze"].as_str().unwrap() {
                "E" => 0,
                "S" => 1,
                "W" => 2,
                "N" => 3,
                _ => unreachable!(),
            },
            events[0]["honba"].as_u64().unwrap() as u8,
            events[0]["kyotaku"].as_u64().unwrap() as u32,
            Some(expert_wall(&events)),
            Some(
                events[0]["scores"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap() as i32)
                    .collect(),
            ),
        );
        loop {
            let begin = game.logs.len();
            game.sync(&mut collector.counts);
            for (index, actual) in game.logs.iter().enumerate().skip(begin) {
                let expected = &events[index];
                for (key, value) in expected.as_object().unwrap() {
                    if key == "tehais" || key == "consumed" {
                        // Physical copies and consumed-tile order are irrelevant.
                        let mut actual = actual[key].clone();
                        let mut expected = value.clone();
                        if key == "tehais" {
                            for hand in actual.as_array_mut().unwrap() {
                                hand.as_array_mut().unwrap().sort_by_key(|v| v.to_string());
                            }
                            for hand in expected.as_array_mut().unwrap() {
                                hand.as_array_mut().unwrap().sort_by_key(|v| v.to_string());
                            }
                        } else {
                            actual
                                .as_array_mut()
                                .unwrap()
                                .sort_by_key(|v| v.to_string());
                            expected
                                .as_array_mut()
                                .unwrap()
                                .sort_by_key(|v| v.to_string());
                        }
                        assert_eq!(actual, expected, "expert prefix event {index} field {key}");
                    } else {
                        assert_eq!(
                            &actual[key], value,
                            "expert prefix event {index} field {key}"
                        );
                    }
                }
            }
            if game.logs.len() == events.len() {
                break;
            }
            if events[game.logs.len()]["type"] == "dora" {
                // Expert logs may reveal an open-kan indicator immediately
                // after rinshan, while this simulator defers it to discard.
                assert!(game.engine.wall.pending_kan_dora_count > 0);
                game.engine.wall.pending_kan_dora_count -= 1;
                game.engine._reveal_kan_dora();
                continue;
            }
            game.recorded = Some(events.clone());
            let choices = game
                .engine
                .active_players
                .clone()
                .into_iter()
                .map(|seat| {
                    let obs = game.engine.get_observation(seat);
                    (seat, game.recorded_action(&obs, seat as usize))
                })
                .collect();
            game.recorded = None;
            game.engine.step(&choices);
            if let Some(error) = game.engine.last_error.take() {
                return Err(pyo3::exceptions::PyValueError::new_err(error));
            }
        }
        // Preserve the corpus's public representation (including consumed-tile
        // order), while retaining the fully simulated hidden transition state.
        game.history = PublicHistory::new();
        for event in &events {
            game.history.apply(event);
        }
        game.logs = events;
        collector.prepare();
        let decision = collector
            .requests
            .iter()
            .position(|request| request.seat == actor)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("expert root has no policy decision")
            })?;
        assert_eq!(
            collector.requests[decision].phase, phase,
            "expert root decision phase"
        );
        collector.submitted = false;
        collector.snapshot(decision)
    }

    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.requests[self.decision].row.bytes)
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
            let candidate = request
                .candidates
                .iter_mut()
                .find(|candidate| {
                    if let Some(selected) = &candidate.action {
                        selected.action_type == action.action_type
                            && selected.to_mjai() == action.to_mjai()
                            && (followup.is_none() || candidate.discard == followup)
                    } else {
                        (request.phase == 4
                            && !matches!(action.action_type, ActionType::Ron | ActionType::Tsumo))
                            || (request.phase == 2
                                && !matches!(
                                    action.action_type,
                                    ActionType::Ankan | ActionType::Kakan
                                ))
                    }
                })
                .unwrap_or_else(|| {
                    panic!("recorded candidate missing at event {}", game.logs.len())
                });
            if candidate.action.is_some() {
                candidate.action = Some(action);
            }
            codes.push(candidate.code);
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
        // Hidden state is a training target only, never part of feature rows.
        let mut shanten = Vec::new();
        let mut ukeire = Vec::new();
        let mut hand_counts = Vec::new();
        let mut ron_legal = Vec::new();
        let mut ron_payments = Vec::new();
        for request in &self.requests {
            let (legal, payments) = self.games[request.game].ron_labels(request);
            ron_legal.push(legal);
            ron_payments.push(payments);
            let mut distances = Vec::new();
            let mut improving = Vec::new();
            let mut holdings = Vec::new();
            for offset in 1..4 {
                let player =
                    &self.games[request.game].history.replay.players[(request.seat + offset) % 4];
                let mut concealed = [0u8; 34];
                let mut counts = vec![0i8; 37];
                for &tile in &player.hand {
                    concealed[(tile / 4) as usize] += 1;
                    counts[candidate::code(tile)] += 1;
                }
                for meld in &player.melds {
                    for &tile in &meld.tiles {
                        counts[candidate::code(tile)] += 1;
                    }
                }
                let (distance, waits) = if player.hand.len() % 3 == 1 {
                    let (distance, mask) = nyanten::improving(&concealed);
                    (
                        distance.min(3) as i8,
                        (0..34).map(|tile| ((mask >> tile) & 1) as i8).collect(),
                    )
                } else {
                    (-1, vec![-1i8; 34])
                };
                distances.push(distance);
                improving.push(waits);
                holdings.push(counts);
            }
            shanten.push(distances);
            ukeire.push(improving);
            hand_counts.push(holdings);
        }
        result.set_item("ron_legal", ron_legal)?;
        result.set_item("ron_payments", ron_payments)?;
        result.set_item("opponent_shanten", shanten)?;
        result.set_item("opponent_ukeire", ukeire)?;
        result.set_item("opponent_hand_counts", hand_counts)?;
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
    fn snapshot(&self, decision: usize) -> PyResult<NativeRoot> {
        if self.submitted || decision >= self.requests.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "snapshot requires a pending decision index",
            ));
        }
        let game = self.requests[decision].game;
        Ok(NativeRoot {
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
        })
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
    fn observations(&self) -> Vec<String> {
        self.requests
            .iter()
            .map(|r| serde_json::to_string(&r.obs).unwrap())
            .collect()
    }
}

#[cfg(test)]
mod root_tests {
    use super::*;

    #[test]
    fn expert_prefix_preserves_public_state_and_physical_inventory() {
        let mut collector = NativeCollector::new(1, 941).unwrap();
        collector.prepare();
        let request = &collector.requests[0];
        let events = serde_json::to_string(&collector.games[0].history.events).unwrap();
        let root =
            NativeRoot::from_expert_prefix(&events, request.seat, request.phase, 123).unwrap();
        assert_eq!(root.requests[root.decision].row.bytes, request.row.bytes);
        let other_seed =
            NativeRoot::from_expert_prefix(&events, request.seat, request.phase, 124).unwrap();
        assert_eq!(
            root.game.engine.wall.tiles,
            other_seed.game.engine.wall.tiles
        );
        let mut inventory = root.game.engine.wall.tiles.clone();
        for player in &root.game.engine.players {
            inventory.extend(&player.hand);
        }
        inventory.sort();
        assert_eq!(inventory, (0..136u8).collect::<Vec<_>>());
        let first = root.fork(1, 17, true, true).unwrap();
        let second = root.fork(1, 17, true, true).unwrap();
        assert_eq!(
            first.games[0].engine.wall.tiles,
            second.games[0].engine.wall.tiles
        );
        assert_eq!(
            first.games[0].engine.wall.dora_indicators,
            root.game.engine.wall.dora_indicators
        );
        for (before, after) in root
            .game
            .engine
            .players
            .iter()
            .zip(&first.games[0].engine.players)
        {
            assert_eq!(before.hand, after.hand);
        }
    }

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
        let root = collector.snapshot(0).unwrap();
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
    fn counterfactual_ron_scores_honba_and_masks_furiten_and_response() {
        let mut collector = NativeCollector::new(1, 941).unwrap();
        collector.prepare();
        let mut request = collector.requests[0].clone();
        request.seat = 0;
        request.phase = 0;
        request.obs.replace_legal_actions(vec![Action::new(
            ActionType::Discard,
            Some(116),
            vec![],
            Some(0),
        )]);
        let game = &mut collector.games[request.game];
        game.engine.oya = 0;
        game.engine.round_wind = 0;
        game.engine.honba = 2;
        game.engine.wall.dora_indicators = vec![120];
        game.engine.wall.pending_kan_dora_count = 0;
        game.engine.wall.drawable_count = 30;
        for player in &mut game.engine.players {
            player.hand.clear();
        }
        game.engine.players[1].hand = vec![0, 4, 8, 36, 40, 44, 96, 100, 104, 124, 125, 126, 117];
        let (legal, payments) = game.ron_labels(&request);
        assert_eq!(legal[0][29], 1);
        // Closed chanta + green dragon: 3 han, 40 fu, plus two honba.
        assert_eq!(payments[0][29], 5800);
        assert_eq!(legal[0][28], -1);
        game.engine.players[1].missed_agari_doujun = true;
        let (legal, payments) = game.ron_labels(&request);
        assert_eq!((legal[0][29], payments[0][29]), (0, 0));
        game.engine.players[1].missed_agari_doujun = false;
        game.engine.players[1].discards.push(118);
        assert_eq!(game.ron_labels(&request).0[0][29], 0);
        request.phase = 1;
        let (legal, payments) = game.ron_labels(&request);
        assert!(legal.iter().flatten().all(|&value| value == -1));
        assert!(payments.iter().flatten().all(|&value| value == -1));
    }

    #[test]
    fn snapshots_preserve_state_and_shuffle_only_unseen_wall() {
        let mut collector = NativeCollector::new(1, 941).unwrap();
        collector.prepare();
        collector.submitted = false;
        let root = collector.snapshot(0).unwrap();
        let unchanged = root.fork(1, 17, false, false).unwrap();
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
