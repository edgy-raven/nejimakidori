use rand::{RngExt, rngs::StdRng};
use riichienv_core::action::{Action, ActionType};
use riichienv_core::observation::Observation;

use crate::candidate::Legal;
pub use crate::candidate::code as tile_code;
use crate::candidate::prefer_red_calls;

pub struct Candidate {
    pub head: usize,
    pub code: usize,
    pub action: Option<Action>,
    pub discard: Option<usize>,
}

pub fn phase_for_actions(legal: &[Action]) -> u8 {
    for (phase, kinds) in [
        (4, &[ActionType::Ron, ActionType::Tsumo][..]),
        (2, &[ActionType::Ankan, ActionType::Kakan][..]),
        (3, &[ActionType::Riichi][..]),
        (0, &[ActionType::Discard][..]),
    ] {
        if legal
            .iter()
            .any(|action| kinds.contains(&action.action_type))
        {
            return phase;
        }
    }
    1
}

pub fn select_discard(
    legal: &[Action],
    code: usize,
    drawn: Option<u8>,
    rng: &mut StdRng,
) -> Action {
    let matching = legal.iter().filter(|action| {
        action.action_type == ActionType::Discard && tile_code(action.tile.unwrap()) == code
    });
    let mut origin_counts = [0usize; 2];
    for action in matching.clone() {
        origin_counts[usize::from(action.tile == drawn)] += 1;
    }
    // Keep held/drawn origins equally likely, then choose a physical copy.
    // Preserve the two RNG calls and their bounds for reproducible rollouts.
    let origins = origin_counts
        .into_iter()
        .enumerate()
        .filter(|&(_, count)| count > 0);
    let (origin, copies) = origins
        .clone()
        .nth(rng.random_range(0..origins.count()))
        .unwrap();
    matching
        .filter(|action| usize::from(action.tile == drawn) == origin)
        .nth(rng.random_range(0..copies))
        .unwrap()
        .clone()
}

pub fn forced_action(legal: &[Action], drawn: Option<u8>, rng: &mut StdRng) -> Option<Action> {
    if let Some(action) = legal
        .iter()
        .find(|action| action.action_type == ActionType::KyushuKyuhai)
    {
        return Some(action.clone());
    }
    if legal.len() == 1 {
        return Some(legal[0].clone());
    }
    let first = legal.first()?;
    if first.action_type == ActionType::Discard {
        let code = tile_code(first.tile.unwrap());
        if legal.iter().all(|action| {
            action.action_type == ActionType::Discard && tile_code(action.tile.unwrap()) == code
        }) {
            return Some(select_discard(legal, code, drawn, rng));
        }
    } else if legal
        .iter()
        .all(|action| action.to_mjai() == first.to_mjai())
    {
        return Some(first.clone());
    }
    None
}

pub fn candidates(
    obs: &Observation,
    phase: u8,
    choices: &Legal,
    rng: &mut StdRng,
) -> Vec<Candidate> {
    let legal = obs.legal_actions_method();
    let mut result = Vec::new();
    match phase {
        0 | 3 => {
            for code in 0..37 {
                if !choices.discards[code] {
                    continue;
                }
                let discard = select_discard(&legal, code, obs.drawn_tile, rng);
                result.push(Candidate {
                    head: usize::from(phase == 3),
                    code,
                    action: Some(discard),
                    discard: if phase == 3 { Some(code) } else { None },
                });
                if phase == 3 && choices.riichi[code] {
                    result.push(Candidate {
                        head: 1,
                        code: 37 + code,
                        action: Some(
                            legal
                                .iter()
                                .find(|action| action.action_type == ActionType::Riichi)
                                .unwrap()
                                .clone(),
                        ),
                        discard: Some(code),
                    });
                }
            }
        }
        1 => {
            for action in prefer_red_calls(legal) {
                match action.action_type {
                    ActionType::Pass | ActionType::Daiminkan => result.push(Candidate {
                        head: 2,
                        code: if action.action_type == ActionType::Pass {
                            0
                        } else {
                            149
                        },
                        action: Some(action),
                        discard: None,
                    }),
                    ActionType::Chi | ActionType::Pon => {
                        let chii = action.action_type == ActionType::Chi;
                        let option = if chii {
                            (action.tile.unwrap() / 4
                                - action
                                    .consume_tiles
                                    .iter()
                                    .map(|tile| tile / 4)
                                    .chain([action.tile.unwrap() / 4])
                                    .min()
                                    .unwrap()) as usize
                        } else {
                            0
                        };
                        for cut in 0..37 {
                            if choices.calls[if chii { option } else { 3 }][cut] {
                                result.push(Candidate {
                                    head: 2,
                                    code: (if chii { 1 } else { 112 }) + option * 37 + cut,
                                    action: Some(action.clone()),
                                    discard: Some(cut),
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        2 => {
            result.push(Candidate {
                head: 3,
                code: 0,
                action: None,
                discard: None,
            });
            for action in legal {
                if [ActionType::Ankan, ActionType::Kakan].contains(&action.action_type) {
                    result.push(Candidate {
                        head: 3,
                        code: 1 + action.tile.unwrap() as usize / 4,
                        action: Some(action),
                        discard: None,
                    });
                }
            }
        }
        4 => {
            if legal
                .iter()
                .any(|action| !matches!(action.action_type, ActionType::Ron | ActionType::Tsumo))
            {
                result.push(Candidate {
                    head: 4,
                    code: 0,
                    action: None,
                    discard: None,
                });
            }
            result.push(Candidate {
                head: 4,
                code: 1,
                action: Some(
                    legal
                        .into_iter()
                        .find(|action| {
                            matches!(action.action_type, ActionType::Ron | ActionType::Tsumo)
                        })
                        .unwrap(),
                ),
                discard: None,
            });
        }
        _ => unreachable!(),
    }
    let mut unique: Vec<Candidate> = Vec::new();
    for candidate in result {
        if let Some(previous) = unique
            .iter_mut()
            .find(|previous| previous.head == candidate.head && previous.code == candidate.code)
        {
            *previous = candidate;
        } else {
            unique.push(candidate);
        }
    }
    unique
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use riichienv_core::{rule::GameRule, state::GameState};

    #[test]
    fn physical_discard_origins_are_balanced_and_wins_are_optional() {
        let mut rng = StdRng::seed_from_u64(17);
        let legal: Vec<Action> = (0..4)
            .map(|tile| Action::new(ActionType::Discard, Some(tile), vec![], Some(0)))
            .collect();
        let drawn = (0..10000)
            .filter(|_| select_discard(&legal, 0, Some(3), &mut rng).tile == Some(3))
            .count();
        assert!((4800..5200).contains(&drawn));
        assert!(forced_action(&legal, Some(3), &mut rng).is_some());
        let mut mixed = legal;
        mixed.push(Action::new(ActionType::Tsumo, Some(3), vec![], Some(0)));
        assert_eq!(phase_for_actions(&mixed), 4);
        assert!(forced_action(&mixed, Some(3), &mut rng).is_none());
    }

    #[test]
    fn policy_codes_preserve_paired_discard_decisions() {
        let mut state = GameState::new(2, true, Some(17), 0, GameRule::default_tenhou());
        let mut obs = state.get_observation(0);
        let mut rng = StdRng::seed_from_u64(17);
        let mut choices = Legal {
            discards: [false; 37],
            riichi: [false; 37],
            calls: [[false; 37]; 4],
        };
        choices.discards[0] = true;
        choices.riichi[0] = true;
        obs.replace_legal_actions(vec![
            Action::new(ActionType::Discard, Some(0), vec![], Some(0)),
            Action::new(ActionType::Riichi, None, vec![], Some(0)),
        ]);
        assert_eq!(phase_for_actions(&obs.legal_actions_method()), 3);
        assert_eq!(
            candidates(&obs, 3, &choices, &mut rng)
                .iter()
                .map(|c| (c.head, c.code, c.discard))
                .collect::<Vec<_>>(),
            vec![(1, 0, Some(0)), (1, 37, Some(0))]
        );
        obs.replace_legal_actions(vec![
            Action::new(ActionType::Pass, None, vec![], Some(0)),
            Action::new(ActionType::Chi, Some(8), vec![0, 4], Some(0)),
            Action::new(ActionType::Pon, Some(8), vec![9, 10], Some(0)),
            Action::new(ActionType::Daiminkan, Some(8), vec![9, 10, 11], Some(0)),
        ]);
        choices.calls[2][34] = true;
        choices.calls[3][4] = true;
        assert_eq!(
            candidates(&obs, 1, &choices, &mut rng)
                .iter()
                .map(|c| (c.head, c.code, c.discard))
                .collect::<Vec<_>>(),
            vec![
                (2, 0, None),
                (2, 109, Some(34)),
                (2, 116, Some(4)),
                (2, 149, None)
            ]
        );
        obs.replace_legal_actions(vec![Action::new(
            ActionType::Ankan,
            Some(8),
            vec![8, 9, 10, 11],
            Some(0),
        )]);
        let options = candidates(&obs, 2, &choices, &mut rng);
        assert_eq!(
            options.iter().map(|c| (c.head, c.code)).collect::<Vec<_>>(),
            vec![(3, 0), (3, 3)]
        );
        assert!(options[0].action.is_none());
        for (win, continuation) in [
            (ActionType::Ron, ActionType::Pass),
            (ActionType::Tsumo, ActionType::Discard),
        ] {
            obs.replace_legal_actions(vec![
                Action::new(win, Some(8), vec![], Some(0)),
                Action::new(continuation, Some(8), vec![], Some(0)),
            ]);
            let options = candidates(&obs, 4, &choices, &mut rng);
            assert_eq!(
                options.iter().map(|c| (c.head, c.code)).collect::<Vec<_>>(),
                vec![(4, 0), (4, 1)]
            );
            assert!(options[0].action.is_none());
            assert_eq!(options[1].action.as_ref().unwrap().action_type, win);
        }
    }
}
