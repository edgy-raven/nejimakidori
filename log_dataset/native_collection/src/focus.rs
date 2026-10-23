//! Sampling decisions use public state and native analyses before encoding.
use crate::{
    candidate,
    history::{PublicHistory, base, dora},
    nyanten,
};
use riichienv_core::{action::ActionType, observation::Observation, types::MeldType};

pub struct Analysis {
    pub counts: [u8; 34],
    pub unavailable: [u8; 34],
    pub distance: i32,
    pub wind: usize,
    pub kyoku: usize,
    pub oya: usize,
    pub remaining: usize,
    pub order: [usize; 4],
}

#[derive(Default)]
pub struct Focus {
    pub routes: [bool; 3],
    pub fast: bool,
    pub threat: bool,
    pub late: bool,
    pub gap: bool,
    pub phase: u8,
    eligible: bool,
}
impl Focus {
    pub fn analyze(
        state: &PublicHistory,
        obs: &Observation,
        actor: usize,
        phase: u8,
        analysis: &Analysis,
    ) -> Self {
        let legal = obs.legal_actions_method();
        let mut choices = [false; 38];
        if phase != 0 {
            choices[0] = true;
        }
        for action in &legal {
            match (phase, action.action_type) {
                (0, ActionType::Discard) => choices[candidate::code(action.tile.unwrap())] = true,
                (1, ActionType::Chi) => choices[1] = true,
                (1, ActionType::Pon) => choices[2] = true,
                (1, ActionType::Daiminkan) => choices[3] = true,
                (2, ActionType::Ankan | ActionType::Kakan) => {
                    choices[1 + action.tile.unwrap() as usize / 4] = true
                }
                (3, ActionType::Riichi) => choices[1] = true,
                _ => (),
            }
        }
        if choices.iter().filter(|&&value| value).count() < 2 {
            return Self::default();
        }
        let cuts = if phase == 0 || phase == 3 {
            legal
                .iter()
                .filter(|a| a.action_type == ActionType::Discard)
                .fold(0, |mask, a| mask | (1 << (a.tile.unwrap() / 4)))
        } else {
            (0..34)
                .filter(|&tile| analysis.counts[tile] > 0)
                .fold(0, |mask, tile| mask | (1 << tile))
        };
        Self {
            routes: nyanten::focus_routes(
                &analysis.counts,
                &analysis.unavailable,
                &state.replay.players[actor].melds,
                cuts,
            ),
            threat: visible_threat(state, actor, analysis),
            late: analysis.wind >= 2
                || (analysis.wind == 1 && analysis.kyoku >= 2)
                || (state.live_wall <= 16 && analysis.distance <= 1),
            gap: placement_gap(&obs.scores, actor, analysis),
            phase,
            eligible: true,
            fast: false,
        }
    }
    pub fn metadata(&self) -> (u8, bool, bool) {
        if !self.eligible {
            return (0, false, false);
        }
        let selected = [
            self.phase == 1 && self.fast,
            self.routes[0],
            self.routes[1],
            self.routes[2],
            self.threat,
            self.late || self.gap,
        ];
        (
            selected
                .iter()
                .enumerate()
                .fold(0, |mask, (bit, &value)| mask | ((value as u8) << bit)),
            self.late,
            self.gap,
        )
    }
}

fn placement_gap(scores: &[i32], actor: usize, analysis: &Analysis) -> bool {
    if analysis.wind == 0 && analysis.kyoku == 0 {
        return false;
    }
    let ranked = analysis.order.map(|seat| scores[seat]);
    let thresholds = analysis.order.map(|seat| {
        if (seat + 4 - analysis.oya) % 4 <= analysis.remaining {
            16000
        } else {
            10000
        }
    });
    match analysis
        .order
        .iter()
        .position(|&seat| seat == actor)
        .unwrap()
    {
        0 => ranked[0] - ranked[1] > thresholds[1],
        1 => ranked[0] - ranked[1] < thresholds[1] && ranked[1] - ranked[3] > thresholds[3],
        _ => ranked[2] - ranked[3] < thresholds[3],
    }
}

pub fn yakuhai(
    state: &PublicHistory,
    seat: usize,
    analysis: &Analysis,
    opened_only: bool,
) -> usize {
    state.replay.players[seat]
        .melds
        .iter()
        .filter(|meld| !opened_only || meld.meld_type != MeldType::Ankan)
        .map(|meld| {
            let tile = meld.tiles[0] as usize / 4;
            usize::from(tile >= 31)
                + usize::from(tile == 27 + (seat + 4 - analysis.oya) % 4)
                + usize::from(tile == 27 + analysis.wind)
        })
        .sum()
}

fn visible_threat(state: &PublicHistory, actor: usize, analysis: &Analysis) -> bool {
    let dora_counts = dora(&state.dora_markers);
    (0..4).filter(|&seat| seat != actor).any(|seat| {
        if state.declared[seat] || state.accepted[seat] {
            return true;
        }
        let melds = &state.replay.players[seat].melds;
        let opened = melds
            .iter()
            .filter(|meld| meld.meld_type != MeldType::Ankan)
            .count();
        if opened >= 3 || (opened >= 2 && state.rivers[seat].len() >= 9) {
            return true;
        }
        let mut han = yakuhai(state, seat, analysis, false) as i32;
        let mut suits = 0u8;
        for (meld, event) in melds.iter().zip(&state.meld_events[seat]) {
            for &tile in event
                .tile
                .iter()
                .chain(&event.consumed)
                .chain(&event.upgrade_tile)
            {
                han += dora_counts[base(tile)] + i32::from(tile >= 34);
                if meld.meld_type != MeldType::Ankan && base(tile) < 27 {
                    suits |= 1 << (base(tile) / 9);
                }
            }
        }
        if opened >= 2 && suits.count_ones() == 1 {
            let boundary = state.riichi_boundary[seat].unwrap_or(state.rivers[seat].len());
            // Match the retained public river windows used by model features.
            let middle_suits = [
                &state.rivers[seat][..boundary],
                &state.rivers[seat][boundary..],
            ]
            .iter()
            .flat_map(|river| river.iter().rev().take(32))
            .filter(|cut| base(cut.tile) < 27 && (2..=6).contains(&(base(cut.tile) % 9)))
            .fold(0u8, |mask, cut| mask | (1 << (base(cut.tile) / 9)));
            if middle_suits | suits == 7 {
                han += 2;
            }
        }
        han >= 3
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_uses_strict_trailing_player_thresholds_and_recorded_ties() {
        let mut analysis = Analysis {
            counts: [0; 34],
            unavailable: [0; 34],
            distance: 3,
            wind: 1,
            kyoku: 3,
            oya: 0,
            remaining: 0,
            order: [0, 1, 2, 3],
        };
        for (scores, dealer, expected) in [
            ([41000, 30000, 20000, 9000], 2, [true, false, false, false]),
            ([41000, 30000, 20000, 9000], 1, [false, true, false, false]),
            ([41000, 30000, 20000, 9000], 3, [true, false, true, true]),
            ([39000, 30000, 20000, 11000], 0, [false, true, true, true]),
            ([31000, 22000, 20000, 15000], 0, [false, false, true, true]),
            ([40000, 30000, 20000, 10000], 0, [false; 4]),
            ([46000, 30000, 20000, 10000], 1, [false; 4]),
            ([35000, 30000, 26000, 10000], 3, [false, true, false, false]),
            ([30000, 26000, 20000, 10000], 3, [false, false, true, true]),
            ([30000, 30000, 20000, 20000], 0, [false, false, true, true]),
        ] {
            analysis.oya = dealer;
            for actor in 0..4 {
                assert_eq!(placement_gap(&scores, actor, &analysis), expected[actor]);
            }
        }
        analysis.oya = 0;
        analysis.kyoku = 1;
        analysis.remaining = 2;
        assert!(!placement_gap(&[41000, 30000, 20000, 9000], 0, &analysis));
        analysis.oya = 2;
        assert!(placement_gap(&[41000, 30000, 20000, 9000], 0, &analysis));
        analysis.wind = 0;
        analysis.kyoku = 0;
        assert!(!placement_gap(&[41000, 30000, 20000, 9000], 0, &analysis));
    }
}
