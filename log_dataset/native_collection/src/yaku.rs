//! Public feasibility bounds, not probabilities or concealed-hand affinities.
use riichienv_core::{
    observation::Observation,
    types::{Meld, MeldType},
};

pub fn possibilities<const N: usize>(
    obs: &Observation,
    seats: [usize; N],
    claim: Option<(usize, usize)>,
) -> [[f64; 19]; N] {
    let mut exposed = [0usize; 34];
    for tile in obs
        .discards
        .iter()
        .flatten()
        .chain(obs.dora_indicators.iter())
    {
        exposed[*tile as usize / 4] += 1;
    }
    for meld in obs.melds.iter().flatten() {
        for &tile in &meld.tiles {
            exposed[tile as usize / 4] += 1;
        }
        // A called tile is already counted in its discarder’s river.
        if let Some(tile) = meld.called_tile {
            exposed[tile as usize / 4] -= 1;
        }
    }
    seats.map(|seat| {
        let mut outside = exposed;
        for meld in &obs.melds[seat] {
            for &tile in &meld.tiles {
                outside[tile as usize / 4] -= 1;
            }
        }
        if let Some((source, tile)) = claim {
            if source != seat {
                outside[tile] -= 1;
            }
        }
        let available = outside.map(|count| 4 - count);

        flags(
            &obs.melds[seat],
            &available,
            obs.round_wind as usize,
            (seat + 4 - obs.oya as usize) % 4,
        )
    })
}

fn flags(melds: &[Meld], available: &[usize; 34], round: usize, seat: usize) -> [f64; 19] {
    let melds: Vec<_> = melds.iter().map(crate::features::NativeMeld::new).collect();
    let mut output = [0u8; 19];
    unsafe extern "C" {
        fn feature_yaku(
            melds: *const crate::features::NativeMeld,
            size: i32,
            available: *const i32,
            round: i32,
            seat: i32,
            out: *mut u8,
        );
    }
    unsafe {
        feature_yaku(
            melds.as_ptr(),
            melds.len() as i32,
            available.map(|x| x as i32).as_ptr(),
            round as i32,
            seat as i32,
            output.as_mut_ptr(),
        );
    }
    output.map(f64::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn meld(kind: MeldType, base: u8) -> Meld {
        Meld {
            meld_type: kind,
            tiles: (0..if kind == MeldType::Ankan { 4 } else { 3 })
                .map(|i| base * 4 + i)
                .collect(),
            opened: kind != MeldType::Ankan,
            from_who: -1,
            called_tile: None,
        }
    }
    #[test]
    fn availability_uses_all_public_seats_without_double_counting_calls() {
        use riichienv_core::{rule::GameRule, state::GameState};
        let mut game = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
        let mut obs = game.get_observation(0);
        obs.discards = [vec![], vec![124], vec![124], vec![]];
        obs.dora_indicators.clear();
        obs.melds = std::array::from_fn(|_| vec![]);
        obs.last_discard = None;
        assert_eq!(possibilities(&obs, [0, 1, 2, 3], None)[0][1], 0.);
        // The currently offered second copy can still complete a dragon pon.
        obs.last_discard = Some(124);
        assert_eq!(possibilities(&obs, [0, 1, 2, 3], Some((2, 31)))[0][1], 1.);
        obs.last_discard = None;
        obs.discards = [vec![124], vec![124], vec![], vec![]];
        assert_eq!(possibilities(&obs, [0], None)[0][1], 0.);
        assert_eq!(possibilities(&obs, [0], Some((1, 31)))[0][1], 1.);
        obs.discards = [vec![], vec![0, 0], vec![0, 0], vec![]];
        assert_eq!(possibilities(&obs, [0], None)[0][15], 0.);
        obs.discards = [vec![], vec![124], vec![], vec![]];
        // A player's called dragon remains available to their own yaku.
        obs.discards[2].clear();
        obs.melds[0] = vec![meld(MeldType::Pon, 31)];
        obs.melds[0][0].called_tile = Some(124);
        assert_eq!(possibilities(&obs, [0, 1, 2, 3], None)[0][1], 1.);
        assert_eq!(possibilities(&obs, [0, 1, 2, 3], None)[1][1], 0.);
    }
    #[test]
    fn public_bounds_respect_closed_kans_honors_and_group_capacity() {
        let all = [4; 34];
        let honor = flags(&[meld(MeldType::Pon, 31)], &all, 0, 1);
        assert_eq!((honor[6], honor[7]), (1., 0.));
        let kans = [meld(MeldType::Ankan, 0), meld(MeldType::Ankan, 8)];
        assert_eq!(flags(&kans, &all, 0, 1)[18], 1.);
        let four = [
            meld(MeldType::Pon, 0),
            meld(MeldType::Pon, 8),
            meld(MeldType::Pon, 9),
            meld(MeldType::Pon, 17),
        ];
        let values = flags(&four, &all, 0, 1);
        assert_eq!(
            (values[10], values[11], values[16], values[17]),
            (0., 0., 0., 0.)
        );
        let mut limited = all;
        limited[31] = 2;
        assert_eq!(flags(&[], &limited, 0, 1)[1], 0.);
        assert_eq!(flags(&[], &limited, 0, 1)[10], 1.);
        limited[32] = 2;
        assert_eq!(flags(&[], &limited, 0, 1)[10], 0.);
    }
}
