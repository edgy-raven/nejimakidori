use riichienv_core::{
    hand_evaluator::HandEvaluator,
    score,
    types::{Conditions, Meld, Wind},
};

#[cfg(test)]
pub fn public_values(melds: &[Meld], dora: &[i32; 34]) -> (u16, u8) {
    let melds: Vec<_> = melds.iter().map(crate::features::NativeMeld::new).collect();
    let mut out = [0i32; 2];
    unsafe extern "C" {
        fn feature_public_values(
            melds: *const crate::features::NativeMeld,
            size: i32,
            dora: *const i32,
            out: *mut i32,
        );
    }
    unsafe {
        feature_public_values(
            melds.as_ptr(),
            melds.len() as i32,
            dora.as_ptr(),
            out.as_mut_ptr(),
        );
    }
    (out[0] as u16, out[1] as u8)
}

#[derive(Clone)]
pub struct Context {
    pub own_discards: Vec<u8>,
    pub missed_agari: bool,
    pub unavailable: [u8; 34],
    pub unavailable_red: [i32; 3],
    pub dora: Vec<u8>,
    pub player_wind: u8,
    pub round_wind: u8,
}

#[derive(Clone, Default)]
pub struct Ranges {
    // Flattened [closed riichi / open][ron / tsumo][minimum / maximum].
    pub values: [f64; 8],
    pub valid: [u8; 4],
    // Open completions lacking yaku, separately for ron and tsumo.
    pub restricted: [u8; 2],
    pub eligible: bool,
}
pub fn ready_ranges(
    tiles: Vec<u8>,
    melds: Vec<Meld>,
    ctx: &Context,
    availability: &crate::features::RangeContext,
    cut: Option<u8>,
    waits: u64,
) -> Ranges {
    // Nyanten already enumerated structural waits, including exhausted tiles.
    // Reuse that mask instead of testing all 34 completions again.
    let furiten = availability.missed_agari != 0
        || availability.own_discards & waits != 0
        || cut.is_some_and(|tile| waits & (1u64 << tile) != 0);
    let open = melds.iter().any(|meld| meld.opened);
    let mut result = Ranges {
        eligible: true,
        ..Ranges::default()
    };
    // calc adds the winning tile to a thirteen-tile hand, including its red
    // bonus. Normalize the hand and fixed melds once for all winning tiles.
    let evaluator = HandEvaluator::new(tiles, melds);
    for base in (0u8..34).filter(|base| waits & (1u64 << base) != 0) {
        for (winning_tile, _) in availability.available_tiles(base) {
            for (win_kind, is_tsumo) in [false, true].into_iter().enumerate() {
                if !open && !is_tsumo && furiten {
                    continue;
                }
                let scored = evaluator.calc_with_indicators(
                    winning_tile,
                    &ctx.dora,
                    &[],
                    Some(Conditions {
                        riichi: !open,
                        tsumo: is_tsumo,
                        player_wind: Wind::from(ctx.player_wind),
                        round_wind: Wind::from(ctx.round_wind),
                        ..Conditions::default()
                    }),
                );
                // Open ron is scored even under furiten so that missing yaku
                // remains distinguishable from an otherwise legal blocked win.
                if !scored.is_win {
                    if open {
                        result.restricted[win_kind] = 1;
                    }
                    continue;
                }
                if !is_tsumo && furiten {
                    continue;
                }
                let scored = normalize_score(scored, ctx.player_wind == 0, is_tsumo);
                let (ron, ko, oya) = (
                    scored.ron_agari,
                    scored.tsumo_agari_ko,
                    scored.tsumo_agari_oya,
                );
                let points = if !is_tsumo {
                    ron
                } else if ctx.player_wind == 0 {
                    ko * 3
                } else {
                    ko * 2 + oya
                } as f64;
                let scenario = usize::from(open) * 2 + win_kind;
                if result.valid[scenario] == 0 {
                    result.values[scenario * 2..scenario * 2 + 2].fill(points);
                } else {
                    result.values[scenario * 2] = result.values[scenario * 2].min(points);
                    result.values[scenario * 2 + 1] = result.values[scenario * 2 + 1].max(points);
                }
                result.valid[scenario] = 1;
            }
        }
    }
    result
}

// The label pipeline supplies complete concealed hands, with melds separate.
// Bind directly to the core so no Python evaluator adjusts the winning tile.
#[pyo3::pyclass(get_all, frozen)]
pub struct HandScore {
    is_win: bool,
    yakuman: bool,
    ron_agari: u32,
    tsumo_agari_ko: u32,
    tsumo_agari_oya: u32,
    han: u32,
    fu: u32,
    yaku: Vec<u32>,
}

#[pyo3::pyfunction]
pub fn score_hand(
    ctx: &pyo3::Bound<'_, pyo3::PyAny>,
    conditions: &pyo3::Bound<'_, pyo3::PyAny>,
    ura: Vec<u8>,
) -> pyo3::PyResult<HandScore> {
    use pyo3::prelude::*;
    use riichienv_core::types::MeldType;

    let tiles: Vec<u8> = ctx.getattr("tiles")?.extract()?;
    let dora: Vec<u8> = ctx.getattr("dora_indicators")?.extract()?;
    let winning_tile: u8 = ctx.getattr("agari_tile")?.extract()?;
    let melds = ctx
        .getattr("melds")?
        .try_iter()?
        .map(|meld| {
            let meld = meld?;
            Ok(Meld::new(
                match meld
                    .getattr("meld_type")?
                    .call_method0("__int__")?
                    .extract::<u8>()?
                {
                    0 => MeldType::Chi,
                    1 => MeldType::Pon,
                    2 => MeldType::Daiminkan,
                    3 => MeldType::Ankan,
                    4 => MeldType::Kakan,
                    _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid meld type")),
                },
                meld.getattr("tiles")?.extract()?,
                meld.getattr("opened")?.extract()?,
                meld.getattr("from_who")?.extract()?,
                meld.getattr("called_tile")?.extract()?,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let conditions = Conditions {
        tsumo: conditions.getattr("tsumo")?.extract()?,
        riichi: conditions.getattr("riichi")?.extract()?,
        double_riichi: conditions.getattr("double_riichi")?.extract()?,
        ippatsu: conditions.getattr("ippatsu")?.extract()?,
        haitei: conditions.getattr("haitei")?.extract()?,
        houtei: conditions.getattr("houtei")?.extract()?,
        rinshan: conditions.getattr("rinshan")?.extract()?,
        chankan: conditions.getattr("chankan")?.extract()?,
        tsumo_first_turn: conditions.getattr("tsumo_first_turn")?.extract()?,
        player_wind: Wind::from(
            conditions
                .getattr("player_wind")?
                .call_method0("__int__")?
                .extract::<u8>()?,
        ),
        round_wind: Wind::from(
            conditions
                .getattr("round_wind")?
                .call_method0("__int__")?
                .extract::<u8>()?,
        ),
        ..Conditions::default()
    };
    let scored = ctx.py().detach(|| {
        normalize_score(
            HandEvaluator::new(tiles, melds).calc_with_indicators(
                winning_tile,
                &dora,
                if conditions.riichi || conditions.double_riichi {
                    &ura
                } else {
                    &[]
                },
                Some(conditions.clone()),
            ),
            conditions.player_wind == Wind::East,
            conditions.tsumo,
        )
    });
    Ok(HandScore {
        is_win: scored.is_win,
        yakuman: scored.yakuman,
        ron_agari: scored.ron_agari,
        tsumo_agari_ko: scored.tsumo_agari_ko,
        tsumo_agari_oya: scored.tsumo_agari_oya,
        han: scored.han,
        fu: scored.fu,
        yaku: scored.yaku,
    })
}

fn normalize_score(
    mut scored: riichienv_core::types::WinResult,
    dealer: bool,
    tsumo: bool,
) -> riichienv_core::types::WinResult {
    if scored.yakuman {
        // Named double yakuman count once; distinct yakuman still stack.
        scored.han = 13 * scored.yaku.len() as u32;
        scored.fu = 0;
        let payment = score::calculate_score(scored.han as u8, 0, dealer, tsumo, 0, 4);
        scored.ron_agari = payment.pay_ron;
        scored.tsumo_agari_ko = payment.pay_tsumo_ko;
        scored.tsumo_agari_oya = payment.pay_tsumo_oya;
    }
    scored
}

#[cfg(test)]
use crate::features::ranges;

#[cfg(test)]
mod tests {
    use super::*;
    use riichienv_core::types::MeldType;

    #[test]
    fn public_values_include_red_dora_and_all_kan_types_without_fu_overflow() {
        let mut dora = [0; 34];
        dora[4] = 2;
        let melds = [
            Meld::new(MeldType::Chi, vec![8, 12, 16], true, 1, Some(8)),
            Meld::new(MeldType::Pon, vec![108, 109, 110], true, 1, Some(108)),
            Meld::new(MeldType::Daiminkan, vec![36, 37, 38, 39], true, 1, Some(36)),
            Meld::new(MeldType::Kakan, vec![52, 53, 54, 55], true, 1, Some(53)),
        ];
        assert_eq!(public_values(&melds, &dora), (28, 4));
        let kans: Vec<_> = [0, 32, 72, 108]
            .into_iter()
            .map(|tile| Meld::new(MeldType::Ankan, (tile..tile + 4).collect(), false, 0, None))
            .collect();
        assert_eq!(public_values(&kans, &dora), (128, 0));
        dora[0] = 3;
        assert_eq!(public_values(&kans, &dora), (128, 12));
    }

    #[test]
    fn one_shanten_ranges_and_open_yaku_are_action_specific() {
        let mut tiles = vec![0, 4, 8, 12, 17, 20, 60, 64, 68, 92, 108, 40, 41];
        let mut ctx = Context {
            own_discards: vec![],
            missed_agari: false,
            unavailable: std::array::from_fn(|base| {
                tiles.iter().filter(|&&t| t as usize / 4 == base).count() as u8
            }),
            unavailable_red: [0; 3],
            dora: vec![],
            player_wind: 1,
            round_wind: 0,
        };
        let values = ranges(&tiles, &[], &ctx, None);
        assert!(values.eligible);
        assert_eq!(values.valid, [1, 1, 0, 0]);
        assert_eq!(values.values, [1300., 3900., 2000., 5200., 0., 0., 0., 0.]);
        // Exhausted 5s still makes later 5s/8s waits furiten. Retain it in
        // the shared structural mask even though it cannot be drawn.
        let mut exhausted = ctx.clone();
        exhausted.unavailable[22] = 4;
        exhausted.unavailable_red[2] = 1;
        exhausted.own_discards.push(22);
        assert_eq!(
            ranges(&tiles, &[], &exhausted, None).values,
            [1300., 2600., 2000., 4000., 0., 0., 0., 0.],
        );
        tiles[10] = 96;
        ctx.unavailable[27] -= 1;
        ctx.unavailable[24] += 1;
        let meld = Meld {
            meld_type: riichienv_core::types::MeldType::Chi,
            tiles: vec![0, 4, 8],
            opened: true,
            from_who: -1,
            called_tile: Some(0),
        };
        let values = ranges(&tiles[3..], &[meld], &ctx, None);
        assert!(values.eligible);
        assert_eq!(values.valid, [0; 4]);
        assert_eq!(values.restricted, [1; 2]);
    }

    #[test]
    fn ranges_exclude_exhausted_copies_and_keep_structural_furiten() {
        let tiles = [0, 4, 8, 12, 17, 20, 60, 64, 68, 92, 96, 40, 41];
        let mut ctx = Context {
            own_discards: vec![],
            missed_agari: false,
            unavailable: std::array::from_fn(|base| {
                tiles.iter().filter(|&&tile| tile / 4 == base as u8).count() as u8
                    + u8::from(base == 22)
            }),
            unavailable_red: [0, 0, 1],
            dora: vec![],
            player_wind: 1,
            round_wind: 0,
        };
        let values = ranges(&tiles, &[], &ctx, None);
        assert_eq!(values.values, [2000., 2000., 2700., 2700., 0., 0., 0., 0.]);
        ctx.unavailable[22] = 4;
        ctx.own_discards.push(22);
        let values = ranges(&tiles, &[], &ctx, None);
        assert_eq!(values.valid, [0, 1, 0, 0]);
        assert_eq!(&values.values[..4], &[0., 0., 2700., 2700.]);
        ctx.unavailable[25] = 4;
        assert_eq!(ranges(&tiles, &[], &ctx, None).valid, [0; 4]);
    }
}
