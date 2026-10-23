#[cfg(test)]
#[repr(C)]
#[derive(Clone)]
pub struct Analysis {
    pub shanten: [i8; 34],
    pub ukeire_mask: [u64; 34],
    pub ukeire_count: [u16; 34],
    pub upgrade_mask: [u64; 34],
    pub upgrade_type_count: [u8; 34],
    pub upgrade_tile_count: [u16; 34],
    pub upgrade_weighted_gain: [u16; 34],
    pub completion_probability: [f32; 34],
    pub completion_valid: [u8; 34],
}

unsafe extern "C" {
    #[cfg(test)]
    fn nyanten_analyze_features(
        counts: *const u8,
        unavailable: *const u8,
        mask: u64,
        modes: *const u8,
        size: usize,
        out: *mut Analysis,
        structural: *mut u64,
    );
    fn nyanten_tenpai_paths(
        counts: *const u8,
        unavailable: *const u8,
        mask: u64,
        max_draws: u32,
        nodes: u32,
        hands: *mut u8,
        weights: *mut u64,
        capacity: u32,
        size: *mut u32,
    ) -> bool;
    #[cfg(test)]
    fn nyanten_analyze_modes(
        counts: *const u8,
        unavailable: *const u8,
        mask: u64,
        upgrades: bool,
        completion: bool,
        modes: *const u8,
        mode_count: usize,
        output: *mut Analysis,
    );
    #[cfg(test)]
    fn nyanten_tenpai_discards(counts: *const u8, waits: *mut u64);
    fn nyanten_draw(counts: *const u8, mode: u32) -> u64;
    fn nyanten_yaku_shanten(counts: *const u8, melds: *const u8, count: u32) -> u32;
}

#[cfg(test)]
pub fn analyze(
    counts: &[u8; 34],
    unavailable: &[u8; 34],
    cuts: &[usize],
    upgrades: bool,
    modes: &[u8],
) -> Vec<Analysis> {
    let mut output: Vec<Analysis> = modes
        .iter()
        .map(|_| unsafe { std::mem::zeroed() })
        .collect();
    let mask = cuts.iter().fold(0, |mask, &cut| mask | 1u64 << cut);
    unsafe {
        nyanten_analyze_modes(
            counts.as_ptr(),
            unavailable.as_ptr(),
            mask,
            upgrades,
            false,
            modes.as_ptr(),
            modes.len(),
            output.as_mut_ptr(),
        );
    }
    output
}

/// Structural waits by discard; zero for absent or non-tenpai discards.
#[cfg(test)]
pub fn tenpai_discards(counts: &[u8; 34]) -> [u64; 34] {
    let mut waits = [0; 34];
    unsafe { nyanten_tenpai_discards(counts.as_ptr(), waits.as_mut_ptr()) };
    waits
}

pub fn improving(counts: &[u8; 34]) -> (i32, u64) {
    let value = unsafe { nyanten_draw(counts.as_ptr(), 1) };
    ((value >> 34) as i32 - 1, value & ((1u64 << 34) - 1))
}

#[cfg(test)]
pub fn efficiency(counts: &[u8; 34], unavailable: &[u8; 34]) -> [f64; 3] {
    let value = unsafe { nyanten_draw(counts.as_ptr(), 1) };
    let mask = value & ((1u64 << 34) - 1);
    [
        (value >> 34) as f64 / 8.0 - 1.0 / 8.0,
        mask.count_ones() as f64 / 34.0,
        (0..34)
            .filter(|tile| mask & (1 << tile) != 0)
            .map(|tile| 4i32 - unavailable[tile] as i32)
            .sum::<i32>() as f64
            / 80.0,
    ]
}

pub fn yaku(counts: &[u8; 34], melds: &[riichienv_core::types::Meld]) -> [i32; 3] {
    let mut fixed = [34u8; 16];
    for (index, meld) in melds.iter().enumerate() {
        for (copy, tile) in meld.tiles.iter().enumerate() {
            fixed[index * 4 + copy] = tile / 4;
        }
    }
    let value =
        unsafe { nyanten_yaku_shanten(counts.as_ptr(), fixed.as_ptr(), melds.len() as u32) };
    std::array::from_fn(|index| ((value >> (index * 8)) & 255) as i32 - 1)
}

/// Positive shortest-route mass for tanyao, flush and outside families.
/// Incomplete searches select nothing, matching the dataset sampling contract.
pub fn focus_routes(
    counts: &[u8; 34],
    unavailable: &[u8; 34],
    melds: &[riichienv_core::types::Meld],
    cuts: u64,
) -> [bool; 3] {
    let mut hands = vec![[0u8; 34]; 4096];
    let mut weights = vec![0u64; 4096];
    let mut size = 0;
    let mask = if counts.iter().map(|&n| n as usize).sum::<usize>() % 3 == 1 {
        1 << 34
    } else {
        cuts
    };
    let complete = unsafe {
        nyanten_tenpai_paths(
            counts.as_ptr(),
            unavailable.as_ptr(),
            mask,
            3,
            512,
            hands.as_mut_ptr().cast(),
            weights.as_mut_ptr(),
            4096,
            &mut size,
        )
    };
    if !complete {
        return [false; 3];
    }
    let mut routes = [false; 3];
    for hand in &mut hands[..size as usize] {
        // Family-route labels override scorer yaku with inclusive structural
        // families and intersect all winning waits. Only positive support is
        // needed for sampling; the path weights are positive by construction.
        let waits = improving(hand).1;
        let mut supported = [waits != 0; 3];
        for tile in (0..34).filter(|tile| waits & (1 << tile) != 0) {
            hand[tile] += 1;
            let distances = yaku(hand, melds);
            for family in 0..3 {
                supported[family] &= distances[family] == -1;
            }
            hand[tile] -= 1;
        }
        for family in 0..3 {
            routes[family] |= supported[family];
        }
        if routes.iter().all(|&value| value) {
            break;
        }
    }
    routes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenpai_discards_match_full_analysis_for_all_hand_forms() {
        for tiles in [
            vec![0, 1, 2, 9, 10, 11, 18, 19, 20, 24, 25, 26, 27, 28],
            vec![0, 0, 0, 0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27],
            vec![0, 0, 9, 9, 18, 18, 27, 27, 28, 28, 31, 31, 32, 33],
            vec![0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 4],
            vec![0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28],
            vec![0, 1, 2, 9, 10, 11, 27, 28],
            vec![0, 1, 2, 27, 28],
            vec![27, 28],
        ] {
            let mut counts = [0; 34];
            for tile in tiles {
                counts[tile] += 1;
            }
            let cuts: Vec<usize> = (0..34).filter(|&tile| counts[tile] > 0).collect();
            // Feature packing reuses structural waits even when public
            // availability removes every tile from the ukeire mask.
            let mut structural = [0u64; 34];
            let mut shared: [Analysis; 2] = unsafe { std::mem::zeroed() };
            unsafe {
                nyanten_analyze_features(
                    counts.as_ptr(),
                    [4u8; 34].as_ptr(),
                    cuts.iter().fold(0, |mask, &cut| mask | (1 << cut)),
                    [0u8, 1].as_ptr(),
                    2,
                    shared.as_mut_ptr(),
                    structural.as_mut_ptr(),
                );
            }
            for &cut in &cuts {
                let mut after = counts;
                after[cut] -= 1;
                let (distance, waits) = improving(&after);
                assert_eq!(structural[cut], waits | (((distance + 1) as u64) << 34));
                assert_eq!(shared[1].shanten[cut], distance as i8);
                assert_eq!(shared[1].ukeire_mask[cut], 0);
            }
            let full = analyze(&counts, &[0; 34], &cuts, false, &[1]).remove(0);
            let expected = std::array::from_fn(|cut| {
                if counts[cut] > 0 && full.shanten[cut] == 0 {
                    full.ukeire_mask[cut]
                } else {
                    0
                }
            });
            assert_eq!(tenpai_discards(&counts), expected, "counts={counts:?}");
        }
    }

    #[test]
    fn action_analysis_matches_c_abi_and_known_wait() {
        assert_eq!(std::mem::size_of::<Analysis>(), 1000);
        assert_eq!(std::mem::offset_of!(Analysis, ukeire_mask), 40);
        assert_eq!(std::mem::offset_of!(Analysis, upgrade_mask), 384);
        assert_eq!(std::mem::offset_of!(Analysis, completion_probability), 828);
        assert_eq!(std::mem::offset_of!(Analysis, completion_valid), 964);
        let mut counts = [0u8; 34];
        for tile in [0, 1, 2, 9, 10, 11, 18, 19, 20, 24, 25, 26, 27] {
            counts[tile] += 1;
        }
        let efficiency = efficiency(&counts, &counts);
        assert_eq!(efficiency, [0.0, 1.0 / 34.0, 3.0 / 80.0]);
        counts[28] += 1;
        let result = analyze(&counts, &counts, &[28], true, &[0, 1]);
        for analysis in result {
            assert_eq!(analysis.shanten[28], 0);
            assert_eq!(analysis.ukeire_mask[28], 1 << 27);
            assert_eq!(analysis.ukeire_count[28], 3);
            assert_eq!(analysis.completion_valid, [0; 34]);
        }
    }
}
