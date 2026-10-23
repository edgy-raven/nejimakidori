use riichienv_core::action::{Action, ActionType};

pub fn code(tile: u8) -> usize {
    if matches!(tile, 16 | 52 | 88) {
        34 + (tile / 36) as usize
    } else {
        (tile / 4) as usize
    }
}
pub fn base(code: usize) -> usize {
    if code >= 34 {
        [4, 13, 22][code - 34]
    } else {
        code
    }
}
pub fn prefer_red_calls(legal: Vec<Action>) -> Vec<Action> {
    let mut selected: Vec<Action> = vec![];
    for action in legal {
        if matches!(action.action_type, ActionType::Chi | ActionType::Pon) {
            let mut consumed = [action.consume_tiles[0] / 4, action.consume_tiles[1] / 4];
            consumed.sort_unstable();
            if let Some(index) = selected.iter().position(|previous| {
                if previous.action_type != action.action_type
                    || previous.tile.unwrap() / 4 != action.tile.unwrap() / 4
                {
                    return false;
                }
                let mut prior = [previous.consume_tiles[0] / 4, previous.consume_tiles[1] / 4];
                prior.sort_unstable();
                prior == consumed
            }) {
                if !selected[index]
                    .consume_tiles
                    .iter()
                    .any(|tile| matches!(tile, 16 | 52 | 88))
                {
                    selected[index] = action;
                }
                continue;
            }
        }
        selected.push(action);
    }
    selected
}

pub struct Legal {
    pub discards: [bool; 37],
    pub riichi: [bool; 37],
    pub calls: [[bool; 37]; 4],
}
#[cfg(test)]
mod bank_tests {
    use crate::{
        features::{Bank, bank_hand},
        packed::Row,
        scoring,
    };
    use riichienv_core::types::{Meld, MeldType};

    #[test]
    fn bank_preserves_dense_meld_equivalence() {
        let mut row = Row::new();
        let mut bank = Bank::default();
        let hand = [36, 40, 44, 72, 76, 80, 108];
        let mut melds = [
            Meld::new(MeldType::Chi, vec![8, 12, 16], true, 1, Some(8)),
            Meld::new(MeldType::Chi, vec![17, 20, 24], true, 1, Some(20)),
        ];
        assert_eq!(
            bank_hand(
                &mut row,
                &mut bank,
                &hand,
                &melds,
                scoring::public_values(&melds, &[0; 34])
            ),
            0
        );
        let original = row.bytes.clone();
        melds.swap(0, 1);
        // Move the red five between overlapping chii without changing totals.
        melds[0].tiles[0] = 16;
        melds[1].tiles[2] = 17;
        assert_eq!(
            bank_hand(
                &mut row,
                &mut bank,
                &hand,
                &melds,
                scoring::public_values(&melds, &[0; 34])
            ),
            0
        );
        assert_eq!(row.bytes, original);
        assert_eq!(row.get("candidate_melds", 2 * 6), 1.0);
        assert_eq!(row.get("candidate_melds", 4 * 6), 1.0);
        assert_eq!(row.get("candidate_melds", 4 * 6 + 5), 1.0);

        melds[0].tiles = vec![9, 13, 16];
        assert_eq!(
            bank_hand(
                &mut row,
                &mut bank,
                &hand,
                &melds,
                scoring::public_values(&melds, &[0; 34])
            ),
            1
        );
        assert_eq!(row.get("candidate_melds", 204 + 2 * 6), 2.0);
        for meld_type in [
            MeldType::Pon,
            MeldType::Daiminkan,
            MeldType::Ankan,
            MeldType::Kakan,
        ] {
            let tiles = if meld_type == MeldType::Pon {
                vec![16, 17, 18]
            } else {
                vec![16, 17, 18, 19]
            };
            let meld = Meld::new(meld_type, tiles, meld_type != MeldType::Ankan, 1, None);
            let index = bank_hand(
                &mut row,
                &mut bank,
                &[36, 40, 44, 72, 76, 80, 108, 109, 112, 113],
                &[meld.clone()],
                scoring::public_values(&[meld], &[0; 34]),
            );
            assert_eq!(index, meld_type as usize + 1);
            assert_eq!(
                row.get("candidate_melds", index * 204 + 4 * 6 + meld_type as usize),
                1.0
            );
        }
    }
}
