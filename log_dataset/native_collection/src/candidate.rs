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

#[derive(Clone)]
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
        assert_eq!(bank_hand(&mut row, &mut bank, &hand, &melds, &[0; 19]), 0);
        let original = row.bytes.clone();
        melds.swap(0, 1);
        // Move the red five between overlapping chii without changing totals.
        melds[0].tiles[0] = 16;
        melds[1].tiles[2] = 17;
        assert_eq!(bank_hand(&mut row, &mut bank, &hand, &melds, &[0; 19]), 0);
        assert_eq!(row.bytes, original);
        assert_eq!(row.get("candidate_meld_tiles", 2), 2.0);
        assert_eq!(row.get("candidate_meld_tiles", 3), 4.0);
        assert_eq!(row.get("candidate_meld_red", 0), 1.0);

        melds[0].tiles = vec![9, 13, 16];
        assert_eq!(bank_hand(&mut row, &mut bank, &hand, &melds, &[0; 19]), 1);
        assert_eq!(row.get("candidate_meld_tiles", 6), 2.0);
        assert_eq!(row.get("candidate_meld_tiles", 7), 2.0);
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
                &[meld],
                &[0; 19],
            );
            assert_eq!(index, meld_type as usize + 1);
            assert_eq!(
                row.get("candidate_meld_types", index * 4 + 3),
                meld_type as usize as f64
            );
        }
    }

    #[test]
    fn bank_keeps_distinct_public_yaku_availability() {
        let mut row = Row::new();
        let mut bank = Bank::default();
        let hand = [0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 124, 125];
        let mut yaku = [0; 19];
        assert_eq!(bank_hand(&mut row, &mut bank, &hand, &[], &yaku), 0);
        yaku[1] = 1;
        assert_eq!(bank_hand(&mut row, &mut bank, &hand, &[], &yaku), 1);
        assert_eq!(bank_hand(&mut row, &mut bank, &hand, &[], &yaku), 1);
        assert_eq!(row.get("candidate_hand_yaku_possibility", 1), 0.0);
        assert_eq!(row.get("candidate_hand_yaku_possibility", 20), 1.0);
    }

    #[test]
    fn local_bank_fits_base_and_fourteen_distinct_discards() {
        let tiles = [0, 4, 8, 12, 16, 17, 20, 24, 28, 32, 36, 40, 44, 48];
        let mut row = Row::new();
        let mut bank = Bank::default();
        assert_eq!(bank_hand(&mut row, &mut bank, &tiles, &[], &[0; 19]), 0);
        for cut in 0..tiles.len() {
            let mut hand = tiles.to_vec();
            hand.remove(cut);
            assert_eq!(
                bank_hand(&mut row, &mut bank, &hand, &[], &[0; 19]),
                cut + 1
            );
        }
        assert_eq!(row.get("candidate_hand_valid", 14), 1.0);
        assert_eq!(row.get("candidate_hand_valid", 15), 0.0);
    }
}

impl serde::Serialize for Legal {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let rows = std::iter::once(self.discards.as_slice())
            .chain(std::iter::once(self.riichi.as_slice()))
            .chain(self.calls.iter().map(|row| row.as_slice()))
            .collect::<Vec<_>>();
        serde::Serialize::serialize(&rows, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Legal {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let rows = <Vec<Vec<bool>> as serde::Deserialize>::deserialize(deserializer)?;
        let rows = rows
            .into_iter()
            .map(|row| {
                row.try_into()
                    .map_err(|_| serde::de::Error::custom("expected 37 legal tiles"))
            })
            .collect::<Result<Vec<[bool; 37]>, D::Error>>()?;
        let rows: [[bool; 37]; 6] = rows
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected six legal rows"))?;
        Ok(Self {
            discards: rows[0],
            riichi: rows[1],
            calls: [rows[2], rows[3], rows[4], rows[5]],
        })
    }
}
