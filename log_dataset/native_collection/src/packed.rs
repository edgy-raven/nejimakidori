pub use crate::schema::*;
#[cfg(test)]
use half::f16;
use std::{collections::HashMap, sync::LazyLock};
static INDEX: LazyLock<HashMap<&'static str, &'static Field>> =
    LazyLock::new(|| FIELDS.iter().map(|f| (f.name, f)).collect());
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Row {
    pub bytes: Vec<u8>,
    pub post_calls: Vec<(u8, Row)>,
    pub reference_dora: [u8; 10],
}
impl Row {
    pub fn new() -> Self {
        let mut bytes = vec![0; ROW_BYTES];
        for (name, sentinel) in [
            ("candidate_meld_tiles", 34),
            ("meld_tiles", 37),
            ("last_tedashi_tile_id", 37),
        ] {
            let field = INDEX[name];
            bytes[field.offset..field.offset + field.size].fill(sentinel);
        }
        Self {
            bytes,
            post_calls: Vec::new(),
            reference_dora: [0; 10],
        }
    }
    #[cfg(test)]
    pub fn get(&self, name: &str, index: usize) -> f64 {
        let f = INDEX[name];
        assert!(index < f.count);
        match f.dtype {
            Dtype::INT8 => self.bytes[f.offset + index] as i8 as f64,
            Dtype::INT16 => {
                i16::from_le_bytes(self.bytes[f.offset + index * 2..][..2].try_into().unwrap())
                    as f64
            }
            Dtype::FLOAT16 => {
                f16::from_le_bytes(self.bytes[f.offset + index * 2..][..2].try_into().unwrap())
                    .to_f64()
            }
            Dtype::INT32 => {
                i32::from_le_bytes(self.bytes[f.offset + index * 4..][..4].try_into().unwrap())
                    as f64
            }
            Dtype::FLOAT32 => {
                f32::from_le_bytes(self.bytes[f.offset + index * 4..][..4].try_into().unwrap())
                    as f64
            }
        }
    }
    pub fn set(&mut self, name: &str, index: usize, value: f64) {
        self.write_values(INDEX[name], index, &[value]);
    }
    fn write_values(&mut self, field: &Field, offset: usize, values: &[f64]) {
        assert!(offset + values.len() <= field.count);
        unsafe extern "C" {
            fn feature_row_set(row: *mut u8, offset: usize, dtype: i32, index: usize, value: f64);
        }
        for (i, &value) in values.iter().enumerate() {
            unsafe {
                feature_row_set(
                    self.bytes.as_mut_ptr(),
                    field.offset,
                    field.dtype as i32,
                    offset + i,
                    value,
                );
            }
        }
    }
    #[cfg(test)]
    pub fn put_at(&mut self, name: &str, offset: usize, values: &[f64]) {
        self.write_values(INDEX[name], offset, values);
    }
    #[cfg(test)]
    pub fn put(&mut self, name: &str, values: &[f64]) {
        let field = INDEX[name];
        assert_eq!(values.len(), field.count);
        self.write_values(field, 0, values);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_writes_preserve_wire_bytes_and_partial_field_offsets() {
        let mut row = Row::new();
        row.put_at("candidate_yaku_shanten", 3, &[-1., 14.]);
        row.set("honba", 0, -300.);
        row.put_at("scores", 1, &[-50000., 125000.]);
        row.put_at("meld_call_age", 2, &[1.5, -2.]);
        row.put("self_efficiency", &[0.5, -2., 3.25]);
        for (name, offset, expected) in [
            ("candidate_yaku_shanten", 2, &[0, 255, 14, 0][..]),
            ("honba", 0, &[212, 254][..]),
            ("scores", 4, &[176, 60, 255, 255, 72, 232, 1, 0][..]),
            ("meld_call_age", 2, &[0, 0, 0, 62, 0, 192, 0, 0][..]),
            (
                "self_efficiency",
                0,
                &[0, 0, 0, 63, 0, 0, 0, 192, 0, 0, 80, 64][..],
            ),
        ] {
            let start = INDEX[name].offset + offset;
            assert_eq!(&row.bytes[start..start + expected.len()], expected);
        }
        assert_eq!(row.get("candidate_yaku_shanten", 3), -1.);
        assert_eq!(row.get("honba", 0), -300.);
        assert_eq!(row.get("scores", 1), -50000.);
        assert_eq!(row.get("scores", 2), 125000.);
        assert_eq!(row.get("meld_call_age", 2), 1.5);
        assert_eq!(row.get("self_efficiency", 2), 3.25);
    }

    #[test]
    #[should_panic]
    fn partial_write_cannot_cross_into_the_next_field() {
        Row::new().put_at("honba", 0, &[1., 2.]);
    }
}
