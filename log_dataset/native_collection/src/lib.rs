mod actions;
mod candidate;
mod collector;
mod features;
mod focus;
mod history;
mod nyanten;
mod packed;
mod schema;
mod scoring;
#[cfg(test)]
mod yaku;
use packed::{FIELDS, Row};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
};
fn columns<'py>(py: Python<'py>, rows: &[&Row]) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    for field in FIELDS {
        let bytes = PyBytes::new_with(py, field.size * rows.len(), |bytes| {
            for (destination, row) in bytes.chunks_exact_mut(field.size).zip(rows) {
                destination.copy_from_slice(&row.bytes[field.offset..field.offset + field.size]);
            }
            Ok(())
        })?;
        output.set_item(field.name, bytes)?;
    }
    Ok(output)
}
#[pyclass]
struct FeatureReplay {
    history: history::PublicHistory,
}
#[pymethods]
impl FeatureReplay {
    #[new]
    fn new() -> Self {
        Self {
            history: history::PublicHistory::new(),
        }
    }
    fn apply(&mut self, events: &str) -> PyResult<()> {
        let events: Vec<serde_json::Value> = serde_json::from_str(events)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        for event in events {
            self.history.apply(&event);
            if event["type"] == "tsumo" {
                if let Some(left) = event["tiles_left"].as_i64() {
                    self.history.live_wall = left as i32;
                }
            }
        }
        Ok(())
    }
    #[pyo3(signature=(actor,phase,east_only=false,include_focus=false))]
    fn pack<'py>(
        &mut self,
        py: Python<'py>,
        actor: usize,
        phase: u8,
        east_only: bool,
        include_focus: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (row, metadata) = py.detach(|| {
            let obs = self.history.legal_observation(actor);
            let mut row = Row::new();
            let (_, focus) = features::pack(
                &mut row,
                &mut self.history,
                &obs,
                actor,
                phase,
                east_only,
                include_focus,
            );
            (row, focus.metadata())
        });
        let output = columns(py, &[&row])?;
        if include_focus {
            output.set_item("focus", metadata)?;
        }
        Ok(output)
    }
}
#[pymodule]
fn _native_collection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let inputs = PyDict::new(m.py());
    for field in FIELDS {
        let spec = PyDict::new(m.py());
        spec.set_item("shape", field.shape.to_vec())?;
        spec.set_item("dtype", field.dtype.name())?;
        inputs.set_item(field.name, spec)?;
    }
    m.add("INPUTS", inputs)?;
    m.add(
        "IGNORED_INPUTS",
        FIELDS
            .iter()
            .filter(|field| {
                field.name.starts_with("discard_action_") && field.name.contains("_upgrade_")
            })
            .map(|field| field.name)
            .collect::<Vec<_>>(),
    )?;
    m.add_function(wrap_pyfunction!(scoring::score_hand, m)?)?;
    m.add_class::<FeatureReplay>()?;
    m.add_class::<collector::NativeCollector>()?;
    Ok(())
}
