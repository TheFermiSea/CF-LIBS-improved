use pyo3::prelude::*;

mod comb_matching;
mod partition;
mod types;

/// High-performance computational core for CF-LIBS.
///
/// Provides Rust-accelerated implementations of:
/// - Comb matching (scan_comb_shifts, kdet_filter_elements)
/// - Partition function evaluation (batch_partition_functions)
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(comb_matching::scan_comb_shifts, m)?)?;
    m.add_function(wrap_pyfunction!(comb_matching::kdet_filter_elements, m)?)?;
    m.add_function(wrap_pyfunction!(partition::batch_partition_functions, m)?)?;
    Ok(())
}
