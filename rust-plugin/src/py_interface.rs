use anyhow::{Context, Result};
use numpy::ToPyArray;
use pyo3::prelude::*;

pub struct CflibsInterface {
    module: PyObject,
}

impl CflibsInterface {
    pub fn new(model_path: &str) -> Result<Self> {
        Python::with_gil(|py| {
            // Add current directory to sys.path to find local modules if needed
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            path.call_method1("append", (".",))?;

            // Allow adding the model path if it's different
            if !model_path.is_empty() {
                path.call_method1("append", (model_path,))?;
            }

            // Import cflibs (assuming it's installed in the environment)
            // cflibs.inversion.daq_interface was created earlier
            let cflibs = py.import("cflibs.inversion.daq_interface").context(
                "Failed to import cflibs.inversion.daq_interface. Ensure CF-LIBS is installed.",
            )?;

            Ok(Self {
                module: cflibs.into(),
            })
        })
    }

    pub fn invert_spectrum(
        &self,
        wavelength: &[f64],
        intensity: &[f64],
    ) -> Result<serde_json::Value> {
        Python::with_gil(|py| {
            let wavelength_py = wavelength.to_pyarray(py);
            let intensity_py = intensity.to_pyarray(py);

            let args = (wavelength_py, intensity_py);

            let result = self.module.call_method1(py, "process_spectrum", args)?;

            // Simple serialization via string for now to avoid complex Py->Rust struct mapping
            // In a production plugin, we might decode the dict directly
            let json_str = result.call_method0(py, "__str__")?.extract::<String>(py)?;

            // For now, just return a dummy success to verify the call worked
            Ok(serde_json::json!({ "status": "ok", "raw": json_str }))
        })
    }
}
