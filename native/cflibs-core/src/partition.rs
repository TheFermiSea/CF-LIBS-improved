use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Evaluate partition functions for all species at all temperatures.
///
/// Computes `log(U) = sum_n( a_n * (ln T)^n )` for each species, then returns `exp(log(U))`.
///
/// Parameters
/// ----------
/// coefficients : numpy array, shape (N_species, N_coeffs)
///     Partition function polynomial coefficients. N_coeffs is typically 5.
/// temperatures : numpy array, shape (N_temps,)
///     Temperatures in Kelvin.
///
/// Returns
/// -------
/// numpy array, shape (N_species, N_temps)
///     Partition function values U(T) for each species at each temperature.
#[pyfunction]
pub fn batch_partition_functions<'py>(
    py: Python<'py>,
    coefficients: PyReadonlyArray2<'py, f64>,
    temperatures: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let coeffs = coefficients.as_array();
    let temps = temperatures.as_array();
    let n_species = coeffs.shape()[0];
    let n_coeffs = coeffs.shape()[1];
    let n_temps = temps.len();

    // Build temperatures slice for rayon
    let temps_slice: Vec<f64> = temps.iter().cloned().collect();

    // Build coefficients as Vec<Vec<f64>> for rayon access
    let coeffs_vec: Vec<Vec<f64>> = (0..n_species)
        .map(|s| (0..n_coeffs).map(|c| coeffs[[s, c]]).collect())
        .collect();

    // Parallel computation over species
    let rows: Vec<Vec<f64>> = coeffs_vec
        .par_iter()
        .map(|species_coeffs| {
            temps_slice
                .iter()
                .map(|&temp| {
                    if temp <= 1.0 {
                        return 1.0;
                    }
                    let ln_t = temp.ln();
                    let mut ln_u = 0.0;
                    let mut ln_t_power = 1.0;
                    for &c in species_coeffs {
                        ln_u += c * ln_t_power;
                        ln_t_power *= ln_t;
                    }
                    ln_u.exp()
                })
                .collect()
        })
        .collect();

    // Build ndarray from rows
    let mut result = Array2::<f64>::zeros((n_species, n_temps));
    for (s, row) in rows.iter().enumerate() {
        for (t, &val) in row.iter().enumerate() {
            result[[s, t]] = val;
        }
    }

    Ok(PyArray2::from_owned_array(py, result))
}

#[cfg(test)]
mod tests {
    /// Test the core partition function computation logic (without Python/PyO3).
    fn eval_partition(coeffs: &[f64], temp: f64) -> f64 {
        if temp <= 1.0 {
            return 1.0;
        }
        let ln_t = temp.ln();
        let mut ln_u = 0.0;
        let mut ln_t_power = 1.0;
        for &c in coeffs {
            ln_u += c * ln_t_power;
            ln_t_power *= ln_t;
        }
        ln_u.exp()
    }

    #[test]
    fn test_single_species_constant() {
        // coeffs = [1.0, 0, 0, 0, 0] -> ln(U) = 1.0 -> U = e^1 = 2.718...
        let coeffs = [1.0, 0.0, 0.0, 0.0, 0.0];
        let result = eval_partition(&coeffs, 10000.0);
        assert!((result - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_single_species_linear() {
        // coeffs = [0.0, 1.0, 0, 0, 0] -> ln(U) = ln(T) -> U = T
        let coeffs = [0.0, 1.0, 0.0, 0.0, 0.0];
        let result = eval_partition(&coeffs, 5000.0);
        assert!((result - 5000.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_temperature() {
        let coeffs = [1.0, 0.5, 0.1, 0.0, 0.0];
        let result = eval_partition(&coeffs, 0.0);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_one_temperature() {
        let coeffs = [1.0, 0.5, 0.1, 0.0, 0.0];
        let result = eval_partition(&coeffs, 1.0);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_negative_temperature() {
        // T <= 1.0 should return 1.0
        let coeffs = [1.0, 0.5, 0.1, 0.0, 0.0];
        let result = eval_partition(&coeffs, -100.0);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_known_value() {
        // coeffs = [0.0, 0.0, 0.0, 0.0, 0.0] -> ln(U) = 0 -> U = 1.0
        let coeffs = [0.0, 0.0, 0.0, 0.0, 0.0];
        let result = eval_partition(&coeffs, 10000.0);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic() {
        // coeffs = [0.0, 0.0, 1.0, 0, 0] -> ln(U) = (ln T)^2 -> U = exp((ln T)^2)
        let coeffs = [0.0, 0.0, 1.0, 0.0, 0.0];
        let temp: f64 = 100.0;
        let ln_t = temp.ln();
        let expected = (ln_t * ln_t).exp();
        let result = eval_partition(&coeffs, temp);
        assert!((result - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_full_polynomial() {
        // Test with all 5 coefficients
        let coeffs: [f64; 5] = [0.5, 0.3, -0.02, 0.001, -0.00005];
        let temp: f64 = 8000.0;
        let ln_t = temp.ln();
        let expected_ln_u = coeffs[0]
            + coeffs[1] * ln_t
            + coeffs[2] * ln_t.powi(2)
            + coeffs[3] * ln_t.powi(3)
            + coeffs[4] * ln_t.powi(4);
        let expected = expected_ln_u.exp();
        let result = eval_partition(&coeffs, temp);
        assert!(
            (result - expected).abs() / expected < 1e-10,
            "result={result}, expected={expected}"
        );
    }

    #[test]
    fn test_varying_coefficient_count() {
        // Fewer than 5 coefficients
        let coeffs: [f64; 2] = [1.0, 0.5];
        let temp: f64 = 5000.0;
        let ln_t = temp.ln();
        let expected = (1.0_f64 + 0.5 * ln_t).exp();
        let result = eval_partition(&coeffs, temp);
        assert!((result - expected).abs() / expected < 1e-10);
    }
}
