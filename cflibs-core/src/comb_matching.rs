#![allow(clippy::too_many_arguments)]

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::types::CombScoreResult;

/// Count how many peaks match transitions within tolerance at a given shift.
///
/// Uses binary search (partition_point) on the sorted transitions array
/// for O(N log M) complexity where N=peaks, M=transitions.
fn count_matches(peaks: &[f64], transitions: &[f64], tolerance: f64, shift: f64) -> usize {
    let mut count = 0;
    let mut used = vec![false; transitions.len()];
    for &peak in peaks {
        let shifted = peak + shift;
        // Binary search for the insertion point
        let idx = transitions.partition_point(|&t| t < shifted);

        // Determine the best (closest) candidate among right and left neighbors,
        // but only if the transition has not already been consumed.
        let right_ok = idx < transitions.len()
            && !used[idx]
            && (transitions[idx] - shifted).abs() <= tolerance;
        let left_ok =
            idx > 0 && !used[idx - 1] && (transitions[idx - 1] - shifted).abs() <= tolerance;

        let chosen = match (right_ok, left_ok) {
            (true, true) => {
                // Both candidates available; pick the closer one
                let d_right = (transitions[idx] - shifted).abs();
                let d_left = (transitions[idx - 1] - shifted).abs();
                if d_left <= d_right {
                    Some(idx - 1)
                } else {
                    Some(idx)
                }
            }
            (true, false) => Some(idx),
            (false, true) => Some(idx - 1),
            (false, false) => None,
        };

        if let Some(ti) = chosen {
            used[ti] = true;
            count += 1;
        }
    }
    count
}

/// Score a single element at a given shift.
fn score_element(
    peak_wavelengths: &[f64],
    transition_wavelengths: &[f64],
    shift: f64,
    total_peaks: usize,
    tolerance: f64,
    min_matches: usize,
    min_precision: f64,
    min_recall: f64,
    max_missing_fraction: f64,
) -> CombScoreResult {
    let expected_lines = transition_wavelengths.len();
    if expected_lines == 0 {
        return CombScoreResult {
            element: String::new(),
            matched_lines: 0,
            expected_lines: 0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            missing_fraction: 1.0,
            passes: false,
        };
    }

    let matched_lines = count_matches(peak_wavelengths, transition_wavelengths, tolerance, shift);
    let precision = matched_lines as f64 / total_peaks.max(1) as f64;
    let recall = matched_lines as f64 / expected_lines.max(1) as f64;
    let f1_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let missing_fraction = 1.0 - recall;
    let passes = matched_lines >= min_matches
        && precision >= min_precision
        && recall >= min_recall
        && missing_fraction <= max_missing_fraction;

    CombScoreResult {
        element: String::new(), // filled in by caller
        matched_lines,
        expected_lines,
        precision,
        recall,
        f1_score,
        missing_fraction,
        passes,
    }
}

/// Scan all shifts in the grid and return the best and fallback summaries.
///
/// This is the Rust equivalent of `_scan_comb_shifts` in line_detection.py.
///
/// Parameters
/// ----------
/// peak_wavelengths : numpy array (1D float64)
///     Detected peak positions in nm.
/// transition_wavelengths : list of lists
///     Per-element transition wavelengths (sorted).
/// element_names : list of str
///     Element name for each transition list.
/// shift_grid : numpy array (1D float64)
///     Shifts to scan in nm.
/// wavelength_tolerance : float
///     Matching tolerance in nm.
/// min_matches : int
///     Minimum matched lines for acceptance.
/// min_precision : float
///     Minimum precision for acceptance.
/// min_recall : float
///     Minimum recall for acceptance.
/// max_missing_fraction : float
///     Maximum missing fraction for acceptance.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys: best_shift, best_scores, fallback_shift, fallback_scores.
///     Each *_scores is a dict mapping element name to a dict with keys:
///     matched_lines, expected_lines, precision, recall, f1_score, missing_fraction, passes.
#[pyfunction]
#[pyo3(signature = (
    peak_wavelengths,
    transition_wavelengths,
    element_names,
    shift_grid,
    wavelength_tolerance,
    min_matches,
    min_precision,
    min_recall,
    max_missing_fraction,
))]
pub fn scan_comb_shifts<'py>(
    py: Python<'py>,
    peak_wavelengths: PyReadonlyArray1<'py, f64>,
    transition_wavelengths: &Bound<'py, PyList>,
    element_names: Vec<String>,
    shift_grid: PyReadonlyArray1<'py, f64>,
    wavelength_tolerance: f64,
    min_matches: usize,
    min_precision: f64,
    min_recall: f64,
    max_missing_fraction: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let peaks = peak_wavelengths.as_slice()?;
    let shifts = shift_grid.as_slice()?;
    let total_peaks = peaks.len();

    // Extract per-element transition wavelengths into sorted Vec<Vec<f64>>
    let n_elements = element_names.len();
    if transition_wavelengths.len() != n_elements {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "element_names ({}) and transition_wavelengths ({}) must have the same length",
            n_elements,
            transition_wavelengths.len(),
        )));
    }
    let mut all_transitions: Vec<Vec<f64>> = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let item = transition_wavelengths.get_item(i)?;
        let inner_list = item.downcast::<PyList>()?;
        let mut wls: Vec<f64> = Vec::with_capacity(inner_list.len());
        for j in 0..inner_list.len() {
            let val: f64 = inner_list.get_item(j)?.extract()?;
            wls.push(val);
        }
        wls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_transitions.push(wls);
    }

    // For each shift, compute scores for all elements (parallelized over shifts)
    struct ShiftResult {
        shift: f64,
        scores: Vec<(String, CombScoreResult)>,
        total_matches_all: usize,
        total_matches_pass: usize,
        total_f1: f64,
        passed_elements: Vec<String>,
    }

    let shift_results: Vec<ShiftResult> = shifts
        .par_iter()
        .map(|&shift_nm| {
            let mut scores = Vec::with_capacity(n_elements);
            let mut total_matches_all = 0usize;
            let mut total_matches_pass = 0usize;
            let mut total_f1 = 0.0f64;
            let mut passed_elements = Vec::new();

            for (i, element) in element_names.iter().enumerate() {
                let trans = &all_transitions[i];
                if trans.is_empty() {
                    continue;
                }
                let mut score = score_element(
                    peaks,
                    trans,
                    shift_nm,
                    total_peaks,
                    wavelength_tolerance,
                    min_matches,
                    min_precision,
                    min_recall,
                    max_missing_fraction,
                );
                score.element = element.clone();
                total_matches_all += score.matched_lines;
                if score.passes {
                    passed_elements.push(element.clone());
                    total_f1 += score.f1_score;
                    total_matches_pass += score.matched_lines;
                }
                scores.push((element.clone(), score));
            }

            ShiftResult {
                shift: shift_nm,
                scores,
                total_matches_all,
                total_matches_pass,
                total_f1,
                passed_elements,
            }
        })
        .collect();

    // Find best (highest total_f1 among passed) and fallback (highest total_matches_all)
    let mut best: Option<&ShiftResult> = None;
    let mut fallback: Option<&ShiftResult> = None;

    for sr in &shift_results {
        // Update fallback (highest total_matches_all, tiebreak by smallest |shift|)
        match fallback {
            None => fallback = Some(sr),
            Some(prev) => {
                if sr.total_matches_all > prev.total_matches_all
                    || (sr.total_matches_all == prev.total_matches_all
                        && sr.shift.abs() < prev.shift.abs())
                {
                    fallback = Some(sr);
                }
            }
        }

        // Update best (highest total_f1, tiebreak by matches_pass, then smallest |shift|)
        match best {
            None => best = Some(sr),
            Some(prev) => {
                let better = if sr.total_f1 > prev.total_f1 {
                    true
                } else if (sr.total_f1 - prev.total_f1).abs() < 1e-12 {
                    if sr.total_matches_pass > prev.total_matches_pass {
                        true
                    } else {
                        sr.total_matches_pass == prev.total_matches_pass
                            && sr.shift.abs() < prev.shift.abs()
                    }
                } else {
                    false
                };
                if better {
                    best = Some(sr);
                }
            }
        }
    }

    // Build result dict
    let result = PyDict::new(py);

    fn build_scores_dict<'py>(py: Python<'py>, sr: &ShiftResult) -> PyResult<Bound<'py, PyDict>> {
        let scores_dict = PyDict::new(py);
        for (name, score) in &sr.scores {
            let entry = PyDict::new(py);
            entry.set_item("matched_lines", score.matched_lines)?;
            entry.set_item("expected_lines", score.expected_lines)?;
            entry.set_item("precision", score.precision)?;
            entry.set_item("recall", score.recall)?;
            entry.set_item("f1_score", score.f1_score)?;
            entry.set_item("missing_fraction", score.missing_fraction)?;
            entry.set_item("passes", score.passes)?;
            scores_dict.set_item(name, entry)?;
        }
        Ok(scores_dict)
    }

    if let Some(b) = best {
        result.set_item("best_shift", b.shift)?;
        result.set_item("best_scores", build_scores_dict(py, b)?)?;
        result.set_item("best_total_f1", b.total_f1)?;
        result.set_item("best_total_matches", b.total_matches_pass)?;
        let passed = PyList::new(py, &b.passed_elements)?;
        result.set_item("best_passed_elements", passed)?;
    } else {
        result.set_item("best_shift", py.None())?;
        result.set_item("best_scores", py.None())?;
        result.set_item("best_total_f1", py.None())?;
        result.set_item("best_total_matches", py.None())?;
        result.set_item("best_passed_elements", py.None())?;
    }

    if let Some(f) = fallback {
        result.set_item("fallback_shift", f.shift)?;
        result.set_item("fallback_scores", build_scores_dict(py, f)?)?;
        result.set_item("fallback_total_matches", f.total_matches_all)?;
        let passed = PyList::new(py, &f.passed_elements)?;
        result.set_item("fallback_passed_elements", passed)?;
    } else {
        result.set_item("fallback_shift", py.None())?;
        result.set_item("fallback_scores", py.None())?;
        result.set_item("fallback_total_matches", py.None())?;
        result.set_item("fallback_passed_elements", py.None())?;
    }

    Ok(result)
}

/// Filter elements using rarity-weighted kdet scoring.
///
/// This is the Rust equivalent of `_kdet_filter_elements` in line_detection.py.
///
/// Parameters
/// ----------
/// peak_wavelengths : numpy array (1D float64)
///     Detected peak positions in nm.
/// transition_wavelengths : list of lists
///     Per-element transition wavelengths (sorted).
/// element_names : list of str
///     Element name for each transition list.
/// shift_grid : numpy array (1D float64)
///     Shifts to scan in nm.
/// wavelength_tolerance : float
///     Matching tolerance in nm.
/// min_score : float
///     Minimum kdet score (after rarity weighting) to keep an element.
/// min_candidates : int
///     Minimum candidate peaks required for kdet acceptance.
/// rarity_power : float
///     Exponent for rarity (line-density) weighting.
/// weight_clip : (float, float)
///     Clamp for rarity weighting factor (min, max).
///
/// Returns
/// -------
/// list of str
///     Element names that pass the kdet filter.
#[pyfunction]
#[pyo3(signature = (
    peak_wavelengths,
    transition_wavelengths,
    element_names,
    shift_grid,
    wavelength_tolerance,
    min_score,
    min_candidates,
    rarity_power,
    weight_clip,
))]
pub fn kdet_filter_elements<'py>(
    py: Python<'py>,
    peak_wavelengths: PyReadonlyArray1<'py, f64>,
    transition_wavelengths: &Bound<'py, PyList>,
    element_names: Vec<String>,
    shift_grid: PyReadonlyArray1<'py, f64>,
    wavelength_tolerance: f64,
    min_score: f64,
    min_candidates: usize,
    rarity_power: f64,
    weight_clip: (f64, f64),
) -> PyResult<Bound<'py, PyList>> {
    let peaks = peak_wavelengths.as_slice()?;
    let shifts = shift_grid.as_slice()?;
    let total_peaks = peaks.len();

    if total_peaks == 0 || element_names.is_empty() {
        return Ok(PyList::empty(py));
    }

    // Extract per-element transition wavelengths
    let n_elements = element_names.len();
    if transition_wavelengths.len() != n_elements {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "element_names ({}) and transition_wavelengths ({}) must have the same length",
            n_elements,
            transition_wavelengths.len(),
        )));
    }
    let mut all_transitions: Vec<Vec<f64>> = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let item = transition_wavelengths.get_item(i)?;
        let inner_list = item.downcast::<PyList>()?;
        let mut wls: Vec<f64> = Vec::with_capacity(inner_list.len());
        for j in 0..inner_list.len() {
            let val: f64 = inner_list.get_item(j)?.extract()?;
            wls.push(val);
        }
        wls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_transitions.push(wls);
    }

    // Compute wavelength range from peaks
    let wl_min = peaks.iter().cloned().fold(f64::INFINITY, f64::min);
    let wl_max = peaks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let wl_range = (wl_max - wl_min).max(1e-6);

    // Compute line densities
    let densities: Vec<f64> = all_transitions
        .iter()
        .map(|trans| trans.len() as f64 / wl_range)
        .collect();

    // Compute median density
    let median_density = {
        let mut sorted_densities = densities.clone();
        sorted_densities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted_densities.len();
        if n == 0 {
            1.0
        } else if n % 2 == 0 {
            (sorted_densities[n / 2 - 1] + sorted_densities[n / 2]) / 2.0
        } else {
            sorted_densities[n / 2]
        }
    };

    // Count candidates for each peak within tolerance of any transition
    // using binary search. For each element, find the best candidate count
    // across all shifts.
    let mut filtered_names: Vec<String> = Vec::new();

    for (i, element) in element_names.iter().enumerate() {
        let trans = &all_transitions[i];
        if trans.is_empty() {
            continue;
        }

        // For each shift, count how many peaks have at least one transition match
        let best_candidates: usize = shifts
            .par_iter()
            .map(|&shift_nm| {
                let mut candidate_count = 0usize;
                for &peak in peaks {
                    let shifted = peak + shift_nm;
                    let idx = trans.partition_point(|&t| t < shifted);
                    let within = if idx < trans.len()
                        && (trans[idx] - shifted).abs() <= wavelength_tolerance
                    {
                        true
                    } else {
                        idx > 0 && (trans[idx - 1] - shifted).abs() <= wavelength_tolerance
                    };
                    if within {
                        candidate_count += 1;
                    }
                }
                candidate_count
            })
            .max()
            .unwrap_or(0);

        let kdet_fraction = best_candidates as f64 / total_peaks as f64;
        let density = densities[i];
        let (clip_lo, clip_hi) = if weight_clip.0 <= weight_clip.1 {
            weight_clip
        } else {
            (weight_clip.1, weight_clip.0)
        };
        let rarity_weight = (median_density / density.max(1e-6))
            .powf(rarity_power)
            .clamp(clip_lo, clip_hi);
        let kdet_score = kdet_fraction * rarity_weight;

        if best_candidates >= min_candidates && kdet_score >= min_score {
            filtered_names.push(element.clone());
        }
    }

    PyList::new(py, &filtered_names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_matches_exact() {
        let peaks = vec![400.0, 500.0, 600.0];
        let transitions = vec![400.0, 500.0, 600.0];
        assert_eq!(count_matches(&peaks, &transitions, 0.1, 0.0), 3);
    }

    #[test]
    fn test_count_matches_with_tolerance() {
        let peaks = vec![400.05, 500.08, 600.12];
        let transitions = vec![400.0, 500.0, 600.0];
        // 400.05 within 0.1 of 400.0 -> match
        // 500.08 within 0.1 of 500.0 -> match
        // 600.12 not within 0.1 of 600.0 -> no match
        assert_eq!(count_matches(&peaks, &transitions, 0.1, 0.0), 2);
    }

    #[test]
    fn test_count_matches_with_shift() {
        let peaks = vec![399.0, 499.0, 599.0];
        let transitions = vec![400.0, 500.0, 600.0];
        // shift=1.0: 399+1=400, 499+1=500, 599+1=600 -> all match
        assert_eq!(count_matches(&peaks, &transitions, 0.1, 1.0), 3);
    }

    #[test]
    fn test_count_matches_no_matches() {
        let peaks = vec![100.0, 200.0, 300.0];
        let transitions = vec![400.0, 500.0, 600.0];
        assert_eq!(count_matches(&peaks, &transitions, 0.1, 0.0), 0);
    }

    #[test]
    fn test_count_matches_empty_peaks() {
        let peaks: Vec<f64> = vec![];
        let transitions = vec![400.0, 500.0];
        assert_eq!(count_matches(&peaks, &transitions, 0.1, 0.0), 0);
    }

    #[test]
    fn test_count_matches_empty_transitions() {
        let peaks = vec![400.0, 500.0];
        let transitions: Vec<f64> = vec![];
        assert_eq!(count_matches(&peaks, &transitions, 0.1, 0.0), 0);
    }

    #[test]
    fn test_score_element_all_match() {
        let peaks = vec![400.0, 500.0, 600.0];
        let transitions = vec![400.0, 500.0, 600.0];
        let score = score_element(&peaks, &transitions, 0.0, 3, 0.1, 1, 0.01, 0.1, 0.85);
        assert_eq!(score.matched_lines, 3);
        assert_eq!(score.expected_lines, 3);
        assert!((score.precision - 1.0).abs() < 1e-10);
        assert!((score.recall - 1.0).abs() < 1e-10);
        assert!((score.f1_score - 1.0).abs() < 1e-10);
        assert!((score.missing_fraction - 0.0).abs() < 1e-10);
        assert!(score.passes);
    }

    #[test]
    fn test_score_element_no_match() {
        let peaks = vec![100.0, 200.0, 300.0];
        let transitions = vec![400.0, 500.0, 600.0];
        let score = score_element(&peaks, &transitions, 0.0, 3, 0.1, 1, 0.01, 0.1, 0.85);
        assert_eq!(score.matched_lines, 0);
        assert!(!score.passes);
    }

    #[test]
    fn test_score_element_empty_transitions() {
        let peaks = vec![400.0, 500.0];
        let transitions: Vec<f64> = vec![];
        let score = score_element(&peaks, &transitions, 0.0, 2, 0.1, 1, 0.01, 0.1, 0.85);
        assert_eq!(score.matched_lines, 0);
        assert_eq!(score.expected_lines, 0);
        assert!(!score.passes);
    }

    #[test]
    fn test_score_element_threshold_rejection() {
        // Only 1 match out of 10 expected lines -> recall 0.1, but need min_matches=3
        let peaks = vec![400.0, 500.0, 600.0];
        let transitions = vec![
            400.0, 410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0,
        ];
        let score = score_element(&peaks, &transitions, 0.0, 3, 0.1, 3, 0.01, 0.1, 0.85);
        // Only 1 peak matches (400.0)
        assert_eq!(score.matched_lines, 1);
        assert!(!score.passes);
    }

    #[test]
    fn test_count_matches_boundary_tolerance() {
        // Test that near-boundary cases work correctly
        let peaks = vec![400.09];
        let transitions = vec![400.0];
        // Clearly within tolerance (0.09 < 0.1)
        assert_eq!(count_matches(&peaks, &transitions, 0.1, 0.0), 1);
        // Clearly beyond tolerance (0.15 > 0.1)
        let peaks2 = vec![400.15];
        assert_eq!(count_matches(&peaks2, &transitions, 0.1, 0.0), 0);
    }

    #[test]
    fn test_count_matches_one_to_one_no_double_counting() {
        // Two peaks close to the same transition: only one should match (one-to-one).
        let peaks = vec![400.02, 400.05];
        let transitions = vec![400.0];
        // Both peaks are within tolerance=0.1 of transition 400.0,
        // but only the first peak processed should consume that transition.
        assert_eq!(count_matches(&peaks, &transitions, 0.1, 0.0), 1);
    }

    #[test]
    fn test_count_matches_one_to_one_distinct_transitions() {
        // Two peaks each near a distinct transition: both should match.
        let peaks = vec![400.02, 500.03];
        let transitions = vec![400.0, 500.0];
        assert_eq!(count_matches(&peaks, &transitions, 0.1, 0.0), 2);
    }

    #[test]
    fn test_count_matches_one_to_one_three_peaks_two_transitions() {
        // Three peaks near two transitions: at most 2 matches.
        let peaks = vec![400.01, 400.05, 500.03];
        let transitions = vec![400.0, 500.0];
        // First peak consumes transition 400.0, second peak has no available
        // transition (400.0 already used), third peak matches 500.0.
        assert_eq!(count_matches(&peaks, &transitions, 0.1, 0.0), 2);
    }
}
