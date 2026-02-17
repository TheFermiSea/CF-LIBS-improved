"""
Line detection utilities for CF-LIBS inversion.

Provides a lightweight peak detection + line matching pipeline to convert
raw spectra into LineObservation objects for classic CF-LIBS solvers.

Uses the canonical ``preprocessing.detect_peaks_auto`` pipeline for
baseline-subtracted, noise-calibrated peak detection.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import Transition
from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation
from cflibs.inversion.preprocessing import detect_peaks_auto

logger = get_logger("inversion.line_detection")


@dataclass
class LineDetectionResult:
    """Result of line detection and matching."""

    observations: List[LineObservation]
    resonance_lines: Set[Tuple[str, int, float]]
    total_peaks: int
    matched_peaks: int
    unmatched_peaks: int
    warnings: List[str] = field(default_factory=list)


def detect_line_observations(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    atomic_db: AtomicDatabase,
    elements: List[str],
    wavelength_tolerance_nm: float = 0.1,
    min_peak_height: float = 0.01,
    peak_width_nm: float = 0.2,
    min_relative_intensity: Optional[float] = None,
    ground_state_threshold_ev: float = 0.1,
    resolving_power: Optional[float] = None,
    min_aki_gk: float = 0.0,
    threshold_factor: float = 4.0,
) -> LineDetectionResult:
    """
    Detect spectral peaks and match them to known atomic transitions.

    Uses baseline-subtracted peak detection with noise-calibrated
    thresholds.  Enforces one-to-one peak-transition matching and
    integrates line area on baseline-subtracted intensity.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength axis in nm (monotonic)
    intensity : np.ndarray
        Spectral intensity values
    atomic_db : AtomicDatabase
        Atomic database instance
    elements : List[str]
        Elements to match against
    wavelength_tolerance_nm : float
        Matching tolerance for known lines in nm
    min_peak_height : float
        Minimum peak height as fraction of max intensity (legacy, used
        as fallback if baseline estimation fails)
    peak_width_nm : float
        Expected peak width for integration (nm).  Overridden by
        ``resolving_power`` when provided.
    min_relative_intensity : float, optional
        Minimum relative intensity threshold for database lines
    ground_state_threshold_ev : float
        Lower-level energy threshold for resonance detection
    resolving_power : float, optional
        Instrument resolving power R = lambda/delta_lambda.  When set,
        integration widths are computed per-line as wavelength/R.
    min_aki_gk : float
        Minimum A_ki * g_k product for transition filtering.  Removes
        unobservable weak lines from the matching pool (default 0.0).
    threshold_factor : float
        Peak detection threshold in noise sigma units (default 4.0)

    Returns
    -------
    LineDetectionResult
        Detected line observations and resonance set
    """
    if wavelength.size == 0 or intensity.size == 0:
        return LineDetectionResult([], set(), 0, 0, 0, ["empty_spectrum"])

    if wavelength.size != intensity.size:
        raise ValueError("Wavelength and intensity arrays must be the same length")

    if not elements:
        return LineDetectionResult([], set(), 0, 0, 0, ["no_elements_specified"])

    wl_min = float(np.min(wavelength))
    wl_max = float(np.max(wavelength))

    transitions = _load_transitions(
        atomic_db,
        elements,
        wavelength_min=wl_min,
        wavelength_max=wl_max,
        min_relative_intensity=min_relative_intensity,
        min_aki_gk=min_aki_gk,
    )

    if not transitions:
        return LineDetectionResult([], set(), 0, 0, 0, ["no_transitions_found"])

    # Canonical peak detection with baseline subtraction
    peaks, baseline, noise = detect_peaks_auto(
        wavelength,
        intensity,
        resolving_power=resolving_power,
        threshold_factor=threshold_factor,
    )

    observations: List[LineObservation] = []
    resonance_lines: Set[Tuple[str, int, float]] = set()
    seen_keys: Set[Tuple[str, int, float]] = set()

    wl_step = _estimate_wl_step(wavelength)

    # One-to-one matching: assign each peak to at most one transition
    peak_assignments = _match_peaks_to_transitions(peaks, transitions, wavelength_tolerance_nm)

    matched_peaks = 0

    for peak_idx, peak_wl, transition in peak_assignments:
        key = (transition.element, transition.ionization_stage, transition.wavelength_nm)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Resolution-aware integration half-width
        if resolving_power is not None and resolving_power > 0:
            line_fwhm_nm = peak_wl / resolving_power
        else:
            line_fwhm_nm = peak_width_nm

        half_width_px = max(int((line_fwhm_nm / max(wl_step, 1e-9)) / 2), 1)

        start_idx = max(0, peak_idx - half_width_px)
        end_idx = min(len(intensity), peak_idx + half_width_px + 1)
        segment_wl = wavelength[start_idx:end_idx]

        # Integrate baseline-subtracted intensity
        segment_corrected = intensity[start_idx:end_idx] - baseline[start_idx:end_idx]
        segment_corrected = np.maximum(segment_corrected, 0.0)

        line_area = float(np.trapezoid(segment_corrected, segment_wl))
        if line_area <= 0:
            continue

        matched_peaks += 1

        # Poisson noise approximation for integrated intensity
        counts = np.maximum(intensity[start_idx:end_idx], 1.0)
        line_unc = float(np.sqrt(np.sum(counts)) * wl_step)

        observations.append(
            LineObservation(
                wavelength_nm=float(transition.wavelength_nm),
                intensity=line_area,
                intensity_uncertainty=max(line_unc, 1e-6),
                element=transition.element,
                ionization_stage=transition.ionization_stage,
                E_k_ev=transition.E_k_ev,
                g_k=transition.g_k,
                A_ki=transition.A_ki,
            )
        )

        is_resonance = transition.is_resonance
        if is_resonance is None:
            is_resonance = transition.E_i_ev < ground_state_threshold_ev
        if is_resonance:
            resonance_lines.add(key)

    total_peaks = len(peaks)
    unmatched_peaks = max(total_peaks - matched_peaks, 0)

    warnings: List[str] = []
    if matched_peaks == 0 and total_peaks > 0:
        warnings.append("no_peaks_matched")
    if matched_peaks == 0 and total_peaks == 0:
        warnings.append("no_peaks_detected")

    return LineDetectionResult(
        observations=observations,
        resonance_lines=resonance_lines,
        total_peaks=total_peaks,
        matched_peaks=matched_peaks,
        unmatched_peaks=unmatched_peaks,
        warnings=warnings,
    )


def _load_transitions(
    atomic_db: AtomicDatabase,
    elements: List[str],
    wavelength_min: float,
    wavelength_max: float,
    min_relative_intensity: Optional[float],
    min_aki_gk: float = 0.0,
) -> List[Transition]:
    transitions: List[Transition] = []
    for element in elements:
        trans_list = atomic_db.get_transitions(
            element,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            min_relative_intensity=min_relative_intensity,
        )
        if min_aki_gk > 0:
            trans_list = [t for t in trans_list if t.A_ki * t.g_k >= min_aki_gk]
        transitions.extend(trans_list)
    return transitions


def _match_peaks_to_transitions(
    peaks: List[Tuple[int, float]],
    transitions: List[Transition],
    tolerance_nm: float,
) -> List[Tuple[int, float, Transition]]:
    """One-to-one greedy matching of peaks to transitions.

    Each peak is assigned to at most one transition and vice versa,
    with conflicts resolved by closest wavelength distance
    (no emissivity weighting).

    Returns list of (peak_index, peak_wavelength, transition) tuples.
    """
    if not peaks or not transitions:
        return []

    peak_wls = np.array([p[1] for p in peaks])

    # Build ALL candidate matches within tolerance: (distance, peak_idx, transition)
    # Using all pairs ensures the second-closest peak can still be matched
    # if the nearest was claimed by another transition.
    candidates = []
    for trans in transitions:
        distances = np.abs(peak_wls - trans.wavelength_nm)
        for p_idx in range(len(peak_wls)):
            if distances[p_idx] <= tolerance_nm:
                candidates.append((distances[p_idx], p_idx, trans))

    # Sort by distance (best matches first)
    candidates.sort(key=lambda c: c[0])

    # Greedy one-to-one assignment
    claimed_peaks: set = set()
    claimed_transitions: set = set()
    assignments: List[Tuple[int, float, Transition]] = []

    for _dist, p_idx, trans in candidates:
        t_key = (trans.element, trans.ionization_stage, trans.wavelength_nm)
        if p_idx in claimed_peaks or t_key in claimed_transitions:
            continue
        claimed_peaks.add(p_idx)
        claimed_transitions.add(t_key)
        assignments.append((peaks[p_idx][0], peaks[p_idx][1], trans))

    return assignments


def _estimate_wl_step(wavelength: np.ndarray) -> float:
    if wavelength.size < 2:
        return 1.0
    diffs = np.diff(wavelength)
    diffs = diffs[np.isfinite(diffs)]
    return float(np.median(diffs)) if diffs.size else 1.0
