"""
Line detection utilities for CF-LIBS inversion.

Provides a lightweight peak detection + line matching pipeline to convert
raw spectra into LineObservation objects for classic CF-LIBS solvers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TypedDict

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import Transition
from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation

logger = get_logger("inversion.line_detection")

try:
    from scipy.signal import find_peaks

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    find_peaks = None

try:
    from cflibs._core import scan_comb_shifts as _scan_comb_shifts_rust
    from cflibs._core import kdet_filter_elements as _kdet_filter_elements_rust

    HAS_RUST_CORE = True
except ImportError:
    try:
        from _core import scan_comb_shifts as _scan_comb_shifts_rust
        from _core import kdet_filter_elements as _kdet_filter_elements_rust

        HAS_RUST_CORE = True
    except ImportError:
        HAS_RUST_CORE = False


def _build_observation_from_fit(
    transition: Transition,
    fit_result: "VoigtFitResult",
    ground_state_threshold_ev: float,
) -> Optional[Tuple[LineObservation, bool]]:
    """Build a LineObservation from a VoigtFitResult.

    Parameters
    ----------
    transition : Transition
        The matched atomic transition.
    fit_result : VoigtFitResult
        Voigt deconvolution result for this peak.
    ground_state_threshold_ev : float
        Lower-level energy threshold for resonance detection.

    Returns
    -------
    Optional[Tuple[LineObservation, bool]]
        (observation, is_resonance) tuple, or None if area is invalid.
    """
    from cflibs.inversion.deconvolution import VoigtFitResult  # noqa: F811

    line_area = fit_result.area
    if line_area <= 0:
        return None

    obs = LineObservation(
        wavelength_nm=float(transition.wavelength_nm),
        intensity=line_area,
        intensity_uncertainty=max(fit_result.area_uncertainty, 1e-6),
        element=transition.element,
        ionization_stage=transition.ionization_stage,
        E_k_ev=transition.E_k_ev,
        g_k=transition.g_k,
        A_ki=transition.A_ki,
    )

    is_resonance = transition.is_resonance
    if is_resonance is None:
        is_resonance = transition.E_i_ev < ground_state_threshold_ev
    return (obs, is_resonance)


def _build_observation(
    transition: Transition,
    peak_idx: int,
    wavelength: np.ndarray,
    intensity: np.ndarray,
    half_width_px: int,
    wl_step: float,
    ground_state_threshold_ev: float,
) -> Optional[Tuple[LineObservation, bool]]:
    """
    Build a LineObservation from a matched transition.

    Parameters
    ----------
    transition : Transition
        The matched atomic transition
    peak_idx : int
        Index of the peak in the spectrum
    wavelength : np.ndarray
        Wavelength array
    intensity : np.ndarray
        Intensity array
    half_width_px : int
        Half-width in pixels for integration window
    wl_step : float
        Wavelength step size
    ground_state_threshold_ev : float
        Lower-level energy threshold for resonance detection

    Returns
    -------
    Optional[Tuple[LineObservation, bool]]
        (observation, is_resonance) tuple, or None if line area is invalid
    """
    start_idx = max(0, peak_idx - half_width_px)
    end_idx = min(len(intensity), peak_idx + half_width_px + 1)
    segment_wl = wavelength[start_idx:end_idx]
    segment_intensity = intensity[start_idx:end_idx]

    line_area = float(np.trapezoid(segment_intensity, segment_wl))
    line_area = max(line_area, float(segment_intensity.max()))

    counts = np.maximum(segment_intensity, 1.0)
    line_unc = float(np.sqrt(np.sum(counts)) * wl_step)
    if line_area <= 0:
        return None

    obs = LineObservation(
        wavelength_nm=float(transition.wavelength_nm),
        intensity=line_area,
        intensity_uncertainty=max(line_unc, 1e-6),
        element=transition.element,
        ionization_stage=transition.ionization_stage,
        E_k_ev=transition.E_k_ev,
        g_k=transition.g_k,
        A_ki=transition.A_ki,
    )

    is_resonance = transition.is_resonance
    if is_resonance is None:
        is_resonance = transition.E_i_ev < ground_state_threshold_ev
    return (obs, is_resonance)


@dataclass
class LineDetectionResult:
    """Result of line detection and matching."""

    observations: List[LineObservation]
    resonance_lines: Set[Tuple[str, int, float]]
    total_peaks: int
    matched_peaks: int
    unmatched_peaks: int
    applied_shift_nm: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class CombScore:
    """Summary of comb scoring for a single element."""

    element: str
    matched_lines: int
    expected_lines: int
    precision: float
    recall: float
    f1_score: float
    missing_fraction: float
    passes: bool


class CombShiftSummary(TypedDict):
    shift_nm: float
    scores: Dict[str, CombScore]
    total_matches: int
    passed_elements: List[str]
    total_f1: float


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
    shift_scan_nm: float = 0.5,
    shift_step_nm: Optional[float] = None,
    comb_max_lines_per_element: int = 30,
    comb_min_matches: int = 3,
    comb_min_precision: float = 0.02,
    comb_min_recall: float = 0.1,
    comb_max_missing_fraction: float = 0.85,
    comb_fallback_to_nearest: bool = True,
    comb_fallback_max_elements: int = 5,
    kdet_enabled: bool = True,
    kdet_min_score: float = 0.05,
    kdet_min_candidates: int = 2,
    kdet_rarity_power: float = 0.5,
    kdet_weight_clip: Tuple[float, float] = (0.25, 4.0),
    use_deconvolution: bool = False,
) -> LineDetectionResult:
    """
    Detect spectral peaks and match them to known atomic transitions.

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
        Minimum peak height as fraction of max intensity
    peak_width_nm : float
        Expected peak width for integration (nm)
    min_relative_intensity : float, optional
        Minimum relative intensity threshold for database lines
    ground_state_threshold_ev : float
        Lower-level energy threshold for resonance detection
    shift_scan_nm : float
        Global wavelength shift scan range (+/- in nm)
    shift_step_nm : float, optional
        Step size for shift scan (defaults to wavelength/line tolerance derived step)
    comb_max_lines_per_element : int
        Maximum number of lines per element used for comb scoring
    comb_min_matches : int
        Minimum matched lines for comb acceptance
    comb_min_precision : float
        Minimum precision for comb acceptance
    comb_min_recall : float
        Minimum recall for comb acceptance
    comb_max_missing_fraction : float
        Maximum fraction of missing comb lines allowed
    comb_fallback_to_nearest : bool
        Fallback to nearest-line matching when comb scoring rejects all elements
    comb_fallback_max_elements : int
        Maximum elements to include in comb fallback selection
    kdet_enabled : bool
        Apply kdet filtering before comb scoring
    kdet_min_score : float
        Minimum kdet score (after rarity weighting) to keep an element
    kdet_min_candidates : int
        Minimum candidate peaks required for kdet acceptance
    kdet_rarity_power : float
        Exponent for rarity (line-density) weighting
    kdet_weight_clip : Tuple[float, float]
        Clamp for rarity weighting factor (min, max)
    use_deconvolution : bool
        If True, apply Voigt deconvolution to resolve overlapping peaks
        before building line observations (default: False).

    Returns
    -------
    LineDetectionResult
        Detected line observations and resonance set
    """
    if wavelength.size == 0 or intensity.size == 0:
        return LineDetectionResult(
            observations=[],
            resonance_lines=set(),
            total_peaks=0,
            matched_peaks=0,
            unmatched_peaks=0,
            applied_shift_nm=0.0,
            warnings=["empty_spectrum"],
        )

    if wavelength.size != intensity.size:
        raise ValueError("Wavelength and intensity arrays must be the same length")

    if not elements:
        return LineDetectionResult(
            observations=[],
            resonance_lines=set(),
            total_peaks=0,
            matched_peaks=0,
            unmatched_peaks=0,
            applied_shift_nm=0.0,
            warnings=["no_elements_specified"],
        )

    wl_min = float(np.min(wavelength))
    wl_max = float(np.max(wavelength))

    transitions = _load_transitions(
        atomic_db,
        elements,
        wavelength_min=wl_min,
        wavelength_max=wl_max,
        min_relative_intensity=min_relative_intensity,
    )

    if not transitions:
        return LineDetectionResult(
            observations=[],
            resonance_lines=set(),
            total_peaks=0,
            matched_peaks=0,
            unmatched_peaks=0,
            applied_shift_nm=0.0,
            warnings=["no_transitions_found"],
        )

    peaks = _find_peaks(wavelength, intensity, min_peak_height, peak_width_nm)

    wl_step = _estimate_wl_step(wavelength)
    half_width_px = max(int((peak_width_nm / max(wl_step, 1e-9)) / 2), 1)

    observations: List[LineObservation] = []
    resonance_lines: Set[Tuple[str, int, float]] = set()
    seen_keys: Set[Tuple[str, int, float]] = set()

    warnings: List[str] = []
    total_peaks = len(peaks)
    if total_peaks == 0:
        return LineDetectionResult(
            observations=[],
            resonance_lines=set(),
            total_peaks=0,
            matched_peaks=0,
            unmatched_peaks=0,
            applied_shift_nm=0.0,
            warnings=["no_peaks_detected"],
        )

    transitions_by_element: Dict[str, List[Transition]] = {}
    for transition in transitions:
        transitions_by_element.setdefault(transition.element, []).append(transition)

    if kdet_enabled:
        filtered_elements, kdet_warnings = _kdet_filter_elements(
            peaks=peaks,
            transitions_by_element=transitions_by_element,
            shift_scan_nm=shift_scan_nm,
            shift_step_nm=shift_step_nm,
            wavelength_tolerance_nm=wavelength_tolerance_nm,
            wl_step=wl_step,
            kdet_min_score=kdet_min_score,
            kdet_min_candidates=kdet_min_candidates,
            kdet_rarity_power=kdet_rarity_power,
            kdet_weight_clip=kdet_weight_clip,
        )
        warnings.extend(kdet_warnings)
        if filtered_elements:
            transitions_by_element = filtered_elements
        else:
            warnings.append("kdet_filtered_all_elements")

    comb_transitions_by_element = {
        element: _select_comb_transitions(transitions, comb_max_lines_per_element)
        for element, transitions in transitions_by_element.items()
    }

    shift_grid = _build_shift_grid(shift_scan_nm, shift_step_nm, wl_step, wavelength_tolerance_nm)
    best_shift_summary, fallback_shift_summary = _scan_comb_shifts(
        peaks=peaks,
        transitions_by_element=comb_transitions_by_element,
        shift_grid=shift_grid,
        total_peaks=total_peaks,
        wavelength_tolerance_nm=wavelength_tolerance_nm,
        comb_min_matches=comb_min_matches,
        comb_min_precision=comb_min_precision,
        comb_min_recall=comb_min_recall,
        comb_max_missing_fraction=comb_max_missing_fraction,
    )

    applied_shift_nm = 0.0
    element_scores: Dict[str, CombScore] = {}
    accepted_elements: List[str] = []

    if best_shift_summary is not None and best_shift_summary["passed_elements"]:
        applied_shift_nm = float(best_shift_summary["shift_nm"])
        element_scores = best_shift_summary["scores"]
        accepted_elements = list(best_shift_summary["passed_elements"])
    else:
        warnings.append("comb_no_elements_passed")
        if comb_fallback_to_nearest and fallback_shift_summary is not None:
            applied_shift_nm = float(fallback_shift_summary["shift_nm"])
            element_scores = fallback_shift_summary["scores"]
            accepted_elements = [
                element
                for element, score in element_scores.items()
                if score.matched_lines >= max(1, comb_min_matches)
            ]
            if not accepted_elements:
                accepted_elements = [
                    element
                    for element, score in sorted(
                        element_scores.items(),
                        key=lambda item: item[1].matched_lines,
                        reverse=True,
                    )[:comb_fallback_max_elements]
                    if score.matched_lines > 0
                ]
        else:
            warnings.append("comb_fallback_disabled")

    matched_peak_indices: Set[int] = set()

    # Optionally run deconvolution on matched peaks
    deconv_results_by_wl: Optional[Dict[float, "VoigtFitResult"]] = None
    if use_deconvolution:
        try:
            from cflibs.inversion.deconvolution import deconvolve_peaks, VoigtFitResult

            # Collect all peak wavelengths for deconvolution
            peak_wl_arr = np.array([pw for _, pw in peaks], dtype=float)
            if len(peak_wl_arr) > 0:
                from cflibs.inversion.preprocessing import estimate_baseline

                baseline = estimate_baseline(wavelength, intensity)
                baseline_subtracted = intensity - baseline
                deconv = deconvolve_peaks(
                    wavelength,
                    baseline_subtracted,
                    peak_wl_arr,
                    fwhm_estimate=peak_width_nm,
                )
                # Map fit results to nearest original peak wavelength
                deconv_results_by_wl = {}
                for fr in deconv.fit_results:
                    best_peak_wl = float(peak_wl_arr[np.argmin(np.abs(peak_wl_arr - fr.center_nm))])
                    deconv_results_by_wl[best_peak_wl] = fr
                logger.debug(
                    "Deconvolution produced %d fit results for %d peaks",
                    len(deconv.fit_results),
                    len(peak_wl_arr),
                )
        except Exception as exc:
            logger.warning("Deconvolution failed, falling back to trapezoid: %s", exc)
            deconv_results_by_wl = None

    if accepted_elements:

        def _score_key(element: str) -> Tuple[float, int]:
            score = element_scores.get(element)
            if score is None:
                return (0.0, 0)
            return (score.f1_score, score.matched_lines)

        accepted_elements.sort(key=_score_key, reverse=True)

        used_peaks: Set[int] = set()
        for element in accepted_elements:
            transitions = comb_transitions_by_element.get(element, [])
            if not transitions:
                continue
            matches = _match_transitions_to_peaks(
                peaks=peaks,
                transitions=transitions,
                tolerance_nm=wavelength_tolerance_nm,
                shift_nm=applied_shift_nm,
                used_peaks=used_peaks,
            )
            for transition, peak_idx, peak_wl, _delta in matches:
                key = (transition.element, transition.ionization_stage, transition.wavelength_nm)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                matched_peak_indices.add(peak_idx)

                # Use deconvolution result if available
                if deconv_results_by_wl is not None and peak_wl in deconv_results_by_wl:
                    result = _build_observation_from_fit(
                        transition,
                        deconv_results_by_wl[peak_wl],
                        ground_state_threshold_ev,
                    )
                else:
                    result = _build_observation(
                        transition,
                        peak_idx,
                        wavelength,
                        intensity,
                        half_width_px,
                        wl_step,
                        ground_state_threshold_ev,
                    )
                if result is None:
                    continue
                obs, is_resonance = result
                observations.append(obs)
                if is_resonance:
                    resonance_lines.add(key)
            for peak_idx, peak_wl in peaks:
                transition = _match_transition(
                    peak_wl + applied_shift_nm, transitions, wavelength_tolerance_nm
                )
                if transition is None:
                    continue

                key = (transition.element, transition.ionization_stage, transition.wavelength_nm)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                matched_peak_indices.add(peak_idx)

                # Use deconvolution result if available
                if deconv_results_by_wl is not None and peak_wl in deconv_results_by_wl:
                    result = _build_observation_from_fit(
                        transition,
                        deconv_results_by_wl[peak_wl],
                        ground_state_threshold_ev,
                    )
                else:
                    result = _build_observation(
                        transition,
                        peak_idx,
                        wavelength,
                        intensity,
                        half_width_px,
                        wl_step,
                        ground_state_threshold_ev,
                    )
                if result is None:
                    continue
                obs, is_resonance = result
                observations.append(obs)
                if is_resonance:
                    resonance_lines.add(key)

    matched_peaks = len(matched_peak_indices)
    unmatched_peaks = max(total_peaks - matched_peaks, 0)

    if matched_peaks == 0 and total_peaks > 0:
        warnings.append("no_peaks_matched")

    return LineDetectionResult(
        observations=observations,
        resonance_lines=resonance_lines,
        total_peaks=total_peaks,
        matched_peaks=matched_peaks,
        unmatched_peaks=unmatched_peaks,
        applied_shift_nm=applied_shift_nm,
        warnings=warnings,
    )


def _load_transitions(
    atomic_db: AtomicDatabase,
    elements: List[str],
    wavelength_min: float,
    wavelength_max: float,
    min_relative_intensity: Optional[float],
) -> List[Transition]:
    transitions: List[Transition] = []
    for element in elements:
        transitions.extend(
            atomic_db.get_transitions(
                element,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
                min_relative_intensity=min_relative_intensity,
            )
        )
    return transitions


def _transition_strength(transition: Transition) -> float:
    if transition.relative_intensity is not None and transition.relative_intensity > 0:
        return float(transition.relative_intensity)
    if transition.A_ki is not None and transition.A_ki > 0:
        return float(transition.A_ki)
    return 0.0


def _select_comb_transitions(
    transitions: List[Transition],
    max_lines: int,
) -> List[Transition]:
    sorted_t = sorted(
        transitions,
        key=lambda t: (_transition_strength(t), -t.wavelength_nm),
        reverse=True,
    )
    if max_lines > 0 and len(sorted_t) > max_lines:
        return sorted_t[:max_lines]
    return sorted_t


def _build_shift_grid(
    shift_scan_nm: float,
    shift_step_nm: Optional[float],
    wl_step: float,
    tolerance_nm: float,
) -> np.ndarray:
    if shift_scan_nm <= 0:
        return np.array([0.0])
    if shift_step_nm is None:
        shift_step_nm = max(min(tolerance_nm / 2.0, 0.05), max(wl_step, 1e-3))
    shift_step_nm = max(float(shift_step_nm), 1e-6)
    num_steps = int(np.floor((2 * shift_scan_nm) / shift_step_nm)) + 1
    if num_steps <= 1:
        return np.array([0.0])
    shifts = np.linspace(-shift_scan_nm, shift_scan_nm, num_steps)
    if not np.any(np.isclose(shifts, 0.0)):
        shifts = np.sort(np.append(shifts, 0.0))
    return shifts


def _score_comb_for_element(
    peaks: List[Tuple[int, float]],
    transitions: List[Transition],
    shift_nm: float,
    total_peaks: int,
    wavelength_tolerance_nm: float,
    comb_min_matches: int,
    comb_min_precision: float,
    comb_min_recall: float,
    comb_max_missing_fraction: float,
) -> CombScore:
    if not transitions:
        return CombScore(
            element="",
            matched_lines=0,
            expected_lines=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            missing_fraction=1.0,
            passes=False,
        )

    matches = _match_transitions_to_peaks(
        peaks=peaks,
        transitions=transitions,
        tolerance_nm=wavelength_tolerance_nm,
        shift_nm=shift_nm,
        used_peaks=None,
    )
    matched_lines = len(matches)
    expected_lines = len(transitions)
    precision = matched_lines / max(total_peaks, 1)
    recall = matched_lines / max(expected_lines, 1)
    if precision + recall > 0:
        f1_score = 2.0 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0
    missing_fraction = 1.0 - recall
    passes = (
        matched_lines >= comb_min_matches
        and precision >= comb_min_precision
        and recall >= comb_min_recall
        and missing_fraction <= comb_max_missing_fraction
    )
    return CombScore(
        element=transitions[0].element,
        matched_lines=matched_lines,
        expected_lines=expected_lines,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        missing_fraction=missing_fraction,
        passes=passes,
    )


def _scan_comb_shifts_dispatch_rust(
    peaks: List[Tuple[int, float]],
    transitions_by_element: Dict[str, List[Transition]],
    shift_grid: np.ndarray,
    total_peaks: int,
    wavelength_tolerance_nm: float,
    comb_min_matches: int,
    comb_min_precision: float,
    comb_min_recall: float,
    comb_max_missing_fraction: float,
) -> Tuple[Optional[CombShiftSummary], Optional[CombShiftSummary]]:
    """Dispatch _scan_comb_shifts to Rust implementation."""
    peak_wavelengths = np.array([p[1] for p in peaks], dtype=np.float64)
    element_names = list(transitions_by_element.keys())
    transition_wls = [
        sorted([t.wavelength_nm for t in transitions_by_element[el]])
        for el in element_names
    ]

    result = _scan_comb_shifts_rust(
        peak_wavelengths,
        transition_wls,
        element_names,
        np.asarray(shift_grid, dtype=np.float64),
        wavelength_tolerance_nm,
        comb_min_matches,
        comb_min_precision,
        comb_min_recall,
        comb_max_missing_fraction,
    )

    def _build_summary(
        shift_key: str,
        scores_key: str,
        matches_key: str,
        passed_key: str,
        f1_key: Optional[str] = None,
    ) -> Optional[CombShiftSummary]:
        shift = result.get(shift_key)
        if shift is None:
            return None
        scores_raw = result.get(scores_key, {})
        scores: Dict[str, CombScore] = {}
        for el, sdict in scores_raw.items():
            scores[el] = CombScore(
                element=el,
                matched_lines=sdict["matched_lines"],
                expected_lines=sdict["expected_lines"],
                precision=sdict["precision"],
                recall=sdict["recall"],
                f1_score=sdict["f1_score"],
                missing_fraction=sdict["missing_fraction"],
                passes=sdict["passes"],
            )
        passed = list(result.get(passed_key, []))
        total_matches = result.get(matches_key, 0)
        total_f1 = result.get(f1_key, 0.0) if f1_key else sum(
            s.f1_score for s in scores.values() if s.passes
        )
        return {
            "shift_nm": float(shift),
            "scores": scores,
            "total_matches": total_matches,
            "passed_elements": passed,
            "total_f1": total_f1,
        }

    best = _build_summary(
        "best_shift", "best_scores", "best_total_matches",
        "best_passed_elements", "best_total_f1",
    )
    fallback = _build_summary(
        "fallback_shift", "fallback_scores", "fallback_total_matches",
        "fallback_passed_elements",
    )
    return best, fallback


def _scan_comb_shifts(
    peaks: List[Tuple[int, float]],
    transitions_by_element: Dict[str, List[Transition]],
    shift_grid: np.ndarray,
    total_peaks: int,
    wavelength_tolerance_nm: float,
    comb_min_matches: int,
    comb_min_precision: float,
    comb_min_recall: float,
    comb_max_missing_fraction: float,
) -> Tuple[Optional[CombShiftSummary], Optional[CombShiftSummary]]:
    if HAS_RUST_CORE:
        try:
            return _scan_comb_shifts_dispatch_rust(
                peaks=peaks,
                transitions_by_element=transitions_by_element,
                shift_grid=shift_grid,
                total_peaks=total_peaks,
                wavelength_tolerance_nm=wavelength_tolerance_nm,
                comb_min_matches=comb_min_matches,
                comb_min_precision=comb_min_precision,
                comb_min_recall=comb_min_recall,
                comb_max_missing_fraction=comb_max_missing_fraction,
            )
        except Exception as e:
            logger.warning(f"Rust comb shift scan failed, falling back to Python: {e}")

    best_summary: Optional[CombShiftSummary] = None
    fallback_summary: Optional[CombShiftSummary] = None

    for shift_nm in shift_grid:
        scores: Dict[str, CombScore] = {}
        total_f1 = 0.0
        total_matches_pass = 0
        total_matches_all = 0
        passed_elements: List[str] = []

        for element, transitions in transitions_by_element.items():
            if not transitions:
                continue
            score = _score_comb_for_element(
                peaks=peaks,
                transitions=transitions,
                shift_nm=shift_nm,
                total_peaks=total_peaks,
                wavelength_tolerance_nm=wavelength_tolerance_nm,
                comb_min_matches=comb_min_matches,
                comb_min_precision=comb_min_precision,
                comb_min_recall=comb_min_recall,
                comb_max_missing_fraction=comb_max_missing_fraction,
            )
            score.element = element
            scores[element] = score
            total_matches_all += score.matched_lines
            if score.passes:
                passed_elements.append(element)
                total_f1 += score.f1_score
                total_matches_pass += score.matched_lines

        if fallback_summary is None:
            fallback_summary = {
                "shift_nm": float(shift_nm),
                "scores": scores,
                "total_matches": total_matches_all,
                "passed_elements": passed_elements,
                "total_f1": total_f1,
            }
        else:
            prev_matches = fallback_summary["total_matches"]
            if total_matches_all > prev_matches or (
                total_matches_all == prev_matches
                and abs(shift_nm) < abs(float(fallback_summary["shift_nm"]))
            ):
                fallback_summary = {
                    "shift_nm": float(shift_nm),
                    "scores": scores,
                    "total_matches": total_matches_all,
                    "passed_elements": passed_elements,
                    "total_f1": total_f1,
                }

        if best_summary is None:
            best_summary = {
                "shift_nm": float(shift_nm),
                "scores": scores,
                "total_matches": total_matches_pass,
                "passed_elements": passed_elements,
                "total_f1": total_f1,
            }
        else:
            prev_f1 = best_summary["total_f1"]
            prev_matches = best_summary["total_matches"]
            better = False
            if total_f1 > prev_f1:
                better = True
            elif np.isclose(total_f1, prev_f1):
                if total_matches_pass > prev_matches:
                    better = True
                elif total_matches_pass == prev_matches and abs(shift_nm) < abs(
                    float(best_summary["shift_nm"])
                ):
                    better = True
            if better:
                best_summary = {
                    "shift_nm": float(shift_nm),
                    "scores": scores,
                    "total_matches": total_matches_pass,
                    "passed_elements": passed_elements,
                    "total_f1": total_f1,
                }

    return best_summary, fallback_summary


def _match_transitions_to_peaks(
    peaks: List[Tuple[int, float]],
    transitions: List[Transition],
    tolerance_nm: float,
    shift_nm: float,
    used_peaks: Optional[Set[int]] = None,
) -> List[Tuple[Transition, int, float, float]]:
    if not peaks or not transitions:
        return []
    used = used_peaks if used_peaks is not None else set()

    peak_indices = np.array([p[0] for p in peaks], dtype=int)
    peak_wavelengths = np.array([p[1] for p in peaks], dtype=float)
    shifted_peaks = peak_wavelengths + shift_nm

    matches: List[Tuple[Transition, int, float, float]] = []
    for transition in transitions:
        deltas = np.abs(shifted_peaks - transition.wavelength_nm)
        candidate_indices = np.where(deltas <= tolerance_nm)[0]
        if candidate_indices.size == 0:
            continue
        # Exclude already used peaks
        available = [idx for idx in candidate_indices if int(peak_indices[idx]) not in used]
        if not available:
            continue
        best_idx = min(available, key=lambda idx: deltas[idx])
        peak_idx = int(peak_indices[best_idx])
        used.add(peak_idx)
        matches.append(
            (
                transition,
                peak_idx,
                float(peak_wavelengths[best_idx]),
                float(deltas[best_idx]),
            )
        )
    return matches


def _kdet_filter_elements(
    peaks: List[Tuple[int, float]],
    transitions_by_element: Dict[str, List[Transition]],
    shift_scan_nm: float,
    shift_step_nm: Optional[float],
    wavelength_tolerance_nm: float,
    wl_step: float,
    kdet_min_score: float,
    kdet_min_candidates: int,
    kdet_rarity_power: float,
    kdet_weight_clip: Tuple[float, float],
) -> Tuple[Dict[str, List[Transition]], List[str]]:
    warnings: List[str] = []
    if not peaks or not transitions_by_element:
        return transitions_by_element, warnings

    peak_wavelengths = np.array([p[1] for p in peaks], dtype=float)
    total_peaks = len(peak_wavelengths)
    if total_peaks == 0:
        return transitions_by_element, warnings

    shift_grid = _build_shift_grid(shift_scan_nm, shift_step_nm, wl_step, wavelength_tolerance_nm)

    if HAS_RUST_CORE:
        try:
            element_names = list(transitions_by_element.keys())
            transition_wls = [
                sorted([t.wavelength_nm for t in transitions_by_element[el]])
                for el in element_names
            ]
            passed_names = list(
                _kdet_filter_elements_rust(
                    np.asarray(peak_wavelengths, dtype=np.float64),
                    transition_wls,
                    element_names,
                    np.asarray(shift_grid, dtype=np.float64),
                    wavelength_tolerance_nm,
                    kdet_min_score,
                    kdet_min_candidates,
                    kdet_rarity_power,
                    kdet_weight_clip,
                )
            )
            filtered = {el: transitions_by_element[el] for el in passed_names}
            if filtered and len(filtered) < len(transitions_by_element):
                warnings.append("kdet_filtered_elements")
            return filtered, warnings
        except Exception as e:
            logger.warning(f"Rust kdet filter failed, falling back to Python: {e}")

    densities = []
    element_density: Dict[str, float] = {}
    wl_range = max(float(peak_wavelengths.max() - peak_wavelengths.min()), 1e-6)
    for element, transitions in transitions_by_element.items():
        density = len(transitions) / wl_range
        element_density[element] = density
        densities.append(density)
    median_density = float(np.median(densities)) if densities else 1.0

    filtered: Dict[str, List[Transition]] = {}
    for element, transitions in transitions_by_element.items():
        if not transitions:
            continue
        transitions_wl = np.array([t.wavelength_nm for t in transitions], dtype=float)
        transitions_wl.sort()
        best_candidates = 0
        for shift_nm in shift_grid:
            shifted_peaks = peak_wavelengths + shift_nm
            candidate_mask = _peaks_within_tolerance(
                shifted_peaks, transitions_wl, wavelength_tolerance_nm
            )
            candidate_count = int(np.sum(candidate_mask))
            if candidate_count > best_candidates:
                best_candidates = candidate_count
        kdet_fraction = best_candidates / total_peaks
        density = element_density.get(element, median_density)
        rarity_weight = (median_density / max(density, 1e-6)) ** kdet_rarity_power
        rarity_weight = float(np.clip(rarity_weight, kdet_weight_clip[0], kdet_weight_clip[1]))
        kdet_score = kdet_fraction * rarity_weight
        if best_candidates >= kdet_min_candidates and kdet_score >= kdet_min_score:
            filtered[element] = transitions

    if filtered and len(filtered) < len(transitions_by_element):
        warnings.append("kdet_filtered_elements")

    return filtered, warnings


def _peaks_within_tolerance(
    peaks: np.ndarray,
    transitions_sorted: np.ndarray,
    tolerance_nm: float,
) -> np.ndarray:
    if transitions_sorted.size == 0 or peaks.size == 0:
        return np.zeros_like(peaks, dtype=bool)
    idx = np.searchsorted(transitions_sorted, peaks)
    idx = np.clip(idx, 0, transitions_sorted.size - 1)
    nearest = transitions_sorted[idx]
    left_idx = np.clip(idx - 1, 0, transitions_sorted.size - 1)
    left_nearest = transitions_sorted[left_idx]
    min_dist = np.minimum(np.abs(nearest - peaks), np.abs(left_nearest - peaks))
    return min_dist <= tolerance_nm


def _match_transition(
    peak_wavelength: float,
    transitions: List[Transition],
    tolerance_nm: float,
) -> Optional[Transition]:
    best_match = None
    best_distance = float("inf")
    for transition in transitions:
        distance = abs(transition.wavelength_nm - peak_wavelength)
        if distance <= tolerance_nm and distance < best_distance:
            best_match = transition
            best_distance = distance
    return best_match


def _estimate_wl_step(wavelength: np.ndarray) -> float:
    if wavelength.size < 2:
        return 1.0
    diffs = np.diff(wavelength)
    diffs = diffs[np.isfinite(diffs)]
    return float(np.median(diffs)) if diffs.size else 1.0


def _find_peaks(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    min_peak_height: float,
    peak_width_nm: float,
) -> List[Tuple[int, float]]:
    max_intensity = float(np.max(intensity))
    if max_intensity <= 0:
        return []

    normalized = intensity / max_intensity
    threshold = max(min_peak_height, 0.0)

    if HAS_SCIPY and find_peaks is not None:
        wl_step = _estimate_wl_step(wavelength)
        min_distance_px = max(int(peak_width_nm / max(wl_step, 1e-9)), 1)
        peak_indices, _ = find_peaks(
            normalized,
            height=threshold,
            distance=min_distance_px,
            prominence=threshold / 2.0,
        )
        return [(int(idx), float(wavelength[idx])) for idx in peak_indices]

    # Simple fallback: local maxima above threshold
    peaks: List[Tuple[int, float]] = []
    for i in range(1, len(intensity) - 1):
        if (
            normalized[i] >= threshold
            and intensity[i] > intensity[i - 1]
            and intensity[i] > intensity[i + 1]
        ):
            peaks.append((i, float(wavelength[i])))
    return peaks
