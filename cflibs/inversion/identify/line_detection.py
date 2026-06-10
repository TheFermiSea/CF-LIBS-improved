"""
Line detection utilities for CF-LIBS inversion.

Provides a lightweight peak detection + line matching pipeline to convert
raw spectra into LineObservation objects for classic CF-LIBS solvers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from cflibs.inversion.deconvolution import VoigtFitResult

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import Transition
from cflibs.core.logging_config import get_logger
from cflibs.inversion.physics.boltzmann import LineObservation

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

# JAX is an optional fast path for the dense numerical kernels in this module
# (peak-tolerance matching, kdet shift-scan reductions, local-maxima
# fallback peak finding). The default code paths remain pure NumPy / SciPy /
# Rust so behavior is unchanged unless callers explicitly opt in via the
# ``use_jax_kdet`` / ``use_jax_peak_fallback`` kwargs of
# ``detect_line_observations``.
#
# Note: ``scipy.signal.find_peaks`` is deliberately NOT ported. Its
# ``distance`` + ``prominence`` semantics produce variable-length output and
# are non-local; reproducing them faithfully in JAX is brittle and the
# scipy C implementation already runs in ~0.2 ms on a 4096-sample spectrum
# (vs ~100 ms JAX JIT warmup). See
# ``docs/jax-port/line-detection-consultation.md`` for the full audit and
# reviewer recommendations (Codex + Gemini).
try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only when jax missing
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False


if _HAS_JAX:

    @jax.jit
    def _jax_peaks_within_tolerance(
        peaks: "jnp.ndarray",
        transitions_sorted: "jnp.ndarray",
        tolerance_nm: float,
    ) -> "jnp.ndarray":
        """JAX-vectorized analogue of :func:`_peaks_within_tolerance`.

        Returns a boolean mask the same length as ``peaks`` indicating
        whether each peak has *any* transition within ``tolerance_nm``.

        Algorithm (identical to the NumPy path): for each peak, find its
        ``searchsorted`` insertion index in ``transitions_sorted``, then
        compare against the two adjacent transitions (``idx`` and
        ``idx - 1``). The peak is "in tolerance" if the minimum of those
        two |distance| values is <= ``tolerance_nm``.

        Parameters
        ----------
        peaks : (N,) jnp.ndarray
            Peak wavelengths (already shifted if applicable). May contain
            arbitrary values — empty arrays must be handled by the caller.
        transitions_sorted : (M,) jnp.ndarray
            Transition wavelengths, **sorted ascending**. Must contain at
            least one element; callers handle the M=0 case in Python.
        tolerance_nm : float
            Match tolerance in nm.

        Returns
        -------
        (N,) jnp.ndarray of bool
            ``True`` where ``min(|peak - left|, |peak - right|) <= tolerance``.
        """
        m_max = transitions_sorted.shape[0] - 1
        idx = jnp.searchsorted(transitions_sorted, peaks)
        idx_clipped = jnp.clip(idx, 0, m_max)
        left_idx = jnp.clip(idx - 1, 0, m_max)
        nearest = transitions_sorted[idx_clipped]
        left_nearest = transitions_sorted[left_idx]
        min_dist = jnp.minimum(jnp.abs(nearest - peaks), jnp.abs(left_nearest - peaks))
        return min_dist <= tolerance_nm

    @jax.jit
    def _jax_kdet_candidate_counts(
        peak_wavelengths: "jnp.ndarray",
        transitions_sorted: "jnp.ndarray",
        shift_grid: "jnp.ndarray",
        tolerance_nm: float,
    ) -> "jnp.ndarray":
        """Count candidate peaks per shift for a single element.

        For each shift in ``shift_grid``, computes the number of peaks
        (after applying the shift) that fall within ``tolerance_nm`` of any
        transition. Replaces the inner Python loop in
        :func:`_kdet_filter_elements` for a single element.

        Parameters
        ----------
        peak_wavelengths : (N,) jnp.ndarray
            Observed peak wavelengths.
        transitions_sorted : (M,) jnp.ndarray
            Element's transition wavelengths, sorted ascending. The caller
            handles the M=0 case in Python.
        shift_grid : (S,) jnp.ndarray
            Wavelength shifts to scan.
        tolerance_nm : float
            Match tolerance.

        Returns
        -------
        (S,) jnp.ndarray of int32
            Candidate count for each shift.
        """

        def _per_shift(shift):
            shifted = peak_wavelengths + shift
            mask = _jax_peaks_within_tolerance(shifted, transitions_sorted, tolerance_nm)
            return jnp.sum(mask.astype(jnp.int32))

        return jax.vmap(_per_shift)(shift_grid)

    @jax.jit
    def _jax_local_maxima_mask(
        intensity: "jnp.ndarray",
        normalized: "jnp.ndarray",
        threshold: float,
    ) -> "jnp.ndarray":
        """Length-N boolean mask of strict local maxima above ``threshold``.

        Matches the simple-fallback path of :func:`_find_peaks`:
        ``intensity[i] > intensity[i-1]`` and ``intensity[i] >
        intensity[i+1]`` and ``normalized[i] >= threshold``, for
        ``i in [1, N-1)``. Endpoint samples (i=0 and i=N-1) are never
        considered peaks.

        Parameters
        ----------
        intensity : (N,) jnp.ndarray
            Raw spectral intensity.
        normalized : (N,) jnp.ndarray
            ``intensity / intensity.max()``.
        threshold : float
            Minimum relative height in [0, 1].

        Returns
        -------
        (N,) jnp.ndarray of bool
            Local-maximum mask.
        """
        n = intensity.shape[0]
        above = normalized >= threshold
        left_lower = jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.bool_), intensity[1:] > intensity[:-1]]
        )
        right_lower = jnp.concatenate(
            [intensity[:-1] > intensity[1:], jnp.zeros((1,), dtype=jnp.bool_)]
        )
        # Force the endpoints to False — the NumPy fallback only checks
        # i in [1, N-1).
        interior = jnp.arange(n)
        interior_mask = (interior > 0) & (interior < n - 1)
        return above & left_lower & right_lower & interior_mask


def _peaks_within_tolerance_jax(
    peaks: np.ndarray,
    transitions_sorted: np.ndarray,
    tolerance_nm: float,
) -> np.ndarray:
    """JAX-backed wrapper around :func:`_jax_peaks_within_tolerance`.

    Behaviorally identical to :func:`_peaks_within_tolerance` (same return
    dtype and shape, same edge cases for empty inputs). Falls back to the
    NumPy implementation if JAX is unavailable.

    See Also
    --------
    _peaks_within_tolerance : the NumPy reference implementation.
    """
    if not _HAS_JAX:
        return _peaks_within_tolerance(peaks, transitions_sorted, tolerance_nm)
    if transitions_sorted.size == 0 or peaks.size == 0:
        return np.zeros_like(peaks, dtype=bool)
    peaks_j = jnp.asarray(peaks, dtype=jnp.float64)
    trans_j = jnp.asarray(transitions_sorted, dtype=jnp.float64)
    mask = _jax_peaks_within_tolerance(peaks_j, trans_j, float(tolerance_nm))
    return np.asarray(mask, dtype=bool)


def _kdet_candidate_counts_jax(
    peak_wavelengths: np.ndarray,
    transitions_sorted: np.ndarray,
    shift_grid: np.ndarray,
    tolerance_nm: float,
) -> np.ndarray:
    """JAX-backed candidate-counts-per-shift for one element.

    Returns an array of length ``len(shift_grid)`` where each entry is the
    number of peaks (after the corresponding shift) that fall within
    ``tolerance_nm`` of any transition.

    Falls back to the NumPy ``_peaks_within_tolerance`` loop when JAX is
    unavailable. The empty-input edge cases (no transitions, no peaks, no
    shifts) return an all-zeros vector of the appropriate length, matching
    the semantics of the NumPy path's inner loop.
    """
    n_shifts = shift_grid.size
    if not _HAS_JAX:
        out = np.zeros(n_shifts, dtype=np.int64)
        for i, shift in enumerate(shift_grid):
            shifted = peak_wavelengths + float(shift)
            mask = _peaks_within_tolerance(shifted, transitions_sorted, tolerance_nm)
            out[i] = int(np.sum(mask))
        return out
    if transitions_sorted.size == 0 or peak_wavelengths.size == 0 or n_shifts == 0:
        return np.zeros(n_shifts, dtype=np.int64)
    peaks_j = jnp.asarray(peak_wavelengths, dtype=jnp.float64)
    trans_j = jnp.asarray(transitions_sorted, dtype=jnp.float64)
    shifts_j = jnp.asarray(shift_grid, dtype=jnp.float64)
    counts = _jax_kdet_candidate_counts(peaks_j, trans_j, shifts_j, float(tolerance_nm))
    return np.asarray(counts, dtype=np.int64)


def _find_peaks_jax_fallback(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    min_peak_height: float,
) -> List[Tuple[int, float]]:
    """JAX-backed analogue of the simple local-maxima fallback in
    :func:`_find_peaks`.

    This is **only** used when SciPy is unavailable. The default
    ``scipy.signal.find_peaks`` path is intentionally preserved — see
    ``docs/jax-port/line-detection-consultation.md`` for the rationale
    (variable-length output, non-local prominence semantics, sub-millisecond
    C implementation).

    Returns a Python list of ``(index, wavelength)`` tuples to match the
    return contract of :func:`_find_peaks`.
    """
    if intensity.size == 0:
        return []
    max_intensity = float(np.max(intensity))
    if max_intensity <= 0:
        return []
    normalized = intensity / max_intensity
    threshold = max(min_peak_height, 0.0)
    if not _HAS_JAX:
        peaks: List[Tuple[int, float]] = []
        for i in range(1, len(intensity) - 1):
            if (
                normalized[i] >= threshold
                and intensity[i] > intensity[i - 1]
                and intensity[i] > intensity[i + 1]
            ):
                peaks.append((i, float(wavelength[i])))
        return peaks
    intensity_j = jnp.asarray(intensity, dtype=jnp.float64)
    normalized_j = jnp.asarray(normalized, dtype=jnp.float64)
    mask = _jax_local_maxima_mask(intensity_j, normalized_j, float(threshold))
    mask_np = np.asarray(mask, dtype=bool)
    peak_indices = np.flatnonzero(mask_np)
    return [(int(idx), float(wavelength[idx])) for idx in peak_indices]


# Area of a unit-height Gaussian expressed per unit FWHM:
# A = h * sigma * sqrt(2*pi) with FWHM = 2*sigma*sqrt(2*ln 2)
#   => A = h * FWHM * sqrt(pi / (4*ln 2)) ~= 1.0645 * h * FWHM.
_GAUSSIAN_AREA_PER_HEIGHT_FWHM = math.sqrt(math.pi / (4.0 * math.log(2.0)))


def _estimate_fwhm_nm(
    segment_wl: np.ndarray,
    segment_intensity: np.ndarray,
    peak_rel_idx: int,
    wl_step: float,
) -> float:
    """Estimate a line FWHM (nm) from half-maximum crossings within a segment.

    Walks outward from the peak sample and linearly interpolates the first
    crossing of half the peak height on each side. When a side never crosses
    half-maximum inside the window (line truncated by the integration window)
    the window edge is used, and a degenerate (non-positive) width falls back
    to the Nyquist-limited ``2 * wl_step``.

    Parameters
    ----------
    segment_wl : np.ndarray
        Wavelength sub-array of the integration window (nm).
    segment_intensity : np.ndarray
        Intensity sub-array of the integration window.
    peak_rel_idx : int
        Peak index relative to the segment start.
    wl_step : float
        Local wavelength step (nm), used as the resolution floor.

    Returns
    -------
    float
        Estimated FWHM in nm (always positive, >= ``wl_step``).
    """
    step_floor = max(wl_step, 1e-6)
    n = len(segment_intensity)
    if n == 0:
        return 2.0 * step_floor
    # Anchor on the tallest sample of the window (the local-maximum peak).
    apex = int(np.argmax(segment_intensity))
    if 0 <= peak_rel_idx < n and segment_intensity[peak_rel_idx] >= segment_intensity[apex]:
        apex = peak_rel_idx
    height = float(segment_intensity[apex])
    if not np.isfinite(height) or height <= 0.0:
        return 2.0 * step_floor
    half = 0.5 * height

    def _crossing(start: int, stop: int, step: int, edge: float) -> float:
        for i in range(start, stop, step):
            y1 = float(segment_intensity[i])
            y0 = float(segment_intensity[i + step])
            if y0 <= half < y1:
                x1 = float(segment_wl[i])
                x0 = float(segment_wl[i + step])
                frac = (half - y0) / (y1 - y0) if y1 != y0 else 0.0
                return x0 + frac * (x1 - x0)
        return edge

    left_wl = _crossing(apex, 0, -1, float(segment_wl[0]))
    right_wl = _crossing(apex, n - 1, 1, float(segment_wl[-1]))
    fwhm = right_wl - left_wl
    if not np.isfinite(fwhm) or fwhm <= 0.0:
        fwhm = 2.0 * step_floor
    return max(float(fwhm), step_floor)


def _poisson_area_floor(line_area: float, wl_step: float, scale: float) -> float:
    """Shot-noise-equivalent 1-sigma floor for an integrated line area.

    The trapezoid (Poisson) line-building path floors its uncertainty at the
    shot-noise level ``sqrt(sum counts) * wl_step``. For an integrated area
    ``A ~ (sum counts) * wl_step`` this is equivalent to ``sqrt(A * wl_step)``
    (i.e. ``sum counts ~ A / wl_step`` so ``sqrt(A / wl_step) * wl_step
    = sqrt(A * wl_step)``). Applying the same floor to the Voigt-fit path makes
    the two physically-distinct line builders produce commensurate
    inverse-variance weights, instead of letting an over-confident Voigt fit
    report a sigma arbitrarily far below the shot-noise limit.

    Parameters
    ----------
    line_area : float
        Fitted (integrated) line area in the same intensity*wavelength units
        used by the trapezoid path.
    wl_step : float
        Local wavelength step (nm) of the spectrum grid.
    scale : float
        Multiplicative scale on the floor. ``1.0`` reproduces the Poisson floor
        exactly; ``0.0`` disables the floor (legacy behaviour). Values in
        between soften it for deconvolution-heavy spectra.

    Returns
    -------
    float
        Shot-noise-equivalent 1-sigma floor (>= 0).
    """
    if scale <= 0.0 or wl_step <= 0.0:
        return 0.0
    return float(scale * math.sqrt(max(line_area, 0.0) * wl_step))


def _build_observation_from_fit(
    transition: Transition,
    fit_result: VoigtFitResult,
    ground_state_threshold_ev: float,
    wl_step: float = 0.0,
    poisson_floor_scale: float = 1.0,
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
    wl_step : float
        Local wavelength step (nm) used to compute the shot-noise-equivalent
        floor on the fitted-area uncertainty. ``0.0`` disables the floor (the
        function then reproduces legacy behaviour with only the ``1e-6`` guard).
    poisson_floor_scale : float
        Scale on the Poisson-equivalent floor (see :func:`_poisson_area_floor`).
        Default ``1.0`` matches the trapezoid path's shot-noise floor; ``0.0``
        disables it. The floor only ever *raises* the reported uncertainty, so
        optically-thin synthetic spectra with honest covariance-based
        uncertainties are unaffected at the limit.

    Returns
    -------
    Optional[Tuple[LineObservation, bool]]
        (observation, is_resonance) tuple, or None if area is invalid.
    """
    line_area = fit_result.area
    if line_area <= 0:
        return None

    poisson_floor = _poisson_area_floor(line_area, wl_step, poisson_floor_scale)
    intensity_uncertainty = max(fit_result.area_uncertainty, poisson_floor, 1e-6)

    obs = LineObservation(
        wavelength_nm=float(transition.wavelength_nm),
        intensity=line_area,
        intensity_uncertainty=intensity_uncertainty,
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

    # Boltzmann/Saha-Boltzmann ordinates require the wavelength-INTEGRATED
    # line intensity (counts*nm) for every line on a common scale. The
    # historical ``max(line_area, peak_height)`` floor silently swapped the
    # integrated area for a bare peak height (counts) whenever the line was
    # narrower than ~1 nm equivalent, mixing incompatible quantities across
    # lines and distorting ln(I*lambda/gA) by |ln(FWHM_eff/1 nm)| ~ 1.5-3
    # ln-units (audit 2026-06-09 02-F6). The floor is removed: when the
    # trapezoid integral is unusable (non-finite or <= 0, e.g. an
    # over-subtracted baseline), fall back to a UNITS-CONSISTENT
    # Gaussian-equivalent area h * FWHM * sqrt(pi/(4 ln 2)) — never the bare
    # height.
    if not np.isfinite(line_area) or line_area <= 0.0:
        peak_height = float(np.max(segment_intensity))
        if not np.isfinite(peak_height) or peak_height <= 0.0:
            return None
        fwhm_nm = _estimate_fwhm_nm(
            segment_wl, segment_intensity, peak_idx - start_idx, wl_step
        )
        line_area = peak_height * fwhm_nm * _GAUSSIAN_AREA_PER_HEIGHT_FWHM

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


def _empty_line_detection_result(warning: str) -> LineDetectionResult:
    """Empty :class:`LineDetectionResult` carrying a single warning code."""
    return LineDetectionResult(
        observations=[],
        resonance_lines=set(),
        total_peaks=0,
        matched_peaks=0,
        unmatched_peaks=0,
        applied_shift_nm=0.0,
        warnings=[warning],
    )


def _resolve_adaptive_tolerances(
    wavelength_tolerance_nm: Optional[float],
    peak_width_nm: Optional[float],
    wl_step: float,
    lambda_mid: float,
    resolving_power: Optional[float],
) -> Tuple[float, float]:
    """Resolve adaptive ``wavelength_tolerance_nm`` / ``peak_width_nm`` defaults.

    When the caller leaves either unset (``None``), derive R-aware defaults: one
    FWHM at the band midpoint when ``resolving_power`` is provided, else the
    legacy fixed constants (0.1 and 0.2 nm) so existing callers see
    byte-identical behaviour (CF-LIBS-improved-s1qr.2).
    """
    if wavelength_tolerance_nm is None:
        if resolving_power is not None and resolving_power > 0:
            wavelength_tolerance_nm = max(2.0 * wl_step, lambda_mid / resolving_power)
        else:
            wavelength_tolerance_nm = 0.1
    if peak_width_nm is None:
        if resolving_power is not None and resolving_power > 0:
            peak_width_nm = max(2.0 * wl_step, lambda_mid / resolving_power)
        else:
            peak_width_nm = 0.2
    return wavelength_tolerance_nm, peak_width_nm


def _select_accepted_elements(
    best_shift_summary: Optional[CombShiftSummary],
    fallback_shift_summary: Optional[CombShiftSummary],
    comb_fallback_to_nearest: bool,
    comb_min_matches: int,
    comb_fallback_max_elements: int,
    warnings: List[str],
) -> Tuple[float, Dict[str, CombScore], List[str]]:
    """Pick the applied shift, element scores, and accepted elements.

    Mirrors the comb-pass / comb-fallback selection exactly; appends the same
    ``comb_no_elements_passed`` / ``comb_fallback_disabled`` warnings as the
    inline code did.
    """
    if best_shift_summary is not None and best_shift_summary["passed_elements"]:
        applied_shift_nm = float(best_shift_summary["shift_nm"])
        element_scores = best_shift_summary["scores"]
        accepted_elements = list(best_shift_summary["passed_elements"])
        return applied_shift_nm, element_scores, accepted_elements

    warnings.append("comb_no_elements_passed")
    if not (comb_fallback_to_nearest and fallback_shift_summary is not None):
        warnings.append("comb_fallback_disabled")
        return 0.0, {}, []

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
    return applied_shift_nm, element_scores, accepted_elements


def _run_deconvolution(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    peaks: List[Tuple[int, float]],
    peak_width_nm: float,
) -> Optional[Dict[float, "VoigtFitResult"]]:
    """Voigt-deconvolve detected peaks, mapping fits to nearest peak wavelength.

    Returns ``None`` when deconvolution is not applicable or fails (the caller
    then falls back to the trapezoid line builder), matching the inline
    behaviour exactly.
    """
    try:
        from cflibs.inversion.deconvolution import deconvolve_peaks

        # Collect all peak wavelengths for deconvolution
        peak_wl_arr = np.array([pw for _, pw in peaks], dtype=float)
        if len(peak_wl_arr) == 0:
            return None
        from cflibs.inversion.preprocess.preprocessing import estimate_baseline

        baseline = estimate_baseline(wavelength, intensity)
        baseline_subtracted = intensity - baseline
        deconv = deconvolve_peaks(
            wavelength,
            baseline_subtracted,
            peak_wl_arr,
            fwhm_estimate=peak_width_nm,
        )
        # Map fit results to nearest original peak wavelength
        deconv_results_by_wl: Dict[float, "VoigtFitResult"] = {}
        for fr in deconv.fit_results:
            best_peak_wl = float(peak_wl_arr[np.argmin(np.abs(peak_wl_arr - fr.center_nm))])
            deconv_results_by_wl[round(best_peak_wl, 4)] = fr
        logger.debug(
            "Deconvolution produced %d fit results for %d peaks",
            len(deconv.fit_results),
            len(peak_wl_arr),
        )
        return deconv_results_by_wl
    except Exception as exc:
        logger.warning("Deconvolution failed, falling back to trapezoid: %s", exc)
        return None


@dataclass
class _ObservationBuildContext:
    """Constant inputs shared by the per-match observation builders."""

    wavelength: np.ndarray
    intensity: np.ndarray
    half_width_px: int
    wl_step: float
    ground_state_threshold_ev: float
    poisson_floor_scale: float
    deconv_results_by_wl: Optional[Dict[float, "VoigtFitResult"]]


def _build_observation_for_match(
    ctx: _ObservationBuildContext,
    transition: Transition,
    peak_idx: int,
    peak_wl: float,
) -> Optional[Tuple[LineObservation, bool]]:
    """Build one observation, preferring the deconvolution fit when available.

    Identical dispatch to the inline ``if deconv_results_by_wl ... else ...``
    blocks: a Voigt-fit observation when this peak has a deconvolution result,
    otherwise the trapezoid builder.
    """
    rounded_wl = round(peak_wl, 4)
    if ctx.deconv_results_by_wl is not None and rounded_wl in ctx.deconv_results_by_wl:
        return _build_observation_from_fit(
            transition,
            ctx.deconv_results_by_wl[rounded_wl],
            ctx.ground_state_threshold_ev,
            wl_step=ctx.wl_step,
            poisson_floor_scale=ctx.poisson_floor_scale,
        )
    return _build_observation(
        transition,
        peak_idx,
        ctx.wavelength,
        ctx.intensity,
        ctx.half_width_px,
        ctx.wl_step,
        ctx.ground_state_threshold_ev,
    )


def _register_observation(
    ctx: _ObservationBuildContext,
    transition: Transition,
    peak_idx: int,
    peak_wl: float,
    seen_keys: Set[Tuple[str, int, float]],
    matched_peak_indices: Set[int],
    observations: List[LineObservation],
    resonance_lines: Set[Tuple[str, int, float]],
) -> None:
    """Dedup, build, and append one observation for a (transition, peak) match.

    Captures the shared body of the two inner loops in
    :func:`detect_line_observations` verbatim: key dedup against ``seen_keys``,
    recording the matched peak, building the observation, and appending it plus
    any resonance key.
    """
    key = (transition.element, transition.ionization_stage, transition.wavelength_nm)
    if key in seen_keys:
        return
    seen_keys.add(key)
    matched_peak_indices.add(peak_idx)

    result = _build_observation_for_match(ctx, transition, peak_idx, peak_wl)
    if result is None:
        return
    obs, is_resonance = result
    observations.append(obs)
    if is_resonance:
        resonance_lines.add(key)


def _collect_observations(
    ctx: _ObservationBuildContext,
    accepted_elements: List[str],
    element_scores: Dict[str, CombScore],
    comb_transitions_by_element: Dict[str, List[Transition]],
    peaks: List[Tuple[int, float]],
    wavelength_tolerance_nm: float,
    applied_shift_nm: float,
    seen_keys: Set[Tuple[str, int, float]],
    matched_peak_indices: Set[int],
    observations: List[LineObservation],
    resonance_lines: Set[Tuple[str, int, float]],
) -> None:
    """Build observations for every accepted element (in-place accumulation).

    Reproduces the ``if accepted_elements:`` block exactly: sort accepted
    elements by (f1, matched_lines) descending, then for each element run the
    comb-transition matcher followed by the per-peak nearest matcher, building
    one observation per unique line key.
    """

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
            _register_observation(
                ctx,
                transition,
                peak_idx,
                peak_wl,
                seen_keys,
                matched_peak_indices,
                observations,
                resonance_lines,
            )
        for peak_idx, peak_wl in peaks:
            transition = _match_transition(
                peak_wl + applied_shift_nm, transitions, wavelength_tolerance_nm
            )
            if transition is None:
                continue
            _register_observation(
                ctx,
                transition,
                peak_idx,
                peak_wl,
                seen_keys,
                matched_peak_indices,
                observations,
                resonance_lines,
            )


def _apply_kdet_filter(
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
    use_jax_kdet: bool,
    shift_coherence_veto: bool,
    coherence_min_lines: int,
    coherence_min_fraction: float,
    warnings: List[str],
) -> Dict[str, List[Transition]]:
    """Run the kdet pre-filter, extending ``warnings`` and returning the kept map.

    Mirrors the inline ``if kdet_enabled:`` block exactly: extend ``warnings``
    with the kdet warnings, keep the filtered element map when non-empty, else
    append ``kdet_filtered_all_elements`` and keep the original map.
    """
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
        use_jax=use_jax_kdet,
        shift_coherence=shift_coherence_veto,
        coherence_min_lines=coherence_min_lines,
        coherence_min_fraction=coherence_min_fraction,
    )
    warnings.extend(kdet_warnings)
    if filtered_elements:
        return filtered_elements
    warnings.append("kdet_filtered_all_elements")
    return transitions_by_element


def _apply_shift_coherence_veto(
    accepted_elements: List[str],
    peaks: List[Tuple[int, float]],
    comb_transitions_by_element: Dict[str, List[Transition]],
    applied_shift_nm: float,
    wavelength_tolerance_nm: float,
    coherence_min_lines: int,
    coherence_min_fraction: float,
    warnings: List[str],
) -> List[str]:
    """Apply the shift-coherence veto, returning the surviving accepted elements.

    Mirrors the inline veto block exactly: when any element is vetoed, swap in
    the kept set and append ``shift_coherence_vetoed_elements``; otherwise leave
    ``accepted_elements`` unchanged.
    """
    kept, vetoed = _shift_coherence_veto(
        elements=accepted_elements,
        peaks=peaks,
        transitions_by_element=comb_transitions_by_element,
        applied_shift_nm=applied_shift_nm,
        tolerance_nm=wavelength_tolerance_nm,
        min_coherent_lines=coherence_min_lines,
        min_coherent_fraction=coherence_min_fraction,
    )
    if vetoed:
        warnings.append("shift_coherence_vetoed_elements")
        return kept
    return accepted_elements


def _assemble_line_detection_result(
    observations: List[LineObservation],
    resonance_lines: Set[Tuple[str, int, float]],
    total_peaks: int,
    matched_peak_indices: Set[int],
    applied_shift_nm: float,
    warnings: List[str],
) -> LineDetectionResult:
    """Compute matched/unmatched counts and build the final result.

    Mirrors the inline tail of :func:`detect_line_observations`: append
    ``no_peaks_matched`` when nothing matched on a non-empty peak set, then
    package the :class:`LineDetectionResult`.
    """
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


def detect_line_observations(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    atomic_db: AtomicDatabase,
    elements: List[str],
    wavelength_tolerance_nm: Optional[float] = None,
    min_peak_height: float = 0.01,
    peak_width_nm: Optional[float] = None,
    min_relative_intensity: Optional[float] = None,
    top_k_per_element: Optional[int] = None,
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
    shift_coherence_veto: bool = True,
    coherence_min_lines: int = 2,
    coherence_min_fraction: float = 0.5,
    use_deconvolution: bool = False,
    poisson_floor_scale: float = 1.0,
    use_jax_kdet: bool = False,
    use_jax_peak_fallback: bool = False,
    resolving_power: Optional[float] = None,
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
    wavelength_tolerance_nm : float, optional
        Matching tolerance for known lines in nm. When ``None`` (default), an
        adaptive value is used: ``max(2 * wl_step, lambda_mid / resolving_power)``
        if ``resolving_power`` is provided, else the legacy ``0.1`` constant.
        This avoids R-independent false matches in dense atomic catalogs
        (see CF-LIBS-improved-s1qr.2).
    min_peak_height : float
        Minimum peak height as fraction of max intensity
    peak_width_nm : float, optional
        Expected peak width for integration (nm). When ``None`` (default), an
        adaptive value is used: ``max(2 * wl_step, lambda_mid / resolving_power)``
        if ``resolving_power`` is provided, else the legacy ``0.2`` constant.
        The default 0.2 nm imposes an R-independent resolution floor at
        high-R; the adaptive form tracks the actual instrumental FWHM
        (see CF-LIBS-improved-s1qr.2).
    min_relative_intensity : float, optional
        Minimum relative intensity threshold for database lines. Prefer
        ``top_k_per_element`` (element-relative) over this absolute floor:
        the latter silently deletes whole elements whose tabulated rel_int is
        small or NULL (see :func:`_load_transitions`).
    top_k_per_element : int, optional
        Keep only each element's ``K`` strongest in-band lines (by the
        gA-Boltzmann :func:`_transition_strength`) instead of an absolute
        rel_int floor. Bounds catalog richness without deleting elements.
        ``None`` keeps all in-band lines. Default 60.
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
    shift_coherence_veto : bool
        If True (default), reject accepted elements whose matched lines do not
        agree on the single instrument residual shift (see
        :func:`_shift_coherence_veto`). This suppresses dense-catalog
        false-positive confounders that accrue matches by random wavelength
        coincidence, without re-tuning the comb gates.
    coherence_min_lines : int
        Minimum number of coherent (consensus-aligned) matched lines an element
        must have to survive the shift-coherence veto.
    coherence_min_fraction : float
        Minimum fraction of an element's matched lines that must be coherent
        with the consensus residual shift to survive the veto.
    use_deconvolution : bool
        If True, apply Voigt deconvolution to resolve overlapping peaks
        before building line observations (default: False).
    poisson_floor_scale : float
        Scale on the shot-noise-equivalent floor applied to the
        *Voigt-fit* line-building path's area uncertainty
        (:func:`_build_observation_from_fit`). The trapezoid path is already
        shot-noise floored at ``sqrt(sum counts) * wl_step``; a confident Voigt
        fit could otherwise report a covariance-based sigma far below that
        floor, producing an over-weighted Boltzmann-plot point. ``1.0``
        (default) reproduces the trapezoid path's Poisson floor so the two
        builders yield commensurate inverse-variance weights; ``0.0`` disables
        the floor (legacy behaviour). Has no effect when ``use_deconvolution``
        is False (the Voigt path is never taken).
    use_jax_kdet : bool
        If True, run the kdet shift-scan inner loop through the JAX
        backend (:func:`_kdet_candidate_counts_jax`). Default is False so
        existing benchmark results stay reproducible. Has no effect when
        ``kdet_enabled=False`` or when the Rust core is available (Rust
        path is always preferred).
    use_jax_peak_fallback : bool
        If True, run the local-maxima fallback peak-finder
        (:func:`_find_peaks_jax_fallback`) instead of the Python fallback.
        Only relevant when SciPy is unavailable — the default
        ``scipy.signal.find_peaks`` path is intentionally preserved
        regardless of this flag (see ``docs/jax-port/line-detection-
        consultation.md``).
    resolving_power : float, optional
        Instrument resolving power (lambda / FWHM). When provided and the
        caller leaves ``wavelength_tolerance_nm`` / ``peak_width_nm``
        unset (``None``), those defaults become 1 FWHM at the band
        midpoint instead of the legacy fixed values. Explicit overrides
        from the caller take precedence (see CF-LIBS-improved-s1qr.2).

    Returns
    -------
    LineDetectionResult
        Detected line observations and resonance set
    """
    if wavelength.size == 0 or intensity.size == 0:
        return _empty_line_detection_result("empty_spectrum")

    if wavelength.size != intensity.size:
        raise ValueError("Wavelength and intensity arrays must be the same length")

    if not elements:
        return _empty_line_detection_result("no_elements_specified")

    wl_min = float(np.min(wavelength))
    wl_max = float(np.max(wavelength))

    # --- Adaptive defaults (CF-LIBS-improved-s1qr.2) -------------------
    # When the caller leaves ``wavelength_tolerance_nm`` or ``peak_width_nm``
    # unset (``None``), derive R-aware defaults: one FWHM at the band
    # midpoint when ``resolving_power`` is provided, else fall back to the
    # legacy fixed constants (0.1 and 0.2 nm respectively) so existing
    # callers see byte-identical behaviour. The 1-FWHM scale (``lambda/R``)
    # matches the instrument resolution element: two database transitions
    # closer than this cannot be separated by the spectrometer, so
    # admitting matches inside a tighter window would amplify
    # ambiguous-catalog noise (CF-LIBS-improved-s1qr.2).
    wl_step = _estimate_wl_step(wavelength)
    lambda_mid = 0.5 * (wl_min + wl_max)
    wavelength_tolerance_nm, peak_width_nm = _resolve_adaptive_tolerances(
        wavelength_tolerance_nm,
        peak_width_nm,
        wl_step,
        lambda_mid,
        resolving_power,
    )

    transitions = _load_transitions(
        atomic_db,
        elements,
        wavelength_min=wl_min,
        wavelength_max=wl_max,
        min_relative_intensity=min_relative_intensity,
        top_k_per_element=top_k_per_element,
    )

    if not transitions:
        return _empty_line_detection_result("no_transitions_found")

    peaks = _find_peaks(
        wavelength,
        intensity,
        min_peak_height,
        peak_width_nm,
        use_jax_fallback=use_jax_peak_fallback,
    )

    half_width_px = max(int((peak_width_nm / max(wl_step, 1e-9)) / 2), 1)

    observations: List[LineObservation] = []
    resonance_lines: Set[Tuple[str, int, float]] = set()
    seen_keys: Set[Tuple[str, int, float]] = set()

    warnings: List[str] = []
    total_peaks = len(peaks)
    if total_peaks == 0:
        return _empty_line_detection_result("no_peaks_detected")

    transitions_by_element: Dict[str, List[Transition]] = {}
    for transition in transitions:
        transitions_by_element.setdefault(transition.element, []).append(transition)

    if kdet_enabled:
        transitions_by_element = _apply_kdet_filter(
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
            use_jax_kdet=use_jax_kdet,
            shift_coherence_veto=shift_coherence_veto,
            coherence_min_lines=coherence_min_lines,
            coherence_min_fraction=coherence_min_fraction,
            warnings=warnings,
        )

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

    applied_shift_nm, element_scores, accepted_elements = _select_accepted_elements(
        best_shift_summary,
        fallback_shift_summary,
        comb_fallback_to_nearest,
        comb_min_matches,
        comb_fallback_max_elements,
        warnings,
    )

    # Shift-coherence veto: drop accepted elements whose matched comb lines do
    # not agree on the single instrument residual shift (random-coincidence
    # confounders), applied uniformly to both the comb-pass and comb-fallback
    # selections. Runs after the (Rust or Python) comb scan so it gates both
    # backends identically.
    if shift_coherence_veto and accepted_elements:
        accepted_elements = _apply_shift_coherence_veto(
            accepted_elements=accepted_elements,
            peaks=peaks,
            comb_transitions_by_element=comb_transitions_by_element,
            applied_shift_nm=applied_shift_nm,
            wavelength_tolerance_nm=wavelength_tolerance_nm,
            coherence_min_lines=coherence_min_lines,
            coherence_min_fraction=coherence_min_fraction,
            warnings=warnings,
        )

    matched_peak_indices: Set[int] = set()

    # Optionally run deconvolution on matched peaks
    deconv_results_by_wl: Optional[Dict[float, VoigtFitResult]] = None
    if use_deconvolution:
        deconv_results_by_wl = _run_deconvolution(wavelength, intensity, peaks, peak_width_nm)

    if accepted_elements:
        build_ctx = _ObservationBuildContext(
            wavelength=wavelength,
            intensity=intensity,
            half_width_px=half_width_px,
            wl_step=wl_step,
            ground_state_threshold_ev=ground_state_threshold_ev,
            poisson_floor_scale=poisson_floor_scale,
            deconv_results_by_wl=deconv_results_by_wl,
        )
        _collect_observations(
            build_ctx,
            accepted_elements,
            element_scores,
            comb_transitions_by_element,
            peaks,
            wavelength_tolerance_nm,
            applied_shift_nm,
            seen_keys,
            matched_peak_indices,
            observations,
            resonance_lines,
        )

    return _assemble_line_detection_result(
        observations=observations,
        resonance_lines=resonance_lines,
        total_peaks=total_peaks,
        matched_peak_indices=matched_peak_indices,
        applied_shift_nm=applied_shift_nm,
        warnings=warnings,
    )


def _load_transitions(
    atomic_db: AtomicDatabase,
    elements: List[str],
    wavelength_min: float,
    wavelength_max: float,
    min_relative_intensity: Optional[float],
    top_k_per_element: Optional[int] = None,
) -> List[Transition]:
    """Load in-band transitions per element, optionally capped to the K strongest.

    The legacy ``min_relative_intensity`` SQL floor is an *absolute* cut on
    NIST ``rel_int``: because that column is incomparable across elements and
    NULL for ~4500 DB lines, a fixed floor silently deletes whole real elements
    whose lines simply happen to have small (or absent) tabulated rel_int —
    e.g. on BHVO-2 every Mg/K line and all 4 Al I resonance lines (rel_int
    24-26) fall below a floor of 100, so those elements never reach the solver.

    ``top_k_per_element`` replaces that with an element-*relative* bound: keep
    each element's ``K`` strongest in-band lines by the gA-Boltzmann
    :func:`_transition_strength` (a single consistent scale), with no absolute
    threshold. Every requested element therefore contributes its best lines,
    and the per-element cap keeps dense catalogs (e.g. Fe, Ti) from swamping
    the comb/kdet stages. ``None`` keeps all in-band lines.
    """
    transitions: List[Transition] = []
    for element in elements:
        element_transitions = atomic_db.get_transitions(
            element,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            min_relative_intensity=min_relative_intensity,
        )
        if top_k_per_element is not None and len(element_transitions) > top_k_per_element:
            element_transitions = _rank_transitions_by_strength(element_transitions)[
                :top_k_per_element
            ]
        transitions.extend(element_transitions)
    return transitions


# Reference excitation temperature (eV) for the gA-Boltzmann line-strength
# proxy used to rank database transitions for the comb / top-K selection.
# ~1 eV (~11600 K) is a typical LIBS plasma excitation temperature; the
# ranking is only weakly sensitive to its exact value over the 0.7-1.2 eV
# range (verified on BHVO-2: the same low-E_k resonance lines top the comb).
_COMB_STRENGTH_T_REF_EV = 1.0


def _ga_boltzmann_weight(transition: Transition) -> float:
    """gA-Boltzmann emission weight ``g_k * A_ki * exp(-E_k / kT_ref)``.

    The population-weighted emissivity proxy the forward model itself uses
    (``epsilon proportional to A_ki * n_k``, ``n_k proportional to
    g_k * exp(-E_k/kT)``). A single internally-consistent scale across every
    transition, including the ~4500 DB lines that have no NIST ``rel_int``.
    """
    a_ki = transition.A_ki
    if a_ki is None or a_ki <= 0:
        return 0.0
    g_k = transition.g_k if transition.g_k else 1
    e_k = transition.E_k_ev if transition.E_k_ev is not None else 0.0
    return float(g_k) * float(a_ki) * math.exp(-max(e_k, 0.0) / _COMB_STRENGTH_T_REF_EV)


def _transition_strength(transition: Transition) -> float:
    """Single-transition line-strength proxy (gA-Boltzmann weight).

    Kept for callers that need a per-line scalar. The comb / top-K *ranking*
    uses :func:`_rank_transitions_by_strength`, which combines this with the
    tabulated ``relative_intensity`` on a shared normalized scale (see there).
    """
    return _ga_boltzmann_weight(transition)


def _rank_transitions_by_strength(transitions: List[Transition]) -> List[Transition]:
    """Rank an element's transitions strongest-first for comb / top-K selection.

    The previous ranking returned NIST ``relative_intensity`` when present ELSE
    ``A_ki`` from a single ``key`` — mixing two scales 6-9 orders of magnitude
    apart, so the ~4500 NULL-rel_int lines (ranked by raw ``A_ki ~ 1e8``)
    spuriously outranked real lines (``rel_int ~ tens``) and crowded genuine
    transitions out of the comb. Worse, NIST ``rel_int`` is *not comparable
    across ion stages*: an element's hot ion lines carry rel_int in the
    thousands while its bright neutral resonance lines (the persistent lines
    that dominate a LIBS plasma) carry rel_int ~ tens, so any rel_int-led
    ranking buries the resonance lines (e.g. the Al I 394.4/396.2 nm doublet
    drops out of the top-30 entirely).

    Fix: rank by the gA-Boltzmann emission weight
    (:func:`_ga_boltzmann_weight`) — ``g_k * A_ki * exp(-E_k / kT_ref)`` — the
    same stage-consistent, population-weighted emissivity proxy the forward
    model uses. This is a single internally-consistent scale across every
    transition (no unit-mixing), ranks low-E_k resonance lines above high-E_k
    ones (restoring the Al doublet to the comb) and suppresses Boltzmann-faint
    high-E_k Rydberg lines (e.g. the Na 413-421 nm series) that are a known
    false-match source. Ties broken by shorter wavelength (deterministic).
    """
    if not transitions:
        return []
    return sorted(
        transitions,
        key=lambda t: (_ga_boltzmann_weight(t), -t.wavelength_nm),
        reverse=True,
    )


def _select_comb_transitions(
    transitions: List[Transition],
    max_lines: int,
) -> List[Transition]:
    ranked = _rank_transitions_by_strength(transitions)
    if max_lines > 0 and len(ranked) > max_lines:
        return ranked[:max_lines]
    return ranked


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
        sorted([t.wavelength_nm for t in transitions_by_element[el]]) for el in element_names
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
        total_f1 = (
            result.get(f1_key, 0.0)
            if f1_key
            else sum(s.f1_score for s in scores.values() if s.passes)
        )
        return {
            "shift_nm": float(shift),
            "scores": scores,
            "total_matches": total_matches,
            "passed_elements": passed,
            "total_f1": total_f1,
        }

    best = _build_summary(
        "best_shift",
        "best_scores",
        "best_total_matches",
        "best_passed_elements",
        "best_total_f1",
    )
    fallback = _build_summary(
        "fallback_shift",
        "fallback_scores",
        "fallback_total_matches",
        "fallback_passed_elements",
    )
    return best, fallback


@dataclass
class _ShiftScore:
    """Per-shift comb scoring aggregate for the pure-Python scan path."""

    scores: Dict[str, CombScore]
    total_f1: float
    total_matches_pass: int
    total_matches_all: int
    passed_elements: List[str]


def _score_comb_shift(
    shift_nm: float,
    peaks: List[Tuple[int, float]],
    transitions_by_element: Dict[str, List[Transition]],
    total_peaks: int,
    wavelength_tolerance_nm: float,
    comb_min_matches: int,
    comb_min_precision: float,
    comb_min_recall: float,
    comb_max_missing_fraction: float,
) -> _ShiftScore:
    """Score every element at a single shift (the inner element loop)."""
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

    return _ShiftScore(
        scores=scores,
        total_f1=total_f1,
        total_matches_pass=total_matches_pass,
        total_matches_all=total_matches_all,
        passed_elements=passed_elements,
    )


def _fallback_summary_improves(
    shift_nm: float,
    total_matches_all: int,
    current: CombShiftSummary,
) -> bool:
    """Whether the current shift beats the existing fallback summary.

    Higher total match count wins; ties broken by smaller absolute shift.
    """
    prev_matches = current["total_matches"]
    return total_matches_all > prev_matches or (
        total_matches_all == prev_matches and abs(shift_nm) < abs(float(current["shift_nm"]))
    )


def _best_summary_improves(
    shift_nm: float,
    total_f1: float,
    total_matches_pass: int,
    current: CombShiftSummary,
) -> bool:
    """Whether the current shift beats the existing best summary.

    Higher total F1 wins; on (np.isclose) F1 ties, higher passing-match count
    wins; on a further tie, smaller absolute shift wins.
    """
    prev_f1 = current["total_f1"]
    prev_matches = current["total_matches"]
    if total_f1 > prev_f1:
        return True
    if np.isclose(total_f1, prev_f1):
        if total_matches_pass > prev_matches:
            return True
        if total_matches_pass == prev_matches and abs(shift_nm) < abs(float(current["shift_nm"])):
            return True
    return False


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
        ss = _score_comb_shift(
            shift_nm=shift_nm,
            peaks=peaks,
            transitions_by_element=transitions_by_element,
            total_peaks=total_peaks,
            wavelength_tolerance_nm=wavelength_tolerance_nm,
            comb_min_matches=comb_min_matches,
            comb_min_precision=comb_min_precision,
            comb_min_recall=comb_min_recall,
            comb_max_missing_fraction=comb_max_missing_fraction,
        )

        if fallback_summary is None or _fallback_summary_improves(
            shift_nm, ss.total_matches_all, fallback_summary
        ):
            fallback_summary = {
                "shift_nm": float(shift_nm),
                "scores": ss.scores,
                "total_matches": ss.total_matches_all,
                "passed_elements": ss.passed_elements,
                "total_f1": ss.total_f1,
            }

        if best_summary is None or _best_summary_improves(
            shift_nm, ss.total_f1, ss.total_matches_pass, best_summary
        ):
            best_summary = {
                "shift_nm": float(shift_nm),
                "scores": ss.scores,
                "total_matches": ss.total_matches_pass,
                "passed_elements": ss.passed_elements,
                "total_f1": ss.total_f1,
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


def _kdet_dispatch_rust(
    peak_wavelengths: np.ndarray,
    transitions_by_element: Dict[str, List[Transition]],
    shift_grid: np.ndarray,
    wavelength_tolerance_nm: float,
    kdet_min_score: float,
    kdet_min_candidates: int,
    kdet_rarity_power: float,
    kdet_weight_clip: Tuple[float, float],
    warnings: List[str],
) -> Optional[Dict[str, List[Transition]]]:
    """Run the Rust kdet filter; return ``None`` to fall through to Python.

    Appends ``kdet_filtered_elements`` to ``warnings`` exactly as the inline
    path did. Returns ``None`` only when the Rust call raises (caller then runs
    the pure-NumPy path).
    """
    try:
        element_names = list(transitions_by_element.keys())
        transition_wls = [
            sorted([t.wavelength_nm for t in transitions_by_element[el]]) for el in element_names
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
        return filtered
    except Exception as e:
        logger.warning(f"Rust kdet filter failed, falling back to Python: {e}")
        return None


def _kdet_best_candidates(
    peak_wavelengths: np.ndarray,
    transitions_wl: np.ndarray,
    shift_grid: np.ndarray,
    wavelength_tolerance_nm: float,
    use_jax: bool,
) -> int:
    """Max in-tolerance candidate count across the shift grid for one element."""
    if use_jax and _HAS_JAX:
        # Single vmapped call across the entire shift grid replaces the
        # Python loop below. Behaviorally identical: we take the max of
        # the per-shift candidate counts.
        counts = _kdet_candidate_counts_jax(
            peak_wavelengths, transitions_wl, shift_grid, wavelength_tolerance_nm
        )
        return int(counts.max()) if counts.size > 0 else 0
    best_candidates = 0
    for shift_nm in shift_grid:
        shifted_peaks = peak_wavelengths + shift_nm
        candidate_mask = _peaks_within_tolerance(
            shifted_peaks, transitions_wl, wavelength_tolerance_nm
        )
        candidate_count = int(np.sum(candidate_mask))
        if candidate_count > best_candidates:
            best_candidates = candidate_count
    return best_candidates


def _kdet_element_passes(
    best_candidates: int,
    kdet_fraction: float,
    element_density: float,
    median_density: float,
    kdet_rarity_power: float,
    kdet_weight_clip: Tuple[float, float],
    kdet_min_score: float,
    kdet_min_candidates: int,
    shift_coherence: bool,
    coherence_min_lines: int,
) -> bool:
    """Whether an element survives the kdet filter (coherence or density score)."""
    rarity_weight = (median_density / max(element_density, 1e-6)) ** kdet_rarity_power
    rarity_weight = float(np.clip(rarity_weight, kdet_weight_clip[0], kdet_weight_clip[1]))
    kdet_score = kdet_fraction * rarity_weight
    if shift_coherence:
        # Coherence mode: keep any element with enough candidate matches and
        # do NOT apply the density-scaled score (which wrongly drops sparse
        # real majors). The authoritative shift-coherence veto downstream
        # rejects the incoherent confounders this admits.
        return best_candidates >= max(kdet_min_candidates, coherence_min_lines)
    return best_candidates >= kdet_min_candidates and kdet_score >= kdet_min_score


def _kdet_element_densities(
    transitions_by_element: Dict[str, List[Transition]],
    peak_wavelengths: np.ndarray,
) -> Tuple[Dict[str, float], float]:
    """Per-element line density and the median density across elements."""
    densities = []
    element_density: Dict[str, float] = {}
    wl_range = max(float(peak_wavelengths.max() - peak_wavelengths.min()), 1e-6)
    for element, transitions in transitions_by_element.items():
        density = len(transitions) / wl_range
        element_density[element] = density
        densities.append(density)
    median_density = float(np.median(densities)) if densities else 1.0
    return element_density, median_density


def _kdet_passing_elements(
    transitions_by_element: Dict[str, List[Transition]],
    peak_wavelengths: np.ndarray,
    shift_grid: np.ndarray,
    total_peaks: int,
    wavelength_tolerance_nm: float,
    element_density: Dict[str, float],
    median_density: float,
    kdet_min_score: float,
    kdet_min_candidates: int,
    kdet_rarity_power: float,
    kdet_weight_clip: Tuple[float, float],
    use_jax: bool,
    shift_coherence: bool,
    coherence_min_lines: int,
) -> Dict[str, List[Transition]]:
    """Per-element kdet pass/fail loop (pure-NumPy path)."""
    filtered: Dict[str, List[Transition]] = {}
    for element, transitions in transitions_by_element.items():
        if not transitions:
            continue
        transitions_wl = np.array([t.wavelength_nm for t in transitions], dtype=float)
        transitions_wl.sort()

        best_candidates = _kdet_best_candidates(
            peak_wavelengths, transitions_wl, shift_grid, wavelength_tolerance_nm, use_jax
        )
        kdet_fraction = best_candidates / total_peaks
        density = element_density.get(element, median_density)
        if _kdet_element_passes(
            best_candidates,
            kdet_fraction,
            density,
            median_density,
            kdet_rarity_power,
            kdet_weight_clip,
            kdet_min_score,
            kdet_min_candidates,
            shift_coherence,
            coherence_min_lines,
        ):
            filtered[element] = transitions
    return filtered


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
    use_jax: bool = False,
    shift_coherence: bool = False,
    coherence_min_lines: int = 2,
    coherence_min_fraction: float = 0.5,
) -> Tuple[Dict[str, List[Transition]], List[str]]:
    warnings: List[str] = []
    if not peaks or not transitions_by_element:
        return transitions_by_element, warnings

    peak_wavelengths = np.array([p[1] for p in peaks], dtype=float)
    total_peaks = len(peak_wavelengths)
    if total_peaks == 0:
        return transitions_by_element, warnings

    shift_grid = _build_shift_grid(shift_scan_nm, shift_step_nm, wl_step, wavelength_tolerance_nm)

    # The kdet density-scaled fraction (best_candidates / total_peaks * rarity)
    # scales with spectrum density, so on a rich multi-element spectrum it drops
    # sparse-but-real majors (e.g. Al at 0.042 < the 0.05 floor) while admitting
    # dense confounders that coincide by chance. When ``shift_coherence`` is on
    # we instead keep elements whose candidate matches *agree on one shift* (the
    # shared instrument residual) and drop the incoherent ones — a
    # catalog-density-invariant discriminator. The Rust fast path computes only
    # the legacy density score (it receives wavelengths, not the per-line
    # residuals coherence needs), so we route through the pure-NumPy path when
    # coherence is requested to keep both backends behaviourally identical.
    if HAS_RUST_CORE and not shift_coherence:
        rust_filtered = _kdet_dispatch_rust(
            peak_wavelengths,
            transitions_by_element,
            shift_grid,
            wavelength_tolerance_nm,
            kdet_min_score,
            kdet_min_candidates,
            kdet_rarity_power,
            kdet_weight_clip,
            warnings,
        )
        if rust_filtered is not None:
            return rust_filtered, warnings

    element_density, median_density = _kdet_element_densities(
        transitions_by_element, peak_wavelengths
    )

    filtered = _kdet_passing_elements(
        transitions_by_element,
        peak_wavelengths,
        shift_grid,
        total_peaks,
        wavelength_tolerance_nm,
        element_density,
        median_density,
        kdet_min_score,
        kdet_min_candidates,
        kdet_rarity_power,
        kdet_weight_clip,
        use_jax,
        shift_coherence,
        coherence_min_lines,
    )

    if filtered and len(filtered) < len(transitions_by_element):
        warnings.append("kdet_filtered_elements")

    return filtered, warnings


def _element_match_residuals(
    peak_wavelengths: np.ndarray,
    transitions: List[Transition],
    shift_nm: float,
    tolerance_nm: float,
) -> np.ndarray:
    """Per-line nearest-peak residual offsets (signed nm) within tolerance.

    For each transition, the signed offset ``(peak + shift) - line`` of its
    nearest detected peak, kept only if within ``tolerance_nm``. The spread of
    these residuals measures whether the element's matches *agree on one
    shift*: a real element's true lines cluster at the single instrument
    residual offset, whereas a confounder matching peaks by coincidence
    scatters across the whole tolerance window.
    """
    if peak_wavelengths.size == 0 or not transitions:
        return np.empty(0, dtype=float)
    shifted = peak_wavelengths + shift_nm
    residuals: List[float] = []
    for t in transitions:
        deltas = shifted - t.wavelength_nm
        abs_deltas = np.abs(deltas)
        j = int(np.argmin(abs_deltas))
        if abs_deltas[j] <= tolerance_nm:
            residuals.append(float(deltas[j]))
    return np.asarray(residuals, dtype=float)


def _shift_coherence_veto(
    elements: List[str],
    peaks: List[Tuple[int, float]],
    transitions_by_element: Dict[str, List[Transition]],
    applied_shift_nm: float,
    tolerance_nm: float,
    min_coherent_lines: int = 2,
    min_coherent_fraction: float = 0.5,
) -> Tuple[List[str], List[str]]:
    """Reject elements whose matched lines do not agree on one shift.

    A genuinely-present element's lines all match at the *same* sub-tolerance
    residual offset (the instrument's residual dispersion error, shared by
    every real species). A false-positive confounder — typically a dense
    catalog (Ag, Sn, W, Bi) — accrues matches by random wavelength coincidence,
    so its residual offsets scatter across the full ``+/- tolerance`` window.

    The veto computes the consensus residual (median of all candidate elements'
    pooled match residuals — the instrument offset) and keeps an element only
    if at least ``min_coherent_fraction`` of its matched lines (and at least
    ``min_coherent_lines``) fall within one resolution element
    (``tolerance_nm / 3``) of that consensus. This is intentionally a coarse,
    catalog-agnostic rule (no per-element or per-spectrum tuning): it rejects
    incoherent confounders while admitting real majors that share the offset.

    Returns ``(kept_elements, vetoed_elements)``.
    """
    if not elements:
        return elements, []
    peak_wavelengths = np.array([p[1] for p in peaks], dtype=float)
    if peak_wavelengths.size == 0:
        return elements, []

    residuals_by_element: Dict[str, np.ndarray] = {}
    pooled: List[float] = []
    for element in elements:
        res = _element_match_residuals(
            peak_wavelengths,
            transitions_by_element.get(element, []),
            applied_shift_nm,
            tolerance_nm,
        )
        residuals_by_element[element] = res
        pooled.extend(res.tolist())

    if not pooled:
        return elements, []

    consensus = float(np.median(pooled))
    band = tolerance_nm / 3.0

    kept: List[str] = []
    vetoed: List[str] = []
    for element in elements:
        res = residuals_by_element[element]
        n = res.size
        if n == 0:
            # No in-tolerance matches at the applied shift; leave the element
            # to the downstream matcher rather than vetoing on no evidence.
            kept.append(element)
            continue
        coherent = int(np.sum(np.abs(res - consensus) <= band))
        fraction = coherent / n
        # An element survives if a majority of its matches cohere with the
        # consensus shift. The dominant FP mode is a dense catalog whose many
        # matches are random coincidences (low coherent fraction); a genuine
        # element's matches concentrate at the shared instrument residual
        # (high fraction) even when there are few of them. We therefore gate
        # primarily on the coherent *fraction* (catalog-density-invariant) and
        # only additionally require ``min_coherent_lines`` when the element has
        # enough matches to make that count meaningful — so a sparse real
        # element (e.g. one dominant line on a thin synthetic) is not vetoed
        # for want of a second line.
        enough_evidence = n >= min_coherent_lines
        passes_fraction = fraction >= min_coherent_fraction
        passes_count = coherent >= min_coherent_lines or not enough_evidence
        if passes_fraction and passes_count:
            kept.append(element)
        else:
            vetoed.append(element)
    return kept, vetoed


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
    use_jax_fallback: bool = False,
) -> List[Tuple[int, float]]:
    """Detect spectral peaks.

    The default path uses ``scipy.signal.find_peaks`` (height + distance +
    prominence). This is **intentionally not** ported to JAX — the C
    implementation runs in ~0.2 ms on a 4096-sample spectrum and produces
    variable-length output that is hostile to JAX's static-shape model.
    See ``docs/jax-port/line-detection-consultation.md`` for the full
    audit + Codex/Gemini reviewer recommendations.

    When SciPy is unavailable, the simple local-maxima fallback runs. The
    ``use_jax_fallback`` flag picks between the Python and JAX
    implementations of that fallback (both behaviorally identical;
    the JAX one is useful when the fallback is hot for some reason).
    """
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

    # Fallback: local maxima above threshold. Two implementations available.
    if use_jax_fallback and _HAS_JAX:
        return _find_peaks_jax_fallback(wavelength, intensity, min_peak_height)

    peaks: List[Tuple[int, float]] = []
    for i in range(1, len(intensity) - 1):
        if (
            normalized[i] >= threshold
            and intensity[i] > intensity[i - 1]
            and intensity[i] > intensity[i + 1]
        ):
            peaks.append((i, float(wavelength[i])))
    return peaks
