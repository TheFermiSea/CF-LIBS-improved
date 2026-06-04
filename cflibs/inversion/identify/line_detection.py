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


def _build_observation_from_fit(
    transition: Transition,
    fit_result: VoigtFitResult,
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

    transitions = _load_transitions(
        atomic_db,
        elements,
        wavelength_min=wl_min,
        wavelength_max=wl_max,
        min_relative_intensity=min_relative_intensity,
        top_k_per_element=top_k_per_element,
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
            use_jax=use_jax_kdet,
            shift_coherence=shift_coherence_veto,
            coherence_min_lines=coherence_min_lines,
            coherence_min_fraction=coherence_min_fraction,
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

    # Shift-coherence veto: drop accepted elements whose matched comb lines do
    # not agree on the single instrument residual shift (random-coincidence
    # confounders), applied uniformly to both the comb-pass and comb-fallback
    # selections. Runs after the (Rust or Python) comb scan so it gates both
    # backends identically.
    if shift_coherence_veto and accepted_elements:
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
            accepted_elements = kept
            warnings.append("shift_coherence_vetoed_elements")

    matched_peak_indices: Set[int] = set()

    # Optionally run deconvolution on matched peaks
    deconv_results_by_wl: Optional[Dict[float, VoigtFitResult]] = None
    if use_deconvolution:
        try:
            from cflibs.inversion.deconvolution import deconvolve_peaks

            # Collect all peak wavelengths for deconvolution
            peak_wl_arr = np.array([pw for _, pw in peaks], dtype=float)
            if len(peak_wl_arr) > 0:
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
                deconv_results_by_wl = {}
                for fr in deconv.fit_results:
                    best_peak_wl = float(peak_wl_arr[np.argmin(np.abs(peak_wl_arr - fr.center_nm))])
                    deconv_results_by_wl[round(best_peak_wl, 4)] = fr
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
                rounded_wl = round(peak_wl, 4)
                if deconv_results_by_wl is not None and rounded_wl in deconv_results_by_wl:
                    result = _build_observation_from_fit(
                        transition,
                        deconv_results_by_wl[rounded_wl],
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
                rounded_wl = round(peak_wl, 4)
                if deconv_results_by_wl is not None and rounded_wl in deconv_results_by_wl:
                    result = _build_observation_from_fit(
                        transition,
                        deconv_results_by_wl[rounded_wl],
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

        if use_jax and _HAS_JAX:
            # Single vmapped call across the entire shift grid replaces the
            # Python loop below. Behaviorally identical: we take the max of
            # the per-shift candidate counts.
            counts = _kdet_candidate_counts_jax(
                peak_wavelengths, transitions_wl, shift_grid, wavelength_tolerance_nm
            )
            best_candidates = int(counts.max()) if counts.size > 0 else 0
        else:
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
        if shift_coherence:
            # Coherence mode: keep any element with enough candidate matches and
            # do NOT apply the density-scaled score (which wrongly drops sparse
            # real majors). The authoritative shift-coherence veto downstream
            # rejects the incoherent confounders this admits.
            if best_candidates >= max(kdet_min_candidates, coherence_min_lines):
                filtered[element] = transitions
        elif best_candidates >= kdet_min_candidates and kdet_score >= kdet_min_score:
            filtered[element] = transitions

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
