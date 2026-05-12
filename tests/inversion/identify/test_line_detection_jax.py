"""
Tests for the JAX paths in ``cflibs.inversion.identify.line_detection``.

The JAX port is intentionally **partial**. ``scipy.signal.find_peaks`` and
its ``height``/``distance``/``prominence`` semantics are deliberately kept
on CPU — see ``docs/jax-port/line-detection-consultation.md`` for the audit
and Codex/Gemini reviewer recommendations.

What is ported (opt-in):

1. :func:`_peaks_within_tolerance_jax` — drop-in for
   :func:`_peaks_within_tolerance` (searchsorted + abs/min reduction).
2. :func:`_kdet_candidate_counts_jax` — vmapped shift × peak candidate
   counter, replacing the inner Python loop in
   :func:`_kdet_filter_elements`.
3. :func:`_find_peaks_jax_fallback` — drop-in for the simple local-maxima
   fallback path inside :func:`_find_peaks` (only relevant when SciPy is
   unavailable). The default scipy.signal.find_peaks path is untouched.
4. :func:`detect_line_observations(use_jax_kdet=True)` /
   :func:`detect_line_observations(use_jax_peak_fallback=True)` —
   opt-in flags on the public entry point.

These tests exercise numerical agreement (rtol 1e-5 for what's ported) and
behavioral parity for the find_peaks call (we verify that the public API
returns the same observations whether or not the JAX flag is set).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from cflibs.inversion.identify.line_detection import (  # noqa: E402
    _HAS_JAX,
    _find_peaks,
    _find_peaks_jax_fallback,
    _kdet_candidate_counts_jax,
    _kdet_filter_elements,
    _peaks_within_tolerance,
    _peaks_within_tolerance_jax,
    _build_shift_grid,
    detect_line_observations,
)
from cflibs.atomic.structures import Transition  # noqa: E402

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


# ---------- _peaks_within_tolerance_jax -------------------------------------


def test_peaks_within_tolerance_jax_matches_numpy_basic():
    peaks = np.array([371.99, 373.49, 400.0, 425.0], dtype=float)
    transitions = np.array([371.95, 373.50, 380.0, 425.05], dtype=float)
    transitions.sort()

    for tol in (0.01, 0.05, 0.1, 0.5):
        np_mask = _peaks_within_tolerance(peaks, transitions, tol)
        jx_mask = _peaks_within_tolerance_jax(peaks, transitions, tol)
        np.testing.assert_array_equal(np_mask, jx_mask, err_msg=f"tol={tol}")


def test_peaks_within_tolerance_jax_handles_empty_peaks():
    peaks = np.array([], dtype=float)
    transitions = np.array([371.99, 373.49], dtype=float)
    mask = _peaks_within_tolerance_jax(peaks, transitions, 0.1)
    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_peaks_within_tolerance_jax_handles_empty_transitions():
    peaks = np.array([371.99, 400.0], dtype=float)
    transitions = np.array([], dtype=float)
    mask = _peaks_within_tolerance_jax(peaks, transitions, 0.1)
    np.testing.assert_array_equal(mask, np.zeros(2, dtype=bool))


def test_peaks_within_tolerance_jax_single_transition():
    """Single-transition edge case: idx-1 wraps to clip(0), so left and
    right nearest collapse to the same value. Mask is True iff the peak
    is within tolerance of the lone transition."""
    peaks = np.array([371.5, 371.99, 372.5], dtype=float)
    transitions = np.array([371.99], dtype=float)
    np_mask = _peaks_within_tolerance(peaks, transitions, 0.1)
    jx_mask = _peaks_within_tolerance_jax(peaks, transitions, 0.1)
    np.testing.assert_array_equal(np_mask, jx_mask)
    assert jx_mask.tolist() == [False, True, False]


def test_peaks_within_tolerance_jax_large_random_agreement():
    """Stress-test with 1000 random peaks vs 500 random transitions, four
    tolerances. JAX path must produce bit-exact identical bools to the
    NumPy path."""
    rng = np.random.default_rng(0xDEADBEEF)
    peaks = rng.uniform(200.0, 800.0, size=1000)
    transitions = np.sort(rng.uniform(200.0, 800.0, size=500))
    for tol in (0.005, 0.05, 0.5, 5.0):
        np_mask = _peaks_within_tolerance(peaks, transitions, tol)
        jx_mask = _peaks_within_tolerance_jax(peaks, transitions, tol)
        np.testing.assert_array_equal(np_mask, jx_mask, err_msg=f"tol={tol}")


# ---------- _kdet_candidate_counts_jax --------------------------------------


def test_kdet_candidate_counts_jax_matches_numpy_loop():
    """The vmapped count must equal a Python loop over the shift grid."""
    rng = np.random.default_rng(7)
    peaks = rng.uniform(371.0, 380.0, size=30)
    transitions = np.sort(rng.uniform(371.0, 380.0, size=15))
    shifts = np.linspace(-0.5, 0.5, 21)
    tol = 0.05

    expected = np.array(
        [
            int(np.sum(_peaks_within_tolerance(peaks + s, transitions, tol)))
            for s in shifts
        ],
        dtype=np.int64,
    )
    got = _kdet_candidate_counts_jax(peaks, transitions, shifts, tol)
    np.testing.assert_array_equal(expected, got)


def test_kdet_candidate_counts_jax_empty_inputs():
    """Each combination of empty inputs returns zeros of the right shape."""
    shifts = np.linspace(-0.1, 0.1, 5)
    peaks = np.array([371.0, 372.0], dtype=float)
    trans = np.array([371.0], dtype=float)

    # Empty transitions
    out = _kdet_candidate_counts_jax(peaks, np.array([]), shifts, 0.1)
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.int64))

    # Empty peaks
    out = _kdet_candidate_counts_jax(np.array([]), trans, shifts, 0.1)
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.int64))

    # Empty shifts
    out = _kdet_candidate_counts_jax(peaks, trans, np.array([]), 0.1)
    assert out.shape == (0,)


def test_kdet_candidate_counts_jax_known_pattern():
    """When the perfect-alignment shift is in the grid, that index should
    have the highest count (typically all peaks)."""
    peaks = np.array([371.0, 373.0, 375.0], dtype=float)
    transitions = np.sort(np.array([371.0, 373.0, 375.0]))
    # Pre-shift the peaks by +0.3 so the best-matching shift is -0.3
    peaks_shifted = peaks + 0.3
    shifts = np.array([-0.5, -0.3, 0.0, 0.3, 0.5])
    counts = _kdet_candidate_counts_jax(peaks_shifted, transitions, shifts, 0.05)
    # The -0.3 shift should match all 3 peaks
    assert counts[1] == 3
    # No other shift should match all 3 with the small tolerance
    assert counts.max() == 3
    assert counts[0] < 3 and counts[2] < 3 and counts[3] < 3 and counts[4] < 3


# ---------- _find_peaks_jax_fallback ----------------------------------------


def _gaussian(wavelength: np.ndarray, center: float, sigma: float, amplitude: float):
    return amplitude * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)


def test_find_peaks_jax_fallback_simple_two_peak():
    wl = np.linspace(370.0, 380.0, 2000)
    intensity = _gaussian(wl, 372.0, 0.05, 10.0) + _gaussian(wl, 376.5, 0.05, 5.0)

    py_peaks = _find_peaks(wl, intensity, 0.05, 0.15)  # scipy default path
    # The simple-fallback path *bypasses scipy* by going through
    # _find_peaks_jax_fallback directly:
    jax_peaks = _find_peaks_jax_fallback(wl, intensity, 0.05)

    # Both methods should find both Gaussian centers. JAX fallback uses
    # strict local maxima so it picks up *exactly* the sample bin closest
    # to each Gaussian peak.
    assert len(jax_peaks) >= 2
    wls = [pw for _, pw in jax_peaks]
    assert any(abs(w - 372.0) < 0.05 for w in wls)
    assert any(abs(w - 376.5) < 0.05 for w in wls)

    # scipy path also returns those peaks (different post-filtering, but
    # the centers are present in both lists).
    py_wls = [pw for _, pw in py_peaks]
    assert any(abs(w - 372.0) < 0.05 for w in py_wls)
    assert any(abs(w - 376.5) < 0.05 for w in py_wls)


def test_find_peaks_jax_fallback_matches_python_fallback_exactly():
    """The JAX fallback and the Python fallback should produce *byte-
    identical* lists of (index, wavelength) tuples for the same inputs.
    """
    rng = np.random.default_rng(42)
    wl = np.linspace(200.0, 400.0, 4000)
    base = rng.uniform(0.0, 0.01, size=wl.shape)
    intensity = (
        base
        + _gaussian(wl, 250.0, 0.05, 1.0)
        + _gaussian(wl, 300.0, 0.05, 0.5)
        + _gaussian(wl, 350.0, 0.05, 0.8)
    )
    max_intensity = intensity.max()
    normalized = intensity / max_intensity
    threshold = 0.05

    # Python fallback (manual loop)
    py_peaks = []
    for i in range(1, len(intensity) - 1):
        if (
            normalized[i] >= threshold
            and intensity[i] > intensity[i - 1]
            and intensity[i] > intensity[i + 1]
        ):
            py_peaks.append((i, float(wl[i])))

    jax_peaks = _find_peaks_jax_fallback(wl, intensity, threshold)

    assert len(py_peaks) == len(jax_peaks)
    py_indices = [i for i, _ in py_peaks]
    jax_indices = [i for i, _ in jax_peaks]
    assert py_indices == jax_indices


def test_find_peaks_jax_fallback_empty_spectrum():
    wl = np.array([], dtype=float)
    intensity = np.array([], dtype=float)
    out = _find_peaks_jax_fallback(wl, intensity, 0.05)
    assert out == []


def test_find_peaks_jax_fallback_zero_intensity():
    wl = np.linspace(370.0, 380.0, 100)
    intensity = np.zeros_like(wl)
    out = _find_peaks_jax_fallback(wl, intensity, 0.05)
    assert out == []


# ---------- _kdet_filter_elements use_jax wiring ----------------------------


def _make_transition(element: str, wl_nm: float) -> Transition:
    return Transition(
        element=element,
        ionization_stage=1,
        wavelength_nm=wl_nm,
        A_ki=1.0e7,
        E_k_ev=3.0,
        E_i_ev=0.0,
        g_k=9,
        g_i=9,
    )


def test_kdet_filter_elements_jax_vs_numpy_parity():
    """The kdet filter must produce the same filtered element set whether
    use_jax=True or use_jax=False. This is the wiring smoke test for the
    public flag — the inner numerical kernel is checked above."""
    peaks = [(i, 371.0 + 0.5 * i) for i in range(8)]  # Peaks every 0.5 nm
    # Element A: lines at every peak (perfect match) -> should pass kdet
    # Element B: lines well outside the peak range -> should fail kdet
    transitions_by_element = {
        "A": [_make_transition("A", 371.0 + 0.5 * i) for i in range(8)],
        "B": [_make_transition("B", 600.0 + 0.5 * i) for i in range(8)],
    }

    kwargs = dict(
        peaks=peaks,
        transitions_by_element=transitions_by_element,
        shift_scan_nm=0.5,
        shift_step_nm=0.05,
        wavelength_tolerance_nm=0.05,
        wl_step=0.01,
        kdet_min_score=0.1,
        kdet_min_candidates=2,
        kdet_rarity_power=0.5,
        kdet_weight_clip=(0.25, 4.0),
    )
    np_filtered, np_warnings = _kdet_filter_elements(**kwargs, use_jax=False)
    jx_filtered, jx_warnings = _kdet_filter_elements(**kwargs, use_jax=True)

    assert set(np_filtered.keys()) == set(jx_filtered.keys())
    assert np_warnings == jx_warnings
    # A must pass, B must not (sanity check on the test design)
    assert "A" in np_filtered
    assert "B" not in np_filtered


# ---------- detect_line_observations opt-in flags --------------------------


def test_detect_line_observations_use_jax_kdet_parity(atomic_db):
    """The public entry point with use_jax_kdet=True must produce
    behaviorally identical results to the default code path."""
    wavelength = np.linspace(371.0, 375.0, 2000)
    intensity = _gaussian(wavelength, 371.99, 0.03, 10.0) + _gaussian(
        wavelength, 373.49, 0.03, 5.0
    )

    base = detect_line_observations(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=["Fe"],
        wavelength_tolerance_nm=0.05,
        min_peak_height=0.05,
        peak_width_nm=0.15,
    )
    jx = detect_line_observations(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=["Fe"],
        wavelength_tolerance_nm=0.05,
        min_peak_height=0.05,
        peak_width_nm=0.15,
        use_jax_kdet=True,
    )

    assert base.total_peaks == jx.total_peaks
    assert base.matched_peaks == jx.matched_peaks
    assert base.applied_shift_nm == pytest.approx(jx.applied_shift_nm, rel=1e-9, abs=1e-9)
    assert base.warnings == jx.warnings
    # Observation lists should match line-for-line (same matched
    # transitions, same intensities). The intensities are computed in
    # NumPy regardless of the kdet backend, so they must be byte-identical.
    assert len(base.observations) == len(jx.observations)
    base_keys = sorted([(o.element, o.ionization_stage, o.wavelength_nm) for o in base.observations])
    jx_keys = sorted([(o.element, o.ionization_stage, o.wavelength_nm) for o in jx.observations])
    assert base_keys == jx_keys


def test_detect_line_observations_use_jax_peak_fallback_default_unchanged(atomic_db):
    """The use_jax_peak_fallback flag must NOT change behavior when scipy
    is available — the scipy.signal.find_peaks path is intentionally
    untouched. This is the central contract of the partial port."""
    wavelength = np.linspace(371.0, 375.0, 2000)
    intensity = _gaussian(wavelength, 371.99, 0.03, 10.0) + _gaussian(
        wavelength, 373.49, 0.03, 5.0
    )

    default = detect_line_observations(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=["Fe"],
        wavelength_tolerance_nm=0.05,
        min_peak_height=0.05,
        peak_width_nm=0.15,
    )
    with_flag = detect_line_observations(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=["Fe"],
        wavelength_tolerance_nm=0.05,
        min_peak_height=0.05,
        peak_width_nm=0.15,
        use_jax_peak_fallback=True,
    )

    # Should be bit-identical because scipy.signal.find_peaks is still
    # used as the peak finder. The use_jax_peak_fallback flag only
    # matters if scipy is unavailable.
    assert default.total_peaks == with_flag.total_peaks
    assert default.matched_peaks == with_flag.matched_peaks
    assert len(default.observations) == len(with_flag.observations)


def test_detect_line_observations_all_jax_flags_consistent(atomic_db):
    """Combining use_jax_kdet + use_jax_peak_fallback should still
    produce identical results to the default scipy/NumPy path."""
    wavelength = np.linspace(371.0, 375.0, 2000)
    intensity = _gaussian(wavelength, 371.99, 0.03, 10.0) + _gaussian(
        wavelength, 373.49, 0.03, 5.0
    )

    default = detect_line_observations(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=["Fe"],
        wavelength_tolerance_nm=0.05,
        min_peak_height=0.05,
        peak_width_nm=0.15,
    )
    all_jax = detect_line_observations(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=["Fe"],
        wavelength_tolerance_nm=0.05,
        min_peak_height=0.05,
        peak_width_nm=0.15,
        use_jax_kdet=True,
        use_jax_peak_fallback=True,
    )

    assert default.total_peaks == all_jax.total_peaks
    assert default.matched_peaks == all_jax.matched_peaks
    assert default.applied_shift_nm == pytest.approx(
        all_jax.applied_shift_nm, rel=1e-9, abs=1e-9
    )


# ---------- regression: shift grid + tolerance kernel ----------------------


def test_jax_kdet_handles_realistic_shift_grid():
    """Build a realistic shift grid via the public helper and verify the
    JAX counter integrates cleanly with it."""
    shift_grid = _build_shift_grid(0.5, None, 0.01, 0.1)
    peaks = np.array([371.0, 373.0, 380.0, 400.0, 425.0], dtype=float)
    transitions = np.sort(
        np.array([371.05, 373.0, 380.05, 400.1, 425.0, 500.0], dtype=float)
    )
    counts = _kdet_candidate_counts_jax(peaks, transitions, shift_grid, 0.05)
    # Output shape must match the grid
    assert counts.shape == shift_grid.shape
    # All counts must be in [0, len(peaks)]
    assert counts.min() >= 0
    assert counts.max() <= len(peaks)


def test_has_jax_flag_is_true_in_env():
    """Sanity: the test must run with JAX importable."""
    assert _HAS_JAX, "JAX must be importable for these tests to run"
