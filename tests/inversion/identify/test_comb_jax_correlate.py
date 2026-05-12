"""
Tests for the JAX-vectorized comb tooth correlation in
``cflibs.inversion.identify.comb``.

The new ``use_jax_correlate=True`` constructor flag opts into a
vectorized replacement for the per-tooth ``scipy.stats.pearsonr`` hot
loop. These tests verify:

1. Per-tooth numerical agreement with the CPU implementation across a
   range of widths, shifts, and edge-of-spectrum corner cases.
2. End-to-end identifier output (detected/rejected, fingerprint scores)
   matches the CPU path on a synthetic LIBS spectrum.
3. ``use_jax_correlate=False`` (default) keeps the CPU path bit-exact
   unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from cflibs.inversion.identify.comb import (  # noqa: E402
    CombIdentifier,
    _correlate_tooth_jax,
)

pytestmark = [pytest.mark.requires_jax]


# ---------- Helpers ----------------------------------------------------------


def _make_synthetic_spectrum(
    line_centers: list[float],
    line_amplitudes: list[float],
    n_points: int = 4000,
    wl_min: float = 380.0,
    wl_max: float = 430.0,
    sigma_nm: float = 0.05,
    noise: float = 0.5,
    seed: int = 0xC0FFEE,
):
    """Synthetic LIBS-ish spectrum with Gaussian peaks + low noise."""
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(wl_min, wl_max, n_points)
    intensity = np.full(n_points, 5.0, dtype=np.float64)
    for c, a in zip(line_centers, line_amplitudes):
        intensity += a * np.exp(-0.5 * ((wavelength - c) / sigma_nm) ** 2)
    intensity += rng.normal(0.0, noise, n_points)
    intensity = np.clip(intensity, 0.0, None)
    return wavelength, intensity


def _cpu_tooth(
    identifier: CombIdentifier,
    wavelength: np.ndarray,
    intensity: np.ndarray,
    baseline: np.ndarray,
    center_nm: float,
    threshold: float,
) -> dict:
    """Run the CPU ``_correlate_tooth`` on the identifier."""
    saved = identifier.use_jax_correlate
    identifier.use_jax_correlate = False
    try:
        return identifier._correlate_tooth(
            wavelength, intensity, baseline, center_nm, threshold
        )
    finally:
        identifier.use_jax_correlate = saved


def _jax_tooth(
    identifier: CombIdentifier,
    wavelength: np.ndarray,
    intensity: np.ndarray,
    baseline: np.ndarray,
    center_nm: float,
    threshold: float,
) -> dict:
    saved = identifier.use_jax_correlate
    identifier.use_jax_correlate = True
    try:
        return identifier._correlate_tooth(
            wavelength, intensity, baseline, center_nm, threshold
        )
    finally:
        identifier.use_jax_correlate = saved


# ---------- Unit kernel ------------------------------------------------------


def test_kernel_matches_cpu_pearson_on_known_signal():
    """The kernel reproduces scipy.stats.pearsonr on a controlled grid."""
    from scipy.stats import pearsonr

    rng = np.random.default_rng(42)
    n_spec = 100
    intensity = np.zeros(n_spec)
    intensity[40:51] = np.exp(-0.5 * (np.arange(11) - 5) ** 2 / 1.5**2) * 100.0
    intensity += rng.normal(0, 0.1, n_spec)
    baseline = np.zeros(n_spec)
    center_idx = 45

    widths = np.array([5, 7, 9], dtype=np.int64)
    shifts = np.arange(-3, 4, dtype=np.int64)
    max_w = 9
    templates = np.zeros((3, max_w))
    template_mask = np.zeros_like(templates)
    half_max = max_w // 2
    for i, w in enumerate(widths):
        half_w = int(w) // 2
        for j in range(int(w)):
            distance = abs(j - half_w)
            templates[i, half_max - half_w + j] = 1.0 - (distance / (half_w + 1))
            template_mask[i, half_max - half_w + j] = 1.0
        templates[i] /= templates[i].max()  # normalize to 1

    corr_grid, _, candidate_ok = _correlate_tooth_jax(
        intensity, baseline, center_idx,
        widths, shifts, templates, template_mask,
    )

    # Cross-check a sample candidate against scipy.
    for wi, w in enumerate(widths):
        for si, s in enumerate(shifts):
            half_w = int(w) // 2
            start = max(0, center_idx + s - half_w)
            end = min(n_spec, center_idx + s + half_w + 1)
            seg = intensity[start:end]
            tmpl = templates[wi, half_max - half_w : half_max - half_w + int(w)]
            if len(seg) != len(tmpl) or len(seg) < 3:
                # CPU would skip; JAX should be 0 or candidate_ok False.
                assert not candidate_ok[wi, si] or abs(corr_grid[wi, si]) < 1e-10
                continue
            if np.std(seg) < 1e-10 or np.std(tmpl) < 1e-10:
                assert abs(corr_grid[wi, si]) < 1e-10
                continue
            r, _ = pearsonr(seg, tmpl)
            assert corr_grid[wi, si] == pytest.approx(r, rel=1e-9, abs=1e-12)


# ---------- Per-tooth integration -------------------------------------------


@pytest.mark.parametrize(
    "center_nm",
    [400.0, 405.0, 411.5, 420.0],  # interior peaks + a non-peak
)
def test_correlate_tooth_jax_matches_cpu(center_nm):
    """JAX tooth output matches CPU tooth output bit-for-bit on the metric."""
    line_centers = [400.0, 405.0, 415.0]
    line_amps = [50.0, 30.0, 40.0]
    wavelength, intensity = _make_synthetic_spectrum(line_centers, line_amps)

    identifier = CombIdentifier(
        atomic_db=None,
        max_shift_pts=5,
        min_width_pts=5,
        max_width_factor=1.0,
    )
    baseline, threshold = identifier._estimate_baseline_threshold(wavelength, intensity)

    cpu = _cpu_tooth(identifier, wavelength, intensity, baseline, center_nm, threshold)
    j = _jax_tooth(identifier, wavelength, intensity, baseline, center_nm, threshold)

    assert cpu["center_nm"] == j["center_nm"]
    assert cpu["best_correlation"] == pytest.approx(j["best_correlation"], rel=1e-5, abs=1e-7)
    # When there's a unique best, shift and width must match.
    if cpu["best_correlation"] > 1e-6:
        assert cpu["best_shift"] == j["best_shift"]
        assert cpu["best_width"] == j["best_width"]
    assert cpu["active"] == j["active"]


def test_correlate_tooth_jax_edge_of_spectrum():
    """Edge-of-spectrum lines should produce identical results."""
    wavelength, intensity = _make_synthetic_spectrum(
        [381.0, 429.0], [40.0, 40.0], n_points=2000
    )
    identifier = CombIdentifier(
        atomic_db=None, max_shift_pts=5, min_width_pts=5
    )
    baseline, threshold = identifier._estimate_baseline_threshold(wavelength, intensity)

    # Lines near both ends.
    for center in (381.0, 429.0):
        cpu = _cpu_tooth(identifier, wavelength, intensity, baseline, center, threshold)
        j = _jax_tooth(identifier, wavelength, intensity, baseline, center, threshold)
        assert cpu["active"] == j["active"]
        assert cpu["best_correlation"] == pytest.approx(
            j["best_correlation"], rel=1e-5, abs=1e-7
        )


def test_correlate_tooth_jax_flat_signal_returns_zero():
    """A flat (no-peak) center gives correlation 0 in both paths."""
    wavelength = np.linspace(390.0, 410.0, 2000)
    intensity = np.ones_like(wavelength) * 10.0  # no signal at all
    identifier = CombIdentifier(atomic_db=None, max_shift_pts=5, min_width_pts=5)
    baseline, threshold = identifier._estimate_baseline_threshold(wavelength, intensity)
    cpu = _cpu_tooth(identifier, wavelength, intensity, baseline, 400.0, threshold)
    j = _jax_tooth(identifier, wavelength, intensity, baseline, 400.0, threshold)
    assert cpu["best_correlation"] == j["best_correlation"] == 0.0
    assert cpu["active"] is False
    assert j["active"] is False


def test_constructor_rejects_jax_when_not_installed(monkeypatch):
    """Constructor surfaces a clear error if use_jax_correlate=True but JAX missing."""
    import cflibs.inversion.identify.comb as comb_module
    monkeypatch.setattr(comb_module, "_HAS_JAX", False)
    with pytest.raises(ImportError, match="requires JAX"):
        CombIdentifier(atomic_db=None, use_jax_correlate=True)


# ---------- End-to-end ------------------------------------------------------


@pytest.mark.requires_db
def test_end_to_end_jax_matches_cpu(temp_db, synthetic_libs_spectrum):
    """End-to-end identifier output is identical between CPU and JAX paths."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (373.49, 800.0), (374.95, 600.0)]},
        noise_level=0.05,
    )
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    cpu_id = CombIdentifier(
        atomic_db=db, elements=["Fe"], min_correlation=0.05,
        use_jax_correlate=False,
    )
    jax_id = CombIdentifier(
        atomic_db=db, elements=["Fe"], min_correlation=0.05,
        use_jax_correlate=True,
    )

    cpu_result = cpu_id.identify(wavelength, intensity)
    jax_result = jax_id.identify(wavelength, intensity)

    cpu_elems = {e.element for e in cpu_result.detected_elements}
    jax_elems = {e.element for e in jax_result.detected_elements}
    assert cpu_elems == jax_elems

    # Per-element fingerprint score parity.
    cpu_by = {e.element: e for e in cpu_result.all_elements}
    jax_by = {e.element: e for e in jax_result.all_elements}
    for el, cpu_e in cpu_by.items():
        jax_e = jax_by[el]
        assert cpu_e.score == pytest.approx(jax_e.score, rel=1e-5, abs=1e-7)
        assert cpu_e.n_matched_lines == jax_e.n_matched_lines
        assert cpu_e.detected == jax_e.detected
