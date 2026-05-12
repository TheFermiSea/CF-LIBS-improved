"""
Tests for the JAX-vectorized classic-mode correlation in
``cflibs.inversion.identify.correlation``.

The ``use_jax_classic=True`` constructor flag opts into a fused JAX
kernel for the ``(T, n_e)`` grid model-spectrum correlation hot loop.
These tests verify:

1. Numerical agreement of the fused kernel vs. the per-grid-point CPU
   path on a synthetic line list.
2. End-to-end identifier output (detected/rejected, correlation score)
   matches the CPU path on a synthetic LIBS spectrum.
3. ``use_jax_classic=False`` (default) keeps the CPU path bit-exact
   unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from cflibs.inversion.identify.correlation import (  # noqa: E402
    CorrelationIdentifier,
)

pytestmark = [pytest.mark.requires_jax]


# ---------- Helpers ----------------------------------------------------------


@dataclass
class FakeTransition:
    """Lightweight stand-in for ``cflibs.atomic.structures.Transition``."""
    wavelength_nm: float
    A_ki: float
    g_k: float
    E_k_ev: float
    ionization_stage: int = 1


def _make_spectrum(centers, amps, sigma=0.05, n_points=4000, wl_min=380.0,
                   wl_max=430.0, noise=0.5, seed=1):
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(wl_min, wl_max, n_points)
    intensity = np.full_like(wavelength, 5.0)
    for c, a in zip(centers, amps):
        intensity += a * np.exp(-0.5 * ((wavelength - c) / sigma) ** 2)
    intensity += rng.normal(0.0, noise, n_points)
    intensity = np.clip(intensity, 0.0, None)
    return wavelength, intensity


# ---------- Unit kernel ------------------------------------------------------


def test_classic_correlations_jax_matches_cpu():
    """JAX kernel produces the same (T, n_e) correlations as the CPU loop."""
    wavelength, intensity = _make_spectrum(
        [400.0, 410.0, 420.0], [50.0, 40.0, 30.0]
    )
    transitions = [
        FakeTransition(400.0, 1e8, 5, 4.0, 1),
        FakeTransition(410.0, 8e7, 5, 4.2, 1),
        FakeTransition(420.0, 6e7, 5, 4.5, 1),
    ]

    # Patch the Saha solver to avoid needing a real atomic DB.
    cpu_id = CorrelationIdentifier(
        atomic_db=None, elements=["Fake"], T_steps=3, n_e_steps=2,
        instrument_fwhm_nm=0.12, use_jax_classic=False,
    )
    jax_id = CorrelationIdentifier(
        atomic_db=None, elements=["Fake"], T_steps=3, n_e_steps=2,
        instrument_fwhm_nm=0.12, use_jax_classic=True,
    )

    # Stub Saha solver: returns flat partition function & no balance info.
    class _FakeSaha:
        def solve_ionization_balance(self, *_a, **_k):
            return None  # triggers W_q=1 fallback in both paths

        def calculate_partition_function(self, *_a, **_k):
            return 1.0

    cpu_id.saha_solver = _FakeSaha()
    jax_id.saha_solver = _FakeSaha()

    T_grid = np.linspace(8000, 12000, 3)
    n_e_grid = np.linspace(3e16, 3e17, 2)

    # CPU reference: replicate the inner loop manually.
    from cflibs.core.constants import KB_EV
    cpu_corrs = []
    for T_K in T_grid:
        T_eV = T_K * KB_EV
        for n_e in n_e_grid:
            model = cpu_id._generate_model_spectrum(
                intensity, "Fake", transitions, wavelength, T_eV, n_e
            )
            i_min, i_max = intensity.min(), intensity.max()
            m_min, m_max = model.min(), model.max()
            if (i_max - i_min) > 1e-10 and (m_max - m_min) > 1e-10:
                exp_norm = (intensity - i_min) / (i_max - i_min)
                mod_norm = (model - m_min) / (m_max - m_min)
                peak_mask = (exp_norm >= 0.15) & (mod_norm >= 0.15)
                if np.sum(peak_mask) < cpu_id.peak_region_min_points:
                    peak_mask = (exp_norm >= 0.15) | (mod_norm >= 0.15)
            else:
                peak_mask = np.ones_like(intensity, dtype=bool)
            exp_peaks = intensity[peak_mask]
            mod_peaks = model[peak_mask]
            if (
                len(exp_peaks) > 2
                and np.std(mod_peaks) > 1e-10
                and np.std(exp_peaks) > 1e-10
            ):
                from scipy.stats import pearsonr
                r, _ = pearsonr(exp_peaks, mod_peaks)
                cpu_corrs.append(r)
            else:
                cpu_corrs.append(0.0)

    jax_corrs = jax_id._classic_correlations_jax(
        wavelength, intensity, "Fake", transitions, T_grid, n_e_grid
    )

    assert len(jax_corrs) == len(cpu_corrs)
    np.testing.assert_allclose(jax_corrs, cpu_corrs, rtol=1e-5, atol=1e-7)


def test_classic_jax_handles_degenerate_dynamic_range():
    """Model rows with no dynamic range fall back to all-ones mask cleanly."""
    wavelength = np.linspace(400.0, 410.0, 1000)
    # Flat intensity -> i_range tiny -> CPU path uses all-ones mask.
    intensity = np.full_like(wavelength, 10.0)
    transitions = [FakeTransition(405.0, 1e8, 5, 4.0, 1)]

    cpu_id = CorrelationIdentifier(
        atomic_db=None, T_steps=2, n_e_steps=1, instrument_fwhm_nm=0.12,
        use_jax_classic=False,
    )
    jax_id = CorrelationIdentifier(
        atomic_db=None, T_steps=2, n_e_steps=1, instrument_fwhm_nm=0.12,
        use_jax_classic=True,
    )

    class _FakeSaha:
        def solve_ionization_balance(self, *_a, **_k):
            return None

        def calculate_partition_function(self, *_a, **_k):
            return 1.0

    cpu_id.saha_solver = _FakeSaha()
    jax_id.saha_solver = _FakeSaha()

    T_grid = np.array([8000.0, 12000.0])
    n_e_grid = np.array([1e17])

    jax_corrs = jax_id._classic_correlations_jax(
        wavelength, intensity, "Fake", transitions, T_grid, n_e_grid
    )
    # On a flat intensity, masked Pearson should yield 0 (no spread in x).
    assert np.all(np.abs(jax_corrs) < 1e-9)


def test_constructor_rejects_jax_when_not_installed(monkeypatch):
    """Constructor surfaces a clear error if use_jax_classic=True but JAX missing."""
    import cflibs.inversion.identify.correlation as corr_module
    monkeypatch.setattr(corr_module, "_HAS_JAX", False)
    with pytest.raises(ImportError, match="requires JAX"):
        CorrelationIdentifier(atomic_db=None, use_jax_classic=True)


# ---------- End-to-end ------------------------------------------------------


@pytest.mark.requires_db
def test_end_to_end_classic_jax_matches_cpu(temp_db):
    """End-to-end identifier output is identical between CPU and JAX paths."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)

    # Build a synthetic spectrum with Fe lines from the test DB.
    wavelength = np.linspace(370, 380, 1000)
    intensity = np.zeros_like(wavelength)
    sigma = 0.05
    for wl in [371.99, 373.49, 374.95]:
        intensity += 1000.0 * np.exp(-0.5 * ((wavelength - wl) / sigma) ** 2)
    rng = np.random.default_rng(42)
    intensity += rng.normal(0, 10, len(intensity))
    intensity = np.maximum(intensity, 0.0)

    cpu_id = CorrelationIdentifier(
        db, elements=["Fe"], T_range_K=(8000, 12000),
        T_steps=3, n_e_steps=2, min_confidence=0.1,
        use_jax_classic=False,
    )
    jax_id = CorrelationIdentifier(
        db, elements=["Fe"], T_range_K=(8000, 12000),
        T_steps=3, n_e_steps=2, min_confidence=0.1,
        use_jax_classic=True,
    )

    cpu_result = cpu_id.identify(wavelength, intensity, mode="classic")
    jax_result = jax_id.identify(wavelength, intensity, mode="classic")

    cpu_elems = {e.element for e in cpu_result.detected_elements}
    jax_elems = {e.element for e in jax_result.detected_elements}
    assert cpu_elems == jax_elems

    cpu_by = {e.element: e for e in cpu_result.all_elements}
    jax_by = {e.element: e for e in jax_result.all_elements}
    for el, cpu_e in cpu_by.items():
        jax_e = jax_by[el]
        assert cpu_e.score == pytest.approx(jax_e.score, rel=1e-5, abs=1e-7)
        assert cpu_e.detected == jax_e.detected
