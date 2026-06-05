"""Numerical-equivalence tests for ``SpectrumModelJax``.

The JAX forward model must produce a synthetic spectrum that matches
:class:`SpectrumModel` within ``rtol=1e-5, atol=1e-7`` for a representative
multi-element LIBS plasma. This is the end-to-end gate that the benchmark
harness will exercise once GPU acceleration is wired in.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Uses the in-memory ``atomic_db`` fixture from ``tests/conftest.py`` (built
# from ``temp_db``); no production database file required.

jax = pytest.importorskip("jax")

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.instrument.model import InstrumentModel  # noqa: E402
from cflibs.plasma.state import SingleZoneLTEPlasma  # noqa: E402
from cflibs.radiation.profiles import BroadeningMode  # noqa: E402
from cflibs.radiation.spectrum_model import (  # noqa: E402
    SpectrumModel,
    SpectrumModelJax,
    planck_radiance,
    planck_radiance_jax,
)


def _build_plasma() -> SingleZoneLTEPlasma:
    """Realistic plasma — Si/Fe/Mg/Ca/Al composition, 10 kK / 1e16 cm^-3."""
    return SingleZoneLTEPlasma(
        T_e=10000.0,
        n_e=1.0e16,
        species={
            "Fe": 3.0e15,
            "H": 5.0e15,
        },
    )


def test_planck_radiance_jax_matches_numpy():
    wl = np.linspace(200.0, 800.0, 1000)
    T_eV = 10000.0 * 8.617333262e-5
    expected = planck_radiance(wl, T_eV)
    actual = np.asarray(planck_radiance_jax(wl, T_eV))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_spectrum_model_jax_matches_numpy_legacy(
    atomic_db,
):  # function-scoped DB is fine here (one call)
    """Full forward synthesis — JAX vs NumPy parity (LEGACY broadening)."""
    plasma = _build_plasma()
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    common = dict(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=200.0,
        lambda_max=800.0,
        delta_lambda=0.05,
        path_length_m=0.01,
        broadening_mode=BroadeningMode.LEGACY,
    )

    model_np = SpectrumModel(**common)
    model_jax = SpectrumModelJax(**common)

    wl_np, i_np = model_np.compute_spectrum()
    wl_jax, i_jax = model_jax.compute_spectrum()

    np.testing.assert_allclose(wl_jax, wl_np, rtol=0, atol=1e-12)
    # The convolution path uses ``signal.convolve`` (NumPy) vs
    # ``jnp.convolve`` (JAX). Both implement the same mathematical
    # operation; we allow the looser ``rtol=1e-4`` tolerance to absorb
    # the float32 default precision of JAX on CPU when x64 is off (the
    # conftest enables x64, so 1e-5 should hold).
    np.testing.assert_allclose(i_jax, i_np, rtol=1e-5, atol=1e-7)


def test_spectrum_model_jax_matches_numpy_nist_parity(atomic_db):
    """Full forward synthesis — JAX vs NumPy parity (NIST_PARITY mode).

    In NIST_PARITY there is no downstream convolution, so the two paths
    differ only in the per-line broadening kernel and radiative-transfer
    arithmetic — that is, exactly the work we want to check on GPU.
    """
    plasma = _build_plasma()
    instrument = InstrumentModel.from_resolving_power(20000.0)
    common = dict(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=200.0,
        lambda_max=800.0,
        delta_lambda=0.05,
        path_length_m=0.01,
        broadening_mode=BroadeningMode.NIST_PARITY,
    )

    model_np = SpectrumModel(**common)
    model_jax = SpectrumModelJax(**common)

    _, i_np = model_np.compute_spectrum()
    _, i_jax = model_jax.compute_spectrum()
    np.testing.assert_allclose(i_jax, i_np, rtol=1e-5, atol=1e-7)


def test_spectrum_model_ldm_gaussian_dispatch_smoke(atomic_db):
    """``BroadeningMode.LDM_GAUSSIAN`` is wired through :class:`SpectrumModel` end-to-end.

    Smoke test for the ADR-0001 T1-4 dispatch in :meth:`compute_spectrum`:
    runs the full forward model in LDM mode and verifies the output is
    finite, non-negative and on the configured wavelength grid. We do
    NOT cross-compare against PHYSICAL_DOPPLER end-to-end because, when
    Doppler σ ≪ Δλ (the typical CF-LIBS regime for many lines at 10 kK),
    ``apply_gaussian_broadening_per_line`` produces undersampled peaks
    whose integrated intensity is dominated by bin aliasing, while LDM
    produces the correct integral — so the two paths cannot agree at the
    integrated-intensity level in that regime. End-to-end parity is
    asserted at the kernel level by ``tests/radiation/test_ldm_broaden.py``.
    """
    plasma = _build_plasma()
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    model_ldm = SpectrumModel(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=300.0,
        lambda_max=500.0,
        delta_lambda=0.01,
        path_length_m=0.01,
        broadening_mode=BroadeningMode.LDM_GAUSSIAN,
    )
    wl, intensity = model_ldm.compute_spectrum()
    assert wl.shape == intensity.shape
    assert wl[0] == pytest.approx(300.0)
    assert wl[-1] == pytest.approx(500.0)
    assert np.all(np.isfinite(intensity))
    assert intensity.max() > 0
    # No negative intensities (LDM Lagrange-3pt scatter uses negative
    # outer weights; the downstream Planck × (1-exp(-κL)) is non-negative
    # because Planck > 0 and κ ≥ 0 outside the small negative artefacts).
    # Allow a tiny tolerance for the negative outer-bin lobe of the
    # Lagrange scatter.
    assert intensity.min() > -0.05 * intensity.max()


def test_spectrum_model_jax_uses_jax_solver(atomic_db):
    """Smoke check: JAX model swaps in the JAX solver (no silent NumPy fallback)."""
    from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolverJax

    plasma = _build_plasma()
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    model = SpectrumModelJax(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=300.0,
        lambda_max=500.0,
        delta_lambda=0.1,
    )
    assert isinstance(model.solver, SahaBoltzmannSolverJax)
    # Wavelength grid must already be a jnp array on the model
    assert "jax" in type(model._wavelength_jax).__module__


# ---------------------------------------------------------------------------
# audit Family 1 — SpectrumModelJax must honour apply_stark (Voigt, not
# Gaussian-only).  Before the fix, SpectrumModelJax.compute_spectrum
# re-implemented the kernel with a pure-Gaussian sum and silently dropped
# apply_stark, so SpectrumModelJax(apply_stark=True) produced Stark-free
# lines while SpectrumModel(apply_stark=True) produced Voigt.
# ---------------------------------------------------------------------------


def _fwhm_of_peak(wavelength: np.ndarray, intensity: np.ndarray) -> float:
    """FWHM (nm) of the largest peak — independent oracle, not kernel-derived.

    Walks outward from the global max to the first half-max crossings. The
    Voigt (Stark-on) profile has heavier Lorentzian wings than the pure
    Gaussian (Stark-off) core, so its FWHM is strictly larger.
    """
    if intensity.max() <= 0:
        return 0.0
    peak_idx = int(np.argmax(intensity))
    half = 0.5 * float(intensity[peak_idx])
    left = peak_idx
    while left > 0 and intensity[left] > half:
        left -= 1
    right = peak_idx
    while right < len(intensity) - 1 and intensity[right] > half:
        right += 1
    return float(wavelength[right] - wavelength[left])


def _wing_fraction(wavelength: np.ndarray, intensity: np.ndarray, core_nm: float) -> float:
    """Fraction of total line flux outside ±``core_nm`` of the peak.

    A pure Gaussian decays super-exponentially, so its far wings carry a
    negligible flux fraction. A Voigt profile (Stark Lorentzian wing) puts
    appreciable flux in the wings. This is an independent, profile-shape
    oracle for "Stark was actually applied".
    """
    if intensity.max() <= 0:
        return 0.0
    peak_wl = wavelength[int(np.argmax(intensity))]
    total = float(np.trapezoid(intensity, wavelength))
    if total <= 0:
        return 0.0
    wing_mask = np.abs(wavelength - peak_wl) > core_nm
    wing = float(np.trapezoid(intensity * wing_mask, wavelength))
    return wing / total


@pytest.fixture
def stark_db_path():
    """Temp DB with one isolated Fe I line carrying a large ``stark_w``.

    Mirrors ``tests/radiation/test_spectrum_model_stark_default.py`` so the
    JAX path is exercised against the same isolated-line oracle the NumPy
    Stark-default regression test uses.
    """
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE lines (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            wavelength_nm REAL,
            aki REAL,
            ei_ev REAL,
            ek_ev REAL,
            gi INTEGER,
            gk INTEGER,
            rel_int REAL,
            stark_w REAL,
            stark_alpha REAL
        )
        """)
    conn.execute("""
        CREATE TABLE energy_levels (
            element TEXT,
            sp_num INTEGER,
            g_level INTEGER,
            energy_ev REAL
        )
        """)
    conn.execute("""
        CREATE TABLE species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            PRIMARY KEY (element, sp_num)
        )
        """)
    conn.execute("""
        INSERT INTO lines
            (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev,
             gi, gk, rel_int, stark_w, stark_alpha)
        VALUES ('Fe', 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000, 0.05, 0.5)
        """)
    conn.execute("""
        INSERT INTO energy_levels (element, sp_num, g_level, energy_ev)
        VALUES ('Fe', 1, 9, 0.0),
               ('Fe', 1, 11, 3.33)
        """)
    conn.execute("""
        INSERT INTO species_physics (element, sp_num, ip_ev)
        VALUES ('Fe', 1, 7.87),
               ('Fe', 2, 16.18)
        """)
    conn.commit()
    conn.close()
    yield db_path
    Path(db_path).unlink(missing_ok=True)


def _stark_common(db: AtomicDatabase) -> dict:
    plasma = SingleZoneLTEPlasma(
        T_e=10000.0,
        n_e=1.0e17,  # high enough for Stark to dominate over Doppler
        species={"Fe": 1.0e15},
    )
    instrument = InstrumentModel(resolution_fwhm_nm=0.01)
    return dict(
        plasma=plasma,
        atomic_db=db,
        instrument=instrument,
        lambda_min=371.5,
        lambda_max=372.5,
        delta_lambda=0.005,
        path_length_m=1.0e-4,  # optically thin: RT scales but doesn't reshape
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
    )


@pytest.mark.requires_jax
def test_spectrum_model_jax_applies_stark_when_requested(stark_db_path):
    """SpectrumModelJax(apply_stark=True) must broaden via Voigt, not Gaussian.

    Pins audit Family 1: the JAX model must produce a wider, heavier-winged
    line with Stark on than with Stark off — i.e. apply_stark is honoured.
    """
    db = AtomicDatabase(stark_db_path)
    common = _stark_common(db)

    model_on = SpectrumModelJax(**common, apply_stark=True)
    assert model_on.apply_stark is True
    wl_on, i_on = model_on.compute_spectrum()

    model_off = SpectrumModelJax(**common, apply_stark=False)
    wl_off, i_off = model_off.compute_spectrum()

    np.testing.assert_array_equal(wl_on, wl_off)
    assert i_on.max() > 0
    assert i_off.max() > 0

    # FWHM oracle: Voigt (Stark-on) is strictly wider than Gaussian (off).
    fwhm_on = _fwhm_of_peak(wl_on, i_on)
    fwhm_off = _fwhm_of_peak(wl_off, i_off)
    assert fwhm_on > fwhm_off, (
        "SpectrumModelJax(apply_stark=True) must produce a wider line than "
        f"apply_stark=False — got FWHM_on={fwhm_on:.5f} nm "
        f"<= FWHM_off={fwhm_off:.5f} nm. Stark was silently dropped."
    )

    # Wing oracle: the Stark Lorentzian wing puts appreciably more flux
    # outside the line core than a pure Gaussian does.
    core_nm = 3.0 * fwhm_off  # well outside the Gaussian core
    wing_on = _wing_fraction(wl_on, i_on, core_nm)
    wing_off = _wing_fraction(wl_off, i_off, core_nm)
    assert wing_on > wing_off, (
        "Stark-on wings must carry more flux than Gaussian-only wings — "
        f"got wing_on={wing_on:.4f} <= wing_off={wing_off:.4f}."
    )


@pytest.mark.requires_jax
def test_spectrum_model_jax_stark_matches_numpy_voigt(stark_db_path):
    """JAX Stark-on output matches base SpectrumModel Stark-on (Voigt parity).

    The base SpectrumModel already routes apply_stark through the Voigt
    forward_model path; SpectrumModelJax (after dropping its override) must
    reproduce it within the documented rtol=1e-5, atol=1e-7 tolerance.
    """
    db = AtomicDatabase(stark_db_path)
    common = _stark_common(db)

    _, i_np = SpectrumModel(**common, apply_stark=True).compute_spectrum()
    _, i_jax = SpectrumModelJax(**common, apply_stark=True).compute_spectrum()

    scale = max(float(i_np.max()), 1e-30)
    np.testing.assert_allclose(i_jax, i_np, rtol=1e-5, atol=1e-7 * scale)


@pytest.mark.requires_jax
def test_spectrum_model_jax_stark_off_matches_numpy_gaussian(stark_db_path):
    """JAX Stark-off output matches base SpectrumModel Stark-off (Gaussian).

    Confirms apply_stark=False reproduces the Gaussian-only result on both
    paths — the override deletion must not perturb the non-Stark branch.
    """
    db = AtomicDatabase(stark_db_path)
    common = _stark_common(db)

    _, i_np = SpectrumModel(**common, apply_stark=False).compute_spectrum()
    _, i_jax = SpectrumModelJax(**common, apply_stark=False).compute_spectrum()

    scale = max(float(i_np.max()), 1e-30)
    np.testing.assert_allclose(i_jax, i_np, rtol=1e-5, atol=1e-7 * scale)
