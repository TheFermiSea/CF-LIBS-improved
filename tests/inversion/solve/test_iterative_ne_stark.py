"""Tests for the Stark-broadening n_e diagnostic + silent-failure gates.

Audit Family 12 (supersedes PR #220). Covers:

1. Stark-width inversion recovers a known n_e within ~15% (independent oracle).
2. With no Stark line supplied, the 1-atm pressure-balance FALLBACK fires and
   emits a warning; ``ne_from_stark`` is reported 0.0.
3. A degenerate composition (one element > 0.8 with >1 element) or a non-physical
   (positive/near-zero-slope) Boltzmann fit yields ``converged=False`` and
   ``closure_degenerate``/``boltzmann_degenerate`` quality flags set.

The expected values are derived from INDEPENDENT oracles (the closed-form Stark
law re-implemented in the test, the Olivero-Longbothum Voigt FWHM), never from
the production code under test.
"""

import logging

import numpy as np
import pytest
from unittest.mock import MagicMock

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import PartitionFunction
from cflibs.inversion.solve.iterative import (
    IterativeCFLIBSSolver,
    LineObservation,
    StarkDiagnosticLine,
)
from cflibs.radiation.stark import (
    REF_NE,
    REF_T_K,
    estimate_ne_from_stark,
    deconvolve_stark_fwhm,
)


@pytest.fixture
def mock_db():
    db = MagicMock(spec=AtomicDatabase)
    db.get_ionization_potential.return_value = 7.0  # eV
    coeffs_I = [3.2188, 0, 0, 0, 0]  # log(25)
    db.get_partition_coefficients.side_effect = lambda el, sp: PartitionFunction(
        element=el,
        ionization_stage=sp,
        coefficients=coeffs_I,
        t_min=1000,
        t_max=20000,
        source="test",
    )
    return db


# Independent oracle constants (NOT imported from the code under test).
_OL_A = 0.5346  # Olivero & Longbothum (1977)
_OL_B = 0.2166


def _voigt_fwhm_oracle(f_lorentz: float, f_gauss: float) -> float:
    """Olivero-Longbothum (1977) Voigt FWHM from Lorentz + Gauss FWHMs."""
    return _OL_A * f_lorentz + np.sqrt(_OL_B * f_lorentz**2 + f_gauss**2)


def _stark_fwhm_oracle(n_e_cm3: float, T_K: float, w_ref: float, alpha: float) -> float:
    """Forward electron-impact Stark FWHM, re-derived independently."""
    T_eff = max(T_K, 1000.0)
    return w_ref * (n_e_cm3 / REF_NE) * (T_eff / REF_T_K) ** (-alpha)


# --------------------------------------------------------------------------- #
# 1. Stark-width inversion recovers a known n_e                                #
# --------------------------------------------------------------------------- #


def test_estimate_ne_from_stark_recovers_known_density_no_gaussian():
    """At the reference T with no instrument/Doppler broadening, the inverse of
    the Stark law must return the exact density used to forge the FWHM."""
    n_e_true = 5.0e17
    w_ref = 0.1  # nm, FWHM at REF_NE, REF_T_K
    alpha = 0.5
    # Independent forward oracle: 0.1 * (5e17/1e17) * (10000/10000)^-0.5 = 0.5 nm
    fwhm = _stark_fwhm_oracle(n_e_true, REF_T_K, w_ref, alpha)
    assert np.isclose(fwhm, 0.5)

    n_e_est = estimate_ne_from_stark(fwhm, REF_T_K, w_ref, stark_alpha=alpha)
    assert n_e_est is not None
    assert abs(n_e_est - n_e_true) / n_e_true < 0.01


def test_estimate_ne_from_stark_recovers_within_15pct_with_deconvolution():
    """A realistic measured (Voigt) FWHM with instrument+Doppler Gaussian
    broadening must invert back to n_e within ~15% after deconvolution."""
    n_e_true = 5.0e17
    T_K = REF_T_K
    w_ref = 0.1
    alpha = 0.5

    # True Lorentzian Stark FWHM from the independent forward oracle.
    f_stark = _stark_fwhm_oracle(n_e_true, T_K, w_ref, alpha)  # 0.5 nm
    f_inst = 0.05
    f_dopp = 0.03
    f_gauss = float(np.hypot(f_inst, f_dopp))
    # Independent Voigt-FWHM oracle for the "measured" width.
    f_measured = _voigt_fwhm_oracle(f_stark, f_gauss)

    # Deconvolution should recover the Lorentzian component closely.
    f_recovered = deconvolve_stark_fwhm(f_measured, f_inst, f_dopp)
    assert abs(f_recovered - f_stark) / f_stark < 0.05

    n_e_est = estimate_ne_from_stark(
        f_measured, T_K, w_ref, stark_alpha=alpha, instrument_fwhm_nm=f_inst, doppler_fwhm_nm=f_dopp
    )
    assert n_e_est is not None
    assert abs(n_e_est - n_e_true) / n_e_true < 0.15


def test_estimate_ne_from_stark_returns_none_without_reference_width():
    assert estimate_ne_from_stark(0.5, REF_T_K, None) is None
    assert estimate_ne_from_stark(0.5, REF_T_K, 0.0) is None


def test_estimate_ne_from_stark_returns_none_when_gaussian_dominates():
    """If instrument+Doppler already exceeds the measured width, there is no
    recoverable Stark component."""
    assert estimate_ne_from_stark(0.05, REF_T_K, 0.1, instrument_fwhm_nm=0.1) is None


def test_solver_uses_stark_diagnostic_as_primary_ne(mock_db):
    """A supplied Stark line drives the final n_e (within 15% of the true 5e17),
    and the quality_metrics record that Stark, not pressure balance, was used."""
    n_e_true = 5.0e17
    w_ref = 0.1
    alpha = 0.5
    fwhm = _stark_fwhm_oracle(n_e_true, REF_T_K, w_ref, alpha)  # 0.5 nm

    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)
    T_eV = 0.8617  # ~10000 K so the (T/Tref)^alpha correction is ~1
    obs = [
        LineObservation(500.0, np.exp(-E / T_eV + 10.0), 0.1, "A", 1, E, 1, 1e8)
        for E in (1.0, 2.0, 3.0, 4.0, 5.0)
    ]
    stark = StarkDiagnosticLine(measured_fwhm_nm=fwhm, stark_w_ref_nm=w_ref, stark_alpha=alpha)

    res = solver.solve(obs, stark_diagnostic=stark)

    assert res.quality_metrics["ne_from_stark"] == pytest.approx(1.0)
    assert abs(res.electron_density_cm3 - n_e_true) / n_e_true < 0.15
    # Sanity: a pure pressure balance at 1 atm / 10000 K would land far from
    # 5e17 (it is density-driven, not Stark-driven) — confirm we are nowhere
    # near the fallback regime by checking the Stark value dominates.


# --------------------------------------------------------------------------- #
# 2. Pressure-balance FALLBACK emits a warning                                #
# --------------------------------------------------------------------------- #


def test_pressure_balance_fallback_emits_warning_when_no_stark_line(mock_db, caplog):
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=3)
    T_eV = 1.0
    obs = [
        LineObservation(500.0, np.exp(-E / T_eV + 10.0), 0.1, "A", 1, E, 1, 1e8)
        for E in (1.0, 2.0, 3.0)
    ]
    with caplog.at_level(logging.WARNING):
        res = solver.solve(obs)  # no stark_diagnostic

    assert res.quality_metrics["ne_from_stark"] == pytest.approx(0.0)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "pressure-balance fallback" in msgs or "pressure balance" in msgs


# --------------------------------------------------------------------------- #
# 3. Silent-failure gates: degenerate composition / non-physical slope         #
# --------------------------------------------------------------------------- #


def test_positive_slope_sets_boltzmann_degenerate_and_unconverged(mock_db):
    """A non-physical (populations rise with E_k) Boltzmann plane must report
    converged=False with boltzmann_degenerate flagged in quality_metrics."""
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)
    obs = []
    for el in ("A", "B"):
        for E in (1.0, 2.0, 3.0, 4.0, 5.0):
            obs.append(LineObservation(500.0, np.exp(10.0 + 0.05 * E), 0.1, el, 1, E, 1, 1e8))

    res = solver.solve(obs)
    assert res.converged is False
    assert res.quality_metrics["boltzmann_degenerate"] == pytest.approx(1.0)
    assert res.quality_metrics["n_elements_fit"] == pytest.approx(2.0)


def test_degenerate_composition_flagged_and_unconverged(mock_db):
    """One element soaking >0.8 of the closure mass (with >1 element) yields
    converged=False and closure_degenerate=True, even with a clean slope.

    Independent oracle for the degeneracy threshold: validate_degeneracy uses
    0.8; we construct A's intercept far above B's so closure mass A >> 0.8."""
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)
    T_eV = 1.0
    obs = []
    # Element A: clean negative-slope plane with a high intercept (~13).
    for E in (1.0, 2.0, 3.0, 4.0, 5.0):
        obs.append(LineObservation(500.0, np.exp(-E / T_eV + 13.0), 0.1, "A", 1, E, 1, 1e8))
    # Element B: same clean slope but a much lower intercept (~5) so its closure
    # mass is exp(5-13) ~ 3e-4 of A's — A soaks >> 0.8.
    for E in (1.0, 2.0, 3.0, 4.0, 5.0):
        obs.append(LineObservation(500.0, np.exp(-E / T_eV + 5.0), 0.1, "B", 1, E, 1, 1e8))

    res = solver.solve(obs)

    # Independent check of the dominance the closure produced.
    assert max(res.concentrations.values()) > 0.8
    assert res.quality_metrics["closure_degenerate"] == pytest.approx(1.0)
    assert res.converged is False


def test_clean_two_element_fit_is_not_degenerate(mock_db):
    """Regression guard: a balanced, clean two-element fit must NOT be flagged
    degenerate and must converge (the gates are no-ops on good data)."""
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)
    T_eV = 1.0
    obs = []
    for el in ("A", "B"):
        for E in (1.0, 2.0, 3.0, 4.0, 5.0):
            obs.append(LineObservation(500.0, np.exp(-E / T_eV + 10.0), 0.1, el, 1, E, 1, 1e8))

    res = solver.solve(obs)
    assert res.quality_metrics["closure_degenerate"] == pytest.approx(0.0)
    assert res.quality_metrics["boltzmann_degenerate"] == pytest.approx(0.0)
    assert res.converged is True
