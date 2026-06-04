"""
Composition-tied regression tests for the self-absorption correction (defect
**B1**) wired into :class:`IterativeCFLIBSSolver`.

Background
----------
``SelfAbsorptionCorrector`` (Bulajic et al. 2002, *Spectrochim. Acta B* 57,
339, doi:10.1016/S0584-8547(01)00398-6) was dead code: it had zero callers in
the production inversion loop. The 2026-05-27 physics audit flagged this as the
single highest-leverage composition fix (~one order of magnitude accuracy on
the self-absorbed major lines that dominate basalt). These tests pin the
*outer* recursion of the algorithm: on every iteration after the first the
solver recomputes the line-center optical depth ``tau`` from the current plasma
state (T, concentrations, partition functions) and divides the observed line
intensities by the curve-of-growth escape factor ``f(tau) = (1 - e^-tau)/tau``
*before* the Boltzmann/closure fit.

Validation strategy (the audit's John & Anoop 2023 Mg-Ca gate)
--------------------------------------------------------------
We forward-model a known 50:50 Mg/Ca plasma in the optically-thin limit, then
deliberately attenuate each line by its physical escape factor to synthesise a
*self-absorbed* observed spectrum. The strong low-lying Ca lines are more
self-absorbed than the Mg lines, so an uncorrected Boltzmann/closure inversion
under-counts Ca (its intercepts are pulled down). The test asserts that turning
the self-absorption correction ON:

1. actually fires (a non-zero optical depth is recorded), and
2. reduces the total composition error and moves the under-counted element
   (Ca) back toward its true 50% — i.e. the correction improves the
   north-star composition metric, not just some proxy.

The atomic database is mocked (constant U = 25, as in ``tests/test_solver.py``)
so the test is fast and DB-independent; the self-absorption physics uses only
the per-line atomic data carried on each ``LineObservation``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import PartitionFunction
from cflibs.core.constants import C_LIGHT, H_PLANCK_EV, KB_EV
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.physics.self_absorption import SelfAbsorptionCorrector, _escape_factor
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

# True plasma state for the synthetic Mg/Ca 50:50 forward model.
_T_K = 9000.0
_U = 25.0  # constant partition function (matches the mock DB)
_TRUE_NFRAC = {"Mg": 0.5, "Ca": 0.5}
# Heavy-particle column density used both to synthesise the self-absorption and
# (by default) inside the solver — they only need to be the same order of
# magnitude for the correction to be in the discriminating regime.
_COLUMN_DENSITY_CM3 = 1.0e16

# Six neutral lines (3 per element) spanning E_k for a usable Boltzmann lever
# arm. The low-E_i lines (Ca 430/487.8, Mg 383.8) land in the moderately
# self-absorbed regime (tau ~ 1-8); the high-E_k lines stay essentially thin.
# (element, wavelength_nm, A_ki [s^-1], g_k, E_k [eV])
_LINES = [
    ("Mg", 383.80, 1.0e8, 5, 5.95),
    ("Mg", 517.27, 1.2e7, 3, 5.108),
    ("Mg", 470.30, 1.6e7, 5, 7.18),
    ("Ca", 430.00, 6.0e7, 7, 4.78),
    ("Ca", 487.81, 1.0e8, 5, 4.78),
    ("Ca", 445.48, 8.7e7, 7, 5.49),
]


def _lower_level_energy_ev(wavelength_nm: float, E_k_ev: float) -> float:
    photon_ev = (H_PLANCK_EV * C_LIGHT) / (wavelength_nm * 1e-9)
    return max(0.0, E_k_ev - photon_ev)


def _make_mock_db() -> MagicMock:
    db = MagicMock(spec=AtomicDatabase)
    db.get_ionization_potential.return_value = 7.6  # eV (Mg/Ca-like)
    coeffs = [np.log(_U), 0.0, 0.0, 0.0, 0.0]  # log U = const -> U = 25
    db.get_partition_coefficients.side_effect = lambda el, sp: PartitionFunction(
        element=el,
        ionization_stage=sp,
        coefficients=coeffs,
        t_min=1000,
        t_max=20000,
        source="test",
    )
    return db


def _thin_intensity(n_frac: float, g_k: int, A_ki: float, E_k_ev: float, lam_nm: float) -> float:
    """Optically-thin line area: I ∝ n_frac · gA/λ · exp(-E_k/kT) / U."""
    kT = _T_K * KB_EV
    return n_frac * (g_k * A_ki / lam_nm) * np.exp(-E_k_ev / kT) / _U


@pytest.fixture
def self_absorbed_observations() -> list[LineObservation]:
    """Synthesise a self-absorbed observed spectrum of a 50:50 Mg/Ca plasma.

    Each line's optically-thin intensity is attenuated by its physical escape
    factor ``f(tau)`` (computed from the same plasma state the solver will see),
    so the strong low-E_i Ca lines are suppressed relative to the Mg lines.
    """
    # Estimator used purely to compute the *true* per-line optical depth for the
    # forward attenuation (uncapped, never masks).
    fwd = SelfAbsorptionCorrector(plasma_length_cm=0.1, mask_threshold=1.0e12)
    pf = {"Mg": _U, "Ca": _U}

    observations: list[LineObservation] = []
    for element, lam, A_ki, g_k, E_k in _LINES:
        i_thin = _thin_intensity(_TRUE_NFRAC[element], g_k, A_ki, E_k, lam)
        E_i = _lower_level_energy_ev(lam, E_k)
        thin_obs = LineObservation(lam, i_thin, i_thin * 0.02, element, 1, E_k, g_k, A_ki)
        tau = fwd._estimate_optical_depth(thin_obs, _T_K, _TRUE_NFRAC, _COLUMN_DENSITY_CM3, pf, E_i)
        i_obs = i_thin * _escape_factor(tau)
        observations.append(LineObservation(lam, i_obs, i_obs * 0.02, element, 1, E_k, g_k, A_ki))
    return observations


def _composition_error(result, truth=_TRUE_NFRAC) -> float:
    """Sum of absolute per-element deviations from the true number fractions."""
    return sum(abs(result.concentrations.get(el, 0.0) - frac) for el, frac in truth.items())


def _solve(observations, *, apply_self_absorption: bool):
    solver = IterativeCFLIBSSolver(
        _make_mock_db(),
        max_iterations=40,
        apply_self_absorption=apply_self_absorption,
        self_absorption_column_density_cm3=_COLUMN_DENSITY_CM3,
    )
    return solver.solve(observations)


def test_self_absorption_opt_in_default_off():
    """Self-absorption is opt-in (default off) and builds no corrector when off.

    The plasma-state optical-depth estimate cannot distinguish a thick line
    from a thin one, so on an optically-thin spectrum a default-on correction
    would over-boost the low-E_k lines (a false positive). It therefore stays
    opt-in; callers enable it explicitly for known optically-thick samples.
    """
    solver_default = IterativeCFLIBSSolver(_make_mock_db())
    assert solver_default.apply_self_absorption is False
    assert solver_default.self_absorption_corrector is None

    solver_on = IterativeCFLIBSSolver(_make_mock_db(), apply_self_absorption=True)
    assert solver_on.apply_self_absorption is True
    assert solver_on.self_absorption_corrector is not None


def test_self_absorption_correction_fires(self_absorbed_observations):
    """With self-absorbed input the correction records a non-trivial tau."""
    result = _solve(self_absorbed_observations, apply_self_absorption=True)
    assert result.quality_metrics["self_absorption_applied"] == 1.0
    # A strong low-E_i line in this fixture is genuinely thick (tau > 1).
    assert result.quality_metrics["self_absorption_max_tau"] > 1.0


def test_self_absorption_reduces_composition_error(self_absorbed_observations):
    """Self-absorption correction improves the Mg/Ca composition vs truth.

    This is the composition-tied (north-star) assertion: on a self-absorbed
    Mg/Ca 50:50 spectrum the corrected inversion must be closer to the true
    composition than the uncorrected one, and must move the under-counted Ca
    back up toward 50%.
    """
    res_off = _solve(self_absorbed_observations, apply_self_absorption=False)
    res_on = _solve(self_absorbed_observations, apply_self_absorption=True)

    err_off = _composition_error(res_off)
    err_on = _composition_error(res_on)

    # Sanity: the uncorrected inversion really does under-count the more
    # self-absorbed element (Ca), otherwise the test proves nothing.
    assert res_off.concentrations["Ca"] < 0.45, (
        "Fixture is not self-absorbed enough: uncorrected Ca should be biased "
        f"low, got {res_off.concentrations['Ca']:.3f}."
    )

    # The correction must reduce the total composition error...
    assert err_on < err_off, (
        f"Self-absorption correction did not improve composition: "
        f"err_off={err_off:.3f} err_on={err_on:.3f}."
    )
    # ...and specifically pull the under-counted Ca back toward truth.
    assert res_on.concentrations["Ca"] > res_off.concentrations["Ca"], (
        "Self-absorption correction should raise the under-counted Ca fraction "
        f"({res_off.concentrations['Ca']:.3f} -> {res_on.concentrations['Ca']:.3f})."
    )


def test_self_absorption_noop_on_thin_spectrum():
    """On an optically-thin plasma the correction must not change the result.

    Optical depth is a property of the absorbing column (n·L), not of the
    observed intensity, so "thin" means a *low column density*. With a tiny
    column density every tau ~ 0, the escape factor is ~1, and SA-on must match
    SA-off to high precision — guarding against the correction accidentally
    perturbing already-thin, well-behaved data.
    """
    kT = _T_K * KB_EV
    obs = []
    for element in ("Mg", "Ca"):
        for E_k in (2.0, 4.0, 6.0):
            y = -E_k / kT + 1.0
            intensity = float(np.exp(y))
            obs.append(LineObservation(500.0, intensity, intensity * 0.02, element, 1, E_k, 1, 1e8))

    # Thin plasma: a column density 1e6x below the self-absorbed regime.
    thin_column = 1.0e10
    solver_off = IterativeCFLIBSSolver(
        _make_mock_db(), max_iterations=40, apply_self_absorption=False
    )
    solver_on = IterativeCFLIBSSolver(
        _make_mock_db(),
        max_iterations=40,
        apply_self_absorption=True,
        self_absorption_column_density_cm3=thin_column,
    )
    res_off = solver_off.solve(obs)
    res_on = solver_on.solve(obs)

    assert res_on.quality_metrics["self_absorption_max_tau"] < 0.1
    for el in res_off.concentrations:
        assert res_on.concentrations[el] == pytest.approx(res_off.concentrations[el], abs=1e-3)
