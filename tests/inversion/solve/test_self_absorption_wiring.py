"""Composition-tied tests for the OBSERVABLE-GATED self-absorption wiring
(bead CF-LIBS-improved-0jvr) in :class:`IterativeCFLIBSSolver`.

Background
----------
The previous wiring recomputed the line-center optical depth from the
*recovered* composition on every solver iteration — a positive feedback loop
(over-attributed element -> bigger tau -> bigger intensity boost -> bigger
intercept -> more mass at closure; audit 2026-06-09, 02-inversion-solver.md
F4) that measurably worsened intercept inflation on real ChemCam BHVO-2.
That path was deleted. The replacement corrects the measured line list ONCE,
BEFORE the Boltzmann fit, using only observables (doublet intensity ratios,
Pace 2025); SA-suspect lines without observables are down-weighted, never
boosted.

Validation strategy
-------------------
We forward-model a known 50:50 Mg/Ca plasma in the optically-thin limit,
then attenuate a Ca doublet (two lines sharing an upper level) by the
physical escape factors with the model's tau link (tau_2 = tau_1 / rho).
The uncorrected inversion under-counts Ca; the observable-gated correction
must recover the doublet's thin intensities from the measured ratio alone
and move the composition back toward truth.

The atomic database is mocked (constant U = 25) so the test is fast and
DB-independent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import PartitionFunction
from cflibs.core.constants import KB_EV
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.physics.self_absorption import _escape_factor
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

# True plasma state for the synthetic Mg/Ca 50:50 forward model.
_T_K = 9000.0
_U = 25.0  # constant partition function (matches the mock DB)
_TRUE_NFRAC = {"Mg": 0.5, "Ca": 0.5}

# (element, wavelength_nm, A_ki [s^-1], g_k, E_k [eV])
# The two Ca 4.78 eV lines share an upper level -> usable doublet pair.
_LINES = [
    ("Mg", 383.80, 1.0e8, 5, 5.95),
    ("Mg", 517.27, 1.2e7, 3, 5.108),
    ("Mg", 470.30, 1.6e7, 5, 7.18),
    ("Ca", 430.00, 6.0e7, 5, 4.78),  # doublet member 1
    ("Ca", 487.81, 1.0e8, 5, 4.78),  # doublet member 2
    ("Ca", 445.48, 8.7e7, 7, 5.49),
]

#: Injected line-center optical depth of the shorter-wavelength Ca doublet
#: member. The longer-wavelength member receives tau_2 = tau_1 / rho with
#: rho = (g1 A1 lam1^3) / (g2 A2 lam2^3) — the same line-strength link the
#: doublet solver assumes, so recovery is exact in the noiseless limit.
_TAU_1 = 2.0


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


def _make_observations(attenuate_ca_doublet: bool) -> list[LineObservation]:
    g1, a1, lam1 = 5, 6.0e7, 430.00
    g2, a2, lam2 = 5, 1.0e8, 487.81
    rho = (g1 * a1 * lam1**3) / (g2 * a2 * lam2**3)
    tau_by_wl = {lam1: _TAU_1, lam2: _TAU_1 / rho}

    observations: list[LineObservation] = []
    for element, lam, A_ki, g_k, E_k in _LINES:
        i_thin = _thin_intensity(_TRUE_NFRAC[element], g_k, A_ki, E_k, lam)
        i_obs = i_thin
        if attenuate_ca_doublet and lam in tau_by_wl:
            i_obs = i_thin * _escape_factor(tau_by_wl[lam])
        observations.append(LineObservation(lam, i_obs, i_obs * 0.02, element, 1, E_k, g_k, A_ki))
    return observations


def _composition_error(result, truth=_TRUE_NFRAC) -> float:
    return sum(abs(result.concentrations.get(el, 0.0) - frac) for el, frac in truth.items())


def _solve(observations, *, apply_self_absorption):
    solver = IterativeCFLIBSSolver(
        _make_mock_db(),
        max_iterations=40,
        apply_self_absorption=apply_self_absorption,
    )
    return solver.solve(observations)


# ---------------------------------------------------------------------------
# knob semantics
# ---------------------------------------------------------------------------


def test_self_absorption_mode_knob():
    """apply_self_absorption accepts 'off'|'observable' (+ legacy booleans)."""
    db = _make_mock_db()

    default = IterativeCFLIBSSolver(db)
    assert default.self_absorption_mode == "off"
    assert default.apply_self_absorption is False
    assert default.self_absorption_corrector is None

    for value in (True, "observable"):
        on = IterativeCFLIBSSolver(db, apply_self_absorption=value)
        assert on.self_absorption_mode == "observable"
        assert on.apply_self_absorption is True
        assert on.self_absorption_corrector is not None

    with pytest.raises(ValueError, match="apply_self_absorption"):
        IterativeCFLIBSSolver(db, apply_self_absorption="feedback")


def test_mode_off_bit_identical_to_default():
    """SA mode 'off' must be bit-identical to the default solver behaviour."""
    obs = _make_observations(attenuate_ca_doublet=True)
    res_default = _solve(obs, apply_self_absorption=False)
    res_off = _solve(obs, apply_self_absorption="off")

    assert res_off.temperature_K == res_default.temperature_K
    assert res_off.electron_density_cm3 == res_default.electron_density_cm3
    assert res_off.converged == res_default.converged
    assert res_off.iterations == res_default.iterations
    assert res_off.concentrations == res_default.concentrations
    assert res_off.quality_metrics == res_default.quality_metrics
    # Off mode reports zeroed SA diagnostics.
    assert res_off.quality_metrics["self_absorption_applied"] == 0.0
    assert res_off.quality_metrics["self_absorption_max_tau"] == 0.0
    assert res_off.quality_metrics["max_tau_estimate"] == 0.0
    assert res_off.quality_metrics["n_lines_sa_corrected"] == 0.0
    assert res_off.quality_metrics["n_lines_sa_suspect"] == 0.0


def test_observable_mode_noop_without_observables():
    """With no doublets / SA-risk lines, observable mode == off bit-identical.

    The thin fixture's lines have distinct upper levels (no pairs) and high
    lower levels (no SA-risk signature), so the pre-fit correction must be a
    pure pass-through — guarding against the correction perturbing
    well-behaved data (the old composition-fed path failed exactly this).
    """
    obs = _make_observations(attenuate_ca_doublet=False)
    # Remove one Ca doublet member so no same-upper-level pair exists.
    obs_no_doublet = [o for o in obs if o.wavelength_nm != 487.81]

    res_off = _solve(obs_no_doublet, apply_self_absorption="off")
    res_on = _solve(obs_no_doublet, apply_self_absorption="observable")

    assert res_on.temperature_K == res_off.temperature_K
    assert res_on.concentrations == res_off.concentrations
    assert res_on.quality_metrics["n_lines_sa_corrected"] == 0.0
    assert res_on.quality_metrics["n_lines_sa_suspect"] == 0.0


# ---------------------------------------------------------------------------
# composition-tied behaviour (north star)
# ---------------------------------------------------------------------------


def test_observable_correction_fires_and_records_metrics():
    obs = _make_observations(attenuate_ca_doublet=True)
    result = _solve(obs, apply_self_absorption="observable")

    assert result.quality_metrics["self_absorption_applied"] == pytest.approx(1.0)
    assert result.quality_metrics["n_lines_sa_corrected"] == pytest.approx(2.0)
    assert result.quality_metrics["n_lines_sa_suspect"] == pytest.approx(0.0)
    # max tau is the longer-wavelength doublet member's tau_2 = tau_1/rho.
    g1, a1, lam1 = 5, 6.0e7, 430.00
    g2, a2, lam2 = 5, 1.0e8, 487.81
    rho = (g1 * a1 * lam1**3) / (g2 * a2 * lam2**3)
    assert result.quality_metrics["max_tau_estimate"] == pytest.approx(_TAU_1 / rho, rel=1e-2)
    assert result.quality_metrics["self_absorption_max_tau"] == pytest.approx(
        result.quality_metrics["max_tau_estimate"]
    )


def test_observable_correction_reduces_composition_error():
    """The doublet-gated correction must move Mg/Ca back toward 50:50."""
    obs = _make_observations(attenuate_ca_doublet=True)
    res_off = _solve(obs, apply_self_absorption="off")
    res_on = _solve(obs, apply_self_absorption="observable")

    # Sanity: uncorrected inversion under-counts the self-absorbed Ca.
    assert res_off.concentrations["Ca"] < 0.45, (
        "Fixture is not self-absorbed enough: uncorrected Ca should be biased "
        f"low, got {res_off.concentrations['Ca']:.3f}."
    )

    err_off = _composition_error(res_off)
    err_on = _composition_error(res_on)
    assert err_on < err_off, (
        f"Observable SA correction did not improve composition: "
        f"err_off={err_off:.3f} err_on={err_on:.3f}."
    )
    assert res_on.concentrations["Ca"] > res_off.concentrations["Ca"]
    # Noiseless doublet -> near-exact recovery of the 50:50 truth.
    assert res_on.concentrations["Ca"] == pytest.approx(0.5, abs=0.05)


def test_observable_mode_never_boosts_unpaired_lines():
    """Attenuate a NON-doublet Ca line: no observable -> no boost.

    The deleted composition-fed path would have boosted it from the
    recovered composition. The observable path must leave it uncorrected
    (it carries no doublet partner and its E_i is above the resonance-risk
    threshold), demonstrating correction factors never come from the
    composition.
    """
    obs = []
    for element, lam, A_ki, g_k, E_k in _LINES:
        i_thin = _thin_intensity(_TRUE_NFRAC[element], g_k, A_ki, E_k, lam)
        i_obs = i_thin * (_escape_factor(2.0) if lam == 445.48 else 1.0)
        obs.append(LineObservation(lam, i_obs, i_obs * 0.02, element, 1, E_k, g_k, A_ki))
    # Remove a doublet member so the 4.78 eV pair cannot fire either.
    obs = [o for o in obs if o.wavelength_nm != 487.81]

    result = _solve(obs, apply_self_absorption="observable")
    assert result.quality_metrics["n_lines_sa_corrected"] == 0.0
