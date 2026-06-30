"""STRICT / no-fallback mode for the iterative CF-LIBS solver.

Fast unit tests (FakeDB, no real DB / no XLA) that assert:

(i)  strict OFF == current production behaviour on a healthy case (the silent
     fallbacks still fire: IP=15.0 eV default, isobaric pressure-balance n_e),
     and two strict-off solves are byte-identical.
(ii) strict ON refuses the relevant fallbacks: missing atomic data
     (MissingAtomicData), an unobserved n_e stage / no Stark diagnostic
     (UnobservedStage), a degenerate / no-fit Boltzmann plane (NonIdentifiable),
     and an unknown closure mode (SolverFailure).

The strict machinery lives in ``cflibs.inversion.common.strict``; the wiring is
in ``cflibs.inversion.solve.iterative``.
"""

from __future__ import annotations

import math
from typing import List, Optional

import pytest

from cflibs.core.constants import EV_TO_K
from cflibs.inversion.common.data_structures import LineObservation
from cflibs.inversion.common.strict import (
    MissingAtomicData,
    NonIdentifiable,
    SolverFailure,
    UnobservedStage,
)
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

T_TRUE_K = 10000.0
_T_EV = T_TRUE_K / EV_TO_K  # ~0.8617 eV


class _Provider:
    """Minimal partition-function provider with the ``.at(T)`` contract."""

    def __init__(self, value: float):
        self._value = float(value)

    def at(self, _T_K: float) -> float:
        return self._value


class FakeDB:
    """Tiny in-memory atomic-data stub.

    ``ip_value=None`` simulates an incomplete DB (no IP row) so the IP=15.0
    fallback site is exercised. ``partition_function_for`` short-circuits the
    full fallback ladder so no real DB/levels are needed.
    """

    def __init__(self, ip_value: Optional[float] = 7.9):
        self._ip = ip_value

    def get_ionization_potential(self, _element: str, _stage: int) -> Optional[float]:
        return self._ip

    def partition_function_for(self, _element: str, stage: int):
        return _Provider(20.0 if stage == 1 else 10.0)

    def get_transitions(self, _element: str, _stage: int):  # used by reliability annotation
        return []


def _line(element: str, E_k_ev: float, intercept: float) -> LineObservation:
    """Build a neutral line whose Boltzmann y sits exactly on the T_TRUE plane.

    y = ln(I*lambda/(g*A)) = intercept - E_k / kT  =>  I solved back out, so a
    pooled fit recovers slope = -1/kT (T = T_TRUE) with R^2 ~ 1.
    """
    g_k = 5
    A_ki = 1.0e7
    wavelength_nm = 400.0
    y_target = intercept - E_k_ev / _T_EV
    intensity = math.exp(y_target) * g_k * A_ki / wavelength_nm
    return LineObservation(
        wavelength_nm=wavelength_nm,
        intensity=intensity,
        intensity_uncertainty=0.03 * intensity,
        element=element,
        ionization_stage=1,
        E_k_ev=E_k_ev,
        g_k=g_k,
        A_ki=A_ki,
    )


def _healthy_obs() -> List[LineObservation]:
    obs: List[LineObservation] = []
    for e in (2.0, 3.0, 4.0, 5.0):
        obs.append(_line("Fe", e, intercept=10.0))
    for e in (2.5, 3.5, 4.5, 5.5):
        obs.append(_line("Ti", e, intercept=9.0))
    return obs


def _degenerate_obs() -> List[LineObservation]:
    # One line per element -> no element has >= 2 points -> fit returns None.
    return [_line("Fe", 3.0, intercept=10.0), _line("Ti", 3.0, intercept=9.0)]


def _solver(db: FakeDB, strict: bool) -> IterativeCFLIBSSolver:
    return IterativeCFLIBSSolver(
        atomic_db=db,
        max_iterations=12,
        use_lax_while_loop=False,
        strict=strict,
    )


# --------------------------------------------------------------------------- #
# (i) strict OFF == production behaviour                                       #
# --------------------------------------------------------------------------- #
def test_strict_off_uses_pressure_balance_and_is_repeatable():
    """No Stark -> pressure-balance n_e fallback fires (ne_from_stark False),
    no exception, and two strict-off solves are byte-identical."""
    obs = _healthy_obs()
    r1 = _solver(FakeDB(), strict=False).solve(obs, closure_mode="standard")
    r2 = _solver(FakeDB(), strict=False).solve(obs, closure_mode="standard")

    # Pressure-balance fallback was used (no Stark diagnostic supplied).
    assert r1.quality_metrics.get("ne_from_stark", 0.0) == 0.0
    assert r1.electron_density_cm3 > 0.0
    # Recovered T is in the physical window and near the planted truth.
    assert 2000.0 <= r1.temperature_K <= 50000.0
    assert r1.temperature_K == pytest.approx(T_TRUE_K, rel=0.05)

    # Byte-identical numerics across two strict-off runs.
    assert r1.temperature_K == r2.temperature_K
    assert r1.electron_density_cm3 == r2.electron_density_cm3
    assert r1.concentrations == r2.concentrations
    assert r1.converged == r2.converged
    # Default-path FAILED markers are inert.
    assert r1.failed is False
    assert r1.failure_reason is None


def test_strict_off_missing_ip_defaults_to_15ev_without_raising():
    """An incomplete DB (no IP) silently defaults IP=15.0 in production; the
    diagnostics record it but the solve still returns a result."""
    obs = _healthy_obs()
    result = _solver(FakeDB(ip_value=None), strict=False).solve(obs)
    assert result is not None
    # Visibility: the default substitution is recorded even when off.
    assert result.diagnostics is not None
    assert "Fe" in result.diagnostics["extra"].get("ip_defaulted", [])


# --------------------------------------------------------------------------- #
# (ii) strict ON refuses the fallbacks                                         #
# --------------------------------------------------------------------------- #
def test_strict_missing_ip_raises_missing_atomic_data():
    obs = _healthy_obs()
    with pytest.raises(MissingAtomicData):
        _solver(FakeDB(ip_value=None), strict=True).solve(obs)


def test_strict_no_stark_raises_unobserved_stage():
    """Healthy fit but no Stark diagnostic -> the imputed pressure-balance n_e
    is refused (physically there is no electron-density measurement)."""
    obs = _healthy_obs()
    with pytest.raises(UnobservedStage):
        _solver(FakeDB(), strict=True).solve(obs, closure_mode="standard")


def test_strict_degenerate_fit_raises_non_identifiable():
    obs = _degenerate_obs()
    with pytest.raises(NonIdentifiable):
        _solver(FakeDB(), strict=True).solve(obs)


def test_strict_unknown_closure_mode_raises():
    obs = _healthy_obs()
    with pytest.raises(SolverFailure):
        _solver(FakeDB(), strict=True).solve(obs, closure_mode="bogus_mode")


def test_env_var_seeds_strict(monkeypatch):
    """resolve_strict() reads CFLIBS_NO_FALLBACK when strict is not passed."""
    monkeypatch.setenv("CFLIBS_NO_FALLBACK", "1")
    solver = IterativeCFLIBSSolver(atomic_db=FakeDB(), use_lax_while_loop=False)
    assert solver.strict is True
    with pytest.raises(UnobservedStage):
        solver.solve(_healthy_obs())
