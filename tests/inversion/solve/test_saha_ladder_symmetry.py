"""Saha ladder forward/inverse symmetry (Issue 5, physics-first-principles audit).

The FORWARD ``SahaBoltzmannSolver`` populates three ionization stages
(``denom = 1 + S1 + S1*S2``). The historical inverse abundance multiplier
truncated at ``1 + S1`` and the charge balance assumed ``Z <= 1``. These tests
pin the corrected inverse ladder against the forward, using a tiny FakeDB (no
SQLite, no full atomic DB) so the algebra is exercised in isolation:

* ``_compute_abundance_multipliers`` returns ``1 + S1 + S1*S2`` and matches the
  forward's ``(n_I + n_II + n_III) / n_I`` density completion.
* the charge-balance ``eps_s`` in ``_pressure_balance_ne`` uses
  ``(S1 + 2*S1*S2)/(1 + S1 + S1*S2)`` (avg free electrons per atom).
* ``include_stage_iii=False`` recovers the legacy ``1 + S1`` two-stage inverse.
* an element whose DB lacks ``IP_II``/``U_III`` silently degrades to two stages.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.core.constants import EV_TO_K, SAHA_CONST_CM3
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver


class FakeDB:
    """Minimal atomic-data stub: constant IPs and partition functions per stage.

    ``ips`` maps stage -> IP (eV) (stage 1 = neutral->I->II boundary, i.e. the
    first ionization potential; stage 2 = second ionization potential). ``us``
    maps stage -> U(T) (stage 1 = U_I, 2 = U_II, 3 = U_III). A ``None`` value
    models a DB gap (returns None from get_ionization_potential).
    """

    def __init__(self, ips: dict, us: dict):
        self._ips = ips
        self._us = us

    def get_ionization_potential(self, element: str, stage: int):
        return self._ips.get(stage)

    # The solver evaluates partition functions through _evaluate_partition_function,
    # which we monkeypatch on the instance to hit these values.
    def partition(self, stage: int) -> float:
        return self._us.get(stage, 1.0)


def _make_solver(db: FakeDB, *, include_stage_iii: bool = True) -> IterativeCFLIBSSolver:
    solver = IterativeCFLIBSSolver.__new__(IterativeCFLIBSSolver)
    solver.atomic_db = db
    solver.include_stage_iii = include_stage_iii
    solver.apply_ipd = False
    solver.pressure_pa = 101325.0
    # _evaluate_partition_function -> FakeDB.partition(stage)
    solver._evaluate_partition_function = lambda el, stage, T_K: db.partition(stage)
    return solver


def _forward_stage_densities(T_eV: float, n_e: float, U, ip):
    """Reference three-stage forward balance (mirrors SahaBoltzmannSolver)."""
    S1 = (SAHA_CONST_CM3 / n_e) * (T_eV**1.5) * (U[2] / U[1]) * np.exp(-ip[1] / T_eV)
    S2 = (SAHA_CONST_CM3 / n_e) * (T_eV**1.5) * (U[3] / U[2]) * np.exp(-ip[2] / T_eV)
    return S1, S2


def test_multiplier_matches_forward_three_stage_ladder():
    T_K = 1.2 * EV_TO_K  # hot core: stage III is non-negligible
    n_e = 2e16
    U = {1: 25.0, 2: 30.0, 3: 20.0}
    ip = {1: 6.77, 2: 16.5}  # Cr-like: IP_I=6.77, IP_II=16.5 eV
    db = FakeDB(ips=ip, us=U)
    solver = _make_solver(db)

    mult = solver._compute_abundance_multipliers(
        ["Cr"], T_K, n_e, {"Cr": U[1]}, {"Cr": U[2]}, {"Cr": ip[1]}
    )

    T_eV = T_K / EV_TO_K
    S1, S2 = _forward_stage_densities(T_eV, n_e, U, ip)
    expected = 1.0 + S1 + S1 * S2  # forward denom = (n_I+n_II+n_III)/n_I

    assert mult["Cr"] == pytest.approx(expected, rel=1e-10)
    # The stage-III term is a large fraction of the total here (hot core).
    f_iii = (S1 * S2) / expected
    assert f_iii > 0.15, f"expected a sizeable f_III at hot-core T, got {f_iii:.3f}"


def test_two_stage_flag_recovers_legacy_multiplier():
    T_K = 1.2 * EV_TO_K
    n_e = 2e16
    U = {1: 25.0, 2: 30.0, 3: 20.0}
    ip = {1: 6.77, 2: 16.5}
    db = FakeDB(ips=ip, us=U)

    solver3 = _make_solver(db, include_stage_iii=True)
    solver2 = _make_solver(db, include_stage_iii=False)

    m3 = solver3._compute_abundance_multipliers(
        ["Cr"], T_K, n_e, {"Cr": U[1]}, {"Cr": U[2]}, {"Cr": ip[1]}
    )["Cr"]
    m2 = solver2._compute_abundance_multipliers(
        ["Cr"], T_K, n_e, {"Cr": U[1]}, {"Cr": U[2]}, {"Cr": ip[1]}
    )["Cr"]

    T_eV = T_K / EV_TO_K
    S1, S2 = _forward_stage_densities(T_eV, n_e, U, ip)
    assert m2 == pytest.approx(1.0 + S1, rel=1e-10)  # legacy two-stage
    assert m3 == pytest.approx(1.0 + S1 + S1 * S2, rel=1e-10)  # full ladder
    # The three-stage multiplier is strictly larger, and the excess is exactly
    # the neglected S1*S2 (stage-III) mass fraction the legacy inverse dropped.
    assert m3 > m2
    assert (m3 - m2) == pytest.approx(S1 * S2, rel=1e-10)


def test_missing_ip_ii_degrades_to_two_stage():
    """No IP_II in the DB -> S2 = 0, so the multiplier is the two-stage 1 + S1."""
    T_K = 1.2 * EV_TO_K
    n_e = 2e16
    U = {1: 25.0, 2: 30.0, 3: 20.0}
    ip = {1: 6.77, 2: None}  # DB gap on the second ionization potential
    db = FakeDB(ips=ip, us=U)
    solver = _make_solver(db, include_stage_iii=True)

    mult = solver._compute_abundance_multipliers(
        ["Cr"], T_K, n_e, {"Cr": U[1]}, {"Cr": U[2]}, {"Cr": ip[1]}
    )["Cr"]

    T_eV = T_K / EV_TO_K
    S1 = (SAHA_CONST_CM3 / n_e) * (T_eV**1.5) * (U[2] / U[1]) * np.exp(-ip[1] / T_eV)
    assert mult == pytest.approx(1.0 + S1, rel=1e-10)


def test_charge_balance_uses_three_stage_epsilon():
    """avg free electrons per atom = (S1 + 2 S1 S2)/(1 + S1 + S1 S2)."""
    T_K = 1.2 * EV_TO_K
    n_e = 2e16
    U = {1: 25.0, 2: 30.0, 3: 20.0}
    ip = {1: 6.77, 2: 16.5}
    db = FakeDB(ips=ip, us=U)
    solver = _make_solver(db)
    # partition_funcs / partition_funcs_II lookups also route through the DB
    # helper lookup_partition_function; pass explicit dicts so no DB call is made.
    ne_out = solver._pressure_balance_ne(
        {"Cr": 1.0}, T_K, n_e, {"Cr": U[1]}, {"Cr": U[2]}, {"Cr": ip[1]}
    )

    T_eV = T_K / EV_TO_K
    S1, S2 = _forward_stage_densities(T_eV, n_e, U, ip)
    ladder = 1.0 + S1 + S1 * S2
    eps = (S1 + 2.0 * S1 * S2) / ladder
    # Reconstruct the expected n_e from the same isobaric relation.
    from cflibs.core.constants import KB

    n_tot = solver.pressure_pa / (KB * T_K * (1.0 + eps))
    expected = eps * n_tot * 1e-6
    assert ne_out == pytest.approx(expected, rel=1e-10)

    # Sanity: a two-stage eps would be strictly smaller (ignores the doubly-
    # charged electrons), so the three-stage charge balance yields a larger n_e.
    eps2 = S1 / (1.0 + S1)
    assert eps > eps2
