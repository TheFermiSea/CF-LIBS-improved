"""Measured n_e from the Saha-Boltzmann inter-stage offset (Issue 4).

The multi-element Saha-Boltzmann plot already stacks neutral + ion lines on one
shared-slope plane; the vertical offset between the two stages' intercepts
measures n_e (Aguilera & Aragon 2007). These tests synthesize clean neutral +
ion line observations at a KNOWN n_e and verify the offset inversion recovers
it, while the Earth-STP pressure balance is off by its usual (~decade) factor.

The estimator (``_estimate_ne_from_sb_offset``) is partition-function-free (the
U's cancel in the offset), so these tests need no atomic DB.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cflibs.core.constants import EV_TO_K, SAHA_CONST_CM3
from cflibs.inversion.common import LineObservation
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver


def _bare_solver(**attrs) -> IterativeCFLIBSSolver:
    solver = IterativeCFLIBSSolver.__new__(IterativeCFLIBSSolver)
    solver.aki_uncertainty_weighting = False
    solver.prefer_sb_offset_ne = True
    for k, v in attrs.items():
        setattr(solver, k, v)
    return solver


def _line(
    E_k: float,
    y_target: float,
    element: str,
    stage: int,
    wl: float = 400.0,
    g_k: float = 5.0,
    A_ki: float = 1e8,
    rel_unc: float = 0.05,
) -> LineObservation:
    """LineObservation whose y_value == y_target at upper energy E_k."""
    intensity = math.exp(y_target) * g_k * A_ki / wl
    return LineObservation(
        wavelength_nm=wl,
        intensity=intensity,
        intensity_uncertainty=rel_unc * intensity,
        element=element,
        ionization_stage=stage,
        E_k_ev=E_k,
        g_k=g_k,
        A_ki=A_ki,
    )


def _synth_two_stage(
    element, T_eV, n_e, ip, U_I, U_II, e_neutral, e_ion, ln_C=0.0, rng=None, noise=0.0
):
    """Forward-synthesize neutral + ion lines at (T, n_e).

    y_stage(E) = ln_C + ln(n_stage / U_stage) - E / T_eV, with
    n_II / n_I = S1 = (SAHA/n_e) T^1.5 (U_II/U_I) exp(-IP/T_eV) and n_I := 1.
    """
    S1 = (SAHA_CONST_CM3 / n_e) * (T_eV**1.5) * (U_II / U_I) * math.exp(-ip / T_eV)
    n_I, n_II = 1.0, S1
    obs = []
    for E in e_neutral:
        y = ln_C + math.log(n_I / U_I) - E / T_eV
        if noise and rng is not None:
            y += rng.normal(0.0, noise)
        obs.append(_line(E, y, element, 1))
    for E in e_ion:
        y = ln_C + math.log(n_II / U_II) - E / T_eV
        if noise and rng is not None:
            y += rng.normal(0.0, noise)
        obs.append(_line(E, y, element, 2))
    return obs


def test_sb_offset_recovers_known_ne_clean():
    """Noise-free synthetic -> the offset inversion recovers n_e near-exactly."""
    T_eV = 1.0
    n_e_true = 2e16
    ip = 6.82  # Fe-like first IP
    obs = _synth_two_stage(
        "Fe",
        T_eV,
        n_e_true,
        ip,
        U_I=30.0,
        U_II=45.0,
        e_neutral=[2.5, 3.2, 4.1, 4.9],
        e_ion=[5.5, 6.4, 7.8],
    )
    solver = _bare_solver()
    ne, n_el, scatter = solver._estimate_ne_from_sb_offset(
        {"Fe": obs}, T_K=T_eV * EV_TO_K, effective_ips={"Fe": ip}
    )
    assert n_el == 1
    assert ne == pytest.approx(n_e_true, rel=1e-6)


def test_sb_offset_partition_independent():
    """The recovered n_e is INDEPENDENT of the partition values (they cancel)."""
    T_eV = 1.0
    n_e_true = 5e15
    ip = 7.9
    common = dict(
        element="Al",
        T_eV=T_eV,
        n_e=n_e_true,
        ip=ip,
        e_neutral=[3.0, 3.8, 4.6],
        e_ion=[6.0, 7.1, 8.0],
    )
    obs_a = _synth_two_stage(U_I=10.0, U_II=12.0, **common)
    obs_b = _synth_two_stage(U_I=55.0, U_II=8.0, **common)  # wildly different U's
    solver = _bare_solver()
    ne_a = solver._estimate_ne_from_sb_offset({"Al": obs_a}, T_eV * EV_TO_K, {"Al": ip})[0]
    ne_b = solver._estimate_ne_from_sb_offset({"Al": obs_b}, T_eV * EV_TO_K, {"Al": ip})[0]
    assert ne_a == pytest.approx(n_e_true, rel=1e-6)
    assert ne_b == pytest.approx(n_e_true, rel=1e-6)


def test_sb_offset_within_30pct_under_noise():
    """With realistic per-line scatter the estimate stays well within ~30%."""
    T_eV = 1.0
    n_e_true = 2e16
    ip = 6.82
    rng = np.random.default_rng(7)
    recovered = []
    for _ in range(40):
        obs = _synth_two_stage(
            "Fe",
            T_eV,
            n_e_true,
            ip,
            U_I=30.0,
            U_II=45.0,
            e_neutral=[2.5, 3.2, 4.1, 4.9, 5.3],
            e_ion=[5.5, 6.4, 7.0, 7.8],
            rng=rng,
            noise=0.10,
        )
        ne = _bare_solver()._estimate_ne_from_sb_offset({"Fe": obs}, T_eV * EV_TO_K, {"Fe": ip})[0]
        recovered.append(ne)
    med = float(np.median(recovered))
    assert med == pytest.approx(n_e_true, rel=0.30)


def test_pressure_balance_is_off_by_a_large_factor():
    """The Earth-STP pressure balance mis-estimates the SAME plasma's n_e.

    Late-gate LIBS n_e ~ 2e16; the isobaric 1-atm balance imputes ~1e17-3e17
    (roughly a decade high), which is the very defect Issue 4 replaces.
    """
    from tests.inversion.solve.test_saha_ladder_symmetry import FakeDB

    T_K = 1.0 * EV_TO_K
    n_e_true = 2e16
    U = {1: 30.0, 2: 45.0, 3: 25.0}
    ip = {1: 6.82, 2: 16.2}
    db = FakeDB(ips=ip, us=U)
    solver = IterativeCFLIBSSolver.__new__(IterativeCFLIBSSolver)
    solver.atomic_db = db
    solver.include_stage_iii = True
    solver.apply_ipd = False
    solver.pressure_pa = 101325.0
    solver._evaluate_partition_function = lambda el, stage, T: db.partition(stage)

    ne_pb = solver._pressure_balance_ne(
        {"Fe": 1.0}, T_K, n_e_true, {"Fe": U[1]}, {"Fe": U[2]}, {"Fe": ip[1]}
    )
    # Pressure balance lands well above the true late-gate value.
    assert ne_pb > 5.0 * n_e_true


def test_update_ne_prefers_sb_offset_over_stark():
    """_update_ne_python returns ne_source='sb_offset' when both stages exist,
    even with a Stark diagnostic supplied (SB-offset outranks Stark)."""
    from cflibs.inversion.solve.iterative import StarkDiagnosticLine

    T_eV = 1.0
    n_e_true = 2e16
    ip = 6.82
    obs = _synth_two_stage(
        "Fe",
        T_eV,
        n_e_true,
        ip,
        U_I=30.0,
        U_II=45.0,
        e_neutral=[2.5, 3.2, 4.1],
        e_ion=[5.5, 6.4, 7.8],
    )
    solver = _bare_solver(strict=False)
    stark = StarkDiagnosticLine(
        measured_fwhm_nm=0.05,
        stark_w_ref_nm=0.01,
        stark_alpha=0.5,
        instrument_fwhm_nm=0.0,
        doppler_fwhm_nm=0.0,
    )
    ne, source = solver._update_ne_python(
        [stark],
        {"Fe": obs},
        T_eV * EV_TO_K,
        n_e_true,
        {"Fe": 1.0},
        {"Fe": 30.0},
        {"Fe": 45.0},
        {"Fe": ip},
    )
    assert source == "sb_offset"
    assert ne == pytest.approx(n_e_true, rel=1e-6)


def test_update_ne_falls_through_when_no_ion_lines():
    """Neutral-only observations -> no SB-offset; falls to the Stark path."""
    from cflibs.inversion.solve.iterative import StarkDiagnosticLine

    T_eV = 1.0
    obs = [_line(E, -E / T_eV + 10.0, "Fe", 1) for E in (2.0, 3.0, 4.0)]
    solver = _bare_solver(strict=False)
    stark = StarkDiagnosticLine(
        measured_fwhm_nm=0.05,
        stark_w_ref_nm=0.01,
        stark_alpha=0.5,
        instrument_fwhm_nm=0.0,
        doppler_fwhm_nm=0.0,
    )
    _, source = solver._update_ne_python(
        [stark],
        {"Fe": obs},
        T_eV * EV_TO_K,
        2e16,
        {"Fe": 1.0},
        {"Fe": 30.0},
        {"Fe": 45.0},
        {"Fe": 6.82},
    )
    assert source == "stark"
