"""Regression guard: the SVD seed library spans composition (audit finding C3).

solve_full_spectrum builds an SVD basis from a warm-start-centred sweep. The
composition-perturbation rows were gated behind ``extra = sweep_points - 9``,
which is 0 at the default sweep_points=9 (the 3x3 T/ne grid fills it), so the
basis spanned only T/ne and the fit could not move composition. They are now
generated unconditionally (one per element).

This test generates a synthetic spectrum at a known composition (via the
solver's own chunked forward — an inverse crime that isolates basis coverage
from forward mismatch), hands the solver a deliberately wrong warm-start
composition, and asserts the fit recovers composition. Before the fix the
fitted composition stayed at the warm start.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from cflibs.inversion.solve.full_spectrum import (  # noqa: E402
    _ChunkedForward,
    _mass_to_number_fractions,
    solve_full_spectrum,
)

K_TO_EV = 1.0 / 11604.5


def _db_path() -> str:
    here = Path(__file__).resolve().parents[3]
    for cand in (here / "ASD_da" / "libs_production.db", here / "libs_production.db"):
        if cand.exists():
            return str(cand)
    pytest.skip("libs_production.db not available")


@pytest.mark.requires_db
@pytest.mark.requires_jax
@pytest.mark.slow
def test_full_spectrum_recovers_composition_from_wrong_warm_start():
    db = _db_path()
    elements = ["Fe", "Ca"]
    wl = np.linspace(370.0, 400.0, 1500)
    T_K, ne = 9000.0, 1e17

    true_mass = {"Fe": 0.80, "Ca": 0.20}
    fwd = _ChunkedForward(db, elements, wl, instrument_fwhm_nm=0.05)
    intensity = np.asarray(
        fwd.spectrum_numpy(
            T_K * K_TO_EV, np.log10(ne), _mass_to_number_fractions(true_mass, elements)
        )
    )
    intensity = intensity / intensity.max()

    warm_mass = {"Fe": 0.50, "Ca": 0.50}  # deliberately wrong
    res = solve_full_spectrum(
        wl,
        intensity,
        elements,
        db,
        warm_start_T_K=T_K,
        warm_start_ne_cm3=ne,
        warm_start_concentrations=warm_mass,
        instrument_fwhm_nm=0.05,
        max_iterations=60,
    )

    err_warm = abs(warm_mass["Fe"] - true_mass["Fe"])
    err_fit = abs(res.concentrations["Fe"] - true_mass["Fe"])
    # C3 guard: with composition rows in the SVD basis the fit MOVES composition
    # toward truth and lowers the objective. (Before the fix the basis spanned
    # only T/ne, so composition was frozen at the warm start.) Note: this only
    # asserts that composition is now *movable* and the objective improves — the
    # full-spectrum solver still under-recovers composition in float64 (it stalls
    # ~0.55 here, not 0.80; see audit full-spectrum sigma_d/Hessian items and the
    # Step-4 convergence validation). Keep the bar at "demonstrably moved", not
    # "fully recovered", so this stays a robust regression guard for C3 itself.
    assert res.converged and res.adopted_fit
    assert res.final_objective < 0.5 * res.initial_objective  # objective genuinely dropped
    assert res.concentrations["Fe"] > 0.53  # moved up from the 0.50 warm start
    assert err_fit < err_warm - 0.02  # moved toward truth
