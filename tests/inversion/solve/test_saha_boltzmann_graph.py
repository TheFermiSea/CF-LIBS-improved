"""Pooled Saha-Boltzmann GRAPH intercept extraction (Aguilera & Aragon 2004).

These tests pin the productionized ``IterativeCFLIBSSolver._fit_saha_boltzmann_graph``
math, independent of any atomic database:

* The pooled global regression recovers a planted shared slope and per-element
  intercepts from clean synthetic Boltzmann data.
* Ionic (stage > 1) lines are shifted onto the neutral plane by the same Saha
  transform the per-element path uses (x += IP; y -= ln S), so a planted ion
  line lands on its element's neutral line and does not bias the intercept.
* The fit is UNWEIGHTED (the validated SB-graph property): a single
  artificially-bright line does NOT hijack its element's intercept, unlike the
  inverse-variance-weighted common-slope plane.
* The intercept it returns is q_s = ln(n_I / U_I), the quantity the closure step
  consumes -- so SB-graph + standard closure reproduces a planted composition.

The solver instance is built with ``__new__`` so no SQLite DB is needed; the fit
only touches the observation x/y values and the unweighted geometry.
"""

from __future__ import annotations

import math

import pytest

from cflibs.inversion.common import LineObservation
from cflibs.inversion.physics.closure import (
    ClosureEquation,
    OXIDE_OXYGEN_PER_CATION,
    default_oxide_stoichiometry,
)
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver
from cflibs.core.constants import SAHA_CONST_CM3, EV_TO_K


def _bare_solver() -> IterativeCFLIBSSolver:
    """An IterativeCFLIBSSolver with only the attributes _fit_saha_boltzmann_graph needs."""
    solver = IterativeCFLIBSSolver.__new__(IterativeCFLIBSSolver)
    solver.aki_uncertainty_weighting = False
    solver.boltzmann_weight_cap = 5.0  # ignored on the unweighted SB-graph path
    solver.saha_boltzmann_graph = True
    return solver


def _line_from_xy(
    x_ek: float,
    y_target: float,
    element: str,
    stage: int = 1,
    g_k: float = 5.0,
    A_ki: float = 1.0e8,
    wavelength_nm: float = 400.0,
    rel_unc: float = 0.1,
) -> LineObservation:
    """Build a neutral-plane LineObservation whose y_value == y_target at E_k=x_ek.

    y_value = ln(I * lambda / (g * A)) => choose I so the log equals y_target.
    """
    intensity = math.exp(y_target) * g_k * A_ki / wavelength_nm
    return LineObservation(
        wavelength_nm=wavelength_nm,
        intensity=intensity,
        intensity_uncertainty=rel_unc * intensity,
        element=element,
        ionization_stage=stage,
        E_k_ev=x_ek,
        g_k=g_k,
        A_ki=A_ki,
    )


# ---------------------------------------------------------------------------
# 1. Pooled-regression math: recover a planted slope + per-element intercepts.
# ---------------------------------------------------------------------------


def test_pooled_regression_recovers_planted_slope_and_intercepts():
    slope = -1.3  # eV^-1  (=> T = -1/(slope*KB_EV))
    intercepts = {"A": 20.0, "B": 18.0, "C": 16.0}
    xs = [2.0, 3.0, 4.0, 5.0]

    obs_by_el: dict[str, list[LineObservation]] = {}
    for el, b in intercepts.items():
        obs_by_el[el] = [
            _line_from_xy(x, slope * x + b, element=el, wavelength_nm=400.0 + i)
            for i, x in enumerate(xs)
        ]

    solver = _bare_solver()
    # T, n_e only set the ion shift; with all-neutral lines they are irrelevant.
    fit = solver._fit_saha_boltzmann_graph(obs_by_el, T_K=10000.0, n_e=1e17, ips={})

    assert fit is not None
    assert fit.slope == pytest.approx(slope, abs=1e-9)
    for el, b in intercepts.items():
        assert fit.intercepts[el] == pytest.approx(b, abs=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Ion-line shift onto the neutral plane.
# ---------------------------------------------------------------------------


def test_ion_line_is_shifted_onto_neutral_plane():
    """A stage-2 line planted to satisfy the NEUTRAL line after the Saha shift
    must not perturb the recovered intercept/slope."""
    slope = -1.1
    b_A = 19.0
    ip_A = 7.9  # eV
    T_K = 10000.0
    n_e = 1.0e17
    T_eV = T_K / EV_TO_K
    ln_S = math.log((SAHA_CONST_CM3 / n_e) * (T_eV**1.5))

    # Neutral lines on element A's line: y = slope*E_k + b_A.
    neutrals = [_line_from_xy(x, slope * x + b_A, element="A") for x in (2.0, 3.0, 4.0)]

    # One ION line at E_k_ion. After the SB-graph shift it sits at
    #   x* = E_k_ion + ip_A,   y* = y_raw - ln_S
    # We want (x*, y*) to land exactly on the neutral line: y* = slope*x* + b_A.
    e_k_ion = 5.0
    x_star = e_k_ion + ip_A
    y_star = slope * x_star + b_A
    y_raw = y_star + ln_S  # invert the y shift so the stored y_value gives y_raw
    ion = _line_from_xy(e_k_ion, y_raw, element="A", stage=2, wavelength_nm=350.0)

    solver = _bare_solver()
    fit_neutral_only = solver._fit_saha_boltzmann_graph(
        {"A": neutrals}, T_K=T_K, n_e=n_e, ips={"A": ip_A}
    )
    fit_with_ion = solver._fit_saha_boltzmann_graph(
        {"A": neutrals + [ion]}, T_K=T_K, n_e=n_e, ips={"A": ip_A}
    )
    assert fit_neutral_only is not None and fit_with_ion is not None
    # The on-line ion point leaves slope and intercept unchanged (still R^2=1).
    assert fit_with_ion.slope == pytest.approx(fit_neutral_only.slope, abs=1e-6)
    assert fit_with_ion.intercepts["A"] == pytest.approx(b_A, abs=1e-6)
    assert fit_with_ion.r_squared == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. Unweighted: a single bright line does NOT hijack the intercept.
# ---------------------------------------------------------------------------


def test_sb_graph_is_unweighted_bright_line_does_not_hijack_intercept():
    """SB-graph uses an unweighted lstsq; an artificially-bright (tiny rel_unc)
    line off the true line must move the intercept the SAME as an equally-placed
    faint line would -- i.e. weighting is ignored."""
    slope = -1.0
    b = 18.0
    xs = [2.0, 3.0, 4.0, 5.0]
    base = [
        _line_from_xy(x, slope * x + b, element="A", wavelength_nm=400.0 + i)
        for i, x in enumerate(xs)
    ]

    # An OFF-line point at x=2.0 sitting +2 above the true line.
    off_y = slope * 2.0 + b + 2.0
    bright = _line_from_xy(
        2.0, off_y, element="A", wavelength_nm=420.0, rel_unc=1e-3
    )  # huge weight
    faint = _line_from_xy(2.0, off_y, element="A", wavelength_nm=420.0, rel_unc=1.0)  # tiny weight

    solver = _bare_solver()
    fit_bright = solver._fit_saha_boltzmann_graph({"A": base + [bright]}, 1e4, 1e17, ips={})
    fit_faint = solver._fit_saha_boltzmann_graph({"A": base + [faint]}, 1e4, 1e17, ips={})

    assert fit_bright is not None and fit_faint is not None
    # Identical because the fit ignores the (rel_unc-derived) weights.
    assert fit_bright.intercepts["A"] == pytest.approx(fit_faint.intercepts["A"], abs=1e-9)
    assert fit_bright.slope == pytest.approx(fit_faint.slope, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Identity recovery: planted n_I/U_I -> intercept -> closure -> composition.
# ---------------------------------------------------------------------------


def test_intercept_is_ln_n_over_U_and_closes_to_planted_composition():
    """Plant intercepts q_s = ln(n_I/U_I); SB-graph recovers them, and standard
    closure with mult=(1+S) and the same U recovers the planted number-density
    ratios (composition)."""
    slope = -1.2
    U = {"A": 25.0, "B": 10.0}
    n_I = {"A": 4.0e14, "B": 1.0e14}  # planted neutral number densities
    intercepts = {el: math.log(n_I[el] / U[el]) for el in U}
    xs = [2.0, 3.0, 4.0]

    obs_by_el = {
        el: [_line_from_xy(x, slope * x + intercepts[el], element=el) for x in xs] for el in U
    }
    solver = _bare_solver()
    fit = solver._fit_saha_boltzmann_graph(obs_by_el, 1e4, 1e17, ips={})
    assert fit is not None
    for el in U:
        assert fit.intercepts[el] == pytest.approx(intercepts[el], abs=1e-9)

    # Closure with mult=1 (no ionization) => rel_C = U*exp(q) = n_I; normalized.
    res = ClosureEquation.apply_standard(fit.intercepts, U)
    total = sum(n_I.values())
    for el in U:
        assert res.concentrations[el] == pytest.approx(n_I[el] / total, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Degenerate inputs return None (caller falls back).
# ---------------------------------------------------------------------------


def test_too_few_lines_returns_none():
    solver = _bare_solver()
    # 1 element, 1 line: under-determined (n_rows < 1 + E).
    obs = {"A": [_line_from_xy(2.0, 16.0, element="A")]}
    assert solver._fit_saha_boltzmann_graph(obs, 1e4, 1e17, ips={}) is None


def test_empty_obs_returns_none():
    solver = _bare_solver()
    assert solver._fit_saha_boltzmann_graph({}, 1e4, 1e17, ips={}) is None


# ---------------------------------------------------------------------------
# 6. default_oxide_stoichiometry helper.
# ---------------------------------------------------------------------------


def test_default_oxide_stoichiometry_full_and_filtered():
    full = default_oxide_stoichiometry()
    assert full == OXIDE_OXYGEN_PER_CATION
    assert full["Si"] == pytest.approx(2.0)
    assert full["Al"] == pytest.approx(1.5)
    assert full["Na"] == pytest.approx(0.5)

    # Filtering drops unknown elements (treated as elemental by oxide closure).
    filt = default_oxide_stoichiometry(["Si", "Fe", "Xx"])
    assert set(filt) == {"Si", "Fe"}
    assert filt["Fe"] == pytest.approx(1.5)


def test_oxide_closure_molar_oxygen_balance():
    """With factor = O-atoms-per-cation, sum_s (rel_C_s * factor_s) is the molar
    oxygen count; the returned cation fractions are the rel_C normalized by it."""
    intercepts = {"Si": 0.0, "Mg": 0.0}  # rel_C = U*exp(0) = U
    U = {"Si": 2.0, "Mg": 3.0}
    factors = default_oxide_stoichiometry(["Si", "Mg"])  # Si:2, Mg:1
    res = ClosureEquation.apply_oxide_mode(intercepts, U, oxide_stoichiometry=factors)
    # F = U_Si*2 + U_Mg*1 = 4 + 3 = 7; C_Si = 2/7, C_Mg = 3/7.
    assert res.concentrations["Si"] == pytest.approx(2.0 / 7.0)
    assert res.concentrations["Mg"] == pytest.approx(3.0 / 7.0)
