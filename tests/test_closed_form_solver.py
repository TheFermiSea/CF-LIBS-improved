"""Tests for the closed-form ILR-based CF-LIBS solver."""

import time

import numpy as np
import pytest
from unittest.mock import MagicMock

from cflibs.inversion.solve.closed_form import ClosedFormILRSolver, ClosedFormConfig
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import PartitionFunction
from cflibs.core.constants import EV_TO_K, SAHA_CONST_CM3


@pytest.fixture
def mock_db():
    db = MagicMock(spec=AtomicDatabase)
    db.get_ionization_potential.return_value = 7.0  # eV

    # log(25) ≈ 3.2188 → constant U(T) = 25 for all T
    coeffs = [3.2188, 0, 0, 0, 0]
    db.get_partition_coefficients.side_effect = lambda el, sp: PartitionFunction(
        element=el,
        ionization_stage=sp,
        coefficients=coeffs,
        t_min=1000,
        t_max=20000,
        source="test",
    )
    return db


def _make_obs(element, E_k, T_eV, intercept, ion_stage=1, wavelength=500.0):
    """Create a synthetic line on the Boltzmann plane.

    With g_k=1, A_ki=1.0, dividing intensity by wavelength gives
    y_value = ln(I*lam/(g*A)) = y exactly.
    """
    y = -E_k / T_eV + intercept
    intensity = np.exp(y) / wavelength
    return LineObservation(
        wavelength_nm=wavelength,
        intensity=intensity,
        intensity_uncertainty=max(intensity * 0.01, 1e-8),
        element=element,
        ionization_stage=ion_stage,
        E_k_ev=E_k,
        g_k=1,
        A_ki=1.0,
    )


# ------------------------------------------------------------------
# 1. Single element, neutral only — T recovery
# ------------------------------------------------------------------


def test_single_element_neutral_t_recovery(mock_db):
    T_eV = 1.0
    T_K_true = T_eV * EV_TO_K

    obs = [_make_obs("Fe", E, T_eV, 8.0) for E in [1.0, 2.0, 3.0, 4.0]]

    solver = ClosedFormILRSolver(mock_db, ClosedFormConfig(saha_passes=1))
    result = solver.solve(obs)

    assert result.converged
    assert abs(result.temperature_K - T_K_true) < 100.0
    assert result.concentrations["Fe"] == pytest.approx(1.0, abs=1e-10)


# ------------------------------------------------------------------
# 2. Multi-element, known compositions — exact recovery
# ------------------------------------------------------------------


def test_multi_element_equal_compositions(mock_db):
    """Three elements with identical intercepts → equal concentrations."""
    T_eV = 0.9
    T_K_true = T_eV * EV_TO_K
    intercept = 8.0

    obs = []
    for E in [1.0, 2.0, 3.0]:
        obs.append(_make_obs("A", E, T_eV, intercept))
    for E in [1.5, 2.5, 3.5]:
        obs.append(_make_obs("B", E, T_eV, intercept))
    for E in [2.0, 3.0, 4.0]:
        obs.append(_make_obs("C", E, T_eV, intercept))

    solver = ClosedFormILRSolver(mock_db, ClosedFormConfig(saha_passes=1))
    result = solver.solve(obs)

    assert result.converged
    assert abs(result.temperature_K - T_K_true) < 100.0
    for el in ["A", "B", "C"]:
        assert result.concentrations[el] == pytest.approx(1.0 / 3, abs=0.01)


def test_multi_element_unequal_compositions(mock_db):
    """Recover Fe=0.7, Cu=0.2, Al=0.1 from synthetic Boltzmann data."""
    T_eV = 1.0
    U_s = 25.0  # constant from mock_db
    target = {"Al": 0.1, "Cu": 0.2, "Fe": 0.7}

    # y_value = -E/T + ln(C_s) + beta - ln(U_s)
    # so y_adj = y + ln(U) = -E/T + ln(C_s) + beta
    beta = 10.0

    obs = []
    for E in [1.0, 2.0, 3.0, 4.0]:
        y = -E / T_eV + np.log(target["Fe"]) + beta - np.log(U_s)
        inten = np.exp(y) / 500.0
        obs.append(LineObservation(500.0, inten, max(inten * 0.01, 1e-8), "Fe", 1, E, 1, 1.0))
    for E in [1.5, 2.5, 3.5]:
        y = -E / T_eV + np.log(target["Cu"]) + beta - np.log(U_s)
        inten = np.exp(y) / 500.0
        obs.append(LineObservation(500.0, inten, max(inten * 0.01, 1e-8), "Cu", 1, E, 1, 1.0))
    for E in [2.0, 3.0]:
        y = -E / T_eV + np.log(target["Al"]) + beta - np.log(U_s)
        inten = np.exp(y) / 500.0
        obs.append(LineObservation(500.0, inten, max(inten * 0.01, 1e-8), "Al", 1, E, 1, 1.0))

    solver = ClosedFormILRSolver(mock_db, ClosedFormConfig(saha_passes=1))
    result = solver.solve(obs)

    assert result.converged
    for el, expected in target.items():
        assert result.concentrations[el] == pytest.approx(
            expected, abs=0.01
        ), f"{el}: got {result.concentrations[el]:.4f}, expected {expected}"


# ------------------------------------------------------------------
# 3. Mixed ionization stages — end-to-end with Saha correction
# ------------------------------------------------------------------


def test_mixed_ionization_stages(mock_db):
    T_eV = 1.0
    T_K = T_eV * EV_TO_K
    n_e = 1e17
    ip = 7.0

    saha_offset = np.log((SAHA_CONST_CM3 / n_e) * (T_eV**1.5))
    common_intercept = 8.0

    obs = []
    for E_k in [1.0, 2.0, 3.0]:
        y = common_intercept - E_k / T_eV
        inten = np.exp(y) / 500.0
        obs.append(LineObservation(500.0, inten, max(inten * 0.01, 1e-8), "A", 1, E_k, 1, 1.0))
    for E_k in [4.0, 5.0, 6.0]:
        y = common_intercept + saha_offset - (ip + E_k) / T_eV
        inten = np.exp(y) / 500.0
        obs.append(LineObservation(500.0, inten, max(inten * 0.01, 1e-8), "A", 2, E_k, 1, 1.0))

    solver = ClosedFormILRSolver(mock_db, ClosedFormConfig(saha_passes=2))
    result = solver.solve(obs, initial_ne_cm3=n_e)

    assert result.converged
    assert abs(result.temperature_K - T_K) < 500.0
    assert "A" in result.concentrations


# ------------------------------------------------------------------
# 4. Uncertainty propagation
# ------------------------------------------------------------------


def test_uncertainty_propagation(mock_db):
    """σ_T and σ_C should be finite and positive with noisy data."""
    T_eV = 1.0
    rng = np.random.default_rng(42)

    obs = []
    for E in [1.0, 2.0, 3.0, 4.0, 5.0]:
        y = -E / T_eV + 8.0 + rng.normal(0, 0.05)
        inten = np.exp(y) / 500.0
        obs.append(LineObservation(500.0, inten, inten * 0.05, "Fe", 1, E, 1, 1.0))
    for E in [1.5, 2.5, 3.5, 4.5]:
        y = -E / T_eV + 7.0 + rng.normal(0, 0.05)
        inten = np.exp(y) / 500.0
        obs.append(LineObservation(500.0, inten, inten * 0.05, "Cu", 1, E, 1, 1.0))

    solver = ClosedFormILRSolver(mock_db, ClosedFormConfig(saha_passes=1))
    result = solver.solve(obs)

    assert result.temperature_uncertainty_K > 0
    assert np.isfinite(result.temperature_uncertainty_K)
    for el in ["Fe", "Cu"]:
        assert el in result.concentration_uncertainties
        assert result.concentration_uncertainties[el] >= 0
        assert np.isfinite(result.concentration_uncertainties[el])


# ------------------------------------------------------------------
# 5. Equivalence with iterative solver
# ------------------------------------------------------------------


def test_equivalence_with_iterative_solver(mock_db):
    T_eV = 1.0
    intercept = 8.0

    obs = []
    for E in [1.0, 2.0, 3.0, 4.0]:
        obs.append(_make_obs("A", E, T_eV, intercept))
    for E in [1.5, 2.5, 3.5]:
        obs.append(_make_obs("B", E, T_eV, intercept))

    iterative = IterativeCFLIBSSolver(mock_db, max_iterations=20)
    cf = ClosedFormILRSolver(mock_db, ClosedFormConfig(saha_passes=1))

    res_iter = iterative.solve(obs)
    res_cf = cf.solve(obs)

    # Temperature within 5% or 500K
    tol = max(500.0, 0.05 * res_iter.temperature_K)
    assert abs(res_cf.temperature_K - res_iter.temperature_K) < tol

    # Compositions within 5%
    for el in ["A", "B"]:
        assert abs(res_cf.concentrations[el] - res_iter.concentrations[el]) < 0.05


# ------------------------------------------------------------------
# 6. Edge cases
# ------------------------------------------------------------------


def test_two_elements(mock_db):
    T_eV = 0.8
    obs = []
    for E in [1.0, 2.0, 3.0]:
        obs.append(_make_obs("Fe", E, T_eV, 8.0))
    for E in [1.5, 2.5]:
        obs.append(_make_obs("Cu", E, T_eV, 7.5))

    solver = ClosedFormILRSolver(mock_db, ClosedFormConfig(saha_passes=1))
    result = solver.solve(obs)

    assert result.converged
    assert len(result.concentrations) == 2
    total = sum(result.concentrations.values())
    assert total == pytest.approx(1.0, abs=1e-10)
    assert all(c > 0 for c in result.concentrations.values())


def test_single_element_single_ion_stage(mock_db):
    """Single element → concentration trivially 1.0."""
    T_eV = 1.0
    obs = [_make_obs("Fe", E, T_eV, 8.0) for E in [1.0, 3.0, 5.0]]

    solver = ClosedFormILRSolver(mock_db, ClosedFormConfig(saha_passes=1))
    result = solver.solve(obs)

    assert result.converged
    assert result.concentrations == {"Fe": 1.0}


def test_element_with_one_line(mock_db):
    """Element with 1 line still contributes to compositions."""
    T_eV = 1.0
    obs = [_make_obs("Fe", E, T_eV, 8.0) for E in [1.0, 2.0, 3.0]]
    obs.append(_make_obs("Cu", 2.0, T_eV, 8.0))  # only 1 line

    solver = ClosedFormILRSolver(mock_db, ClosedFormConfig(saha_passes=1))
    result = solver.solve(obs)

    assert result.converged
    assert "Cu" in result.concentrations
    assert sum(result.concentrations.values()) == pytest.approx(1.0, abs=1e-10)


def test_empty_observations(mock_db):
    solver = ClosedFormILRSolver(mock_db)
    result = solver.solve([])

    assert not result.converged
    assert result.concentrations == {}


# ------------------------------------------------------------------
# 7. ILR simplex round-trip — sum to 1, all positive
# ------------------------------------------------------------------


def test_ilr_simplex_properties(mock_db):
    """Compositions sum to 1 and are all positive for random data."""
    T_eV = 1.0
    rng = np.random.default_rng(123)

    elements = ["Al", "Cu", "Fe", "Mg", "Si"]
    obs = []
    for el in elements:
        intercept = rng.uniform(6, 9)
        for E in rng.uniform(1, 5, size=3):
            obs.append(_make_obs(el, float(E), T_eV, intercept))

    solver = ClosedFormILRSolver(mock_db, ClosedFormConfig(saha_passes=1))
    result = solver.solve(obs)

    total = sum(result.concentrations.values())
    assert total == pytest.approx(1.0, abs=1e-10)
    for el, c in result.concentrations.items():
        assert c > 0, f"{el} has non-positive concentration {c}"


# ------------------------------------------------------------------
# 8. Speed benchmark
# ------------------------------------------------------------------


@pytest.mark.slow
def test_speed_benchmark(mock_db):
    """Closed-form should complete without error; log relative speed."""
    T_eV = 1.0
    rng = np.random.default_rng(42)

    elements = ["Al", "Ca", "Cu", "Fe", "Mg", "Si"]
    obs = []
    for el in elements:
        intercept = rng.uniform(6, 9)
        n_lines = rng.integers(5, 9)
        for E in rng.uniform(1, 6, size=n_lines):
            obs.append(_make_obs(el, float(E), T_eV, intercept))

    iterative = IterativeCFLIBSSolver(mock_db, max_iterations=20)
    cf = ClosedFormILRSolver(mock_db, ClosedFormConfig(saha_passes=2))

    # Warm up
    iterative.solve(obs)
    cf.solve(obs)

    n_trials = 50

    t0 = time.perf_counter()
    for _ in range(n_trials):
        iterative.solve(obs)
    t_iter = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_trials):
        cf.solve(obs)
    t_cf = time.perf_counter() - t0

    speedup = t_iter / t_cf
    # Record but don't hard-assert; wall-clock varies in CI
    print(f"Closed-form speedup: {speedup:.1f}×  (iter={t_iter:.3f}s, cf={t_cf:.3f}s)")
