"""
Parity tests for IterativeCFLIBSSolverJax vs IterativeCFLIBSSolver.

Verifies that the JAX-accelerated solver produces composition output within
``np.testing.assert_allclose(rtol=1e-3, atol=1e-4)`` of the numpy solver on
representative multi-element fixtures (Aalto-style spectra, T~8000 K,
n_e~1e16 cm^-3).

The looser tolerance reflects the iterative nature of the solver: tiny
float-precision differences in the per-iteration WLS reduction can shift the
convergence trajectory by one or two iterations even when the per-step
algebra matches to machine precision.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

# Force CPU JAX backend for deterministic tests (matches conftest default,
# but pin explicitly so this file works when run in isolation).
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

# Imports placed after the importorskip + jax.config.update above so the
# module short-circuits cleanly when jax is missing and configures x64 once
# before any solver module pulls in jax.numpy.
from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.atomic.structures import PartitionFunction  # noqa: E402
from cflibs.core.constants import EV_TO_K, SAHA_CONST_CM3  # noqa: E402
from cflibs.inversion.solve.iterative import (  # noqa: E402
    IterativeCFLIBSSolver,
    IterativeCFLIBSSolverJax,
    LineObservation,
)

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


@pytest.fixture
def mock_db():
    """Mirror tests/test_solver.py mock_db fixture exactly."""
    db = MagicMock(spec=AtomicDatabase)
    db.get_ionization_potential.return_value = 7.0  # eV
    coeffs_I = [3.2188, 0, 0, 0, 0]  # log(25)

    def _pf(el, sp):
        return PartitionFunction(
            element=el,
            ionization_stage=sp,
            coefficients=coeffs_I,
            t_min=1000,
            t_max=20000,
            source="test",
        )

    db.get_partition_coefficients.side_effect = _pf
    return db


def _make_neutral_only_observations(T_eV: float, intercept: float = 10.0) -> list:
    """Generate noiseless single-element neutral-only Boltzmann data."""
    obs = []
    for E in [1.0, 2.0, 3.0]:
        y = -E / T_eV + intercept
        obs.append(
            LineObservation(
                wavelength_nm=500.0,
                intensity=np.exp(y),
                intensity_uncertainty=0.1,
                element="A",
                ionization_stage=1,
                E_k_ev=E,
                g_k=1,
                A_ki=1e8,
            )
        )
    return obs


def _make_two_element_observations(T_eV: float = 1.0) -> list:
    """Two-element noiseless data with equal concentrations."""
    obs = []
    for E in [1, 3, 5]:
        y = -E / T_eV + 10.0
        obs.append(LineObservation(500, np.exp(y), 0.1, "A", 1, E, 1, 1e8))
    for E in [1, 3, 5]:
        y = -E / T_eV + 10.0
        obs.append(LineObservation(500, np.exp(y), 0.1, "B", 1, E, 1, 1e8))
    return obs


def _make_aalto_style_multi_element_observations() -> list:
    """
    Multi-element fixture roughly matching Aalto-style steel/alloy spectra.

    Uses T = 8000 K, n_e = 1e16 cm^-3, and a mixture of neutral + ionic
    lines spanning multiple elements with realistic intensity ratios.
    """
    T_K = 8000.0
    T_eV = T_K / EV_TO_K  # ~0.689 eV
    n_e = 1.0e16
    ip = 7.0  # mock_db returns IP=7 for every element
    saha_offset = np.log((SAHA_CONST_CM3 / n_e) * (T_eV**1.5))

    # Three "elements" with different intercepts (~ different concentrations
    # via the closure equation: rel_C_s = U_s * exp(intercept))
    intercepts_true = {"Fe": 10.0, "Ni": 9.5, "Cr": 9.0}
    obs = []
    rng = np.random.default_rng(20260507)
    for el, intercept in intercepts_true.items():
        # Neutral lines: various E_k
        for E_k in [1.0, 2.5, 4.0, 5.5]:
            y = intercept - E_k / T_eV
            # Add 0.5% noise on intensity to simulate real fixtures
            intensity = np.exp(y) * (1.0 + rng.normal(0.0, 0.005))
            wavelength_nm = 500.0
            obs.append(
                LineObservation(
                    wavelength_nm=wavelength_nm,
                    intensity=intensity / wavelength_nm,
                    intensity_uncertainty=max(intensity * 0.005 / wavelength_nm, 1e-8),
                    element=el,
                    ionization_stage=1,
                    E_k_ev=E_k,
                    g_k=1,
                    A_ki=1.0,
                )
            )
        # Ionic lines (stage 2): higher E_k effectively due to IP shift
        for E_k in [3.0, 4.0]:
            y = intercept + saha_offset - (ip + E_k) / T_eV
            intensity = np.exp(y) * (1.0 + rng.normal(0.0, 0.005))
            wavelength_nm = 500.0
            obs.append(
                LineObservation(
                    wavelength_nm=wavelength_nm,
                    intensity=intensity / wavelength_nm,
                    intensity_uncertainty=max(intensity * 0.005 / wavelength_nm, 1e-8),
                    element=el,
                    ionization_stage=2,
                    E_k_ev=E_k,
                    g_k=1,
                    A_ki=1.0,
                )
            )
    return obs


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_jax_solver_instantiable(mock_db):
    """Solver builds and reports its backend."""
    solver = IterativeCFLIBSSolverJax(mock_db, max_iterations=5)
    assert solver.backend in {"jax", "numpy_fallback"}
    if solver.backend == "jax":
        assert solver.jax_backend is not None


def test_jax_solver_runs_basic(mock_db):
    """JAX solver returns sensible output for a single-element neutral fit."""
    solver = IterativeCFLIBSSolverJax(mock_db, max_iterations=5)
    obs = _make_neutral_only_observations(T_eV=0.8617)  # 10 000 K
    res = solver.solve(obs)
    assert res.converged
    assert abs(res.temperature_K - 10000.0) < 500.0
    assert abs(res.concentrations["A"] - 1.0) < 1e-9


def test_jax_solver_quality_metrics_reports_backend(mock_db):
    solver = IterativeCFLIBSSolverJax(mock_db, max_iterations=5)
    obs = _make_neutral_only_observations(T_eV=0.8617)
    res = solver.solve(obs)
    assert "backend" in res.quality_metrics
    assert "jax_backend" in res.quality_metrics


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


def _solve_pair(mock_db, observations):
    # The lax/JAX backend does not run the self-absorption correction (B1),
    # so the numpy reference must disable it too for an apples-to-apples
    # iteration-backend parity comparison. (SA is exercised separately in
    # tests/inversion/solve/test_self_absorption_wiring.py.)
    solver_np = IterativeCFLIBSSolver(mock_db, max_iterations=20, apply_self_absorption=False)
    solver_jax = IterativeCFLIBSSolverJax(mock_db, max_iterations=20)
    res_np = solver_np.solve(observations)
    res_jax = solver_jax.solve(observations)
    return res_np, res_jax


def test_parity_neutral_only(mock_db):
    obs = _make_neutral_only_observations(T_eV=0.8617)
    res_np, res_jax = _solve_pair(mock_db, obs)
    assert res_np.converged and res_jax.converged
    np.testing.assert_allclose(res_jax.temperature_K, res_np.temperature_K, rtol=1e-3, atol=1.0)
    np.testing.assert_allclose(
        res_jax.electron_density_cm3, res_np.electron_density_cm3, rtol=1e-3, atol=0.0
    )
    for el in res_np.concentrations:
        np.testing.assert_allclose(
            res_jax.concentrations[el],
            res_np.concentrations[el],
            rtol=1e-3,
            atol=1e-4,
        )


def test_parity_two_element_equal_concentration(mock_db):
    obs = _make_two_element_observations(T_eV=1.0)
    res_np, res_jax = _solve_pair(mock_db, obs)
    assert sorted(res_np.concentrations.keys()) == sorted(res_jax.concentrations.keys())
    np.testing.assert_allclose(res_jax.temperature_K, res_np.temperature_K, rtol=1e-3, atol=2.0)
    np.testing.assert_allclose(res_jax.electron_density_cm3, res_np.electron_density_cm3, rtol=1e-3)
    for el in res_np.concentrations:
        np.testing.assert_allclose(
            res_jax.concentrations[el],
            res_np.concentrations[el],
            rtol=1e-3,
            atol=1e-4,
        )


def test_parity_aalto_multi_element(mock_db):
    """
    Real-plasma-style fixture: T~8000K, n_e~1e16, multi-element neutral+ionic.

    The numpy and JAX solvers must agree on composition output to
    ``rtol=1e-3, atol=1e-4`` -- this is the contract documented in the
    JAX solver docstring.
    """
    obs = _make_aalto_style_multi_element_observations()
    res_np, res_jax = _solve_pair(mock_db, obs)
    assert sorted(res_np.concentrations.keys()) == sorted(res_jax.concentrations.keys())
    # Allow a slightly looser temperature tolerance because the iteration
    # trajectory can differ by 1 step.
    np.testing.assert_allclose(res_jax.temperature_K, res_np.temperature_K, rtol=2e-3, atol=10.0)
    np.testing.assert_allclose(res_jax.electron_density_cm3, res_np.electron_density_cm3, rtol=2e-3)
    for el in res_np.concentrations:
        np.testing.assert_allclose(
            res_jax.concentrations[el],
            res_np.concentrations[el],
            rtol=1e-3,
            atol=1e-4,
        )


def test_parity_matrix_closure(mock_db):
    """Matrix-mode closure must produce parity composition output."""
    obs = _make_two_element_observations(T_eV=1.0)
    solver_np = IterativeCFLIBSSolver(mock_db, max_iterations=20)
    solver_jax = IterativeCFLIBSSolverJax(mock_db, max_iterations=20)
    res_np = solver_np.solve(obs, closure_mode="matrix", matrix_element="A", matrix_fraction=0.5)
    res_jax = solver_jax.solve(obs, closure_mode="matrix", matrix_element="A", matrix_fraction=0.5)
    for el in res_np.concentrations:
        np.testing.assert_allclose(
            res_jax.concentrations[el],
            res_np.concentrations[el],
            rtol=1e-3,
            atol=1e-4,
        )
