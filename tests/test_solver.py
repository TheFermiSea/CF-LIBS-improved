"""
Tests for iterative solver.
"""

import pytest
from unittest.mock import MagicMock
import numpy as np
from cflibs.inversion.solver import IterativeCFLIBSSolver, LineObservation
from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import PartitionFunction
from cflibs.core.constants import EV_TO_K, SAHA_CONST_CM3


@pytest.fixture
def mock_db():
    db = MagicMock(spec=AtomicDatabase)
    # Setup standard IP
    db.get_ionization_potential.return_value = 7.0  # eV

    # Setup simple partition coeffs (constant)
    # log(25) = 3.2188
    coeffs_I = [3.2188, 0, 0, 0, 0]
    db.get_partition_coefficients.side_effect = lambda el, sp: PartitionFunction(
        element=el,
        ionization_stage=sp,
        coefficients=coeffs_I,
        t_min=1000,
        t_max=20000,
        source="test",
    )
    return db


def test_solver_basic(mock_db):
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=5)

    # Create synthetic data for T=10000K (0.86 eV)
    # Element A (Neutral)
    # y = -E/T + const
    T_eV = 0.8617

    obs = []
    # Neutral lines (Stage 1)
    for E in [1.0, 2.0, 3.0]:
        # y = -E/0.8617 + 10.0
        y = -E / T_eV + 10.0
        # Inverse: I = exp(y) * gA / lambda (simplification)
        # We set y_value directly via intensity manipulation
        # y = ln(I*lam/gA) => I = exp(y) if lam=gA=1
        intensity = np.exp(y)
        obs.append(
            LineObservation(
                wavelength_nm=500.0,
                intensity=intensity,
                intensity_uncertainty=0.1,
                element="A",
                ionization_stage=1,
                E_k_ev=E,
                g_k=1,
                A_ki=1e8,
            )
        )

    res = solver.solve(obs)

    assert res.converged
    assert abs(res.temperature_K - 10000.0) < 500.0
    assert "A" in res.concentrations
    assert res.concentrations["A"] == 1.0


def test_solver_multielement(mock_db):
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)

    # Elements A and B, equal concentration
    T_eV = 1.0  # ~11600 K

    obs = []
    # Element A
    for E in [1, 3, 5]:
        y = -E / T_eV + 10.0
        obs.append(LineObservation(500, np.exp(y), 0.1, "A", 1, E, 1, 1e8))

    # Element B
    for E in [1, 3, 5]:
        y = -E / T_eV + 10.0  # Same intercept -> same concentration if U is same
        obs.append(LineObservation(500, np.exp(y), 0.1, "B", 1, E, 1, 1e8))

    res = solver.solve(obs)

    assert abs(res.temperature_K - 11604.0) < 500.0
    assert abs(res.concentrations["A"] - 0.5) < 0.05
    assert abs(res.concentrations["B"] - 0.5) < 0.05


def test_solver_mixed_stage_transform_recovers_temperature(mock_db):
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)

    T_eV = 1.0
    T_K = T_eV * EV_TO_K
    n_e = 1.0e17
    ip = 7.0
    wavelength_nm = 500.0

    # The classic Saha-Boltzmann transform should subtract only the ionic
    # prefactor from y while carrying IP on the x-axis.
    saha_offset = np.log(2.0 * (SAHA_CONST_CM3 / n_e) * (T_eV**1.5))
    common_intercept = 8.0

    observations = []

    for E_k in [1.0, 2.0, 3.0]:
        y = common_intercept - E_k / T_eV
        intensity = np.exp(y) / wavelength_nm
        observations.append(
            LineObservation(
                wavelength_nm=wavelength_nm,
                intensity=intensity,
                intensity_uncertainty=max(intensity * 0.01, 1e-8),
                element="A",
                ionization_stage=1,
                E_k_ev=E_k,
                g_k=1,
                A_ki=1.0,
            )
        )

    for E_k in [4.0, 5.0, 6.0]:
        y = common_intercept + saha_offset - (ip + E_k) / T_eV
        intensity = np.exp(y) / wavelength_nm
        observations.append(
            LineObservation(
                wavelength_nm=wavelength_nm,
                intensity=intensity,
                intensity_uncertainty=max(intensity * 0.01, 1e-8),
                element="A",
                ionization_stage=2,
                E_k_ev=E_k,
                g_k=1,
                A_ki=1.0,
            )
        )

    res = solver.solve(observations)

    assert res.converged
    assert res.temperature_K == pytest.approx(T_K, rel=0.08)
    assert res.concentrations["A"] == pytest.approx(1.0, abs=1e-8)
