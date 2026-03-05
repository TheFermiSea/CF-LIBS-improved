"""
Tests for partition function utilities.

Tests the polynomial partition function evaluator and integration with
the Saha-Boltzmann solver for computing ionization equilibrium.
"""

import numpy as np
from cflibs.plasma.partition import polynomial_partition_function
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from unittest.mock import MagicMock


def test_polynomial_evaluator():
    # log(U) = a0
    coeffs = [np.log(10), 0, 0, 0, 0]
    assert np.isclose(polynomial_partition_function(5000, coeffs), 10.0)

    # log(U) = a0 + a1*log(T)
    # let a0=0, a1=1 => log(U) = log(T) => U = T
    coeffs = [0, 1, 0, 0, 0]
    assert np.isclose(polynomial_partition_function(100, coeffs), 100.0)


def test_solver_integration():
    # Mock database
    db = MagicMock()
    # Mock partition coefficients
    pf_mock = MagicMock()
    pf_mock.coefficients = [np.log(100), 0, 0, 0, 0]
    # Ensure get_partition_coefficients exists on the mock
    db.get_partition_coefficients.return_value = pf_mock

    # Mock energy levels (should NOT be called if coeffs exist)
    db.get_energy_levels.return_value = []

    solver = SahaBoltzmannSolver(db)

    # Test using __wrapped__ to bypass caching decorator (which fails with mocks)
    # T = 1 eV ~ 11604 K
    # U should be 100 regardless of T because only a0 is set
    U = solver.calculate_partition_function.__wrapped__(solver, "H", 1, 1.0)
    assert np.isclose(U, 100.0)

    # Verify get_partition_coefficients was called
    db.get_partition_coefficients.assert_called_with("H", 1)


def test_solver_fallback():
    db = MagicMock()
    # No coefficients
    db.get_partition_coefficients.return_value = None
    # Ionization potential for the IP-based energy cutoff
    db.get_ionization_potential.return_value = 13.6

    # Levels: g=1, E=0; g=3, E=1.0
    level1 = MagicMock(g=1, energy_ev=0.0)
    level2 = MagicMock(g=3, energy_ev=1.0)
    db.get_energy_levels.return_value = [level1, level2]

    solver = SahaBoltzmannSolver(db)

    T_eV = 1.0
    # U = 1*exp(0) + 3*exp(-1/1) = 1 + 3*0.3678 = 2.103
    U = solver.calculate_partition_function.__wrapped__(solver, "H", 1, T_eV)

    expected = 1.0 + 3.0 * np.exp(-1.0)
    assert np.isclose(U, expected)
