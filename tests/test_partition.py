"""
Tests for partition function utilities.

Tests the polynomial partition function evaluator and integration with
the Saha-Boltzmann solver for computing ionization equilibrium.
"""

import json
from pathlib import Path

import numpy as np
import pytest
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


# ---- Expanded partition function coverage tests ----

REFERENCE_FILE = Path(__file__).parent / "data" / "nist_reference" / "partition_functions.json"
KEY_TEMPS = [5000, 10000, 15000, 20000]


@pytest.fixture(scope="module")
def nist_reference():
    """Load NIST reference partition function values."""
    with open(REFERENCE_FILE) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def test_reference_json_has_68_elements(nist_reference):
    """Reference JSON should cover at least 68 elements."""
    assert len(nist_reference) >= 68


def test_reference_json_has_106_species(nist_reference):
    """Reference JSON should cover at least 106 element-stage combos."""
    n_species = sum(len(stages) for stages in nist_reference.values())
    assert n_species >= 106


@pytest.mark.parametrize(
    "element,expected_stages",
    [
        ("Fe", ["1", "2"]),
        ("Cu", ["1", "2"]),
        ("Al", ["2"]),
        ("Cr", ["1", "2"]),
        ("Ti", ["1", "2"]),
        ("Ni", ["1", "2"]),
        ("Si", ["1", "2"]),
        ("Mg", ["1"]),
        ("Ca", ["1", "2"]),
        ("Na", ["1"]),
        ("Li", ["1"]),
        ("Ba", ["1", "2"]),
        ("Mn", ["1", "2"]),
        ("Co", ["1", "2"]),
        ("V", ["1", "2"]),
        ("W", ["1", "2"]),
    ],
)
def test_key_elements_in_reference(nist_reference, element, expected_stages):
    """Key LIBS elements must have partition function data."""
    assert element in nist_reference, f"{element} missing from reference data"
    for stage in expected_stages:
        assert stage in nist_reference[element], f"{element} stage {stage} missing"


@pytest.mark.parametrize(
    "element,stage,T_str,U_min,U_max",
    [
        # Fe I at 10000 K: well-known value ~42 from NIST
        ("Fe", "1", "10000", 30.0, 80.0),
        # Cu I at 5000 K: should be ~2 (ground state dominated)
        ("Cu", "1", "5000", 1.5, 3.5),
        # H-like ions: noble gases at 5000 K should have U ≈ ground state degeneracy
        ("Kr", "1", "5000", 0.9, 1.1),
        ("Xe", "1", "5000", 0.9, 1.1),
        # Rare earths have large partition functions
        ("Ce", "2", "10000", 100.0, 500.0),
    ],
)
def test_partition_function_physical_bounds(nist_reference, element, stage, T_str, U_min, U_max):
    """Partition function values should be physically reasonable."""
    U = nist_reference[element][stage][T_str]
    assert U_min <= U <= U_max, f"U({element} {stage}, T={T_str}K) = {U} not in [{U_min}, {U_max}]"


@pytest.mark.requires_db
def test_polynomial_fit_accuracy_at_key_temps(nist_reference):
    """Polynomial fits must reproduce reference values within 2% at key temperatures."""
    import sqlite3

    db_path = Path(__file__).parent.parent / "libs_production.db"
    if not db_path.exists():
        pytest.skip("libs_production.db not available")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT element, sp_num, a0, a1, a2, a3, a4 FROM partition_functions")
    coefficients = {(r[0], r[1]): list(r[2:]) for r in cursor}
    conn.close()

    max_error = 0.0
    worst_species = ""

    for element, stages in nist_reference.items():
        for stage_str, temp_values in stages.items():
            stage = int(stage_str)
            key = (element, stage)
            if key not in coefficients:
                continue

            coeffs = coefficients[key]
            for T_str in [str(t) for t in KEY_TEMPS]:
                if T_str not in temp_values:
                    continue
                U_ref = temp_values[T_str]
                T_K = float(T_str)
                U_fit = polynomial_partition_function(T_K, coeffs)
                rel_err = abs(U_fit - U_ref) / U_ref
                if rel_err > max_error:
                    max_error = rel_err
                    worst_species = f"{element} {stage_str} at T={T_str}K"

    assert max_error < 0.02, (
        f"Max relative error {max_error:.4%} exceeds 2% threshold " f"(worst: {worst_species})"
    )


def test_partition_function_monotonic_with_temperature(nist_reference):
    """Partition functions should generally increase with temperature."""
    n_non_monotonic = 0
    n_total = 0

    for element, stages in nist_reference.items():
        for stage_str, temp_values in stages.items():
            available_temps = sorted([int(t) for t in temp_values.keys()])
            if len(available_temps) < 3:
                continue
            U_values = [temp_values[str(t)] for t in available_temps]
            # Check overall trend: U at max T > U at min T
            n_total += 1
            if U_values[-1] < U_values[0] * 0.99:
                n_non_monotonic += 1

    assert (
        n_non_monotonic == 0
    ), f"{n_non_monotonic}/{n_total} species have decreasing U(T) trend (U_max < 0.99 * U_min)"


@pytest.mark.requires_db
def test_database_has_expanded_coefficients():
    """Database should have 106+ partition function coefficient sets."""
    import sqlite3

    db_path = Path(__file__).parent.parent / "libs_production.db"
    if not db_path.exists():
        pytest.skip("libs_production.db not available")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT COUNT(*) FROM partition_functions")
    count = cursor.fetchone()[0]
    conn.close()

    assert count >= 106, f"Database has {count} coefficient sets, expected >= 106"
