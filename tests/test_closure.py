"""
Tests for closure equation.
"""

import math

from cflibs.inversion.closure import ClosureEquation


def test_standard_closure():
    # Setup
    intercepts = {"Fe": 1.0, "Ti": 0.5}
    partition_funcs = {"Fe": 25.0, "Ti": 30.0}

    # Calculate expected
    # Fe: 25 * exp(1) = 67.96
    # Ti: 30 * exp(0.5) = 49.46
    # Total = 117.42
    # C_Fe = 67.96 / 117.42 = 0.578
    # C_Ti = 49.46 / 117.42 = 0.421

    res = ClosureEquation.apply_standard(intercepts, partition_funcs)

    assert res.mode == "standard"
    assert abs(sum(res.concentrations.values()) - 1.0) < 1e-6
    assert abs(res.concentrations["Fe"] - 0.578) < 0.01


def test_matrix_closure():
    intercepts = {"Fe": 1.0, "Ti": 0.5}
    partition_funcs = {"Fe": 25.0, "Ti": 30.0}

    # Fix Fe = 0.9
    # F = (25 * e^1) / 0.9 = 75.51
    # C_Ti = (30 * e^0.5) / 75.51 = 0.655
    # Wait, total > 1 is possible if matrix fraction + others > 1?
    # No, C_Ti should be small if Fe is large.
    # Ah, F is fixed by the Matrix element.
    # U_m * exp(q_m) / F = C_matrix
    # F = U_m * exp(q_m) / C_matrix
    # Then C_other = U_other * exp(q_other) / F

    res = ClosureEquation.apply_matrix_mode(
        intercepts, partition_funcs, matrix_element="Fe", matrix_fraction=0.9
    )

    assert res.concentrations["Fe"] == 0.9
    assert abs(res.concentrations["Ti"] - (49.46 / 75.51)) < 0.01


def test_oxide_closure():
    intercepts = {"Si": 1.0}
    partition_funcs = {"Si": 10.0}
    stoichiometry = {"Si": 2.139}  # SiO2

    # Si: 10 * e^1 = 27.18
    # Oxide mass = 27.18 * 2.139 = 58.14
    # F = 58.14
    # C_Si = 27.18 / 58.14 = 0.467

    res = ClosureEquation.apply_oxide_mode(intercepts, partition_funcs, stoichiometry)

    assert abs(res.concentrations["Si"] * 2.139 - 1.0) < 1e-6


def test_standard_closure_applies_abundance_multipliers():
    intercepts = {"Fe": 1.0, "Cu": 1.0}
    partition_funcs = {"Fe": 10.0, "Cu": 10.0}
    abundance_multipliers = {"Fe": 3.0, "Cu": 1.0}

    res = ClosureEquation.apply_standard(
        intercepts,
        partition_funcs,
        abundance_multipliers=abundance_multipliers,
    )

    expected_fe = 3.0 * 10.0 * math.exp(1.0)
    expected_cu = 1.0 * 10.0 * math.exp(1.0)
    expected_total = expected_fe + expected_cu

    assert res.concentrations["Fe"] == expected_fe / expected_total
    assert res.concentrations["Cu"] == expected_cu / expected_total
