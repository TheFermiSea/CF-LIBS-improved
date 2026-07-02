"""
Tests for closure equation.
"""

import math

import pytest

from cflibs.inversion.physics.closure import ClosureEquation, log_ratios

# ---------------------------------------------------------------------------
# Aitchison log-ratios (Issue 2 — matrix-invariant DED deliverable)
# ---------------------------------------------------------------------------


def test_log_ratios_basic_values():
    comp = {"Ti": 0.90, "Al": 0.06, "V": 0.04}
    lr = log_ratios(comp, "Ti")
    assert set(lr) == {"Al", "V"}  # reference omitted by default
    assert lr["Al"] == pytest.approx(math.log(0.06 / 0.90))
    assert lr["V"] == pytest.approx(math.log(0.04 / 0.90))


def test_log_ratios_denominator_cancels_matrix_invariant():
    # The theorem: ln(C_s/C_ref) is identical whether computed from raw relatives
    # or from closure-normalized fractions, and is invariant to the detected set.
    # Add a fourth (spurious) element -> renormalize -> ratios among the original
    # three are unchanged (MatrixEffects.lean recoveredComposition_ratio_matrix_invariant).
    rel = {"Ti": 90.0, "Al": 6.0, "V": 4.0}
    norm = {k: v / sum(rel.values()) for k, v in rel.items()}
    lr_rel = log_ratios(rel, "Ti")
    lr_norm = log_ratios(norm, "Ti")
    for k in ("Al", "V"):
        assert lr_rel[k] == pytest.approx(lr_norm[k])
    # Add matrix contamination (a new detected element sloshes absolute fractions)
    rel2 = {**rel, "O": 50.0}
    lr_contaminated = log_ratios(rel2, "Ti")
    for k in ("Al", "V"):
        assert lr_contaminated[k] == pytest.approx(lr_rel[k])


def test_log_ratios_include_reference():
    lr = log_ratios({"Ti": 0.9, "V": 0.1}, "Ti", include_reference=True)
    assert lr["Ti"] == 0.0
    assert lr["V"] == pytest.approx(math.log(0.1 / 0.9))


def test_log_ratios_zero_or_missing_numerator_is_nan():
    lr = log_ratios({"Ti": 0.9, "V": 0.0, "Al": -1.0}, "Ti")
    assert math.isnan(lr["V"])  # zero numerator -> NaN, not a clipped finite value
    assert math.isnan(lr["Al"])  # negative numerator -> NaN


def test_log_ratios_nonfinite_numerator_is_nan():
    lr = log_ratios({"Ti": 0.9, "V": float("nan"), "Al": float("inf")}, "Ti")
    assert math.isnan(lr["V"])
    assert math.isnan(lr["Al"])


def test_log_ratios_missing_reference_raises():
    with pytest.raises(KeyError):
        log_ratios({"Ti": 0.9, "V": 0.1}, "Zr")


def test_log_ratios_nonpositive_reference_raises():
    with pytest.raises(ValueError):
        log_ratios({"Ti": 0.0, "V": 0.1}, "Ti")
    with pytest.raises(ValueError):
        log_ratios({"Ti": -0.1, "V": 0.1}, "Ti")
    with pytest.raises(ValueError):
        log_ratios({"Ti": float("nan"), "V": 0.1}, "Ti")


def test_log_ratios_scale_invariance():
    # Uniform scaling of all concentrations leaves every log-ratio unchanged.
    comp = {"Ti": 0.9, "Al": 0.06, "V": 0.04}
    scaled = {k: 137.0 * v for k, v in comp.items()}
    lr_a = log_ratios(comp, "Ti")
    lr_b = log_ratios(scaled, "Ti")
    for k in lr_a:
        assert lr_a[k] == pytest.approx(lr_b[k])


def test_cflibsresult_log_ratios_method_and_autofield():
    from cflibs.inversion.solve.iterative import CFLIBSResult

    res = CFLIBSResult(
        temperature_K=11000.0,
        temperature_uncertainty_K=0.0,
        electron_density_cm3=1e17,
        concentrations={"Ti": 0.90, "Al": 0.06, "V": 0.04},
        concentration_uncertainties={},
        iterations=1,
        converged=True,
    )
    # Explicit-reference method
    lr = res.log_ratios("Ti")
    assert lr["V"] == pytest.approx(math.log(0.04 / 0.90))
    # Auto-populated diagnostic against the dominant (matrix) element
    assert res.log_ratio_reference == "Ti"
    assert res.log_ratios_vs_dominant["V"] == pytest.approx(math.log(0.04 / 0.90))
    # Issue-6 completeness marker (light-touch annotation)
    assert res.closure_completeness == "detected_set_only"


def test_cflibsresult_log_ratios_empty_composition_safe():
    from cflibs.inversion.solve.iterative import CFLIBSResult

    res = CFLIBSResult(
        temperature_K=float("nan"),
        temperature_uncertainty_K=0.0,
        electron_density_cm3=0.0,
        concentrations={},
        concentration_uncertainties={},
        iterations=0,
        converged=False,
        failed=True,
    )
    assert res.log_ratio_reference is None
    assert res.log_ratios_vs_dominant == {}


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

    # Log-sum-exp stabilization is identical up to floating-point rounding
    # (~1 ULP); the closure parity gate is defined at 1e-12.
    assert abs(res.concentrations["Fe"] - 0.9) < 1e-12
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

    # Log-sum-exp stabilization is identical up to floating-point rounding
    # (~1 ULP); the closure parity gate is defined at 1e-12.
    assert abs(res.concentrations["Fe"] - expected_fe / expected_total) < 1e-12
    assert abs(res.concentrations["Cu"] - expected_cu / expected_total) < 1e-12


def test_validate_degeneracy_flags_dominant_element():
    # One element soaking > 0.8 with > 1 element present is degenerate.
    assert ClosureEquation.validate_degeneracy({"Fe": 0.95, "Cu": 0.05}) is True
    # Exactly at the threshold is NOT degenerate (strict >).
    assert ClosureEquation.validate_degeneracy({"Fe": 0.8, "Cu": 0.2}) is False
    # Balanced composition is fine.
    assert ClosureEquation.validate_degeneracy({"Fe": 0.5, "Cu": 0.5}) is False


def test_validate_degeneracy_single_element_never_degenerate():
    # A pure sample legitimately closes to a single 1.0 — never flagged.
    assert ClosureEquation.validate_degeneracy({"Fe": 1.0}) is False
    assert ClosureEquation.validate_degeneracy({}) is False


def test_validate_degeneracy_custom_threshold():
    assert ClosureEquation.validate_degeneracy({"Fe": 0.7, "Cu": 0.3}, threshold=0.6) is True
    assert ClosureEquation.validate_degeneracy({"Fe": 0.7, "Cu": 0.3}, threshold=0.9) is False
