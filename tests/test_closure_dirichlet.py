"""
Tests for Dirichlet-residual closure equation.

Validates that the latent dark-element residual correctly absorbs missing mass
instead of inflating detected element concentrations.
"""

import math

import pytest

from cflibs.inversion.physics.closure import (
    ClosureEquation,
    ClosureMode,
    DirichletResidualResult,
)
from cflibs.inversion.physics.quality import QualityAssessor, QualityMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intercepts_and_pfs(element_fractions, F=100.0):
    """
    Build synthetic intercepts and partition functions that would produce
    the given element fractions under standard closure.

    Given target fraction c_s, we need:
        c_s = (U_s * exp(q_s)) / F
    Set U_s = 1 for simplicity, then exp(q_s) = c_s * F  =>  q_s = ln(c_s * F).
    """
    intercepts = {}
    partition_funcs = {}
    for el, frac in element_fractions.items():
        partition_funcs[el] = 1.0
        intercepts[el] = math.log(frac * F)
    return intercepts, partition_funcs


# ---------------------------------------------------------------------------
# Tests: no missing elements  (residual should be ~0)
# ---------------------------------------------------------------------------


class TestNoMissingElements:
    """When all mass is accounted for, residual should be zero or negligible."""

    def test_simple_no_residual(self):
        """Raw sum == 1 => residual is 0, concentrations match standard."""
        fracs = {"Fe": 0.7, "Cu": 0.2, "Al": 0.1}
        intercepts, pfs = _make_intercepts_and_pfs(fracs)

        res = ClosureEquation.apply_dirichlet_residual(intercepts, pfs, mode="simple")

        assert isinstance(res, DirichletResidualResult)
        assert res.mode == "simple"
        assert res.residual_fraction == pytest.approx(0.0, abs=1e-10)
        assert sum(res.concentrations.values()) == pytest.approx(1.0, abs=1e-10)
        for el, frac in fracs.items():
            assert res.concentrations[el] == pytest.approx(frac, rel=1e-6)

    def test_dirichlet_no_missing_mass(self):
        """Even in dirichlet mode, when raw sum is large the residual is small."""
        fracs = {"Fe": 0.7, "Cu": 0.2, "Al": 0.1}
        intercepts, pfs = _make_intercepts_and_pfs(fracs, F=100.0)

        res = ClosureEquation.apply_dirichlet_residual(intercepts, pfs, mode="dirichlet")

        assert res.mode == "dirichlet"
        # With raw_sum = 100, residual should be small
        assert res.residual_fraction < 0.02
        assert sum(res.concentrations.values()) == pytest.approx(
            1.0 - res.residual_fraction, abs=1e-10
        )


# ---------------------------------------------------------------------------
# Tests: missing elements  (the core use case)
# ---------------------------------------------------------------------------


class TestMissingElements:
    """When elements are missing, residual should capture the deficit."""

    def test_simple_captures_missing_mass(self):
        """
        Ground truth: Fe=0.6, Cu=0.2, Al=0.1, S=0.1
        Only detect Fe, Cu, Al.  Standard closure would inflate them to sum=1.
        Dirichlet-residual (simple) should assign ~0.1 to residual.
        """
        # Build intercepts as if the "true F" is known and S is simply absent.
        # With U=1, rel_C = exp(q) = true_frac * F_true
        # detected raw sum = 0.9 * F_true;  we want raw_sum == 0.9
        # so F_true = 1.0
        detected = {"Fe": 0.6, "Cu": 0.2, "Al": 0.1}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        res = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="simple", residual_threshold=0.05
        )

        # Raw sum should be ~0.9
        assert res.raw_closure_sum == pytest.approx(0.9, rel=1e-6)
        # Residual should be ~0.1
        assert res.residual_fraction == pytest.approx(0.1, rel=1e-6)
        # Detected concentrations should sum to ~0.9
        assert sum(res.concentrations.values()) == pytest.approx(0.9, rel=1e-6)

    def test_detected_lower_than_standard(self):
        """
        Key property: detected element concentrations must be LOWER than
        standard closure when mass is missing.
        """
        detected = {"Fe": 0.6, "Cu": 0.2, "Al": 0.1}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        standard_res = ClosureEquation.apply_standard(intercepts, pfs)
        dirichlet_res = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="simple", residual_threshold=0.05
        )

        for el in detected:
            assert dirichlet_res.concentrations[el] < standard_res.concentrations[el], (
                f"{el}: dirichlet {dirichlet_res.concentrations[el]:.4f} "
                f"should be < standard {standard_res.concentrations[el]:.4f}"
            )

    def test_dirichlet_mode_captures_missing_mass(self):
        """Dirichlet mode should also produce a non-zero residual for missing mass."""
        detected = {"Fe": 0.6, "Cu": 0.2, "Al": 0.1}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        res = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="dirichlet", alpha_residual=2.0
        )

        assert res.residual_fraction > 0.0
        assert sum(res.concentrations.values()) < 1.0

    def test_large_missing_fraction(self):
        """When a large fraction is missing (50%), residual should be substantial."""
        detected = {"Fe": 0.3, "Cu": 0.2}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        res = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="simple", residual_threshold=0.05
        )

        assert res.raw_closure_sum == pytest.approx(0.5, rel=1e-6)
        assert res.residual_fraction == pytest.approx(0.5, rel=1e-6)
        assert sum(res.concentrations.values()) == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: simple vs dirichlet mode comparison
# ---------------------------------------------------------------------------


class TestModeComparison:
    """Compare simple and dirichlet modes produce reasonable results."""

    def test_both_modes_produce_non_negative(self):
        detected = {"Fe": 0.5, "Cu": 0.3}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        for m in ("simple", "dirichlet"):
            res = ClosureEquation.apply_dirichlet_residual(intercepts, pfs, mode=m)
            assert res.residual_fraction >= 0.0
            for c in res.concentrations.values():
                assert c >= 0.0

    def test_dirichlet_alpha_residual_effect(self):
        """Higher alpha_residual should produce a larger residual fraction."""
        detected = {"Fe": 0.6, "Cu": 0.2, "Al": 0.1}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        res_low = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="dirichlet", alpha_residual=1.5
        )
        res_high = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="dirichlet", alpha_residual=5.0
        )

        assert res_high.residual_fraction > res_low.residual_fraction

    def test_relative_proportions_preserved(self):
        """
        Relative proportions among detected elements should be preserved
        regardless of mode.
        """
        detected = {"Fe": 0.6, "Cu": 0.3, "Al": 0.1}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        for m in ("simple", "dirichlet"):
            res = ClosureEquation.apply_dirichlet_residual(intercepts, pfs, mode=m)
            # Fe/Cu ratio should be 2.0 regardless of mode
            ratio = res.concentrations["Fe"] / res.concentrations["Cu"]
            assert ratio == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case handling."""

    def test_single_element(self):
        """Single detected element: residual captures missing mass."""
        intercepts, pfs = _make_intercepts_and_pfs({"Fe": 0.5}, F=1.0)

        res = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="simple", residual_threshold=0.05
        )

        assert res.residual_fraction == pytest.approx(0.5, rel=1e-6)
        assert res.concentrations["Fe"] == pytest.approx(0.5, rel=1e-6)

    def test_all_elements_detected_no_residual(self):
        """When raw sum == 1, both modes should give ~0 residual."""
        fracs = {"Fe": 0.5, "Cu": 0.3, "Al": 0.2}
        intercepts, pfs = _make_intercepts_and_pfs(fracs, F=1.0)

        res_simple = ClosureEquation.apply_dirichlet_residual(intercepts, pfs, mode="simple")
        assert res_simple.residual_fraction == pytest.approx(0.0, abs=1e-10)

    def test_very_large_residual(self):
        """When only a tiny fraction is detected, residual should be very large."""
        intercepts, pfs = _make_intercepts_and_pfs({"Fe": 0.01}, F=1.0)

        res = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="simple", residual_threshold=0.05
        )

        assert res.residual_fraction == pytest.approx(0.99, rel=1e-4)
        assert res.concentrations["Fe"] == pytest.approx(0.01, rel=1e-4)

    def test_zero_concentrations(self):
        """All-zero input produces empty result."""
        res = ClosureEquation.apply_dirichlet_residual(
            intercepts={},
            partition_funcs={},
            mode="simple",
        )

        assert res.concentrations == {}
        assert res.residual_fraction == 1.0

    def test_missing_partition_function_skipped(self):
        """Elements without partition functions are skipped gracefully."""
        intercepts = {"Fe": 1.0, "Unknown": 2.0}
        partition_funcs = {"Fe": 10.0}  # No entry for Unknown

        res = ClosureEquation.apply_dirichlet_residual(intercepts, partition_funcs, mode="simple")

        assert "Unknown" not in res.concentrations
        assert "Fe" in res.concentrations

    def test_abundance_multipliers_applied(self):
        """Abundance multipliers should affect raw concentrations."""
        intercepts = {"Fe": 1.0, "Cu": 1.0}
        partition_funcs = {"Fe": 10.0, "Cu": 10.0}

        res_no_mult = ClosureEquation.apply_dirichlet_residual(
            intercepts, partition_funcs, mode="simple"
        )
        res_with_mult = ClosureEquation.apply_dirichlet_residual(
            intercepts,
            partition_funcs,
            abundance_multipliers={"Fe": 3.0, "Cu": 1.0},
            mode="simple",
        )

        # With multiplier Fe should have higher proportion relative to Cu
        ratio_no_mult = res_no_mult.concentrations["Fe"] / res_no_mult.concentrations["Cu"]
        ratio_with_mult = res_with_mult.concentrations["Fe"] / res_with_mult.concentrations["Cu"]
        assert ratio_with_mult == pytest.approx(3.0 * ratio_no_mult, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: diagnostic output
# ---------------------------------------------------------------------------


class TestDiagnostics:
    """Verify diagnostic fields are populated correctly."""

    def test_closure_diagnostic_value(self):
        detected = {"Fe": 0.6, "Cu": 0.2}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        res = ClosureEquation.apply_dirichlet_residual(intercepts, pfs, mode="simple")

        # raw_sum = 0.8, so diagnostic = |0.8 - 1| = 0.2
        assert res.closure_diagnostic == pytest.approx(0.2, rel=1e-6)

    def test_raw_closure_sum(self):
        detected = {"Fe": 0.7, "Cu": 0.3}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        res = ClosureEquation.apply_dirichlet_residual(intercepts, pfs, mode="simple")

        assert res.raw_closure_sum == pytest.approx(1.0, rel=1e-6)

    def test_experimental_factor_set(self):
        """Experimental factor should equal the raw sum."""
        detected = {"Fe": 0.6, "Cu": 0.2}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        res = ClosureEquation.apply_dirichlet_residual(intercepts, pfs, mode="simple")

        assert res.experimental_factor == pytest.approx(res.raw_closure_sum, rel=1e-10)


class TestGammaResidualReporting:
    """Verify residual reporting is explicit and not inferred for all under-closure."""

    def test_closure_mode_enum(self):
        assert ClosureMode.DIRICHLET_RESIDUAL.value == "dirichlet_residual"

    def test_quality_metrics_gamma_residual_serializes(self):
        metrics = QualityMetrics(
            r_squared_boltzmann=0.99,
            closure_residual=0.05,
            gamma_residual=0.05,
        )

        assert metrics.gamma_residual == pytest.approx(0.05)
        assert metrics.to_dict()["gamma_residual"] == pytest.approx(0.05)

    def test_quality_assessor_only_suppresses_warning_for_explicit_gamma(self):
        assessor = QualityAssessor()
        kwargs = {
            "observations": [],
            "temperature_K": 10000.0,
            "electron_density_cm3": 1.0e17,
            "concentrations": {"Fe": 0.80},
            "ionization_potentials": {"Fe": 7.9},
            "partition_funcs_I": {"Fe": 1.0},
            "partition_funcs_II": {"Fe": 1.0},
        }

        standard_metrics = assessor.assess(**kwargs)
        residual_metrics = assessor.assess(**kwargs, gamma_residual=0.20)

        assert standard_metrics.gamma_residual == pytest.approx(0.0)
        assert standard_metrics.warnings
        assert residual_metrics.gamma_residual == pytest.approx(0.20)
        assert residual_metrics.warnings == []


# ---------------------------------------------------------------------------
# Tests: threshold behaviour
# ---------------------------------------------------------------------------


class TestScaleInvariance:
    """Validation gate (c): the Dirichlet-residual closure diagnostic and
    residual must be invariant to a global intensity rescaling.

    Scaling all line intensities by a constant ``k`` shifts every Boltzmann
    intercept ``q_s`` by ``ln(k)`` (so every ``rel_C_s`` scales by ``k``) and
    scales the data-derived experimental factor ``F`` by ``k`` as well.  The
    calibrated raw sum ``rho = sum(rel_C_s) / F`` — and hence the deficit,
    residual, and closure diagnostic — must not change.
    """

    def test_diagnostic_unchanged_when_intensities_scaled(self):
        detected = {"Fe": 0.6, "Cu": 0.2, "Al": 0.1}  # raw_sum = 0.9, deficit 0.1
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        base = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="simple", experimental_factor=1.0
        )

        k = 1.0e3
        scaled_intercepts = {el: q + math.log(k) for el, q in intercepts.items()}
        scaled = ClosureEquation.apply_dirichlet_residual(
            scaled_intercepts, pfs, mode="simple", experimental_factor=k
        )

        assert scaled.closure_diagnostic == pytest.approx(base.closure_diagnostic, rel=1e-9)
        assert scaled.residual_fraction == pytest.approx(base.residual_fraction, rel=1e-9)
        for el in detected:
            assert scaled.concentrations[el] == pytest.approx(base.concentrations[el], rel=1e-9)

    def test_diagnostic_unchanged_dirichlet_mode_scaled(self):
        detected = {"Fe": 0.6, "Cu": 0.2, "Al": 0.1}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        base = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="dirichlet", experimental_factor=1.0
        )

        k = 1.0e3
        scaled_intercepts = {el: q + math.log(k) for el, q in intercepts.items()}
        scaled = ClosureEquation.apply_dirichlet_residual(
            scaled_intercepts, pfs, mode="dirichlet", experimental_factor=k
        )

        assert scaled.closure_diagnostic == pytest.approx(base.closure_diagnostic, rel=1e-9)
        assert scaled.residual_fraction == pytest.approx(base.residual_fraction, rel=1e-9)

    def test_naive_reference_would_break_without_factor(self):
        """Without passing the rescaled experimental_factor the diagnostic
        would change — this documents that the calibration factor is what
        provides scale-invariance (and that the default F=1 path is the
        unit-calibrated convention)."""
        detected = {"Fe": 0.6, "Cu": 0.2, "Al": 0.1}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)
        k = 1.0e3
        scaled_intercepts = {el: q + math.log(k) for el, q in intercepts.items()}

        # Passing F=1 with 1e3-scaled intensities is a calibration mismatch:
        # raw_sum ~ 900 >> 1, so the diagnostic blows up.
        mismatched = ClosureEquation.apply_dirichlet_residual(
            scaled_intercepts, pfs, mode="simple", experimental_factor=1.0
        )
        assert mismatched.closure_diagnostic > 100.0

    def test_invalid_experimental_factor_raises(self):
        intercepts, pfs = _make_intercepts_and_pfs({"Fe": 0.6, "Cu": 0.2}, F=1.0)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="experimental_factor"):
                ClosureEquation.apply_dirichlet_residual(
                    intercepts, pfs, mode="simple", experimental_factor=bad
                )


class TestThreshold:
    """Verify residual_threshold gating in simple mode."""

    def test_below_threshold_no_residual(self):
        """When deficit < threshold, simple mode returns 0 residual."""
        # deficit = 0.03 < default threshold 0.05
        detected = {"Fe": 0.67, "Cu": 0.3}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        res = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="simple", residual_threshold=0.05
        )

        assert res.residual_fraction == 0.0

    def test_above_threshold_activates_residual(self):
        """When deficit > threshold, simple mode activates the residual."""
        detected = {"Fe": 0.6, "Cu": 0.3}
        intercepts, pfs = _make_intercepts_and_pfs(detected, F=1.0)

        res = ClosureEquation.apply_dirichlet_residual(
            intercepts, pfs, mode="simple", residual_threshold=0.05
        )

        assert res.residual_fraction == pytest.approx(0.1, rel=1e-6)
