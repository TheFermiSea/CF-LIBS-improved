"""
Parity tests for Rust-accelerated CF-LIBS core functions.

Verifies that the Rust implementations of scan_comb_shifts,
kdet_filter_elements, and batch_partition_functions produce
results consistent with the pure-Python implementations.
"""

import pytest
import numpy as np

from cflibs.plasma.partition import polynomial_partition_function

pytestmark = pytest.mark.requires_rust

try:
    from cflibs._core import (
        scan_comb_shifts,
        kdet_filter_elements,
        batch_partition_functions,
    )

    HAS_RUST = True
except ImportError:
    try:
        from _core import (
            scan_comb_shifts,
            kdet_filter_elements,
            batch_partition_functions,
        )

        HAS_RUST = True
    except ImportError:
        HAS_RUST = False


@pytest.mark.skipif(not HAS_RUST, reason="cflibs-core not installed")
class TestCombRustParity:
    """Verify Rust comb matching outputs match Python outputs."""

    def test_scan_comb_shifts_basic(self):
        """Test scan_comb_shifts with synthetic peaks that have known matches."""
        # Peaks at known wavelengths
        peak_wavelengths = np.array([400.0, 500.0, 600.0, 700.0], dtype=np.float64)

        # Element A: transitions at 400.0, 500.0 (should match 2)
        # Element B: transitions at 600.0, 700.0, 800.0 (should match 2 out of 3)
        transition_wavelengths = [
            [400.0, 500.0],
            [600.0, 700.0, 800.0],
        ]
        element_names = ["A", "B"]
        shift_grid = np.array([0.0], dtype=np.float64)

        result = scan_comb_shifts(
            peak_wavelengths,
            transition_wavelengths,
            element_names,
            shift_grid,
            wavelength_tolerance=0.1,
            min_matches=1,
            min_precision=0.01,
            min_recall=0.1,
            max_missing_fraction=0.9,
        )

        assert "best_shift" in result
        assert "best_scores" in result
        assert "fallback_shift" in result

        # Best shift should be 0.0
        assert result["best_shift"] == 0.0

        # Check element scores
        scores = result["best_scores"]
        assert "A" in scores
        assert "B" in scores
        assert scores["A"]["matched_lines"] == 2
        assert scores["B"]["matched_lines"] == 2

    def test_scan_comb_shifts_with_shift(self):
        """Test that shift scanning finds the correct offset."""
        # Peaks shifted by +0.5 nm from transitions
        peak_wavelengths = np.array([399.5, 499.5, 599.5], dtype=np.float64)
        transition_wavelengths = [[400.0, 500.0, 600.0]]
        element_names = ["Fe"]
        # Shift grid includes 0.5 which should align peaks
        shift_grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)

        result = scan_comb_shifts(
            peak_wavelengths,
            transition_wavelengths,
            element_names,
            shift_grid,
            wavelength_tolerance=0.1,
            min_matches=2,
            min_precision=0.01,
            min_recall=0.3,
            max_missing_fraction=0.9,
        )

        assert result["best_shift"] == 0.5
        assert result["best_scores"]["Fe"]["matched_lines"] == 3

    def test_scan_comb_shifts_no_matches(self):
        """Test with peaks that don't match any transitions."""
        peak_wavelengths = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        transition_wavelengths = [[400.0, 500.0, 600.0]]
        element_names = ["Fe"]
        shift_grid = np.array([0.0], dtype=np.float64)

        result = scan_comb_shifts(
            peak_wavelengths,
            transition_wavelengths,
            element_names,
            shift_grid,
            wavelength_tolerance=0.1,
            min_matches=1,
            min_precision=0.01,
            min_recall=0.1,
            max_missing_fraction=0.9,
        )

        assert result["best_scores"]["Fe"]["matched_lines"] == 0
        assert not result["best_scores"]["Fe"]["passes"]

    def test_scan_comb_shifts_empty_inputs(self):
        """Test with empty peak array."""
        peak_wavelengths = np.array([], dtype=np.float64)
        transition_wavelengths = [[400.0]]
        element_names = ["Fe"]
        shift_grid = np.array([0.0], dtype=np.float64)

        result = scan_comb_shifts(
            peak_wavelengths,
            transition_wavelengths,
            element_names,
            shift_grid,
            wavelength_tolerance=0.1,
            min_matches=1,
            min_precision=0.01,
            min_recall=0.1,
            max_missing_fraction=0.9,
        )

        assert result["best_scores"]["Fe"]["matched_lines"] == 0

    def test_scan_comb_shifts_multiple_elements(self):
        """Test scoring with multiple elements, checking F1 ranking."""
        peak_wavelengths = np.array([400.0, 410.0, 420.0, 500.0, 510.0], dtype=np.float64)
        # Element A: matches 3 out of 3 peaks at 400, 410, 420
        transition_wavelengths_a = [400.0, 410.0, 420.0]
        # Element B: matches 2 out of 5 peaks at 500, 510
        transition_wavelengths_b = [500.0, 510.0, 520.0, 530.0, 540.0]

        result = scan_comb_shifts(
            peak_wavelengths,
            [transition_wavelengths_a, transition_wavelengths_b],
            ["A", "B"],
            np.array([0.0], dtype=np.float64),
            wavelength_tolerance=0.1,
            min_matches=1,
            min_precision=0.01,
            min_recall=0.1,
            max_missing_fraction=0.9,
        )

        scores = result["best_scores"]
        # Element A: recall = 3/3 = 1.0, precision = 3/5 = 0.6
        assert scores["A"]["matched_lines"] == 3
        assert abs(scores["A"]["recall"] - 1.0) < 1e-10
        # Element B: recall = 2/5 = 0.4, precision = 2/5 = 0.4
        assert scores["B"]["matched_lines"] == 2


@pytest.mark.skipif(not HAS_RUST, reason="cflibs-core not installed")
class TestKdetRustParity:
    """Verify Rust kdet filter outputs match Python outputs."""

    def test_kdet_filter_basic(self):
        """Test kdet filter with elements that should pass/fail."""
        peak_wavelengths = np.array([400.0, 410.0, 420.0, 500.0, 510.0], dtype=np.float64)
        # Element A: 3 transitions matching 3 peaks
        # Element B: 1 transition matching 1 peak (sparse, should fail)
        transition_wavelengths = [
            [400.0, 410.0, 420.0],
            [500.0],
        ]
        element_names = ["A", "B"]
        shift_grid = np.array([0.0], dtype=np.float64)

        passed = list(
            kdet_filter_elements(
                peak_wavelengths,
                transition_wavelengths,
                element_names,
                shift_grid,
                wavelength_tolerance=0.1,
                min_score=0.05,
                min_candidates=2,
                rarity_power=0.5,
                weight_clip=(0.25, 4.0),
            )
        )

        # Element A should pass (3 candidates >= 2), Element B should fail (1 < 2)
        assert "A" in passed
        assert "B" not in passed

    def test_kdet_filter_all_pass(self):
        """Test when all elements meet the threshold."""
        peak_wavelengths = np.array([400.0, 410.0, 500.0, 510.0], dtype=np.float64)
        transition_wavelengths = [
            [400.0, 410.0],
            [500.0, 510.0],
        ]
        element_names = ["A", "B"]
        shift_grid = np.array([0.0], dtype=np.float64)

        passed = list(
            kdet_filter_elements(
                peak_wavelengths,
                transition_wavelengths,
                element_names,
                shift_grid,
                wavelength_tolerance=0.1,
                min_score=0.01,
                min_candidates=2,
                rarity_power=0.5,
                weight_clip=(0.25, 4.0),
            )
        )

        assert "A" in passed
        assert "B" in passed

    def test_kdet_filter_empty_peaks(self):
        """Test with empty peak array."""
        peak_wavelengths = np.array([], dtype=np.float64)
        transition_wavelengths = [[400.0, 410.0]]
        element_names = ["A"]
        shift_grid = np.array([0.0], dtype=np.float64)

        passed = list(
            kdet_filter_elements(
                peak_wavelengths,
                transition_wavelengths,
                element_names,
                shift_grid,
                wavelength_tolerance=0.1,
                min_score=0.05,
                min_candidates=2,
                rarity_power=0.5,
                weight_clip=(0.25, 4.0),
            )
        )

        assert len(passed) == 0


@pytest.mark.skipif(not HAS_RUST, reason="cflibs-core not installed")
class TestPartitionRustParity:
    """Verify Rust partition function matches Python implementation."""

    def test_batch_partition_functions_single_species(self):
        """Test batch evaluation for a single species at multiple temperatures."""
        coefficients = np.array([[0.5, 0.3, -0.02, 0.001, -0.00005]], dtype=np.float64)
        temperatures = np.array([5000.0, 8000.0, 10000.0, 15000.0], dtype=np.float64)

        result = np.asarray(batch_partition_functions(coefficients, temperatures))

        assert result.shape == (1, 4)

        # Compare with Python implementation
        for t_idx, temp in enumerate(temperatures):
            expected = polynomial_partition_function(float(temp), list(coefficients[0]))
            assert (
                abs(result[0, t_idx] - expected) / expected < 1e-10
            ), f"Mismatch at T={temp}: rust={result[0, t_idx]}, python={expected}"

    def test_batch_partition_functions_multiple_species(self):
        """Test batch evaluation for multiple species."""
        coefficients = np.array(
            [
                [0.5, 0.3, -0.02, 0.001, -0.00005],
                [1.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        temperatures = np.array([5000.0, 10000.0, 20000.0], dtype=np.float64)

        result = np.asarray(batch_partition_functions(coefficients, temperatures))

        assert result.shape == (3, 3)

        for s in range(3):
            for t_idx, temp in enumerate(temperatures):
                expected = polynomial_partition_function(float(temp), list(coefficients[s]))
                assert (
                    abs(result[s, t_idx] - expected) / expected < 1e-10
                ), f"Mismatch at species={s}, T={temp}"

    def test_batch_partition_functions_low_temperature(self):
        """Test edge case with T <= 1 (should return 1.0)."""
        coefficients = np.array([[1.0, 0.5, 0.1, 0.0, 0.0]], dtype=np.float64)
        temperatures = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        result = np.asarray(batch_partition_functions(coefficients, temperatures))

        assert result.shape == (1, 3)
        for t_idx in range(3):
            assert (
                result[0, t_idx] == 1.0
            ), f"Expected 1.0 for T={temperatures[t_idx]}, got {result[0, t_idx]}"

    def test_batch_partition_functions_identity(self):
        """Test with zero coefficients (should give U=1.0 for all T)."""
        coefficients = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        temperatures = np.array([1000.0, 5000.0, 10000.0], dtype=np.float64)

        result = np.asarray(batch_partition_functions(coefficients, temperatures))

        for t_idx in range(3):
            assert abs(result[0, t_idx] - 1.0) < 1e-10

    def test_batch_partition_functions_linear(self):
        """Test coeffs [0, 1, 0, 0, 0] => U = T (since ln(U) = ln(T))."""
        coefficients = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        temperatures = np.array([100.0, 5000.0, 10000.0], dtype=np.float64)

        result = np.asarray(batch_partition_functions(coefficients, temperatures))

        for t_idx, temp in enumerate(temperatures):
            assert abs(result[0, t_idx] - temp) / temp < 1e-10

    def test_evaluate_batch_method(self):
        """Test the PartitionFunctionEvaluator.evaluate_batch method."""
        from cflibs.plasma.partition import PartitionFunctionEvaluator

        coefficients = np.array(
            [
                [0.5, 0.3, -0.02, 0.001, -0.00005],
                [1.0, 0.1, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        temperatures = np.array([5000.0, 10000.0], dtype=np.float64)

        result = PartitionFunctionEvaluator.evaluate_batch(coefficients, temperatures)

        assert result.shape == (2, 2)
        for s in range(2):
            for t_idx, temp in enumerate(temperatures):
                expected = polynomial_partition_function(float(temp), list(coefficients[s]))
                assert abs(result[s, t_idx] - expected) / expected < 1e-10