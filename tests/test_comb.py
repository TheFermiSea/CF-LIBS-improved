"""
Tests for comb template correlation algorithm.
"""

import pytest
import numpy as np
from cflibs.inversion.comb_identifier import CombIdentifier
from cflibs.inversion.element_id import ElementIdentificationResult


def test_build_triangular_template():
    """Test triangular template generation."""
    identifier = CombIdentifier(atomic_db=None)  # No DB needed for template test

    # Test width=5
    template = identifier._build_triangular_template(5)
    assert len(template) == 5
    assert template[2] == 1.0  # Peak at center
    assert template[0] < template[1] < template[2]
    assert template[4] < template[3] < template[2]
    # Should be symmetric
    np.testing.assert_allclose(template[0], template[4])
    np.testing.assert_allclose(template[1], template[3])

    # Test width=3
    template = identifier._build_triangular_template(3)
    assert len(template) == 3
    assert template[1] == 1.0  # Peak at center


def test_estimate_baseline_threshold(atomic_db, synthetic_libs_spectrum):
    """Test baseline estimation on synthetic data."""
    identifier = CombIdentifier(atomic_db)

    # Create synthetic spectrum with baseline
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(400.0, 1000.0), (500.0, 500.0)]},
        noise_level=0.05,
    )
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]

    baseline, threshold = identifier._estimate_baseline_threshold(wavelength, intensity)

    # Baseline should be smooth
    assert len(baseline) == len(intensity)
    assert np.all(baseline >= 0)

    # Threshold should be positive
    assert threshold > 0

    # Residual should capture peaks
    residual = intensity - baseline
    assert np.max(residual) > threshold


def test_correlate_tooth_with_peak(atomic_db):
    """Test tooth correlation when a peak is present."""
    identifier = CombIdentifier(atomic_db, min_correlation=0.5)

    # Create spectrum with sharp Gaussian peak
    wavelength = np.linspace(370.0, 380.0, 1000)
    center_nm = 375.0
    sigma = 0.05
    intensity = 10.0 + 1000.0 * np.exp(-0.5 * ((wavelength - center_nm) / sigma) ** 2)
    baseline = np.ones_like(wavelength) * 10.0

    tooth = identifier._correlate_tooth(wavelength, intensity, baseline, center_nm, threshold=50.0)

    # Should have high correlation
    assert tooth["center_nm"] == center_nm
    assert tooth["best_correlation"] > 0.5
    assert tooth["active"] == True  # Use == instead of 'is' for numpy bool
    assert abs(tooth["best_shift"]) <= identifier.max_shift_pts


def test_correlate_tooth_far_from_peak(atomic_db):
    """Test tooth correlation far from any peak."""
    identifier = CombIdentifier(atomic_db, min_correlation=0.7)  # Higher threshold

    # Create spectrum with peak at 375nm
    wavelength = np.linspace(370.0, 450.0, 2000)
    center_nm = 375.0
    sigma = 0.05
    intensity = 10.0 + 1000.0 * np.exp(-0.5 * ((wavelength - center_nm) / sigma) ** 2)
    baseline = np.ones_like(wavelength) * 10.0

    # Search far from peak at 440nm
    tooth = identifier._correlate_tooth(
        wavelength, intensity, baseline, center_nm=440.0, threshold=50.0
    )

    # Should have low correlation and be inactive with high threshold
    # (baseline-subtracted data will be near zero far from peak)
    assert tooth["best_correlation"] < 0.7


def test_identify_basic(atomic_db, synthetic_libs_spectrum):
    """Test full identify() on synthetic spectrum."""
    identifier = CombIdentifier(atomic_db, elements=["Fe", "H"])

    # Create spectrum with Fe and H lines
    spectrum = synthetic_libs_spectrum(
        elements={
            "Fe": [(371.99, 1000.0), (373.49, 500.0), (374.95, 200.0)],
            "H": [(656.28, 5000.0), (486.13, 1000.0)],
        },
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Should be ElementIdentificationResult
    assert isinstance(result, ElementIdentificationResult)

    # Should have some detected elements
    assert len(result.all_elements) > 0

    # Should detect Fe (has 3 lines in this range)
    fe_detected = any(e.element == "Fe" and e.detected for e in result.all_elements)
    assert fe_detected

    # Should have experimental peaks
    assert result.n_peaks > 0


def test_identify_returns_comb_algorithm(atomic_db, synthetic_libs_spectrum):
    """Verify algorithm field is set to 'comb'."""
    identifier = CombIdentifier(atomic_db, elements=["Fe"])

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    assert result.algorithm == "comb"
    assert "min_correlation" in result.parameters
    assert "baseline_window_nm" in result.parameters


def test_analyze_interferences(atomic_db):
    """Test interference detection between overlapping element teeth."""
    identifier = CombIdentifier(atomic_db)

    # Create two elements with overlapping active teeth
    element_teeth = {
        "Fe": [
            {"center_nm": 400.0, "best_correlation": 0.8, "active": True},
            {"center_nm": 500.0, "best_correlation": 0.7, "active": True},
        ],
        "Cu": [
            {"center_nm": 400.05, "best_correlation": 0.75, "active": True},  # Overlap with Fe
            {"center_nm": 600.0, "best_correlation": 0.6, "active": True},
        ],
    }

    updated_teeth = identifier._analyze_interferences(element_teeth, wl_tolerance_nm=0.1)

    # Fe at 400.0 should be interfered by Cu
    fe_tooth_400 = updated_teeth["Fe"][0]
    assert fe_tooth_400.get("is_interfered", False) is True
    assert "Cu" in fe_tooth_400.get("interfering_elements", [])

    # Cu at 400.05 should be interfered by Fe
    cu_tooth_400 = updated_teeth["Cu"][0]
    assert cu_tooth_400.get("is_interfered", False) is True
    assert "Fe" in cu_tooth_400.get("interfering_elements", [])

    # Fe at 500.0 should NOT be interfered
    fe_tooth_500 = updated_teeth["Fe"][1]
    assert fe_tooth_500.get("is_interfered", False) is False

    # Cu at 600.0 should NOT be interfered
    cu_tooth_600 = updated_teeth["Cu"][1]
    assert cu_tooth_600.get("is_interfered", False) is False


def test_fingerprint_computation(atomic_db):
    """Test fingerprint as mean of active tooth correlations."""
    identifier = CombIdentifier(atomic_db)

    # Test with active teeth
    teeth = [
        {"best_correlation": 0.8, "active": True},
        {"best_correlation": 0.6, "active": True},
        {"best_correlation": 0.3, "active": False},  # Should be excluded
        {"best_correlation": 0.7, "active": True},
    ]

    fingerprint = identifier._compute_fingerprint(teeth)
    expected = (0.8 + 0.6 + 0.7) / 3
    assert abs(fingerprint - expected) < 1e-6

    # Test with no active teeth
    teeth_inactive = [{"best_correlation": 0.3, "active": False}]
    fingerprint = identifier._compute_fingerprint(teeth_inactive)
    assert fingerprint == 0.0


def test_identify_with_no_elements(atomic_db, synthetic_libs_spectrum):
    """Test identify when spectrum has no matching elements."""
    # Search for elements not in spectrum
    identifier = CombIdentifier(atomic_db, elements=["Ca", "Mg"])

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Should have all elements in rejected (or none detected)
    assert len(result.detected_elements) == 0
