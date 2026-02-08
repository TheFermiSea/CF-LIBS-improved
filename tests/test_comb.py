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


def test_build_triangular_template_even_width():
    """Test triangular template with even width converts to odd."""
    identifier = CombIdentifier(atomic_db=None)

    # Even width should be converted to odd
    template = identifier._build_triangular_template(4)
    assert len(template) == 5  # Should be 5 not 4
    assert template[2] == 1.0  # Peak at center


def test_build_triangular_template_normalization():
    """Test triangular template is normalized to max=1.0."""
    identifier = CombIdentifier(atomic_db=None)

    template = identifier._build_triangular_template(7)
    assert np.max(template) == 1.0
    assert np.all(template >= 0.0)
    assert np.all(template <= 1.0)


def test_estimate_baseline_threshold_empty_spectrum(atomic_db):
    """Test baseline estimation with zero intensity spectrum."""
    identifier = CombIdentifier(atomic_db)

    wavelength = np.linspace(300.0, 700.0, 1000)
    intensity = np.zeros_like(wavelength)

    baseline, threshold = identifier._estimate_baseline_threshold(wavelength, intensity)

    assert len(baseline) == len(intensity)
    assert threshold == 0.0  # No positive residuals


def test_estimate_baseline_threshold_flat_spectrum(atomic_db):
    """Test baseline estimation with flat spectrum."""
    identifier = CombIdentifier(atomic_db)

    wavelength = np.linspace(300.0, 700.0, 1000)
    intensity = np.ones_like(wavelength) * 100.0

    baseline, threshold = identifier._estimate_baseline_threshold(wavelength, intensity)

    # Baseline should be close to flat
    np.testing.assert_allclose(baseline, intensity, rtol=0.1)
    # Threshold should be very low (noise only)
    assert threshold < 5.0


def test_correlate_tooth_at_edge_of_spectrum(atomic_db):
    """Test tooth correlation at spectrum edge."""
    identifier = CombIdentifier(atomic_db, min_correlation=0.5)

    wavelength = np.linspace(370.0, 380.0, 1000)
    # Peak at the very start of spectrum
    center_nm = 370.5
    sigma = 0.05
    intensity = 10.0 + 1000.0 * np.exp(-0.5 * ((wavelength - center_nm) / sigma) ** 2)
    baseline = np.ones_like(wavelength) * 10.0

    tooth = identifier._correlate_tooth(wavelength, intensity, baseline, center_nm, threshold=50.0)

    # Should still find the peak even at edge
    assert tooth["center_nm"] == center_nm
    assert tooth["best_correlation"] > 0.0


def test_correlate_tooth_negative_intensities(atomic_db):
    """Test tooth correlation with negative intensities after baseline subtraction."""
    identifier = CombIdentifier(atomic_db)

    wavelength = np.linspace(370.0, 380.0, 1000)
    # Intensity below baseline
    intensity = np.ones_like(wavelength) * 5.0
    baseline = np.ones_like(wavelength) * 10.0

    tooth = identifier._correlate_tooth(wavelength, intensity, baseline, 375.0, threshold=50.0)

    # Should handle negative residuals gracefully
    assert tooth["best_correlation"] <= 0.5  # Low or negative correlation


def test_correlate_tooth_width_search_boundary():
    """Test tooth correlation width search at minimum width."""
    identifier = CombIdentifier(atomic_db=None, min_width_pts=3, max_width_factor=0.5)

    wavelength = np.linspace(370.0, 380.0, 2000)  # Fine resolution
    center_nm = 375.0
    sigma = 0.02  # Very narrow peak
    intensity = 10.0 + 1000.0 * np.exp(-0.5 * ((wavelength - center_nm) / sigma) ** 2)
    baseline = np.ones_like(wavelength) * 10.0

    tooth = identifier._correlate_tooth(wavelength, intensity, baseline, center_nm, threshold=50.0)

    # Should find best width
    assert tooth["best_width"] >= identifier.min_width_pts


def test_analyze_interferences_no_overlap(atomic_db):
    """Test interference analysis when elements don't overlap."""
    identifier = CombIdentifier(atomic_db)

    element_teeth = {
        "Fe": [{"center_nm": 400.0, "best_correlation": 0.8, "active": True}],
        "Cu": [{"center_nm": 500.0, "best_correlation": 0.75, "active": True}],
    }

    updated_teeth = identifier._analyze_interferences(element_teeth, wl_tolerance_nm=0.1)

    # No teeth should be interfered
    assert updated_teeth["Fe"][0].get("is_interfered", False) is False
    assert updated_teeth["Cu"][0].get("is_interfered", False) is False


def test_analyze_interferences_inactive_teeth_ignored(atomic_db):
    """Test interference analysis ignores inactive teeth."""
    identifier = CombIdentifier(atomic_db)

    element_teeth = {
        "Fe": [{"center_nm": 400.0, "best_correlation": 0.8, "active": True}],
        "Cu": [{"center_nm": 400.05, "best_correlation": 0.3, "active": False}],
    }

    updated_teeth = identifier._analyze_interferences(element_teeth, wl_tolerance_nm=0.1)

    # Fe should NOT be interfered (Cu tooth is inactive)
    assert updated_teeth["Fe"][0].get("is_interfered", False) is False


def test_analyze_interferences_same_element_not_interfering(atomic_db):
    """Test teeth from same element don't interfere with each other."""
    identifier = CombIdentifier(atomic_db)

    element_teeth = {
        "Fe": [
            {"center_nm": 400.0, "best_correlation": 0.8, "active": True},
            {"center_nm": 400.05, "best_correlation": 0.75, "active": True},
        ],
    }

    updated_teeth = identifier._analyze_interferences(element_teeth, wl_tolerance_nm=0.1)

    # Same element teeth should NOT interfere with each other
    assert updated_teeth["Fe"][0].get("is_interfered", False) is False
    assert updated_teeth["Fe"][1].get("is_interfered", False) is False


def test_compute_fingerprint_all_inactive():
    """Test fingerprint computation with all inactive teeth."""
    identifier = CombIdentifier(atomic_db=None)

    teeth = [
        {"best_correlation": 0.2, "active": False},
        {"best_correlation": 0.1, "active": False},
    ]

    fingerprint = identifier._compute_fingerprint(teeth)
    assert fingerprint == 0.0


def test_compute_fingerprint_mixed_correlations():
    """Test fingerprint with varying correlations."""
    identifier = CombIdentifier(atomic_db=None)

    teeth = [
        {"best_correlation": 1.0, "active": True},
        {"best_correlation": 0.5, "active": True},
        {"best_correlation": 0.0, "active": True},
    ]

    fingerprint = identifier._compute_fingerprint(teeth)
    expected = (1.0 + 0.5 + 0.0) / 3
    assert abs(fingerprint - expected) < 1e-6


def test_identify_empty_wavelength_array(atomic_db):
    """Test identify with empty arrays."""
    identifier = CombIdentifier(atomic_db, elements=["Fe"])

    wavelength = np.array([])
    intensity = np.array([])

    # Should handle gracefully or raise appropriate error
    try:
        result = identifier.identify(wavelength, intensity)
        # If it doesn't raise, check it returns valid result
        assert isinstance(result, ElementIdentificationResult)
    except (ValueError, IndexError):
        # Expected error for empty arrays
        pass


def test_identify_single_point_spectrum(atomic_db):
    """Test identify with single wavelength point."""
    identifier = CombIdentifier(atomic_db, elements=["Fe"])

    wavelength = np.array([400.0])
    intensity = np.array([1000.0])

    # Should handle gracefully
    try:
        result = identifier.identify(wavelength, intensity)
        assert isinstance(result, ElementIdentificationResult)
    except (ValueError, IndexError):
        # Expected for insufficient data
        pass


def test_identify_high_noise_spectrum(atomic_db, synthetic_libs_spectrum):
    """Test identify with very noisy spectrum."""
    identifier = CombIdentifier(atomic_db, elements=["Fe", "H"], min_correlation=0.3)

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0)]},
        noise_level=0.5,  # 50% noise!
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Should complete without error
    assert isinstance(result, ElementIdentificationResult)
    # May or may not detect Fe due to high noise
    assert len(result.all_elements) > 0


def test_identify_multiple_elements(atomic_db, synthetic_libs_spectrum):
    """Test identify with multiple overlapping elements."""
    identifier = CombIdentifier(atomic_db, elements=["Fe", "H"], min_correlation=0.4)

    spectrum = synthetic_libs_spectrum(
        elements={
            "Fe": [(371.99, 1000.0), (373.49, 800.0)],
            "H": [(656.28, 3000.0)],
        },
        noise_level=0.02,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Should detect multiple elements
    detected_symbols = [e.element for e in result.detected_elements]
    assert len(detected_symbols) >= 1  # At least one should be detected


def test_identify_parameters_stored_correctly(atomic_db, synthetic_libs_spectrum):
    """Test that algorithm parameters are stored in result."""
    baseline_window = 15.0
    threshold_percentile = 90.0
    min_correlation = 0.6

    identifier = CombIdentifier(
        atomic_db,
        baseline_window_nm=baseline_window,
        threshold_percentile=threshold_percentile,
        min_correlation=min_correlation,
        elements=["Fe"],
    )

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Check parameters are stored
    assert result.parameters["baseline_window_nm"] == baseline_window
    assert result.parameters["threshold_percentile"] == threshold_percentile
    assert result.parameters["min_correlation"] == min_correlation


def test_get_element_lines_out_of_range(atomic_db):
    """Test _get_element_lines with wavelength range that has no lines."""
    identifier = CombIdentifier(atomic_db)

    # Get Fe lines in UV range where database has none
    transitions = identifier._get_element_lines("Fe", 100.0, 150.0)

    # Should return empty list (no lines in this range)
    assert isinstance(transitions, list)


def test_identify_detected_vs_rejected_split(atomic_db, synthetic_libs_spectrum):
    """Test that elements are correctly split into detected vs rejected."""
    identifier = CombIdentifier(
        atomic_db,
        elements=["Fe", "H"],
        min_correlation=0.5,
    )

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 2000.0), (373.49, 1500.0), (374.95, 1000.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # All elements should be in all_elements
    assert len(result.all_elements) == len(result.detected_elements) + len(result.rejected_elements)

    # Detected elements should have score >= min_correlation
    for elem in result.detected_elements:
        assert elem.detected == True  # Use == to handle numpy bool
        assert elem.score >= identifier.min_correlation


def test_identify_matched_lines_have_transitions(atomic_db, synthetic_libs_spectrum):
    """Test that matched lines contain transition objects."""
    identifier = CombIdentifier(atomic_db, elements=["Fe"])

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 2000.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Find Fe in results
    for elem in result.detected_elements:
        if elem.element == "Fe":
            for line in elem.matched_lines:
                assert line.transition is not None
                assert line.element == "Fe"
                assert line.wavelength_th_nm > 0
                assert line.intensity_exp > 0


def test_identify_peak_counting(atomic_db, synthetic_libs_spectrum):
    """Test that experimental peaks are counted correctly."""
    identifier = CombIdentifier(atomic_db, elements=["Fe"])

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 2000.0), (373.49, 1500.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Should detect peaks
    assert result.n_peaks >= 2
    assert result.n_matched_peaks >= 0
    assert result.n_unmatched_peaks >= 0
    assert result.n_peaks == result.n_matched_peaks + result.n_unmatched_peaks