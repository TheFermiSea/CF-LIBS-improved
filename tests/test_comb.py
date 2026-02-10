"""
Tests for comb template correlation algorithm.
"""

import pytest
import numpy as np
from cflibs.inversion.comb_identifier import CombIdentifier
from cflibs.inversion.element_id import ElementIdentificationResult

pytestmark = pytest.mark.requires_db


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
    """Test fingerprint as sum(active correlations) / n_total_teeth."""
    identifier = CombIdentifier(atomic_db)

    # Test with active teeth: score = sum(active) / total
    teeth = [
        {"best_correlation": 0.8, "active": True},
        {"best_correlation": 0.6, "active": True},
        {"best_correlation": 0.3, "active": False},  # Excluded from sum
        {"best_correlation": 0.7, "active": True},
    ]

    fingerprint = identifier._compute_fingerprint(teeth)
    expected = (0.8 + 0.6 + 0.7) / 4  # Divide by total teeth (4), not active (3)
    assert abs(fingerprint - expected) < 1e-6

    # Test with no active teeth
    teeth_inactive = [{"best_correlation": 0.3, "active": False}]
    fingerprint = identifier._compute_fingerprint(teeth_inactive)
    assert fingerprint == 0.0

    # Test with empty list
    fingerprint = identifier._compute_fingerprint([])
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


# ============================================================================
# Additional comprehensive tests
# ============================================================================


def test_triangular_template_even_width(atomic_db):
    """Test that even widths are converted to odd."""
    identifier = CombIdentifier(atomic_db)

    # Pass even width, should become odd
    template = identifier._build_triangular_template(4)
    assert len(template) == 5  # 4+1 to make odd

    template = identifier._build_triangular_template(6)
    assert len(template) == 7  # 6+1 to make odd


def test_triangular_template_normalization(atomic_db):
    """Test template normalization to max=1.0."""
    identifier = CombIdentifier(atomic_db)

    for width in [3, 5, 7, 9]:
        template = identifier._build_triangular_template(width)
        assert np.max(template) == 1.0
        assert np.all(template >= 0)
        assert np.all(template <= 1.0)


def test_estimate_baseline_with_noisy_spectrum(atomic_db, synthetic_libs_spectrum):
    """Test baseline estimation with high noise."""
    identifier = CombIdentifier(atomic_db)

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(400.0, 1000.0)]},
        noise_level=0.2,  # 20% noise
    )

    baseline, threshold = identifier._estimate_baseline_threshold(
        spectrum["wavelength"], spectrum["intensity"]
    )

    # Should still produce valid results
    assert len(baseline) == len(spectrum["intensity"])
    assert threshold > 0


def test_estimate_baseline_flat_spectrum(atomic_db):
    """Test baseline estimation on flat spectrum."""
    identifier = CombIdentifier(atomic_db)

    wavelength = np.linspace(300.0, 400.0, 1000)
    intensity = np.ones_like(wavelength) * 100.0

    baseline, threshold = identifier._estimate_baseline_threshold(wavelength, intensity)

    # Baseline should be close to flat value
    np.testing.assert_allclose(baseline, 100.0, rtol=0.1)
    # No positive residuals, threshold should be 0
    assert threshold == 0.0


def test_correlate_tooth_at_spectrum_edge(atomic_db):
    """Test tooth correlation at spectrum edge."""
    identifier = CombIdentifier(atomic_db)

    wavelength = np.linspace(370.0, 380.0, 1000)
    intensity = np.ones_like(wavelength) * 10.0
    baseline = np.ones_like(wavelength) * 10.0

    # Try to correlate at the very edge
    tooth = identifier._correlate_tooth(
        wavelength, intensity, baseline, center_nm=370.0, threshold=50.0
    )

    # Should handle edge without crashing
    assert "best_correlation" in tooth
    assert "active" in tooth


def test_correlate_tooth_with_multiple_peaks(atomic_db):
    """Test tooth correlation with multiple nearby peaks."""
    identifier = CombIdentifier(atomic_db)

    wavelength = np.linspace(370.0, 380.0, 1000)
    sigma = 0.05
    # Create two nearby peaks
    intensity = 10.0 + 1000.0 * np.exp(-0.5 * ((wavelength - 375.0) / sigma) ** 2)
    intensity += 800.0 * np.exp(-0.5 * ((wavelength - 375.5) / sigma) ** 2)
    baseline = np.ones_like(wavelength) * 10.0

    tooth = identifier._correlate_tooth(
        wavelength, intensity, baseline, center_nm=375.0, threshold=50.0
    )

    # Should still find correlation at 375.0
    assert tooth["best_correlation"] > 0.3


def test_correlate_tooth_with_negative_baseline(atomic_db):
    """Test tooth correlation when baseline subtraction gives negative values."""
    identifier = CombIdentifier(atomic_db)

    wavelength = np.linspace(370.0, 380.0, 1000)
    intensity = np.ones_like(wavelength) * 10.0
    # Overestimated baseline
    baseline = np.ones_like(wavelength) * 20.0

    tooth = identifier._correlate_tooth(
        wavelength, intensity, baseline, center_nm=375.0, threshold=50.0
    )

    # Should handle negative residuals
    assert tooth["best_correlation"] <= 0.5  # Won't match well


def test_comb_identifier_custom_parameters(atomic_db, synthetic_libs_spectrum):
    """Test CombIdentifier with custom parameters."""
    identifier = CombIdentifier(
        atomic_db,
        baseline_window_nm=5.0,
        threshold_percentile=90.0,
        min_correlation=0.6,
        max_shift_pts=3,
        min_width_pts=5,
        max_width_factor=0.8,
        elements=["Fe"],
    )

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Check parameters are stored in result
    assert result.parameters["baseline_window_nm"] == 5.0
    assert result.parameters["threshold_percentile"] == 90.0
    assert result.parameters["min_correlation"] == 0.6


def test_identify_empty_wavelength_range(atomic_db):
    """Test identify with empty wavelength range."""
    identifier = CombIdentifier(atomic_db, elements=["Fe"])

    # Very narrow range with no lines
    wavelength = np.linspace(200.0, 201.0, 100)
    intensity = np.ones_like(wavelength) * 10.0

    result = identifier.identify(wavelength, intensity)

    # Should complete without error
    assert isinstance(result, ElementIdentificationResult)


def test_identify_high_correlation_threshold(atomic_db, synthetic_libs_spectrum):
    """Test identify with very high correlation threshold."""
    identifier = CombIdentifier(
        atomic_db,
        elements=["Fe", "H"],
        min_correlation=0.95  # Very high threshold
    )

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0)]},
        noise_level=0.05,  # Some noise
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # With high threshold, detection might be more strict
    assert isinstance(result, ElementIdentificationResult)


def test_analyze_interferences_no_overlap(atomic_db):
    """Test interference analysis with well-separated lines."""
    identifier = CombIdentifier(atomic_db)

    element_teeth = {
        "Fe": [{"center_nm": 400.0, "best_correlation": 0.8, "active": True}],
        "Cu": [{"center_nm": 500.0, "best_correlation": 0.75, "active": True}],
    }

    updated_teeth = identifier._analyze_interferences(element_teeth)

    # No teeth should be interfered
    assert updated_teeth["Fe"][0].get("is_interfered", False) is False
    assert updated_teeth["Cu"][0].get("is_interfered", False) is False


def test_analyze_interferences_inactive_teeth(atomic_db):
    """Test that inactive teeth don't cause interferences."""
    identifier = CombIdentifier(atomic_db)

    element_teeth = {
        "Fe": [{"center_nm": 400.0, "best_correlation": 0.8, "active": True}],
        "Cu": [
            {"center_nm": 400.05, "best_correlation": 0.3, "active": False}  # Inactive
        ],
    }

    updated_teeth = identifier._analyze_interferences(element_teeth)

    # Fe should NOT be interfered by inactive Cu tooth
    assert updated_teeth["Fe"][0].get("is_interfered", False) is False


def test_analyze_interferences_custom_tolerance(atomic_db):
    """Test interference analysis with custom wavelength tolerance."""
    identifier = CombIdentifier(atomic_db)

    element_teeth = {
        "Fe": [{"center_nm": 400.0, "best_correlation": 0.8, "active": True}],
        "Cu": [{"center_nm": 400.15, "best_correlation": 0.75, "active": True}],
    }

    # With tight tolerance, should not interfere
    updated_teeth = identifier._analyze_interferences(element_teeth, wl_tolerance_nm=0.1)
    assert updated_teeth["Fe"][0].get("is_interfered", False) is False

    # With loose tolerance, should interfere
    updated_teeth = identifier._analyze_interferences(element_teeth, wl_tolerance_nm=0.2)
    assert updated_teeth["Fe"][0].get("is_interfered", False) is True


def test_fingerprint_with_mixed_correlations(atomic_db):
    """Test fingerprint computation with mixed correlation values."""
    identifier = CombIdentifier(atomic_db)

    teeth = [
        {"best_correlation": 0.9, "active": True},
        {"best_correlation": 0.7, "active": True},
        {"best_correlation": 0.5, "active": True},
        {"best_correlation": 0.1, "active": False},
        {"best_correlation": 0.0, "active": False},
    ]

    fingerprint = identifier._compute_fingerprint(teeth)
    expected = (0.9 + 0.7 + 0.5) / 5  # Divide by total teeth (5)
    np.testing.assert_allclose(fingerprint, expected)


def test_fingerprint_all_active(atomic_db):
    """Test fingerprint when all teeth are active (sum/total = mean)."""
    identifier = CombIdentifier(atomic_db)

    teeth = [
        {"best_correlation": 0.8, "active": True},
        {"best_correlation": 0.8, "active": True},
        {"best_correlation": 0.8, "active": True},
    ]

    fingerprint = identifier._compute_fingerprint(teeth)
    # When all active: (0.8*3)/3 = 0.8 (same as mean)
    np.testing.assert_allclose(fingerprint, 0.8)


def test_identify_matched_lines_have_transitions(atomic_db, synthetic_libs_spectrum):
    """Test that matched lines have associated Transition objects."""
    identifier = CombIdentifier(atomic_db, elements=["Fe"])

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (373.49, 500.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Check detected elements have transitions
    for elem_id in result.detected_elements:
        for line in elem_id.matched_lines:
            assert line.transition is not None
            assert line.transition.element == elem_id.element


def test_identify_peak_counting(atomic_db, synthetic_libs_spectrum):
    """Test experimental peak counting."""
    identifier = CombIdentifier(atomic_db, elements=["Fe"])

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (373.49, 500.0), (374.95, 200.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Should detect multiple peaks
    assert result.n_peaks >= 3
    assert result.n_matched_peaks >= 0
    assert result.n_unmatched_peaks >= 0
    assert result.n_peaks == result.n_matched_peaks + result.n_unmatched_peaks


def test_identify_metadata_fields(atomic_db, synthetic_libs_spectrum):
    """Test that metadata fields are populated correctly."""
    identifier = CombIdentifier(atomic_db, elements=["Fe"])

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Check for metadata in element identifications
    for elem_id in result.all_elements:
        assert "fingerprint" in elem_id.metadata
        assert "n_active_teeth" in elem_id.metadata
        assert "n_total_teeth" in elem_id.metadata
        assert elem_id.metadata["fingerprint"] >= 0.0
        assert elem_id.metadata["fingerprint"] <= 1.0


def test_identify_score_equals_confidence(atomic_db, synthetic_libs_spectrum):
    """Test that score equals confidence for comb algorithm."""
    identifier = CombIdentifier(atomic_db, elements=["Fe"])

    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # For comb algorithm, score should equal confidence (both are fingerprint)
    for elem_id in result.all_elements:
        assert elem_id.score == elem_id.confidence


def test_correlate_tooth_zero_std_deviation(atomic_db):
    """Test tooth correlation when data has zero standard deviation."""
    identifier = CombIdentifier(atomic_db)

    wavelength = np.linspace(370.0, 380.0, 1000)
    intensity = np.ones_like(wavelength) * 10.0  # Constant
    baseline = np.ones_like(wavelength) * 10.0  # Residual will be zero

    tooth = identifier._correlate_tooth(
        wavelength, intensity, baseline, center_nm=375.0, threshold=50.0
    )

    # Should handle zero std gracefully
    assert tooth["best_correlation"] == 0.0
    assert tooth["active"] is False


def test_identify_with_single_element(atomic_db, synthetic_libs_spectrum):
    """Test identify with only one element in search list."""
    identifier = CombIdentifier(atomic_db, elements=["H"])

    spectrum = synthetic_libs_spectrum(
        elements={"H": [(656.28, 5000.0), (486.13, 1000.0)]},
        noise_level=0.01,
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Should have exactly one element analyzed
    assert len(result.all_elements) == 1
    assert result.all_elements[0].element == "H"


def test_identify_spectrum_range_validation(atomic_db, synthetic_libs_spectrum):
    """Test that spectrum range is logged correctly."""
    identifier = CombIdentifier(atomic_db, elements=["Fe"])

    spectrum = synthetic_libs_spectrum(
        wavelength_range=(350.0, 400.0),
        elements={"Fe": [(371.99, 1000.0)]},
    )

    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    # Should complete successfully
    assert result.algorithm == "comb"
    assert len(spectrum["wavelength"]) > 0
    assert spectrum["wavelength"][0] >= 350.0
    assert spectrum["wavelength"][-1] <= 400.0


def test_false_positive_noise_only(atomic_db):
    """Noise-only input should not detect any element."""
    identifier = CombIdentifier(atomic_db, elements=["Fe", "H"], min_correlation=0.5)

    # Create pure noise spectrum
    rng = np.random.default_rng(42)
    wavelength = np.linspace(200, 400, 2000)
    intensity = 100 + rng.normal(0, 10, 2000)

    result = identifier.identify(wavelength, intensity)
    # No element should be detected in pure noise
    assert len(result.detected_elements) == 0, (
        f"False positives on noise: {[e.element for e in result.detected_elements]}"
    )


def test_coverage_penalty_reduces_score(atomic_db):
    """Elements with few active teeth out of many should score low."""
    identifier = CombIdentifier(atomic_db)

    # Build teeth list: 3 active out of 50 total
    teeth = []
    for i in range(50):
        if i < 3:
            teeth.append({"active": True, "best_correlation": 0.9})
        else:
            teeth.append({"active": False, "best_correlation": 0.1})

    score = identifier._compute_fingerprint(teeth)
    # 3 * 0.9 / 50 = 0.054, should be much less than min_correlation
    assert score < 0.1, f"Score {score} too high for 3/50 active teeth"

def test_max_lines_per_element_parameter(atomic_db):
    """Test that max_lines_per_element caps transition count."""
    identifier = CombIdentifier(atomic_db, max_lines_per_element=5)
    assert identifier.max_lines_per_element == 5

    # Default should be 50
    identifier_default = CombIdentifier(atomic_db)
    assert identifier_default.max_lines_per_element == 50


def test_default_min_correlation_lowered(atomic_db):
    """Test that default min_correlation is 0.10."""
    identifier = CombIdentifier(atomic_db)
    assert identifier.min_correlation == 0.10
