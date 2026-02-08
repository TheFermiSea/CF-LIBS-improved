"""
Tests for ALIAS element identification algorithm.
"""

import pytest
import numpy as np
from cflibs.inversion.alias_identifier import ALIASIdentifier
from cflibs.inversion.element_id import ElementIdentificationResult


def test_detect_peaks(atomic_db, synthetic_libs_spectrum):
    """Test peak detection with 2nd derivative enhancement."""
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (400.0, 500.0), (450.0, 200.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db)
    peaks = identifier._detect_peaks(spectrum["wavelength"], spectrum["intensity"])

    # Should detect 3 peaks
    assert len(peaks) > 0
    assert isinstance(peaks, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in peaks)

    # Peaks should be (index, wavelength) tuples
    peak_wavelengths = [p[1] for p in peaks]

    # Should find peaks near expected positions (within 1 nm)
    expected_wls = [371.99, 400.0, 450.0]
    for expected_wl in expected_wls:
        closest = min(peak_wavelengths, key=lambda x: abs(x - expected_wl))
        assert (
            abs(closest - expected_wl) < 1.0
        ), f"Expected peak at {expected_wl}, closest found at {closest}"


def test_compute_element_emissivities(atomic_db):
    """Test emissivity calculation for Fe I lines."""
    identifier = ALIASIdentifier(atomic_db)

    # Compute emissivities for Fe in 370-376 nm range (covers test lines)
    line_data = identifier._compute_element_emissivities("Fe", 370.0, 376.0)

    assert len(line_data) > 0
    for line in line_data:
        assert "transition" in line
        assert "avg_emissivity" in line
        assert "wavelength_nm" in line
        assert line["avg_emissivity"] > 0
        assert 370.0 <= line["wavelength_nm"] <= 376.0


def test_fuse_lines(atomic_db):
    """Test line fusion within resolution element."""
    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # Get Fe lines
    line_data = identifier._compute_element_emissivities("Fe", 370.0, 376.0)
    wavelength = np.linspace(370.0, 376.0, 1000)

    # Fuse lines
    fused = identifier._fuse_lines(line_data, wavelength)

    assert len(fused) > 0
    for line in fused:
        assert "transition" in line
        assert "avg_emissivity" in line
        assert "wavelength_nm" in line
        assert "n_fused" in line
        assert line["n_fused"] >= 1


def test_match_lines(atomic_db):
    """Test matching theoretical lines to experimental peaks."""
    identifier = ALIASIdentifier(atomic_db)

    # Create fused lines at specific wavelengths
    from cflibs.atomic.structures import Transition

    trans1 = Transition("Fe", 1, 372.0, 1e7, 3.33, 0.0, 11, 9)
    trans2 = Transition("Fe", 1, 373.5, 5e6, 3.32, 0.0, 9, 9)
    fused_lines = [
        {"transition": trans1, "avg_emissivity": 1000.0, "wavelength_nm": 372.0},
        {"transition": trans2, "avg_emissivity": 500.0, "wavelength_nm": 373.5},
    ]

    # Create peaks near theoretical wavelengths
    peaks = [(100, 372.01), (200, 373.49)]  # (index, wavelength)

    matched_mask, wavelength_shifts = identifier._match_lines(fused_lines, peaks)

    # Both lines should match
    assert matched_mask[0] == True
    assert matched_mask[1] == True

    # Shifts should be small
    assert abs(wavelength_shifts[0]) < 0.1
    assert abs(wavelength_shifts[1]) < 0.1


def test_identify_basic(atomic_db, synthetic_libs_spectrum):
    """Test full identify() with synthetic spectrum containing Fe lines."""
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (373.49, 500.0), (374.95, 200.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db, elements=["Fe"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    assert isinstance(result, ElementIdentificationResult)
    assert result.algorithm == "alias"
    assert result.n_peaks > 0

    # Fe should be in the results (detected or rejected)
    fe_elements = [e for e in result.all_elements if e.element == "Fe"]
    assert len(fe_elements) == 1

    fe_result = fe_elements[0]
    assert fe_result.n_matched_lines > 0


def test_identify_returns_result_type(atomic_db, synthetic_libs_spectrum):
    """Test that identify() returns ElementIdentificationResult."""
    spectrum = synthetic_libs_spectrum()

    identifier = ALIASIdentifier(atomic_db, elements=["Fe"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    assert isinstance(result, ElementIdentificationResult)
    assert hasattr(result, "detected_elements")
    assert hasattr(result, "rejected_elements")
    assert hasattr(result, "all_elements")
    assert hasattr(result, "experimental_peaks")
    assert hasattr(result, "algorithm")
    assert result.algorithm == "alias"


def test_identify_no_elements(atomic_db, synthetic_libs_spectrum):
    """Test identify with no matching elements (edge case)."""
    # Create spectrum with only H line, but search for Cu
    spectrum = synthetic_libs_spectrum(
        elements={"H": [(656.28, 5000.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db, elements=["Cu"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    assert isinstance(result, ElementIdentificationResult)
    # Cu should not be detected (no Cu lines in wavelength range of test DB)
    cu_elements = [e for e in result.detected_elements if e.element == "Cu"]
    assert len(cu_elements) == 0


def test_scores_between_zero_and_one(atomic_db, synthetic_libs_spectrum):
    """Test that all scores are in [0, 1] range."""
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (373.49, 500.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db, elements=["Fe", "H"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    for element_id in result.all_elements:
        # Check main scores
        assert 0.0 <= element_id.score <= 1.0
        assert 0.0 <= element_id.confidence <= 1.0

        # Check metadata scores
        metadata = element_id.metadata
        if "k_sim" in metadata:
            assert 0.0 <= metadata["k_sim"] <= 1.0
        if "k_rate" in metadata:
            assert 0.0 <= metadata["k_rate"] <= 1.0
        if "k_shift" in metadata:
            assert 0.0 <= metadata["k_shift"] <= 1.0
        if "k_det" in metadata:
            assert 0.0 <= metadata["k_det"] <= 1.0
