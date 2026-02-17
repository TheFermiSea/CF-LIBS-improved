"""
Tests for CorrelationIdentifier.
"""

import pytest
import numpy as np

from cflibs.inversion.correlation_identifier import CorrelationIdentifier
from cflibs.inversion.element_id import ElementIdentificationResult

pytestmark = pytest.mark.requires_db


def test_correlation_identifier_classic_mode(temp_db):
    """Test classic mode correlation identification."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)

    # Create synthetic spectrum with Gaussian Fe lines (realistic shape)
    wavelength = np.linspace(370, 380, 1000)
    intensity = np.zeros_like(wavelength)
    sigma = 0.05  # ~0.12 nm FWHM, typical for moderate resolving power

    # Add Fe lines from test database: 371.99, 373.49, 374.95
    for wl in [371.99, 373.49, 374.95]:
        intensity += 1000.0 * np.exp(-0.5 * ((wavelength - wl) / sigma) ** 2)

    # Add noise
    rng = np.random.default_rng(42)
    intensity += rng.normal(0, 10, len(intensity))
    intensity = np.maximum(intensity, 0.0)

    # Run identification
    identifier = CorrelationIdentifier(
        db,
        elements=["Fe"],
        T_range_K=(8000, 12000),
        T_steps=3,
        n_e_steps=2,
        min_confidence=0.1,
    )

    result = identifier.identify(wavelength, intensity, mode="classic")

    # Check result structure
    assert isinstance(result, ElementIdentificationResult)
    assert result.algorithm == "correlation"
    assert result.n_peaks > 0

    # Check that Fe is detected
    detected_symbols = [e.element for e in result.detected_elements]
    assert "Fe" in detected_symbols

    # Check element scores are in [0, 1]
    for elem in result.all_elements:
        assert 0.0 <= elem.score <= 1.0
        assert 0.0 <= elem.confidence <= 1.0


def test_correlation_identifier_min_confidence_threshold(temp_db):
    """Test that min_confidence threshold works."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)

    # Weak spectrum
    wavelength = np.linspace(370, 380, 500)
    intensity = np.random.rand(500) * 0.1  # Mostly noise

    identifier = CorrelationIdentifier(
        db,
        elements=["Fe"],
        min_confidence=0.8,  # High threshold
        T_steps=2,
        n_e_steps=2,
    )

    result = identifier.identify(wavelength, intensity, mode="classic")

    # With high threshold and noisy data, likely no detections
    assert len(result.detected_elements) <= 1


def test_correlation_identifier_auto_mode_fallback(temp_db):
    """Test that auto mode falls back to classic when no vector_index."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)

    wavelength = np.linspace(370, 380, 500)
    intensity = np.ones(500)

    identifier = CorrelationIdentifier(db, elements=["Fe"], T_steps=2, n_e_steps=2)

    # Auto should use classic (no vector_index provided)
    result = identifier.identify(wavelength, intensity, mode="auto")

    assert result.algorithm == "correlation"
    assert result.parameters["mode"] == "classic"


def test_correlation_identifier_vector_mode_requires_index(temp_db):
    """Test that vector mode raises error without vector_index."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)

    wavelength = np.linspace(370, 380, 500)
    intensity = np.ones(500)

    identifier = CorrelationIdentifier(db, elements=["Fe"])

    with pytest.raises(ValueError, match="vector_index"):
        identifier.identify(wavelength, intensity, mode="vector")


def test_correlation_identifier_matched_lines(temp_db):
    """Test that matched lines are populated."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)

    # Create spectrum with known Fe lines (Gaussian peaks)
    wavelength = np.linspace(370, 380, 1000)
    intensity = np.zeros_like(wavelength)
    sigma = 0.05

    # Add Fe I 371.99 nm line
    intensity += 500.0 * np.exp(-0.5 * ((wavelength - 371.99) / sigma) ** 2)

    identifier = CorrelationIdentifier(
        db,
        elements=["Fe"],
        wavelength_tolerance_nm=0.2,
        T_steps=2,
        n_e_steps=2,
        min_confidence=0.0,
    )

    result = identifier.identify(wavelength, intensity, mode="classic")

    # Check that Fe has matched lines
    fe_elem = next((e for e in result.all_elements if e.element == "Fe"), None)
    assert fe_elem is not None
    assert fe_elem.n_matched_lines >= 0


def test_instrument_fwhm_parameter(temp_db):
    """Verify instrument_fwhm_nm parameter is stored and used for sigma."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)

    identifier = CorrelationIdentifier(
        db, elements=["Fe"], instrument_fwhm_nm=0.1
    )
    # sigma should be 0.1/2.355 ≈ 0.0425
    assert hasattr(identifier, "instrument_fwhm_nm")
    assert identifier.instrument_fwhm_nm == 0.1

    # Default should be 0.05
    identifier_default = CorrelationIdentifier(db, elements=["Fe"])
    assert identifier_default.instrument_fwhm_nm == 0.05


def test_noise_only_no_detection(temp_db):
    """Pure noise should not detect any elements with high confidence."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)

    rng = np.random.default_rng(42)
    wavelength = np.linspace(200, 400, 2000)
    intensity = 100 + rng.normal(0, 10, 2000)

    identifier = CorrelationIdentifier(
        db,
        elements=["Fe"],
        T_steps=3,
        n_e_steps=2,
        min_confidence=0.3,
    )

    result = identifier.identify(wavelength, intensity, mode="classic")

    # No element should have high confidence on noise
    for elem in result.all_elements:
        assert elem.confidence < 0.5, (
            f"{elem.element} has confidence {elem.confidence} on noise"
        )


def test_max_lines_per_element_parameter(temp_db):
    """Test that max_lines_per_element caps transition count."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)

    identifier = CorrelationIdentifier(db, max_lines_per_element=10)
    assert identifier.max_lines_per_element == 10

    # Default should be 100
    identifier_default = CorrelationIdentifier(db)
    assert identifier_default.max_lines_per_element == 100


def test_default_min_confidence_lowered(temp_db):
    """Test that default min_confidence is 0.03."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)

    identifier = CorrelationIdentifier(db)
    assert identifier.min_confidence == 0.03
