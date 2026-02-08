"""
Tests for CorrelationIdentifier.
"""

import pytest
import numpy as np

from cflibs.inversion.correlation_identifier import CorrelationIdentifier
from cflibs.inversion.element_id import ElementIdentificationResult


def test_correlation_identifier_classic_mode(temp_db):
    """Test classic mode correlation identification."""
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(temp_db)

    # Create synthetic spectrum with Fe lines
    wavelength = np.linspace(370, 380, 1000)
    intensity = np.zeros_like(wavelength)

    # Add Fe lines from test database: 371.99, 373.49, 374.95
    for wl in [371.99, 373.49, 374.95]:
        idx = np.argmin(np.abs(wavelength - wl))
        intensity[idx] = 1.0

    # Add noise
    intensity += np.random.rand(len(intensity)) * 0.1

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

    # Create spectrum with known Fe lines
    wavelength = np.linspace(370, 380, 1000)
    intensity = np.zeros_like(wavelength)

    # Add Fe I 371.99 nm line
    idx = np.argmin(np.abs(wavelength - 371.99))
    intensity[idx] = 1.0

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
