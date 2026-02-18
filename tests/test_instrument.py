"""
Tests for instrument modeling.
"""

import os
import pytest
import numpy as np
from pathlib import Path
import tempfile
import yaml

from cflibs.instrument.model import InstrumentModel
from cflibs.instrument.convolution import apply_instrument_function


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def test_instrument_model_creation():
    """Test creating instrument model."""
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    assert instrument.resolution_fwhm_nm == 0.05
    assert instrument.resolution_sigma_nm > 0


def test_instrument_model_resolution_sigma():
    """Test resolution sigma property."""
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    # FWHM = 2.355 * sigma
    expected_sigma = 0.05 / 2.355
    assert instrument.resolution_sigma_nm == pytest.approx(expected_sigma, rel=1e-6)


def test_instrument_model_from_file(temp_config_file):
    """Test loading instrument model from config file."""
    instrument = InstrumentModel.from_file(Path(temp_config_file))
    assert instrument.resolution_fwhm_nm == 0.05


def test_instrument_model_from_file_missing_section():
    """Test loading instrument model from invalid config."""
    config_fd, config_path = tempfile.mkstemp(suffix=".yaml")
    os.close(config_fd)  # Close file descriptor to prevent leaks

    try:
        with open(config_path, "w") as f:
            yaml.dump({"plasma": {}}, f)

        with pytest.raises(ValueError, match="must contain 'instrument' section"):
            InstrumentModel.from_file(Path(config_path))
    finally:
        Path(config_path).unlink()


def test_instrument_model_apply_response_no_curve():
    """Test applying response with no response curve."""
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    wavelength = np.linspace(200, 800, 100)
    intensity = np.ones_like(wavelength)

    result = instrument.apply_response(wavelength, intensity)
    assert np.allclose(result, intensity)


def test_instrument_model_apply_response_with_curve():
    """Test applying response curve."""
    # Create mock response curve
    response_curve = np.array([[200.0, 0.5], [500.0, 1.0], [800.0, 0.8]])

    instrument = InstrumentModel(resolution_fwhm_nm=0.05, response_curve=response_curve)

    wavelength = np.linspace(200, 800, 100)
    intensity = np.ones_like(wavelength)

    result = instrument.apply_response(wavelength, intensity)
    assert len(result) == len(intensity)
    assert np.all(result >= 0)
    # Response should modify intensity
    assert not np.allclose(result, intensity)


def test_apply_instrument_function():
    """Test applying instrument function."""
    wavelength = np.linspace(200, 800, 1000)  # Evenly spaced
    intensity = np.zeros_like(wavelength)
    intensity[500] = 1.0  # Single peak

    sigma = 2.0
    convolved = apply_instrument_function(wavelength, intensity, sigma)

    assert len(convolved) == len(intensity)
    assert np.all(convolved >= 0)
    # Peak should be broadened
    assert convolved[500] < 1.0
    # But should still be the maximum
    assert np.argmax(convolved) == 500


def test_apply_instrument_function_jax():
    """Test applying instrument function with JAX."""
    try:
        import jax  # noqa: F401
    except ImportError:
        pytest.skip("JAX not installed")

    from cflibs.instrument.convolution import apply_instrument_function_jax

    wavelength = np.linspace(200, 800, 1000)
    intensity = np.zeros_like(wavelength)
    intensity[500] = 1.0

    sigma = 2.0
    convolved = apply_instrument_function_jax(wavelength, intensity, sigma)

    assert len(convolved) == len(intensity)
    assert np.all(convolved >= 0)
    assert convolved[500] < 1.0
    assert np.argmax(convolved) == 500


def test_apply_instrument_function_non_uniform():
    """Test that non-uniform grids raise error."""
    wavelength = np.array([200, 201, 203, 206, 210])  # Non-uniform
    intensity = np.ones_like(wavelength)

    with pytest.raises(ValueError, match="evenly spaced"):
        apply_instrument_function(wavelength, intensity, 0.05)


def test_apply_instrument_function_preserves_integral():
    """Test that convolution approximately preserves integral."""
    wavelength = np.linspace(200, 800, 1000)
    intensity = np.exp(-(((wavelength - 500) / 10) ** 2))  # Gaussian peak

    original_integral = _trapezoid(intensity, wavelength)
    convolved = apply_instrument_function(wavelength, intensity, 0.05)
    convolved_integral = _trapezoid(convolved, wavelength)

    # Integrals should be approximately equal (within 5%)
    assert abs(convolved_integral - original_integral) / original_integral < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
