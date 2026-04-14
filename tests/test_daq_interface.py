import numpy as np
import pytest

from cflibs.inversion.runtime.daq_interface import process_spectrum


def test_process_spectrum_happy_path():
    """Test process_spectrum with valid input arrays."""
    wavelength = np.array([200.0, 250.0, 300.0, 350.0])
    intensity = np.array([10.0, 50.0, 100.0, 20.0])

    result = process_spectrum(wavelength, intensity)

    assert result["status"] == "success"
    assert result["metrics"]["max_intensity"] == 100.0
    assert result["metrics"]["peak_wavelength"] == 300.0
    assert result["plasma_parameters"]["temperature_K"] == 12000.0
    assert result["plasma_parameters"]["electron_density_cm3"] == 1e17


def test_process_spectrum_types():
    """Test process_spectrum returns native python floats for JSON serialization."""
    wavelength = np.array([200.0, 250.0, 300.0, 350.0])
    intensity = np.array([10.0, 50.0, 100.0, 20.0])

    result = process_spectrum(wavelength, intensity)

    assert type(result["metrics"]["max_intensity"]) is float
    assert type(result["metrics"]["peak_wavelength"]) is float


def test_process_spectrum_empty():
    """Test process_spectrum raises ValueError on empty input."""
    wavelength = np.array([])
    intensity = np.array([])

    with pytest.raises(ValueError):
        process_spectrum(wavelength, intensity)


def test_shim_exports_process_spectrum():
    """Test the backward compatibility shim exports the function."""
    import cflibs.inversion.daq_interface as shim

    assert hasattr(shim, "process_spectrum")
    assert shim.process_spectrum is process_spectrum
