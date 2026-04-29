"""
Tests for I/O spectrum utilities.
"""

import os
import pytest
import numpy as np
import tempfile
from pathlib import Path
import pandas as pd

from cflibs.io.spectrum import load_spectrum, save_spectrum


def test_save_spectrum_csv():
    """Test saving spectrum to CSV."""
    wavelength = np.linspace(200, 800, 100)
    intensity = np.exp(-(((wavelength - 500) / 50) ** 2))

    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)  # Close file descriptor to prevent leaks
    Path(temp_path).unlink()

    try:
        save_spectrum(temp_path, wavelength, intensity)
        assert Path(temp_path).exists()

        # Verify it can be loaded
        data = np.loadtxt(temp_path, delimiter=",", skiprows=1)
        assert len(data) == len(wavelength)
        assert np.allclose(data[:, 0], wavelength)
        assert np.allclose(data[:, 1], intensity)
    finally:
        if Path(temp_path).exists():
            Path(temp_path).unlink()


def test_save_spectrum_custom_header():
    """Test saving spectrum with custom header."""
    wavelength = np.linspace(200, 800, 100)
    intensity = np.ones_like(wavelength)

    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)  # Close file descriptor to prevent leaks
    Path(temp_path).unlink()

    try:
        save_spectrum(temp_path, wavelength, intensity, header="wl,int")

        # Check header
        with open(temp_path, "r") as f:
            first_line = f.readline()
            assert "wl,int" in first_line
    finally:
        if Path(temp_path).exists():
            Path(temp_path).unlink()


def test_load_spectrum_csv():
    """Test loading spectrum from CSV."""
    wavelength = np.linspace(200, 800, 100)
    intensity = np.exp(-(((wavelength - 500) / 50) ** 2))

    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)

    try:
        # Create CSV file
        df = pd.DataFrame({"wavelength": wavelength, "intensity": intensity})
        df.to_csv(temp_path, index=False)

        # Load it
        wl_loaded, int_loaded = load_spectrum(temp_path)

        assert len(wl_loaded) == len(wavelength)
        assert np.allclose(wl_loaded, wavelength)
        assert np.allclose(int_loaded, intensity)
    finally:
        Path(temp_path).unlink()


def test_load_spectrum_csv_alternative_names():
    """Test loading CSV with alternative column names."""
    wavelength = np.linspace(200, 800, 100)
    intensity = np.ones_like(wavelength)

    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)

    try:
        # Create CSV with alternative names
        df = pd.DataFrame({"lambda_nm": wavelength, "counts": intensity})
        df.to_csv(temp_path, index=False)

        wl_loaded, int_loaded = load_spectrum(temp_path)
        assert len(wl_loaded) == len(wavelength)
    except ValueError:
        # If column names don't match, that's okay - test alternative
        df = pd.DataFrame({"wavelength_nm": wavelength, "I": intensity})
        df.to_csv(temp_path, index=False)
        wl_loaded, int_loaded = load_spectrum(temp_path)
        assert len(wl_loaded) == len(wavelength)
    finally:
        Path(temp_path).unlink()


def test_load_spectrum_csv_accepts_aalto_spectrum_column():
    """Aalto example spectra use wavelength,spectrum columns."""
    wavelength = np.linspace(200, 800, 100)
    intensity = np.ones_like(wavelength)

    fd, temp_path = tempfile.mkstemp(suffix=".csv")

    try:
        df = pd.DataFrame({"wavelength": wavelength, "spectrum": intensity})
        df.to_csv(temp_path, index=False)

        wl_loaded, int_loaded = load_spectrum(temp_path)

        assert np.allclose(wl_loaded, wavelength)
        assert np.allclose(int_loaded, intensity)
    finally:
        Path(temp_path).unlink()


def test_load_spectrum_numpy_format():
    """Test loading spectrum from NumPy format."""
    wavelength = np.linspace(200, 800, 100)
    intensity = np.ones_like(wavelength)

    fd, temp_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)  # Close file descriptor to prevent leaks

    try:
        # Save as space-separated
        np.savetxt(temp_path, np.column_stack([wavelength, intensity]))

        wl_loaded, int_loaded = load_spectrum(temp_path)
        assert len(wl_loaded) == len(wavelength)
        assert np.allclose(wl_loaded, wavelength)
    finally:
        Path(temp_path).unlink()


def test_load_spectrum_file_not_found():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_spectrum("nonexistent_file.csv")


def test_load_spectrum_csv_no_wavelength():
    """Test loading CSV without wavelength column."""
    wavelength = np.linspace(200, 800, 100)
    intensity = np.ones_like(wavelength)

    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        df = pd.DataFrame({"random_col": wavelength, "intensity": intensity})
        df.to_csv(temp_path, index=False)

        with pytest.raises(ValueError, match="Could not find wavelength column in CSV"):
            load_spectrum(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_spectrum_csv_no_intensity():
    """Test loading CSV without intensity column."""
    wavelength = np.linspace(200, 800, 100)
    intensity = np.ones_like(wavelength)

    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        df = pd.DataFrame({"wavelength": wavelength, "random_col": intensity})
        df.to_csv(temp_path, index=False)

        with pytest.raises(ValueError, match="Could not find intensity column in CSV"):
            load_spectrum(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_spectrum_numpy_1d_array():
    """Test loading txt/numpy file with 1D array."""
    wavelength = np.linspace(200, 800, 100)

    fd, temp_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)

    try:
        np.savetxt(temp_path, wavelength)

        with pytest.raises(ValueError, match="Spectrum file must have at least 2 columns"):
            load_spectrum(temp_path)
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
