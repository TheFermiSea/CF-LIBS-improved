"""
Integration tests for SpectrumModel.
"""

import pytest
import numpy as np
from cflibs.plasma import SingleZoneLTEPlasma
from cflibs.instrument import InstrumentModel
from cflibs.radiation import SpectrumModel

pytestmark = pytest.mark.requires_db


def test_spectrum_model_init(atomic_db, sample_plasma):
    """Test initializing spectrum model."""
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    model = SpectrumModel(
        plasma=sample_plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=200.0,
        lambda_max=800.0,
        delta_lambda=0.01,
    )

    assert model.plasma == sample_plasma
    assert model.atomic_db == atomic_db
    assert model.instrument == instrument
    assert len(model.wavelength) > 0


def test_spectrum_model_compute_spectrum(atomic_db, sample_plasma):
    """Test computing a complete spectrum."""
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    model = _create_test_model(
        plasma=sample_plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        delta_lambda=0.01,
    )

    wavelength, intensity = model.compute_spectrum()

    assert len(wavelength) == len(intensity)
    assert len(wavelength) > 0
    assert np.all(wavelength >= 370.0)
    assert np.all(wavelength <= 375.0)
    assert np.all(intensity >= 0)


def test_spectrum_model_validation(atomic_db):
    """Test that invalid plasma is caught."""
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    invalid_plasma = SingleZoneLTEPlasma(T_e=-1000.0, n_e=1e17, species={"Fe": 1e15})  # Invalid

    model = SpectrumModel(
        plasma=invalid_plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=200.0,
        lambda_max=800.0,
        delta_lambda=0.01,
    )

    with pytest.raises(ValueError):
        model.compute_spectrum()


def test_spectrum_model_path_length_scaling(atomic_db, sample_plasma):
    """Test that intensity scales with path length."""
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    def get_spectrum(path_length):
        model = _create_test_model(
            plasma=sample_plasma,
            atomic_db=atomic_db,
            instrument=instrument,
            path_length_m=path_length,
        )
        return model.compute_spectrum()

    wl1, I1 = get_spectrum(1e-5)
    wl2, I2 = get_spectrum(2e-5)

    # Intensities should scale with path length
    # (allowing for numerical differences)
    ratio = I2.max() / I1.max() if I1.max() > 0 else 1.0
    assert 1.8 < ratio < 2.2  # Approximately 2x


def test_spectrum_model_temperature_dependence(atomic_db):
    """Test that spectrum changes with temperature."""
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    plasma1 = SingleZoneLTEPlasma(T_e=8000.0, n_e=1e17, species={"Fe": 1e15})

    plasma2 = SingleZoneLTEPlasma(T_e=12000.0, n_e=1e17, species={"Fe": 1e15})

    model1 = _create_test_model(plasma1, atomic_db, instrument)
    model2 = _create_test_model(plasma2, atomic_db, instrument)

    wl1, I1 = model1.compute_spectrum()
    wl2, I2 = model2.compute_spectrum()

    # Spectra should be different
    assert not np.allclose(I1, I2, rtol=0.1)


# --- Broadening Mode Tests ---


def test_spectrum_model_nist_parity_mode(atomic_db, sample_plasma):
    """Test NIST_PARITY broadening mode produces spectrum without convolution."""
    from cflibs.radiation.profiles import BroadeningMode

    instrument = InstrumentModel.from_resolving_power(1000)

    model = _create_test_model(
        plasma=sample_plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        delta_lambda=0.01,
        broadening_mode=BroadeningMode.NIST_PARITY,
    )

    wavelength, intensity = model.compute_spectrum()

    assert len(wavelength) == len(intensity)
    assert np.all(intensity >= 0)


def test_spectrum_model_nist_parity_requires_resolving_power(atomic_db, sample_plasma):
    """Test NIST_PARITY raises error without resolving_power."""
    from cflibs.radiation.profiles import BroadeningMode

    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    with pytest.raises(ValueError, match="resolving_power"):
        _create_test_model(
            plasma=sample_plasma,
            atomic_db=atomic_db,
            instrument=instrument,
            delta_lambda=0.01,
            broadening_mode=BroadeningMode.NIST_PARITY,
        )


def test_spectrum_model_physical_doppler_mode(atomic_db, sample_plasma):
    """Test PHYSICAL_DOPPLER broadening mode produces spectrum."""
    from cflibs.radiation.profiles import BroadeningMode

    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    model = _create_test_model(
        plasma=sample_plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        delta_lambda=0.01,
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
    )

    wavelength, intensity = model.compute_spectrum()

    assert len(wavelength) == len(intensity)
    assert np.all(intensity >= 0)


def _create_test_model(
    plasma,
    atomic_db,
    instrument,
    lambda_min=370.0,
    lambda_max=375.0,
    delta_lambda=0.1,
    **kwargs
):
    """Helper to create SpectrumModel with common test defaults."""
    return SpectrumModel(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        delta_lambda=delta_lambda,
        **kwargs
    )


def test_spectrum_model_legacy_mode_unchanged(atomic_db, sample_plasma):
    """Test LEGACY mode produces identical results to old behavior."""
    from cflibs.radiation.profiles import BroadeningMode

    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    model_default = _create_test_model(sample_plasma, atomic_db, instrument)
    model_legacy = _create_test_model(
        sample_plasma, atomic_db, instrument, broadening_mode=BroadeningMode.LEGACY
    )

    wl1, I1 = model_default.compute_spectrum()
    wl2, I2 = model_legacy.compute_spectrum()

    np.testing.assert_array_equal(wl1, wl2)
    np.testing.assert_allclose(I1, I2, rtol=1e-10)


def test_spectrum_model_nist_parity_vs_legacy_differ(atomic_db, sample_plasma):
    """Test NIST_PARITY and LEGACY produce different spectra.

    Uses the same resolving-power instrument for both modes so the
    difference is purely from the broadening mode, not the instrument.
    """
    from cflibs.radiation.profiles import BroadeningMode

    instrument = InstrumentModel.from_resolving_power(1000)

    model_legacy = _create_test_model(
        sample_plasma, atomic_db, instrument, delta_lambda=0.01, broadening_mode=BroadeningMode.LEGACY
    )
    model_nist = _create_test_model(
        sample_plasma, atomic_db, instrument, delta_lambda=0.01, broadening_mode=BroadeningMode.NIST_PARITY
    )

    _, I_legacy = model_legacy.compute_spectrum()
    _, I_nist = model_nist.compute_spectrum()

    # They should produce different spectra (LEGACY uses scalar sigma +
    # convolution; NIST_PARITY uses per-line sigma, no convolution)
    assert not np.allclose(I_legacy, I_nist, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
