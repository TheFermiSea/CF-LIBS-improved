import numpy as np

from cflibs.inversion.identify.line_detection import detect_line_observations


def _gaussian(wavelength: np.ndarray, center: float, sigma: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)


def test_detect_line_observations_matches_known_lines(atomic_db):
    wavelength = np.linspace(371.0, 375.0, 2000)
    intensity = _gaussian(wavelength, 371.99, 0.03, 10.0) + _gaussian(wavelength, 373.49, 0.03, 5.0)

    result = detect_line_observations(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=["Fe"],
        wavelength_tolerance_nm=0.05,
        min_peak_height=0.05,
        peak_width_nm=0.15,
    )

    assert len(result.observations) >= 2

    matched_wavelengths = [obs.wavelength_nm for obs in result.observations]
    assert any(abs(wl - 371.99) < 0.05 for wl in matched_wavelengths)
    assert any(abs(wl - 373.49) < 0.05 for wl in matched_wavelengths)
    assert result.resonance_lines
