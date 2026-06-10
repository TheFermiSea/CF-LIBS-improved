"""Tests for Voigt deconvolution module."""

import pytest
import numpy as np

from cflibs.inversion.preprocess.deconvolution import (
    VoigtFitResult,
    DeconvolutionResult,
    group_peaks,
    deconvolve_peaks,
)
from cflibs.radiation.profiles import voigt_profile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synth_voigt_spectrum(
    wavelength: np.ndarray,
    centers: list,
    amplitudes: list,
    sigma: float = 0.03,
    gamma: float = 0.02,
    noise_std: float = 0.0,
    rng_seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic spectrum from known Voigt peaks."""
    spectrum = np.zeros_like(wavelength)
    for c, a in zip(centers, amplitudes):
        spectrum += voigt_profile(wavelength, c, sigma, gamma, a)
    if noise_std > 0:
        rng = np.random.default_rng(rng_seed)
        spectrum += rng.normal(0, noise_std, len(wavelength))
    return spectrum


# ---------------------------------------------------------------------------
# group_peaks tests
# ---------------------------------------------------------------------------


class TestGroupPeaks:
    def test_empty(self):
        assert group_peaks(np.array([]), fwhm_estimate=0.1) == []

    def test_single_peak(self):
        groups = group_peaks(np.array([300.0]), fwhm_estimate=0.1)
        assert groups == [[0]]

    def test_well_separated(self):
        wl = np.array([300.0, 310.0, 320.0])
        groups = group_peaks(wl, fwhm_estimate=0.1, grouping_factor=2.0)
        assert len(groups) == 3

    def test_close_peaks_merged(self):
        wl = np.array([300.0, 300.15, 310.0])
        groups = group_peaks(wl, fwhm_estimate=0.1, grouping_factor=2.0)
        # First two should be grouped together
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 1

    def test_all_merged(self):
        wl = np.array([300.0, 300.05, 300.10])
        groups = group_peaks(wl, fwhm_estimate=0.1, grouping_factor=2.0)
        assert len(groups) == 1
        assert len(groups[0]) == 3


# ---------------------------------------------------------------------------
# deconvolve_peaks tests — SciPy fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeconvolveSciPy:
    """Tests using the SciPy (curve_fit) backend."""

    def test_single_isolated_peak(self):
        """Single peak recovery: center and amplitude within 5%."""
        wavelength = np.linspace(299, 301, 500)
        true_center = 300.0
        true_amp = 1000.0
        sigma = 0.03
        gamma = 0.02

        intensity = _synth_voigt_spectrum(
            wavelength, [true_center], [true_amp], sigma=sigma, gamma=gamma
        )
        peak_wl = np.array([true_center])
        fwhm_est = 0.12  # rough FWHM estimate

        result = deconvolve_peaks(
            wavelength, intensity, peak_wl, fwhm_estimate=fwhm_est, use_jax=False
        )

        assert isinstance(result, DeconvolutionResult)
        assert len(result.fit_results) == 1

        fr = result.fit_results[0]
        assert (
            abs(fr.center_nm - true_center) / true_center < 0.05
        ), f"Center off: {fr.center_nm:.4f} vs {true_center}"
        assert (
            abs(fr.area - true_amp) / true_amp < 0.10
        ), f"Amplitude off: {fr.area:.1f} vs {true_amp}"

    def test_two_overlapping_peaks(self):
        """Two peaks separated by 1.5 FWHM should be resolved."""
        sigma = 0.03
        gamma = 0.02
        # Approximate FWHM for these widths (Voigt ~0.10 nm)
        fwhm_est = 0.10
        separation = 1.5 * fwhm_est

        center1 = 300.0
        center2 = center1 + separation
        amp1 = 800.0
        amp2 = 600.0

        wavelength = np.linspace(299.5, 301.0, 800)
        intensity = _synth_voigt_spectrum(
            wavelength,
            [center1, center2],
            [amp1, amp2],
            sigma=sigma,
            gamma=gamma,
        )
        peak_wl = np.array([center1, center2])

        result = deconvolve_peaks(
            wavelength, intensity, peak_wl, fwhm_estimate=fwhm_est, use_jax=False
        )

        assert len(result.fit_results) == 2

        # Sort by center for consistent comparison
        fits = sorted(result.fit_results, key=lambda f: f.center_nm)
        assert abs(fits[0].center_nm - center1) < 0.02
        assert abs(fits[1].center_nm - center2) < 0.02
        assert abs(fits[0].area - amp1) / amp1 < 0.15
        assert abs(fits[1].area - amp2) / amp2 < 0.15

    def test_scipy_fallback_produces_results(self):
        """SciPy backend should produce reasonable results."""
        wavelength = np.linspace(299, 301, 300)
        intensity = _synth_voigt_spectrum(wavelength, [300.0], [500.0], sigma=0.03, gamma=0.02)
        result = deconvolve_peaks(
            wavelength,
            intensity,
            np.array([300.0]),
            fwhm_estimate=0.1,
            use_jax=False,
        )
        assert len(result.fit_results) == 1
        assert result.fit_results[0].area > 0
        assert result.rms_residual < 50.0

    def test_empty_wavelength_array(self):
        """Edge case: empty arrays should return empty result."""
        result = deconvolve_peaks(
            np.array([]),
            np.array([]),
            np.array([300.0]),
            fwhm_estimate=0.1,
        )
        assert len(result.fit_results) == 0
        assert result.rms_residual == 0.0

    def test_empty_peak_array(self):
        """Edge case: no peaks should return empty result."""
        wavelength = np.linspace(299, 301, 100)
        intensity = np.ones_like(wavelength)
        result = deconvolve_peaks(
            wavelength,
            intensity,
            np.array([]),
            fwhm_estimate=0.1,
        )
        assert len(result.fit_results) == 0

    def test_noisy_spectrum(self):
        """Fitting should still work with moderate noise."""
        wavelength = np.linspace(299, 301, 500)
        true_amp = 1000.0
        intensity = _synth_voigt_spectrum(
            wavelength,
            [300.0],
            [true_amp],
            sigma=0.03,
            gamma=0.02,
            noise_std=5.0,
        )
        result = deconvolve_peaks(
            wavelength,
            intensity,
            np.array([300.0]),
            fwhm_estimate=0.1,
            use_jax=False,
        )
        assert len(result.fit_results) == 1
        # Allow 10% tolerance due to noise
        assert abs(result.fit_results[0].area - true_amp) / true_amp < 0.10


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_voigt_fit_result(self):
        r = VoigtFitResult(
            center_nm=300.0,
            amplitude=100.0,
            sigma_gaussian=0.03,
            gamma_lorentzian=0.02,
            area=100.0,
            area_uncertainty=5.0,
            residual_rms=1.0,
        )
        assert r.center_nm == 300.0
        assert r.area == 100.0

    def test_deconvolution_result(self):
        r = DeconvolutionResult(
            fit_results=[],
            wavelength=np.array([1.0, 2.0]),
            fitted_spectrum=np.array([0.0, 0.0]),
            residuals=np.array([1.0, 2.0]),
            rms_residual=1.5,
        )
        assert r.rms_residual == 1.5
        assert len(r.fit_results) == 0
