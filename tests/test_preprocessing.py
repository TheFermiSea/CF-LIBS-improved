"""Tests for shared preprocessing module."""
import numpy as np
from cflibs.inversion.preprocessing import (
    estimate_baseline,
    estimate_noise,
    detect_peaks,
    robust_normalize,
)


class TestEstimateBaseline:
    def test_removes_sloped_continuum(self):
        wavelength = np.linspace(200, 400, 2000)
        # Sloped baseline + sharp peaks
        baseline_true = 100 + 0.5 * wavelength
        peaks = np.zeros_like(wavelength)
        peaks[500] = 1000  # sharp peak
        peaks[1000] = 800
        peaks[1500] = 600
        intensity = baseline_true + peaks

        result = estimate_baseline(wavelength, intensity)
        # Baseline should track the slope, not the peaks
        assert np.corrcoef(result, baseline_true)[0, 1] > 0.95

    def test_flat_spectrum(self):
        wavelength = np.linspace(200, 400, 1000)
        intensity = np.full_like(wavelength, 500.0)
        result = estimate_baseline(wavelength, intensity)
        np.testing.assert_allclose(result, 500.0, atol=1.0)


class TestEstimateNoise:
    def test_sigma_clipped_lower_than_raw(self):
        rng = np.random.default_rng(42)
        n = 5000
        wavelength = np.linspace(200, 400, n)
        true_noise = 10.0
        baseline = np.full(n, 500.0)
        noise = rng.normal(0, true_noise, n)
        # Add peaks that would inflate raw MAD
        peaks = np.zeros(n)
        for i in range(20):
            idx = rng.integers(0, n)
            peaks[idx] = rng.uniform(200, 1000)
        intensity = baseline + noise + peaks

        sigma = estimate_noise(intensity, baseline)
        raw_mad = np.median(np.abs(intensity - baseline - np.median(intensity - baseline))) * 1.4826
        assert sigma < raw_mad
        assert abs(sigma - true_noise) < 5.0  # within 5 of true

    def test_pure_noise(self):
        rng = np.random.default_rng(123)
        true_noise = 15.0
        baseline = np.zeros(3000)
        intensity = baseline + rng.normal(0, true_noise, 3000)
        sigma = estimate_noise(intensity, baseline)
        assert abs(sigma - true_noise) < 3.0


class TestDetectPeaks:
    def test_finds_known_peaks(self):
        rng = np.random.default_rng(42)
        wavelength = np.linspace(200, 400, 2000)
        baseline = np.full(2000, 100.0)
        noise_level = 10.0
        noise = rng.normal(0, noise_level, 2000)

        # Add peaks at known locations with SNR ~10
        peak_locations = [300, 600, 900, 1200, 1500]
        peak_wavelengths = [wavelength[i] for i in peak_locations]
        signal = np.zeros(2000)
        for loc in peak_locations:
            signal[loc - 2 : loc + 3] = 100.0  # SNR = 100/10 = 10

        intensity = baseline + signal + noise
        peaks = detect_peaks(wavelength, intensity, baseline, noise_level)
        found_indices = [p[0] for p in peaks]

        for loc in peak_locations:
            assert any(abs(f - loc) <= 3 for f in found_indices), f"Peak at {loc} not found"

    def test_no_peaks_in_noise(self):
        rng = np.random.default_rng(42)
        wavelength = np.linspace(200, 400, 2000)
        baseline = np.full(2000, 100.0)
        noise_level = 10.0
        intensity = baseline + rng.normal(0, noise_level, 2000)

        peaks = detect_peaks(wavelength, intensity, baseline, noise_level)
        # With threshold_factor=4, very few false peaks expected
        assert len(peaks) < 5


class TestRobustNormalize:
    def test_basic(self):
        intensity = np.array([0, 25, 50, 75, 100], dtype=float)
        result = robust_normalize(intensity, percentile=95.0)
        p95 = np.percentile(intensity, 95.0)
        np.testing.assert_allclose(result, intensity / p95)

    def test_cosmic_ray_robustness(self):
        rng = np.random.default_rng(42)
        intensity = rng.uniform(10, 100, 1000)
        # Add cosmic ray spike
        intensity[500] = 100000.0

        result = robust_normalize(intensity, percentile=95.0)
        # Most values should be near 0-1 range, not compressed to ~0
        non_spike = np.delete(result, 500)
        assert np.median(non_spike) > 0.3
        assert np.median(non_spike) < 2.0

    def test_zero_spectrum(self):
        intensity = np.zeros(100)
        result = robust_normalize(intensity)
        np.testing.assert_array_equal(result, np.zeros(100))
