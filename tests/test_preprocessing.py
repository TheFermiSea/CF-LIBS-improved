"""Tests for shared preprocessing module."""

import pytest
import numpy as np
from cflibs.inversion.preprocessing import (
    estimate_baseline,
    estimate_baseline_snip,
    estimate_baseline_als,
    estimate_noise,
    detect_peaks,
    detect_peaks_auto,
    robust_normalize,
    BaselineMethod,
)

# ---------------------------------------------------------------------------
# Existing baseline / noise / peak / normalize tests
# ---------------------------------------------------------------------------


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
        np.linspace(200, 400, n)
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

    def test_zero_noise_floor(self):
        # Case where intensity and baseline are identical
        n = 1000
        baseline = np.full(n, 100.0)
        intensity = baseline.copy()
        sigma = estimate_noise(intensity, baseline)
        assert sigma >= 1e-10
        # When noise is 0, it uses 95th percentile of abs(diff) * 1e-6, floored at 1e-10
        # here diff is 0, so it should be exactly 1e-10
        assert sigma == 1e-10

    def test_small_array_break(self):
        # Case where number of points is initially small (< 10)
        # The loop should break immediately if mask count < 10
        n = 8
        baseline = np.zeros(n)
        intensity = np.arange(n, dtype=float)
        sigma = estimate_noise(intensity, baseline)
        assert sigma > 0

    def test_extreme_outlier(self):
        # Case where one huge outlier exists
        rng = np.random.default_rng(42)
        n = 1000
        true_noise = 10.0
        baseline = np.zeros(n)
        intensity = rng.normal(0, true_noise, n)
        intensity[500] = 1e6  # Extreme outlier

        sigma = estimate_noise(intensity, baseline)
        # Should ignore the outlier and get close to true noise
        assert abs(sigma - true_noise) < 3.0

    def test_constant_offset(self):
        # Case where there's a constant offset between intensity and baseline
        rng = np.random.default_rng(42)
        n = 1000
        true_noise = 10.0
        baseline = np.zeros(n)
        intensity = rng.normal(100.0, true_noise, n)  # 100.0 offset

        sigma = estimate_noise(intensity, baseline)
        # MAD should be robust to the constant offset (it subtracts median)
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
        [wavelength[i] for i in peak_locations]
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


# ---------------------------------------------------------------------------
# SNIP baseline tests
# ---------------------------------------------------------------------------


def _make_gaussian_peaks(wavelength, centers, amplitudes, sigma=0.5):
    """Helper: build Gaussian peaks on a wavelength grid."""
    y = np.zeros_like(wavelength)
    for c, a in zip(centers, amplitudes):
        y += a * np.exp(-0.5 * ((wavelength - c) / sigma) ** 2)
    return y


@pytest.mark.unit
class TestSNIPBaseline:
    def test_recovers_flat_baseline_under_peaks(self):
        wavelength = np.linspace(200, 400, 2000)
        flat_baseline = 100.0
        peaks = _make_gaussian_peaks(
            wavelength, centers=[250, 300, 350], amplitudes=[500, 800, 600]
        )
        intensity = flat_baseline + peaks

        result = estimate_baseline_snip(wavelength, intensity, num_iterations=40)

        # The SNIP baseline should be close to the true flat baseline
        # where there are no peaks (avoid peak regions)
        non_peak_mask = peaks < 10.0
        np.testing.assert_allclose(
            result[non_peak_mask],
            flat_baseline,
            atol=20.0,
            err_msg="SNIP baseline deviates from true flat baseline in non-peak regions",
        )

    def test_recovers_sloped_baseline(self):
        wavelength = np.linspace(200, 400, 2000)
        true_baseline = 50 + 0.3 * wavelength
        peaks = _make_gaussian_peaks(wavelength, [250, 310, 370], [400, 600, 300])
        intensity = true_baseline + peaks

        result = estimate_baseline_snip(wavelength, intensity, num_iterations=60)

        # Correlation with true baseline should be high
        non_peak_mask = peaks < 10.0
        corr = np.corrcoef(result[non_peak_mask], true_baseline[non_peak_mask])[0, 1]
        assert corr > 0.90, f"SNIP baseline correlation too low: {corr:.3f}"

    def test_short_array(self):
        wavelength = np.array([300.0])
        intensity = np.array([100.0])
        result = estimate_baseline_snip(wavelength, intensity)
        np.testing.assert_array_equal(result, intensity)

    def test_constant_array(self):
        wavelength = np.linspace(200, 400, 500)
        intensity = np.full_like(wavelength, 42.0)
        result = estimate_baseline_snip(wavelength, intensity)
        np.testing.assert_allclose(result, 42.0, atol=1.0)

    def test_order_zero_disables_transform(self):
        wavelength = np.linspace(200, 400, 1000)
        intensity = np.full_like(wavelength, 100.0)
        result = estimate_baseline_snip(wavelength, intensity, order=0)
        np.testing.assert_allclose(result, 100.0, atol=1.0)

    def test_smoothing_window(self):
        wavelength = np.linspace(200, 400, 1000)
        # Create spectrum with peaks on a noisy baseline so SNIP produces
        # a non-trivial baseline that benefits from smoothing.
        rng = np.random.default_rng(99)
        baseline = 100.0 + rng.normal(0, 10, 1000)
        peaks = _make_gaussian_peaks(wavelength, [250, 300, 350], [400, 600, 300])
        intensity = np.maximum(baseline + peaks, 0)

        result_smooth = estimate_baseline_snip(
            wavelength, intensity, num_iterations=20, smoothing_window=21
        )
        result_raw = estimate_baseline_snip(
            wavelength, intensity, num_iterations=20, smoothing_window=0
        )

        # Both should return valid arrays of the right shape
        assert result_smooth.shape == wavelength.shape
        assert result_raw.shape == wavelength.shape
        # Smoothed baseline should differ from unsmoothed
        assert not np.array_equal(result_smooth, result_raw)


# ---------------------------------------------------------------------------
# ALS baseline tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestALSBaseline:
    def test_converges_and_tracks_slow_continuum(self):
        wavelength = np.linspace(200, 400, 1000)
        true_baseline = 80 + 20 * np.sin(2 * np.pi * wavelength / 200)
        peaks = _make_gaussian_peaks(wavelength, [250, 300, 350], [300, 500, 200])
        intensity = true_baseline + peaks

        result = estimate_baseline_als(wavelength, intensity, lam=1e6, p=0.01)

        # ALS should track the slow continuum, staying below or near the peaks
        non_peak_mask = peaks < 5.0
        residual = np.abs(result[non_peak_mask] - true_baseline[non_peak_mask])
        assert (
            np.mean(residual) < 30.0
        ), f"ALS baseline deviates too much from true continuum: mean={np.mean(residual):.1f}"

    def test_flat_baseline(self):
        wavelength = np.linspace(200, 400, 500)
        intensity = np.full_like(wavelength, 200.0)
        result = estimate_baseline_als(wavelength, intensity)
        np.testing.assert_allclose(result, 200.0, atol=5.0)

    def test_short_array(self):
        wavelength = np.array([300.0, 301.0])
        intensity = np.array([100.0, 110.0])
        result = estimate_baseline_als(wavelength, intensity)
        np.testing.assert_array_equal(result, intensity)

    def test_constant_array(self):
        wavelength = np.linspace(200, 400, 300)
        intensity = np.full_like(wavelength, 55.0)
        result = estimate_baseline_als(wavelength, intensity)
        np.testing.assert_allclose(result, 55.0, atol=2.0)


# ---------------------------------------------------------------------------
# BaselineMethod dispatch tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBaselineMethodDispatch:
    def test_enum_values(self):
        assert BaselineMethod.MEDIAN.value == "median"
        assert BaselineMethod.SNIP.value == "snip"
        assert BaselineMethod.ALS.value == "als"

    def test_median_default(self):
        wavelength = np.linspace(200, 400, 500)
        intensity = np.full_like(wavelength, 100.0)
        peaks_default, bl_default, _ = detect_peaks_auto(wavelength, intensity)
        peaks_median, bl_median, _ = detect_peaks_auto(
            wavelength, intensity, baseline_method=BaselineMethod.MEDIAN
        )
        np.testing.assert_array_equal(bl_default, bl_median)

    def test_snip_dispatch(self):
        wavelength = np.linspace(200, 400, 1000)
        rng = np.random.default_rng(42)
        baseline_true = 100.0
        peaks = _make_gaussian_peaks(wavelength, [300], [500])
        intensity = baseline_true + peaks + rng.normal(0, 5, 1000)

        result_peaks, bl_snip, noise = detect_peaks_auto(
            wavelength, intensity, baseline_method=BaselineMethod.SNIP
        )
        # SNIP should produce a different baseline than median
        _, bl_median, _ = detect_peaks_auto(
            wavelength, intensity, baseline_method=BaselineMethod.MEDIAN
        )
        # They should not be identical (different algorithms)
        assert not np.allclose(bl_snip, bl_median, atol=1e-6)

    def test_als_dispatch(self):
        wavelength = np.linspace(200, 400, 500)
        rng = np.random.default_rng(42)
        intensity = 100.0 + rng.normal(0, 5, 500)

        result_peaks, bl_als, noise = detect_peaks_auto(
            wavelength, intensity, baseline_method=BaselineMethod.ALS
        )
        # ALS should return a valid baseline array
        assert bl_als.shape == wavelength.shape
        assert np.all(np.isfinite(bl_als))

    def test_unknown_baseline_method_raises(self):
        wavelength = np.linspace(200, 400, 500)
        intensity = np.full_like(wavelength, 100.0)
        with pytest.raises(ValueError, match="Unknown baseline_method"):
            detect_peaks_auto(wavelength, intensity, baseline_method="invalid")
