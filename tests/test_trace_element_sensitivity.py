import numpy as np
import pytest
from cflibs.inversion.preprocess.preprocessing import (
    detect_peaks_auto,
    BaselineMethod,
)

def test_low_snr_peak_preservation():
    """Test that ALS/Percentile baseline preserves weak peaks that Median clips."""
    # Create synthetic spectrum: flat baseline + noise + one weak peak
    x = np.linspace(300, 310, 1000)
    # Baseline with some curvature
    base = 10.0 + 2.0 * np.sin(x / 5.0)
    # Noise
    np.random.seed(42)
    noise_sigma = 0.5
    noise = np.random.normal(0, noise_sigma, len(x))
    
    # Weak peak: height 2.5 (SNR = 5 relative to noise, but might be clipped by baseline)
    peak_pos = 305.0
    peak_idx = np.argmin(np.abs(x - peak_pos))
    peak = 2.5 * np.exp(-((x - peak_pos) ** 2) / (2 * 0.05**2))
    
    intensity = base + noise + peak
    
    # 1. Detect with Median (current default before change)
    peaks_med, _, _ = detect_peaks_auto(
        x, intensity, baseline_method=BaselineMethod.MEDIAN, threshold_factor=3.0
    )
    
    # 2. Detect with ALS (new default)
    peaks_als, _, _ = detect_peaks_auto(
        x, intensity, baseline_method=BaselineMethod.ALS, threshold_factor=3.0
    )
    
    # 3. Detect with Percentile
    peaks_perc, _, _ = detect_peaks_auto(
        x, intensity, baseline_method=BaselineMethod.PERCENTILE, 
        baseline_percentile=10.0, threshold_factor=3.0
    )
    
    # Check if peak at 305.0 is found
    def is_found(peaks, target_wl):
        return any(abs(p[1] - target_wl) < 0.1 for p in peaks)
    
    # In this synthetic case, we expect ALS and Percentile to be more sensitive
    # than Median if the window is large or if the noise is structured.
    assert is_found(peaks_als, 305.0), "ALS should find the weak peak"
    assert is_found(peaks_perc, 305.0), "Percentile should find the weak peak"

def test_min_intensity_floor():
    """Test that min_intensity_floor correctly gates peak detection."""
    x = np.linspace(300, 310, 1000)
    intensity = np.ones_like(x) * 10.0
    # Add a peak with height 5.0 above baseline
    intensity[500] = 15.0
    
    # Noise is zero, so noise-based threshold is zero (or floor)
    # If we set min_intensity_floor=10.0, it should NOT find the peak (height 5.0 < 10.0)
    # If we set min_intensity_floor=2.0, it SHOULD find the peak (height 5.0 > 2.0)
    
    peaks_high, _, _ = detect_peaks_auto(
        x, intensity, min_intensity_floor=10.0, threshold_factor=3.0
    )
    assert len(peaks_high) == 0
    
    peaks_low, _, _ = detect_peaks_auto(
        x, intensity, min_intensity_floor=2.0, threshold_factor=3.0
    )
    assert len(peaks_low) == 1
