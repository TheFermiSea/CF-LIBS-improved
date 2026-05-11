import numpy as np
import pytest
from cflibs.inversion.preprocess.preprocessing import detect_peaks_auto, BaselineMethod

def test_als_preserves_weak_peaks():
    """Verify that ALS baseline preserves weak peaks that median filter might clip."""
    # Create a synthetic spectrum with a broad baseline and a weak peak
    x = np.linspace(200, 300, 1000)
    # Broad Gaussian baseline
    baseline_true = 10 * np.exp(-((x - 250) / 50)**2)
    # Weak peak at 220 nm
    peak_weak = 0.2 * np.exp(-((x - 220) / 0.1)**2)
    # Strong peak at 280 nm
    peak_strong = 5.0 * np.exp(-((x - 280) / 0.1)**2)
    
    # Use a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.05, x.size)
    intensity = baseline_true + peak_weak + peak_strong + noise
    
    # Detect peaks with ALS (new default)
    peaks_als, _, _ = detect_peaks_auto(
        x, intensity, baseline_method=BaselineMethod.ALS, threshold_factor=3.0
    )
    
    als_wavelengths = [p[1] for p in peaks_als]
    
    # Check if 220nm is found by ALS
    found_als = any(abs(w - 220) < 0.5 for w in als_wavelengths)
    assert found_als, "ALS should find the weak peak at 220nm"

def test_min_intensity_floor():
    """Verify that min_intensity_floor allows detecting peaks below noise threshold."""
    x = np.linspace(200, 300, 1000)
    # High noise
    noise_sigma = 0.5
    rng = np.random.RandomState(42)
    noise = rng.normal(0, noise_sigma, x.size)
    # Weak peak that is below 4*sigma (2.0) but above a floor (e.g. 1.0)
    peak = 1.2 * np.exp(-((x - 250) / 0.1)**2)
    intensity = peak + noise
    
    # Without floor, threshold is ~4 * 0.5 = 2.0. Peak at 1.2 should be missed.
    peaks_no_floor, _, _ = detect_peaks_auto(
        x, intensity, threshold_factor=4.0, min_intensity_floor=0.0
    )
    
    # With floor at 1.0, it should be found.
    peaks_with_floor, _, _ = detect_peaks_auto(
        x, intensity, threshold_factor=4.0, min_intensity_floor=1.0
    )
    
    assert len(peaks_no_floor) == 0
    assert len(peaks_with_floor) > 0
    assert any(abs(p[1] - 250) < 0.5 for p in peaks_with_floor)

def test_reproducibility():
    """Verify that peak detection is reproducible across runs."""
    x = np.linspace(200, 300, 1000)
    intensity = np.sin(x) + 1.0
    
    peaks1, b1, n1 = detect_peaks_auto(x, intensity)
    peaks2, b2, n2 = detect_peaks_auto(x, intensity)
    
    assert peaks1 == peaks2
    np.testing.assert_array_equal(b1, b2)
    assert n1 == n2
