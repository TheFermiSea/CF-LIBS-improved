import numpy as np
from cflibs.inversion.preprocess.preprocessing import detect_peaks_auto, BaselineMethod

def test_detect_peaks_low_snr_sensitivity():
    """Verify that min_intensity_floor allows detecting weak peaks in noise."""
    # Create synthetic spectrum with high noise and one weak peak
    # Peak at 500nm, height 2.0
    # Noise sigma 1.0 -> threshold at factor 4 is 4.0.
    # Without min_intensity_floor, the peak at 2.0 should be missed.
    # With min_intensity_floor=1.5, it should be detected.
    
    np.random.seed(42)
    wavelength = np.linspace(400, 600, 2000)
    
    # Baseline
    baseline_true = 10.0 + 0.01 * wavelength
    
    # Noise
    noise_sigma = 1.0
    noise = np.random.normal(0, noise_sigma, len(wavelength))
    
    # Peak
    peak_pos = 500.0
    peak_idx = np.argmin(np.abs(wavelength - peak_pos))
    intensity_peak = 2.0
    # Lorentzian peak
    peak = intensity_peak / (1 + ((wavelength - peak_pos) / 0.1)**2)
    
    intensity = baseline_true + noise + peak
    
    # 1. Standard detection (default threshold_factor=4.0)
    peaks_std, _, est_noise = detect_peaks_auto(
        wavelength, intensity, threshold_factor=4.0, baseline_method=BaselineMethod.MEDIAN
    )
    
    # Verify standard detection misses the weak peak at 500nm
    peak_found_std = any(abs(p[1] - peak_pos) < 0.5 for p in peaks_std)
    assert not peak_found_std, f"Standard detection should have missed the weak peak (noise={est_noise})"
    
    # 2. Sensitive detection with min_intensity_floor
    peaks_sens, _, _ = detect_peaks_auto(
        wavelength, intensity, threshold_factor=4.0, min_intensity_floor=1.5
    )
    
    # Verify sensitive detection finds it
    peak_found_sens = any(abs(p[1] - peak_pos) < 0.5 for p in peaks_sens)
    assert peak_found_sens, "Sensitive detection with min_intensity_floor should have found the weak peak"

def test_baseline_percentile_preservation():
    """Verify that PERCENTILE baseline estimation works correctly."""
    np.random.seed(42)
    wavelength = np.linspace(400, 600, 2000)
    
    # Sparse but strong peaks + one weak peak
    intensity = np.zeros_like(wavelength)
    # Strong peaks
    for pos in [420, 450, 480, 520, 550, 580]:
        intensity += 50.0 / (1 + ((wavelength - pos) / 0.05)**2)
    
    # Weak trace peak
    trace_pos = 500.0
    intensity += 1.0 / (1 + ((wavelength - trace_pos) / 0.05)**2)
    
    # Small continuum
    intensity += 5.0
    
    # Detect peaks using percentile baseline
    peaks_perc, base_perc, _ = detect_peaks_auto(
        wavelength, intensity, threshold_factor=0.5, 
        baseline_method=BaselineMethod.PERCENTILE, baseline_window_nm=1.0
    )
    
    # 10th percentile should be lower than median in a peak-rich spectrum
    peaks_median, base_median, _ = detect_peaks_auto(
        wavelength, intensity, threshold_factor=0.5, 
        baseline_method=BaselineMethod.MEDIAN, baseline_window_nm=1.0
    )
    
    assert np.mean(base_perc) < np.mean(base_median), "Percentile baseline should be lower than median baseline in emission-rich spectrum"
    
    # Ensure trace peak is detected with percentile method
    found_perc = any(abs(p[1] - trace_pos) < 0.2 for p in peaks_perc)
    assert found_perc, "Trace peak should be detected with percentile baseline method"
