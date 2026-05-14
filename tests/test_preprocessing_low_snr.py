from unittest.mock import patch

import numpy as np

from cflibs.inversion.preprocess import preprocessing as pp
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


# ---------------------------------------------------------------------------
# Regression guard: detect_peaks_auto must dispatch to the requested estimator.
#
# Closed PR #114 silently rewrote estimate_baseline() to always call ALS and
# flipped the detect_peaks_auto default to BaselineMethod.ALS, while leaving
# the BaselineMethod.{MEDIAN,SNIP,PERCENTILE} branches in place.  The branches
# still *looked* correct in the dispatch table, so a reader would assume
# MEDIAN went through median_filter -- but it did not.  These tests pin the
# contract by patching each estimator and asserting that exactly one of them
# is invoked per BaselineMethod value.
# ---------------------------------------------------------------------------

def _tiny_spectrum():
    wavelength = np.linspace(400.0, 410.0, 256)
    rng = np.random.RandomState(0)
    intensity = 1.0 + 0.05 * rng.standard_normal(wavelength.size)
    intensity[128] += 5.0
    return wavelength, intensity


def test_baseline_method_median_dispatches_to_median_estimator():
    wavelength, intensity = _tiny_spectrum()
    with patch.object(pp, "estimate_baseline", wraps=pp.estimate_baseline) as med, \
         patch.object(pp, "estimate_baseline_snip", wraps=pp.estimate_baseline_snip) as snip, \
         patch.object(pp, "estimate_baseline_als", wraps=pp.estimate_baseline_als) as als, \
         patch.object(pp, "estimate_baseline_percentile", wraps=pp.estimate_baseline_percentile) as pct:
        detect_peaks_auto(wavelength, intensity, baseline_method=BaselineMethod.MEDIAN)
    assert med.call_count == 1
    assert snip.call_count == 0
    assert als.call_count == 0
    assert pct.call_count == 0


def test_baseline_method_snip_dispatches_to_snip_estimator():
    wavelength, intensity = _tiny_spectrum()
    with patch.object(pp, "estimate_baseline", wraps=pp.estimate_baseline) as med, \
         patch.object(pp, "estimate_baseline_snip", wraps=pp.estimate_baseline_snip) as snip, \
         patch.object(pp, "estimate_baseline_als", wraps=pp.estimate_baseline_als) as als, \
         patch.object(pp, "estimate_baseline_percentile", wraps=pp.estimate_baseline_percentile) as pct:
        detect_peaks_auto(wavelength, intensity, baseline_method=BaselineMethod.SNIP)
    assert med.call_count == 0
    assert snip.call_count == 1
    assert als.call_count == 0
    assert pct.call_count == 0


def test_baseline_method_als_dispatches_to_als_estimator():
    wavelength, intensity = _tiny_spectrum()
    with patch.object(pp, "estimate_baseline", wraps=pp.estimate_baseline) as med, \
         patch.object(pp, "estimate_baseline_snip", wraps=pp.estimate_baseline_snip) as snip, \
         patch.object(pp, "estimate_baseline_als", wraps=pp.estimate_baseline_als) as als, \
         patch.object(pp, "estimate_baseline_percentile", wraps=pp.estimate_baseline_percentile) as pct:
        detect_peaks_auto(wavelength, intensity, baseline_method=BaselineMethod.ALS)
    assert med.call_count == 0
    assert snip.call_count == 0
    assert als.call_count == 1
    assert pct.call_count == 0


def test_baseline_method_percentile_dispatches_to_percentile_estimator():
    wavelength, intensity = _tiny_spectrum()
    with patch.object(pp, "estimate_baseline", wraps=pp.estimate_baseline) as med, \
         patch.object(pp, "estimate_baseline_snip", wraps=pp.estimate_baseline_snip) as snip, \
         patch.object(pp, "estimate_baseline_als", wraps=pp.estimate_baseline_als) as als, \
         patch.object(pp, "estimate_baseline_percentile", wraps=pp.estimate_baseline_percentile) as pct:
        detect_peaks_auto(wavelength, intensity, baseline_method=BaselineMethod.PERCENTILE)
    assert med.call_count == 0
    assert snip.call_count == 0
    assert als.call_count == 0
    assert pct.call_count == 1


def test_default_baseline_method_is_median_not_als():
    """Lock the default in: detect_peaks_auto's default must stay MEDIAN.

    Closed PR #114 flipped this to ALS, which changed peak detection
    semantics for every caller that did not pass baseline_method explicitly.
    """
    wavelength, intensity = _tiny_spectrum()
    with patch.object(pp, "estimate_baseline", wraps=pp.estimate_baseline) as med, \
         patch.object(pp, "estimate_baseline_als", wraps=pp.estimate_baseline_als) as als:
        detect_peaks_auto(wavelength, intensity)
    assert med.call_count == 1, "Default detect_peaks_auto must call the median estimator"
    assert als.call_count == 0, "Default detect_peaks_auto must NOT call ALS"


# ---------------------------------------------------------------------------
# BaselineMethod.AUTO: opt-in adaptive selector.
# Picks ALS for low-SNR spectra (the PR #114 intent) and MEDIAN otherwise.
# Crucially: AUTO is only consulted when the caller asks for it by name.
# ---------------------------------------------------------------------------

def test_auto_method_picks_als_for_low_snr():
    """A noisy spectrum with weak peaks should route AUTO -> ALS."""
    rng = np.random.RandomState(1)
    wavelength = np.linspace(400.0, 600.0, 2000)
    # Slowly varying continuum + heavy noise + a couple of low-amplitude peaks.
    continuum = 10.0 + 0.005 * wavelength
    noise = rng.normal(0.0, 1.0, wavelength.size)
    peak_a = 2.0 / (1.0 + ((wavelength - 480.0) / 0.1) ** 2)
    peak_b = 2.2 / (1.0 + ((wavelength - 520.0) / 0.1) ** 2)
    intensity = continuum + noise + peak_a + peak_b

    selected = pp._select_auto_baseline_method(
        wavelength, intensity, window_nm=10.0
    )
    assert selected == BaselineMethod.ALS, (
        f"low-SNR spectrum should route to ALS, got {selected}"
    )


def test_auto_method_picks_median_for_high_snr():
    """A clean spectrum with strong peaks should route AUTO -> MEDIAN."""
    wavelength = np.linspace(400.0, 600.0, 2000)
    continuum = 10.0 + 0.005 * wavelength
    rng = np.random.RandomState(2)
    noise = rng.normal(0.0, 0.05, wavelength.size)  # tiny noise
    peak = 200.0 / (1.0 + ((wavelength - 500.0) / 0.1) ** 2)  # strong peak
    intensity = continuum + noise + peak

    selected = pp._select_auto_baseline_method(
        wavelength, intensity, window_nm=10.0
    )
    assert selected == BaselineMethod.MEDIAN, (
        f"high-SNR spectrum should route to MEDIAN, got {selected}"
    )


def test_auto_method_runs_end_to_end_via_detect_peaks_auto():
    """detect_peaks_auto with baseline_method=AUTO returns valid output and
    does not raise.  Also verifies the AUTO branch resolves before dispatch
    (no AUTO leaks into the explicit ladder)."""
    rng = np.random.RandomState(3)
    wavelength = np.linspace(400.0, 600.0, 2000)
    intensity = 5.0 + rng.normal(0.0, 1.0, wavelength.size)
    # Inject one moderately strong peak so the result is non-trivial.
    peak_pos = 500.0
    peak_idx = int(np.argmin(np.abs(wavelength - peak_pos)))
    intensity += 30.0 / (1.0 + ((wavelength - peak_pos) / 0.1) ** 2)

    peaks, baseline, noise = detect_peaks_auto(
        wavelength, intensity, baseline_method=BaselineMethod.AUTO
    )
    assert baseline.shape == intensity.shape
    assert noise > 0
    # We injected a real peak, so AUTO should not return an empty list.
    assert len(peaks) >= 1
    # And the injected peak should be near the top of the list.
    assert any(abs(p[1] - peak_pos) < 0.5 for p in peaks)
