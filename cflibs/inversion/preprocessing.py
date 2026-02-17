"""Shared preprocessing for element identification algorithms.

Provides the canonical peak detection pipeline for all CF-LIBS identifiers.
All identification algorithms should use ``detect_peaks`` (or the convenience
wrapper ``detect_peaks_auto``) rather than implementing custom peak detection.

Pipeline:
    1. Baseline estimation via median filter
    2. Sigma-clipped MAD noise estimation
    3. Peak detection with baseline subtraction, height/prominence thresholds
    4. Optional cosmic ray rejection (minimum FWHM filter)
    5. Optional second-derivative confirmation
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import median_filter
from typing import List, Optional, Tuple

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.preprocessing")


def estimate_baseline(
    wavelength: np.ndarray, intensity: np.ndarray, window_nm: float = 10.0
) -> np.ndarray:
    """Robust baseline estimation via median filter.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm
    intensity : np.ndarray
        Intensity array
    window_nm : float
        Filter window width in nm (default 10.0)

    Returns
    -------
    np.ndarray
        Estimated baseline
    """
    spacing = np.median(np.diff(wavelength))
    window_pts = max(3, int(window_nm / spacing))
    if window_pts % 2 == 0:
        window_pts += 1  # ensure odd
    return median_filter(intensity, size=window_pts)


def estimate_noise(intensity: np.ndarray, baseline: np.ndarray) -> float:
    """Iterative sigma-clipped MAD noise estimation.

    Uses 3 iterations of 3-sigma clipping to remove peak contributions
    before computing noise. Critical for LIBS spectra where raw MAD
    overestimates noise due to emission peaks and continuum.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity array
    baseline : np.ndarray
        Baseline estimate

    Returns
    -------
    float
        Noise level (sigma)
    """
    residuals = intensity - baseline
    for _ in range(3):
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        sigma = mad * 1.4826
        if sigma < 1e-10:
            break
        mask = np.abs(residuals - med) < 3.0 * sigma
        if np.sum(mask) < 10:
            break
        residuals = residuals[mask]
    # Final estimate
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    noise = mad * 1.4826

    # Floor noise to prevent zero thresholds that cause find_peaks to
    # return many trivial local maxima.
    if noise < 1e-10:
        noise = max(1e-10, float(np.nanpercentile(np.abs(intensity - baseline), 95)) * 1e-6)

    return noise


def detect_peaks(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    baseline: np.ndarray,
    noise: float,
    threshold_factor: float = 4.0,
    prominence_factor: float = 1.5,
    resolving_power: Optional[float] = None,
    min_fwhm_pixels: float = 1.5,
    use_second_derivative: bool = False,
) -> List[Tuple[int, float]]:
    """Unified peak detection above baseline.

    This is the canonical peak detection function for all CF-LIBS
    identification algorithms. It operates on baseline-subtracted
    intensity with noise-calibrated thresholds and optional filters
    for cosmic ray rejection and second-derivative confirmation.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm
    intensity : np.ndarray
        Intensity array (raw, not baseline-subtracted)
    baseline : np.ndarray
        Baseline estimate from ``estimate_baseline``
    noise : float
        Noise level from ``estimate_noise``
    threshold_factor : float
        Peak height threshold = noise * threshold_factor (default 4.0)
    prominence_factor : float
        Peak prominence threshold = noise * prominence_factor (default 1.5)
    resolving_power : float, optional
        Instrument resolving power R = lambda/delta_lambda.  If provided,
        the minimum peak distance is set to one resolution element and
        cosmic ray rejection uses resolution-aware FWHM thresholds.
    min_fwhm_pixels : float
        Minimum peak full-width at half-maximum in pixels.  Peaks narrower
        than this are rejected as cosmic ray artifacts (default 1.5).
        Set to 0 to disable cosmic ray filtering.
    use_second_derivative : bool
        If True, apply second-derivative confirmation: peaks must have
        positive curvature (-d2I/dlambda2 > 0) within a +/-2 pixel
        neighborhood (default False).

    Returns
    -------
    List[Tuple[int, float]]
        List of (index, wavelength_nm) tuples for detected peaks
    """
    corrected = intensity - baseline
    threshold = noise * threshold_factor
    prominence = noise * prominence_factor

    # Minimum distance between peaks: at least one resolution element
    if resolving_power is not None and resolving_power > 0:
        mean_wl = float(np.mean(wavelength))
        spacing = float(np.median(np.diff(wavelength)))
        resolution_nm = mean_wl / resolving_power
        min_distance = max(3, int(resolution_nm / max(spacing, 1e-9)))
    else:
        min_distance = 3

    peak_indices, properties = find_peaks(
        corrected,
        height=threshold,
        prominence=prominence,
        distance=min_distance,
    )

    if len(peak_indices) == 0:
        return []

    # Cosmic ray rejection: filter by minimum FWHM
    if min_fwhm_pixels > 0:
        try:
            widths_result = peak_widths(corrected, peak_indices, rel_height=0.5)
            fwhm_pixels = widths_result[0]
            width_mask = fwhm_pixels >= min_fwhm_pixels
            peak_indices = peak_indices[width_mask]
        except (ValueError, RuntimeError) as exc:
            logger.debug("peak_widths failed for %d peaks: %s", len(peak_indices), exc)

    if len(peak_indices) == 0:
        return []

    # Second-derivative confirmation
    if use_second_derivative:
        if wavelength.size > 1:
            spacing = float(np.median(np.diff(wavelength)))
        else:
            spacing = 1.0
        d1 = np.gradient(corrected, spacing, edge_order=2)
        d2 = -np.gradient(d1, spacing, edge_order=2)
        d2[d2 < 0] = 0.0
        confirmed = []
        for idx in peak_indices:
            lo = max(0, idx - 2)
            hi = min(len(d2), idx + 3)
            if np.max(d2[lo:hi]) > 0:
                confirmed.append(idx)
        peak_indices = np.array(confirmed, dtype=int) if confirmed else np.array([], dtype=int)

    return [(int(idx), float(wavelength[idx])) for idx in peak_indices]


def detect_peaks_auto(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    resolving_power: Optional[float] = None,
    threshold_factor: float = 4.0,
    prominence_factor: float = 1.5,
    baseline_window_nm: float = 10.0,
    min_fwhm_pixels: float = 1.5,
    use_second_derivative: bool = False,
) -> Tuple[List[Tuple[int, float]], np.ndarray, float]:
    """Convenience wrapper: estimate baseline/noise, then detect peaks.

    This is the recommended entry point for peak detection across all
    identification algorithms.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm
    intensity : np.ndarray
        Intensity array
    resolving_power : float, optional
        Instrument resolving power R = lambda/delta_lambda
    threshold_factor : float
        Peak height threshold in noise units (default 4.0)
    prominence_factor : float
        Peak prominence threshold in noise units (default 1.5)
    baseline_window_nm : float
        Baseline median filter window in nm (default 10.0)
    min_fwhm_pixels : float
        Minimum FWHM in pixels for cosmic ray rejection (default 1.5)
    use_second_derivative : bool
        Apply second-derivative confirmation (default False)

    Returns
    -------
    peaks : List[Tuple[int, float]]
        List of (index, wavelength_nm) tuples
    baseline : np.ndarray
        Estimated baseline array
    noise : float
        Estimated noise level (sigma)
    """
    baseline = estimate_baseline(wavelength, intensity, window_nm=baseline_window_nm)
    noise = estimate_noise(intensity, baseline)

    peaks = detect_peaks(
        wavelength,
        intensity,
        baseline,
        noise,
        threshold_factor=threshold_factor,
        prominence_factor=prominence_factor,
        resolving_power=resolving_power,
        min_fwhm_pixels=min_fwhm_pixels,
        use_second_derivative=use_second_derivative,
    )
    return peaks, baseline, noise


def robust_normalize(intensity: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Percentile-based normalization robust to cosmic ray artifacts.

    Uses the given percentile instead of max() to avoid sensitivity
    to single-pixel cosmic ray spikes.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity array
    percentile : float
        Normalization percentile (default 95.0)

    Returns
    -------
    np.ndarray
        Normalized intensity array
    """
    scale = np.percentile(intensity, percentile)
    if scale > 1e-10:
        return intensity / scale
    return intensity.copy()
