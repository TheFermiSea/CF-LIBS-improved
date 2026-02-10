"""Shared preprocessing for element identification algorithms.

This module provides common preprocessing steps used by ALIAS, Comb, and
Correlation identifiers: baseline estimation, noise estimation, peak detection,
and robust normalization.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from typing import List, Tuple


def estimate_baseline(
    wavelength: np.ndarray, intensity: np.ndarray, window_nm: float = 10.0
) -> np.ndarray:
    """Robust baseline estimation via median filter.

    Estimates the continuum background by applying a median filter with a
    window sized appropriately for the wavelength spacing. This is more
    robust than polynomial fitting for LIBS spectra with strong emission lines.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm, shape (N,)
    intensity : np.ndarray
        Intensity array, shape (N,)
    window_nm : float
        Window size in nm (default: 10.0)

    Returns
    -------
    baseline : np.ndarray
        Estimated baseline, shape (N,)

    Notes
    -----
    The window size is converted from nm to points using the median wavelength
    spacing. The window is forced to be odd to ensure symmetric filtering.
    """
    # Convert window_nm to points
    spacing = np.median(np.diff(wavelength))
    window_pts = max(3, int(window_nm / spacing))

    # Ensure odd window size for symmetric filtering
    if window_pts % 2 == 0:
        window_pts += 1

    # Apply median filter
    baseline = median_filter(intensity, size=window_pts, mode="nearest")

    return baseline


def estimate_noise(intensity: np.ndarray, baseline: np.ndarray) -> float:
    """Iterative sigma-clipped MAD noise estimation.

    Estimates the noise level by computing the median absolute deviation (MAD)
    on residuals after iterative sigma clipping. This removes contributions
    from emission lines, providing a robust estimate of the baseline noise.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity array, shape (N,)
    baseline : np.ndarray
        Estimated baseline, shape (N,)

    Returns
    -------
    sigma : float
        Estimated noise level (standard deviation)

    Notes
    -----
    Uses 3 iterations of 3-sigma clipping. The MAD is converted to standard
    deviation via the factor 1.4826 (for Gaussian distributions).

    This is critical for LIBS data: raw MAD on the full spectrum overestimates
    noise due to Poisson statistics on peaks. Sigma clipping removes peak
    contributions to isolate the baseline noise.
    """
    residuals = intensity - baseline

    # Iterative sigma clipping (3 iterations, 3-sigma threshold)
    for _ in range(3):
        median_val = np.median(residuals)
        mad = np.median(np.abs(residuals - median_val))
        sigma = mad * 1.4826  # Convert MAD to std for Gaussian

        if sigma < 1e-10:
            break

        # 3-sigma mask
        mask = np.abs(residuals - median_val) < 3 * sigma
        if np.sum(mask) < 10:  # Safety: need enough points
            break
        residuals = residuals[mask]

    # Final MAD -> sigma conversion
    median_val = np.median(residuals)
    mad = np.median(np.abs(residuals - median_val))
    sigma = mad * 1.4826

    return sigma


def detect_peaks(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    baseline: np.ndarray,
    noise: float,
    threshold_factor: float = 4.0,
    prominence_factor: float = 1.5,
) -> List[Tuple[int, float]]:
    """Unified peak detection.

    Detects peaks in the baseline-corrected spectrum using scipy.signal.find_peaks
    with SNR-based thresholds.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm, shape (N,)
    intensity : np.ndarray
        Intensity array, shape (N,)
    baseline : np.ndarray
        Estimated baseline, shape (N,)
    noise : float
        Estimated noise level
    threshold_factor : float
        Minimum SNR for peak detection (default: 4.0)
    prominence_factor : float
        Minimum prominence in units of noise (default: 1.5)

    Returns
    -------
    peaks : List[Tuple[int, float]]
        List of (index, wavelength_nm) tuples for detected peaks

    Notes
    -----
    The corrected intensity is computed as ``intensity - baseline``.
    Peaks must satisfy both height and prominence thresholds to be detected.
    """
    # Baseline-corrected spectrum
    corrected = intensity - baseline

    # Thresholds
    height_threshold = noise * threshold_factor
    prominence_threshold = noise * prominence_factor

    # Detect peaks
    peak_indices, _ = find_peaks(
        corrected,
        height=height_threshold,
        prominence=prominence_threshold,
    )

    # Build list of (index, wavelength) tuples
    peaks = [(int(idx), float(wavelength[idx])) for idx in peak_indices]

    return peaks


def robust_normalize(intensity: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Percentile-based normalization.

    Normalizes intensity by a high percentile value rather than the maximum.
    This is robust to cosmic rays and outliers.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity array, shape (N,)
    percentile : float
        Percentile to use for scaling (default: 95.0)

    Returns
    -------
    normalized : np.ndarray
        Normalized intensity, shape (N,)

    Notes
    -----
    NOT max-based (vulnerable to cosmic rays). Uses percentile-based scaling
    for robustness.
    """
    scale = np.percentile(intensity, percentile)

    if scale > 1e-10:
        return intensity / scale

    return intensity
