"""Shared preprocessing for element identification algorithms."""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from typing import List, Tuple


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
    return mad * 1.4826


def detect_peaks(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    baseline: np.ndarray,
    noise: float,
    threshold_factor: float = 4.0,
    prominence_factor: float = 1.5,
) -> List[Tuple[int, float]]:
    """Unified peak detection above baseline.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm
    intensity : np.ndarray
        Intensity array
    baseline : np.ndarray
        Baseline estimate
    noise : float
        Noise level from estimate_noise
    threshold_factor : float
        Peak height threshold = noise * threshold_factor (default 4.0)
    prominence_factor : float
        Peak prominence threshold = noise * prominence_factor (default 1.5)

    Returns
    -------
    List[Tuple[int, float]]
        List of (index, wavelength_nm) tuples for detected peaks
    """
    corrected = intensity - baseline
    threshold = noise * threshold_factor
    prominence = noise * prominence_factor

    peak_indices, _ = find_peaks(corrected, height=threshold, prominence=prominence)
    return [(int(idx), float(wavelength[idx])) for idx in peak_indices]


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
