"""Shared preprocessing for element identification algorithms.

Provides the canonical peak detection pipeline for all CF-LIBS identifiers.
All identification algorithms should use ``detect_peaks`` (or the convenience
wrapper ``detect_peaks_auto``) rather than implementing custom peak detection.

Pipeline:
    1. Baseline estimation via median filter, SNIP, or ALS
    2. Sigma-clipped MAD noise estimation
    3. Peak detection with baseline subtraction, height/prominence thresholds
    4. Optional cosmic ray rejection (minimum FWHM filter)
    5. Optional second-derivative confirmation
"""

import numpy as np
from enum import Enum
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
    if wavelength.size < 2:
        return intensity.copy()
    spacing = float(np.median(np.abs(np.diff(wavelength))))
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 1.0  # safe fallback instead of 1e-10
    window_pts = max(3, int(window_nm / spacing))
    # Clamp to array length to prevent hangs
    max_window = len(intensity) if len(intensity) % 2 == 1 else len(intensity) - 1
    window_pts = min(window_pts, max(3, max_window))
    if window_pts % 2 == 0:
        window_pts += 1  # ensure odd
    return median_filter(intensity, size=window_pts)


class BaselineMethod(Enum):
    """Baseline estimation method for peak detection.

    Attributes
    ----------
    MEDIAN : str
        Median filter baseline (fast, robust default).
    SNIP : str
        Statistics-sensitive Non-linear Iterative Peak-clipping with LLS
        transform.  Best for spectra with sharp peaks on slowly varying
        continuum.
    ALS : str
        Asymmetric Least Squares smoothing.  Best for spectra with broad
        continuum features.
    """

    MEDIAN = "median"
    SNIP = "snip"
    ALS = "als"


def estimate_baseline_snip(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    num_iterations: int = 40,
    order: int = 2,
    smoothing_window: int = 0,
) -> np.ndarray:
    """Baseline estimation via SNIP (Statistics-sensitive Non-linear
    Iterative Peak-clipping) with LLS transform.

    Implements the algorithm of Ryan et al. (1988) with the Log-Log-Square
    root (LLS) transform to equalise peak widths before clipping.

    Algorithm
    ---------
    1. Apply LLS transform:
       :math:`v_i = \\log(\\log(\\sqrt{y_i + 1} + 1) + 1)`
    2. For *p* from ``num_iterations`` down to 1:
       :math:`v_i = \\min(v_i, (v_{i-p} + v_{i+p}) / 2)`
    3. Invert the LLS transform to recover the baseline in intensity
       space.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm (used only for consistency with other
        baseline estimators; spacing is not used).
    intensity : np.ndarray
        Intensity array (non-negative values expected).
    num_iterations : int
        Number of clipping iterations (default 40).  Larger values remove
        broader features.
    order : int
        Transform order.  ``2`` selects the LLS (Log-Log-Sqrt) transform;
        ``1`` applies a single-log transform; ``0`` disables the transform.
    smoothing_window : int
        If > 0, apply a moving-average smoothing of this width (in pixels)
        to the result.  ``0`` disables smoothing (default).

    Returns
    -------
    np.ndarray
        Estimated baseline array (same length as *intensity*).

    References
    ----------
    Ryan, C.G., Clayton, E., Griffin, W.L., Sie, S.H. & Cousens, D.R.
    (1988). SNIP, a statistics-sensitive background treatment for the
    quantitative analysis of PIXE spectra in geoscience applications.
    *Nucl. Instrum. Methods B*, 34, 396--402.
    """
    if intensity.size < 2:
        return intensity.copy()

    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2, got {order}")

    y = np.maximum(intensity.astype(float), 0.0)

    # --- LLS forward transform ---
    if order >= 1:
        y = np.log(np.sqrt(y + 1.0) + 1.0)
    if order >= 2:
        y = np.log(y + 1.0)

    # Pad edges to handle boundary effects; cap pad_width for short arrays
    pad_width = min(num_iterations, len(y) - 1) if len(y) > 1 else 0
    v = np.pad(y, pad_width=pad_width, mode="reflect")

    # --- Iterative peak clipping (p from pad_width down to 1) ---
    for p in range(pad_width, 0, -1):
        avg = (v[: len(v) - 2 * p] + v[2 * p :]) / 2.0
        mid = v[p : len(v) - p]
        v[p : len(v) - p] = np.minimum(mid, avg)

    # Remove padding
    v = v[pad_width : pad_width + len(y)]

    # --- LLS inverse transform ---
    if order >= 2:
        v = np.exp(v) - 1.0
    if order >= 1:
        v = (np.exp(v) - 1.0) ** 2 - 1.0
        v = np.maximum(v, 0.0)

    # Optional smoothing
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        v = np.convolve(v, kernel, mode="same")

    return v


def estimate_baseline_als(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    max_iterations: int = 20,
    tol: float = 1e-4,
) -> np.ndarray:
    """Baseline estimation via Asymmetric Least Squares (ALS) smoothing.

    Implements the algorithm of Eilers & Boelens (2005).  A smoothing
    penalty (second-difference matrix) is combined with asymmetric
    weights that penalise positive residuals less than negative ones,
    causing the baseline to track below the peaks.

    The system solved at each iteration is:

    .. math::

        (W + \\lambda D^\\top D) z = W y

    where *W* = diag(w), *D* is the second-difference matrix, and the
    asymmetric weights are updated as
    :math:`w_i = p` if :math:`y_i > z_i`, else :math:`w_i = 1 - p`.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm (unused; kept for API consistency).
    intensity : np.ndarray
        Intensity array.
    lam : float
        Smoothness penalty parameter :math:`\\lambda` (default 1e6).
        Larger values produce smoother baselines.
    p : float
        Asymmetry parameter (default 0.01).  Smaller values push the
        baseline further below the peaks.
    max_iterations : int
        Maximum IRLS iterations (default 20).
    tol : float
        Convergence tolerance on the weight vector (default 1e-4).

    Returns
    -------
    np.ndarray
        Estimated baseline array (same length as *intensity*).

    References
    ----------
    Eilers, P.H.C. & Boelens, H.F.M. (2005). Baseline Correction with
    Asymmetric Least Squares Smoothing.  Leiden University Medical Centre
    Report.
    """
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")
    if not (0 < p < 1):
        raise ValueError(f"p must be in (0, 1), got {p}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    n = intensity.size
    if n < 3:
        return intensity.copy()

    y = intensity.astype(float)

    # Second-difference matrix D (shape (n-2) x n)
    e = sparse.eye(n, format="csc")
    D = e[2:] - 2 * e[1:-1] + e[:-2]
    DTD = lam * D.T.dot(D)

    w = np.ones(n)
    z = y.copy()

    for _ in range(max_iterations):
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        z_new = spsolve(W + DTD, w * y)
        # Update asymmetric weights
        w_new = np.where(y > z_new, p, 1 - p)
        if float(np.linalg.norm(w_new - w)) / max(float(np.linalg.norm(w)), 1e-10) < tol:
            z = z_new
            break
        w = w_new
        z = z_new

    return z


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
    min_distance_px: Optional[int] = None,
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
        the minimum peak distance is set to one resolution element.
    min_distance_px : int, optional
        Explicit minimum peak distance in pixels. Overrides resolving_power-
        derived value when provided.
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

    # Minimum distance between peaks
    if min_distance_px is not None:
        min_distance = max(1, int(min_distance_px))
    elif resolving_power is not None and resolving_power > 0:
        mean_wl = float(np.mean(wavelength))
        spacing = float(np.median(np.abs(np.diff(wavelength))))
        resolution_nm = mean_wl / resolving_power
        min_distance = max(3, int(resolution_nm / max(spacing, 1e-9)))
    else:
        min_distance = 3

    peak_indices, _ = find_peaks(
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
    if use_second_derivative and len(peak_indices) > 0:
        if wavelength.size >= 3:  # edge_order=2 requires >= 3 points
            spacing = float(np.median(np.abs(np.diff(wavelength))))
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
        # else: skip second-derivative filtering for short arrays

    return [(int(idx), float(wavelength[idx])) for idx in peak_indices]


def detect_peaks_auto(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    resolving_power: Optional[float] = None,
    min_distance_px: Optional[int] = None,
    threshold_factor: float = 4.0,
    prominence_factor: float = 1.5,
    baseline_window_nm: float = 10.0,
    min_fwhm_pixels: float = 1.5,
    use_second_derivative: bool = False,
    baseline_method: BaselineMethod = BaselineMethod.MEDIAN,
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
    min_distance_px : int, optional
        Explicit minimum peak distance in pixels. Overrides resolving_power
        when provided.
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
    baseline_method : BaselineMethod
        Baseline estimation method (default ``BaselineMethod.MEDIAN``).
        ``SNIP`` uses Statistics-sensitive Non-linear Iterative Peak-clipping;
        ``ALS`` uses Asymmetric Least Squares smoothing.

    Returns
    -------
    peaks : List[Tuple[int, float]]
        List of (index, wavelength_nm) tuples
    baseline : np.ndarray
        Estimated baseline array
    noise : float
        Estimated noise level (sigma)
    """
    if wavelength.size < 2:
        return [], intensity.copy(), 0.0

    if baseline_method == BaselineMethod.SNIP:
        baseline = estimate_baseline_snip(wavelength, intensity)
    elif baseline_method == BaselineMethod.ALS:
        baseline = estimate_baseline_als(wavelength, intensity)
    elif baseline_method == BaselineMethod.MEDIAN:
        baseline = estimate_baseline(wavelength, intensity, window_nm=baseline_window_nm)
    else:
        raise ValueError(f"Unknown baseline_method: {baseline_method!r}")
    noise = estimate_noise(intensity, baseline)

    peaks = detect_peaks(
        wavelength,
        intensity,
        baseline,
        noise,
        threshold_factor=threshold_factor,
        prominence_factor=prominence_factor,
        resolving_power=resolving_power,
        min_distance_px=min_distance_px,
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
