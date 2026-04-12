"""
Outlier detection methods for LIBS spectra.

This module provides methods for identifying and removing anomalous spectra
from replicate measurements, which is essential for reliable CF-LIBS analysis.

Methods implemented:
- **SAM (Spectral Angle Mapper)**: Angle-based similarity insensitive to intensity scaling
- **MAD (Median Absolute Deviation)**: Robust univariate outlier detection

**SAM** is useful when spectral *shape* matters (shot-to-shot intensity variations
are acceptable, but shape changes indicate anomalies).

**MAD** is useful for detecting outliers in univariate data or when intensity
values themselves need to be cleaned (per-channel or per-spectrum).

References:
    - Kruse et al. (1993), "The Spectral Image Processing System (SIPS)"
    - Zhang et al. (2017), "Spectral preprocessing for LIBS"
    - Leys et al. (2013), "Detecting outliers: Do not use standard deviation"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.outliers")


class OutlierMethod(Enum):
    """Method for outlier detection."""

    SAM = "sam"  # Spectral Angle Mapper
    MAD = "mad"  # Median Absolute Deviation


@dataclass
class SAMResult:
    """
    Results of SAM-based outlier detection.

    Attributes
    ----------
    angles : np.ndarray
        SAM angle (radians) from each spectrum to the reference.
        Range: [0, pi/2]. Lower = more similar.
    outlier_mask : np.ndarray
        Boolean mask where True indicates an outlier spectrum.
    threshold : float
        Threshold angle (radians) used for outlier detection.
    reference_spectrum : np.ndarray
        The reference spectrum used for comparison.
    n_outliers : int
        Number of outliers detected.
    inlier_indices : np.ndarray
        Indices of inlier (non-outlier) spectra.
    outlier_indices : np.ndarray
        Indices of outlier spectra.
    method : str
        Thresholding method used ('mad', 'percentile', 'fixed').
    """

    angles: np.ndarray
    outlier_mask: np.ndarray
    threshold: float
    reference_spectrum: np.ndarray
    n_outliers: int
    inlier_indices: np.ndarray
    outlier_indices: np.ndarray
    method: str = "mad"

    @property
    def n_inliers(self) -> int:
        """Number of inlier spectra."""
        return len(self.inlier_indices)

    @property
    def outlier_fraction(self) -> float:
        """Fraction of spectra flagged as outliers."""
        total = len(self.angles)
        return self.n_outliers / total if total > 0 else 0.0

    def angles_degrees(self) -> np.ndarray:
        """SAM angles in degrees for easier interpretation."""
        return np.degrees(self.angles)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "SAM Outlier Detection Results:",
            f"  Total spectra: {len(self.angles)}",
            f"  Outliers: {self.n_outliers} ({self.outlier_fraction*100:.1f}%)",
            f"  Threshold: {np.degrees(self.threshold):.2f} degrees ({self.method})",
            f"  Angle range: {np.degrees(self.angles.min()):.2f} - {np.degrees(self.angles.max()):.2f} degrees",
        ]
        if self.n_outliers > 0:
            lines.append(f"  Outlier indices: {list(self.outlier_indices)}")
        return "\n".join(lines)


class SpectralAngleMapper:
    """
    Spectral Angle Mapper (SAM) for outlier detection in LIBS spectra.

    SAM calculates the angle between two spectra treated as vectors in
    n-dimensional space (n = number of wavelength channels):

        SAM(s1, s2) = arccos(s1 · s2 / (||s1|| × ||s2||))

    Key properties:
    - Range: [0, π/2] radians (0° to 90°)
    - Intensity-invariant: SAM(s, k*s) = 0 for any scalar k > 0
    - Small angles = similar spectral shapes
    - Large angles = different spectral shapes

    Parameters
    ----------
    threshold_method : str
        Method for automatic threshold selection:
        - 'mad': Median + k × MAD (default, robust)
        - 'percentile': Use percentile of angle distribution
        - 'fixed': Use fixed threshold value
    threshold_sigma : float
        For 'mad' method: number of MAD units above median (default: 3.0)
    threshold_percentile : float
        For 'percentile' method: percentile value (default: 95.0)
    threshold_fixed : float
        For 'fixed' method: angle threshold in radians (default: 0.1 ~ 5.7°)
    reference_method : str
        Method for computing reference spectrum:
        - 'mean': Use mean spectrum (default)
        - 'median': Use element-wise median (more robust)

    Examples
    --------
    >>> sam = SpectralAngleMapper(threshold_sigma=3.0)
    >>> result = sam.detect_outliers(spectra)
    >>> clean_spectra = spectra[~result.outlier_mask]
    >>> print(result.summary())
    """

    def __init__(
        self,
        threshold_method: str = "mad",
        threshold_sigma: float = 3.0,
        threshold_percentile: float = 95.0,
        threshold_fixed: float = 0.1,
        reference_method: str = "mean",
    ):
        if threshold_method not in ("mad", "percentile", "fixed"):
            raise ValueError(f"Unknown threshold_method: {threshold_method}")
        if reference_method not in ("mean", "median"):
            raise ValueError(f"Unknown reference_method: {reference_method}")

        self.threshold_method = threshold_method
        self.threshold_sigma = threshold_sigma
        self.threshold_percentile = threshold_percentile
        self.threshold_fixed = threshold_fixed
        self.reference_method = reference_method

    def spectral_angle(
        self,
        spectrum1: np.ndarray,
        spectrum2: np.ndarray,
    ) -> float:
        """
        Calculate SAM angle between two spectra.

        Parameters
        ----------
        spectrum1 : np.ndarray
            First spectrum (1D array of intensities)
        spectrum2 : np.ndarray
            Second spectrum (1D array, same length as spectrum1)

        Returns
        -------
        float
            Angle in radians [0, π/2]

        Raises
        ------
        ValueError
            If spectra have different lengths or are zero vectors
        """
        s1 = np.asarray(spectrum1, dtype=np.float64)
        s2 = np.asarray(spectrum2, dtype=np.float64)

        if s1.shape != s2.shape:
            raise ValueError(f"Spectrum shape mismatch: {s1.shape} vs {s2.shape}")

        # Compute norms
        norm1 = np.linalg.norm(s1)
        norm2 = np.linalg.norm(s2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            raise ValueError("Cannot compute SAM for zero or near-zero spectrum")

        # Compute cosine of angle
        cos_angle = np.dot(s1, s2) / (norm1 * norm2)

        # Clamp to [-1, 1] for numerical stability
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.arccos(cos_angle))

    def pairwise_angles(
        self,
        spectra: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise SAM angles between all spectra.

        Parameters
        ----------
        spectra : np.ndarray
            2D array of shape (n_spectra, n_wavelengths)

        Returns
        -------
        np.ndarray
            Symmetric matrix of shape (n_spectra, n_spectra) with SAM angles.
            Diagonal elements are 0.
        """
        spectra = np.asarray(spectra, dtype=np.float64)
        # Normalize all spectra
        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        normalized = spectra / norms

        # Compute cosine similarity matrix
        cos_matrix = normalized @ normalized.T

        # Clamp for numerical stability
        cos_matrix = np.clip(cos_matrix, -1.0, 1.0)

        # Convert to angles
        angles = np.arccos(cos_matrix)

        return angles

    def angles_from_reference(
        self,
        spectra: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SAM angles from each spectrum to a reference.

        Parameters
        ----------
        spectra : np.ndarray
            2D array of shape (n_spectra, n_wavelengths)
        reference : np.ndarray, optional
            Reference spectrum. If None, computed from spectra using
            reference_method.

        Returns
        -------
        angles : np.ndarray
            SAM angle from each spectrum to reference
        reference : np.ndarray
            The reference spectrum used
        """
        spectra = np.asarray(spectra, dtype=np.float64)

        if reference is None:
            reference = self._compute_reference(spectra)
        else:
            reference = np.asarray(reference, dtype=np.float64)

        # Normalize reference
        ref_norm = np.linalg.norm(reference)
        if ref_norm < 1e-10:
            raise ValueError("Reference spectrum is zero or near-zero")
        ref_normalized = reference / ref_norm

        # Normalize all spectra
        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = spectra / norms

        # Compute cosine similarities
        cos_angles = normalized @ ref_normalized

        # Clamp and convert to angles
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)

        return angles, reference

    def detect_outliers(
        self,
        spectra: np.ndarray,
        reference: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> SAMResult:
        """
        Detect outlier spectra using SAM distance from reference.

        Parameters
        ----------
        spectra : np.ndarray
            2D array of shape (n_spectra, n_wavelengths)
        reference : np.ndarray, optional
            Reference spectrum. If None, computed from spectra.
        threshold : float, optional
            Override threshold (in radians). If None, computed automatically.

        Returns
        -------
        SAMResult
            Detection results including outlier mask and statistics
        """
        spectra = np.asarray(spectra, dtype=np.float64)

        if spectra.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {spectra.shape}")

        n_spectra = spectra.shape[0]

        if n_spectra < 2:
            logger.warning("Need at least 2 spectra for outlier detection")
            return SAMResult(
                angles=np.array([0.0]),
                outlier_mask=np.array([False]),
                threshold=0.0,
                reference_spectrum=spectra[0] if n_spectra > 0 else np.array([]),
                n_outliers=0,
                inlier_indices=np.array([0]) if n_spectra > 0 else np.array([]),
                outlier_indices=np.array([], dtype=int),
                method=self.threshold_method,
            )

        # Compute angles from reference
        angles, ref_spectrum = self.angles_from_reference(spectra, reference)

        # Determine threshold
        if threshold is not None:
            thresh = threshold
            method_used = "fixed"
        else:
            thresh = self._compute_threshold(angles)
            method_used = self.threshold_method

        # Identify outliers
        outlier_mask = angles > thresh
        outlier_indices = np.where(outlier_mask)[0]
        inlier_indices = np.where(~outlier_mask)[0]

        n_outliers = int(np.sum(outlier_mask))

        if n_outliers > 0:
            logger.info(
                f"SAM detected {n_outliers}/{n_spectra} outliers "
                f"(threshold: {np.degrees(thresh):.2f}°)"
            )

        return SAMResult(
            angles=angles,
            outlier_mask=outlier_mask,
            threshold=thresh,
            reference_spectrum=ref_spectrum,
            n_outliers=n_outliers,
            inlier_indices=inlier_indices,
            outlier_indices=outlier_indices,
            method=method_used,
        )

    def filter_spectra(
        self,
        spectra: np.ndarray,
        reference: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, SAMResult]:
        """
        Filter out outlier spectra, returning only inliers.

        Convenience method that combines detection and filtering.

        Parameters
        ----------
        spectra : np.ndarray
            2D array of shape (n_spectra, n_wavelengths)
        reference : np.ndarray, optional
            Reference spectrum
        threshold : float, optional
            Override threshold (radians)

        Returns
        -------
        filtered_spectra : np.ndarray
            Spectra with outliers removed
        result : SAMResult
            Detection results for diagnostics
        """
        result = self.detect_outliers(spectra, reference, threshold)
        filtered = spectra[~result.outlier_mask]
        return filtered, result

    def _compute_reference(self, spectra: np.ndarray) -> np.ndarray:
        """Compute reference spectrum from collection."""
        if self.reference_method == "median":
            return np.median(spectra, axis=0)
        else:  # mean
            return np.mean(spectra, axis=0)

    def _compute_threshold(self, angles: np.ndarray) -> float:
        """Compute threshold based on angle distribution."""
        if self.threshold_method == "fixed":
            return self.threshold_fixed

        elif self.threshold_method == "percentile":
            return float(np.percentile(angles, self.threshold_percentile))

        else:  # mad
            median_angle = float(np.median(angles))
            # MAD = median(|x - median(x)|)
            mad = float(np.median(np.abs(angles - median_angle)))

            # Scale MAD to approximate standard deviation for normal distribution
            # For normal distribution: MAD ≈ 0.6745 × σ
            mad_scaled = mad * 1.4826

            threshold = median_angle + self.threshold_sigma * mad_scaled

            # Ensure threshold is at least median + small margin
            threshold = max(threshold, median_angle * 1.1)

            return threshold


def sam_distance(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
) -> float:
    """
    Convenience function to compute SAM angle between two spectra.

    Parameters
    ----------
    spectrum1 : np.ndarray
        First spectrum
    spectrum2 : np.ndarray
        Second spectrum

    Returns
    -------
    float
        SAM angle in radians
    """
    mapper = SpectralAngleMapper()
    return float(mapper.spectral_angle(spectrum1, spectrum2))


def detect_outlier_spectra(
    spectra: np.ndarray,
    threshold_sigma: float = 3.0,
    reference_method: str = "mean",
) -> SAMResult:
    """
    Convenience function for SAM-based outlier detection.

    Parameters
    ----------
    spectra : np.ndarray
        2D array of shape (n_spectra, n_wavelengths)
    threshold_sigma : float
        Number of MAD units for threshold (default: 3.0)
    reference_method : str
        'mean' or 'median' for computing reference

    Returns
    -------
    SAMResult
        Detection results
    """
    mapper = SpectralAngleMapper(
        threshold_method="mad",
        threshold_sigma=threshold_sigma,
        reference_method=reference_method,
    )
    return mapper.detect_outliers(spectra)


# =============================================================================
# MAD (Median Absolute Deviation) Outlier Detection
# =============================================================================

# MAD scale factor: for normal distribution, MAD ≈ 0.6745 × σ
# So σ ≈ MAD × 1.4826
MAD_SCALE_FACTOR = 1.4826


@dataclass
class MADResult:
    """
    Results of MAD-based outlier detection.

    Attributes
    ----------
    outlier_mask : np.ndarray
        Boolean mask where True indicates an outlier.
        Shape depends on mode: (n_spectra,) for spectrum mode,
        (n_spectra, n_wavelengths) for channel mode.
    median : np.ndarray or float
        Median value(s) used as center.
    mad : np.ndarray or float
        MAD value(s) computed.
    threshold : float
        Number of scaled MAD units used for detection.
    n_outliers : int
        Total number of outlier values detected.
    mode : str
        Detection mode ('spectrum', 'channel', or '1d').
    statistic : str
        Statistic used for spectrum mode ('total_intensity', 'max_intensity', etc.)
    inlier_indices : np.ndarray, optional
        Indices of inlier items (for spectrum/1d mode). None for channel mode.
    outlier_indices : np.ndarray, optional
        Indices of outlier items (for spectrum/1d mode). None for channel mode.
    """

    outlier_mask: np.ndarray
    median: np.ndarray
    mad: np.ndarray
    threshold: float
    n_outliers: int
    mode: str
    statistic: str = ""
    inlier_indices: Optional[np.ndarray] = None
    outlier_indices: Optional[np.ndarray] = None

    def __post_init__(self):
        """Compute indices if not provided."""
        if self.inlier_indices is None and self.outlier_mask.ndim == 1:
            self.outlier_indices = np.where(self.outlier_mask)[0]
            self.inlier_indices = np.where(~self.outlier_mask)[0]

    @property
    def n_inliers(self) -> int:
        """Number of inlier items."""
        if self.inlier_indices is not None:
            return len(self.inlier_indices)
        return int(np.sum(~self.outlier_mask))

    @property
    def outlier_fraction(self) -> float:
        """Fraction of values flagged as outliers."""
        total = self.outlier_mask.size
        return self.n_outliers / total if total > 0 else 0.0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"MAD Outlier Detection Results ({self.mode} mode):",
            f"  Total items: {self.outlier_mask.size}",
            f"  Outliers: {self.n_outliers} ({self.outlier_fraction*100:.1f}%)",
            f"  Threshold: {self.threshold} scaled MAD units",
        ]
        if self.statistic:
            lines.append(f"  Statistic: {self.statistic}")
        if (
            self.mode in ("spectrum", "1d")
            and self.n_outliers > 0
            and self.outlier_indices is not None
        ):
            lines.append(f"  Outlier indices: {list(self.outlier_indices)}")
        return "\n".join(lines)


class MADOutlierDetector:
    """
    Median Absolute Deviation (MAD) outlier detector for LIBS spectra.

    MAD is a robust measure of statistical dispersion:

        MAD = median(|X_i - median(X)|)

    For normally distributed data, MAD ≈ 0.6745 × σ. We scale MAD by 1.4826
    to estimate the standard deviation, then flag values beyond k × scaled_MAD
    from the median as outliers.

    MAD is more robust than standard deviation because:
    1. Median is not affected by extreme outliers
    2. 50% of data can be outliers before MAD breaks down (vs ~0% for std)

    Parameters
    ----------
    threshold : float
        Number of scaled MAD units for outlier detection (default: 3.0).
        Points beyond median ± threshold × scaled_MAD are flagged.
    mode : str
        Detection mode:
        - 'spectrum': Flag entire spectra based on summary statistic
        - 'channel': Flag individual (spectrum, wavelength) values
        - '1d': Operate on 1D array directly
    statistic : str
        For 'spectrum' mode, which statistic to compute per spectrum:
        - 'total_intensity': Sum of all intensities (default)
        - 'max_intensity': Maximum intensity
        - 'mean_intensity': Mean intensity
        - 'std_intensity': Standard deviation of intensities

    Examples
    --------
    >>> # Detect outlier spectra by total intensity
    >>> mad = MADOutlierDetector(threshold=3.0, mode='spectrum')
    >>> result = mad.detect_outliers(spectra)
    >>> clean_spectra = spectra[~result.outlier_mask]

    >>> # Clean individual channel values
    >>> mad = MADOutlierDetector(threshold=3.0, mode='channel')
    >>> result = mad.detect_outliers(spectra)
    >>> # result.outlier_mask has shape (n_spectra, n_wavelengths)

    >>> # Simple 1D outlier detection
    >>> mad = MADOutlierDetector(threshold=3.0, mode='1d')
    >>> result = mad.detect_outliers(intensity_values)
    """

    def __init__(
        self,
        threshold: float = 3.0,
        mode: str = "spectrum",
        statistic: str = "total_intensity",
    ):
        if mode not in ("spectrum", "channel", "1d"):
            raise ValueError(f"Unknown mode: {mode}. Use 'spectrum', 'channel', or '1d'")
        if statistic not in ("total_intensity", "max_intensity", "mean_intensity", "std_intensity"):
            raise ValueError(f"Unknown statistic: {statistic}")
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")

        self.threshold = threshold
        self.mode = mode
        self.statistic = statistic

    def compute_mad(
        self, data: np.ndarray, axis: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute median and MAD of data.

        Parameters
        ----------
        data : np.ndarray
            Input data
        axis : int, optional
            Axis along which to compute. None for flattened array.

        Returns
        -------
        median : np.ndarray
            Median value(s)
        mad : np.ndarray
            MAD value(s)
        """
        median = np.median(data, axis=axis, keepdims=True if axis is not None else False)
        deviations = np.abs(data - median)
        mad = np.median(deviations, axis=axis, keepdims=False)

        # Squeeze median if keepdims was used
        if axis is not None:
            median = np.squeeze(median, axis=axis)

        return median, mad

    def detect_outliers(
        self,
        data: np.ndarray,
    ) -> MADResult:
        """
        Detect outliers using MAD.

        Parameters
        ----------
        data : np.ndarray
            Input data. Shape depends on mode:
            - '1d': 1D array of values
            - 'spectrum'/'channel': 2D array of shape (n_spectra, n_wavelengths)

        Returns
        -------
        MADResult
            Detection results
        """
        data = np.asarray(data, dtype=np.float64)

        if self.mode == "1d":
            return self._detect_1d(data)
        elif self.mode == "spectrum":
            return self._detect_spectrum(data)
        else:  # channel
            return self._detect_channel(data)

    def _detect_1d(self, data: np.ndarray) -> MADResult:
        """Detect outliers in 1D data."""
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array for mode='1d', got shape {data.shape}")

        median, mad = self.compute_mad(data)
        scaled_mad = mad * MAD_SCALE_FACTOR

        if scaled_mad < 1e-10:
            # All values essentially identical
            logger.warning("MAD is near zero - data may be constant")
            outlier_mask = np.zeros(len(data), dtype=bool)
        else:
            deviations = np.abs(data - median)
            outlier_mask = deviations > self.threshold * scaled_mad

        outlier_indices = np.where(outlier_mask)[0]
        inlier_indices = np.where(~outlier_mask)[0]
        n_outliers = int(np.sum(outlier_mask))

        if n_outliers > 0:
            logger.info(f"MAD (1d) detected {n_outliers}/{len(data)} outliers")

        return MADResult(
            outlier_mask=outlier_mask,
            median=np.array([median]),
            mad=np.array([mad]),
            threshold=self.threshold,
            n_outliers=n_outliers,
            mode="1d",
            statistic="",
            inlier_indices=inlier_indices,
            outlier_indices=outlier_indices,
        )

    def _detect_spectrum(self, data: np.ndarray) -> MADResult:
        """Detect outlier spectra based on summary statistic."""
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array for mode='spectrum', got shape {data.shape}")

        # Compute statistic per spectrum
        if self.statistic == "total_intensity":
            values = np.sum(data, axis=1)
        elif self.statistic == "max_intensity":
            values = np.max(data, axis=1)
        elif self.statistic == "mean_intensity":
            values = np.mean(data, axis=1)
        else:  # std_intensity
            values = np.std(data, axis=1)

        median, mad = self.compute_mad(values)
        scaled_mad = mad * MAD_SCALE_FACTOR

        if scaled_mad < 1e-10:
            logger.warning("MAD is near zero - spectra may be nearly identical")
            outlier_mask = np.zeros(len(values), dtype=bool)
        else:
            deviations = np.abs(values - median)
            outlier_mask = deviations > self.threshold * scaled_mad

        outlier_indices = np.where(outlier_mask)[0]
        inlier_indices = np.where(~outlier_mask)[0]
        n_outliers = int(np.sum(outlier_mask))

        if n_outliers > 0:
            logger.info(
                f"MAD (spectrum/{self.statistic}) detected {n_outliers}/{len(values)} outliers"
            )

        return MADResult(
            outlier_mask=outlier_mask,
            median=np.array([median]),
            mad=np.array([mad]),
            threshold=self.threshold,
            n_outliers=n_outliers,
            mode="spectrum",
            statistic=self.statistic,
            inlier_indices=inlier_indices,
            outlier_indices=outlier_indices,
        )

    def _detect_channel(self, data: np.ndarray) -> MADResult:
        """Detect outliers independently at each wavelength channel."""
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array for mode='channel', got shape {data.shape}")

        n_spectra, n_channels = data.shape

        # Compute MAD along spectrum axis (axis=0) for each channel
        median, mad = self.compute_mad(data, axis=0)
        scaled_mad = mad * MAD_SCALE_FACTOR

        # Compute deviations from median for each (spectrum, channel)
        deviations = np.abs(data - median)

        # Handle channels with zero MAD (constant or nearly constant across spectra)
        # When MAD=0, we still want to detect outliers if any deviation exists
        # Use the maximum deviation as a fallback scale, or flag any deviation > 0
        zero_mad_mask = scaled_mad < 1e-10
        if np.any(zero_mad_mask):
            # For channels with zero MAD, flag any point that deviates significantly
            # from the median (use absolute threshold based on median magnitude)
            median_magnitude = np.abs(median)
            # Fallback: flag if deviation > threshold% of median (or > 1 if median ~ 0)
            fallback_threshold = np.maximum(median_magnitude * 0.01, 1.0)
            safe_scaled_mad = np.where(zero_mad_mask, fallback_threshold, scaled_mad)
        else:
            safe_scaled_mad = scaled_mad

        outlier_mask = deviations > self.threshold * safe_scaled_mad

        n_outliers = int(np.sum(outlier_mask))

        if n_outliers > 0:
            logger.info(
                f"MAD (channel) detected {n_outliers}/{data.size} outlier values "
                f"across {n_spectra} spectra × {n_channels} channels"
            )

        return MADResult(
            outlier_mask=outlier_mask,
            median=median,
            mad=mad,
            threshold=self.threshold,
            n_outliers=n_outliers,
            mode="channel",
            statistic="",
            inlier_indices=None,
            outlier_indices=None,
        )

    def filter_spectra(
        self,
        spectra: np.ndarray,
    ) -> Tuple[np.ndarray, MADResult]:
        """
        Filter out outlier spectra (spectrum mode only).

        Parameters
        ----------
        spectra : np.ndarray
            2D array of shape (n_spectra, n_wavelengths)

        Returns
        -------
        filtered_spectra : np.ndarray
            Spectra with outliers removed
        result : MADResult
            Detection results
        """
        if self.mode != "spectrum":
            raise ValueError("filter_spectra only works with mode='spectrum'")

        result = self.detect_outliers(spectra)
        filtered = spectra[~result.outlier_mask]
        return filtered, result

    def clean_channels(
        self,
        spectra: np.ndarray,
        replacement: str = "median",
    ) -> Tuple[np.ndarray, MADResult]:
        """
        Replace outlier channel values with robust estimate.

        Parameters
        ----------
        spectra : np.ndarray
            2D array of shape (n_spectra, n_wavelengths)
        replacement : str
            Replacement method:
            - 'median': Replace with channel median (default)
            - 'nan': Replace with NaN
            - 'interpolate': Linear interpolation from neighbors

        Returns
        -------
        cleaned_spectra : np.ndarray
            Spectra with outlier values replaced
        result : MADResult
            Detection results
        """
        if self.mode != "channel":
            raise ValueError("clean_channels only works with mode='channel'")
        if replacement not in ("median", "nan", "interpolate"):
            raise ValueError(f"Unknown replacement method: {replacement}")

        result = self.detect_outliers(spectra)
        cleaned = spectra.copy()

        if replacement == "median":
            # Broadcast median to match outlier positions
            for i in range(spectra.shape[0]):
                mask = result.outlier_mask[i]
                cleaned[i, mask] = result.median[mask]
        elif replacement == "nan":
            cleaned[result.outlier_mask] = np.nan
        else:  # interpolate
            for i in range(spectra.shape[0]):
                mask = result.outlier_mask[i]
                if np.any(mask):
                    good_idx = np.where(~mask)[0]
                    bad_idx = np.where(mask)[0]
                    if len(good_idx) >= 2:
                        cleaned[i, bad_idx] = np.interp(bad_idx, good_idx, cleaned[i, good_idx])
                    else:
                        # Not enough good points for interpolation
                        cleaned[i, bad_idx] = result.median[bad_idx]

        return cleaned, result


def mad_outliers_1d(
    data: np.ndarray,
    threshold: float = 3.0,
) -> MADResult:
    """
    Convenience function for MAD outlier detection on 1D data.

    Parameters
    ----------
    data : np.ndarray
        1D array of values
    threshold : float
        Number of scaled MAD units (default: 3.0)

    Returns
    -------
    MADResult
        Detection results
    """
    detector = MADOutlierDetector(threshold=threshold, mode="1d")
    return detector.detect_outliers(data)


def mad_outliers_spectra(
    spectra: np.ndarray,
    threshold: float = 3.0,
    statistic: str = "total_intensity",
) -> MADResult:
    """
    Convenience function for MAD outlier detection on spectra.

    Flags entire spectra based on a summary statistic.

    Parameters
    ----------
    spectra : np.ndarray
        2D array of shape (n_spectra, n_wavelengths)
    threshold : float
        Number of scaled MAD units (default: 3.0)
    statistic : str
        Statistic to use: 'total_intensity', 'max_intensity',
        'mean_intensity', or 'std_intensity'

    Returns
    -------
    MADResult
        Detection results
    """
    detector = MADOutlierDetector(threshold=threshold, mode="spectrum", statistic=statistic)
    return detector.detect_outliers(spectra)


def mad_clean_channels(
    spectra: np.ndarray,
    threshold: float = 3.0,
    replacement: str = "median",
) -> Tuple[np.ndarray, MADResult]:
    """
    Clean outlier channel values in spectra using MAD.

    Parameters
    ----------
    spectra : np.ndarray
        2D array of shape (n_spectra, n_wavelengths)
    threshold : float
        Number of scaled MAD units (default: 3.0)
    replacement : str
        'median', 'nan', or 'interpolate'

    Returns
    -------
    cleaned_spectra : np.ndarray
        Spectra with outlier values replaced
    result : MADResult
        Detection results
    """
    detector = MADOutlierDetector(threshold=threshold, mode="channel")
    return detector.clean_channels(spectra, replacement=replacement)
