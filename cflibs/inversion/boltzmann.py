"""
Boltzmann plot generation and fitting for CF-LIBS.

This module provides robust Boltzmann plot fitting with multiple outlier rejection
methods based on literature recommendations:

- **Sigma-clipping**: Iterative rejection based on residual standard deviation
- **RANSAC**: Random Sample Consensus for robust fitting with gross outliers
- **Huber**: M-estimation with Huber loss function for moderate outliers

References:
    - El Sherbini et al. (2020): Curve-of-growth self-absorption detection
    - Huber, P.J. (1981): Robust Statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.boltzmann")


class FitMethod(Enum):
    """Fitting method for Boltzmann plot regression."""

    SIGMA_CLIP = "sigma_clip"  # Iterative sigma clipping (default)
    RANSAC = "ransac"  # Random Sample Consensus
    HUBER = "huber"  # Huber M-estimation


try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = object  # Dummy for type hinting if missing


@dataclass
class LineObservation:
    """
    Represents a single spectral line observation.
    """

    wavelength_nm: float
    intensity: float  # measured intensity (integrated area)
    intensity_uncertainty: float
    element: str
    ionization_stage: int
    E_k_ev: float  # upper level energy in eV
    g_k: int  # statistical weight of upper level
    A_ki: float  # Einstein coefficient in s^-1

    @property
    def y_value(self) -> float:
        """Calculate y-axis value: ln(I * lambda / (g * A))."""
        # Note: Intensity units are arbitrary, but consistent relative to each other.
        # wavelength in nm is fine as long as consistent.
        # Term inside log must be dimensionless or units handled in intercept.
        # Standard usage: ln( I * lambda[nm] / (g * A[s^-1]) )
        if self.intensity <= 0:
            return -np.inf
        return np.log(self.intensity * self.wavelength_nm / (self.g_k * self.A_ki))

    @property
    def y_uncertainty(self) -> float:
        """
        Calculate uncertainty in y-axis value.
        dy = d(ln x) = dx / x
        Here x = I * ...
        dx/x = dI/I (assuming errors in lambda, g, A are negligible)
        """
        if self.intensity == 0:
            return 0.0
        return self.intensity_uncertainty / self.intensity


@dataclass
class BoltzmannFitResult:
    """
    Results of a Boltzmann plot fit.

    Attributes
    ----------
    temperature_K : float
        Excitation temperature in Kelvin
    temperature_uncertainty_K : float
        1-sigma uncertainty in temperature
    intercept : float
        Y-intercept of Boltzmann plot (related to column density)
    intercept_uncertainty : float
        1-sigma uncertainty in intercept
    r_squared : float
        Coefficient of determination (goodness of fit)
    n_points : int
        Number of points used in final fit
    rejected_points : list[int]
        Indices of rejected outlier points
    slope : float
        Slope of Boltzmann plot = -1/(kB*T)
    slope_uncertainty : float
        1-sigma uncertainty in slope
    fit_method : str
        Method used for fitting ('sigma_clip', 'ransac', 'huber')
    n_iterations : int
        Number of iterations performed
    inlier_mask : np.ndarray
        Boolean mask of inlier points (True = used in fit)
    covariance_matrix : np.ndarray, optional
        2x2 covariance matrix [[var(slope), cov], [cov, var(intercept)]].
        Used for propagating correlated uncertainties.
    """

    temperature_K: float
    temperature_uncertainty_K: float
    intercept: float
    intercept_uncertainty: float
    r_squared: float
    n_points: int
    rejected_points: list[int]  # Indices of rejected points
    slope: float
    slope_uncertainty: float
    fit_method: str = "sigma_clip"
    n_iterations: int = 1
    inlier_mask: np.ndarray | None = field(default=None, repr=False)
    covariance_matrix: np.ndarray | None = field(default=None, repr=False)


class BoltzmannPlotFitter:
    """
    Fitter for Boltzmann plots to determine excitation temperature.

    Supports multiple robust fitting methods:

    - **sigma_clip** (default): Iterative sigma-clipping of outliers
    - **ransac**: Random Sample Consensus for gross outlier rejection
    - **huber**: M-estimation with Huber loss function

    Examples
    --------
    >>> fitter = BoltzmannPlotFitter(method=FitMethod.RANSAC)
    >>> result = fitter.fit(observations)
    >>> print(f"Temperature: {result.temperature_K:.0f} K")
    """

    def __init__(
        self,
        outlier_sigma: float = 3.0,
        max_iterations: int = 5,
        method: FitMethod = FitMethod.SIGMA_CLIP,
        ransac_min_samples: int = 2,
        ransac_residual_threshold: float | None = None,
        ransac_max_trials: int = 100,
        huber_epsilon: float = 1.35,
    ):
        """
        Initialize fitter.

        Parameters
        ----------
        outlier_sigma : float
            Sigma threshold for outlier rejection (sigma_clip method)
        max_iterations : int
            Maximum iterations for sigma clipping
        method : FitMethod
            Fitting method to use (SIGMA_CLIP, RANSAC, or HUBER)
        ransac_min_samples : int
            Minimum number of samples for RANSAC model fitting (default: 2)
        ransac_residual_threshold : float, optional
            Maximum residual for a point to be considered an inlier in RANSAC.
            If None, uses median absolute deviation of residuals.
        ransac_max_trials : int
            Maximum RANSAC iterations (default: 100)
        huber_epsilon : float
            Huber loss transition point. Points with standardized residuals
            below this use squared loss; above use linear loss. Default: 1.35
            (95% efficiency for Gaussian errors)
        """
        self.outlier_sigma = outlier_sigma
        self.max_iterations = max_iterations
        self.method = method
        self.ransac_min_samples = ransac_min_samples
        self.ransac_residual_threshold = ransac_residual_threshold
        self.ransac_max_trials = ransac_max_trials
        self.huber_epsilon = huber_epsilon

    def fit(self, observations: list[LineObservation]) -> BoltzmannFitResult:
        """
        Perform robust linear regression on Boltzmann plot data.

        The fitting method is determined by the `method` parameter set during
        initialization. All methods return consistent results with outlier
        identification.

        Parameters
        ----------
        observations : list[LineObservation]
            List of line observations

        Returns
        -------
        BoltzmannFitResult
            Fit results including temperature, uncertainties, and outlier info

        Raises
        ------
        ValueError
            If fewer than 2 valid observations provided
        """
        if len(observations) < 2:
            raise ValueError("Need at least 2 points for a fit")

        # Prepare arrays
        x_all = np.array([obs.E_k_ev for obs in observations])
        y_all = np.array([obs.y_value for obs in observations])
        y_err_all = np.array([obs.y_uncertainty for obs in observations])

        # Handle cases where y calculation failed (e.g. negative intensity)
        valid_mask = np.isfinite(y_all)
        if not np.all(valid_mask):
            logger.warning(f"Excluding {np.sum(~valid_mask)} points with invalid Y values")

        # Route to appropriate fitting method
        if self.method == FitMethod.RANSAC:
            return self._fit_ransac(x_all, y_all, y_err_all, valid_mask)
        elif self.method == FitMethod.HUBER:
            return self._fit_huber(x_all, y_all, y_err_all, valid_mask)
        else:  # SIGMA_CLIP (default)
            return self._fit_sigma_clip(x_all, y_all, y_err_all, valid_mask)

    def _fit_sigma_clip(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        y_err_all: np.ndarray,
        valid_mask: np.ndarray,
    ) -> BoltzmannFitResult:
        """Iterative sigma-clipping fit."""
        indices = np.arange(len(x_all))
        mask = valid_mask.copy()

        slope = 0.0
        intercept = 0.0
        slope_err = 0.0
        intercept_err = 0.0
        r_squared = 0.0
        n_iterations = 0
        covariance_matrix = None

        for iteration in range(self.max_iterations):
            n_iterations = iteration + 1
            x = x_all[mask]
            y = y_all[mask]
            y_err = y_err_all[mask]

            if len(x) < 2:
                logger.warning("Too few points remaining after rejection")
                break

            # Weighted least squares
            weights = self._compute_weights(y_err)

            try:
                # Need >2 points for covariance estimation
                if len(x) > 2:
                    (m, c), cov = np.polyfit(x, y, 1, w=weights, cov=True)
                    slope_err = np.sqrt(cov[0, 0])
                    intercept_err = np.sqrt(cov[1, 1])
                    covariance_matrix = cov
                else:
                    m, c = np.polyfit(x, y, 1, w=weights)
                    slope_err = np.inf
                    intercept_err = np.inf
                    covariance_matrix = None
            except np.linalg.LinAlgError:
                logger.error("Linear regression failed")
                return self._empty_result()

            slope = m
            intercept = c

            # Calculate R^2

            y_pred = m * x + c
            r_squared = self._compute_r_squared(y, y_pred, weights)

            # Check for outliers

            residuals = y - y_pred
            std_res = np.std(residuals)
            if std_res == 0:
                break

            bad_indices = np.abs(residuals) > self.outlier_sigma * std_res

            if not np.any(bad_indices):
                break

            current_indices = indices[mask]
            outlier_global_indices = current_indices[bad_indices]
            mask[outlier_global_indices] = False

            logger.debug(f"Iteration {iteration}: Rejected {len(outlier_global_indices)} outliers")

        return self._create_result(
            slope,
            slope_err,
            intercept,
            intercept_err,
            r_squared,
            mask,
            indices,
            "sigma_clip",
            n_iterations,
            covariance_matrix,
        )

    def _fit_ransac(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        y_err_all: np.ndarray,
        valid_mask: np.ndarray,
    ) -> BoltzmannFitResult:
        """
        RANSAC (Random Sample Consensus) robust fitting.

        RANSAC is effective for data with gross outliers (self-absorbed lines,
        misidentified lines) that would severely bias least squares.
        """
        indices = np.arange(len(x_all))
        mask = valid_mask.copy()

        x = x_all[mask]
        y = y_all[mask]
        y_err = y_err_all[mask]
        n_valid = len(x)

        if n_valid < self.ransac_min_samples:
            logger.warning(f"Too few valid points ({n_valid}) for RANSAC")
            return self._empty_result()

        # Compute threshold if not specified
        threshold = self.ransac_residual_threshold
        if threshold is None:
            # Use median absolute deviation as robust scale estimate
            weights = self._compute_weights(y_err)
            m_init, c_init = np.polyfit(x, y, 1, w=weights)
            residuals_init = np.abs(y - (m_init * x + c_init))
            mad = np.median(residuals_init)
            threshold = 3.0 * mad * 1.4826  # Scale MAD to std dev estimate

        best_inliers = None
        best_n_inliers = 0

        rng = np.random.default_rng(42)  # Reproducible results

        for _ in range(self.ransac_max_trials):
            # Random sample
            sample_idx = rng.choice(n_valid, size=self.ransac_min_samples, replace=False)
            x_sample = x[sample_idx]
            y_sample = y[sample_idx]

            # Fit model to sample
            if len(np.unique(x_sample)) < 2:
                continue  # Degenerate sample

            try:
                m, c = np.polyfit(x_sample, y_sample, 1)
            except np.linalg.LinAlgError:
                continue

            # Count inliers
            residuals = np.abs(y - (m * x + c))
            inliers = residuals < threshold
            n_inliers = np.sum(inliers)

            if n_inliers > best_n_inliers:
                best_n_inliers = n_inliers
                best_inliers = inliers

        if best_inliers is None or best_n_inliers < 2:
            logger.warning("RANSAC failed to find sufficient inliers")
            return self._empty_result()

        # Refit using all inliers with weights
        x_inliers = x[best_inliers]
        y_inliers = y[best_inliers]
        y_err_inliers = y_err[best_inliers]

        slope, intercept, slope_err, intercept_err, covariance_matrix = self._weighted_fit(
            x_inliers, y_inliers, y_err_inliers
        )

        # Compute R^2 on inliers
        weights = self._compute_weights(y_err_inliers)
        y_pred = slope * x_inliers + intercept
        r_squared = self._compute_r_squared(y_inliers, y_pred, weights)

        # Map inliers back to original indices
        valid_indices = indices[mask]
        final_mask = np.zeros_like(mask)
        final_mask[valid_indices[best_inliers]] = True

        return self._create_result(
            slope,
            slope_err,
            intercept,
            intercept_err,
            r_squared,
            final_mask,
            indices,
            "ransac",
            self.ransac_max_trials,
            covariance_matrix,
        )

    def _fit_huber(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        y_err_all: np.ndarray,
        valid_mask: np.ndarray,
    ) -> BoltzmannFitResult:
        """
        Huber M-estimation robust fitting.

        Uses iteratively reweighted least squares (IRLS) with Huber weights.
        More efficient than RANSAC for moderate outliers.
        """
        indices = np.arange(len(x_all))
        mask = valid_mask.copy()

        x = x_all[mask]
        y = y_all[mask]
        y_err = y_err_all[mask]

        if len(x) < 2:
            logger.warning("Too few valid points for Huber fit")
            return self._empty_result()

        # Initial weights from measurement uncertainties
        base_weights = self._compute_weights(y_err)
        combined_weights = base_weights.copy()  # Initialize for case with no iterations

        # IRLS with Huber weights
        slope, intercept = np.polyfit(x, y, 1, w=base_weights)
        n_iterations = 0

        for iteration in range(self.max_iterations):
            n_iterations = iteration + 1

            # Compute residuals and scale
            residuals = y - (slope * x + intercept)
            scale = np.median(np.abs(residuals)) * 1.4826  # MAD-based scale

            if scale < 1e-10:
                break  # Perfect fit

            # Standardized residuals
            u = residuals / scale

            # Huber weights: 1 for |u| <= epsilon, epsilon/|u| for |u| > epsilon
            huber_weights = np.where(
                np.abs(u) <= self.huber_epsilon, 1.0, self.huber_epsilon / np.abs(u)
            )

            # Combined weights
            combined_weights = base_weights * huber_weights

            # Refit
            try:
                new_slope, new_intercept = np.polyfit(x, y, 1, w=combined_weights)
            except np.linalg.LinAlgError:
                break

            # Check convergence
            if abs(new_slope - slope) < 1e-8 * abs(slope) + 1e-12:
                slope, intercept = new_slope, new_intercept
                break

            slope, intercept = new_slope, new_intercept

        # Final fit with covariance (using final weights)
        covariance_matrix = None
        try:
            (slope, intercept), cov = np.polyfit(x, y, 1, w=combined_weights, cov=True)
            slope_err = np.sqrt(cov[0, 0])
            intercept_err = np.sqrt(cov[1, 1])
            covariance_matrix = cov
        except np.linalg.LinAlgError:
            slope_err = np.inf
            intercept_err = np.inf

        # Identify outliers (points with very low Huber weight)
        residuals = y - (slope * x + intercept)
        scale = np.median(np.abs(residuals)) * 1.4826
        if scale > 0:
            u = np.abs(residuals / scale)
            inlier_points = u <= 3 * self.huber_epsilon  # Mark extreme outliers
        else:
            inlier_points = np.ones(len(x), dtype=bool)

        # Compute R^2 on all points
        y_pred = slope * x + intercept
        r_squared = self._compute_r_squared(y, y_pred, base_weights)

        # Map to original indices
        valid_indices = indices[mask]
        final_mask = np.zeros_like(mask)
        final_mask[valid_indices[inlier_points]] = True

        return self._create_result(
            slope,
            slope_err,
            intercept,
            intercept_err,
            r_squared,
            final_mask,
            indices,
            "huber",
            n_iterations,
            covariance_matrix,
        )

    def _compute_weights(self, y_err: np.ndarray) -> np.ndarray:
        """Compute regression weights from measurement uncertainties."""
        if np.all(y_err == 0):
            return np.ones_like(y_err)
        safe_err = np.where(y_err > 0, y_err, np.inf)
        return 1.0 / safe_err**2

    def _compute_r_squared(self, y: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted R-squared."""
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def _weighted_fit(self, x: np.ndarray, y: np.ndarray, y_err: np.ndarray) -> tuple:
        """Perform weighted linear fit returning slope, intercept, uncertainties, and covariance.

        Returns
        -------
        tuple
            (slope, intercept, slope_err, intercept_err, covariance_matrix)
            covariance_matrix is 2x2: [[var(slope), cov], [cov, var(intercept)]]
        """
        weights = self._compute_weights(y_err)
        try:
            (m, c), cov = np.polyfit(x, y, 1, w=weights, cov=True)
            return m, c, np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), cov
        except np.linalg.LinAlgError:
            m, c = np.polyfit(x, y, 1, w=weights)
            return m, c, np.inf, np.inf, None

    def _create_result(
        self,
        slope: float,
        slope_err: float,
        intercept: float,
        intercept_err: float,
        r_squared: float,
        mask: np.ndarray,
        indices: np.ndarray,
        method: str,
        n_iterations: int,
        covariance_matrix: np.ndarray | None = None,
    ) -> BoltzmannFitResult:
        """Create BoltzmannFitResult with temperature calculation.

        Parameters
        ----------
        covariance_matrix : np.ndarray, optional
            2x2 covariance matrix [[var(slope), cov], [cov, var(intercept)]].
            Used for propagating correlated uncertainties through Saha/closure.
        """
        if slope >= 0:
            logger.warning(
                f"Positive or zero slope ({slope}) detected. "
                "Population inversion or error. T set to infinity."
            )
            temperature_K = float("inf")
            temp_err_K = float("inf")
        else:
            temperature_K = -1.0 / (KB_EV * slope)
            temp_err_K = (temperature_K**2) * KB_EV * slope_err

        rejected_points = list(indices[~mask])

        return BoltzmannFitResult(
            temperature_K=temperature_K,
            temperature_uncertainty_K=temp_err_K,
            intercept=intercept,
            intercept_uncertainty=intercept_err,
            r_squared=r_squared,
            n_points=int(np.sum(mask)),
            rejected_points=rejected_points,
            slope=slope,
            slope_uncertainty=slope_err,
            fit_method=method,
            n_iterations=n_iterations,
            inlier_mask=mask.copy(),
            covariance_matrix=covariance_matrix,
        )

    def _empty_result(self) -> BoltzmannFitResult:
        """Return empty result for failed fits."""
        return BoltzmannFitResult(
            temperature_K=0.0,
            temperature_uncertainty_K=0.0,
            intercept=0.0,
            intercept_uncertainty=0.0,
            r_squared=0.0,
            n_points=0,
            rejected_points=[],
            slope=0.0,
            slope_uncertainty=0.0,
            fit_method=self.method.value,
            n_iterations=0,
            inlier_mask=None,
            covariance_matrix=None,
        )

    def plot(
        self, observations: list[LineObservation], result: BoltzmannFitResult, ax=None
    ) -> object | None:
        """
        Plot the Boltzmann plot.

        Parameters
        ----------
        observations : list[LineObservation]
            Data points
        result : BoltzmannFitResult
            Fit result
        ax : matplotlib.axes.Axes, optional
            Axes to plot on

        Returns
        -------
        Figure or None
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not installed, cannot plot.")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Extract data
        x_all = np.array([obs.E_k_ev for obs in observations])
        y_all = np.array([obs.y_value for obs in observations])
        y_err_all = np.array([obs.y_uncertainty for obs in observations])

        # Split into accepted and rejected
        rejected_set = set(result.rejected_points)
        accepted_mask = np.array([i not in rejected_set for i in range(len(observations))])

        # Plot accepted
        ax.errorbar(
            x_all[accepted_mask],
            y_all[accepted_mask],
            yerr=y_err_all[accepted_mask],
            fmt="o",
            color="blue",
            label="Accepted",
            alpha=0.7,
            capsize=3,
        )

        # Plot rejected
        if np.any(~accepted_mask):
            ax.scatter(
                x_all[~accepted_mask],
                y_all[~accepted_mask],
                marker="x",
                color="red",
                label="Rejected",
                zorder=5,
            )

        # Plot fit line
        # Use range of x
        x_min, x_max = np.min(x_all), np.max(x_all)
        x_range = np.linspace(x_min, x_max, 100)
        y_fit = result.slope * x_range + result.intercept

        label_fit = f"Fit: T = {result.temperature_K:.0f} ± {result.temperature_uncertainty_K:.0f} K\n$R^2$={result.r_squared:.3f}"
        ax.plot(x_range, y_fit, "k--", label=label_fit)

        ax.set_xlabel("Upper Level Energy $E_k$ (eV)")
        ax.set_ylabel("$\\ln(I \\lambda / g_k A_{ki})$")
        ax.set_title(
            f"Boltzmann Plot - {observations[0].element} {observations[0].ionization_stage}"
        )
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        return fig
