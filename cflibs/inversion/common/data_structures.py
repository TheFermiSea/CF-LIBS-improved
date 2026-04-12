"""
Pure data structures for the CF-LIBS inversion pipeline.

These dataclasses are used across multiple inversion sub-packages (preprocessing,
physics, identification, solving). They live here to avoid circular imports when
the inversion package is split into sub-packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FitMethod(Enum):
    """Fitting method for Boltzmann plot regression."""

    SIGMA_CLIP = "sigma_clip"  # Iterative sigma clipping (default)
    RANSAC = "ransac"  # Random Sample Consensus
    HUBER = "huber"  # Huber M-estimation


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
