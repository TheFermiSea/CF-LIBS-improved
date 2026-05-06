"""
Plasma broadening models for line width analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp1d

from cflibs.atomic.database import AtomicDatabase
from cflibs.core.logging_config import get_logger

logger = get_logger("plasma.broadening")


class NonHydrogenicStarkInterpolator:
    """
    Interpolator for STARK-B broadening widths w(T, ne).

    Supports bilinear interpolation in (T, ne) space and provides
    inversion to solve for electron density given a measured Stark width.
    """

    def __init__(self, t_grid: np.ndarray, ne_grid: np.ndarray, w_grid: np.ndarray):
        """
        Initialize interpolator.

        Parameters
        ----------
        t_grid : np.ndarray
            Temperature grid [K]
        ne_grid : np.ndarray
            Electron density grid [cm^-3]
        w_grid : np.ndarray
            Width grid (T x ne) [pm]
        """
        # Ensure grids are sorted
        t_idx = np.argsort(t_grid)
        ne_idx = np.argsort(ne_grid)

        self._t = t_grid[t_idx]
        self._ne = ne_grid[ne_idx]
        self._w = w_grid[t_idx][:, ne_idx]

        if len(self._ne) > 1 and len(self._t) > 1:
            self._interp = RegularGridInterpolator(
                (self._t, self._ne), self._w, method="linear", bounds_error=False, fill_value=None
            )
            self._mode = "2d"
        else:
            # Fallback to 1D interpolation in T and linear scaling in ne
            # Use the first available ne point as reference
            self._interp_t = interp1d(
                self._t, self._w[:, 0], bounds_error=False, fill_value="extrapolate"
            )
            self._mode = "1d-t"

    def get_width(self, t_k: float, ne_cm3: float) -> float:
        """
        Interpolate Stark width w at given T and ne.

        Parameters
        ----------
        t_k : float
            Temperature in K
        ne_cm3 : float
            Electron density in cm^-3

        Returns
        -------
        float
            Stark width (FWHM) in pm
        """
        if self._mode == "2d":
            point = np.array([[t_k, ne_cm3]])
            return float(self._interp(point)[0])
        else:
            # Linear scaling in ne: w(T, ne) = w(T, ne_ref) * (ne / ne_ref)
            w_at_t = float(self._interp_t(t_k))
            return w_at_t * (ne_cm3 / self._ne[0])

    def solve_ne(self, t_k: float, measured_stark_w_pm: float) -> float:
        """
        Solve for ne given temperature and measured Stark width.

        Leverages the approximately linear scaling of Stark width with ne.

        Parameters
        ----------
        t_k : float
            Temperature in K
        measured_stark_w_pm : float
            Measured Stark component of the FWHM in pm

        Returns
        -------
        float
            Estimated electron density in cm^-3
        """
        # Pick a reference ne from the grid
        idx = len(self._ne) // 2
        ne_ref = self._ne[idx]
        w_ref = self.get_width(t_k, ne_ref)

        if w_ref <= 0 or np.isnan(w_ref):
            # Fallback to the first point
            ne_ref = self._ne[0]
            w_ref = self.get_width(t_k, ne_ref)

        if w_ref <= 0 or np.isnan(w_ref):
            return np.nan

        # Linear scaling: ne = ne_ref * (measured_w / w_ref)
        return ne_ref * (measured_stark_w_pm / w_ref)


def get_stark_interpolator(
    db: AtomicDatabase, element: str, ionization_stage: int, wavelength_nm: float
) -> NonHydrogenicStarkInterpolator | None:
    """
    Create a Stark width interpolator for a specific line if data is available.

    Parameters
    ----------
    db : AtomicDatabase
        Atomic database instance
    element : str
        Element symbol
    ionization_stage : int
        Ionization stage
    wavelength_nm : float
        Wavelength in nm

    Returns
    -------
    NonHydrogenicStarkInterpolator or None
    """
    df = db.get_stark_width_grid(element, ionization_stage, wavelength_nm)
    if df.empty:
        return None

    # Get unique grid points
    t_grid = np.sort(df["t_k"].unique())
    ne_grid = np.sort(df["ne_cm3"].unique())

    try:
        w_pivot = df.pivot(index="t_k", columns="ne_cm3", values="w_pm")
        w_pivot = w_pivot.reindex(index=t_grid, columns=ne_grid)
        w_values = w_pivot.values
    except Exception as e:
        logger.error(
            f"Failed to pivot Stark width grid for {element} {ionization_stage} @ {wavelength_nm}nm: {e}"
        )
        return None

    if np.any(np.isnan(w_values)):
        logger.warning(
            f"Incomplete Stark width grid for {element} {ionization_stage} @ {wavelength_nm}nm"
        )
        if np.isnan(w_values).sum() > (w_values.size * 0.5):
            return None

    return NonHydrogenicStarkInterpolator(t_grid, ne_grid, w_values)


def estimate_ne_from_lines(
    db: AtomicDatabase, t_k: float, line_data: list[tuple[str, int, float, float]]
) -> float:
    """
    Estimate electron density from multiple non-hydrogenic lines.

    Parameters
    ----------
    db : AtomicDatabase
        Atomic database instance
    t_k : float
        Plasma temperature in K
    line_data : list of (element, sp_num, wavelength_nm, measured_stark_w_pm)
        List of measured line Stark widths

    Returns
    -------
    float
        Median estimated electron density in cm^-3
    """
    ne_estimates = []
    for elem, sp, wave, w_meas in line_data:
        interp = get_stark_interpolator(db, elem, sp, wave)
        if interp:
            ne = interp.solve_ne(t_k, w_meas)
            if not np.isnan(ne):
                ne_estimates.append(ne)

    if not ne_estimates:
        return np.nan

    return float(np.median(ne_estimates))
