"""
Instrument model for spectrometer response.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from pathlib import Path

from cflibs.core.logging_config import get_logger

logger = get_logger("instrument.model")


@dataclass
class InstrumentModel:
    """
    Model for spectrometer instrument response.

    Attributes
    ----------
    resolution_fwhm_nm : float
        Instrument resolution (FWHM) in nm
    response_curve : Optional[np.ndarray]
        Spectral response curve (wavelength, response) pairs
    wavelength_calibration : Optional[callable]
        Function to convert pixel to wavelength
    """

    resolution_fwhm_nm: float = 0.0
    response_curve: Optional[np.ndarray] = None
    wavelength_calibration: Optional[Callable[..., float]] = None
    resolving_power: Optional[float] = None

    @property
    def resolution_sigma_nm(self) -> float:
        """Gaussian standard deviation for instrument function."""
        return self.resolution_fwhm_nm / 2.355

    @property
    def is_resolving_power_mode(self) -> bool:
        """True if instrument is configured with a valid resolving power R > 0."""
        return self.resolving_power is not None and self.resolving_power > 0

    def sigma_at_wavelength(self, wavelength_nm: float) -> float:
        """
        Compute Gaussian sigma at a given wavelength.

        In resolving-power mode, FWHM = lambda / R varies with wavelength.
        In fixed-FWHM mode, returns the constant resolution_sigma_nm.

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nm

        Returns
        -------
        float
            Gaussian standard deviation in nm

        Raises
        ------
        ValueError
            If wavelength_nm <= 0 or resolving_power <= 0.
        """
        if wavelength_nm <= 0:
            raise ValueError(f"wavelength_nm must be positive; got {wavelength_nm!r}")
        if self.resolving_power is not None:
            if self.resolving_power <= 0:
                raise ValueError(f"resolving_power must be positive; got {self.resolving_power!r}")
            fwhm = wavelength_nm / self.resolving_power
            return fwhm / 2.355
        return self.resolution_sigma_nm

    @classmethod
    def from_resolving_power(
        cls, resolving_power: float, response_curve: Optional[np.ndarray] = None
    ) -> "InstrumentModel":
        """
        Create InstrumentModel from resolving power R = lambda / FWHM.

        Parameters
        ----------
        resolving_power : float
            Resolving power (dimensionless)
        response_curve : array, optional
            Spectral response curve

        Returns
        -------
        InstrumentModel

        Raises
        ------
        ValueError
            If resolving_power <= 0.
        """
        if resolving_power <= 0:
            raise ValueError(f"resolving_power must be positive; got {resolving_power!r}")
        return cls(
            resolution_fwhm_nm=0.0,
            response_curve=response_curve,
            resolving_power=resolving_power,
        )

    @classmethod
    def from_file(cls, config_path: Path) -> "InstrumentModel":
        """
        Load instrument model from configuration file.

        Parameters
        ----------
        config_path : Path
            Path to configuration file

        Returns
        -------
        InstrumentModel
            Instrument model instance
        """
        from cflibs.core.config import load_config

        config = load_config(config_path)

        if "instrument" not in config:
            raise ValueError("Configuration must contain 'instrument' section")

        instr_config = config["instrument"]

        resolution = instr_config.get("resolution_fwhm_nm")
        if resolution is None:
            raise ValueError("Instrument config must specify 'resolution_fwhm_nm'")

        response_file = instr_config.get("response_curve")
        response_curve = None
        if response_file:
            # Load response curve from file
            response_path = Path(response_file)
            if response_path.exists():
                data = np.loadtxt(response_path, delimiter=",")
                response_curve = data
            else:
                logger.warning(f"Response curve file not found: {response_file}")

        return cls(resolution_fwhm_nm=resolution, response_curve=response_curve)

    def apply_response(self, wavelength: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """
        Apply spectral response curve.

        Parameters
        ----------
        wavelength : array
            Wavelength grid in nm
        intensity : array
            Intensity spectrum

        Returns
        -------
        array
            Intensity with response applied
        """
        if self.response_curve is None:
            return intensity

        # Interpolate response curve onto wavelength grid
        from scipy.interpolate import interp1d

        wl_resp = self.response_curve[:, 0]
        resp = self.response_curve[:, 1]

        # Normalize response to max = 1
        resp = resp / resp.max()

        f = interp1d(wl_resp, resp, kind="linear", bounds_error=False, fill_value=0.0)
        response = f(wavelength)

        return intensity * response