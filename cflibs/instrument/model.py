"""
Instrument model for spectrometer response.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from pathlib import Path

from cflibs.core.jax_runtime import HAS_JAX, jit_if_available, jnp  # noqa: F401
from cflibs.core.logging_config import get_logger

jit = jit_if_available

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


# ---------------------------------------------------------------------------
# JAX-accelerated instrument model
# ---------------------------------------------------------------------------


if HAS_JAX:

    @jit
    def _sigma_at_wavelength_jax(
        wavelength_nm: jnp.ndarray,
        resolving_power: jnp.ndarray,
        resolution_sigma_nm: jnp.ndarray,
        use_resolving_power: jnp.ndarray,
    ) -> jnp.ndarray:
        """Per-wavelength Gaussian sigma — branchless for JIT compatibility.

        ``use_resolving_power`` is a 0/1 mask (jnp scalar) so the function
        can be jit-compiled regardless of which mode is active.
        """
        sigma_R = wavelength_nm / jnp.maximum(resolving_power, 1e-30) / 2.355
        return jnp.where(use_resolving_power > 0.5, sigma_R, resolution_sigma_nm)

    @jit
    def _apply_response_jax(
        wavelength: jnp.ndarray,
        intensity: jnp.ndarray,
        wl_resp: jnp.ndarray,
        resp: jnp.ndarray,
    ) -> jnp.ndarray:
        """Linear interpolation of the (normalized) response curve onto ``wavelength``.

        Uses ``jnp.interp`` (which becomes a fused linear-interp kernel
        under XLA) to avoid the SciPy interp1d dependency in JAX land.
        """
        resp_norm = resp / jnp.maximum(jnp.max(resp), 1e-30)
        # jnp.interp clamps out-of-range to the boundary — match SciPy's
        # ``fill_value=0.0`` behavior by zeroing those points explicitly.
        in_range = (wavelength >= wl_resp[0]) & (wavelength <= wl_resp[-1])
        interpolated = jnp.interp(wavelength, wl_resp, resp_norm)
        response = jnp.where(in_range, interpolated, 0.0)
        return intensity * response

else:  # pragma: no cover - JAX should be installed in this repo

    def _sigma_at_wavelength_jax(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")

    def _apply_response_jax(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")


@dataclass
class InstrumentModelJax(InstrumentModel):
    """JAX-friendly drop-in companion to :class:`InstrumentModel`.

    All public attributes from the parent are preserved. The JAX path
    differs from the NumPy path only in the implementation of the
    arithmetic helpers (``sigma_at_wavelength_array``, ``apply_response``):
    they evaluate on ``jnp.ndarray`` inputs and dispatch to JIT'd kernels
    so the work runs on the active accelerator when ``JAX_PLATFORMS=cuda``.

    Numerical equivalence with the NumPy parent is asserted by
    ``tests/instrument/test_model_jax.py`` within ``rtol=1e-5, atol=1e-7``.
    """

    def __post_init__(self) -> None:
        if not HAS_JAX:  # pragma: no cover - defensive
            raise ImportError("InstrumentModelJax requires JAX. Install with `pip install jax`.")
        # Pre-stage the response curve as jnp arrays so we don't pay the
        # H2D cost on every call.
        if self.response_curve is not None:
            curve = np.asarray(self.response_curve, dtype=np.float64)
            self._response_wl_jax = jnp.asarray(curve[:, 0])
            self._response_resp_jax = jnp.asarray(curve[:, 1])
        else:
            self._response_wl_jax = None
            self._response_resp_jax = None

    def sigma_at_wavelength_array(self, wavelength_nm) -> "jnp.ndarray":
        """Vectorised :py:meth:`InstrumentModel.sigma_at_wavelength`.

        Accepts a scalar or array of wavelengths and returns a ``jnp.ndarray``
        of Gaussian sigmas in nm.
        """
        wl = jnp.asarray(wavelength_nm, dtype=jnp.float64)
        use_R = 1.0 if self.is_resolving_power_mode else 0.0
        R = float(self.resolving_power) if self.resolving_power is not None else 1.0
        return _sigma_at_wavelength_jax(
            wl,
            jnp.asarray(R),
            jnp.asarray(float(self.resolution_sigma_nm)),
            jnp.asarray(float(use_R)),
        )

    def apply_response(self, wavelength: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Apply the spectral response curve via JAX (jnp.interp).

        Returns a NumPy array so callers (e.g. ``SpectrumModel``) don't have
        to special-case the device-array boundary.
        """
        if self.response_curve is None:
            return intensity
        wl_jnp = jnp.asarray(wavelength, dtype=jnp.float64)
        i_jnp = jnp.asarray(intensity, dtype=jnp.float64)
        result = _apply_response_jax(
            wl_jnp,
            i_jnp,
            self._response_wl_jax,
            self._response_resp_jax,
        )
        return np.asarray(result)

    @classmethod
    def from_resolving_power(
        cls, resolving_power: float, response_curve: Optional[np.ndarray] = None
    ) -> "InstrumentModelJax":
        if resolving_power <= 0:
            raise ValueError(f"resolving_power must be positive; got {resolving_power!r}")
        return cls(
            resolution_fwhm_nm=0.0,
            response_curve=response_curve,
            resolving_power=resolving_power,
        )

    @classmethod
    def from_instrument_model(cls, instrument: InstrumentModel) -> "InstrumentModelJax":
        """Promote an existing :class:`InstrumentModel` to the JAX variant."""
        return cls(
            resolution_fwhm_nm=instrument.resolution_fwhm_nm,
            response_curve=instrument.response_curve,
            wavelength_calibration=instrument.wavelength_calibration,
            resolving_power=instrument.resolving_power,
        )


# Wire JAX pytree registration so consumers can vmap over batched
# InstrumentModel instances. Idempotent.
from cflibs.core.jax_runtime import _ensure_pytrees_registered as _register_pytrees  # noqa: E402

_register_pytrees()
