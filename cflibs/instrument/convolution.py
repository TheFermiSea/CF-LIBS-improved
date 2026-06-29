"""
Instrument function convolution.
"""

import numpy as np
from scipy import signal

from cflibs.core.jax_runtime import HAS_JAX, jnp  # noqa: F401
from cflibs.core.logging_config import get_logger

logger = get_logger("instrument.convolution")


def apply_instrument_function(
    wavelength: np.ndarray, intensity: np.ndarray, sigma_nm: float
) -> np.ndarray:
    """
    Apply Gaussian instrument function via convolution.

    Parameters
    ----------
    wavelength : array
        Wavelength grid in nm (must be evenly spaced)
    intensity : array
        Intensity spectrum
    sigma_nm : float
        Gaussian standard deviation in nm

    Returns
    -------
    array
        Convolved intensity spectrum
    """
    # A non-positive sigma means no instrument broadening. The kernel would be
    # degenerate — kernel = exp(-0.5*(0/0)^2) = NaN — so return the input
    # unchanged, matching SpectrumModel._apply_instrument_convolution's
    # sigma<=0 short-circuit (audit M1-A2 / instrument F10).
    if sigma_nm <= 0:
        return np.asarray(intensity)

    # Check if wavelength grid is evenly spaced
    dwl = np.diff(wavelength)
    if not np.allclose(dwl, dwl[0], rtol=1e-6):
        raise ValueError("Wavelength grid must be evenly spaced for convolution")

    delta_wl = dwl[0]

    # Create Gaussian kernel
    # Use 5 sigma on each side
    n_sigma = 5
    kernel_size = int(2 * n_sigma * sigma_nm / delta_wl)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel_wl = np.linspace(-n_sigma * sigma_nm, n_sigma * sigma_nm, kernel_size)
    kernel = np.exp(-0.5 * (kernel_wl / sigma_nm) ** 2)
    kernel = kernel / kernel.sum()  # Normalize

    # Convolve
    convolved = signal.convolve(intensity, kernel, mode="same")

    return convolved


def apply_instrument_function_jax(
    wavelength: np.ndarray, intensity: np.ndarray, sigma_nm: float
) -> np.ndarray:
    """
    Apply Gaussian instrument function via JAX convolution.

    Parameters
    ----------
    wavelength : array
        Wavelength grid in nm (must be evenly spaced)
    intensity : array
        Intensity spectrum
    sigma_nm : float
        Gaussian standard deviation in nm

    Returns
    -------
    array
        Convolved intensity spectrum
    """
    if not HAS_JAX:
        raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")

    # Non-positive sigma => no broadening (degenerate NaN kernel otherwise).
    if sigma_nm <= 0:
        return np.asarray(intensity)

    wavelength_j = jnp.asarray(wavelength)
    intensity_j = jnp.asarray(intensity)

    dwl = jnp.diff(wavelength_j)
    # Use rtol=1e-4 for JAX (float32 default has ~1e-5 precision vs float64's ~1e-13)
    if not bool(jnp.allclose(dwl, dwl[0], rtol=1e-4)):
        raise ValueError("Wavelength grid must be evenly spaced for convolution")

    delta_wl = float(dwl[0])

    n_sigma = 5
    kernel_size = int(2 * n_sigma * sigma_nm / delta_wl)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3

    kernel_wl = jnp.linspace(-n_sigma * sigma_nm, n_sigma * sigma_nm, kernel_size)
    kernel = jnp.exp(-0.5 * (kernel_wl / sigma_nm) ** 2)
    kernel = kernel / jnp.sum(kernel)

    convolved = jnp.convolve(intensity_j, kernel, mode="same")
    return np.array(convolved)
