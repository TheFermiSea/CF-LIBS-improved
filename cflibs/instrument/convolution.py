"""
Instrument function convolution.
"""

import numpy as np
from scipy import signal

try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None  # type: ignore[assignment]

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
