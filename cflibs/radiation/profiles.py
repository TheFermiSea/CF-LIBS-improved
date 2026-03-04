"""
Line profile functions for spectral broadening.
Includes Gaussian, Lorentzian, and Voigt profiles.
"""

import numpy as np
import warnings
from enum import Enum
from typing import Union, Callable

try:
    from scipy.special import wofz as scipy_wofz

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

    def jit(f):
        return f


from cflibs.core.constants import C_LIGHT, EV_TO_J, M_PROTON
from cflibs.core.logging_config import get_logger

logger = get_logger("radiation.profiles")


class BroadeningMode(str, Enum):
    """
    Broadening mode for spectrum computation.

    LEGACY : Original behavior — single scalar sigma for all lines,
        plus separate downstream instrument convolution.
    NIST_PARITY : Per-line Gaussian sigma from resolving power R,
        sigma = lambda / (R * 2.355). No downstream convolution
        (broadening is fully captured in the per-line profile).
    PHYSICAL_DOPPLER : Per-line physical Doppler width via doppler_width(),
        with downstream instrument convolution still applied.
    """

    LEGACY = "legacy"
    NIST_PARITY = "nist_parity"
    PHYSICAL_DOPPLER = "physical_doppler"


def resolving_power_sigma(wavelength_nm: float, resolving_power: float) -> float:
    """
    Compute Gaussian sigma from spectrometer resolving power.

    NIST LIBS simulation defines Resolution = R = lambda / FWHM,
    so FWHM = lambda / R and sigma = FWHM / 2.355.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nm
    resolving_power : float
        Resolving power R = lambda / FWHM (dimensionless)

    Returns
    -------
    float
        Gaussian standard deviation in nm

    Raises
    ------
    ValueError
        If ``wavelength_nm`` <= 0 or ``resolving_power`` <= 0.
    """
    if resolving_power <= 0:
        raise ValueError(f"resolving_power must be positive; got {resolving_power!r}")
    if wavelength_nm <= 0:
        raise ValueError(f"wavelength_nm must be positive; got {wavelength_nm!r}")
    fwhm = wavelength_nm / resolving_power
    return fwhm / 2.355


def apply_gaussian_broadening_per_line(
    wavelength_grid: np.ndarray,
    line_wavelengths: np.ndarray,
    line_intensities: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """
    Apply Gaussian broadening with a per-line sigma array.

    Each line is broadened with its own Gaussian width, allowing
    wavelength-dependent broadening (e.g., from resolving power or
    physical Doppler width).

    Parameters
    ----------
    wavelength_grid : array
        Wavelength grid in nm
    line_wavelengths : array
        Line center wavelengths in nm
    line_intensities : array
        Line intensities (integrated area)
    sigmas : array
        Per-line Gaussian standard deviations in nm

    Returns
    -------
    array
        Broadened spectrum

    Raises
    ------
    ValueError
        If input array lengths differ or any sigma is <= 0.
    """
    if not (len(line_wavelengths) == len(line_intensities) == len(sigmas)):
        raise ValueError(
            f"Input length mismatch: line_wavelengths={len(line_wavelengths)}, "
            f"line_intensities={len(line_intensities)}, sigmas={len(sigmas)}"
        )
    if len(sigmas) > 0 and np.any(np.asarray(sigmas) <= 0):
        raise ValueError("All per-line sigma values must be > 0")

    spectrum = np.zeros_like(wavelength_grid, dtype=float)

    for wl, intensity, sig in zip(line_wavelengths, line_intensities, sigmas):
        profile = gaussian_profile(wavelength_grid, wl, sig, intensity)
        spectrum += profile

    return spectrum


def _raise_jax_missing() -> None:
    raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")


def gaussian_profile(
    wavelength: Union[float, np.ndarray], center: float, sigma: float, amplitude: float = 1.0
) -> Union[float, np.ndarray]:
    """
    Calculate Gaussian line profile.

    Parameters
    ----------
    wavelength : float or array
        Wavelength(s) in nm
    center : float
        Line center wavelength in nm
    sigma : float
        Standard deviation in nm
    amplitude : float
        Integrated area (not peak height).
        Note: Peak height = amplitude / (sigma * sqrt(2*pi))

    Returns
    -------
    float or array
        Profile value(s)
    """
    x = (wavelength - center) / sigma
    return amplitude * np.exp(-0.5 * x**2) / (sigma * np.sqrt(2 * np.pi))


def lorentzian_profile(
    wavelength: Union[float, np.ndarray], center: float, gamma: float, amplitude: float = 1.0
) -> Union[float, np.ndarray]:
    """
    Calculate Lorentzian line profile.

    Parameters
    ----------
    wavelength : float or array
        Wavelength(s) in nm
    center : float
        Line center wavelength in nm
    gamma : float
        HWHM (Half-Width at Half-Maximum) in nm.
    amplitude : float
        Integrated area.
        Peak height = amplitude / (gamma * pi)

    Returns
    -------
    float or array
        Profile value(s)
    """
    return (amplitude / np.pi) * (gamma / ((wavelength - center) ** 2 + gamma**2))


def doppler_width(wavelength_nm: float, T_eV: float, mass_amu: float) -> float:
    """
    Calculate Doppler broadening width (FWHM).
    """
    mass_kg = mass_amu * M_PROTON

    # Doppler FWHM = lambda * sqrt(8kT ln2 / mc^2)
    # sigma = FWHM / 2.355
    # T_eV is energy in eV. Convert to Joules: T_eV * EV_TO_J
    # sigma = lambda/c * sqrt(2 * E / m) ???
    # Standard Gaussian sigma for Doppler is lambda/c * sqrt(kT/m).
    # Roadmap asked for sqrt(2kT/mc^2). This corresponds to v_most_probable.
    # Let's use T_eV * EV_TO_J for kT.

    sigma = wavelength_nm * np.sqrt(2 * (T_eV * EV_TO_J) / (mass_kg * C_LIGHT**2))
    fwhm = 2.355 * sigma

    return fwhm


def voigt_fwhm(sigma: float, gamma: float) -> float:
    """
    Calculate approximate Voigt FWHM.
    Using Olivero and Longbothum approximation (1977).
    accuracy ~0.02%

    Parameters
    ----------
    sigma : float
        Gaussian standard deviation
    gamma : float
        Lorentzian HWHM

    Returns
    -------
    float
        FWHM of Voigt profile
    """
    fwhm_g = 2.35482 * sigma
    fwhm_l = 2.0 * gamma

    return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)


def voigt_profile(
    wavelength: Union[float, np.ndarray],
    center: float,
    sigma: float,
    gamma: float,
    amplitude: float = 1.0,
) -> Union[float, np.ndarray]:
    """
    Calculate Voigt profile.

    Parameters
    ----------
    wavelength : array
        Wavelength grid
    center : float
        Center
    sigma : float
        Gaussian std dev
    gamma : float
        Lorentzian HWHM
    amplitude : float
        Area
    """
    # Avoid division by zero
    if sigma < 1e-12:
        return lorentzian_profile(wavelength, center, gamma, amplitude)
    if gamma < 1e-12:
        return gaussian_profile(wavelength, center, sigma, amplitude)

    if HAS_SCIPY:
        z = (wavelength - center + 1j * gamma) / (sigma * np.sqrt(2))
        w_z = scipy_wofz(z)
        return amplitude * w_z.real / (sigma * np.sqrt(2 * np.pi))
    else:
        # Fallback to Pseudo-Voigt if no scipy
        fL = 2.0 * gamma
        fV = voigt_fwhm(sigma, gamma)

        # Mixing parameter eta (Pseudo-Voigt approximation)
        ratio = fL / fV
        eta = 1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3

        G = gaussian_profile(wavelength, center, sigma, amplitude)
        L = lorentzian_profile(wavelength, center, gamma, amplitude)
        return eta * L + (1 - eta) * G


def apply_gaussian_broadening(
    wavelength_grid: np.ndarray,
    line_wavelengths: np.ndarray,
    line_intensities: np.ndarray,
    sigma_nm: float,
) -> np.ndarray:
    """Apply Gaussian broadening to spectral lines."""
    spectrum = np.zeros_like(wavelength_grid)

    for wl, intensity in zip(line_wavelengths, line_intensities):
        profile = gaussian_profile(wavelength_grid, wl, sigma_nm, intensity)
        spectrum += profile

    return spectrum


def apply_voigt_broadening(
    wavelength_grid: np.ndarray,
    line_wavelengths: np.ndarray,
    line_intensities: np.ndarray,
    sigmas: Union[float, np.ndarray],
    gammas: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Apply Voigt broadening to spectral lines.

    Parameters
    ----------
    wavelength_grid : array
        Wavelength grid in nm
    line_wavelengths : array
        Line center wavelengths in nm
    line_intensities : array
        Line intensities (integrated area)
    sigmas : float or array
        Gaussian standard deviation in nm
    gammas : float or array
        Lorentzian HWHM in nm

    Returns
    -------
    array
        Broadened spectrum
    """
    spectrum = np.zeros_like(wavelength_grid)

    # Broadcast scalar inputs
    if np.isscalar(sigmas):
        sigmas = np.full_like(line_intensities, sigmas)
    if np.isscalar(gammas):
        gammas = np.full_like(line_intensities, gammas)

    for wl, intensity, sig, gam in zip(line_wavelengths, line_intensities, sigmas, gammas):
        p = voigt_profile(wavelength_grid, wl, sig, gam, intensity)
        spectrum += p

    return spectrum


def total_lorentzian_width(
    gamma_natural: float = 0.0, gamma_stark: float = 0.0, gamma_vdw: float = 0.0
) -> float:
    """
    Calculate total Lorentzian width (HWHM).

    Parameters
    ----------
    gamma_natural : float
        Natural broadening HWHM
    gamma_stark : float
        Stark broadening HWHM
    gamma_vdw : float
        Van der Waals broadening HWHM

    Returns
    -------
    float
        Total HWHM
    """
    return gamma_natural + gamma_stark + gamma_vdw


# --- JAX IMPLEMENTATION ---

if HAS_JAX:
    x64_enabled = bool(getattr(jax.config, "jax_enable_x64", False))
    _weideman_dtype = jnp.float64 if x64_enabled else jnp.float32
    if not x64_enabled:
        warnings.warn(
            "JAX x64 mode is disabled. Voigt profile coefficients will use float32 precision. "
            "Enable float64 with jax.config.update('jax_enable_x64', True) before importing "
            "cflibs.radiation.profiles for highest accuracy.",
            UserWarning,
            stacklevel=2,
        )

    # Weideman (1994) coefficients for Faddeeva function approximation
    # Reference: Weideman, SIAM J. Numer. Anal. 31, 1497 (1994)
    # N = 36 terms, provides up to ~15 digits in float64 mode
    _WEIDEMAN_L = 5.0453784915222872
    _WEIDEMAN_COEFFS = jnp.array(
        [
            5.3552841173932895e-14,
            -8.0527261170976810e-14,
            -3.2398883441056261e-13,
            4.4307993809438665e-13,
            2.0979949804464113e-12,
            -2.1169169127002517e-12,
            -1.4312512495461891e-11,
            6.3463874290909676e-12,
            9.9393262862192946e-11,
            3.1971993994865226e-11,
            -6.6348465239446123e-10,
            -9.0922385524685665e-10,
            3.7734430504796621e-09,
            1.1883887203463527e-08,
            -1.0962277931636141e-08,
            -1.1303157199293924e-07,
            -1.2894842925653411e-07,
            6.7416556638248690e-07,
            2.7654086656368491e-06,
            1.4187058478641208e-06,
            -2.1741186565542035e-05,
            -8.8177971418517626e-05,
            -1.1396630644455730e-04,
            4.6290316939990987e-04,
            3.5484447086997187e-03,
            1.3898253763251489e-02,
            4.1051043016576978e-02,
            1.0084293371847958e-01,
            2.1501636320107403e-01,
            4.0734241895033424e-01,
            6.9566219189710010e-01,
            1.0813580371765887e00,
            1.5401625788153652e00,
            2.0193976436113505e00,
            2.4453784928519209e00,
            2.7407450274098601e00,
        ],
        dtype=_weideman_dtype,
    )

    @jit
    def _faddeeva_weideman_jax(z: jnp.ndarray) -> jnp.ndarray:
        """
        Weideman rational approximation to Faddeeva function w(z).

        This is a branch-free implementation with stable gradients for all z,
        making it suitable for use with JAX autodiff (including MCMC sampling).

        The approximation uses N=36 terms and achieves up to ~15 digits in
        float64 mode across the complex plane.

        Parameters
        ----------
        z : complex array
            Complex argument(s)

        Returns
        -------
        complex array
            Faddeeva function w(z) = exp(-z²) * erfc(-iz)

        References
        ----------
        Weideman, J.A.C. (1994) "Computation of the Complex Error Function"
        SIAM J. Numer. Anal. 31, 1497-1518.
        """
        L = _WEIDEMAN_L

        # Möbius transformation to unit disk
        Z = (L + 1j * z) / (L - 1j * z)

        # Polynomial evaluation (coefficients precomputed via FFT)
        p = jnp.polyval(_WEIDEMAN_COEFFS, Z)

        # Final rational formula
        denom = L - 1j * z
        w = 2.0 * p / (denom * denom) + (1.0 / jnp.sqrt(jnp.pi)) / denom

        return w

    @jit
    def gaussian_profile_jax(
        wavelength: jnp.ndarray, center: float, sigma: float, amplitude: float = 1.0
    ) -> jnp.ndarray:
        """JAX-compatible Gaussian profile calculation."""
        x = (wavelength - center) / sigma
        return amplitude * jnp.exp(-0.5 * x**2) / (sigma * jnp.sqrt(2 * jnp.pi))

    @jit
    def lorentzian_profile_jax(
        wavelength: jnp.ndarray, center: float, gamma: float, amplitude: float = 1.0
    ) -> jnp.ndarray:
        """JAX-compatible Lorentzian profile calculation."""
        return (amplitude / jnp.pi) * (gamma / ((wavelength - center) ** 2 + gamma**2))

    @jit
    def _faddeeva_humlicek_jax(z):
        """
        Humlicek W4 approximation to Faddeeva function w(z) for JAX.

        DEPRECATED: This implementation has gradient stability issues at high
        electron densities (log_ne > 17.5) due to JAX evaluating all branches
        of jnp.where during backpropagation. Use _faddeeva_weideman_jax instead.

        Uses rational approximation that works for all z without requiring
        complex erfc. Accurate to ~1e-4 relative error.

        Reference: Humlicek, JQSRT 27 (1982) 437
        """
        x = jnp.real(z)
        y = jnp.imag(z)

        # Work in upper half plane (y >= 0 for Voigt with gamma > 0)
        y = jnp.abs(y)

        # Humlicek W4 algorithm regions
        s = jnp.abs(x) + y
        t = y - 1j * x

        # Region 1: s >= 15 (asymptotic)
        def region1(x, y, t):
            w = t * 0.5641896 / (0.5 + t * t)
            return w

        # Region 2: 5.5 <= s < 15
        def region2(x, y, t):
            u = t * t
            w = t * (1.410474 + u * 0.5641896) / (0.75 + u * (3.0 + u))
            return w

        # Region 3: s < 5.5 and y >= 0.195 * |x| - 0.176
        def region3(x, y, t):
            w = (16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + t * 0.5642236)))) / (
                16.4955 + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t))))
            )
            return w

        # Region 4: s < 5.5 and y < 0.195 * |x| - 0.176
        def region4(x, y, t):
            u = t * t
            w = jnp.exp(u) - t * (
                36183.31
                - u
                * (
                    3321.9905
                    - u
                    * (1540.787 - u * (219.0313 - u * (35.76683 - u * (1.320522 - u * 0.56419))))
                )
            ) / (
                32066.6
                - u
                * (
                    24322.84
                    - u
                    * (
                        9022.228
                        - u * (2186.181 - u * (364.2191 - u * (61.57037 - u * (1.841439 - u))))
                    )
                )
            )
            return w

        # Select region based on conditions
        w = jnp.where(
            s >= 15.0,
            region1(x, y, t),
            jnp.where(
                s >= 5.5,
                region2(x, y, t),
                jnp.where(
                    y >= 0.195 * jnp.abs(x) - 0.176,
                    region3(x, y, t),
                    region4(x, y, t),
                ),
            ),
        )

        return w

    @jit
    def voigt_profile_jax(
        wavelength: jnp.ndarray, center: float, sigma: float, gamma: float, amplitude: float = 1.0
    ) -> jnp.ndarray:
        """
        JAX-compatible Voigt profile calculation using Weideman rational approximation.

        This implementation uses the branch-free Weideman (1994) algorithm which
        provides stable gradients across all parameter ranges, making it suitable
        for gradient-based optimization and MCMC sampling.

        Parameters
        ----------
        wavelength : array
            Wavelength grid in nm
        center : float
            Line center wavelength in nm
        sigma : float
            Gaussian standard deviation (Doppler broadening) in nm
        gamma : float
            Lorentzian HWHM (Stark/pressure broadening) in nm
        amplitude : float
            Integrated area under the profile

        Returns
        -------
        array
            Voigt profile values
        """
        # Ensure positive widths to avoid division by zero
        sigma = jnp.maximum(sigma, 1e-12)
        gamma = jnp.maximum(gamma, 1e-12)

        x = wavelength - center
        z = (x + 1j * gamma) / (sigma * jnp.sqrt(2.0))

        # Use gradient-stable Weideman approximation
        w_z = _faddeeva_weideman_jax(z)

        profile = jnp.real(w_z) / (sigma * jnp.sqrt(2.0 * jnp.pi))
        return amplitude * profile

    @jit
    def doppler_sigma_jax(wavelength_nm: float, T_eV: float, mass_amu: float) -> float:
        """
        JAX-compatible Doppler width calculation (sigma, not FWHM).

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nm
        T_eV : float
            Temperature in eV
        mass_amu : float
            Atomic mass in amu

        Returns
        -------
        float
            Doppler sigma (standard deviation) in nm
        """
        mass_kg = mass_amu * M_PROTON

        # sigma = lambda/c * sqrt(2 * kT / m)
        # T_eV * EV_TO_J gives energy in Joules
        sigma = wavelength_nm * jnp.sqrt(2.0 * T_eV * EV_TO_J / (mass_kg * C_LIGHT**2))
        return sigma

    def _vmap_profile(
        profile_fn: Callable[..., jnp.ndarray],
        wavelength_grid: jnp.ndarray,
        line_wavelengths: jnp.ndarray,
        line_intensities: jnp.ndarray,
        sigmas: jnp.ndarray,
        gammas: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(
            lambda wl, inten, sig, gam: profile_fn(wavelength_grid, wl, sig, gam, inten),
            in_axes=(0, 0, 0, 0),
        )(line_wavelengths, line_intensities, sigmas, gammas)

    @jit
    def apply_gaussian_broadening_jax(
        wavelength_grid: jnp.ndarray,
        line_wavelengths: jnp.ndarray,
        line_intensities: jnp.ndarray,
        sigma_nm: float,
    ) -> jnp.ndarray:
        """JAX-compatible Gaussian broadening."""
        sigmas = jnp.full_like(line_intensities, sigma_nm)
        profiles = jax.vmap(
            lambda wl, inten, sig: gaussian_profile_jax(wavelength_grid, wl, sig, inten),
            in_axes=(0, 0, 0),
        )(line_wavelengths, line_intensities, sigmas)
        return jnp.sum(profiles, axis=0)

    @jit
    def apply_voigt_broadening_jax(
        wavelength_grid: jnp.ndarray,
        line_wavelengths: jnp.ndarray,
        line_intensities: jnp.ndarray,
        sigmas: Union[float, jnp.ndarray],
        gammas: Union[float, jnp.ndarray],
    ) -> jnp.ndarray:
        """JAX-compatible Voigt broadening."""
        sigmas_arr = jnp.asarray(sigmas)
        gammas_arr = jnp.asarray(gammas)

        if sigmas_arr.ndim == 0:
            sigmas_arr = jnp.full_like(line_intensities, sigmas_arr)
        if gammas_arr.ndim == 0:
            gammas_arr = jnp.full_like(line_intensities, gammas_arr)

        profiles = _vmap_profile(
            voigt_profile_jax,
            wavelength_grid,
            line_wavelengths,
            line_intensities,
            sigmas_arr,
            gammas_arr,
        )
        return jnp.sum(profiles, axis=0)

else:

    def voigt_profile_jax(*args, **kwargs):
        _raise_jax_missing()

    def gaussian_profile_jax(*args, **kwargs):
        _raise_jax_missing()

    def lorentzian_profile_jax(*args, **kwargs):
        _raise_jax_missing()

    def doppler_sigma_jax(*args, **kwargs):
        _raise_jax_missing()

    def apply_gaussian_broadening_jax(*args, **kwargs):
        _raise_jax_missing()

    def apply_voigt_broadening_jax(*args, **kwargs):
        _raise_jax_missing()
