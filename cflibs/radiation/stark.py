"""
Stark broadening calculations for spectral lines.
"""

from typing import Optional
import numpy as np
from cflibs.core.logging_config import get_logger
from cflibs.core.constants import EV_TO_K

try:
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None  # type: ignore[assignment]

    def jit(f):
        return f


logger = get_logger("radiation.stark")

# Reference conditions for Stark parameters
REF_NE = 1.0e16  # cm^-3
REF_T_K = 10000.0  # K


def stark_hwhm(
    n_e_cm3: float,
    T_K: float,
    stark_w_ref: Optional[float],
    stark_alpha: Optional[float] = None,
    ref_T_K: float = 10000.0,
    stark_A_ref: Optional[float] = None,
) -> float:
    """
    Calculate Stark Half-Width at Half-Maximum (HWHM).
    Includes optional quasi-static ion-broadening correction based on Griem's theory.

    w_electron = w_ref * (n_e / 10^16) * (T / T_ref)^(-alpha)
    w_total = w_electron * (1 + 1.75 * A_ion * (1 - 0.75 * R_D))

    Parameters
    ----------
    n_e_cm3 : float
        Electron density in cm^-3
    T_K : float
        Temperature in K
    stark_w_ref : float
        Stark width parameter (HWHM) at 10^16 cm^-3
    stark_alpha : float
        Scaling exponent (default 0.5 if None, represents T^(-1/2))
    ref_T_K : float
        Reference temperature
    stark_A_ref : float, optional
        Ion-broadening parameter at 10^16 cm^-3

    Returns
    -------
    float
        Stark HWHM in nm
    """
    if stark_w_ref is None or stark_w_ref <= 0:
        return 0.0

    alpha = stark_alpha if stark_alpha is not None else 0.5

    # Use max(T, 1000) to avoid division by zero or extreme scaling
    T_eff = max(T_K, 1000.0)

    # Electron impact width
    w_e = stark_w_ref * (n_e_cm3 / REF_NE) * (T_eff / ref_T_K) ** (-alpha)

    if stark_A_ref is None or stark_A_ref <= 0:
        return w_e

    # Ion broadening correction
    A_ion = stark_A_ref * (n_e_cm3 / REF_NE) ** 0.25
    # For a typical LIBS plasma, the Debye shielding parameter R_D is ~0.5.
    # A full calculation requires Debye length, but 0.5 is a common approximation.
    R_D = 0.5
    correction = 1.0 + 1.75 * A_ion * (1.0 - 0.75 * R_D)

    return w_e * correction


def stark_width(
    n_e_cm3: float,
    T_K: float,
    stark_w_ref: Optional[float],
    stark_alpha: Optional[float] = None,
    ref_T_K: float = 10000.0,
    stark_A_ref: Optional[float] = None,
) -> float:
    """Calculate Stark Full-Width at Half-Maximum (FWHM)."""
    hwhm = stark_hwhm(n_e_cm3, T_K, stark_w_ref, stark_alpha, ref_T_K, stark_A_ref)
    return 2.0 * hwhm


def stark_shift(n_e_cm3: float, stark_d_ref: Optional[float]) -> float:
    """
    Calculate Stark shift.

    d = d_ref * (n_e / 10^16)
    """
    if stark_d_ref is None:
        return 0.0
    return stark_d_ref * (n_e_cm3 / REF_NE)


def estimate_stark_parameter(
    wavelength_nm: float,
    upper_energy_ev: float,
    ionization_potential_ev: Optional[float],
    ionization_stage: int,
) -> float:
    """
    Estimate Stark broadening parameter w_ref (HWHM at 1e16 cm^-3).
    Based on semi-empirical trends if data is missing.

    Returns
    -------
    float
        Estimated Stark w_ref in nm
    """
    # Fallback if critical data missing
    if ionization_potential_ev is None or upper_energy_ev >= ionization_potential_ev:
        # Default ~0.005 nm at 1e16 for typical lines
        return 0.005

    # Effective principal quantum number n*
    # n* = Z_eff * sqrt(Ry / (IP - E_upper))
    # Ry = 13.605 eV
    binding_energy = ionization_potential_ev - upper_energy_ev
    if binding_energy <= 0.1:
        binding_energy = 0.1

    n_eff = ionization_stage * np.sqrt(13.605 / binding_energy)

    # Rough scaling approx
    # w_ref (nm) ~ C * lambda^2 * n_eff^4
    # Tuned to give reasonable order of magnitude
    w_est = 2.0e-5 * (wavelength_nm / 500.0) ** 2 * (n_eff**4)

    return max(0.0001, min(w_est, 0.5))


class StarkBroadeningCalculator:
    """
    Calculator for Stark parameters using database lookup or estimation.
    """

    def __init__(self, atomic_db):
        """
        Initialize with atomic database.

        Parameters
        ----------
        atomic_db : AtomicDatabase
            Database instance
        """
        self.db = atomic_db

    def get_stark_width(
        self,
        element: str,
        ionization_stage: int,
        wavelength_nm: float,
        n_e_cm3: float,
        T_e_eV: float,
        upper_energy_ev: Optional[float] = None,
    ) -> float:
        """
        Get Stark FWHM for a line.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage
        wavelength_nm : float
            Line wavelength
        n_e_cm3 : float
            Electron density
        T_e_eV : float
            Temperature in eV
        upper_energy_ev : float, optional
            Upper energy level (for estimation fallback)

        Returns
        -------
        float
            Stark FWHM in nm
        """
        # 1. Lookup
        params = self.db.get_stark_parameters(element, ionization_stage, wavelength_nm)
        w_ref, alpha, _ = params

        T_K = T_e_eV * EV_TO_K

        # 2. Estimate if missing
        if w_ref is None and upper_energy_ev is not None:
            ip = self.db.get_ionization_potential(element, ionization_stage)
            w_ref = estimate_stark_parameter(wavelength_nm, upper_energy_ev, ip, ionization_stage)
            alpha = 0.5

        # 3. Calculate
        return stark_width(n_e_cm3, T_K, w_ref, alpha)

    def get_stark_shift(
        self, element: str, ionization_stage: int, wavelength_nm: float, n_e_cm3: float
    ) -> float:
        """Get Stark shift for a line."""
        params = self.db.get_stark_parameters(element, ionization_stage, wavelength_nm)
        _, _, d_ref = params
        return stark_shift(n_e_cm3, d_ref)


if HAS_JAX:

    @jit
    def stark_hwhm_jax(
        n_e_cm3: float,
        T_eV: float,
        stark_w_ref: float,
        stark_alpha: float,
        stark_A_ref: float = 0.0,
    ) -> float:
        """
        JAX-compatible Stark HWHM calculation with ion broadening correction.

        Parameters
        ----------
        n_e_cm3 : float
            Electron density
        T_eV : float
            Temperature in eV
        stark_w_ref : float
            Ref HWHM at 1e16 cm^-3
        stark_alpha : float
            Scaling exponent
        stark_A_ref : float
            Ion-broadening parameter at 10^16 cm^-3

        Returns
        -------
        float
            Stark HWHM in nm
        """
        # REF_T in eV = 10000 / 11604.5 ~ 0.8617 eV
        REF_T_EV = 0.86173

        # Safe defaults
        w_ref = jnp.nan_to_num(stark_w_ref, nan=0.0)
        alpha = jnp.nan_to_num(stark_alpha, nan=0.5)
        A_ref = jnp.nan_to_num(stark_A_ref, nan=0.0)

        factor_ne = n_e_cm3 / 1.0e16
        factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -alpha)

        w_e = w_ref * factor_ne * factor_T

        A_ion = A_ref * jnp.power(factor_ne, 0.25)
        R_D = 0.5
        correction = 1.0 + 1.75 * A_ion * (1.0 - 0.75 * R_D)

        return w_e * correction

    @jit
    def estimate_stark_parameter_jax(
        wavelength_nm: float,
        upper_energy_ev: float,
        ionization_potential_ev: float,
        ionization_stage: int,
    ) -> float:
        """
        JAX-compatible Stark parameter estimation.

        Used as fallback when database lacks Stark data.

        Parameters
        ----------
        wavelength_nm : float
            Line wavelength in nm
        upper_energy_ev : float
            Upper level energy in eV
        ionization_potential_ev : float
            Ionization potential in eV
        ionization_stage : int
            Ionization stage (1=neutral, 2=singly ionized)

        Returns
        -------
        float
            Estimated Stark w_ref (HWHM at 1e16 cm^-3) in nm
        """
        # Binding energy with safety floor
        binding_energy = jnp.maximum(ionization_potential_ev - upper_energy_ev, 0.1)

        # Effective principal quantum number
        n_eff = ionization_stage * jnp.sqrt(13.605 / binding_energy)

        # Semi-empirical scaling
        # w_ref ~ C * lambda^2 * n_eff^4
        w_est = 2.0e-5 * (wavelength_nm / 500.0) ** 2 * (n_eff**4)

        # Clamp to reasonable range
        return jnp.clip(w_est, 0.0001, 0.5)

else:

    def stark_hwhm_jax(*args, **kwargs):
        raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")

    def estimate_stark_parameter_jax(*args, **kwargs):
        raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")
