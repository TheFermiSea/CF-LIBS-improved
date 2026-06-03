"""
Stark broadening calculations for spectral lines.
"""

from typing import Optional
import numpy as np
from cflibs.core.logging_config import get_logger
from cflibs.core.constants import EV_TO_K
from cflibs.core.jax_runtime import HAS_JAX, jit_if_available, jnp

jit = jit_if_available


logger = get_logger("radiation.stark")

# Reference conditions for Stark parameters.
#
# SINGLE SOURCE OF TRUTH for the project-wide Stark-width convention.
# The atomic database stores ``lines.stark_w`` as the electron-impact
# **FWHM** at ``n_e = 1e17 cm^-3``, ``T = 10000 K`` — see
# ``scripts/ingest_stark_b.py`` ("T_e = 10000 K, n_e = 1.0e17 cm^-3")
# and the published-value anchors in ``tests/test_stark_provenance.py``.
#
# Historically the runtime treated the column as **HWHM at 1e16**, which
# over-broadened every Stark line by a factor of 20 at ps-LIBS densities:
#   x10 from the wrong reference density (1e16 vs 1e17)
#   x2  from an extra HWHM->FWHM doubling on an already-FWHM value.
# Verified end-to-end (A4-CONV-2): Al I 396.15 stored 4.5 pm produced an
# omega_stark of 90.0 pm at 1e17/10000 K; the convention-correct value is
# exactly the stored 4.5 pm (the data is *defined* as the FWHM at those
# reference conditions). This module now keeps data + every consumer in
# agreement on FWHM@1e17.
REF_NE = 1.0e17  # cm^-3 — reference density the stored stark_w is tabulated at
REF_T_K = 10000.0  # K — reference temperature the stored stark_w is tabulated at
REF_T_EV = 0.86173  # REF_T_K in eV (10000 K / 11604.5); used by the JAX twins

# The stored ``stark_w`` is a FULL width (FWHM). ``stark_hwhm`` therefore
# returns half of it; ``stark_width`` doubles back to recover the FWHM,
# so at the reference conditions ``stark_width == stored stark_w`` exactly.
_STARK_W_IS_FWHM = True


def stark_hwhm(
    n_e_cm3: float,
    T_K: float,
    stark_w_ref: Optional[float],
    stark_alpha: Optional[float] = None,
    ref_T_K: float = REF_T_K,
    stark_A_ref: Optional[float] = None,
) -> float:
    """
    Calculate Stark Half-Width at Half-Maximum (HWHM).
    Includes optional quasi-static ion-broadening correction based on Griem's theory.

    ``stark_w_ref`` is the stored database value: the electron-impact
    **FWHM** at ``n_e = REF_NE = 1e17 cm^-3``, ``T = REF_T_K = 10000 K``.
    This function returns the corresponding **HWHM** scaled to the
    requested (n_e, T)::

        w_fwhm   = stark_w_ref * (n_e / REF_NE) * (T / T_ref)^(-alpha)
        w_hwhm   = 0.5 * w_fwhm
        w_total  = w_hwhm * (1 + 1.75 * A_ion * (1 - 0.75 * R_D))

    At the reference conditions ``stark_width`` (= 2 * stark_hwhm) returns
    exactly the stored ``stark_w``; see the module-level convention note.

    .. note:: Bead ``CF-LIBS-improved-s1qr.3`` (2026-05-25). Periodic
       cross-exams have suggested this function does linear interpolation
       of electron-impact parameters over a coarse T grid, breaking C^∞
       smoothness. **That is empirically false.** This is an analytic
       power law ``(T/T_ref)^(-alpha)`` with NO tabulated grid, NO
       ``np.interp``, NO ``interp1d``. Verified C^∞ smooth above the
       ``max(T, 1000)`` floor (which is unreachable in realistic LIBS
       plasma at 5000-15000 K). Numerical 2nd-derivative across T =
       5000..15000 K varies smoothly, no sawtooth signature. The
       ``stark_hwhm_jax`` twin at ``stark.py:215+`` uses the identical
       closed form.

    Parameters
    ----------
    n_e_cm3 : float
        Electron density in cm^-3
    T_K : float
        Temperature in K
    stark_w_ref : float
        Stored Stark width: electron-impact FWHM at REF_NE = 1e17 cm^-3,
        T = REF_T_K = 10000 K (nm).
    stark_alpha : float
        Scaling exponent (default 0.5 if None, represents T^(-1/2))
    ref_T_K : float
        Reference temperature (default REF_T_K = 10000 K)
    stark_A_ref : float, optional
        Ion-broadening parameter at REF_NE = 1e17 cm^-3

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

    # Electron-impact width. ``stark_w_ref`` is a FWHM at REF_NE; halve it
    # to obtain the HWHM (stark_width doubles it back to FWHM).
    w_fwhm = stark_w_ref * (n_e_cm3 / REF_NE) * (T_eff / ref_T_K) ** (-alpha)
    w_e = 0.5 * w_fwhm

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
    ref_T_K: float = REF_T_K,
    stark_A_ref: Optional[float] = None,
) -> float:
    """Calculate Stark Full-Width at Half-Maximum (FWHM).

    ``stark_w_ref`` is the stored FWHM at REF_NE = 1e17 cm^-3; at the
    reference conditions this returns exactly that value.
    """
    hwhm = stark_hwhm(n_e_cm3, T_K, stark_w_ref, stark_alpha, ref_T_K, stark_A_ref)
    return 2.0 * hwhm


def stark_shift(n_e_cm3: float, stark_d_ref: Optional[float]) -> float:
    """
    Calculate Stark shift.

    ``stark_d_ref`` is the stored signed shift at REF_NE = 1e17 cm^-3.

    d = d_ref * (n_e / REF_NE)
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
    Estimate Stark broadening parameter w_ref (FWHM at REF_NE = 1e17 cm^-3),
    matching the stored ``lines.stark_w`` convention. Based on semi-empirical
    trends, used only when literature/database data is missing.

    Returns
    -------
    float
        Estimated Stark w_ref (FWHM at REF_NE) in nm
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
            Stored Stark width: FWHM at REF_NE = 1e17 cm^-3
        stark_alpha : float
            Scaling exponent
        stark_A_ref : float
            Ion-broadening parameter at REF_NE = 1e17 cm^-3

        Returns
        -------
        float
            Stark HWHM in nm (half of the FWHM scaled to live n_e, T)
        """
        # Safe defaults
        w_ref = jnp.nan_to_num(stark_w_ref, nan=0.0)
        alpha = jnp.nan_to_num(stark_alpha, nan=0.5)
        A_ref = jnp.nan_to_num(stark_A_ref, nan=0.0)

        factor_ne = n_e_cm3 / REF_NE
        factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -alpha)

        # ``w_ref`` is a FWHM at REF_NE; halve it to return a HWHM, mirroring
        # the float ``stark_hwhm`` above.
        w_e = 0.5 * w_ref * factor_ne * factor_T

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
            Estimated Stark w_ref (FWHM at REF_NE = 1e17 cm^-3) in nm
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
