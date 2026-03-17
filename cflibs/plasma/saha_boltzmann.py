"""
Saha-Boltzmann solver for LTE plasma.
"""

import threading
from typing import Dict, Optional, Tuple
import numpy as np

from cflibs.core.constants import C_LIGHT, E_CHARGE, EV_TO_K, J_TO_EV, KB, SAHA_CONST_CM3
from cflibs.plasma.state import SingleZoneLTEPlasma
from cflibs.core.abc import SolverStrategy, AtomicDataSource
from cflibs.core.cache import cached_partition_function
from cflibs.core.logging_config import get_logger
from cflibs.plasma.partition import PartitionFunctionEvaluator

logger = get_logger("plasma.saha_boltzmann")
_MISSING_LEVEL_WARNED: set[tuple[str, int]] = set()
_MISSING_LEVEL_WARNED_LOCK = threading.Lock()


class SahaBoltzmannSolver(SolverStrategy):
    """
    Solves Saha-Boltzmann equations for LTE plasma.

    For each element, solves:
    1. Saha equation for ionization balance
    2. Boltzmann distribution for level populations
    """

    def __init__(self, atomic_db: AtomicDataSource):
        """
        Initialize solver.

        Parameters
        ----------
        atomic_db : AtomicDatabase
            Atomic database for ionization potentials and energy levels
        """
        self.atomic_db = atomic_db

    def solve_ionization_balance(
        self, element: str, T_e_eV: float, n_e_cm3: float, total_density_cm3: float
    ) -> Dict[int, float]:
        """
        Solve ionization balance using Saha equation.

        Parameters
        ----------
        element : str
            Element symbol
        T_e_eV : float
            Electron temperature in eV
        n_e_cm3 : float
            Electron density in cm^-3
        total_density_cm3 : float
            Total element density (sum over all ionization stages) in cm^-3

        Returns
        -------
        Dict[int, float]
            Dictionary mapping ionization stage to number density in cm^-3
        """
        # Get ionization potentials
        ip_I = self.atomic_db.get_ionization_potential(element, 1)
        ip_II = self.atomic_db.get_ionization_potential(element, 2)

        if ip_I is None:
            logger.warning(f"No ionization potential for {element} I, assuming neutral only")
            return {1: total_density_cm3}

        # Calculate Ionization Potential Depression (IPD)
        # In high-density LIBS plasmas, IPD lowers the effective ionization energy.
        T_K = T_e_eV * EV_TO_K
        delta_chi = ionization_potential_lowering(n_e_cm3, T_K)

        eff_ip_I = max(ip_I - delta_chi, 0.0)

        # Saha equation: n_{z+1} * n_e / n_z = const * T^1.5 * (U_{z+1}/U_z) * exp(-IP/kT)
        # For neutral to singly ionized:
        # n_II * n_e / n_I = SAHA_CONST * T^1.5 * (U_II/U_I) * exp(-IP_I/kT)

        U_I = self.calculate_partition_function(element, 1, T_e_eV, max_energy_ev=eff_ip_I)

        eff_ip_II = max(ip_II - delta_chi, 0.0) if ip_II is not None else None
        U_II = self.calculate_partition_function(element, 2, T_e_eV, max_energy_ev=eff_ip_II)

        if U_I <= 0.0 or U_II <= 0.0:
            logger.warning(
                "Non-positive partition function for %s (U_I=%g, U_II=%g); "
                "falling back to neutral-only.",
                element,
                U_I,
                U_II,
            )
            return {1: total_density_cm3}

        S1 = (SAHA_CONST_CM3 / n_e_cm3) * (T_e_eV**1.5) * (U_II / U_I) * np.exp(-eff_ip_I / T_e_eV)

        # Solve the coupled system:
        #   n_II / n_I = S1
        #   n_III / n_II = S2  (if available)
        #   n_total = n_I + n_II + n_III
        #
        # => n_I = n_total / (1 + S1 + S1*S2)
        # => n_II = S1 * n_I
        # => n_III = S2 * n_II

        S2 = 0.0
        if ip_II is not None:
            ip_III = self.atomic_db.get_ionization_potential(element, 3)
            eff_ip_III = max(ip_III - delta_chi, 0.0) if ip_III is not None else None
            U_III = self.calculate_partition_function(element, 3, T_e_eV, max_energy_ev=eff_ip_III)
            S2 = (
                (SAHA_CONST_CM3 / n_e_cm3)
                * (T_e_eV**1.5)
                * (U_III / U_II)
                * np.exp(-eff_ip_II / T_e_eV)
            )

        denom = 1.0 + S1 + S1 * S2
        n_I = total_density_cm3 / denom
        n_II = S1 * n_I
        n_III = S2 * n_II

        result = {1: n_I, 2: n_II}
        if n_III > total_density_cm3 * 1e-6:
            result[3] = n_III

        return result

    @cached_partition_function
    def calculate_partition_function(
        self,
        element: str,
        ionization_stage: int,
        T_e_eV: float,
        max_energy_ev: float | None = None,
    ) -> float:
        """
        Calculate partition function for a species.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage
        T_e_eV : float
            Electron temperature in eV
        max_energy_ev : float or None
            Maximum energy level to include.  When ``None`` (default), the
            ionization potential of the species is used as the physical cap
            (scaled to 98 % of IP to exclude autoionizing levels).

        Returns
        -------
        float
            Partition function
        """
        # Try to get polynomial coefficients first (Phase 2a)
        if hasattr(self.atomic_db, "get_partition_coefficients"):
            pf = self.atomic_db.get_partition_coefficients(element, ionization_stage)
            if pf is not None:
                T_K = T_e_eV * EV_TO_K
                return PartitionFunctionEvaluator.evaluate(T_K, pf.coefficients)

        # Fallback to summation over levels (Phase 1)
        levels = self.atomic_db.get_energy_levels(element, ionization_stage)

        if not levels:
            # Fallback: assume simple approximation
            # For many elements, U ~ 2 * g_ground at low T
            key = (element, ionization_stage)
            with _MISSING_LEVEL_WARNED_LOCK:
                should_warn = key not in _MISSING_LEVEL_WARNED
                if should_warn:
                    _MISSING_LEVEL_WARNED.add(key)
            if should_warn:
                logger.warning(
                    "No energy levels for %s %s, using approximation", element, ionization_stage
                )
            return 2.0  # Rough approximation

        # Determine the energy cutoff: use ionization potential when available
        # to exclude autoionizing states above the continuum.
        if max_energy_ev is None:
            ip = self.atomic_db.get_ionization_potential(element, ionization_stage)
            max_energy_ev = ip * 0.98 if ip else 50.0

        # Partition function: U = sum(g_i * exp(-E_i / kT))
        U = 0.0
        for level in levels:
            if level.energy_ev <= max_energy_ev:
                U += level.g * np.exp(-level.energy_ev / T_e_eV)

        return U

    def get_ionization_fractions(
        self, element: str, T_e_eV: float, n_e_cm3: float
    ) -> Dict[int, float]:
        """
        Compute ionization stage fractions for an element.

        Returns fractional populations (summing to 1.0) for each ionization
        stage, useful for comparison with NIST LIBS simulation values.

        Parameters
        ----------
        element : str
            Element symbol (e.g. "Fe")
        T_e_eV : float
            Electron temperature in eV
        n_e_cm3 : float
            Electron density in cm^-3

        Returns
        -------
        Dict[int, float]
            Mapping from ionization stage (1=neutral, 2=singly ionized, ...)
            to fractional population (0-1).
        Raises
        ------
        ValueError
            If T_e_eV <= 0, n_e_cm3 <= 0, or solver returns non-positive total.
        """
        if T_e_eV <= 0.0:
            raise ValueError(f"T_e_eV must be positive; got {T_e_eV!r}")
        if n_e_cm3 <= 0.0:
            raise ValueError(f"n_e_cm3 must be positive; got {n_e_cm3!r}")

        # Use arbitrary total density; fractions are independent of it
        total_density = 1.0
        stage_densities = self.solve_ionization_balance(element, T_e_eV, n_e_cm3, total_density)
        total = sum(stage_densities.values())
        if total <= 0.0:
            raise ValueError(
                f"Ionization balance returned non-positive total density "
                f"({total!r}) for element {element!r}"
            )
        return {stage: n / total for stage, n in stage_densities.items()}

    def solve_level_population(
        self,
        element: str,
        ionization_stage: int,
        stage_density_cm3: float,
        T_e_eV: float,
        n_e_cm3: Optional[float] = None,
    ) -> Dict[Tuple[str, int, float], float]:
        """
        Solve Boltzmann distribution for level populations.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage
        stage_density_cm3 : float
            Total density of this ionization stage in cm^-3
        T_e_eV : float
            Electron temperature in eV
        n_e_cm3 : float, optional
            Electron density in cm^-3 (used to compute IPD cutoff)

        Returns
        -------
        Dict[Tuple[str, int, float], float]
            Dictionary mapping (element, stage, energy_ev) to population density
        """
        # Determine max energy based on IPD if electron density is provided
        ip = self.atomic_db.get_ionization_potential(element, ionization_stage)
        if n_e_cm3 is not None and ip is not None:
            T_K = T_e_eV * EV_TO_K
            delta_chi = ionization_potential_lowering(n_e_cm3, T_K)
            max_energy_ev = max(ip - delta_chi, 0.0)
        else:
            max_energy_ev = ip * 0.98 if ip else 50.0

        U = self.calculate_partition_function(
            element, ionization_stage, T_e_eV, max_energy_ev=max_energy_ev
        )

        if U <= 0.0:
            logger.warning(
                "Non-positive partition function for %s stage %d at T_e_eV=%g; "
                "returning empty level populations.",
                element,
                ionization_stage,
                T_e_eV,
            )
            return {}

        levels = self.atomic_db.get_energy_levels(element, ionization_stage)

        populations = {}
        for level in levels:
            # Exclude levels above the lowered IP as they merged into the continuum
            if level.energy_ev > max_energy_ev:
                continue
            # Boltzmann: n_i = n_total * (g_i / U) * exp(-E_i / kT)
            n_i = stage_density_cm3 * (level.g / U) * np.exp(-level.energy_ev / T_e_eV)
            key = (element, ionization_stage, level.energy_ev)
            populations[key] = n_i

        return populations

    def solve_plasma(self, plasma: SingleZoneLTEPlasma) -> Dict[Tuple[str, int, float], float]:
        """
        Solve complete Saha-Boltzmann system for a plasma.

        Parameters
        ----------
        plasma : SingleZoneLTEPlasma
            Plasma state

        Returns
        -------
        Dict[Tuple[str, int, float], float]
            Dictionary mapping (element, stage, energy_ev) to population density in cm^-3
        """
        T_e_eV = plasma.T_e_eV
        n_e_cm3 = plasma.n_e

        all_populations = {}

        for element, total_density in plasma.species.items():
            # Solve ionization balance
            stage_densities = self.solve_ionization_balance(element, T_e_eV, n_e_cm3, total_density)

            # Solve level populations for each stage
            for stage, stage_density in stage_densities.items():
                if stage_density > 0:
                    populations = self.solve_level_population(
                        element, stage, stage_density, T_e_eV, n_e_cm3
                    )
                    all_populations.update(populations)

        logger.debug(f"Solved Saha-Boltzmann for {len(plasma.species)} species")
        return all_populations


# ---------------------------------------------------------------------------
# Ionization Potential Depression (IPD)
# ---------------------------------------------------------------------------

# Derived CGS helpers built from the canonical SI constants in cflibs.core.constants
_E_ESU = E_CHARGE * C_LIGHT * 10.0  # electron charge [esu = statcoulomb]
_KB_ERG = KB * 1.0e7  # Boltzmann constant [erg/K]
_ERG_TO_EV = J_TO_EV * 1.0e-7  # conversion factor erg -> eV


def ionization_potential_lowering(
    n_e_cm3: float,
    T_K: float,
    model: str = "debye_huckel",
) -> float:
    """
    Compute the reduction in effective ionization potential due to plasma screening.

    At high electron densities (n_e ~ 10^17 cm^-3), the Coulomb potential is
    screened by the plasma, effectively lowering the ionization threshold.
    Ignoring this effect leads to a small but systematic bias in the Saha
    equation.

    **Debye-Hückel model** (``model='debye_huckel'``):

    .. math::

        \\Delta\\chi = \\frac{e^2}{4\\pi\\varepsilon_0 \\lambda_D}

    where the Debye length is:

    .. math::

        \\lambda_D = \\sqrt{\\frac{k_B T}{4\\pi n_e e^2}}

    In Gaussian CGS this simplifies to:

    .. math::

        \\Delta\\chi = e^2 \\sqrt{\\frac{4\\pi n_e}{k_B T}}

    Parameters
    ----------
    n_e_cm3 : float
        Electron density [cm^-3]
    T_K : float
        Plasma temperature [K]
    model : str
        IPD model. Currently only ``'debye_huckel'`` is supported.

    Returns
    -------
    float
        Ionization potential depression [eV]. Always >= 0.

    Raises
    ------
    ValueError
        If an unsupported model name is given.

    Examples
    --------
    >>> delta_chi = ionization_potential_lowering(1e17, 10000)
    >>> 0.03 <= delta_chi <= 0.06  # ~0.04 eV at canonical LIBS conditions
    True

    References
    ----------
    - Stewart, J.C. & Pyatt, K.D. (1966) ApJ 144, 1203
    - Kramida, A. et al. NIST Atomic Spectra Database
    """
    if model != "debye_huckel":
        raise ValueError(f"Unsupported IPD model: {model!r}. Use 'debye_huckel'.")

    if n_e_cm3 <= 0.0 or T_K <= 0.0:
        return 0.0

    # Debye length: lambda_D = sqrt(kT / (4*pi*n_e*e^2))  [cm]
    lambda_D = np.sqrt(_KB_ERG * T_K / (4.0 * np.pi * n_e_cm3 * _E_ESU**2))

    # IPD = e^2 / lambda_D  [erg]
    delta_chi_erg = _E_ESU**2 / lambda_D

    # Convert to eV
    return float(delta_chi_erg * _ERG_TO_EV)
