"""
Saha-Boltzmann solver for LTE plasma.
"""

import threading
from typing import Dict, Tuple
import numpy as np

from cflibs.core.constants import SAHA_CONST_CM3, EV_TO_K
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

        # Saha equation: n_{z+1} * n_e / n_z = const * T^1.5 * (U_{z+1}/U_z) * exp(-IP/kT)
        # For neutral to singly ionized:
        # n_II * n_e / n_I = SAHA_CONST * T^1.5 * (U_II/U_I) * exp(-IP_I/kT)

        U_I = self.calculate_partition_function(element, 1, T_e_eV)
        U_II = self.calculate_partition_function(element, 2, T_e_eV)

        saha_ratio_I = (
            (SAHA_CONST_CM3 / n_e_cm3) * (T_e_eV**1.5) * (U_II / U_I) * np.exp(-ip_I / T_e_eV)
        )

        # n_I + n_II = n_total
        # n_II = saha_ratio_I * n_I
        # n_I + saha_ratio_I * n_I = n_total
        # n_I = n_total / (1 + saha_ratio_I)

        n_I = total_density_cm3 / (1.0 + saha_ratio_I)
        n_II = total_density_cm3 - n_I

        result = {1: n_I}

        # If we have IP for stage II and n_II is significant, calculate stage III
        if ip_II is not None and n_II > total_density_cm3 * 1e-6:
            U_III = self.calculate_partition_function(element, 3, T_e_eV)
            saha_ratio_II = (
                (SAHA_CONST_CM3 / n_e_cm3)
                * (T_e_eV**1.5)
                * (U_III / U_II)
                * np.exp(-ip_II / T_e_eV)
            )
            n_III = n_II * saha_ratio_II / (1.0 + saha_ratio_II)
            n_II = n_II - n_III
            result[2] = n_II
            if n_III > total_density_cm3 * 1e-6:
                result[3] = n_III
        else:
            result[2] = n_II

        return result

    @cached_partition_function
    def calculate_partition_function(
        self, element: str, ionization_stage: int, T_e_eV: float, max_energy_ev: float = 50.0
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
        max_energy_ev : float
            Maximum energy level to include (default: 50 eV)

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

        # Partition function: U = sum(g_i * exp(-E_i / kT))
        U = 0.0
        for level in levels:
            if level.energy_ev <= max_energy_ev:
                U += level.g * np.exp(-level.energy_ev / T_e_eV)

        return U

    def solve_level_population(
        self, element: str, ionization_stage: int, stage_density_cm3: float, T_e_eV: float
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

        Returns
        -------
        Dict[Tuple[str, int, float], float]
            Dictionary mapping (element, stage, energy_ev) to population density
        """
        U = self.calculate_partition_function(element, ionization_stage, T_e_eV)
        levels = self.atomic_db.get_energy_levels(element, ionization_stage)

        populations = {}
        for level in levels:
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
                    populations = self.solve_level_population(element, stage, stage_density, T_e_eV)
                    all_populations.update(populations)

        logger.debug(f"Solved Saha-Boltzmann for {len(plasma.species)} species")
        return all_populations
