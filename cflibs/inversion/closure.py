"""
Closure equation implementation for CF-LIBS.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.closure")


def _validated_abundance_multiplier(
    abundance_multipliers: Optional[Dict[str, float]],
    element: str,
) -> float:
    """Return a finite positive abundance multiplier for an element."""
    multiplier = abundance_multipliers.get(element, 1.0) if abundance_multipliers else 1.0
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"abundance_multipliers[{element!r}] must be finite and positive")
    return float(multiplier)


@dataclass
class ClosureResult:
    """
    Result of applying the closure equation.
    """

    concentrations: Dict[str, float]  # element -> mass/molar fraction (sum=1)
    experimental_factor: float  # The eliminated factor F (scaling factor)
    total_measured: float  # Sum of relative concentrations before normalization
    mode: str  # Mode used ('standard', 'matrix', 'oxide')


class ClosureEquation:
    """
    Handles the closure equation (normalization) step of CF-LIBS.

    The fundamental equation is: C_s = (U_s(T) * exp(q_s)) / F
    where q_s is the intercept from the Boltzmann plot.

    The closure condition determines F.
    """

    @staticmethod
    def apply_standard(
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        abundance_multipliers: Optional[Dict[str, float]] = None,
    ) -> ClosureResult:
        """
        Apply standard closure: sum(C_s) = 1.

        F = sum(U_s * exp(q_s))

        Parameters
        ----------
        intercepts : Dict[str, float]
            Boltzmann plot intercepts q_s for each element
        partition_funcs : Dict[str, float]
            Partition function values U_s(T) for each element
        abundance_multipliers : Dict[str, float], optional
            Optional per-element scaling that maps the fitted intercept from
            the neutral Saha-Boltzmann plane back to total elemental abundance.
            Defaults to unity for all elements.

        Returns
        -------
        ClosureResult
        """
        # Calculate relative concentrations (unscaled)
        # rel_C_s = U_s * exp(q_s)
        rel_concentrations = {}
        total_measured = 0.0

        for element, q_s in intercepts.items():
            if element not in partition_funcs:
                logger.warning(f"Missing partition function for {element} in closure")
                continue

            U_s = partition_funcs[element]
            multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
            rel_C = multiplier * U_s * np.exp(q_s)
            rel_concentrations[element] = rel_C
            total_measured += rel_C

        if total_measured == 0:
            logger.error("Total measured concentration is zero")
            return ClosureResult({}, 0.0, 0.0, "standard")

        # F is effectively total_measured
        F = total_measured

        concentrations = {el: val / F for el, val in rel_concentrations.items()}

        return ClosureResult(
            concentrations=concentrations,
            experimental_factor=F,
            total_measured=total_measured,
            mode="standard",
        )

    @staticmethod
    def apply_matrix_mode(
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        matrix_element: str,
        matrix_fraction: float = 0.9,
        abundance_multipliers: Optional[Dict[str, float]] = None,
    ) -> ClosureResult:
        """
        Apply matrix closure: One element has fixed concentration.

        C_matrix = (U_m * exp(q_m)) / F = matrix_fraction
        => F = (U_m * exp(q_m)) / matrix_fraction

        Parameters
        ----------
        intercepts : Dict[str, float]
            Intercepts
        partition_funcs : Dict[str, float]
            Partition functions
        matrix_element : str
            Element with known concentration
        matrix_fraction : float
            Concentration of matrix element (0.0 to 1.0)
        abundance_multipliers : Dict[str, float], optional
            Optional per-element scaling that maps the fitted intercept from
            the neutral Saha-Boltzmann plane back to total elemental abundance.
            Keys should match the intercept and partition-function mappings.
            Defaults to unity for all elements when omitted.

        Returns
        -------
        ClosureResult
        """
        if matrix_element not in intercepts or matrix_element not in partition_funcs:
            logger.error(f"Matrix element {matrix_element} missing from data")
            return ClosureEquation.apply_standard(
                intercepts,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )

        U_m = partition_funcs[matrix_element]
        q_m = intercepts[matrix_element]
        matrix_multiplier = _validated_abundance_multiplier(abundance_multipliers, matrix_element)
        rel_C_m = matrix_multiplier * U_m * np.exp(q_m)

        F = rel_C_m / matrix_fraction

        concentrations = {}
        total_measured = 0.0

        for element, q_s in intercepts.items():
            if element in partition_funcs:
                U_s = partition_funcs[element]
                multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
                rel_C = multiplier * U_s * np.exp(q_s)
                total_measured += rel_C
                concentrations[element] = rel_C / F

        return ClosureResult(
            concentrations=concentrations,
            experimental_factor=F,
            total_measured=total_measured,
            mode=f"matrix({matrix_element}={matrix_fraction})",
        )

    @staticmethod
    def apply_oxide_mode(
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        oxide_stoichiometry: Dict[str, float],
        abundance_multipliers: Optional[Dict[str, float]] = None,
    ) -> ClosureResult:
        """
        Apply oxide closure: Elements exist as oxides, sum(Oxides) = 1.

        Used for geological samples.
        e.g. Si -> SiO2 (factor ~2.14), Al -> Al2O3

        C_oxide_s = C_s * oxide_factor_s
        sum(C_oxide_s) = 1
        sum( (U_s * exp(q_s) / F) * oxide_factor_s ) = 1
        F = sum( U_s * exp(q_s) * oxide_factor_s )

        Parameters
        ----------
        intercepts : Dict[str, float]
            Boltzmann plot intercepts q_s for each element.
        partition_funcs : Dict[str, float]
            Partition function values U_s(T) for each element.
        oxide_stoichiometry : Dict[str, float]
            Map of element to oxide conversion factor (e.g. ``{"Si": 2.139}``
            for SiO2).
        abundance_multipliers : Dict[str, float], optional
            Optional per-element scaling that maps the fitted intercept from
            the neutral Saha-Boltzmann plane back to total elemental abundance
            before oxide weighting. Defaults to unity for all elements when
            omitted.

        Returns
        -------
        ClosureResult
            Concentrations are ELEMENTAL fractions, but normalized such that oxides sum to 1.
        """
        rel_concentrations = {}
        total_oxide_rel = 0.0

        for element, q_s in intercepts.items():
            if element not in partition_funcs:
                continue

            U_s = partition_funcs[element]
            multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
            rel_C = multiplier * U_s * np.exp(q_s)
            rel_concentrations[element] = rel_C

            factor = oxide_stoichiometry.get(element, 1.0)  # Default to metal if no oxide
            total_oxide_rel += rel_C * factor

        if total_oxide_rel == 0:
            return ClosureResult({}, 0.0, 0.0, "oxide")

        F = total_oxide_rel

        concentrations = {el: val / F for el, val in rel_concentrations.items()}

        return ClosureResult(
            concentrations=concentrations,
            experimental_factor=F,
            total_measured=total_oxide_rel,
            mode="oxide",
        )
