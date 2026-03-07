"""
Plasma state representations and composition conversion helpers.
"""

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger

logger = get_logger("plasma.state")


def _normalize_positive_fractions(
    fractions: Mapping[str, float],
    quantity_name: str,
) -> Dict[str, float]:
    """Normalize positive fractions to sum to one."""
    non_finite = [el for el, v in fractions.items() if not math.isfinite(float(v))]
    if non_finite:
        raise ValueError(
            f"{quantity_name} contains non-finite values: {', '.join(sorted(non_finite))}"
        )
    negative_elements = [element for element, value in fractions.items() if float(value) < 0.0]
    if negative_elements:
        elements_str = ", ".join(sorted(negative_elements))
        raise ValueError(f"{quantity_name} cannot contain negative components: {elements_str}")

    normalized_input = {
        element: float(value) for element, value in fractions.items() if value > 0.0
    }
    total = sum(normalized_input.values())
    if total <= 0.0:
        raise ValueError(f"{quantity_name} must contain at least one positive component")
    return {element: value / total for element, value in normalized_input.items()}


def species_densities_to_number_fractions(species: Mapping[str, float]) -> Dict[str, float]:
    """
    Convert element number densities to normalized number fractions.

    Parameters
    ----------
    species : Mapping[str, float]
        Element number densities in cm^-3.

    Returns
    -------
    Dict[str, float]
        Element number fractions summing to 1.0.
    """
    return _normalize_positive_fractions(species, "species")


def number_fractions_to_species_densities(
    number_fractions: Mapping[str, float],
    total_number_density_cm3: float,
) -> Dict[str, float]:
    """
    Convert number fractions to element number densities.

    Parameters
    ----------
    number_fractions : Mapping[str, float]
        Element number fractions.
    total_number_density_cm3 : float
        Total heavy-particle number density in cm^-3.

    Returns
    -------
    Dict[str, float]
        Element number densities in cm^-3.
    """
    if not math.isfinite(total_number_density_cm3) or total_number_density_cm3 <= 0.0:
        raise ValueError("total_number_density_cm3 must be finite and positive")

    normalized = _normalize_positive_fractions(number_fractions, "number_fractions")
    return {
        element: total_number_density_cm3 * fraction for element, fraction in normalized.items()
    }


def mass_fractions_to_number_fractions(
    mass_fractions: Mapping[str, float],
    atomic_masses_amu: Mapping[str, float],
) -> Dict[str, float]:
    """
    Convert mass fractions to number fractions.

    Parameters
    ----------
    mass_fractions : Mapping[str, float]
        Element mass fractions.
    atomic_masses_amu : Mapping[str, float]
        Atomic masses in amu keyed by element symbol.

    Returns
    -------
    Dict[str, float]
        Element number fractions summing to 1.0.
    """
    weighted: Dict[str, float] = {}
    for element, mass_fraction in mass_fractions.items():
        if mass_fraction < 0.0:
            raise ValueError(f"mass_fractions cannot contain negative components: {element}")
        if mass_fraction == 0.0:
            continue
        if element not in atomic_masses_amu:
            raise KeyError(f"Missing atomic mass for element {element!r}")
        atomic_mass = float(atomic_masses_amu[element])
        if atomic_mass <= 0.0:
            raise ValueError(f"Atomic mass for element {element!r} must be positive")
        weighted[element] = float(mass_fraction) / atomic_mass
    return _normalize_positive_fractions(weighted, "mass_fractions")


def mass_fractions_to_species_densities(
    mass_fractions: Mapping[str, float],
    total_number_density_cm3: float,
    atomic_masses_amu: Mapping[str, float],
) -> Dict[str, float]:
    """
    Convert mass fractions directly to element number densities.

    Parameters
    ----------
    mass_fractions : Mapping[str, float]
        Element mass fractions.
    total_number_density_cm3 : float
        Total heavy-particle number density in cm^-3.
    atomic_masses_amu : Mapping[str, float]
        Atomic masses in amu keyed by element symbol.

    Returns
    -------
    Dict[str, float]
        Element number densities in cm^-3.
    """
    number_fractions = mass_fractions_to_number_fractions(mass_fractions, atomic_masses_amu)
    return number_fractions_to_species_densities(number_fractions, total_number_density_cm3)


@dataclass
class PlasmaState:
    """
    Base plasma state representation.

    Attributes
    ----------
    T_e : float
        Electron temperature in K
    T_g : float
        Gas/ion temperature in K (optional, defaults to T_e)
    n_e : float
        Electron density in cm^-3
    species : Dict[str, float]
        Element number densities in cm^-3 (key: element symbol). These are
        heavy-particle densities and are distinct from mass fractions, number
        fractions, and electron density.
    pressure : Optional[float]
        Pressure in atm (optional)
    """

    T_e: float  # K
    n_e: float  # cm^-3
    species: Dict[str, float]  # cm^-3
    T_g: Optional[float] = None
    pressure: Optional[float] = None

    @property
    def T_e_eV(self) -> float:
        """Electron temperature in eV."""
        return self.T_e * KB_EV

    @property
    def T_g_eV(self) -> float:
        """Gas temperature in eV."""
        if self.T_g is None:
            return self.T_e_eV
        return self.T_g * KB_EV

    @property
    def total_species_density_cm3(self) -> float:
        """Total heavy-particle number density in cm^-3."""
        return float(sum(self.species.values()))

    @property
    def species_number_fractions(self) -> Dict[str, float]:
        """Element number fractions implied by ``species``."""
        return species_densities_to_number_fractions(self.species)


class SingleZoneLTEPlasma(PlasmaState):
    """
    Single-zone LTE plasma model.

    This is the simplest plasma model: a homogeneous, optically thin
    plasma in local thermodynamic equilibrium.
    """

    def __init__(
        self,
        T_e: float,
        n_e: float,
        species: Dict[str, float],
        T_g: Optional[float] = None,
        pressure: Optional[float] = None,
    ):
        """
        Initialize single-zone LTE plasma.

        Parameters
        ----------
        T_e : float
            Electron temperature in K
        n_e : float
            Electron density in cm^-3
        species : Dict[str, float]
            Element number densities in cm^-3
        T_g : float, optional
            Gas temperature in K (defaults to T_e)
        pressure : float, optional
            Pressure in atm
        """
        super().__init__(T_e, n_e, species, T_g, pressure)
        logger.info(
            f"Created SingleZoneLTEPlasma: T_e={T_e:.1f} K, n_e={n_e:.2e} cm^-3, "
            f"species={list(species.keys())}"
        )

    @classmethod
    def from_number_fractions(
        cls,
        T_e: float,
        n_e: float,
        number_fractions: Mapping[str, float],
        total_species_density_cm3: float,
        T_g: Optional[float] = None,
        pressure: Optional[float] = None,
    ) -> "SingleZoneLTEPlasma":
        """
        Build a plasma state from element number fractions.

        Parameters
        ----------
        T_e : float
            Electron temperature in K.
        n_e : float
            Electron density in cm^-3.
        number_fractions : Mapping[str, float]
            Element number fractions on a heavy-particle basis.
        total_species_density_cm3 : float
            Total heavy-particle number density in cm^-3.
        T_g : float, optional
            Gas temperature in K. Defaults to ``T_e`` when omitted.
        pressure : float, optional
            Pressure in atm.

        Returns
        -------
        SingleZoneLTEPlasma
            Plasma state with element number densities derived from the
            supplied number fractions.
        """
        species = number_fractions_to_species_densities(
            number_fractions,
            total_number_density_cm3=total_species_density_cm3,
        )
        return cls(T_e=T_e, n_e=n_e, species=species, T_g=T_g, pressure=pressure)

    @classmethod
    def from_mass_fractions(
        cls,
        T_e: float,
        n_e: float,
        mass_fractions: Mapping[str, float],
        total_species_density_cm3: float,
        atomic_masses_amu: Mapping[str, float],
        T_g: Optional[float] = None,
        pressure: Optional[float] = None,
    ) -> "SingleZoneLTEPlasma":
        """
        Build a plasma state from element mass fractions.

        Parameters
        ----------
        T_e : float
            Electron temperature in K.
        n_e : float
            Electron density in cm^-3.
        mass_fractions : Mapping[str, float]
            Element mass fractions.
        total_species_density_cm3 : float
            Total heavy-particle number density in cm^-3.
        atomic_masses_amu : Mapping[str, float]
            Atomic masses in amu keyed by element symbol.
        T_g : float, optional
            Gas temperature in K. Defaults to ``T_e`` when omitted.
        pressure : float, optional
            Pressure in atm.

        Returns
        -------
        SingleZoneLTEPlasma
            Plasma state with element number densities derived from the
            supplied mass fractions.
        """
        species = mass_fractions_to_species_densities(
            mass_fractions,
            total_number_density_cm3=total_species_density_cm3,
            atomic_masses_amu=atomic_masses_amu,
        )
        return cls(T_e=T_e, n_e=n_e, species=species, T_g=T_g, pressure=pressure)

    def validate(self) -> bool:
        """
        Validate plasma state.

        Returns
        -------
        bool
            True if valid

        Raises
        ------
        ValueError
            If plasma state is invalid
        """
        if self.T_e <= 0:
            raise ValueError("Electron temperature must be positive")

        if self.n_e <= 0:
            raise ValueError("Electron density must be positive")

        if not self.species:
            raise ValueError("At least one species must be specified")

        for element, density in self.species.items():
            if density <= 0:
                raise ValueError(f"Species density for {element} must be positive")

        total_species_density = self.total_species_density_cm3
        if total_species_density <= 0.0:
            raise ValueError("Total species density must be positive")

        # This is only a rough plausibility check. ``species`` stores heavy-particle
        # densities rather than charge density, so we only warn on extreme mismatch.
        if abs(self.n_e - total_species_density) / max(self.n_e, total_species_density) > 0.5:
            logger.warning(
                f"Charge neutrality check: n_e={self.n_e:.2e}, "
                f"total_species_density={total_species_density:.2e}"
            )

        return True
