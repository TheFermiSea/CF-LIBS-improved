"""
Abstract base classes and protocols for extensibility.

ABCs are used for core interfaces (AtomicDataSource, SolverStrategy) that must
be inherited. Protocols are used for structural typing of classes that may
implement the interface without explicit inheritance.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Protocol, TYPE_CHECKING, runtime_checkable
import numpy as np

from cflibs.atomic.structures import Transition, EnergyLevel

if TYPE_CHECKING:
    from cflibs.plasma.state import SingleZoneLTEPlasma


class AtomicDataSource(ABC):
    """
    Abstract interface for atomic data sources.

    This allows plugging in different data sources (SQLite, NIST API, HDF5, etc.)
    without changing the rest of the codebase.
    """

    @abstractmethod
    def get_transitions(
        self,
        element: str,
        ionization_stage: Optional[int] = None,
        wavelength_min: Optional[float] = None,
        wavelength_max: Optional[float] = None,
        min_relative_intensity: Optional[float] = None,
    ) -> List[Transition]:
        """Get transitions for an element."""
        pass

    @abstractmethod
    def get_energy_levels(self, element: str, ionization_stage: int) -> List[EnergyLevel]:
        """Get energy levels for a species."""
        pass

    @abstractmethod
    def get_ionization_potential(self, element: str, ionization_stage: int) -> Optional[float]:
        """Get ionization potential for a species in eV."""
        pass

    @abstractmethod
    def get_available_elements(self) -> List[str]:
        """Get list of available element symbols."""
        pass


class SolverStrategy(ABC):
    """
    Abstract interface for plasma solvers.

    This allows implementing different solver algorithms (Saha-Boltzmann,
    non-LTE solvers, multi-zone solvers, etc.) as interchangeable strategies.
    """

    @abstractmethod
    def solve_ionization_balance(
        self, element: str, T_e_eV: float, n_e_cm3: float, total_density_cm3: float
    ) -> Dict[int, float]:
        """Solve ionization balance, returning stage -> density mapping."""
        pass

    @abstractmethod
    def solve_level_population(
        self,
        element: str,
        ionization_stage: int,
        stage_density_cm3: float,
        T_e_eV: float,
        n_e_cm3: Optional[float] = None,
    ) -> Dict[Tuple[str, int, float], float]:
        """Solve level population distribution."""
        pass

    @abstractmethod
    def solve_plasma(self, plasma: "SingleZoneLTEPlasma") -> Dict[Tuple[str, int, float], float]:
        """Solve complete plasma system."""
        pass


@runtime_checkable
class PlasmaModel(Protocol):
    """
    Protocol for plasma models (structural typing).

    Any class with these properties/methods can be used as a PlasmaModel
    without explicit inheritance.
    """

    def validate(self) -> None:
        """Validate plasma state."""
        ...

    @property
    def T_e_eV(self) -> float:
        """Electron temperature in eV."""
        ...

    @property
    def n_e(self) -> float:
        """Electron density in cm^-3."""
        ...

    @property
    def species(self) -> Dict[str, float]:
        """Species densities in cm^-3."""
        ...


@runtime_checkable
class InstrumentModelProtocol(Protocol):
    """
    Protocol for instrument models (structural typing).

    Any class with these methods/properties can be used as an instrument model
    without explicit inheritance.
    """

    def apply_response(self, wavelength: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Apply spectral response curve."""
        ...

    def apply_instrument_function(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> np.ndarray:
        """Apply instrument function (convolution)."""
        ...

    @property
    def resolution_sigma_nm(self) -> float:
        """Instrument resolution (Gaussian sigma) in nm."""
        ...


# Backward compatibility alias
InstrumentModelInterface = InstrumentModelProtocol
