"""
Abstract base classes and protocols for extensibility.

ABCs are used for core interfaces (SolverStrategy) that must be inherited.
Protocols are used for structural typing of classes that may implement the
interface without explicit inheritance (AtomicDataSource).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Protocol, TYPE_CHECKING, runtime_checkable

from cflibs.atomic.structures import Transition, EnergyLevel

if TYPE_CHECKING:
    from cflibs.plasma.state import SingleZoneLTEPlasma


@runtime_checkable
class AtomicDataSource(Protocol):
    """
    Structural interface for atomic data sources.

    Any class with these methods (SQLite, NIST API, HDF5, ...) satisfies it
    without explicit inheritance — which is how it is actually used: production
    code duck-types the concrete ``AtomicDatabase`` and tests pass minimal fakes.
    It was an ABC, but with a single concrete adapter and no polymorphic
    dispatch the inheritance earned nothing; a ``Protocol`` keeps the structural
    contract (and ``isinstance`` via ``runtime_checkable``) without forcing the
    base class. Note the real variation seam, ``partition_function_for``, is a
    convenience method on ``AtomicDatabase`` and intentionally NOT part of this
    protocol (call sites hasattr-guard it).

    The methods stay ``@abstractmethod`` so the best of both holds: a duck-typed
    class satisfies it structurally without inheritance, while a class that
    *explicitly* subclasses it still fails fast at instantiation if it forgets a
    method (Protocol uses ABCMeta).
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
        ...

    @abstractmethod
    def get_energy_levels(self, element: str, ionization_stage: int) -> List[EnergyLevel]:
        """Get energy levels for a species."""
        ...

    @abstractmethod
    def get_ionization_potential(self, element: str, ionization_stage: int) -> Optional[float]:
        """Get ionization potential for a species in eV."""
        ...

    @abstractmethod
    def get_available_elements(self) -> List[str]:
        """Get list of available element symbols."""
        ...


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
