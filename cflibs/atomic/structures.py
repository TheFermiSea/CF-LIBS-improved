"""
Data structures for atomic physics data.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class EnergyLevel:
    """
    Represents an atomic energy level.

    Attributes
    ----------
    element : str
        Element symbol (e.g., 'Fe', 'Ti')
    ionization_stage : int
        Ionization stage (1=neutral, 2=singly ionized, etc.)
    energy_ev : float
        Energy above ground state in eV
    g : int
        Statistical weight (degeneracy)
    j : Optional[float]
        Total angular momentum quantum number
    """

    element: str
    ionization_stage: int
    energy_ev: float
    g: int
    j: Optional[float] = None


@dataclass
class Transition:
    """
    Represents an atomic transition.

    Attributes
    ----------
    element : str
        Element symbol
    ionization_stage : int
        Ionization stage (1=neutral, 2=singly ionized, etc.)
    wavelength_nm : float
        Transition wavelength in nm
    A_ki : float
        Einstein A coefficient (spontaneous emission rate) in s^-1
    E_k_ev : float
        Upper level energy in eV
    E_i_ev : float
        Lower level energy in eV
    g_k : int
        Upper level statistical weight
    g_i : int
        Lower level statistical weight
    relative_intensity : Optional[float]
        Relative intensity (from database, if available)
    stark_w : Optional[float]
        Stark width at reference density (10^16 cm^-3)
    stark_alpha : Optional[float]
        Stark width scaling exponent (typically ~1.0)
    stark_shift : Optional[float]
        Stark shift coefficient
    is_resonance : Optional[bool]
        Boolean flag for ground-state transitions
    aki_uncertainty : Optional[float]
        Fractional uncertainty in A_ki (0-1), derived from NIST accuracy grade
    accuracy_grade : Optional[str]
        NIST accuracy grade (AAA/AA/A/B/C/D/E) for transition probability
    """

    element: str
    ionization_stage: int
    wavelength_nm: float
    A_ki: float
    E_k_ev: float
    E_i_ev: float
    g_k: int
    g_i: int
    relative_intensity: Optional[float] = None
    stark_w: Optional[float] = None
    stark_alpha: Optional[float] = None
    stark_shift: Optional[float] = None
    is_resonance: Optional[bool] = False
    aki_uncertainty: Optional[float] = None
    accuracy_grade: Optional[str] = None

    @property
    def energy_diff_ev(self) -> float:
        """Energy difference between upper and lower levels."""
        return self.E_k_ev - self.E_i_ev


@dataclass
class SpeciesPhysics:
    """
    Physical properties of an atomic species.

    Attributes
    ----------
    element : str
        Element symbol
    ionization_stage : int
        Ionization stage
    ionization_potential_ev : float
        Ionization potential from this stage to next in eV
    atomic_mass : Optional[float]
        Atomic mass in amu
    """

    element: str
    ionization_stage: int
    ionization_potential_ev: float
    atomic_mass: Optional[float] = None


@dataclass
class PartitionFunction:
    """
    Partition function polynomial coefficients.
    log(U) = Σ a_n (log T)^n

    Attributes
    ----------
    element : str
        Element symbol
    ionization_stage : int
        Ionization stage
    coefficients : List[float]
        Polynomial coefficients [a0, a1, a2, a3, a4]
    t_min : float
        Minimum valid temperature (K)
    t_max : float
        Maximum valid temperature (K)
    source : str
        Data source
    """

    element: str
    ionization_stage: int
    coefficients: List[float]
    t_min: float
    t_max: float
    source: str
