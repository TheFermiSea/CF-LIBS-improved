"""
Atomic data structures and database interfaces.

This module provides:
- Energy level representations
- Transition probability data structures
- Stark and broadening parameters
- Interfaces to external databases (NIST ASD, Kurucz, etc.)
"""

from cflibs.atomic.masses import (
    DEFAULT_ATOMIC_MASS_AMU,
    STANDARD_ATOMIC_MASSES,
    resolve_element_mass,
)
from cflibs.atomic.structures import EnergyLevel, Transition, SpeciesPhysics, PartitionFunction
from cflibs.atomic.wavelength_conversion import (
    AIR_VACUUM_BOUNDARY_NM,
    air_to_vacuum_nm,
    vacuum_to_air_nm,
)


# Lazy import to avoid circular dependency
def __getattr__(name):
    if name == "AtomicDatabase":
        from cflibs.atomic.database import AtomicDatabase

        return AtomicDatabase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EnergyLevel",
    "Transition",
    "SpeciesPhysics",
    "PartitionFunction",
    "AtomicDatabase",
    "STANDARD_ATOMIC_MASSES",
    "DEFAULT_ATOMIC_MASS_AMU",
    "resolve_element_mass",
    "air_to_vacuum_nm",
    "vacuum_to_air_nm",
    "AIR_VACUUM_BOUNDARY_NM",
]
