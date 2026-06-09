"""Canonical standard atomic-mass table and a single mass-resolution ladder.

Historically the same ``STANDARD_MASSES``-style dictionary plus the
``database -> table -> 50.0 amu`` fallback ladder was independently redefined in
several modules (``manifold/generator.py``, ``manifold/basis_library.py``,
``inversion/solve/bayesian/atomic.py``). Those copies were byte-identical on
every overlapping element (verified element-by-element), so they are consolidated
here without any change in numeric behaviour.

This module is the *deep* module behind a small interface: callers cross one seam
(:data:`STANDARD_ATOMIC_MASSES` and :func:`resolve_element_mass`) instead of each
maintaining its own table and ladder.

The masses are NIST standard atomic weights (amu). The Doppler width scales as
``1 / sqrt(M)``, so a coarse mass is adequate; the generic
:data:`DEFAULT_ATOMIC_MASS_AMU` placeholder covers elements absent from both the
database and this table.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

__all__ = [
    "STANDARD_ATOMIC_MASSES",
    "DEFAULT_ATOMIC_MASS_AMU",
    "AtomicMassSource",
    "resolve_element_mass",
]


# Generic fallback mass (amu) for elements with neither a database value nor a
# tabulated standard weight. Preserved verbatim from every prior copy.
DEFAULT_ATOMIC_MASS_AMU: float = 50.0


# Canonical standard atomic masses (amu). This is the superset of every prior
# per-module table; each old copy was a (value-identical) subset of this one.
STANDARD_ATOMIC_MASSES: dict[str, float] = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.01,
    "N": 14.01,
    "O": 16.00,
    "F": 19.00,
    "Ne": 20.18,
    "Na": 22.99,
    "Mg": 24.31,
    "Al": 26.98,
    "Si": 28.09,
    "P": 30.97,
    "S": 32.07,
    "Cl": 35.45,
    "Ar": 39.95,
    "K": 39.10,
    "Ca": 40.08,
    "Sc": 44.96,
    "Ti": 47.87,
    "V": 50.94,
    "Cr": 52.00,
    "Mn": 54.94,
    "Fe": 55.85,
    "Co": 58.93,
    "Ni": 58.69,
    "Cu": 63.55,
    "Zn": 65.38,
    "Ga": 69.72,
    "Ge": 72.63,
    "As": 74.92,
    "Se": 78.97,
    "Br": 79.90,
    "Kr": 83.80,
    "Rb": 85.47,
    "Sr": 87.62,
    "Y": 88.91,
    "Zr": 91.22,
    "Nb": 92.91,
    "Mo": 95.95,
    "Ru": 101.1,
    "Rh": 102.9,
    "Pd": 106.4,
    "Ag": 107.9,
    "Cd": 112.4,
    "In": 114.8,
    "Sn": 118.7,
    "Sb": 121.8,
    "Te": 127.6,
    "I": 126.9,
    "Xe": 131.3,
    "Cs": 132.9,
    "Ba": 137.3,
    "La": 138.9,
    "Ce": 140.1,
    "Pr": 140.9,
    "Nd": 144.2,
    "Sm": 150.4,
    "Eu": 152.0,
    "Gd": 157.3,
    "Tb": 158.9,
    "Dy": 162.5,
    "Ho": 164.9,
    "Er": 167.3,
    "Tm": 168.9,
    "Yb": 173.0,
    "Lu": 175.0,
    "Hf": 178.5,
    "Ta": 180.9,
    "W": 183.8,
    "Re": 186.2,
    "Os": 190.2,
    "Ir": 192.2,
    "Pt": 195.1,
    "Au": 197.0,
    "Hg": 200.6,
    "Tl": 204.4,
    "Pb": 207.2,
    "Bi": 209.0,
    "U": 238.0,
}


@runtime_checkable
class AtomicMassSource(Protocol):
    """Structural type for a database that can supply an atomic mass.

    Matches :class:`cflibs.atomic.database.AtomicDatabase`, whose
    ``get_atomic_mass`` returns the mass in amu or ``None`` when absent.
    """

    def get_atomic_mass(self, element: str) -> Optional[float]: ...


def resolve_element_mass(
    element: str,
    atomic_db: Optional[AtomicMassSource] = None,
    *,
    require_positive: bool = False,
    table: Optional[dict[str, float]] = None,
) -> float:
    """Resolve an element's atomic mass (amu) via the canonical fallback ladder.

    The ladder is ``database value -> standard table -> generic 50.0 amu``:

    1. If ``atomic_db`` is provided, query :meth:`AtomicMassSource.get_atomic_mass`.
       A database hit is used directly. With ``require_positive=False`` (the
       :class:`ManifoldGenerator` policy) any non-``None`` value is accepted; with
       ``require_positive=True`` (the :class:`BasisLibraryGenerator` policy) the
       value must additionally be ``> 0`` to be accepted.
    2. Otherwise fall back to :data:`STANDARD_ATOMIC_MASSES`.
    3. Otherwise return :data:`DEFAULT_ATOMIC_MASS_AMU`.

    Passing ``atomic_db=None`` yields the pure table-only ladder
    (``STANDARD_ATOMIC_MASSES.get(element, 50.0)``) used by the Bayesian forward
    model's snapshot path.

    Parameters
    ----------
    element : str
        Element symbol (e.g. ``"Fe"``).
    atomic_db : AtomicMassSource, optional
        Database to consult first. When ``None``, only the table is consulted.
    require_positive : bool, keyword-only
        When ``True``, a database value is accepted only if it is strictly
        positive (otherwise the ladder continues to the table). Defaults to
        ``False`` to preserve the historical ManifoldGenerator semantics.
    table : dict[str, float], optional
        Override for the standard-mass fallback table. Defaults to the canonical
        :data:`STANDARD_ATOMIC_MASSES`. Callers that historically used a strict
        subset of the canonical table pass that subset here to preserve their
        exact table-miss (50.0 amu fallback) behaviour.

    Returns
    -------
    float
        The atomic mass in amu.
    """
    masses = STANDARD_ATOMIC_MASSES if table is None else table
    if atomic_db is not None:
        db_mass = atomic_db.get_atomic_mass(element)
        if db_mass is not None and (not require_positive or db_mass > 0.0):
            return float(db_mass)
    if element in masses:
        return masses[element]
    return DEFAULT_ATOMIC_MASS_AMU
