"""
Plasma state definitions and solvers.

This module provides:
- Plasma state representations
- LTE / partial-LTE solvers
- Saha-Boltzmann solvers with constraint enforcement
- Multi-zone plasma models
"""

from cflibs.plasma.state import PlasmaState, SingleZoneLTEPlasma, SingleZoneLTEPlasmaJax
from cflibs.plasma.saha_boltzmann import (
    SahaBoltzmannSolver,
    SahaBoltzmannSolverJax,
    SpeciesStageState,
    ionization_potential_lowering,
)
from cflibs.plasma.lte_validator import LTEValidator, LTEReport, LTECheckResult
from cflibs.plasma.ipd import (
    StewartPyattIPD,
    make_ipd_model,
    stewart_pyatt_lowering,
)

__all__ = [
    "PlasmaState",
    "SingleZoneLTEPlasma",
    "SingleZoneLTEPlasmaJax",
    "SahaBoltzmannSolver",
    "SahaBoltzmannSolverJax",
    "SpeciesStageState",
    "ionization_potential_lowering",
    "LTEValidator",
    "LTEReport",
    "LTECheckResult",
    "StewartPyattIPD",
    "make_ipd_model",
    "stewart_pyatt_lowering",
]
