"""Run the constrained-element iterative solver with injected n_e (DED step 5).

The synthetic forward applies Stark broadening, but at n_e=1e17 the Stark width
(~0.005 nm) is far below a realistic instrument FWHM (0.1 nm) and is therefore
unmeasurable -- exactly as in real instrument-limited LIBS. So for a fair test
of the SOLVER's composition accuracy we inject the known n_e via a synthetic
Stark diagnostic (the Saha correction needs n_e). Composition is reported as
absolute wt% via the clean number->mass conversion (constrained set, no oxygen).
"""

from __future__ import annotations

from typing import Dict, Sequence

from cflibs.inversion.common import LineObservation
from cflibs.inversion.pipeline import _number_to_mass_fractions
from cflibs.inversion.solve.iterative import (
    CFLIBSResult,
    IterativeCFLIBSSolver,
    StarkDiagnosticLine,
)


def make_ne_diagnostic(
    ne_cm3: float, stark_w_ref_nm: float = 0.01, stark_alpha: float = 0.5
) -> StarkDiagnosticLine:
    """A synthetic diagnostic that makes the solver recover exactly ``ne_cm3``.

    n_e = 1e17 * (measured/stark_w_ref)^(1/alpha); set measured accordingly so
    the reference width cancels and any ``stark_w_ref_nm`` yields ``ne_cm3``.
    """
    measured = stark_w_ref_nm * (ne_cm3 / 1e17) ** stark_alpha
    return StarkDiagnosticLine(
        measured_fwhm_nm=measured,
        stark_w_ref_nm=stark_w_ref_nm,
        stark_alpha=stark_alpha,
        instrument_fwhm_nm=0.0,
        doppler_fwhm_nm=0.0,
    )


def run_constrained_solver(
    db,
    observations: Sequence[LineObservation],
    ne_cm3: float,
    *,
    max_iterations: int = 30,
    saha_boltzmann_graph: bool = True,
    closure_mode: str = "standard",
) -> CFLIBSResult:
    """Solve on a fixed known element set with the true n_e injected."""
    solver = IterativeCFLIBSSolver(
        db, saha_boltzmann_graph=saha_boltzmann_graph, max_iterations=max_iterations
    )
    res = solver.solve(
        list(observations),
        closure_mode=closure_mode,
        stark_diagnostics=[make_ne_diagnostic(ne_cm3)],
    )
    if not res.mass_fractions:
        res.mass_fractions = _number_to_mass_fractions(res.concentrations)
    return res


def recovered_wt(res: CFLIBSResult) -> Dict[str, float]:
    """Absolute recovered composition in wt% (mass fractions x 100)."""
    return {k: 100.0 * v for k, v in res.mass_fractions.items()}
