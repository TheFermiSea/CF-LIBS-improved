"""Stage 7 — iterative CF-LIBS solve (J7 stub; ADR-0004 §5.1.1).

Jittable iterative CF-LIBS loop (Saha correction -> common-slope fit ->
closure -> charge/pressure balance -> n_e update) as a ``lax.while_loop``.
J7 reuses the lax solve kernels verbatim against the unified snapshot via
:meth:`cflibs.jitpipe.snapshot.PipelineSnapshot.to_lax_snapshot`. Logic lands
in bead J7. No SQLite, no host imports (import-hygiene test).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot


def iterative_solve(
    line_intensities: Any,
    line_index: Any,
    temperature_eV: Any,
    electron_density: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Run the iterative CF-LIBS plasma-parameter solve (J7).

    Parameters
    ----------
    line_intensities : array
        Self-absorption-corrected intensities per line.
    line_index : array
        Catalog-line indices into ``snapshot`` per line.
    temperature_eV : array
        Initial temperature, eV (from the Boltzmann fit).
    electron_density : array
        Initial electron density, cm^-3 (from the Stark diagnostic).
    snapshot : PipelineSnapshot
        Atomic-data snapshot.
    params : PipelineParams
        Traced knobs (``t_tolerance_k``, ``ne_tolerance_frac``,
        ``pressure_pa``, ``max_iterations``).
    static : StaticConfig
        Static config (``max_iters`` loop bound, ``closure_mode``).

    Returns
    -------
    tuple
        ``(temperature_eV, electron_density, concentrations, converged,
        iterations)``.

    Raises
    ------
    NotImplementedError
        Stage logic is implemented in bead J7.
    """
    raise NotImplementedError("jitpipe.solve is a J0 skeleton; logic lands in J7")
