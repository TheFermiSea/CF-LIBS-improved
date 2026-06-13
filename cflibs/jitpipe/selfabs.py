"""Stage 6a — self-absorption correction (J5/J6 stub; ADR-0004 §5.1.1).

Jittable doublet-ratio + curve-of-growth self-absorption correction using the
``doublet_pairs`` / ``doublet_rho`` / ``doublet_r_thin`` blocks of the
snapshot. Logic lands in bead J5. No SQLite, no host imports (import-hygiene
test).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot


def correct_self_absorption(
    line_intensities: Any,
    line_index: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Apply doublet-ratio / CDSB self-absorption correction (J5).

    Parameters
    ----------
    line_intensities : array
        Measured integrated intensities per line.
    line_index : array
        Catalog-line indices into ``snapshot`` per line.
    snapshot : PipelineSnapshot
        Atomic-data snapshot (``doublet_pairs``, ``doublet_rho``,
        ``doublet_r_thin``).
    params : PipelineParams
        Traced knobs.
    static : StaticConfig
        Static shape/mode config (jit cache key).

    Returns
    -------
    array
        Self-absorption-corrected intensities.

    Raises
    ------
    NotImplementedError
        Stage logic is implemented in bead J5.
    """
    raise NotImplementedError("jitpipe.selfabs is a J0 skeleton; logic lands in J5")
