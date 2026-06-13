"""Stage 5 — Boltzmann-plot fitting (J5 stub; ADR-0004 §5.1.1).

Jittable multi-element Boltzmann-plot fit (ln(I lambda / gA) vs E_k) yielding
per-species temperature and intercept. Logic lands in bead J5. No SQLite, no
host imports (import-hygiene test).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot


def boltzmann_fit(
    line_intensities: Any,
    line_index: Any,
    line_mask: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Fit per-species Boltzmann plots for the identified lines (J5).

    Parameters
    ----------
    line_intensities : array
        Integrated intensities per identified line.
    line_index : array
        Catalog-line indices into ``snapshot`` per identified line.
    line_mask : array of bool
        Valid-line mask (padding).
    snapshot : PipelineSnapshot
        Atomic-data snapshot (g, A, E_k).
    params : PipelineParams
        Traced knobs (``boltzmann_weight_cap``, ``min_boltzmann_r2``).
    static : StaticConfig
        Static shape/mode config (jit cache key).

    Returns
    -------
    tuple of array
        ``(temperature_eV, intercept, r2)`` per species.

    Raises
    ------
    NotImplementedError
        Stage logic is implemented in bead J5.
    """
    raise NotImplementedError("jitpipe.fit is a J0 skeleton; logic lands in J5")
