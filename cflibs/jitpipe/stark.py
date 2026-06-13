"""Stage 6b — Stark n_e diagnostic (J6 stub; ADR-0004 §5.1.1).

Jittable electron-density estimate from observed Stark line widths, using the
snapshot's ``line_stark_w`` / ``line_stark_alpha`` / atomic-mass (Doppler)
blocks. Logic lands in bead J6. No SQLite, no host imports (import-hygiene
test).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot


def stark_electron_density(
    line_widths_nm: Any,
    line_index: Any,
    temperature_eV: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Estimate electron density from observed Stark widths (J6).

    Parameters
    ----------
    line_widths_nm : array
        Measured line FWHM per line, nm.
    line_index : array
        Catalog-line indices into ``snapshot`` per line.
    temperature_eV : array
        Plasma temperature, eV (for the Stark alpha temperature factor and
        Doppler deconvolution).
    snapshot : PipelineSnapshot
        Atomic-data snapshot (``line_stark_w``, ``line_stark_alpha``,
        ``species_physics`` masses).
    params : PipelineParams
        Traced knobs.
    static : StaticConfig
        Static shape/mode config (jit cache key).

    Returns
    -------
    array
        Electron density estimate(s), cm^-3.

    Raises
    ------
    NotImplementedError
        Stage logic is implemented in bead J6.
    """
    raise NotImplementedError("jitpipe.stark is a J0 skeleton; logic lands in J6")
