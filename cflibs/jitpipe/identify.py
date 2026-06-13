"""Stage 4 — element / line identification (J4 stub; ADR-0004 §5.1.1).

Jittable line identification (ALIAS / comb / correlation scoring) of detected
peaks against catalog candidates. Logic lands in bead J4. No SQLite, no host
imports (import-hygiene test).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot


def identify_lines(
    peak_wavelengths_nm: Any,
    peak_intensities: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Identify catalog lines for each detected peak (J4).

    Parameters
    ----------
    peak_wavelengths_nm : array
        Calibrated peak wavelengths, nm.
    peak_intensities : array
        Peak intensities.
    snapshot : PipelineSnapshot
        Atomic-data snapshot (candidate catalog).
    params : PipelineParams
        Traced knobs (``wavelength_tolerance_nm``, score weights).
    static : StaticConfig
        Static shape/mode config (jit cache key).

    Returns
    -------
    tuple of array
        ``(line_index, match_score, match_mask)`` per peak, padded to the
        line bucket.

    Raises
    ------
    NotImplementedError
        Stage logic is implemented in bead J4.
    """
    raise NotImplementedError("jitpipe.identify is a J0 skeleton; logic lands in J4")
