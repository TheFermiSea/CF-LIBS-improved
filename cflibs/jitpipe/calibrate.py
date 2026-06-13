"""Stage 3 — wavelength calibration (J3 stub; signatures only, ADR-0004 §5.1.1).

Jittable shift/affine wavelength calibration of detected peaks against the
catalog. Logic lands in bead J3. No SQLite, no host imports (import-hygiene
test).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot


def calibrate_wavelengths(
    peak_wavelengths_nm: Any,
    snapshot: "PipelineSnapshot",
    candidate_line_index: Any,
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Calibrate the detected-peak wavelength axis against the catalog (J3).

    Parameters
    ----------
    peak_wavelengths_nm : array
        Detected peak wavelengths, nm.
    snapshot : PipelineSnapshot
        Atomic-data snapshot (catalog wavelengths).
    candidate_line_index : array
        Per-peak candidate catalog-line indices into ``snapshot``.
    params : PipelineParams
        Traced knobs (``global_shift_scan_nm``, ``residual_shift_scan_nm``).
    static : StaticConfig
        Static shape/mode config (jit cache key).

    Returns
    -------
    tuple of array
        ``(calibrated_wavelengths_nm, shift_nm)``.

    Raises
    ------
    NotImplementedError
        Stage logic is implemented in bead J3.
    """
    raise NotImplementedError("jitpipe.calibrate is a J0 skeleton; logic lands in J3")
