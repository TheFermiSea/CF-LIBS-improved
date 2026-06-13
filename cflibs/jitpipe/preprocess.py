"""Stage 1 — preprocessing (J1 stub; signatures only, ADR-0004 §5.1.1).

Jittable baseline removal, noise estimation, and normalization of a raw
spectrum. Logic lands in bead J1; this module exists at J0 only to pin the
stage signature and keep the package layout complete. No SQLite, no host
imports (import-hygiene test).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig


def preprocess_spectrum(
    intensities: Any,
    wavelengths_nm: Any,
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Baseline-remove, denoise and normalize a raw spectrum (J1).

    Parameters
    ----------
    intensities : array
        Raw measured intensities, shape ``(N_pix,)``.
    wavelengths_nm : array
        Wavelength axis, nm, shape ``(N_pix,)``.
    params : PipelineParams
        Traced continuous knobs (noise/baseline thresholds).
    static : StaticConfig
        Static shape/mode config (jit cache key).

    Returns
    -------
    array
        Preprocessed intensities, shape ``(N_pix,)``.

    Raises
    ------
    NotImplementedError
        Stage logic is implemented in bead J1.
    """
    raise NotImplementedError("jitpipe.preprocess is a J0 skeleton; logic lands in J1")
