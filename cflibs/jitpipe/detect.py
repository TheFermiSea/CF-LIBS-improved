"""Stage 2 — peak detection (J2 stub; signatures only, ADR-0004 §5.1.1).

Jittable peak finding over a preprocessed spectrum. Logic lands in bead J2.
No SQLite, no host imports (import-hygiene test).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig


def detect_peaks(
    intensities: Any,
    wavelengths_nm: Any,
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Detect candidate peaks in a preprocessed spectrum (J2).

    Parameters
    ----------
    intensities : array
        Preprocessed intensities, shape ``(N_pix,)``.
    wavelengths_nm : array
        Wavelength axis, nm, shape ``(N_pix,)``.
    params : PipelineParams
        Traced knobs (``min_peak_height``, ``peak_width_nm``, ``min_snr``).
    static : StaticConfig
        Static shape/mode config (jit cache key).

    Returns
    -------
    tuple of array
        ``(peak_wavelengths_nm, peak_intensities, peak_mask)`` padded to the
        line bucket (``static.bucket_id``).

    Raises
    ------
    NotImplementedError
        Stage logic is implemented in bead J2.
    """
    raise NotImplementedError("jitpipe.detect is a J0 skeleton; logic lands in J2")
