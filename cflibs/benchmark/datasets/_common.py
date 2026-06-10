"""Shared helpers for the per-dataset adapter modules (bead A2).

The dataset submodules used to lazy-import these names from their parent
(:mod:`cflibs.benchmark.adapters_extended`) inside ``iter_spectra`` to dodge
an import cycle; hosting them here (contract pieces re-exported from
:mod:`cflibs.benchmark.scoreboard_registry`, plus the wavelength-grid
helper) lets every submodule use plain top-level imports.
"""

from __future__ import annotations

import numpy as np

from cflibs.benchmark.scoreboard_registry import (
    PRESENCE_CUTOFF_WT,
    SpectrumTruth,
    presence_set,
)

__all__ = [
    "PRESENCE_CUTOFF_WT",
    "SpectrumTruth",
    "enforce_strictly_increasing",
    "presence_set",
]


def enforce_strictly_increasing(
    wavelength_nm: np.ndarray, intensity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a strictly increasing wavelength grid and matching intensities.

    Multi-channel spectrometers occasionally repeat (or slightly fold back)
    wavelengths at channel boundaries. This helper stable-sorts by wavelength
    and drops any point whose wavelength does not strictly exceed the previous
    kept point, keeping the first occurrence. Deterministic; a no-op for grids
    that are already strictly increasing.
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    if wl.size and np.all(np.diff(wl) > 0):
        return wl, inten
    order = np.argsort(wl, kind="stable")
    wl, inten = wl[order], inten[order]
    keep = np.empty(wl.shape, dtype=bool)
    keep[0] = True
    np.greater(wl[1:], np.maximum.accumulate(wl)[:-1], out=keep[1:])
    return wl[keep], inten[keep]
