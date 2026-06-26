"""Extract integrated line intensities at KNOWN positions (DED-PLAN step 4).

Bypasses peak detection and element ID entirely: for each curated LineSpec,
integrate the (noisy) spectrum over a +/-1.5 sigma window after a local linear
baseline subtraction, yielding a LineObservation carrying the line's atomic
parameters. This isolates the SOLVER's absolute accuracy from detection/ID
error budgets (which are separate layers).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from cflibs.inversion.common import LineObservation

from .line_lists import LineSpec

FWHM_TO_SIGMA = 1.0 / 2.354820045

# NumPy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def extract_line_intensities(
    wl: np.ndarray,
    intensity: np.ndarray,
    line_specs: Sequence[LineSpec],
    instrument_fwhm_nm: float = 0.1,
    n_sigma: float = 1.5,
    local_baseline: bool = True,
    skip_blended: bool = False,
) -> List[LineObservation]:
    """Integrate each known line and return LineObservation list.

    Parameters
    ----------
    wl, intensity : arrays
        Spectrum (noisy or clean) on a shared wavelength grid.
    line_specs : sequence of LineSpec
    instrument_fwhm_nm : float
        Sets the integration half-width (max of n_sigma*sigma and 3 grid steps).
    skip_blended : bool
        If True, drop LineSpecs flagged ``blended`` (overlap another element).
    """
    wl = np.asarray(wl, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    if wl.size < 3:
        return []
    step = float(np.median(np.diff(wl)))
    sigma = instrument_fwhm_nm * FWHM_TO_SIGMA
    half = max(n_sigma * sigma, 3.0 * step)

    obs: List[LineObservation] = []
    for ls in line_specs:
        if skip_blended and ls.blended:
            continue
        lo, hi = ls.wavelength_nm - half, ls.wavelength_nm + half
        mask = (wl >= lo) & (wl <= hi)
        if int(mask.sum()) < 3:
            continue
        x = wl[mask]
        y = inten[mask].astype(float).copy()
        if local_baseline:
            base = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
            y = np.clip(y - base, 0.0, None)
        area = float(_trapz(y, x))
        if not np.isfinite(area) or area <= 0.0:
            continue
        obs.append(
            LineObservation(
                wavelength_nm=ls.wavelength_nm,
                intensity=area,
                intensity_uncertainty=max(area * 0.05, 1e-12),
                element=ls.element,
                ionization_stage=ls.ionization_stage,
                E_k_ev=ls.E_k_ev,
                g_k=ls.g_k,
                A_ki=ls.A_ki,
                aki_uncertainty=ls.aki_uncertainty,
            )
        )
    return obs
