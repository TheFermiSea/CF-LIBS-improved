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


def _fit_line_area(x, y, center, sigma):
    """Least-squares Gaussian (known center+sigma) + linear baseline -> area.

    Model y ~ A*exp(-(x-c)^2/2sigma^2) + b0 + b1*(x-c), linear in (A,b0,b1).
    The shape-matched fit rejects noise that does not look like the line, so
    it is far less biased on weak lines than windowed integration + a hard
    positivity clip (which rectifies/selects on noise). Returns area =
    A*sigma*sqrt(2*pi) (may be <=0; caller filters).
    """
    g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    M = np.column_stack([g, np.ones_like(x), (x - center)])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    amp = float(coef[0])
    return amp * sigma * np.sqrt(2.0 * np.pi)


def extract_line_intensities(
    wl: np.ndarray,
    intensity: np.ndarray,
    line_specs: Sequence[LineSpec],
    instrument_fwhm_nm: float = 0.1,
    n_sigma: float = 1.5,
    local_baseline: bool = True,
    skip_blended: bool = False,
    method: str = "integrate",
) -> List[LineObservation]:
    """Measure each known line and return a LineObservation list.

    Parameters
    ----------
    wl, intensity : arrays
        Spectrum (noisy or clean) on a shared wavelength grid.
    line_specs : sequence of LineSpec
    instrument_fwhm_nm : float
        Line width; sets the window and the fit sigma.
    method : {"integrate", "fit"}
        "integrate" (default): trapz over a +/-n_sigma window after a robust
        median-edge local baseline, no per-pixel clip. Gives the best clean
        floor here. "fit": shape-matched Gaussian least-squares amplitude;
        a pure-Gaussian model mismatches the Stark-broadened profile so it
        biases the clean floor -- kept for noisy-data experiments only.
    skip_blended : bool
        If True, drop LineSpecs flagged ``blended`` (overlap another element).
    """
    wl = np.asarray(wl, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    if wl.size < 3:
        return []
    step = float(np.median(np.diff(wl)))
    sigma = instrument_fwhm_nm * FWHM_TO_SIGMA
    # Fitting needs the full profile; integration uses a tight window.
    half = max((3.0 if method == "fit" else n_sigma) * sigma, 3.0 * step)

    obs: List[LineObservation] = []
    for ls in line_specs:
        if skip_blended and ls.blended:
            continue
        lo, hi = ls.wavelength_nm - half, ls.wavelength_nm + half
        mask = (wl >= lo) & (wl <= hi)
        if int(mask.sum()) < 4:
            continue
        x = wl[mask]
        y = inten[mask].astype(float).copy()
        if method == "fit":
            area = _fit_line_area(x, y, ls.wavelength_nm, sigma)
        else:
            if local_baseline:
                # robust median-edge baseline; subtract WITHOUT a per-pixel clip
                # (clipping rectifies readout noise -> inflates weak lines)
                k = max(1, x.size // 6)
                left = float(np.median(y[:k]))
                right = float(np.median(y[-k:]))
                base = np.interp(x, [x[0], x[-1]], [left, right])
                y = y - base
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
