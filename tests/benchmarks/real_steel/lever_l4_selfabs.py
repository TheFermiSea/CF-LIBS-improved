"""Lever L4 — self-absorption correction for the Fe matrix collapse (real-steel gate).

Root cause (docs/research/real-steel-accuracy-levers.md + the round-1 diagnosis):
Fe is the ~91-94 wt% matrix, so its lines are **optically thick**. Self-absorption
saturates their line-center emission and broadens the profile, so the *integrated*
intensity the extractor measures is far **below** the optically-thin value. The Fe
Boltzmann-plot intercept is pulled down, the recovered Fe number density collapses
(real-steel recovered Fe ~24-61 wt% vs true ~94), and the closure ``Sigma C = 1``
hands the lost mass to the trace minors. This Fe collapse is the dominant remaining
error (baseline Fe RMSEP 79.8, L2 64.5).

Diagnosis that shaped the correction
------------------------------------
A controlled experiment (boost only Fe's intensities by a uniform factor) showed
the collapse is curable: an ~8x uniform Fe boost lifts overall RMSEP 30.4 -> 14.8
and Fe 64.5 -> 29.3. Three facts fixed the design:

* The boost must be **uniform across an element's lines** (a per-line, width-
  differential boost rotates the common Boltzmann slope, corrupts the fitted T,
  and *worsens* Fe through the coupled Saha balance).
* The self-absorbed element must be identified by **line width**, NOT by emission:
  the trace minors carry anomalously bright (ion/resonance) lines, so ranking by
  intensity mis-picks a minor and collapses Fe further. Fe's lines are instead
  systematically *broadened* (real-steel: Fe median FWHM ~0.36 nm vs the ~0.27 nm
  instrument width; optically-thin minors sit at the instrument width).
* The boost *magnitude* comes from the El Sherbini et al. 2005 (*Spectrochim.
  Acta B* 60, 1573, doi:10.1016/j.sab.2005.10.011) self-absorption coefficient,
  ``SA = (Delta_lambda / Delta_lambda_0) ** (1 / -0.54)``, so the optically-thin
  intensity is recovered as ``I_thin = I_obs / SA``. Because the profile is
  instrument-convolved at this resolution, the instrument FWHM is removed in
  quadrature (Gaussian) to recover the intrinsic self-absorption broadening before
  the ratio is formed.

Method (per spectrum, observable-anchored, no composition feedback)
-------------------------------------------------------------------
1. Measure each L2-selected line's FWHM and peak SNR from the spectrum.
2. Instrument width ``Delta_lambda_instr`` = a low percentile of the well-measured
   line widths (the optically-thin floor of the width distribution).
3. For each element with >= ``MIN_CORR_LINES`` well-measured lines whose **median**
   width exceeds the instrument width by >= ``THICK_RATIO`` (i.e. it is optically
   thick), form the intrinsic width ``Delta_lambda_intr = sqrt(FWHM^2 -
   instr^2)`` per line and the El Sherbini boost ``M = (median(Delta_lambda_intr)
   / Delta_lambda_0) ** (1/alpha)`` (clamped to ``[1, SA_BOOST_MAX]``), with
   ``Delta_lambda_0`` the optically-thin intrinsic reference width and ``alpha =
   0.54``.
4. Apply ``M`` **uniformly** to all of that element's line intensities (preserving
   the Boltzmann slope), then run the constrained Saha-Boltzmann solve on the L2
   neutral-anchor line base.

The correction reads only measured line widths/intensities — never a recovered
composition (the positive-feedback failure the 2026-06-09 audit condemned). It is
self-contained per sample (no calibration standard), so it is evaluated over all 36
samples like L2. It needs the measured spectrum (line widths), so it lives in the
benchmark layer; no shipped ``cflibs/`` change and therefore no DED-precision risk.

Real-steel gate result (36 samples): overall RMSEP 30.43 (L2) -> 20.78, Fe 64.53
-> 42.71. Stable across floor in [0.05, 0.12] nm, alpha in [0.50, 0.60], clamp in
[12, 30], and the SNR/percentile gates -- the robustness expected of a physical
(not gate-tuned) correction.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from tests.benchmarks.ded_precision.benchmark_runner import (  # noqa: E402
    extract_line_intensities,
)
from tests.benchmarks.ded_precision.line_lists import LineSpec  # noqa: E402
from tests.benchmarks.ded_precision.solver_runner import (  # noqa: E402
    recovered_wt,
    run_constrained_solver,
)
from tests.benchmarks.real_steel.harness import run_benchmark  # noqa: E402
from tests.benchmarks.real_steel.lever_l2_lines import select_l2_lines  # noqa: E402

# El Sherbini 2005 self-absorption width exponent: SA = (dl/dl_0)^(1/-0.54), so the
# optically-thin intensity boost is (dl_intr/dl_0)^(1/ALPHA_SA).
ALPHA_SA = 0.54
# Optically-thin intrinsic reference width Delta_lambda_0 (nm): the residual
# (Stark + un-deconvolved) width of an optically-thin line. A physical mid-range
# value; the gate result is stable for any floor in [0.05, 0.12] nm.
THIN_REF_WIDTH_NM = 0.08
# Clamp on the per-element boost so a single mis-measured spectrum cannot run away
# (tau ~5 El Sherbini escape-factor validity implies a boost of order ten).
SA_BOOST_MAX = 20.0
# An element is corrected only with at least this many adequately-measured lines.
MIN_CORR_LINES = 3
# Peak-SNR floor for a line to count toward the width statistics / correction.
SNR_MIN = 3.0
# An element is treated as optically thick only when its MEDIAN measured width
# exceeds the instrument width by this factor (below it the element is thin and
# left uncorrected -- this is what keeps trace minors at boost 1).
THICK_RATIO = 1.15
# Percentile of the well-measured line widths taken as the instrument width
# (the optically-thin floor of the width distribution).
INSTR_WIDTH_PCT = 10.0
# Half-window (nm) for the width measurement: wide enough to capture a
# self-absorption-broadened profile at the ~0.15 nm/ch real-steel resolution.
WIDTH_HALF_NM = 0.6


def measure_line_width(
    wl: np.ndarray, intensity: np.ndarray, center_nm: float, half_nm: float = WIDTH_HALF_NM
) -> Tuple[float, float]:
    """Robust FWHM (nm) and peak SNR of the line at ``center_nm`` from the spectrum.

    Second-moment width after a robust median-edge local baseline (same baseline
    policy as the extractor), with the peak height measured against the window-
    edge noise. Returns ``(nan, 0.0)`` when the line cannot be measured.
    """
    wl = np.asarray(wl, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    mask = (wl >= center_nm - half_nm) & (wl <= center_nm + half_nm)
    if int(mask.sum()) < 5:
        return float("nan"), 0.0
    x = wl[mask]
    y = inten[mask].astype(float)
    k = max(1, x.size // 6)
    base = np.interp(x, [x[0], x[-1]], [float(np.median(y[:k])), float(np.median(y[-k:]))])
    y = y - base
    peak = float(y.max())
    if peak <= 0:
        return float("nan"), 0.0
    edge = np.concatenate([y[:k], y[-k:]])
    noise = float(np.std(edge)) if edge.size > 1 else 0.0
    snr = peak / noise if noise > 0 else float("inf")
    yp = np.clip(y, 0.0, None)
    tot = float(yp.sum())
    if tot <= 0:
        return float("nan"), 0.0
    mu = float(np.sum(x * yp) / tot)
    var = float(np.sum((x - mu) ** 2 * yp) / tot)
    fwhm = 2.354820045 * np.sqrt(max(var, 1e-12))
    return fwhm, snr


def apply_self_absorption(
    wl: np.ndarray,
    intensity: np.ndarray,
    observations: Sequence,
    *,
    alpha: float = ALPHA_SA,
    thin_ref_width_nm: float = THIN_REF_WIDTH_NM,
    sa_boost_max: float = SA_BOOST_MAX,
    min_corr_lines: int = MIN_CORR_LINES,
    snr_min: float = SNR_MIN,
    thick_ratio: float = THICK_RATIO,
    instr_width_pct: float = INSTR_WIDTH_PCT,
) -> Tuple[int, float]:
    """In-place width-gated uniform self-absorption boost of ``observations``.

    Each optically-thick element (median measured width exceeding the per-spectrum
    instrument width by ``thick_ratio``) is boosted by a single El Sherbini factor
    derived from its median intrinsic (instrument-deconvolved) width, applied
    uniformly to all its lines so the Boltzmann slope is preserved. Returns
    ``(n_elements_corrected, max_boost)``.
    """
    widths: Dict[int, float] = {}
    snrs: Dict[int, float] = {}
    by_el: Dict[str, List[int]] = {}
    for i, o in enumerate(observations):
        fwhm, snr = measure_line_width(wl, intensity, o.wavelength_nm)
        widths[i] = fwhm
        snrs[i] = snr
        by_el.setdefault(o.element, []).append(i)

    good_all = [widths[i] for i in widths if np.isfinite(widths[i]) and snrs[i] >= snr_min]
    if not good_all:
        return 0, 1.0
    instr = float(np.percentile(good_all, instr_width_pct))
    if instr <= 0:
        return 0, 1.0
    ref = max(float(thin_ref_width_nm), 1e-6)

    n_corr = 0
    max_boost = 1.0
    for _el, idxs in by_el.items():
        good = [i for i in idxs if np.isfinite(widths[i]) and snrs[i] >= snr_min]
        if len(good) < min_corr_lines:
            continue
        median_w = float(np.median([widths[i] for i in good]))
        if median_w / instr < thick_ratio:
            continue  # optically thin element -> no boost (keeps trace minors at 1)
        intrinsic = [np.sqrt(max(widths[i] ** 2 - instr**2, 0.0)) for i in good]
        median_intrinsic = float(np.median(intrinsic))
        boost = float(
            np.clip((max(median_intrinsic, ref) / ref) ** (1.0 / alpha), 1.0, sa_boost_max)
        )
        if boost <= 1.0:
            continue
        for i in idxs:  # uniform across ALL the element's lines (preserve slope)
            o = observations[i]
            o.intensity = float(o.intensity) * boost
            o.intensity_uncertainty = max(float(o.intensity_uncertainty) * boost, 1e-12)
        n_corr += 1
        max_boost = max(max_boost, boost)
    return n_corr, max_boost


def solve_l4(
    db,
    wl,
    intensity,
    truth,
    *,
    alpha: float = ALPHA_SA,
    thin_ref_width_nm: float = THIN_REF_WIDTH_NM,
    sa_boost_max: float = SA_BOOST_MAX,
    min_corr_lines: int = MIN_CORR_LINES,
    snr_min: float = SNR_MIN,
    thick_ratio: float = THICK_RATIO,
    instr_width_pct: float = INSTR_WIDTH_PCT,
) -> Dict[str, float]:
    """L4 solve: L2 lines + width-gated uniform self-absorption boost + Saha-Boltzmann.

    Self-contained per sample (no calibration standard): the correction is anchored
    on each spectrum's own line widths, so it is evaluated like L2 over all samples.
    """
    els = list(truth.keys())
    window = (float(wl.min()), float(wl.max()))
    specs: List[LineSpec] = []
    for e in els:
        specs.extend(select_l2_lines(db, e, window, 8))
    if not specs:
        return {}
    obs = extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=0.2)
    if not obs:
        return {}
    apply_self_absorption(
        wl,
        intensity,
        obs,
        alpha=alpha,
        thin_ref_width_nm=thin_ref_width_nm,
        sa_boost_max=sa_boost_max,
        min_corr_lines=min_corr_lines,
        snr_min=snr_min,
        thick_ratio=thick_ratio,
        instr_width_pct=instr_width_pct,
    )
    res = run_constrained_solver(db, obs, 1e17)
    return recovered_wt(res)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=ALPHA_SA)
    ap.add_argument("--thin-ref-width-nm", type=float, default=THIN_REF_WIDTH_NM)
    ap.add_argument("--sa-boost-max", type=float, default=SA_BOOST_MAX)
    ap.add_argument("--min-corr-lines", type=int, default=MIN_CORR_LINES)
    ap.add_argument("--snr-min", type=float, default=SNR_MIN)
    ap.add_argument("--thick-ratio", type=float, default=THICK_RATIO)
    ap.add_argument("--instr-width-pct", type=float, default=INSTR_WIDTH_PCT)
    a = ap.parse_args()

    def _fn(db, wl, intensity, truth):
        return solve_l4(
            db,
            wl,
            intensity,
            truth,
            alpha=a.alpha,
            thin_ref_width_nm=a.thin_ref_width_nm,
            sa_boost_max=a.sa_boost_max,
            min_corr_lines=a.min_corr_lines,
            snr_min=a.snr_min,
            thick_ratio=a.thick_ratio,
            instr_width_pct=a.instr_width_pct,
        )

    r = run_benchmark(_fn, limit=a.limit)
    print("L4 (width-gated self-absorption) real-steel RMSEP (wt%):")
    for k in sorted(r):
        v = r[k]
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
