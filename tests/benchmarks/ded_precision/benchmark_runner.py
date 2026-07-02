"""Monte-Carlo composition-series runner (DED-PLAN step 5).

For each composition in a drift series, generate ``n_monte_carlo`` realizations
(each the average of ``n_shots`` noisy shots), extract at known line positions,
solve on the constrained element set with injected n_e, and record the absolute
recovered wt%. Returns a long-form DataFrame; ``summarize_series`` aggregates it
into per-element accuracy (RMSEP/bias) and precision (std).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cflibs.atomic.database import AtomicDatabase

from .alloy_definitions import ALLOY_WINDOWS_NM, COMPOSITION_SERIES, elements_of
from .line_extractor import extract_line_intensities
from .line_lists import build_alloy_line_list
from .noise_model import DEDNoiseParams
from .solver_runner import recovered_wt, run_constrained_solver
from .spectrum_generator import clean_spectrum, default_grid, make_forward, noisy_shot


def run_composition_series(
    db_path: str,
    alloy: str,
    axis: str,
    *,
    n_shots: int = 1,
    n_monte_carlo: int = 20,
    T_K: float = 11000.0,
    ne_cm3: float = 1e17,
    noise: Optional[DEDNoiseParams] = None,
    instrument_fwhm_nm: float = 0.1,
    grid_step_nm: float = 0.02,
    clean: bool = False,
    seed: int = 0,
    budget=None,
    prefer_spread: bool = False,
    line_T_K=None,
    closure_mode: str = "standard",
    max_iterations: int = 30,
    saha_boltzmann_graph: bool = True,
) -> pd.DataFrame:
    """Run one drift axis (e.g. Ti-6Al-4V Al scan); return a long-form DataFrame.

    ``clean=True`` skips noise (one realization per point) -- a solver-accuracy
    floor with no noise. Otherwise each point gets ``n_monte_carlo`` noisy
    realizations, each averaging ``n_shots`` shots.
    """
    db = AtomicDatabase(db_path)
    els = elements_of(alloy)
    wl = default_grid(ALLOY_WINDOWS_NM[alloy], grid_step_nm)
    fwd = make_forward(db_path, els, wl, instrument_fwhm_nm)
    line_specs = [
        s
        for v in build_alloy_line_list(
            db,
            alloy,
            T_K=(line_T_K if line_T_K is not None else T_K),
            budget=budget,
            prefer_spread=prefer_spread,
        ).values()
        for s in v
    ]
    noise = noise or DEDNoiseParams()
    rng = np.random.default_rng(seed)
    n_real = 1 if clean else n_monte_carlo

    rows: List[Dict[str, object]] = []
    for ci, comp in enumerate(COMPOSITION_SERIES[alloy][axis]):
        target_val = comp[axis]
        for mc in range(n_real):
            if clean:
                spec = clean_spectrum(fwd, comp, els, T_K, ne_cm3)
            else:
                shots = [
                    noisy_shot(fwd, comp, els, T_K, ne_cm3, noise, rng) for _ in range(n_shots)
                ]
                spec = np.mean(shots, axis=0)
            obs = extract_line_intensities(
                wl, spec, line_specs, instrument_fwhm_nm=instrument_fwhm_nm
            )
            try:
                res = run_constrained_solver(
                    db,
                    obs,
                    ne_cm3,
                    max_iterations=max_iterations,
                    saha_boltzmann_graph=saha_boltzmann_graph,
                    closure_mode=closure_mode,
                )
                pred = recovered_wt(res)
                conv = bool(res.converged)
            except Exception as exc:  # solver failures are data, not crashes
                pred = {e: float("nan") for e in els}
                conv = False
                rows.append({"error": str(exc)[:80]})
            for e in els:
                rows.append(
                    {
                        "alloy": alloy,
                        "axis": axis,
                        "comp_index": ci,
                        "target_value": float(target_val),
                        "mc": mc,
                        "element": e,
                        "truth_wt": float(comp[e]),
                        "pred_wt": float(pred.get(e, float("nan"))),
                        "error": float(pred.get(e, float("nan")) - comp[e]),
                        "converged": conv,
                        "n_shots": n_shots,
                    }
                )
    return pd.DataFrame([r for r in rows if "element" in r])


def summarize_log_ratios(df: pd.DataFrame, reference: str = "Ti") -> pd.DataFrame:
    """Per-numerator Aitchison log-ratio accuracy ``ln(el/reference)`` (Issue 2).

    Pivots the long-form run into per-(composition, realization) compositions and
    reports RMSEP/bias/std of ``ln(pred_el/pred_ref) - ln(truth_el/truth_ref)``
    for every element other than ``reference``. This is the matrix-invariant DED
    tracking deliverable that is reported alongside the absolute wt% RMSEP from
    :func:`summarize_series`: the ratio cancels the shared closure denominator,
    so it is stable under per-element mass-slosh (``MatrixEffects.lean``
    ``recoveredComposition_ratio_matrix_invariant``).

    The log-ratio error is computed from the recorded wt% columns; the
    atomic-weight offset cancels in the predicted-minus-truth difference, so the
    result is identical to a number-fraction log-ratio (see
    :func:`tests.benchmarks.ded_precision.metrics.log_ratio_metrics`).
    """
    from .metrics import log_ratio_metrics

    predicted: List[Dict[str, float]] = []
    truth: List[Dict[str, float]] = []
    for _, g in df.groupby(["comp_index", "mc"]):
        predicted.append({row.element: float(row.pred_wt) for row in g.itertuples()})
        truth.append({row.element: float(row.truth_wt) for row in g.itertuples()})
    if not predicted:
        return pd.DataFrame()
    m = log_ratio_metrics(predicted, truth, reference)
    rows = [
        {"numerator": num, "reference": reference, **stats}
        for num, stats in m["per_numerator"].items()  # type: ignore[union-attr]
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("numerator")


def summarize_series(df: pd.DataFrame) -> pd.DataFrame:
    """Per-element accuracy (RMSEP/bias/MAE/MaxAE) + precision (std) across a run."""
    out = []
    for el, g in df.groupby("element"):
        err = g["error"].to_numpy(dtype=float)
        err = err[np.isfinite(err)]
        if err.size == 0:
            out.append({"element": el, "rmsep": np.nan, "bias": np.nan, "n": 0})
            continue
        # precision: std of predictions within each composition point, averaged
        stds = [
            float(np.nanstd(sub["pred_wt"].to_numpy(dtype=float)))
            for _, sub in g.groupby("comp_index")
        ]
        out.append(
            {
                "element": el,
                "rmsep": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
                "mae": float(np.mean(np.abs(err))),
                "maxae": float(np.max(np.abs(err))),
                "sigma_mean": float(np.nanmean(stds)),
                "n": int(err.size),
            }
        )
    return pd.DataFrame(out).set_index("element")
