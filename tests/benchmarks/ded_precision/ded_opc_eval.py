"""Known-matrix OPC evaluation on the DED real goal (Ti-6Al-4V drift tracking).

The DED deployment target is *drift tracking on a known matrix*: the feedstock
composition (Ti-6Al-4V Grade 5) is known a priori, so a matrix-matched
calibration is available by construction. This is exactly the regime the shipped
known-matrix OPC mode (``cflibs.inversion.physics.opc``) was promoted for. This
module measures whether applying OPC to the DED Ti-6Al-4V series helps the V/Ti
and Al/Ti drift-tracking residual, vs the calibration-free baseline.

Methodology (structurally honest, mirrors ``real_steel/test_opc_shipped.py``):

1. **Standard = nominal feedstock.** The matrix-matched standard is the nominal
   Ti-6Al-4V composition ({Ti:90, Al:6, V:4} wt%). Its ``recover(T)`` callback
   forward-models the *standard's own* clean spectrum and solves the constrained
   Saha-Boltzmann balance at a fixed T -- it never sees a drift (unknown) point,
   so the calibration cannot peek.
2. **Calibrate.** ``calibrate_opc([standard])`` picks the robust temperature
   (optimal-T scan over the standard's own data) and the per-element factor
   ``F = geomean(C_true / C_rec)``.
3. **Apply + score the drift series.** For each drift composition (held-out: the
   nominal point is the only calibration data, the rest are drift away from it),
   solve WITHOUT OPC (calibration-free default: recover T from the Boltzmann
   slope) and WITH OPC (``apply_opc`` rescales intensities, solve at the robust
   fixed T). Track recovered V/Ti and Al/Ti ratios vs truth.

HONEST CAVEAT (in code + in the report). The DED synthetic forward is CLEAN: it
has none of the real-data systematic biases (ion-stage-only sampling, matrix
self-absorption) that OPC's per-element ``F`` was built to absorb. So on this
synthetic, OPC may be NEUTRAL or even slightly NEGATIVE -- it corrects a bias the
synthetic does not contain, and folds in a fixed-T assumption the clean solver
does not need. The value of OPC is on REAL data (real-steel held-out: 4x RMSEP
reduction). This test's purpose is twofold: (a) confirm OPC does NOT HARM the DED
real-goal path, and (b) report whether it moves the V/Ti residual at all.

Run::

    PYTHONPATH=$PWD python tests/benchmarks/ded_precision/ded_opc_eval.py
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.inversion.physics.opc import (  # noqa: E402
    OPCCalibration,
    Standard,
    StandardRecovery,
    apply_opc,
    calibrate_opc,
)

from .alloy_definitions import (  # noqa: E402
    ALLOY_COMPOSITIONS,
    ALLOY_WINDOWS_NM,
    COMPOSITION_SERIES,
    elements_of,
)
from .line_extractor import extract_line_intensities  # noqa: E402
from .line_lists import build_alloy_line_list  # noqa: E402
from .noise_model import DEDNoiseParams  # noqa: E402
from .solver_runner import recovered_wt, run_constrained_solver  # noqa: E402
from .spectrum_generator import clean_spectrum, default_grid, make_forward, noisy_shot  # noqa: E402

DB_PATH = "ASD_da/libs_production.db"
ALLOY = "Ti-6Al-4V"


# --- standard construction (un-peekable: standard's own spectrum only) --------


def _build_standard(
    db,
    fwd,
    wl: np.ndarray,
    line_specs,
    els: Sequence[str],
    *,
    T_K: float,
    ne_cm3: float,
    instrument_fwhm_nm: float,
) -> Standard:
    """Matrix-matched standard = nominal feedstock; recover() solves at fixed T.

    The standard's clean spectrum is forward-modeled ONCE (it is a fixed
    "measurement" of the nominal feedstock). ``recover(T)`` re-solves that fixed
    spectrum at the scanned solve temperature -- exactly the optimal-T scan the
    shipped ``choose_optimal_temperature`` needs.
    """
    nominal = ALLOY_COMPOSITIONS[ALLOY]
    spec_std = clean_spectrum(fwd, nominal, els, T_K, ne_cm3)
    obs_std = extract_line_intensities(
        wl, spec_std, line_specs, instrument_fwhm_nm=instrument_fwhm_nm
    )

    @lru_cache(maxsize=None)
    def _recover(T: float) -> StandardRecovery:
        if not obs_std:
            return StandardRecovery(composition={}, converged=False, degenerate=True)
        res = run_constrained_solver(db, obs_std, ne_cm3, fixed_temperature_K=float(T))
        degenerate = float((res.quality_metrics or {}).get("degenerate_composition", 0.0)) >= 0.5
        return StandardRecovery(
            composition=recovered_wt(res),
            converged=bool(res.converged),
            degenerate=degenerate,
        )

    return Standard(name=f"{ALLOY}-nominal", certified=dict(nominal), recover=_recover)


# --- per-point recovery (with / without OPC) ---------------------------------


def _recover_point(
    db,
    fwd,
    wl: np.ndarray,
    line_specs,
    els: Sequence[str],
    comp: Dict[str, float],
    *,
    T_K: float,
    ne_cm3: float,
    instrument_fwhm_nm: float,
    calibration: OPCCalibration,
    clean: bool,
    noise: Optional[DEDNoiseParams],
    rng: Optional[np.random.Generator],
    n_monte_carlo: int,
    n_shots: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (pred_without_opc, pred_with_opc) wt% dicts for one composition.

    For a noisy run the two predictions are the per-MC-realization means of the
    recovered wt%; for clean, a single realization.
    """
    n_real = 1 if clean else n_monte_carlo
    acc_without: Dict[str, List[float]] = {e: [] for e in els}
    acc_with: Dict[str, List[float]] = {e: [] for e in els}

    for _ in range(n_real):
        if clean:
            spec = clean_spectrum(fwd, comp, els, T_K, ne_cm3)
        else:
            shots = [noisy_shot(fwd, comp, els, T_K, ne_cm3, noise, rng) for _ in range(n_shots)]
            spec = np.mean(shots, axis=0)

        # WITHOUT OPC: calibration-free default path (recover T from slope).
        obs0 = extract_line_intensities(wl, spec, line_specs, instrument_fwhm_nm=instrument_fwhm_nm)
        try:
            res0 = run_constrained_solver(db, obs0, ne_cm3)
            p0 = recovered_wt(res0)
        except Exception:
            p0 = {}
        for e in els:
            acc_without[e].append(float(p0.get(e, np.nan)))

        # WITH OPC: rescale intensities by F, solve at the robust fixed T.
        obs1 = extract_line_intensities(wl, spec, line_specs, instrument_fwhm_nm=instrument_fwhm_nm)
        apply_opc(obs1, calibration)
        try:
            res1 = run_constrained_solver(
                db, obs1, ne_cm3, fixed_temperature_K=float(calibration.robust_T_K)
            )
            p1 = recovered_wt(res1)
        except Exception:
            p1 = {}
        for e in els:
            acc_with[e].append(float(p1.get(e, np.nan)))

    pred_without = {e: float(np.nanmean(v)) if v else np.nan for e, v in acc_without.items()}
    pred_with = {e: float(np.nanmean(v)) if v else np.nan for e, v in acc_with.items()}
    return pred_without, pred_with


# --- ratio-tracking metrics ---------------------------------------------------


def _ratio(comp: Dict[str, float], num: str, den: str) -> float:
    d = comp.get(den, np.nan)
    if not np.isfinite(d) or d <= 0:
        return np.nan
    return float(comp.get(num, np.nan)) / d


def _track_metrics(true_r: np.ndarray, rec_r: np.ndarray) -> Dict[str, float]:
    """Drift-tracking metrics of recovered ratio vs true ratio.

    Returns mean relative error of the ratio, plus the slope and R^2 of the
    linear fit recovered = slope*true + intercept (slope~1, R^2~1 => good
    drift tracking).
    """
    m = np.isfinite(true_r) & np.isfinite(rec_r) & (true_r != 0)
    t, r = true_r[m], rec_r[m]
    if t.size < 2:
        return {"relerr": np.nan, "slope": np.nan, "r2": np.nan, "n": float(t.size)}
    relerr = float(np.mean(np.abs(r - t) / np.abs(t)))
    if np.ptp(t) == 0:
        slope, r2 = np.nan, np.nan
    else:
        slope, intercept = np.polyfit(t, r, 1)
        pred = slope * t + intercept
        ss_res = float(np.sum((r - pred) ** 2))
        ss_tot = float(np.sum((r - np.mean(r)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        slope = float(slope)
    return {"relerr": relerr, "slope": float(slope), "r2": float(r2), "n": float(t.size)}


# --- top-level evaluation -----------------------------------------------------


def evaluate(
    db_path: str = DB_PATH,
    *,
    T_K: float = 11000.0,
    ne_cm3: float = 1e17,
    instrument_fwhm_nm: float = 0.1,
    grid_step_nm: float = 0.02,
    clean: bool = True,
    n_monte_carlo: int = 5,
    n_shots: int = 1,
    seed: int = 0,
) -> Dict[str, object]:
    """Evaluate WITH vs WITHOUT known-matrix OPC on the Ti-6Al-4V drift series.

    Tracks V/Ti on the V drift axis and Al/Ti on the Al drift axis (each ratio
    is measured on the axis where it actually drifts). Returns a dict with the
    OPC calibration provenance and the with/without tracking metrics.
    """
    db = AtomicDatabase(db_path)
    els = elements_of(ALLOY)
    wl = default_grid(ALLOY_WINDOWS_NM[ALLOY], grid_step_nm)
    fwd = make_forward(db_path, els, wl, instrument_fwhm_nm)
    line_specs = [s for v in build_alloy_line_list(db, ALLOY, T_K=T_K).values() for s in v]

    # 1) Build the matrix-matched standard from the nominal feedstock and calibrate.
    standard = _build_standard(
        db, fwd, wl, line_specs, els, T_K=T_K, ne_cm3=ne_cm3, instrument_fwhm_nm=instrument_fwhm_nm
    )
    calibration = calibrate_opc([standard])

    noise = None if clean else DEDNoiseParams()
    rng = None if clean else np.random.default_rng(seed)

    # 2) Walk each drift axis; collect true + recovered ratios (with/without OPC).
    axis_ratio = {"V": ("V", "Ti"), "Al": ("Al", "Ti")}
    per_axis: Dict[str, Dict[str, np.ndarray]] = {}
    for axis, (num, den) in axis_ratio.items():
        true_r, rec0_r, rec1_r = [], [], []
        for comp in COMPOSITION_SERIES[ALLOY][axis]:
            pred0, pred1 = _recover_point(
                db,
                fwd,
                wl,
                line_specs,
                els,
                comp,
                T_K=T_K,
                ne_cm3=ne_cm3,
                instrument_fwhm_nm=instrument_fwhm_nm,
                calibration=calibration,
                clean=clean,
                noise=noise,
                rng=rng,
                n_monte_carlo=n_monte_carlo,
                n_shots=n_shots,
            )
            true_r.append(_ratio(comp, num, den))
            rec0_r.append(_ratio(pred0, num, den))
            rec1_r.append(_ratio(pred1, num, den))
        per_axis[axis] = {
            "true": np.asarray(true_r),
            "without": np.asarray(rec0_r),
            "with": np.asarray(rec1_r),
        }

    vti_without = _track_metrics(per_axis["V"]["true"], per_axis["V"]["without"])
    vti_with = _track_metrics(per_axis["V"]["true"], per_axis["V"]["with"])
    alti_without = _track_metrics(per_axis["Al"]["true"], per_axis["Al"]["without"])
    alti_with = _track_metrics(per_axis["Al"]["true"], per_axis["Al"]["with"])

    return {
        "clean": clean,
        "calibration": {
            "robust_T_K": calibration.robust_T_K,
            "F": dict(calibration.F),
            "selected_standards": list(calibration.selected_standards),
        },
        "vti_without": vti_without,
        "vti_with": vti_with,
        "alti_without": alti_without,
        "alti_with": alti_with,
        "per_axis": {a: {k: v.tolist() for k, v in d.items()} for a, d in per_axis.items()},
    }


def _print_report(res: Dict[str, object]) -> None:
    cal = res["calibration"]
    tag = "CLEAN" if res["clean"] else "NOISY"
    print(f"\n=== DED Ti-6Al-4V known-matrix OPC eval ({tag}) ===")
    print(f"  robust_T_K = {cal['robust_T_K']:.0f} K   selected = {cal['selected_standards']}")
    print("  F = {" + ", ".join(f"{k}:{v:.3f}" for k, v in cal["F"].items()) + "}")
    for name, lbl in (("vti", "V/Ti (V axis)"), ("alti", "Al/Ti (Al axis)")):
        w0 = res[f"{name}_without"]
        w1 = res[f"{name}_with"]
        print(f"  {lbl}:")
        print(f"    WITHOUT  relerr={w0['relerr']:.4f}  slope={w0['slope']:.3f}  r2={w0['r2']:.4f}")
        print(f"    WITH     relerr={w1['relerr']:.4f}  slope={w1['slope']:.3f}  r2={w1['r2']:.4f}")


def main() -> Dict[str, object]:
    clean_res = evaluate(clean=True)
    _print_report(clean_res)
    out = {"clean": clean_res}
    try:
        noisy_res = evaluate(clean=False, n_monte_carlo=5)
        _print_report(noisy_res)
        out["noisy"] = noisy_res
    except Exception as exc:  # noisy run is best-effort
        print(f"\n[noisy run skipped: {type(exc).__name__}: {str(exc)[:120]}]")
    return out


if __name__ == "__main__":
    main()
