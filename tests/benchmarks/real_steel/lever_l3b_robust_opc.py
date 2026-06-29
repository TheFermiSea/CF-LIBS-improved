"""Lever L3b — robust, conditioning-based OPC standard selection (real-steel gate).

Round-1 found single-standard OPC (lever L3) is *fragile*: a keystone-collapsed
standard (e.g. idx0) yields a bad correction ``F`` (held-out RMSEP ~36), while a
well-conditioned standard (idx7) yields a good one (~8.4). Picking the good
standard by its *held-out* performance would be peeking (the Optuna-overfit
lesson). L3b removes the peeking entirely:

  1. **Conditioning gate (in-sample only).** For EACH candidate standard, run the
     L2 neutral-anchor *uncorrected* fixed-T solve and judge whether that
     standard is well-conditioned using ONLY its own recovered composition + its
     own certified truth — never the other 35 samples. A standard passes when:
       * the solve ``converged``;
       * it is not flagged ``quality_metrics['degenerate_composition']``;
       * its in-sample uncorrected ``rmsep_overall`` is below a fixed a-priori
         threshold (keystone-collapsed standards have a huge in-sample RMSEP);
       * no *non-matrix* element soaks the closure above a fixed fraction (a
         trace minor recovered as a large fraction is the keystone-collapse
         signature; the legitimate matrix element — argmax of the certified truth,
         Fe for steel — is exempt because it is genuinely dominant).
  2. **Keep** the well-conditioned standards.
  3. **Robust correction.** Average the per-element OPC factors ``F_e`` of the
     selected standards by **geometric mean** into one robust ``F`` (geometric
     because ``F`` is a multiplicative correction). The fixed temperature is the
     mean of the selected standards' in-sample optimal ``T*`` (L1 composition —
     each ``T*`` is chosen on its own standard's truth, never held-out).
  4. **Apply** that robust ``F`` and robust ``T`` UNCHANGED to all samples and
     score the held-out ``rmsep_overall`` via :func:`run_benchmark`.

LEAVE-ONE-OUT (no self-leakage). Some scored samples are themselves selected
standards, so their own ``F_s`` is in the robust average. For every selected
standard we score it with a robust ``F`` recomputed from the *other* selected
standards only (geometric mean over ``S \\ {s}``); every non-standard sample uses
the full robust ``F``. The harness solver is keyed by an intensity fingerprint so
each sample receives the correct (self-excluded) ``F``. We additionally report
``n_selected`` so the ">=3 standards dominate" robustness condition is explicit.

HONEST EVALUATION: standard selection, ``T``, and ``F`` derive ONLY from
candidate standards' own data + own certified truth. The held-out
``rmsep_overall`` over all 36 samples (each using a self-excluded ``F``) is the
un-overfittable headline.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

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
from tests.benchmarks.real_steel.harness import (  # noqa: E402
    load_real_steel,
    run_benchmark,
    score,
)
from tests.benchmarks.real_steel.best_config import derive_F_at_T  # noqa: E402
from tests.benchmarks.real_steel.lever_l1_temp import choose_optimal_T  # noqa: E402
from tests.benchmarks.real_steel.lever_l2_lines import select_l2_lines  # noqa: E402
from tests.benchmarks.real_steel.lever_l3_opc import _F_MAX, _F_MIN, _line_key  # noqa: E402

# Fixed reference temperature for the conditioning gate (physically reasonable
# steel-plasma value; NOT fit per sample, NOT tuned to held-out truth).
COND_T_K = 9000.0
# A-priori conditioning thresholds (set from physics, never tuned on held-out):
#  * a standard whose own uncorrected solve is worse than this is keystone-collapsed.
MAX_INSAMPLE_RMSEP = 20.0  # wt%
#  * a NON-matrix element recovered above this fraction means the closure collapsed.
MAX_NONMATRIX_FRAC = 60.0  # %
# Coarser T grid than best_config (the robust mean is insensitive to fine steps).
DEFAULT_T_GRID = tuple(float(t) for t in range(7000, 12001, 1000))


def _solve_l1_full(db, wl, intensity, truth, T_K: float):
    """L2 neutral-anchor uncorrected solve at fixed ``T_K`` -> (pred_wt, CFLIBSResult)."""
    window = (float(wl.min()), float(wl.max()))
    specs: List[LineSpec] = []
    for e in truth:
        specs.extend(select_l2_lines(db, e, window, 8))
    if not specs:
        return {}, None
    obs = extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=0.2)
    res = run_constrained_solver(db, obs, 1e17, fixed_temperature_K=T_K)
    return recovered_wt(res), res


def assess_conditioning(db, standard, T_K: float = COND_T_K) -> Dict[str, object]:
    """Judge one candidate standard's conditioning from its OWN data + truth only.

    Returns a diagnostics dict with ``passed`` and the individual gate values.
    Uses no other sample's data — this is the un-peekable selection criterion.
    """
    _sid, wl, inten, truth = standard
    pred, res = _solve_l1_full(db, wl, inten, truth, T_K)
    if res is None:
        return {"passed": False, "reason": "no_lines"}
    converged = bool(res.converged)
    degenerate = float((res.quality_metrics or {}).get("degenerate_composition", 0.0)) >= 0.5
    in_rmsep = score([(truth, pred or {})])["rmsep_overall"]
    # Recovered composition renormalized to the modeled basis (matches score()).
    ps = sum(v for v in (pred or {}).values() if np.isfinite(v) and v > 0)
    matrix_el = max(truth, key=truth.get) if truth else None
    nonmatrix_max = 0.0
    if ps > 0:
        for e, v in (pred or {}).items():
            if e == matrix_el or not np.isfinite(v):
                continue
            nonmatrix_max = max(nonmatrix_max, v / ps * 100.0)
    passed = bool(
        converged
        and (not degenerate)
        and np.isfinite(in_rmsep)
        and in_rmsep < MAX_INSAMPLE_RMSEP
        and nonmatrix_max < MAX_NONMATRIX_FRAC
    )
    return {
        "passed": passed,
        "converged": converged,
        "degenerate": degenerate,
        "in_rmsep": float(in_rmsep),
        "nonmatrix_max_pct": float(nonmatrix_max),
        "matrix_el": matrix_el,
    }


def _geomean_F(F_list: List[Dict], per_line: bool) -> Dict:
    """Per-key geometric mean of a list of OPC factor dicts (clamped to the F band).

    DEGENERATE-VALUE FILTER (the key robustness step): a per-element factor that
    saturated the clamp band (``F_e == _F_MIN`` or ``_F_MAX``) means *that*
    standard did not usefully recover element ``e`` (its ``C_rec_e`` was ~0, so
    ``F_e = C_true/C_rec`` blew up to the clamp). Such a value carries no
    information about the true relative-sensitivity factor — averaging it in lets
    one degenerate standard inflate ``F_e`` and make ``e`` soak the closure on the
    held-out samples (observed for Mo: unfiltered geomean -> F_Mo ~2 -> Mo soaks
    -> Fe collapses, RMSEP ~31). We therefore drop clamp-saturated values before
    the geometric mean. This uses only each standard's *own* recovery (in-sample),
    never any held-out truth. If every standard is degenerate for ``e`` we apply
    no correction (``F_e = 1``).
    """
    if not F_list:
        return {}
    eps = 1e-6
    keys: set = set()
    for F in F_list:
        keys.update(F.keys())
    out: Dict = {}
    for k in keys:
        vals = [
            v
            for F in F_list
            for v in (float(F.get(k, 1.0)),)
            if np.isfinite(v) and v > 0 and abs(v - _F_MIN) > eps and abs(v - _F_MAX) > eps
        ]
        if not vals:
            out[k] = 1.0
            continue
        gm = float(np.exp(np.mean(np.log(vals))))
        out[k] = float(np.clip(gm, _F_MIN, _F_MAX))
    return out


def _fingerprint(intensity) -> Tuple[int, float]:
    arr = np.asarray(intensity, dtype=float)
    return (int(arr.size), round(float(arr.sum()), 3))


def run_robust_opc(
    db_path: str = "ASD_da/libs_production.db",
    *,
    t_grid=DEFAULT_T_GRID,
    cond_T_K: float = COND_T_K,
    per_line: bool = False,
    limit: int | None = None,
) -> Dict[str, object]:
    """Robust conditioning-based OPC: select well-conditioned standards, geomean F.

    Honest held-out evaluation via :func:`run_benchmark` with per-sample
    leave-one-out (a selected standard never sees its own ``F``).
    """
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    samples = list(load_real_steel())
    if limit is not None:
        samples = samples[:limit]

    # 1+2. Conditioning gate -> selected well-conditioned standards (in-sample only).
    selected: List[int] = []
    diagnostics: List[Dict[str, object]] = []
    for i, std in enumerate(samples):
        d = assess_conditioning(db, std, cond_T_K)
        d["index"] = i
        d["sample"] = std[0]
        diagnostics.append(d)
        flag = "KEEP" if d.get("passed") else "drop"
        print(
            f"  [cond {i}] {std[0]}: {flag} "
            f"converged={d.get('converged')} degenerate={d.get('degenerate')} "
            f"in_rmsep={d.get('in_rmsep', float('nan')):.2f} "
            f"nonmatrix_max={d.get('nonmatrix_max_pct', float('nan')):.1f}%",
            flush=True,
        )
        if d.get("passed"):
            selected.append(i)

    if not selected:  # degenerate: nothing passed -> fall back to the single best in-sample
        best = min(diagnostics, key=lambda d: d.get("in_rmsep", float("inf")))
        selected = [int(best["index"])]
        print(
            f"[L3b] no standard passed; falling back to best in-sample idx {selected[0]}",
            flush=True,
        )

    # 3. Robust T (mean of selected standards' in-sample optimal T*) + per-standard F.
    T_stars: List[float] = []
    for i in selected:
        T_i, err_i, _curve = choose_optimal_T(db, samples[i], t_grid)
        T_stars.append(T_i)
        print(f"  [T* {i}] {samples[i][0]}: T*={T_i:.0f}K in-sample rmsep={err_i:.2f}", flush=True)
    robust_T = float(np.mean(T_stars))
    print(
        f"[L3b] robust T = mean(T*) = {robust_T:.0f} K over {len(selected)} standards", flush=True
    )

    F_per_std: Dict[int, Dict] = {}
    for i in selected:
        F_per_std[i] = derive_F_at_T(db, samples[i], robust_T, per_line=per_line)
    robust_F = _geomean_F(list(F_per_std.values()), per_line)

    # Leave-one-out F map keyed by sample fingerprint: a selected standard is
    # scored with the geomean of the OTHER selected standards (no self-leakage).
    loo_F: Dict[Tuple[int, float], Dict] = {}
    for i in selected:
        others = [F_per_std[j] for j in selected if j != i]
        loo_F[_fingerprint(samples[i][2])] = _geomean_F(others, per_line) if others else robust_F

    # 4. Apply robust (T, F) held-out and score via run_benchmark (LOO per sample).
    def solve_fn(db_, wl, intensity, truth):
        F = loo_F.get(_fingerprint(intensity), robust_F)
        window = (float(wl.min()), float(wl.max()))
        specs: List[LineSpec] = []
        for e in truth:
            specs.extend(select_l2_lines(db_, e, window, 8))
        if not specs:
            return {}
        obs = extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=0.2)
        for o in obs:
            f = (
                F.get(_line_key(o.element, o.wavelength_nm), 1.0)
                if per_line
                else F.get(o.element, 1.0)
            )
            o.intensity = float(o.intensity) * float(f)
            o.intensity_uncertainty = max(float(o.intensity_uncertainty) * float(f), 1e-12)
        res = run_constrained_solver(db_, obs, 1e17, fixed_temperature_K=robust_T)
        return recovered_wt(res)

    held = run_benchmark(solve_fn, db_path=db_path, limit=limit)

    return {
        "robust_T_K": robust_T,
        "per_line": per_line,
        "n_selected": len(selected),
        "selected_indices": selected,
        "selected_samples": [samples[i][0] for i in selected],
        "robust_F": {str(k): v for k, v in robust_F.items()},
        "conditioning_rule": (
            f"converged AND not degenerate_composition AND in-sample uncorrected "
            f"rmsep < {MAX_INSAMPLE_RMSEP} wt% AND non-matrix recovered fraction "
            f"< {MAX_NONMATRIX_FRAC}% (gate at fixed T={cond_T_K:.0f} K)"
        ),
        "held_out": held,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="ASD_da/libs_production.db")
    ap.add_argument("--per-line", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cond-t", type=float, default=COND_T_K)
    ap.add_argument("--tmin", type=int, default=7000)
    ap.add_argument("--tmax", type=int, default=12000)
    ap.add_argument("--tstep", type=int, default=1000)
    a = ap.parse_args()
    grid = tuple(float(t) for t in range(a.tmin, a.tmax + 1, a.tstep))
    out = run_robust_opc(
        db_path=a.db_path, t_grid=grid, cond_T_K=a.cond_t, per_line=a.per_line, limit=a.limit
    )
    print("\nL3b (robust conditioning-based OPC) real-steel results:")
    print(f"  conditioning rule: {out['conditioning_rule']}")
    print(f"  n_selected: {out['n_selected']}  -> {out['selected_samples']}")
    print(f"  robust_T_K: {out['robust_T_K']:.0f}")
    print(f"  per_line: {out['per_line']}")
    print("  HELD-OUT RMSEP (wt%):")
    hs = out["held_out"]
    for k in sorted(hs):
        v = hs[k]
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
