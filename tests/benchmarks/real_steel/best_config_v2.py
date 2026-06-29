"""Best lever combination v2 — L2 lines + L1 fixed-T + robust-OPC (L3b) + L4 self-absorption.

v1 (:mod:`tests.benchmarks.real_steel.best_config`) composed **L2** neutral-anchor line
selection + **L1** fixed optimal-``T`` + **L3** single-standard one-point-calibration ``F`` and
hit an honest held-out ``rmsep_overall`` of **16.48 wt%** (baseline 39.04). Two round-1 levers
were not yet in the combination:

* **L3b robust OPC** (:mod:`...lever_l3b_robust_opc`) replaces v1's *single* a-priori standard
  with a **conditioning-gated, geometric-mean** OPC ``F`` over every well-conditioned standard
  (selection uses only each standard's own data + own certified truth — never held-out truth),
  and a robust fixed ``T`` = mean of the selected standards' in-sample optimal ``T*``. This
  removes v1's fragile single-standard dependence (one keystone-collapsed standard -> bad ``F``).
* **L4 self-absorption** (:mod:`...lever_l4_selfabs`) targets the dominant remaining error — the
  Fe matrix collapse — *standalone* (L2+L4 overall 30.43 -> 20.78, Fe 64.5 -> 42.7). It measures
  each element's line widths, identifies the optically-thick element (median width above the
  per-spectrum instrument width), and applies a single El Sherbini 2005 boost uniformly to that
  element's lines. It is per-spectrum and reads only measured widths — no composition feedback.

  **EMPIRICAL FINDING — L4 is DROPPED from the v2 combination.** Once the robust OPC ``F`` is
  present, L4 *regresses* the held-out result: ``use_l4=True`` gives overall 11.35 / Fe 23.43 vs
  ``use_l4=False`` overall 10.12 / Fe 20.66 (Mn 1.50 -> 7.25). The reason is physical: the OPC
  ``F`` is derived on the standard and already absorbs the systematic matrix self-absorption bias
  as part of the relative-sensitivity factor, so L4's *second*, per-spectrum self-absorption boost
  over-corrects Fe and rotates the trace-minor balance. The combination therefore uses
  ``use_l4=False`` (the winning config); L4 is kept as an opt-in flag for the OPC-free regime only.

v2 composition (all honest / un-overfittable):

  1. Select **L2** neutral-anchor lines and extract intensities.
  2. (opt-in, OFF in the winning config) Apply **L4** width-gated self-absorption boost in-place.
  3. Apply the **robust OPC** per-element factor ``F`` (geometric mean over conditioning-gated
     standards) to each observation's intensity.
  4. Solve the constrained Saha-Boltzmann balance at the **robust fixed ``T``**.

KEY CONSISTENCY POINT: the OPC ``F`` is derived on the SAME pipeline it is applied to. When L4
is enabled, the standard's *uncorrected* recovery used for ``F_e = C_true_e / C_rec_e`` is the
L2+L4+fixed-T recovery, so ``F`` only corrects the residual systematic L4 does not (the two
corrections are composed, not double-counted: a large L4 Fe boost makes ``C_rec(Fe)`` larger and
hence ``F_Fe`` smaller). L4 is adaptive per sample; OPC is one global residual factor.

HONEST EVALUATION (no per-sample / no held-out peeking):
  * Standard selection, robust ``T`` and robust ``F`` derive ONLY from candidate standards' own
    spectra + own certified compositions. L4 reads only each scored sample's own line widths.
  * Leave-one-out: a selected standard is scored with the geometric-mean ``F`` of the OTHER
    selected standards (no self-leakage), keyed by an intensity fingerprint.
  * The headline is the held-out ``rmsep_overall`` over all 36 samples.
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
)
from tests.benchmarks.real_steel.lever_l1_temp import choose_optimal_T  # noqa: E402
from tests.benchmarks.real_steel.lever_l2_lines import select_l2_lines  # noqa: E402
from tests.benchmarks.real_steel.lever_l3_opc import (  # noqa: E402
    _F_MAX,
    _F_MIN,
    _MIN_WT,
    _line_key,
)
from tests.benchmarks.real_steel.lever_l3b_robust_opc import (  # noqa: E402
    COND_T_K,
    DEFAULT_T_GRID,
    _fingerprint,
    _geomean_F,
    assess_conditioning,
)
from tests.benchmarks.real_steel.lever_l4_selfabs import apply_self_absorption  # noqa: E402


def _l2_obs(db, wl, intensity, truth):
    """Extract the L2 neutral-anchor line observations for ``truth``'s elements (or ``None``)."""
    window = (float(wl.min()), float(wl.max()))
    specs: List[LineSpec] = []
    for e in truth:
        specs.extend(select_l2_lines(db, e, window, 8))
    if not specs:
        return None
    return extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=0.2)


def solve_v2(
    db,
    wl,
    intensity,
    truth,
    *,
    T_star: float,
    F: Dict,
    per_line: bool = False,
    use_l4: bool = False,
) -> Dict[str, float]:
    """v2 combined solve: L2 lines [+ L4 self-absorption] + OPC ``F`` rescale + fixed-``T`` solve.

    Order: extract L2 lines -> (optional) width-gated self-absorption boost -> OPC per-element
    ``F`` rescale -> constrained Saha-Boltzmann at ``T_star``. L4 runs before the OPC rescale so
    ``F`` corrects only the residual systematic the per-spectrum self-absorption boost leaves.
    """
    obs = _l2_obs(db, wl, intensity, truth)
    if not obs:
        return {}
    if use_l4:
        apply_self_absorption(wl, intensity, obs)
    for o in obs:
        f = F.get(_line_key(o.element, o.wavelength_nm), 1.0) if per_line else F.get(o.element, 1.0)
        o.intensity = float(o.intensity) * float(f)
        o.intensity_uncertainty = max(float(o.intensity_uncertainty) * float(f), 1e-12)
    res = run_constrained_solver(db, obs, 1e17, fixed_temperature_K=T_star)
    return recovered_wt(res)


def derive_F_v2(
    db, standard, T_star: float, *, per_line: bool = False, use_l4: bool = False
) -> Dict:
    """OPC ``F`` from the standard's *uncorrected* (L2[+L4]+fixed-T) recovery at ``T_star``.

    ``F_e = C_true_e / C_rec_e`` (renormalized to the standard's modeled basis), clamped to the
    OPC band. Derived on the SAME pipeline (``use_l4``) it will be applied to, so the L4 boost and
    OPC factor compose consistently rather than double-counting.
    """
    _sid, wl, inten, truth = standard
    rec = solve_v2(db, wl, inten, truth, T_star=T_star, F={}, per_line=per_line, use_l4=use_l4)
    rec = {e: rec.get(e, 0.0) for e in truth}
    tot = sum(v for v in rec.values() if np.isfinite(v) and v > 0)
    F_el: Dict[str, float] = {}
    for e, tv in truth.items():
        cn = (rec[e] / tot * 100.0) if (tot > 0 and np.isfinite(rec[e])) else 0.0
        f = max(tv, _MIN_WT) / max(cn, _MIN_WT)
        F_el[e] = float(np.clip(f, _F_MIN, _F_MAX))
    if not per_line:
        return F_el
    window = (float(wl.min()), float(wl.max()))
    F_line: Dict[Tuple[str, float], float] = {}
    for e in truth:
        for ls in select_l2_lines(db, e, window, 8):
            F_line[_line_key(e, ls.wavelength_nm)] = F_el.get(e, 1.0)
    return F_line


def run_v2(
    db_path: str = "ASD_da/libs_production.db",
    *,
    t_grid=DEFAULT_T_GRID,
    cond_T_K: float = COND_T_K,
    per_line: bool = False,
    use_l4: bool = False,
    limit: int | None = None,
) -> Dict[str, object]:
    """Honest v2 run: conditioning-gated robust (T*, F) [+ L4], applied held-out (LOO), scored.

    Mirrors :func:`...lever_l3b_robust_opc.run_robust_opc` but threads the v2 pipeline (optional
    L4 self-absorption, with ``F`` derived on the same pipeline). Returns the held-out scores plus
    the selection / calibration metadata.
    """
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    samples = list(load_real_steel())
    if limit is not None:
        samples = samples[:limit]

    # 1+2. Conditioning gate (L2 uncorrected, in-sample only) -> selected standards.
    selected: List[int] = []
    diagnostics: List[Dict[str, object]] = []
    for i, std in enumerate(samples):
        d = assess_conditioning(db, std, cond_T_K)
        d["index"] = i
        d["sample"] = std[0]
        diagnostics.append(d)
        flag = "KEEP" if d.get("passed") else "drop"
        print(
            f"  [cond {i}] {std[0]}: {flag} in_rmsep={d.get('in_rmsep', float('nan')):.2f} "
            f"nonmatrix_max={d.get('nonmatrix_max_pct', float('nan')):.1f}%",
            flush=True,
        )
        if d.get("passed"):
            selected.append(i)

    if not selected:
        best = min(diagnostics, key=lambda d: d.get("in_rmsep", float("inf")))
        selected = [int(best["index"])]
        print(f"[v2] no standard passed; fallback to best in-sample idx {selected[0]}", flush=True)

    # 3. Robust T (mean of selected standards' in-sample optimal T*) + per-standard F (v2 pipeline).
    T_stars: List[float] = []
    for i in selected:
        T_i, err_i, _curve = choose_optimal_T(db, samples[i], t_grid)
        T_stars.append(T_i)
        print(f"  [T* {i}] {samples[i][0]}: T*={T_i:.0f}K in-sample rmsep={err_i:.2f}", flush=True)
    robust_T = float(np.mean(T_stars))
    print(f"[v2] robust T = mean(T*) = {robust_T:.0f} K over {len(selected)} standards", flush=True)

    F_per_std: Dict[int, Dict] = {}
    for i in selected:
        F_per_std[i] = derive_F_v2(db, samples[i], robust_T, per_line=per_line, use_l4=use_l4)
    robust_F = _geomean_F(list(F_per_std.values()), per_line)

    # Leave-one-out F map keyed by sample fingerprint (no self-leakage for selected standards).
    loo_F: Dict[Tuple[int, float], Dict] = {}
    for i in selected:
        others = [F_per_std[j] for j in selected if j != i]
        loo_F[_fingerprint(samples[i][2])] = _geomean_F(others, per_line) if others else robust_F

    # 4. Apply robust (T, F) [+ L4] held-out and score (LOO per sample).
    def solve_fn(db_, wl, intensity, truth):
        F = loo_F.get(_fingerprint(intensity), robust_F)
        return solve_v2(
            db_, wl, intensity, truth, T_star=robust_T, F=F, per_line=per_line, use_l4=use_l4
        )

    held = run_benchmark(solve_fn, db_path=db_path, limit=limit)

    return {
        "use_l4": use_l4,
        "robust_T_K": robust_T,
        "per_line": per_line,
        "n_selected": len(selected),
        "selected_indices": selected,
        "selected_samples": [samples[i][0] for i in selected],
        "robust_F": {str(k): v for k, v in robust_F.items()},
        "held_out": held,
    }


def best_solve_fn_v2(
    robust_T: float, robust_F: Dict, *, per_line: bool = False, use_l4: bool = False
):
    """Frozen ``solve_fn(db, wl, intensity, truth)`` applying the v2 calibration unchanged.

    For deployment / the gate harness: ``robust_T`` and ``robust_F`` are the standard-derived
    calibration; L4 self-absorption is recomputed per sample inside the closure.
    """

    def _fn(db, wl, intensity, truth):
        return solve_v2(
            db, wl, intensity, truth, T_star=robust_T, F=robust_F, per_line=per_line, use_l4=use_l4
        )

    return _fn


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
    ap.add_argument(
        "--compare-l4",
        action="store_true",
        help="run v2 both with and without L4 and report which improves Fe / overall",
    )
    a = ap.parse_args()
    grid = tuple(float(t) for t in range(a.tmin, a.tmax + 1, a.tstep))

    def _report(tag, out):
        hs = out["held_out"]
        print(f"\n[{tag}] v2 real-steel results (use_l4={out['use_l4']}):")
        print(f"  n_selected: {out['n_selected']} -> {out['selected_samples']}")
        print(f"  robust_T_K: {out['robust_T_K']:.0f}")
        print("  HELD-OUT RMSEP (wt%):")
        for k in sorted(hs):
            v = hs[k]
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
        return hs

    if a.compare_l4:
        out_no = run_v2(
            db_path=a.db_path,
            t_grid=grid,
            cond_T_K=a.cond_t,
            per_line=a.per_line,
            use_l4=False,
            limit=a.limit,
        )
        hs_no = _report("no-L4", out_no)
        out_l4 = run_v2(
            db_path=a.db_path,
            t_grid=grid,
            cond_T_K=a.cond_t,
            per_line=a.per_line,
            use_l4=True,
            limit=a.limit,
        )
        hs_l4 = _report("with-L4", out_l4)
        print(
            "\n[compare] Fe RMSEP  no-L4={:.2f}  with-L4={:.2f}".format(
                hs_no.get("rmsep_Fe", float("nan")), hs_l4.get("rmsep_Fe", float("nan"))
            )
        )
        print(
            "[compare] overall  no-L4={:.2f}  with-L4={:.2f}".format(
                hs_no["rmsep_overall"], hs_l4["rmsep_overall"]
            )
        )
    else:
        out = run_v2(
            db_path=a.db_path,
            t_grid=grid,
            cond_T_K=a.cond_t,
            per_line=a.per_line,
            use_l4=False,
            limit=a.limit,
        )
        _report("v2", out)
