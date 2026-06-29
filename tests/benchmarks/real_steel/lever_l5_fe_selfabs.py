"""Lever L5 — per-spectrum, per-line Fe self-absorption *inside* the OPC solve.

Motivation (docs/research/real-steel-opc-promotion.md "Open items"): in the v2
winning config (L2 neutral-anchor lines + L1 fixed-``T`` + robust OPC ``F``) Fe is
still the dominant held-out residual (Fe RMSEP 20.66, overall 10.12). The cause is
physical: real steel has **variable Fe content (85-95 wt%)**, so the Fe matrix
lines are optically thick to a *different degree in every spectrum*. A single
shared OPC ``F_Fe`` is a global scalar — it captures the *average* matrix
self-absorption bias but cannot adapt to the per-spectrum variation.

Why standalone L4 + OPC double-corrects (and how L5 avoids it)
-------------------------------------------------------------
L4 (:mod:`...lever_l4_selfabs`) applies a **uniform** per-element boost (one factor
for all of Fe's lines) precisely so it does not rotate the *fitted* Boltzmann slope
and corrupt the recovered ``T``. But a uniform per-element scalar is **degenerate
with the OPC ``F_Fe`` scalar**: re-deriving ``F`` on the L4 pipeline only shrinks
``F_Fe`` to compensate, while L4's noisy per-spectrum *magnitude* leaks into the
trace-minor balance (v2: use_l4=True regresses overall 10.12 -> 11.35, Mn 1.50 ->
7.25). The two corrections live in the same (scalar) subspace, so stacking them
adds noise, not signal.

L5's key unlock: **in the OPC regime the temperature is FIXED** (``robust_T``),
so there is no Boltzmann slope-fit to corrupt. That removes the exact constraint
that forced L4 to be uniform. L5 can therefore apply a genuine **per-line**
self-absorption correction — each Fe line corrected by *its own* measured width —
which is NOT degenerate with the OPC scalar: it carries the per-line (and hence
per-spectrum) differential self-absorption that ``F_Fe`` structurally cannot.

Method (NotebookLM-grounded, notebook ``f1d2a053``; El Sherbini 2005,
Spectrochim. Acta B 60 1573; CD-SB / IRSAC reviews)
---------------------------------------------------
Per spectrum, observable-anchored (reads only measured line widths — never a
recovered composition):

1. Measure each L2-selected line's FWHM and peak SNR from the spectrum.
2. Instrument width ``Delta_lambda_instr`` = a low percentile of the well-measured
   widths (the optically-thin floor of the width distribution).
3. Identify each optically-thick element (>= ``min_corr_lines`` lines whose MEDIAN
   width exceeds ``thick_ratio * instr`` — Fe for steel). Thin elements (trace
   minors at the instrument width) are left uncorrected.
4. For each line of a thick element form the intrinsic (instrument-deconvolved)
   width ``Delta_lambda_intr = sqrt(FWHM^2 - instr^2)`` and the **per-line** El
   Sherbini self-absorption boost (``I_thin = I_obs / SA``,
   ``SA = (Delta_lambda / Delta_lambda_0) ** (1 / -0.54)``)::

       boost_i = clip((max(Delta_lambda_intr_i, ref) / ref) ** (1 / alpha), 1, max)

   with ``ref = Delta_lambda_0`` the optically-thin intrinsic reference width and
   ``alpha = 0.54``. The boost is applied to **each line individually** (not the
   element median), so broader (more self-absorbed) lines are boosted harder.

The OPC ``F`` is then re-derived on this *same* per-line-corrected pipeline (the
honest decoupling NotebookLM prescribes: correct self-absorption on both the
standard and the unknown first, THEN fit OPC ``F`` so ``F`` carries only the
static residual relative-sensitivity — inaccurate ``A_ki`` / detector efficiency —
not the dynamic per-spectrum self-absorption the physical correction already
removed). Composition is solved at the robust fixed ``T``.

HONEST EVALUATION (mirrors best_config_v2): conditioning-gated standard selection,
robust ``T`` = mean of selected standards' in-sample optimal ``T*``, robust ``F`` =
geomean over selected standards (clamp-saturated values filtered), per-sample
leave-one-out so a selected standard never sees its own ``F``. The headline is the
held-out ``rmsep_overall`` over all 36 samples. No parameter is tuned to held-out
truth; the SA constants are the same physical El Sherbini values L4 uses.

EMPIRICAL FINDING (full 36-sample honest held-out gate, robust T=10500 K)
------------------------------------------------------------------------
* ``mode="off"`` (v2 reproduction): overall 10.124, Fe 20.660.
* ``mode="per_line"`` (per-line intensity boost): overall **11.900**, Fe **24.387**
  -> REGRESSES. The "fixed T removes the slope constraint" hypothesis does not save
  it: re-deriving ``F`` only re-centers the *average* boost, while the per-line /
  per-spectrum boost *magnitude* still lives in the same scalar subspace as ``F_Fe``
  and its variance leaks into the trace-minor balance (exactly the L4+OPC failure).
* ``mode="thin_filter"`` (drop self-absorbed Fe lines, anchor on the optically-thin
  subset; IRSAC / SC-LIBS): overall **9.561**, Fe **19.615** -> IMPROVES (overall
  -0.56, Fe -1.05; Si 7.20->5.85, Mn 1.50->1.16, Mo 10.81->9.78). Because it is a
  line *selection* change — not an intensity scaling — it does not occupy the OPC
  scalar subspace and so does not double-correct. **This is the winning L5 mode.**

The lesson: against an already-OPC-calibrated pipeline, *correcting* matrix
self-absorption intensities double-counts ``F``; *excluding* the self-absorbed lines
(keeping an optically-thin anchor) is the per-spectrum lever that composes cleanly.

This is a benchmark-layer lever (needs the measured spectrum / line widths). The
winning ``thin_filter`` mode is pure NumPy and promotes into the shipped ``cflibs/``
OPC pre-solve step (physics-only; DED no-regression gated) as an opt-in known-matrix
line filter. No shipped change is made in this commit (benchmark layer only), so the
DED-precision gate is unaffected.
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
from tests.benchmarks.real_steel.lever_l4_selfabs import (  # noqa: E402
    measure_line_width,
)

# Physical El Sherbini 2005 self-absorption constants (identical to L4; NOT tuned
# to the held-out gate). SA = (dl/dl_0)^(1/-0.54), so boost = (dl_intr/dl_0)^(1/ALPHA_SA).
ALPHA_SA = 0.54
THIN_REF_WIDTH_NM = 0.08  # optically-thin intrinsic reference width Delta_lambda_0 (nm)
SA_BOOST_MAX = 20.0  # per-line clamp so a mis-measured line cannot run away
MIN_CORR_LINES = 3  # an element is corrected only with >= this many well-measured lines
SNR_MIN = 3.0  # peak-SNR floor for a line to count toward width statistics / correction
THICK_RATIO = 1.15  # element is "thick" only when median width / instr exceeds this
INSTR_WIDTH_PCT = 10.0  # percentile of well-measured widths taken as the instrument width


def apply_self_absorption_per_line(
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
    """In-place **per-line** width-based self-absorption boost of ``observations``.

    Differs from :func:`...lever_l4_selfabs.apply_self_absorption` only in that the
    El Sherbini boost is formed from *each line's own* intrinsic width rather than
    the element median, so the correction is differential across an element's lines.
    Safe here because the OPC solve fixes ``T`` (no Boltzmann slope to rotate).
    Element-thickness gating is identical to L4. Returns ``(n_lines_boosted, max_boost)``.
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

    n_boosted = 0
    max_boost = 1.0
    for _el, idxs in by_el.items():
        good = [i for i in idxs if np.isfinite(widths[i]) and snrs[i] >= snr_min]
        if len(good) < min_corr_lines:
            continue
        median_w = float(np.median([widths[i] for i in good]))
        if median_w / instr < thick_ratio:
            continue  # optically thin element -> no boost (keeps trace minors at 1)
        for i in idxs:
            # Per-line boost from this line's own intrinsic (deconvolved) width.
            # Un-measurable / low-SNR lines fall back to the element-median width so
            # they still receive the bulk matrix correction (never < 1).
            w = widths[i] if (np.isfinite(widths[i]) and snrs[i] >= snr_min) else median_w
            intrinsic = np.sqrt(max(w**2 - instr**2, 0.0))
            boost = float(np.clip((max(intrinsic, ref) / ref) ** (1.0 / alpha), 1.0, sa_boost_max))
            if boost <= 1.0:
                continue
            o = observations[i]
            o.intensity = float(o.intensity) * boost
            o.intensity_uncertainty = max(float(o.intensity_uncertainty) * boost, 1e-12)
            n_boosted += 1
            max_boost = max(max_boost, boost)
    return n_boosted, max_boost


def select_thin_lines(
    wl: np.ndarray,
    intensity: np.ndarray,
    observations: Sequence,
    *,
    min_corr_lines: int = MIN_CORR_LINES,
    snr_min: float = SNR_MIN,
    thick_ratio: float = THICK_RATIO,
    instr_width_pct: float = INSTR_WIDTH_PCT,
    min_keep: int = 2,
) -> List:
    """Drop the width-broadened (self-absorbed) lines of each optically-thick element.

    IRSAC / SC-LIBS anchor strategy (NotebookLM notebook ``f1d2a053``): instead of
    *correcting* the intensities of self-absorbed matrix (Fe) lines, **exclude** them
    and keep only the optically-thin subset (lines whose measured width is at the
    instrument floor). This is a line-*selection* change — not an intensity scaling —
    so it cannot live in the same scalar subspace as the OPC ``F_Fe`` and therefore
    cannot double-correct with it. Thin elements are untouched. An element is only
    filtered if at least ``min_keep`` thin lines remain (else its lines are kept as-is
    so a noisy spectrum never strips the matrix anchor entirely). Reads only measured
    widths — no composition feedback. Returns the kept ``observations`` subset.
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
        return list(observations)
    instr = float(np.percentile(good_all, instr_width_pct))
    if instr <= 0:
        return list(observations)

    drop: set = set()
    for _el, idxs in by_el.items():
        good = [i for i in idxs if np.isfinite(widths[i]) and snrs[i] >= snr_min]
        if len(good) < min_corr_lines:
            continue
        median_w = float(np.median([widths[i] for i in good]))
        if median_w / instr < thick_ratio:
            continue  # optically thin element -> keep all its lines
        thin = [i for i in good if widths[i] / instr < thick_ratio]
        if len(thin) < min_keep:
            continue  # too few thin anchors -> keep the element's lines as-is
        keep = set(thin)
        for i in idxs:
            if i not in keep:
                drop.add(i)
    return [o for i, o in enumerate(observations) if i not in drop]


def _l2_obs(db, wl, intensity, truth):
    window = (float(wl.min()), float(wl.max()))
    specs: List[LineSpec] = []
    for e in truth:
        specs.extend(select_l2_lines(db, e, window, 8))
    if not specs:
        return None
    return extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=0.2)


def solve_l5(
    db,
    wl,
    intensity,
    truth,
    *,
    T_star: float,
    F: Dict,
    per_line: bool = False,
    mode: str = "thin_filter",
) -> Dict[str, float]:
    """v2 solve with a per-spectrum Fe self-absorption handling before the OPC rescale.

    ``mode`` selects the self-absorption treatment of the optically-thick matrix:

    * ``"off"`` -- reproduce v2 (no per-spectrum self-absorption handling).
    * ``"per_line"`` -- per-line El Sherbini intensity boost (regresses on the gate;
      double-corrects with the OPC scalar -- kept for the record).
    * ``"thin_filter"`` -- drop the width-broadened (self-absorbed) matrix lines and
      anchor on the optically-thin subset (line selection, not intensity scaling).

    Order: extract L2 lines -> self-absorption handling (``mode``) -> OPC per-element
    ``F`` rescale -> constrained Saha-Boltzmann at fixed ``T_star``. The handling runs
    before the OPC rescale so ``F`` corrects only the residual static sensitivity.
    """
    obs = _l2_obs(db, wl, intensity, truth)
    if not obs:
        return {}
    if mode == "per_line":
        apply_self_absorption_per_line(wl, intensity, obs)
    elif mode == "thin_filter":
        obs = select_thin_lines(wl, intensity, obs)
        if not obs:
            return {}
    for o in obs:
        f = F.get(_line_key(o.element, o.wavelength_nm), 1.0) if per_line else F.get(o.element, 1.0)
        o.intensity = float(o.intensity) * float(f)
        o.intensity_uncertainty = max(float(o.intensity_uncertainty) * float(f), 1e-12)
    res = run_constrained_solver(db, obs, 1e17, fixed_temperature_K=T_star)
    return recovered_wt(res)


def derive_F_l5(
    db, standard, T_star: float, *, per_line: bool = False, mode: str = "thin_filter"
) -> Dict:
    """OPC ``F`` from the standard's self-absorption-handled (L2+mode+fixed-T) recovery.

    ``F_e = C_true_e / C_rec_e`` (renormalized), clamped to the OPC band, derived on
    the SAME pipeline (``mode``) it is applied to (so the handling + OPC compose).
    """
    _sid, wl, inten, truth = standard
    rec = solve_l5(db, wl, inten, truth, T_star=T_star, F={}, per_line=per_line, mode=mode)
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


def run_l5(
    db_path: str = "ASD_da/libs_production.db",
    *,
    t_grid=DEFAULT_T_GRID,
    cond_T_K: float = COND_T_K,
    per_line: bool = False,
    mode: str = "thin_filter",
    limit: int | None = None,
) -> Dict[str, object]:
    """Honest L5 run: conditioning-gated robust (T*, F) on the ``mode`` pipeline, LOO-scored.

    Mirrors :func:`...best_config_v2.run_v2` but threads the per-spectrum self-absorption
    handling (``mode``; ``F`` derived on the same pipeline). ``mode="off"`` reproduces v2.
    """
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    samples = list(load_real_steel())
    if limit is not None:
        samples = samples[:limit]

    # 1+2. Conditioning gate (uncorrected L2, in-sample only) -> selected standards.
    selected: List[int] = []
    diagnostics: List[Dict[str, object]] = []
    for i, std in enumerate(samples):
        d = assess_conditioning(db, std, cond_T_K)
        d["index"] = i
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
        print(f"[L5] no standard passed; fallback to best in-sample idx {selected[0]}", flush=True)

    # 3. Robust T + per-standard F on the selected self-absorption-handling pipeline.
    T_stars: List[float] = []
    for i in selected:
        T_i, err_i, _curve = choose_optimal_T(db, samples[i], t_grid)
        T_stars.append(T_i)
        print(f"  [T* {i}] {samples[i][0]}: T*={T_i:.0f}K in-sample rmsep={err_i:.2f}", flush=True)
    robust_T = float(np.mean(T_stars))
    print(f"[L5] robust T = mean(T*) = {robust_T:.0f} K over {len(selected)} standards", flush=True)

    F_per_std: Dict[int, Dict] = {}
    for i in selected:
        F_per_std[i] = derive_F_l5(db, samples[i], robust_T, per_line=per_line, mode=mode)
    robust_F = _geomean_F(list(F_per_std.values()), per_line)

    loo_F: Dict[Tuple[int, float], Dict] = {}
    for i in selected:
        others = [F_per_std[j] for j in selected if j != i]
        loo_F[_fingerprint(samples[i][2])] = _geomean_F(others, per_line) if others else robust_F

    # 4. Apply robust (T, F) + self-absorption handling held-out and score (LOO per sample).
    def solve_fn(db_, wl, intensity, truth):
        F = loo_F.get(_fingerprint(intensity), robust_F)
        return solve_l5(
            db_, wl, intensity, truth, T_star=robust_T, F=F, per_line=per_line, mode=mode
        )

    held = run_benchmark(solve_fn, db_path=db_path, limit=limit)

    return {
        "mode": mode,
        "robust_T_K": robust_T,
        "per_line": per_line,
        "n_selected": len(selected),
        "selected_samples": [samples[i][0] for i in selected],
        "robust_F": {str(k): v for k, v in robust_F.items()},
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
    ap.add_argument(
        "--mode",
        default="thin_filter",
        choices=("off", "per_line", "thin_filter"),
        help="self-absorption handling for the single-run path",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="run mode=off (v2 baseline) vs each SA mode and report Fe / overall deltas",
    )
    a = ap.parse_args()
    grid = tuple(float(t) for t in range(a.tmin, a.tmax + 1, a.tstep))

    def _report(tag, out):
        hs = out["held_out"]
        print(f"\n[{tag}] L5 real-steel results (mode={out['mode']}):")
        print(f"  n_selected: {out['n_selected']} -> {out['selected_samples']}")
        print(f"  robust_T_K: {out['robust_T_K']:.0f}")
        print("  HELD-OUT RMSEP (wt%):")
        for k in sorted(hs):
            v = hs[k]
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
        return hs

    def _run(mode):
        return run_l5(
            db_path=a.db_path,
            t_grid=grid,
            cond_T_K=a.cond_t,
            per_line=a.per_line,
            mode=mode,
            limit=a.limit,
        )

    if a.compare:
        hs_off = _report("v2 (mode=off)", _run("off"))
        for m in ("thin_filter", "per_line"):
            hs_m = _report(f"L5 ({m})", _run(m))
            print(
                "\n[compare:{0}] Fe RMSEP  v2={1:.2f}  L5={2:.2f}".format(
                    m, hs_off.get("rmsep_Fe", float("nan")), hs_m.get("rmsep_Fe", float("nan"))
                )
            )
            print(
                "[compare:{0}] overall  v2={1:.2f}  L5={2:.2f}".format(
                    m, hs_off["rmsep_overall"], hs_m["rmsep_overall"]
                )
            )
    else:
        _report("L5", _run(a.mode))
