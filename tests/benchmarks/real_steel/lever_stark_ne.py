"""Lever Stark-n_e — per-spectrum electron density from measured Stark widths.

The shipped real-steel solves inject a *fixed* ``n_e = 1e17 cm^-3`` into the Saha
balance (see :func:`...solver_runner.run_constrained_solver`). That value is an
a-priori 1-atm pressure-balance guess, not a measurement. Now that the atomic DB
carries complete Stark-broadening coverage (commit 5967206), this lever measures
``n_e`` *per spectrum* from the Stark widths of literature-grade (``stark_b``)
isolated lines and feeds that into the balance instead of the fixed 1e17.

Physics (single source of truth: :mod:`cflibs.radiation.stark`). The
electron-impact width is linear in density,

    w_fwhm = w_ref * (n_e / 1e17) * (T / T_ref)^(-alpha),

so a measured Lorentzian (Stark) FWHM inverts to

    n_e = 1e17 * (w_stark / w_ref) * (T / T_ref)^(alpha)

(Tognoni 2010; Aragon & Aguilera 2010; for hydrogen the canonical
``n_e = 8.02e12 * (Delta lambda_FWHM)^1.5`` is the H-specific special case of the
same width->density inversion). The measurement is delegated UNCHANGED to the
shipped :func:`cflibs.inversion.physics.stark_ne.measure_stark_ne`, which
pinned-Voigt-fits each candidate line (Gaussian fixed to instrument+Doppler,
Lorentzian free), gates on ``stark_b`` provenance + SNR + isolation, inverts each
line, and robustly combines (median + MAD). Only the *injection point* changes.

COMPOSITION WITH THE SHIPPED OPC (best_config_v2): the pipeline is unchanged
except for the injected ``n_e`` ---

  1. L2 neutral-anchor line selection (``select_l2_lines``).
  2. Measure ``n_e`` from the same spectrum's Stark widths (this lever); fall back
     to 1e17 only when no line qualifies, and clamp to a physical band.
  3. Robust OPC ``F`` rescale (geometric mean over conditioning-gated standards).
  4. Constrained Saha-Boltzmann solve at the robust fixed ``T`` and the measured
     ``n_e``.

KEY CONSISTENCY POINT (same as v2's L4 composition): the OPC ``F`` is DERIVED on
the *same* pipeline it is applied to. When ``use_stark_ne=True`` the standard's
uncorrected recovery used for ``F_e = C_true_e / C_rec_e`` is the L2+measured-n_e
recovery, so ``F`` only corrects the residual the measured ``n_e`` leaves. The two
are composed, not double-counted.

HONEST EVALUATION (un-overfittable; no held-out peeking):
  * Standard selection (conditioning gate at fixed T) and the robust ``T`` /
    robust ``F`` derive ONLY from candidate standards' own spectra + own
    certified truth. The selection/robust-T machinery is IDENTICAL to v2 so the
    delta isolates the n_e change. The per-spectrum n_e reads only each scored
    sample's own measured line widths -- never any truth.
  * Leave-one-out ``F`` map (a selected standard never sees its own ``F``).
  * Headline: held-out ``rmsep_overall`` over all 36 samples, compared to the
    fixed-1e17 (``use_stark_ne=False``) result on the identical pipeline and to
    the shipped 8.383 wt% reference.
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
from tests.benchmarks.real_steel.harness import load_real_steel, run_benchmark  # noqa: E402
from tests.benchmarks.real_steel.lever_l1_temp import choose_optimal_T  # noqa: E402
from tests.benchmarks.real_steel.lever_l2_lines import select_l2_lines  # noqa: E402
from tests.benchmarks.real_steel.lever_l3_opc import (
    _F_MAX,
    _F_MIN,
    _MIN_WT,
    _line_key,
)  # noqa: E402
from tests.benchmarks.real_steel.lever_l3b_robust_opc import (  # noqa: E402
    COND_T_K,
    DEFAULT_T_GRID,
    _fingerprint,
    _geomean_F,
    assess_conditioning,
)

# Instrument FWHM the L2 extractor + Stark fit assume (the real-steel rig value).
INSTR_FWHM_NM = 0.2
# The fixed a-priori density the shipped pipeline injects (fallback when no line
# qualifies for a Stark measurement).
NE_FIXED_CM3 = 1.0e17
# Physical clamp band for the per-spectrum measured n_e (cm^-3). LIBS plasmas in
# this window sit ~1e16-1e18; values outside indicate a failed fit / blend
# dominating the median (e.g. one sample's raw median was 1.7e19 with comparable
# scatter). Clamping keeps a single bad line from wrecking the Saha balance. The
# band is set from plasma physics, NEVER tuned to held-out truth.
NE_CLAMP_MIN_CM3 = 3.0e16
NE_CLAMP_MAX_CM3 = 5.0e18
# Stark-measurement gates (delegated to measure_stark_ne).
STARK_MIN_SNR = 3.0
STARK_MAX_LINES = 8


def _l2_specs(db, truth, window) -> List[LineSpec]:
    specs: List[LineSpec] = []
    for e in truth:
        specs.extend(select_l2_lines(db, e, window, 8))
    return specs


def measure_ne(
    db, wl, intensity, obs, *, T_K: float, instr: float = INSTR_FWHM_NM
) -> Tuple[float, int]:
    """Per-spectrum n_e from Stark widths -> (n_e_cm3, n_diagnostic_lines).

    Delegates the measurement to the shipped :func:`measure_stark_ne`; clamps the
    robust median to the physical band and falls back to ``NE_FIXED_CM3`` when no
    line qualifies. Returns ``(NE_FIXED_CM3, 0)`` on fallback.
    """
    from cflibs.inversion.physics.stark_ne import measure_stark_ne

    diag = measure_stark_ne(
        wl,
        intensity,
        obs,
        db,
        instrument_fwhm_nm=instr,
        T_K=T_K,
        max_lines=STARK_MAX_LINES,
        min_snr=STARK_MIN_SNR,
    )
    if not diag.usable or diag.ne_median_cm3 is None:
        return NE_FIXED_CM3, 0
    ne = float(np.clip(diag.ne_median_cm3, NE_CLAMP_MIN_CM3, NE_CLAMP_MAX_CM3))
    return ne, int(diag.n_lines)


def solve_stark(
    db,
    wl,
    intensity,
    truth,
    *,
    T_star: float,
    F: Dict,
    per_line: bool = False,
    use_stark_ne: bool = True,
    instr: float = INSTR_FWHM_NM,
) -> Dict[str, float]:
    """L2 lines + (measured n_e) + OPC ``F`` rescale + fixed-``T`` constrained solve.

    When ``use_stark_ne`` is False this is byte-for-byte the v2 pipeline (fixed
    1e17), so the same function backs both the lever and its control.
    """
    window = (float(wl.min()), float(wl.max()))
    specs = _l2_specs(db, truth, window)
    if not specs:
        return {}
    obs = extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=instr)
    if not obs:
        return {}
    if use_stark_ne:
        ne_cm3, _n = measure_ne(db, wl, intensity, obs, T_K=T_star, instr=instr)
    else:
        ne_cm3 = NE_FIXED_CM3
    for o in obs:
        f = F.get(_line_key(o.element, o.wavelength_nm), 1.0) if per_line else F.get(o.element, 1.0)
        o.intensity = float(o.intensity) * float(f)
        o.intensity_uncertainty = max(float(o.intensity_uncertainty) * float(f), 1e-12)
    res = run_constrained_solver(db, obs, ne_cm3, fixed_temperature_K=T_star)
    return recovered_wt(res)


def derive_F_stark(
    db,
    standard,
    T_star: float,
    *,
    per_line: bool = False,
    use_stark_ne: bool = True,
    instr: float = INSTR_FWHM_NM,
) -> Dict:
    """OPC ``F`` from the standard's *uncorrected* (L2 + measured-n_e) recovery.

    ``F_e = C_true_e / C_rec_e`` clamped to the OPC band, derived on the SAME
    pipeline (``use_stark_ne``) it will be applied to so the n_e change and the
    OPC factor compose consistently rather than double-counting.
    """
    _sid, wl, inten, truth = standard
    rec = solve_stark(
        db,
        wl,
        inten,
        truth,
        T_star=T_star,
        F={},
        per_line=per_line,
        use_stark_ne=use_stark_ne,
        instr=instr,
    )
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


def run_stark_ne(
    db_path: str = "ASD_da/libs_production.db",
    *,
    t_grid=DEFAULT_T_GRID,
    cond_T_K: float = COND_T_K,
    per_line: bool = False,
    use_stark_ne: bool = True,
    instr: float = INSTR_FWHM_NM,
    limit: int | None = None,
) -> Dict[str, object]:
    """Honest run: v2 selection/robust-(T,F) with per-spectrum measured n_e injected.

    The conditioning gate and robust-``T`` are IDENTICAL to v2 (fixed-1e17 gating);
    only the F-derivation and final solve swap the injected n_e for the measured
    value (``use_stark_ne=True``). With ``use_stark_ne=False`` this reproduces v2
    exactly -- the apples-to-apples control on the same selection.
    """
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    samples = list(load_real_steel())
    if limit is not None:
        samples = samples[:limit]

    # 1+2. Conditioning gate (identical to v2; fixed-1e17 in-sample only).
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
        print(
            f"[stark_ne] no standard passed; fallback to best in-sample idx {selected[0]}",
            flush=True,
        )

    # 3. Robust T (identical to v2) + per-standard F on the measured-n_e pipeline.
    T_stars: List[float] = []
    for i in selected:
        T_i, err_i, _curve = choose_optimal_T(db, samples[i], t_grid)
        T_stars.append(T_i)
        print(f"  [T* {i}] {samples[i][0]}: T*={T_i:.0f}K in-sample rmsep={err_i:.2f}", flush=True)
    robust_T = float(np.mean(T_stars))
    print(
        f"[stark_ne] robust T = mean(T*) = {robust_T:.0f} K over {len(selected)} standards",
        flush=True,
    )

    # Diagnostic: report the measured n_e per selected standard (in-sample).
    ne_report: Dict[int, Tuple[float, int]] = {}
    for i in selected:
        _sid, wl_i, inten_i, truth_i = samples[i]
        window = (float(wl_i.min()), float(wl_i.max()))
        obs_i = extract_line_intensities(
            wl_i, inten_i, _l2_specs(db, truth_i, window), instrument_fwhm_nm=instr
        )
        ne_i, nl_i = measure_ne(db, wl_i, inten_i, obs_i, T_K=robust_T, instr=instr)
        ne_report[i] = (ne_i, nl_i)
        print(f"  [n_e {i}] {samples[i][0]}: n_e={ne_i:.3e} cm^-3 ({nl_i} lines)", flush=True)

    F_per_std: Dict[int, Dict] = {}
    for i in selected:
        F_per_std[i] = derive_F_stark(
            db, samples[i], robust_T, per_line=per_line, use_stark_ne=use_stark_ne, instr=instr
        )
    robust_F = _geomean_F(list(F_per_std.values()), per_line)

    loo_F: Dict[Tuple[int, float], Dict] = {}
    for i in selected:
        others = [F_per_std[j] for j in selected if j != i]
        loo_F[_fingerprint(samples[i][2])] = _geomean_F(others, per_line) if others else robust_F

    # 4. Apply robust (T, F) + measured n_e held-out and score (LOO per sample).
    def solve_fn(db_, wl, intensity, truth):
        F = loo_F.get(_fingerprint(intensity), robust_F)
        return solve_stark(
            db_,
            wl,
            intensity,
            truth,
            T_star=robust_T,
            F=F,
            per_line=per_line,
            use_stark_ne=use_stark_ne,
            instr=instr,
        )

    held = run_benchmark(solve_fn, db_path=db_path, limit=limit)

    return {
        "use_stark_ne": use_stark_ne,
        "robust_T_K": robust_T,
        "per_line": per_line,
        "n_selected": len(selected),
        "selected_indices": selected,
        "selected_samples": [samples[i][0] for i in selected],
        "selected_ne_cm3": {samples[i][0]: ne_report[i][0] for i in selected},
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
        "--compare",
        action="store_true",
        help="run both fixed-1e17 (control) and measured-n_e on the identical selection",
    )
    a = ap.parse_args()
    grid = tuple(float(t) for t in range(a.tmin, a.tmax + 1, a.tstep))

    def _report(tag, out):
        hs = out["held_out"]
        print(f"\n[{tag}] stark_ne real-steel results (use_stark_ne={out['use_stark_ne']}):")
        print(f"  n_selected: {out['n_selected']} -> {out['selected_samples']}")
        print(f"  robust_T_K: {out['robust_T_K']:.0f}")
        print(f"  selected n_e: { {k: f'{v:.2e}' for k, v in out['selected_ne_cm3'].items()} }")
        print("  HELD-OUT RMSEP (wt%):")
        for k in sorted(hs):
            v = hs[k]
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
        return hs

    if a.compare:
        out_fixed = run_stark_ne(
            db_path=a.db_path,
            t_grid=grid,
            cond_T_K=a.cond_t,
            per_line=a.per_line,
            use_stark_ne=False,
            limit=a.limit,
        )
        hs_fixed = _report("fixed-1e17", out_fixed)
        out_stark = run_stark_ne(
            db_path=a.db_path,
            t_grid=grid,
            cond_T_K=a.cond_t,
            per_line=a.per_line,
            use_stark_ne=True,
            limit=a.limit,
        )
        hs_stark = _report("measured-n_e", out_stark)
        print(
            "\n[compare] overall  fixed-1e17={:.3f}  measured-n_e={:.3f}".format(
                hs_fixed["rmsep_overall"], hs_stark["rmsep_overall"]
            )
        )
    else:
        out = run_stark_ne(
            db_path=a.db_path,
            t_grid=grid,
            cond_T_K=a.cond_t,
            per_line=a.per_line,
            use_stark_ne=True,
            limit=a.limit,
        )
        _report("stark_ne", out)
