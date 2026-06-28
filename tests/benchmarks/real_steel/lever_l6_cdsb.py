"""Lever L6 -- Columnar-Density Saha-Boltzmann (CD-SB) for the self-absorbed Fe matrix.

Motivation (round-3 closing question, docs/research/real-steel-opc-promotion.md):
the v2 winner uses ``thin_filter`` (L5) -- it *drops* the width-broadened Fe matrix
lines and anchors on the optically-thin Fe subset (held-out overall 9.561, Fe
19.615). Dropping lines is a line-*selection* change, so it composes cleanly with
the OPC ``F`` scalar instead of double-correcting it. The open question this lever
closes: can we do *better* by KEEPING the self-absorbed Fe lines and USING them via
the Columnar-Density Saha-Boltzmann method, rather than discarding their
information?

CD-SB grounding (NotebookLM notebook ``f1d2a053``; Cristoforetti & Tognoni 2013,
Spectrochim. Acta B 86, 51; the Pisa-group COG relation)
--------------------------------------------------------
In classical CF-LIBS the Boltzmann-plot ordinate ``ln(I*lambda / (g_k*A_ki))``
is proportional to the UPPER-level number density ``n_k`` and is valid only for
optically-thin lines. The CD-SB method (Cristoforetti & Tognoni 2013) overcomes
self-absorption by rewriting the plot in terms of the LOWER-level *columnar
density* ``n_i^l`` (the density of the absorbing lower level integrated along the
line of sight ``l``), which is read from the self-absorbed line's *shape* rather
than its (suppressed) peak intensity. NotebookLM, quoting the source:

* Modified Saha-Boltzmann line ``y* = m x* + q`` with ``m = -1/(k_B T)`` and
  ``q = ln(N*l / U(T))``. For atomic lines ``x* = E_i`` (the LOWER level) and
  ``y* = ln(n_i^l / g_i)``.
* Columnar density from the integrated absorption coefficient ``kl``:
  ``n_i^l / g_i = 8*pi*c*Delta_lambda_0*(kl) / (lambda_0^4 * A_ki)``  (eqn 12).
* ``kl`` is recovered from the measured self-absorption broadening via the
  Pisa-group curve-of-growth (COG) relation (eqn 13):
  ``Delta_lambda / Delta_lambda_0 = ((1 - e^{-kl}) / kl)^{-0.56}``, where
  ``Delta_lambda_0`` is the optically-thin FWHM (``= w_s * n_e`` from Stark, here
  the optically-thin intrinsic reference width).

The unlock: ``y*`` is *decoupled from the raw self-absorbed peak intensity* -- it
is built from the line WIDTH (via ``kl``) and the atomic constants. The broader
(more self-absorbed) a matrix line is, the larger its ``kl`` and the larger the
columnar density it testifies to. This is exactly the per-spectrum signal a single
OPC scalar ``F_Fe`` structurally cannot carry (variable 85-95 wt% Fe -> different
optical depth, hence different line width, in every spectrum).

How CD-SB enters this intensity-based solver (and why it is NOT the L5 per_line trap)
------------------------------------------------------------------------------------
The shipped constrained Saha-Boltzmann solver ingests per-line *intensities* and
internally forms ``n_k`` from ``I*lambda/(g_k*A_ki)``. So to put a thick element on
the CD-SB ordinate we inject, for each of its lines, the intensity the solver would
need to recover the CD-SB columnar density. Requiring
``(I_cdsb*lambda/(g_k*A_ki)) * U(T) * exp(E_k/k_BT) == N*l`` (the columnar density)
with ``N*l = (n_i^l/g_i)*U(T)*exp(E_i/k_BT)`` and substituting eqn 12 gives, after
the ``A_ki`` and the per-element constants cancel::

    I_cdsb(line) proportional_to  g_k * kl * exp(-(E_k - E_i)/(k_B T)) / lambda^5

with ``E_k - E_i = hc/lambda`` the photon energy and ``kl`` from the COG inversion
of the measured width. The MEASURED self-absorbed Fe intensity does NOT appear in
this ordinate (only the width does) -- this is the categorical difference from L5's
``per_line`` boost, which *scaled* the measured intensity by a width factor and so
lived in the same multiplicative subspace as ``F_Fe`` (double-correcting; regressed
overall 10.12 -> 11.90). Because CD-SB *replaces* the ordinate with a width-derived
columnar density rather than scaling the intensity, it is neither a pure selection
nor a pure per-element intensity scale.

Scale / OPC composition (the honest decoupling)
-----------------------------------------------
The CD-SB ordinate is in physical (width/atomic) units while the trace minors keep
their measured detector-unit intensities, so the thick element's CD-SB intensities
are placed on a comparable footing by ONE GLOBAL scale ``S`` (the same constant for
every spectrum) derived from the calibration STANDARDS only:
``S = geomean_std[ geomean(measured Fe areas) / geomean(CD-SB raw Fe) ]``. ``S`` is
a unit constant (it does NOT vary per spectrum, so it does not remove the per-
spectrum ``kl`` magnitude that carries the Fe-content adaptivity), and any residual
relative-sensitivity is then absorbed by the OPC ``F`` **re-derived on this same
CD-SB pipeline** -- so ``F`` corrects only the static residual, never the dynamic
per-spectrum self-absorption the CD-SB ordinate already encodes. This is the only
honest way to test CD-SB against an already-OPC-calibrated pipeline (round-3 design
insight): apply CD-SB inside the solve, THEN re-fit ``F`` on the CD-SB recovery.

HONEST EVALUATION (mirrors best_config_v2 / lever_l5):
  * Conditioning-gated standard selection (in-sample only), robust fixed ``T`` =
    mean of selected standards' in-sample optimal ``T*``, the global scale ``S`` and
    the robust ``F`` (geomean over selected standards, clamp-saturated values
    filtered) derived ONLY from the standards' own data + own certified truth.
  * Per-sample leave-one-out so a selected standard never sees its own ``F``.
  * The headline is the held-out ``rmsep_overall`` over all 36 samples. No parameter
    is tuned to held-out truth; the COG exponent (0.56) and the optically-thin
    reference width (0.08 nm) are the physical CD-SB / El Sherbini constants.

Benchmark-layer only (needs the measured line widths). No shipped ``cflibs/`` change
in this commit, so the DED-precision gate is unaffected. ``mode="off"`` reproduces
the v2 baseline through the identical machinery for an apples-to-apples comparison.
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
from tests.benchmarks.real_steel.lever_l4_selfabs import measure_line_width  # noqa: E402

# --- Physical CD-SB constants (NOT tuned to the held-out gate) -----------------
HC_EV_NM = 1239.84193  # photon energy E[eV] = HC_EV_NM / lambda[nm]
K_B_EV = 8.617333262e-5  # Boltzmann constant, eV/K
COG_EXP = 0.56  # Pisa-group curve-of-growth exponent: dl/dl0 = ((1-e^-kl)/kl)^-0.56
THIN_REF_WIDTH_NM = 0.08  # optically-thin intrinsic reference width Delta_lambda_0 (nm)
LAMBDA_REF_NM = 370.0  # numeric reference for the lambda^5 factor (cancels into S/F)
KL_MAX = 50.0  # cap on the recovered optical depth kl
KL_FLOOR = 0.1  # floor so a marginally-broadened thick-element line still contributes

# --- Element-thickness gating (identical physical values to L4/L5) ------------
MIN_CORR_LINES = 3  # an element is "thick" only with >= this many well-measured lines
SNR_MIN = 3.0  # peak-SNR floor for a line to count toward width statistics
THICK_RATIO = 1.15  # element is thick only when median width / instr exceeds this
INSTR_WIDTH_PCT = 10.0  # percentile of well-measured widths taken as the instrument width


def _geomean(vals: Sequence[float]) -> float:
    """Geometric mean of strictly-positive finite values (1.0 if none)."""
    v = [float(x) for x in vals if np.isfinite(x) and x > 0]
    return float(np.exp(np.mean(np.log(v)))) if v else 1.0


def _solve_kl_from_ratio(
    ratio: float, *, cog_exp: float = COG_EXP, kl_max: float = KL_MAX, n_iter: int = 60
) -> float:
    """Invert the COG relation dl/dl0 = ((1-e^-kl)/kl)^-cog_exp for the optical depth kl.

    ``ratio`` is the measured self-absorption broadening Delta_lambda/Delta_lambda_0
    (>= 1). Returns 0 for an unbroadened (optically-thin) line and a bisection root
    otherwise. ``g(kl) = (1-e^-kl)/kl`` decreases monotonically from 1 (kl->0) to 0.
    """
    if not np.isfinite(ratio) or ratio <= 1.0:
        return 0.0
    target = ratio ** (-1.0 / cog_exp)  # g(kl) we must match, in (0, 1)

    def g(kl: float) -> float:
        return 1.0 if kl <= 0.0 else (1.0 - np.exp(-kl)) / kl

    if g(kl_max) >= target:  # even at kl_max not thick enough -> cap
        return kl_max
    lo, hi = 0.0, kl_max
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if g(mid) > target:
            lo = mid  # g too high -> need larger kl
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cdsb_raw_for_thick(
    wl: np.ndarray,
    intensity: np.ndarray,
    observations: Sequence,
    T_K: float,
    *,
    thin_ref_width_nm: float = THIN_REF_WIDTH_NM,
    min_corr_lines: int = MIN_CORR_LINES,
    snr_min: float = SNR_MIN,
    thick_ratio: float = THICK_RATIO,
    instr_width_pct: float = INSTR_WIDTH_PCT,
) -> Tuple[Dict[int, float], Dict[str, float]]:
    """Per-line CD-SB raw columnar-density ordinate for every optically-thick element.

    Returns ``({obs_index: raw_value}, diagnostics)``. ``raw`` (before the global
    unit scale ``S``) is ``g_k * kl * exp(-(E_k-E_i)/(k_B T)) * (lref/lambda)^5`` with
    ``kl`` from the COG inversion of the instrument-deconvolved line width. Only the
    width enters -- never the measured (self-absorbed) intensity. Optically-thin
    elements (median width at the instrument floor) get no entries (left on measured
    intensity by the caller). Mirrors L4/L5's thickness gating exactly.
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
        return {}, {}
    instr = float(np.percentile(good_all, instr_width_pct))
    if instr <= 0:
        return {}, {}
    ref = max(float(thin_ref_width_nm), 1e-6)

    raw: Dict[int, float] = {}
    diag: Dict[str, float] = {}
    for el, idxs in by_el.items():
        good = [i for i in idxs if np.isfinite(widths[i]) and snrs[i] >= snr_min]
        if len(good) < min_corr_lines:
            continue
        median_w = float(np.median([widths[i] for i in good]))
        if median_w / instr < thick_ratio:
            continue  # optically thin element -> not CD-SB corrected
        diag[el] = median_w / instr
        for i in idxs:
            # Marginal / unmeasurable lines fall back to the element median width so
            # every thick-element line still gets a (floored) columnar-density vote.
            w = widths[i] if (np.isfinite(widths[i]) and snrs[i] >= snr_min) else median_w
            intrinsic = float(np.sqrt(max(w**2 - instr**2, 0.0)))
            kl = max(_solve_kl_from_ratio(intrinsic / ref), KL_FLOOR)
            o = observations[i]
            lam = float(o.wavelength_nm)
            photon_ev = HC_EV_NM / lam
            boltz = float(np.exp(-photon_ev / (K_B_EV * T_K)))
            raw_i = float(o.g_k) * kl * boltz * (LAMBDA_REF_NM / lam) ** 5
            if np.isfinite(raw_i) and raw_i > 0:
                raw[i] = raw_i
    return raw, diag


def _cdsb_scale(observations: Sequence, raw: Dict[int, float]) -> float:
    """Per-standard unit factor: geomean(measured thick areas) / geomean(CD-SB raw).

    Placing the CD-SB ordinate at the same average brightness as the measured Fe
    lines ON THE STANDARD keeps the re-derived ``F_Fe`` near O(1) (inside the OPC
    clamp band). It is a single constant per standard; the run-level ``S`` is the
    geomean across selected standards and is applied UNCHANGED to every spectrum, so
    it never removes the per-spectrum ``kl`` magnitude (the Fe-content adaptivity).
    """
    if not raw:
        return 1.0
    meas = [float(observations[i].intensity) for i in raw]
    return _geomean(meas) / max(_geomean([raw[i] for i in raw]), 1e-300)


def apply_cdsb(
    wl: np.ndarray,
    intensity: np.ndarray,
    observations: Sequence,
    T_K: float,
    scale: float,
) -> Tuple[int, float]:
    """In-place: replace each thick-element line's intensity with its CD-SB ordinate.

    The self-absorbed Fe lines are KEPT (vs L5 thin_filter, which drops them) and
    placed on the columnar-density ordinate ``scale * raw`` (width-derived, NOT the
    measured intensity). Thin elements are untouched. Returns ``(n_replaced, max_kl_raw)``.
    """
    raw, _diag = _cdsb_raw_for_thick(wl, intensity, observations, T_K)
    if not raw:
        return 0, 0.0
    n = 0
    mx = 0.0
    for i, raw_i in raw.items():
        o = observations[i]
        o.intensity = float(scale) * raw_i
        o.intensity_uncertainty = max(o.intensity * 0.05, 1e-12)
        n += 1
        mx = max(mx, raw_i)
    return n, mx


def _l2_obs(db, wl, intensity, truth):
    window = (float(wl.min()), float(wl.max()))
    specs: List[LineSpec] = []
    for e in truth:
        specs.extend(select_l2_lines(db, e, window, 8))
    if not specs:
        return None
    return extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=0.2)


def solve_l6(
    db,
    wl,
    intensity,
    truth,
    *,
    T_star: float,
    F: Dict,
    scale: float = 1.0,
    per_line: bool = False,
    mode: str = "cdsb",
) -> Dict[str, float]:
    """v2 solve with a per-spectrum CD-SB ordinate for the thick matrix before OPC.

    ``mode``:
      * ``"off"``  -- reproduce v2 (no CD-SB; measured intensities throughout).
      * ``"cdsb"`` -- replace the thick element's lines with their columnar-density
        ordinate (kept, width-derived), then OPC ``F`` rescale, then fixed-``T`` solve.

    Order: extract L2 lines -> CD-SB ordinate (``mode``, at ``scale``) -> OPC
    per-element ``F`` rescale -> constrained Saha-Boltzmann at fixed ``T_star``.
    """
    obs = _l2_obs(db, wl, intensity, truth)
    if not obs:
        return {}
    if mode == "cdsb":
        apply_cdsb(wl, intensity, obs, T_star, scale)
    for o in obs:
        f = F.get(_line_key(o.element, o.wavelength_nm), 1.0) if per_line else F.get(o.element, 1.0)
        o.intensity = float(o.intensity) * float(f)
        o.intensity_uncertainty = max(float(o.intensity_uncertainty) * float(f), 1e-12)
    res = run_constrained_solver(db, obs, 1e17, fixed_temperature_K=T_star)
    return recovered_wt(res)


def derive_F_l6(
    db,
    standard,
    T_star: float,
    scale: float,
    *,
    per_line: bool = False,
    mode: str = "cdsb",
) -> Dict:
    """OPC ``F`` from the standard's CD-SB-handled (L2+mode+fixed-T) recovery.

    ``F_e = C_true_e / C_rec_e`` (renormalized), clamped to the OPC band, derived on
    the SAME pipeline (``mode``, ``scale``) it is applied to, so the CD-SB ordinate
    and the OPC factor compose instead of double-counting.
    """
    _sid, wl, inten, truth = standard
    rec = solve_l6(
        db, wl, inten, truth, T_star=T_star, F={}, scale=scale, per_line=per_line, mode=mode
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


def _standard_scale(db, standard, T_star: float) -> float:
    """Per-standard CD-SB unit factor S_s = geomean(measured Fe) / geomean(raw Fe)."""
    _sid, wl, inten, truth = standard
    obs = _l2_obs(db, wl, inten, truth)
    if not obs:
        return 1.0
    raw, _diag = _cdsb_raw_for_thick(wl, inten, obs, T_star)
    return _cdsb_scale(obs, raw)


def run_l6(
    db_path: str = "ASD_da/libs_production.db",
    *,
    t_grid=DEFAULT_T_GRID,
    cond_T_K: float = COND_T_K,
    per_line: bool = False,
    mode: str = "cdsb",
    limit: int | None = None,
) -> Dict[str, object]:
    """Honest L6 run: conditioning-gated robust (T*, S, F) on the CD-SB pipeline, LOO-scored.

    Mirrors :func:`...best_config_v2.run_v2` / :func:`...lever_l5_fe_selfabs.run_l5`
    but threads the CD-SB columnar-density ordinate (``mode="cdsb"``) and the global
    unit scale ``S`` (standards only). ``mode="off"`` reproduces v2.
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
        print(f"[L6] no standard passed; fallback to best in-sample idx {selected[0]}", flush=True)

    # 3. Robust T = mean of the selected standards' in-sample optimal T*.
    T_stars: List[float] = []
    for i in selected:
        T_i, err_i, _curve = choose_optimal_T(db, samples[i], t_grid)
        T_stars.append(T_i)
        print(f"  [T* {i}] {samples[i][0]}: T*={T_i:.0f}K in-sample rmsep={err_i:.2f}", flush=True)
    robust_T = float(np.mean(T_stars))
    print(f"[L6] robust T = mean(T*) = {robust_T:.0f} K over {len(selected)} standards", flush=True)

    # 3b. Global CD-SB unit scale S (standards only; one constant for all spectra).
    if mode == "cdsb":
        S_list = [_standard_scale(db, samples[i], robust_T) for i in selected]
        scale = _geomean(S_list)
        print(f"[L6] global CD-SB scale S = geomean over standards = {scale:.4g}", flush=True)
    else:
        scale = 1.0

    # 4. Per-standard F on the CD-SB pipeline; robust + leave-one-out geomean F.
    F_per_std: Dict[int, Dict] = {}
    for i in selected:
        F_per_std[i] = derive_F_l6(db, samples[i], robust_T, scale, per_line=per_line, mode=mode)
    robust_F = _geomean_F(list(F_per_std.values()), per_line)

    loo_F: Dict[Tuple[int, float], Dict] = {}
    for i in selected:
        others = [F_per_std[j] for j in selected if j != i]
        loo_F[_fingerprint(samples[i][2])] = _geomean_F(others, per_line) if others else robust_F

    # 5. Apply robust (T, S, F) + CD-SB held-out and score (LOO per sample).
    def solve_fn(db_, wl, intensity, truth):
        F = loo_F.get(_fingerprint(intensity), robust_F)
        return solve_l6(
            db_,
            wl,
            intensity,
            truth,
            T_star=robust_T,
            F=F,
            scale=scale,
            per_line=per_line,
            mode=mode,
        )

    held = run_benchmark(solve_fn, db_path=db_path, limit=limit)

    return {
        "mode": mode,
        "robust_T_K": robust_T,
        "cdsb_scale": scale,
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
        default="cdsb",
        choices=("off", "cdsb"),
        help="CD-SB ordinate (cdsb) or v2 reproduction (off) for the single-run path",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="run mode=off (v2 baseline) vs mode=cdsb and report Fe / overall deltas",
    )
    a = ap.parse_args()
    grid = tuple(float(t) for t in range(a.tmin, a.tmax + 1, a.tstep))

    def _report(tag, out):
        hs = out["held_out"]
        print(f"\n[{tag}] L6 real-steel results (mode={out['mode']}):")
        print(f"  n_selected: {out['n_selected']} -> {out['selected_samples']}")
        print(f"  robust_T_K: {out['robust_T_K']:.0f}")
        print(f"  cdsb_scale: {out['cdsb_scale']:.4g}")
        print("  HELD-OUT RMSEP (wt%):")
        for k in sorted(hs):
            v = hs[k]
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
        return hs

    def _run(mode):
        return run_l6(
            db_path=a.db_path,
            t_grid=grid,
            cond_T_K=a.cond_t,
            per_line=a.per_line,
            mode=mode,
            limit=a.limit,
        )

    if a.compare:
        hs_off = _report("v2 (mode=off)", _run("off"))
        hs_cd = _report("L6 (cdsb)", _run("cdsb"))
        print(
            "\n[compare] Fe RMSEP  v2={:.2f}  L6={:.2f}".format(
                hs_off.get("rmsep_Fe", float("nan")), hs_cd.get("rmsep_Fe", float("nan"))
            )
        )
        print(
            "[compare] overall  v2={:.2f}  L6={:.2f}".format(
                hs_off["rmsep_overall"], hs_cd["rmsep_overall"]
            )
        )
    else:
        _report("L6", _run(a.mode))
