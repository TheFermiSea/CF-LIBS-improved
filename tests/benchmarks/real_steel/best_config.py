"""Best lever combination — L2 line-selection + L1 optimal-T + L3 OPC F (real-steel gate).

Composes the three real-steel levers that each improved on the baseline
(``rmsep_overall`` 39.04 wt%), in the physically-sensible order:

  1. **L2** neutral-anchor wide-E_k line selection
     (:func:`tests.benchmarks.real_steel.lever_l2_lines.select_l2_lines`) — removes the
     ion-only Cu Saha trap. Baseline 39.04 -> 30.43 (deterministic, no calibration).
  2. **L1** hold the plasma at a fixed *optimal* temperature ``T*`` derived from one
     matrix-matched standard (:mod:`...lever_l1_temp`) — composition is hypersensitive
     to T (Zhao 2018). -> 29.98 held-out.
  3. **L3** one-point-calibration per-element correction ``F_e = C_true/C_rec`` derived
     from the SAME standard at ``T*`` (:mod:`...lever_l3_opc`) — fixes the absolute-closure
     relative-error amplification for a known matrix (Cavalcanti 2013). -> mean 24.7,
     best 8.4 held-out.

The combined solve selects L2 lines, scales each observation's intensity by its element's
OPC factor ``F``, and runs the constrained Saha-Boltzmann solve at the fixed ``T*``.

HONEST EVALUATION (no per-sample overfitting):
  * Both ``T*`` and ``F`` are derived from ONE matrix-matched certified standard and applied
    UNCHANGED to every other sample. Nothing is tuned to a held-out sample's truth.
  * The standard is chosen *a priori* by the lowest in-sample uncorrected-L2 RMSEP (the most
    self-consistent calibration sample) — a criterion that uses only the candidate standards'
    own certified compositions, never any held-out truth. (L3 found single-standard OPC is
    sensitive to whether the standard's own uncorrected solve is keystone-collapsed; this
    selection picks the least-collapsed standard without peeking at the test set.)
  * The held-out ``rmsep_overall`` (over the 35 non-standard samples) is the un-overfittable
    headline. The in-sample (standard) RMSEP at ``T*`` with ``F`` is reported for context.
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
from tests.benchmarks.real_steel.harness import load_real_steel, score  # noqa: E402
from tests.benchmarks.real_steel.lever_l1_temp import (  # noqa: E402
    DEFAULT_T_GRID,
    choose_optimal_T,
    solve_l1,
)
from tests.benchmarks.real_steel.lever_l2_lines import select_l2_lines  # noqa: E402
from tests.benchmarks.real_steel.lever_l3_opc import (  # noqa: E402
    _F_MAX,
    _F_MIN,
    _MIN_WT,
    _line_key,
)


def _specs(db, truth, window) -> List[LineSpec]:
    specs: List[LineSpec] = []
    for e in truth:
        specs.extend(select_l2_lines(db, e, window, 8))
    return specs


def solve_best(
    db, wl, intensity, truth, *, T_star: float, F: Dict, per_line: bool = False
) -> Dict[str, float]:
    """Combined solve: L2 lines, OPC ``F`` rescale, constrained Saha-Boltzmann at fixed ``T*``."""
    window = (float(wl.min()), float(wl.max()))
    specs = _specs(db, truth, window)
    if not specs:
        return {}
    obs = extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=0.2)
    for o in obs:
        f = F.get(_line_key(o.element, o.wavelength_nm), 1.0) if per_line else F.get(o.element, 1.0)
        o.intensity = float(o.intensity) * float(f)
        o.intensity_uncertainty = max(float(o.intensity_uncertainty) * float(f), 1e-12)
    res = run_constrained_solver(db, obs, 1e17, fixed_temperature_K=T_star)
    return recovered_wt(res)


def derive_F_at_T(db, standard, T_star: float, *, per_line: bool = False) -> Dict:
    """OPC correction from the standard's *uncorrected* L2+fixed-T recovery at ``T*``.

    ``F_e = C_true_e / C_rec_e`` (renormalized to the standard's modeled basis), clamped to a
    sane band. Per-line keys every line of element ``e`` with ``F_e`` (selection is
    sample-independent, so per-line F is portable).
    """
    _sid, wl, inten, truth = standard
    rec = solve_l1(db, wl, inten, truth, T_star)  # uncorrected L2 + fixed T recovery
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


def select_standard_index(db, samples) -> Tuple[int, float]:
    """A-priori standard pick: the sample with the lowest in-sample uncorrected-L2 RMSEP.

    Uses only each candidate standard's own certified composition (the calibration sample is
    allowed to know its own truth) — never any held-out sample's truth.
    """
    best_i, best_err = 0, float("inf")
    for i, (sid, wl, inten, truth) in enumerate(samples):
        from tests.benchmarks.real_steel.lever_l2_lines import solve_l2

        pred = solve_l2(db, wl, inten, truth) or {}
        err = score([(truth, pred)])["rmsep_overall"]
        print(f"  [std-pick {i}] sample {sid} in-sample L2 rmsep={err:.3f}", flush=True)
        if np.isfinite(err) and err < best_err:
            best_err, best_i = err, i
    return best_i, best_err


def run_best(
    db_path: str = "ASD_da/libs_production.db",
    std_index: int | None = None,
    *,
    t_grid=DEFAULT_T_GRID,
    per_line: bool = False,
    limit: int | None = None,
) -> Dict[str, object]:
    """Honest combined run: pick/derive (T*, F) on one standard, apply held-out, score.

    ``std_index=None`` selects the standard a priori by lowest in-sample uncorrected-L2 RMSEP.
    """
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    samples = list(load_real_steel())
    if limit is not None:
        samples = samples[:limit]

    if std_index is None:
        print(
            "[best] selecting standard a priori (lowest in-sample uncorrected-L2 RMSEP)...",
            flush=True,
        )
        std_index, std_pick_err = select_standard_index(db, samples)
        print(
            f"[best] standard = idx {std_index} (in-sample L2 rmsep {std_pick_err:.3f})", flush=True
        )
    if std_index >= len(samples):
        std_index = 0
    standard = samples[std_index]

    print(f"[best] standard sample {standard[0]} (idx {std_index}); scanning T* ...", flush=True)
    T_star, std_T_rmsep, curve = choose_optimal_T(db, standard, t_grid)
    print(
        f"[best] T* = {T_star:.0f} K (standard L2+T rmsep {std_T_rmsep:.3f}); deriving F ...",
        flush=True,
    )
    F = derive_F_at_T(db, standard, T_star, per_line=per_line)

    in_pred = solve_best(
        db, standard[1], standard[2], standard[3], T_star=T_star, F=F, per_line=per_line
    )
    in_rmsep = score([(standard[3], in_pred or {})])["rmsep_overall"]

    held: List[Tuple[Dict[str, float], Dict[str, float]]] = []
    for i, (sid, wl, inten, truth) in enumerate(samples):
        if i == std_index:
            continue
        pred = solve_best(db, wl, inten, truth, T_star=T_star, F=F, per_line=per_line)
        held.append((truth, pred or {}))
        print(f"  [held-out {len(held)}] sample {sid} done", flush=True)
    held_scores = score(held)

    return {
        "standard_sample": standard[0],
        "std_index": std_index,
        "T_star_K": T_star,
        "per_line": per_line,
        "F": {str(k): v for k, v in F.items()},
        "standard_in_sample_rmsep": in_rmsep,
        "scan_curve": curve,
        "held_out": held_scores,
    }


def best_solve_fn(T_star: float, F: Dict, *, per_line: bool = False):
    """Return a frozen ``solve_fn(db, wl, intensity, truth)`` for the gate harness.

    ``T_star`` / ``F`` are the standard-derived calibration; the returned closure applies them
    unchanged to whatever sample the harness feeds it (the held-out evaluation contract).
    """

    def _fn(db, wl, intensity, truth):
        return solve_best(db, wl, intensity, truth, T_star=T_star, F=F, per_line=per_line)

    return _fn


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="ASD_da/libs_production.db")
    ap.add_argument(
        "--std-index",
        type=int,
        default=None,
        help="standard index; default = a-priori pick (lowest in-sample L2 RMSEP)",
    )
    ap.add_argument("--per-line", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tmin", type=int, default=7000)
    ap.add_argument("--tmax", type=int, default=12000)
    ap.add_argument("--tstep", type=int, default=500)
    a = ap.parse_args()
    grid = tuple(float(t) for t in range(a.tmin, a.tmax + 1, a.tstep))
    out = run_best(
        db_path=a.db_path, std_index=a.std_index, t_grid=grid, per_line=a.per_line, limit=a.limit
    )
    print("\nBEST (L2 lines + L1 optimal-T + L3 OPC F) real-steel results:")
    print(f"  standard: {out['standard_sample']} (idx {out['std_index']})")
    print(f"  T_star_K: {out['T_star_K']:.0f}")
    print(f"  per_line: {out['per_line']}")
    print(f"  standard_in_sample_rmsep: {out['standard_in_sample_rmsep']:.3f}")
    print("  HELD-OUT RMSEP (wt%):")
    hs = out["held_out"]
    for k in sorted(hs):
        v = hs[k]
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
