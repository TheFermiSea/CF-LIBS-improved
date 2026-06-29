"""Lever L1 — optimal-temperature / OPC (real-steel gate; Zhao 2018).

Root cause (docs/research/real-steel-accuracy-levers.md): CF-LIBS composition is
**hypersensitive to plasma T**. Trace minors observed only via ION lines bias the
Boltzmann-slope fit low (~6705 K); the Saha ion->total back-correction
``N_I ∝ N_II · N_e · exp(E_ion / kT)`` then explodes and the minor soaks the
closure. Zhao et al. 2018 (Plasma Sci. Technol. 20 035502) show the same effect
on a Cu-Pb alloy (Pb-only lines -> 6700 K, Cu lines -> 9300 K) and that picking
the *optimal* T (which minimizes a matrix-matched standard's composition error)
cuts relative errors from 12-32% down to 1.8-13.4%.

L1 HOLDS the plasma temperature at a fixed value instead of recovering it from
the (ion-biased) slope, via the physics-only ``fixed_temperature_K`` option added
to ``IterativeCFLIBSSolver`` (threaded through ``run_constrained_solver``).

HONEST OPC EVALUATION (no per-sample overfitting):
  1. Treat ONE sample as the matrix-matched standard.
  2. Scan T over [7000, 12000] K; pick T* that minimizes THAT standard's
     composition error vs its certified truth.
  3. Apply T* UNCHANGED to ALL OTHER samples and report the HELD-OUT
     ``rmsep_overall``. T* is never tuned to any held-out sample's truth.

The held-out RMSEP is the un-overfittable headline. The in-sample (standard) RMSEP
at T* is reported for context only.

Line policy: imports L2's committed neutral-anchor selection (``select_l2_lines``).
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
    score,
)
from tests.benchmarks.real_steel.lever_l2_lines import select_l2_lines  # noqa: E402

# Optimal-T scan grid (K). A coarse, physically-reasonable steel-plasma range;
# the chosen T* is derived from ONE standard only, never from held-out truth.
DEFAULT_T_GRID = tuple(float(t) for t in range(7000, 12001, 500))


def solve_l1(db, wl, intensity, truth, fixed_temperature_K: float) -> Dict[str, float]:
    """L1 solve: L2 neutral-anchor lines + constrained Saha-Boltzmann at fixed T."""
    els = list(truth.keys())
    window = (float(wl.min()), float(wl.max()))
    specs: List[LineSpec] = []
    for e in els:
        specs.extend(select_l2_lines(db, e, window, 8))
    if not specs:
        return {}
    obs = extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=0.2)
    res = run_constrained_solver(db, obs, 1e17, fixed_temperature_K=fixed_temperature_K)
    return recovered_wt(res)


def _sample_rmsep(db, wl, inten, truth, T_K: float) -> float:
    """Single-sample overall RMSEP (wt%) at fixed ``T_K`` (renormalized like score)."""
    pred = solve_l1(db, wl, inten, truth, T_K)
    return score([(truth, pred or {})])["rmsep_overall"]


def choose_optimal_T(
    db, standard, t_grid=DEFAULT_T_GRID
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """Scan T on the matrix-matched standard; return (T*, in-sample RMSEP at T*, curve)."""
    _sid, wl, inten, truth = standard
    curve: List[Tuple[float, float]] = []
    best_T, best_err = float(t_grid[0]), float("inf")
    for T_K in t_grid:
        err = _sample_rmsep(db, wl, inten, truth, T_K)
        curve.append((float(T_K), float(err)))
        print(f"  scan T={T_K:.0f}K  standard RMSEP={err:.3f}", flush=True)
        if np.isfinite(err) and err < best_err:
            best_err, best_T = err, float(T_K)
    return best_T, best_err, curve


def run_opc(
    db_path: str = "ASD_da/libs_production.db",
    std_index: int = 0,
    t_grid=DEFAULT_T_GRID,
    limit: int | None = None,
) -> Dict[str, object]:
    """Full honest-OPC run: derive T* on one standard, apply held-out, score."""
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    samples = list(load_real_steel())
    if limit is not None:
        samples = samples[:limit]
    if std_index >= len(samples):
        std_index = 0
    standard = samples[std_index]

    print(f"[L1] standard = sample {standard[0]} (index {std_index}); scanning T ...", flush=True)
    T_star, in_sample_rmsep, curve = choose_optimal_T(db, standard, t_grid)
    print(f"[L1] T* = {T_star:.0f} K (standard in-sample RMSEP {in_sample_rmsep:.3f})", flush=True)

    # Held-out: apply T* unchanged to every OTHER sample.
    held: List[Tuple[Dict[str, float], Dict[str, float]]] = []
    for i, (sid, wl, inten, truth) in enumerate(samples):
        if i == std_index:
            continue
        pred = solve_l1(db, wl, inten, truth, T_star)
        held.append((truth, pred or {}))
        print(f"  [held-out {len(held)}] sample {sid} done", flush=True)
    held_scores = score(held)

    return {
        "T_star_K": T_star,
        "standard_sample": standard[0],
        "standard_in_sample_rmsep": in_sample_rmsep,
        "held_out": held_scores,
        "scan_curve": curve,
    }


def scan_heldout_T(db, samples, std_index, t_grid) -> List[Tuple[float, float]]:
    """DIAGNOSTIC ONLY (never used to pick T*): held-out overall RMSEP vs fixed T.

    Characterizes the best-achievable fixed-T ceiling so the honest single-standard
    OPC result can be reported against it. This function peeks at held-out truth and
    is therefore strictly informational — ``run_opc`` selects T* from the standard
    alone.
    """
    out: List[Tuple[float, float]] = []
    for T_K in t_grid:
        held = []
        for i, (_sid, wl, inten, truth) in enumerate(samples):
            if i == std_index:
                continue
            held.append((truth, solve_l1(db, wl, inten, truth, T_K) or {}))
        ov = score(held)["rmsep_overall"]
        out.append((float(T_K), float(ov)))
        print(f"  [diag] held-out overall @ T={T_K:.0f}K = {ov:.3f}", flush=True)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="ASD_da/libs_production.db")
    ap.add_argument("--std-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tmin", type=int, default=7000)
    ap.add_argument("--tmax", type=int, default=12000)
    ap.add_argument("--tstep", type=int, default=500)
    ap.add_argument(
        "--diag-heldout",
        action="store_true",
        help="diagnostic: scan held-out overall vs T (NOT used to pick T*)",
    )
    a = ap.parse_args()
    grid = tuple(float(t) for t in range(a.tmin, a.tmax + 1, a.tstep))
    if a.diag_heldout:
        import warnings as _w

        _w.filterwarnings("ignore")
        from cflibs.atomic.database import AtomicDatabase as _ADB

        _db = _ADB(a.db_path)
        _samples = list(load_real_steel())
        if a.limit is not None:
            _samples = _samples[: a.limit]
        print("[L1 DIAG] held-out overall RMSEP vs fixed T (informational only):")
        scan_heldout_T(_db, _samples, a.std_index, grid)
    out = run_opc(db_path=a.db_path, std_index=a.std_index, t_grid=grid, limit=a.limit)
    print("\nL1 (optimal-T / OPC) real-steel results:")
    print(f"  T_star_K: {out['T_star_K']:.0f}")
    print(f"  standard: {out['standard_sample']}")
    print(f"  standard_in_sample_rmsep: {out['standard_in_sample_rmsep']:.3f}")
    print("  HELD-OUT RMSEP (wt%):")
    hs = out["held_out"]
    for k in sorted(hs):
        v = hs[k]
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
