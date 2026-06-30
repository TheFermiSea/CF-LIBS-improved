"""Honest ON/OFF benchmark of the opt-in DB-isolation gate on the DED floor.

Reuses the ``tests/benchmarks/ded_precision`` harness for spectrum generation,
line extraction, and metrics, but constructs ``IterativeCFLIBSSolver`` with
``db_isolation_gate`` True/False and ``saha_boltzmann_graph`` True/False so the
gate's effect can be measured on:

  * the **SB-graph** path (the DED + shipped real-Ti production default), and
  * the **weighted common-slope** path (where the gate's inverse-variance
    mechanism actually bites).

Run from the worktree root (avoids the scripts/ sys.path worktree trap):

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/bench_db_isolation_gate.py

Requires ``ASD_da/libs_production.db``. Prints per-element RMSEP/bias plus the
ON-minus-OFF delta (negative = gate helps).
"""

from __future__ import annotations

import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.pipeline import _number_to_mass_fractions
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

from tests.benchmarks.ded_precision.alloy_definitions import (
    ALLOY_WINDOWS_NM,
    COMPOSITION_SERIES,
    elements_of,
)
from tests.benchmarks.ded_precision.benchmark_runner import summarize_series
from tests.benchmarks.ded_precision.line_extractor import extract_line_intensities
from tests.benchmarks.ded_precision.line_lists import build_alloy_line_list
from tests.benchmarks.ded_precision.solver_runner import make_ne_diagnostic
from tests.benchmarks.ded_precision.spectrum_generator import (
    clean_spectrum,
    default_grid,
    make_forward,
)

DB_PATH = "ASD_da/libs_production.db"


def run_series(
    db_path: str,
    alloy: str,
    axis: str,
    *,
    saha_boltzmann_graph: bool,
    db_isolation_gate: bool,
    T_K: float = 11000.0,
    ne_cm3: float = 1e17,
    instrument_fwhm_nm: float = 0.1,
    grid_step_nm: float = 0.02,
    max_iterations: int = 30,
) -> pd.DataFrame:
    db = AtomicDatabase(db_path)
    els = elements_of(alloy)
    wl = default_grid(ALLOY_WINDOWS_NM[alloy], grid_step_nm)
    fwd = make_forward(db_path, els, wl, instrument_fwhm_nm)
    line_specs = [
        s
        for v in build_alloy_line_list(db, alloy, T_K=T_K, prefer_spread=False).values()
        for s in v
    ]
    rows: List[Dict[str, object]] = []
    for ci, comp in enumerate(COMPOSITION_SERIES[alloy][axis]):
        target_val = comp[axis]
        spec = clean_spectrum(fwd, comp, els, T_K, ne_cm3)
        obs = extract_line_intensities(wl, spec, line_specs, instrument_fwhm_nm=instrument_fwhm_nm)
        solver = IterativeCFLIBSSolver(
            db,
            saha_boltzmann_graph=saha_boltzmann_graph,
            max_iterations=max_iterations,
            apply_self_absorption="off",
            db_isolation_gate=db_isolation_gate,
            db_isolation_fwhm_nm=instrument_fwhm_nm,
        )
        try:
            res = solver.solve(
                list(obs),
                closure_mode="standard",
                stark_diagnostics=[make_ne_diagnostic(ne_cm3)],
            )
            if not res.mass_fractions:
                res.mass_fractions = _number_to_mass_fractions(res.concentrations)
            pred = {k: 100.0 * v for k, v in res.mass_fractions.items()}
            conv = bool(res.converged)
        except Exception as exc:  # noqa: BLE001 — a failed solve is data, not a crash
            pred = {e: float("nan") for e in els}
            conv = False
            print(f"  solve failed at {axis}={target_val}: {str(exc)[:90]}")
        for e in els:
            rows.append(
                {
                    "element": e,
                    "comp_index": ci,
                    "target_value": float(target_val),
                    "truth_wt": float(comp[e]),
                    "pred_wt": float(pred.get(e, float("nan"))),
                    "error": float(pred.get(e, float("nan")) - comp[e]),
                    "converged": conv,
                }
            )
    return pd.DataFrame(rows)


def overall_rmsep(summ: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(summ["rmsep"].to_numpy(dtype=float) ** 2)))


def main() -> int:
    alloy, axis = "Ti-6Al-4V", "Al"
    for label, sbg in (("SB-graph (DED default)", True), ("common-slope (weighted)", False)):
        print(f"\n=== fit path: {label} | saha_boltzmann_graph={sbg} ===")
        summaries = {}
        for gate in (False, True):
            summ = summarize_series(
                run_series(DB_PATH, alloy, axis, saha_boltzmann_graph=sbg, db_isolation_gate=gate)
            )
            summaries[gate] = summ
            print(
                f"\n  gate {'ON ' if gate else 'OFF'}: overall_RMSEP={overall_rmsep(summ):.4f} wt%"
            )
            print(summ[["rmsep", "bias", "maxae"]].round(4).to_string().replace("\n", "\n    "))
        off, on = summaries[False], summaries[True]
        print("\n  --- delta (ON - OFF), negative = gate helps ---")
        for el in off.index:
            print(
                f"    {el:>3}: dRMSEP={on.loc[el, 'rmsep'] - off.loc[el, 'rmsep']:+.4f}  "
                f"dbias={on.loc[el, 'bias'] - off.loc[el, 'bias']:+.4f}"
            )
        print(f"    overall dRMSEP={overall_rmsep(on) - overall_rmsep(off):+.4f} wt%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
