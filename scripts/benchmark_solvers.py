#!/usr/bin/env python
"""Solver head-to-head: classic iterative vs ILR closed-form vs corrected C-sigma, on the SAME
detected lines, scored in BOTH bases (user choice). On supercam_labcal (the only multi-element
dataset with local data).

Bases (resolved from the code, not guessed):
- CFLIBSResult.concentrations are MASS fractions (composition_errors x100 -> wt%, vs element-wt%
  truth). solve_csigma_composition returns NUMBER fractions -> converted to mass for the
  production comparison, and the others converted to number for the element-number comparison.
- PRODUCTION (mass): the scoreboard metric -- RMSE of 100*mass_frac vs certified wt% over the
  certified elements (dropped element = full error). Includes iterative/ILR oxide closure.
- ELEMENT-NUMBER: the raw-solver metric -- same RMSE in number-fraction space (truth de-converted
  with mass_to_number_fractions). Isolates the inversion from the oxide-closure step.

Shared front-end: run_pipeline (iterative baseline) with a spy on LineSelector.select captures the
exact LineObservation list; ILR + C-sigma run on THOSE lines so the only variable is the solver.
C-sigma uses solve_csigma_composition (the U_s(T)-CORRECTED solver), NOT the raw fit_csigma whose
relative_concentrations are C_s/U_s.

    PYTHONPATH=$PWD .venv/bin/python scripts/benchmark_solvers.py
"""

from __future__ import annotations

import json
import math
import statistics as st

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.scoreboard import (
    _sample_indices,
    ensure_default_datasets,
    iter_datasets,
)
from cflibs.benchmark.synthetic_corpus import STANDARD_MASSES, mass_to_number_fractions
from cflibs.inversion.physics.csigma import solve_csigma_composition
from cflibs.inversion.physics.line_selection import LineSelector
from cflibs.inversion.pipeline import build_pipeline_config, run_pipeline
from cflibs.inversion.solve.closed_form import ClosedFormConfig, ClosedFormILRSolver

DB = "ASD_da/libs_production.db"
DATASET = "supercam_labcal"
MAX_SPECTRA = 15
SEED = 7


def _number_to_mass(num: dict[str, float]) -> dict[str, float]:
    w = {el: f * STANDARD_MASSES[el] for el, f in num.items() if el in STANDARD_MASSES}
    tot = sum(w.values())
    return {el: v / tot for el, v in w.items()} if tot > 0 else {}


def _mass_to_number(mass: dict[str, float]) -> dict[str, float]:
    pos = {el: v for el, v in mass.items() if v > 0 and el in STANDARD_MASSES}
    return mass_to_number_fractions(pos) if pos else {}


def _rmse_wt(pred_frac: dict[str, float], truth_wt: dict[str, float]) -> float:
    """RMSE of 100*pred_frac vs certified wt% over the certified elements (scoreboard metric)."""
    errs = [100.0 * pred_frac.get(el, 0.0) - cert for el, cert in truth_wt.items()]
    return math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else float("nan")


def main() -> None:
    ensure_default_datasets()
    db = AtomicDatabase(DB)
    entry = next(iter_datasets(names=[DATASET]))
    items = list(entry.adapter_factory())
    items = [items[i] for i in _sample_indices(len(items), MAX_SPECTRA, SEED)]

    orig = LineSelector.select
    cap: dict = {}

    def spy(self, observations, **kw):
        r = orig(self, observations, **kw)
        cap["obs"] = r.selected_lines
        return r

    mass_rmse: dict[str, list] = {"iterative": [], "ilr": [], "csigma": []}
    num_rmse: dict[str, list] = {"iterative": [], "ilr": [], "csigma": []}
    fails: dict[str, int] = {"ilr": 0, "csigma": 0}

    LineSelector.select = spy
    try:
        for sid, wl, inten, truth in items:
            cap.clear()
            truth_wt = dict(truth.composition_wt)
            truth_num_wt = {el: 100.0 * f for el, f in _mass_to_number(truth_wt).items()}
            try:
                res, _diag = run_pipeline(
                    wl, inten, db, build_pipeline_config(list(truth_wt.keys()))
                )
            except Exception:
                continue
            obs = cap.get("obs", [])
            if not obs:
                continue

            itr_mass = dict(res.concentrations)  # iterative, oxide closure, MASS fractions
            mass_rmse["iterative"].append(_rmse_wt(itr_mass, truth_wt))
            num_rmse["iterative"].append(_rmse_wt(_mass_to_number(itr_mass), truth_num_wt))

            try:
                ilr = ClosedFormILRSolver(db, config=ClosedFormConfig(closure_mode="standard"))
                ilr_mass = dict(ilr.solve(obs).concentrations)
                mass_rmse["ilr"].append(_rmse_wt(ilr_mass, truth_wt))
                num_rmse["ilr"].append(_rmse_wt(_mass_to_number(ilr_mass), truth_num_wt))
            except Exception:
                fails["ilr"] += 1

            try:
                out = solve_csigma_composition(obs, db)  # U_s(T)-corrected; NUMBER fractions
                if out is not None:
                    cs_num = out[0]
                    mass_rmse["csigma"].append(_rmse_wt(_number_to_mass(cs_num), truth_wt))
                    num_rmse["csigma"].append(_rmse_wt(dict(cs_num), truth_num_wt))
                else:
                    fails["csigma"] += 1
            except Exception:
                fails["csigma"] += 1
    finally:
        LineSelector.select = orig

    def med(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return st.median(xs) if xs else float("nan")

    print(
        f"\nsupercam_labcal n={MAX_SPECTRA} -- median RMSE_wt vs truth (lower=better). Solver head-to-head:\n"
    )
    print(f"  {'solver':12s} {'PRODUCTION (mass)':>18s} {'ELEMENT-NUMBER':>16s} {'n':>4s}")
    summary = {}
    for s in ("iterative", "ilr", "csigma"):
        m, nrm = med(mass_rmse[s]), med(num_rmse[s])
        summary[s] = {"production_mass_rmse": m, "element_number_rmse": nrm, "n": len(mass_rmse[s])}
        print(f"  {s:12s} {m:>18.3f} {nrm:>16.3f} {len(mass_rmse[s]):>4d}")
    print(f"\n  failures: {fails}")
    with open("output/solver_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote output/solver_comparison.json")


if __name__ == "__main__":
    main()
