"""Optuna DED composition-ACCURACY optimization (physics-first; tuning-tooling only).

PHYSICS-FIRST: this script lives in scripts/ and is NEVER imported by shipped cflibs/.
Optuna is process-tooling. The objective only tunes physics-meaningful knobs
(line-selection + solver config) and is gated by composition accuracy ALONE
(no latency term) — accuracy is the absolute priority.

Objective (MINIMIZE), Ti-6Al-4V clean Al-scan:
  2.0*V/Ti_ratio_err + 1.0*Al/Ti_ratio_err + 1.0*|drift_slope-1| + 1.0*(1-R2)
Baseline (current HEAD): V/Ti 0.25, Al/Ti 0.099, slope 1.003, R2 0.939 -> ~0.663.
V/Ti is up-weighted: it is the limiting channel.

Run (one worker, shared NFS-safe journal storage):
  PYTHONPATH=<repo>:<optlibs> JAX_PLATFORMS=cpu python scripts/optuna_ded_precision.py \
      --storage /cluster/shared/cf-libs-v4/optuna_ded.journal --n-trials 12
Many SLURM array workers share the same --storage journal (Optuna handles concurrency).
"""
from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import optuna  # noqa: E402

from tests.benchmarks.ded_precision.benchmark_runner import run_composition_series  # noqa: E402

DB = os.environ.get("CFLIBS_DB", "ASD_da/libs_production.db")
ALLOY, AXIS = "Ti-6Al-4V", "Al"


def _metrics(df):
    """Return (Al/Ti rel-err, V/Ti rel-err, drift slope, R2) for the clean Al-scan."""
    pts = sorted(df["target_value"].unique())
    al = df[df["element"] == "Al"]
    true_al = al["target_value"].to_numpy(dtype=float)
    pred_al = al["pred_wt"].to_numpy(dtype=float)
    if len(set(true_al)) < 2 or not np.all(np.isfinite(pred_al)):
        return float("nan"), float("nan"), float("nan"), float("nan")
    slope, inter = np.polyfit(true_al, pred_al, 1)
    ss_res = np.sum((pred_al - (slope * true_al + inter)) ** 2)
    ss_tot = np.sum((pred_al - pred_al.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    alti, vti = [], []
    for p in pts:
        sub = df[df["target_value"] == p]

        def m(el, col, _sub=sub):
            r = _sub[_sub["element"] == el]
            return float(r[col].mean()) if len(r) else float("nan")

        truth = {el: m(el, "pred_wt") - m(el, "error") for el in ("Ti", "Al", "V")}
        pred = {el: m(el, "pred_wt") for el in ("Ti", "Al", "V")}
        alti.append(abs(pred["Al"] / pred["Ti"] - truth["Al"] / truth["Ti"]) / (truth["Al"] / truth["Ti"]))
        vti.append(abs(pred["V"] / pred["Ti"] - truth["V"] / truth["Ti"]) / (truth["V"] / truth["Ti"]))
    return float(np.mean(alti)), float(np.mean(vti)), float(slope), float(r2)


def objective(trial):
    budget = {
        "Ti": trial.suggest_int("Ti", 8, 16),
        "Al": trial.suggest_int("Al", 4, 12),
        "V": trial.suggest_int("V", 4, 14),
    }
    prefer_spread = trial.suggest_categorical("prefer_spread", [True, False])
    line_T_K = trial.suggest_float("line_T_K", 8000.0, 14000.0)
    closure_mode = trial.suggest_categorical("closure_mode", ["standard", "ilr", "matrix"])
    max_iterations = trial.suggest_int("max_iterations", 20, 60)
    saha_graph = trial.suggest_categorical("saha_graph", [True, False])
    try:
        df = run_composition_series(
            DB, ALLOY, AXIS, clean=True,
            budget=budget, prefer_spread=prefer_spread, line_T_K=line_T_K,
            closure_mode=closure_mode, max_iterations=max_iterations,
            saha_boltzmann_graph=saha_graph,
        )
        alti, vti, slope, r2 = _metrics(df)
    except Exception as exc:  # noqa: BLE001 — a bad config is data, not a crash
        trial.set_user_attr("error", str(exc)[:120])
        return 1e3
    if not all(np.isfinite(x) for x in (alti, vti, slope, r2)):
        return 1e3
    trial.set_user_attr("alti", alti)
    trial.set_user_attr("vti", vti)
    trial.set_user_attr("slope", slope)
    trial.set_user_attr("r2", r2)
    return 2.0 * vti + 1.0 * alti + 1.0 * abs(slope - 1.0) + 1.0 * (1.0 - r2)


def _make_storage(path):
    try:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        return JournalStorage(JournalFileBackend(path))
    except Exception:  # noqa: BLE001 — fallback for older/newer optuna layouts
        return f"sqlite:///{path}.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", required=True, help="journal file path (shared across workers)")
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--study", default="ded_precision")
    a = ap.parse_args()
    study = optuna.create_study(
        study_name=a.study, storage=_make_storage(a.storage),
        direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
    )
    study.optimize(objective, n_trials=a.n_trials, gc_after_trial=True)
    print("WORKER DONE. best_value=", study.best_value, "best_params=", study.best_params,
          "best_attrs=", study.best_trial.user_attrs, flush=True)


if __name__ == "__main__":
    main()
