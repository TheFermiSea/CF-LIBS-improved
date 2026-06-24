#!/usr/bin/env python
"""Optuna over the SOLVER x full-param-space, via the unified dispatch (config_overrides={'solver':X}).
Optimizes the DEV split (lower composition RMSE = better); the best config is then validated on the
HELD-OUT tier (the merge gate). The forward-model solvers (joint/bayesian) do NOT converge on real
data (they fall back to the iterative warm-start), so this study searches the solvers that actually
produce real fits: iterative (full 18-knob space) + closed_form (ILR)."""

from __future__ import annotations
import argparse
import json
import optuna


def _score(db, datasets, n, ov):
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.benchmark.scoreboard import run_scoreboard, ensure_default_datasets

    ensure_default_datasets()
    board = run_scoreboard(
        AtomicDatabase(db),
        datasets=datasets,
        max_spectra=n,
        seed=7,
        pipeline_impl="reference",
        config_overrides=ov,
    )
    rs = [(d.get("composition") or {}).get("rmse_wt_median") for d in board.get("datasets", [])]
    rs = [r for r in rs if r is not None]
    return sum(rs) / len(rs) if rs else None


def objective(trial, db, n):
    solver = trial.suggest_categorical("solver", ["iterative", "closed_form"])
    ov = {"solver": solver}
    if solver == "iterative":
        ov.update(
            closure_mode=trial.suggest_categorical("closure_mode", ["standard", "oxide", "matrix"]),
            max_iterations=trial.suggest_int("max_iterations", 8, 40),
            t_tolerance_k=trial.suggest_float("t_tolerance_k", 20.0, 300.0),
            boltzmann_weight_cap=trial.suggest_float("boltzmann_weight_cap", 0.0, 15.0),
            min_boltzmann_r2=trial.suggest_float("min_boltzmann_r2", 0.0, 0.9),
            apply_self_absorption=trial.suggest_categorical(
                "apply_self_absorption", ["off", "observable"]
            ),
            saha_boltzmann_graph=trial.suggest_categorical("saha_boltzmann_graph", [True, False]),
            ransac_early_exit=True,  # the merged held-out win
        )
        ov["solver_overrides"] = {
            "apply_ipd": trial.suggest_categorical("apply_ipd", [True, False]),
            "aki_uncertainty_weighting": trial.suggest_categorical("aki_unc_w", [True, False]),
        }
    else:
        ov["solver_overrides"] = {
            "closure_mode": trial.suggest_categorical("cf_closure", ["standard", "oxide"]),
            "saha_passes": trial.suggest_int("cf_saha_passes", 1, 5),
            "apply_ipd": trial.suggest_categorical("cf_ipd", [True, False]),
        }
    r = _score(db, ["supercam_labcal", "aalto"], n, ov)
    trial.set_user_attr("overrides", json.dumps(ov))
    return r if (r and r == r) else 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--max-spectra", type=int, default=12)
    ap.add_argument("--db", default="ASD_da/libs_production.db")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--study-name", default="solvers")
    ap.add_argument("--out", default="/cluster/shared/ai/cf-libs-bench/optuna_solvers_result.json")
    a = ap.parse_args()
    study = optuna.create_study(
        direction="minimize",
        study_name=a.study_name,
        storage=a.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=7, multivariate=True),
    )

    def cb(st, tr):
        print(
            f"  trial {tr.number}: {tr.params.get('solver')} rmse={tr.value} (best={st.best_value:.3f})",
            flush=True,
        )

    study.optimize(lambda t: objective(t, a.db, a.max_spectra), n_trials=a.n_trials, callbacks=[cb])
    best = json.loads(study.best_trial.user_attrs["overrides"])
    print(f"\nBEST dev RMSE={study.best_value:.3f}  config={best}", flush=True)
    held = _score(a.db, ["supercam_scct", "bhvo2_chemcam"], a.max_spectra, best)
    print(f"BEST on HELD-OUT={held}  (trunk held-out ~2.42 wt%)", flush=True)
    json.dump(
        {
            "best_dev": study.best_value,
            "best_config": best,
            "best_held_out": held,
            "n_trials": len(study.trials),
        },
        open(a.out, "w"),
        indent=2,
    )
    print(f"STUDY COMPLETE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
