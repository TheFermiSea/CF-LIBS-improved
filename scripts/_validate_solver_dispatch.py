"""Validate unified solver dispatch: each solver returns a finite wt% RMSE.

Run as:  PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/_validate_solver_dispatch.py [solver ...]
"""

import sys

from cflibs.atomic import AtomicDatabase
from cflibs.benchmark.scoreboard import run_scoreboard

SOLVERS = sys.argv[1:] or ["iterative", "closed_form", "joint", "bayesian"]


def main() -> None:
    db = AtomicDatabase("ASD_da/libs_production.db")
    for solver in SOLVERS:
        try:
            board = run_scoreboard(
                db,
                datasets=["supercam_labcal"],
                max_spectra=3,
                seed=7,
                pipeline_impl="reference",
                config_overrides={"solver": solver},
            )
            comp = board["datasets"][0].get("composition") or {}
            rmse = comp.get("rmse_wt_median")
            n_failed = board["datasets"][0].get("n_failed")
            print(f"{solver:14s} rmse_wt_median={rmse}  n_failed={n_failed}")
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"{solver:14s} ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
