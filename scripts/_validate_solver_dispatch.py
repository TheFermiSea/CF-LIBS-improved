"""Validate unified solver dispatch: each solver returns a finite wt% RMSE.

Run as:  PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/_validate_solver_dispatch.py [solver ...]
"""

import sys

from cflibs.atomic import AtomicDatabase
from cflibs.benchmark.scoreboard import run_scoreboard

SOLVERS = sys.argv[1:] or ["iterative", "closed_form", "joint", "bayesian"]

#: Keep the full-spectrum solvers cheap enough for a smoke run (they are the
#: GPU-targeted solvers; on a CPU 7933-pixel SuperCam axis they are slow).
_SOLVER_OVERRIDES = {
    "joint": {"max_iterations_joint": 40},
    "bayesian": {"num_warmup": 50, "num_samples": 50},
}


def main() -> None:
    db = AtomicDatabase("ASD_da/libs_production.db")
    for solver in SOLVERS:
        overrides = {"solver": solver}
        if solver in _SOLVER_OVERRIDES:
            overrides["solver_overrides"] = _SOLVER_OVERRIDES[solver]
        try:
            board = run_scoreboard(
                db,
                datasets=["supercam_labcal"],
                max_spectra=3,
                seed=7,
                pipeline_impl="reference",
                config_overrides=overrides,
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
