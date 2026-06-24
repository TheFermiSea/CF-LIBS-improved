"""M5 core confound-controlled comparison: NIST vs VALD-complete.

Identical run_scoreboard call for both DBs; vary ONLY the db path.
"""
import json
import sys

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.scoreboard import run_scoreboard

DBS = {
    "NIST": "/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db",
    "VALD": "/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/output/vald_complete.db",
}
DATASETS = ["bhvo2_chemcam", "supercam_labcal"]
MAX_SPECTRA = int(sys.argv[1]) if len(sys.argv) > 1 else 4

results = {}
for tag, path in DBS.items():
    print(f"=== {tag} :: {path} ===", flush=True)
    db = AtomicDatabase(path)
    print(f"  cflibs={AtomicDatabase.__module__}", flush=True)
    board = run_scoreboard(
        db,
        datasets=DATASETS,
        include_holdout=True,
        max_spectra=MAX_SPECTRA,
        pipeline_impl="reference",
    )
    results[tag] = board
    for ds in board["datasets"]:
        comp = ds["composition"]
        print(
            f"  {tag} {ds['name']}: n_ok={ds['n_ok']} n_failed={ds['n_failed']} "
            f"rmse_med={comp['rmse_wt_median'] if comp else None}",
            flush=True,
        )

out = "/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/output/m5_core_compare.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"WROTE {out}", flush=True)
