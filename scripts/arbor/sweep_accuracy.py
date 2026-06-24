#!/usr/bin/env python
"""Overnight LLM-free ACCURACY-lever sweep (companion to sweep_calib.py). The CLIAPIProxy cooldowns
under sustained load so `arbor run` can't complete overnight; this evaluates the accuracy levers
directly via run_scoreboard's config_overrides on dev + HELD-OUT, no model dependency.

Levers (these can genuinely LOWER composition RMSE, unlike the calib levers which only trade speed):
  - apply_self_absorption='observable'  (self-absorption correction; a measured dev win historically,
    but it REGRESSED held-out before -- so held-out is the real test)
  - grade_aware_selection=True          (weight the Boltzmann fit by atomic-data accuracy grade)
Each config runs in a fresh subprocess; we aggregate exactly like run_eval.score_split (per-dataset
rmse_wt_median, mean across datasets). HELD-OUT decides.

  SWEEP_REPO=<repo> JAX_PLATFORMS=cpu python scripts/arbor/sweep_accuracy.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

REPO = os.environ.get("SWEEP_REPO", os.getcwd())
DB = os.environ.get("SWEEP_DB", "ASD_da/libs_production.db")
MAXSPEC = int(os.environ.get("SWEEP_MAXSPEC", "20"))
OUT = os.environ.get("SWEEP_OUT", "/mnt/nfs/shared/ai/cf-libs-bench/accuracy_sweep_results.json")

# config name -> run_scoreboard config_overrides
CONFIGS = {
    "baseline": {},
    "self_absorption_observable": {"apply_self_absorption": "observable"},
    "grade_aware": {"grade_aware_selection": True},
    "self_abs+grade": {"apply_self_absorption": "observable", "grade_aware_selection": True},
}
SPLITS = ["dev", "test"]
SPLIT_DATASETS = {
    "dev": (["supercam_labcal", "aalto"], False),
    "test": (["supercam_scct", "bhvo2_chemcam"], True),
}

SNIP = """
import time, json
from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.scoreboard import run_scoreboard, ensure_default_datasets
ensure_default_datasets()
datasets, incl = {ds!r}
db = AtomicDatabase({db!r})
t = time.time()
board = run_scoreboard(db, datasets=datasets, max_spectra={n}, seed=7,
                       pipeline_impl="reference", include_holdout=incl, config_overrides={cfg!r})
per = {{}}
for d in board.get("datasets", []):
    m = (d.get("composition") or {{}}).get("rmse_wt_median")
    if m is not None:
        per[d["name"]] = float(m)
rmse = sum(per.values())/len(per) if per else None
print("SWEEPJSON" + json.dumps({{"rmse": rmse, "per": per, "sec": round(time.time()-t, 1)}}))
"""


def run_one(cfg: dict, split: str) -> dict:
    ds = SPLIT_DATASETS[split]
    code = SNIP.format(ds=ds, db=DB, n=MAXSPEC, cfg=cfg)
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=REPO)
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=5400,
        )
        for line in res.stdout.splitlines():
            if line.startswith("SWEEPJSON"):
                return json.loads(line[len("SWEEPJSON") :])
        return {"error": (res.stdout + res.stderr)[-400:]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> None:
    results: dict = {"_meta": {"repo": REPO, "db": DB, "max_spectra": MAXSPEC}}
    print(
        f"accuracy sweep: {len(CONFIGS)} configs x {len(SPLITS)} splits, n={MAXSPEC}\n", flush=True
    )
    for name, cfg in CONFIGS.items():
        results[name] = {}
        for split in SPLITS:
            r = run_one(cfg, split)
            results[name][split] = r
            print(
                f"  {name:28s} {split:4s}  rmse={r.get('rmse')}  sec={r.get('sec')}"
                + (f"  ERR {r['error'][:80]}" if r.get("error") else ""),
                flush=True,
            )
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
    base = results.get("baseline", {}).get("test", {}).get("rmse")
    print(f"\nHELD-OUT baseline RMSE = {base}")
    for name in CONFIGS:
        t = results.get(name, {}).get("test", {}).get("rmse")
        if t is not None and base is not None and name != "baseline":
            verdict = (
                "IMPROVES" if t < base - 0.01 else ("regresses" if t > base + 0.01 else "~tie")
            )
            print(f"  {name:28s} held-out {t:.3f} vs {base:.3f} -> {verdict}")
    print(f"\nDONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
