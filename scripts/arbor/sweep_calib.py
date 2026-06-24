#!/usr/bin/env python
"""Overnight LLM-free calibration-lever sweep — the robust alternative to `arbor run` (the CLIAPIProxy
cooldowns under sustained load: both claude and codex go auth_unavailable after ~30 turns).

The new calibration levers are env-toggleable (CFLIBS_HOUGH_CALIB / CFLIBS_RANSAC_EARLY_EXIT /
CFLIBS_CALIB_POOL_CACHE), so we can evaluate every flag combination on BOTH the dev and the HELD-OUT
(test) tiers via the production score_split, with NO code changes and no model dependency. Each config
runs in a FRESH subprocess so the env flags are read cleanly at pipeline-config build time. We log the
composition RMSE (accuracy) AND wall-time (speed) so the decision is: does any config hold-or-improve
held-out RMSE while being faster?

  SWEEP_REPO=<repo> JAX_PLATFORMS=cpu python scripts/arbor/sweep_calib.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO = os.environ.get("SWEEP_REPO", os.getcwd())
DB = os.environ.get("SWEEP_DB", "ASD_da/libs_production.db")
MAXSPEC = int(os.environ.get("SWEEP_MAXSPEC", "20"))
OUT = os.environ.get("SWEEP_OUT", "/mnt/nfs/shared/ai/cf-libs-bench/calib_sweep_results.json")

CONFIGS = {
    "baseline (all off)": {},
    "pool_cache": {"CFLIBS_CALIB_POOL_CACHE": "1"},
    "hough": {"CFLIBS_HOUGH_CALIB": "1"},
    "early_exit": {"CFLIBS_RANSAC_EARLY_EXIT": "1"},
    "hough+early": {"CFLIBS_HOUGH_CALIB": "1", "CFLIBS_RANSAC_EARLY_EXIT": "1"},
    "hough+early+pool": {
        "CFLIBS_HOUGH_CALIB": "1",
        "CFLIBS_RANSAC_EARLY_EXIT": "1",
        "CFLIBS_CALIB_POOL_CACHE": "1",
    },
}
SPLITS = ["dev", "test"]

# Fresh-process snippet: score one split, print the result as a tagged JSON line.
SNIP = (
    "import sys,time,json; sys.path.insert(0,'scripts/arbor'); "
    "from run_eval import score_split; "
    "t=time.time(); r,p=score_split('{split}',{n},'{db}'); "
    "print('SWEEPJSON'+json.dumps({{'rmse':r,'per':p,'sec':round(time.time()-t,1)}}))"
)


def run_one(flags: dict, split: str) -> dict:
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=REPO, **flags)
    cmd = [sys.executable, "-c", SNIP.format(split=split, n=MAXSPEC, db=DB)]
    try:
        res = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=5400)
        for line in res.stdout.splitlines():
            if line.startswith("SWEEPJSON"):
                return json.loads(line[len("SWEEPJSON") :])
        return {"error": (res.stdout + res.stderr)[-400:]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> None:
    results: dict = {"_meta": {"repo": REPO, "db": DB, "max_spectra": MAXSPEC}}
    print(
        f"sweep: {len(CONFIGS)} configs x {len(SPLITS)} splits, max_spectra={MAXSPEC}\n", flush=True
    )
    for name, flags in CONFIGS.items():
        results[name] = {}
        for split in SPLITS:
            r = run_one(flags, split)
            results[name][split] = r
            print(
                f"  {name:20s} {split:4s}  rmse={r.get('rmse')}  sec={r.get('sec')}"
                + (f"  ERR {r['error'][:80]}" if r.get("error") else ""),
                flush=True,
            )
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
    # verdict
    base = results.get("baseline (all off)", {}).get("test", {}).get("rmse")
    print(f"\nHELD-OUT baseline RMSE = {base}")
    for name in CONFIGS:
        t = results.get(name, {}).get("test", {}).get("rmse")
        d = results.get(name, {}).get("dev", {}).get("rmse")
        if t is not None and base is not None and name != "baseline (all off)":
            tag = "HOLDS/IMPROVES" if t <= base + 0.02 else "REGRESSES"
            print(f"  {name:20s} held-out {t:.3f} (dev {d}) -> {tag} accuracy")
    print(f"\nDONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
