#!/usr/bin/env python
"""Benchmark the alternative atomic databases on BOTH the optimization (dev) and the held-out
(test) splits — the comparison the earlier M5 work never did (it measured optimization-tier only).

The self-absorption cycle proved a lever can win on dev and regress on held-out; the same could be
true in reverse for a database. So the DECISION metric is the HELD-OUT split: does any alternative
atomic DB beat NIST on data it was never tuned against?

    PYTHONPATH=$PWD .venv/bin/python scripts/benchmark_atomic_dbs.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "scripts/arbor")
from run_eval import score_split  # noqa: E402

DBS = {
    "NIST (28.7k, graded)": "ASD_da/libs_production.db",
    "R4 NIST+VALD (935k)": "output/r4_nist_vald_backfill.db",
    "VALD-complete (1.09M)": "output/vald_complete.db",
    "VALD-gradeB (118k)": "output/vald_gradeB.db",
}
MAX_SPECTRA = 20


def main() -> None:
    print(
        f"\n{'database':24s} {'dev RMSE':>9s} {'HELD-OUT RMSE':>14s}  (lower=better; held-out decides)\n"
    )
    results = {}
    for name, db in DBS.items():
        try:
            dev, _ = score_split("dev", MAX_SPECTRA, db)
        except Exception as e:
            dev = float("nan")
            print(f"  {name:24s} dev FAILED: {type(e).__name__}: {e}")
        try:
            test, per = score_split("test", MAX_SPECTRA, db)
        except Exception as e:
            test, per = float("nan"), {}
            print(f"  {name:24s} test FAILED: {type(e).__name__}: {e}")
        results[name] = {"db": db, "dev_rmse": dev, "test_rmse": test, "test_per_dataset": per}
        print(f"{name:24s} {dev:>9.3f} {test:>14.3f}")

    nist = results["NIST (28.7k, graded)"]["test_rmse"]
    print(f"\nNIST held-out RMSE = {nist:.3f} wt%")
    winners = [
        (n, r["test_rmse"])
        for n, r in results.items()
        if "NIST" not in n and r["test_rmse"] == r["test_rmse"] and r["test_rmse"] < nist - 0.01
    ]
    print("HELD-OUT winners over NIST:", winners or "NONE — NIST still best on held-out")
    with open("output/atomic_db_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote output/atomic_db_benchmark.json")


if __name__ == "__main__":
    main()
