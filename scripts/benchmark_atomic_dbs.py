#!/usr/bin/env python
"""Benchmark the alternative atomic databases on BOTH dev (optimization) and held-out (test)
splits — the comparison the earlier M5 work never did (optimization-tier only). Held-out decides.

SPECTRUM-LEVEL parallelism: every (db, split, spectrum) inversion is independent, so we fan ALL of
them across every core (the reference solver is single-threaded CPU; there is no GPU kernel for it).
The big DBs also needed the wavelength index NIST has (added separately).

    PYTHONPATH=$PWD .venv/bin/python scripts/benchmark_atomic_dbs.py
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import statistics as st
from collections import defaultdict

DBS = {
    "NIST (28.7k, graded)": "ASD_da/libs_production.db",
    "R4 NIST+VALD (935k)": "output/r4_nist_vald_backfill.db",
    "VALD-complete (1.09M)": "output/vald_complete.db",
    "VALD-gradeB (118k)": "output/vald_gradeB.db",
}
SPLITS = {
    "dev": ["supercam_labcal", "aalto"],
    "test": ["supercam_scct", "bhvo2_chemcam"],
}
MAX_SPECTRA = 20
WORKERS = max(1, (os.cpu_count() or 16))

_DBC: dict = {}  # per-process AtomicDatabase cache


def _score_one(task):
    """One inversion: (db, split, dataset, sid, wl, inten, truth) -> rmse_wt or None."""
    db_path, split, dataset, sid, wl, inten, truth = task
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.benchmark.scoreboard import _score_spectrum

    db = _DBC.get(db_path)
    if db is None:
        db = AtomicDatabase(db_path)
        _DBC[db_path] = db
    try:
        rec = _score_spectrum(db, sid, wl, inten, truth, config_overrides=None)
        r = rec.get("rmse_wt") if rec.get("status") == "ok" else None
    except Exception:  # noqa: BLE001
        r = None
    return (db_path, split, dataset, r)


def _load_split(split):
    from cflibs.benchmark.scoreboard import _sample_indices, ensure_default_datasets, iter_datasets

    ensure_default_datasets()
    out = []
    for name in SPLITS[split]:
        entry = next(iter(iter_datasets(names=[name])), None)
        if entry is None:
            continue
        items = list(entry.adapter_factory())
        if not items:
            continue
        idx = _sample_indices(len(items), MAX_SPECTRA, 7)
        if idx is not None:
            items = [items[i] for i in idx]
        for sid, wl, inten, truth in items:
            out.append((name, sid, wl, inten, truth))
    return out


def main() -> None:
    tasks = []
    for split in SPLITS:
        for dataset, sid, wl, inten, truth in _load_split(split):
            for db in DBS.values():
                tasks.append((db, split, dataset, sid, wl, inten, truth))
    print(f"{len(tasks)} inversions across {WORKERS} workers (spectrum-level)...\n")

    # collect rmse per (db, split, dataset)
    bucket = defaultdict(list)
    ctx = mp.get_context("spawn")
    done = 0
    with cf.ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        for db_path, split, dataset, r in ex.map(_score_one, tasks, chunksize=2):
            done += 1
            if r is not None:
                bucket[(db_path, split, dataset)].append(r)
            if done % 40 == 0:
                print(f"  {done}/{len(tasks)} inversions done")

    # metric per (db, split): median per dataset, then mean across datasets (matches run_eval)
    def metric(db, split):
        meds = [
            st.median(bucket[(db, split, ds)])
            for ds in SPLITS[split]
            if bucket.get((db, split, ds))
        ]
        return sum(meds) / len(meds) if meds else float("nan")

    print(f"\n{'database':24s} {'dev RMSE':>9s} {'HELD-OUT RMSE':>14s}   (held-out decides)\n")
    res = {}
    for name, db in DBS.items():
        d, t = metric(db, "dev"), metric(db, "test")
        res[name] = {"dev": d, "test": t}
        print(f"{name:24s} {d:>9.3f} {t:>14.3f}")

    nist = res["NIST (28.7k, graded)"]["test"]
    print(f"\nNIST held-out RMSE = {nist:.3f} wt%")
    winners = [
        (n, r["test"])
        for n, r in res.items()
        if "NIST" not in n and r["test"] == r["test"] and r["test"] < nist - 0.01
    ]
    print("HELD-OUT winners over NIST:", winners or "NONE — NIST still best on held-out")
    with open("output/atomic_db_benchmark.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote output/atomic_db_benchmark.json")


if __name__ == "__main__":
    main()
