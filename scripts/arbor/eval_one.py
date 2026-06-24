#!/usr/bin/env python
"""One SLURM-array unit of the atomic-DB benchmark: score ONE (db, dataset, split) by running its
spectra in parallel across the task's allocated cores, and write the median composition RMSE.

Used by scripts/benchmarks/db_bench.sbatch — one array task per (db, dataset), fanned across
vasp-01/02/03. Aggregate the per-unit JSONs afterward (held-out = the test-split datasets).

    python scripts/arbor/eval_one.py --db DB --dataset NAME --split dev|test --workers N --out F.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import shutil
import statistics as st
import tempfile

_DBC: dict = {}  # per-process AtomicDatabase cache


def _score(task):
    db_path, sid, wl, inten, truth = task
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.benchmark.scoreboard import _score_spectrum

    db = _DBC.get(db_path)
    if db is None:
        # Each WORKER PROCESS gets its OWN node-local copy of the DB. One shared copy (NFS or /tmp)
        # still collides under concurrent SQLite reads ("database is locked"); a per-process copy
        # has no shared file, so no lock contention.
        local = os.path.join(
            tempfile.gettempdir(), f"dbw_{os.getpid()}_{os.path.basename(db_path)}"
        )
        if not os.path.exists(local):
            shutil.copy(db_path, local)
        db = AtomicDatabase(local)
        _DBC[db_path] = db
    try:
        rec = _score_spectrum(db, sid, wl, inten, truth, config_overrides=None)
        return rec.get("rmse_wt") if rec.get("status") == "ok" else None
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-spectra", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from cflibs.benchmark.scoreboard import (
        _sample_indices,
        ensure_default_datasets,
        iter_datasets,
    )

    ensure_default_datasets()
    entry = next(iter(iter_datasets(names=[a.dataset])), None)
    items = list(entry.adapter_factory()) if entry is not None else []
    if items:
        idx = _sample_indices(len(items), a.max_spectra, 7)
        if idx is not None:
            items = [items[i] for i in idx]
    # Copy the DB to node-local disk: SQLite file locking is unreliable over NFS, and 8 concurrent
    # workers reading a large NFS DB collide with "database is locked". Local copy = no NFS locks.
    db_path = a.db
    if os.path.exists(a.db):
        local = os.path.join(tempfile.gettempdir(), f"dbb_{os.getpid()}_{os.path.basename(a.db)}")
        shutil.copy(a.db, local)
        db_path = local

    tasks = [(db_path, sid, wl, inten, truth) for sid, wl, inten, truth in items]

    rmses = []
    if tasks:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=max(1, a.workers), mp_context=ctx) as ex:
            for r in ex.map(_score, tasks, chunksize=1):
                if r is not None:
                    rmses.append(r)

    if db_path != a.db:
        try:
            os.remove(db_path)
        except OSError:
            pass

    out = {
        "db": a.db,
        "dataset": a.dataset,
        "split": a.split,
        "rmse_median": (st.median(rmses) if rmses else None),
        "n_ok": len(rmses),
        "n": len(tasks),
    }
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
