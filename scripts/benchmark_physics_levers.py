#!/usr/bin/env python
"""Benchmark the levers that act on the INVERSION (not line selection) — where accuracy
actually lives. Paired same-element-set conditioned RMSE vs the default, on supercam_labcal.

Selection levers were no-ops (the pipeline uses every line). These touch the estimator/physics:
  - apply_self_absorption: off -> observable  (selfAbsorbed_corrects_bias: undoes the downward
    bias of optically-thick major-element lines)
  - closure_mode: oxide -> standard / ilr      (directly changes the composition mapping)

    PYTHONPATH=$PWD .venv/bin/python scripts/benchmark_physics_levers.py
"""

from __future__ import annotations

import json
import statistics as st

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.scoreboard import (
    _sample_indices,
    _score_spectrum,
    ensure_default_datasets,
    iter_datasets,
)

DB = "ASD_da/libs_production.db"
DATASET = "supercam_labcal"
MAX_SPECTRA = 20
SEED = 7

LEVERS = {
    "self_absorption": {"apply_self_absorption": "observable"},
    "closure_standard": {"closure_mode": "standard"},
    "closure_ilr": {"closure_mode": "ilr"},
}


def _cond_rmse(rec_a: dict, rec_b: dict):
    if rec_a.get("status") != "ok" or rec_b.get("status") != "ok":
        return None
    shared = set(rec_a.get("called_present", [])) & set(rec_b.get("called_present", []))
    if not shared:
        return None

    def rmse(rec):
        errs = [rec.get("signed_errors_wt", {}).get(el) for el in shared]
        errs = [e for e in errs if e is not None]
        return (sum(e * e for e in errs) / len(errs)) ** 0.5 if errs else None

    ra, rb = rmse(rec_a), rmse(rec_b)
    return None if (ra is None or rb is None) else (ra, rb)


def main() -> None:
    ensure_default_datasets()
    db = AtomicDatabase(DB)
    entry = next(iter_datasets(names=[DATASET]))
    items = list(entry.adapter_factory())
    idx = _sample_indices(len(items), MAX_SPECTRA, SEED)
    items = [items[i] for i in idx]

    base, lever_recs = [], {k: [] for k in LEVERS}
    for sid, wl, inten, truth in items:
        base.append(_score_spectrum(db, sid, wl, inten, truth, config_overrides=None))
        for name, ov in LEVERS.items():
            lever_recs[name].append(_score_spectrum(db, sid, wl, inten, truth, config_overrides=ov))

    print(
        f"\nsupercam_labcal n={MAX_SPECTRA}, conditioned RMSE vs DEFAULT. Δ<0 = lever IMPROVES:\n"
    )
    print(
        f"  {'lever':18s} {'med_base':>9s} {'med_lever':>10s} {'Δmedian':>9s} {'n':>3s} {'wins':>7s}"
    )
    summary = {}
    for name in LEVERS:
        pairs = [
            p for p in (_cond_rmse(b, l) for b, l in zip(base, lever_recs[name])) if p is not None
        ]
        if not pairs:
            print(f"  {name:18s} {'--':>9s} {'--':>10s} {'--':>9s} {0:>3d}")
            continue
        bs, ls = [p[0] for p in pairs], [p[1] for p in pairs]
        med_b, med_l = st.median(bs), st.median(ls)
        wins = sum(1 for b, l in pairs if l < b - 1e-9)
        summary[name] = {
            "med_base": med_b,
            "med_lever": med_l,
            "delta": med_l - med_b,
            "n": len(pairs),
            "wins": wins,
        }
        flag = (
            "  <== IMPROVES"
            if med_l < med_b - 0.01
            else ("  (regress)" if med_l > med_b + 0.01 else "")
        )
        print(
            f"  {name:18s} {med_b:>9.3f} {med_l:>10.3f} {med_l - med_b:>+9.3f} "
            f"{len(pairs):>3d} {wins:>3d}/{len(pairs):<3d}{flag}"
        )

    with open("output/physics_levers.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote output/physics_levers.json")
    winners = [k for k, v in summary.items() if v["delta"] < -0.01]
    print(
        f"IMPROVING levers (Δ<-0.01 wt%): {winners or 'NONE — dig deeper (atomic-data weighting, solver)'}"
    )


if __name__ == "__main__":
    main()
