#!/usr/bin/env python
"""Benchmark the derived-threshold knob: target_sigma_t OFF (tuned magic numbers) vs ON
(thresholds derived from a target σ_T/T via the verified ErrorBudget).

Accuracy-first gate: paired, SAME-ELEMENT-SET conditioned RMSE — each spectrum is scored under
OFF and every ON config, then compared over the intersection of elements both runs called
present (removes the ID-flip confound). Drives `_score_spectrum` directly to get per-spectrum
records (run_scoreboard only exposes per-dataset aggregates).

Only `supercam_labcal` (1139) and `aalto` (74) have local data; the other optimization-tier
datasets are public-data gaps. Usage:
    PYTHONPATH=$PWD .venv/bin/python scripts/benchmark_target_sigma.py
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
DATASETS = ["supercam_labcal", "aalto"]
TARGETS = [0.10, 0.05, 0.03]
MAX_SPECTRA = 20
SEED = 7


def _cond_rmse(rec_off: dict, rec_on: dict):
    """Conditioned RMSE pair over the intersection of called-present elements."""
    if rec_off.get("status") != "ok" or rec_on.get("status") != "ok":
        return None
    shared = set(rec_off.get("called_present", [])) & set(rec_on.get("called_present", []))
    if not shared:
        return None

    def rmse(rec):
        errs = [rec.get("signed_errors_wt", {}).get(el) for el in shared]
        errs = [e for e in errs if e is not None]
        return (sum(e * e for e in errs) / len(errs)) ** 0.5 if errs else None

    ro, rn = rmse(rec_off), rmse(rec_on)
    return None if (ro is None or rn is None) else (ro, rn)


def main() -> None:
    ensure_default_datasets()
    db = AtomicDatabase(DB)
    entries = {e.name: e for e in iter_datasets(names=DATASETS)}

    # per (dataset, target): list of (rmse_off, rmse_on) conditioned pairs
    pairs: dict[tuple[str, float], list] = {(ds, t): [] for ds in DATASETS for t in TARGETS}

    for ds in DATASETS:
        items = list(entries[ds].adapter_factory())
        idx = _sample_indices(len(items), MAX_SPECTRA, SEED)
        if idx is not None:
            items = [items[i] for i in idx]
        for sid, wl, inten, truth in items:
            rec_off = _score_spectrum(db, sid, wl, inten, truth, config_overrides=None)
            for t in TARGETS:
                rec_on = _score_spectrum(
                    db, sid, wl, inten, truth, config_overrides={"target_sigma_t": t}
                )
                p = _cond_rmse(rec_off, rec_on)
                if p is not None:
                    pairs[(ds, t)].append(p)

    print(
        f"\n{'dataset':18s} {'target':>7s} {'med_off':>8s} {'med_on':>8s} "
        f"{'Δmedian':>8s} {'n':>3s} {'wins':>7s}"
    )
    summary = {}
    for ds in DATASETS:
        for t in TARGETS:
            pr = pairs[(ds, t)]
            if not pr:
                print(f"{ds:18s} {t:>7.2f} {'--':>8s} {'--':>8s} {'--':>8s} {0:>3d} {'--':>7s}")
                continue
            offs, ons = [p[0] for p in pr], [p[1] for p in pr]
            med_off, med_on = st.median(offs), st.median(ons)
            wins = sum(1 for o, n in pr if n < o - 1e-9)
            summary[f"{ds}@{t}"] = {
                "med_off": med_off,
                "med_on": med_on,
                "delta": med_on - med_off,
                "n": len(pr),
                "wins": wins,
            }
            print(
                f"{ds:18s} {t:>7.2f} {med_off:>8.3f} {med_on:>8.3f} "
                f"{med_on - med_off:>+8.3f} {len(pr):>3d} {wins:>3d}/{len(pr):<3d}"
            )

    with open("output/target_sigma_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote output/target_sigma_benchmark.json")
    deltas = [v["delta"] for v in summary.values()]
    if deltas:
        improved = sum(1 for d in deltas if d < -0.01)
        regressed = sum(1 for d in deltas if d > 0.01)
        print(
            f"VERDICT: {improved} improved, {regressed} regressed, "
            f"{len(deltas) - improved - regressed} flat (±0.01 wt%). "
            f"median Δ = {st.median(deltas):+.3f} wt%"
        )


if __name__ == "__main__":
    main()
