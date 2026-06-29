#!/usr/bin/env python
"""Head-to-head comparison of the formalization-derived line-selection levers, on the same
paired same-element-set conditioned-RMSE harness (supercam_labcal).

The earlier finding: at the DEFAULT config the selection gates never bind on this data
(SNR~1e6 => min_snr inert; <=16 lines/element => max_lines=20 cap never binds), so every
selection lever is a structural no-op. To test the selection CRITERION itself we force a
BINDING cap (max_lines=6) and compare:
  - baseline   : top-6 by score (SNR x isolation x 1/atomic_unc)
  - reliability: 6 lines maximizing upper-level energy spread (twoLineBeta_stable_sharp)
plus cap=6-vs-default(20) to show what capping costs. Usage:
    PYTHONPATH=$PWD .venv/bin/python scripts/benchmark_lever_comparison.py
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
CAP = 6

CONFIGS = {
    "default20": None,
    "score_cap6": {"max_lines_per_element": CAP},
    "reliability_cap6": {"max_lines_per_element": CAP, "reliability_ranked_selection": True},
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


def _compare(recs_a, recs_b, label):
    pairs = [p for p in (_cond_rmse(a, b) for a, b in zip(recs_a, recs_b)) if p is not None]
    if not pairs:
        print(f"{label:32s}  no usable pairs")
        return None
    a_s, b_s = [p[0] for p in pairs], [p[1] for p in pairs]
    med_a, med_b = st.median(a_s), st.median(b_s)
    wins = sum(1 for a, b in pairs if b < a - 1e-9)
    print(
        f"{label:32s}  med_A={med_a:6.3f}  med_B={med_b:6.3f}  Δ(B-A)={med_b - med_a:+6.3f}  "
        f"n={len(pairs):2d}  B_wins={wins:2d}/{len(pairs):<2d}"
    )
    return {"med_a": med_a, "med_b": med_b, "delta": med_b - med_a, "n": len(pairs), "b_wins": wins}


def main() -> None:
    ensure_default_datasets()
    db = AtomicDatabase(DB)
    entry = next(iter_datasets(names=[DATASET]))
    items = list(entry.adapter_factory())
    idx = _sample_indices(len(items), MAX_SPECTRA, SEED)
    items = [items[i] for i in idx]

    recs = {name: [] for name in CONFIGS}
    for sid, wl, inten, truth in items:
        for name, ov in CONFIGS.items():
            recs[name].append(_score_spectrum(db, sid, wl, inten, truth, config_overrides=ov))

    print(
        f"\nsupercam_labcal, n={MAX_SPECTRA}, conditioned RMSE (wt%). 'A vs B' => Δ=med_B-med_A:\n"
    )
    summary = {}
    # The lever test: does reliability-ranked beat score-ranked when the cap binds?
    summary["reliability_vs_score@cap6"] = _compare(
        recs["score_cap6"], recs["reliability_cap6"], "score_cap6 (A) vs reliability_cap6 (B)"
    )
    # Context: what does capping at 6 cost vs the default (cap never binds)?
    summary["score_cap6_vs_default"] = _compare(
        recs["default20"], recs["score_cap6"], "default20 (A) vs score_cap6 (B)"
    )
    summary["reliability_cap6_vs_default"] = _compare(
        recs["default20"], recs["reliability_cap6"], "default20 (A) vs reliability_cap6 (B)"
    )

    with open("output/lever_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote output/lever_comparison.json")
    lev = summary["reliability_vs_score@cap6"]
    if lev:
        verdict = (
            "reliability-ranked BEATS score-ranked"
            if lev["delta"] < -0.01
            else (
                "reliability-ranked LOSES to score-ranked"
                if lev["delta"] > 0.01
                else "reliability-ranked ~ score-ranked (flat)"
            )
        )
        print(
            f"\nLEVER VERDICT (cap binds): {verdict} (Δ={lev['delta']:+.3f} wt%, {lev['b_wins']}/{lev['n']} wins)"
        )


if __name__ == "__main__":
    main()
