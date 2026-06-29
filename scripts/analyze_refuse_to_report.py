#!/usr/bin/env python
"""Refuse-to-report lever: does flagging non-identifiable spectra (Step-3 identifiability
guards) remove the high-error ones? Measures the coverage/accuracy tradeoff on supercam_labcal.

For each spectrum: run the default pipeline, capture the SELECTED lines, apply the verified
identifiability guards (temperature needs >=2 distinct-E lines per species; composition needs
a T anchor), then compare per-spectrum RMSE of RETAINED (identifiable) vs FLAGGED spectra. If
flagged RMSE > retained RMSE, abstaining improves the reported accuracy.

    PYTHONPATH=$PWD .venv/bin/python scripts/analyze_refuse_to_report.py
"""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.scoreboard import (
    _sample_indices,
    _score_spectrum,
    ensure_default_datasets,
    iter_datasets,
)
from cflibs.inversion.physics.identifiability import refuse_to_report
from cflibs.inversion.physics.line_selection import LineSelector

DB = "ASD_da/libs_production.db"
DATASET = "supercam_labcal"
MAX_SPECTRA = 30
SEED = 7


def main() -> None:
    ensure_default_datasets()
    db = AtomicDatabase(DB)
    entry = next(iter_datasets(names=[DATASET]))
    items = list(entry.adapter_factory())
    idx = _sample_indices(len(items), MAX_SPECTRA, SEED)
    items = [items[i] for i in idx]

    # spy: capture the selected lines of the most recent selection
    orig = LineSelector.select
    cap: dict = {}

    def spy(self, observations, **kw):
        res = orig(self, observations, **kw)
        cap["sel"] = res.selected_lines
        return res

    LineSelector.select = spy

    rows = []  # (identifiable, reason, rmse_wt)
    for sid, wl, inten, truth in items:
        cap.clear()
        rec = _score_spectrum(db, sid, wl, inten, truth, config_overrides=None)
        if rec.get("status") != "ok" or "rmse_wt" not in rec:
            continue
        by_el: dict[str, list[float]] = defaultdict(list)
        for o in cap.get("sel", []):
            by_el[o.element].append(o.E_k_ev)
        lines_by_element = {el: len(es) for el, es in by_el.items()}
        # T anchor exists iff some species has >=2 distinct-E lines (temperature_identifiable)
        has_anchor = any(
            refuse_to_report(upper_level_energies_ev=es).identifiable for es in by_el.values()
        )
        res = refuse_to_report(lines_by_element=lines_by_element, has_temperature_anchor=has_anchor)
        rows.append((res.identifiable, res.reason, rec["rmse_wt"]))

    if not rows:
        print("no usable spectra")
        return
    retained = [r[2] for r in rows if r[0]]
    flagged = [r[2] for r in rows if not r[0]]
    allr = [r[2] for r in rows]
    print(f"\nsupercam_labcal refuse-to-report (identifiability guards), n={len(rows)}:")
    print(
        f"  identifiable (retained): {len(retained)}  flagged: {len(flagged)}  "
        f"({100 * len(flagged) / len(rows):.0f}% coverage loss)"
    )

    def med(xs):
        return f"{st.median(xs):.3f}" if xs else "--"

    print(f"  median RMSE_wt  all={med(allr)}  retained={med(retained)}  flagged={med(flagged)}")
    print(f"  flag reasons: {dict(Counter(r[1] for r in rows if not r[0]))}")
    if flagged and retained:
        better = st.median(flagged) > st.median(retained)
        print(
            f"\n  VERDICT: flagged spectra have {'HIGHER' if better else 'NOT higher'} RMSE than "
            f"retained -> abstaining {'IMPROVES' if better else 'does NOT improve'} reported accuracy."
        )
    elif not flagged:
        print(
            "\n  VERDICT: 0 spectra flagged on this data — the guards never fire here "
            "(every spectrum has an identifiable T anchor). Refuse-to-report is a no-op on "
            "supercam_labcal (healthy multi-line spectra); it is a safety net for degenerate inputs."
        )


if __name__ == "__main__":
    main()
