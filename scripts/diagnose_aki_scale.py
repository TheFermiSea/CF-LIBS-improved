#!/usr/bin/env python
"""Per-species A_ki systematic-scale-bias diagnostic (audit Issue 1, Fix #1b).

Emits the falsifiable "scale-spread table": for each CF-LIBS target species,
the NIST accuracy-grade composition and the grade-weighted RMS fractional A_ki
error, which is (to first order) the within-species Boltzmann-plot RMS residual
FLOOR attributable to A_ki bias. High sigma_scale_rms => this species' composition
intercept carries a large systematic scale error that no WLS down-weighting can
remove; only lifetime/branching-fraction anchoring can.

Usage:
    PYTHONPATH=$PWD python scripts/diagnose_aki_scale.py \
        --db ASD_da/libs_production.db --output output/validation/aki_scale_diag.json

    # optional NIST-vs-Kurucz cross-source scale comparison (needs an ingested
    # Kurucz DB from scripts/ingest_kurucz_atomic.py):
    PYTHONPATH=$PWD python scripts/diagnose_aki_scale.py \
        --db ASD_da/libs_production.db --compare-db output/kurucz_atomic.db
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cflibs
from cflibs.atomic.aki_anchor import (
    DEFAULT_TARGET_SPECIES,
    compare_aki_sources,
    species_scale_diagnostic,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="ASD_da/libs_production.db")
    ap.add_argument("--compare-db", default=None, help="Independent atomic DB (e.g. Kurucz)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    print(f"cflibs={Path(cflibs.__file__).resolve()}")  # worktree-provenance guard
    print(f"db={Path(args.db).resolve()}\n")

    reports = species_scale_diagnostic(args.db, DEFAULT_TARGET_SPECIES)

    hdr = (
        f"{'species':>8} {'n_aki':>7} {'D/E%':>6} {'sig_rms':>8} {'sig_med':>8} "
        f"{'sharedLv':>9} {'grades'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        gc = ",".join(f"{k}:{v}" for k, v in sorted(r.grade_counts.items()))
        print(
            f"{r.element+' '+str(r.sp_num):>8} {r.n_with_aki:>7} "
            f"{100*r.frac_de_grade:>5.1f} {r.sigma_scale_rms:>8.3f} "
            f"{r.sigma_scale_median:>8.3f} {r.n_shared_upper_levels:>9} {gc}"
        )

    payload: dict = {"diagnostic": [r.as_dict() for r in reports]}

    if args.compare_db:
        print("\n=== NIST vs comparison-source ln(A_ref/A_cmp) ===")
        comps = compare_aki_sources(args.db, args.compare_db, DEFAULT_TARGET_SPECIES)
        print(f"{'species':>8} {'n_shared':>9} {'med':>8} {'mad':>8} {'scaleF':>8}")
        for c in comps:
            d = c.as_dict()
            print(
                f"{d['species']:>8} {d['n_shared']:>9} {d['ln_ratio_median']:>8.3f} "
                f"{d['ln_ratio_mad']:>8.3f} {d['scale_factor_median']:>8.3f}"
            )
        payload["source_comparison"] = [c.as_dict() for c in comps]

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
