#!/usr/bin/env python
"""Benchmark-gate a single opt-in pipeline flag.

Runs the goal-metric scoreboard twice on the same datasets/seed — once with the
production default (flag off) and once with the flag forced on via
``config_overrides`` — then reports whether to ADOPT the flag as the new default.

Adoption guard (the repo has regressed 3x on ungated algorithm changes):
ADOPT iff aggregate mean micro-F1 does not decrease AND no dataset regresses F1
by more than 0.02 AND aggregate mean composition RMSE does not increase.

Usage
-----
    PYTHONPATH=$PWD python scripts/benchmarks/gate_flag.py \
        --db ASD_da/libs_production.db --override saha_boltzmann_graph=true \
        --datasets aalto csa_planetary chemcam_calib silva2022 synthetic_fixedforward \
        --max-spectra 100 --seed 20260610 --out output/gate/saha_boltzmann_graph
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.scoreboard import run_scoreboard


def _parse_override(spec: str) -> tuple[str, Any]:
    """Parse a ``key=value`` override; coerce bools/ints/floats, else str."""
    key, raw = spec.split("=", 1)
    low = raw.lower()
    if low in ("true", "false"):
        return key, low == "true"
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            continue
    return key, raw


def _rows(board: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {d["name"]: d for d in board.get("datasets", [])}


def _f1(d: dict[str, Any]) -> float:
    return float((d.get("id_metrics") or {}).get("f1", 0.0))


def _rmse(d: dict[str, Any]) -> Optional[float]:
    comp = d.get("composition")
    if comp and comp.get("rmse_wt_median") is not None:
        return float(comp["rmse_wt_median"])
    return None


def _nfail(d: dict[str, Any]) -> int:
    return int(d.get("n_failed", 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--override", action="append", default=[], help="key=value (repeatable)")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--max-spectra", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--include-holdout", action="store_true")
    ap.add_argument("--out", default="output/gate/flag")
    args = ap.parse_args()

    overrides = dict(_parse_override(s) for s in args.override)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db = AtomicDatabase(args.db)

    common = dict(
        datasets=args.datasets,
        max_spectra=args.max_spectra,
        seed=args.seed,
        preset=args.preset,
        include_holdout=args.include_holdout,
    )
    print("[gate] baseline (production default) ...", flush=True)
    base = run_scoreboard(db, config_overrides=None, **common)
    json.dump(base, open(out / "baseline.json", "w"), indent=2, default=str)
    print(f"[gate] flag-on {overrides} ...", flush=True)
    flag = run_scoreboard(db, config_overrides=overrides, **common)
    json.dump(flag, open(out / "flag_on.json", "w"), indent=2, default=str)

    b, f = _rows(base), _rows(flag)
    names = sorted(set(b) & set(f))
    print("\n| dataset | F1 base | F1 on | dF1 | RMSE base | RMSE on | fail base | fail on |")
    print("|---|---|---|---|---|---|---|---|")
    df1s: list[float] = []
    drmse: list[float] = []
    regress: list[str] = []
    for n in names:
        d0, d1 = b[n], f[n]
        a, c = _f1(d0), _f1(d1)
        df = c - a
        df1s.append(df)
        r0, r1 = _rmse(d0), _rmse(d1)
        if r0 is not None and r1 is not None:
            drmse.append(r1 - r0)
        if df < -0.02:
            regress.append(n)
        rs0 = f"{r0:.3f}" if r0 is not None else "—"
        rs1 = f"{r1:.3f}" if r1 is not None else "—"
        print(
            f"| {n} | {a:.3f} | {c:.3f} | {df:+.3f} | {rs0} | {rs1} | {_nfail(d0)} | {_nfail(d1)} |"
        )

    mean_df1 = float(np.mean(df1s)) if df1s else 0.0
    mean_drmse = float(np.mean(drmse)) if drmse else 0.0
    # No-harm guard: no dataset F1 regression > 0.02, no aggregate F1/RMSE harm.
    no_harm = (mean_df1 >= -1e-6) and (not regress) and (mean_drmse <= 1e-6)
    # Improvement: helps on at least one axis (don't flip a default for zero gain).
    improved = (mean_df1 > 1e-3) or (bool(drmse) and mean_drmse < -1e-3)
    adopt = no_harm and improved
    verdict = {
        "override": overrides,
        "mean_dF1": mean_df1,
        "mean_dRMSE_wt": mean_drmse,
        "regressions_dF1_gt_0.02": regress,
        "no_harm": no_harm,
        "improved": improved,
        "ADOPT": adopt,
    }
    json.dump(verdict, open(out / "verdict.json", "w"), indent=2)
    print(
        f"\n[gate] mean dF1={mean_df1:+.4f}  mean dRMSE_wt={mean_drmse:+.4f}  regressions={regress}"
    )
    print(f"[gate] VERDICT: {'ADOPT (flip default)' if adopt else 'REJECT (keep opt-in)'}")


if __name__ == "__main__":
    main()
