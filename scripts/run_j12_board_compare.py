"""J12: run the scoreboard for BOTH pipelines and emit a jit-vs-reference delta table.

Runs ``run_scoreboard`` with pipeline_impl='reference' and ='jit' over the
scoring datasets (optimization tier; holdout excluded unless --include-holdout),
then writes a per-dataset delta table (F1, median RMSE wt%, failures, median
wall s) plus the two raw board JSONs. This is the J12/M3 superiority-run harness
(ADR-0004 §8.2); the promotion decision reads its output.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD python -u \
        scripts/run_j12_board_compare.py --datasets aalto,synthetic_fixedforward \
        --max-spectra 6 --output-dir output/j12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.scoreboard import run_scoreboard, write_artifacts

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "ASD_da" / "libs_production.db"


def _row(board: dict, name: str) -> dict | None:
    for d in board["datasets"]:
        if d["name"] == name:
            return d
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None, help="comma list; default = all scoring")
    ap.add_argument("--max-spectra", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--include-holdout", action="store_true")
    ap.add_argument("--output-dir", default="output/j12")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets = (
        [s.strip() for s in args.datasets.split(",") if s.strip()] if args.datasets else None
    )
    db = AtomicDatabase(str(DB))

    boards = {}
    for impl in ("reference", "jit"):
        print(f"\n{'='*70}\nRunning board: pipeline={impl}\n{'='*70}", flush=True)
        board = run_scoreboard(
            db,
            datasets=datasets,
            max_spectra=args.max_spectra,
            seed=args.seed,
            include_holdout=args.include_holdout,
            pipeline_impl=impl,
        )
        jpath, _ = write_artifacts(board, out / impl)
        boards[impl] = board
        print(f"  wrote {jpath}", flush=True)

    # ---- delta table ----
    names = [d["name"] for d in boards["reference"]["datasets"]]
    lines = [
        "# J12 jit-vs-reference board delta",
        "",
        f"max_spectra={args.max_spectra} seed={args.seed} "
        f"include_holdout={args.include_holdout}",
        "",
        "| dataset | n | F1 ref | F1 jit | ΔF1 | RMSE ref | RMSE jit | fail ref | fail jit "
        "| wall ref | wall jit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        r, j = _row(boards["reference"], name), _row(boards["jit"], name)
        if r is None or j is None:
            continue
        rf = r["id_metrics"]["f1"]
        jf = j["id_metrics"]["f1"]
        rr = (r.get("composition") or {}).get("rmse_wt_median")
        jr = (j.get("composition") or {}).get("rmse_wt_median")
        rw = (r.get("runtime") or {}).get("median_wall_s")
        jw = (j.get("runtime") or {}).get("median_wall_s")

        def f(x, s=".3f"):
            return format(x, s) if isinstance(x, (int, float)) else "—"

        lines.append(
            f"| {name} | {r['n_run']} | {f(rf)} | {f(jf)} | {f(jf-rf,'+.3f')} | "
            f"{f(rr)} | {f(jr)} | {r['n_failed']} | {j['n_failed']} | "
            f"{f(rw,'.2f')} | {f(jw,'.2f')} |"
        )
    table = "\n".join(lines) + "\n"
    (out / "J12-delta.md").write_text(table)
    (out / "boards.json").write_text(
        json.dumps({"reference": boards["reference"], "jit": boards["jit"]}, default=str)
    )
    print("\n" + table)
    print(f"wrote {out/'J12-delta.md'}")


if __name__ == "__main__":
    main()
