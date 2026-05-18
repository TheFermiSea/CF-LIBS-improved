#!/usr/bin/env python
"""8-cell × N-seed sweep of the alias-fix experiment.

Phase C of jaunty-weaving-mist. Enumerates the 2^3 = 8 combinations of the
three opt-in fix flags landed in PRs #175 / #177 / #176 (ftp1 / 762f / dj6y),
runs each cell × seed via ``scripts/run_unified_benchmark.py``, then
aggregates ``id_summary.json`` outputs into a CSV and a Markdown report.

Idempotent: reruns skip (cell, seed) pairs whose output directory already
contains ``id_summary.json``, so a partial sweep can resume in place.

Usage (cluster):
    python scripts/sweep_alias_fixes.py \\
        --seeds 5 \\
        --output-dir /scratch/cf-libs-bench/alias-fix-sweep \\
        --basis-dir /cluster/shared/cf-libs-bench/basis_libraries \\
        --dataset-shard 1/3

Usage (smoke-test off-cluster):
    python scripts/sweep_alias_fixes.py --smoke-mode

The promotion rule (codified in the Markdown report):
    Pick the cell with the highest macro_f1 lift on
    ``vrabel2020_soil_benchmark`` subject to no regression > 0.02 on any
    metric on ``aalto_libs`` or ``bhvo2_usgs``. Ties within 0.005 macro_f1
    -> prefer the simpler config (fewer fix flags). If ``baseline`` wins,
    the diagnosis was over-confident and the report says so explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_UNIFIED = _REPO_ROOT / "scripts" / "run_unified_benchmark.py"

# When this script ships inside a git worktree (e.g. /tmp/cf-libs-wt-sweep)
# but the venv's editable ``cflibs`` install points at a sibling worktree
# (e.g. /home/brian/code/CF-LIBS-improved), the wrong copy of cflibs gets
# imported. Force the worktree's repo root to the front of sys.path so the
# in-tree ``cflibs/`` package wins. This is critical for the smoke-mode
# registry check and for any aggregator that imports cflibs.benchmark.
_REPO_STR = str(_REPO_ROOT)
if _REPO_STR not in sys.path:
    sys.path.insert(0, _REPO_STR)
elif sys.path.index(_REPO_STR) != 0:
    sys.path.remove(_REPO_STR)
    sys.path.insert(0, _REPO_STR)

# Canonical cell order — must stay in sync with
# ``cflibs.benchmark.unified._ALIAS_SWEEP_CELLS``.
_CELL_ORDER: Tuple[str, ...] = (
    "baseline",
    "ftp1",
    "762f",
    "dj6y",
    "ftp1+762f",
    "ftp1+dj6y",
    "762f+dj6y",
    "all_three",
)

# Metrics extracted from id_summary.json["overall"][<workflow>].
_OVERALL_METRICS: Tuple[str, ...] = (
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "false_positives_per_spectrum",
)

# Per-dataset breakdown lives under id_summary.json["stratified"]["dataset_id"][<workflow>].
# Datasets called out by the promotion rule.
_PROMOTION_TARGET_DATASET = "vrabel2020_soil_benchmark"
_PROMOTION_GUARD_DATASETS = ("aalto_libs", "bhvo2_usgs")
_PROMOTION_GUARD_TOL = 0.02
_PROMOTION_TIE_TOL = 0.005

# Per-element diff targets (the elements flagged in jaunty-weaving-mist as
# under-suspicion for the alias-fix experiment).
_FOCUS_ELEMENTS: Tuple[str, ...] = ("Si", "Mg", "Al", "Ti")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 8-cell × N-seed alias-fix sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of RNG seeds per cell. Each seed is run as a separate "
        "subprocess; results are aggregated mean ± stdev across seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Top-level output directory. Each (cell, seed) gets a subdir "
        "named ``cell-<cell>-seed-<seed>``. REQUIRED unless --smoke-mode.",
    )
    parser.add_argument(
        "--basis-dir",
        type=Path,
        default=Path("/cluster/shared/cf-libs-bench/basis_libraries"),
        help="Path to the basis-library .h5 files (cluster shared NFS).",
    )
    parser.add_argument(
        "--dataset-shard",
        type=str,
        default="1/3",
        help="Dataset shard N/K in run_unified_benchmark.py's convention.",
    )
    parser.add_argument(
        "--cells",
        type=str,
        default="all",
        help="Comma-separated cell names (or 'all'). Useful for smoke "
        "tests with a single cell. Cells: " + ",".join(_CELL_ORDER) + ".",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed value (seeds run as [seed_start, seed_start+seeds)).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter used to invoke run_unified_benchmark.py.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip benchmark execution; only aggregate existing "
        "id_summary.json files in --output-dir.",
    )
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help="Registry-validation-only check: confirm all 8 alias_sweep_<cell> "
        "workflows are registered, then exit. No benchmarks run. Use for "
        "off-cluster validation.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=_REPO_ROOT / "docs" / "research" / "findings",
        help="Directory where the aggregated CSV and Markdown report land.",
    )
    parser.add_argument(
        "--report-date",
        type=str,
        default=None,
        help="ISO date string for the report filename "
        "(defaults to today's date in UTC).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run_unified_benchmark.py invocations that WOULD "
        "be executed and exit. No benchmarks or aggregation run.",
    )
    return parser


# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------


def _run_smoke() -> int:
    """Validate the registry exposes all 8 sweep workflows. No benchmarks."""
    try:
        from cflibs.benchmark.unified import build_id_workflow_registry
    except ImportError as exc:
        print(f"smoke: failed to import unified registry: {exc}", file=sys.stderr)
        return 2

    registry = build_id_workflow_registry(quick=True)
    missing: List[str] = []
    for cell in _CELL_ORDER:
        name = f"alias_sweep_{cell}"
        if name not in registry:
            missing.append(name)
    if missing:
        print(f"smoke: missing workflows: {missing}", file=sys.stderr)
        return 1
    print("smoke: all 8 alias_sweep_<cell> workflows registered:")
    for cell in _CELL_ORDER:
        print(f"  - alias_sweep_{cell}")
    return 0


# ---------------------------------------------------------------------------
# Run dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellRun:
    cell: str
    seed: int
    output_dir: Path

    @property
    def workflow_name(self) -> str:
        return f"alias_sweep_{self.cell}"

    @property
    def id_summary_path(self) -> Path:
        return self.output_dir / "id_summary.json"


def _resolve_cells(arg: str) -> List[str]:
    if arg.strip().lower() == "all":
        return list(_CELL_ORDER)
    requested = [c.strip() for c in arg.split(",") if c.strip()]
    bad = [c for c in requested if c not in _CELL_ORDER]
    if bad:
        raise SystemExit(
            f"--cells: unknown cell(s) {bad}. Valid cells: {list(_CELL_ORDER)}"
        )
    return requested


def _check_basis_dir(basis_dir: Path) -> None:
    if not basis_dir.exists():
        raise SystemExit(
            f"--basis-dir does not exist: {basis_dir}\n"
            "On the cluster this should be /cluster/shared/cf-libs-bench/basis_libraries.\n"
            "Off-cluster, use --smoke-mode to validate the harness without running "
            "actual benchmarks."
        )
    h5s = list(basis_dir.glob("basis_fwhm_*nm.h5"))
    if not h5s:
        raise SystemExit(
            f"--basis-dir {basis_dir} contains no basis_fwhm_*nm.h5 files. "
            "Has the basis library been built?"
        )


def _build_run_command(
    *,
    python: str,
    run_unified: Path,
    cell: str,
    seed: int,
    basis_dir: Path,
    dataset_shard: str,
    output_dir: Path,
) -> List[str]:
    return [
        python,
        str(run_unified),
        "--quick",
        "--max-outer-folds",
        "1",
        "--sections",
        "id",
        "--id-workflows",
        f"alias_sweep_{cell}",
        "--basis-dir",
        str(basis_dir),
        "--dataset-shard",
        dataset_shard,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
    ]


def _run_one_cell(run: CellRun, cmd: Sequence[str]) -> int:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = run.output_dir / "stdout.log"
    err_path = run.output_dir / "stderr.log"
    print(
        f"[sweep] running cell={run.cell!r} seed={run.seed} -> {run.output_dir}",
        flush=True,
    )
    with log_path.open("wb") as out_f, err_path.open("wb") as err_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=err_f, check=False)
    if result.returncode != 0:
        print(
            f"[sweep] FAILED cell={run.cell!r} seed={run.seed} "
            f"(returncode={result.returncode}, see {err_path})",
            file=sys.stderr,
            flush=True,
        )
    return result.returncode


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class CellSeedRow:
    cell: str
    seed: int
    workflow: str
    overall: Dict[str, float] = field(default_factory=dict)
    per_dataset: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_element: Dict[str, Dict[str, float]] = field(default_factory=dict)
    failure: Optional[str] = None


def _load_id_summary(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    try:
        with path.open("r") as fp:
            return json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[sweep] failed to parse {path}: {exc}", file=sys.stderr)
        return None


def _extract_overall(summary: Dict[str, object], workflow: str) -> Dict[str, float]:
    overall_block = summary.get("overall")
    if not isinstance(overall_block, dict):
        return {}
    wf = overall_block.get(workflow)
    if not isinstance(wf, dict):
        return {}
    out: Dict[str, float] = {}
    for key in _OVERALL_METRICS:
        val = wf.get(key)
        if isinstance(val, (int, float)) and not (
            isinstance(val, float) and math.isnan(val)
        ):
            out[key] = float(val)
    return out


def _extract_per_dataset(
    summary: Dict[str, object], workflow: str
) -> Dict[str, Dict[str, float]]:
    stratified = summary.get("stratified")
    if not isinstance(stratified, dict):
        return {}
    ds_block = stratified.get("dataset_id")
    if not isinstance(ds_block, dict):
        return {}
    wf = ds_block.get(workflow)
    if not isinstance(wf, dict):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for dataset_name, metrics in wf.items():
        if not isinstance(metrics, dict):
            continue
        row: Dict[str, float] = {}
        for key in _OVERALL_METRICS:
            val = metrics.get(key)
            if isinstance(val, (int, float)) and not (
                isinstance(val, float) and math.isnan(val)
            ):
                row[key] = float(val)
        if row:
            out[dataset_name] = row
    return out


def _extract_per_element(
    summary: Dict[str, object], workflow: str
) -> Dict[str, Dict[str, float]]:
    per_elem_block = summary.get("per_element")
    if not isinstance(per_elem_block, dict):
        return {}
    wf = per_elem_block.get(workflow)
    if not isinstance(wf, dict):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for element, metrics in wf.items():
        if not isinstance(metrics, dict):
            continue
        row: Dict[str, float] = {}
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and not (
                isinstance(val, float) and math.isnan(val)
            ):
                row[key] = float(val)
        if row:
            out[element] = row
    return out


def _aggregate(
    runs: Sequence[CellRun],
) -> List[CellSeedRow]:
    rows: List[CellSeedRow] = []
    for run in runs:
        row = CellSeedRow(cell=run.cell, seed=run.seed, workflow=run.workflow_name)
        summary = _load_id_summary(run.id_summary_path)
        if summary is None:
            row.failure = f"id_summary.json missing or unparseable at {run.id_summary_path}"
        else:
            row.overall = _extract_overall(summary, run.workflow_name)
            row.per_dataset = _extract_per_dataset(summary, run.workflow_name)
            row.per_element = _extract_per_element(summary, run.workflow_name)
            if not row.overall:
                row.failure = (
                    f"no overall metrics for {run.workflow_name} in "
                    f"{run.id_summary_path}"
                )
        rows.append(row)
    return rows


def _mean_stdev(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), 0.0)
    return (float(statistics.mean(values)), float(statistics.stdev(values)))


def _summarize_per_cell(
    rows: Sequence[CellSeedRow],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Return cell -> metric -> (mean, stdev) across seeds for OVERALL metrics."""
    by_cell: Dict[str, List[CellSeedRow]] = {}
    for r in rows:
        by_cell.setdefault(r.cell, []).append(r)

    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for cell, cell_rows in by_cell.items():
        cell_summary: Dict[str, Tuple[float, float]] = {}
        for metric in _OVERALL_METRICS:
            vals = [r.overall[metric] for r in cell_rows if metric in r.overall]
            cell_summary[metric] = _mean_stdev(vals)
        summary[cell] = cell_summary
    return summary


def _summarize_per_cell_per_dataset(
    rows: Sequence[CellSeedRow],
    datasets: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    """cell -> dataset -> metric -> (mean, stdev)."""
    by_cell: Dict[str, List[CellSeedRow]] = {}
    for r in rows:
        by_cell.setdefault(r.cell, []).append(r)

    out: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    for cell, cell_rows in by_cell.items():
        per_ds: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for ds in datasets:
            metric_map: Dict[str, Tuple[float, float]] = {}
            for metric in _OVERALL_METRICS:
                vals = [
                    r.per_dataset[ds][metric]
                    for r in cell_rows
                    if ds in r.per_dataset and metric in r.per_dataset[ds]
                ]
                metric_map[metric] = _mean_stdev(vals)
            per_ds[ds] = metric_map
        out[cell] = per_ds
    return out


def _summarize_per_cell_per_element(
    rows: Sequence[CellSeedRow],
    elements: Sequence[str],
    metric: str = "f1",
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """cell -> element -> (mean f1, stdev f1)."""
    by_cell: Dict[str, List[CellSeedRow]] = {}
    for r in rows:
        by_cell.setdefault(r.cell, []).append(r)

    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for cell, cell_rows in by_cell.items():
        per_el: Dict[str, Tuple[float, float]] = {}
        for el in elements:
            vals = [
                r.per_element[el][metric]
                for r in cell_rows
                if el in r.per_element and metric in r.per_element[el]
            ]
            per_el[el] = _mean_stdev(vals)
        out[cell] = per_el
    return out


# ---------------------------------------------------------------------------
# Promotion rule
# ---------------------------------------------------------------------------


def _fix_flag_count(cell: str) -> int:
    if cell == "baseline":
        return 0
    return len(cell.split("+"))


def _apply_promotion_rule(
    cell_summary: Dict[str, Dict[str, Tuple[float, float]]],
    per_ds: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
) -> Tuple[Optional[str], List[str]]:
    """Return (winner_cell, log_lines).

    Promotion rule:
      1. Compute macro_f1 lift on _PROMOTION_TARGET_DATASET vs baseline.
      2. Reject any cell whose metric on _PROMOTION_GUARD_DATASETS regresses
         by > _PROMOTION_GUARD_TOL on any of _OVERALL_METRICS.
         (For metrics where lower-is-better — fp_per_spectrum — regression
         means a >tol increase rather than decrease.)
      3. Pick the surviving cell with the highest lift on the target dataset.
      4. Ties within _PROMOTION_TIE_TOL macro_f1 -> simpler (fewer flags).
    """
    log: List[str] = []
    baseline_target = (
        per_ds.get("baseline", {}).get(_PROMOTION_TARGET_DATASET, {}).get("macro_f1")
    )
    if baseline_target is None or math.isnan(baseline_target[0]):
        log.append(
            f"!! baseline lacks macro_f1 on {_PROMOTION_TARGET_DATASET!r}; "
            "cannot evaluate promotion rule."
        )
        return None, log

    baseline_mean = baseline_target[0]
    baseline_guards: Dict[str, Dict[str, float]] = {}
    for ds in _PROMOTION_GUARD_DATASETS:
        block = per_ds.get("baseline", {}).get(ds, {})
        baseline_guards[ds] = {
            metric: mean for metric, (mean, _) in block.items() if not math.isnan(mean)
        }

    survivors: List[Tuple[str, float]] = []
    for cell in _CELL_ORDER:
        if cell == "baseline":
            continue
        target_block = per_ds.get(cell, {}).get(_PROMOTION_TARGET_DATASET, {})
        target = target_block.get("macro_f1")
        if target is None or math.isnan(target[0]):
            log.append(f"-- {cell}: no macro_f1 on {_PROMOTION_TARGET_DATASET!r}; skipping.")
            continue
        cell_mean = target[0]
        lift = cell_mean - baseline_mean

        guard_violations: List[str] = []
        for ds in _PROMOTION_GUARD_DATASETS:
            cell_guard = per_ds.get(cell, {}).get(ds, {})
            for metric in _OVERALL_METRICS:
                ref = baseline_guards.get(ds, {}).get(metric)
                if ref is None:
                    continue
                cur_pair = cell_guard.get(metric)
                if cur_pair is None or math.isnan(cur_pair[0]):
                    continue
                cur = cur_pair[0]
                if metric == "false_positives_per_spectrum":
                    # Lower-is-better: regression = increase
                    delta = cur - ref
                else:
                    # Higher-is-better: regression = decrease
                    delta = ref - cur
                if delta > _PROMOTION_GUARD_TOL:
                    guard_violations.append(
                        f"{ds}.{metric} {ref:.4f}->{cur:.4f} (delta={delta:+.4f})"
                    )
        if guard_violations:
            log.append(
                f"-- {cell}: lift={lift:+.4f} on target but FAILS guard: "
                + "; ".join(guard_violations)
            )
            continue
        log.append(f"++ {cell}: lift={lift:+.4f} on target, guards OK.")
        survivors.append((cell, lift))

    if not survivors:
        log.append("== no fix-on cell passes the guard. Baseline wins by default.")
        return "baseline", log

    survivors.sort(key=lambda x: x[1], reverse=True)
    best_lift = survivors[0][1]
    tied = [s for s in survivors if abs(s[1] - best_lift) <= _PROMOTION_TIE_TOL]
    if len(tied) > 1:
        tied.sort(key=lambda x: (_fix_flag_count(x[0]), x[0]))
        log.append(
            f"== tie within {_PROMOTION_TIE_TOL} macro_f1; picking simplest: "
            + ", ".join(c for c, _ in tied)
        )
    winner = tied[0][0]
    log.append(f"== winner: {winner} (lift={tied[0][1]:+.4f} on target)")
    return winner, log


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _fmt_mean_stdev(pair: Tuple[float, float], digits: int = 4) -> str:
    mean, stdev = pair
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {stdev:.{digits}f}"


def _write_csv(rows: Sequence[CellSeedRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell",
        "seed",
        "scope",
        "key",
        "metric",
        "value",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            for metric, val in r.overall.items():
                writer.writerow(
                    dict(cell=r.cell, seed=r.seed, scope="overall", key="overall",
                         metric=metric, value=val)
                )
            for ds, metrics in r.per_dataset.items():
                for metric, val in metrics.items():
                    writer.writerow(
                        dict(cell=r.cell, seed=r.seed, scope="dataset", key=ds,
                             metric=metric, value=val)
                    )
            for elem, metrics in r.per_element.items():
                for metric, val in metrics.items():
                    writer.writerow(
                        dict(cell=r.cell, seed=r.seed, scope="element", key=elem,
                             metric=metric, value=val)
                    )
            if r.failure:
                writer.writerow(
                    dict(cell=r.cell, seed=r.seed, scope="failure", key="-",
                         metric="-", value=r.failure)
                )


def _write_markdown(
    *,
    rows: Sequence[CellSeedRow],
    cell_summary: Dict[str, Dict[str, Tuple[float, float]]],
    per_ds: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    per_elem: Dict[str, Dict[str, Tuple[float, float]]],
    winner: Optional[str],
    promotion_log: Sequence[str],
    path: Path,
    seed_count: int,
    dataset_shard: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Empirical 07: alias-fix sweep (Phase C of jaunty-weaving-mist)")
    lines.append("")
    lines.append(
        f"- Cells: {len(_CELL_ORDER)} (2³ combinations of ftp1/762f/dj6y)"
    )
    lines.append(f"- Seeds per cell: {seed_count}")
    lines.append(f"- Dataset shard: `{dataset_shard}`")
    lines.append(f"- Promotion target: `{_PROMOTION_TARGET_DATASET}`")
    lines.append(
        f"- Promotion guards: {', '.join(repr(d) for d in _PROMOTION_GUARD_DATASETS)} "
        f"(no metric regression > {_PROMOTION_GUARD_TOL})"
    )
    lines.append("")
    lines.append("## Overall metrics (mean ± stdev across seeds)")
    lines.append("")
    header = ["cell"] + list(_OVERALL_METRICS)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for cell in _CELL_ORDER:
        if cell not in cell_summary:
            lines.append("| " + " | ".join([cell] + ["n/a"] * len(_OVERALL_METRICS)) + " |")
            continue
        row_cells = [cell] + [
            _fmt_mean_stdev(cell_summary[cell][m]) for m in _OVERALL_METRICS
        ]
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    # Per-dataset breakdown for the promotion target + guard datasets
    relevant_dsets = (_PROMOTION_TARGET_DATASET,) + _PROMOTION_GUARD_DATASETS
    lines.append("## Per-dataset macro_f1 (mean ± stdev)")
    lines.append("")
    header_ds = ["cell"] + list(relevant_dsets)
    lines.append("| " + " | ".join(header_ds) + " |")
    lines.append("|" + "|".join(["---"] * len(header_ds)) + "|")
    for cell in _CELL_ORDER:
        cell_block = per_ds.get(cell, {})
        row_cells = [cell]
        for ds in relevant_dsets:
            metric_pair = cell_block.get(ds, {}).get("macro_f1", (float("nan"), float("nan")))
            row_cells.append(_fmt_mean_stdev(metric_pair))
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    # Per-element f1 delta vs baseline for the focus elements
    lines.append("## Per-element f1 delta vs baseline (focus: " + ", ".join(_FOCUS_ELEMENTS) + ")")
    lines.append("")
    baseline_per_elem = per_elem.get("baseline", {})
    header_el = ["cell"] + [f"{el} f1 (delta)" for el in _FOCUS_ELEMENTS]
    lines.append("| " + " | ".join(header_el) + " |")
    lines.append("|" + "|".join(["---"] * len(header_el)) + "|")
    for cell in _CELL_ORDER:
        cell_block = per_elem.get(cell, {})
        row_cells = [cell]
        for el in _FOCUS_ELEMENTS:
            mean_pair = cell_block.get(el, (float("nan"), float("nan")))
            base_pair = baseline_per_elem.get(el, (float("nan"), float("nan")))
            if math.isnan(mean_pair[0]) or math.isnan(base_pair[0]):
                row_cells.append("n/a")
                continue
            delta = mean_pair[0] - base_pair[0]
            row_cells.append(f"{mean_pair[0]:.4f} ({delta:+.4f})")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    lines.append("## Promotion rule decision")
    lines.append("")
    lines.append("> Pick the cell with the highest macro_f1 lift on")
    lines.append(f"> `{_PROMOTION_TARGET_DATASET}` subject to no regression")
    lines.append(
        f"> > {_PROMOTION_GUARD_TOL} on any metric on `"
        + "` / `".join(_PROMOTION_GUARD_DATASETS)
        + "`."
    )
    lines.append(
        f"> Ties within {_PROMOTION_TIE_TOL} macro_f1 → prefer simpler (fewer flags)."
    )
    lines.append("")
    if winner is None:
        lines.append("**Result:** undetermined — see log below.")
    elif winner == "baseline":
        lines.append(
            "**Result:** `baseline` wins. The alias-fix diagnosis was "
            "over-confident — none of the fix-on cells beat baseline on "
            f"`{_PROMOTION_TARGET_DATASET}` without regressing the guards."
        )
    else:
        lines.append(f"**Winner:** `{winner}`")
    lines.append("")
    lines.append("### Decision log")
    lines.append("")
    lines.append("```")
    for line in promotion_log:
        lines.append(line)
    lines.append("```")
    lines.append("")

    # Failure summary
    failures = [r for r in rows if r.failure]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            lines.append(f"- cell=`{r.cell}` seed={r.seed}: {r.failure}")
        lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if args.smoke_mode:
        return _run_smoke()

    if args.output_dir is None:
        parser.error("--output-dir is required unless --smoke-mode is set.")
    output_dir: Path = args.output_dir

    cells = _resolve_cells(args.cells)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    runs: List[CellRun] = [
        CellRun(cell=cell, seed=seed, output_dir=output_dir / f"cell-{cell}-seed-{seed}")
        for cell in cells
        for seed in seeds
    ]

    if not args.aggregate_only:
        if not _RUN_UNIFIED.is_file():
            raise SystemExit(f"run_unified_benchmark.py not found at {_RUN_UNIFIED}")
        if args.dry_run:
            for run in runs:
                cmd = _build_run_command(
                    python=args.python,
                    run_unified=_RUN_UNIFIED,
                    cell=run.cell,
                    seed=run.seed,
                    basis_dir=args.basis_dir,
                    dataset_shard=args.dataset_shard,
                    output_dir=run.output_dir,
                )
                print("DRY-RUN:", " ".join(cmd))
            return 0
        _check_basis_dir(args.basis_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_run = 0
        n_skip = 0
        n_fail = 0
        for run in runs:
            if run.id_summary_path.is_file():
                print(
                    f"[sweep] SKIP cell={run.cell!r} seed={run.seed} "
                    f"(id_summary.json already exists at {run.id_summary_path})",
                    flush=True,
                )
                n_skip += 1
                continue
            cmd = _build_run_command(
                python=args.python,
                run_unified=_RUN_UNIFIED,
                cell=run.cell,
                seed=run.seed,
                basis_dir=args.basis_dir,
                dataset_shard=args.dataset_shard,
                output_dir=run.output_dir,
            )
            rc = _run_one_cell(run, cmd)
            n_run += 1
            if rc != 0:
                n_fail += 1
        print(
            f"[sweep] dispatch complete: ran={n_run} skipped={n_skip} failed={n_fail}"
        )

    # Aggregate whatever we have
    rows = _aggregate(runs)

    # Date for filenames
    if args.report_date:
        date_str = args.report_date
    else:
        # Prefer GIT_AUTHOR_DATE-style ISO; fall back to today UTC.
        from datetime import datetime, timezone

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    csv_path = args.report_dir / f"{date_str}-empirical-07-alias-fix-sweep.csv"
    md_path = args.report_dir / f"{date_str}-empirical-07-alias-fix-sweep.md"

    _write_csv(rows, csv_path)
    print(f"[sweep] wrote CSV: {csv_path}")

    cell_summary = _summarize_per_cell(rows)
    per_ds = _summarize_per_cell_per_dataset(
        rows, (_PROMOTION_TARGET_DATASET, *_PROMOTION_GUARD_DATASETS)
    )
    per_elem = _summarize_per_cell_per_element(rows, _FOCUS_ELEMENTS, metric="f1")

    winner, promotion_log = _apply_promotion_rule(cell_summary, per_ds)

    _write_markdown(
        rows=rows,
        cell_summary=cell_summary,
        per_ds=per_ds,
        per_elem=per_elem,
        winner=winner,
        promotion_log=promotion_log,
        path=md_path,
        seed_count=len(seeds),
        dataset_shard=args.dataset_shard,
    )
    print(f"[sweep] wrote Markdown: {md_path}")

    n_fail = sum(1 for r in rows if r.failure)
    if n_fail:
        print(f"[sweep] note: {n_fail} cell-seed runs failed to produce metrics.")
    return 0 if not n_fail else 1


if __name__ == "__main__":
    sys.exit(main())
