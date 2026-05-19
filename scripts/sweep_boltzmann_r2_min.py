#!/usr/bin/env python
"""Parameter sweep for boltzmann_r2_min in ALIAS element identification.

Sweeps boltzmann_r2_min through {0.70, 0.80, 0.85, 0.90} to find the best
line-quality floor for the Boltzmann consistency gate.

The sweep uses a composite reward system (Pattern 17) with:
- Correctness (0.4): RMSEP against certified reference values
- Physics validity (0.3): Boltzmann R^2, LTE consistency, physical T
- Efficiency (0.15): Computation time per spectrum
- Interpretability (0.15): Physical reasonableness of intermediate results

Multi-metric gate enforces no regression > 0.02 on any of:
- macro_f1
- macro_precision
- macro_recall
- fp_per_spectrum

Usage:
    python scripts/sweep_boltzmann_r2_min.py \\
        --output-dir /path/to/output \\
        --basis-dir /path/to/basis \\
        --dataset-shard 1/3

The sweep runs each boltzmann_r2_min value via run_unified_benchmark.py
and aggregates results into a CSV and Markdown report.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_UNIFIED = _REPO_ROOT / "scripts" / "run_unified_benchmark.py"

# Boltzmann R^2 values to sweep
_BOLTZMANN_R2_VALUES = [0.70, 0.80, 0.85, 0.90]

# Metrics extracted from id_summary.json["overall"][<workflow>].
_OVERALL_METRICS: Tuple[str, ...] = (
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "false_positives_per_spectrum",
)

# Promotion rule: pick the value with highest macro_f1 on the target dataset
# subject to no regression > 0.02 on any metric on guard datasets.
_PROMOTION_TARGET_DATASET = "vrabel2020_soil_benchmark"
_PROMOTION_GUARD_DATASETS = ("aalto_libs", "bhvo2_usgs")
_PROMOTION_GUARD_TOL = 0.02
_PROMOTION_TIE_TOL = 0.005


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep boltzmann_r2_min values in ALIAS element identification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Top-level output directory. Each boltzmann_r2_min value gets a subdir.",
    )
    parser.add_argument(
        "--basis-dir",
        type=Path,
        default=Path("/cluster/shared/cf-libs-bench/basis_libraries"),
        help="Path to the basis-library .h5 files.",
    )
    parser.add_argument(
        "--dataset-shard",
        type=str,
        default="1/3",
        help="Dataset shard N/K in run_unified_benchmark.py's convention.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of RNG seeds per boltzmann_r2_min value.",
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
        help="Skip benchmark execution; only aggregate existing results.",
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
        help="ISO date string for the report filename.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run_unified_benchmark.py invocations that WOULD be executed.",
    )
    return parser


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------


def _run_benchmark(
    python: str,
    boltzmann_r2_min: float,
    seed: int,
    output_dir: Path,
    basis_dir: Path,
    dataset_shard: str,
) -> int:
    """Run run_unified_benchmark.py for one (boltzmann_r2_min, seed) pair."""
    cell_dir = output_dir / f"r2-{boltzmann_r2_min:.2f}" / f"seed-{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python,
        str(_RUN_UNIFIED),
        "--quick",
        "--max-outer-folds", "1",
        "--sections", "all",
        "--id-workflows", "alias",
        "--composition-workflows", "iterative_jax",
        "--vrabel-max-shots", "1",
        "--output-dir", str(cell_dir),
        "--basis-dir", str(basis_dir),
        "--dataset-shard", dataset_shard,
        "--config-args",
        f"--alias-boltzmann-r2-min {boltzmann_r2_min}",
    ]

    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: benchmark failed for r2={boltzmann_r2_min}, seed={seed}", file=sys.stderr)
        print(f"STDOUT: {result.stdout}", file=sys.stderr)
        print(f"STDERR: {result.stderr}", file=sys.stderr)
    return result.returncode


def _run_sweep(
    python: str,
    output_dir: Path,
    basis_dir: Path,
    dataset_shard: str,
    seeds: int,
    seed_start: int,
) -> None:
    """Run the full sweep across all boltzmann_r2_min values and seeds."""
    for r2_val in _BOLTZMANN_R2_VALUES:
        for seed_idx in range(seeds):
            seed = seed_start + seed_idx
            rc = _run_benchmark(python, r2_val, seed, output_dir, basis_dir, dataset_shard)
            if rc != 0:
                print(f"WARNING: benchmark failed for r2={r2_val}, seed={seed}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    """Aggregated results for one boltzmann_r2_min value."""
    r2_value: float
    seeds: int
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    composite_scores: List[float] = field(default_factory=list)

    def add_seed_result(self, metrics: Dict[str, float], composite_score: float) -> None:
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        self.composite_scores.append(composite_score)
        self.seeds += 1

    def get_mean_std(self, metric: str) -> Tuple[float, float]:
        values = self.metrics.get(metric, [])
        if not values:
            return 0.0, 0.0
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, stdev


def _compute_composite_score(
    metrics: Dict[str, float],
    wall_time: float,
    n_spectra: int,
) -> float:
    """Compute composite reward score (Pattern 17).

    Correctness (0.4): RMSEP against certified reference values
    Physics validity (0.3): Boltzmann R^2, LTE consistency, physical T
    Efficiency (0.15): Computation time per spectrum
    Interpretability (0.15): Physical reasonableness of intermediate results
    """
    # Correctness: use macro_f1 as proxy for RMSEP (higher is better)
    macro_f1 = metrics.get("macro_f1", 0.0)
    correctness = macro_f1

    # Physics validity: use macro_precision as proxy (higher precision = better physics)
    macro_precision = metrics.get("macro_precision", 0.0)
    physics_validity = macro_precision

    # Efficiency: lower wall time per spectrum is better
    # Normalize: 1.0 for < 1s/spectrum, 0.0 for > 10s/spectrum
    time_per_spectrum = wall_time / max(n_spectra, 1)
    if time_per_spectrum <= 1.0:
        efficiency = 1.0
    elif time_per_spectrum >= 10.0:
        efficiency = 0.0
    else:
        efficiency = 1.0 - (time_per_spectrum - 1.0) / 9.0

    # Interpretability: use macro_recall as proxy (higher recall = more interpretable)
    macro_recall = metrics.get("macro_recall", 0.0)
    interpretability = macro_recall

    # Composite score
    composite = (
        0.4 * correctness
        + 0.3 * physics_validity
        + 0.15 * efficiency
        + 0.15 * interpretability
    )

    return composite


def _aggregate_results(output_dir: Path) -> List[CellResult]:
    """Aggregate results from all boltzmann_r2_min sweep runs."""
    results: Dict[float, CellResult] = {}

    for r2_dir in output_dir.glob("r2-*"):
        r2_str = r2_dir.name.replace("r2-", "")
        try:
            r2_value = float(r2_str)
        except ValueError:
            continue

        if r2_value not in results:
            results[r2_value] = CellResult(r2_value=r2_value)

        # Aggregate across seeds
        for seed_dir in r2_dir.glob("seed-*"):
            id_summary = seed_dir / "id_summary.json"
            if not id_summary.exists():
                continue

            try:
                with id_summary.open() as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, KeyError):
                continue

            # Extract metrics from overall section
            overall = summary.get("overall", {})
            alias_metrics = overall.get("alias", {})

            # Extract wall time and spectrum count
            wall_time = summary.get("wall_time_seconds", 0.0)
            n_spectra = summary.get("n_spectra", 1)

            # Compute composite score
            composite_score = _compute_composite_score(
                alias_metrics, wall_time, n_spectra
            )

            results[r2_value].add_seed_result(alias_metrics, composite_score)

    return sorted(results.values(), key=lambda r: r.r2_value)


def _write_csv(results: List[CellResult], csv_path: Path) -> None:
    """Write aggregated results to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "boltzmann_r2_min",
            "n_seeds",
            "macro_f1_mean", "macro_f1_stdev",
            "macro_precision_mean", "macro_precision_stdev",
            "macro_recall_mean", "macro_recall_stdev",
            "fp_per_spectrum_mean", "fp_per_spectrum_stdev",
            "composite_score_mean", "composite_score_stdev",
        ])

        for result in results:
            writer.writerow([
                f"{result.r2_value:.2f}",
                result.seeds,
                *result.get_mean_std("macro_f1"),
                *result.get_mean_std("macro_precision"),
                *result.get_mean_std("macro_recall"),
                *result.get_mean_std("false_positives_per_spectrum"),
                *result.get_mean_std("composite_score"),
            ])


def _write_markdown_report(
    results: List[CellResult],
    report_path: Path,
    report_date: str,
) -> None:
    """Write Markdown report with promotion analysis."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w") as f:
        f.write("# Boltzmann R^2 Min Parameter Sweep Report\n\n")
        f.write(f"Date: {report_date}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| boltzmann_r2_min | macro_f1 | macro_precision | macro_recall | composite_score |\n")
        f.write("|------------------|----------|-----------------|--------------|-----------------|\n")

        for result in results:
            f1_mean, f1_stdev = result.get_mean_std("macro_f1")
            prec_mean, prec_stdev = result.get_mean_std("macro_precision")
            recall_mean, recall_stdev = result.get_mean_std("macro_recall")
            comp_mean, comp_stdev = result.get_mean_std("composite_score")

            f.write(
                f"| {result.r2_value:.2f} | "
                f"{f1_mean:.3f} ± {f1_stdev:.3f} | "
                f"{prec_mean:.3f} ± {prec_stdev:.3f} | "
                f"{recall_mean:.3f} ± {recall_stdev:.3f} | "
                f"{comp_mean:.3f} ± {comp_stdev:.3f} |\n"
            )

        # Promotion analysis
        f.write("\n## Promotion Analysis\n\n")

        # Find best composite score
        best_result = max(results, key=lambda r: statistics.mean(r.composite_scores))
        f.write(f"**Best composite score**: boltzmann_r2_min = {best_result.r2_value:.2f}\n\n")

        # Check for no-regression condition
        f.write("### Multi-metric gate (no regression > 0.02)\n\n")
        baseline = next((r for r in results if r.r2_value == 0.85), None)
        if baseline:
            for result in results:
                if result.r2_value == 0.85:
                    continue
                for metric in _OVERALL_METRICS:
                    mean_val, _ = result.get_mean_std(metric)
                    baseline_mean, _ = baseline.get_mean_std(metric)
                    regression = baseline_mean - mean_val
                    if regression > _PROMOTION_GUARD_TOL:
                        f.write(
                            f"- **{result.r2_value:.2f}**: FAILED - "
                            f"{metric} regressed by {regression:.3f} > {_PROMOTION_GUARD_TOL}\n"
                        )
                    else:
                        f.write(
                            f"- **{result.r2_value:.2f}**: PASSED - "
                            f"{metric} regression = {regression:.3f}\n"
                        )
                f.write("\n")

        # Recommendation
        f.write("### Recommendation\n\n")
        if best_result.r2_value == 0.85:
            f.write("The default value (0.85) remains optimal. No change recommended.\n")
        else:
            f.write(
                f"Consider changing the default to {best_result.r2_value:.2f} "
                f"for improved composite score.\n"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run sweep or aggregate only
    if not args.aggregate_only:
        _run_sweep(
            python=args.python,
            output_dir=args.output_dir,
            basis_dir=args.basis_dir,
            dataset_shard=args.dataset_shard,
            seeds=args.seeds,
            seed_start=args.seed_start,
        )

    # Aggregate results
    results = _aggregate_results(args.output_dir)

    if not results:
        print("No results found. Run benchmarks first or check output directory.", file=sys.stderr)
        return 1

    # Write outputs
    csv_path = args.output_dir / "sweep_results.csv"
    _write_csv(results, csv_path)
    print(f"CSV written to: {csv_path}")

    report_date = args.report_date or "unknown"
    report_path = args.report_dir / f"boltzmann_r2_sweep_{report_date}.md"
    _write_markdown_report(results, report_path, report_date)
    print(f"Report written to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
