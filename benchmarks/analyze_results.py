#!/usr/bin/env python3
"""Benchmark results analysis and figure-data generation.

Reads all 6 benchmark JSON outputs and produces figure-ready CSV data files
and a master summary JSON for the paper. Does NOT generate plots -- only
clean tabular data for Phase 5 (paper writing).

Expected inputs (from benchmarks/results/):
  - voigt_results.json         (BENCH-01: Voigt throughput)
  - boltzmann_results.json     (BENCH-02: Boltzmann fitting)
  - anderson_results.json      (BENCH-03: Anderson convergence)
  - faiss_results.json         (BENCH-04: FAISS latency)
  - batch_forward_results.json (BENCH-05: Batch forward scaling)
  - e2e_pipeline_results.json  (BENCH-06: End-to-end pipeline)

Outputs (to benchmarks/figures/):
  - fig2_voigt_throughput.csv
  - fig3_boltzmann_speedup.csv
  - fig4a_anderson_iteration_counts.csv
  - fig4b_anderson_residual_trajectories.csv
  - fig5_faiss_latency.csv
  - fig6_batch_scaling.csv
  - fig7_e2e_breakdown.csv
  - benchmark_summary.json

Usage
-----
    python benchmarks/analyze_results.py --input-dir benchmarks/results/ --output-dir benchmarks/figures/
    python benchmarks/analyze_results.py --mock --output-dir /tmp/test_figures/

References
----------
Kawahara et al. (2022) arXiv:2105.14782 -- ExoJAX benchmark comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import print_table  # noqa: E402


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------
def load_json(path: str) -> dict | None:
    """Load a JSON file, returning None if missing or invalid."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  WARNING: {path} not found -- skipping")
        return None
    except json.JSONDecodeError as e:
        print(f"  WARNING: {path} has invalid JSON: {e} -- skipping")
        return None


def write_csv(path: str, headers: list[str], rows: list[list]) -> None:
    """Write a CSV file with given headers and rows."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  Wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Mock data generation (for pipeline testing only)
# ---------------------------------------------------------------------------
def generate_mock_data() -> dict[str, dict]:
    """Generate synthetic mock data matching expected JSON schemas.

    WARNING: This produces FAKE data for testing the analysis pipeline only.
    It must NEVER be presented as real benchmark results.
    """
    print("=" * 60)
    print("WARNING: USING MOCK DATA -- NOT REAL BENCHMARK RESULTS")
    print("WARNING: USING MOCK DATA -- NOT REAL BENCHMARK RESULTS")
    print("=" * 60)

    hw = {
        "cpu_model": "MOCK",
        "gpu_model": "MOCK V100S",
        "jax_backend": "MOCK",
        "timestamp_utc": "MOCK",
    }

    # Voigt (BENCH-01)
    grid_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    voigt = {
        "benchmark": "voigt_throughput",
        "hardware": hw,
        "parameters": {"grid_sizes": grid_sizes, "n_lines": 10, "n_runs": 10},
        "results": {
            "cpu_throughput": [1e7 * (1 + 0.1 * i) for i in range(7)],
            "gpu_throughput": [5e6 * (1 + 0.5 * i) for i in range(7)],
            "cpu_time_mean": [g * 10 / 1e7 for g in grid_sizes],
            "gpu_time_mean": [g * 10 / 5e6 for g in grid_sizes],
            "cpu_time_std": [0.001] * 7,
            "gpu_time_std": [0.001] * 7,
            "max_relative_error": [5e-14] * 7,
            "speedup": [
                c / g
                for c, g in zip(
                    [1e7 * (1 + 0.1 * i) for i in range(7)],
                    [5e6 * (1 + 0.5 * i) for i in range(7)],
                )
            ],
        },
    }

    # Boltzmann (BENCH-02)
    elem_counts = [1, 2, 3, 5, 7, 10, 15, 20]
    lines_per = [10, 25, 50, 100]
    boltz_results = []
    for ec in elem_counts:
        for lp in lines_per:
            cpu_t = ec * lp * 0.001
            gpu_t = max(0.03, cpu_t / (1 + ec * 0.5))
            boltz_results.append(
                {
                    "element_count": ec,
                    "lines_per_element": lp,
                    "cpu_time_ms_mean": cpu_t,
                    "cpu_time_ms_std": cpu_t * 0.1,
                    "gpu_time_ms_mean": gpu_t,
                    "gpu_time_ms_std": gpu_t * 0.1,
                    "speedup": cpu_t / gpu_t,
                }
            )
    boltzmann = {
        "benchmark": "boltzmann_fitting",
        "hardware": hw,
        "parameters": {
            "element_counts": elem_counts,
            "lines_per_element": lines_per,
            "n_runs": 10,
        },
        "results": boltz_results,
    }

    # Anderson (BENCH-03)
    M_values = list(range(11))
    conditions = [
        {"T_eV": 0.6, "n_e_init": 1e15, "label": "low-T/low-ne"},
        {"T_eV": 0.8, "n_e_init": 1e16, "label": "mid-T/mid-ne"},
        {"T_eV": 1.0, "n_e_init": 1e17, "label": "T=1eV/high-ne"},
        {"T_eV": 1.5, "n_e_init": 1e16, "label": "highest-T"},
    ]
    n_cond = len(conditions)
    # iteration_counts[condition][M]
    iter_counts = []
    for ci in range(n_cond):
        picard = 5 + ci * 2
        row = [picard] + [max(3, picard - M) for M in range(1, 11)]
        iter_counts.append(row)
    # residual trajectories: [condition][M] -> list of residuals per iteration
    residuals = []
    for ci in range(n_cond):
        cond_res = []
        for M in M_values:
            n_iter = iter_counts[ci][M]
            cond_res.append([10 ** (-i) for i in range(n_iter + 1)])
        residuals.append(cond_res)

    anderson = {
        "benchmark": "anderson_convergence",
        "hardware": hw,
        "parameters": {
            "M_values": M_values,
            "test_conditions": conditions,
            "tol": 1e-6,
            "max_iter": 200,
            "n_runs": 5,
        },
        "results": {
            "iteration_counts": iter_counts,
            "converged": [[True] * 11 for _ in range(n_cond)],
            "residual_trajectories": residuals,
            "wall_times": [[0.1 * (M + 1) for M in M_values] for _ in range(n_cond)],
        },
    }

    # FAISS (BENCH-04)
    db_sizes = [1000, 10000, 100000]
    faiss_results = []
    for ds in db_sizes:
        faiss_results.append(
            {
                "db_size": ds,
                "cpu_flat_search_ms_mean": ds * 0.0002,
                "cpu_flat_search_ms_std": ds * 0.00002,
                "cpu_ivf_search_ms_mean": max(0.5, ds * 0.00003),
                "cpu_ivf_search_ms_std": 0.1,
                "cpu_ivf_recall_at_1": max(0.5, 1.0 - ds * 3e-6),
                "gpu_flat_search_ms_mean": ds * 0.00005,
                "gpu_flat_search_ms_std": ds * 0.000005,
                "gpu_ivf_search_ms_mean": max(0.1, ds * 0.000008),
                "gpu_ivf_search_ms_std": 0.02,
                "gpu_ivf_recall_at_1": max(0.5, 1.0 - ds * 3e-6),
            }
        )
    faiss = {
        "benchmark": "faiss_query_latency",
        "hardware": hw,
        "parameters": {"db_sizes": db_sizes, "d": 30, "k": 10},
        "results": faiss_results,
    }

    # Batch forward (BENCH-05)
    batch_sizes_bf = [1, 5, 10, 50, 100, 500, 1000]
    bf_results = []
    for bs in batch_sizes_bf:
        cpu_ms = bs * 6.0
        gpu_ms = max(5.0, bs * 0.5)
        bf_results.append(
            {
                "batch_size": bs,
                "cpu_total_time_ms_mean": cpu_ms,
                "cpu_total_time_ms_std": cpu_ms * 0.05,
                "cpu_per_spectrum_ms": 6.0,
                "cpu_throughput_spectra_per_sec": 1000.0 / 6.0,
                "gpu_total_time_ms_mean": gpu_ms,
                "gpu_total_time_ms_std": gpu_ms * 0.1,
                "gpu_per_spectrum_ms": gpu_ms / bs,
                "gpu_throughput_spectra_per_sec": bs / (gpu_ms / 1000.0),
                "estimated_memory_gb": bs * 4096 * 50 * 8 / 1e9,
                "speedup": cpu_ms / gpu_ms,
            }
        )
    batch_forward = {
        "benchmark": "batch_forward_model",
        "hardware": hw,
        "parameters": {"batch_sizes": batch_sizes_bf, "n_lines": 50, "n_wl": 4096},
        "results": bf_results,
    }

    # E2E pipeline (BENCH-06)
    batch_sizes_e2e = [1, 10, 100, 1000, 10000]
    e2e_results = []
    for bs in batch_sizes_e2e:
        cpu_ms = bs * 8.0
        gpu_ms = max(10.0, bs * 0.3)
        e2e_results.append(
            {
                "batch_size": bs,
                "cpu_total_ms_mean": cpu_ms,
                "cpu_total_ms_std": cpu_ms * 0.05,
                "gpu_total_ms_mean": gpu_ms,
                "gpu_total_ms_std": gpu_ms * 0.1,
                "gpu_compute_ms_mean": gpu_ms * 0.9,
                "gpu_transfer_ms_mean": gpu_ms * 0.1,
                "speedup": cpu_ms / gpu_ms,
                "speedup_no_transfer": cpu_ms / (gpu_ms * 0.9),
                "cpu_component_breakdown": {
                    "data_prep": cpu_ms * 0.01,
                    "saha_boltzmann": cpu_ms * 0.35,
                    "boltzmann_fit": cpu_ms * 0.05,
                    "voigt_profiles": cpu_ms * 0.45,
                    "assembly": cpu_ms * 0.09,
                    "closure": cpu_ms * 0.05,
                },
                "gpu_component_breakdown": {
                    "transfer": gpu_ms * 0.1,
                    "forward_model": gpu_ms * 0.7,
                    "boltzmann_fit": gpu_ms * 0.15,
                    "closure": gpu_ms * 0.05,
                },
            }
        )
    e2e = {
        "benchmark": "end_to_end_pipeline",
        "hardware": hw,
        "parameters": {
            "batch_sizes": batch_sizes_e2e,
            "n_elements": 5,
            "n_lines": 50,
            "n_wl": 4096,
        },
        "results": e2e_results,
        "summary": {
            "crossover_batch_size": 10,
            "max_speedup": 26.67,
            "max_speedup_batch_size": 10000,
            "bottleneck_cpu": "voigt_profiles",
            "bottleneck_gpu": "forward_model",
        },
    }

    return {
        "voigt": voigt,
        "boltzmann": boltzmann,
        "anderson": anderson,
        "faiss": faiss,
        "batch_forward": batch_forward,
        "e2e_pipeline": e2e,
    }


# ---------------------------------------------------------------------------
# Figure data extraction
# ---------------------------------------------------------------------------
def extract_fig2_voigt(data: dict, output_dir: str) -> dict | None:
    """Figure 2: Voigt throughput vs grid size."""
    r = data["results"]
    grid_sizes = data["parameters"]["grid_sizes"]
    headers = ["grid_size", "cpu_throughput", "gpu_throughput", "speedup", "max_relative_error"]
    rows = []
    for i, gs in enumerate(grid_sizes):
        rows.append(
            [
                gs,
                round(r["cpu_throughput"][i], 2),
                round(r["gpu_throughput"][i], 2),
                round(r["speedup"][i], 4),
                f"{r['max_relative_error'][i]:.2e}",
            ]
        )
    write_csv(os.path.join(output_dir, "fig2_voigt_throughput.csv"), headers, rows)
    # Return headline numbers
    speedups = r["speedup"]
    max_idx = int(np.argmax(speedups))
    return {
        "voigt_max_speedup": round(max(speedups), 2),
        "voigt_max_speedup_grid_size": grid_sizes[max_idx],
    }


def extract_fig3_boltzmann(data: dict, output_dir: str) -> dict | None:
    """Figure 3: Boltzmann fitting speedup vs element count."""
    results = data["results"]
    headers = ["element_count", "lines_per_element", "cpu_time_ms", "gpu_time_ms", "speedup"]
    rows = []
    max_speedup = 0.0
    for r in results:
        rows.append(
            [
                r["element_count"],
                r["lines_per_element"],
                round(r["cpu_time_ms_mean"], 4),
                round(r["gpu_time_ms_mean"], 4),
                round(r["speedup"], 2),
            ]
        )
        if r["speedup"] > max_speedup:
            max_speedup = r["speedup"]
    write_csv(os.path.join(output_dir, "fig3_boltzmann_speedup.csv"), headers, rows)
    return {"boltzmann_max_speedup": round(max_speedup, 2)}


def extract_fig4_anderson(data: dict, output_dir: str) -> dict | None:
    """Figure 4: Anderson convergence.

    4a: iteration counts vs M for each condition
    4b: residual trajectories for selected conditions and M values
    """
    params = data["parameters"]
    results = data["results"]
    M_values = params["M_values"]
    conditions = params["test_conditions"]
    iter_counts = results["iteration_counts"]

    # Fig 4a: iteration counts
    headers_a = ["M"] + [c["label"] for c in conditions] + ["mean_iters"]
    rows_a = []
    for mi, M in enumerate(M_values):
        row = [M]
        iters_at_M = []
        for ci in range(len(conditions)):
            it = iter_counts[ci][mi]
            row.append(it)
            iters_at_M.append(it)
        row.append(round(np.mean(iters_at_M), 2))
        rows_a.append(row)
    write_csv(os.path.join(output_dir, "fig4a_anderson_iteration_counts.csv"), headers_a, rows_a)

    # Fig 4b: residual trajectories
    residuals_raw = results.get("residual_trajectories", {})
    if residuals_raw:
        # Normalize the structure: can be list-of-list or dict-of-dict
        M_select = [0, 3] if 3 in M_values else [0, min(M_values[-1], 3)]

        # Extract trajectories into a uniform structure: traj[ci][M] -> list
        traj_data: dict[int, dict[int, list]] = {}
        if isinstance(residuals_raw, dict):
            # Format: {"condition_0": {"M_0": [...], "M_3": [...]}, ...}
            for ci in range(len(conditions)):
                key = f"condition_{ci}"
                cond_dict = residuals_raw.get(key, {})
                traj_data[ci] = {}
                for M in M_select:
                    m_key = f"M_{M}"
                    traj_data[ci][M] = cond_dict.get(m_key, [])
        elif isinstance(residuals_raw, list):
            # Format: list[condition][M_index] -> list of residuals
            for ci in range(min(len(residuals_raw), len(conditions))):
                traj_data[ci] = {}
                for M in M_select:
                    mi = M_values.index(M) if M in M_values else 0
                    if isinstance(residuals_raw[ci], list) and mi < len(residuals_raw[ci]):
                        traj_data[ci][M] = residuals_raw[ci][mi]
                    elif isinstance(residuals_raw[ci], dict):
                        traj_data[ci][M] = residuals_raw[ci].get(f"M_{M}", [])
                    else:
                        traj_data[ci][M] = []

        # Find max trajectory length
        max_iter = 0
        for ci_trajs in traj_data.values():
            for traj in ci_trajs.values():
                max_iter = max(max_iter, len(traj))

        if max_iter > 0:
            headers_b = ["iteration"]
            for ci, c in enumerate(conditions):
                for M in M_select:
                    headers_b.append(f"{c['label']}_M{M}")

            rows_b = []
            for it_idx in range(max_iter):
                row = [it_idx]
                for ci in range(len(conditions)):
                    for M in M_select:
                        traj = traj_data.get(ci, {}).get(M, [])
                        if it_idx < len(traj):
                            row.append(f"{traj[it_idx]:.6e}")
                        else:
                            row.append("")
                rows_b.append(row)
            write_csv(
                os.path.join(output_dir, "fig4b_anderson_residual_trajectories.csv"),
                headers_b,
                rows_b,
            )

    # Headline: optimal M and average iteration reduction
    picard_iters = [iter_counts[ci][0] for ci in range(len(conditions))]
    best_M = 0
    best_reduction = 0.0
    for mi, M in enumerate(M_values):
        if M == 0:
            continue
        aa_iters = [iter_counts[ci][mi] for ci in range(len(conditions))]
        reduction = np.mean(picard_iters) / np.mean(aa_iters)
        if reduction > best_reduction:
            best_reduction = reduction
            best_M = M

    return {
        "anderson_optimal_M": best_M,
        "anderson_avg_iteration_reduction": round(float(best_reduction), 2),
    }


def extract_fig5_faiss(data: dict, output_dir: str) -> dict | None:
    """Figure 5: FAISS query latency vs database size."""
    results = data["results"]
    headers = [
        "db_size",
        "cpu_flat_ms",
        "cpu_ivf_ms",
        "gpu_flat_ms",
        "gpu_ivf_ms",
        "recall_ivf",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r["db_size"],
                round(r["cpu_flat_search_ms_mean"], 4),
                round(r["cpu_ivf_search_ms_mean"], 4),
                _fmt_nullable(r.get("gpu_flat_search_ms_mean")),
                _fmt_nullable(r.get("gpu_ivf_search_ms_mean")),
                _fmt_nullable(r.get("cpu_ivf_recall_at_1") or r.get("gpu_ivf_recall_at_1")),
            ]
        )
    write_csv(os.path.join(output_dir, "fig5_faiss_latency.csv"), headers, rows)

    # GPU vs CPU speedup at largest db_size
    largest = results[-1]
    gpu_flat = largest.get("gpu_flat_search_ms_mean")
    cpu_flat = largest.get("cpu_flat_search_ms_mean")
    if gpu_flat and cpu_flat and gpu_flat > 0:
        speedup = round(cpu_flat / gpu_flat, 2)
    else:
        speedup = None

    return {"faiss_gpu_vs_cpu_speedup_largest": speedup}


def extract_fig6_batch_forward(data: dict, output_dir: str) -> dict | None:
    """Figure 6: Batch forward model scaling."""
    results = data["results"]
    headers = ["batch_size", "cpu_spectra_per_sec", "gpu_spectra_per_sec", "speedup", "memory_gb"]
    rows = []
    max_throughput = 0.0
    for r in results:
        if r.get("skipped"):
            continue
        cpu_sps = r.get("cpu_throughput_spectra_per_sec", 0) or 0
        gpu_sps = r.get("gpu_throughput_spectra_per_sec") or 0
        speedup = r.get("speedup")
        mem = r.get("estimated_memory_gb", 0)
        rows.append(
            [
                r["batch_size"],
                round(cpu_sps, 2),
                _fmt_nullable(gpu_sps if gpu_sps > 0 else None),
                _fmt_nullable(speedup),
                round(mem, 4),
            ]
        )
        if gpu_sps > max_throughput:
            max_throughput = gpu_sps
        if cpu_sps > max_throughput:
            max_throughput = cpu_sps
    write_csv(os.path.join(output_dir, "fig6_batch_scaling.csv"), headers, rows)
    return {"batch_forward_max_throughput_spectra_per_sec": round(max_throughput, 2)}


def extract_fig7_e2e(data: dict, output_dir: str) -> dict | None:
    """Figure 7: End-to-end pipeline breakdown."""
    results = data["results"]
    headers = [
        "batch_size",
        "cpu_total_ms",
        "gpu_total_ms",
        "speedup",
        "cpu_saha_ms",
        "cpu_voigt_ms",
        "cpu_assembly_ms",
        "cpu_boltzmann_ms",
        "cpu_closure_ms",
        "gpu_transfer_ms",
        "gpu_forward_ms",
        "gpu_boltzmann_ms",
        "gpu_closure_ms",
    ]
    rows = []
    for r in results:
        cpu_bd = r.get("cpu_component_breakdown", {})
        gpu_bd = r.get("gpu_component_breakdown", {})
        rows.append(
            [
                r["batch_size"],
                _fmt_nullable(r.get("cpu_total_ms_mean")),
                _fmt_nullable(r.get("gpu_total_ms_mean")),
                _fmt_nullable(r.get("speedup")),
                _fmt_nullable(cpu_bd.get("saha_boltzmann")),
                _fmt_nullable(cpu_bd.get("voigt_profiles")),
                _fmt_nullable(cpu_bd.get("assembly")),
                _fmt_nullable(cpu_bd.get("boltzmann_fit")),
                _fmt_nullable(cpu_bd.get("closure")),
                _fmt_nullable(gpu_bd.get("transfer")),
                _fmt_nullable(gpu_bd.get("forward_model")),
                _fmt_nullable(gpu_bd.get("boltzmann_fit")),
                _fmt_nullable(gpu_bd.get("closure")),
            ]
        )
    write_csv(os.path.join(output_dir, "fig7_e2e_breakdown.csv"), headers, rows)

    # Headline numbers from summary or compute them
    summary = data.get("summary", {})
    return {
        "e2e_crossover_batch_size": summary.get("crossover_batch_size"),
        "e2e_max_speedup": summary.get("max_speedup", 0),
        "e2e_max_speedup_batch_size": summary.get("max_speedup_batch_size"),
    }


def _fmt_nullable(val, digits: int = 4) -> str:
    """Format a nullable numeric value."""
    if val is None:
        return ""
    if isinstance(val, float):
        return str(round(val, digits))
    return str(val)


# ---------------------------------------------------------------------------
# Master summary
# ---------------------------------------------------------------------------
def build_summary(
    headlines: dict,
    all_data: dict[str, dict | None],
    is_mock: bool,
) -> dict:
    """Build the master benchmark_summary.json."""
    # Hardware from first available data
    hw = {}
    for d in all_data.values():
        if d is not None:
            hw = d.get("hardware", {})
            break

    # Paper claims evaluation
    claims: dict = {}

    # Check if any kernel shows GPU speedup > 1
    voigt_data = all_data.get("voigt")
    boltz_data = all_data.get("boltzmann")
    all_show_speedup = True

    # Voigt: check if GPU speedup exists
    if voigt_data:
        max_voigt_su = max(voigt_data["results"]["speedup"])
        claims["voigt_gpu_speedup"] = max_voigt_su > 1.0
        if max_voigt_su <= 1.0:
            all_show_speedup = False
    else:
        all_show_speedup = False

    # Boltzmann: check speedup
    if boltz_data:
        max_boltz_su = max(r["speedup"] for r in boltz_data["results"])
        claims["boltzmann_gpu_speedup"] = max_boltz_su > 1.0
        if max_boltz_su <= 1.0:
            all_show_speedup = False
    else:
        all_show_speedup = False

    # E2E claims
    e2e_data = all_data.get("e2e_pipeline")
    if e2e_data:
        # Check if GPU faster at batch>=100
        batch_100_speedup = None
        for r in e2e_data["results"]:
            if r["batch_size"] >= 100 and r.get("speedup") is not None:
                batch_100_speedup = r["speedup"]
                break
        if batch_100_speedup is not None:
            claims["e2e_batch_100_gpu_faster"] = batch_100_speedup > 1.0
            claims["e2e_batch_100_speedup"] = round(batch_100_speedup, 2)
        else:
            claims["e2e_batch_100_gpu_faster"] = None
            claims["e2e_batch_100_speedup"] = None
    else:
        claims["e2e_batch_100_gpu_faster"] = None

    # Accuracy maintained (from Voigt error)
    if voigt_data:
        max_err = max(voigt_data["results"]["max_relative_error"])
        claims["accuracy_maintained"] = max_err < 1e-6
        claims["max_relative_error"] = f"{max_err:.2e}"
    else:
        claims["accuracy_maintained"] = None

    claims["all_kernels_show_speedup"] = all_show_speedup

    summary = {
        "hardware": hw,
        "is_mock_data": is_mock,
        "headline_numbers": headlines,
        "paper_claims": claims,
    }

    if is_mock:
        summary["WARNING"] = "MOCK DATA -- NOT REAL BENCHMARK RESULTS"

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and produce figure-ready CSV data"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="benchmarks/results/",
        help="Directory containing benchmark JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/figures/",
        help="Directory for figure-ready CSV output",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate and use mock data (for testing the analysis pipeline only)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CF-LIBS Benchmark Results Analysis")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    if args.mock:
        mock = generate_mock_data()
        all_data = {
            "voigt": mock["voigt"],
            "boltzmann": mock["boltzmann"],
            "anderson": mock["anderson"],
            "faiss": mock["faiss"],
            "batch_forward": mock["batch_forward"],
            "e2e_pipeline": mock["e2e_pipeline"],
        }
    else:
        file_map = {
            "voigt": "voigt_results.json",
            "boltzmann": "boltzmann_results.json",
            "anderson": "anderson_results.json",
            "faiss": "faiss_results.json",
            "batch_forward": "batch_forward_results.json",
            "e2e_pipeline": "e2e_pipeline_results.json",
        }
        all_data = {}
        for key, fname in file_map.items():
            all_data[key] = load_json(os.path.join(args.input_dir, fname))

    # Check if we have any data
    available = [k for k, v in all_data.items() if v is not None]
    if not available:
        print("\nERROR: No benchmark data found. Run benchmark scripts first.")
        print(f"  Searched in: {args.input_dir}")
        sys.exit(1)

    print(f"\nData available: {', '.join(available)}")
    missing = [k for k, v in all_data.items() if v is None]
    if missing:
        print(f"Data missing:   {', '.join(missing)}")

    # Extract figure data
    headlines: dict = {}

    print("\nExtracting figure data:")

    if all_data["voigt"]:
        h = extract_fig2_voigt(all_data["voigt"], args.output_dir)
        if h:
            headlines.update(h)

    if all_data["boltzmann"]:
        h = extract_fig3_boltzmann(all_data["boltzmann"], args.output_dir)
        if h:
            headlines.update(h)

    if all_data["anderson"]:
        h = extract_fig4_anderson(all_data["anderson"], args.output_dir)
        if h:
            headlines.update(h)

    if all_data["faiss"]:
        h = extract_fig5_faiss(all_data["faiss"], args.output_dir)
        if h:
            headlines.update(h)

    if all_data["batch_forward"]:
        h = extract_fig6_batch_forward(all_data["batch_forward"], args.output_dir)
        if h:
            headlines.update(h)

    if all_data["e2e_pipeline"]:
        h = extract_fig7_e2e(all_data["e2e_pipeline"], args.output_dir)
        if h:
            headlines.update(h)

    # Build and write master summary
    summary = build_summary(headlines, all_data, is_mock=args.mock)
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Wrote {summary_path}")

    # Print text summary table
    print("\n")
    headers_t = ["Benchmark", "Key Metric", "Value"]
    rows_t = []
    for key, val in headlines.items():
        rows_t.append([key, "", str(val)])
    print_table(headers_t, rows_t, title="Headline Numbers")

    if args.mock:
        print("=" * 60)
        print("WARNING: ALL ABOVE NUMBERS ARE MOCK DATA")
        print("WARNING: DO NOT USE IN PUBLICATIONS")
        print("=" * 60)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
