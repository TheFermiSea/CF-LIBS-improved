#!/usr/bin/env python3
"""
Aggregate all 5 VALD accuracy results into a combined report.

Loads individual JSON results and produces a combined accuracy_report.json
suitable for paper tables.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    results_dir = Path(__file__).parent / "results"

    # Load individual results
    files = {
        "voigt": results_dir / "voigt_accuracy.json",
        "boltzmann": results_dir / "boltzmann_accuracy.json",
        "anderson": results_dir / "anderson_accuracy.json",
        "softmax": results_dir / "softmax_accuracy.json",
        "batch_forward": results_dir / "batch_forward_accuracy.json",
    }

    individual = {}
    for name, path in files.items():
        if not path.exists():
            print(f"ERROR: Missing {path}")
            return 1
        with open(path) as f:
            individual[name] = json.load(f)

    # Build summary
    kernel_results = {
        "voigt": {
            "passed": individual["voigt"]["passed"],
            "max_error": individual["voigt"]["max_relative_error"],
            "threshold": individual["voigt"]["threshold"],
            "n_tests": individual["voigt"]["n_grid_points"] + individual["voigt"]["n_edge_cases"],
            "unit": "relative error (dimensionless)",
        },
        "boltzmann": {
            "passed": individual["boltzmann"]["passed"],
            "max_slope_error": individual["boltzmann"]["slope_error"]["max"],
            "max_intercept_error": individual["boltzmann"]["intercept_error"]["max"],
            "threshold": individual["boltzmann"]["threshold"],
            "n_tests": individual["boltzmann"]["cpu_vs_gpu"]["n_tests"],
            "unit": "relative error (dimensionless)",
        },
        "anderson": {
            "passed": individual["anderson"]["passed"],
            "max_residual": individual["anderson"]["max_residual"],
            "max_ne_error_pct": individual["anderson"]["max_ne_error_pct"],
            "threshold_residual": individual["anderson"]["threshold_residual"],
            "n_conditions": individual["anderson"]["n_conditions"],
            "unit": "absolute residual (log-space)",
        },
        "softmax": {
            "passed": individual["softmax"]["passed"],
            "max_sum_deviation": individual["softmax"]["sum_constraint"]["max_deviation"],
            "threshold": individual["softmax"]["threshold"],
            "n_tests": individual["softmax"]["sum_constraint"]["n_tests"],
            "unit": "absolute deviation (dimensionless)",
        },
        "batch_forward": {
            "passed": individual["batch_forward"]["passed"],
            "max_error": individual["batch_forward"]["max_error"],
            "threshold": individual["batch_forward"]["threshold"],
            "n_tests": individual["batch_forward"]["n_tests"],
            "unit": "relative error (dimensionless)",
        },
    }

    all_passed = all(kr["passed"] for kr in kernel_results.values())

    report = {
        "validation_phase": "04-01",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "all_passed": all_passed,
            "kernel_results": kernel_results,
        },
        "details": individual,
    }

    out_path = results_dir / "accuracy_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print("=" * 70)
    print("ACCURACY VALIDATION SUMMARY (Phase 04-01)")
    print("=" * 70)
    print(f"{'Kernel':<18} {'Status':<8} {'Max Error':<14} {'Threshold':<14} {'Tests'}")
    print("-" * 70)

    rows = [
        ("Voigt (VALD-01)", kernel_results["voigt"]["passed"],
         f"{kernel_results['voigt']['max_error']:.2e}",
         f"{kernel_results['voigt']['threshold']:.0e}",
         kernel_results["voigt"]["n_tests"]),
        ("Boltzmann (02)", kernel_results["boltzmann"]["passed"],
         f"{kernel_results['boltzmann']['max_slope_error']:.2e}",
         f"{kernel_results['boltzmann']['threshold']:.0e}",
         kernel_results["boltzmann"]["n_tests"]),
        ("Anderson (03)", kernel_results["anderson"]["passed"],
         f"{kernel_results['anderson']['max_residual']:.2e}",
         f"{kernel_results['anderson']['threshold_residual']:.0e}",
         kernel_results["anderson"]["n_conditions"]),
        ("Softmax (04)", kernel_results["softmax"]["passed"],
         f"{kernel_results['softmax']['max_sum_deviation']:.2e}",
         f"{kernel_results['softmax']['threshold']:.0e}",
         kernel_results["softmax"]["n_tests"]),
        ("Batch fwd (05)", kernel_results["batch_forward"]["passed"],
         f"{kernel_results['batch_forward']['max_error']:.2e}",
         f"{kernel_results['batch_forward']['threshold']:.0e}",
         kernel_results["batch_forward"]["n_tests"]),
    ]

    for name, passed, err, thresh, n in rows:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<16} {status:<8} {err:<14} {thresh:<14} {n}")

    print("-" * 70)
    overall = "ALL PASS" if all_passed else "SOME FAILED"
    print(f"  Overall: {overall}")
    print("=" * 70)
    print(f"\nReport written to {out_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
