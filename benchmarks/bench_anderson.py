#!/usr/bin/env python3
"""
BENCH-03: Anderson acceleration convergence benchmark.

Measures iteration counts, residual trajectories, and wall-clock times
for the Anderson-accelerated Saha-Boltzmann solver across memory depths
M=0..10 and 10 plasma conditions spanning the CF-LIBS parameter space.

M=0 is pure Picard iteration (no acceleration).

Output: JSON with hardware metadata, iteration counts, convergence flags,
residual histories, and wall times for all (M, condition) combinations.

Usage:
    python benchmarks/bench_anderson.py --output benchmarks/results/anderson_results.json

References:
    Walker & Ni (2011) SIAM J. Numer. Anal. 49(4), 1715-1735.
    Evans et al. (2018) arXiv:1810.08455 -- convergence rate improvement.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.common import (
    benchmark_function,
    hardware_metadata,
    print_table,
    save_results,
)


def make_synthetic_atomic_data():
    """Create synthetic Fe+Cu atomic data for benchmarking.

    Same approach as tests/test_anderson_solver.py -- no database needed.
    Fe-like: IP_I=7.9 eV, IP_II=16.2 eV; U_I~25, U_II~10, U_III~1
    Cu-like: IP_I=7.7 eV, IP_II=20.3 eV; U_I~4, U_II~2, U_III~1
    """
    import jax.numpy as jnp
    from cflibs.plasma.anderson_solver import AtomicDataJAX

    ip_arr = np.array(
        [
            [7.9, 16.2],  # Fe: neutral->I, I->II
            [7.7, 20.3],  # Cu: neutral->I, I->II
        ],
        dtype=np.float64,
    )

    # Partition function coefficients: constant log(U) only (coeff[0])
    pf_arr = np.zeros((2, 3, 5), dtype=np.float64)
    pf_arr[0, 0, 0] = np.log(25.0)  # Fe I
    pf_arr[0, 1, 0] = np.log(10.0)  # Fe II
    pf_arr[0, 2, 0] = np.log(1.0)  # Fe III
    pf_arr[1, 0, 0] = np.log(4.0)  # Cu I
    pf_arr[1, 1, 0] = np.log(2.0)  # Cu II
    pf_arr[1, 2, 0] = np.log(1.0)  # Cu III

    ns_arr = np.array([3, 3], dtype=np.int32)

    return AtomicDataJAX(
        ionization_potentials=jnp.array(ip_arr),
        partition_coefficients=jnp.array(pf_arr),
        n_stages=jnp.array(ns_arr),
    )


# Test conditions spanning CF-LIBS parameter space
# From Phase 2 (02-02-SUMMARY.md) iteration count table
TEST_CONDITIONS = [
    {"T_eV": 0.6, "n_e_init": 1e15, "label": "low-T/low-ne"},
    {"T_eV": 0.7, "n_e_init": 1e16, "label": "low-T/mid-ne"},
    {"T_eV": 0.8, "n_e_init": 1e16, "label": "mid-T/mid-ne(a)"},
    {"T_eV": 0.9, "n_e_init": 5e16, "label": "mid-T/high-ne"},
    {"T_eV": 1.0, "n_e_init": 1e16, "label": "T=1eV/mid-ne"},
    {"T_eV": 1.0, "n_e_init": 1e17, "label": "T=1eV/high-ne"},
    {"T_eV": 1.1, "n_e_init": 5e15, "label": "high-T/low-ne"},
    {"T_eV": 1.2, "n_e_init": 5e15, "label": "high-T/low-ne(b)"},
    {"T_eV": 1.3, "n_e_init": 2e16, "label": "high-T/mid-ne"},
    {"T_eV": 1.5, "n_e_init": 1e16, "label": "highest-T"},
]


def run_anderson_benchmark(
    M_values: list[int],
    n_warmup: int = 3,
    n_runs: int = 5,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> dict:
    """Run the full Anderson convergence benchmark.

    Parameters
    ----------
    M_values : list of int
        Anderson memory depths to sweep.
    n_warmup : int
        JIT warmup calls for timing.
    n_runs : int
        Timed runs for wall-time measurement.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    dict
        Full benchmark results.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from cflibs.plasma.anderson_solver import anderson_solve

    atomic_data = make_synthetic_atomic_data()
    compositions = jnp.array([0.7, 0.3])  # 70% Fe, 30% Cu

    n_conditions = len(TEST_CONDITIONS)
    n_M = len(M_values)

    # Storage: iteration_counts[cond_idx][M_idx]
    iteration_counts = [[0] * n_M for _ in range(n_conditions)]
    converged_flags = [[False] * n_M for _ in range(n_conditions)]
    final_residuals = [[0.0] * n_M for _ in range(n_conditions)]
    wall_times_ms = [[0.0] * n_M for _ in range(n_conditions)]

    # Residual trajectories for selected M values (M=0, 3, 5)
    trajectory_M_values = [0, 3, 5]
    residual_trajectories: dict[str, dict[str, list[float]]] = {}

    for ci, cond in enumerate(TEST_CONDITIONS):
        print(f"\n--- Condition {ci}: T={cond['T_eV']} eV, n_e_init={cond['n_e_init']:.0e} ---")
        residual_trajectories[f"condition_{ci}"] = {}

        for mi, m in enumerate(M_values):
            # Run once to get iteration count and residual history
            result = anderson_solve(
                T_eV=cond["T_eV"],
                compositions=compositions,
                atomic_data=atomic_data,
                n_e_init=cond["n_e_init"],
                m=m,
                tol=tol,
                max_iter=max_iter,
            )

            iters = int(result.iterations)
            conv = bool(result.converged)
            res = float(result.residual)

            iteration_counts[ci][mi] = iters
            converged_flags[ci][mi] = conv
            final_residuals[ci][mi] = res

            # Wall time measurement (with warmup)
            def _solve():
                return anderson_solve(
                    T_eV=cond["T_eV"],
                    compositions=compositions,
                    atomic_data=atomic_data,
                    n_e_init=cond["n_e_init"],
                    m=m,
                    tol=tol,
                    max_iter=max_iter,
                )

            timing = benchmark_function(
                _solve,
                n_warmup=n_warmup,
                n_runs=n_runs,
            )
            wall_times_ms[ci][mi] = timing["mean_s"] * 1e3

            # Store residual trajectory for selected M values
            if m in trajectory_M_values:
                hist = np.array(result.residual_history)
                # Trim to actual iterations (rest is zero-padded)
                hist_trimmed = hist[:iters].tolist()
                residual_trajectories[f"condition_{ci}"][f"M_{m}"] = hist_trimmed

            if m in [0, 3, 5, 10]:
                print(
                    f"  M={m:2d}: {iters:3d} iters, "
                    f"res={res:.2e}, conv={conv}, "
                    f"time={wall_times_ms[ci][mi]:.2f} ms"
                )

    # Summary table: Picard vs Anderson(M=3) vs Anderson(M=5)
    headers = [
        "Condition",
        "T [eV]",
        "n_e_init",
        "Picard",
        "AA(M=3)",
        "AA(M=5)",
        "Speedup(3)",
    ]
    rows = []
    for ci, cond in enumerate(TEST_CONDITIONS):
        picard_iters = iteration_counts[ci][M_values.index(0)] if 0 in M_values else "N/A"
        aa3_iters = iteration_counts[ci][M_values.index(3)] if 3 in M_values else "N/A"
        aa5_iters = iteration_counts[ci][M_values.index(5)] if 5 in M_values else "N/A"

        if isinstance(picard_iters, int) and isinstance(aa3_iters, int) and aa3_iters > 0:
            speedup = f"{picard_iters / aa3_iters:.1f}x"
        else:
            speedup = "N/A"

        rows.append(
            [
                cond["label"],
                str(cond["T_eV"]),
                f"{cond['n_e_init']:.0e}",
                str(picard_iters),
                str(aa3_iters),
                str(aa5_iters),
                speedup,
            ]
        )

    print_table(headers, rows, title="Anderson Convergence Benchmark (BENCH-03)")

    results = {
        "benchmark": "anderson_convergence",
        "hardware": hardware_metadata(),
        "parameters": {
            "M_values": M_values,
            "test_conditions": TEST_CONDITIONS,
            "tol": tol,
            "max_iter": max_iter,
            "n_warmup": n_warmup,
            "n_runs": n_runs,
            "compositions": [0.7, 0.3],
            "elements": ["Fe-like", "Cu-like"],
        },
        "results": {
            "iteration_counts": iteration_counts,
            "converged": converged_flags,
            "final_residuals": final_residuals,
            "wall_times_ms": wall_times_ms,
            "residual_trajectories": residual_trajectories,
        },
        "references": {
            "method_ref": "Walker & Ni (2011) SIAM J. Numer. Anal. 49(4), 1715-1735",
            "convergence_ref": "Evans et al. (2018) arXiv:1810.08455",
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="BENCH-03: Anderson convergence benchmark")
    parser.add_argument(
        "--output",
        default="benchmarks/results/anderson_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of timed runs for wall-time measurement (default: 5)",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use reduced M sweep for quick testing",
    )
    args = parser.parse_args()

    if args.small:
        M_values = [0, 1, 3, 5]
    else:
        M_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    results = run_anderson_benchmark(
        M_values=M_values,
        n_runs=args.n_runs,
    )

    save_results(results, args.output)


if __name__ == "__main__":
    main()
