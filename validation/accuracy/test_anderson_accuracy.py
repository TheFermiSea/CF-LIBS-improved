#!/usr/bin/env python3
"""
VALD-03: Anderson-accelerated Saha solver accuracy validation.

Compares Anderson-accelerated solver (m=1,3,5) against Picard iteration (m=0)
as ground truth across 20 plasma conditions spanning T=0.5-3 eV, n_e=1e15-1e18.

Reference: Evans et al. (2018) arXiv:1810.08455 -- Anderson acceleration convergence.

# ASSERT_CONVENTION: n_e [cm^-3], T_eV [eV], C_i dimensionless sum-to-1,
#   SAHA_CONST_CM3 from cflibs.core.constants
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Force CPU + float64
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp

from cflibs.plasma.anderson_solver import (
    anderson_solve,
    picard_solve,
    AtomicDataJAX,
)


def make_synthetic_atomic_data(n_elements: int, max_stages: int = 3) -> AtomicDataJAX:
    """Create synthetic atomic data for testing.

    Uses physically plausible but simplified partition functions and IPs.
    """
    # Ionization potentials [eV] -- realistic range
    ip_values = [
        [7.87, 16.18],   # Fe-like
        [7.73, 20.29],   # Cu-like
        [6.11, 11.87],   # Ca-like
        [8.15, 16.35],   # Si-like
    ]

    # Partition function coefficients (log(U) = a0 + a1*ln(T) + ...)
    # Simplified: constant partition functions at reasonable values
    pf_defaults = [
        [[np.log(25.0), 0.3, 0.0, 0.0, 0.0],   # neutral (Fe I ~ g0=25)
         [np.log(30.0), 0.35, 0.0, 0.0, 0.0],   # singly ionized
         [np.log(25.0), 0.25, 0.0, 0.0, 0.0]],   # doubly ionized
        [[np.log(2.0), 0.1, 0.0, 0.0, 0.0],
         [np.log(1.0), 0.05, 0.0, 0.0, 0.0],
         [np.log(6.0), 0.15, 0.0, 0.0, 0.0]],
        [[np.log(1.0), 0.0, 0.0, 0.0, 0.0],
         [np.log(2.0), 0.1, 0.0, 0.0, 0.0],
         [np.log(1.0), 0.0, 0.0, 0.0, 0.0]],
        [[np.log(9.0), 0.2, 0.0, 0.0, 0.0],
         [np.log(6.0), 0.15, 0.0, 0.0, 0.0],
         [np.log(1.0), 0.0, 0.0, 0.0, 0.0]],
    ]

    n_elem = min(n_elements, len(ip_values))
    ip_arr = np.zeros((n_elem, max_stages - 1), dtype=np.float64)
    pf_arr = np.zeros((n_elem, max_stages, 5), dtype=np.float64)
    ns_arr = np.full(n_elem, max_stages, dtype=np.int32)

    for i in range(n_elem):
        for j in range(max_stages - 1):
            ip_arr[i, j] = ip_values[i][j]
        for s in range(max_stages):
            pf_arr[i, s, :] = pf_defaults[i][s]

    return AtomicDataJAX(
        ionization_potentials=jnp.array(ip_arr),
        partition_coefficients=jnp.array(pf_arr),
        n_stages=jnp.array(ns_arr),
    )


def main():
    t0 = time.time()

    print("=" * 60)
    print("VALD-03: Anderson Solver Accuracy Validation")
    print("=" * 60)

    # Define 20 plasma conditions
    T_eV_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
    ne_init_values = [1e15, 1e16, 1e17, 1e18]

    # Build 20 conditions (10 T values x 2 ne values)
    conditions = []
    for i, T in enumerate(T_eV_values):
        ne = ne_init_values[i % len(ne_init_values)]
        conditions.append((T, ne))
    # Fill remaining
    for i in range(len(conditions), 20):
        T = T_eV_values[i % len(T_eV_values)]
        ne = ne_init_values[(i // len(T_eV_values)) % len(ne_init_values)]
        conditions.append((T, ne))
    conditions = conditions[:20]

    # Three composition sets
    compositions_sets = {
        "Fe_only": np.array([1.0]),
        "Fe_Cu": np.array([0.7, 0.3]),
        "Fe_Cu_Ca_Si": np.array([0.5, 0.2, 0.15, 0.15]),
    }

    anderson_depths = [1, 3, 5]
    picard_tol = 1e-14
    picard_max_iter = 200

    results = []

    print(f"\nRunning {len(conditions)} plasma conditions x {len(compositions_sets)} compositions...")

    for comp_name, comp in compositions_sets.items():
        n_elem = len(comp)
        atomic_data = make_synthetic_atomic_data(n_elem)

        for idx, (T_eV, ne_init) in enumerate(conditions):
            # Picard ground truth
            picard_result = picard_solve(
                T_eV=T_eV,
                compositions=jnp.array(comp),
                atomic_data=atomic_data,
                n_e_init=ne_init,
                tol=picard_tol,
                max_iter=picard_max_iter,
            )
            ne_picard = float(picard_result.n_e)
            picard_iters = int(picard_result.iterations)
            picard_residual = float(picard_result.residual)
            picard_converged = bool(picard_result.converged)

            for m in anderson_depths:
                anderson_result = anderson_solve(
                    T_eV=T_eV,
                    compositions=jnp.array(comp),
                    atomic_data=atomic_data,
                    n_e_init=ne_init,
                    m=m,
                    tol=picard_tol,
                    max_iter=picard_max_iter,
                )
                ne_anderson = float(anderson_result.n_e)
                anderson_iters = int(anderson_result.iterations)
                anderson_residual = float(anderson_result.residual)
                anderson_converged = bool(anderson_result.converged)

                # Fixed-point agreement
                ne_rel_err = abs(ne_anderson - ne_picard) / max(abs(ne_picard), 1e-300)
                ne_pct_err = ne_rel_err * 100.0

                results.append({
                    "condition_idx": idx,
                    "T_eV": T_eV,
                    "ne_init": ne_init,
                    "composition": comp_name,
                    "anderson_m": m,
                    "ne_picard": ne_picard,
                    "ne_anderson": ne_anderson,
                    "ne_relative_error": ne_rel_err,
                    "ne_pct_error": ne_pct_err,
                    "picard_residual": picard_residual,
                    "anderson_residual": anderson_residual,
                    "picard_iterations": picard_iters,
                    "anderson_iterations": anderson_iters,
                    "picard_converged": picard_converged,
                    "anderson_converged": anderson_converged,
                    "residual_comparison": anderson_residual,
                    "fixed_point_agreement": ne_rel_err,
                })

    # Aggregate
    all_residuals = [r["anderson_residual"] for r in results]
    all_ne_errors = [r["ne_relative_error"] for r in results]
    all_ne_pct = [r["ne_pct_error"] for r in results]
    all_converged = [r["anderson_converged"] for r in results]

    max_residual = max(all_residuals)
    max_ne_error = max(all_ne_errors)
    max_ne_pct = max(all_ne_pct)
    all_conv = all(all_converged)

    elapsed = time.time() - t0

    passed = max_residual < 1e-12 and max_ne_pct < 0.01 and all_conv

    print(f"\n  Max Anderson residual:     {max_residual:.2e} (threshold: 1e-12)")
    print(f"  Max n_e relative error:   {max_ne_error:.2e}")
    print(f"  Max n_e percent error:    {max_ne_pct:.4f}% (threshold: 0.01%)")
    print(f"  All converged:            {all_conv}")

    # Iteration comparison summary
    print("\n  Iteration counts (mean) by Anderson depth:")
    for m in anderson_depths:
        m_results = [r for r in results if r["anderson_m"] == m]
        mean_iters = np.mean([r["anderson_iterations"] for r in m_results])
        picard_mean = np.mean([r["picard_iterations"] for r in m_results])
        speedup = picard_mean / max(mean_iters, 1)
        print(f"    m={m}: {mean_iters:.1f} iters (Picard: {picard_mean:.1f}, speedup: {speedup:.1f}x)")

    result_dict = {
        "test_id": "VALD-03",
        "kernel": "anderson",
        "passed": passed,
        "threshold_residual": 1e-12,
        "threshold_ne_pct": 0.01,
        "max_residual": max_residual,
        "max_ne_error_pct": max_ne_pct,
        "n_conditions": len(conditions),
        "n_compositions": len(compositions_sets),
        "n_total_tests": len(results),
        "all_converged": all_conv,
        "residual_comparison": {
            "max": max_residual,
            "mean": float(np.mean(all_residuals)),
        },
        "fixed_point_agreement": {
            "max_relative_error": max_ne_error,
            "max_pct_error": max_ne_pct,
            "mean_pct_error": float(np.mean(all_ne_pct)),
        },
        "details": results,
        "elapsed_seconds": round(elapsed, 2),
        "reference": "Evans et al. (2018) arXiv:1810.08455",
    }

    out_path = Path(__file__).parent / "results" / "anderson_accuracy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nResults written to {out_path}")

    status = "PASS" if passed else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"VALD-03 RESULT: {status}")
    print(f"  Max residual: {max_residual:.2e}, Max n_e error: {max_ne_pct:.4f}%")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
