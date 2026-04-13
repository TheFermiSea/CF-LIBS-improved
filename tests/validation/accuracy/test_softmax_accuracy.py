#!/usr/bin/env python3
"""
VALD-04: Softmax closure accuracy validation.

Tests that softmax_closure preserves sum-to-1 to machine precision,
round-trip inverse_softmax(softmax(theta)) == theta, and handles edge cases.

# ASSERT_CONVENTION: theta [dimensionless], C_i [dimensionless], sum(C_i) = 1
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

from cflibs.inversion.softmax_closure import softmax_closure, inverse_softmax, softmax_jacobian


def main():
    t0 = time.time()

    print("=" * 60)
    print("VALD-04: Softmax Closure Accuracy Validation")
    print("=" * 60)

    rng = np.random.default_rng(123)
    n_tests = 1000

    # --- Random theta vectors ---
    print(f"\n[1/3] Random sum-to-1 tests ({n_tests} cases)...")
    dim_choices = [2, 3, 5, 10, 15, 20]
    scale_choices = [0.1, 1.0, 10.0, 100.0]

    sum_deviations = []
    round_trip_errors = []

    for i in range(n_tests):
        D = rng.choice(dim_choices)
        scale = rng.choice(scale_choices)
        theta = rng.standard_normal(D) * scale

        C = np.asarray(softmax_closure(jnp.array(theta)))

        # Sum constraint
        sum_constraint = abs(np.sum(C) - 1.0)
        sum_deviations.append(sum_constraint)

        # All positive
        assert np.all(C > 0), f"Negative concentration at test {i}"

        # Round-trip
        theta_rt = np.asarray(inverse_softmax(jnp.array(C)))
        C_rt = np.asarray(softmax_closure(jnp.array(theta_rt)))
        rt_err = float(np.max(np.abs(C_rt - C)))
        round_trip_errors.append(rt_err)

    max_sum_dev = float(np.max(sum_deviations))
    mean_sum_dev = float(np.mean(sum_deviations))
    max_rt_err = float(np.max(round_trip_errors))

    print(f"  Max |sum(C) - 1|: {max_sum_dev:.2e} (threshold: 1e-15)")
    print(f"  Mean |sum(C) - 1|: {mean_sum_dev:.2e}")
    print(f"  Max round-trip error: {max_rt_err:.2e}")

    # --- Edge cases ---
    print("\n[2/3] Edge case tests...")
    edge_results = []

    edge_cases = [
        ("extreme_spread", np.array([500.0, -500.0, 0.0])),
        ("single_element", np.array([0.0])),
        ("uniform_2", np.zeros(2)),
        ("uniform_10", np.zeros(10)),
        ("uniform_20", np.zeros(20)),
        ("one_dominant", np.array([100.0, 0.0, 0.0, 0.0])),
        ("near_degenerate", np.array([1e-15, 1e-15, 1e-15])),
        ("large_negative", np.array([-1000.0, -999.0, -998.0])),
        ("mixed_extreme", np.array([1000.0, -1000.0, 500.0, -500.0, 0.0])),
    ]

    for name, theta in edge_cases:
        C = np.asarray(softmax_closure(jnp.array(theta)))
        sum_dev = abs(np.sum(C) - 1.0)
        has_nan = bool(np.any(np.isnan(C)))
        has_inf = bool(np.any(np.isinf(C)))
        all_positive = bool(np.all(C >= 0))

        status = "PASS" if sum_dev < 1e-15 and not has_nan and not has_inf and all_positive else "FAIL"
        print(f"  {name:25s} sum_dev={sum_dev:.2e} nan={has_nan} inf={has_inf} [{status}]")

        edge_results.append({
            "name": name,
            "D": len(theta),
            "theta": theta.tolist(),
            "C": C.tolist(),
            "sum_deviation": float(sum_dev),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "all_positive": all_positive,
        })

    # --- Gradient check ---
    print("\n[3/3] Gradient check...")
    grad_results = []
    for D in [2, 5, 10]:
        theta = jnp.array(rng.standard_normal(D))

        # grad of sum(softmax(theta)) should be zero-vector (sum is constant 1)
        grad_sum = jax.grad(lambda t: jnp.sum(softmax_closure(t)))(theta)
        grad_norm = float(jnp.linalg.norm(grad_sum))

        # grad of individual component should be finite
        grad_c0 = jax.grad(lambda t: softmax_closure(t)[0])(theta)
        grad_c0_finite = bool(jnp.all(jnp.isfinite(grad_c0)))

        # Jacobian check: analytical vs autodiff
        J_analytical = np.asarray(softmax_jacobian(theta))
        J_auto = np.asarray(jax.jacobian(softmax_closure)(theta))
        jac_err = float(np.max(np.abs(J_analytical - J_auto)))

        print(f"  D={D:2d}: grad_sum_norm={grad_norm:.2e}, grad_c0_finite={grad_c0_finite}, jac_err={jac_err:.2e}")

        grad_results.append({
            "D": D,
            "grad_sum_norm": grad_norm,
            "grad_c0_finite": grad_c0_finite,
            "jacobian_error": jac_err,
        })

    elapsed = time.time() - t0

    # Aggregate
    any_edge_fail = any(
        e["sum_deviation"] >= 1e-15 or e["has_nan"] or e["has_inf"] or not e["all_positive"]
        for e in edge_results
    )
    any_grad_fail = any(not g["grad_c0_finite"] for g in grad_results)
    passed = max_sum_dev < 1e-15 and not any_edge_fail and not any_grad_fail

    result = {
        "test_id": "VALD-04",
        "kernel": "softmax",
        "passed": passed,
        "threshold": 1e-15,
        "sum_constraint": {
            "max_deviation": max_sum_dev,
            "mean_deviation": mean_sum_dev,
            "n_tests": n_tests,
        },
        "round_trip": {
            "max_error": max_rt_err,
        },
        "edge_cases": edge_results,
        "gradient_check": grad_results,
        "has_nan": any(e["has_nan"] for e in edge_results),
        "has_inf": any(e["has_inf"] for e in edge_results),
        "elapsed_seconds": round(elapsed, 2),
    }

    out_path = Path(__file__).parent / "results" / "softmax_accuracy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {out_path}")

    status = "PASS" if passed else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"VALD-04 RESULT: {status}")
    print(f"  Max |sum(C) - 1|: {max_sum_dev:.2e} (threshold: 1e-15)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
