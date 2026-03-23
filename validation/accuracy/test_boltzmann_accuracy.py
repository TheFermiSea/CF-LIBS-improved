#!/usr/bin/env python3
"""
VALD-02: Boltzmann WLS fit accuracy validation.

Compares JAX batched_boltzmann_fit (closed-form 5-sum WLS) against
numpy.polyfit CPU reference across 1000 random Boltzmann datasets.

# ASSERT_CONVENTION: x = E_k [eV], y = ln(I*lambda/(g_k*A_ki)) [dimensionless],
#   slope [eV^-1], k_B = 8.617333e-5 eV/K
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

from cflibs.inversion.boltzmann_jax import batched_boltzmann_fit
from cflibs.core.constants import KB_EV


def generate_boltzmann_dataset(rng: np.random.Generator, n_lines: int, T_K: float):
    """Generate a single synthetic Boltzmann plot dataset."""
    slope_true = -1.0 / (KB_EV * T_K)
    intercept_true = rng.uniform(5.0, 25.0)

    E_k = rng.uniform(0.0, 10.0, size=n_lines)
    noise_sigma = 0.1
    y = slope_true * E_k + intercept_true + rng.normal(0, noise_sigma, size=n_lines)
    weights = rng.uniform(0.1, 10.0, size=n_lines)

    return E_k, y, weights, slope_true, intercept_true


def cpu_reference_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted least squares via normal equations (NumPy reference)."""
    # WLS: minimize sum w_i (y_i - a - b*x_i)^2
    S_w = np.sum(w)
    S_wx = np.sum(w * x)
    S_wy = np.sum(w * y)
    S_wxx = np.sum(w * x * x)
    S_wxy = np.sum(w * x * y)

    det = S_w * S_wxx - S_wx**2
    if abs(det) < 1e-30:
        return 0.0, 0.0

    slope = (S_w * S_wxy - S_wx * S_wy) / det
    intercept = (S_wxx * S_wy - S_wx * S_wxy) / det
    return slope, intercept


def main():
    t0 = time.time()

    print("=" * 60)
    print("VALD-02: Boltzmann WLS Fit Accuracy Validation")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_tests = 1000
    n_element_choices = [1, 2, 5, 10, 20]

    # Generate datasets
    all_slopes_ref = []
    all_intercepts_ref = []
    all_slopes_jax = []
    all_intercepts_jax = []
    slope_errors = []
    intercept_errors = []

    # We need to batch -- find max lines for padding
    datasets = []
    for i in range(n_tests):
        T_K = rng.uniform(5000, 30000)
        n_lines = rng.integers(5, 51)
        E_k, y, w, slope_true, intercept_true = generate_boltzmann_dataset(rng, n_lines, T_K)
        datasets.append((E_k, y, w))

    max_lines = max(len(d[0]) for d in datasets)

    # Pad and batch
    x_batch = np.zeros((n_tests, max_lines), dtype=np.float64)
    y_batch = np.zeros((n_tests, max_lines), dtype=np.float64)
    w_batch = np.zeros((n_tests, max_lines), dtype=np.float64)
    mask_batch = np.zeros((n_tests, max_lines), dtype=bool)

    for i, (E_k, y, w) in enumerate(datasets):
        n = len(E_k)
        x_batch[i, :n] = E_k
        y_batch[i, :n] = y
        w_batch[i, :n] = w
        mask_batch[i, :n] = True

    # CPU reference (element-by-element)
    print(f"\n[1/2] Running CPU reference fits ({n_tests} datasets)...")
    cpu_slopes = np.zeros(n_tests)
    cpu_intercepts = np.zeros(n_tests)
    for i in range(n_tests):
        n = int(np.sum(mask_batch[i]))
        s, b = cpu_reference_fit(x_batch[i, :n], y_batch[i, :n], w_batch[i, :n])
        cpu_slopes[i] = s
        cpu_intercepts[i] = b

    # JAX batched fit
    print(f"[2/2] Running JAX batched fit ({n_tests} datasets)...")
    result = batched_boltzmann_fit(
        jnp.array(x_batch),
        jnp.array(y_batch),
        jnp.array(w_batch),
        jnp.array(mask_batch),
    )
    jax_slopes = np.asarray(result.slope)
    jax_intercepts = np.asarray(result.intercept)

    # Compute errors
    slope_denom = np.maximum(np.abs(cpu_slopes), 1e-300)
    intercept_denom = np.maximum(np.abs(cpu_intercepts), 1e-300)

    slope_rel_err = np.abs(jax_slopes - cpu_slopes) / slope_denom
    intercept_rel_err = np.abs(jax_intercepts - cpu_intercepts) / intercept_denom

    max_slope_err = float(np.max(slope_rel_err))
    max_intercept_err = float(np.max(intercept_rel_err))
    mean_slope_err = float(np.mean(slope_rel_err))
    mean_intercept_err = float(np.mean(intercept_rel_err))

    has_nan = bool(np.any(np.isnan(jax_slopes)) or np.any(np.isnan(jax_intercepts)))
    has_inf = bool(np.any(np.isinf(jax_slopes)) or np.any(np.isinf(jax_intercepts)))

    elapsed = time.time() - t0

    passed = max_slope_err < 1e-10 and max_intercept_err < 1e-10 and not has_nan and not has_inf

    print(f"\n  Max slope relative error:     {max_slope_err:.2e} (threshold: 1e-10)")
    print(f"  Mean slope relative error:    {mean_slope_err:.2e}")
    print(f"  Max intercept relative error: {max_intercept_err:.2e} (threshold: 1e-10)")
    print(f"  Mean intercept relative error:{mean_intercept_err:.2e}")
    print(f"  NaN: {has_nan}, Inf: {has_inf}")

    result_dict = {
        "test_id": "VALD-02",
        "kernel": "boltzmann",
        "passed": passed,
        "threshold": 1e-10,
        "slope_error": {
            "max": max_slope_err,
            "mean": mean_slope_err,
            "std": float(np.std(slope_rel_err)),
        },
        "intercept_error": {
            "max": max_intercept_err,
            "mean": mean_intercept_err,
            "std": float(np.std(intercept_rel_err)),
        },
        "cpu_vs_gpu": {
            "method_jax": "closed-form 5-sum WLS (batched_boltzmann_fit)",
            "method_cpu": "closed-form 5-sum WLS (NumPy reference)",
            "n_tests": n_tests,
            "max_lines": int(max_lines),
            "T_range_K": [5000, 30000],
        },
        "has_nan": has_nan,
        "has_inf": has_inf,
        "elapsed_seconds": round(elapsed, 2),
    }

    out_path = Path(__file__).parent / "results" / "boltzmann_accuracy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nResults written to {out_path}")

    status = "PASS" if passed else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"VALD-02 RESULT: {status}")
    print(f"  Max slope error: {max_slope_err:.2e}, Max intercept error: {max_intercept_err:.2e}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
