#!/usr/bin/env python3
"""
VALD-01: Voigt profile accuracy validation.

Systematic sweep of the Weideman N=36 Faddeeva approximation against
scipy.special.wofz across the (x, y) parameter space.

Reference: Zaghloul (2024) arXiv:2411.00917 -- accuracy benchmark.
           Weideman (1994) SIAM J. Numer. Anal. 31, 1497.

# ASSERT_CONVENTION: gamma = HWHM [nm], sigma = std dev [nm], V(lambda) = [nm^-1]
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

from scipy.special import wofz as scipy_wofz

# Import the JAX Voigt kernel internals for direct (x, y) testing
from cflibs.radiation.profiles import _faddeeva_weideman_complex_jax, voigt_spectrum_jax


def faddeeva_scipy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reference Faddeeva w(z) via scipy (uses algorithm 680 / Poppe & Wijers)."""
    z = x + 1j * y
    return scipy_wofz(z).real


def faddeeva_jax(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """JAX Weideman N=36 Faddeeva Re[w(z)]."""
    z = jnp.asarray(x + 1j * y, dtype=jnp.complex128)
    w = _faddeeva_weideman_complex_jax(z)
    return np.asarray(jnp.real(w))


def parameter_sweep(
    x_points: np.ndarray, y_points: np.ndarray
) -> dict:
    """Sweep (x, y) grid, compute relative errors."""
    xx, yy = np.meshgrid(x_points, y_points, indexing="ij")
    x_flat = xx.ravel()
    y_flat = yy.ravel()

    ref = faddeeva_scipy(x_flat, y_flat)
    jax_val = faddeeva_jax(x_flat, y_flat)

    # Relative error (avoid division by zero)
    denom = np.maximum(np.abs(ref), 1e-300)
    rel_err = np.abs(jax_val - ref) / denom

    # Check for NaN/Inf
    has_nan = bool(np.any(np.isnan(jax_val)))
    has_inf = bool(np.any(np.isinf(jax_val)))

    return {
        "x_grid": x_flat.tolist(),
        "y_grid": y_flat.tolist(),
        "ref_values": ref.tolist(),
        "jax_values": jax_val.tolist(),
        "relative_errors": rel_err.tolist(),
        "max_relative_error": float(np.max(rel_err)),
        "mean_relative_error": float(np.mean(rel_err)),
        "std_relative_error": float(np.std(rel_err)),
        "n_points": len(x_flat),
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def edge_case_tests() -> list[dict]:
    """Test specific edge cases in the (x, y) parameter space."""
    cases = [
        ("pure_lorentzian", 0.0, 1.0),
        ("pure_gaussian_limit", 0.0, 1e-8),
        ("origin", 0.0, 0.0 + 1e-10),  # y must be > 0 for physical Voigt
        ("far_wing", 1000.0, 1.0),
        ("narrow_lorentzian", 5.0, 1e-6),
        ("very_narrow_lorentzian", 5.0, 1e-10),  # outside stated validity range
        ("large_both", 50.0, 50.0),
        ("moderate", 3.0, 3.0),
        ("asymmetric_large_x", 100.0, 0.1),
        ("asymmetric_large_y", 0.1, 100.0),
    ]

    results = []
    for name, x, y in cases:
        ref = faddeeva_scipy(np.array([x]), np.array([y]))[0]
        jax_v = faddeeva_jax(np.array([x]), np.array([y]))[0]
        denom = max(abs(ref), 1e-300)
        rel_err = abs(jax_v - ref) / denom

        results.append({
            "name": name,
            "x": x,
            "y": y,
            "ref_value": float(ref),
            "jax_value": float(jax_v),
            "relative_error": float(rel_err),
            "has_nan": bool(np.isnan(jax_v)),
            "has_inf": bool(np.isinf(jax_v)),
        })

    return results


def voigt_spectrum_parity_test() -> dict:
    """Test voigt_spectrum_jax against scipy-based voigt for a realistic spectrum."""
    # Realistic LIBS parameters
    wl_grid = np.linspace(250.0, 260.0, 500)
    line_centers = np.array([251.6, 252.3, 253.8, 255.1, 257.6, 259.3])
    line_intensities = np.array([1e5, 5e4, 8e4, 3e4, 6e4, 2e4])
    sigmas = np.array([0.01, 0.012, 0.011, 0.009, 0.013, 0.01])
    gammas = np.array([0.02, 0.015, 0.025, 0.018, 0.022, 0.012])

    # JAX batch spectrum
    spec_jax = np.asarray(voigt_spectrum_jax(
        jnp.array(wl_grid),
        jnp.array(line_centers),
        jnp.array(line_intensities),
        jnp.array(sigmas),
        jnp.array(gammas),
    ))

    # scipy reference: line-by-line
    spec_ref = np.zeros_like(wl_grid)
    for wl0, inten, sig, gam in zip(line_centers, line_intensities, sigmas, gammas):
        z = (wl_grid - wl0 + 1j * gam) / (sig * np.sqrt(2))
        w = scipy_wofz(z)
        spec_ref += inten * w.real / (sig * np.sqrt(2 * np.pi))

    denom = np.maximum(np.abs(spec_ref), 1e-300)
    rel_err = np.abs(spec_jax - spec_ref) / denom

    # Only consider points where signal is appreciable
    mask = spec_ref > 1e-10 * np.max(spec_ref)
    rel_err_signal = rel_err[mask] if np.any(mask) else rel_err

    return {
        "n_wavelengths": len(wl_grid),
        "n_lines": len(line_centers),
        "max_relative_error": float(np.max(rel_err_signal)),
        "mean_relative_error": float(np.mean(rel_err_signal)),
        "max_absolute_error": float(np.max(np.abs(spec_jax - spec_ref))),
        "has_nan": bool(np.any(np.isnan(spec_jax))),
        "has_inf": bool(np.any(np.isinf(spec_jax))),
    }


def main():
    t0 = time.time()

    print("=" * 60)
    print("VALD-01: Voigt Profile Accuracy Validation")
    print("=" * 60)

    # --- Grid sweep: 20x20 = 400 points ---
    print("\n[1/3] Parameter space sweep (20x20 grid)...")
    x_points = np.linspace(0, 50, 20)
    y_points = np.logspace(-6, np.log10(50), 20)
    sweep = parameter_sweep(x_points, y_points)
    print(f"  Grid points: {sweep['n_points']}")
    print(f"  Max relative error: {sweep['max_relative_error']:.2e}")
    print(f"  Mean relative error: {sweep['mean_relative_error']:.2e}")
    print(f"  NaN: {sweep['has_nan']}, Inf: {sweep['has_inf']}")

    # --- Edge cases ---
    print("\n[2/3] Edge case tests...")
    edges = edge_case_tests()
    # Only count edge cases within validity range for pass/fail
    valid_edges = [e for e in edges if "very_narrow" not in e["name"]]
    max_edge_err = max(e["relative_error"] for e in valid_edges) if valid_edges else 0.0
    any_nan = any(e["has_nan"] for e in valid_edges)
    any_inf = any(e["has_inf"] for e in valid_edges)
    for e in edges:
        # Cases outside stated validity range are informational only
        outside_validity = "very_narrow" in e["name"]
        if outside_validity:
            status = "INFO" if e["relative_error"] >= 1e-6 else "PASS"
        else:
            status = "PASS" if e["relative_error"] < 1e-6 and not e["has_nan"] and not e["has_inf"] else "FAIL"
        print(f"  {e['name']:30s} rel_err={e['relative_error']:.2e} [{status}]")

    # --- Full spectrum parity ---
    print("\n[3/3] Full spectrum parity test...")
    spec_test = voigt_spectrum_parity_test()
    print(f"  Max relative error (signal region): {spec_test['max_relative_error']:.2e}")
    print(f"  NaN: {spec_test['has_nan']}, Inf: {spec_test['has_inf']}")

    elapsed = time.time() - t0

    # --- Aggregate results ---
    overall_max_err = max(sweep["max_relative_error"], max_edge_err, spec_test["max_relative_error"])
    overall_nan = sweep["has_nan"] or any_nan or spec_test["has_nan"]
    overall_inf = sweep["has_inf"] or any_inf or spec_test["has_inf"]
    passed = overall_max_err < 1e-6 and not overall_nan and not overall_inf

    result = {
        "test_id": "VALD-01",
        "kernel": "voigt",
        "passed": passed,
        "threshold": 1e-6,
        "max_relative_error": overall_max_err,
        "mean_relative_error": sweep["mean_relative_error"],
        "n_grid_points": sweep["n_points"],
        "n_edge_cases": len(edges),
        "zaghloul_comparison": "scipy.special.wofz used as reference (Poppe-Wijers algorithm 680)",
        "parameter_sweep": {
            "max_error": sweep["max_relative_error"],
            "mean_error": sweep["mean_relative_error"],
            "std_error": sweep["std_relative_error"],
            "n_tests": sweep["n_points"],
        },
        "edge_cases": edges,
        "spectrum_parity": spec_test,
        "has_nan": overall_nan,
        "has_inf": overall_inf,
        "elapsed_seconds": round(elapsed, 2),
    }

    # Write JSON
    out_path = Path(__file__).parent / "results" / "voigt_accuracy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Summary
    status = "PASS" if passed else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"VALD-01 RESULT: {status}")
    print(f"  Max relative error: {overall_max_err:.2e} (threshold: 1e-6)")
    print(f"  NaN/Inf: {overall_nan or overall_inf}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
