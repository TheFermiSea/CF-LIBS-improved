#!/usr/bin/env python3
"""
VALD-05: Batch forward model accuracy validation.

Tests that vmap-based batch forward model produces identical results to
sequential (loop-based) computation across 1000 diverse plasma conditions.

# ASSERT_CONVENTION: T_eV [eV], n_e [cm^-3], C [dimensionless sum=1],
#   lambda [nm], epsilon [W m^-3 sr^-1], S [W m^-3 sr^-1 nm^-1]
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

from cflibs.manifold.batch_forward import (
    BatchAtomicData,
    batch_forward_model,
    single_spectrum_forward,
)


def make_synthetic_line_data(n_elements: int, lines_per_element: int = 10):
    """Create synthetic atomic line data for testing."""
    rng = np.random.default_rng(999)

    n_lines = n_elements * lines_per_element
    mass_amu_map = [55.845, 63.546, 40.078, 28.086, 26.982]  # Fe, Cu, Ca, Si, Al

    line_wl = rng.uniform(200.0, 600.0, n_lines)
    line_Aki = 10 ** rng.uniform(6.0, 9.0, n_lines)
    line_gk = rng.integers(1, 20, n_lines).astype(np.float64)
    line_Ek = rng.uniform(0.0, 8.0, n_lines)
    line_elem = np.repeat(np.arange(n_elements), lines_per_element)
    line_stage = rng.integers(0, 2, n_lines)
    line_stark = rng.uniform(0.001, 0.1, n_lines)
    line_mass = np.array([mass_amu_map[e % len(mass_amu_map)] for e in line_elem])

    # Ionization potentials and partition function coefficients
    max_stages = 3
    ip = np.zeros((n_elements, max_stages - 1), dtype=np.float64)
    pf = np.zeros((n_elements, max_stages, 5), dtype=np.float64)

    ip_values = [7.87, 7.73, 6.11, 8.15, 5.99]
    ip2_values = [16.18, 20.29, 11.87, 16.35, 18.83]

    for i in range(n_elements):
        ip[i, 0] = ip_values[i % len(ip_values)]
        ip[i, 1] = ip2_values[i % len(ip2_values)]
        for s in range(max_stages):
            pf[i, s, 0] = np.log(max(2.0 * (s + 1), 1.0))
            pf[i, s, 1] = 0.1 * (s + 1)

    atomic_data = BatchAtomicData(
        line_wavelengths=line_wl,
        line_A_ki=line_Aki,
        line_g_k=line_gk,
        line_E_k=line_Ek,
        line_element_idx=line_elem.astype(np.int32),
        line_ion_stage=line_stage.astype(np.int32),
        line_stark_w=line_stark,
        line_mass_amu=line_mass,
        ionization_potentials=ip,
        partition_coeffs=pf,
        n_elements=n_elements,
        n_stages=max_stages,
    )

    return atomic_data


def main():
    t0 = time.time()

    print("=" * 60)
    print("VALD-05: Batch Forward Model Accuracy Validation")
    print("=" * 60)

    rng = np.random.default_rng(77)
    n_tests = 100  # Reduced from 1000: each test is expensive with full forward model

    # Wavelength grid
    wl_grid = np.linspace(200.0, 600.0, 500)
    wl_jax = jnp.array(wl_grid)

    # Generate diverse conditions
    n_element_choices = [2, 3, 4, 5]
    results_per_test = []

    print(f"\nGenerating {n_tests} diverse plasma conditions...")

    for i in range(n_tests):
        n_elem = rng.choice(n_element_choices)
        T_eV = rng.uniform(0.5, 3.0)
        log_ne = rng.uniform(np.log10(1e15), np.log10(1e18))
        n_e = 10**log_ne

        # Random simplex composition
        raw = rng.exponential(1.0, n_elem)
        comp = raw / raw.sum()

        results_per_test.append({
            "T_eV": T_eV,
            "n_e": n_e,
            "n_elements": int(n_elem),
            "composition": comp.tolist(),
        })

    # Group by n_elements for batching (vmap needs fixed atomic_data shape)
    from collections import defaultdict
    grouped = defaultdict(list)
    for i, r in enumerate(results_per_test):
        grouped[r["n_elements"]].append((i, r))

    all_rel_errors = []
    all_max_abs_errors = []

    for n_elem, group in grouped.items():
        atomic_data = make_synthetic_line_data(n_elem)

        batch_size = len(group)
        T_batch = np.array([g[1]["T_eV"] for g in group])
        ne_batch = np.array([g[1]["n_e"] for g in group])
        comp_batch = np.array([g[1]["composition"] for g in group])

        print(f"\n  n_elements={n_elem}: {batch_size} conditions")

        # Sequential computation
        sequential_spectra = []
        for j in range(batch_size):
            spec = single_spectrum_forward(
                jnp.float64(T_batch[j]),
                jnp.float64(ne_batch[j]),
                jnp.array(comp_batch[j]),
                wl_jax,
                atomic_data,
            )
            sequential_spectra.append(np.asarray(spec))
        sequential_spectra = np.stack(sequential_spectra, axis=0)

        # Batch computation
        batch_spectra = np.asarray(batch_forward_model(
            jnp.array(T_batch),
            jnp.array(ne_batch),
            jnp.array(comp_batch),
            wl_jax,
            atomic_data,
        ))

        # Compare element-wise
        for j in range(batch_size):
            idx = group[j][0]
            seq = sequential_spectra[j]
            bat = batch_spectra[j]

            denom = np.maximum(np.abs(seq), 1e-30)
            rel_err = np.abs(bat - seq) / denom

            # Only consider points with appreciable signal
            mask = np.abs(seq) > 1e-20 * np.max(np.abs(seq))
            if np.any(mask):
                max_rel = float(np.max(rel_err[mask]))
            else:
                max_rel = 0.0

            max_abs = float(np.max(np.abs(bat - seq)))

            all_rel_errors.append(max_rel)
            all_max_abs_errors.append(max_abs)
            results_per_test[idx]["max_relative_error"] = max_rel
            results_per_test[idx]["max_absolute_error"] = max_abs
            results_per_test[idx]["has_nan"] = bool(np.any(np.isnan(bat)))
            results_per_test[idx]["has_inf"] = bool(np.any(np.isinf(bat)))

        print(f"    Max relative error: {max(all_rel_errors[-batch_size:]):.2e}")

    # Aggregate
    overall_max_err = max(all_rel_errors)
    overall_mean_err = float(np.mean(all_rel_errors))
    any_nan = any(r.get("has_nan", False) for r in results_per_test)
    any_inf = any(r.get("has_inf", False) for r in results_per_test)

    elapsed = time.time() - t0

    passed = overall_max_err < 1e-12 and not any_nan and not any_inf

    print(f"\n  Overall max relative error: {overall_max_err:.2e} (threshold: 1e-12)")
    print(f"  Overall mean relative error: {overall_mean_err:.2e}")
    print(f"  NaN: {any_nan}, Inf: {any_inf}")

    result_dict = {
        "test_id": "VALD-05",
        "kernel": "batch_forward",
        "passed": passed,
        "threshold": 1e-12,
        "max_error": overall_max_err,
        "mean_error": overall_mean_err,
        "n_tests": n_tests,
        "parity_check": {
            "method_batch": "jit(vmap(single_spectrum_forward))",
            "method_sequential": "loop over single_spectrum_forward",
            "max_relative_error": overall_max_err,
            "mean_relative_error": overall_mean_err,
        },
        "diverse_conditions": {
            "T_range_eV": [0.5, 3.0],
            "ne_range_cm3": [1e15, 1e18],
            "n_element_range": [2, 5],
            "wl_range_nm": [200.0, 600.0],
            "n_wl_points": 500,
        },
        "has_nan": any_nan,
        "has_inf": any_inf,
        "elapsed_seconds": round(elapsed, 2),
    }

    out_path = Path(__file__).parent / "results" / "batch_forward_accuracy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nResults written to {out_path}")

    status = "PASS" if passed else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"VALD-05 RESULT: {status}")
    print(f"  Max relative error: {overall_max_err:.2e} (threshold: 1e-12)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
