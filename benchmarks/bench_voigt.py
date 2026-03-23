#!/usr/bin/env python3
"""
BENCH-01: Voigt profile throughput benchmark.

Measures wall-clock throughput (evaluations/sec) of the JAX Voigt kernel
(voigt_spectrum_jax) vs the SciPy CPU baseline (scipy.special.wofz) across
wavelength grid sizes from 100 to 100,000 points with 10 spectral lines.

Also measures accuracy: max |V_jax - V_scipy| / |V_scipy|.

Output: JSON with hardware metadata, timing arrays, throughput, speedup,
and relative error per grid size.

Usage:
    python benchmarks/bench_voigt.py --output benchmarks/results/voigt_results.json
    python benchmarks/bench_voigt.py --cpu-only --output /tmp/voigt_test.json

References:
    Weideman (1994) SIAM J. Numer. Anal. 31, 1497-1518.
    Zaghloul (2024) arXiv:2411.00917 -- accuracy reference for Voigt.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Ensure cflibs is importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.common import (
    benchmark_function,
    hardware_metadata,
    print_table,
    save_results,
)


def scipy_voigt_spectrum(
    wl_grid: np.ndarray,
    centers: np.ndarray,
    intensities: np.ndarray,
    sigmas: np.ndarray,
    gammas: np.ndarray,
) -> np.ndarray:
    """CPU baseline: compute Voigt spectrum using scipy.special.wofz.

    Uses the same outer-product approach as voigt_spectrum_jax for fair
    comparison, but with NumPy arrays and scipy Faddeeva function.
    """
    from scipy.special import wofz

    # (N_wl, N_lines)
    diff = wl_grid[:, None] - centers[None, :]
    sigma_2d = sigmas[None, :]
    gamma_2d = gammas[None, :]

    z = (diff + 1j * gamma_2d) / (sigma_2d * np.sqrt(2.0))
    w = wofz(z)
    profiles = w.real / (sigma_2d * np.sqrt(2.0 * np.pi))

    spectrum = np.sum(intensities[None, :] * profiles, axis=1)
    return spectrum


def run_voigt_benchmark(
    grid_sizes: list[int],
    n_lines: int = 10,
    n_warmup: int = 3,
    n_runs: int = 10,
    cpu_only: bool = False,
) -> dict:
    """Run the full Voigt throughput benchmark.

    Parameters
    ----------
    grid_sizes : list of int
        Number of wavelength grid points to sweep.
    n_lines : int
        Number of spectral lines.
    n_warmup : int
        JIT warmup calls (excluded from timing).
    n_runs : int
        Number of timed calls per configuration.
    cpu_only : bool
        If True, skip JAX GPU benchmarks.

    Returns
    -------
    dict
        Full benchmark results with hardware, parameters, results.
    """
    # Generate test line parameters
    rng = np.random.default_rng(42)
    centers = np.linspace(250.0, 550.0, n_lines)  # nm
    intensities = rng.uniform(0.5, 2.0, n_lines)
    sigmas_val = np.full(n_lines, 0.01)  # nm (Doppler)
    gammas_val = np.full(n_lines, 0.02)  # nm (Stark HWHM)

    # Check JAX availability
    has_jax = False
    if not cpu_only:
        try:
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            from cflibs.radiation.profiles import voigt_spectrum_jax

            has_jax = True
        except ImportError:
            print("JAX not available, running CPU-only benchmarks.")

    # Storage
    cpu_throughput = []
    gpu_throughput = []
    cpu_time_mean = []
    cpu_time_std = []
    gpu_time_mean = []
    gpu_time_std = []
    max_rel_errors = []
    speedups = []

    headers = ["Grid Size", "CPU (ms)", "GPU (ms)", "Speedup", "Max Rel Err"]
    rows = []

    for n_wl in grid_sizes:
        print(f"\n--- Grid size: {n_wl} ---")
        wl = np.linspace(200.0, 600.0, n_wl)

        # --- CPU baseline (scipy) ---
        cpu_result = benchmark_function(
            scipy_voigt_spectrum,
            args=(wl, centers, intensities, sigmas_val, gammas_val),
            n_warmup=1,
            n_runs=n_runs,
        )
        cpu_spec = scipy_voigt_spectrum(wl, centers, intensities, sigmas_val, gammas_val)

        cpu_tp = n_wl * n_lines / cpu_result["mean_s"]
        cpu_throughput.append(cpu_tp)
        cpu_time_mean.append(cpu_result["mean_s"])
        cpu_time_std.append(cpu_result["std_s"])

        print(f"  CPU: {cpu_result['mean_s']*1e3:.3f} +/- {cpu_result['std_s']*1e3:.3f} ms")

        # --- GPU (JAX) ---
        if has_jax:
            import jax.numpy as jnp
            from cflibs.radiation.profiles import voigt_spectrum_jax

            wl_jax = jnp.array(wl)
            centers_jax = jnp.array(centers)
            intensities_jax = jnp.array(intensities)
            sigmas_jax = jnp.array(sigmas_val)
            gammas_jax = jnp.array(gammas_val)

            gpu_result = benchmark_function(
                voigt_spectrum_jax,
                args=(wl_jax, centers_jax, intensities_jax, sigmas_jax, gammas_jax),
                n_warmup=n_warmup,
                n_runs=n_runs,
            )

            gpu_tp = n_wl * n_lines / gpu_result["mean_s"]
            gpu_throughput.append(gpu_tp)
            gpu_time_mean.append(gpu_result["mean_s"])
            gpu_time_std.append(gpu_result["std_s"])

            # Accuracy: compare JAX output to scipy reference
            jax_spec = np.array(
                voigt_spectrum_jax(wl_jax, centers_jax, intensities_jax, sigmas_jax, gammas_jax)
            )
            nonzero = np.abs(cpu_spec) > 1e-30
            if np.any(nonzero):
                rel_err = float(
                    np.max(
                        np.abs(jax_spec[nonzero] - cpu_spec[nonzero]) / np.abs(cpu_spec[nonzero])
                    )
                )
            else:
                rel_err = 0.0
            max_rel_errors.append(rel_err)

            sp = cpu_result["mean_s"] / gpu_result["mean_s"]
            speedups.append(sp)

            print(f"  GPU: {gpu_result['mean_s']*1e3:.3f} +/- {gpu_result['std_s']*1e3:.3f} ms")
            print(f"  Speedup: {sp:.2f}x")
            print(f"  Max relative error: {rel_err:.2e}")

            rows.append(
                [
                    str(n_wl),
                    f"{cpu_result['mean_s']*1e3:.3f}",
                    f"{gpu_result['mean_s']*1e3:.3f}",
                    f"{sp:.2f}x",
                    f"{rel_err:.2e}",
                ]
            )
        else:
            gpu_throughput.append(None)
            gpu_time_mean.append(None)
            gpu_time_std.append(None)
            max_rel_errors.append(None)
            speedups.append(None)

            rows.append(
                [
                    str(n_wl),
                    f"{cpu_result['mean_s']*1e3:.3f}",
                    "N/A",
                    "N/A",
                    "N/A",
                ]
            )

    print_table(headers, rows, title="Voigt Throughput Benchmark (BENCH-01)")

    results = {
        "benchmark": "voigt_throughput",
        "hardware": hardware_metadata(),
        "parameters": {
            "grid_sizes": grid_sizes,
            "n_lines": n_lines,
            "n_warmup": n_warmup,
            "n_runs": n_runs,
            "sigma_nm": 0.01,
            "gamma_nm": 0.02,
            "wl_range_nm": [200.0, 600.0],
        },
        "results": {
            "cpu_throughput": cpu_throughput,
            "gpu_throughput": gpu_throughput,
            "cpu_time_mean": cpu_time_mean,
            "gpu_time_mean": gpu_time_mean,
            "cpu_time_std": cpu_time_std,
            "gpu_time_std": gpu_time_std,
            "max_relative_error": max_rel_errors,
            "speedup": speedups,
        },
        "references": {
            "accuracy_ref": "Zaghloul (2024) arXiv:2411.00917",
            "method_ref": "Weideman (1994) SIAM J. Numer. Anal. 31, 1497-1518",
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="BENCH-01: Voigt profile throughput benchmark")
    parser.add_argument(
        "--output",
        default="benchmarks/results/voigt_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip GPU benchmarks (CPU scipy baseline only)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of timed runs per configuration (default: 10)",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use smaller grid sizes for quick testing",
    )
    args = parser.parse_args()

    if args.small:
        grid_sizes = [100, 500, 1000, 5000]
    else:
        grid_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]

    results = run_voigt_benchmark(
        grid_sizes=grid_sizes,
        n_runs=args.n_runs,
        cpu_only=args.cpu_only,
    )

    save_results(results, args.output)


if __name__ == "__main__":
    main()
