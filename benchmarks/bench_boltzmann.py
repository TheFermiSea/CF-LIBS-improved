#!/usr/bin/env python3
"""
BENCH-02: Boltzmann fitting time benchmark.

Measures wall-clock time for batched Boltzmann plot fitting using the JAX
batched_boltzmann_fit kernel vs the CPU baseline (np.polyfit in a Python
loop) across element counts and lines-per-element configurations.

Output: JSON with hardware metadata, timing for all (element_count,
lines_per_element) combinations, speedup ratios.

Usage:
    python benchmarks/bench_boltzmann.py --output benchmarks/results/boltzmann_results.json
    python benchmarks/bench_boltzmann.py --cpu-only --output /tmp/boltzmann_test.json

References:
    Tognoni et al. (2010) Spectrochim. Acta B 65 -- CF-LIBS methodology.
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

# ASSERT_CONVENTION: x = E_k [eV], y = ln(I*lambda/(g_k*A_ki)) [dimensionless],
#   slope [eV^-1], k_B = 8.617333e-5 eV/K

KB_EV = 8.617333e-5  # eV/K


def generate_boltzmann_data(
    n_elements: int,
    n_lines: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic Boltzmann plot data for benchmarking.

    Parameters
    ----------
    n_elements : int
        Number of elements (batch size).
    n_lines : int
        Number of spectral lines per element.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    x : ndarray, shape (n_elements, n_lines)
        Upper-level energies E_k [eV].
    y : ndarray, shape (n_elements, n_lines)
        Boltzmann plot values ln(I*lambda/(g_k*A_ki)).
    w : ndarray, shape (n_elements, n_lines)
        Weights (1/noise^2).
    mask : ndarray, shape (n_elements, n_lines)
        Boolean mask (all True for this benchmark -- no padding).
    """
    # Random temperatures in [8000, 15000] K per element
    T_K = rng.uniform(8000.0, 15000.0, n_elements)
    slopes = -1.0 / (KB_EV * T_K)  # [eV^-1]
    intercepts = rng.uniform(10.0, 20.0, n_elements)

    # Energy levels sorted per element
    x = np.sort(rng.uniform(0.0, 10.0, (n_elements, n_lines)), axis=1)

    # Boltzmann values: y = slope * E_k + intercept + noise
    noise_std = 0.1
    noise = rng.normal(0.0, noise_std, (n_elements, n_lines))
    y = slopes[:, None] * x + intercepts[:, None] + noise

    # Weights = 1 / noise_variance
    w = np.full((n_elements, n_lines), 1.0 / noise_std**2)

    # All valid (no padding)
    mask = np.ones((n_elements, n_lines), dtype=bool)

    return x, y, w, mask


def cpu_boltzmann_baseline(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> list[tuple[float, float]]:
    """CPU baseline: np.polyfit in a Python loop.

    This replicates the actual CPU codepath -- a sequential loop calling
    np.polyfit for each element. NOT a vectorized approach.

    Parameters
    ----------
    x : ndarray, shape (B, N)
    y : ndarray, shape (B, N)
    w : ndarray, shape (B, N)

    Returns
    -------
    list of (slope, intercept) tuples
    """
    results = []
    for i in range(x.shape[0]):
        # np.polyfit with weights (polyfit wants sqrt(weights))
        coeffs = np.polyfit(x[i], y[i], 1, w=np.sqrt(w[i]))
        results.append((coeffs[0], coeffs[1]))
    return results


def run_boltzmann_benchmark(
    element_counts: list[int],
    lines_per_element: list[int],
    n_warmup: int = 3,
    n_runs: int = 10,
    cpu_only: bool = False,
) -> dict:
    """Run the full Boltzmann fitting benchmark.

    Parameters
    ----------
    element_counts : list of int
        Number of elements (batch sizes) to sweep.
    lines_per_element : list of int
        Lines per element to sweep.
    n_warmup : int
        JIT warmup calls.
    n_runs : int
        Timed runs per configuration.
    cpu_only : bool
        If True, skip JAX benchmarks.

    Returns
    -------
    dict
        Full benchmark results.
    """
    rng = np.random.default_rng(123)

    # Check JAX
    has_jax = False
    if not cpu_only:
        try:
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            from cflibs.inversion.boltzmann_jax import batched_boltzmann_fit

            has_jax = True
        except ImportError:
            print("JAX not available, running CPU-only benchmarks.")

    all_results = []
    headers = ["Elements", "Lines", "CPU (ms)", "GPU (ms)", "Speedup"]
    rows = []

    for n_elem in element_counts:
        for n_lines in lines_per_element:
            print(f"\n--- {n_elem} elements x {n_lines} lines ---")
            x, y, w, mask = generate_boltzmann_data(n_elem, n_lines, rng)

            # CPU baseline
            cpu_timing = benchmark_function(
                cpu_boltzmann_baseline,
                args=(x, y, w),
                n_warmup=1,
                n_runs=n_runs,
            )
            cpu_ms = cpu_timing["mean_s"] * 1e3
            cpu_std_ms = cpu_timing["std_s"] * 1e3
            print(f"  CPU: {cpu_ms:.3f} +/- {cpu_std_ms:.3f} ms")

            entry = {
                "element_count": n_elem,
                "lines_per_element": n_lines,
                "cpu_time_ms_mean": cpu_ms,
                "cpu_time_ms_std": cpu_std_ms,
                "gpu_time_ms_mean": None,
                "gpu_time_ms_std": None,
                "speedup": None,
            }

            # GPU (JAX)
            if has_jax:
                import jax.numpy as jnp
                from cflibs.inversion.boltzmann_jax import batched_boltzmann_fit

                x_jax = jnp.array(x)
                y_jax = jnp.array(y)
                w_jax = jnp.array(w)
                mask_jax = jnp.array(mask)

                gpu_timing = benchmark_function(
                    batched_boltzmann_fit,
                    args=(x_jax, y_jax, w_jax, mask_jax),
                    n_warmup=n_warmup,
                    n_runs=n_runs,
                )
                gpu_ms = gpu_timing["mean_s"] * 1e3
                gpu_std_ms = gpu_timing["std_s"] * 1e3
                sp = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")

                entry["gpu_time_ms_mean"] = gpu_ms
                entry["gpu_time_ms_std"] = gpu_std_ms
                entry["speedup"] = sp

                print(f"  GPU: {gpu_ms:.3f} +/- {gpu_std_ms:.3f} ms")
                print(f"  Speedup: {sp:.2f}x")

                rows.append(
                    [
                        str(n_elem),
                        str(n_lines),
                        f"{cpu_ms:.3f}",
                        f"{gpu_ms:.3f}",
                        f"{sp:.2f}x",
                    ]
                )
            else:
                rows.append(
                    [
                        str(n_elem),
                        str(n_lines),
                        f"{cpu_ms:.3f}",
                        "N/A",
                        "N/A",
                    ]
                )

            all_results.append(entry)

    print_table(headers, rows, title="Boltzmann Fitting Benchmark (BENCH-02)")

    results = {
        "benchmark": "boltzmann_fitting",
        "hardware": hardware_metadata(),
        "parameters": {
            "element_counts": element_counts,
            "lines_per_element": lines_per_element,
            "n_warmup": n_warmup,
            "n_runs": n_runs,
            "noise_std": 0.1,
            "T_range_K": [8000, 15000],
            "E_k_range_eV": [0, 10],
        },
        "results": all_results,
        "references": {
            "method_ref": "Tognoni et al. (2010) Spectrochim. Acta B 65",
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="BENCH-02: Boltzmann fitting time benchmark")
    parser.add_argument(
        "--output",
        default="benchmarks/results/boltzmann_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip GPU benchmarks",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of timed runs (default: 10)",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use smaller sweep for quick testing",
    )
    args = parser.parse_args()

    if args.small:
        element_counts = [1, 5, 10]
        lines_per_element = [10, 50]
    else:
        element_counts = [1, 2, 3, 5, 7, 10, 15, 20]
        lines_per_element = [10, 25, 50, 100]

    results = run_boltzmann_benchmark(
        element_counts=element_counts,
        lines_per_element=lines_per_element,
        n_runs=args.n_runs,
        cpu_only=args.cpu_only,
    )

    save_results(results, args.output)


if __name__ == "__main__":
    main()
