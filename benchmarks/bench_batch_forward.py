#!/usr/bin/env python3
"""Batch forward model throughput benchmark (BENCH-05).

Measures spectra/sec vs batch size for CPU sequential loop vs JAX GPU vmap,
using the batch_forward_model from cflibs.manifold.batch_forward.

Output: JSON with throughput, timing, and memory data.

Usage
-----
    python benchmarks/bench_batch_forward.py --output benchmarks/results/batch_forward_results.json
    python benchmarks/bench_batch_forward.py --small --cpu-only --output /tmp/test_batch.json

References
----------
Kawahara et al. (2022) arXiv:2105.14782 -- ExoJAX: GPU spectral model throughput.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# JAX availability (enable float64 before any JAX import)
# ---------------------------------------------------------------------------
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hardware_metadata, save_results, print_table  # noqa: E402

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Add project root to path for cflibs imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cflibs.manifold.batch_forward import (  # noqa: E402
    BatchAtomicData,
    single_spectrum_forward,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_LINES = 50  # Representative line count for multi-element LIBS
N_WL = 4096  # Wavelength grid points (manifold standard)
N_ELEMENTS = 5  # Number of elements in mixture
N_RUNS_DEFAULT = 10
N_RUNS_LARGE = 3  # Reduced runs for batch_size >= 5000

BATCH_SIZES_FULL = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
BATCH_SIZES_SMALL = [1, 5, 10, 50, 100]


def make_synthetic_atomic_data(
    n_lines: int = N_LINES,
    n_elements: int = N_ELEMENTS,
    seed: int = 42,
) -> BatchAtomicData:
    """Create synthetic atomic data for benchmarking.

    Generates physically reasonable (but not real) atomic parameters
    for n_lines spectral lines across n_elements elements.

    Parameters
    ----------
    n_lines : int
        Number of spectral lines.
    n_elements : int
        Number of elements.
    seed : int
        Random seed.

    Returns
    -------
    BatchAtomicData
        Packed atomic data arrays.
    """
    rng = np.random.default_rng(seed)

    wl = np.sort(rng.uniform(200.0, 600.0, n_lines))  # nm
    A_ki = 10.0 ** rng.uniform(6.0, 9.0, n_lines)  # s^-1
    E_k = rng.uniform(0.0, 10.0, n_lines)  # eV
    g_k = rng.integers(1, 11, n_lines).astype(np.float64)
    element_idx = rng.integers(0, n_elements, n_lines).astype(np.int32)
    ion_stage = rng.integers(0, 2, n_lines).astype(np.int32)  # 0=I, 1=II
    stark_w = rng.uniform(0.001, 0.1, n_lines)  # nm HWHM at 1e16
    mass_amu = np.array([55.85, 63.55, 26.98, 47.87, 58.69])[element_idx]  # Fe,Cu,Al,Ti,Ni

    # Ionization potentials [eV] for 3 stages
    ip = np.array([
        [7.9, 16.2],   # Fe-like
        [7.7, 20.3],   # Cu-like
        [6.0, 18.8],   # Al-like
        [6.8, 13.6],   # Ti-like
        [7.6, 18.2],   # Ni-like
    ])

    # Partition function coefficients (Irwin polynomial, 5 coeffs per stage)
    pf = np.zeros((n_elements, 3, 5), dtype=np.float64)
    pf[:, 0, 0] = np.log(25.0)  # Neutral: U ~ 25
    pf[:, 1, 0] = np.log(15.0)  # Singly ionized: U ~ 15
    pf[:, 2, 0] = np.log(10.0)  # Doubly ionized: U ~ 10

    return BatchAtomicData(
        line_wavelengths=wl,
        line_A_ki=A_ki,
        line_g_k=g_k,
        line_E_k=E_k,
        line_element_idx=element_idx,
        line_ion_stage=ion_stage,
        line_stark_w=stark_w,
        line_mass_amu=mass_amu,
        ionization_potentials=ip,
        partition_coeffs=pf,
        n_elements=n_elements,
        n_stages=3,
    )


def make_plasma_params(
    batch_size: int,
    n_elements: int = N_ELEMENTS,
    seed: int = 77,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random plasma parameters for benchmarking.

    Parameters
    ----------
    batch_size : int
        Number of parameter sets.
    n_elements : int
        Number of elements.
    seed : int
        Random seed.

    Returns
    -------
    T_eV : ndarray, shape (batch_size,)
        Temperature in eV.
    n_e : ndarray, shape (batch_size,)
        Electron density in cm^-3.
    C : ndarray, shape (batch_size, n_elements)
        Number fractions (sum = 1).
    """
    rng = np.random.default_rng(seed)

    # T in eV: 8000-15000 K -> 0.69-1.29 eV
    T_eV = rng.uniform(0.69, 1.29, batch_size)
    # n_e: log-uniform in [1e15, 1e17]
    n_e = 10.0 ** rng.uniform(15.0, 17.0, batch_size)
    # Dirichlet for simplex compositions
    C = rng.dirichlet(np.ones(n_elements), batch_size)

    return T_eV, n_e, C


def estimate_memory_gb(batch_size: int, n_wl: int, n_lines: int) -> float:
    """Estimate peak memory for batch computation in GB.

    Dominant term: batch_size * n_wl * n_lines * 8 bytes (float64 intermediate).
    """
    return batch_size * n_wl * n_lines * 8 / 1e9


def run_cpu_sequential(
    T_eV: np.ndarray,
    n_e: np.ndarray,
    C: np.ndarray,
    wl_grid: np.ndarray,
    atomic_data: BatchAtomicData,
    n_runs: int,
) -> dict:
    """Time CPU sequential loop (one spectrum at a time).

    This is the baseline: a Python for-loop calling single_spectrum_forward.
    """
    batch_size = T_eV.shape[0]

    def run_once():
        spectra = []
        for i in range(batch_size):
            s = single_spectrum_forward(
                float(T_eV[i]),
                float(n_e[i]),
                C[i],
                wl_grid,
                atomic_data,
            )
            if hasattr(s, "block_until_ready"):
                s.block_until_ready()
            spectra.append(np.asarray(s))
        return np.stack(spectra, axis=0)

    # Warmup (1 call)
    _ = run_once()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = run_once()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    arr = np.array(times)
    mean_s = float(np.mean(arr))

    return {
        "total_time_ms_mean": round(mean_s * 1000, 4),
        "total_time_ms_std": round(float(np.std(arr)) * 1000, 4),
        "per_spectrum_ms": round(mean_s / batch_size * 1000, 4),
        "throughput_spectra_per_sec": round(batch_size / mean_s, 2) if mean_s > 0 else 0.0,
        "n_runs": n_runs,
        "output_shape": list(result.shape),
    }


def run_gpu_batch(
    T_eV: np.ndarray,
    n_e: np.ndarray,
    C: np.ndarray,
    wl_grid: np.ndarray,
    atomic_data: BatchAtomicData,
    n_runs: int,
) -> dict:
    """Time GPU batched vmap forward model.

    Includes separate measurement of data transfer time.
    """
    from cflibs.manifold.batch_forward import batch_forward_model

    # Convert to JAX arrays
    T_jax = jnp.asarray(T_eV, dtype=jnp.float64)
    ne_jax = jnp.asarray(n_e, dtype=jnp.float64)
    C_jax = jnp.asarray(C, dtype=jnp.float64)
    wl_jax = jnp.asarray(wl_grid, dtype=jnp.float64)

    # Measure transfer time separately
    transfer_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        T_d = jax.device_put(jnp.asarray(T_eV, dtype=jnp.float64))
        ne_d = jax.device_put(jnp.asarray(n_e, dtype=jnp.float64))
        C_d = jax.device_put(jnp.asarray(C, dtype=jnp.float64))
        T_d.block_until_ready()
        ne_d.block_until_ready()
        C_d.block_until_ready()
        t1 = time.perf_counter()
        transfer_times.append(t1 - t0)
    transfer_ms = float(np.mean(transfer_times)) * 1000

    batch_size = T_eV.shape[0]

    # Warmup (triggers JIT compilation)
    result = batch_forward_model(T_jax, ne_jax, C_jax, wl_jax, atomic_data)
    result.block_until_ready()

    # Timed runs (compute only -- data already on device)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = batch_forward_model(T_jax, ne_jax, C_jax, wl_jax, atomic_data)
        result.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    arr = np.array(times)
    mean_s = float(np.mean(arr))

    return {
        "total_time_ms_mean": round(mean_s * 1000, 4),
        "total_time_ms_std": round(float(np.std(arr)) * 1000, 4),
        "per_spectrum_ms": round(mean_s / batch_size * 1000, 4),
        "throughput_spectra_per_sec": round(batch_size / mean_s, 2) if mean_s > 0 else 0.0,
        "transfer_time_ms": round(transfer_ms, 4),
        "n_runs": n_runs,
        "output_shape": list(np.asarray(result).shape),
    }


def benchmark_batch_size(
    batch_size: int,
    wl_grid: np.ndarray,
    atomic_data: BatchAtomicData,
    n_runs: int,
    cpu_only: bool,
    max_batch: int | None,
) -> dict:
    """Run benchmark for a single batch size."""
    entry: dict = {
        "batch_size": batch_size,
        "estimated_memory_gb": round(estimate_memory_gb(batch_size, N_WL, N_LINES), 4),
    }

    if max_batch is not None and batch_size > max_batch:
        entry["skipped"] = True
        entry["skip_reason"] = f"batch_size > max_batch ({max_batch})"
        print(f"  batch={batch_size}: SKIPPED (> max_batch)")
        return entry

    T_eV, n_e, C = make_plasma_params(batch_size)

    # -----------------------------------------------------------------------
    # CPU sequential baseline
    # -----------------------------------------------------------------------
    print(f"  batch={batch_size}: CPU sequential...", end=" ", flush=True)
    try:
        cpu_result = run_cpu_sequential(T_eV, n_e, C, wl_grid, atomic_data, n_runs)
        entry["cpu_total_time_ms_mean"] = cpu_result["total_time_ms_mean"]
        entry["cpu_total_time_ms_std"] = cpu_result["total_time_ms_std"]
        entry["cpu_per_spectrum_ms"] = cpu_result["per_spectrum_ms"]
        entry["cpu_throughput_spectra_per_sec"] = cpu_result["throughput_spectra_per_sec"]
        print(
            f"{cpu_result['throughput_spectra_per_sec']:.1f} spec/s "
            f"({cpu_result['per_spectrum_ms']:.2f} ms/spec)"
        )
    except Exception as e:
        entry["cpu_error"] = str(e)
        print(f"ERROR: {e}")

    # -----------------------------------------------------------------------
    # GPU batch (JAX vmap)
    # -----------------------------------------------------------------------
    if HAS_JAX and not cpu_only:
        print(f"  batch={batch_size}: GPU batch...", end=" ", flush=True)
        try:
            gpu_result = run_gpu_batch(T_eV, n_e, C, wl_grid, atomic_data, n_runs)
            entry["gpu_total_time_ms_mean"] = gpu_result["total_time_ms_mean"]
            entry["gpu_total_time_ms_std"] = gpu_result["total_time_ms_std"]
            entry["gpu_per_spectrum_ms"] = gpu_result["per_spectrum_ms"]
            entry["gpu_throughput_spectra_per_sec"] = gpu_result["throughput_spectra_per_sec"]
            entry["gpu_transfer_time_ms"] = gpu_result["transfer_time_ms"]

            # Speedup
            if entry.get("cpu_throughput_spectra_per_sec", 0) > 0:
                entry["speedup"] = round(
                    gpu_result["throughput_spectra_per_sec"]
                    / entry["cpu_throughput_spectra_per_sec"],
                    2,
                )
            else:
                entry["speedup"] = None

            print(
                f"{gpu_result['throughput_spectra_per_sec']:.1f} spec/s "
                f"({gpu_result['per_spectrum_ms']:.2f} ms/spec)  "
                f"speedup={entry.get('speedup', 'N/A')}x"
            )
        except Exception as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "oom" in err_str or "resource" in err_str:
                entry["gpu_oom"] = True
                print(f"OOM ({e})")
            else:
                entry["gpu_error"] = str(e)
                print(f"ERROR: {e}")
    else:
        entry["gpu_total_time_ms_mean"] = None
        entry["gpu_total_time_ms_std"] = None
        entry["gpu_per_spectrum_ms"] = None
        entry["gpu_throughput_spectra_per_sec"] = None
        entry["gpu_transfer_time_ms"] = None
        entry["speedup"] = None

    gc.collect()
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch forward model throughput benchmark (BENCH-05)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/batch_forward_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use reduced batch sizes (1, 5, 10, 50, 100)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip GPU benchmarks",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS_DEFAULT,
        help=f"Number of timed runs (default {N_RUNS_DEFAULT})",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="Maximum batch size to test (skip larger sizes)",
    )
    args = parser.parse_args()

    batch_sizes = BATCH_SIZES_SMALL if args.small else BATCH_SIZES_FULL

    print("=" * 60)
    print("Batch Forward Model Throughput Benchmark (BENCH-05)")
    print("=" * 60)
    print(f"  N_lines:       {N_LINES}")
    print(f"  N_wl:          {N_WL}")
    print(f"  N_elements:    {N_ELEMENTS}")
    print(f"  n_runs:        {args.n_runs}")
    print(f"  Batch sizes:   {batch_sizes}")
    print(f"  JAX available: {HAS_JAX}")
    if HAS_JAX:
        print(f"  JAX backend:   {jax.default_backend()}")
        print(f"  JAX devices:   {jax.devices()}")
    print(f"  CPU-only mode: {args.cpu_only}")

    hw = hardware_metadata()

    # Prepare shared data
    wl_grid = np.linspace(200.0, 600.0, N_WL)
    atomic_data = make_synthetic_atomic_data()

    results_list = []
    for bs in batch_sizes:
        n_runs = N_RUNS_LARGE if bs >= 5000 else args.n_runs
        entry = benchmark_batch_size(
            batch_size=bs,
            wl_grid=wl_grid,
            atomic_data=atomic_data,
            n_runs=n_runs,
            cpu_only=args.cpu_only,
            max_batch=args.max_batch,
        )
        results_list.append(entry)

    output = {
        "benchmark": "batch_forward_model",
        "hardware": hw,
        "parameters": {
            "batch_sizes": batch_sizes,
            "n_lines": N_LINES,
            "n_wl": N_WL,
            "n_elements": N_ELEMENTS,
            "n_runs_default": args.n_runs,
            "n_runs_large": N_RUNS_LARGE,
        },
        "results": results_list,
    }

    save_results(output, args.output)

    # Summary table
    headers = [
        "batch",
        "cpu_ms/spec",
        "cpu_spec/s",
        "gpu_ms/spec",
        "gpu_spec/s",
        "speedup",
        "mem_GB",
    ]
    rows = []
    for r in results_list:
        if r.get("skipped"):
            rows.append([str(r["batch_size"]), "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", "SKIP"])
            continue

        def fmt(val, digits=2):
            if isinstance(val, (int, float)):
                return f"{val:.{digits}f}"
            return "N/A"

        rows.append([
            str(r["batch_size"]),
            fmt(r.get("cpu_per_spectrum_ms")),
            fmt(r.get("cpu_throughput_spectra_per_sec"), 1),
            fmt(r.get("gpu_per_spectrum_ms")),
            fmt(r.get("gpu_throughput_spectra_per_sec"), 1),
            fmt(r.get("speedup"), 1),
            fmt(r.get("estimated_memory_gb"), 4),
        ])
    print_table(headers, rows, title="Batch Forward Model Benchmark Summary")

    # Note about ExoJAX comparison
    print("Note: For comparison with ExoJAX (arXiv:2105.14782) GPU throughput,")
    print("see the gpu_throughput_spectra_per_sec values above.")


if __name__ == "__main__":
    main()
