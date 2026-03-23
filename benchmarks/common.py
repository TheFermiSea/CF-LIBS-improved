"""
Shared benchmark utilities for CF-LIBS GPU kernel benchmarks.

Provides timing, hardware metadata collection, warmup logic, and
JSON output helpers used by bench_voigt.py, bench_boltzmann.py, and
bench_anderson.py.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def hardware_metadata() -> dict[str, Any]:
    """Collect hardware and software metadata for benchmark provenance.

    Returns
    -------
    dict
        Keys: cpu_model, gpu_model, gpu_driver, cuda_version,
        jax_version, jax_backend, jax_platforms_env, python_version,
        numpy_version, timestamp_utc.
    """
    meta: dict[str, Any] = {
        "cpu_model": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # NumPy version
    try:
        import numpy as np

        meta["numpy_version"] = np.__version__
    except ImportError:
        meta["numpy_version"] = "N/A"

    # JAX info
    try:
        import jax

        meta["jax_version"] = jax.__version__
        meta["jax_backend"] = jax.default_backend()
        devices = jax.devices()
        meta["jax_devices"] = [
            {"id": d.id, "kind": str(d.device_kind)} for d in devices
        ]
    except ImportError:
        meta["jax_version"] = "N/A"
        meta["jax_backend"] = "N/A"
        meta["jax_devices"] = []

    meta["jax_platforms_env"] = os.environ.get("JAX_PLATFORMS", "not set")

    # GPU info via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            meta["gpu_model"] = parts[0] if len(parts) > 0 else "unknown"
            meta["gpu_driver"] = parts[1] if len(parts) > 1 else "unknown"
        else:
            meta["gpu_model"] = "N/A"
            meta["gpu_driver"] = "N/A"
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        meta["gpu_model"] = "N/A"
        meta["gpu_driver"] = "N/A"

    # CUDA version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    meta["cuda_version"] = line.strip()
                    break
            else:
                meta["cuda_version"] = "N/A"
        else:
            meta["cuda_version"] = "N/A"
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        meta["cuda_version"] = "N/A"

    return meta


def benchmark_function(
    fn: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> dict[str, Any]:
    """Time a function with warmup and multiple runs.

    Parameters
    ----------
    fn : callable
        Function to benchmark.
    args : tuple
        Positional arguments for fn.
    kwargs : dict or None
        Keyword arguments for fn.
    n_warmup : int
        Number of warmup calls (excluded from timing).
    n_runs : int
        Number of timed calls.

    Returns
    -------
    dict
        Keys: mean_s, std_s, min_s, max_s, all_times_s, n_runs.
    """
    if kwargs is None:
        kwargs = {}

    # Warmup: run and discard (triggers JIT compilation)
    for _ in range(n_warmup):
        result = fn(*args, **kwargs)
        _block_until_ready(result)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        _block_until_ready(result)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    import numpy as np

    arr = np.array(times)
    return {
        "mean_s": float(np.mean(arr)),
        "std_s": float(np.std(arr)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
        "all_times_s": [float(t) for t in times],
        "n_runs": n_runs,
    }


def _block_until_ready(result: Any) -> None:
    """Call block_until_ready() on JAX arrays to ensure GPU sync."""
    try:
        import jax.numpy as jnp

        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for item in result:
                _block_until_ready(item)
        elif hasattr(result, "__dict__"):
            # Handle dataclasses/namedtuples with JAX array fields
            for val in (
                result._asdict().values()
                if hasattr(result, "_asdict")
                else vars(result).values()
            ):
                if hasattr(val, "block_until_ready"):
                    val.block_until_ready()
    except ImportError:
        pass


def save_results(results: dict, output_path: str) -> None:
    """Write results dict to JSON file with indent=2.

    Parameters
    ----------
    results : dict
        Benchmark results to save.
    output_path : str
        Path to output JSON file.
    """
    ensure_output_dir(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


def ensure_output_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def print_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    """Print a formatted ASCII table to stdout."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))
    print()
