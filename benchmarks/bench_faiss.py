#!/usr/bin/env python3
"""FAISS query latency benchmark (BENCH-04).

Measures FAISS index build time and k-NN search latency across multiple
database sizes, index types (FlatL2, IVFFlat), and backends (CPU, GPU).

Output: JSON with latency arrays, index build times, and recall values.

Usage
-----
    python benchmarks/bench_faiss.py --output benchmarks/results/faiss_results.json
    python benchmarks/bench_faiss.py --small --cpu-only --output /tmp/test_faiss.json
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# FAISS import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore[import-untyped]

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

# Check for GPU support
HAS_FAISS_GPU = False
if HAS_FAISS:
    HAS_FAISS_GPU = hasattr(faiss, "StandardGpuResources")

# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hardware_metadata, save_results, print_table  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_DIM = 30  # PCA-reduced embedding dimension (SpectralEmbedder)
N_QUERIES = 100  # Batch of queries per measurement
K = 10  # Number of nearest neighbors
N_RUNS = 10  # Repetitions for timing statistics

DB_SIZES_FULL = [100_000, 1_000_000, 10_000_000, 100_000_000]
DB_SIZES_SMALL = [1_000, 10_000, 100_000]


def generate_vectors(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Generate L2-normalized random float32 vectors.

    Parameters
    ----------
    n : int
        Number of vectors.
    d : int
        Vector dimension.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n, d), dtype float32
        L2-normalized vectors.
    """
    rng = np.random.default_rng(seed)
    # For very large arrays, generate in chunks to avoid memory spikes
    chunk_size = min(n, 1_000_000)
    vectors = np.empty((n, d), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = rng.standard_normal((end - start, d)).astype(np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vectors[start:end] = chunk / norms
    return vectors


def compute_recall_at_1(gt_indices: np.ndarray, test_indices: np.ndarray) -> float:
    """Compute recall@1: fraction of queries where the top-1 result matches ground truth.

    Parameters
    ----------
    gt_indices : np.ndarray, shape (n_queries, k)
        Ground-truth neighbor indices from exact search.
    test_indices : np.ndarray, shape (n_queries, k)
        Neighbor indices from approximate search.

    Returns
    -------
    float
        Recall@1 in [0, 1].
    """
    return float(np.mean(gt_indices[:, 0] == test_indices[:, 0]))


def estimate_memory_gb(n: int, d: int) -> float:
    """Estimate memory for float32 vector array in GB."""
    return n * d * 4 / 1e9


def benchmark_single_size(
    db_size: int, d: int, k: int, n_queries: int, n_runs: int, cpu_only: bool
) -> dict:
    """Run all FAISS benchmarks for a single database size.

    Returns a dict with CPU flat, CPU IVF, GPU flat, GPU IVF results.
    """
    result: dict = {
        "db_size": db_size,
        "estimated_memory_gb": round(estimate_memory_gb(db_size, d), 3),
        "gpu_available": HAS_FAISS_GPU and not cpu_only,
        "gpu_oom": False,
    }

    print(f"\n--- db_size = {db_size:,} (est. {result['estimated_memory_gb']:.3f} GB) ---")

    # Generate database and query vectors
    print("  Generating vectors...", end=" ", flush=True)
    t0 = time.perf_counter()
    db_vectors = generate_vectors(db_size, d, seed=42)
    queries = generate_vectors(n_queries, d, seed=123)
    t_gen = time.perf_counter() - t0
    print(f"done ({t_gen:.2f}s)")

    # -----------------------------------------------------------------------
    # CPU FlatL2 (exact search -- ground truth)
    # -----------------------------------------------------------------------
    print("  CPU FlatL2: building...", end=" ", flush=True)
    t0 = time.perf_counter()
    cpu_flat = faiss.IndexFlatL2(d)
    cpu_flat.add(db_vectors)
    build_time = time.perf_counter() - t0
    result["cpu_flat_build_s"] = round(build_time, 4)
    print(f"{build_time:.3f}s")

    # Search (warmup + timed runs)
    print(f"  CPU FlatL2: searching (n_runs={n_runs})...", end=" ", flush=True)
    # Warmup
    _, gt_indices = cpu_flat.search(queries, k)

    search_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _, gt_indices = cpu_flat.search(queries, k)
        search_times.append(time.perf_counter() - t0)

    arr = np.array(search_times)
    result["cpu_flat_search_ms_mean"] = round(float(np.mean(arr)) * 1000, 4)
    result["cpu_flat_search_ms_std"] = round(float(np.std(arr)) * 1000, 4)
    result["cpu_flat_per_query_us"] = round(float(np.mean(arr)) / n_queries * 1e6, 4)
    print(f"{result['cpu_flat_search_ms_mean']:.3f} +/- {result['cpu_flat_search_ms_std']:.3f} ms")

    # -----------------------------------------------------------------------
    # CPU IVFFlat (approximate search)
    # -----------------------------------------------------------------------
    nlist = max(4, int(sqrt(db_size)))
    nprobe = min(nlist, 10)
    print(f"  CPU IVFFlat (nlist={nlist}, nprobe={nprobe}): training...", end=" ", flush=True)

    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    cpu_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)

    # Train on subset for large databases
    train_size = min(db_size, 500_000)
    if train_size < db_size:
        rng = np.random.default_rng(99)
        train_idx = rng.choice(db_size, train_size, replace=False)
        train_vecs = db_vectors[train_idx]
    else:
        train_vecs = db_vectors
    cpu_ivf.train(train_vecs)
    cpu_ivf.add(db_vectors)
    build_time_ivf = time.perf_counter() - t0
    result["cpu_ivf_build_s"] = round(build_time_ivf, 4)
    print(f"{build_time_ivf:.3f}s")

    cpu_ivf.nprobe = nprobe

    # Search
    print(f"  CPU IVFFlat: searching (n_runs={n_runs})...", end=" ", flush=True)
    # Warmup
    _, ivf_indices = cpu_ivf.search(queries, k)

    search_times_ivf = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _, ivf_indices = cpu_ivf.search(queries, k)
        search_times_ivf.append(time.perf_counter() - t0)

    arr_ivf = np.array(search_times_ivf)
    result["cpu_ivf_search_ms_mean"] = round(float(np.mean(arr_ivf)) * 1000, 4)
    result["cpu_ivf_search_ms_std"] = round(float(np.std(arr_ivf)) * 1000, 4)
    result["cpu_ivf_per_query_us"] = round(float(np.mean(arr_ivf)) / n_queries * 1e6, 4)
    result["cpu_ivf_recall_at_1"] = round(compute_recall_at_1(gt_indices, ivf_indices), 4)
    result["cpu_ivf_nlist"] = nlist
    result["cpu_ivf_nprobe"] = nprobe
    print(
        f"{result['cpu_ivf_search_ms_mean']:.3f} +/- {result['cpu_ivf_search_ms_std']:.3f} ms  "
        f"recall@1={result['cpu_ivf_recall_at_1']:.4f}"
    )

    # -----------------------------------------------------------------------
    # GPU FlatL2
    # -----------------------------------------------------------------------
    if HAS_FAISS_GPU and not cpu_only:
        try:
            print("  GPU FlatL2: transferring...", end=" ", flush=True)
            res = faiss.StandardGpuResources()
            t0 = time.perf_counter()
            gpu_flat = faiss.index_cpu_to_gpu(res, 0, cpu_flat)
            transfer_time = time.perf_counter() - t0
            print(f"transfer {transfer_time:.3f}s, searching...", end=" ", flush=True)

            # Warmup
            gpu_flat.search(queries, k)

            search_times_gpu = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                gpu_flat.search(queries, k)
                search_times_gpu.append(time.perf_counter() - t0)

            arr_gpu = np.array(search_times_gpu)
            result["gpu_flat_search_ms_mean"] = round(float(np.mean(arr_gpu)) * 1000, 4)
            result["gpu_flat_search_ms_std"] = round(float(np.std(arr_gpu)) * 1000, 4)
            result["gpu_flat_per_query_us"] = round(float(np.mean(arr_gpu)) / n_queries * 1e6, 4)
            result["gpu_flat_transfer_s"] = round(transfer_time, 4)
            print(
                f"{result['gpu_flat_search_ms_mean']:.3f} +/- "
                f"{result['gpu_flat_search_ms_std']:.3f} ms"
            )
            del gpu_flat
        except Exception as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                result["gpu_oom"] = True
                print(f"OOM ({e})")
            else:
                print(f"ERROR: {e}")
                result["gpu_flat_error"] = str(e)

        # GPU IVFFlat
        try:
            print("  GPU IVFFlat: transferring...", end=" ", flush=True)
            res2 = faiss.StandardGpuResources()
            t0 = time.perf_counter()
            gpu_ivf = faiss.index_cpu_to_gpu(res2, 0, cpu_ivf)
            transfer_time_ivf = time.perf_counter() - t0
            print(f"transfer {transfer_time_ivf:.3f}s, searching...", end=" ", flush=True)

            gpu_ivf.nprobe = nprobe

            # Warmup
            gpu_ivf.search(queries, k)

            search_times_gpu_ivf = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _, gpu_ivf_indices = gpu_ivf.search(queries, k)
                search_times_gpu_ivf.append(time.perf_counter() - t0)

            arr_gpu_ivf = np.array(search_times_gpu_ivf)
            result["gpu_ivf_search_ms_mean"] = round(float(np.mean(arr_gpu_ivf)) * 1000, 4)
            result["gpu_ivf_search_ms_std"] = round(float(np.std(arr_gpu_ivf)) * 1000, 4)
            result["gpu_ivf_per_query_us"] = round(float(np.mean(arr_gpu_ivf)) / n_queries * 1e6, 4)
            result["gpu_ivf_recall_at_1"] = round(
                compute_recall_at_1(gt_indices, gpu_ivf_indices), 4
            )
            result["gpu_ivf_transfer_s"] = round(transfer_time_ivf, 4)
            print(
                f"{result['gpu_ivf_search_ms_mean']:.3f} +/- "
                f"{result['gpu_ivf_search_ms_std']:.3f} ms  "
                f"recall@1={result['gpu_ivf_recall_at_1']:.4f}"
            )
            del gpu_ivf
        except Exception as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                result["gpu_oom"] = True
                print(f"OOM ({e})")
            else:
                print(f"ERROR: {e}")
                result["gpu_ivf_error"] = str(e)
    else:
        result["gpu_flat_search_ms_mean"] = None
        result["gpu_flat_search_ms_std"] = None
        result["gpu_flat_per_query_us"] = None
        result["gpu_ivf_search_ms_mean"] = None
        result["gpu_ivf_search_ms_std"] = None
        result["gpu_ivf_recall_at_1"] = None

    # Cleanup
    del cpu_flat, cpu_ivf, db_vectors, queries
    gc.collect()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS query latency benchmark (BENCH-04)")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/faiss_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use reduced database sizes (1K, 10K, 100K) for quick testing",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip GPU benchmarks even if faiss-gpu is available",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS,
        help=f"Number of timed search runs (default {N_RUNS})",
    )
    args = parser.parse_args()

    if not HAS_FAISS:
        print("ERROR: faiss is not installed. Install with: pip install faiss-cpu")
        sys.exit(1)

    db_sizes = DB_SIZES_SMALL if args.small else DB_SIZES_FULL
    n_runs = args.n_runs

    print("=" * 60)
    print("FAISS Query Latency Benchmark (BENCH-04)")
    print("=" * 60)
    print(f"  Vector dimension: {VECTOR_DIM}")
    print(f"  k (neighbors):    {K}")
    print(f"  n_queries:        {N_QUERIES}")
    print(f"  n_runs:           {n_runs}")
    print(f"  Database sizes:   {db_sizes}")
    print(f"  FAISS version:    {faiss.__version__}")
    print(f"  FAISS GPU:        {HAS_FAISS_GPU}")
    print(f"  CPU-only mode:    {args.cpu_only}")

    hw = hardware_metadata()
    results_list = []

    for db_size in db_sizes:
        mem_gb = estimate_memory_gb(db_size, VECTOR_DIM)
        # Skip if estimated memory exceeds 14 GB (V100S has 32 GB but leave margin)
        if mem_gb > 14.0:
            print(f"\n--- db_size = {db_size:,}: SKIPPING (est. {mem_gb:.1f} GB > 14 GB limit) ---")
            results_list.append(
                {
                    "db_size": db_size,
                    "estimated_memory_gb": round(mem_gb, 3),
                    "skipped": True,
                    "skip_reason": f"estimated memory {mem_gb:.1f} GB exceeds 14 GB limit",
                }
            )
            continue

        entry = benchmark_single_size(
            db_size=db_size,
            d=VECTOR_DIM,
            k=K,
            n_queries=N_QUERIES,
            n_runs=n_runs,
            cpu_only=args.cpu_only,
        )
        results_list.append(entry)

    output = {
        "benchmark": "faiss_query_latency",
        "hardware": hw,
        "parameters": {
            "db_sizes": db_sizes,
            "d": VECTOR_DIM,
            "k": K,
            "n_queries": N_QUERIES,
            "n_runs": n_runs,
        },
        "results": results_list,
    }

    save_results(output, args.output)

    # Summary table
    headers = [
        "db_size",
        "flat_build_s",
        "flat_ms",
        "ivf_build_s",
        "ivf_ms",
        "ivf_R@1",
        "gpu_flat_ms",
        "gpu_ivf_ms",
    ]
    rows = []
    for r in results_list:
        if r.get("skipped"):
            rows.append(
                [
                    f"{r['db_size']:,}",
                    "SKIP",
                    "SKIP",
                    "SKIP",
                    "SKIP",
                    "SKIP",
                    "SKIP",
                    "SKIP",
                ]
            )
            continue
        rows.append(
            [
                f"{r['db_size']:,}",
                (
                    f"{r.get('cpu_flat_build_s', 'N/A'):.3f}"
                    if isinstance(r.get("cpu_flat_build_s"), (int, float))
                    else "N/A"
                ),
                (
                    f"{r.get('cpu_flat_search_ms_mean', 'N/A'):.3f}"
                    if isinstance(r.get("cpu_flat_search_ms_mean"), (int, float))
                    else "N/A"
                ),
                (
                    f"{r.get('cpu_ivf_build_s', 'N/A'):.3f}"
                    if isinstance(r.get("cpu_ivf_build_s"), (int, float))
                    else "N/A"
                ),
                (
                    f"{r.get('cpu_ivf_search_ms_mean', 'N/A'):.3f}"
                    if isinstance(r.get("cpu_ivf_search_ms_mean"), (int, float))
                    else "N/A"
                ),
                (
                    f"{r.get('cpu_ivf_recall_at_1', 'N/A'):.4f}"
                    if isinstance(r.get("cpu_ivf_recall_at_1"), (int, float))
                    else "N/A"
                ),
                (
                    f"{r.get('gpu_flat_search_ms_mean', 'N/A'):.3f}"
                    if isinstance(r.get("gpu_flat_search_ms_mean"), (int, float))
                    else "N/A"
                ),
                (
                    f"{r.get('gpu_ivf_search_ms_mean', 'N/A'):.3f}"
                    if isinstance(r.get("gpu_ivf_search_ms_mean"), (int, float))
                    else "N/A"
                ),
            ]
        )
    print_table(headers, rows, title="FAISS Benchmark Summary")


if __name__ == "__main__":
    main()
