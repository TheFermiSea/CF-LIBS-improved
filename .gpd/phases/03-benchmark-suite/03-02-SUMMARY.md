# Plan 03-02 Summary: FAISS + Batch Forward Benchmarks

## Status: COMPLETE

**Plan contract ref:** 03-02-PLAN.md
**Completed:** 2026-03-23T21:36:13Z
**Duration:** ~15 min
**Tasks:** 2/2

## One-Liner

FAISS query latency scales linearly with index size (0.3 ms at 1K to 19 ms at 100K for FlatL2); batch forward model achieves ~170 spectra/sec on CPU with JAX vmap showing no speedup without GPU hardware.

## Key Results

### FAISS Query Latency (BENCH-04)

| db_size | FlatL2 search (ms) | IVF search (ms) | IVF recall@1 | IVF nlist |
|---------|-------------------|-----------------|--------------|-----------|
| 1,000   | 0.31 +/- 0.03    | 0.55 +/- 0.13  | 0.88         | 31        |
| 10,000  | 1.90 +/- 0.10    | 1.02 +/- 0.01  | 0.70         | 100       |
| 100,000 | 18.89 +/- 2.73   | 2.97 +/- 0.89  | 0.56         | 316       |

[CONFIDENCE: HIGH] -- all numbers from actual FAISS execution, reproducible with seed=42/123, latency monotonically increases with db_size as expected.

**Key observations:**
- FlatL2 scales linearly with database size (brute-force O(N*d))
- IVFFlat is 2-6x faster than FlatL2 at 100K but recall drops to 0.56 with nprobe=10
- For production (1M-100M), IVF with tuned nprobe will be essential
- GPU benchmarks require faiss-gpu (not available on this machine)

### Batch Forward Model Throughput (BENCH-05)

| batch_size | CPU seq (spec/s) | JAX vmap (spec/s) | speedup |
|-----------|-------------------|-------------------|---------|
| 1         | 132               | 153               | 1.2x    |
| 5         | 182               | 97                | 0.5x    |
| 10        | 161               | 108               | 0.7x    |
| 50        | 170               | 94                | 0.6x    |
| 100       | 169               | 105               | 0.6x    |

[CONFIDENCE: HIGH] -- all numbers from actual JAX execution on CPU, warmup excluded, n_runs=10.

**Key observations:**
- CPU sequential throughput ~170 spec/s, roughly constant per-spectrum cost (~6 ms/spec)
- JAX vmap on CPU is SLOWER than sequential loop (expected: vmap dispatches to XLA which has overhead on CPU without parallelism benefit)
- On actual GPU hardware, vmap should show significant speedup due to parallel warp execution
- ExoJAX (arXiv:2105.14782) reports ~10,000 spectra/sec on V100 for similar spectral models; our GPU benchmark will provide direct comparison when run on V100S

## Contract Results

### Claims

| Claim ID | Status | Evidence |
|----------|--------|----------|
| claim-faiss-bench-script | DELIVERED | `benchmarks/bench_faiss.py` with FlatL2 + IVFFlat + GPU support |
| claim-batch-forward-bench-script | DELIVERED | `benchmarks/bench_batch_forward.py` with CPU sequential + GPU vmap |

### Deliverables

| Deliverable ID | Status | Path |
|---------------|--------|------|
| deliv-faiss-bench | PRODUCED | `benchmarks/bench_faiss.py` |
| deliv-batch-forward-bench | PRODUCED | `benchmarks/bench_batch_forward.py` |

### Acceptance Tests

| Test ID | Outcome | Notes |
|---------|---------|-------|
| test-faiss-runs | PASS | `--small --cpu-only` produces valid JSON with latency arrays, build times, recall values |
| test-batch-forward-runs | PASS | `--small --cpu-only` produces valid JSON with throughput arrays and timing data |

### References

| Reference ID | Status | Action Taken |
|-------------|--------|-------------|
| ref-exojax | NOTED | ExoJAX comparison noted in bench_batch_forward.py output; direct throughput comparison deferred to GPU run |

### Forbidden Proxies

| Proxy ID | Status |
|----------|--------|
| fp-fabricated-benchmarks | REJECTED -- all numbers from actual FAISS/JAX execution |
| fp-isolated-kernel | REJECTED -- GPU path includes device_put transfer time measurement |

## Files Created/Modified

| File | Purpose |
|------|---------|
| `benchmarks/bench_faiss.py` | FAISS query latency benchmark (BENCH-04) |
| `benchmarks/bench_batch_forward.py` | Batch forward throughput benchmark (BENCH-05) |
| `benchmarks/common.py` | Fix PermissionError in nvidia-smi/nvcc calls |
| `benchmarks/results/faiss_results.json` | CPU-only benchmark results (small mode) |
| `benchmarks/results/batch_forward_results.json` | CPU-only benchmark results (small mode) |

## Checkpoints

| Task | Hash | Description |
|------|------|-------------|
| 1 | 7a62b59 | FAISS benchmark script + common.py fix |
| 2 | d515804 | Batch forward benchmark + results JSON |
| fmt | fc556a9 | Black formatting |

## Conventions

- T [eV], n_e [cm^-3], lambda [nm] per project convention lock
- Timing: wall-clock seconds via time.perf_counter()
- JAX: block_until_ready() for GPU sync
- Statistics: N_runs >= 10, warmup excluded, report mean +/- std
- FAISS vectors: float32, L2-normalized, d=30

## Deviations

1. [Rule 4 - Missing component] common.py `hardware_metadata()` raised PermissionError on nvidia-smi; added PermissionError/OSError to exception handler.
2. [Observation] JAX vmap on CPU is slower than sequential loop -- this is expected behavior, not a bug. GPU hardware required for meaningful speedup measurement.
3. [Observation] IVF recall@1 drops to 0.56 at 100K with nprobe=10 and d=30 -- higher nprobe or HNSW index may be needed for production.

## Self-Check: PASSED

- [x] benchmarks/bench_faiss.py exists and runs
- [x] benchmarks/bench_batch_forward.py exists and runs
- [x] benchmarks/results/faiss_results.json exists with valid JSON
- [x] benchmarks/results/batch_forward_results.json exists with valid JSON
- [x] All checkpoint hashes verified in git log
- [x] ruff check clean
- [x] black formatted
- [x] No fabricated numbers

```yaml
gpd_return:
  status: completed
  files_written:
    - benchmarks/bench_faiss.py
    - benchmarks/bench_batch_forward.py
    - benchmarks/common.py
    - benchmarks/results/faiss_results.json
    - benchmarks/results/batch_forward_results.json
    - .gpd/phases/03-benchmark-suite/03-02-SUMMARY.md
  issues:
    - "FAISS GPU benchmarks require faiss-gpu installation on V100S nodes"
    - "Batch forward GPU benchmarks need re-run on actual GPU hardware for meaningful speedup numbers"
    - "IVF recall@1 at d=30 may need nprobe tuning or different index type (HNSW) for production"
  next_actions:
    - "Run bench_faiss.py on V100S with faiss-gpu for GPU latency numbers"
    - "Run bench_batch_forward.py on V100S for GPU throughput vs ExoJAX comparison"
    - "Tune IVF nprobe or test HNSW index for better recall/latency tradeoff"
  phase: "03"
  plan: "02"
  tasks_completed: 2
  tasks_total: 2
  duration_seconds: 900
```
