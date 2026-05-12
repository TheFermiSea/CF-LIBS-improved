# Autonomous Experiment Plan (Exp 1-4)

Started 2026-05-12 21:00 EDT. User offline ~18 hours.

Parallel context: another agent is working on `feat/adr-0001-pattern-survey-impl`
(JAX-internals refactor; Radis/jaxrts/Exojax/petitRADTRANS pattern survey).
Their work has not landed on dev. They may produce merge conflicts in
`scripts/run_unified_benchmark.py` and `cflibs/inversion/` when they rebase.
I am not coordinating with them; I work only on dev.

## Exp 1 — Identifier shootout (IN PROGRESS)

- Cells (5): alias_jax, comb_jax, correlation_jax, spectral_nnls, hybrid_union
- Dataset: full Vrabel (sharded), --vrabel-max-shots 50
- Bandit: warmup 2/arm, 30 iters total per shard
- GPU enabled (JAX_PLATFORMS=cuda)
- Output: `/cluster/shared/cf-libs-bench/results/exp001/shard{1,2,3}/`
- Status: `bash scripts/exp001_status.sh`
- Per-shard PIDs at launch:
  - vasp-01: python ~278504
  - vasp-02: python ~168447
  - vasp-03: python ~675625
- ETA: ~4-6 hours per shard

When all 3 shards finish: aggregate via
`python scripts/aggregate_shards.py /cluster/shared/cf-libs-bench/results/exp001/shard{1,2,3}/results.parquet --output /cluster/shared/cf-libs-bench/results/exp001/merged.parquet`

Then DuckDB:
```sql
SELECT identifier, dataset,
       AVG(f1) AS f1_mean, STDDEV(f1) AS f1_std,
       AVG(false_positives_per_spectrum) AS fp_mean,
       AVG(latency_mean_s) AS lat,
       COUNT(*) AS n
FROM read_parquet('merged.parquet')
WHERE record_kind = 'identification'
GROUP BY identifier, dataset
ORDER BY identifier, dataset;
```

Top-3 by f1_mean on vrabel2020_soil_benchmark → input to Exp 2.

## Exp 2 — Composition workflow shootout (DEPENDS ON EXP 1)

After Exp 1 names the top-3 identifiers, run:
- Cells: top-3 identifiers × {iterative_jax, bayesian}
- 6 cells, --bandit 2 --n-iters 30, same sharding
- Output: `/cluster/shared/cf-libs-bench/results/exp002/`
- Metric: Aitchison distance (d_A), bayesian posterior coverage

Note: bayesian workflow uses NumPyro NUTS; ~50× slower per spectrum.
Use --vrabel-max-shots 25 (half of Exp 1) to keep wall-time comparable.

## Exp 3 — Basis-FWHM gap rebuild (INDEPENDENT; CAN PARALLEL WITH EXP 2)

Vrabel smoke showed basis_fwhm_mismatch ~0.032 nm vs ~0.008 nm for BHVO-2.
Hypothesis: add 4 more FWHMs in the gap: 0.03, 0.04, 0.06, 0.08 nm.

Trigger:
```bash
ssh root@10.0.0.21 'cd /scratch/cf-libs-build-temp 2>/dev/null || rsync ...; \
  JAX_PLATFORMS=cpu .venv/bin/python scripts/build_basis_library.py \
    --fwhm 0.03 0.04 0.06 0.08'
```
~20-30 min CPU on vasp-02. Output goes directly to NFS.

After rebuild: re-run Exp 1's winning cell with the expanded basis grid;
compare F1 / fwhm_mismatch.

## Exp 4 — Final number on winner

Take (winning identifier, winning composition, winning basis grid).
Run --bandit 0 --n-iters 24 (equal allocation) on full sharded Vrabel.
Single cell, 3 shards. Bootstrap CI on F1 + d_A.

## Failure modes to watch

- A shard process dies: relaunch from /scratch/cf-libs-exp001 with the
  same JSON config. Bandit state is per-process so seed reuse gives a
  comparable trajectory.
- NFS basis dir permission issue: chmod 1777 on slurm-ctl, same fix as
  T2.4 incident. See docs/basis-libraries.md.
- Bench-pause flag at /tmp/cf-libs-bench-paused should be present
  throughout to keep post-merge gate from stealing GPU.
- ADR-0001 agent rebasing onto dev may produce merge conflicts in
  scripts/run_unified_benchmark.py. If detected, do NOT auto-resolve;
  flag for human review.

## Critical findings during run-2 (post-relaunch with --vrabel-max-shots=5)

**1. GPU is idle.** Telemetry from PR #145's sampler shows gpu_util ≈ 0% on all 3 vasp nodes during Exp 1 v2, despite JAX_PLATFORMS=cuda being set and JAX preallocating 24 GB of GPU memory. Diagnostic probe confirmed `jax.devices() == [CudaDevice(id=0)]` and `(a*2).sum()` computed correctly on cuda:0. Conclusion: the cf-libs identifier pipeline is CPU-bound — NNLS solvers (scipy.optimize), BoltzmannPlotFitter (numpy), peak detection (numpy), and CSV/JSON I/O dominate. JAX kernels are brief bursts on the GPU; nvidia-smi at 5s cadence rarely catches them. The 0% util is real but doesn't mean GPU is malfunctioning.

**2. JIT-recompile pressure is the real bottleneck.** jax_cache_files = 7394 in 44 MB on vasp-01 mid-run. Each unique spectrum shape (varying n_lines) forces a new compile. This is exactly what ADR-0001's T1-5 (chunked lax.scan over wavelength grid) is designed to fix. We are paying the cost they are addressing in parallel.

**3. Run-1 (vrabel-max-shots=50) was infeasible.** iter-000 ran 2+ hours per shard without completing. 30 iters × 5 cells × ~2 hr cold-JIT per cell ≫ 18 hr budget. Killed and relaunched with --vrabel-max-shots=5 (~167 spec/shard) and --n-iters=20.

**4. Observability layer landed during the run.** PR #145 (`feat/observability-gpu-sampler`) gives cluster-wide telemetry. Surfaced finding #1 within 3 min of deployment — without this we'd have wasted 18 hours assuming GPU was being used.
