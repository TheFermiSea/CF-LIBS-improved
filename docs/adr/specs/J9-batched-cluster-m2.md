# J9 Implementation Spec — Batched Execution on V100S, LDM Validation, Cluster Discipline → Milestone M2

**Bead:** J9 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §5.2/§5.5, §8.2 (M2), risks R3/R4/R6 · **Track:** spine · **Depends:** J8 · **Estimated effort:** 4–6 pd / ~1 wk

## 1. Goals

- `run_batch(bucket) = jit(vmap(stage_chain, in_axes=(0, None, None)))` — the manifold pattern (`cflibs/manifold/batch_forward.py:13,459`) generalized to the full graph; per-bucket fixed B; final partial batch padded with a spectrum-validity mask.
- Implement the bucket dispatcher over the 7 measured instrument grids (ADR-0004 §5.2 table; exact N_wl, zero wavelength padding) and the memory planner reusing `available_device_bytes` (`cflibs/radiation/host.py:116-181`) + `plan_chunks` (`:189-226`) — monotone in budget, degrades gracefully off-GPU.
- **LDM real-data validation (risk R4, early in this bead):** LDM (`cflibs/radiation/ldm.py`, n_sigma=24 at `:69`) is the batching enabler (B≈64–128 on silva2022 vs 8–16 chunked Voigt) but is default-OFF pending real-data validation (`ADR-0001-HANDOFF.md` T1-4). Run LDM-vs-PHYSICAL_DOPPLER parity + a scoreboard shadow on real datasets; record the adopt/fallback decision in the ledger. Fallback = chunked Voigt at B=8–16 — still orders of magnitude over the 0.4–5.1 s/spectrum CPU baseline.
- Memory hygiene: `donate_argnums` on the B×N_wl intensity buffers (none exists in `cflibs/` today — free win); keep the chunked forward's `jax.checkpoint` policy (`kernels.py:1055-1076`); remat the broadening stage for reverse-mode end-to-end gradients (J11 consumer).
- SLURM job assets: batched-inversion sbatch template and the 3-long-lived-worker campaign harness (queue-directory candidate pull; zero recompiles across candidates because continuous knobs are traced — ADR-0004 §5.5); AOT `jit.lower().compile()` for the campaign harness; warmup batch per bucket.
- Cluster discipline baked into every script: `--gpus-per-task=1 --ntasks=1 --cpus-per-task=4 --mem=32G`, never `--nodelist`; explicit `export JAX_COMPILATION_CACHE_DIR=$HOME/jax-cache` (the shared-path uid-skew pathology — J0 fixed the code default, scripts must still be explicit); `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.5`; scored runs set `XLA_FLAGS=--xla_gpu_deterministic_ops=true`; `==`-pin jax/jaxlib in `[cluster]` (`pyproject.toml:41`).

## 2. Memory envelope to enforce (design to 16 GB, fp64, ~40 % XLA headroom)

Per ADR-0004 §5.2: dense Voigt forbidden on large grids (7.0 GB/spectrum on silva2022); chunked scan 537 MB live/chunk → B 8–16; LDM ≈60 MB (silva2022) → B 64–128, ≈8 MB (chemcam) → B 512+; identify + solve batch the whole registry. Sliding-median and RANSAC intermediates are row-/H-chunked upstream (J1/J2). An **OOM-canary test** asserts planned bytes < 0.6× device bytes per bucket before launch.

## 3. Acceptance criteria — **Milestone M2 gate**

1. **Throughput:** ≥50 spectra/s end-to-end on one V100S, amortized over ≥1,000 spectra **including compile**; full corpus (~1,264 registered spectra) < 60 s/GPU wall. (Promotion's hard floor remains ≥10× the 0.4–5.1 s/spectrum baseline — ADR-0004 §3 C5; missing 50 but clearing 10× files a gap bead, does not block.)
2. Per-bucket memory headroom documented in a committed artifact (planned vs measured peak bytes per bucket, both LDM and chunked paths).
3. LDM decision recorded: parity numbers (existing rtol 1e-4 relaxation per ADR-0001 runbook §3 applies), shadow-board delta, adopt/fallback verdict.
4. Batch-vs-single bit-consistency: `run_batch` output for spectrum i equals `run_one` on spectrum i (same bucket, deterministic ops) — the batching must add nothing but an axis.
5. Padded partial batch correctness: validity-masked tail spectra produce no records and no NaN leakage.
6. Determinism: two identical scored GPU runs produce identical board JSON (seeded sampling + deterministic ops).
7. SLURM templates land in `scripts/jitpipe/` with warmup + private-cache exports; campaign-worker harness demonstrates ≥2 candidate evaluations with zero recompiles (cache-stats assert).

## 4. Test plan

`tests/jitpipe/test_batching.py` (batch-vs-single equality, partial-batch masks, OOM canary with a mocked budget); GPU acceptance runs on vasp nodes via sbatch (background from the parent session per CLAUDE.md watchdog rules; never inside a sub-agent). Timing artifacts committed under `docs/jitpipe/m2-throughput.md`.

## 5. Risks

R3 (memory blowup) — bounded by the canary + planner; R4 (LDM gap) — explicit early validation with a fallback that still meets the promotion floor; R6 (compile tax) — ≤7 buckets × few static variants, private cache, warmup; R10 (nondeterminism) — deterministic-ops flag + AC-6.

## 6. Dependencies / files

Depends J8. Enables J12; J10/J11 consume the batch harness opportunistically. Files: `cflibs/jitpipe/host.py` (planner/dispatcher), `scripts/jitpipe/*.sbatch`, tests, `docs/jitpipe/m2-throughput.md`. Reference untouched.
