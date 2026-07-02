---
slug: two-backend-pipeline
title: "Two Backends of One Pipeline: inversion/ vs jitpipe/"
chapter: architecture
order: 40
status: stable
register: handbook
summary: >
  The single most load-bearing architectural fact: cflibs/inversion/ (NumPy reference) and
  cflibs/jitpipe/ (fixed-shape JAX vmap) are TWO BACKENDS of ONE inversion pipeline, per ADR-0004.
  They meet only at the scoreboard dispatch and the parity tests. jitpipe requires JAX and raises a
  clear ImportError — no NumPy fallback, by design.
tags: [jitpipe, jax, vmap, adr-0004, adr-0005, backends, parity]
updated: 2026-07-02
sources:
  - docs/adr/ADR-0004-jittable-inversion-pipeline.md
  - docs/adr/ADR-0005-jitpipe-promotion.md
  - cflibs/jitpipe/__init__.py
related: [data-flow, inversion-subpackages, jax-backend-matrix, adr-index, change-taxonomy]
code_refs:
  - cflibs/jitpipe/__init__.py::run_batch
  - cflibs/jitpipe/pipeline.py::run_one
  - cflibs/jitpipe/snapshot.py::PipelineSnapshot
  - cflibs/jitpipe/params.py::PipelineParams
  - cflibs/inversion/pipeline.py::run_pipeline
---

*[Wiki home](../index.md) · [Glossary](../glossary.md) · [Bibliography](../bibliography.md)*

# Two Backends of One Pipeline: `inversion/` vs `jitpipe/`

CF-LIBS has **one inversion pipeline** and **two backends that implement it**. The reference backend
(`cflibs/inversion/`) is NumPy/SciPy/Rust with data-dependent shapes; the second backend
(`cflibs/jitpipe/`) is a fixed-shape, end-to-end `jit`/`vmap`-able JAX re-implementation of the same
stage graph. This is the single most important structural fact in the codebase, and it is governed by
[ADR-0004](../../adr/ADR-0004-jittable-inversion-pipeline.md) (build) and
[ADR-0005](../../adr/ADR-0005-jitpipe-promotion.md) (promotion).

## Why two backends exist

The reference pipeline's wall-time is **front-end-dominated**: on `bhvo2_chemcam` the stage medians
are calibration 1.55 s + detection/ID 0.89 s versus a **solve of 0.01 s** out of 2.64 s total
(ADR-0004 §1.1). During identifier sweeps on the V100S cluster, GPU utilization was 0–24 % (mean
~3 %) — Amdahl-bound on the non-JAX per-spectrum fraction — with 7,394 `jit`-cache entries
accumulated because variable per-spectrum shapes force recompiles. Incremental kernel ports cannot
fix an Amdahl-plus-shape-instability problem; a ground-up fixed-shape rewrite can. The three
motivations (ADR-0004 §1.2) are: **batched throughput**, **forward-eval scale for identification**
(the Gornushkin–Völker Monte-Carlo regime [@gornushkin2022], $10^3$–$10^4$ forward evals/spectrum),
and **differentiability** (joint gradient solve, HMC, gradient knob-tuning).

## They are the same pipeline, not a fork

The two backends realize the same measurement→physics→inference flow stage-for-stage:

| Reference (`inversion/`) | jitpipe (`jitpipe/`) | Contract |
|--------------------------|----------------------|----------|
| `preprocess/` baseline+noise | `preprocess.py` | rtol 1e-12 (SNIP/median), exact noise |
| `preprocess/` peak detect | `detect.py` | **byte-identical index lists** vs scipy |
| `preprocess/` wl calibration | `calibrate.py` | corrected axis max\|Δλ\| ≤ 0.04 nm |
| `identify/` line matching | `identify.py` | identical accepted element sets |
| `physics/` Boltzmann + SB-graph | `fit.py` | SB-graph slope rtol 1e-10 (algebraic identity) |
| `physics/` self-absorption | `selfabs.py` | τ atol 1e-6 |
| `physics/` Stark n_e | `stark.py` | per-line n_e rtol 1e-3 |
| `physics/` closure (6 modes) | `fit.py` | std/matrix/oxide rtol 1e-12 |
| `solve/iterative` | `solve.py` | joint WLS GN-step-0 ≡ SB-graph rtol 1e-10 |
| `radiation/kernels.py::forward_model` | `forward.py` (thin wrapper) | **shared** — no physics duplication |

The forward model is *literally shared*: `jitpipe/forward.py` is a thin wrapper over
`cflibs/radiation/kernels.py`, which stays the single source of truth. The re-implementation is of the
*orchestration and the discrete front-end logic*, done as fixed-shape scans with masks — never a
re-derivation of the physics.

## The one-way import rule (ADR-0004 D3)

> [!WARNING] BENCHMARK-GATED — the two backends may diverge only within documented parity tolerances;
> any change to a scoring-relevant stage in either backend is benchmark-gated (the repo has regressed
> identification F1 before, −0.041, PR #229).

- `jitpipe` **may import** `cflibs.radiation.{kernels,ldm,host}`, `cflibs.core.jax_runtime`,
  `cflibs.inversion.common` dataclasses, and the preset constants from `inversion/pipeline.py`
  (single source of preset truth — never copied).
- **Nothing outside `jitpipe` imports `jitpipe`.** Enforced by an import-hygiene test.
- The two implementations meet at exactly **two points**: the scoreboard CLI dispatch
  (`--pipeline {reference,jit}`) and the parity tests. There is deliberately **no `use_jax` flag**
  inside the reference — the kwarg-flag anti-pattern was rejected up front.

This one-way rule is what keeps the two backends from drifting into an unmaintainable fork: the
physics lives in shared library code, the reference is frozen as the parity oracle, and every
reference bug-fix is paired with a parity-test update.

## No NumPy fallback — jitpipe requires JAX by design

Unlike the rest of `cflibs/` (which degrades gracefully when JAX is absent), `jitpipe` **requires**
JAX and says so loudly. Importing the package without JAX raises a clear `ImportError` with install
guidance rather than silently degrading (ADR-0004 §5.5):

```
cflibs.jitpipe requires JAX, which is not installed. The jittable inversion pipeline
has no NumPy fallback (ADR-0004 §5.5). Install it with one of:
    uv pip install -e '.[local]'    # Apple Silicon (JAX Metal)
    uv pip install -e '.[cluster]'  # NVIDIA GPU (JAX CUDA)
    pip install 'jax[cpu]'          # CPU-only
If you only need the reference (non-jit) pipeline, use cflibs.inversion instead.
```

This is a deliberate design choice: a fixed-shape JAX pipeline that silently fell back to NumPy would
give a false sense of the batched/GPU path being exercised. Fail loud, fail early.

## Public API and the snapshot contract

`jitpipe`'s public surface is small (`cflibs/jitpipe/__init__.py`):

- `PipelineSnapshot` — one frozen, pytree-registered struct-of-arrays bundle of the **whole atomic
  DB** (~5–6 MB, ≈0.04 % of a 16 GB V100S), host-built once per process and `.npz`-cached by DB
  content hash. Per-spectrum work is then pure gather; candidate sets are element **masks** over one
  per-bucket superset snapshot, never per-spectrum rebuilds — this is what makes `vmap` over
  heterogeneous candidate sets possible without recompiles.
- `PipelineParams` — a traced pytree of **every** continuous knob (the "killer feature" for
  campaigns: zero recompiles across candidate configs because all knobs are traced leaves).
- `StaticConfig` — the hashable `jit` cache key (statics only).
- `run_one`, `run_batch` — single / batched entry points.

## Promotion status (ADR-0005)

The jit pipeline was **promoted (2026-06-19)** as the engine for **batched/GPU inference and
full-population campaign evaluation**, having cleared the M3 gate (ID F1 ≥ reference on 8/9 datasets;
composition RMSE non-regressing; ~82,000 spectra/s batched on V100S, `vmap == loop`; parity suite
green). The reference pipeline remains:

- the **default for single-spectrum CLI/interactive use** (the jit per-spectrum CPU path is
  compile-dominated), and
- the **parity oracle** (`--pipeline=reference`), bug-fixes-only, retained for ≥ 2 releases.

The jit backend is **not** the universal default — it is the *selected* engine where its measured
strength (throughput at parity accuracy) applies. Demotion criteria are restated in ADR-0005 §4: a
full-board regression past tolerance, or a red parity suite on a release SHA, reverts it to a parallel
evaluator.

## Fixed-shape discipline (the design rule)

Every jit-breaking construct in the reference (variable-length peak lists, mutable `used` sets, greedy
tie-breaks, `raise ValueError` on failure) becomes, in `jitpipe`, a fixed-shape array + validity mask
+ `uint8` reason code. Failure is signaled by a `failed` mask interpreted host-side, never an
exception. Overflow on any padded axis keeps the top-score entries and sets a mandatory truncation
flag. This is the non-negotiable rule derived from the 7,394-cache-entry / 3%-GPU finding.

## See also

- [Data flow](data-flow.md) — the pipeline these two backends implement.
- [JAX backend matrix](jax-backend-matrix.md) — the CPU/Metal/CUDA backends jitpipe runs on.
- [ADR index](adr-index.md) — ADR-0004 and ADR-0005 in context.
- [Change taxonomy](change-taxonomy.md) — why every cross-backend change is benchmark-gated.
