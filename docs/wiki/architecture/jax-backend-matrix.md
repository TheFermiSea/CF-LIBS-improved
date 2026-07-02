---
slug: jax-backend-matrix
title: "JAX Backend Matrix"
chapter: architecture
order: 60
status: stable
register: reference
summary: >
  The three JAX backends CF-LIBS runs on — jax-cpu, jax-metal (Apple Silicon), jax-cuda (NVIDIA) —
  their capabilities and gotchas. float64 (x64) is MANDATORY for the plasma physics; jax-metal lacks
  it. JAX is optional throughout the reference path (graceful degradation); jitpipe is the one
  package that requires it.
tags: [jax, x64, float64, metal, cuda, backend, graceful-degradation]
updated: 2026-07-02
sources:
  - CLAUDE.md
  - docs/adr/ADR-0004-jittable-inversion-pipeline.md
  - cflibs/core/jax_runtime.py
related: [two-backend-pipeline, module-map, physics-only-constraint]
code_refs:
  - cflibs/core/jax_runtime.py
  - cflibs/hpc/gpu_config.py
---

*[Wiki home](../index.md) · [Glossary](../glossary.md) · [Bibliography](../bibliography.md)*

# JAX Backend Matrix

JAX is optional throughout the reference CF-LIBS path — code degrades gracefully to NumPy when JAX is
absent — with one exception: `cflibs/jitpipe/` *requires* JAX ([two-backend-pipeline](two-backend-pipeline.md)).
This page is the reference for the three backends, their capabilities, and the one non-negotiable
requirement: **float64**.

## The three backends

| Backend | Hardware | float64 (x64) | Notes |
|---------|----------|---------------|-------|
| `jax-cpu` | any CPU | ✅ yes | The parity/CI backend. `conftest.py` forces CPU + `jax_enable_x64=True`. Correct but not fast; the reference single-spectrum default. |
| `jax-metal` | Apple Silicon GPU | ❌ **no** (also no complex) | Development convenience on Mac; **cannot run the scored plasma physics** because it lacks float64. Use for shape/plumbing work, not numerics. |
| `jax-cuda` | NVIDIA GPU (V100S cluster) | ✅ yes | The production batched/GPU backend. `hpc/gpu_config.py` declares x64 required for cluster GPU. ~82,000 spectra/s batched for the jit core (ADR-0005). |

Backend detection lives in `cflibs/core/jax_runtime.py`. The `JaxMemoryPolicy` defaults
`allow_32bit=False` — the codebase votes x64 by default.

## Why float64 is mandatory {#x64-mandatory}

> [!IMPORTANT] x64 is not a preference — it is a correctness requirement for the plasma physics.

The Saha three-stage population balance carries exponents that reach ~50; the partition-function sums
run over IPD-truncated level ladders; the Boltzmann/common-slope weighted-least-squares needs rtol
1e-5 that is simply unreachable in fp32. A historical log-basis partition bug produced an
**18-orders-of-magnitude** error — the kind of failure fp32's ~7 significant digits invites. ADR-0004
§5.3 pins the non-negotiable-fp64 list:

- Saha three-stage populations + IPD ([target-architecture](target-architecture.md#pf-ipd-invariant))
- partition functions (polynomial [@irwin1981] + direct sums over ≤676 levels)
- Boltzmann / common-slope WLS and the solve
- Stark-width $n_e$ [@gigosos2003]
- LDM log-$\sigma$ interpolation weights, the preprocessing LLS round-trip, and the ALS Gram solve

**fp32-tolerable (opt-in, policy-gated) only:** the broadening/profile matrix *values* (compute
populations in fp64, cast at the broadening input boundary, accumulate bin sums back in fp64) and
identify-stage template scoring (thresholded at ~1e-2). Never fp32 in the Stark-width Voigt path.

This is why `jax-metal` is a dead end for scored work: no float64 means no trustworthy Saha exponents.

## Graceful degradation (the reference path)

Every physics computation in the reference path has a working NumPy fallback. JAX is an *accelerator*,
not a requirement (ARCHITECTURE.md §0 principle 4). The capability flags in
`cflibs/inversion/__init__.py` (`HAS_BAYESIAN`, `HAS_JOINT_OPTIMIZER`, …) are computed with
`find_spec` so that importing `cflibs.inversion` never forces a JAX import. A machine with no JAX runs
the full reference inversion; it simply cannot use the jit backend, the manifold generator, or the
NumPyro sampler.

JAX paths must be **parity-tested against NumPy** (ARCHITECTURE.md §0): radiation forward, manifold
generation, and the inv-solve lax path all carry parity fixtures. The tiered tolerance contracts
(kernel rtol 1e-5 down to 1e-12 for pure algebra; exact for discrete decisions) are in ADR-0004 §5.4.

## Determinism and the compile cache

Scored runs set `XLA_FLAGS=--xla_gpu_deterministic_ops=true`; all stochastic stages are keyed by a
counter-based `jax.random` seed derived from a per-spectrum hash. CPU↔GPU agreement is
**tolerance-bounded** (fp64 reduction order, ≲1e-12 relative), never bit-identical.

> [!CAUTION] DO-NOT: never call `jax.clear_caches()` in a test — it is a process-global wipe that
> evicts every other test's compiled graphs. Per-jitted-function cache-size assertions need no global
> clear (a freshly-defined `@jax.jit` starts at 0). The NFS-shared compile cache developed uid skew
> and hung cluster jobs; the standing rule is the user-private `~/.cache/cflibs/jax` path with
> `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.5`, and jax/jaxlib `==`-pinned in the `[cluster]`
> extra so cache keys survive node-image drift.

## See also

- [Two-backend pipeline](two-backend-pipeline.md) — the JAX backend that consumes all this.
- [Target architecture](target-architecture.md) — the IPD/PF invariant that demands fp64.
- [Module map](module-map.md) — `core/jax_runtime.py` and `hpc/gpu_config.py`.
