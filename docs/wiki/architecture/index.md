---
slug: architecture
title: "Codebase Architecture: Current & Target"
chapter: architecture
order: 0
status: stable
register: orientation
summary: >
  System-level map of the CF-LIBS codebase: 20 top-level packages plus the inversion
  sub-packages and the Rust core, two interchangeable inversion backends (NumPy reference
  vs. jitpipe JAX-vmap, ADR-0004), the physics-only constraint, and the target refactor
  (domain/ leaf, CFLIBSConfig, IPD/PF truncation invariant). This is an engineering chapter,
  not a physics chapter — physics claims link out to libs-physics and formal-spec.
tags: [architecture, module-map, jitpipe, physics-only, adr, target-architecture, data-flow]
updated: 2026-07-02
sources:
  - CLAUDE.md
  - docs/v4/overhaul/ARCHITECTURE.md
  - docs/v4/overhaul/BLUEPRINT.md
  - docs/v4/overhaul/BLUEPRINT-ADDENDUM.md
  - docs/v4/overhaul/CRITIC.md
  - docs/adr/ADR-0004-jittable-inversion-pipeline.md
  - docs/adr/ADR-0005-jitpipe-promotion.md
  - cflibs/jitpipe/__init__.py
  - cflibs/inversion/__init__.py
related: [module-map, data-flow, inversion-subpackages, two-backend-pipeline, abstractions, jax-backend-matrix, target-architecture, physics-only-constraint, change-taxonomy, adr-index]
---

*[Wiki home](../index.md) · [Glossary](../glossary.md) · [Bibliography](../bibliography.md)*

# Codebase Architecture: Current & Target

This chapter is the engineering map of CF-LIBS: what the packages are, how data flows through
them, which abstractions hold the system together, and where the architecture is deliberately
heading. It is explicitly **not** a physics chapter — every physics claim links out to
[libs-physics](../libs-physics.md) or the machine-checked [formal-spec](../formal-spec.md);
here the concern is *structure*, *contracts*, and *safe-change discipline*.

CF-LIBS is a physics-based Calibration-Free Laser-Induced Breakdown Spectroscopy library. It
implements the full pipeline: forward modeling (plasma parameters → synthetic spectrum),
inversion (measured spectrum → plasma composition), and GPU-accelerated manifold pre-computation
for fast inference. The reference algorithm is the classic Ciucci–Corsi–Palleschi calibration-free
procedure [@ciucci1999], hardened with a multi-element Saha–Boltzmann temperature fit [@aguilera2007]
and an explicit LTE-validity gate [@cristoforetti2010].

## The one-paragraph mental model

There is **one CF-LIBS pipeline** with **two interchangeable backends**. The measurement→physics→
inference flow (preprocess → identify → Boltzmann/Saha physics → closure → iterate) is realized once
in NumPy/SciPy under `cflibs/inversion/` (the reference, single-spectrum default and parity oracle)
and again, fixed-shape and end-to-end `jit`/`vmap`-able, under `cflibs/jitpipe/` (the batched/GPU
engine, ADR-0004/ADR-0005). They meet at exactly two points: the scoreboard CLI dispatch and the
parity tests. Everything else — atomic data, plasma physics, radiation forward model, instrument
model — is shared library code that both backends call.

## Chapter map

| Page | What it answers |
|------|-----------------|
| [Module map](module-map.md) | What every package is for — all 20 top-level packages, the inversion sub-packages, and the Rust core (including the six the old docs omit). |
| [Data flow](data-flow.md) | Forward, inversion, and manifold pipelines end-to-end, stage by stage. |
| [Inversion sub-packages](inversion-subpackages.md) | The six-way split of `cflibs/inversion/`; the flat-path shims are **gone** — import canonical paths only. |
| [Two-backend pipeline](two-backend-pipeline.md) | `inversion/` (NumPy) vs. `jitpipe/` (JAX-vmap) as two backends of one pipeline; ADR-0004 contract. |
| [Key abstractions](abstractions.md) | `AtomicDataSource`, `SolverStrategy`, `InstrumentModelProtocol`, `SingleZoneLTEPlasma`, `BayesianForwardModel`, `select_candidate_elements`. |
| [JAX backend matrix](jax-backend-matrix.md) | `jax-cpu` / `jax-metal` / `jax-cuda`; x64 is mandatory; graceful degradation. |
| [Target architecture](target-architecture.md) | `domain/` leaf, single-source-of-truth constants, `CFLIBSConfig`, IPD/PF truncation invariant, air↔vacuum convention, physical `NoiseModel`. |
| [Physics-only constraint](physics-only-constraint.md) | The banned-API list and its two-point enforcement (Ruff TID251 + AST scanner); enforcement history. |
| [Change taxonomy](change-taxonomy.md) | SAFE-NOW / BENCHMARK-GATED / DESIGN-DECISION, and the FALSE-exclusion list of things **not** to "fix". |
| [ADR index](adr-index.md) | Annotated index into `docs/adr/`, which stays canonical. |

## Guiding principles (the invariants this chapter defends)

These four principles govern every architectural decision and are restated on the pages that
implement them (source: `docs/v4/overhaul/ARCHITECTURE.md` §0):

1. **Physics-only is inviolable.** No ML library in shipped `cflibs/`; the evolution tooling is the
   sole exception, and even it may not import ML. See [physics-only-constraint](physics-only-constraint.md).
2. **Mission priority governs ordering:** ACCURACY > PRECISION > RELIABILITY > latency. Every
   performance-only change is deferred behind correctness in the same area.
3. **Single source of truth for every physics quantity.** One canonical function/constant for
   emissivity, IPD, Voigt FWHM, the Saha constant, molar-mass conversion, and air↔vacuum wavelength.
4. **JAX optional, graceful degradation** — with one deliberate exception: `jitpipe` *requires* JAX
   and raises a clear `ImportError` rather than silently falling back to NumPy (ADR-0004 §5.5).

## What this chapter deliberately excludes

- The physics of Saha–Boltzmann, closure, and self-absorption → [libs-physics](../libs-physics.md)
  and [cf-libs-family](../cf-libs-family.md).
- Machine-checked statements of the forward/inverse relations → [formal-spec](../formal-spec.md).
- The falsification ledger (what we tried and it failed) → [error-budget-and-falsification](../error-budget-and-falsification.md).
- Benchmark numbers and reliability workflows → [benchmarks-reliability-workflows](../benchmarks-reliability-workflows.md).

> [!IMPORTANT] RESET LINE — the atomic DB was rebuilt from the ASD59 gold-standard dump (203k lines,
> 62k levels). Any composition/F1/RMSE figure predating 2026-07-02 is pre-reset; those pages carry the
> PRE-RESET banner. This architecture chapter cites *mechanisms and structure*, not magnitudes, so it is
> reset-neutral except where explicitly flagged.

## See also

- [Module map](module-map.md) — start here for "where does X live".
- [Two-backend pipeline](two-backend-pipeline.md) — the single most load-bearing architectural fact.
- [Orientation chapter](../orientation.md) — if you are new to the whole project.
