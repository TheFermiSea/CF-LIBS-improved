---
slug: module-map
title: "Module Map: Every Package and Its Role"
chapter: architecture
order: 10
status: stable
register: reference
summary: >
  Complete package inventory of cflibs/ — 20 top-level Python packages, the 7 inversion
  sub-packages, the inversion/solve/bayesian sub-sub-package, and the Rust core. Includes the
  six packages the old CLAUDE.md module map omits: bandit, hardware, observability,
  parameter_sweep_server, visualization, and inversion/forward_models.
tags: [module-map, packages, reference, jitpipe, rust, bandit, observability]
updated: 2026-07-02
sources:
  - CLAUDE.md
  - docs/v4/overhaul/ARCHITECTURE.md
  - cflibs/bandit/__init__.py
  - cflibs/hardware/__init__.py
  - cflibs/observability/__init__.py
  - cflibs/parameter_sweep_server/__init__.py
  - cflibs/visualization/__init__.py
  - cflibs/inversion/forward_models/__init__.py
code_refs:
  - cflibs/jitpipe/__init__.py
  - cflibs/inversion/forward_models/__init__.py::get_forward_model
  - cflibs/parameter_sweep_server/__init__.py::serve
related: [inversion-subpackages, two-backend-pipeline, data-flow, target-architecture]
---

*[Wiki home](../index.md) · [Glossary](../glossary.md) · [Bibliography](../bibliography.md)*

# Module Map: Every Package and Its Role

This is the "where does X live" reference. CF-LIBS ships **20 top-level Python packages** under
`cflibs/`, the **7 sub-packages** of `cflibs/inversion/`, the `inversion/solve/bayesian`
sub-sub-package, and a **Rust core** at `native/cflibs-core/` — 29 importable packages in all. The
legacy CLAUDE.md table lists only the "headline" fourteen; this page is the complete inventory, and
it explicitly names the six the old map omits.

## Top-level packages (the shared library + both backends)

| Package | Role |
|---------|------|
| `cflibs/core/` | Constants (SI + plasma units), config loading/validation (YAML), JAX runtime detection, logging, caching. The single-source-of-truth home for physical constants (§ [target-architecture](target-architecture.md)). |
| `cflibs/atomic/` | `AtomicDatabase` (SQLite: lines, energy_levels, species_physics, partition_functions), the `AtomicDataSource` ABC, connection pooling, query caching. |
| `cflibs/plasma/` | Plasma state (`SingleZoneLTEPlasma`, `TwoRegionPlasma`), `SahaBoltzmannSolver`, partition functions, LTE validator (McWhirter + Cristoforetti [@cristoforetti2010]). |
| `cflibs/radiation/` | `SpectrumModel` forward-model orchestrator, canonical line emissivity, profile broadening (Gaussian/Doppler/Voigt/resolving-power), Stark, and the JAX-jittable `kernels.py` used by both backends. |
| `cflibs/instrument/` | Instrument models (fixed FWHM or resolving-power mode), response curves, JIT-compiled broadening kernels. |
| `cflibs/inversion/` | The reference (NumPy) inversion backend — 7 physics-aligned sub-packages (see [inversion-subpackages](inversion-subpackages.md)). |
| `cflibs/jitpipe/` | The **second inversion backend**: fixed-shape, end-to-end `jit`/`vmap`-able re-implementation for batched/GPU inference (ADR-0004/0005). Requires JAX; no NumPy fallback. See [two-backend-pipeline](two-backend-pipeline.md). |
| `cflibs/manifold/` | JAX-accelerated batch spectrum generation over a (T, n_e, composition) grid; HDF5/Zarr storage; loader; FAISS nearest-neighbor lookup. |
| `cflibs/benchmark/` | Unified benchmark harness, synthetic corpus generation, composition metrics (Aitchison/ILR), dataset adapters. |
| `cflibs/validation/` | Round-trip validation (`GoldenSpectrum`, `RoundTripValidator`), NIST parity checks. |
| `cflibs/io/` | Spectrum I/O, config loading, exporters (CSV, JSON, HDF5) with unit labels + basis annotation. |
| `cflibs/cli/` | `main.py` — argparse CLI: `forward`, `invert`, `analyze`, `bayesian`, `batch`, `generate-db`, `generate-manifold`, plus the internal `scoreboard`. |
| `cflibs/pds/` | ChemCam/SuperCam planetary data system interface and validation corpus. |
| `cflibs/hpc/` | Cluster utilities, distributed MCMC, SLURM integration. |
| `cflibs/evolution/` | Hierarchical-ES driver config + the physics-only AST blocklist scanner for LLM-driven algorithm optimization. Optimization-process tooling only. See [physics-only-constraint](physics-only-constraint.md). |

## The six packages the old module map omits

These are real, importable, shipped packages that never made it into the legacy CLAUDE.md table.
They are documented here so the code-map is complete.

| Package | Role | Entry point |
|---------|------|-------------|
| `cflibs/bandit/` | Multi-armed bandit allocators for parameter sweeps. Thompson-sampling allocator used by `scripts/parameter_sweep.py --bandit N` to spend a sweep budget on the most promising configs. | `ThompsonAllocator` |
| `cflibs/hardware/` | Hardware-interface ABCs and placeholders for spectrographs/detectors, lasers, motion stages, and powder-flow regulators. Designed to integrate custom GUIs and real-time DED control. | `HardwareComponent`, `SpectrographInterface`, `FlowRegulatorInterface` |
| `cflibs/observability/` | Observability utilities for benchmark pipelines (structured run metrics/telemetry). | — |
| `cflibs/parameter_sweep_server/` | A long-lived asyncio TCP daemon that loads the atomic DB + datasets + `UnifiedBenchmarkRunner` **once**, then serves JSON benchmark configs over a length-prefixed protocol. Turns per-iteration cost from "spin up a fresh interpreter + reload 2 GB + recompile JAX" into "send JSON, get JSON". | `serve`, `serve_sync`, `client` |
| `cflibs/visualization/` | Jupyter-compatible interactive widgets (spectra, Boltzmann plots, Bayesian posteriors, quality metrics). Optional deps: `ipywidgets`, `plotly`. | `SpectrumViewer`, `BoltzmannPlotWidget` |
| `cflibs/inversion/forward_models/` | A string-keyed **forward-model registry** (ADR-0001 T1-6): select a kernel by name (`"single_zone_lte"`, `"hermann_two_region"`, `"lte_with_self_absorption"`) instead of importing a class. The canonical kernel is `cflibs.radiation.kernels.forward_model`; the registry entries are thin wrappers, never physics duplication. | `get_forward_model`, `FORWARD_MODELS` |

The `hermann_two_region` registry entry realizes the uniform-LTE two-region idea of Hermann et al.
[@hermann2017] at the kernel level; full two-zone Bayesian inference lives in
`inversion/solve/bayesian/forward.py`.

## Inversion sub-packages (the reference backend, expanded)

`cflibs/inversion/` is not a flat module — it is 7 sub-packages. Full detail on
[inversion-subpackages](inversion-subpackages.md); the one-line version:

| Sub-package | Role |
|-------------|------|
| `common/` | Pure data structures (`LineObservation`, `BoltzmannFitResult`, `FitMethod`, `ElementIdentification`), PCA pipeline. |
| `preprocess/` | Baseline removal, noise estimation, deconvolution, wavelength calibration, outlier detection. |
| `physics/` | Boltzmann fitting, closure (standard/matrix/oxide/ILR), CDSB self-absorption, Stark, line selection, quality metrics, uncertainty. |
| `identify/` | ALIAS, comb, correlation, spectral NNLS, hybrid, BIC model selection, line detection. |
| `solve/` | Iterative CF-LIBS loop, closed-form ILR solver, Bayesian (NUTS + dynesty), joint L-BFGS-B, manifold coarse-to-fine. |
| `solve/bayesian/` | The NumPyro/dynesty sub-sub-package: `BayesianForwardModel`, samplers, likelihood, two-zone forward. |
| `runtime/` | Real-time DAQ streaming, temporal gate optimization, hardware interface. |

`cflibs/inversion/pipeline.py` and `cflibs/inversion/candidate_prefilter.py` are **real top-level
modules**, not shims — `run_pipeline` is the reference orchestrator and `select_candidate_elements`
is the mandatory NNLS prefilter for Bayesian MCMC.

## The Rust core

| Crate | Role |
|-------|------|
| `native/cflibs-core/` | Rust computational core (PyO3): comb matching, partition functions. Serves the **reference pipeline only** — PyO3 callbacks inside `jit` kill fusion and vmap, so `jitpipe` uses pure-JAX ports of the same algorithms (ADR-0004 D7). Serena's Pyright does not cover Rust; edit these crates with grep/`cargo`. |

## Package dependency direction (the layering rule)

The intended dependency direction is a DAG rooted at `core/`:

```
core  ←  atomic  ←  plasma  ←  radiation  ←  instrument
                        ↑            ↑
                        └── inversion (reference)  ─┐
                        └── jitpipe   (JAX backend) ─┤→ cli / benchmark / validation / hpc
manifold ← radiation                                │
evolution → (calls the pipeline as an evaluator; nothing imports evolution)
```

Two rules make this real: `plasma/` must **not** import from `inversion/` (currently clean), and
**nothing outside `jitpipe/` imports `jitpipe/`** (ADR-0004 D3, enforced by an import-hygiene test).
The target refactor adds a `domain/` leaf beneath `core/` for the shared result/observation types —
see [target-architecture](target-architecture.md).

## See also

- [Inversion sub-packages](inversion-subpackages.md) — the 6/7-way split in detail.
- [Two-backend pipeline](two-backend-pipeline.md) — why there are two inversion packages.
- [Data flow](data-flow.md) — how these packages chain at runtime.
- [Target architecture](target-architecture.md) — where the module boundaries are heading.
