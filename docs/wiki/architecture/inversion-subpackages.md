---
slug: inversion-subpackages
title: "Inversion Sub-Packages"
chapter: architecture
order: 30
status: stable
register: reference
summary: >
  The six-plus-one sub-package split of cflibs/inversion/ (common, preprocess, physics, identify,
  solve, runtime, and the forward_models registry), mapped to the measurement→physics→inference
  pipeline. The old flat module paths are GONE — import canonical sub-package paths only.
tags: [inversion, sub-packages, imports, shims, reference]
updated: 2026-07-02
sources:
  - CLAUDE.md
  - cflibs/inversion/__init__.py
  - docs/v4/overhaul/verified/inv-top.md
related: [module-map, data-flow, abstractions, two-backend-pipeline]
code_refs:
  - cflibs/inversion/__init__.py
  - cflibs/inversion/pipeline.py::run_pipeline
  - cflibs/inversion/candidate_prefilter.py::select_candidate_elements
---

*[Wiki home](../index.md) · [Glossary](../glossary.md) · [Bibliography](../bibliography.md)*

# Inversion Sub-Packages

`cflibs/inversion/` is organized into sub-packages that mirror the CF-LIBS
measurement→physics→inference pipeline. This is a reference page for "which sub-package owns this
stage" and, critically, the rule for how to import from it. This page describes the **reference
(NumPy) backend**; the parallel JAX backend is [two-backend-pipeline](two-backend-pipeline.md).

## The seven sub-packages

| Sub-package | Role | Key symbols |
|-------------|------|-------------|
| `common/` | Pure data structures shared across the pipeline; the PCA pipeline. No heavy imports at module load. | `LineObservation`, `BoltzmannFitResult`, `FitMethod`, `ElementIdentification`, `IdentifiedLine` |
| `preprocess/` | Signal processing: baseline removal (SNIP/ALS), noise estimation, deconvolution, wavelength calibration, outlier detection (SAM/MAD). | `estimate_baseline`, `estimate_noise`, `detect_peaks`, wavelength-calibration segmented solver |
| `physics/` | Saha–Boltzmann plane physics: Boltzmann fitting, closure (standard/matrix/oxide/ILR), CDSB self-absorption, Stark broadening, line selection, quality metrics, uncertainty propagation. | `BoltzmannPlotFitter`, `ClosureEquation`, `QualityAssessor`, `LineSelector` |
| `identify/` | Element identification: ALIAS, comb, correlation, spectral NNLS, hybrid (NNLS+ALIAS), BIC model selection, line detection (transition matching). | `ALIASIdentifier`, `CombIdentifier`, `CorrelationIdentifier`, `detect_line_observations` |
| `solve/` | Plasma-parameter inference: iterative CF-LIBS loop, closed-form ILR solver, Bayesian (NUTS + dynesty), joint L-BFGS-B, manifold coarse-to-fine. | `IterativeCFLIBSSolver`, `ClosedFormILRSolver`, `CFLIBSResult` |
| `solve/bayesian/` | The NumPyro/dynesty sub-sub-package. | `BayesianForwardModel`, samplers, likelihood, `TwoZoneBayesianForwardModel` |
| `runtime/` | Real-time: DAQ streaming, temporal gate optimization, hardware interface. | `FastAnalyzer`, streaming ingest |
| `forward_models/` | String-keyed forward-model registry (thin wrappers over `radiation/kernels.py`). | `get_forward_model`, `FORWARD_MODELS` |

Two members of `cflibs/inversion/` are **not** sub-packages and **not** shims — they are real
top-level modules:

- `cflibs/inversion/pipeline.py` — `run_pipeline`, the reference orchestrator.
- `cflibs/inversion/candidate_prefilter.py` — `select_candidate_elements`, the mandatory NNLS
  top-K prefilter for Bayesian MCMC (full-element MCMC is intractable). See
  [abstractions](abstractions.md#select-candidate-elements).

## Lazy public API

`cflibs/inversion/__init__.py` exposes the public inversion API **lazily**: symbols load only when
accessed, so beginner CLI workflows stay fast and quiet and optional stacks (JAX, NumPyro, dynesty,
`uncertainties`) are not imported at package import time. Capability flags — `HAS_HYBRID`,
`HAS_JOINT_OPTIMIZER`, `HAS_BAYESIAN`, `HAS_NESTED`, `HAS_UNCERTAINTIES` — are computed with
`importlib.util.find_spec`, i.e. they report availability **without importing** the optional module.

## The flat-path shims are GONE — import canonical paths only

> [!CAUTION] DO-NOT: The old flat module paths (`cflibs.inversion.solver`,
> `cflibs.inversion.line_detection`, `cflibs.inversion.boltzmann`, …) **no longer exist**. PR #210
> removed the first batch; the 2026-06 cleanliness sweep removed the remaining 18 shims. There is no
> back-compat shim to import through. A prior version of this guidance claimed "flat-path import shims
> still work" — that is a **contradicted claim** and is corrected here: it is false.

Always import from the canonical sub-package path:

```python
# WRONG — these modules were deleted:
# from cflibs.inversion.solver import IterativeCFLIBSSolver
# from cflibs.inversion.line_detection import detect_line_observations

# RIGHT — canonical sub-package paths:
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver
from cflibs.inversion.identify.line_detection import detect_line_observations
from cflibs.inversion.common.data_structures import LineObservation
```

When searching for a symbol, use the Serena `find_symbol` / `find_referencing_symbols` ladder rather
than grepping for the old flat path — the LSP follows the canonical location and will not surface a
dead shim.

## Target refactor: a `domain/` leaf for the shared data types

`CFLIBSResult` currently lives in `solve/iterative.py` but is imported by `io/`, `jitpipe/`, and
`closed_form.py`; `LineObservation` is imported through a `physics/boltzmann` shim by ~11 modules.
The target architecture moves these pure data types into a new `cflibs/domain/` leaf package so the
result contract can be imported without pulling in solver internals. `inversion/common/` then
re-exports them for back-compat. Details on [target-architecture](target-architecture.md#domain-leaf).

## See also

- [Module map](module-map.md) — the full package inventory.
- [Data flow](data-flow.md) — how these sub-packages chain at runtime.
- [Two-backend pipeline](two-backend-pipeline.md) — the JAX re-implementation of this flow.
- [Abstractions](abstractions.md) — the ABCs the `solve/` strategies implement.
