---
slug: data-flow
title: "Data Flow: Forward, Inversion, and Manifold Pipelines"
chapter: architecture
order: 20
status: stable
register: handbook
summary: >
  End-to-end stage graphs for the three CF-LIBS pipelines: forward (plasma state → synthetic
  spectrum), inversion (measured spectrum → composition), and manifold (batch pre-computation →
  fast nearest-neighbor inference). Names the exact module boundary each stage crosses.
tags: [data-flow, forward-model, inversion, manifold, pipeline, boltzmann-plot]
updated: 2026-07-02
sources:
  - CLAUDE.md
  - docs/adr/ADR-0004-jittable-inversion-pipeline.md
  - cflibs/inversion/pipeline.py
related: [module-map, two-backend-pipeline, inversion-subpackages]
code_refs:
  - cflibs/inversion/pipeline.py::run_pipeline
  - cflibs/radiation/kernels.py::forward_model
  - cflibs/radiation/spectrum_model.py::SpectrumModel
  - cflibs/manifold/generator.py
lean_refs:
  - CflibsFormal/Boltzmann.lean#boltzmann_plot
---

*[Wiki home](../index.md) · [Glossary](../glossary.md) · [Bibliography](../bibliography.md)*

# Data Flow: Forward, Inversion, and Manifold Pipelines

Three pipelines move data through CF-LIBS: **forward** turns a plasma state into a synthetic
spectrum, **inversion** turns a measured spectrum into a composition, and **manifold** pre-computes
a grid of forward spectra so inference can be a lookup. This page is the stage-by-stage map; the
physics of each stage lives in [libs-physics](../libs-physics.md), and the two inversion
*backends* that realize the same inversion flow are covered in
[two-backend-pipeline](two-backend-pipeline.md).

Wavelengths in every stage below are **air** wavelengths (the DB stores air, per NIST/ASD
provenance); a single conversion utility handles vacuum detectors — see
[target-architecture](target-architecture.md#air-vacuum).

## Forward model: plasma state → synthetic spectrum

```
PlasmaState(T, n_e, composition)
   │  cflibs/plasma/state.py  (SingleZoneLTEPlasma)
   ▼
SahaBoltzmannSolver                         cflibs/plasma/saha_boltzmann.py
   │  Saha equation → ion-stage ratios (with IPD Δχ, Stewart–Pyatt/Debye–Hückel [@stewart1966])
   │  Boltzmann distribution → level populations n_k
   ▼
line emissivity   ε_ki = (hc / 4π λ) · A_ki · n_k        cflibs/radiation/emissivity.py
   │
   ▼
line broadening   Gaussian/Doppler/Voigt + Stark          cflibs/radiation/profiles.py
   │
   ▼
instrument convolution (fixed FWHM or R-mode)             cflibs/instrument/
   ▼
synthetic spectrum  I(λ)
```

The partition function $U_s(T)$ that normalizes the Boltzmann factor is evaluated by
`cflibs/plasma/partition.py` (polynomial $\ln U = \sum_n a_n (\ln T)^n$ [@irwin1981], validated
against Barklem & Collet [@barklem2016], or direct summation over IPD-truncated levels). The
**partition-function truncation must share the Saha IPD** — this invariant is defended on
[target-architecture](target-architecture.md#pf-ipd-invariant).

`SpectrumModel` (`cflibs/radiation/spectrum_model.py`) is the NumPy orchestrator; the JAX path uses
the single unified kernel `forward_model` in `cflibs/radiation/kernels.py`, which is *the* source of
truth for the emissivity formula. Both call the same physics — only the vectorization differs.

## Inversion: measured spectrum → composition

```
measured spectrum  I(λ)
   │  PREPROCESS   cflibs/inversion/preprocess/
   ▼  baseline removal · noise estimation · response correction · wavelength calibration
peaks
   │  IDENTIFY     cflibs/inversion/identify/
   ▼  ALIAS / comb / correlation / NNLS-hybrid line identification → LineObservations
identified lines
   │  PHYSICS + SOLVE   cflibs/inversion/physics/ + cflibs/inversion/solve/
   ▼
   ┌─────────────────── IterativeCFLIBSSolver ───────────────────┐
   │ Boltzmann plot:  y = ln(I·λ / (g_k A_ki))  vs  x = E_k      │  lean:CflibsFormal/Boltzmann.lean#boltzmann_plot
   │   slope = −1/(k_B T)  →  temperature T                       │
   │ Saha correction: map ionic lines onto the neutral plane      │
   │ multi-element common-slope fit  [@aguilera2007]              │
   │ closure equation  Σ C_s = 1                                  │
   │ charge / pressure balance  →  update n_e                     │
   │ iterate until convergence                                    │
   └──────────────────────────────────────────────────────────────┘
   ▼
CFLIBSResult(T, n_e, concentrations, uncertainties)
```

The canonical Boltzmann ordinate is **always** $y = \ln(I_{ki}\,\lambda / (g_k A_{ki}))$ with slope
$-1/(k_B T)$; the $\lambda$ factor is load-bearing whenever wavelength varies across the fit and must
never be silently dropped (see [notation](../formal-spec.md)). The reference orchestrator is
`run_pipeline` (`cflibs/inversion/pipeline.py`); its measured wall-time is front-end-dominated
(calibration + identification ≫ solve), which is the motivation for the second backend
([two-backend-pipeline](two-backend-pipeline.md)).

Three alternative solvers plug into the same `solve/` stage behind the `SolverStrategy` abstraction
([abstractions](abstractions.md)): the closed-form ILR solver, the joint L-BFGS-B optimizer, and the
Bayesian NUTS sampler. All consume the same identified `LineObservation`s and emit the same
`CFLIBSResult`.

## Manifold: batch pre-computation → fast lookup

```
parameter grid  {(T_i, n_e,i, C_i)}
   │  cflibs/manifold/generator.py   (jit + vmap over the grid)
   ▼  batched forward_model  →  {I_i(λ)}
stored manifold  (HDF5 or Zarr)      cflibs/manifold/  storage backends
   │  build FAISS index
   ▼
nearest-neighbor inference: measured I(λ) → closest grid point → (T, n_e, C) seed
```

The manifold is a **fast initializer / coarse-to-fine seed**, not a foundation the accuracy rests on
— the "manifold-as-foundation" framing is a contradicted claim and is not carried forward (see
[change-taxonomy](change-taxonomy.md)). The Monte-Carlo full-spectrum forward-fit paradigm that
motivates dense manifolds is Gornushkin & Völker [@gornushkin2022]; in this codebase the manifold
feeds the `solve/manifold` coarse-to-fine path, which then hands off to a physics solver.

## Where the three pipelines share code

| Shared component | Used by |
|------------------|---------|
| `radiation/kernels.py::forward_model` | forward, manifold, and (as the differentiable residual) the joint/Bayesian solvers |
| `plasma/saha_boltzmann.py` | forward model **and** the inversion Saha correction |
| `atomic/` DB + snapshot | every stage that needs $A_{ki}$, $E_k$, $g_k$, $U_s(T)$ |
| `core/constants.py` | all of the above (single-source-of-truth constants) |

This sharing is why `plasma/` must not import `inversion/`: the closed-form forward physics is a leaf
that both directions depend on. Breaking that rule would create an import cycle between the forward
model and the inverter.

## See also

- [Two-backend pipeline](two-backend-pipeline.md) — the NumPy vs. JAX realizations of the inversion flow.
- [Inversion sub-packages](inversion-subpackages.md) — the packages each inversion stage lives in.
- [libs-physics](../libs-physics.md) — the physics of each stage.
