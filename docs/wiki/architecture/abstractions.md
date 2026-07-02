---
slug: abstractions
title: "Key Abstractions"
chapter: architecture
order: 50
status: stable
register: reference
summary: >
  The six load-bearing abstractions that hold CF-LIBS together: AtomicDataSource (pluggable atomic
  backend), SolverStrategy (pluggable solver), InstrumentModelProtocol (structural typing),
  SingleZoneLTEPlasma (core plasma state + composition conversions), BayesianForwardModel
  (JAX MCMC forward), and select_candidate_elements (mandatory NNLS prefilter).
tags: [abstractions, abc, protocol, solver-strategy, plasma-state, bayesian]
updated: 2026-07-02
sources:
  - CLAUDE.md
  - docs/v4/overhaul/ARCHITECTURE.md
  - docs/adr/ADR-0002-no-generic-predictor-factory.md
related: [module-map, inversion-subpackages, two-backend-pipeline, target-architecture]
code_refs:
  - cflibs/atomic/database.py::AtomicDataSource
  - cflibs/plasma/state.py::SingleZoneLTEPlasma
  - cflibs/inversion/solve/bayesian/forward.py::BayesianForwardModel
  - cflibs/inversion/candidate_prefilter.py::select_candidate_elements
---

*[Wiki home](../index.md) · [Glossary](../glossary.md) · [Bibliography](../bibliography.md)*

# Key Abstractions

CF-LIBS is a Python codebase with heavy ABCs, protocols, and pluggable strategies. Six abstractions
carry most of the structural weight. This page is the reference for what each one is *for* and where
the seam is intended to flex.

## `AtomicDataSource` (ABC)

**Where:** `cflibs/atomic/database.py`. **Role:** pluggable backend for atomic-data queries.

The forward model and inverter need $A_{ki}$, $E_k$, $g_k$, and $U_s(T)$ for arbitrary species. The
`AtomicDataSource` ABC abstracts *where* that data comes from: the shipped implementation is
`AtomicDatabase` (SQLite over `ASD_da/libs_production.db`), but the seam allows a NIST-API or HDF5
backend without touching the physics. Connection pooling and query caching live behind this interface.
The target architecture keeps the ABC in `atomic/` and moves the `AtomicSnapshot` pytree (the
device-resident struct-of-arrays form) out of `core/jax_runtime.py` into `atomic/snapshot.py` so that
`core/` no longer carries a domain type ([target-architecture](target-architecture.md#domain-leaf)).

## `SolverStrategy` (ABC)

**Where:** `cflibs/core/abc.py` (type), implementations in `cflibs/inversion/solve/`. **Role:**
pluggable plasma solvers.

Every solver — iterative Saha–Boltzmann, closed-form ILR, joint L-BFGS-B, Bayesian NUTS,
manifold coarse-to-fine — implements the same `SolverStrategy` contract: consume identified
`LineObservation`s, emit a `CFLIBSResult(T, n_e, concentrations, uncertainties)`. This is what lets
`RoundTripValidator` gate regressions across solvers (the target adds a `solver_factory` injection
seam so the validator is not hardwired to the iterative solver).

> [!NOTE] `SolverStrategy` is a *strategy* seam, not a *factory* seam. ADR-0002 explicitly rejects a
> generic `_make_predictor` factory for the standalone non-ALIAS identifiers: "symmetry is not depth".
> The abstraction exists where it earns leverage (interchangeable solvers behind one result contract),
> not where it would only add visual symmetry. See [ADR index](adr-index.md).

## `InstrumentModelProtocol` (Protocol)

**Where:** `cflibs/core/abc.py`. **Role:** structural typing for instrument models.

Instrument models come in two shapes — fixed-FWHM and resolving-power (per-wavelength
$\sigma = \lambda / R / 2\sqrt{2\ln 2}$) — plus JAX and NumPy variants. Rather than a class hierarchy,
CF-LIBS uses a `Protocol`: anything that structurally provides the instrument-response surface
satisfies the type. This keeps the JAX `InstrumentModelJax` and the NumPy `InstrumentModel`
interchangeable at call sites without a shared base class forcing import-order coupling. The type is
imported under `TYPE_CHECKING` only, so `core/` does not gain a runtime dependency on `instrument/`.

## `SingleZoneLTEPlasma`

**Where:** `cflibs/plasma/state.py`. **Role:** the core plasma-state dataclass and the home of the
composition-conversion helpers.

`SingleZoneLTEPlasma` holds $(T, n_e, \text{composition})$ and the conversion machinery between the
three composition representations that the pipeline juggles: **mass fractions ↔ number/mole fractions
↔ number densities**. Getting these conversions right and applying them consistently is the single
highest-impact accuracy lever in the whole system — the solvers emit number fractions while every
truth corpus stores mass fractions, and a missing conversion silently compares the wrong bases
([change-taxonomy](change-taxonomy.md), M3). The sibling `TwoRegionPlasma` is **experimental /
unvalidated** and warns on construction; its magic-number zone constants have no literature basis
([target-architecture](target-architecture.md)).

## `BayesianForwardModel`

**Where:** `cflibs/inversion/solve/bayesian/forward.py`. **Role:** the JAX-compiled forward model for
MCMC.

`BayesianForwardModel` wraps the shared `radiation/kernels.py::forward_model` into a NumPyro-compatible
differentiable likelihood. It supports resolving-power mode and a Chebyshev baseline, and it is what
NUTS samples over. The sampler *loop* stays host-side (the ExoJAX pattern: NUTS consumes the jit
likelihood, it does not wrap the host driver). The physics-forward Bayesian route places calibrated
posteriors on $(T, n_e, \text{composition})$ rather than point estimates — a physics-only alternative
to ML surrogates, staying inside the banned-API constraint
([physics-only-constraint](physics-only-constraint.md)). `TwoZoneBayesianForwardModel` extends it for
the Hermann-style two-region case [@hermann2017].

## `select_candidate_elements`

**Where:** `cflibs/inversion/candidate_prefilter.py`. **Role:** the mandatory NNLS top-K prefilter for
Bayesian MCMC. {#select-candidate-elements}

Full-element MCMC is intractable — the posterior is exponential in $T$ and the composition simplex is
high-dimensional. `select_candidate_elements` runs a non-negative-least-squares spectral decomposition
to pick the top-K candidate elements before MCMC ever starts. It is **not optional**: the Bayesian and
joint solvers depend on it to make the problem finite. It is a real top-level module, not a shim.

## How the abstractions compose

```
AtomicDataSource ─→ SahaBoltzmannSolver ─→ forward_model (kernels.py)
                                                 ▲
InstrumentModelProtocol ─────────────────────────┘
                                                 │
SingleZoneLTEPlasma (state + conversions) ───────┤
                                                 ▼
select_candidate_elements ─→ SolverStrategy ∈ {iterative, closed-form, joint, Bayesian}
                                   │  (Bayesian uses BayesianForwardModel)
                                   ▼
                              CFLIBSResult
```

The through-line: **one atomic-data seam feeds one forward model; one result contract is emitted by
interchangeable solvers.** Everything else is orchestration.

## See also

- [Module map](module-map.md) — where these types live.
- [Inversion sub-packages](inversion-subpackages.md) — the solvers behind `SolverStrategy`.
- [ADR index](adr-index.md) — ADR-0002 on when *not* to add a factory abstraction.
- [Target architecture](target-architecture.md) — the `domain/` leaf for the shared result type.
