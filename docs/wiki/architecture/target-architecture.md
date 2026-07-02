---
slug: target-architecture
title: "Target Architecture"
chapter: architecture
order: 70
status: review
register: handbook
summary: >
  Where the architecture is deliberately heading: a domain/ pure-physics leaf package,
  single-source-of-truth constants, a typed CFLIBSConfig replacing 15+ CFLIBS_* env vars, the
  IPD/PF truncation-consistency invariant, the air<->vacuum wavelength convention, and a physical
  NoiseModel. Synthesizes the v4 overhaul ARCHITECTURE/BLUEPRINT/CRITIC set.
tags: [target-architecture, domain, cflibsconfig, ipd, air-vacuum, noise-model, single-source-of-truth]
updated: 2026-07-02
sources:
  - docs/v4/overhaul/ARCHITECTURE.md
  - docs/v4/overhaul/BLUEPRINT.md
  - docs/v4/overhaul/BLUEPRINT-ADDENDUM.md
  - docs/v4/overhaul/CRITIC.md
related: [module-map, abstractions, change-taxonomy, adr-index, jax-backend-matrix]
code_refs:
  - cflibs/core/constants.py
  - cflibs/plasma/saha_boltzmann.py::SahaBoltzmannSolver
  - cflibs/plasma/partition.py
  - cflibs/core/config.py
---

*[Wiki home](../index.md) · [Glossary](../glossary.md) · [Bibliography](../bibliography.md)*

# Target Architecture

This page is the target-state map: the refactor CF-LIBS is deliberately heading toward, synthesized
from the v4 overhaul `ARCHITECTURE.md`, `BLUEPRINT.md`, `BLUEPRINT-ADDENDUM.md`, and `CRITIC.md`. It
collapses that layered document set into one place. Every item here is classified by the
[change taxonomy](change-taxonomy.md) (SAFE-NOW / BENCHMARK-GATED / DESIGN-DECISION / AUDIT-FIRST) and,
per mission priority, correctness precedes latency everywhere. Wavelengths below are **air** unless a
vacuum detector calibration is explicitly in play (see [§ air↔vacuum](#air-vacuum)).

## 1. The `domain/` pure-physics leaf {#domain-leaf}

**Problem:** the pipeline's shared data types live inside solver/runtime modules. `CFLIBSResult` is
defined in `inversion/solve/iterative.py` but imported by `io/`, `jitpipe/`, and `closed_form.py`;
`LineObservation` is imported through a `physics/boltzmann` shim by ~11 modules; the `AtomicSnapshot`
pytree is a domain carrier stranded in `core/jax_runtime.py`. This creates import cycles (e.g.
`physics/matrix_effects` ↔ `solve/iterative`) and forces consumers to pull in solver internals just to
read a result.

**Target (SAFE-NOW, behavior-preserving):** create `cflibs/domain/` as a **leaf package** — it imports
nothing from `cflibs` except `core.constants`.

| Type | From | To |
|------|------|----|
| `CFLIBSResult` | `inversion/solve/iterative.py` | `domain/result.py` |
| `LineObservation`, `BoltzmannFitResult` | `inversion/common/data_structures.py` (shim) | `domain/observation.py` |
| `AtomicSnapshot` | `core/jax_runtime.py` | `atomic/snapshot.py` |

`inversion/common/` re-exports for back-compat. Rationale follows RADIS's loose-coupling principle
(database objects separate from computation objects): the result contract can then be imported by
`io/`, `jitpipe/`, `benchmark/`, and `validation/` without dragging in solver internals.

## 2. Single source of truth for every physics quantity

One canonical function/constant per quantity; all other call sites import it. The overhaul found the
same constant re-defined or drifted across files:

| Quantity | Canonical home | Was duplicated in |
|----------|----------------|-------------------|
| Olivero-1977 Voigt constants (0.5343 / 0.2169) | `core/constants.py` | `profiles.py`, `radiation/stark.py`, `jitpipe/stark.py`, instrument (2.355 vs exact $2\sqrt{2\ln2}$) |
| McWhirter prefactor `1.6e12` | `core/constants.py::MCWHIRTER_CONST` | 3 sites (physical_consistency, line_selection, temporal) |
| Saha constant, $k_B$ (eV/K), cm⁻¹→eV | `core/constants.py` | scattered literals |
| number↔mass fraction conversion | one tested converter | 3 scoring paths, each raw |
| air↔vacuum wavelength | single `air_to_vacuum` / `vacuum_to_air` in `core/` | none yet (audit-first) |

De-duplicating constants is SAFE-NOW (parity-verified). The number↔mass converter is SAFE-NOW to
*build* but its *application* on the scoring path is BENCHMARK-GATED because it moves every composition
number ([change-taxonomy](change-taxonomy.md)).

## 3. Configuration over 15+ environment variables {#cflibsconfig}

**Problem:** the pipeline reads 15+ `CFLIBS_*` env vars, 8 inline `CFLIBS_FF_*` benchmark flags, plus
config fictions, with inconsistent truthy parsing (`CFLIBS_USE_LAX_WHILE_LOOP` accepts only `"1"`,
while `CFLIBS_RELIABILITY_FROM_UNCERTAINTY` accepts `1/true/yes/on`). Effective configuration is
invisible, so benchmark runs are not reproducible.

**Target:** a single frozen, validated `cflibs.core.config.CFLIBSConfig` dataclass that replaces all
env-var reads. It is:

- loaded once at solver/pipeline construction,
- logged at INFO with all effective values,
- embedded in `CFLIBSResult.metadata` so a run is reproducible from its own output,
- the place feature-flag promotions become **mechanical config changes**, not env-var toggles.

Env vars are honored via an adapter for one release cycle, then deprecated. The three highest-value
flags to migrate first are `CFLIBS_REFUSE_TO_REPORT`, `CFLIBS_MCWHIRTER_RESONANCE_DE`, and
`CFLIBS_USE_LAX_WHILE_LOOP` — they gate the accuracy/reliability items. Building `CFLIBSConfig` with
constructor defaults matching current env defaults is SAFE-NOW (no behavior change on construct).

## 4. The IPD / partition-function truncation-consistency invariant {#pf-ipd-invariant}

> [!IMPORTANT] This invariant is flagged CRITICAL in the Saha–Boltzmann LTE literature and is the most
> important physics-correctness invariant in the whole forward model.

**The invariant:** the partition-function sum cutoff ($E_{\max} = \chi - \Delta\chi$) MUST use the
**same** ionization-potential depression $\Delta\chi$ as the Saha-equation exponent
($\chi_{\text{eff}} = \chi - \Delta\chi$). Break it and the Saha equation becomes internally
inconsistent: the forward model produces $n_{II}/n_{I}$ ratios that do not correspond to the
population calculation in the Boltzmann plot. The IPD itself is the Debye–Hückel / Stewart–Pyatt
$\Delta\chi(n_e, T)$ correction [@stewart1966].

**Architectural enforcement (SAFE-NOW once audited):**

1. A single `compute_ipd(n_e, T, model)` in `cflibs/plasma/` is the **only** site that computes
   $\Delta\chi$. Every caller — the Saha solver, the partition-function truncation, the level-population
   filter — imports and calls it.
2. The Saha solver passes the computed $\Delta\chi$ to the partition evaluator as an **explicit
   argument**; the evaluator does not independently recompute IPD.
3. An invariant test asserts PF cutoff $= \chi - \Delta\chi$ for both the polynomial [@irwin1981] and
   direct-sum paths, and that the Saha ratio agrees within 0.1 % with an independently-derived value at
   $(n_e = 10^{17}\,\text{cm}^{-3}, T = 10^4\,\text{K})$ for Fe I/II.

The risk today is two divergent population code paths in `radiation/kernels.py` (precomputed vs.
kernel-internal); if they use different IPD cutoffs they produce different Saha ratios for the same
inputs. The physics of this invariant is owned by [libs-physics](../libs-physics.md); this page
owns the *architectural* enforcement (one function, threaded argument, invariant test).

## 5. Air ↔ vacuum wavelength convention {#air-vacuum}

**Context:** NIST ASD reports **air** wavelengths above 200 nm. The Boltzmann ordinate
$y = \ln(I\lambda/(g_kA_{ki}))$, the Doppler/Stark formulas, and the line-matching tolerance all need a
consistent $\lambda$. A 0.05–0.1 nm systematic air/vacuum offset would dominate the sub-pixel centroid
gains and create false identification peaks below ~0.1 nm tolerance.

**Convention target (AUDIT-FIRST, then enforce):**

- the DB stores wavelengths in **air** (matching NIST/ASD provenance);
- line-matching tolerance is in **air nm**;
- the forward-model emissivity uses the stored (air) $\lambda$ for the $hc/\lambda$ factor;
- conversion to vacuum is applied **only** for an explicitly-vacuum detector calibration;
- a **single** `air_to_vacuum(wl_nm)` / `vacuum_to_air(wl_nm)` utility lives in `cflibs/core/` using the
  Edlén (1953) formula (same as NIST);
- every module that stores or compares wavelengths annotates the convention in its docstring.

This is AUDIT-FIRST because no verified finding has yet confirmed the DB, tolerance, and forward model
are consistent; the fix classifies as BENCHMARK-GATED if the audit finds an inconsistency (it changes
identification).

## 6. Physical `NoiseModel`

**Problem:** the round-trip validator adds background **before** shot noise, applies Poisson to
signal+background (inflating variance), then uses the noise-realized intensity as its own variance
estimate — a circular estimator. The Bayesian likelihood uses a fixed-$\sigma$ Gaussian (a Pearson
bias on shot-noise-dominated lines).

**Target `NoiseModel` (SAFE-NOW):** shot noise on the **clean** signal, variance from model
parameters:

```python
var_shot  = max(clean_signal, 0)          # Poisson: Var[photons] = E[photons]
var_total = var_shot + sigma_readout**2   # Gaussian readout in quadrature
noisy     = clean_signal + background + rng.normal(scale=sqrt(var_total))  # background AFTER variance
```

The Bayesian likelihood then uses pixel-dependent variance $\sigma_k^2(x) = \lambda_k(x) + \sigma_{RON}^2$
evaluated at each MCMC proposal, including the $\log(\sigma_k^2)$ normalization term so BIC/WAIC model
comparison is unbiased. The gate condition: if any line's SNR < 100, the fixed-$\sigma$ approximation is
inaccurate. Physics detail is in [libs-physics](../libs-physics.md) and the error budget in
[error-budget-and-falsification](../error-budget-and-falsification.md).

## 7. What the target architecture does NOT change

- The CF-LIBS algorithmic flow (Saha–Boltzmann + Boltzmann-plot + closure + $n_e$ iteration) is
  preserved and validated; any physics change is benchmark-gated.
- JAX stays optional; the refactor adds no new JAX dependencies.
- `cflibs/evolution/` stays a separate optimization-tooling package; the physics-only constraint is
  not relaxed ([physics-only-constraint](physics-only-constraint.md)).
- The CLI and YAML schema stay backward-compatible through the `CFLIBSConfig` migration.

## What correct code MUST do (target-architecture checklist)

- [ ] Import shared result/observation types from `domain/`, never from a solver module.
- [ ] Read every tunable through `CFLIBSConfig`, never `os.environ` directly.
- [ ] Compute IPD exactly once via `compute_ipd`; thread $\Delta\chi$ to both Saha and PF truncation.
- [ ] Truncate the partition sum at $\chi - \Delta\chi$ — the same $\Delta\chi$ the Saha exponent uses.
- [ ] State air-vs-vacuum in the first paragraph of any module that stores/compares wavelengths.
- [ ] Estimate variance from the noise model, never from the noise-realized intensity.
- [ ] Import every physical constant from `core/constants.py` — no local literals.

## See also

- [Change taxonomy](change-taxonomy.md) — the classification each item above carries.
- [ADR index](adr-index.md) — the design decisions (DA-1…DA-8) behind these targets.
- [Abstractions](abstractions.md) — the `domain/` types and the seams they cross.
- [libs-physics](../libs-physics.md) — the physics behind the IPD/PF and noise items.
