---
slug: impl-novel-techniques
title: "Code: Implementations of Our Novel Techniques"
chapter: impl-novel-techniques
order: 0
status: review
register: code-walkthrough
summary: >
  Maps each frontier CF-LIBS technique to the shipped module that realizes it: Lean-derived
  line-selection thresholds and error budget that supersede tuned magic numbers, four
  certified reliability constants, observable-anchored self-absorption, Aitchison log-ratio
  reporting, in-plasma g·A self-calibration, the jittable differentiable pipeline, the
  manifold (a fast-inference OPTION, not the pipeline spine), and the M7/M8 refuse-to-report
  gates. Governing rule: every threshold is DERIVEd, ADAPTed, JOINTLY-OPTIMIZEd, or DROPped,
  leaving <=3-5 free parameters.
tags: [implementation, derived-thresholds, error-budget, reliability, self-absorption, log-ratio, jitpipe, manifold, refuse-to-report, parameter-taxonomy]
updated: 2026-07-02
benchmarks_pre_reset: false
sources:
  - "@aragon2008"
  - "@ciucci1999"
  - "@tognoni2010"
  - "@cristoforetti2010"
  - "@bulajic2002"
  - "@sun2009"
  - "@elsherbini2005"
  - "@volker2023"
  - "@cavalcanti2013"
  - "@zhao2018"
  - "@aitchison1982"
  - "@egozcue2003"
  - "@kawahara2022"
  - "@kawahara2025"
  - cflibs/inversion/physics/derived_thresholds.py
  - cflibs/inversion/physics/error_budget.py
  - cflibs/inversion/physics/reliability.py
  - cflibs/inversion/physics/self_absorption_observable.py
  - cflibs/inversion/physics/opc.py
  - cflibs/inversion/physics/ga_selfcal.py
  - cflibs/atomic/aki_anchor.py
  - cflibs/inversion/physics/closure.py
  - cflibs/inversion/physics/quality.py
  - cflibs/inversion/solve/joint_optimizer.py
  - cflibs/jitpipe/__init__.py
  - cflibs/manifold/__init__.py
  - docs/M5-parameter-optimization-plan.md
  - docs/research/realtime/2026-06-20-realtime-plan-v4-real-data-accuracy.md
  - scripts/research/rtval/README_rtval.md
  - docs/v4/overhaul/verified/manifold.md
code_refs:
  - cflibs/inversion/physics/derived_thresholds.py::min_lines_per_element_for
  - cflibs/inversion/physics/error_budget.py::derive_line_selection_thresholds
  - cflibs/inversion/physics/reliability.py::temperature_conditioning
  - cflibs/inversion/physics/reliability.py::composition_error_bound
  - cflibs/inversion/physics/reliability.py::mcwhirter_min_ne
  - cflibs/inversion/physics/reliability.py::stark_saha_lte_gate
  - cflibs/inversion/physics/self_absorption_observable.py::ObservableSelfAbsorptionCorrector
  - cflibs/inversion/physics/self_absorption_observable.py::correct_intensity_planck
  - cflibs/inversion/physics/opc.py::calibrate_opc
  - cflibs/inversion/physics/ga_selfcal.py::find_shared_upper_groups
  - cflibs/atomic/aki_anchor.py::anchor_aki_to_lifetimes
  - cflibs/inversion/physics/closure.py::log_ratios
  - cflibs/inversion/physics/closure.py::ilr_transform
  - cflibs/inversion/physics/quality.py::per_element_reliability_from_uncertainty
  - cflibs/inversion/physics/quality.py::downgrade_quality_flag
  - cflibs/inversion/solve/joint_optimizer.py
  - cflibs/jitpipe/__init__.py
  - cflibs/manifold/generator.py::ManifoldGenerator
lean_refs:
  - CflibsFormal/ErrorBudget.lean#olsSlope_stable_l2
  - CflibsFormal/ErrorBudget.lean#requiredEnergySpread_sufficient
  - CflibsFormal/ErrorBudget.lean#maxPerLineError_sufficient
  - CflibsFormal/ErrorBudget.lean#temp_rel_error_eq
  - CflibsFormal/ErrorBudget.lean#composition_target_sufficient
  - CflibsFormal/Robustness.lean#twoLineBeta_stable_sharp
  - CflibsFormal/CompositionRobustness.lean#composition_dist_vector_le
  - CflibsFormal/StarkBroadening.lean#mcWhirterBound
  - CflibsFormal/StarkBroadening.lean#stark_saha_lte_consistent
related: [frontier-methods, formal-spec, error-budget-and-falsification, benchmarks-reliability-workflows, impl-literature-methods, architecture]
supersedes: []
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Code: Implementations of Our Novel Techniques

This chapter is the bridge between the theory in [Frontier Methods](frontier-methods.md) and
the running code: for each novel technique it names the shipped module, the exact function, and
the physics or Lean theorem it realizes. The single governing design rule is that **no free
number survives without a job** — every threshold is either *derived* from a proven error
budget, *adapted* from a runtime feature, *jointly optimized* offline, or *dropped* as inert,
which leaves the pipeline with at most three to five genuinely free parameters
([`docs/M5-parameter-optimization-plan.md`](../M5-parameter-optimization-plan.md)).

> [!NOTE] FORMAL — several techniques below are backed by machine-checked proofs in the
> companion `cflibs-formal` repo (Lean 4 + Mathlib, axiom-clean, no `sorry`). Where a formula is
> a verbatim mirror of a theorem, the theorem id is cited in monospace `lean:...`; see
> [Formal Spec](formal-spec.md) for the proof-level treatment and
> [Error Budget & Falsification](error-budget-and-falsification.md) for the negative-result
> ledger.

Wavelength convention: every module discussed here consumes upper-level **energies in eV** and
intensities in instrument units; wavelengths follow the repo convention (the DB stores **air**
wavelengths per NIST/ASD, converted through the single `cflibs/core` util). The canonical
Boltzmann ordinate is $y = \ln\!\big(I_{ki}\,\lambda / (g_k A_{ki})\big)$ versus $x = E_k$ with
slope $-1/(k_B T)$ (`lean:CflibsFormal/Boltzmann.lean#boltzmann_plot`); $\lambda$ is load-bearing
whenever it varies across a fit and is never silently dropped.

---

## 1. The parameter taxonomy: DERIVE / ADAPT / JOINTLY-OPTIMIZE / DROP {#taxonomy}

Classical CF-LIBS pipelines accrete "magic numbers" — a minimum line count of 3, a minimum
energy spread of 2 eV, a minimum SNR of 10 — that were tuned on one corpus and silently
overfit. The M5 plan
([`docs/M5-parameter-optimization-plan.md`](../M5-parameter-optimization-plan.md)) replaces this
with a **three-tier routing plus an offline optimizer over the residue**. Each parameter is
classified as exactly one of:

| Tier | Meaning | Realized by |
|------|---------|-------------|
| **DERIVE** | fixed by a physics formula (a target accuracy determines it) | `cflibs/inversion/physics/derived_thresholds.py`, `error_budget.py` |
| **ADAPT** | `param = clip(k · feature, lo, hi)` — a bounded, interpretable rule keyed on a runtime observable (resolving power $R$, per-spectrum noise $\sigma$, calibration RMS) | `cflibs/inversion/pipeline.py` `_resolve_adaptive_tolerances` |
| **JOINTLY OPTIMIZE** | genuinely free; tuned offline by Optuna/CMA-ES in `scripts/campaign1/` and shipped as plain preset numbers with a provenance comment | `cflibs/inversion/pipeline.py` presets |
| **DROP** | study-confirmed inert — pin at default, remove from the search space | (no code; removed from knob space) |

The net collapse is dramatic: `min_lines_per_element`, `min_energy_spread_ev`, and `min_snr`
fold into **one** physics target ($\sigma_T/T$); four detection thresholds become ADAPT rules
sharing ~6 dimensionless coefficients; five inert gates are DROPped. Only ~3–5 free knobs remain
— small enough that a few hundred optimizer trials genuinely saturate the space, and (critically)
few enough that overfitting on ~8–12 effective spectra is structurally avoided
([`docs/M5-parameter-optimization-plan.md`](../M5-parameter-optimization-plan.md) §2).

> [!WARNING] BENCHMARK-GATED — the JOINTLY-OPTIMIZE presets and any ADAPT default that changes
> the shipped path must clear a flag-off/flag-on scoreboard non-regression run with a paired
> bootstrap ΔRMSE whose 95% CI excludes zero (the repo has regressed on identifier changes
> three times; see [Benchmarks & Reliability Workflows](benchmarks-reliability-workflows.md)).
> No sub-0.1-wt%-RMSE win is trustworthy.

The DERIVE tier is where our most distinctive work lives, so we treat it first.

---

## 2. Derived thresholds and the proven error budget {#derived-thresholds}

`cflibs/inversion/physics/derived_thresholds.py` · `cflibs/inversion/physics/error_budget.py`
· `lean:CflibsFormal/ErrorBudget.lean`

These two modules turn a **target temperature accuracy** into the line-count, energy-spread, and
SNR gates that `line_selection.py` used to hardcode. Every function is a verbatim mirror of a
machine-verified theorem, and `tests/oracle/test_derived_thresholds.py` conformance-tests the
Python against the Float fixtures the Lean spec emits, so a drift from the proof fails CI.

### 2.1 The propagation chain {#error-chain}

The physics is ordinary least squares on the Boltzmann plot. With a per-line ordinate error
$\varepsilon$ (dimensionless; $\varepsilon \approx 1/\mathrm{SNR}$ since $y=\ln I + \text{const}$),
$n$ lines, and an energy spread $s_E = \sum_k (E_k - \bar E)^2$ in eV², the worst-case
inverse-temperature slope error is bounded:

$$|\Delta\beta| \;\le\; \varepsilon\,\frac{\sqrt{n}}{\sqrt{s_E}}
\qquad(\text{`lean:CflibsFormal/ErrorBudget.lean#olsSlope_stable_l2`}).$$

The relative temperature error is then **exact** (not a bound):

$$\frac{|\Delta T|}{T} \;=\; k_B\,T\,|\Delta\beta|
\qquad(\text{`lean:CflibsFormal/ErrorBudget.lean#temp_rel_error_eq`}),$$

because $T = -1/(k_B\beta)$ so $dT/T = -d\beta/\beta = k_B T\, d\beta$. Inverting the chain turns
a *target* $\sigma_T/T$ into the required energy spread and SNR:

$$s_E \ge \frac{\varepsilon^2 n}{\tau_\beta^2}
\;(\text{`#requiredEnergySpread_sufficient`}),\qquad
\varepsilon \le \tau_\beta\sqrt{s_E/n}
\;(\text{`#maxPerLineError_sufficient`}),$$

with the slope target $\tau_\beta = (\sigma_T/T)/(k_B T)$. The line-count benefit is *statistical*
rather than deterministic: only the Gauss–Markov noise-gain kernel $\sum_k w_k^2 = 1/s_E$ is
proven (`#olsSlope_noise_gain`), from which `required_min_lines_stat` derives
$n \ge \varepsilon^2/(\mathrm{var}_{\!E}\cdot\tau_\beta^2)$.

### 2.2 The code, line for line {#derived-code}

The dimensionless core in `error_budget.py:56` (`slope_error_bound`) and `error_budget.py:98`
(`slope_target_from_temp_rel`) is the exact mirror; the *physical adoption layer* below it maps
the chain into Kelvin/eV and — importantly — converts between the statistical spread $s_E$ and
the energy **span** $R = \max E - \min E$ that the pipeline actually gates on. That conversion is
distribution-dependent, and the module is honest about the assumption:

| line distribution | $s_E$ vs span $R$ | when |
|-------------------|-------------------|------|
| `"uniform"` (default) | $s_E \approx n R^2/12$ | lines spread evenly — the practitioner case |
| `"endpoints"` | $s_E = n R^2/4$ | extreme, maximal $s_E$ for a span |

`required_energy_span_ev` (`error_budget.py:157`) and `min_snr_for_target`
(`error_budget.py:175`) invert the sufficiency theorems under the chosen distribution, and
`derive_line_selection_thresholds` (`error_budget.py:205`) packages them into a frozen
`LineSelectionThresholds`. The convenience wrappers in
`derived_thresholds.py::min_lines_per_element_for` (`derived_thresholds.py:81`) and
`min_energy_spread_ev_for` (`derived_thresholds.py:97`) are what a caller uses to replace the
tuned constants `3`/`20` and `2.0 eV` with *consequences of a stated target*.

The composition side of the budget closes the loop: `density_budget_from_composition`
(`derived_thresholds.py:71`) implements

$$\delta \;\le\; \frac{\tau_C\,\hat S}{\text{card}+1}
\qquad(\text{`lean:CflibsFormal/ErrorBudget.lean#composition_target_sufficient`}),$$

the per-species absolute density-error budget guaranteeing a target composition accuracy
$\tau_C$ over `card` species at recovered total density $\hat S$.

> [!NOTE] FORMAL — the whole chain (`olsSlope_stable_l2` → `temp_rel_error_eq` →
> `composition_target_sufficient`) is proven end-to-end and oracle-conformance-tested. This is
> the concrete meaning of "our thresholds are derived, not tuned." See
> [Formal Spec](formal-spec.md#part-3-error-budget) for the proofs.

### 2.3 What correct code MUST do {#derived-checklist}

- State a **target accuracy** ($\sigma_T/T$ or $\tau_C$) and derive gates from it — never inline
  `min_lines = 3` as a bare constant.
- Carry $\lambda$ inside the ordinate; use $s_E$ (eV²), not a bare energy span, when applying the
  slope-error bound, and declare the span↔$s_E$ distribution assumption.
- Keep the Python a byte-faithful mirror of the Lean theorem it cites (conformance test is the
  contract).

---

## 3. Four certified reliability constants {#reliability}

`cflibs/inversion/physics/reliability.py` · `lean:CflibsFormal/Robustness.lean`,
`CompositionRobustness.lean`, `StarkBroadening.lean`

`reliability.py` turns *proven robustness constants* into practical selectors and refuse/flag
gates. Each formula names its theorem; nothing here touches NumPy state, a DB, or ML — it is pure
`math`.

### 3.1 Sharp temperature conditioning {#temp-conditioning}

For a two-line slope estimate $\beta = (y_j - y_i)/(E_i - E_j)$, the worst-case opposite-sign
ordinate perturbation gives an **exact** (sharp) error amplification:

$$|\Delta\beta| = \frac{2\varepsilon}{|E_i - E_j|}
\qquad(\text{`lean:CflibsFormal/Robustness.lean#twoLineBeta_stable_sharp`}).$$

`temperature_conditioning(e_i, e_j)` (`reliability.py:48`) returns $2/|E_i-E_j|$ (or $+\infty$ for
coincident energies); `rank_line_pairs_by_conditioning` and `best_temperature_pair` sort pairs so
a selector prefers the **widest** upper-level energy separation — strictly better conditioned for
temperature. This is the rigorous justification for the folklore that a wide $E_k$ lever recovers
$T$ more robustly [@aragon2008].

### 3.2 Whole-vector composition bound {#comp-bound}

`composition_error_bound(card, delta, total_density)` (`reliability.py:138`) mirrors the
$\ell^1$ whole-vector certificate

$$\sum_s |\tilde C_s - C_s| \;\le\; \frac{2\,\text{card}\,\delta}{\hat S}
\qquad(\text{`lean:CflibsFormal/CompositionRobustness.lean#composition_dist_vector_le`}),$$

the certified worst-case composition error for a per-species density error $\le \delta$ over
`card` species at recovered total density $\hat S$. It shrinks with larger recovered density and
grows linearly in both species count and per-species uncertainty — the exact levers M8 reports on.

### 3.3 The McWhirter LTE floor {#mcwhirter}

`mcwhirter_min_ne(t_k, d_e_ev)` (`reliability.py:173`) implements the classical necessary
condition for collisional (LTE) processes to dominate radiative ones [@cristoforetti2010]:

$$n_e \;\ge\; 1.6\times10^{12}\,\sqrt{T}\,(\Delta E)^3\ \mathrm{cm^{-3}}
\qquad(\text{`lean:CflibsFormal/StarkBroadening.lean#mcWhirterBound`}),$$

with $\Delta E$ the relevant transition energy interval in eV, $T$ in K. The constant lives
once in `cflibs/core/constants.py::MCWHIRTER_CONST`. The bound is proven monotone increasing in
both $T$ (`#mcWhirterBound_mono_T`) and $\Delta E$ (`#mcWhirterBound_mono_dE`): a hotter plasma or
larger energy interval demands a higher $n_e$ for LTE.

> [!IMPORTANT] The McWhirter criterion is *necessary, not sufficient* for LTE in a transient,
> inhomogeneous laser plasma [@cristoforetti2010]. The physically correct $\Delta E$ is the
> **resonance-line energy**, not `max(E_k)` (see [`reference_mcwhirter_delta_e_physics`]). That
> resonance-$\Delta E$ convention is **opt-in**, gated behind `CFLIBS_MCWHIRTER_RESONANCE_DE`; the
> **default** solver path still passes $\Delta E = \max(E_k)$ (see
> [impl-literature-methods §8](impl-literature-methods.md#iterative)). Prefer the flag.

### 3.4 The Stark↔Saha LTE cross-check M7 consumes {#stark-saha-gate}

`stark_saha_lte_gate(ne_stark, ne_saha, t_k, d_e_ev, rtol=0.5)` (`reliability.py:205`) mirrors
`lean:CflibsFormal/StarkBroadening.lean#stark_saha_lte_consistent`. Two **physically independent**
electron-density diagnostics — a line **width** (Stark) versus a stage-intensity **ratio** (Saha)
— certify LTE self-consistency iff *both* hypotheses hold:

1. **Agreement** — the two estimates lie within `rtol` of their mean (empirical evidence, not a
   definitional identity, precisely because they consume different observations).
2. **Clears McWhirter** — their mean is at least `mcwhirter_min_ne(T, ΔE)`.

On failure the returned reason names the violated hypothesis (`disagree` / `below-mcwhirter` /
`invalid-ne`), which is exactly the signal the M7 refuse-to-report path (§9) flags on. This is the
data-side answer to "should we trust this fit at all?" — two independent $n_e$ measurements that
disagree are a hard reliability failure no goodness-of-fit metric would catch.

### 3.5 What correct code MUST do {#reliability-checklist}

- Rank temperature pairs by **energy separation**, never by intensity alone.
- Feed the Stark↔Saha gate two genuinely independent $n_e$ diagnostics (width vs ratio); never
  wire the same estimate into both slots (the agreement test becomes vacuous).
- Treat the McWhirter value as a floor, not a target; a plasma may clear it and still fail LTE.

---

## 4. Observable-anchored self-absorption {#self-absorption}

`cflibs/inversion/physics/self_absorption_observable.py`

This module is a **falsification made permanent in code**. The 2026-06-09 overhaul audit
condemned the previous composition-fed self-absorption corrector: it computed each line's optical
depth $\tau$ from the *recovered* composition, forming a positive-feedback loop
(over-attributed Fe → larger $\tau$ → larger intensity boost → larger Boltzmann intercept → closure
gives Fe still more mass), and empirically it *worsened* intercept inflation on real ChemCam
BHVO-2.

> [!CAUTION] FALSIFIED: Composition-derived per-line τ improves accuracy
> - **Claim:** compute $\tau$ from the composition-derived lower-level column density and correct
>   each line for thick-line saturation.
> - **Predicted:** lower held-out RMSEP on ChemCam BHVO-2.
> - **Observed:** RMSEP *increased* — positive-feedback loop ($\tau$ feeds composition feeds $\tau$).
> - **Verdict:** REJECTED; replaced by the observable-anchored corrector below.
> - **Evidence:** `docs/research/physics-first-principles-audit.md` Issue 3 (finding F4);
>   `cflibs/inversion/physics/self_absorption_observable.py`.
> - **Date:** 2026-07-02

The literature requirement is that **every per-line correction anchor to an observable of the
measured spectrum, never only to the current composition iterate** [@bulajic2002; @sun2009]. The
module implements a correction ladder, applied to the observed line list **once, before** the
Saha–Boltzmann fit (the old per-iteration placement was itself part of the feedback problem):

| Rung | Method | Observable | Validity |
|------|--------|-----------|----------|
| (a) | Doublet/multiplet intensity ratio | a measured line pair sharing the same upper level; deviation from the optically-thin ratio $(g_kA_{ki}/\lambda)_1/(g_kA_{ki}/\lambda)_2$ gives $\tau$ with **no plasma state at all** [citation-needed] | $\tau \in [0.1, 5]$ (`DOUBLET_TAU_VALIDITY_MAX`) |
| (b) | Planck-ceiling closed form | measured peak spectral radiance + a $T$ estimate → $\tau$ from the homogeneous-slab solution $I_\lambda = B_\lambda(1-e^{-\tau_\lambda})$ [@volker2023] | $\tau \le 3$ (`PLANCK_TAU_VALIDITY_MAX`, their 10% RSD budget) |
| (c) | No correction + SA-suspect flag | lines matching the published SA-risk signature (low lower-level energy $E_i < 0.74$ eV and high intensity) are **down-weighted** via inflated uncertainty, never silently boosted | — |

The Planck rung is `correct_intensity_planck` (`self_absorption_observable.py:214`), built on
`planck_ceiling_optical_depth` (`:115`) and the Doppler curve-of-growth escape factor
`doppler_cog_escape_factor` (`:175`); the ladder orchestrator is
`ObservableSelfAbsorptionCorrector` (`:287`). The El Sherbini measured/Stark width-ratio route
[@elsherbini2005] and the doublet route [citation-needed] are the two width- and ratio-based observables
the module prefers before falling back to down-weighting. **Nothing in this module reads a
recovered composition** — that invariant is the whole point.

Cross-reference: the *known-matrix* alternative to calibration-free self-absorption is the
opt-in one-point calibration in `cflibs/inversion/physics/opc.py::calibrate_opc` [@cavalcanti2013;
@zhao2018], which by construction sees only certified standards and never a recovered composition
— the same structural-honesty property. See [Impl: Literature Methods](impl-literature-methods.md)
for OPC as a classical-quantification lever.

### What correct code MUST do {#sa-checklist}

- Anchor every $\tau$ to a spectral observable (doublet ratio, Planck ceiling, Stark/measured
  width); if none exists, **down-weight**, never boost.
- Apply the correction once, before the fit; never per-iteration after a composition update.
- Respect the per-method $\tau$ validity ceiling ($\le 3$ Planck, $\le 5$ doublet).

---

## 5. Log-ratio reporting: the DED deliverable {#log-ratio}

`cflibs/inversion/physics/closure.py`

For the real goal — real-time composition-**drift** tracking on a known/constrained element set
(e.g. {Ti, Al, V} in DED feedstock) — the load-bearing output is not the absolute wt% vector but
the **Aitchison log-ratios** of the closure relatives. `log_ratios(concentrations, reference)`
(`closure.py:101`) returns $\ln(C_s / C_\text{ref})$ for each element, which is *provably matrix-
and detected-set invariant*: an unknown scalar calibration factor $F$ and any missing part cancel
in the ratio [@aitchison1982]. Reporting drift as $\Delta\ln(N_i/N_j)$ instead of $\Delta$wt%
removes the closure-induced spurious correlation that plagues raw composition tracking.

The isometric route is `ilr_transform` (`closure.py:255`, Helmert basis) with `clr_transform`
(`:236`) and `ilr_inverse` (`:277`); `ilr_propagate_covariance` (`:299`) maps a composition
covariance into ILR coordinates, removing the degenerate sum-to-one direction so uncertainty
propagation is non-singular [@egozcue2003]. These map the constrained simplex to real coordinates
where ordinary Euclidean statistics (means, distances, covariances) are meaningful — the correct
geometry for compositional data.

> [!NOTE] Prefer log-ratios $\ln(N_i/N_j)$ over closure wt% for the DED/tracking deliverable. The
> ratio is what survives the CF-LIBS scalar factor $F$ and the always-incomplete detected element
> set; absolute wt% does not.

---

## 6. In-plasma g·A self-calibration (planned/partial) {#ga-self-cal}

`cflibs/inversion/physics/ga_selfcal.py` · `cflibs/atomic/aki_anchor.py`

Two lines $i,j$ sharing the **same upper level** $k$ have an intensity ratio fixed *exactly* by
atomic constants — the shared $g_k$ **and** the shared population $n_k$ cancel:

$$\frac{I_i}{I_j} = \frac{A_i\,\lambda_j}{A_j\,\lambda_i}.$$

This identity holds independently of $T$, $n_e$, the level populations, and even LTE — one of the
very few model-free statements in CF-LIBS. Consequently, in the Boltzmann ordinate every
shared-upper-level line must land on the **same** $y$-value; any spread within the group is a
direct measurement of the **relative** $A_{ki}$ error (plus noise and *differential*
self-absorption). `find_shared_upper_groups` (`ga_selfcal.py:237`) forms the groups (by DB
`upp_level_id`, else by the physical $(\text{element}, \text{stage}, E_k, g_k)$ fingerprint) and
the module refines the per-line $A_{ki}$ so the group agrees, shrinking corrections toward 1 by
the per-line noise estimate (empirical Bayes). Optically-thick lines are excluded so differential
self-absorption never contaminates the anchor.

This is a **relative** calibration — it removes line-to-line $A_{ki}$ scatter but cannot recover
the group's *absolute* scale. That reproduces, from the plasma itself, the observed benefit of an
internally-consistent single-source line set ("Kurucz beat NIST" on SuperCam; see notes
`reference_exojax_kurucz_atomic`). The absolute-scale sub-step is `aki_anchor.py`: renormalize
each upper level's $A_{ki}$ set to a measured radiative lifetime, $A_{ki} = \mathrm{BR}_{ki}/\tau_k$
(`anchor_aki_to_lifetimes`), following Wisconsin-group branching-fraction/lifetime practice, whose
relative branching fractions and laser-measured lifetimes are accurate to a few percent
[@aragon2008]. Corrections are written to a **separate overlay DB** (`anchored_lines` table); the
production DB is opened read-only and never mutated. `species_scale_diagnostic` quantifies the
grade-weighted RMS fractional $A_{ki}$ error as the Boltzmann-plot residual floor attributable to
atomic data — the falsifiable "scale-spread table."

> [!NOTE] Status: the relative-g·A refinement and the anchoring diagnostics are shipped; the
> absolute lifetime anchoring depends on an independent second line source (Kurucz/VALD ingest via
> `scripts/ingest_kurucz_atomic.py`) and a staged overlay build, and is **partial**. Theory and
> motivation live in [Frontier Methods](frontier-methods.md#ga-self-cal).

---

## 7. The jittable differentiable pipeline {#jitpipe}

`cflibs/jitpipe/` · `scripts/research/rtval/` · `lean:` (n/a — numerical, benchmark-gated)

`cflibs.jitpipe` is a **parallel**, end-to-end `jit`/`vmap`-able re-implementation of the CF-LIBS
inversion pipeline (ADR-0004). Per its contract in `cflibs/jitpipe/__init__.py`, nothing outside
`jitpipe` imports it, it never mutates the reference pipeline, and — unlike the rest of `cflibs`,
which degrades gracefully without JAX — it **requires** JAX and raises a clear `ImportError` with
install guidance if absent (there is no NumPy fallback). Its public surface is a frozen,
pytree-registered `PipelineSnapshot` (the whole atomic DB as a struct-of-arrays, `.npz`-cached),
a traced `PipelineParams`, a hashable `StaticConfig` jit-cache key, and `run_one`/`run_batch`.

### 7.1 Structured Gauss–Newton, K=1, and the forward-difference lever {#structured-gn}

The reproduction harness `scripts/research/rtval/` is the how-to for the fast path. It is an
**independent ExoJAX-based reference** forward + a structured-Jacobian **K=1 Gauss–Newton**
inversion, used to *validate* the production core-JAX `jitpipe` forward, not duplicate it. The key
numerical lever: the spectrum is **linear in composition** — `forward_with_basis(T, ne)` returns a
per-species basis $B$ so $S = B\,\text{comp}$. The concentration columns of the Jacobian are then
just $B$ (no autodiff needed); only the **two** nonlinear columns ($T$, $\log n_e$) are computed,
and they use **forward differences** rather than a JVP. The reason is measured, not assumed: a
single JVP through the ExoJAX `hjert` Voigt-Hjerting kernel is ~3.4 ms — about 9× a plain forward
— so two extra forward evaluations for the FD columns are the cheaper structured lever
([`scripts/research/rtval/README_rtval.md`](../../scripts/research/rtval/README_rtval.md)).

Measured on a V100S (float32, K=1, median of 120 single-shot calls): a 1000-channel synthetic ROI
inverts in **708 µs** and a real ChemCam 240–300 nm ROI (1163 ch) in **906 µs** — sub-ms is
reached at LIBS-realistic ROI width. The honest caveats are load-bearing: $n_e$ is the
weakest-recovered direction (~29% relative error at K=1) because Stark broadening only weakly
modulates the spectrum at $n_e \sim 10^{17}\,\mathrm{cm^{-3}}$ — a genuine CF-LIBS degeneracy, not
a bug — and K=1 from a *flat* composition prior gives a ~0.10 simplex RMSE, so production use must
warm-start composition from a Boltzmann/NNLS prior. K>1 did not help from the flat start
(overshoot on the ill-conditioned $n_e$ axis).

> [!IMPORTANT] "Sub-ms" is **ROI-conditional** and holds for the batched jit core at ~1000
> channels; the full 240–850 nm path is ~2 ms. Real-data accuracy is **atomic-data-limited**, not
> solver-limited (the algorithmic floor is ~0 while the line-list mismatch dominates). Latency is
> deferred until accuracy is robust; see notes `project_realtime_forward_model_state`.

### 7.2 ExoJAX venv isolation {#exojax-quarantine}

ExoJAX is **not** a declared CF-LIBS dependency and importing it pollutes the test suite.
`scripts/research/rtval/` therefore lives outside `pytest` collection (`pytest.ini` sets
`testpaths = tests`; no file is named `test_*.py`) and runs only in a dedicated ExoJAX venv. The
ExoJAX design patterns we borrow — loose coupling of the line-database (`Adb*`) from the opacity
calculators (`Opa*`), snapshot-based opacity so the full line list need not sit in GPU memory —
are catalogued in [`docs/v4/overhaul/reference/exojax.md`](../v4/overhaul/reference/exojax.md)
and trace to the ExoJAX papers [@kawahara2022; @kawahara2025]. ExoJAX ships **no** Saha, so the
harness supplies Saha–Boltzmann level populations in log-space and drives ExoJAX's Voigt-Hjerting
opacity with them.

---

## 8. Manifold generation — a fast-inference OPTION, not the spine {#manifold}

`cflibs/manifold/` · [`docs/v4/overhaul/verified/manifold.md`](../v4/overhaul/verified/manifold.md)

`ManifoldGenerator` (`cflibs/manifold/generator.py`) JAX-`jit`/`vmap` batch-generates synthetic
spectra over a $(T, n_e, \text{composition})$ grid, stored as HDF5 or Zarr, enabling
nearest-neighbor inference via a FAISS index. The `BasisLibraryGenerator` provides an NNLS-oriented
per-element basis with a **reference-S0 pattern** (area-normalized basis spectra).

> [!IMPORTANT] CORRECTED FRAMING — the manifold is **one path to fast inference, not the pipeline
> spine.** Earlier framing that treated a precomputed manifold as the foundation of inference is
> not carried forward. The reference iterative/joint solvers are the accuracy-bearing path; the
> manifold is an optional coarse-to-fine accelerator.

The adversarial verification of the manifold code
([`docs/v4/overhaul/verified/manifold.md`](../v4/overhaul/verified/manifold.md)) confirmed real
findings that gate its use: the total heavy-particle density is approximated as $n_e$ (standard
CF-LIBS, $\le 20\%$ error at typical 0.7–1.0 eV, degrading above 1.2 eV where Fe/Ca double-ionize —
finding F1); the configured **gate delay was never applied** (integration started at $t=0$,
defeating its purpose — F2, high severity); and the Stark $T^{-\alpha}$ scaling was missing in the
`batch_forward.py` path (F3). There are three parallel physics implementations
(`generator.py`, `batch_forward.py`, `basis_library.py`) that must be kept consistent (F5), and the
LDM time-integration path lacks an integration-level parity test (F10, high).

> [!NOTE] Status correction: the **1e6× C1 conditioning fix landed on branch**; manifold
> **regeneration is pending**. Do not quote manifold-derived composition numbers as current until
> regeneration completes. The manifold is a retrieval/pattern-match accelerator where relative
> spectral shape matters more than absolute scale, which is why the $N_\text{total}\approx n_e$
> approximation is tolerable there but not in the accuracy-bearing solver.

---

## 9. Refuse-to-report gates (M7) and per-element reliability flags (M8) {#refuse-to-report}

`cflibs/inversion/physics/quality.py` · `reliability.py::stark_saha_lte_gate` ·
`cflibs/inversion/pipeline.py`

The single most important reliability behavior is that the pipeline can **refuse to report** when
it cannot trust its own answer, rather than emitting a confidently-wrong composition. Two levers
combine into one trust signal.

**M7 — overall refuse-to-report (Lever 6).** The post-loop Cristoforetti reliability re-fit
[@cristoforetti2010] populates `QualityMetrics` (`quality.py:20`): the inter-element temperature
spread `inter_element_t_std_frac`, the Saha–Boltzmann self-consistency
`saha_boltzmann_consistency`, and a `quality_flag` in
{`excellent`, `good`, `acceptable`, `poor`, `reject`} assigned by `_determine_quality_flag`
(`quality.py:601`). The `stark_saha_lte_gate` (§3.4) supplies the independent-diagnostic LTE
check that this verdict consumes: a `disagree`/`below-mcwhirter` failure forces the flag down.
When the flag reaches `reject`, `overall_reliable` is false and the result is withheld/flagged
rather than reported. (The re-fit is a perf knob defaulting on; disabling it drops the annotation.)

**M8 — per-element reliability (Lever 7).** `per_element_reliability_from_uncertainty`
(`quality.py:656`) labels each element from its **relative** concentration uncertainty
$\text{rel} = \sigma_C/C$: `rel > reject` → `"reject"`, `rel > poor` → `"poor"`, else `"ok"`. A
weak emitter with a huge CI is unreliable *even when its fit metrics look fine* — a zero
concentration with nonzero $\sigma$ is an infinite relative CI and is conservatively rejected;
malformed/non-finite values are rejected. `downgrade_quality_flag` (`quality.py:708`) then folds
the worst per-element label into the overall `quality_flag`, never upgrading — coupling M8 into the
single M7 trust signal.

> [!NOTE] The refuse-to-report path is the operational payoff of the certified constants in §3:
> the Stark↔Saha gate (`twoLineBeta`/`stark_saha_lte_consistent`) and the composition $\ell^1$
> bound (`composition_dist_vector_le`) are exactly the quantities M7/M8 threshold on. See
> [Benchmarks & Reliability Workflows](benchmarks-reliability-workflows.md) for how the gate is
> exercised and [`cflibs-verification-gate`] for the PASS/FAIL discipline.

### What correct code MUST do {#refuse-checklist}

- Compute `overall_reliable` from *executed* diagnostics (independent-$n_e$ agreement + LTE floor
  + composition CI), never assert it.
- Let per-element CIs **only downgrade** the overall flag, never upgrade it.
- Prefer withholding a `reject`-flagged result to reporting a confidently-wrong composition.

---

## 10. The joint-optimizer parameterization recipe (neutral) {#joint-optimizer}

`cflibs/inversion/solve/joint_optimizer.py`

The joint optimizer simultaneously fits $T$, $n_e$, and all concentrations, accounting for
parameter correlations the sequential single-element analysis ignores [@tognoni2010; @ciucci1999].
Its value is the **parameterization recipe**, which any solver in this repo reuses:

| Physical quantity | Parameterization | Why |
|-------------------|------------------|-----|
| Temperature | $\log(T_\text{eV})$ | positivity + scale invariance |
| Electron density | $\log_{10} n_e$ | wide dynamic range |
| Concentrations | $\text{softmax}(\theta)$ via `softmax_closure` | enforces the sum-to-one simplex ($\sum_s C_s = 1$) exactly |

This is presented **neutrally**: the joint optimizer does *not* categorically beat the iterative
solver. Our own finding is that a converged joint fit can beat iterative on some conditioning
regimes (SVD-conditioned, Hebert-style), but the load-bearing open problem is the **adoption gate**
— *when to trust a converged fit* — and posterior/uncertainty width is the signal, not a raw loss
value. Claiming "joint beats iterative" without that adoption gate is a contradicted position and
is not carried forward.

> [!WARNING] BENCHMARK-GATED — swapping the default solver, or adopting a joint fit, requires the
> M7 refuse-to-report gate (§9) to certify the fit *and* a flag-off/flag-on scoreboard
> non-regression. See notes `reference_forward_model_solvers_work` and
> [Error Budget & Falsification](error-budget-and-falsification.md).

---

## See also

- [Frontier Methods](frontier-methods.md) — the theory each technique here realizes.
- [Formal Spec](formal-spec.md) — the Lean theorems cited by the DERIVE and reliability tiers.
- [Error Budget & Falsification](error-budget-and-falsification.md) — the negative-result ledger
  (composition-fed τ, joint-beats-iterative, manifold-as-foundation).
- [Benchmarks & Reliability Workflows](benchmarks-reliability-workflows.md) — how the M7/M8 gates
  and derived thresholds are exercised operationally.
- [Impl: Literature Methods](impl-literature-methods.md) — OPC and the classical levers that sit
  alongside these novel techniques.
- [Architecture](architecture/index.md) — where `jitpipe`, `manifold`, and the physics sub-packages sit
  in the module map.
