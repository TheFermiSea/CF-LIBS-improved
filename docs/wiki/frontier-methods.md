---
slug: frontier-methods
title: "Frontier Methods We Are Deriving"
chapter: frontier-methods
order: 0
status: frontier
register: review
summary: >
  The active research frontier of CF-LIBS, framed by two goals (DED Ti-6Al-4V drift
  tracking by ratios; absolute composition generally). Nine theorem-licensed thrusts,
  each with a physics rationale, a falsifiable first experiment, the cflibs-formal theorem
  that licenses it, and an effort/dependency order. Load-bearing claim: the inversion
  algorithm is already at its floor; accuracy now lives in the atomic data, the reported
  coordinate (log-ratios not closure wt%), a measured n_e, and theorem-pinned refuse-to-report.
tags: [frontier, estimator-routing, log-ratio, self-calibration, electron-density, refuse-to-report, differentiable-forward, instrument-calibration, formal-methods]
updated: 2026-07-02
benchmarks_pre_reset: true
sources:
  - "@aragon2008"
  - "@ciucci1999"
  - "@tognoni2010"
  - "@aguilera2007"
  - "@aitchison1982"
  - "@ruffoni2014"
  - "@gigosos2003"
  - "@aragon2014"
  - "@kawahara2022"
  - "@cristoforetti2010"
  - "@hermann2018"
  - "@maali2019"
  - docs/research/physics-first-principles-audit.md
  - docs/research/accuracy-first-roadmap.md
  - docs/M-spec-estimator-routing-policy.md
  - docs/research/realtime/2026-06-20-realtime-plan-v4-real-data-accuracy.md
  - docs/research/realtime/varpro-datavoyager-analysis.md
  - docs/v4/overhaul/reference/exojax.md
  - docs/adr/ADR-0006-instrument-calibration-first-class.md
code_refs:
  - cflibs/inversion/physics/identifiability.py
  - cflibs/inversion/physics/reliability.py
  - cflibs/inversion/physics/self_absorption_observable.py
  - cflibs/inversion/physics/stark_ne.py
  - cflibs/inversion/physics/closure.py::ClosureEquation
  - cflibs/inversion/solve/iterative.py::IterativeCFLIBSSolver
  - cflibs/inversion/preprocess/response_correction.py
lean_refs:
  - CflibsFormal/SelfAbsorptionInverse.lean#selfAbsorption_breaks_identifiability
  - CflibsFormal/MatrixEffects.lean#recoveredComposition_ratio_matrix_invariant
  - CflibsFormal/SahaInverse.lean#saha_joint_identifiability
  - CflibsFormal/StarkBroadening.lean#stark_saha_lte_consistent
  - CflibsFormal/ErrorBudget.lean#temp_rel_error_eq
  - CflibsFormal/CompositionIdentifiability.lean#compositionIdentifiable
related: [formal-spec, error-budget-and-falsification, atomic-data-and-datasets, impl-novel-techniques, benchmarks-reliability-workflows, cf-libs-family]
supersedes: []
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Frontier Methods We Are Deriving

This chapter is the intellectual center of the wiki: the **theory of the methods we are
still deriving**, not the shipped pipeline. It is written in the review register — equation-
forward, cited, and explicitly labelled active research — and every thrust is pinned to (i)
the physics deficiency it repairs, (ii) a *falsifiable first experiment*, and (iii) the
`cflibs-formal` theorem that licenses it or forbids the naive alternative. The shipped
realizations of anything that has landed live in [impl-novel-techniques](impl-novel-techniques.md)
and [impl-literature-methods](impl-literature-methods.md); the *rejected* variants are recorded
once, with their falsification evidence, in
[error-budget-and-falsification](error-budget-and-falsification.md) and are never re-argued here.

> [!IMPORTANT] PRE-RESET NUMBERS — the atomic DB was rebuilt (ASD59, 203k lines); the
> composition/RMSE/wt% figures quoted below (e.g. the ~0.171 cross-DB RMSE, the 19–26 wt%
> SuperCam floor, the V/Ti 15.2→3.6 OPC gain, the 0.65 wt% SCCT result) come from
> pre-reset campaigns. **The mechanisms are retained; treat the magnitudes as dead** and
> re-measure on the current DB before quoting any number as live.

## The two goals that rank everything {#two-goals}

Every priority in this chapter is ranked against two *separate* program goals
([`project_ded_drift_tracking_goal`], [`project_formalization_optimization_mission`]):

- **(a) DED Ti-6Al-4V drift tracking** — a *constrained, known* element set {Ti, Al, V};
  **precision and ratios** matter far more than absolute wt%; the nominal feedstock is a
  prior; oxides/geology are out of scope.
- **(b) Absolute composition generally** — steel minors, geology, unknown matrices; the
  classical calibration-free quant problem [@ciucci1999; @tognoni2010].

The decisive empirical fact reframing the whole frontier is the controlled cross-database
round-trip in the real-data program (`docs/research/realtime/2026-06-20-realtime-plan-v4-real-data-accuracy.md` §3):
with the same line list on both sides, a clean inversion closes to RMSE $\approx 2.9\times10^{-6}$ —
**the algorithm floor is essentially zero** — while swapping the atomic line list alone
injects $\approx 0.171$ RMSE. *There is no meaningful accuracy left to win by improving the
solver arithmetic.* Accuracy now lives in four places: the **atomic data**, the **reported
coordinate** (log-ratios, not closure wt%), a **measured** rather than imputed $n_e$, and a
**theorem-pinned trust surface** (refuse-to-report). That is the shape of this chapter.

### Wavelength and symbol conventions {#conventions}

Wavelengths are quoted in **air** (the atomic DB stores air wavelengths per NIST/ASD; the
single `air_to_vacuum`/`vacuum_to_air` utility in `cflibs/core/` converts). All symbols
($T$, $n_e$, $E_k$, $g_k$, $A_{ki}$, $U_s(T)$, $N_s$, $\tau$, $C_s$, $F$) are the canonical
ones defined once in [formal-spec/notation](formal-spec.md) — this chapter never redefines
them. The canonical Boltzmann ordinate is always

$$
y \;=\; \ln\!\left(\frac{I_{ki}\,\lambda_{ki}}{g_k A_{ki}}\right)
\;=\; -\frac{E_k}{k_B T} \;+\; \ln\!\left(F\,\frac{hc}{4\pi}\,\frac{N_s}{U_s(T)}\right),
\qquad \text{slope} = -\frac{1}{k_B T},
$$

lean:`CflibsFormal/SahaInverse.lean#sahaBoltzmann_plot`. The factor $\lambda_{ki}$ is
load-bearing when it varies across the fit and must never be silently dropped.

## The research-frontier map {#frontier-map}

The nine thrusts, in the consolidated action order derived in the physics-first-principles
audit (its §"Consolidated action ordering"). "Effort" and "Depends" are the audit's; "Status"
is the frontier maturity tag; the engineering-decision tag (SAFE-NOW / BENCHMARK-GATED /
DESIGN-DECISION / AUDIT-FIRST) marks how it may be adopted.

| # | Thrust (section) | Goal | Effort | Depends on | Decision tag | Status |
|---|------------------|------|--------|-----------|--------------|--------|
| 1 | [Log-ratio tracking](#log-ratio-tracking) | (a) | Low | — | SAFE-NOW | frontier |
| 2 | [Formal-verified inversion](#formal-verified-inversion) | (a)(b) | Med | — | DESIGN-DECISION | frontier |
| 3 | [Estimator-routing policy](#estimator-routing-policy) | (a)(b) | Med | 2 | DESIGN-DECISION | frontier |
| 4 | [In-plasma relative-gA self-calibration](#ga-self-cal) | (a)(b) | Med | — | BENCHMARK-GATED | frontier |
| 5 | [Measured n_e (SB offset + Balmer)](#measured-ne-sb-offset) | (b), trust | Med | — | BENCHMARK-GATED | frontier |
| 6 | [Strict gates / refuse-to-report](#strict-gates) | (a)(b), trust | Low–Med | 2,5 | SAFE-NOW | frontier |
| 7 | [Instrument calibration first-class](#instrument-calibration-first-class) | in-house | Low–Med | — | AUDIT-FIRST | frontier |
| 8 | [Manifold emulator (S0-at-Tref)](#manifold-emulator) | speed | Med | — | DESIGN-DECISION | frontier |
| 9 | [Differentiable forward + structured GN](#differentiable-forward-gn) | speed, triage | Med | 7 | BENCHMARK-GATED | frontier |

Two further clusters — the **inverse Saha stage-III symmetry** (Cluster C) and the
**observable-anchored self-absorption corrector on the fit path** (Cluster D) — are cheap
correctness fixes covered inline under [Estimator routing](#estimator-routing-policy) and
[Strict gates](#strict-gates); their *rejected* naive forms (composition-fed per-line $\tau$;
single-gate cooling quadrature) are in
[error-budget-and-falsification](error-budget-and-falsification.md#part-3), not repeated here.

---

## 1. Log-ratio tracking: the near-free DED deliverable {#log-ratio-tracking}

**Status:** frontier · **Goal (a), top lever · SAFE-NOW · lowest effort.**

### Physics rationale

The default estimator normalises $\sum_s C_s = 1$ over the *detected* species set
(`ClosureEquation.apply_standard` in `cflibs/inversion/physics/closure.py`). Every fraction
shares the denominator $\sum_t \mathrm{rel}_t$, so any per-element intensity, atomic-data, or
self-absorption error in *one* element moves *all* fractions — the closure "mass slosh."
Formally the whole-vector error is bounded by the shared-denominator perturbation
(lean:`CflibsFormal/CompositionRobustness.lean#composition_abs_sub_le`), and the recovered
*absolute* fractions are matrix-dependent
(lean:`CflibsFormal/MatrixEffects.lean#recoveredComposition_absolute_matrix_dependent`).

But the DED deliverable is a **ratio**, and the ratio cancels the shared denominator exactly:

$$
\frac{C_V}{C_{Ti}} \;=\; \frac{\mathrm{rel}_V}{\mathrm{rel}_{Ti}}
\;=\; \frac{N_V/U_V(T)}{N_{Ti}/U_{Ti}(T)}\cdot\frac{U_V(T)}{U_{Ti}(T)}.
$$

The closure-normalised *ratio* is provably matrix- and detected-set-invariant
(lean:`CflibsFormal/MatrixEffects.lean#recoveredComposition_ratio_matrix_invariant`). The
proper coordinate for tracking is therefore the **Aitchison log-ratio** [@aitchison1982;
@egozcue2003]:

$$
\ell_{V/Ti} \;=\; \ln\!\frac{N_V}{N_{Ti}}, \qquad \ell_{Al/Ti} \;=\; \ln\!\frac{N_{Al}}{N_{Ti}},
$$

computed **directly from the closure relatives**, with $\sum C_s = 1$ demoted to an *optional
final projection* — never the estimator — for the tracking goal. This is the reinterpretation
of the program's own history: the DED V/Ti limiter improved 15.2 → 3.6 % *(pre-reset)* only
via one-point calibration [@cavalcanti2013], a per-matrix band-aid that was compensating for
absolute-fraction slosh the ratio never needed.

### Falsifiable first experiment

On Ti-6Al-4V (real SCCT + synthetic): perturb/down-weight the Ti (or Fe) line set and measure
the change in recovered V. **Prediction:** the *absolute* V fraction moves substantially (slosh)
while $\ell_{V/Ti}$ is stable — and the log-ratio is already as stable in closure output as in a
C-σ columnar solve, **falsifying** the "need C-σ for invariant ratios" claim (C-σ's real edge is
optical thickness, not denominator avoidance; see [cf-libs-family](cf-libs-family.md)). Compare
V/Ti scatter across sols under wt% vs log-ratio reporting.

### Licensing theorem

lean:`CflibsFormal/MatrixEffects.lean#recoveredComposition_ratio_matrix_invariant` (ratios are
matrix-invariant) and its absolute-fraction counterpart
`recoveredComposition_absolute_matrix_dependent` (why wt% is not). The ILR coordinate machinery
is the standard compositional-data infrastructure [@egozcue2003].

### Effort / dependency

**Lowest effort, top DED impact** — a pure output/reporting change plus a small benchmark, no
dependency on any other thrust. The composition-metric infrastructure already exists in
`cflibs/benchmark/` (Aitchison/ILR). Reporting must carry the ILR coordinate and its covariance,
not a re-projected wt% vector.

---

## 2. Formal-verified inversion: refuse-to-report as a proof obligation {#formal-verified-inversion}

**Status:** frontier · **Goal (a)(b) · DESIGN-DECISION.**

### Physics rationale

The `cflibs-formal` Lean spec is used as a **soundness envelope**: each estimator's correctness
is a theorem whose *hypotheses* are physical preconditions, and the runtime's job is to check
those preconditions before it runs the estimator. Where a precondition provably fails, the
correct output is **not a number but a refusal** — the M7 refuse-to-report path. This inverts the
usual "always emit a best-guess" posture: a result is emitted only when a theorem licenses it.

The keystone is the self-absorption identifiability wall:

> `selfAbsorption_breaks_identifiability` — there exist $N_1 \neq N_2$ and $\tau_1,\tau_2 \ge 0$
> with $I^{\text{thick}}(N_1,\tau_1) = I^{\text{thick}}(N_2,\tau_2)$. One measured intensity does
> **not** pin down $(N,\tau)$.

lean:`CflibsFormal/SelfAbsorptionInverse.lean#selfAbsorption_breaks_identifiability`. A single
thick line with unknown $\tau$ is non-identifiable *as a theorem*, so reporting a composition
from it is not conservative-but-imperfect — it is unlicensed. The same discipline covers the
composition and electron-density preconditions
(lean:`CflibsFormal/CompositionIdentifiability.lean#compositionIdentifiable`;
lean:`CflibsFormal/SahaInverse.lean#saha_joint_identifiability`, which requires an *observed* ion
stage — exactly what SuperCam SCCT lacks). This is the formal grounding of the reliability flag:
`overall_reliable` must never certify a result whose licensing hypothesis is violated (e.g. an
imputed $n_e$; see [§5](#measured-ne-sb-offset)).

The float mirrors of these theorems are the reference formulas in `oracle/check_fixtures.py`,
regression-tested in `tests/oracle/test_spec_regression.py`; the runtime gates live in
`cflibs/inversion/physics/identifiability.py` (routing preconditions) and
`cflibs/inversion/physics/reliability.py` (within-regime line ranking). This is the
theorem→float-mirror→runtime-gate chain that makes "formal-verified inversion" operational
rather than aspirational (see [formal-spec](formal-spec.md) for the full map).

### Falsifiable first experiment

Construct two forward spectra $(N_1,\tau_1)$ and $(N_2,\tau_2)$ that the theorem guarantees are
intensity-identical on a single thick line, feed both to the inversion, and confirm the
identifiability gate routes **both** to X (refuse) rather than returning two different
compositions. Then relax to a two-line curve-of-growth pair and confirm the gate now *permits*
the solve (τ recoverable). A pass is: refuse iff the licensing hypothesis is false, permit iff
true — no false certificates, no spurious refusals.

### Licensing theorem

The refusal itself is licensed by the *negative* theorem
`selfAbsorption_breaks_identifiability`; the permits by the *positive* well-posedness theorems
(`compositionIdentifiable`, `density_identifiability`, `saha_joint_identifiability`,
`thick_composition_identifiability`).

### Effort / dependency

Medium; the theorems and float mirrors exist, the work is wiring every emission path through the
identifiability gate (some already are). Depends on the routing policy ([§3](#estimator-routing-policy))
being explicit. This is a **DESIGN-DECISION**: accuracy-changing results through a
non-identifiable gate are forbidden by project policy (the `cflibs-verification-gate`).

---

## 3. Estimator-routing policy: regime → estimator → theorem {#estimator-routing-policy}

**Status:** frontier · **Goal (a)(b) · DESIGN-DECISION.**

### Physics rationale

Different physical regimes are *identifiable* by different estimators, and each estimator is a
sound left-inverse only inside its regime. The routing policy
(`docs/M-spec-estimator-routing-policy.md`) makes the choice explicit and pins each decision to
the theorem that licenses it. The keystone decision table:

| # | Regime (precondition) | Estimator | Licensing theorem(s) |
|---|-----------------------|-----------|----------------------|
| **R1** | Optically thin, few clean lines ($\tau\approx0$, low noise, ≥1 isolated line/species) | Classic / C-σ (one inverse) | `Classic.classic_sound`, `classicDensity_recovers` |
| **R2** | Thin, **many noisy** lines with energy spread $\sum(E_k-\bar E)^2>0$ | OLS Boltzmann plot | `Alt.leastSquares_sound`, `olsSlope_noise_gain` |
| **R3** | Self-absorbed, $\tau$ **known** (or fit from a pair) | Classic ∘ SA(τ)-correction | `Alt.selfAbsorbed_eq_classic_corrected`, `selfAbsorbed_corrects_bias` |
| **R4** | Thick, $\tau$ unknown, ≥2 lines of differing width/strength | CoG ratio inversion (recover τ → R3) | `cogRatio_strictAntiOn`, `cogRatio_injOn` |
| **R5** | $n_e$ from an ion/neutral stage ratio | Saha inversion $n_e = S(T)/R$ | `Saha.electronDensity_antitone`, `electron_density_identifiability` |
| **R6** | $n_e$ reliability / LTE gate (Stark **and** Saha available) | Cross-check + McWhirter floor | `stark_saha_lte_consistent`, `mcWhirterBound_mono_T` |
| **X** | Single thick line, $\tau$ unknown | **REFUSE / flag** | `selfAbsorption_breaks_identifiability` |

The escape factor is $SA(\tau) = (1-e^{-\tau})/\tau \in (0,1]$, so R3 is literally "classic, then
divide out $SA(\tau)$"; in the thin limit it degrades gracefully to R1
(`selfAbsorbed_eq_classic_thin`). Two disciplines are non-negotiable:

1. **C-σ is not a second method.** `Alt.CSigma.csigmaComposition_eq_classicComposition` is an
   *unconditional identity*: C-σ and the classic per-species inverse are the same algebraic
   left-inverse in two packagings. Do **not** count them as two votes or average them — their
   agreement carries zero independent corroboration. Keep exactly one, chosen by numerical
   conditioning only. The genuinely independent same-spectrum cross-check is **OLS vs classic**
   (`Alt.leastSquares_agrees_classic`), which differs off the noise-free fixpoint and is the
   *only* legitimate two-estimator data-quality signal.
2. **Reliability ranks lines, not estimators.** Once a regime is chosen, the conditioning result
   `temp_rel_error_eq` gives the *exact* identity $|\Delta T|/T = k_B \hat T\,|\Delta\beta|$
   (lean:`CflibsFormal/ErrorBudget.lean#temp_rel_error_eq`); a two-line temperature error scales
   as $2/|\Delta E|$ in the energy spread, so `reliability.py` prefers wide-$|\Delta E|$,
   high-SNR line sets. It never changes *which* estimator runs.

**Cluster C fold-in (inverse Saha stage-III symmetry).** The forward populates three stages
($\,\mathrm{denom}=1+S_1+S_1 S_2$) but the inverse abundance multiplier truncates at
$1+\max(S,0)$, a genuine forward/inverse asymmetry that *explains the perverse Cr-partition
regression* (improving stage-III partitions on the forward side while the inverse drops
stage-III mass widens the mismatch). The fix is to make the inverse ladder identical to the
forward: $\sum_z N_z = N_I\sum_z\prod_{k\le z}(S_k/n_e)$ and
$\langle Z\rangle=(S_1+2S_1 S_2)/(1+S_1+S_1 S_2)$, fetching $\chi_{III}/U_{III}$ exactly as the
forward already does. Magnitude is strongly $T$-dependent (up to ~40 % for hot-core/high-$Z$; a
few % at late-gate steel $T$).

### Falsifiable first experiment

Round-trip test: synthesise a Ti-6Al-4V + Cr spectrum with the forward three-stage model at
$T=1.2$ eV, $n_e=2\times10^{16}$ cm$^{-3}$, then invert. **Prediction:** the current two-stage
inverse under-recovers Cr/Ti by the neglected $S_1 S_2$ fraction; the three-stage inverse closes
the round-trip to $<0.3$ wt%; low-$f_{III}$ elements (Cu, Ni) are unchanged; and re-running at
late-gate steel $T$ confirms the effect is small there — a clean $T$-dependence signature.
Separately, feed a single thick line with unknown τ and confirm route X (refuse), then a
two-line CoG pair and confirm route R4 recovers a unique τ (monotone ratio).

### Licensing theorem

The whole table is theorem-pinned; the two prunes rest on `csigmaComposition_eq_classicComposition`
(collapse C-σ) and `stark_saha_lte_consistent` (the only real $n_e$ cross-check). The stage-III
extension mirrors the forward Saha chain (`MatrixEffects.lean#sahaSplit_sum`).

### Effort / dependency

Medium for the policy wiring; the stage-III fix is **low effort** (mirror the forward code into
the inverse) and best done *after* the $n_e$ fix ([§5](#measured-ne-sb-offset)) so the multiplier
does not ride a wrong $n_e$.

---

## 4. In-plasma relative-gA self-calibration: the atomic-data BIAS, not variance {#ga-self-cal}

**Status:** frontier · **Goal (a)(b), dominant real-data floor · BENCHMARK-GATED.**

### Physics rationale

CF-LIBS composition is $C_s \propto U_s(T)\,e^{\text{intercept}_s}$, and the intercept carries the
**absolute scale** of that species' $g_k A_{ki}$ values. A poorly-graded transition probability is
a fixed 50–100 % *bias* in an unknown direction, correlated within a multiplet and within a source
paper — **not** a zero-mean independent Gaussian. The shipped code models it as independent random
variance and *down-weights* poorly-graded lines (inverse-variance WLS). You cannot average or
weight your way out of a scale error: down-weighting a biased line lowers its leverage but leaves
the surviving (also-biased) ensemble mean shifted. This is why the one-point-calibration F-factor
"works" on matrix-matched standards — it is band-aiding a per-species coherent scale bias
[@cavalcanti2013; @borduchi2019] — and why **Kurucz beat NIST on SuperCam** *(pre-reset:
~5–6 wt%)*: a single-source list carries *one coherent, closure-absorbable* scale error, versus
NIST's *many independent* per-source offsets that closure cannot absorb.

Two design changes attack the root cause:

1. **Absolute-scale anchoring** via lifetime/branching sum-rules. For each species, renormalise the
   adopted $A_{ki}$ set so the summed decay rate from each upper level matches an independently
   measured radiative lifetime, $A_{ki}=\mathrm{BR}_{ki}/\tau_k$ — the Wisconsin-group practice, in
   which relative branching fractions and laser lifetimes are accurate to a few percent while the
   absolute scale is not [@ruffoni2014]. The intercept then carries a lifetime-limited (~few %)
   scale error the closure can absorb.
2. **In-plasma relative-gA self-calibration.** Group observed lines by shared upper level (already
   in the DB `energy_levels`) and use the exact $T$-, $n_e$-, population-**independent** identity

   $$
   \frac{I_i}{I_j} \;=\; \frac{g_i A_i \lambda_j}{g_j A_j \lambda_i}
   $$

   to measure the relative-$A_{ki}$ residual per group and refine each species' relative $g_kA_{ki}$
   to minimise cross-line (and cross-shot) residual *before* the slope fit. This reproduces "why
   Kurucz beat NIST" **from your own plasma**, no external standard, and replaces the fabricated
   NIST-relative-intensity→grade uncertainty with a physically derived per-line error. It is
   unaffected by the anchoring falsification below because it measures whatever lines are actually
   used. The multiplet-aware Boltzmann extension [@volker2023multiplet] is the companion for
   handling blended multiplets in the same fit.

> [!CAUTION] FALSIFIED: Lifetime-anchoring (Lawler/Den Hartog overlay) yields a coherent scale correction on the pipeline's selected lines
> - **Claim:** renormalising each species' $A_{ki}$ to Wisconsin-group lifetime sum-rules corrects a real, coherent scale bias vs NIST and lowers held-out RMSEP.
> - **Predicted:** $\ln(A_{\text{anchored}}/A_{\text{NIST}})$ materially non-zero per species; held-out RMSEP drops.
> - **Observed:** median $\ln(A_{\text{anchored}}/A_{\text{NIST}}) \approx 0$ for every covered species — **NIST ASD already sources the Wisconsin gf-values there** [@ruffoni2014], so no coherent correction exists; coverage of the *actually-selected* lines is the binding limiter (Fe I/Fe II: 0 %), and a net held-out **regression** (+0.40 wt%, pre-reset) appeared via an Fe selection cascade.
> - **Verdict:** REJECTED as a default path (kept OPT-IN, off the default). The ~0.171 cross-DB RMSE is **not** Lawler-vs-NIST scale error; suspects are the specific older-source gf's of the selected persistent lines (testable per-line vs an independent Kurucz ingest) or a non-$A_{ki}$ mechanism.
> - **Evidence:** physics-first-principles-audit.md Addendum (2026-07-02); 2,304-line lab-anchored overlay DB.
> - **Date:** 2026-07-02

The surviving, unfalsified arm is the **in-plasma relative self-calibration** (2) plus a
**curated, gf-grade-aware line list** — the roadmap's re-pointed lever
(`docs/research/realtime/2026-06-20-realtime-plan-v4-real-data-accuracy.md` §5.2): build
NIST-A/B-only, Kurucz-complete, and VALD3 bundles over the same line set, carry a per-line gf-grade
column as a first-class line-selection input, and down-weight Kurucz-back-filled (ungraded) and
weak-emitter lines. Completeness wins the aggregate fit on dense real spectra (Kurucz's LIN/POS
completeness back-fills weak lines NIST lacks); per-line accuracy is what would let a *curated* list
beat both [@ruffoni2014]. ExoJAX is the offline ingestion/validation oracle for these bundles
[@kawahara2022], never a hot-path dependency.

### Falsifiable first experiment

On steel_266 and SuperCam SCCT: (1) compute per-species Boltzmann-plot RMS residual with NIST vs
Kurucz $g_kA_{ki}$ — predict Kurucz's within-species scatter is materially lower and tracks the
~0.171 loss reduction; (2) run the relative-gA self-calibration on same-upper-level groups and
predict the refined relative $g_kA_{ki}$ lowers cross-shot intercept residual and held-out RMSEP
*without* adopting Kurucz wholesale. A curated NIST-A/B + graded-back-fill bundle should beat raw
Kurucz on the *controlled* cross-DB test.

### Licensing theorem

`CompositionIdentifiability.lean#compositionIdentifiable` assumes correct/identical $A_{ki}$ (the
`hAeq` hypothesis), so the estimator's soundness proof is *vacuous* under real $A_{ki}$ bias — the
formal statement that atomic-data scale error sits **outside** the verified envelope until anchored
or self-calibrated. This is the frontier where physics and formal-methods meet: the theorem tells
you *which hypothesis you are violating*.

### Effort / dependency

Medium (the self-calibration pass) to high (the curated bundle + VALD evaluation). **Highest value
for both goals, independent of the other clusters.** BENCHMARK-GATED: any change to the line list or
line weighting has regressed the scoreboard before and must run a flag-off/flag-on non-regression
benchmark ([benchmarks-reliability-workflows](benchmarks-reliability-workflows.md)).

---

## 5. Measured n_e: Stark/Balmer seed and the SB inter-stage offset {#measured-ne-sb-offset}

**Status:** frontier · **Goal (b) + trust surface · BENCHMARK-GATED · Cluster B.**

### Physics rationale

On real data $n_e$ is **not measured** — it is imputed from an isobaric pressure balance at
hard-coded Earth STP ($P=101325$ Pa in `iterative.py`), which yields
$n_e\sim1.4$–$2.7\times10^{17}$ cm$^{-3}$ against realistic late-gate LIBS
$1$–$3\times10^{16}$ — roughly a decade high — and every ion line is then remapped to the neutral
plane with that wrong $n_e$. Meanwhile the multi-element Saha-Boltzmann plot *already stacks*
neutral+ion lines on one plane [@aguilera2007] but consumes $n_e$ as an **input** and discards the
inter-stage ordinate offset that *measures* it. The offset is exactly the log-Saha shift:

> `sahaBoltzmann_shift_eq_log_saha` — the vertical split between the neutral and ion intercepts on
> the shared-slope plane equals $\ln[S_z(T)/n_e]$, so once $T$ is fixed by the common slope, $n_e$
> is recoverable from the split.

lean:`CflibsFormal/SahaInverse.lean#sahaBoltzmann_shift_eq_log_saha`, and jointly identifiable with
composition via `saha_joint_identifiability` **whenever an ion stage is observed**. Three design
changes:

1. **Invert the SB inter-stage offset** for $n_e$ whenever any element shows both a neutral and an
   ion line (steel Fe I + Fe II) — choose $n_e$ minimising the neutral/ion intercept split
   [@aguilera2007]. Reuses machinery already in `_fit_saha_boltzmann_graph`.
2. **Balmer (Hα/Hβ) Stark $n_e$** via the Gigosos-Cardeñoso computer-simulated width relation
   $w \propto n_e^{0.68}$, nearly $T$- and atomic-data-independent [@gigosos2003]; Hβ 486.1 nm is
   preferred for weaker self-absorption. H is present from hydrated minerals, organics, and
   shield-gas/ambient — the one diagnostic needing neither ion lines nor tabulated Stark-B widths.
   For non-hydrogenic lines the STARK-B semiclassical widths [@sahalbrechot2015] and the NIST
   critically-evaluated experimental Stark data [@konjevic2002] supply the $w(n_e,T)$.
3. **Retire Earth-STP pressure balance** to a last resort, and **never let `overall_reliable` pass
   on an imputed $n_e$** — the trust-surface link to [§6](#strict-gates).

**Second-order for DED by cancellation; first-order for the trust surface.** For the constrained,
all-low-IP DED target, $n_e$ enters the ion→neutral correction as $\times n_e$ and the abundance
multiplier as $\times 1/n_e$; these cancel for strongly-ionised low-IP metals (total abundance
$\propto n_e^0$), so a full-decade $n_e$ sweep moved Ti-6Al-4V by only ~1–2 % *(pre-reset)*. The
$n_e$ work therefore matters for (i) the LTE/McWhirter trust surface and (ii) ion-observed minors
(steel Cr) — **not** for the DED composition magnitude. Do **not** spend the accuracy budget chasing
$n_e$ magnitude for DED (that dead-end is recorded in
[error-budget-and-falsification](error-budget-and-falsification.md#part-3)); do it for trust and
minors.

### Falsifiable first experiment

On steel_245nm: solve with (a) pressure-balance $n_e$ vs (b) $n_e$ from the SB inter-stage offset;
compare recovered Fe/Cr against certified values and the neutral-vs-ion intercept residual.
**Prediction:** (b) reduces the intercept split and shifts multi-stage majors; on SCCT (zero ion
lines) both give identical composition, confirming the offset path is *inactive, not harmful* there.
Separately, on any spectrum with a 656.3/486.1 nm line, compare Gigosos-Balmer $n_e$ to
pressure-balance (predict ~decade disagreement) and check whether it flips the McWhirter verdict.

### Licensing theorem

`SahaInverse.lean#sahaBoltzmann_shift_eq_log_saha` (offset → $n_e$) and `#saha_joint_identifiability`
(joint with composition given an observed ion stage). The reliability cross-check is
`StarkBroadening.lean#stark_saha_lte_consistent` ([§6](#strict-gates)).

### Effort / dependency

Medium. The SB-offset inversion is low-medium (reuses existing graph machinery); the Balmer path is
a new estimator (medium). Independent of Clusters A/D. Higher priority for goal (b) and the trust
surface than for DED composition.

---

## 6. Strict gates: LTE-validity, reliability, and refuse-to-report as a first-class output {#strict-gates}

**Status:** frontier · **Goal (a)(b) + trust · SAFE-NOW.**

### Physics rationale

The McWhirter criterion is *necessary, not sufficient* for LTE; the temporal/spatial relaxation
criteria of Cristoforetti et al. [@cristoforetti2010; @cristoforetti2013] and the collisional-
radiative departure quantification of Pietanza et al. [@pietanza2010] (~10 % at
$n_e\sim10^{18}$–$10^{19}$, ~42 % at $10^{15}$ cm$^{-3}$) define a validity envelope the fit must
respect. The blackbody-limit diagnostic of Hermann et al. gives a cheap LTE check from the strongest
lines and simultaneously self-calibrates the response [@hermann2018], and the SuperCam-specific
plasma-diagnostic envelope [@manelski2024] closes the gap the generic-lab criteria leave for
planetary standoff LIBS. The gate logic (R6 in [§3](#estimator-routing-policy)):

1. Compute $n_e^{\text{Stark}}$ (`stark_ne.py`) and $n_e^{\text{Saha}}$ (R5).
2. **Disagreement** ⇒ the LTE/single-zone assumption is suspect ⇒ downgrade / flag (do not certify
   `overall_reliable`).
3. **Below McWhirter** $n_e < 1.6\times10^{12}\sqrt{T}\,(\Delta E)^3$ (monotone in $T$ and $\Delta E$,
   using the resonance-line $\Delta E$ convention, [`reference_mcwhirter_delta_e_physics`]) ⇒ LTE
   invalid ⇒ flag.
4. Both agree **and** clear McWhirter ⇒ a single trusted $n_e$ is licensed.

The consistency theorem makes the empirical status explicit:

> `stark_saha_lte_consistent` — *if* the Stark-width $n_e$ and the Saha-ratio $n_e$ agree **and** the
> result clears the McWhirter bound, then a single consistent $n_e$ exists satisfying both forward
> laws. The agreement hypothesis is *assumed*, not proven — which is exactly why it is a **runtime
> gate**, not an identity.

lean:`CflibsFormal/StarkBroadening.lean#stark_saha_lte_consistent`,
`#mcWhirterBound_mono_T`, `#mcWhirterBound_mono_dE`. Because Stark and Saha consume *different*
observations (a WIDTH vs a RATIO), their agreement is genuine evidence — the second of the only two
legitimate cross-checks in the pipeline (the first is OLS-vs-classic, [§3](#estimator-routing-policy)).

**M7 refuse-to-report is a first-class output.** When identifiability fails (route X) or the LTE
gate is not cleared, the pipeline emits a structured *refusal with reason* — not a silent
best-guess. Per-element M8 reliability flags carry the same discipline down to individual elements.
This is what makes "absolutely reliable" a defensible claim: the pipeline reports a number **only**
when a theorem licenses it and the trust gate clears.

**Cluster D fold-in (observable-anchored self-absorption on the fit path).** Every *inference*
forward is optically thin with no fitted optical-depth degree of freedom, so the prior-free
full-spectrum fit rides $T$ to a box edge to fake the missing saturation. The audited fix is to wire
the **existing observable-anchored** corrector (doublet-ratio + Planck-ceiling curve-of-growth
[@volker2023multiplet], and the Planck-function $I_{\text{thin}} = -B\ln(1-I/B)$ inversion of
Völker & Gornushkin, `@volker2023`) into the full-spectrum/joint/Bayesian objective — pre-correcting
thick observed lines to their thin-equivalent intensities. The naive composition-fed per-line $\tau$
is a **verified positive-feedback dead-end** (it made ChemCam BHVO-2 worse) and lives in
[error-budget-and-falsification](error-budget-and-falsification.md#part-3), not here.

### Falsifiable first experiment

Run the joint/full-spectrum solver twice on steel_266 / SuperCam Ti-6Al-4V: current thin objective vs
observable-corrector-enabled. **Prediction:** the corrected fit reduces the V over-estimate and the
Ti/Al ratio error and stops $T$ pinning at the box edge (inspect the T-at-bound rate on held-out).
Separately, inject a spectrum below McWhirter and confirm the gate flags rather than reports; inject
one that clears it and confirm certification.

### Licensing theorem

`stark_saha_lte_consistent` (the gate) and `selfAbsorption_breaks_identifiability` (the refuse-to-
report wall, [§2](#formal-verified-inversion)). The error-budget conditioning
`temp_rel_error_eq`/`_le` (lean:`CflibsFormal/ErrorBudget.lean#temp_rel_error_le`) sets the line-
quality thresholds the gate enforces.

### Effort / dependency

Low–Med; the McWhirter validator and Stark estimator exist. Depends on [§5](#measured-ne-sb-offset)
supplying a *real* $n_e$ for the cross-check (a gate on an imputed $n_e$ is itself a trust defect).

---

## 7. Instrument calibration as a first-class input {#instrument-calibration-first-class}

**Status:** frontier · **in-house instruments · AUDIT-FIRST · ADR-0006.**

### Physics rationale

Calibration-free removes exactly **one** instrumental term. The integrated line intensity is

$$
I_{ki} = F\cdot\frac{hc}{4\pi\lambda_{ki}}\,A_{ki}\,g_k\,\frac{N_s}{U_s(T)}\,e^{-E_k/k_BT},
$$

and the *scalar, wavelength-independent* $F$ (plasma volume in view, solid angle, throughput,
detector gain) cancels under closure $\sum_s C_s=1$. **That is the entire content of
"calibration-free": $F$ cancels; nothing else does** (ADR-0006 §1). Two instrumental effects
survive:

1. **Wavelength-dependent spectral response $E(\lambda)$.** Grating efficiency, optics
   transmission, and detector QE make $I_{\text{meas}}(\lambda)=E(\lambda)\,I_{\text{true}}(\lambda)$,
   contributing $\ln E(\lambda_{ki})$ to the Boltzmann ordinate. Because lines of different $E_k$ sit
   at different $\lambda_{ki}$, $E(\lambda)$ injects an **$E_k$-correlated perturbation that rotates
   and scatters the Boltzmann plot** — biasing slope ($\to T$) *and* intercept ($\to$ composition).
   It does **not** cancel under closure (wavelength-dependent, and different species populate
   different regions). **$E(\lambda)$ is the dominant un-removed instrumental systematic in
   quantitative CF-LIBS** (ADR-0006 §1.2).
2. **Line-spread function (LSF).** Area-conserving under convolution, so it leaves the
   integrated-intensity intercept invariant *in principle*; the residual is the per-channel /
   per-segment LSF (constant-FWHM-in-nm per grating segment vs one global resolving power), a real
   but sub-wt% refinement that barely bites neutral-dominated DED alloys.

ADR-0006's decision is to model the spectrometer as an explicit, provenance-tracked
`InstrumentCalibration` object consumed **identically** by every forward and inversion path
(legacy solver, differentiable hot path, jitpipe, Bayesian, manifold/ExoJAX reference), calibrating
*relative* response only (never absolute radiometry). The formal grounding: `ForwardMap.lean`
hard-codes a *scalar* $F_{\text{cal}}$, so a wavelength-dependent $E(\lambda)$ sits **outside every
closure theorem** — the theorem-level statement that $E(\lambda)$ is unmodeled. The frontier method
is **branching-ratio auto-recovery**: jointly infer a smooth $E(\lambda)$ (with a per-grating seam
step) from same-upper-level line pairs, whose *true* intensity ratio is fixed by atomic constants,
so any measured deviation is $E(\lambda_i)/E(\lambda_j)$ [@hermann2018]. This is the same physics as
the relative-gA self-calibration ([§4](#ga-self-cal)) read in the wavelength direction, and the two
must be co-fit to avoid absorbing one into the other.

> [!CAUTION] DEAD-END / DO-NOT: re-applying $E(\lambda)$ to vendor-corrected products.
> `SpectralResponseCorrection` is correctly **default-off** for ChemCam/SuperCam CL5/CCS products —
> they are vendor-radiometrically corrected, so re-applying double-corrects. Auto-recovery is for
> *in-house* spectra lacking a supplied curve only.

### Falsifiable first experiment

Inject a known $E(\lambda)$ with a channel step into a clean synthetic spectrum and confirm
composition error scales with the step size; then branching-ratio-recover $E(\lambda)$ on an
*in-house* dataset and re-benchmark, predicting recovered composition converges toward the
supplied-curve result. Cross-link the atomic-data ingest ([atomic-data-and-datasets](atomic-data-and-datasets.md))
because the branching-ratio anchors need the same-upper-level groupings.

### Licensing theorem

`ForwardMap.lean` scalar-$F_{\text{cal}}$ closure (the negative result: $E(\lambda)$ is outside it),
and the same-upper-level intensity-ratio identity used in [§4](#ga-self-cal).

### Effort / dependency

Low–Med; the `SpectralResponseCorrection` math exists (`cflibs/inversion/preprocess/response_correction.py`),
the work is the auto-recovery joint fit. **AUDIT-FIRST**: confirm default-off for vendor-corrected
products before enabling anywhere. See [architecture](architecture/index.md) for how
`InstrumentCalibration` threads every path.

---

## 8. Manifold emulator: offline reference-S0-at-Tref with per-T scaling {#manifold-emulator}

**Status:** frontier · **speed / pre-compute · DESIGN-DECISION.**

### Physics rationale

> [!CAUTION] CONTRADICTED FRAMING: the manifold is **not** the accuracy foundation. Nearest-neighbour
> manifold inference is a *speed* pattern; accuracy comes from the physics forward and the atomic
> data. Do not treat a stale manifold as ground truth.

The design pattern worth carrying forward from ExoJAX [@kawahara2022] is the **two-step line-strength
factorisation** (`docs/v4/overhaul/reference/exojax.md` §2.2): compute a reference line strength
$S_0$ at a fixed $T_{\text{ref}}$ **once** (on CPU), then apply a cheap per-$T$ scaling in the hot
loop:

$$
S_0 = \frac{-A_{ki}\,g_k\,e^{-c_2 E_{\text{low}}/T_{\text{ref}}}\,\mathrm{expm1}(-c_2\nu/T_{\text{ref}})}
{8\pi c\,\nu^2\,U(T_{\text{ref}})},
\qquad c_2 = hc/k_B = 1.4388\ \text{cm·K},
$$

$$
\ln S(T) = \ln S_0 - c_2 E_{\text{low}}\!\left(\tfrac{1}{T}-\tfrac{1}{T_{\text{ref}}}\right)
+ \ln\frac{1-e^{-c_2\nu/T}}{1-e^{-c_2\nu/T_{\text{ref}}}} - \ln\frac{U(T)}{U(T_{\text{ref}})}.
$$

Storing $\ln S_0$ per line (computed once) and scaling in the log domain with `expm1` for the
stimulated-emission term avoids float32 overflow for weak lines at low $T$ and decouples the line
database from the GPU (the PreMODIT "Line Basis Density at $T_{\text{ref}}$" idea). This is the
principled amortisation for the manifold generator and — critically — for the **precomputed
self-absorption table** that Qiu et al. show is the highest-leverage physics-only combination when
paired with one-point calibration [@qiu2026]. For LIBS the broadening term is electron-impact
(Stark), scaled by $n_e$ [@sahalbrechot2015] — never the air-broadening HITRAN parameters.

### Falsifiable first experiment

Generate a manifold slice with the $S_0$-at-$T_{\text{ref}}$ factorisation and compare, line-by-line,
against a direct full-recompute forward across the $(T,n_e)$ grid; **prediction:** agreement to
float tolerance ($\le 10^{-5}$ relative) at $\ge 10\times$ throughput, with the weak-line float32
overflow (present in a naive linear-domain recompute) absent. Then benchmark manifold nearest-neighbour
inference against the direct solver on held-out spectra and confirm it is a *speed* win with *no*
accuracy claim of its own.

### Licensing theorem

No new inverse theorem — the factorisation is an identity in the forward line-strength law
(`ForwardMap.lean`/`ForwardMapEnergy.lean`); its correctness is the algebraic equality above, oracle-
testable against the direct forward. The manifold's *outputs* are still gated by the same
identifiability/reliability theorems as any other estimator.

### Effort / dependency

Medium; ExoJAX supplies the reference implementation offline. Independent of the accuracy thrusts —
this buys latency, not accuracy.

---

## 9. Differentiable forward + structured Gauss-Newton (K=1) {#differentiable-forward-gn}

**Status:** frontier · **speed / real-time triage · BENCHMARK-GATED.**

### Physics rationale

For the real-time hot path, the forward model is made differentiable (JAX) and inverted with a
**structured (analytic-block) Jacobian and a single damped Gauss-Newton step (K=1)**:

$$
\Delta\theta = \big(J^\top J + \lambda I\big)^{-1} J^\top r,
\qquad \theta = (T, n_e, \{C_s\}),
$$

with $r$ the residual against the measured spectrum and $J$ assembled from analytic Saha-Boltzmann
blocks rather than full autodiff. This is optimal-estimation in the Rodgers sense (a linearised
inverse with Tikhonov damping) [@rodgers2000], warm-started so one step suffices. The DataVoyager
latency/accuracy sweep (`docs/research/realtime/varpro-datavoyager-analysis.md`, verified section)
settles the method choice:

| Method | comp_rmse range | latency range (µs) |
|--------|-----------------|--------------------|
| structured GN | 0.05783–0.06511 | 321–1097 |
| autodiff | 0.05783–0.06511 | 348–1509 |
| VarPro | 0.22902–0.23465 | 710–1712 |

- **Structured K=1 dominates.** Best <1 ms config is `structured, K=1, roi_ch=2000` at 353.9 µs,
  comp_rmse 0.05783 *(synthetic; pre-reset)*. Autodiff gives *identical* accuracy but is slower;
  $K>1$ buys nothing on this grid (strictly dominated).
- **Forward-difference lever.** The structured Jacobian is the "same solve" as autodiff at lower
  latency — the analytic blocks replace the autodiff trace, and a single forward-difference refinement
  of the ill-conditioned rows is enough.

> [!CAUTION] FALSIFIED: Variable projection (VarPro) is competitive for single-shot LIBS composition.
> - **Claim:** eliminating the linear composition parameters (VarPro) improves the latency/accuracy
>   trade-off for single-shot inversion.
> - **Predicted:** lower comp_rmse at equal or lower latency.
> - **Observed:** VarPro's best comp_rmse is **0.229 (~4× worse** than structured/autodiff's 0.058)
>   at any $K$/ROI; it fits $T$ and $n_e$ *well* at high resolution but **mis-apportions composition**,
>   and it degenerates entirely at roi_ch=500 (NaN; $n_e$-error 523 at K=1). It is also the slowest.
> - **Verdict:** REJECTED for single-shot composition; retained only as a $T,n_e$-diagnostic curiosity.
> - **Evidence:** `docs/research/realtime/varpro-datavoyager-analysis.md` (verified section; DataVoyager's
>   own Pareto synthesis was partly hallucinated and is corrected there).
> - **Date:** 2026-06-20

**ROI-conditional sub-ms, not unconditional.** The sub-ms claim holds only at ~250–500-channel ROI
(min 407 µs on the real Voigt path); a 1500-channel window is ~1.2 ms and the full window ~2.0 ms.
And it is a **triage/warm-start** path, not the accuracy path: on real spectra the legacy integrated-
intensity solver *beat* structured GN 4/5 (median 0.076 vs 0.151 wt%, ~2× — pre-reset), because the
real-data floor is atomic-data-limited, not solver-limited. The honest scope: **sub-ms at
250–500-channel ROI, plausible-not-calibrated composition, ~2000× faster than legacy, NaN-free across
80 adversarial runs**, wired *behind* the Stark-$n_e$ seed ([§5](#measured-ne-sb-offset)) and the
upstream SA corrector ([§6](#strict-gates)). Full-spectrum forward-fitting CF (Monte Carlo /
MERLIN-style) [@gornushkin2022; @favre2024merlin] is the accuracy-oriented sibling that folds SA and
response into one forward — a heavier, offline path.

### Falsifiable first experiment

Re-run the structured / autodiff / VarPro sweep on the **current (ASD59)** DB and confirm: (i)
structured K=1 reproduces autodiff accuracy at lower latency; (ii) VarPro stays ~4× worse on
composition and NaNs at small ROI; (iii) the legacy solver still beats structured GN on *real* held-out
composition, isolating the atomic-data floor from the solver. A pass requires the structured-vs-autodiff
oracle agreement $\le 10^{-5}$ against `jacfwd`.

### Licensing theorem

No inverse-existence theorem beyond the composition/electron-density identifiability results the GN
step shares with every estimator (`compositionIdentifiable`, `electron_density_identifiability`); the
Gauss-Newton conditioning is governed by the same error-budget bounds
(lean:`CflibsFormal/ErrorBudget.lean#temp_rel_error_le`) that rank line quality — wide-$|\Delta E|$,
high-SNR lines make $J^\top J$ well-conditioned.

### Effort / dependency

Medium; the structured Jacobian and `lax.scan` K=1 loop are largely built and oracle-tested. Depends on
[§7](#instrument-calibration-first-class) (the same `InstrumentCalibration` must feed the differentiable
forward). BENCHMARK-GATED against the legacy solver on real held-out data — latency is deferred until
the pipeline is accuracy-robust ([`feedback_accuracy_before_latency`]).

---

## What correct code MUST do {#must-do-checklist}

A frontier method is only "landed" when its shipped realisation satisfies **all** of the following;
each is a benchmark- or theorem-checkable obligation, not a stylistic preference.

- [ ] **Report the log-ratio coordinate for tracking.** Emit $\ln(N_i/N_j)$ (with covariance) from the
  closure relatives; treat $\sum C_s=1$ as an optional final projection, never the tracking estimator
  ([§1](#log-ratio-tracking); `recoveredComposition_ratio_matrix_invariant`).
- [ ] **Gate every emission through identifiability.** Route a single thick line with unknown $\tau$ to
  X (refuse); permit only when a positive well-posedness theorem holds
  ([§2](#formal-verified-inversion); `selfAbsorption_breaks_identifiability`).
- [ ] **Keep exactly one of classic/C-σ; never count them as two votes.** Use OLS-vs-classic and
  Stark-vs-Saha as the *only* cross-checks, each a gate not an assumed pass ([§3](#estimator-routing-policy)).
- [ ] **Make the inverse Saha ladder identical to the forward** (stage III where $f_{III}$ is
  non-negligible at the fitted $T,n_e$); do not let the multiplier ride a wrong $n_e$ ([§3](#estimator-routing-policy)).
- [ ] **Treat $g_kA_{ki}$ error as a systematic bias.** Prefer in-plasma relative-gA self-calibration on
  same-upper-level groups and a gf-grade-aware curated line list over inverse-variance down-weighting;
  do **not** re-attempt naive lifetime-anchoring as a default ([§4](#ga-self-cal)).
- [ ] **Measure $n_e$; never impute it and certify.** Invert the SB inter-stage offset or the Balmer
  Stark width; `overall_reliable` must fail on an imputed $n_e$ ([§5](#measured-ne-sb-offset), [§6](#strict-gates)).
- [ ] **Emit refuse-to-report as a structured, reasoned output** (M7) with per-element M8 flags; clear
  McWhirter and the Stark-Saha cross-check before certifying ([§6](#strict-gates)).
- [ ] **Wire the observable-anchored SA corrector into the fit path**; never add composition-fed per-line
  $\tau$ ([§6](#strict-gates); dead-end recorded in [error-budget-and-falsification](error-budget-and-falsification.md#part-3)).
- [ ] **Consume one provenance-tracked `InstrumentCalibration`** in every forward/inverse path; keep
  $E(\lambda)$ auto-recovery default-off for vendor-corrected products ([§7](#instrument-calibration-first-class)).
- [ ] **Keep the manifold and differentiable-GN paths honest about scope** — speed, not accuracy;
  oracle-test the forward factorisation and the structured Jacobian to $\le 10^{-5}$; BENCHMARK-GATE any
  adoption against the legacy solver on real held-out data ([§8](#manifold-emulator), [§9](#differentiable-forward-gn)).

## See also {#see-also}

- [formal-spec](formal-spec.md) — the theorem catalogue and notation authority these thrusts pin to.
- [error-budget-and-falsification](error-budget-and-falsification.md) — the falsification ledger and the
  do-NOT-do list (composition-fed $\tau$, single-gate quadrature, $n_e$-magnitude chasing for DED).
- [atomic-data-and-datasets](atomic-data-and-datasets.md) — the gf-grade-aware bundle, VALD/Kurucz
  ingest, and same-upper-level groupings the self-calibration needs.
- [impl-novel-techniques](impl-novel-techniques.md) — shipped realisations of the landed pieces.
- [cf-libs-family](cf-libs-family.md) — where C-σ, CD-SB, OPC, and full-spectrum forward-fitting sit.
- [benchmarks-reliability-workflows](benchmarks-reliability-workflows.md) — the flag-off/flag-on
  scoreboard discipline every BENCHMARK-GATED thrust must clear.
