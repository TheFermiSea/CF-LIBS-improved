---
slug: formal-spec
title: "Formal Specification & Notation Authority (cflibs-formal)"
chapter: formal-spec
order: 10
status: stable
register: reference
summary: >
  The machine-verified Lean 4 + mathlib specification (cflibs-formal) that grounds the CF-LIBS
  pipeline, and the AUTHORITATIVE notation table every other chapter imports. Each runtime gate is
  pinned to a proven theorem; the soundness envelope names which hypothesis real data violates.
tags: [formal-spec, lean, notation, identifiability, error-budget, soundness-envelope, verification]
updated: 2026-07-02
benchmarks_pre_reset: false
sources:
  - "@aragon2008"
  - "@ciucci1999"
  - "@aguilera2007"
  - "@yalcin1999"
  - "@cristoforetti2010"
  - "@gigosos2003"
  - "@volker2024"
  - "@barklem2016"
  - "@aragon2014"
  - "@borduchi2019"
  - "@olivero1977"
  - "~/code/cflibs-formal/CflibsFormal/"
  - "~/code/cflibs-formal/docs/SOLVER_FORMALIZATION_GAPS.md"
  - docs/M-spec-estimator-routing-policy.md
  - docs/v4/overhaul/adversarial/ADJUDICATION.md
  - cflibs/inversion/physics/reliability.py
  - cflibs/inversion/physics/derived_thresholds.py
  - cflibs/inversion/physics/error_budget.py
code_refs:
  - cflibs/inversion/physics/reliability.py::temperature_conditioning
  - cflibs/inversion/physics/reliability.py::stark_saha_lte_gate
  - cflibs/inversion/physics/derived_thresholds.py::required_energy_spread
  - cflibs/inversion/physics/error_budget.py::derive_line_selection_thresholds
  - cflibs/inversion/physics/quality.py::QualityMetrics
  - cflibs/plasma/saha_boltzmann.py::SahaBoltzmannSolver
lean_refs:
  - CflibsFormal/Boltzmann.lean#boltzmann_plot
  - CflibsFormal/Saha.lean#saha_relation
  - CflibsFormal/ForwardMap.lean#boltzmann_plot_intensity
  - CflibsFormal/Identifiability.lean#temperature_identifiability
  - CflibsFormal/Identifiability.lean#temperature_not_identifiable_of_degenerate
  - CflibsFormal/CompositionIdentifiability.lean#compositionIdentifiable
  - CflibsFormal/SelfAbsorptionInverse.lean#selfAbsorption_breaks_identifiability
  - CflibsFormal/ErrorBudget.lean#requiredEnergySpread_sufficient
  - CflibsFormal/ErrorBudget.lean#temp_rel_error_eq
  - CflibsFormal/Robustness.lean#twoLineBeta_stable_sharp
  - CflibsFormal/CompositionRobustness.lean#composition_dist_vector_le
  - CflibsFormal/StarkBroadening.lean#stark_saha_lte_consistent
  - CflibsFormal/MatrixEffects.lean#recoveredComposition_ratio_matrix_invariant
related: [libs-physics, classical-quantification, cf-libs-family, error-budget-and-falsification, benchmarks-reliability-workflows]
supersedes:
  - docs/v4/overhaul/verified/atomic.md
  - docs/v4/overhaul/adversarial/partition-functions.md
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Formal Specification & Notation Authority (cflibs-formal)

`cflibs-formal` is a machine-verified Lean 4 + mathlib specification of the CF-LIBS forward model,
the inverse composition-recovery problem, and the identifiability / reliability theorems that
establish *when and why* the inversion is well-posed. This chapter is two things at once: the
**rigor backbone** — a theorem-by-theorem tour of what is proven and which runtime decision each
proof licenses — and the **canonical notation table** that every other wiki chapter imports rather
than redefining.

> [!NOTE] FORMAL — the `.lean` sources live in a **separate repository** (`~/code/cflibs-formal`,
> Lean `v4.31.0` + mathlib `v4.31.0`). This chapter narrates and links theorem identifiers; it
> never copies Lean source. To read a proof, open the named module in that repo.

> [!IMPORTANT] SUPERSEDES the stale "Part 1 only" memory. An older session note recorded that only
> Part 1 (the Boltzmann distribution) was formalized, with "Saha → closure → inverse well-posedness"
> as future work. **That is out of date.** The spec is now end-to-end: forward (Boltzmann, Saha,
> ForwardMap, broadening), inverse (Boltzmann-plot, Saha-plane, self-absorption, OLS/least-squares),
> the composition estimators (Classic, C-σ, Closure, MultiSpecies), and the identifiability /
> robustness / error-budget layer are all present and axiom-clean.

## Verification status {#verification-status}

| Fact | Value | Provenance |
|------|-------|------------|
| Lean modules | 39 `.lean` files (33 in the shared core `CflibsFormal.*`, 6 alternative-estimator modules in `CflibsFormal.Alt.*`) | `find CflibsFormal -name '*.lean'` |
| Declarations | ~692 declarations, all axiom-clean (`lake env axiom-audit` exit 0, foundation-audit 0/0/0) | `SOLVER_FORMALIZATION_GAPS.md` |
| Axioms | **Only** the three standard mathlib foundations: `propext`, `Classical.choice`, `Quot.sound` | `SOLVER_FORMALIZATION_GAPS.md` |
| `sorry` / `admit` | **Zero** proof obligations open. The "2 sorry / 2 admit / 11 axiom" sometimes quoted are *phantom textual counts* (docstrings + the vendored `tools/AxiomAudit.lean`), not open goals. | `SOLVER_FORMALIZATION_GAPS.md` |

The design intent is stated plainly in the repo's own README: **"The goal is rigor, not numerical
accuracy."** Real-data CF-LIBS accuracy is limited by atomic data and plasma modeling, not by the
spec — so the investment is in *provable structure* (soundness, identifiability, error bounds), each
result grounded in the peer-reviewed literature and audited so that the *statement* faithfully
encodes the intended physics. Everything is dimensionless (bare `ℝ`); a separate additive layer
(`Dimensions.lean`) machine-checks dimensional homogeneity.

The load-bearing consequence for the pipeline: 51 runtime solver checks were **derived from proven
theorems**, and the `SOLVER_FORMALIZATION_GAPS.md` ledger names exactly where a *refuse* today is a
heuristic that a future theorem would make rigorous ([Part 5](#part-5-gaps)).

---

# Part 1 — Notation authority {#notation-table}

This table is the **single source of truth** for CF-LIBS notation across the wiki. Every symbol is
pinned to its `cflibs-formal` Lean identifier. `shared_conventions` reproduces the compact version;
every other chapter links here (`../formal-spec.md#notation-table`) and **never redefines a canonical
symbol**. A page introducing a genuinely new symbol adds it *here* first.

| Symbol | Meaning | Units | cflibs-formal id |
|--------|---------|-------|------------------|
| $T$ | excitation / plasma temperature | K (eV where noted) | `T` |
| $k_B$ | Boltzmann constant | eV/K | `kB` |
| $E_k$ | upper-level energy | eV (cm⁻¹ in DB) | `E k` |
| $g_k$ | upper-level statistical weight ($2J+1$) | – | `g k` |
| $U_s(T)$ | partition function of species $s$ | – | `partitionFunction` |
| $n_k$ | level population number density | cm⁻³ | `population` |
| $N_s$ | species (columnar) number density | cm⁻³ | `N` |
| $n_e$ | electron density | cm⁻³ | `Saha.electronDensity` |
| $A_{ki}$ | Einstein transition probability | s⁻¹ | `A` |
| $\lambda$ | transition wavelength (VACUUM unless flagged air) | nm | `lambda` |
| $I_{ki}$ | integrated line intensity | inst. units | `intensity` (`lineIntensity`) |
| $\tau$ | optical depth; escape factor $\mathrm{SA}(\tau)=(1-e^{-\tau})/\tau$ | – | `tau` (`selfAbsorbedIntensity`) |
| $C_s$ | species mass/number fraction ($\sum_s C_s = 1$) | – | `composition` |
| $F$ (`Fcal`) | scalar experimental/calibration factor (cancels via closure) | – | `F_cal` (`Fcal`) |
| $\chi$ | ionization energy between adjacent stages | eV | `chi` |
| $S(T)$ | Saha factor (everything except $n_e$ and the stage ratio) | cm⁻³ | `sahaFactor` |
| $R$ | measured ion/neutral stage intensity ratio | – | `electronDensityFromRatio` |
| $\beta$ | inverse temperature $\beta = 1/(k_B T)$ (Boltzmann-plot slope $= -\beta$) | eV⁻¹ | `twoLineBeta` |
| $\mathrm{SS}_E$ | upper-level energy spread $\sum_k (E_k - \bar E)^2$ | eV² | `ss_e` |

## Conventions that bind every page {#conventions}

- **Canonical Boltzmann ordinate.** The Boltzmann plot is *always*
  $$ y = \ln\!\left(\frac{I_{ki}\,\lambda}{g_k A_{ki}}\right) \quad\text{versus}\quad x = E_k, \qquad \text{slope} = -\frac{1}{k_B T}. $$
  The $\lambda$ factor is **load-bearing whenever it varies across the fitted line set** and must
  never be silently dropped. The scalar-`Fcal` convention (`lineIntensity`, no explicit $\lambda$)
  and the explicit photon-energy convention (`lineIntensityEnergy`, ordinate $\ln(I\lambda/gA)$)
  are proven to yield the *same* slope $-1/(k_B T)$ by `lean:CflibsFormal/ForwardMapEnergy.lean#lineIntensityEnergy_eq_lineIntensity`.
- **Wavelength convention.** The atomic DB stores **air** wavelengths (NIST/ASD convention); the
  spec's `lambda` is dimensionless and convention-agnostic. Any page that stores or compares
  wavelengths must state air-vs-vacuum in its first paragraph and route through the single
  `air_to_vacuum` / `vacuum_to_air` utility in `cflibs/core/`.
- **Prefer log-ratios.** For the DED / drift-tracking deliverable, prefer $\ln(N_i/N_j)$ over
  closure wt% — the ratio is matrix-invariant ([Part 2, MatrixEffects](#matrixeffects)) and free of
  the completeness inflation factor.

---

# Part 2 — The specification, module by module {#part-2-modules}

Modules are grouped by pipeline stage. For each, the prose states the theorem, then its Lean name,
then **which runtime decision it licenses**, then the **soundness-envelope** note: which hypothesis
real data can violate (making the guarantee *vacuous* rather than false).

## Forward model {#forward}

### Boltzmann.lean — level populations {#boltzmann}

The forward LTE level population $n_k = N \cdot g_k \exp(-E_k/k_B T)/U(T)$. Three cornerstones:
`population_sum` (populations sum to the total number density $N$), `boltzmann_plot`
($\ln(n_k/g_k)$ is affine in $E_k$ with slope $-1/(k_B T)$), and `temperature_from_two_levels` (the
slope between any two distinct-energy levels recovers $1/(k_B T)$ exactly).

- **Lean:** `lean:CflibsFormal/Boltzmann.lean#boltzmann_plot`, `#population_sum`, `#temperature_from_two_levels`.
- **Licenses:** the Boltzmann-plot temperature estimator (`BoltzmannPlotFitter.fit`).
- **Envelope:** assumes a single-temperature LTE population. Real plasmas can be non-LTE or
  multi-temperature; LTE validity is a *measurement-dependent hypothesis* checked at runtime
  ([StarkBroadening](#starkbroadening)), not something the theorem can supply.

### Saha.lean — ionization balance {#saha}

The Saha law $n_{z+1}\,n_e/n_z = 2(U_{z+1}/U_z)(2\pi m_e k_B T/h^2)^{3/2}\exp(-\chi/k_B T)$, with the
$n_e$-free part packaged as `sahaFactor`. Proven: `sahaFactor_pos` ($S>0$), `saha_relation` (the
diagnostic form $n_e = S/R$ is equivalent to $R\,n_e = S$), `electronDensity_antitone` ($R \mapsto
S/R$ is strictly antitone, hence a measured stage ratio pins a *unique* $n_e$), and `log_sahaFactor`
(the Saha-plot identity: $\ln S$ is affine in $1/(k_B T)$ with slope $-\chi$ plus a $(3/2)\ln T$
term). [@yalcin1999]

- **Lean:** `lean:CflibsFormal/Saha.lean#saha_relation`, `#electronDensity_antitone`, `#log_sahaFactor`.
- **Licenses:** `SahaBoltzmannSolver` and the Saha inversion $n_e = S(T)/R$ (routing regime R5).
- **Envelope:** requires *both* an observed neutral and an observed ion line. Single-stage
  pressure-balance $n_e$ imputation is **outside the envelope** — the hypothesis is the gate
  ([Part 5, Tier-3](#part-5-gaps)).

### ForwardMap.lean — optically-thin line intensity {#forwardmap}

Lifts the population Boltzmann plot to *observable* intensity $I_{ki} = F_\mathrm{cal}\,A_{ki}\,n_k$.
`boltzmann_plot_intensity`: $\ln(I_{ki}/(g_k A_{ki}))$ is affine in $E_k$, slope $-1/(k_B T)$,
intercept $\ln(F_\mathrm{cal} N / U(T))$ — the concentration-bearing intercept. `temperature_from_two_lines`:
the intensity-plot slope recovers $1/(k_B T)$ with $F_\mathrm{cal}$, $N$, $U$, $g$, $A$ all
cancelling. This is the **calibration-absorbed photon-rate convention** of [@ciucci1999]; the
per-line $h c/4\pi\lambda$ factor lives in `ForwardMapEnergy` and cancels in the slope only when
$\lambda$ is constant across the fit or the intensity is genuinely photon-rate.

- **Lean:** `lean:CflibsFormal/ForwardMap.lean#boltzmann_plot_intensity`, `#temperature_from_two_lines`.
- **Licenses:** the entire temperature leg of every classical estimator; the intercept is the
  density leg fed to closure.
- **Envelope:** assumes optically-thin emission ($\tau \approx 0$). Self-absorption breaks it
  ([SelfAbsorptionInverse](#selfabsorptioninverse)).

### LineBroadening.lean — Doppler width & the Gaussian budget {#linebroadening}

The thermal Doppler FWHM $\Delta\lambda_D = \lambda_0\sqrt{8\ln 2\,k_B T/(mc^2)}$ is positive,
strictly increasing in $T$, and invertible (`doppler_recovers`). Gaussian profiles convolve with
variance-addition, so FWHMs combine as $w=\sqrt{w_1^2+w_2^2}$ (`gaussQuadrature`), and
`deconvolveGaussian_quadrature` removes a known Gaussian component exactly — how instrument +
Doppler contributions are stripped to expose the Stark Lorentzian. [@olivero1977]

- **Lean:** `lean:CflibsFormal/LineBroadening.lean#doppler_recovers`, `#deconvolveGaussian_quadrature`.
- **Licenses:** the deconvolution feeding `deconvolve_stark_fwhm` before Stark $n_e$.
- **Envelope (honest):** the Gaussian-quadrature law is *exact* for Gaussian⊗Gaussian; using it to
  extract the Stark *Lorentzian* from a Voigt is the standard **approximation**, exact only in the
  Gaussian-dominated limit. The full Voigt combination (Olivero–Longbothum) is out of scope.

### StarkBroadening.lean — electron-impact width & the McWhirter LTE floor {#starkbroadening}

A second, physically independent $n_e$ diagnostic. `starkFWHM` is Griem-linear in $n_e$
($\Delta\lambda = 2w(n_e/n_\mathrm{ref})$), stated as an `IsLinearMap` (`starkFWHM_isLinear`);
`starkDensity` inverts it from a *measured* width and `starkDensity_recovers` proves exact inversion.
`mcWhirterBound` is the LTE floor $n_e \ge 1.6\times10^{12}\sqrt{T}(\Delta E)^3$, proven monotone in
both $T$ and $\Delta E$. The keystone is `stark_saha_lte_consistent`: **IF** the Stark route (width)
and the Saha route (stage ratio) yield the same $n_e$ **AND** it clears the McWhirter bound, **THEN**
one $n_e$ is simultaneously consistent with both independent forward laws and LTE. [@cristoforetti2010]

- **Lean:** `lean:CflibsFormal/StarkBroadening.lean#stark_saha_lte_consistent`, `#mcWhirterBound_mono_T`, `#mcWhirterBound_mono_dE`.
- **Licenses:** the `stark_saha_lte_gate` reliability gate the M7 refuse-to-report path consumes
  (`cflibs/inversion/physics/reliability.py::stark_saha_lte_gate`).
- **Envelope (honest, in the theorem itself):** agreement (`hagree`) is *assumed*, not proven — the
  two diagnostics are not shown to necessarily coincide. That is exactly right: their agreement is
  *evidence* precisely because they consume genuinely different observations (a width vs a ratio),
  and the runtime must *evaluate* it, not assume it.

## Inverse problem {#inverse}

### Inverse.lean — the algorithm-agnostic core {#inverse-core}

The shared inverse framework common to *every* extraction method: `PlasmaParams`, the forward
`observe` map, an abstract `CompositionEstimator`, and the `Sound` contract (a sound estimator
returns `trueComposition` on any forward-model observation from an admissible state).
`general_identifiability`: equal observations ⇒ equal temperature and composition. `sound_estimators_agree`:
any two sound estimators agree on forward data — what makes classic and C-σ *comparable*.

- **Lean:** `lean:CflibsFormal/Inverse.lean#general_identifiability`, `#sound_estimators_agree`.
- **Licenses:** the abstract cross-method agreement check (regime R1/R2 vote design).
- **Envelope (honest scoping):** `general_identifiability` pins $T$ only by *assuming* an external
  two-line ratio `hTratio` on a one-line-per-species map — the honest caveat removed by
  [CompositionIdentifiability](#compositionidentifiability).

### SahaInverse.lean — the joint Saha–Boltzmann plot {#sahainverse}

Couples Saha into the Boltzmann-plot inverse: neutral and ion ordinates lie on one line of common
slope $-1/(k_B T)$, and the inter-stage intercept shift carries $\ln(N_{z+1}/N_z)$
(`sahaBoltzmann_plot`), which equals the Saha term with $n_e$ explicit
(`sahaBoltzmann_shift_eq_log_saha`). `saha_joint_identifiability`: from a distinct-energy neutral
pair *plus* an ion line, **both** $T$ and $n_e$ are uniquely determined — $n_e$ recovered from
observations, never smuggled in. [@aguilera2007]

- **Lean:** `lean:CflibsFormal/SahaInverse.lean#saha_joint_identifiability`, `#sahaBoltzmann_shift_eq_log_saha`.
- **Licenses:** joint $(T, n_e)$ recovery when an ion stage is observed.
- **Envelope:** *requires* an observed ion line. No ion line ⇒ refuse $n_e$; do not impute a stage
  ratio ([Part 5, Tier-3, "unobserved stage"](#part-5-gaps)).

### SelfAbsorptionInverse.lean — the identifiability wall {#selfabsorptioninverse}

Folds the optically-thick forward map into the inverse and proves a **two-sided** result. PRESERVED:
with *known, matched* $\tau$, thick observations still identify density and composition
(`thick_composition_identifiability`) — the known $\mathrm{SA}(\tau)$ cancels per line. LOST: with
$\tau$ *unknown*, two genuinely different densities at different $\tau$ produce the same measured
thick intensity (`selfAbsorption_breaks_identifiability`) — a single line constrains only the product
$N\cdot\mathrm{SA}(\tau)$.

- **Lean:** `lean:CflibsFormal/SelfAbsorptionInverse.lean#selfAbsorption_breaks_identifiability`, `#thick_composition_identifiability`.
- **Licenses:** routing regime **X** — *refuse/flag* a single thick line with unknown $\tau$ (the M7
  path). This theorem is the mathematical justification for refuse-to-report.
- **Envelope:** the wall is unconditional. The escape from it is a multi-line curve-of-growth ratio
  (regime R4), whose invertibility is proven in `CurveOfGrowth` / `EquivalentWidth`.

### OLS.lean & LeastSquaresFit.lean — the projection inverse {#leastsquares}

`OLS.lean` is the pure-algebra foundation (`mean`, `olsSlope`, `olsIntercept`, the centering
identities, the noise gain `olsSlope_noise_gain: $\sum w_k^2 = 1/\mathrm{SS}_E$`, and the noise-free
recovery `ols_recovers_line`). `LeastSquaresFit.lean` supplies the property real spectra need — the
**off-manifold** projection inverse: `ols_minimizes_rss` (the closed-form OLS estimate *globally
minimizes* the residual sum of squares for arbitrary noisy $y$, existence exhibited, no
compactness argument), `leastSquaresFeasible_iff_exists` (a residual-based feasibility gate),
`leastSquaresResidual_eq_zero_iff` (zero minimal residual ⟺ on-manifold), and `ols_minimizer_eq_inverse`
(on-manifold, the projection inverse equals the identifiable inverse).

- **Lean:** `lean:CflibsFormal/LeastSquaresFit.lean#ols_minimizes_rss`, `#leastSquaresFeasible_iff_exists`, `#ols_minimizer_eq_inverse`.
- **Licenses:** the strict-mode line-fit feasibility gate — *a fit is admissible only when its
  minimal residual is small*; refuse when it exceeds tolerance rather than reporting a projection.
- **Envelope:** off the manifold there is **no ground-truth** $(T,N)$ to be `Sound` against; the
  estimator returns the orthogonal projection and the minimal residual quantifies model mismatch.

## Composition estimators {#estimators}

### Classic.lean — the calibration-free algorithm, assembled & sound {#classic}

The classical CF-LIBS estimator as an explicit map: temperature from the Boltzmann-plot slope
(`classic_temperature_correct`), relative densities from the exponentiated intercept
(`classicDensity_recovers`), absolute scale from closure $\sum_s C_s = 1$. `classic_sound`: run on the
genuine forward spectrum it returns the true composition. `classic_calibration_free`: the shared
$F_\mathrm{cal}$ cancels in the ratio — the calibration-free property made *literally true* via
`Closure.composition_smul_invariant`. [@ciucci1999]

- **Lean:** `lean:CflibsFormal/Classic.lean#classic_sound`, `#classic_calibration_free`, `#classic_temperature_correct`.
- **Licenses:** routing regime R1 (`ClosureEquation.apply_standard`).
- **Envelope:** the clean case — shared $F_\mathrm{cal}$, one clean emitting level per species,
  positive $g,A,N$. Noise and self-absorption route elsewhere (R2/R3).

### Alt/CSigma.lean — C-σ *is* the classic inverse, re-packaged {#csigma}

The multi-element master-line construction of [@aguilera2007] and the Saha-coupled cross-stage
collapse of [@aragon2014]: after subtracting the per-species offset $q_s = \ln(F_\mathrm{cal}N_s/U_s)$,
every line of every species collapses onto one master line $Y_{s,k} = -E_k/(k_B T)$
(`csigma_master_line`). The **honest content**: `csigmaComposition_eq_classicComposition` is an
*unconditional identity* — for all positive intensities, C-σ and classic are the **same algebraic
left-inverse** in log/exp vs direct-division packaging.

- **Lean:** `lean:CflibsFormal/Alt/CSigma.lean#csigmaComposition_eq_classicComposition`, `#csigma_master_line`, `#csigma_cross_stage_collapse`.
- **Licenses — a *prune*, not a method:** C-σ and classic are **not two votes**. Do not average
  them, do not report "two estimators agree" — their agreement is definitional and carries **zero**
  independent corroboration. Keep exactly one; choose by floating-point conditioning only
  (`docs/M-spec-estimator-routing-policy.md` §0). The only legitimate two-estimator vote is **OLS vs
  classic**, which differ off the noise-free fixpoint.
- **Envelope:** same thin-LTE hypotheses as classic; the Saha cross-stage section additionally uses
  `Saha.sahaFactor` / `log_sahaFactor`.

### Closure.lean & MultiSpecies.lean — the simplex constraint {#closure}

`Closure.composition` maps densities to fractions; `composition_sum_one`, `composition_mem_stdSimplex`
(the CF-LIBS closure $\sum_s C_s = 1$ is faithful membership in the probability simplex), and
`composition_smul_invariant` (fractions are intensive — invariant under rescaling all densities).
`MultiSpecies` reuses this per-element and proves `density_ratio_from_intensities` (at known $T$, the
ratio of two species' de-normalized intensities equals $N_s/N_t$).

- **Lean:** `lean:CflibsFormal/Closure.lean#composition_mem_stdSimplex`, `#composition_smul_invariant`; `lean:CflibsFormal/MultiSpecies.lean#density_ratio_from_intensities`.
- **Licenses:** the closure step (absolute scale) and the ratio deliverable.
- **Envelope (honest):** `MultiSpecies` shares **one** $(g,E,A)$ family and one $U$ across species;
  genuine per-species $U_s(T)$ and per-species atomic data are deferred ([Part 5, gap #7](#part-5-gaps)).

## Robustness & identifiability {#robustness}

### Identifiability.lean — when $(T, n_e, N)$ are recoverable {#identifiability}

Turns the forward model into injectivity statements. `temperature_identifiability`: two same-species
lines of **distinct** upper-level energy fix $T$ uniquely. `temperature_degeneracy` /
`temperature_not_identifiable_of_degenerate`: with $E_i = E_j$ the ratio collapses to a
$T$-*independent* constant, so the distinct-energy hypothesis is provably **necessary** — the "small
$\Delta E$ ⇒ refuse" gate is a *theorem*, not a heuristic. `density_identifiability`,
`electron_density_identifiability` complete the set.

- **Lean:** `lean:CflibsFormal/Identifiability.lean#temperature_identifiability`, `#temperature_not_identifiable_of_degenerate`, `#density_identifiability`.
- **Licenses:** the temperature-pair conditioning ranking and the $\Delta E = 0$ refuse gate
  (`cflibs/inversion/physics/reliability.py::temperature_conditioning`).

### CompositionIdentifiability.lean — many-element composition, $T$ from the data {#compositionidentifiability}

Removes `general_identifiability`'s honest caveat. A richer observation map `observeMulti` exposes
two anchor lines on a distinct-energy pair, so `compositionIdentifiable` extracts $T$ **from the
observations themselves** and forces equal full composition; `compositionIdentifiable_T` delivers
$T$ from any valid anchor.

- **Lean:** `lean:CflibsFormal/CompositionIdentifiability.lean#compositionIdentifiable`, `#compositionIdentifiable_T`.
- **Licenses:** trusting a multi-line, many-element composition solve without an externally supplied
  temperature.
- **Envelope — THE key soundness caveat:** identifiability assumes both parameter sets share
  *identical, correct* $g, A, E, U(T)$ (`CompositionIdentifiability.lean:132`). Under D/E-grade
  atomic-data bias — the documented real-data floor (~0.171 RMSEP, NULL-$A_{ki}$, incomplete stages)
  — that hypothesis is **false**, so the theorem is **vacuous on real data**. This is the single most
  important envelope statement in the spec: identifiability is proven, but the premise it rests on is
  what real spectra violate. The runtime corollary is *hard-FAIL on missing/zero/NULL atomic data*,
  not a fallback ([Part 5, gap #2](#part-5-gaps)).

### Robustness.lean & CompositionRobustness.lean — explicit-constant error propagation {#robustness-modules}

Quantitative perturbation with *no asymptotics*. `twoLineBeta_stable` / `twoLineBeta_stable_sharp`:
the two-line slope is Lipschitz in the ordinates with the **sharp** constant $2/|E_i-E_j|$ — wider
energy spacing ⇒ tighter bound. `composition_dist_vector_le`: the whole-vector $\ell^1$ composition
error obeys $\sum_s|\tilde C_s - C_s| \le 2\,(\mathrm{card})\,\delta/\hat S$ for per-species density
error $\le\delta$.

- **Lean:** `lean:CflibsFormal/Robustness.lean#twoLineBeta_stable_sharp`; `lean:CflibsFormal/CompositionRobustness.lean#composition_dist_vector_le`.
- **Licenses:** the temperature-pair conditioning number ($2/|E_i-E_j|$) and the certified
  worst-case composition error bound in `reliability.py`.
- **Envelope:** `CompositionRobustness` takes the per-species density error $\delta$ as a *given
  hypothesis* — the chain closing raw noise $\to\delta$ is the error-budget layer
  ([Part 3](#part-3-error-budget)), and is not yet fully closed ([Part 5, gap #5](#part-5-gaps)).

### ErrorBudget.lean — thresholds *derived*, not tuned {#errorbudget}

Turns empirical "magic-number" line-selection thresholds into corollaries of one deterministic
error-propagation chain. See [Part 3](#part-3-error-budget) for the full chain.

### MatrixEffects.lean — what CF-LIBS provably removes {#matrixeffects}

Makes matrix-independence precise. Model the matrix as the *detected subset* $D$ of species. The
recovered **subcomposition** (pairwise ratios among detected species) is matrix-**invariant** for any
$D$ (`recoveredComposition_ratio_matrix_invariant` — Aitchison subcompositional coherence). The
recovered **absolute** fractions are matrix-**dependent**, over-estimated by the exact factor
$1/(1-m)\ge 1$ where $m$ is the undetected mass fraction (`recoveredComposition_eq_inflation`).
Ionization suppression: the ion density $n_\mathrm{ion} = N_\mathrm{tot}S/(S+n_e)$ is strictly
decreasing in $n_e$ (`sahaIonDensity_antitone`). [@ciucci1999]

- **Lean:** `lean:CflibsFormal/MatrixEffects.lean#recoveredComposition_ratio_matrix_invariant`, `#recoveredComposition_eq_inflation`, `#sahaIonDensity_antitone`.
- **Licenses:** the DED / drift deliverable's preference for **ratios over absolute wt%** — report
  absolute fractions as upper bounds; flag non-reliable unless completeness is externally certified.
- **Envelope:** invariance holds with detected densities $n$ and temperature $T$ **held fixed**; it is
  NOT unconditional matrix-independence. Per-shot $T$/$n_e$ shifts are handled separately, and
  CF-LIBS alone leaves documented residual matrix effects (Borduchi et al. [@borduchi2019]).

---

# Part 3 — The certified error-budget chain {#part-3-error-budget}

The empirical line-selection thresholds (min lines per element, min energy spread, min SNR) are not
tuned constants — they are **proven corollaries of a single deterministic error-propagation chain**,
so they follow *from a target accuracy*. The Python mirrors in
`cflibs/inversion/physics/{error_budget,derived_thresholds}.py` are conformance-pinned to the spec's
emitted oracle fixtures, so a drift from the proof fails CI.

$$
\underbrace{\varepsilon}_{\substack{\text{per-line}\\\text{ordinate error}}}
\;\xrightarrow[\text{\lean{olsSlope\_stable\_l2}}]{}\;
\underbrace{|\Delta\beta|}_{\substack{\text{slope}\\\text{error}}}
\;\xrightarrow[\text{\lean{temp\_rel\_error\_eq}}]{}\;
\underbrace{\frac{|\Delta T|}{T}}_{\substack{\text{rel. temp}\\\text{error}}}
\;\xrightarrow[\text{\lean{composition\_abs\_sub\_le}}]{}\;
\underbrace{|\Delta C_s| \le \tau}_{\text{target}}
$$

| Link | Statement | Lean lemma | Kind |
|------|-----------|------------|------|
| $\varepsilon \to |\Delta\beta|$ | $|\Delta\beta| \le \varepsilon\sqrt{n}/\sqrt{\mathrm{SS}_E}$ (Cauchy–Schwarz); sharp $\ell^1$ variant also proven | `olsSlope_stable_l2`, `olsSlope_stable_l1` | upper bound |
| $|\Delta\beta| \to |\Delta T|/T$ | $|\Delta T|/T = k_B T\,|\Delta\beta|$ | `temp_rel_error_eq` | **exact identity** |
| $|\Delta T|/T \to |\Delta C_s|$ | per-fraction $\ell^1$ composition bound | `composition_abs_sub_le`, `composition_dist_vector_le` | upper bound |
| noise-gain kernel | $\sum_k w_k^2 = 1/\mathrm{SS}_E$ (Gauss–Markov kernel) | `olsSlope_noise_gain` | **exact identity** |

**Inverting the chain** yields the thresholds — the energy spread / SNR that *guarantee* a target
accuracy:

| Derived threshold | Guarantee | Lean lemma | Python mirror |
|-------------------|-----------|------------|---------------|
| $\mathrm{SS}_E \ge \varepsilon^2 n/\tau_\beta^2$ (min energy spread) | slope error $\le \tau_\beta$ | `requiredEnergySpread_sufficient` | `derived_thresholds.required_energy_spread` |
| $\varepsilon \le \tau_\beta\sqrt{\mathrm{SS}_E/n}$ (min SNR) | slope error $\le \tau_\beta$ | `maxPerLineError_sufficient` | `derived_thresholds.max_per_line_error` |
| $n \ge \varepsilon^2/(v\,\tau_\beta^2)$ (statistical min lines) | slope *std* $\le \tau_\beta$ | `olsSlope_noise_gain` (+ `Alt.OLSVariance.olsSlope_variance_eq`) | `derived_thresholds.required_min_lines` |
| $\delta$ budget for target $\sigma_C$ | $|\Delta C_s| \le \tau_C$ | `composition_target_sufficient` | `error_budget.density_budget_from_composition` |

> [!IMPORTANT] Honest scope of the line-count threshold. The deterministic bounds are **worst-case**
> (adversarial, perfectly-correlated ordinate errors). In that regime the dominant levers are the
> **energy spread** $\mathrm{SS}_E$ and the **per-line error** $\varepsilon$ — adding more lines *at
> the same energies* does **not** improve the worst case (`olsSlope_stable_l2_sq` carries `card ι` in
> the numerator). The familiar "more lines ⇒ better" rule is a **statistical** statement (variance
> $\sigma^2/\mathrm{SS}_E$, Gauss–Markov); its deterministic kernel is the noise gain, and the spec
> does **not** claim a deterministic min-line-count bound the worst case cannot support.

The reliability layer (`reliability.py::temperature_conditioning`) consumes the *sharp* two-line
constant $2/|E_i-E_j|$ from `twoLineBeta_stable_sharp` to rank candidate temperature pairs; the
`stark_saha_lte_gate` consumes `stark_saha_lte_consistent` for the M7 refuse/flag decision.

---

# Part 4 — Verified false-positive callouts {#part-4-false-positives}

The adversarial physics audit (`docs/v4/overhaul/adversarial/ADJUDICATION.md`) had a "skeptic of the
skeptics" adjudicator re-derive every "flawed" verdict. Three claims that a *prior* audit had raised
were re-checked and **rejected as false alarms** — they are recorded here so the wiki does not
resurrect them.

> [!NOTE] NOT A BUG — the $\ln(U_{II}/U_I)$ cancellation is CORRECT.
> A prior audit flagged a "missing $\ln(U_{II}/U_I)$ term" in the Saha→neutral-plane $y$-shift. The
> adjudicator re-derived it with sympy/astropy: the cancellation is **algebraically real**, the
> round-trip recovers $T$ exactly, and the prior "missing term" alarm was itself the error. The
> spec's `sahaBoltzmann_shift_eq_log_saha` carries exactly the correct
> $(\ln U_z - \ln U_{z+1})$ structure. **Verdict: upheld correct, no fix.**

> [!NOTE] LIVE, NOT DEAD — `QualityAssessor` / `QualityMetrics` is a live gate.
> The quality-metrics module (`cflibs/inversion/physics/quality.py::QualityMetrics`) is wired into
> `iterative.py` and the `inversion/__init__` surface; its Boltzmann $R^2$ / conditioning thresholds
> are consumed, not dead code. It is the runtime realization of the `Robustness` / `ErrorBudget`
> conditioning bounds. **Verdict: live; do not prune.**

> [!NOTE] CORRECT — the Saha $S2$ (stage III/II) guard is right.
> `SahaBoltzmannSolver` (`saha_boltzmann.py`) sets the stage-III/II Saha ratio `S2 = 0.0` when
> stage-III atomic data is unavailable, giving $n_I = n_\mathrm{tot}/(1 + S_1 + S_1 S_2)$. Zeroing an
> **unobserved** stage ratio (rather than imputing one) is exactly the conservative, envelope-faithful
> behavior the spec licenses (`Saha.lean` proves the two-stage relation only for observed stages).
> **Verdict: correct guard; not a physics error.**

For the *genuine* bugs the same audit confirmed (H-α Stark reference width, mole-vs-mass reporting,
Fe I level completeness, the non-default `temporal.py` optical-depth prefactor), see
[error-budget-and-falsification.md](error-budget-and-falsification.md) and
[atomic-data-and-datasets.md](atomic-data-and-datasets.md). Those are real and benchmark-gated; the
three above are not.

---

# Part 5 — Formalization gaps (the honest soundness envelope) {#part-5-gaps}

The `SOLVER_FORMALIZATION_GAPS.md` ledger is the authoritative list of **properties the solvers rely
on that are not yet proven** — where the strict solver would step *outside* the verified envelope. A
gap is not a broken proof; it marks where a *refuse* today is a heuristic a future theorem would make
rigorous. Two gaps are load-bearing for real-data honesty.

## Tier 1 — load-bearing {#gaps-tier-1}

| # | Gap | Why it matters | Runtime posture |
|---|-----|----------------|-----------------|
| 2 | **Atomic-data perturbation channel unmodeled.** Identifiability assumes *identical, correct* $g,A,E,U(T)$; no theorem bounds composition error by $\Delta(gA)$, $\Delta E_k$, $\Delta U_s$. | This is the **documented real-data accuracy floor** (~0.171 RMSEP, NULL-$A_{ki}$, incomplete stages). Highest physics impact. | Hard-FAIL on missing/zero/NULL atomic data — do **not** fall back to `IP = 15.0 eV` or crude-$U$. |
| 1 | **Fully nonlinear joint least-squares inverse** over the coupled multi-species map (fit $(T, n_e, \text{composition})$ from raw intensities). The log-linearized per-species case *is* now proven (`LeastSquaresFit`, `ols_minimizes_rss`). | The joint solver's convergence is not yet theorem-backed. | Trust the log-linear feasibility gate; treat joint fits as advisory pending an adoption gate. |
| 3 | **$n_e$/Saha Lipschitz + multi-line conditioning.** Only exact injectivity (`electronDensity_antitone`) and the 2-line slope (`Robustness`) are proven — no condition-number bound for the multi-element design matrix. | Ill-conditioned design matrices give unstable composition. | Runtime rank/condition monitor on the design matrix. |
| 5 | **End-to-end noise → composition not closed.** The chain raw-$\varepsilon \to \delta$ is not fully composed; no $U_s(T)$ Lipschitz lemma; deterministic bound assumes a single global $\varepsilon$ (no heteroscedastic per-line variant). | The error-budget chain has an open link. | Use the proven per-link bounds; do not over-claim an end-to-end certificate. |
| 6 | **Coupled Saha–closure–charge fixed point.** Only static facts proven; no existence/uniqueness/convergence of the self-consistent $(T, n_e, \text{composition})$ loop. | This is what would license *trusting the iterative solver's convergence flag*. | Convergence flag is heuristic, not certified. |

## Tier 2 — self-absorption / regime coverage {#gaps-tier-2}

- **Composition-level self-absorption non-identifiability** (gap #9): a per-line density-LOST
  theorem exists (`selfAbsorption_breaks_identifiability`) but no *composition-level* one — the
  dominant failure mode for concentrated / high-entropy alloys.
- **Thick-regime curve of growth (the $\sqrt\tau$ branch) unformalized** (gap #10): only the
  optically-thin / saturation-onset branch of `EquivalentWidth` / `Alt/CSigmaCurveOfGrowth` is
  proven; the thick $\sqrt\tau$ asymptotic and a conditioning bound for the COG inverse are open.
- **Matrix-effect invariance under per-shot $(T, n_e)$ variation** (gap #11): invariance is proven
  only at fixed $T, n_e$ ([MatrixEffects envelope](#matrixeffects)).

## Tier 3 — the hypothesis *is* the runtime check {#gaps-tier-3}

These need **wiring, not a Lean fix** — the spec already provides the criterion:

- **LTE validity** — a hypothesis in every Boltzmann/Saha theorem (correctly, it is
  measurement-dependent). Strict mode must evaluate McWhirter on the measured $n_e$ and cross-check
  Stark/Saha agreement (`stark_saha_lte_consistent`), and refuse when LTE is unsupported.
- **Unobserved ionization stage** — `saha_joint_identifiability` *requires* an observed ion line;
  single-stage pressure-balance $n_e$ is outside the envelope. Refuse $n_e$ rather than impute a
  stage ratio.
- **Completeness / missing mass $m$** — absolute composition is provably inflated by $1/(1-m)$
  (`recoveredComposition_eq_inflation`); $m$ depends on undetected species. Report absolute fractions
  as upper bounds; prefer ratios (`recoveredComposition_ratio_matrix_invariant`).

## Not gaps — recorded, do not chase {#gaps-not}

- The three standard mathlib axioms are the trust base (no fix possible or needed).
- The "2 sorry / 11 axiom" premise is *phantom textual counts*; the spec is axiom-clean.
- `oracle/Generate.lean` is a **Float bridge / test artifact**, not verified ground truth — trust the
  $\mathbb{R}$ theorems, not `fixtures.json`.

---

## Soundness envelope — one-line summary {#soundness-envelope}

Everything in `cflibs-formal` is proven over exact $\mathbb{R}$ under explicit nondegeneracy
hypotheses. The envelope is defined by **which hypothesis real data violates**:

| Guarantee | Holds when | Real data violates via |
|-----------|-----------|------------------------|
| Composition identifiability | atomic data $g,A,E,U$ is identical & correct | D/E-grade bias, NULL-$A_{ki}$, incomplete stages ⇒ **vacuous** |
| Temperature identifiability | $E_i \ne E_j$ (distinct energies) | near-degenerate line pairs ⇒ ill-conditioned |
| Thin composition soundness | $\tau \approx 0$ | self-absorption; unknown $\tau$ ⇒ refuse |
| Joint $(T, n_e)$ | an ion line is observed | single-stage spectra ⇒ refuse $n_e$ |
| Absolute composition | all species detected ($m = 0$) | undetected species ⇒ upper bound only; prefer ratios |
| LTE-dependent theorems | McWhirter cleared, Stark≈Saha | non-LTE / transient plasma ⇒ refuse |

## What correct code MUST do {#checklist}

- [ ] Compute the Boltzmann ordinate as $\ln(I\lambda/(g A))$; carry $\lambda$ when it varies across
      the fit (`boltzmann_plot`).
- [ ] Refuse a temperature fit on a degenerate line pair ($\Delta E \to 0$) —
      `temperature_not_identifiable_of_degenerate` makes this a theorem, not a heuristic.
- [ ] Hard-FAIL on missing/zero/NULL atomic data; never substitute `IP = 15.0 eV` or a crude $U$
      (gap #2 — identifiability is *vacuous* under atomic-data bias).
- [ ] Refuse a single thick line with unknown $\tau$; recover $\tau$ from a multi-line COG ratio or
      route to R3 with a measured $\tau$ (`selfAbsorption_breaks_identifiability`).
- [ ] Refuse $n_e$ when no ion line is observed; do not impute a stage ratio
      (`saha_joint_identifiability` premise; the `S2 = 0` guard is correct).
- [ ] Gate the line-fit on the least-squares feasibility residual; refuse when it exceeds tolerance
      (`leastSquaresFeasible_iff_exists`).
- [ ] Never count classic and C-σ as two votes — they are the same left-inverse
      (`csigmaComposition_eq_classicComposition`). The only legitimate vote is OLS vs classic.
- [ ] Report absolute fractions as upper bounds under incomplete detection; prefer $\ln(N_i/N_j)$
      ratios for the DED / drift deliverable (`recoveredComposition_ratio_matrix_invariant`).
- [ ] Evaluate McWhirter + Stark↔Saha agreement before trusting any LTE-dependent result
      (`stark_saha_lte_consistent`).

## See also {#see-also}

- [libs-physics.md](libs-physics.md) — the physics the forward theorems formalize (Saha–Boltzmann,
  broadening).
- [classical-quantification.md](classical-quantification.md) — the classic / C-σ / closure
  estimators these theorems certify.
- [cf-libs-family.md](cf-libs-family.md) — the method family (OPC, CD-SB, C-σ) and its literature.
- [error-budget-and-falsification.md](error-budget-and-falsification.md) — the falsification ledger
  and the confirmed (non-false-positive) bugs.
- [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) — how the reliability
  gates and derived thresholds are exercised in practice.
