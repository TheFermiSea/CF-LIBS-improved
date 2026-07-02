---
slug: cf-libs-family
title: "The Calibration-Free Family: CF-LIBS, OPC, C-sigma, CD-SB"
chapter: cf-libs-family
order: 0
status: stable
register: review
summary: >
  The family of standardless LIBS quantification methods — classic CF-LIBS, one-point
  calibration, C-sigma graphs, and columnar-density Saha-Boltzmann — as one algebraic
  object seen through different packagings. Load-bearing verdict: C-sigma is the classic
  inverse repackaged (an unconditional Lean identity), so "C-sigma agrees with classic"
  is zero independent corroboration; the single lever CF-LIBS provably removes is the
  scalar calibration factor F, and closure is a projection, not the estimator.
tags: [cf-libs, one-point-calibration, csigma, cdsb, closure, self-absorption, boltzmann-plot, saha-boltzmann, ilr, calibration-free, review]
updated: 2026-07-02
benchmarks_pre_reset: true
sources:
  - "@ciucci1999"
  - "@tognoni2010"
  - "@aguilera2007"
  - "@aragon2008"
  - "@aragon2014"
  - "@cdsb2013"
  - "@cavalcanti2013"
  - "@volker2024"
  - "@aitchison1982"
  - docs/v4/overhaul/literature/cflibs-method.md
  - docs/v4/overhaul/literature/bayesian-oe.md
  - docs/M-spec-boltzmann-convention-literature-verdict.md
  - docs/M-spec-estimator-routing-policy.md
  - docs/M-spec-solver-head-to-head.md
  - docs/research/real-steel-opc-promotion.md
  - docs/research/2026-06-19-cross-discipline-sota.md
  - cflibs/inversion/physics/csigma.py
  - cflibs/inversion/physics/opc.py
  - cflibs/inversion/physics/closure.py
  - cflibs/inversion/physics/self_absorption_observable.py
code_refs:
  - cflibs/inversion/physics/boltzmann.py::BoltzmannPlotFitter
  - cflibs/inversion/physics/closure.py::ClosureEquation
  - cflibs/inversion/physics/closure_strategy.py
  - cflibs/inversion/physics/softmax_closure.py
  - cflibs/inversion/physics/csigma.py
  - cflibs/inversion/physics/opc.py
  - cflibs/inversion/physics/matrix_effects.py
  - cflibs/inversion/physics/self_absorption.py
  - cflibs/inversion/physics/self_absorption_observable.py
  - cflibs/plasma/saha_boltzmann.py::SahaBoltzmannSolver
  - cflibs/inversion/solve/bayesian.py::BayesianForwardModel
lean_refs:
  - CflibsFormal/ForwardMap.lean#boltzmann_plot_intensity
  - CflibsFormal/ForwardMap.lean#temperature_from_two_lines
  - CflibsFormal/Saha.lean#saha_relation
  - CflibsFormal/Saha.lean#sahaBoltzmann_plot
  - CflibsFormal/Closure.lean#composition_sum_one
  - CflibsFormal/Alt/CSigma.lean#csigmaComposition_eq_classicComposition
  - CflibsFormal/SelfAbsorption.lean#selfAbsorptionFactor_le_one
  - CflibsFormal/SelfAbsorption.lean#selfAbsorption_breaks_identifiability
  - CflibsFormal/MatrixEffects.lean#recoveredComposition_ratio_matrix_invariant
  - CflibsFormal/CurveOfGrowth.lean#cogRatio_strictAntiOn
related: [classical-quantification, error-budget-and-falsification, frontier-methods, formal-spec, impl-literature-methods, libs-physics]
supersedes: []
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# The Calibration-Free Family: CF-LIBS, OPC, C-sigma, CD-SB

The calibration-free (CF) family is a set of standardless LIBS quantification methods that
share one algebraic skeleton — an LTE line-intensity forward model inverted under the closure
constraint $\sum_s C_s = 1$ — and differ only in how they package the inverse and how they
treat self-absorption. This chapter maps the family (classic CF-LIBS, one-point calibration,
C-sigma graphs, columnar-density Saha-Boltzmann, and the Bayesian/optimal-estimation framing),
states precisely what "calibration-free" removes and what it does not, and records the
load-bearing verdicts our formal spec forces: **C-sigma is the classic inverse in different
packaging (an unconditional identity, hence zero independent corroboration), and closure is a
projection onto the simplex, not the estimator.**

> [!IMPORTANT] PRE-RESET NUMBERS — the atomic DB was rebuilt (ASD59, 203k lines); the
> composition/RMSE figures quoted below (the solver head-to-head 2.27/11.51/17.71 wt%, the
> real-steel OPC 39.04 → 10.12 wt%) are NOT current. The **mechanisms and orderings are
> retained**; treat the magnitudes as dead until re-benchmarked on the reset DB. See
> [benchmarks-reliability-workflows](benchmarks-reliability-workflows.md).

**Wavelength convention.** Every wavelength in this chapter is a NIST/ASD **air** wavelength as
stored in the atomic DB; conversion is via the single `air_to_vacuum`/`vacuum_to_air` utility in
`cflibs/core/`. The canonical Boltzmann ordinate $y=\ln(I_{ki}\lambda/(g_kA_{ki}))$ carries
$\lambda$ explicitly because it is load-bearing whenever the fitted line set spans a wide
wavelength range — see [The canonical ordinate](#canonical-ordinate). Symbols follow
[formal-spec/notation](formal-spec.md); this page introduces no new canonical symbol.

---

## 1. The family map {#family-map}

All members invert the same optically-thin LTE line-intensity relation [@ciucci1999; @tognoni2010]

$$
I_{ki} = F \cdot \frac{hc}{4\pi\lambda_{ki}} \cdot C_s \cdot \frac{A_{ki}\,g_k}{U_s(T)}\,
          e^{-E_k/(k_B T)} ,
$$

where $F$ is the single scalar experimental/calibration factor (optical collection efficiency ×
plasma column × geometric $1/4\pi$), identical for every line in one acquisition. The methods
differ in three axes only: (i) how the per-species scale is read off, (ii) whether
self-absorption is *avoided*, *corrected*, or *modelled*, and (iii) what external anchor (if any)
replaces closure.

| Method | Core idea | Self-absorption | External anchor | Regime it targets | Key refs |
|--------|-----------|-----------------|-----------------|-------------------|----------|
| **Classic CF-LIBS** | Boltzmann/Saha-Boltzmann plot → $T$, closure fixes $F$ | avoid (line selection) | none (closure only) | optically thin, clean lines | [@ciucci1999; @tognoni2010] |
| **Multi-element Saha-Boltzmann** | pool neutral+ion lines of all elements on one plot, common slope | avoid | none | wide $E_k$ span, several species | [@aguilera2007] |
| **One-point calibration (OPC)** | one known concentration fixes $F$ (drop full closure) | avoid / optional correction | one certified element/standard | per-matrix bias removal | [@cavalcanti2013; @borduchi2019] |
| **C-sigma graphs** | generalized curve-of-growth, pool by ionization stage on $\log(C\sigma)$ vs $\log(I/B)$ | model (radiative transfer) | none | self-absorbed, geological | [@aragon2014; @aguilera2015] |
| **CD-SB (columnar density)** | invert self-absorbed lines for column density $N_s\ell$ | exploit | none | resonance/strong lines, high $C$ | [@cdsb2013] |
| **SA-corrected CF** | classic + escape-factor / Planck correction | correct (observable-anchored) | none | thick lines present | [@bulajic2002; @sun2009; @volker2023] |
| **Bayesian / OE** | posterior over $(T,n_e,C_s)$; forward model in the likelihood | model | prior | uncertainty-first, degenerate data | [@kasim2019; @bowman2024; @oliver2024] |

The runtime routing table that selects among these per spectrum — and the solver code — lives in
[impl-literature-methods](impl-literature-methods.md); this chapter is method-level theory only.
The estimator-routing policy (`docs/M-spec-estimator-routing-policy.md`) pins each route to a
proven theorem.

---

## 2. CF-LIBS foundations {#foundations}

### 2.1 From line intensity to the Boltzmann plot

Taking the logarithm of the forward relation linearizes it. For a set of lines of one species
$s$ in one ionization stage:

$$
\underbrace{\ln\!\frac{I_{ki}\,\lambda_{ki}}{g_k A_{ki}}}_{y}
   = \underbrace{-\frac{1}{k_B T}}_{\text{slope}}\,E_k
   + \underbrace{\ln\!\frac{F\,C_s\,hc}{4\pi\,U_s(T)}}_{\text{intercept } b_s}.
$$

This is the **canonical CF-LIBS Boltzmann plot** [@ciucci1999; @tognoni2010; @aragon2008]:
$y = \ln(I_{ki}\lambda/(g_kA_{ki}))$ against $x = E_k$, slope $-1/(k_BT)$, intercept $b_s$. All
lines of one species fall on one line; all species share the slope (one $T$); different species
give parallel lines offset by $\ln(F C_s/U_s(T))$. The identity is machine-checked in the formal
spec: `lean:CflibsFormal/ForwardMap.lean#boltzmann_plot_intensity` proves the affine form and
`lean:CflibsFormal/ForwardMap.lean#temperature_from_two_lines` proves that the two-line slope
recovers $1/(k_BT)$ exactly. The pipeline implements it in
`cflibs/inversion/physics/boltzmann.py::BoltzmannPlotFitter`.

### 2.2 The canonical ordinate — $\lambda$ is load-bearing {#canonical-ordinate}

> [!NOTE] FORMAL — the wavelength-form Boltzmann plot is proven affine in $E_k$ with slope
> $-1/(k_BT)$: `lean:CflibsFormal/ForwardMap.lean#boltzmann_plot_intensity`.

A recurring implementation error is dropping $\lambda$: writing $y=\ln(I/(g_kA_{ki}))$. This is a
*valid unit-reduced form* only when $\lambda$ is constant across the fit, the intensity is truly
photon-rate ($I\propto A_{ki}n_k$, no $hc/\lambda$ factor), or comparisons are restricted to
one $\lambda$. In those cases $hc/\lambda$ is a global constant that cancels in both the slope
($T$) and the closure ($C_s$). But in the real multi-line CF-LIBS regime $\lambda_{ki}$ **varies**
and **correlates with $E_k$** across a species' line set, so the per-line $-\ln\lambda_{ki}$ term
does not cancel: dropping it biases **both** the slope (temperature) and the intercept
(composition), not merely the intercept (`docs/M-spec-boltzmann-convention-literature-verdict.md`).

Quantitatively, over a 240–660 nm window $\ln(\lambda)$ spans $\ln(660/240)\approx 1.01$ — a
spread that tilts the slope directly. A numerical bridge experiment measured a $\lambda$-drop bug
producing slope $-0.689$ against the correct $-0.419$: a gross $T$ error, not rounding. The
literature standard therefore keeps $\lambda$ in the numerator whenever $I$ is a measured
energy/radiance quantity [@aragon2008]. The pipeline's `ASSERT_CONVENTION` markers in
`cflibs/inversion/physics/boltzmann.py` enforce this. **Never silently drop $\lambda$.**

### 2.3 The three founding assumptions

Classic CF-LIBS [@ciucci1999] rests on three assumptions, each a distinct failure mode
(quantified in [error-budget-and-falsification](error-budget-and-falsification.md)):

1. **Stoichiometric ablation** — plasma composition equals bulk sample. No algorithmic
   correction exists for its violation; it is a fundamental limit [@tognoni2010].
2. **Local thermodynamic equilibrium (LTE)** — Boltzmann level populations and Saha ionization
   balance hold at one $T$. Necessary condition is McWhirter's density floor
   $n_e \ge 1.6\times10^{12}\,T^{1/2}(\Delta E)^3\ \mathrm{cm^{-3}}$, but this is necessary and
   **not sufficient**: a relaxation-time criterion must also hold [@cristoforetti2010], and
   partial-LTE effects bias low-lying levels [@cristoforetti2013]. $\Delta E$ is the
   resonance-line energy of the dominant species, not $\max E_k$ or an adjacent gap
   (`reference_mcwhirter_delta_e_physics`).
3. **Optical thinness** — no self-absorption. This is the assumption the self-absorption family
   (§5–6) exists to relax.

The intrinsic error floor even for an *ideal* plasma is dominated by the temperature precision
[@tognoni2007], which is why the multi-element common-slope fit (§3) and wide-$E_k$ line
selection matter more than solver choice.

---

## 3. The Saha-Boltzmann plane {#saha-boltzmann-plane}

Neutral and ionic lines cannot be plotted on the same Boltzmann axes directly — the ion's
upper-level energy is measured from the *ion* ground state. The Saha-Eggert equation supplies the
bridge [@aragon2008]:

$$
\frac{N_{s,1}}{N_{s,0}} = \frac{2}{n_e}\,\frac{U_{s,1}(T)}{U_{s,0}(T)}
   \left(\frac{2\pi m_e k_B T}{h^2}\right)^{3/2} e^{-\chi_s/(k_B T)} ,
$$

proven (factor-by-factor, correct dimensions) in `lean:CflibsFormal/Saha.lean#saha_relation`; the
ratio is antitone in $n_e$ (`electronDensity_antitone`), which is what makes $n_e$ recoverable
from a stage ratio. Solved in `cflibs/plasma/saha_boltzmann.py::SahaBoltzmannSolver`.

**Mapping ionic lines onto the neutral plane.** An ionic line is shifted to effective energy
$E_k^{\text{eff}} = E_k^{\text{ion}} + \chi_s$ on the abscissa, and its ordinate gains the Saha
correction term $\ln\!\big(\tfrac{n_e}{2}(h^2/2\pi m_e k_B T)^{3/2}\big)$. After the shift, ions
and neutrals of all elements lie on **one straight line** with the common slope $-1/(k_BT)$
[@aguilera2007]. This is the multi-element Saha-Boltzmann plot; the formal statement is
`lean:CflibsFormal/Saha.lean#sahaBoltzmann_plot`, with
`sahaBoltzmann_shift_eq_log_saha` proving the shift equals the log-Saha factor.

The common-slope multi-element fit is the workhorse of a robust CF inversion: pooling lines
across species widens the $E_k$ lever-arm and averages per-line noise, directly attacking the
dominant $T$ error [@aguilera2007; @tognoni2007]. Forgetting to add $\chi_s$ (pitfall P7 of
`docs/v4/overhaul/literature/cflibs-method.md`) places ionic points at the wrong abscissa and
corrupts the slope. For doubly-ionized species add $\chi_{s,1}+\chi_{s,2}$.

---

## 4. One-point calibration and what "calibration-free" means {#one-point-calibration}

### 4.1 F cancels via closure — the actual content of "calibration-free"

The scalar $F$ is never measured. Because $C_s \propto U_s(T)\,e^{b_s}/F$, imposing
$\sum_s C_s = 1$ determines $F$ from the fitted intercepts alone:

$$
F = \frac{hc}{4\pi}\sum_s U_s(T)\,e^{b_s}, \qquad
C_s = \frac{U_s(T)\,e^{b_s}}{F\,hc/4\pi}.
$$

This is the *entire* sense in which CF-LIBS is calibration-free: **the single scalar $F$ is
provably removed by the closure normalization.** The formal spec makes the scope explicit — the
classic inverse is calibration-free in that a global rescale of all intensities leaves the
recovered composition invariant (`classic_calibration_free`, and closure's scale invariance
`lean:CflibsFormal/Closure.lean#composition_sum_one` with `composition_smul_invariant`). What
$F$-cancellation does **not** remove: per-line atomic-data error, self-absorption, non-LTE, or
non-stoichiometric ablation — those survive closure untouched and dominate the modern error
budget.

### 4.2 OPC framed honestly — a per-matrix band-aid

One-point calibration replaces closure with a single known concentration $C_{\text{ref}}$
[@cavalcanti2013]: fix $F$ from the reference element's intercept, then all other
concentrations follow. The Saha-Boltzmann-plot variant [@borduchi2019] cross-calibrates $T$ and
$n_e$ from one reference line per stage and is single-line-per-element capable.

OPC posts the best documented trueness among standardless-adjacent methods (sub-1 wt% on
majors for bronzes; ~15% average uncertainty versus ~53% for inverse-CF without SA correction).
**But this is bought with a per-matrix standard, so OPC is a bias-correction layer, not a
calibration-free method** — its accuracy is contingent on the standard being matrix-matched.
The honest framing: OPC is the right tool exactly when the matrix is *known and constrained*
(the DED tracking deliverable — Ti-6Al-4V, steel — is precisely this case), and a category error
when sold as standardless quantification of an unknown matrix.

> [!WARNING] BENCHMARK-GATED — the shipped OPC mode (`cflibs/inversion/physics/opc.py`) is opt-in
> and gated by the real-steel held-out guard (`RMSEP_GUARD=11.0`). On the 36-sample steel gate it
> took held-out RMSEP from 39.04 → 16.48 (single-standard) → **10.12 wt%** (robust geometric-mean
> $F$ over well-conditioned standards) — a 4× reduction (PRE-RESET; mechanism retained). Three
> honesty rules are structural: standard *selection* uses only in-sample certified truth, $F$ is a
> geometric mean robust to one collapsed standard, and leave-one-out prevents a scored standard
> from seeing its own $F$. See `docs/research/real-steel-opc-promotion.md` and cross-link the
> error budget in [error-budget-and-falsification](error-budget-and-falsification.md).

Notably, once OPC is present, an additional per-spectrum self-absorption boost **regresses**
accuracy (10.12 → 11.35 wt%): the matrix-matched $F$ already absorbs the systematic matrix
self-absorption bias, so a second correction over-corrects. Self-absorption stays an opt-in
lever for the OPC-free regime only.

---

## 5. C-sigma and columnar density {#csigma-columnar}

### 5.1 The C-sigma graph

The C-sigma (C$\sigma$) method of Aragón & Aguilera [@aragon2014; @aguilera2015] generalizes the
single-element curve of growth so that lines of *different elements sharing an ionization stage*
pool onto one graph. Each line is one point with

$$
x = \log_{10}(C\,\sigma), \qquad y = \log_{10}\!\big(I/B(\lambda,T)\big),
$$

where $\sigma$ is the absorption cross-section (from atomic data, $T$, $n_e$) and $B(\lambda,T)$
is the Planck blackbody radiance. Because the ordinate is built on the full radiative-transfer
relation for a homogeneous LTE slab, optically thin and thick lines lie on the *same* master
curve: slope $+1$ in the thin limit ($I\propto C\sigma$), saturating to $+1/2$ then flattening as
$C\sigma$ grows — the classic curve-of-growth. A common fit yields $T$ and composition together,
without excluding self-absorbed lines. Implemented (strictly opt-in) in
`cflibs/inversion/physics/csigma.py`; reported average relative error <10% on fused-glass rock
[@aragon2014].

### 5.2 The load-bearing verdict: C-sigma is the classic inverse repackaged {#csigma-is-classic}

> [!NOTE] FORMAL — unconditional identity:
> `lean:CflibsFormal/Alt/CSigma.lean#csigmaComposition_eq_classicComposition`. For all positive
> intensities, `csigmaComposition = Classic.classicComposition`.

Our formal spec proves that the "Saha-Boltzmann C-sigma" master-line estimator and the classic
per-species estimator are the **same algebraic left-inverse in two log/exp packagings** — the
per-species density inverses coincide (`csigmaDensity_offset_eq_classicDensity`), so the composed
compositions are literally equal as functions of all positive intensities
(`docs/M-spec-estimator-routing-policy.md` §0). The routing consequences are mandatory:

- **Do not count classic and C-sigma as two votes.** Do not average them; do not report "two
  estimators agree." Their agreement (`csigma_agrees_classic`) is a *definitional identity*
  applied to a forward spectrum and therefore carries **zero independent corroboration**.
  Counting it as corroboration is a category error — you are counting the same estimator twice.
- **Keep exactly one packaging**, chosen by numerical conditioning only: the C-sigma master-line
  offset form avoids one division and is preferable when per-species partition-function ratios
  are ill-scaled; otherwise the classic form is fine. This is a floating-point choice, never a
  physics choice.
- The genuinely independent same-spectrum cross-check is **OLS Boltzmann vs classic**, which
  differ off the noise-free fixpoint and whose gap *is* a data-quality signal (small ⇒ low noise
  / valid thin assumption; large ⇒ suspect self-absorption or misidentification).

**Methodological principle for this wiki: count independent evidence, not repackaged
identities.** When two methods provably reduce to the same inverse, their agreement is a unit
test of the algebra, not a physics confirmation.

### 5.3 The C-sigma partition-function bug (method verdict) {#csigma-u-bug}

In the head-to-head (`docs/M-spec-solver-head-to-head.md`), C-sigma finished worst in both bases
(median RMSE 17.71 wt% mass, 15.63 element-number, vs classic iterative 2.27 / 4.06; PRE-RESET).
The dominant cause was a real bug the oracle was *built to catch*: the bare fit's
`relative_concentrations` are $C_s/U_s$, **not** a composition — the partition function is omitted
from the cross-section and the bare fit has no DB to restore it. On the verified oracle fixture
($U = 3.88/3.09/4.71$), the raw fit returned $[0.468,0.387,0.145]\approx C/U$, not the truth
$[0.5,0.3,0.2]$; applying $\times U_s(T)$ recovered $[0.491,0.324,0.185]\approx$ truth (a ~25×
worst-case error corrected). `solve_csigma_composition(obs, db)` applies the correction. Even
fixed, C-sigma is intrinsically sensitive on real LIBS (needs accurate cross-sections, the
optically-thin/COG regime, and a well-conditioned common-line fit), so it does not beat the
classic iterative solver on real spectra — but the surviving lesson is
[atomic-data quality and $U(T)$ fidelity](atomic-data-and-datasets.md), the payoff of validating
solvers against the formal oracle.

### 5.4 CD-SB — exploit self-absorption instead of avoiding it

Columnar-density Saha-Boltzmann [@cdsb2013] turns the worst (self-absorbed) lines into the most
informative by inverting them for the elemental columnar density $N_s\ell$, removing the
detector-response dependence and reported error <4% on CaO in limestone. It is the preferred
$T$/composition route when resonance/strong lines dominate (late gates, high concentrations).
The curve-of-growth ratio inversion is well-posed: the ratio of two curve-of-growth line
intensities is strictly antitone (hence injective) in column density
(`lean:CflibsFormal/CurveOfGrowth.lean#cogRatio_strictAntiOn`), so a measured ratio inverts to a
unique optical depth $\tau$, which then feeds the correction.

---

## 6. Self-absorption: the observable-anchored corrector {#cdsb-self-absorption}

Self-absorption multiplies the thin intensity by the escape factor
$\mathrm{SA}(\tau) = (1-e^{-\tau})/\tau \in (0,1]$
(`lean:CflibsFormal/SelfAbsorption.lean#selfAbsorptionFactor_le_one`; derived, not assumed —
`selfAbsorbedIntensity_eq_slab` builds it from radiative transfer). Optically thick lines fall
below the true Boltzmann line, so the naïve slope overestimates $T$ (pitfall P9). Three treatment
tiers exist:

| Tier | Method | Requirement | Refs |
|------|--------|-------------|------|
| Avoid | select low-$gA$, high-$E_k$ lines; doublet-ratio test | line diversity | [@tognoni2010] |
| Correct | escape-factor / internal-reference / Planck-function correction | an **observable** anchor | [@bulajic2002; @sun2009; @elsherbini2005; @volker2023] |
| Model | C-sigma / CD-SB fold $\tau$ into the forward model | COG regime | [@aragon2014; @cdsb2013] |

The **current preferred path is observable-anchored correction**: every per-line correction
factor is tied to a measurement of the spectrum (line profile, high-$E_k$ reference line, or
measured/Stark width ratio), never to the current composition iterate. The distinction is not
cosmetic — it is the difference between a stable corrector and a divergent feedback loop.

> [!CAUTION] FALSIFIED: Composition-derived per-line $\tau$ improves accuracy
> - **Claim:** compute $\tau$ from the composition-derived lower-level column density and correct
>   each line for thick-line saturation.
> - **Predicted:** lower held-out RMSEP on ChemCam BHVO-2.
> - **Observed:** RMSEP *increased* — positive-feedback loop (over-attributed Fe → bigger $\tau$
>   → bigger intensity boost → bigger intercept $b_{\mathrm{Fe}}$ → closure gives Fe even more
>   mass → bigger $\tau$…).
> - **Verdict:** REJECTED; replaced by the observable-anchored corrector.
> - **Evidence:** overhaul audit `02-inversion-solver.md` finding F4;
>   `cflibs/inversion/physics/self_absorption_observable.py`.
> - **Date:** 2026-07-02

The single-thick-line-with-unknown-$\tau$ case is **not identifiable** at all
(`lean:CflibsFormal/SelfAbsorption.lean#selfAbsorption_breaks_identifiability`: two $(N,\tau)$
pairs give one intensity), which is the licensing theorem for the M7 refuse-to-report path — the
solver must refuse rather than invent a number. Either $\tau$ is supplied independently, or it is
recovered from a multi-line curve-of-growth ratio (§5.4), or the pipeline flags the line.

---

## 7. Closure, ILR, and Aitchison geometry {#closure-ilr-aitchison}

### 7.1 Closure is a projection, not the estimator

$\sum_s C_s = 1$ is a **constraint that projects a density vector onto the standard simplex**, not
a mechanism that estimates composition. The estimator is the per-species inverse (§4.1); closure
merely normalizes and fixes the otherwise-unobservable $F$. The formal statement:
`lean:CflibsFormal/Closure.lean#composition_sum_one` (the projected vector sums to one),
`composition_nonneg`, `composition_mem_stdSimplex`, and crucially `composition_smul_invariant` —
closure is invariant to any positive rescaling of the densities. Two consequences:

1. **A missing element inflates every detected element.** Closure over a subset $D$ of species
   gives $C_s^{\text{rec}} = N_s / \sum_{t\in D} N_t$, which exceeds the true fraction by the
   inflation factor $\text{(total density)}/\text{(detected density)} \ge 1$
   (`lean:CflibsFormal/MatrixEffects.lean` `one_le_inflationFactor`,
   `recoveredComposition_eq_inflation`). The absolute recovered composition is *matrix-dependent*
   (`recoveredComposition_absolute_matrix_dependent`), but **ratios of detected elements are
   matrix-invariant** (`lean:CflibsFormal/MatrixEffects.lean#recoveredComposition_ratio_matrix_invariant`).
2. **Prefer log-ratios over closure wt%.** For the DED/tracking deliverable — where the element
   set is known and precision on *ratios* matters more than absolute wt% — report
   $\ln(N_i/N_j)$, which is untouched by the missing-element inflation and by the $F$ ambiguity.

### 7.2 The simplex has log-ratio geometry (Aitchison)

Compositional data live on the simplex, whose natural geometry is not Euclidean but Aitchison
log-ratio geometry [@aitchison1982]. Two coordinate systems matter:

- **Additive log-ratio (ALR):** $\eta_s = \ln(C_s/C_{s_0})$ against a reference element; the
  inverse is a softmax. Simplest for unconstrained optimization/MCMC (closure is automatic).
- **Isometric log-ratio (ILR):** an orthonormal basis on the simplex preserving Aitchison
  distances [@egozcue2003]; the right coordinate for computing composition RMSE and for priors
  that must be isotropic on the simplex.

The pipeline exposes an ILR/softmax closure family (`cflibs/inversion/physics/closure_strategy.py`,
`cflibs/inversion/physics/softmax_closure.py`) alongside the standard closure
(`cflibs/inversion/physics/closure.py::ClosureEquation`). The head-to-head found the closed-form
ILR solver competitive in the element-number basis (5.96 vs classic 4.06) but behind in the
production/mass basis (11.51), partly a fairness caveat — ILR was run with standard closure while
iterative used the geological oxide closure (§8), which is worth a lot on those samples
(PRE-RESET). The Aitchison metric is the correct scoring basis;
see [benchmarks-reliability-workflows](benchmarks-reliability-workflows.md) for the
composition-metric discussion.

---

## 8. Matrix and oxide closure {#matrix-oxide-closure}

For geological/mineral samples oxygen is ubiquitous but its LIBS lines are weak or outside the
window, so plain elemental closure omits a major constituent and inflates the metals (§7.1). Two
stoichiometric variants close the gap [@tognoni2010]:

1. **Oxide closure** — assume each metal $M$ is present as its most stable oxide (SiO$_2$,
   Al$_2$O$_3$, Fe$_2$O$_3$, MgO, CaO, TiO$_2$, …) and impose $\sum_{\text{oxides}} w_{\text{ox}} = 1$
   with $w_{\text{ox}} = w_M\,(M_{\text{ox}}/M_M)$. Used in ChemCam/SuperCam CF-LIBS
   [@manelski2024].
2. **Stoichiometric correction** — known oxide stoichiometry fixes $n_O/n_M$ ratios; oxygen is
   added to the closure sum from the inferred metal oxides, changing the denominator consistently.

These are realized in the matrix/oxide closure strategies
(`cflibs/inversion/physics/matrix_effects.py`, `closure_strategy.py`). The oxide step is the
single largest lever on the SuperCam lab-calibration samples in the head-to-head — the reason the
oxide-closed iterative solver beat standard-closure ILR in the mass basis.

### 8.1 Mole fractions are not mass fractions

> [!IMPORTANT] CF-LIBS closure returns **mole (number) fractions**. Comparison to nominal wt%
> requires $w_s = C_s M_s / \sum_j C_j M_j$.

Omitting the molar-mass conversion is one of the most frequently shipped CF-LIBS bugs; it
produces systematic errors proportional to the deviation from the average molar mass — up to
353% for carbon in steel [@volker2024]. Any reported comparison must state its basis, and the
two must match. This wiki treats "which basis" as a first-class reporting decision, not a
detail.

---

## 9. Bayesian and optimal-estimation framing {#bayesian-oe-inversion}

The classic iterative loop returns a point estimate; the modern framing places a posterior over
the state $x=(T,n_e,C_1,\dots,C_S)$ with the CF-LIBS forward model $F(x)$ in the likelihood
[@kruger2024]. Two tiers:

**Optimal estimation (Rodgers Gauss-Newton).** Minimize
$J(x)=\tfrac12(y-F(x))^\top S_e^{-1}(y-F(x))+\tfrac12(x-x_a)^\top S_a^{-1}(x-x_a)$;
the linearized posterior covariance is $\hat S=(K^\top S_e^{-1}K+S_a^{-1})^{-1}$ and the averaging
kernel $A=\hat S K^\top S_e^{-1}K$ gives degrees of freedom for signal $d_s=\mathrm{tr}(A)$. OE is
a fast first-pass and yields a linearized uncertainty, but is inadequate when $T$ enters
exponentially (the CF-LIBS posterior is non-Gaussian and ridge-shaped). *(Rodgers 2000, Inverse
Methods for Atmospheric Sounding — book, no DOI; cited in prose only per this wiki's DOI rule.)*

**Full posterior (NUTS/HMC).** For the exponential $T$-sensitivity and the $T$–$n_e$ degeneracy,
sample the posterior [@kasim2019; @bowman2024; @oliver2024]. The degeneracy — lowering $T$ can be
compensated by raising total density — is broken only by including at least one ionic line per
element (the Saha leverage). Composition priors must respect the simplex: use ALR/ILR
reparameterization or a Dirichlet, **never** independent Uniform[0,1] priors on each $C_s$ (which
give incorrect geometry and truncation artefacts). The physics-only Bayesian estimators
[@oliver2024; @bowman2024] give calibrated posteriors on $(T,n_e)$ without the banned ML
surrogates. The JAX forward model for MCMC is
`cflibs/inversion/solve/bayesian.py::BayesianForwardModel`; full derivations in
[bayesian-oe details](frontier-methods.md) and `docs/v4/overhaul/literature/bayesian-oe.md`.

The adoption question — *when to trust a converged forward-model fit over the iterative solver* —
is unresolved and is a posterior-uncertainty (not point-accuracy) problem: a converged joint fit
can beat the iterative solver yet lack an adoption gate (`project_forward_model_solvers_work`).
That gate is tracked in [error-budget-and-falsification](error-budget-and-falsification.md), not
asserted here.

---

## 10. Method comparison matrix {#method-comparison-matrix}

What each method assumes, the regime it wins in, and — the column that matters — **what it
provably removes** versus what it leaves in the error budget.

| Method | Assumes | Provably removes | Leaves in the budget | Independent evidence? |
|--------|---------|------------------|----------------------|-----------------------|
| Classic CF-LIBS | thin, LTE, stoichiometric | scalar $F$ (closure) | atomic data, SA, non-LTE, ablation | baseline |
| Multi-element SB | + shared $T$ across species | $F$; reduces $T$ variance | same as classic | pooled-fit robustness |
| OPC | + one matrix-matched standard | matrix bias (relative-sensitivity $F$) | needs the standard; not standardless | per-matrix trueness |
| C-sigma | thin+thick on COG | (same inverse as classic) | **nothing extra** — it *is* classic | **none** (identity §5.2) |
| CD-SB | strong/resonance lines, COG | detector response; uses SA | $\tau$ estimation error | column-density check |
| SA-corrected CF | an observable $\tau$ anchor | thick-line bias (if anchored) | anchor quality; diverges if composition-fed | observable anchor |
| Bayesian/OE | prior + noise model | (nothing new physically) | same physics; adds calibrated UQ | posterior uncertainty |

Two rows carry the load. **C-sigma removes nothing the classic inverse does not** — it is the same
left-inverse, so its "independent evidence" cell is *none* (§5.2). **OPC removes matrix bias but
is not standardless** — it trades the closure assumption for a matrix-matched standard, honest and
excellent when the matrix is known (DED), a category error otherwise (§4.2). The
modern-methods rows of the 123-method cross-discipline catalog
(`docs/research/2026-06-19-cross-discipline-sota.md`) that are physics-only-compatible and rank as
the highest-leverage upgrades over classic CF are: **observable-anchored SA correction**
[@bulajic2002; @volker2023], **precomputed SA + OPC** (2026), **Monte-Carlo/full-spectrum
forward-fitting** [@gornushkin2022; @hermann2017], and **multiplet-aware Boltzmann fits**
[@volker2023multiplet]. The ML-supported CF variant [@favre2024] is reference-only under the
physics-only constraint. Full rows, integration difficulty, and the runtime routing table are in
[impl-literature-methods](impl-literature-methods.md).

---

## 11. What correct code MUST do — checklist {#checklist}

**Boltzmann / Saha-Boltzmann plane**
- [ ] Ordinate is $\ln(I_{ki}\lambda_{ki}/(g_kA_{ki}))$ — carry $\lambda$; it biases slope *and*
      intercept when the line set spans a wide $\lambda$ range (§2.2).
- [ ] $g_k = 2J_k+1$ for the **upper** level; $k_B$ units match $E_k$ units (eV/K vs cm$^{-1}$/K).
- [ ] Shift ionic lines to $E_k^{\text{eff}} = E_k^{\text{ion}} + \chi_s$; apply the Saha
      $y$-correction; fit one common slope across all species (§3).
- [ ] Evaluate $U_s(T)$ at the current iterate's $T$, summed to a cutoff consistent with the Saha
      $\Delta\chi$ [@barklem2016; @stewart1966].

**Closure and basis**
- [ ] Treat closure as a projection: report detected-element **ratios / log-ratios**, which are
      matrix-invariant; flag that absolute wt% is matrix-dependent (§7.1).
- [ ] Convert mole → mass fractions before comparing to wt% [@volker2024]; state the basis.
- [ ] Use ALR/ILR or Dirichlet on the simplex; never independent Uniform[0,1] per element (§7.2, §9).

**Method identity and voting**
- [ ] Do **not** count classic and C-sigma as two agreeing estimators — it is an unconditional
      identity, zero independent corroboration (§5.2). Keep one packaging, chosen by conditioning.
- [ ] If using bare C-sigma output, restore $\times U_s(T)$ — `relative_concentrations` are
      $C_s/U_s$, not composition (§5.3).
- [ ] The only legitimate same-spectrum two-estimator vote is OLS Boltzmann vs classic (differ
      off the noise-free fixpoint).

**Self-absorption**
- [ ] Anchor every per-line correction to an **observable**, never to the recovered composition
      (the composition-fed $\tau$ loop is FALSIFIED, §6).
- [ ] Refuse (do not fabricate) on a single thick line with unknown $\tau$ — not identifiable.

**OPC / known-matrix mode**
- [ ] Keep OPC opt-in and gated; select standards from in-sample truth only; robust ($F$ =
      geometric mean); leave-one-out. Do not stack a second SA boost on top of an OPC-matched $F$ (§4.2).

**LTE validity**
- [ ] McWhirter is necessary, not sufficient; also check the relaxation-time criterion
      [@cristoforetti2010] and partial-LTE level cuts [@cristoforetti2013]; $\Delta E$ = resonance
      energy of the dominant species.

---

## See also

- [classical-quantification](classical-quantification.md) — calibration-curve and internal-standard baselines the CF family is measured against.
- [libs-physics](libs-physics.md) — Saha-Boltzmann forward model, LTE validity, partition functions.
- [frontier-methods](frontier-methods.md) — Bayesian/OE retrieval, full-spectrum forward-fitting, manifold methods.
- [formal-spec](formal-spec.md) — the notation authority and the Lean theorems cited here.
- [error-budget-and-falsification](error-budget-and-falsification.md) — the falsification ledger (composition-fed $\tau$, joint-solver adoption gate) and the dominant modern error terms.
- [impl-literature-methods](impl-literature-methods.md) — the runtime routing table, solver code, and the full 123-method catalog rows.
- [atomic-data-and-datasets](atomic-data-and-datasets.md) — the atomic-data-quality floor that dominates once $F$, SA, and $T$ are handled.
- [benchmarks-reliability-workflows](benchmarks-reliability-workflows.md) — Aitchison/ILR scoring, the ASD59 reset, and the held-out gates.
