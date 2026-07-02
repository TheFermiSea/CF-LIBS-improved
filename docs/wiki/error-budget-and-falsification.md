---
slug: error-budget-and-falsification
title: "Error Budget, Program Conscience & Falsification Ledger"
chapter: error-budget-and-falsification
order: 0
status: review
register: review
summary: >
  The science-first differentiator: the measured CF-LIBS error budget (inputs dominate the
  solver by ~5 orders of magnitude — algorithm floor RMSE 2.9e-6 vs atomic-data mismatch 0.171),
  the eight root-cause clusters, the two-goal ranking, and the durable falsification ledger of
  verified dead-ends (composition-derived tau, Lawler-anchoring, SNR-gating, stage-III Cr,
  "joint beats iterative") so future agents do not re-attempt them.
tags: [error-budget, falsification, do-not-do, accuracy-first, atomic-data, self-absorption, reliability, opc]
updated: 2026-07-02
benchmarks_pre_reset: true
sources:
  - "@tognoni2007"
  - "@tognoni2010"
  - "@ciucci1999"
  - "@cristoforetti2010"
  - "@aragon2014"
  - "@cdsb2013"
  - "@volker2023"
  - "@bulajic2002"
  - "@cavalcanti2013"
  - "@volker2024"
  - docs/research/accuracy-first-roadmap.md
  - docs/research/physics-first-principles-audit.md
  - docs/research/real-data-accuracy-program-summary.md
  - docs/research/real-steel-opc-promotion.md
  - tests/benchmarks/ded_precision/NOISE_FINDINGS.md
  - docs/v4/overhaul/adversarial/ADJUDICATION.md
  - cflibs/inversion/physics/self_absorption_observable.py
  - cflibs/inversion/physics/opc.py
  - cflibs/inversion/solve/iterative.py
code_refs:
  - cflibs/inversion/physics/self_absorption_observable.py::ObservableSelfAbsorptionCorrector
  - cflibs/inversion/physics/opc.py::calibrate_opc
  - cflibs/inversion/solve/iterative.py::IterativeCFLIBSSolver
  - cflibs/inversion/physics/line_selection.py::select_lines_by_policy
  - cflibs/inversion/physics/stark_ne.py::measure_stark_ne
  - cflibs/plasma/lte_validator.py::LTEValidator
lean_refs:
  - CflibsFormal/SahaInverse.lean#saha_joint_identifiability
  - CflibsFormal/MatrixEffects.lean#recoveredComposition_ratio_matrix_invariant
related: [atomic-data-and-datasets, benchmarks-reliability-workflows, classical-quantification, cf-libs-family, formal-spec]
supersedes:
  - docs/research/accuracy-first-roadmap.md
  - docs/research/real-steel-accuracy-levers.md
  - docs/research/real-steel-opc-promotion.md
  - docs/research/real-data-accuracy-program-summary.md
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Error Budget, Program Conscience & Falsification Ledger

This chapter is the CF-LIBS program's **conscience**: the crown-jewel error budget that says *the
inputs dominate, not the solver*, and the durable **falsification ledger** that records every
verified dead-end so no future agent re-spends the budget on it. It is a review-register page —
equation-forward, cited, and organized as a structured register you read *instead of* re-deriving
the same negative results.

> [!IMPORTANT] PRE-RESET NUMBERS — the atomic DB was rebuilt (ASD59, 203k lines) on 2026-07-02;
> the composition/F1/RMSE figures below are **NOT current**. Every magnitude here predates the
> reset. The **mechanisms, rankings, and falsification verdicts are retained**; treat the specific
> wt%/RMSE magnitudes as dead and re-measure on the reset DB before quoting any number as current.
> See [Atomic data & datasets](atomic-data-and-datasets.md) for the reset baseline.

The chapter has three parts, each a distinct register:

1. **[The Error Budget](#part-1)** — where the measured real-data error actually lives (7 levers,
   8 root-cause clusters), the two-goal ranking, and the accuracy-first program spine.
2. **[The Falsification Ledger](#part-2)** — first-class `[!CAUTION] FALSIFIED` blocks for every
   negative result the program has *paid for*.
3. **[The Do-Not-Do Decision Record](#part-3)** — the consolidated dead-end table (Claim /
   Predicted / Observed / Verdict / Evidence / Date).

---

## Part 1 — The Error Budget {#part-1}

### The thesis: inputs dominate, not the solver {#inputs-dominate}

The binding error in CF-LIBS on this codebase is **the inputs**, not the solver arithmetic. A
plan-v4 controlled round-trip (real forward model, real ChemCam/SuperCam spectra) measured three
anchor quantities that set the entire budget:

| Anchor term | Measured (pre-reset) | Meaning |
|---|---|---|
| **Inversion-algorithm floor** | RMSE $\approx 2.9\times10^{-6}$ mass-fraction | When atomic data and forward model agree, the solver recovers truth *essentially exactly*. |
| **Atomic-data line-list mismatch** | RMSE $\approx 0.171$ mass-fraction | Injected by atomic-data quality alone — **~5 orders of magnitude above the solver floor**, same order as the real-data composition error. |
| **Weak-emitter ill-posedness floor** | RMSE $\approx 0.20$ mass-fraction | Na/K/Si at ~$10^6$ dynamic range — fundamental ill-conditioning, not solver math. |

The conclusion is the spine of the whole program:

> The solver is already correct to $\approx 0$. The accuracy ceiling is set by **atomic-data
> quality, instrument calibration, self-absorption on optically-thick majors, and weak-emitter
> ill-posedness** — in that rough order. Optimizing the solver further (and *especially* optimizing
> it for speed) cannot move composition RMSE while the inputs dominate by five orders of magnitude.

Real-data tests were run **uncalibrated** (fixed-FWHM proxy, no wavelength-dependent response
$E(\lambda)$), and SuperCam composition RMSE was **20–26 wt%** — 4–10× the 1–3 wt% CF-LIBS norm
for well-calibrated, optically-thin, LTE plasmas [@tognoni2010]. Tognoni et al. established that
even an *ideal* analytical plasma has an intrinsic precision floor dominated by temperature
uncertainty [@tognoni2007] — so a chunk of the residual is irreducible by any point-estimate
method and must be reported as calibrated uncertainty, not chased as bias.

### The 7-lever error budget {#budget-table}

Ranked by share of the **measured** real-data accuracy gap. Where a lever has not been isolated
with its own flag-off/flag-on experiment, its share is **UNMEASURED** and the isolation experiment
is that lever's headline deliverable. This table supersedes the error-budget duplication that was
scattered across four research docs (roadmap §2, physics-audit Clusters A/B/E,
real-steel summaries).

| # | Lever | Share of measured gap | Expected gain / basis |
|---|-------|----------------------|-----------------------|
| 1 | **Atomic-data quality & gf-grade-aware selection** | **DOMINANT** — owns the 0.171 RMSE term. | Targets the full 0.171. Repo-measured: **Kurucz beat NIST by −5 to −6 wt% on line-rich SuperCam** (completeness). Grade-A (≤3%) vs D/E (50%) is a 3–17× per-line accuracy swing propagating ~1:1 into species bias [@ciucci1999; @tognoni2010]. |
| 2 | **Instrument calibration $E(\lambda)$ / LSF / axis** | **LARGE, partly UNMEASURED.** Uncalibrated → SuperCam 20–26 wt%. | $E(\lambda)$ does **not** cancel under closure; it *rotates* (T) and *scatters* (composition) the Boltzmann plot. Well-calibrated norm 1–3 wt% [@tognoni2010; @hermann2018]. |
| 3 | **Weak-emitter ill-posedness** | **LARGE** — owns the ~0.20 noisy floor (Na/K/Si, ~$10^6$ range). | No point-estimate fix; the honest response is **calibrated large CIs + reliability downgrade**, not a tighter wrong number [@tognoni2007]. |
| 4 | **Self-absorption (optically-thick majors)** | **MODERATE, conditional, UNMEASURED in-repo.** | Bounded to Ca/Na/Mg/Al/Fe. Literature ceiling ~27% → ~2% rel. error when SA dominates; ~10× when SA dominates [@bulajic2002]. Realistic repo gain: single-digit wt% on SA-heavy spectra. |
| 5 | **Stark $n_e$ identifiability & coverage** | **MODERATE, asymmetric. Headline gain already BANKED.** | Pin/seed implemented; residual is DB coverage — only **244/28,727 lines (0.85%)** carry `stark_b`. Stark on graded lines ~10–20% $n_e$ accuracy vs ~30% from a flat prior [@konjevic2002; @sahalbrechot2015; @gigosos2003]. |
| 6 | **LTE validity / multi-zone reliability** | **SMALL on RMSE; OWNS the trust surface.** | No unconditional RMSE drop. Gain = **conditional RMSE** (refuse-to-report) + a reported flag-fraction. Apparent single-T can diverge from true $T_e$ by tens of % in non-LTE conditions [@cristoforetti2010; @pietanza2010]. |
| 7 | **Uncertainty calibration & reporting** | **~0% of the point estimate; ~100% of trustworthiness.** | UQ never moves the mean — it makes error bars honest. Bring classical 95%-CI coverage into **[0.93, 0.97]** [@tognoni2007; @maali2019]. |

**Reconciliation to the anchors.** Levers 1+2+3 account for essentially all of the measured
real-data RMSE. Levers 4 and 5 correct *subsets* (optically-thick majors; $n_e$-degenerate
spectra) and cannot exceed those subsets. Levers 6 and 7 are reliability/reporting, not
RMSE-of-the-mean. This is internally consistent with algorithm-floor $\approx 0$: **every remaining
error is an input or an ill-posedness, never the solver.**

### The 8 root-cause clusters (A–H) {#clusters}

A first-principles physics audit deduplicated 21 raw findings into **eight root-cause clusters**.
Each was required to survive three lenses — shipped code (file:line), LIBS literature, and the
`cflibs-formal` Lean soundness envelope — before being treated as actionable.

| Cluster | Root cause | Strongest verdict |
|---|---|---|
| **A. Atomic-data scale** | Absolute + relative $g\!\cdot\!A$ error is a *systematic, source-correlated bias*, not random variance; partition completeness secondary. | CONFIRMED |
| **B. $n_e$ is imputed, not measured** | On real data $n_e$ is Earth-STP pressure-balance imputed (no Stark-B, no ion lines); real diagnostics exist unused. | CONFIRMED |
| **C. Saha ladder forward/inverse asymmetry** | Inverse abundance multiplier & charge balance truncate at stage II; forward uses stage III. | CONFIRMED |
| **D. Thin fit forward vs thick data** | Every *inference* forward is optically thin; the iterative path corrects observably, the full-spectrum/joint/Bayesian path does not. | WEAKENED |
| **E. Relative closure vs ratio/columnar reporting** | Default estimator normalizes $\sum C_s = 1$ over the *detected* set; ratios (the DED deliverable) already cancel the denominator but are not reported. | WEAKENED |
| **F. Forward completeness (geology)** | Molecular bands (AlO, CN, C2, TiO, CaO) and physical continuum entirely absent from the forward model. | CONFIRMED |
| **G. Oxide redox / instrument response** | Fixed oxidation state; wavelength-dependent $E(\lambda)$ off / unrecoverable on in-house data. | WEAKENED |
| **H. Temporal / spatial integration** | Single-snapshot single-zone fit of a gate-integrated, LOS-integrated cooling plasma. | WEAKENED |

Cluster A is the dominant real-data floor and is *independent* of the others — it should be
attacked first. Full mechanism per cluster lives in `docs/research/physics-first-principles-audit.md`;
the Lean soundness hooks (e.g. `CompositionIdentifiability.lean` assumes correct $A_{ki}$ — vacuous
under D/E-grade bias) are in [formal-spec](formal-spec.md).

### The two-goal ranking {#two-goals}

The program serves **two goals that rank the levers differently.** Conflating them is itself a
recurring error.

**(a) DED Ti-6Al-4V drift tracking** — a *constrained, known* element set {Ti, Al, V}; precision
and *ratios* matter far more than absolute wt%; nominal feedstock is a prior; oxides/geology are
out of scope. Prefer log-ratios $\ln(N_V/N_{Ti})$ over closure wt% (lean:`CflibsFormal/MatrixEffects.lean#recoveredComposition_ratio_matrix_invariant`
proves closure-normalized ratios are matrix-invariant).

| Rank | Lever | Why |
|---|---|---|
| 1 | Report Aitchison log-ratios, not closure wt% (Cluster E) | Near-zero effort; ratios already cancel the shared denominator that OPC was band-aiding. |
| 2 | Atomic-data scale — relative $g\!\cdot\!A$ self-calibration + absolute anchoring (A) | Dominant real-data floor; tightens T hence the Ti/Al ratio. |
| 3 | Wire the *observable* SA corrector into the full-spectrum/joint path (D) | Ti/Al majors saturate; stops the T-box-edge runaway. |
| 4 | Make the inverse Saha ladder = forward 3-stage (C) | Cheap correctness fix; matters for hot-core (≥1.2 eV) spectra. |
| 5 | Measure $n_e$ instead of imputing it (B) | *Second-order for DED composition by cancellation*; first-order for the trust/LTE surface. |

**(b) Absolute composition generally** — steel minors, geology, unknown matrices. Ranking:
A (dominant) → D (matrix Fe, geology majors) → E undetected-mass $1/(1-m)$ inflation → B
(ion-observed minors like steel Cr, LTE trust) → F/G/H (geology forward completeness) → C.

### Legacy-first, latency LAST (the standing directive) {#legacy-first}

> [!WARNING] BENCHMARK-GATED — the standing directive is binding: **ACCURACY / PRECISION /
> RELIABILITY take strict precedence over latency.** Sub-millisecond inference is the eventual goal
> but is **deferred** until the pipeline is scientifically robust. Never trade accuracy for speed.

The measured justification: the legacy iterative solver
(`cflibs/inversion/solve/iterative.py::IterativeCFLIBSSolver`) scored **median composition RMSE
0.076 vs the fast structured-GN 0.151 on real data — ~2× more accurate on 4/5 real spectra** — at
~2000× the cost. Three measured mechanisms make the legacy path the correct **accuracy host**
today:

1. **Stark $n_e$ exploitation** — the legacy solver consumes `measure_stark_ne` as its primary
   per-iteration $n_e$ update; the GN merely pins $n_e$ and cannot identify it from a flat prior
   (`ne_rel` $\approx 0.3$).
2. **Self-absorption placement** — modeling SA *inside* the fast GN **rails $n_e$**; the
   observable-gated SA correction belongs upstream in the legacy path, applied once before the
   Boltzmann fit.
3. **Atomic-data uncertainty consumption** — the legacy solver folds `aki_uncertainty` into
   fit-space $\sigma$; the fast GN *stores* `line_aki_uncertainty` but **never uses it**.

**Operating rule.** All accuracy-changing physics lands in the **legacy iterative path first**, is
benchmark-gated there, and only then is ported to the fast GN under jitpipe parity gates. Latency
is milestone M10 — **last, and explicitly gated on the accuracy program being complete and locked.**
Promoting a 2×-less-accurate solver now to buy speed would violate the directive.

> [!CAUTION] DEAD-END — "the manifold is the foundation / joint beats iterative / a faster solver
> is progress" are **contradicted claims**. Neither solver is uniformly best (see
> [F: joint-beats-iterative](#f-joint)); the manifold generator is stale; and the solver floor is
> $\approx 0$, so there is no accuracy to be won from the solver, only from the inputs. Do not carry
> these forward.

### What correct code MUST do (Part-1 checklist) {#part1-checklist}

- **Gate every accuracy-changing change** flag-off vs flag-on on the scoreboard — the repo has
  regressed **three times** on ungated scoring changes. Non-regression is mandatory, not optional.
- **Never quote a composition/F1/RMSE number without a `benchmarks_pre_reset` tag** and its
  calibration mode. Uncalibrated (Mode-C) results are **non-quantitative** and cannot flip a
  default.
- **Land accuracy physics in the legacy iterative path first**; port to the fast GN only under
  parity gates.
- **Attack Cluster A first** (atomic data) — it is the dominant, independent floor.
- **For the DED deliverable, report log-ratios**, not closure-normalized wt%.
- **Bound each lever's headline by the measured 0.171 atomic-data term** — no lever's asserted gain
  may exceed the input error it corrects.

---

## Part 2 — The Falsification Ledger {#part-2}

Honesty is a feature of this wiki. Every negative result the program *paid for* is recorded here as
a first-class block so it is never silently re-attempted. Each block is `Claim / Predicted /
Observed / Verdict / Evidence / Date`.

### F1 — Composition-derived per-line $\tau$ {#f-tau}

> [!CAUTION] FALSIFIED: Composition-derived per-line $\tau$ improves accuracy
> - **Claim:** compute optical depth $\tau$ from the composition-derived lower-level column density
>   and correct each line for thick-line saturation.
> - **Predicted:** lower held-out RMSEP on ChemCam BHVO-2.
> - **Observed:** RMSEP **increased** — a positive-feedback loop ($\tau$ feeds composition feeds
>   $\tau$). Enabling forward SA this way *collapsed* steel accuracy.
> - **Verdict:** REJECTED; replaced by the observable-anchored corrector, which corrects *observed*
>   intensities (doublet ratio, Planck-ceiling curve-of-growth) and never reads composition
>   [@bulajic2002; @volker2023].
> - **Evidence:** physics-first-principles-audit.md Issue 3 (finding F4);
>   `cflibs/inversion/physics/self_absorption_observable.py::ObservableSelfAbsorptionCorrector`.
> - **Date:** 2026-07-02

The genuine gap is that the *full-spectrum/joint/Bayesian* path does not yet use the observable
corrector the iterative path already runs. Wire the existing observable corrector there — do **not**
add a composition-fed $\tau$ degree of freedom. See [CF-LIBS family](cf-libs-family.md) for the
CD-SB / C-sigma alternatives that handle thickness in the forward model instead
[@aragon2014; @cdsb2013].

### F2 — OPC is a band-aid, not physics progress {#f-opc}

> [!CAUTION] FALSIFIED: One-Point Calibration (OPC) is accuracy progress
> - **Claim:** matrix-matched OPC $F$-factors are a fix for the atomic-data / mass-slosh error.
> - **Predicted:** OPC generalizes as a physics improvement.
> - **Observed:** OPC's per-element geometric-mean $F$ corrects a *constant per-element absolute
>   $g\!\cdot\!A$-scale bias* on matrix-matched standards — the coherent-scale-error signature of
>   Cluster A. It **wins on dominant-matrix alloys** (steel 39.04 → 10.12 wt% held-out; DED V/Ti
>   limiter 15.2% → 3.6%) but **"averages to $\approx 1$ on a full-range Fe-Co binary"** — a
>   per-element constant *cannot* represent a per-species-and-per-line scale error.
> - **Verdict:** OPC is a **calibration band-aid over Cluster A**, valid only in the known-matrix
>   regime; it is not a root-cause fix and its regime limit is a *regime* limit, not a data limit
>   (Co has 1022 usable DB lines). Fixing the atomic data attacks the root cause and should reduce
>   reliance on matrix-matched OPC.
> - **Evidence:** real-steel-opc-promotion.md; real-data-accuracy-program-summary.md ("Regime of
>   validity"); `cflibs/inversion/physics/opc.py::calibrate_opc` [@cavalcanti2013].
> - **Date:** 2026-06-27

Corollary that must survive: **L4 self-absorption is intentionally OFF in the winning OPC config
(v2)** — an intensity SA *correction* double-counts the matrix bias the OPC $F$ already absorbs
(overall 10.12 → 11.35 wt% regression when both are on). A selection change (dropping self-absorbed
lines) composes with OPC; an intensity correction does not.

### F3 — Lawler / Den Hartog lifetime-anchoring {#f-lawler}

> [!CAUTION] FALSIFIED: Lawler-anchoring the $A_{ki}$ scale recovers the 0.171 atomic-data loss
> - **Claim:** renormalize each species' $A_{ki}$ set to independently-measured radiative lifetimes
>   ($A_{ki} = \mathrm{BR}_{ki}/\tau_k$, Wisconsin-group practice) to remove a coherent per-species
>   scale error.
> - **Predicted:** lower held-out RMSEP; recovers most of the "Kurucz-beat-NIST" gap.
> - **Observed (2026-07-02 addendum):** the experiment ran (overlay DB, 2,304 lab-anchored lines,
>   production DB untouched). **Median $\ln(A_\text{anchored}/A_\text{NIST}) \approx 0$ for every
>   covered species** — NIST ASD already sources the Wisconsin gf-values there, so **no coherent
>   scale correction exists to apply.** Coverage of the pipeline's *actually-selected* lines is the
>   binding limiter (Fe I / Fe II: 0%); where covered, micro-improvements (Cr −0.49, Ni −0.19
>   RMSEP) but a **net held-out regression (+0.40)** via an Fe selection cascade.
> - **Verdict:** REJECTED on the default path; overlay stays **OPT-IN**. The 0.171 loss is **NOT**
>   Lawler-vs-NIST scale error. Remaining suspects: the specific older-source gf's of the selected
>   *persistent* lines (testable per-line against an independent Kurucz ingest), or a non-$A_{ki}$
>   mechanism. The *in-plasma relative-$gA$ self-calibration* (a distinct sub-lever) is unaffected —
>   it measures whatever lines are actually used.
> - **Evidence:** physics-first-principles-audit.md Addendum (2026-07-02).
> - **Date:** 2026-07-02

### F4 — SNR-gated line selection for minor-element drift {#f-snr}

> [!CAUTION] FALSIFIED: Dropping low-SNR lines improves noisy minor-element recovery
> - **Claim:** gate out lines below a per-shot SNR threshold to remove the noisy weak-line bias in
>   DED drift tracking.
> - **Predicted:** cleaner recovery of tracked minor elements (V, Al) under DED noise.
> - **Observed:** SNR-gating removes **exactly** the weak minor-element lines being tracked. On
>   noisy 10-shot Ti-6Al-4V: `min_snr=5` drove **V to 0.0 wt%** (all V lines gated out);
>   `min_snr=10` gated every line → solve failure.
> - **Verdict:** COUNTERPRODUCTIVE for minor-element drift. The answer to noise is **shot
>   averaging** (lowers the floor so weak lines stay measurable), NOT dropping low-SNR lines. The
>   `min_snr` parameter is kept (default 0 = OFF) for major-element-only scenarios and **must not**
>   be used when a tracked element is a weak minor constituent.
> - **Evidence:** tests/benchmarks/ded_precision/NOISE_FINDINGS.md ("SNR-gated line selection").
> - **Date:** 2026-06-26

Operational companion (retained): under DED noise the dominant bias is **readout noise on weak
lines** rectified by a positivity clip, which inflates V/Al and pulls Ti down via closure. Remove
the per-pixel clip (unbiased integral + median-edge baseline); the residual is a *selection* bias
(the log-Boltzmann step needs positive intensities). The **clean-floor bias is calibratable** and
largely cancels in *drift* (measured − nominal) — which is why ratios beat absolutes for DED.

### F5 — "Correct" stage-III partitions fix the Cr under-estimate {#f-cr}

> [!CAUTION] FALSIFIED: Fixing Cr III partition functions fixes the Cr under-estimate
> - **Claim:** Cr is under-estimated on steel/Inconel because Cr III partition data is wrong; supply
>   correct stage-III partitions.
> - **Predicted:** the Cr bias shrinks.
> - **Observed:** correcting Cr III partitions made Cr **WORSE**. Cr I/II/III all have real
>   `direct_sum_fit` partitions, so this is *not* a partition-data bug.
> - **Verdict:** the real cause is the **forward/inverse Saha-ladder asymmetry** (Cluster C): the
>   forward populates three stages ($1 + S_1 + S_1 S_2$) while the inverse abundance multiplier
>   truncates at stage II ($1 + S$). Improving one side of an asymmetric ladder *widens* the
>   mismatch. The residual Cr bias at late-gate T is the iterative Boltzmann+Saha **method's**
>   inherent approximation, not a data bug. Fix = make the inverse ladder identical to the forward.
> - **Evidence:** physics-first-principles-audit.md Issue 5 + reinterpreted-conclusion #3;
>   NOISE_FINDINGS.md ("Partition-data gaps"). **NOTE:** an earlier NOISE_FINDINGS note that
>   "partitions are fine" was itself corrected — the point is the *asymmetry*, not the partition
>   values.
> - **Date:** 2026-06-26

### F6 — "Joint beats iterative" (without an adoption gate) {#f-joint}

> [!CAUTION] FALSIFIED: The joint (full-spectrum) solver uniformly beats the iterative solver
> - **Claim:** the converged full-spectrum joint solver, fitting the forward directly with no
>   Saha-correction approximation, is uniformly better.
> - **Predicted:** lower composition RMSE across the board.
> - **Observed:** on clean Inconel625 the joint solver **fixes** the Cr under-estimate (24.3 vs
>   iterative 18.3, truth 22.5) — confirming the Cr bias is a *method* approximation — **but worsens
>   Ni/Mo/Nb**, so its overall RMSE is *higher* (~4.8 vs ~3.2 wt%). On real data the fast structured
>   GN was **2× less accurate** than legacy iterative (0.151 vs 0.076).
> - **Verdict:** OVERSIMPLIFICATION. **Neither solver is uniformly best**; they distribute error
>   differently. The open problem is the **adoption gate** — *when to trust a converged fit*
>   (posterior uncertainty is the signal), not "switch solvers." A per-element or ensemble strategy
>   is the research direction, not a one-line swap.
> - **Evidence:** NOISE_FINDINGS.md ("Iterative vs joint"); accuracy-first-roadmap.md §3;
>   real-data legacy-vs-GN measurement.
> - **Date:** 2026-06-26

### F7 — Selection levers help on healthy high-SNR data {#f-selection}

> [!CAUTION] FALSIFIED: Reliability/selection levers improve healthy high-SNR solves
> - **Claim:** conditioning-aware line selection, reliability ranking, and identifiability routing
>   raise accuracy on the standard pipeline.
> - **Predicted:** a measurable RMSE drop from the selection levers.
> - **Observed:** on healthy, high-SNR data the selection levers are **no-ops** — the pipeline is
>   *not* selection-constrained there. Reliability-ranking wins **only at a forced binding cap**
>   (−0.52 wt%, 17/20 spectra) — i.e. in a constrained/degraded regime, not the default one.
> - **Verdict:** selection/conditioning tools are **regime-specific**: they earn their keep only in
>   constrained or low-SNR conditions. Do not sell them as a general accuracy lever, and do not gate
>   them on healthy-data benchmarks (they will correctly show ~0 and be wrongly discarded).
> - **Evidence:** project formalization-optimization mission summary; oracle/reliability lever
>   benchmarks.
> - **Date:** 2026-06

### F8 — Earth-STP pressure-balance $n_e$ as a primary diagnostic {#f-ne}

> [!CAUTION] FALSIFIED: Earth-STP pressure-balance $n_e$ is a valid primary diagnostic
> - **Claim:** when no Stark-B / ion line is available, impute $n_e$ from an isobaric pressure
>   balance at hardcoded Earth STP (101325 Pa).
> - **Predicted:** a usable $n_e$ for the Saha correction.
> - **Observed:** it imputes $n_e \sim 1.4$–$2.7\times10^{17}\,\mathrm{cm^{-3}}$ versus realistic
>   late-gate LIBS $1$–$3\times10^{16}$ — **roughly a decade high** — and every ion line is remapped
>   to the neutral plane with that wrong $n_e$. Pressure balance is **not** a LIBS $n_e$ diagnostic
>   [@tognoni2010].
> - **Verdict:** RETIRE as anything but a last resort. For the *constrained low-IP DED target*
>   composition is $n_e$-insensitive by cancellation ($\propto n_e^0$), so this does not invalidate
>   the DED result — but **`overall_reliable` must never pass on an imputed $n_e$** (a spectrum
>   passing the trust gate on a fabricated parameter is a trust-surface defect). Measure $n_e$ from
>   the Saha-Boltzmann inter-stage offset or a Balmer Stark line instead
>   (lean:`CflibsFormal/SahaInverse.lean#saha_joint_identifiability`).
> - **Evidence:** physics-first-principles-audit.md Issue 4 + reinterpreted-conclusion #6;
>   `cflibs/inversion/solve/iterative.py` pressure-balance fallback.
> - **Date:** 2026-07-02

### Mechanism retained, magnitudes dead: the identifier-gate diagnoses {#identifier-mechanism}

Two diagnostics are carried forward for **mechanism only** — their F1/precision/recall numbers are
pre-reset *and* pre-identifier-fix, so the magnitudes are dead:

- **Vrábel universal-miss (Boltzmann-$R^2$ × cold-T).** The root cause of universal element misses
  on high-resolution echelle soil spectra was a **downstream physics-consistency gate**, not
  coverage: the ALIAS `boltzmann_r2_min = 0.85` gate, driven by an unphysically cold
  plasma-temperature estimate (~4000 K vs the expected 7000–12000 K). Lines were detected (SNR
  16–38), in the DB, and matched — but the wrong T made the predicted emissivities
  $g_k A_{ki}\exp(-E_k/k_B T)$ wildly wrong, collapsing $R^2$ and killing every element except the
  one whose matched lines sat in a narrow $E_k$ window. Self-absorption of the strong low-$E_k$
  resonance lines is the *upstream* cause of the cold-T mis-estimate. **Lesson:** make identifiers
  paper-faithful and temperature-aware *first*; a physics-consistency gate on a broken T is a
  universal-miss machine.
- **ALIAS specificity paradox (geological "line forest").** The vector-space specificity
  (IDF-style) weight *penalizes* peaks in crowded regions, so in an Fe/Mn/Cr-rich matrix it
  devalues the most reliable high-intensity lines and leans on weak "unique" lines that are often
  noise — the "bag-of-lines" fallacy that ignores Saha-Boltzmann line coupling. **Lesson:** the
  "common" lines are the physically reliable ones; a purely statistical identifier that devalues
  them craters precision on transition-metal matrices. [ALIAS: Noel et al. 2025 — citation-needed]

---

## Part 3 — The Do-Not-Do Decision Record {#part-3}

The durable home for verified dead-ends. **Before proposing any lever, check this table.** It
absorbs the accuracy-first program spine as the decision record; entries are ordered dead-ends
first, then verified-sound "do not fix," then rejected knob-work.

### Verified dead-ends — do NOT re-attempt {#dead-ends}

| # | Claim (what was tried) | Predicted | Observed | Verdict | Evidence | Date |
|---|---|---|---|---|---|---|
| D1 | Composition-derived per-line $\tau$ in the fit forward | Lower BHVO-2 RMSEP | RMSEP **increased** (positive-feedback loop) | REJECTED — use observable-anchored corrector | audit Issue 3 / F4; `self_absorption_observable.py` | 2026-07-02 |
| D2 | Lawler/Den Hartog lifetime-anchoring of $A_{ki}$ scale | Recover the 0.171 loss | Median $\ln(A/A_\text{NIST})\approx 0$; net +0.40 held-out regression | REJECTED default; opt-in overlay | audit Addendum | 2026-07-02 |
| D3 | SNR-gate weak lines for DED minor-element drift | Cleaner V/Al recovery | V → 0.0 wt%; solve fails at higher threshold | COUNTERPRODUCTIVE — shot-average instead | NOISE_FINDINGS | 2026-06-26 |
| D4 | "Correct" Cr III partitions to fix Cr under-estimate | Cr bias shrinks | Cr got **worse** | REJECTED — root cause is Saha-ladder asymmetry (Cluster C) | audit Issue 5; NOISE_FINDINGS | 2026-06-26 |
| D5 | Swap in the joint solver as uniformly better | Lower RMSE everywhere | Fixes Cr, worsens Ni/Mo/Nb; higher overall RMSE; 2× worse on real | OVERSIMPLIFICATION — need an adoption gate | NOISE_FINDINGS; roadmap §3 | 2026-06-26 |
| D6 | Sell selection/reliability levers as general accuracy | RMSE drop on the default pipeline | No-op on healthy high-SNR; win only at a forced binding cap | REGIME-SPECIFIC only | formalization mission | 2026-06 |
| D7 | Earth-STP pressure-balance $n_e$ as primary diagnostic | Usable $n_e$ | ~decade-high $n_e$; corrupts Saha remap; `overall_reliable` passes on a fabricated param | RETIRE to last resort; never pass trust gate on it | audit Issue 4 | 2026-07-02 |
| D8 | Add L4 intensity SA correction on top of OPC | Extra accuracy | Regression (10.12 → 11.35 wt%) — double-counts OPC's $F$ | KEEP SA OFF when OPC is on | opc-promotion | 2026-06-27 |
| D9 | Single-gate "2-point cooling quadrature" T fit | De-bias gate-integrated T | Underdetermined from one integrated spectrum (a knob) | DO NOT BUILD — use multi-gate routing where multiple delays exist | audit Issue 8 | 2026-07-02 |
| D10 | Swap in C-sigma columnar estimator *for the denominator argument* | Matrix-invariant ratios | $N_{Ti}$ still fit from Ti lines → ratio moves in both methods | REJECTED premise — report ratios; C-sigma's value is SA (Cluster D) | audit Issue 2 | 2026-07-02 |
| D11 | "Add $A_{ki}$ uncertainty" as the fix | Better composition | Already tracked/folded into WLS; a better *weight* is a no-op vs a *scale* bias | ANCHOR the scale, don't re-weight | audit Issue 1 | 2026-07-02 |
| D12 | Expect partition-completeness to move Ti/Cr/V/Fe > ~1 wt% | Several wt% | The ≥10–20% U deficits are alkalis/alkaline-earths, absent from DED/steel targets | LOW YIELD for these targets | audit Issue 1 | 2026-07-02 |
| D13 | Chase $n_e$ *magnitude* for the DED all-low-IP target | Composition accuracy | Composition $\propto n_e^0$ by cancellation; a decade sweep moved Ti-6Al-4V ~1–2% | DO $n_e$ work for the trust surface & ion-minors, not DED magnitude | audit Issue 4 | 2026-07-02 |
| D14 | Promote the fast GN / latency work now | Product-ready speed | 2× less accurate on real data; violates the standing directive | DEFER to M10, gated on M1–M9 locked | roadmap §3, §7 | 2026-06-20 |

### Verified physically sound — do NOT "fix" {#do-not-fix}

These were adversarially audited and **upheld**; changing them is regression risk with no upside.

- The single-$(T, n_e)$ Saha-Boltzmann *algebra* and the calibration-free closure ($F_\text{cal}$
  cancels; two-stage $n_e$ cancellation). The Boltzmann-plane inversion's $\ln(U_{II}/U_I)$
  cancellation is algebraically real — a prior "missing term" alarm was the false one.
- The homogeneous isothermal LTE slab RT formula $I = B(1-e^{-\kappa L})$ within its single-zone
  assumption; Voigt profiles (scipy `wofz` / Weideman JAX), Doppler width
  $\sigma = \lambda\sqrt{k_B T/mc^2}$, the Stark-width convention (FWHM@1e17/10000 K,
  Olivero-Longbothum deconvolution), and the McWhirter constant $1.6\times10^{12}$ (exact vs 5
  sources) [@cristoforetti2010].
- IPD / partition truncation at $\mathrm{IP}-\Delta\chi$ with Debye lowering [@stewart1966;
  @barklem2016] — physically correct cutoff; only level *completeness below it* is (minorly) at
  issue.
- `SpectralResponseCorrection` math **and its default-off for vendor-corrected ChemCam/SuperCam** —
  re-applying $E(\lambda)$ would double-correct.
- The McWhirter/Cristoforetti LTE validator framed as **necessary-not-sufficient** — sound; the
  issue is upstream (the fit never uses a self-consistent $n_e$) [@cristoforetti2010].

> [!NOTE] Two real, localized bugs the audit *did* confirm (not dead-ends — fix them): the H-α Stark
> reference width stored as electron-impact-only (~27× deficit vs ion-broadened Gigosos 2003, biases
> $n_e$ on real H-bearing spectra) [@gigosos2003], and the mole-vs-mass composition reporting defect
> — CF-LIBS closure yields **mole** fractions; omitting $w_s = C_s M_s / \sum_j C_j M_j$ can give
> up to +353% error on light elements in steel [@volker2024]. Both are in
> [benchmarks & reliability](benchmarks-reliability-workflows.md).

### Rejected knob-work — stop {#knob-work}

- **Do not** treat OPC $F$-factor tuning as accuracy progress — it band-aids Cluster A ([F2](#f-opc)).
- **Do not** re-run the full `pytest tests/` suite inside a sub-agent (stream-idle watchdog); use a
  narrow subset or background it from the parent.
- **Do not** quote any pre-reset composition/F1/RMSE number as current — the ASD59 rebuild
  invalidated all of them; re-measure first.

### What correct code MUST do (Part-3 checklist) {#part3-checklist}

- **Consult this ledger before proposing a lever.** If it is D1–D14, do not re-attempt without new
  evidence that the *observed* failure mechanism no longer applies.
- **Add a new dead-end here** (with Claim/Predicted/Observed/Verdict/Evidence/Date) the moment a
  lever is falsified — the ledger is only durable if it is kept current.
- **Distinguish a regime limit from a data limit** (OPC/Fe-Co; selection levers/healthy data): a
  lever that no-ops on healthy data is not thereby worthless.
- **Never let a reliability flag pass on an imputed/fabricated parameter** (D7).

---

## See also

- [Atomic data & datasets](atomic-data-and-datasets.md) — the ASD59 reset baseline; Cluster A detail.
- [Benchmarks, reliability & workflows](benchmarks-reliability-workflows.md) — the scoreboard,
  refuse-to-report gates, coverage bands, and the two confirmed bugs.
- [CF-LIBS family](cf-libs-family.md) — OPC, CD-SB, C-sigma, IRSAC self-absorption variants.
- [Classical quantification](classical-quantification.md) — Boltzmann/Saha inversion the budget rests on.
- [Formal spec](formal-spec.md) — the Lean soundness envelope (ratio matrix-invariance, Saha
  identifiability) the clusters are checked against.

---

## References (BibTeX seed for `docs/wiki/references.bib`)

<!-- Consolidator: merge these verified entries into docs/wiki/references.bib. Every DOI below was
     verified (asta/WebSearch/hand-corrected cross-discipline pack) on 2026-07-02. -->

```bibtex
@article{ciucci1999,
  author = {Ciucci, A. and Corsi, M. and Palleschi, V. and Rastelli, S. and Salvetti, A. and Tognoni, E.},
  title = {New Procedure for Quantitative Elemental Analysis by Laser-Induced Plasma Spectroscopy},
  journal = {Applied Spectroscopy}, year = {1999}, volume = {53}, number = {8}, pages = {960--964},
  doi = {10.1366/0003702991947612}, note = {verified: DOI resolved 2026-07-02}
}
@article{tognoni2007,
  author = {Tognoni, E. and Cristoforetti, G. and Legnaioli, S. and Palleschi, V. and Salvetti, A. and Mueller, M. and Panne, U. and Gornushkin, I.},
  title = {A numerical study of expected accuracy and precision in Calibration-Free Laser-Induced Breakdown Spectroscopy in the assumption of ideal analytical plasma},
  journal = {Spectrochimica Acta Part B}, year = {2007}, volume = {62}, number = {12}, pages = {1287--1302},
  doi = {10.1016/J.SAB.2007.10.005}, note = {verified: DOI resolved 2026-07-02}
}
@article{tognoni2010,
  author = {Tognoni, E. and Cristoforetti, G. and Legnaioli, S. and Palleschi, V.},
  title = {Calibration-free laser-induced breakdown spectroscopy: State of the art},
  journal = {Spectrochimica Acta Part B}, year = {2010}, volume = {65}, number = {1}, pages = {1--14},
  doi = {10.1016/j.sab.2009.11.006}, note = {verified: WebSearch 2026-07-02}
}
@article{cristoforetti2010,
  author = {Cristoforetti, G. and De Giacomo, A. and Dell'Aglio, M. and Legnaioli, S. and Tognoni, E. and Palleschi, V. and Omenetto, N.},
  title = {Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy: Beyond the McWhirter criterion},
  journal = {Spectrochimica Acta Part B}, year = {2010}, volume = {65}, number = {1}, pages = {86--95},
  doi = {10.1016/j.sab.2009.11.005}, note = {verified: cross-discipline pack 2026-07-02}
}
@article{pietanza2010,
  author = {Pietanza, L. D. and Colonna, G. and De Giacomo, A. and Capitelli, M.},
  title = {Kinetic processes for laser induced plasma diagnostic: A collisional-radiative model approach},
  journal = {Spectrochimica Acta Part B}, year = {2010}, volume = {65}, number = {8}, pages = {616--626},
  doi = {10.1016/j.sab.2010.03.012}, note = {verified: cross-discipline pack 2026-07-02}
}
@article{bulajic2002,
  author = {Bulajic, D. and Corsi, M. and Cristoforetti, G. and Legnaioli, S. and Palleschi, V. and Salvetti, A. and Tognoni, E.},
  title = {A procedure for correcting self-absorption in calibration free-laser induced breakdown spectroscopy},
  journal = {Spectrochimica Acta Part B}, year = {2002}, volume = {57}, number = {2}, pages = {339--353},
  doi = {10.1016/S0584-8547(01)00398-6}, note = {verified: WebSearch 2026-07-02}
}
@article{aragon2014csigma,
  author = {Aragón, C. and Aguilera, J. A.},
  title = {CSigma graphs: A new approach for plasma characterization in laser-induced breakdown spectroscopy},
  journal = {Journal of Quantitative Spectroscopy and Radiative Transfer}, year = {2014}, volume = {149}, pages = {90--102},
  doi = {10.1016/J.JQSRT.2014.07.026}, note = {verified: cross-discipline pack 2026-07-02}
}
@article{cristoforetti2013cdsb,
  author = {Cristoforetti, G. and Tognoni, E.},
  title = {Calculation of elemental columnar density from self-absorbed lines in laser-induced breakdown spectroscopy: A resource for quantitative analysis},
  journal = {Spectrochimica Acta Part B}, year = {2013}, volume = {79--80}, pages = {63--71},
  doi = {10.1016/J.SAB.2012.11.010}, note = {verified: cross-discipline pack 2026-07-02}
}
@article{voelker2023,
  author = {Völker, T. and Gornushkin, I. B.},
  title = {Investigation of a Method for the Correction of Self-Absorption by Planck Function in LIBS},
  journal = {Journal of Analytical Atomic Spectrometry}, year = {2023}, volume = {38}, pages = {336--349},
  doi = {10.1039/d2ja00352j}, note = {verified: cross-discipline pack 2026-07-02}
}
@article{cavalcanti2013,
  author = {Cavalcanti, G. H. and Teixeira, D. V. and Legnaioli, S. and Lorenzetti, G. and Pardini, L. and Palleschi, V.},
  title = {One-point calibration for calibration-free laser-induced breakdown spectroscopy quantitative analysis},
  journal = {Spectrochimica Acta Part B}, year = {2013}, volume = {87}, pages = {51--56},
  doi = {10.1016/J.SAB.2013.05.016}, note = {verified: cross-discipline pack (DOI-corrected) 2026-07-02}
}
@article{konjevic2002,
  author = {Konjević, N. and Lesage, A. and Fuhr, J. R. and Wiese, W. L.},
  title = {Experimental Stark Widths and Shifts for Spectral Lines of Neutral and Ionized Atoms (A Critical Review of Selected Data for the Period 1989 through 2000)},
  journal = {Journal of Physical and Chemical Reference Data}, year = {2002}, volume = {31}, number = {3}, pages = {819--927},
  doi = {10.1063/1.1525443}, note = {verified: cross-discipline pack (DOI-corrected) 2026-07-02}
}
@article{sahalbrechot2015,
  author = {Sahal-Bréchot, S. and Dimitrijević, M. S. and Moreau, N. and Ben Nessib, N.},
  title = {The STARK-B database VAMDC node: a repository for spectral line broadening and shifts due to collisions with charged particles},
  journal = {Physica Scripta}, year = {2015}, volume = {90}, number = {5}, pages = {054008},
  doi = {10.1088/0031-8949/90/5/054008}, note = {verified: cross-discipline pack 2026-07-02}
}
@article{gigosos2003,
  author = {Gigosos, M. A. and González, M. A. and Cardeñoso, V.},
  title = {Computer simulated Balmer-alpha, -beta and -gamma Stark line profiles for non-equilibrium plasmas diagnostics},
  journal = {Spectrochimica Acta Part B}, year = {2003}, volume = {58}, number = {8}, pages = {1489--1504},
  doi = {10.1016/S0584-8547(03)00097-1}, note = {verified: cross-discipline pack 2026-07-02}
}
@article{barklem2016,
  author = {Barklem, P. S. and Collet, R.},
  title = {Partition functions and equilibrium constants for diatomic molecules and atoms of astrophysical interest},
  journal = {Astronomy \& Astrophysics}, year = {2016}, volume = {588}, pages = {A96},
  doi = {10.1051/0004-6361/201526961}, note = {verified: cross-discipline pack 2026-07-02}
}
@article{stewart1966,
  author = {Stewart, J. C. and Pyatt, K. D.},
  title = {Lowering of Ionization Potentials in Plasmas},
  journal = {Astrophysical Journal}, year = {1966}, volume = {144}, pages = {1203--1211},
  doi = {10.1086/148714}, note = {verified: cross-discipline pack 2026-07-02}
}
@article{hermann2018,
  author = {Hermann, J. and Axente, E. and Craciun, V. and Taleb, A. and Pelascini, F.},
  title = {Local thermodynamic equilibrium in a laser-induced plasma evidenced by blackbody radiation},
  journal = {Spectrochimica Acta Part B}, year = {2018}, volume = {144}, pages = {82--86},
  doi = {10.1016/j.sab.2018.03.013}, note = {verified: cross-discipline pack 2026-07-02}
}
@article{maali2019,
  author = {Maali, A. and Shabanov, S. V.},
  title = {Error analysis in optimization problems relevant for calibration-free laser-induced breakdown spectroscopy},
  journal = {Journal of Quantitative Spectroscopy and Radiative Transfer}, year = {2019}, volume = {222--223}, pages = {236--246},
  doi = {10.1016/J.JQSRT.2018.10.029}, note = {verified: cross-discipline pack 2026-07-02}
}
@article{volker2024,
  author = {Völker, T. and Gornushkin, I. B.},
  title = {On the conversion between number and mass fractions in calibration-free LIBS},
  journal = {Journal of Analytical Atomic Spectrometry}, year = {2024},
  doi = {10.1039/D4JA00028E}, note = {verified: ADJUDICATION.md flaw #3 citation 2026-07-02}
}
```

<!-- UNVERIFIED — needs a DOI before it may be cited in prose; currently marked [citation-needed]:
     Noel et al. 2025 (ALIAS); Vrábel 2020 soil benchmark; Aitchison 1986 (compositional data). -->
