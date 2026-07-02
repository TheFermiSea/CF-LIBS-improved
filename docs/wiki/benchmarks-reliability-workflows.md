---
slug: benchmarks-reliability-workflows
title: "Reliability, Benchmarks, Uncertainty & Operational Workflows"
chapter: benchmarks-reliability-workflows
order: 0
status: stable
register: handbook
summary: >
  The trust-surface, evidence, and how-to-run chapter: refuse-to-report and per-element
  reliability gates, the five UQ surfaces, the benchmark harness and non-regression gate
  discipline, the campaign-1 anti-overfit protocol, the DED clean-floor benchmark, NIST-parity
  round-trip, the CLI, contributing conventions, and cluster deployment. Key operational fact:
  selection/refuse levers are no-ops on healthy high-SNR data and only bind in
  constrained/low-SNR regimes; scoring changes MUST be non-regression-gated (the repo regressed 3x).
tags: [reliability, uncertainty, benchmarks, refuse-to-report, anti-overfit, ded, cli, deployment, workflows]
updated: 2026-07-02
benchmarks_pre_reset: false
sources:
  - "@cristoforetti2010"
  - "@tognoni2010"
  - "@tognoni2007"
  - "@aguilera2007"
  - "@vehtari2017"
  - "@lei2018"
  - "@egozcue2003"
  - "@benjamini1995"
  - "@holtzman2015"
  - "@volker2023"
  - cflibs/inversion/physics/reliability.py
  - cflibs/inversion/physics/derived_thresholds.py
  - cflibs/inversion/physics/uncertainty.py
  - cflibs/inversion/physics/conformal.py
  - cflibs/inversion/physics/coverage.py
  - cflibs/benchmark/posterior_metrics.py
  - cflibs/benchmark/scoreboard.py
  - cflibs/validation/round_trip.py
  - scripts/campaign1/README.md
  - tests/benchmarks/ded_precision/NOISE_FINDINGS.md
  - docs/VALIDATION_METRICS.md
  - docs/User_Guide.md
  - docs/Deployment.md
  - docs/research/physics-first-principles-audit.md
code_refs:
  - cflibs/inversion/physics/reliability.py::stark_saha_lte_gate
  - cflibs/inversion/physics/reliability.py::mcwhirter_min_ne
  - cflibs/inversion/physics/reliability.py::composition_error_bound
  - cflibs/inversion/physics/derived_thresholds.py::min_lines_per_element_for
  - cflibs/inversion/physics/uncertainty.py::MonteCarloUQ
  - cflibs/inversion/physics/conformal.py
  - cflibs/inversion/physics/coverage.py::tarp_coverage
  - cflibs/benchmark/posterior_metrics.py::PosteriorDiagnostics
  - cflibs/benchmark/scoreboard.py
  - cflibs/benchmark/composition_metrics.py::aitchison_distance
  - cflibs/validation/round_trip.py::GoldenSpectrum
lean_refs:
  - CflibsFormal/StarkBroadening.lean#stark_saha_lte_consistent
  - CflibsFormal/StarkBroadening.lean#mcWhirterBound
  - CflibsFormal/ErrorBudget.lean#olsSlope_stable_l2
  - CflibsFormal/CompositionRobustness.lean#composition_dist_vector_le
  - CflibsFormal/Robustness.lean#twoLineBeta_stable_sharp
related: [error-budget-and-falsification, formal-spec, classical-quantification, atomic-data-and-datasets, architecture]
supersedes:
  - docs/User_Guide.md
  - docs/Deployment.md
  - docs/M5-parameter-optimization-plan.md
  - docs/M-spec-target-sigma-wiring-findings.md
  - docs/M-spec-lever-comparison.md
  - docs/VALIDATION_METRICS.md
  - docs/arbor-integration.md
  - docs/dataset-sharding.md
  - docs/jax-compile-cache.md
  - docs/nfs-shared-data.md
  - docs/results-parquet-schema.md
  - docs/federation.md
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Reliability, Benchmarks, Uncertainty & Operational Workflows

This chapter is the trust surface, the evidence base, and the operator's manual for CF-LIBS.
It covers *when to believe a result* (reliability gates and the five uncertainty surfaces),
*how the pipeline is measured* (the benchmark harness, its metrics, and the non-regression
discipline that keeps scoring honest), and *how to run it* (CLI, cluster deployment, and the
contributing conventions). The single load-bearing operational fact, repeated throughout: on
healthy high-SNR laboratory data the reference pipeline is **not selection-constrained**, so the
line-selection and refuse-to-report levers are no-ops; they earn their keep only in
constrained, portable, or low-SNR regimes — and every change to a scoring path must be
non-regression-gated because the repo has regressed **three times** on ungated scoring edits.

Symbols follow [formal-spec.md](./formal-spec.md); this chapter never redefines a canonical
symbol. Wavelengths are academic here — where they appear, the atomic DB stores **air**
wavelengths per NIST/ASD. Numeric accuracy figures below are ASD59-reset baseline unless a
figure is explicitly labelled otherwise.

> [!IMPORTANT] RESET LINE — accuracy numbers in this chapter are ASD59-reset baseline
> (2026-07-02) or are labelled mechanism-only. The DED clean-floor figures (§7) were measured on the
> reset DB and, being a self-consistency benchmark (same DB forward and inverse), are essentially
> reset-independent; they are the same run reported as the ASD59 clean floor in
> [atomic-data-and-datasets.md](atomic-data-and-datasets.md). The campaign-1 F1/RMSE **thresholds** (§6)
> are gate policy, not measured results.

---

## 1. Reliability discipline — the refuse-to-report gate {#reliability-discipline}

The reliability surface answers one question: *is this spectrum something any CF-LIBS estimator
can honestly report on?* It is a **gate**, not a correction. Its formulas are verbatim mirrors of
machine-verified theorems in the companion `cflibs-formal` Lean spec, so the shipped gate cannot
silently drift from the proof (`cflibs/inversion/physics/reliability.py`).

### 1.1 The M7 refuse-to-report path and M8 per-element flags {#m7-m8}

Two tiers of trust flag exist:

| Flag | Scope | Semantics |
|------|-------|-----------|
| `overall_reliable` (M7) | whole spectrum | the estimator is willing to report at all; false ⇒ **refuse to report** |
| per-element reliability (M8) | one element | this element's Boltzmann/Saha anchor is well-conditioned enough to quote |

The gate flags — it does not repair. A refused spectrum is one where the physics is degenerate
(a single-line species, a single thick line with unknown optical depth, no distinct-energy
temperature anchor) — cases where *no* estimator can recover the quantity, so reporting a number
would be dishonest. This is the identifiability boundary made operational
(`lean:CflibsFormal/Identifiability.lean` and its `selfAbsorption_breaks_identifiability`).

> [!CAUTION] DO-NOT: never let `overall_reliable` pass on an imputed n_e.
> On real data lacking Stark-B lines the electron density is currently imputed from an Earth-STP
> pressure balance (§9.3). A spectrum can therefore pass McWhirter on a **fabricated** parameter.
> The physics audit flags this as a trust-surface defect: the reliability flag must be evaluated
> against a *measured* n_e (Saha inter-stage offset or Balmer Stark), never an imputed one. See
> [error-budget-and-falsification.md](./error-budget-and-falsification.md) and
> `docs/research/physics-first-principles-audit.md` Issue 4.

### 1.2 The Stark↔Saha two-diagnostic cross-check is EVIDENCE, not identity {#stark-saha-crosscheck}

The centrepiece of the LTE gate is a cross-check between **two physically independent** electron
density diagnostics:

- **Stark route** — n_e from a measured line **width** (`n_e^Stark`).
- **Saha route** — n_e from a stage-intensity **ratio** (`n_e^Saha`).

These consume genuinely *different* observations (a width versus a ratio), so their agreement is
**empirical evidence** that LTE holds — it is not a definitional identity. The gate
(`stark_saha_lte_gate`, mirroring `lean:CflibsFormal/StarkBroadening.lean#stark_saha_lte_consistent`)
certifies LTE self-consistent iff **both** hypotheses hold:

1. **Agreement** — the two estimates are within `rtol` (default 0.5, i.e. 50 % of their mean).
2. **Clears McWhirter** — their mean is at least the McWhirter LTE floor.

$$n_e \ge 1.6\times10^{12}\,\sqrt{T}\,(\Delta E)^3 \quad\text{(cm}^{-3}\text{, } T \text{ in K, } \Delta E \text{ in eV)}$$

The McWhirter bound (`mcwhirter_min_ne`, `lean:CflibsFormal/StarkBroadening.lean#mcWhirterBound`)
is proven monotone increasing in both $T$ and $\Delta E$ — a hotter plasma or a larger energy gap
demands a higher electron density for collisional processes to dominate radiative ones
[@cristoforetti2010]. On failure the gate names the violated hypothesis so the M7 path can route
the reason (`disagree` vs `below-mcwhirter`), with a degenerate non-positive density
short-circuiting to `invalid-ne`.

> [!NOTE] FORMAL — the agreement-plus-floor gate is proven in
> `lean:CflibsFormal/StarkBroadening.lean#stark_saha_lte_consistent`. The gate function is a
> line-by-line mirror; `tests/oracle/` conformance-pins it so a drift from the proof fails CI.

McWhirter is **necessary, not sufficient** — it can pass while LTE is still violated by temporal
or spatial non-equilibrium [@cristoforetti2010]. That is exactly why the independent-diagnostic
agreement check is layered on top: two channels agreeing *and* clearing the floor is stronger
evidence than the floor alone.

### 1.3 The certified conditioning constants {#conditioning-constants}

The same module exposes the proven error-amplification constants that rank *how well* a given
line set conditions each estimate — these feed both the reliability flag and the selection levers
(§3):

| Quantity | Formula | Lean theorem | Reading |
|----------|---------|--------------|---------|
| Temperature conditioning | $2/\lvert E_i - E_j\rvert$ | `twoLineBeta_stable_sharp` | wider upper-level energy separation ⇒ better-conditioned two-line $T$ |
| Composition $\ell^1$ error bound | $\sum_s \lvert \tilde C_s - C_s\rvert \le 2\,\mathrm{card}\,\delta/\hat S$ | `composition_dist_vector_le` | grows linearly in species count and per-species density error; shrinks with recovered total density |

`temperature_conditioning` is the **sharp** Lipschitz constant of the two-line slope under the
worst-case opposite-sign ordinate perturbation — not a heuristic but the exact per-unit-$\varepsilon$
amplification. It returns $+\infty$ when the two energies coincide (the estimate is undefined).

**What correct reliability code MUST do**

- [ ] Gate on a **measured** n_e, never an imputed one (§1.1 DO-NOT).
- [ ] Treat Stark↔Saha agreement as evidence from two *different* observations, never collapse it to an identity.
- [ ] Enforce **both** gate hypotheses (agreement AND McWhirter), and surface the violated one by name.
- [ ] Keep every formula a verbatim mirror of its Lean theorem, conformance-pinned in `tests/oracle/`.

---

## 2. Uncertainty quantification — the five surfaces {#uncertainty-quantification}

CF-LIBS carries **five** distinct UQ surfaces, each answering a different question. They are
complementary, not redundant; picking the wrong one gives a defensible-looking but meaningless
error bar.

| # | Surface | Question answered | Module |
|---|---------|-------------------|--------|
| 1 | Analytical (Boltzmann covariance) | fast, correlation-aware point error bars | `uncertainty.py` (`uncertainties` pkg) |
| 2 | Monte-Carlo | full-pipeline error incl. non-linear / non-Gaussian effects | `uncertainty.py::MonteCarloUQ` |
| 3 | Conformal | distribution-free, finite-sample prediction intervals | `conformal.py` |
| 4 | TARP coverage | are the Bayesian posteriors calibrated? | `coverage.py::tarp_coverage` |
| 5 | SBC / LOO posterior metrics | MCMC convergence + model-comparison gate | `posterior_metrics.py` |

### 2.1 Analytical propagation {#uq-analytical}

The fast path propagates per-line intensity uncertainty through the Boltzmann regression using the
`uncertainties` package, which tracks correlations via its computational graph. Correlation
tracking is essential here: the Boltzmann slope and intercept come from the *same* regression and
are correlated, so a naive independent-error propagation understates the composition error. The
module explicitly rejects alternatives (`scipp`, `AutoUncertainties`) precisely because they do
not track correlations (`cflibs/inversion/physics/uncertainty.py`). The intrinsic error floor is
dominated by temperature precision — the exponential $\exp(-E_k/k_B T)$ makes $T$ the leading
error driver, so a wide $E_k$ lever-arm and a multi-element common slope are the primary precision
levers [@tognoni2007].

### 2.2 Monte-Carlo propagation {#uq-monte-carlo}

`MonteCarloUQ` re-runs the whole inversion on perturbed inputs (spectral noise, atomic data),
capturing non-linear and non-Gaussian effects the analytical path linearises away. It is the
publication-quality path; `joblib` parallelises it but is an *undocumented soft opt-in* — it is
not pulled in by any `pyproject` extra (not even `cflibs[uncertainty]`) and must be installed
explicitly, else it falls back to serial.

### 2.3 Conformal prediction — distribution-free intervals {#uq-conformal}

Conformal prediction wraps any point estimate with a finite-sample, distribution-free
marginal-coverage guarantee under exchangeability. It is an **additive diagnostic**: it never
changes the inversion point estimate (`cflibs/inversion/physics/conformal.py`). Two estimators
ship:

- **Split conformal** [@lei2018] — the level-$\alpha$ threshold is the
  $\lceil(1-\alpha)(n+1)\rceil/n$ empirical quantile of the calibration nonconformity scores; the
  interval is a constant-width band `point ± q_hat`.
- **Conformalized quantile regression (CQR)** — Romano, Patterson & Candès (2019, arXiv:1905.03222) —
  adapts to heteroscedastic noise and is typically no wider than the split-CP band while retaining
  the same guarantee.

Both give $\Pr\{Y_{n+1}\in\hat C(X_{n+1})\}\ge 1-\alpha$ for an exchangeable draw of size $n$, with
upper bound $1-\alpha+1/(n+1)$ when scores are a.s. distinct.

### 2.4 TARP posterior coverage {#uq-tarp}

TARP (Tests of Accuracy with Random Points) — Lemos, Coogan, Hezaveh & Perreault-Levasseur (2023,
arXiv:2302.03026, ICML) — is an **evaluation-free** coverage test: it operates on posterior
**samples only**, never the density, which fits CF-LIBS' NUTS/dynesty output exactly
(`cflibs/inversion/physics/coverage.py::tarp_coverage`). Its Expected Coverage Probability curve
reads directly:

- ECP **on the diagonal** ⇒ calibrated.
- ECP **below the diagonal** ⇒ credible regions too small ⇒ **over-confident / under-coverage** —
  the most dangerous failure for reported composition error bars.
- ECP **above the diagonal** ⇒ under-confident / conservative.

### 2.5 SBC / LOO posterior metrics — the Bayesian gate {#uq-posterior-metrics}

`posterior_metrics.py::PosteriorDiagnostics` implements the Tier-1 posterior calibration gate
(§5). The hard-gate thresholds:

| Metric | Threshold | Failure mode |
|--------|-----------|--------------|
| $\hat R$ (all params) | < 1.01 | hard fail [@gelman1992; @vehtari2021] |
| ESS bulk (all params) | ≥ 400 | run **INVALID** (re-run, not fail) |
| Divergent transitions (NUTS) | 0 | hard fail |
| PSIS $\hat k$ (all folds) | < 0.7 | hard fail — LOO unreliable [@vehtari2017] |
| 95 % CI coverage | $\in[0.93,0.97]$ | **bidirectional** — over-coverage is also a fail |
| $\Delta$ELPD vs baseline | $\Delta\mathrm{ELPD}-2\,\mathrm{SE}_{\mathrm{diff}}>0$ | required for likelihood-changing PRs |

The bidirectional coverage band is the subtle one: an over-covering posterior is *also* a failure
because an over-confident-that-it's-humble model is mis-calibrated. ArviZ is preferred when
present (it ships in the `bayesian` extra); the module falls back to small numpy implementations
of split-$\hat R$ and batch-means ESS, but PSIS-LOO requires ArviZ (fields are `None` otherwise).

**What correct UQ code MUST do**

- [ ] Track Boltzmann slope/intercept correlation in the analytical path.
- [ ] Keep conformal and TARP strictly additive — never let a diagnostic move the point estimate.
- [ ] Enforce the coverage band **bidirectionally**.
- [ ] Treat low ESS as *invalid* (re-run), distinct from a hard fail.

See [frontier-methods.md](./frontier-methods.md) for the Bayesian inversion the posterior metrics
gate, and [error-budget-and-falsification.md](./error-budget-and-falsification.md) for the analytic
sensitivity budget.

---

## 3. Selection-lever reality — what actually binds {#selection-lever-reality}

This section is the antidote to a recurring trap: assuming that line-selection and
refuse-to-report knobs improve accuracy. On healthy high-SNR lab data they **do not** — they are
measured no-ops. The evidence is a paired, same-element-set, conditioned-RMSE benchmark
(`supercam_labcal`, n = 20–30, seed 7, reference pipeline), with every lever wired gated and
default-off (`docs/M-spec-lever-comparison.md`).

### 3.1 The three levers and their measured regime {#lever-table}

| Lever | Theorem basis | Default regime | When it bites |
|-------|---------------|----------------|---------------|
| `target_sigma_t` (derived SNR/spread/lines gates) | `ErrorBudget` (`olsSlope_stable_l2`, …) | **no-op** (SNR ~10⁶ ⇒ `min_snr` inert) | low-SNR regimes where the gate binds |
| `reliability_ranked_selection` (max-energy-spread subset) | `twoLineBeta_stable_sharp` ($2/\lvert\Delta E\rvert$) | **no-op** at default cap (≤16 < 20 lines/el) | **when the cap binds — and then it WINS** |
| `refuse_to_report` (identifiability guards) | Identifiability + `selfAbsorption_breaks_identifiability` | **no-op** (0/30 flagged on healthy spectra) | degenerate / line-starved inputs |

### 3.2 `min_lines_per_element` is the #1 fragility {#min-lines-fragility}

Of the three tuned "magic numbers" (`min_snr=10`, `min_energy_spread=2`, `min_lines=3`), only
`min_lines_per_element` is load-bearing — and only *downstream* via element handling, not via the
selector gate itself. The wiring study is unambiguous
(`docs/M-spec-target-sigma-wiring-findings.md`):

- `min_snr` is the only **hard** gate the derivation drives, and it **never binds**: detected lines
  on `supercam_labcal` have SNR of **449,000–5,300,000**, five orders of magnitude above any
  threshold (legacy 10 or derived 27). Raising `min_snr` 10→27 rejects nothing.
- `min_energy_spread` and `min_lines` are **advisory-only** in the selector — they emit warnings,
  they do not change the selected set.

Result of wiring `target_sigma_t` ∈ {0.10, 0.05, 0.03}: **0 improved, 0 regressed, 3 flat**,
median $\Delta = +0.000$ wt%. A structurally inert lever, correctly derived from the verified error
budget but with nothing to bite on because the dataset noise model gives ~10⁶ SNR. The derived
thresholds (`derived_thresholds.py::min_lines_per_element_for`, mirroring
`lean:CflibsFormal/ErrorBudget.lean#olsSlope_stable_l2`) are the *principled* replacement for the
tuned constants and generalise to low-SNR regimes — the durable contribution is the conformance
guard, not an accuracy delta.

### 3.3 The one positive result — reliability ranking wins when selection is forced {#reliability-ranking-win}

Force a binding cap (`max_lines_per_element = 6`) to test the *selection criterion itself*:

| Comparison (conditioned RMSE, wt%) | med A | med B | $\Delta$(B−A) | B wins |
|---|---|---|---|---|
| score_cap6 (A) vs **reliability_cap6** (B) | 4.923 | **4.401** | **−0.522** | **17/20** |
| default20 (A) vs score_cap6 (B) | 3.762 | 4.649 | +0.887 | 0/20 |
| default20 (A) vs reliability_cap6 (B) | 3.762 | 4.401 | +0.639 | 5/20 |

When you *must* choose a subset, picking the widest upper-level energy spread (the proven
$2/\lvert\Delta E\rvert$ conditioning) beats picking the highest-SNR-score lines by 0.52 wt%
(17/20 spectra). The conditioning criterion is genuinely the better rule — validated, not assumed.
But note the second row: **capping at all costs accuracy** here (3.76 uncapped vs 4.40–4.65 at
cap 6) because more high-SNR lines give better statistics.

### 3.4 The honest policy {#lever-policy}

On healthy high-SNR multi-line lab data the right policy is: **don't cap, don't gate** — use every
line. The levers are **conditioning-aware tools for constrained regimes**, kept default-off:

- **portable / handheld** instruments (few resolvable lines ⇒ the cap or line count binds),
- **real-time / speed-limited** inference (a deliberate small `max_lines` cap for latency),
- **low-SNR spectra** (the `min_snr` gate binds; `target_sigma_t` adapts it principledly),
- **degenerate inputs** (single-line species, single thick line + unknown $\tau$ ⇒ refuse-to-report
  flags what no estimator can recover).

The accuracy bottleneck on healthy data is elsewhere — atomic-data quality, wavelength
calibration, the solver — consistent with every M5 finding. See
[atomic-data-and-datasets.md](./atomic-data-and-datasets.md).

---

## 4. The benchmark harness {#benchmark-harness}

One command measures the only things that matter: element-identification accuracy
(precision/recall/F1), composition accuracy (RMSE in element wt% vs certified truth), and runtime
per stage, across every registered truth-bearing dataset, running the **production** pipeline
exactly as `cflibs analyze` would (`cflibs/benchmark/scoreboard.py`).

```bash
JAX_PLATFORMS=cpu cflibs scoreboard --output-dir output/scoreboard
```

### 4.1 Tier policy — the gate cannot leak {#tier-policy}

| Tier | Behaviour |
|------|-----------|
| `optimization` | always run; iterate here |
| `holdout` | adoption-gate datasets (e.g. `bhvo2_chemcam`, `emslibs2019`) — EXCLUDED unless `--include-holdout` |
| `vault` | `gibbons2024` — never run by the harness at all |

Explicitly requesting an excluded dataset is a **hard error**, never a silent skip — casual boards
cannot leak the gate.

### 4.2 Candidate-set policy — measuring rejection {#candidate-policy}

For each spectrum the pipeline is given `candidates = truth.elements_present ∪ CONFOUNDER_ELEMENTS`
(Ag/Sn/W/Bi/Th — known false-positive confounders with in-band neutral resonance lines in the same
thermal $E_k$ band as real majors). Truth elements make **recall** measurable; the fixed confounder
set makes **false positives** measurable. This measures *rejection given a candidate superset*, not
open-world identification over the full periodic table.

### 4.3 Composition metrics — Aitchison / ILR {#composition-metrics}

Composition lives on the simplex, so Euclidean wt%-RMSE is the wrong geometry. The primary scalar
is the **Aitchison distance** on the closed composition (`composition_metrics.py::aitchison_distance`):

$$d_A(\hat C, C^*) = \sqrt{\sum_i \big(\mathrm{clr}(\hat C)_i - \mathrm{clr}(C^*)_i\big)^2},\qquad \mathrm{clr}(C)_i = \log\!\frac{C_i}{g(C)}$$

with $g$ the geometric mean. Thresholds: EXCELLENT $d_A<0.05$, GOOD $<0.10$, ACCEPTABLE $<0.20$,
FAIL $\ge0.20$. The ILR (isometric log-ratio) transform [@egozcue2003] gives an orthonormal
coordinate system for the same geometry. The scoring convention is grounded in compositional-data
theory — Aitchison's simplex algebra and subcompositional coherence.

> [!IMPORTANT] For the DED / tracking deliverable, prefer **log-ratios** $\ln(N_i/N_j)$ over
> closure wt%. Ratios cancel the shared closure denominator (matrix- and detected-set invariant,
> `lean:CflibsFormal/MatrixEffects.lean#recoveredComposition_ratio_matrix_invariant`) — the DED
> V/Ti mass-slosh that OPC was band-aiding is structurally sidestepped by ratio reporting. See
> `docs/research/physics-first-principles-audit.md` Issue 2 and §7 below.

Subcompositional ratio errors $\lvert\log(\hat r/r^*)\rvert$ for the Fe/Si, Mg/Si, Ca/Si, Al/Si
pairs are reported as the invariants CF-LIBS physics actually preserves under partial detection.

### 4.4 Dataset adapters and the unified runner {#adapters}

The unified runner (`cflibs/benchmark/unified.py`) centralises dataset adapters (real,
assay-backed, blind, synthetic), leakage-safe grouped splits, workflow registries for
identification and composition, nested evaluation, and statistical tests. Results serialise to a
single `results.parquet` (schema v1, one row per id×composition×split×spectrum) replacing the
legacy JSON triplet — see `docs/results-parquet-schema.md`. The scoring rule itself
(`cflibs/benchmark/scoring.py`) is the *one* don't-care-aware confusion rule shared by every
scoring path, so callers cannot drift on the semantics — in particular the **don't-care band**:
real-but-sub-detection-floor traces that are neither rewarded (TP) nor penalised (FP), never an FN.

---

## 5. Benchmark-gate discipline {#benchmark-gate-discipline}

> [!WARNING] BENCHMARK-GATED — the repo has regressed **three times** on ungated scoring changes.
> Any change to a scoring, identifier, or selection path REQUIRES a flag-off / flag-on scoreboard
> non-regression run before merge. Cite the gate and the measured delta. Do not merge on
> "looks better."

### 5.1 Why the gate is mandatory {#why-gate}

Two concrete regressions anchor the rule:

- A **paper-faithful ALIAS** change (audit Family 5 / PR #229) *regressed* F1 by −0.041 despite
  being more faithful to the source paper.
- Making Comb paper-faithful *lifted* F1 (0.033 → 0.529, recall 0.018 → 0.995) — the opposite
  direction — showing the effect of an identifier change is not predictable from first principles.

The lesson: **benchmark-gate every identifier-scoring change before merge**. A change that is
"more correct" on paper can move the metric either way; only the scoreboard settles it.

### 5.2 The tiered gate framework {#tiered-gate}

The validation framework (`docs/VALIDATION_METRICS.md`) is Goodhart-resistant by construction:

- **Single primary endpoint** — one scalar ($d_A$) gates merge; everything else is diagnostic.
  Avoids the "13 PRs each find a metric they improved" failure.
- **Tier 1** (binding gate), **Tier 2** (alarms — justify, don't block), **Tier 3** (forensic, log
  only).
- **Two baselines** — `last-green-on-main` (rolling A/B, regression detection) and a **frozen-v0**
  reproduction (long-horizon drift), the v0 reference being a Tognoni-2010 CF-LIBS state-of-art
  reproduction [@tognoni2010].
- **Multiplicity control** — Holm-Bonferroni for Tier-1 hard gates (q = 0.05), Benjamini-Hochberg
  FDR at q = 0.10 for Tier-2 alarms [@benjamini1995].
- **Physical-constraint gates** (LTE, McWhirter, charge balance, mass conservation) are
  first-principles and cannot be tuned by an agent without breaking physics.

### 5.3 The overconfidence guardrail {#overconfidence-guardrail}

A Tier-2 alarm borrowed from stellar spectroscopy: the **internal/external precision ratio**. If
internal repeatability $\ll$ external accuracy the model is over-confident and fitting noise
[@holtzman2015]. This is the same failure TARP catches for posteriors (§2.4) — an
over-confident error bar is a defect, not a feature.

### 5.4 Statistical decision protocol {#decision-protocol}

Pre-register `validation/protocol.yaml` on `main`; per-PR test is Wilcoxon paired (per-dataset
$d_A$) vs both baselines; effect-size requirement is ≥5 % relative **and** ≥0.005 absolute
improvement with a 95 % bootstrap CI excluding zero; ≥5 seeds; the **agent cannot pick seeds**, the
protocol does.

---

## 6. Anti-overfit protocol — campaign 1 {#anti-overfit-protocol}

Campaign 1 is the optimization program's cheapest campaign: a seeded Optuna-TPE search over 46
pipeline + detection knobs, fitness = the goal-metric scoreboard on the **optimization split**,
with holdouts and the vault never entering the loop (`scripts/campaign1/README.md`). Nothing under
`cflibs/` imports optuna — it is optimization-layer tooling only; the winning config ships as plain
preset numbers after the adoption gate.

### 6.1 The three tiers — optimization / holdout / vault {#split-tiers}

| Tier | Datasets | Role |
|------|----------|------|
| optimization | (loop-visible) | TPE fitness is computed here |
| holdout | BHVO-2, EMSLIBS-2019, the 40 % target holdouts | adoption gate only (quota: 1/week) |
| vault | `gibbons2024` | never evaluated by this tooling at all |

`splits.py` builds splits **by target identity, never by spectrum**, and structurally refuses a
holdout leak (`HoldoutViolation`). The frozen manifest is committed at
`docs/benchmarks/manifests/campaign1-splits-v1.json` (seed 20260610).

### 6.2 FROZEN_MANIFEST — reproducibility by construction {#frozen-manifest}

Every study writes a `frozen_manifest.json` embedding the exact inputs so a result can never be
silently re-based:

- the split id lists,
- the **atomic-DB sha256**,
- the **seed**,
- the **git SHA**,
- the knob-space definition and (for v2) the fitness-grading constants.

Split manifests carry spectrum ids, not per-file sha256 (several datasets aggregate many spectra
per file), so the atomic-DB sha256 + splits-manifest sha256 + git SHA pin the inputs instead. A
worker started with a contradicting `--fitness-version` is **refused** — a journal must never mix
fitness maths.

### 6.3 The adoption gates G1–G4 {#adoption-gates}

Holdout re-evaluation (top-K + baseline) applies the adoption gate — a candidate ADOPTS only if
**all four** pass:

| Gate | Requirement |
|------|-------------|
| **G1** | optimization pooled $\Delta$F1 ≥ +0.02, paired-bootstrap 95 % CI excludes 0 |
| **G2** | holdout pooled $\Delta$F1 ≥ +0.02, CI excludes 0, **no per-dataset regression** beyond bootstrap noise (BHVO-2 n=4 gated point-wise) |
| **G3** | **zero** new false positives and **zero** new failures on real holdout datasets |
| **G4** | holdout runtime ≤ 1.5× baseline |

Every holdout query is ledger-logged; a second query within 7 days requires a human `--force`. The
verdict recommendation among ADOPT candidates is the best worst-dataset score.

### 6.4 Fitness lessons — graded penalties beat flat deaths {#fitness-versions}

Run 1 used flat death penalties (`−1e9` for any excess FP/failure). Outcome: **79/80 trials at
−1e9**, zero ranking signal for TPE. Fitness-v2 replaced these with **graded** penalties
(`LAMBDA_FP = 0.05` per excess FP, `LAMBDA_FAIL = 0.02` per excess failure) and a
`CATASTROPHIC_FITNESS = −1e3` floor — sized so one excess FP costs more than any plausible
single-step score gain (observed weighted_score range ~0.30–0.56). The hard no-regression
constraint stays at **adoption** (the G-gates) so the search keeps its gradient. A trial with zero
excess counts scores identically under v1 and v2.

### 6.5 The autonomous-search gate (Arbor) {#arbor-gate}

The Arbor integration (`docs/arbor-integration.md`) runs an autonomous, gated, anti-overfit search
in git-worktree isolation. Its keystone is a **correctness gate** ahead of the accuracy score: each
candidate first runs the physics-only import blocklist (`ruff --select TID251 cflibs/`) plus the
`cflibs-formal` oracle conformance; any failure ⇒ `score = -1e9`, `valid = false` ⇒ Arbor never
merges it. This auto-catches C-σ-class silent errors (a candidate that scores well but is
physically wrong). Splits use the project tiers: `--split dev` = optimization, `--split test` =
holdout (the merge gate). The honest bound: Arbor makes the *search* structured and safe; it does
not change the finding that the deeper bottleneck is atomic-data quality, which code-optimization
cannot reach.

See [error-budget-and-falsification.md](./error-budget-and-falsification.md) for the falsification
ledger these gates feed.

---

## 7. The DED precision benchmark {#ded-precision-benchmark}

The DED (Directed Energy Deposition) benchmark is a synthetic, constrained-absolute self-consistency
test on a **known, closed** element set (no oxygen, no oxides), matching the real mission: real-time
composition-drift tracking of a known alloy where **precision and ratios matter far more than
absolute wt%** (`tests/benchmarks/ded_precision/`).

### 7.1 The alloys and the drift model {#ded-alloys}

| Alloy | Constrained set (wt%) |
|-------|-----------------------|
| Ti-6Al-4V (primary) | Ti 90.0, Al 6.0, V 4.0 |
| Inconel 625 | Ni 64.5, Cr 22.5, Mo 9.5, Nb 3.5 |
| 316L | Fe 68.0, Cr 17.0, Ni 12.0, Mo 3.0 |

Drift is modelled by varying one element while the others keep their mutual ratios (one species
evaporates/enriches, the rest re-scale to sum 100) — `make_series` in `alloy_definitions.py`.

### 7.2 The validated clean floor {#ded-clean-floor}

On a **noise-free** spectrum with true n_e injected, the constrained iterative solver recovers the
absolute composition to ~1–2 wt% RMSEP with a small, **stable, calibratable** bias:

| Alloy | Best elements (RMSEP wt%) | Weak element |
|-------|---------------------------|--------------|
| Ti-6Al-4V {Ti,Al,V} | Al 0.82, V 0.87, Ti 1.43 | — (all < 1.5) |
| Inconel 625 {Ni,Cr,Mo,Nb} | Nb 0.08, Mo 0.64 | Cr −4.5, Ni +5.1 |
| 316L {Fe,Cr,Ni,Mo} | Ni 0.12, Mo 0.60 | Cr −2.6, Fe +3.2 |

This is the trustworthy deliverable. The clean-floor bias is stable across the drift scan, so it
**cancels in drift tracking** (measured − nominal) — which is why the DED goal tolerates a bias it
would have to calibrate away for an absolute-accuracy goal.

### 7.3 The readout-noise-on-weak-lines bias mechanism {#readout-noise-mechanism}

Under a (conservative, **guessed**) DED noise model, single-shot recovery collapses (Ti bias
~−30 wt%) and shot-averaging only partially recovers (Ti plateaus ~83 wt%, not the 88 clean floor).
The ablation diagnosis: the bias is driven almost entirely by **readout noise on weak lines**.
Windowed integration with a positivity clip *rectifies* readout noise and inflates weak
minor-element lines (V, Al), pulling the major element (Ti) down via closure. The weakest high-$E_k$
lines sit near the noise floor and the log-Boltzmann step needs positive intensities — a selection
bias that survives averaging.

### 7.4 SNR-gating is COUNTERPRODUCTIVE for minor-element drift {#ded-snr-gating}

The intuitive fix — drop low-SNR lines — is exactly wrong for the DED mission:

| `min_snr` | Ti | Al | V | note |
|-----------|----|----|---|------|
| 0 (off) | 70.7 | 15.6 | 13.6 | noisy baseline |
| 5 | 86.6 | 13.4 | **0.0** | all V lines gated ⇒ **V lost** |
| 10 | 0 | 0 | 0 | all lines gated ⇒ solve fails |

SNR-gating removes exactly the weak minor-element lines (V, low-concentration Al) you are trying to
track. For minor-element drift the answer to noise is **shot averaging** (lower the noise floor so
weak lines stay measurable), **not** dropping low-SNR lines. `min_snr` is kept (default 0 = OFF) for
major-element-only scenarios but must **not** be used when a tracked element is a weak minor
constituent. This is the DED-specific corollary of §3.

### 7.5 The Cr under-estimate — a method limit, not a data bug {#ded-cr}

Cr is systematically under-estimated (~−3 to −5 wt%) across Inconel + 316L. Lines, resonance
handling, stage-III, and partitions were all ruled out (Cr I/II/III have real `direct_sum_fit`
partitions). The residual is the iterative Boltzmann+Saha **method's inherent approximation error**
for high-ionization transition metals. The joint (full-spectrum) solver *fixes* the Cr
under-estimate (24.3 vs iterative 18.3, truth 22.5) — confirming a method, not a data, cause — but
worsens Ni/Mo/Nb, so its overall Inconel RMSE is higher (~4.8 vs ~3.2 wt%). Neither solver is
uniformly best; they distribute error differently.

> [!CAUTION] DO-NOT: do not carry forward "joint beats iterative" as a headline. The joint solver
> wins on Cr but loses overall on Inconel, and adopting it needs an adoption gate (posterior
> uncertainty as the trust signal), not a blanket swap. See
> [error-budget-and-falsification.md](./error-budget-and-falsification.md).

---

## 8. NIST-parity validation — the GoldenSpectrum round-trip {#nist-parity-validation}

Round-trip validation is the pipeline's synthetic ground-truth check: generate a spectrum from
known plasma parameters, add realistic noise, invert, and verify recovery within tolerance
(`cflibs/validation/round_trip.py::GoldenSpectrum`). The reference physics is Tognoni-2010 CF-LIBS
state of the art [@tognoni2010] and the Ciucci-1999 procedure [@ciucci1999].

| Step | What it checks |
|------|----------------|
| forward from known $(T, n_e, C)$ | the forward model is self-consistent |
| add Poisson + Gaussian noise | realistic measurement conditions |
| invert | the solver recovers the injected truth |
| assert within tolerance | no round-trip drift; `n_e > 0`, physical $T$ |

A `GoldenSpectrum` carries the true temperature, electron density, concentrations (sum to 1.0), and
the line observations — so a solver regression is caught against an exact known answer, independent
of any real-data atomic-data floor. NIST-parity scripts (`scripts/validate_nist_parity.py`,
`scripts/run_nist_validation.py`) extend this to cross-check the DB's line data against NIST/ASD.
The multi-element Saha-Boltzmann plot [@aguilera2007] is the temperature core the round-trip
exercises. See [atomic-data-and-datasets.md](./atomic-data-and-datasets.md) for the DB provenance
and [classical-quantification.md](./classical-quantification.md) for the inversion it validates.

---

## 9. CLI and quickstart {#cli-and-quickstart}

The CLI (`cflibs/cli/main.py`) exposes the whole pipeline. Quality-gate and setup commands live in
§10; this section is the run surface.

### 9.1 The subcommands {#cli-subcommands}

```bash
cflibs generate-db                                   # build the atomic SQLite DB (hours)
cflibs forward examples/config_example.yaml --output spectrum.csv
cflibs invert spectrum.csv --elements Fe Cu --config examples/inversion_config_example.yaml
cflibs analyze spectrum.csv --elements Fe,Cu --output result.json
cflibs bayesian spectrum.csv --elements Fe,Cu --output posterior.json
cflibs batch ./spectra --elements Fe,Cu --output-dir output/batch_results
cflibs generate-manifold examples/manifold_config_example.yaml --progress
```

`analyze`, `invert` and `batch` share one pipeline and one set of accuracy-critical knobs, bundled
into presets (`--preset` / `analysis.preset`):

| Preset | `saha_boltzmann_graph` | `closure_mode` | Use for |
|--------|------------------------|----------------|---------|
| `geological` (**default**) | true | `oxide` | rocks, soils, minerals |
| `metallic` | true | `standard` | alloys, metals (oxide stoichiometry would be wrong physics) |
| `raw` | false | `standard` | legacy-default comparison runs |

The default `geological` bundle is the validated-best configuration — on the real ChemCam BHVO-2
basalt standard the legacy defaults scored RMSE 10.29 wt% (mechanism-only figure) while the
geological bundle scored 4.03 wt%. Explicit flags override the preset; every run logs the resolved
preset and all knobs at INFO.

### 9.2 The trust report {#cli-trust-report}

Every output path (`analyze` table, `invert` stdout, batch rows, JSON `trust` block) reports:
`converged`, the Boltzmann-plane $R^2$, the number of elements fit, the **n_e provenance**, the
degeneracy gates (`boltzmann_degenerate`, `closure_degenerate`), and any requested elements dropped
before the fit with the dropping stage (`detection`, `selection`, or `solve`).

### 9.3 The honest n_e fallback path {#ne-fallback}

The n_e provenance line is load-bearing for trust. The primary path uses a Stark-width diagnostic;
when no `stark_b`-flagged line is available it falls back to a **1-atm pressure-balance** estimate —
and the fallback prints a visible `WARNING` because n_e was **ASSUMED, not measured**. This visible
warning is the honest surface of the imputed-n_e defect flagged in §1.1: a downstream consumer must
treat a pressure-balance n_e as unconstrained and must not let `overall_reliable` pass on it.

### 9.4 Spectral-response correction {#cli-response-curve}

CF-LIBS compares line intensities across wide wavelength spans, so the wavelength-dependent
detection efficiency $E(\lambda)$ enters the Boltzmann intercepts ($\Delta q = \ln E$) and biases
both $T$ and concentrations — correcting for it is an experimental prerequisite [@tognoni2010].
Use `--response-curve lamp_response.csv` for in-house instruments; do **not** use it on
ChemCam/SuperCam CCS spectra (already vendor-radiometrically corrected — re-applying double-corrects).
The default is identity (no correction), bit-identical to prior releases.

---

## 10. Contributing {#contributing}

### 10.1 The physics-only constraint {#physics-only}

The shipped algorithm must not import `sklearn`, `torch`, `tensorflow`, `keras`, `flax`, `equinox`,
`transformers`, `jax.nn`, or `jax.experimental.stax`. ML is allowed **only** in `cflibs/evolution/`.
Enforcement is two-layer: (1) the Ruff **TID251** static rule bans these APIs; (2) the AST
blocklist scanner in `cflibs/evolution/evaluator.py` rejects any evolved candidate that violates the
ban (fitness = −inf) before physics evaluation. This is why the reliability, error-budget, and
conformal modules depend on `math`/`numpy` alone.

### 10.2 The quality-gate sequence {#quality-gates}

Run gates in this order (the swarm-profile order):

```bash
ruff check cflibs/ tests/
black --check cflibs/
mypy cflibs/                                  # advisory/non-blocking in the swarm profile
pytest tests/ -x -q -m "not slow and not requires_db"
```

Auto-fix: `black cflibs/` then `ruff check --fix cflibs/`. **JAX x64 is mandatory** — `conftest.py`
forces CPU with `jax_enable_x64=True`; do not pass `JAX_ENABLE_X64=1` as an env var to the full
suite (conftest enables it in-process and the no-hidden-x64 subprocess tests assert a fresh
interpreter). Test markers: `requires_db`, `requires_jax`, `requires_bayesian`,
`requires_uncertainty`, `requires_rust`, `slow`, `unit`, `integration`, `physics`, `nist_parity`.

### 10.3 The worktree PYTHONPATH trap {#worktree-trap}

Running `python scripts/<x>.py` by path puts `scripts/` (not the repo root) at `sys.path[0]`, so a
worktree silently imports `cflibs` from whichever checkout is installed in the venv. **Always** run
scripts as `PYTHONPATH=$PWD python scripts/<x>.py` from the worktree root and check the printed
`cflibs=` provenance line.

### 10.4 The sub-agent pytest watchdog rule {#pytest-watchdog}

The full suite (`pytest tests/`) takes **~57 min** with the `requires_db` suites. The Claude Agent
stream-idle watchdog kills sub-agents after ~600 s (10 min) of no observable stream output, and a
quiet pytest run is invisible to it. **Never** instruct a sub-agent to run the whole suite. Instead:
(1) a narrow subset that finishes well under 60 s, or (2) have the sub-agent commit+push and let CI
run the full suite, or (3) run the full suite in the **parent** via `Bash(run_in_background=True)`.
Commit after each logical step *before* the test step so a watchdog kill does not lose work. Never
call `jax.clear_caches()` in a test — it is a process-global wipe.

### 10.5 references.bib maintenance {#references-bib}

`docs/wiki/references.bib` is the single source of truth for citations. **Integrity is
non-negotiable**: every inline `[@key]` must resolve to exactly one entry, and **an entry without a
verified `doi` may not be cited**. Verify a DOI before adding it (asta CLI, gpd-arxiv, or
WebSearch) and record `note = {verified: ... <date>}`. Cite the **literature** for physics claims,
the **code path** (`code_refs` / inline `path::Symbol`) for what the code does, and the **Lean
theorem** (`lean_refs`) for what is proven. Proceedings papers without a journal DOI (e.g. CQR,
TARP) are cited in prose by author-year + arXiv id, never as a `[@key]` with a fabricated DOI.

### 10.6 Doc-authoring conventions for this wiki {#doc-conventions}

- Paste the frontmatter template verbatim; delete no required key. `slug` is the stable-forever
  filename stem — never renumber in place (a rename leaves a redirect stub).
- Any page citing a composition/F1/RMSE number sets `benchmarks_pre_reset` and, if pre-reset, opens
  with the PRE-RESET banner.
- doc→doc links are relative Markdown **with** the `.md` extension and a stable anchor; doc→code is
  `code_refs` + monospace `path::Symbol` (never a Markdown link across the tree boundary);
  code→doc is meant to be generated by a planned `scripts/build_wiki_codemap.py` (not built yet —
  codemap generation is out of scope for this rebuild, see MIGRATION.md) — never hand-written.
- Use the falsification-record block verbatim for every negative result; never carry a contradicted
  claim forward (state the corrected position; the archived original holds the tombstone).
- Once that generator exists, regenerate `docs/wiki/code-map.md` whenever frontmatter `code_refs`/`lean_refs` change (neither the generator nor `code-map.md` is present today).

### 10.7 Landing the plane {#landing}

Work is not complete until `git push` succeeds. File issues for remaining work, run quality gates
if code changed, then `git pull --rebase && git push && git status` (must show up-to-date). Use
native `bd` for issue tracking (not `bdh` — BeadHub was removed). Beads federation syncs via
`bd dolt push` after state changes (`docs/federation.md`).

---

## 11. Cluster deployment {#deployment-cluster}

> [!IMPORTANT] Posture: **accuracy before latency.** Sub-ms latency is deferred until the pipeline
> is robust. Optimize the accurate legacy solver and input-quality levers first; latency last. GPU
> matters for manifold generation, not for the reference inversion (measured GPU jit is ~8× slower
> on the per-spectrum inversion path).

### 11.1 Environment matrix {#env-matrix}

| Extra | Use case | Includes |
|-------|----------|----------|
| `jax-cpu` | testing, CI | JAX CPU only |
| `jax-metal` | Apple Silicon dev | JAX + Metal (no float64/complex) |
| `jax-cuda` | NVIDIA GPUs | JAX + CUDA 12 |
| `cluster` | production cluster | JAX CUDA + h5py + mpi4py |

Setup: `uv venv --python 3.12` then `uv pip install -e ".[local]"` (Mac) or `".[cluster]"` (NVIDIA).
JAX Metal is experimental; tests run CPU (`JAX_PLATFORMS=cpu pytest tests/`). The target cluster is
3 nodes, dual 20-core Xeon each (120 cores), ~1.1 TB RAM total, one V100S (32 GB HBM2) per node,
InfiniBand.

### 11.2 SLURM / NFS staging gotchas {#slurm-nfs}

- **NFS-shared data** (`docs/nfs-shared-data.md`): a single export replaces per-node data copies
  (3×11 GB ⇒ 1×11 GB). SQLite-over-NFS locks, so benchmark workers use **per-worker DB copies**.
- **JAX compile cache** (`docs/jax-compile-cache.md`): the Linux default is the **per-user**
  `~/.cache/cflibs/jax`, **not** the NFS `/cluster/shared/jax-cache`. The shared path developed uid
  skew across compute nodes and hung jobs; cross-host sharing is opt-in only. A cold XLA compile is
  15–45 s per kernel, so the persistent cache dominates short-iteration sweep warm-up.
- **Dataset sharding** (`docs/dataset-sharding.md`): `--dataset-shard N/K` splits the Vrabel-2020
  50k-spectrum corpus across nodes for population-level precision (bootstrap SE ∝ $1/\sqrt{n}$;
  1k→50k spectra ⇒ ~7× CI tightening). Do **not** shard tiny datasets (BHVO-2 n=12, NIST SRM 612
  <100) — the loader explicitly skips them.
- **Campaign workers** (§6): `--array=0-15%16 --cpus-per-task=8 --mem=16G`, CPU-only, **no
  `--nodelist`** (let SLURM schedule), private JAX cache; NFS-safe journal storage; killing a worker
  loses at most the in-flight trial.

### 11.3 Manifold generation {#manifold-gen}

Manifold builds are the primary JAX consumer — `jit`/`vmap` batch-generate spectra over a
$(T, n_e, C)$ grid to HDF5/Zarr for fast nearest-neighbour inference. Multi-node via MPI
(`mpirun -np 3 ... python generate_manifold.py` or `srun -N 3 --gpus-per-node=1`). Estimated rates:
~50k–100k spectra/s per V100S, ~150k–300k across the 3-GPU cluster. Batch ~10k spectra/GPU on a
V100S (32 GB).

See [architecture.md](architecture/index.md) for the manifold data flow and
[atomic-data-and-datasets.md](./atomic-data-and-datasets.md) for the DB the forward model reads.

---

## See also

- [error-budget-and-falsification.md](./error-budget-and-falsification.md) — the falsification
  ledger and the analytic error budget these gates feed.
- [formal-spec.md](./formal-spec.md) — the notation authority and the Lean theorems the reliability
  gate mirrors.
- [classical-quantification.md](./classical-quantification.md) — the iterative CF-LIBS solver these
  benchmarks measure.
- [atomic-data-and-datasets.md](./atomic-data-and-datasets.md) — the atomic-data floor that is the
  real accuracy bottleneck, and the certified datasets.
- [architecture.md](architecture/index.md) — the pipeline and manifold data flow.
- [frontier-methods.md](./frontier-methods.md) — the Bayesian inversion the posterior metrics gate.
