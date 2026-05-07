# CF-LIBS Validation Metrics — Pre-PR Gate Framework

A scientifically grounded, multi-tier validation framework for evaluating
whether a swarm-generated PR *actually improved* the CF-LIBS pipeline.
Authoritative reference for `validation/protocol.yaml` and for the
benchmark gate run by `python/benchmark_gate.py` in beefcake-swarm.

> **Status:** drafted 2026-05-06 from a parallel literature + statistics
> review (NotebookLM CF-LIBS corpus + Asta + community-practice survey).
> Treat the thresholds as v0 — they will be tightened as data accumulates.

---

## 1. Design principles

1. **Single primary endpoint.** One scalar gates merge; everything else is
   diagnostic. Avoids multiplicity and the "13 PRs each finds a metric they
   improved" failure mode.
2. **Tiered reporting.** Tier 1 (gate; binding pass/fail), Tier 2 (alarms;
   require justification but don't block), Tier 3 (forensic; logged for
   review).
3. **Two reference baselines.**
   - `last-green-on-main` (rolling A/B; detects regressions)
   - **Frozen v0 baseline** (re-run of a fixed published-baseline pipeline
     configuration; detects long-horizon drift across the whole codebase).
4. **Multiplicity control.** Holm-Bonferroni for Tier-1 hard gates within a
   single PR; Benjamini-Hochberg FDR @ q=0.10 for Tier-2 alarms across the
   PR family. Tier-3 metrics are exploratory and never gate.
5. **Goodhart-resistance by construction.** Physical-constraint gates
   (LTE, McWhirter, charge balance, mass conservation) are first-principles
   and cannot be tuned by the agent without breaking the physics.

---

## 2. Tier 1 — gate metrics (binding)

### 2.1 Composition accuracy

**Primary scalar: Aitchison distance $d_A$ on the closed composition.**

$$d_A(\hat C, C^*) = \sqrt{\sum_i \left(\mathrm{clr}(\hat C)_i - \mathrm{clr}(C^*)_i\right)^2}$$

where $\mathrm{clr}(C)_i = \log(C_i/g(C))$ and $g$ is the geometric mean.

| | Threshold |
|---|---|
| EXCELLENT | $d_A < 0.05$ |
| GOOD | $d_A < 0.10$ |
| ACCEPTABLE | $d_A < 0.20$ |
| FAIL | $d_A \geq 0.20$ |

- **Per-spectrum.** Median-aggregated per dataset.
- **Cross-dataset aggregation:** Friedman mean rank.
- **No-regression cap:** no single dataset's median $d_A$ degraded by
  > 10% relative vs `last-green-on-main`.

**Stratified concentration reporting** (mandatory):

| Stratum | RD threshold |
|---|---|
| Majors ($> 5$ wt%) | RD < 5% |
| Minors (0.1–5 wt%) | RD < 20% |
| Traces ($< 0.1$ wt%) | MDL-bounded; report as $\leq \text{LOQ}$ if below LOQ |

A PR that improves majors while wrecking traces does **not** pass.

**Subcompositional ratio errors** (also reported):
$|\log(\hat r/r^*)|$ for the Fe/Si, Mg/Si, Ca/Si, Al/Si pairs — the
invariants CF-LIBS physics actually preserves under partial detection.

**Companion metrics** (informational, not separately gated):
clr-RMSEP, NRMSEP, MAPE, $R^2$, Pearson r — preserved for cross-study
comparability with Takahashi 2017, Anderson 2017, Hao 2018 conventions.

### 2.2 Identification quality

| Threshold | Value |
|---|---|
| Macro-F1 | within 2 abs points of last-green |
| Recall (per element) | $\geq 0.6$ for {Si, Fe, Mg, Ca, Al, Ti, Mn, K, Na} on Aalto, USGS, NIST SRM |
| Wavelength tolerance | $\epsilon(\lambda) = \sqrt{\mathrm{FWHM}_{\mathrm{inst}}^2 + \omega_{\mathrm{Stark}}^2}$ |

The adaptive tolerance replaces fixed $\pm\lambda/(2 R_P)$ — Stark widths
in dense plasmas reach 0.2–0.5 nm vs ~0.1 nm instrumental and silently
mis-classify dense-plasma lines under a fixed tolerance.

### 2.3 Posterior calibration (when MCMC/NS active)

| Threshold | Value | Failure mode |
|---|---|---|
| $\hat R$ (all parameters) | $< 1.01$ | hard fail |
| ESS bulk (all parameters) | $\geq 400$ | run is **INVALID** (re-run, not fail) |
| Divergent transitions | $0$ | hard fail |
| PSIS $\hat k$ (all folds) | $< 0.7$ | hard fail (LOO unreliable) |
| 95% credible-interval coverage | $\in [0.93, 0.97]$ | bidirectional — over-coverage is also a fail |
| ELPD vs baseline | $\Delta\mathrm{ELPD} - 2 \cdot \mathrm{SE}_{\mathrm{diff}} > 0$ | required for likelihood-changing PRs |

Coverage outside the band fires a Tier-2 alarm if not in the hard-fail
range. PIT-histogram $\chi^2$ uniformity test is reported as Tier-3
diagnostic.

**Implementation:** `cflibs.benchmark.posterior_metrics.compute_posterior_diagnostics`
returns a `PosteriorDiagnostics` dataclass covering R-hat / ESS-bulk /
ESS-tail per parameter, divergent-transition count, PSIS-LOO ELPD with
per-fold $\hat k$, 95%-CI empirical coverage with a bidirectional gate
verdict, PIT $\chi^2$ p-value, and CLR-space sharpness. The unified
runner threads this through to `composition_records.json` whenever a
composition workflow emits a `posterior_samples` payload.

### 2.4 Cross-dataset generalization (anti-overfit)

| Requirement | Spec |
|---|---|
| Multi-matrix CRM panel | Steel (NIST SRM 1764 or JK), Basalt (BHVO-2), Mars analog (Norite or Shergottite), Plant/Soil (NIST 1573a) |
| Sequestered held-out fraction | 30% of CRM panel, never shown to the agent or in any prompt |
| Sequestered rotation | quarterly |
| Negative-transfer guard | If Aalto improves but ChemCam regresses, block regardless of mean |

### 2.5 Physical consistency

A PR fails if **any 2** of these trip; one alone fires a Tier-2 alarm
that requires explicit justification.

| Constraint | Spec | Reference |
|---|---|---|
| Multi-T LTE consistency | $|T_{\mathrm{neutral}} - T_{\mathrm{ion}}| / T_{\mathrm{avg}} < 0.15$ | Cristoforetti 2010 |
| McWhirter $n_e$ floor | $n_e \geq 1.6 \times 10^{12} T^{1/2} (\Delta E)^3$ | textbook LTE |
| Plasma-T physicality | $T_e \in [3000, 20000]$ K (catastrophic if any < 1000 K) | NB1 §25–26 |
| Closure residual (un-normalized) | $|\sum_s C_s - 1| < 0.10$ before forced closure | NB1 §24 |

The magnitude of forced normalization is itself a missing-element bias
signal — log it even when within bounds.

### 2.6 Statistical decision protocol

| | Spec |
|---|---|
| Pre-registration | `validation/protocol.yaml` frozen on `main`; changes require versioned bump |
| Per-PR test | Wilcoxon paired (per-dataset $d_A$) vs `last-green-on-main` AND vs frozen-v0 |
| Tier-1 multiplicity | Holm-Bonferroni across the family of Tier-1 gates (q = 0.05) |
| Tier-2 multiplicity | BH-FDR across the alarm family (q = 0.10) |
| Effect-size requirement | $\geq 5\%$ relative AND $\geq 0.005$ absolute improvement on aggregate $d_A$; 95% bootstrap CI excludes zero |
| Reproducibility | $\geq 5$ seeds; two independent runs agree within 1% on $d_A$ |
| Frozen environment | Docker image SHA pinned in protocol.yaml; deterministic CUDA flags; single-GPU per run |
| Seed control | Agent CANNOT pick seeds; the protocol does |

---

## 3. Tier 2 — alarms (BH-FDR q=0.10, don't gate)

| Metric | Threshold |
|---|---|
| Wall-clock per-spectrum | not regressed > 20% |
| Peak RSS | not regressed > 30% |
| MCMC tokens / leapfrog steps | not regressed > 50% |
| Boltzmann-plot linearity $R^2$ | per-element regression alarm |
| Stark-broadening multi-line residual | per-line-family alarm |
| Saha-Boltzmann self-consistency drift | $|T(BC_1) - T(BC_2)| / T_{\mathrm{avg}}$ |
| Per-element FP rate | track Mn, Na, K specifically (current 2.6/14.3/28.6% precision) |
| ALIAS Confidence Level CL components | $k_{\mathrm{det}} \cdot P_{\mathrm{maj}} \cdot P_{\mathrm{SNR}} \cdot P_{\mathrm{ab}}$ |
| Coverage drift outside [0.93, 0.97] | when not a hard fail |
| ELPD regression | $> 4 \cdot \mathrm{SE}$ |
| Internal/external precision ratio | $< 0.3$ → over-confident model fitting noise (Holtzman 2015 stellar guardrail) |

### Robustness perturbation tests (Tier-2)

| Test | Spec |
|---|---|
| Line-dropout robustness | Remove top-3 leverage lines; recompute $d_A$; require $\Delta d_A < 0.02$ |
| Outlier injection | Inject $N(0, 5\sigma)$ into 5% of channels; require $\Delta d_A < 0.05$ |
| Self-absorption stress | Verify $\sigma(T)/T < 0.01$ before applying SA correction (Völker & Gornushkin 2023) |

Student-t likelihoods (PR #58) should beat $L_2$ on the outlier-injection
test — failure to do so is itself a Tier-2 alarm against the PR.

---

## 4. Tier 3 — forensics (log only, never gate)

- Per-spectrum residual table for the top-10 worst spectra.
- PIT-histogram $\chi^2$ uniformity p-value.
- Bayes factor / log-evidence $\ln Z$ from nested sampling.
- Per-element MDL/LOQ alongside RD.
- Subcompositional Aitchison restricted to detected-only elements
  (separate from full-composition $d_A$).
- Resolving-power-stratified scoring (RP buckets 0–500, 500–1k, 1k–3k, 3k+).
- Plasma-parameter-vs-composition correlation (does composition error
  correlate with bad T or $n_e$ recovery?).

---

## 5. Reference baselines

| Baseline | Purpose | Cadence |
|---|---|---|
| `last-green-on-main` | Rolling A/B; regression detection | Per-PR (current) |
| Frozen-v0 (Tognoni 2010 reproduction) | Long-horizon drift; absolute-progress reference | Quarterly re-run from a pinned configuration |

Until the v0 reference pipeline is implemented, use the first commit on
`main` as the v0 baseline.

---

## 6. Goodhart hedges

The agent will optimize for whatever it sees in the prompt. The hedges
below assume the swarm WILL find loopholes given enough iterations.

1. **Sequestered held-out** — 30% of the CRM panel is never shown to the
   agent or in any prompt; the primary endpoint is computed only on the
   held-out split. Rotated quarterly.
2. **Physical-constraint gates** — Tier 1 §2.5 above. Cannot be tuned
   without breaking first-principles physics.
3. **Adversarial-twin run** — A parallel agent is told the metric *and*
   told to game it. If its gains match the candidate PR's, the candidate
   is presumed gamed and rejected.
   - **Edge case: Both fail.** If both candidate and adversarial fail to
     improve the metric, the PR is rejected (no progress).
   - **Edge case: Adversarial improves physics.** If an adversarial run
     accidentally discovers a real scientific improvement, the PR is still
     blocked if overlap > 50%. The improvement must be re-submitted via a
     clean scientific profile to ensure it wasn't a lucky overfit.
4. **Internal/external precision-ratio guardrail** — Tier-2 alarm if
   internal repeatability ≪ external accuracy (model is overconfident,
   fitting noise; stellar-survey practice from Holtzman 2015 / GALAH).
5. **Multi-matrix CRM mandate** — Single-matrix wins do not unlock merge.
6. **No metric leakage to workers** — The architect sees aggregated
   pass/fail; workers see compile/test results only. Per-dataset
   numerics are not threaded into the iteration loop.

---

## 7. Pre-PR vs post-PR gate scope

| Stage | Runs |
|---|---|
| Pre-merge gate (`scripts/post-merge-benchmark.sh nightly` triggered by `python/benchmark_gate.py`) | Tier 1 §2.1–§2.5 minus adversarial-twin |
| Post-merge nightly on `main` | Frozen-v0 reproduction; sequestered-fold rotation; full Tier 1+2+3 |
| Quarterly | Sequestered-fold reshuffle; v0-baseline re-run; Tognoni reference recompute |

---

## 8. Implementation gap (as of 2026-05-06)

The current benchmark already emits:

- Aitchison distance $d_A$ + RMSE composition + closure residual
- Per-element RMSEP, MAE, MAPE, Bias, $R^2$, Pearson r, LOD, LOQ
- Identification: precision, recall, F1, Jaccard, Hamming Loss
- Plasma physics: $T$ error, $n_e$ error
- Per-spectrum elapsed_seconds; total wall-clock
- Resolving-power estimate per spectrum (not yet stratified)

The current benchmark **does not** emit:

- Posterior calibration (R̂, ESS, coverage in [0.93, 0.97], PSIS-LOO)
- Stratified-by-magnitude composition reporting
- Subcompositional ratio errors (Fe/Si, Mg/Si, Ca/Si, Al/Si)
- Adaptive wavelength tolerance $\epsilon(\lambda)$
- Multi-T LTE consistency check (single-T currently — McWhirter floor,
  T physicality and closure residual are now Tier-1-gated via
  `cflibs/benchmark/physical_consistency.py`; the multi-T LTE check
  itself only fires when an inversion populates per-spectrum
  `t_neutral_k` and `t_ion_k`, which is still TODO)
- Robustness perturbation tests (line-dropout, outlier injection)
- Sequestered held-out separation
- Frozen-v0 reference baseline

These gaps are tracked as P1 issues in beads (CF-LIBS-improved-*).

---

## 9. Sources

- **Greenacre, *Statistical Science* 2022** — Aitchison reappraisal,
  subcompositional coherence
- **Egozcue et al., *TEST* 2019** — compositional sample-space structure
- **Tognoni et al., *Spectrochim. Acta Part B* 2010** — CF-LIBS state-of-art
  baseline (frozen-v0 reference)
- **Vehtari, Gelman, Gabry, *Stat Comput* 2017** — PSIS-LOO + WAIC
- **Cousins & Wasserman, PHYSTAT 2024** (arXiv:2402.16229) — nuisance-
  parameter handling
- **Bousquet et al., *Spectrochim. Acta Part B* 2023** — Boltzmann-plot
  trueness/precision/accuracy
- **Völker & Gornushkin, *JAAS* 2023** — multiplet Boltzmann + 1% T-RSD
  limit for SA correction
- **Matsumura et al., *ACS Earth Space Chem* 2024** — robust Student-t
  Boltzmann fitting under outliers
- **Cristoforetti et al., *Spectrochim. Acta Part B* 2010** — LTE-validity
  consistency criteria
- **Pedarnig et al., *JAAS* 2023** — benchmark protocol for CF-LIBS
- **Demšar, *JMLR* 2006** — Friedman + Nemenyi for cross-classifier ranking
- **Benjamini & Hochberg, *J R Stat Soc B* 1995** — FDR multiplicity control
- **Holtzman et al., *AJ* 2015** — APOGEE internal/external scatter
  framework (stellar precision/accuracy guardrail)
- **Aitchison, *J R Stat Soc B* 1982/1986** — subcompositional coherence
  and the simplex
