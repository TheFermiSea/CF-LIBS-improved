---
slug: classical-quantification
title: "Classical Quantitative LIBS: Calibration Curves to Chemometrics"
chapter: classical-quantification
order: 20
status: stable
register: review
summary: >
  The pre-calibration-free landscape: external calibration curves, internal
  standardization, the five-component matrix-effect taxonomy, and chemometrics
  (PLS/PCR/PCA). CF-LIBS exists to remove the standards these methods require;
  it cancels the multiplicative scale channel but NOT ablation non-stoichiometry
  or self-absorption. Chemometrics is described as cited concept only — the
  physics-only constraint bans ML estimators from shipped code.
tags: [classical-libs, calibration-curve, internal-standard, matrix-effects, chemometrics, lod, self-absorption]
updated: 2026-07-02
benchmarks_pre_reset: false
sources:
  - "@hahn2010"
  - "@hahn2012"
  - "@tognoni2010"
  - "@tognoni2007"
  - "@tognoni2016"
  - "@aragon2008"
  - "@ciucci1999"
  - "@aguilera2007"
  - "@long1983"
  - "@cremers2013"
  - "@cavalcanti2013"
  - "@feng2014"
  - "@anderson2012"
  - "@sun2009"
  - "@egozcue2003"
  - docs/REFERENCE_ANALYSIS_LIBSSA.md
  - docs/research/2026-06-19-cross-discipline-sota.md
  - cflibs/inversion/physics/matrix_effects.py
code_refs:
  - cflibs/inversion/physics/matrix_effects.py::MatrixEffectCorrector
  - cflibs/inversion/physics/matrix_effects.py::InternalStandardizer
  - cflibs/inversion/physics/matrix_effects.py::OnePointCalibrator
  - cflibs/inversion/physics/matrix_effects.py::AblationRegime
related: [cf-libs-family, error-budget-and-falsification, libs-physics, atomic-data-and-datasets]
supersedes:
  - docs/REFERENCE_ANALYSIS_LIBSSA.md
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Classical Quantitative LIBS: Calibration Curves to Chemometrics

This chapter is the *before* picture: how LIBS was made quantitative for two decades
before calibration-free methods, and which of those methods' failure modes calibration-free
LIBS actually fixes. It is deliberately a review-register chapter — equation-forward, cited,
and honest about the parts of the problem that no downstream algorithm (ours included) can remove.

The single load-bearing claim: **classical quantification buys accuracy by spending certified
reference standards**, and every classical method (external curves, internal ratios, chemometric
regression) is a different strategy for pricing or hiding the matrix effect. Calibration-free
LIBS ([cf-libs-family.md](cf-libs-family.md)) removes the *standard*, not the *matrix effect* —
a distinction the [matrix-effects](#matrix-effects) section makes precise.

> [!NOTE]
> **Wavelength convention.** Wherever a line wavelength $\lambda$ (nm) is compared to a NIST/ASD
> value in this chapter, it is an **air** wavelength (the DB stores air per NIST/ASD; the single
> `air_to_vacuum` / `vacuum_to_air` util lives in `cflibs/core/`). Line *selection* for a
> calibration curve is an air-wavelength operation; the forward model works in vacuum. This
> chapter never silently mixes the two.

---

## The pre-calibration-free landscape {#landscape}

LIBS emits a spectrum whose line intensity $I_{ki}$ depends on the analyte concentration $C_s$
*and* on plasma temperature $T$ (K), electron density $n_e$ (cm⁻³), ablated mass, optical depth
$\tau$, and instrument response — a chain summarized in [libs-physics.md](libs-physics.md). The
central practical problem, unchanged since the earliest analytical LIBS, is that a raw line
intensity is not proportional to concentration across samples of different composition. The
authoritative modern statements of this problem are the two-part review of [@hahn2010; @hahn2012]
and the LIBS handbook [@cremers2013]; the noise/precision companion is [@tognoni2016].

Three classical families answer it, in increasing sophistication:

| Family | What it needs | What it corrects | What it cannot correct |
|--------|---------------|------------------|------------------------|
| External calibration curve | Matrix-matched certified standards spanning the range | Instrument + average plasma response, *if* standards match the unknown | Any matrix difference between standard and unknown |
| Internal standardization | A reference line of known/constant concentration | Shot-to-shot and slow drift (multiplicative fluctuation) | Differential ablation; genuine composition-dependent line ratios |
| Chemometrics (PLS/PCR) | A large labeled training set | Non-linear, multi-line, matrix-coupled response empirically | Extrapolation outside the training manifold; interpretability |

Calibration-free LIBS was introduced precisely to escape the top-row requirement — standards —
by computing the scale factor $F$ from the closure constraint $\sum_s C_s = 1$ instead of from a
curve [@ciucci1999; @tognoni2010]. This chapter documents what that escape costs and what it
leaves on the table.

---

## Calibration curves {#calibration-curves}

**External calibration** fits, per analyte line, a response function against certified reference
materials (CRMs):

$$ I_{ki} = m\, C_s + b $$

where $m$ (intensity per mass fraction) is the sensitivity/slope and $b$ the blank intercept.
Concentration of an unknown is read back as $\hat{C}_s = (I_{ki} - b)/m$. In practice the curve
is often non-linear at high $C_s$ because of **self-absorption** (the curve of growth bends over;
see [self-absorption in the CF family](cf-libs-family.md)) and at low $C_s$ because of spectral
background — so the linear form holds only over a bounded dynamic range [@hahn2012; @cremers2013].

### Limit of detection and quantification {#lod-loq}

The detection and quantification limits follow the IUPAC convention analysed rigorously by
[@long1983]:

$$ \mathrm{LOD} = \frac{3\,\sigma_{\text{blank}}}{m}, \qquad \mathrm{LOQ} = \frac{10\,\sigma_{\text{blank}}}{m} $$

with $\sigma_{\text{blank}}$ the standard deviation of the blank (or near-blank) signal and $m$
the calibration slope. The factor 3 (LOD) and 10 (LOQ) are the widely-used $k$-values; [@long1983]
is explicit that they encode a *probabilistic* decision (false-positive/false-negative risk) and
must be reported with the underlying blank statistics, not quoted as a bare number. Two consequences
that carry directly into the CF-LIBS error budget ([error-budget-and-falsification.md](error-budget-and-falsification.md)):

- **LOD is slope-limited.** A weak or self-absorbed line has small $m$ and therefore a poor LOD
  regardless of SNR — a line-selection argument identical to the one physics-based line selection
  makes in [@tognoni2010].
- **Blank noise is the denominator's floor.** LIBS source noise (shot-to-shot ablation and coupling
  fluctuation) dominates over shot/detector noise for major elements [@tognoni2016]; this is why
  normalization (next section) matters more than detector cooling for most analytical lines.

### Matrix-matched standards {#matrix-matched}

The unstated premise of an external curve is that the standard and the unknown ablate and radiate
identically — i.e. share a *matrix*. When they do not, the curve's slope $m$ is wrong for the unknown
and the read-back is biased. Matrix-matched CRMs (e.g. certified steels for steel, basalt glasses for
rock) restore validity but are exactly the expensive, sample-specific artefacts CF-LIBS aims to
eliminate [@ciucci1999; @tognoni2010]. The historical `libssa` reference implementation
(narrated in `docs/REFERENCE_ANALYSIS_LIBSSA.md`, absorbed here) is a faithful example of this
classical workflow: Saha-Boltzmann plus univariate/PCA/PLS, all standards-dependent.

> [!IMPORTANT]
> The accuracy numbers quoted in this chapter (e.g. "classic CF ~10–20% relative", "OPC <1 wt%")
> are **literature-reported** figures from the cited papers, not CF-LIBS-pipeline benchmarks. They
> are not subject to the ASD59 reset banner. Do not read them as this repo's measured performance;
> for that see [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md).

---

## Internal standardization {#internal-standardization}

Internal standardization replaces an absolute intensity with a **ratio** to a reference line,
cancelling any *multiplicative* fluctuation common to both lines (ablated mass, laser energy,
collection solid angle):

$$ R = \frac{I_{ki}^{(\text{analyte})}}{I_{k'i'}^{(\text{ref})}} \;\propto\; \frac{C_{\text{analyte}}}{C_{\text{ref}}}\cdot \Phi(T, n_e) $$

The proportionality is exact only if the temperature/density factor $\Phi$ is the same for both
lines — i.e. if analyte and reference lines have similar upper-level energies $E_k$ and belong to
compatible ionization stages [@aragon2008; @cremers2013]. When $C_{\text{ref}}$ is known and roughly
constant (a major matrix element, or a deliberately added spike), the ratio $R$ is a drift-corrected
proxy for the analyte concentration. This is the classical ancestor of two things the shipped
pipeline still does:

| Classical idea | Shipped realization | Code path |
|----------------|---------------------|-----------|
| Reference-line ratio for drift correction | `InternalStandardizer` normalization | `cflibs/inversion/physics/matrix_effects.py::InternalStandardizer` |
| Single-standard scale anchoring | One-point calibration (OPC) of the Boltzmann plot | `cflibs/inversion/physics/matrix_effects.py::OnePointCalibrator` |

The **one-point calibration** layer [@cavalcanti2013] is the modern hinge between the classical and
calibration-free worlds: it uses a *single* certified standard to fix the Boltzmann-plot
intercept/scale, recovering much of the trueness of a full calibration curve at a fraction of the
standard cost. In the source study it delivered sub-wt% trueness on bronzes versus tens-of-percent
uncertainty for un-corrected inverse CF-LIBS [@cavalcanti2013]. It is **opt-in** in the shipped
code — it trades the "no standards at all" purity of CF-LIBS for accuracy, and that trade is a
[DESIGN-DECISION] left to the caller, not a default.

> [!NOTE]
> **Ratios over closure for tracking.** For the composition-*drift* deliverable (a known element
> set, precision over absolute accuracy), prefer log-intensity or log-concentration ratios
> $\ln(N_i/N_j)$ to closure wt% — the internal-standard ratio *is* the natural observable, and its
> shot-to-shot precision is what a DED tracking loop consumes. See
> [cf-libs-family.md](cf-libs-family.md) and the log-ratio (ILR) closure discussion.

---

## Matrix effects: the five-component framework {#matrix-effects}

"Matrix effect" is an umbrella for every way that two samples with the *same* analyte mass fraction
produce *different* analyte line intensities. Following the taxonomy of [@hahn2010; @hahn2012] and
the `matrix_effects.py` module header, it decomposes into five physically distinct channels. The
last column — **what CF-LIBS removes** — is the honest core of this chapter.

| # | Channel | Physical mechanism | Does closure-based CF-LIBS remove it? |
|---|---------|--------------------|----------------------------------------|
| 1 | **Ablation stoichiometry** | Mass removed per pulse is element-dependent (thermal/differential ablation); the plasma composition ≠ the solid composition | **No.** This is the hard limit. Closure normalizes the *plasma*, so any solid→plasma fractionation propagates straight through. |
| 2 | **Plasma-parameter shift** | Matrix changes $T$ and $n_e$, moving every line's Saha-Boltzmann factor | **Partially.** A single self-consistent $(T, n_e)$ fit *per shot* absorbs a uniform shift; it cannot absorb spatial gradients or non-LTE. |
| 3 | **Self-absorption** | Optically thick lines saturate (escape factor $SA(\tau)=(1-e^{-\tau})/\tau$) so intensity is sub-linear in $C_s$ | **No, not by closure alone.** Requires an explicit correction (IRSAC, C-sigma, CD-SB) — see [cf-libs-family.md](cf-libs-family.md). |
| 4 | **Chemical matrix** | Composition-dependent line ratios and molecular recombination independent of $T,n_e$ | **No.** Empirical only; the `MatrixEffectCorrector` factor DB handles it heuristically. |
| 5 | **Instrument** | Wavelength-dependent spectral response, resolving power, detector nonlinearity | **Yes, largely** — the multiplicative $F$ and the response curve cancel in the closure ratio; this is CF-LIBS's cleanest win. |

The module docstring states the assumption plainly: *"Standard CF-LIBS assumes these effects cancel
in the closure equation, but this assumption breaks down for samples with very different
thermal/optical properties, organic matrices, highly reflective metallics, and complex
mineralogy."* That is channels 1, 3, and 4 — the three CF-LIBS does **not** remove.

### Ablation regime is a first-class knob {#ablation-regime}

The default empirical correction factors encode *nanosecond*-pulse fractionation (e.g. the ~15%
carbon-loss baked into `METALLIC.C = 0.85`). Ultrashort (ps/fs) ablation is much closer to
stoichiometric, so the generic ns factors are physically wrong there; the code exposes this as
`AblationRegime.{NS, PS}` and collapses the generic factors to 1.0 for `PS`
(`cflibs/inversion/physics/matrix_effects.py::AblationRegime`). The ps 1.0 defaults are an explicit
*"no generic correction known"* placeholder, **not** a claim of perfect stoichiometry — supply
calibrated factors when ps CRM data exist. Grounding: [@hahn2010; @cremers2013] for the ns empirical
basis and regime dependence.

> [!CAUTION]
> **Channel 1 is where over-claiming happens.** Because closure produces a plausible-looking,
> sums-to-100% answer for *any* input, a fractionating matrix yields a confidently-wrong
> composition. The refuse-to-report / reliability gate
> ([error-budget-and-falsification.md](error-budget-and-falsification.md)) exists partly to catch
> this: a good-looking closure fit is not evidence that ablation was stoichiometric.

Cross-link: the full quantitative error budget for these channels — including which are
falsifiable and which have been falsified in this repo — lives in
[error-budget-and-falsification.md](error-budget-and-falsification.md).

---

## Chemometrics overview (cited concept, not shipped) {#chemometrics}

> [!IMPORTANT]
> **Physics-only constraint.** The shipped CF-LIBS algorithm must not import `sklearn`, `torch`,
> `tensorflow`, `jax.nn`, etc. (enforced by Ruff TID251 + the AST blocklist in
> `cflibs/evolution/evaluator.py`). The methods in this section are therefore **described as
> literature, not implemented in the shipped path**. They carry no `code_refs`. Where their *idea*
> is realized physics-only (e.g. sparsity via NNLS/Tikhonov instead of LASSO), that is noted.

Chemometrics attacks the matrix problem empirically: instead of modeling the plasma, it regresses
composition on the whole spectrum from a labeled training set.

| Method | One-line description | LIBS status | Physics-only analogue |
|--------|----------------------|-------------|-----------------------|
| **Univariate** | Single-line intensity → concentration; the classical curve of [Calibration curves](#calibration-curves) | Baseline; matrix-fragile | The curve itself |
| **PCA** | Unsupervised variance decomposition for exploration / outlier flagging / dimensionality reduction | Standard preprocessing/QC | Retained: `cflibs/inversion/common/` PCA pipeline is used for diagnostics, not regression |
| **PCR** | Regress concentration on principal-component scores | Reduces collinearity; needs training set | None shipped |
| **PLS / PLS-DA** | Latent-variable regression maximizing covariance with the target; the workhorse of quantitative chemometric LIBS | Best empirical accuracy with good coverage; opaque, extrapolation-fragile | Sparsity *prior* via NNLS / L1-regularized SciPy solves |
| **LASSO / elastic net** | Sparse/shrinkage regression, automatic line selection under matrix effects | Strong for geological LIBS [@anderson2012] | NNLS / Tikhonov (`sklearn` barred) |

The most transferable chemometric insight for a *physics-first* pipeline is the **dominant-factor
hybrid**: use the physics forward model as the leading term and let a small multivariate model
correct the residual [@feng2014]. Two halves, two verdicts:

- The **spectrum-standardization preprocessing** half of [@feng2014] is pure signal conditioning —
  physics-only-compatible and portable.
- The **PLS residual-correction** half uses multivariate regression and is therefore
  `physics_only_compatible = false`: keep it strictly as an offline diagnostic, never in the
  shipped inversion.

For interpretability and correct uncertainty on the *closure* output, the compositional-data-analysis
literature ([@egozcue2003], isometric log-ratio transforms) is the physics-compatible replacement for
naive multivariate covariance — it lives in the ILR closure mode
([cf-libs-family.md](cf-libs-family.md)), not in any ML estimator.

The empirical ceiling matters as context, not as a target: modern ML classifiers report >99%
test accuracy on rock-type discrimination and drug spectra (survey in
`docs/research/2026-06-19-cross-discipline-sota.md`), establishing the upper bound a physics
identifier is measured against — but they are knowledge-only here by construction.

---

## What correct code MUST do {#checklist}

A review-register checklist for any module touching classical quantification in this repo:

1. **Never quote an LOD without its blank statistics and slope.** LOD $= 3\sigma_{\text{blank}}/m$ is
   a decision rule, not a constant [@long1983]; report $\sigma_{\text{blank}}$, $m$, and the $k$-value.
2. **Ratio, don't subtract, for drift.** Internal standardization cancels multiplicative noise only
   as a *ratio* of lines with compatible $E_k$ and ionization stage [@aragon2008]; mismatched lines
   reintroduce a $\Phi(T,n_e)$ bias.
3. **Treat the five matrix channels separately.** Closure removes channel 5 (instrument) and part of
   channel 2 (uniform $T,n_e$ shift); it does **not** remove channels 1, 3, 4. Any code claiming
   "matrix-corrected" must say *which* channel.
4. **Make ablation regime explicit.** ns fractionation factors are wrong for ps/fs data; default to
   `AblationRegime.NS` only knowingly, and never present ps 1.0 factors as "stoichiometric proven".
5. **Keep chemometrics out of the shipped path.** PLS/PCR/LASSO are literature; realize their *ideas*
   physics-only (NNLS sparsity, ILR covariance) or keep them as offline diagnostics. Enforcement is
   automatic (Ruff TID251 + AST blocklist) — do not defeat it.
6. **OPC is opt-in, and it spends a standard.** Enabling `OnePointCalibrator` trades CF-LIBS's
   standards-free property for accuracy [@cavalcanti2013]; surface that trade, do not hide it.

---

## See also

- [cf-libs-family.md](cf-libs-family.md) — how calibration-free methods remove the *standard* (and which self-absorption corrections tackle channel 3).
- [error-budget-and-falsification.md](error-budget-and-falsification.md) — quantitative budget for the five matrix channels; falsification ledger.
- [libs-physics.md](libs-physics.md) — the Saha-Boltzmann forward chain that $T$, $n_e$, $\tau$ enter.
- [atomic-data-and-datasets.md](atomic-data-and-datasets.md) — CRMs, line lists, and why matrix-matched standards are expensive.
- [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md) — this repo's *measured* accuracy (post-ASD59-reset), distinct from the literature numbers cited here.
