# J10 forward_fit — literature clues for raising identification PRECISION

**Date:** 2026-06-15 · **Bead:** o3be · Problem: forward_fit (Gornushkin-style population
forward-fitter) reaches P≈0.51/R≈0.46/F1≈0.49 on the 11-candidate confounder-laden ak3.1.3
corpus — ~half of detections are false positives. Goal: physics-only techniques to cut FPs
without killing recall. Citations carry verification caveats (see end); principles are
multiply-confirmed, but verify exact author/venue/numbers before formal (paper) citation.

## Literature corroborates the in-flight levers
- **BIC-margin presence gate** (lever b, testing now): **Wilson, da Silva Castanheira, Lévesque
  Kinder & Baillet 2024**, *A Bayesian Model-Selection Approach for Determining the Number of
  Spectral Peaks in Neural Power Spectra*, **bioRxiv DOI 10.1101/2024.08.01.606216** (PMC11326208 /
  PMID 39149403; preprint — no journal version confirmed) — BIC-gated peak inclusion raised PPV
  **63%→96%** (sensitivity 91→89%) on ~5,000 synthetic spectra (verified verbatim). Right *category*
  of fix, but the domain is **neural power spectra (specparam/FOOOF), not LIBS** — a structural
  analogy, not a quantitative prediction for forward_fit. **Webb et al. 2021** (MNRAS 501:2268)
  argue BIC underperforms a line-strength-weighted **SpIC** → an upgrade path.
- **Diagnostic per-wavelength weights** (lever a, merged): **Amato, Cristoforetti, Legnaioli,
  Lorenzetti, Palleschi, Sorrentino & Tognoni 2010**, *Progress towards an unassisted element
  identification from LIBS with automatic ranking techniques inspired by text retrieval*,
  **Spectrochim. Acta B 65(8):664–670, DOI 10.1016/j.sab.2010.04.019** (verified) — defines
  "selectivity = log(1/element-frequency)" × "peak relevance" = the TF-IDF/IEF crowding weight (the
  exact source for lever #1 below); **Aydin et al. 2008** Boltzmann-residual down-weighting.
- **Multi-line coherence** (lever d): **Hahn & Omenetto 2012** (Appl. Spectrosc. 66:347) —
  "identification should never be done on a single line"; **Labutin & Zaytsev 2013** (Anal.
  Chem. 85:1693) fit all lines of a species globally; **El Haddad et al. 2014** good-practices.
- **Score calibration** (lever e): **Protassov et al. 2002** (ApJ 571:545) — naive LRT
  thresholds are miscalibrated when a parameter is bounded at 0 (concentration ≥ 0), inflating
  FP rate; explains the non-monotonic threshold behavior we measured.

## New, citable, physics-only ideas (ranked)
1. **IEF / TF-IDF spectral-crowding weight** (Amato 2010) — **TOP**. In `_diagnostic_wavelength_weights`,
   after the per-element Gaussian profiles array, compute `f_i = (profiles > noise_floor).sum(axis=0)`,
   `IEF_i = log((n_elements+1)/clip(f_i,1))`, multiply into `w_i`. Penalizes bins shared by many
   candidates (the dominant FP mechanism: confounders borrowing shared-bin signal in the crowded
   380–450 nm Fe/Ti/Cr/Ca region). ~10 lines, computed once (memoized), zero inference cost.
   Expected **+0.06–0.15** precision. Extends the merged lever (a).
2. **Iterative Boltzmann-R² post-gate** (Aydin et al. 2008, Spectrochim. Acta B 63:1060) — the
   Saha-Boltzmann *consistency* filter. Host-side pure-numpy: for each present element with ≥3
   matched lines, regress `ln(I·λ/(gA))` vs `E_k`; if R² < 0.85, veto (a true element's lines obey
   one temperature; a confounder's coincidental lines don't). Expected **+0.04–0.10**. Pairs with
   the min-line-count gate.
3. **Minimum-line-count hard gate** (Hahn & Omenetto 2012; Labutin & Zaytsev 2013) — require ≥2
   unblended in-band predicted lines above an SNR floor before a presence call. Expected +0.05–0.12
   (kills single-coincidence FPs). This *is* lever (d), now literature-grounded.
4. **SpIC** (Webb et al. 2021, MNRAS 501:2268) — replace flat `k·log(N)` in `bic_cost` with a
   per-line-strength penalty `Σ log(N)/I_j` (weak/blended lines pay more). Incremental upgrade to
   the BIC gate; +0.02–0.06.
5. **Simulation-calibrated null threshold** (Protassov et al. 2002) — set `presence_threshold` per
   element at the 95th percentile of the gap under K≈200 element-excluded null forward-model runs
   (cheap, already vmapped). Fixes the miscalibrated/non-monotonic threshold; a precomputed
   (n_elements, SNR-tier) lookup serves production.
6. **NNLS joint sparse decomposition** (Tibshirani 1996 LASSO; NNLS is `scipy.optimize.nnls`,
   pure numerical, physics-OK) — replace per-element-independent LOO gaps with one joint fit:
   build an `N_wl × N_elements` basis at the best (T,n_e), solve NNLS vs the measured spectrum,
   drop elements with `c_e/Σc < ~1%`. A confounder spectrally dominated by a true element zeroes
   out by competition. A more invasive but principled redesign of the presence decision.
7. **Robust Student-t Boltzmann line weighting** (Matsumura et al. 2024, ACS Earth Space Chem.
   8:1259) — second-pass adaptive weights from Boltzmann residuals (down-weights self-absorbed/
   blended lines automatically); +0.03–0.08, an alternative formulation of lever (a).
8. **Self-absorption correction in the forward model** (Gornushkin et al. 1999) — `apply_self_absorption`
   toggle already exists; enabling it deflates over-predicted strong resonance lines of abundant
   confounders. **Benchmark first** (helps only optically-thick regimes; can hurt thin ones).

## Recommended plan (after the BIC-gate sweep result lands)
1. **IEF crowding weight** (#1) — highest ROI, ~10 lines, extends the merged diag-weights; GPU A/B vs current.
2. **Boltzmann-R² veto** (#2) + **min-line-count gate** (#3) — the physics multi-line-coherence FP filter.
3. Then SpIC (#4) / null-calibration (#5) as refinements; NNLS (#6) as a larger redesign if needed.
Every change benchmark-gated on the GPU vs Comb (project rule; a prior identifier change regressed F1 −0.041).

## Citation verification caveats (honest)
- **Wilson 2024** DOI VERIFIED 10.1101/2024.08.01.606216 (bioRxiv preprint, first author Luc E. Wilson;
  the earlier "PLOS Comp Biol" guess was WRONG — no journal version found). Domain is *neural power
  spectra*, not optical emission — the BIC-gated-peak analogy is structural, the 63→96% PPV is not a direct
  LIBS prediction. **Amato 2010** DOI VERIFIED 10.1016/j.sab.2010.04.019 (first author G. Amato; the
  "Andrade" variant was a false lead).
- **El Haddad 2014**, **Gajarska 2024** author-ordering: papers/venues confirmed via search,
  but exact phrases / per-tooth thresholds / first-author ordering not confirmed from full text — verify
  before formal citation. The underlying principles are standard and multiply-confirmed.
- All precision-gain figures are by-analogy estimates; actual gains depend on the corpus line density / SNR.
- Corpus validity (`zfy2`) still caps absolute numbers; same-corpus relative-vs-Comb is the reliable signal.
