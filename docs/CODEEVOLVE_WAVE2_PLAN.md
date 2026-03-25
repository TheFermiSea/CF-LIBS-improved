# CodeEvolve Wave 2 Planning Document

**Date:** 2026-03-25
**Status:** Draft / Brainstorming
**Author:** Brian Squires (via Claude session)

---

## Context: Wave 1 Results

Wave 1 applied GPU-accelerated CMA-ES (Sep-CMA-ES variant) to optimize a 101-parameter multi-algorithm ensemble for element identification on 74 Aalto LIBS spectra at RP 300--1100.

### Key Numbers

| Metric | Value |
|--------|-------|
| Objective | F_beta(0.5) |
| Best result | **0.914** |
| Parameters | 101 (8 groups: algorithm weights, per-algorithm configs, 22 per-element thresholds, decision fusion) |
| Evaluations | 2 x 10^6 configurations in 58 seconds (35,300 evals/s) |
| Population | 4,096 candidates, 500 generations |
| Baseline (manual grid search) | F1 = 0.654 (hybrid NNLS+ALIAS, intersection mode) |
| Baseline (single-algo CMA-ES, 41 params) | F_beta = 0.551 |

### What the Optimizer Discovered

- **Default detection threshold:** 0.30 -> 0.90 (+200%). The hand-tuned baseline was dramatically too permissive.
- **NNLS dominant:** weight 2.93 (next highest: Comb at 0.97). ALIAS, Correlation, Hybrid suppressed with negative weights.
- **Mn/Na nearly vetoed:** per-element thresholds of 0.93 and 0.87 respectively, confirming their pathological false-positive status.
- **Reliable elements kept open:** Fe (0.05), Si (0.12), Al (0.07) -- consistent with the per-element benchmark analysis.
- **Decision fusion SNR center:** 9.5, well above the 3-sigma IUPAC detection limit.

### Key Limitation

The optimization operated on **cached algorithm outputs** (static .npz arrays from `generate_evolve_cache_v2.py`), not end-to-end through the forward model. The five algorithms (ALIAS, NNLS, Correlation, Comb, Hybrid) were run once on all 74 spectra, and the optimizer only learned how to combine/threshold those fixed outputs. This means:

1. The basis spectra are frozen (fixed T, ne, FWHM).
2. Algorithm hyperparameters that affect the raw outputs (e.g., NNLS regularization, ALIAS peak detection sensitivity) are not jointly optimized.
3. The optimizer cannot discover that a different basis library would produce better raw algorithm scores.

The 0.914 result is therefore a **strong local minimum within the combiner parameter space**, but potentially far from the global optimum over the full (basis + algorithm + combiner) parameter space.

---

## Wave 2 Strategy: Escaping the Local Minimum

The core question: how do we get beyond F_beta = 0.914 without simply overfitting harder to 74 spectra?

Three axes of attack: (1) improve the inputs to the algorithms (basis library), (2) expand the algorithm ensemble (new identification pathways), (3) search more broadly in the optimization landscape.

---

### 1. Basis Library Improvements

The current basis library uses a fixed T/ne grid with fixed Gaussian FWHM broadening. Several improvements are possible:

**1a. Joint T/ne Grid Optimization**

Currently the basis is generated at a fixed set of plasma conditions (e.g., T = 0.5--2.0 eV, ne = 1e15--1e18 cm^-3). The grid spacing was chosen by hand. Questions:

- Is the grid dense enough in the parameter regions that matter for element identification (vs. quantification)?
- Could we optimize the grid placement itself? E.g., adaptive mesh refinement: identify which (T, ne) values produce the most discriminative basis spectra for the problematic elements (Mn, Na, Ca), and add grid points there.
- Practical concern: regenerating the full basis library takes hours on GPU. Any grid change requires a full re-cache of algorithm outputs.

**1b. Additional Ionization Stages (III, IV)**

The current basis includes neutral (I) and singly-ionized (II) species. For higher-T plasmas or multi-pulse LIBS:

- Exact partition functions are now available for 83 elements (direct summation from NIST energy levels). Could this enable reliable basis spectra for Fe III, Ti III, etc.?
- At the Aalto RP and temperature range (T ~ 0.8--1.5 eV), Stage III populations are typically negligible. But for elements with low second ionization potentials (Ba, Sr, Ca), Stage III lines may be non-negligible.
- **Decision needed:** Is the added complexity worth it for the current dataset? Probably not for Aalto spectra, but could matter for higher-T plasma sources.

**1c. Wavelength-Dependent Instrument Response**

The current basis uses a fixed Gaussian FWHM (1.0 nm). Real spectrometers have:

- Wavelength-dependent resolving power (RP varies from ~300 at 200 nm to ~1100 at 900 nm in the Aalto system).
- Non-Gaussian line shapes (especially echelle spectrometers).
- Efficiency curves that modulate relative line intensities.

Implementing a wavelength-dependent FWHM(lambda) in the basis generation would produce more realistic basis spectra. This is a relatively straightforward change to `generate_evolve_cache_v2.py` / the `BasisLibrary` class.

**1d. Continuum and Self-Absorption**

The current basis is emission-only. Adding:

- Bremsstrahlung + recombination continuum would provide a more realistic spectral background, potentially improving NNLS decomposition.
- Self-absorption correction for strong lines (especially alkali resonance lines like Na D and Li 670 nm) would reduce the discrepancy between modeled and observed line ratios.

---

### 2. Algorithm Architecture Changes

**2a. New Identification Pathways**

Candidates to add to the ensemble:

| Pathway | Description | Expected Strength | Implementation Effort |
|---------|-------------|-------------------|----------------------|
| Deep learning spectral matching | Train a CNN/transformer on synthetic spectra to output element presence probabilities | Could learn non-linear spectral features invisible to linear methods | High (need training data, architecture search) |
| Template correlation with learned templates | Instead of fixed basis spectra, learn optimal template spectra per element that maximize discriminability | Could overcome basis library limitations | Medium |
| Multi-resolution NNLS | Run NNLS at 2--3 different spectral resolutions (e.g., 0.5 nm, 1.0 nm, 2.0 nm) and combine | Higher resolution may resolve blends; lower resolution may be more robust to noise | Low (just re-bin and re-run) |
| Derivative spectroscopy | Match on first/second derivatives of spectra | Less sensitive to baseline/continuum; highlights sharp features | Low |
| Mutual information scoring | Score element presence by mutual information between basis spectrum and observed spectrum | Non-linear alternative to correlation | Medium |

**2b. Multi-Resolution Approach**

This is low-hanging fruit. The idea:

1. Generate basis libraries at 3 resolutions: FWHM = 0.5, 1.0, 2.0 nm.
2. Run NNLS independently at each resolution.
3. Include all three sets of NNLS outputs in the ensemble cache.
4. Let the optimizer learn resolution-dependent weights per element.

Rationale: At FWHM = 0.5 nm, closely-spaced line multiplets (e.g., Fe 371--375 nm complex) become partially resolved, potentially improving Fe/Mn discrimination. At FWHM = 2.0 nm, noise is smoothed and broad spectral envelope features become more prominent, potentially improving detection of elements with many weak lines.

**2c. Temporal/Spatial Averaging for Noisy Spectra**

Some Aalto spectra are single-shot with high noise. If multi-shot data is available:

- Shot-to-shot averaging with outlier rejection.
- Spatial binning for spatially-resolved data.
- This is a data preprocessing change, not an algorithm change, but could significantly improve raw algorithm outputs.

---

### 3. Optimization Strategy Changes

**3a. Warm-Starting from Wave 1**

Instead of random initialization, start CMA-ES from the Wave 1 optimum:

- Initialize mean at Wave 1 best parameters.
- Inflate the covariance matrix: sigma = 0.3--0.5 (instead of the typical 0.1--0.2 for fine-tuning).
- This explores a neighborhood around the known good solution while allowing escape to nearby basins.

**3b. Large-Perturbation Restart**

More aggressive: start from Wave 1 optimum but with sigma = 0.5--1.0, effectively treating it as a rough starting point. Combine with IPOP-CMA-ES (increasing population restart strategy) to systematically increase the search radius after each convergence.

**3c. Alternative Objective Functions**

Wave 1 used F_beta(0.5), which weights precision 2x over recall. Alternatives:

| Objective | Pros | Cons |
|-----------|------|------|
| F1 (equal precision/recall) | Standard metric, easier to compare with literature | May not match CF-LIBS needs where FPs are more costly |
| F_beta(0.3) | Even stronger precision emphasis | May over-suppress detections |
| Multi-objective Pareto (precision vs recall) | Produces a front of solutions; user picks operating point | More complex, slower convergence |
| Log-loss on per-element probabilities | Smooth, differentiable; penalizes confident wrong answers | Requires probability-calibrated outputs |
| Exact-match rate | Directly optimizes the hardest metric (22/74 = 30% is current best) | Very sparse gradient signal; many plateaus |

**3d. Population-Based Training (PBT)**

Instead of a single CMA-ES run, maintain a population of diverse configurations and evolve them with different mutation strategies:

- Some individuals explore algorithm weight space.
- Some individuals explore per-element threshold space.
- Some individuals explore algorithm hyperparameters.
- Periodically share good strategies across sub-populations.

This is essentially an island-model evolutionary approach. Could be implemented on top of the existing CMA-ES infrastructure.

**3e. Quality-Diversity Search (MAP-Elites)**

Use MAP-Elites or similar quality-diversity algorithms with behavior descriptors:

- Number of elements detected (mean across spectra).
- Precision/recall balance (P/(P+R) ratio).
- Computational cost.

This would produce a diverse archive of high-performing configurations, revealing whether there are fundamentally different strategies at similar performance levels.

**3f. Bayesian Optimization**

For the outer loop (choosing which algorithm configurations to cache and which combiner architecture to use), Bayesian optimization (GP-UCB or TPE) could be more sample-efficient than CMA-ES. Particularly relevant if basis library regeneration is in the loop, since each evaluation is expensive.

---

### 4. Dataset Expansion

**4a. Synthetic Spectra with Known Ground Truth**

Generate synthetic LIBS spectra using the forward model:

- 500--1000 spectra covering diverse compositions (pure elements, binary mixtures, geological compositions).
- Known ground truth by construction.
- Add realistic noise (Poisson + readout + baseline).
- Use for training (optimize on synthetic) / validation (evaluate on Aalto) split.

Risk: synthetic spectra may not capture all the pathologies of real spectra (matrix effects, self-absorption, continuum structure, detector artifacts).

**4b. Cross-Validation on Aalto Spectra**

Current approach: optimize on all 74 spectra, evaluate on all 74 spectra. This is a train-on-test setup.

Proposed: k-fold cross-validation (k=5, so 59 train / 15 test per fold). This gives:

- Honest generalization estimate.
- Per-fold variance as uncertainty on the F_beta estimate.
- Risk: 15 spectra per fold may be too few for stable per-element metrics.

Alternative: leave-one-mineral-out cross-validation (train on 73, test on 1, repeat). Very expensive but mineral-level generalization is what we actually care about.

**4c. ChemCam Calibration Target Spectra**

The CCCT data (6 targets with certified oxide compositions) is already validated in the paper. Additionally:

- ChemCam has published calibration standard spectra (69--332 standards from Clegg 2017).
- If accessible, these would provide a large, independent validation set at RP 2000--4000.
- Caveat: different RP range than Aalto; results would test generalization across instruments, not within-instrument performance.

The `data/chemcam_standards/` directory exists (newly added, untracked) -- this may already be in progress.

**4d. Adversarial Spectra**

Generate spectra specifically designed to challenge the Wave 1 optimum:

- Spectra where Mn is truly present (test whether the near-veto threshold is too aggressive).
- Spectra with trace Na in a geological matrix.
- Binary mixtures of elements with highly overlapping spectra (Fe+Mn, Ca+Na).
- Very low SNR spectra.

---

### 5. Informed by Literature

**5a. Hackem-LIBS (Zeng et al. 2020)**

Heterogeneous stacking ensemble for LIBS quantification. Key idea: train a meta-learner (e.g., ridge regression or gradient boosting) on the outputs of diverse base models. Could we use a stacking architecture where Wave 1's optimized ensemble is one of several "base learners," with a second-level model that learns when to trust each base learner?

Relevance: our current combiner is a fixed weighted-sum with per-element thresholds. A stacking approach could learn non-linear combinations (e.g., "trust NNLS for Fe only when the Comb score is also above X").

**5b. OPSIAL (Tan 2019)**

Forward-model fitting approach to element identification. Instead of matching peaks or decomposing spectra, OPSIAL fits a radiative transfer model line-by-line to the observed spectrum. Could OPSIAL-style line-by-line fitting improve the forward-model identification pathway (currently the weakest at F1 = 0.505)?

Relevance: our current forward-model pathway uses concentration thresholding on NNLS coefficients, which is crude. A proper line-by-line fit with residual analysis could distinguish true contributions from fitting artifacts.

**5c. ExoAtom (Ni et al. 2025)**

Expanded atomic line lists beyond NIST ASD. ExoAtom compiles line data from Kurucz, VALD, and other sources, providing more complete line lists especially for:

- Rare earth elements.
- High-excitation transitions missing from NIST.
- Isotopic shifts.

Could additional line data improve basis spectra completeness and reduce false negatives for elements with incomplete NIST coverage (Zn, Zr are currently undetectable partly due to insufficient database coverage)?

**5d. Barklem & Collet (2016)**

Partition functions for 284 atomic species via direct summation over experimental and theoretical energy levels. This validates our direct-summation approach (currently implemented for 83 elements). Could provide:

- Cross-checks for our partition function values.
- Extension to additional species not in our current database.
- Temperature-dependent partition function derivatives for sensitivity analysis.

**5e. Additional References to Investigate**

- Kim et al. 2025: recent deep learning approaches to LIBS element identification.
- Dai et al. 2026: if available, any advances in low-RP LIBS identification.
- Eum et al. 2021: ensemble approaches for LIBS classification.

---

### 6. Technical Improvements

**6a. End-to-End Differentiable Pipeline**

The most ambitious improvement: make the full pipeline differentiable via JAX so that gradient-based optimization can propagate through:

```
parameters -> basis library -> algorithm outputs -> combiner -> F_beta score
```

Currently only the last two steps (combiner -> F_beta) are optimized. Making the full chain differentiable would allow:

- Joint optimization of basis library parameters (T, ne, FWHM) alongside combiner parameters.
- Gradient-based discovery of which basis configurations are most discriminative.
- Potential for dramatic performance improvements by co-adapting all stages.

Challenges:

- NNLS decomposition is not trivially differentiable (requires implicit differentiation or unrolled optimization).
- ALIAS peak-matching involves discrete operations (peak detection, thresholding) that need relaxation for differentiability.
- The F_beta score involves discrete TP/FP/FN counts that need a soft approximation.
- Full pipeline evaluation is ~0.39s per spectrum (vs. microseconds for cached outputs); 2M evaluations would take weeks instead of 58 seconds.

Compromise: make the basis generation differentiable but keep the algorithm outputs cached. Periodically regenerate the cache at the current basis parameters, then optimize the combiner. This "alternating optimization" approach is much cheaper than full end-to-end.

**6b. Bayesian Optimization for Outer Loop**

Use Gaussian Process-based Bayesian optimization (e.g., via BoTorch) for:

- Selecting which new algorithms to add to the ensemble.
- Choosing basis library parameters.
- Hyperparameter optimization of the CMA-ES itself (population size, sigma, learning rates).

This is appropriate when evaluations are expensive (e.g., each requires regenerating the basis + re-caching).

**6c. Warm-Starting CMA-ES from Wave 1**

Initialize CMA-ES with:

- Mean = Wave 1 best parameters.
- Covariance = Wave 1 final covariance matrix (if saved) inflated by factor of 4--16.
- Sigma = 0.3--0.5 (vs. the converged sigma from Wave 1 which is likely < 0.01).

This preserves the learned correlation structure while allowing broad exploration.

**6d. Adaptive Per-Element Search List**

Currently the search list is fixed at 22 elements. The optimizer could also learn:

- Which elements to include/exclude from the search list.
- Per-spectrum adaptive element lists based on a quick first-pass screening.

This would require expanding the parameter space but could help with the Mn/Na problem (simply exclude them from the search).

---

## Session Planning

### Session 1: Basis Library + New Algorithms (estimated: 4--6 hours)

**Goals:**
- Implement wavelength-dependent FWHM in basis generation.
- Generate multi-resolution basis libraries (0.5, 1.0, 2.0 nm).
- Add derivative spectroscopy and multi-resolution NNLS as new identification pathways.
- Regenerate the evolve cache with all new algorithm outputs.

**Deliverables:**
- Updated `generate_evolve_cache_v3.py` with new algorithms.
- New basis libraries in `output/evolve_cache/`.
- Updated cache .npz with additional algorithm output arrays.

**Prerequisites:**
- GPU access (vasp-01/02/03) for basis generation.
- Decision on which new algorithms to implement.

### Session 2: Optimization Framework (estimated: 3--4 hours)

**Goals:**
- Implement train/test split infrastructure (k-fold and leave-one-mineral-out).
- Add alternative objective functions (F1, multi-objective Pareto, log-loss).
- Implement warm-start from Wave 1 optimum.
- Add IPOP-CMA-ES restart strategy.

**Deliverables:**
- Updated optimization script with cross-validation support.
- Multi-objective optimization mode.
- Warm-start configuration file pointing to Wave 1 parameters.

**Prerequisites:**
- Wave 1 final parameters saved in a loadable format.
- Decision on objective function(s).

### Session 3: Optimization Campaign (estimated: 2--3 hours active, overnight GPU runs)

**Goals:**
- Run Wave 2 optimization with expanded algorithm set + warm start.
- Run cross-validation experiments.
- Compare F_beta across: Wave 1 baseline, Wave 2 warm-start, Wave 2 cold-start, multi-objective Pareto.

**Deliverables:**
- Wave 2 optimization results (best parameters, convergence curves).
- Cross-validation F_beta estimates with confidence intervals.
- Pareto front if multi-objective was run.

**Prerequisites:**
- Sessions 1 and 2 complete.
- GPU allocation for overnight runs.

### Session 4: Analysis + Paper Update (estimated: 3--4 hours)

**Goals:**
- Analyze Wave 2 results vs. Wave 1.
- Per-element breakdown of improvements/regressions.
- Update paper results section if Wave 2 improves on Wave 1.
- Document what worked and what didn't.

**Deliverables:**
- Updated results tables in paper (if improved).
- Wave 2 analysis report in `docs/reports/`.
- Updated CLAUDE.md if workflow changed.

---

## Open Questions for Human Decision

### Priority 1 (blocking Session 1)

1. **Which new identification pathways are worth implementing?**
   - Multi-resolution NNLS is low effort / moderate reward. Recommend yes.
   - Derivative spectroscopy is low effort / unknown reward. Recommend yes.
   - Deep learning is high effort / potentially high reward. Recommend defer to Wave 3.
   - Template correlation with learned templates is medium effort. Recommend defer.

2. **Should we implement wavelength-dependent FWHM?**
   - This requires knowing the Aalto spectrometer's FWHM(lambda) function. Is this characterized?

3. **Budget: how many GPU-hours are available for Wave 2?**
   - Basis generation: ~1--2 hours per resolution per FWHM.
   - Cache generation: ~30 min per algorithm set.
   - Optimization runs: ~1--5 min each, but many runs needed for cross-validation.
   - Estimated total: 10--20 GPU-hours.

### Priority 2 (blocking Session 2)

4. **Should we re-run Wave 1 with a train/test split before Wave 2?**
   - This would establish a honest baseline for comparison.
   - Without it, we cannot claim Wave 2 is better (both could be overfitting).
   - Recommendation: yes, run 5-fold CV on Wave 1 configuration first.

5. **Should we pursue end-to-end differentiability or stay with cached outputs?**
   - End-to-end is the principled approach but dramatically more expensive.
   - Recommendation: stay with cached outputs for Wave 2, but implement the alternating optimization as a stretch goal.

6. **F_beta(0.5) or try a different objective?**
   - F_beta(0.5) strongly favors precision. The Wave 1 result (0.914) may already be near the precision ceiling.
   - Switching to F1 would test whether we can improve recall without sacrificing precision.
   - Multi-objective Pareto would give the most information but is harder to summarize.

### Priority 3 (nice to have)

7. **Should we expand the element search list beyond 22?**
   - Adding more elements increases the false-positive surface area.
   - But some geological applications need rare earths, Be, B, etc.

8. **Should we incorporate the ChemCam standards data** (in `data/chemcam_standards/`) for validation?
   - Different RP range, so it tests cross-instrument generalization.
   - May not be directly comparable with Aalto results.

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Wave 2 shows no improvement over Wave 1 | Medium | Low | Wave 1 is already a strong result; paper is publishable as-is |
| Cross-validation reveals Wave 1 overfitting | Medium | Medium | Expected behavior; honest CV score is still valuable for the paper |
| Basis regeneration takes too long | Low | Medium | Use existing basis as fallback; only regenerate at priority resolutions |
| New algorithms degrade ensemble | Low | Low | Optimizer can learn to ignore them (assign zero weight) |
| End-to-end pipeline is infeasible | High | Low | Stay with cached outputs; defer to Wave 3 |

---

## Success Criteria

- **Minimum:** Establish honest cross-validated F_beta for Wave 1 configuration. This is valuable even if Wave 2 doesn't improve on it.
- **Target:** Wave 2 cross-validated F_beta > Wave 1 cross-validated F_beta by at least 2 percentage points.
- **Stretch:** F_beta(0.5) > 0.93 on cross-validated test folds, or Pareto front that dominates Wave 1 in both precision and recall.
- **Bonus:** Identify a physically interpretable new strategy (e.g., "multi-resolution NNLS resolves Fe/Mn confusion") that merits discussion in the paper.
