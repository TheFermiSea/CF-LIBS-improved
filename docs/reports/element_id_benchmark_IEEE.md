# Comparative Evaluation of Element Identification Algorithms for Low-Resolution Laser-Induced Breakdown Spectroscopy

**B. Squires**

*CF-LIBS Project, Department of Physics*

---

*Abstract* — Automated element identification in Calibration-Free Laser-Induced Breakdown Spectroscopy (CF-LIBS) is fundamentally limited at the low resolving powers (RP < 1000) typical of field-deployed instruments, where severe spectral overlap renders peak-matching unreliable. We present a systematic benchmark of five identification pathways — ALIAS peak-matching, full-spectrum NNLS decomposition, a novel two-stage hybrid NNLS+ALIAS identifier, Voigt deconvolution with ALIAS, and forward-model concentration thresholding — evaluated on 74 LIBS spectra (13 pure elements, 61 minerals) from the Aalto University spectral library at RP = 300–1100. The hybrid NNLS+ALIAS approach in intersection mode achieves the best performance: precision P = 0.604, recall R = 0.713, F₁ = 0.654, representing a 17% F₁ improvement over the ALIAS baseline. Per-element analysis shows perfect precision (P = 1.00) for Si, Al, Fe, Li, Co, and Ni, while Mn and Na remain intractable false-positive sources across all methods. These results establish quantitative performance baselines and demonstrate that two-stage architectures combining global spectral fitting with peak-level confirmation are the current best approach at low resolving power.

*Keywords* — LIBS, calibration-free, element identification, spectral unmixing, NNLS, mineral analysis, peak-matching, resolving power

---

## I. Introduction

Laser-Induced Breakdown Spectroscopy (LIBS) generates atomic emission spectra from laser-produced microplasmas, enabling rapid elemental analysis without sample preparation [1], [2]. The Calibration-Free (CF-LIBS) methodology, introduced by Ciucci *et al.* [3] and reviewed by Tognoni *et al.* [4], eliminates the need for matrix-matched reference standards by modeling plasma emission from first principles using Saha-Boltzmann equilibrium and the closure equation (Σ Cₛ = 1). This approach has proven especially valuable in remote sensing contexts such as the ChemCam and SuperCam instruments on NASA's Mars rovers [5]–[7], where calibration standards are unavailable.

A critical prerequisite for CF-LIBS analysis is the identification of which elements are present — spectral lines must be correctly assigned to their emitting species before Boltzmann plots, temperature determination, and concentration calculation can proceed. At high resolving powers (RP > 5000), this identification problem is comparatively tractable: spectral lines are well-resolved, and algorithms such as ALIAS can achieve high accuracy by matching observed peak positions against theoretical wavelengths from atomic databases [8]. However, at the low resolving powers typical of compact spectrometers (RP = 300–1000), severe spectral overlap, line blending, and the density of emission features render peak-matching unreliable. This is precisely the regime of interest for field-deployed and handheld LIBS instruments.

The Mars LIBS instruments illustrate the state of the art. ChemCam and SuperCam employ multivariate regression (PLS, ICA) trained on libraries of hundreds of geological standards, achieving robust quantification of eight major elements [5], [7], [9]. However, these approaches require large calibration databases and are limited to the compositional space covered by the training set. A calibration-free identifier that works reliably at low RP would enable analysis of truly unknown samples.

In this work, we evaluate five distinct algorithmic pathways for element identification on 74 mineral and elemental LIBS spectra at RP = 300–1100. Our goals are to determine which approach maximizes identification precision while maintaining adequate recall, and to establish quantitative performance baselines for future CF-LIBS pipeline design.

---

## II. Methods

### A. Benchmark Dataset

The benchmark comprises 74 LIBS spectra from the Aalto University mineral spectral library [10], consisting of:

- **13 pure element spectra** (Al, Co, Cr, Cu, Fe, Mg, Mn, Ni, Pb, Sn, Ti, V, Zn) measured on high-purity metallic targets, each with single-element ground truth.
- **61 mineral spectra** covering 45 mineral species. The ground truth for each mineral is the set of metallic/semi-metallic major elements from its stoichiometric formula, restricted to a 22-element search list. Light elements (O, H, C, N, S, F, Cl, B) are excluded as they produce weak or ambiguous LIBS emission in air.

The effective resolving power was estimated from isolated peak FWHM measurements, yielding RP = 300–1100 (median ≈ 600). All spectra cover 200–900 nm.

> **[Fig. 5]** *Distribution of effective resolving power across 74 Aalto LIBS spectra. The median RP ≈ 600 places the benchmark firmly in the low-resolution regime where peak-matching is expected to be unreliable.*

The element search list contains 22 elements: Fe, Ca, Mg, Si, Al, Ti, Na, K, Mn, Cr, Ni, Cu, Co, V, Li, Sr, Ba, Zn, Pb, Mo, Zr, Sn — encompassing the major rock-forming and common trace elements in geological samples.

### B. Scoring Methodology

Each algorithm produces a set of detected elements per spectrum. Scoring follows standard information retrieval metrics computed against the ground-truth element set: True Positive (TP), False Positive (FP), False Negative (FN), and True Negative (TN). Precision, recall, F₁, and false positive rate (FPR) are computed globally (micro-averaged) across all 74 spectra. An "exact match" requires the detected set to equal the expected set exactly.

### C. Basis Library Generation

Pathways requiring basis spectra (NNLS, hybrid, forward model) share a common pre-computed basis library generated using Saha-Boltzmann physics [3], [11].

**TABLE I.** Basis library generation parameters.

| Parameter | Value |
|-----------|-------|
| Wavelength range | 200–900 nm |
| Wavelength pixels | 4,096 |
| Instrument FWHM | 0.5 nm (matched to RP ≈ 600–1000) |
| Temperature grid | 4,000–12,000 K, 30 linear steps |
| Electron density grid | 10¹⁵–5 × 10¹⁷ cm⁻³, 10 logarithmic steps |
| Ionization stages | I, II |
| Grid points per element | 300 |
| Elements | 76 (all available in database) |

For each element and grid point (*T*, *nₑ*), the ionization balance is solved via the Saha equation, level populations are computed from the Boltzmann distribution, and emission spectra are constructed from atomic transition parameters (*A*ₖᵢ, *g*ₖ, *E*ₖ). Each spectrum is convolved with a Gaussian instrument profile (FWHM = 0.5 nm) and area-normalized. Partition functions use polynomial fits to the NIST compilation. The library is stored in HDF5 format.

### D. Identification Pathways

#### 1) ALIAS Peak-Matching (Baseline)

The ALIAS algorithm [8] identifies elements by matching detected experimental peaks against theoretical line positions. For each candidate element, the algorithm: (i) detects peaks via second-derivative enhancement; (ii) auto-calibrates the wavelength axis; (iii) estimates plasma temperature from Boltzmann plot fitting; (iv) screens elements by strength-weighted match rate; and (v) scores each element by an independent confidence level (CL) combining match quality, Boltzmann consistency, and statistical significance against chance coincidence. An element is detected if its CL exceeds the detection threshold. Swept parameters: detection threshold ∈ {0.02, 0.03, 0.05}, intensity threshold factor ∈ {3.0, 3.5}, chance window scale ∈ {0.3, 0.4}.

#### 2) Full-Spectrum NNLS Decomposition

Inspired by spectral unmixing approaches [12], [13], this pathway decomposes the observed spectrum as a non-negative linear combination of single-element basis spectra:

*I*_obs(λ) ≈ Σᵢ *c*ᵢ · *B*ᵢ(λ; *T*, *n*ₑ) + *P*(λ)

where *B*ᵢ are the basis spectra at estimated (*T*, *n*ₑ), *c*ᵢ ≥ 0 are element coefficients enforced by the NNLS constraint [13], and *P*(λ) is a polynomial continuum. An element is detected if its coefficient SNR exceeds a threshold. Swept parameters: detection SNR ∈ {1.0, 1.5, 2.0, 2.5, 3.0}, continuum degree ∈ {2, 3, 4}, fallback *T* ∈ {6000, 8000, 10000} K.

#### 3) Hybrid NNLS+ALIAS (Two-Stage)

This novel approach combines the complementary strengths of NNLS (global spectral context, robust to blending) and ALIAS (individual line validation). The architecture is inspired by ChemCam/SuperCam's two-stage pipeline [5], [7] and hierarchical spectral discrimination [14].

> **[Fig. 4]** *Block diagram of the two-stage hybrid architecture. Stage 1: NNLS decomposes the full spectrum into element basis spectra with a lenient SNR threshold, producing a candidate set. Stage 2: ALIAS peak-matching is restricted to NNLS candidates, validating line-level evidence. In intersection mode, an element must pass both stages.*

**Stage 1 (NNLS Screening)**: Full-spectrum NNLS decomposition with lenient SNR threshold (≥ 1.0–2.0) produces a candidate element set. This stage has high recall but admits false positives.

**Stage 2 (ALIAS Confirmation)**: ALIAS peak-matching restricted to NNLS candidates validates that detected elements have corresponding peak-level evidence.

In **intersection mode**, an element must pass both stages. In **union mode**, an element passes if detected by either stage. Swept parameters: NNLS SNR ∈ {1.0, 1.5, 2.0}, ALIAS detection threshold ∈ {0.03, 0.05, 0.10}, mode ∈ {intersect, union}.

#### 4) Voigt Deconvolution + ALIAS

This pathway pre-processes the spectrum with multi-peak Voigt deconvolution [15], [16] before running ALIAS: (i) baseline subtraction; (ii) peak detection; (iii) grouping of nearby peaks within 2 × FWHM; (iv) multi-peak Voigt fitting per group; (v) spectrum reconstruction from fitted profiles; (vi) ALIAS identification on the cleaned spectrum. Swept parameters: detection threshold ∈ {0.03, 0.05}.

#### 5) Forward-Model Concentration Thresholding

NNLS decomposition with a concentration-based detection criterion: each element's coefficient is normalized to a fractional concentration (*c*ᵢ / Σ*c*ⱼ). Elements above a threshold are detected. Swept parameters: concentration threshold ∈ {0.001, 0.005, 0.01, 0.02, 0.05}, continuum degree ∈ {2, 3}.

### E. Computational Complexity

The computational cost of each pathway scales differently with the number of elements *E*, detected peaks *P*, spectral pixels *N*, basis library elements *M*, peak groups *G*, and peaks per group *K*:

- **ALIAS**: *O*(*E* · *P*) — each element is independently scored against each detected peak.
- **NNLS**: *O*(*N* · *M*²) — dominated by the NNLS active-set algorithm solving a (*N* × *M*) system.
- **Hybrid**: *O*(*N* · *M*² + *E* · *P*) — NNLS stage plus ALIAS stage.
- **Voigt+ALIAS**: *O*(*G* · *K*³ + *E* · *P*) — dominated by nonlinear least-squares fitting of multi-peak Voigt profiles within each group.
- **Forward model**: *O*(*N* · *M*²) — identical to NNLS with post-hoc thresholding.

For the present benchmark (*N* = 4096, *M* = 76, *E* = 22, *P* ≈ 50–100, *G* ≈ 20–40, *K* ≈ 2–5), ALIAS is fastest (0.14 s/spectrum), NNLS and forward model are comparable (0.28 s), hybrid incurs modest overhead (0.39 s), and Voigt deconvolution is an order of magnitude slower (6.8 s) due to repeated nonlinear optimization.

---

## III. Results

### A. Pathway Comparison

Table II presents the best configuration for each pathway, ranked by F₁ score.

**TABLE II.** Best performance per pathway on 74 Aalto LIBS spectra (RP = 300–1100).

| Rank | Pathway | Configuration | *P* | *R* | *F*₁ | FPR | Exact |
|------|---------|--------------|------|------|-------|-------|-------|
| 1 | Hybrid (intersect) | nsnr=1.5, adt=0.05 | **0.604** | 0.713 | **0.654** | 0.053 | **16/74** |
| 2 | ALIAS | dt=0.05, itf=3.0 | 0.505 | 0.629 | 0.560 | 0.070 | 11/74 |
| 3 | Voigt+ALIAS | dt=0.03 | 0.488 | 0.623 | 0.547 | 0.075 | 8/74 |
| 4 | Forward model | ct=0.05 | 0.369 | 0.796 | 0.505 | 0.148 | 0/74 |
| 5 | Spectral NNLS | snr=3.0, T=10kK | 0.293 | **0.940** | 0.447 | 0.236 | 2/74 |

> **[Fig. 1]** *Grouped bar chart comparing precision, recall, F₁, and false positive rate across the five identification pathways (best configuration each). The hybrid NNLS+ALIAS intersection method achieves the best F₁ and the lowest FPR. Spectral NNLS achieves the highest recall but at the cost of unacceptable precision.*

Key observations:

1. **NNLS has near-perfect recall (*R* = 0.94) but unacceptable precision (*P* = 0.29)**: It detects nearly every true element but also flags 7–8 spurious elements per spectrum. False positive elements include O, Na, V, Mg, K, Pb, and Hg — elements with many lines in the 200–900 nm range.

2. **ALIAS has moderate precision (*P* = 0.50) and moderate recall (*R* = 0.63)**: Precision is limited by Mn (30 FP), Na (20 FP), and Mg (15 FP). Recall is limited by Fe (15 FN), Si (17 FN), and Ca (14 FN).

3. **The hybrid intersection effectively gates NNLS false positives**: The ALIAS confirmation suppresses O, H, N, and rare-earth false positives that plague pure NNLS. Mn and Na false positives persist because these elements pass both stages.

4. **Voigt deconvolution provides no net benefit**: At RP < 1000, peak overlap is so severe that the Voigt fitting problem is underdetermined for groups of 3+ peaks.

5. **Forward-model concentration thresholding cannot distinguish trace from absent**: Line-rich database elements (Mn, V, W, Na) receive non-negligible concentrations even when absent.

### B. Per-Element Analysis

Table III presents the per-element precision and recall for the best hybrid and ALIAS configurations.

**TABLE III.** Per-element identification performance on 74 spectra.

| Element | Hybrid *P* | Hybrid *R* | ALIAS *P* | ALIAS *R* |
|---------|----------|----------|---------|---------|
| Si | **1.00** | 0.95 | 0.96 | 0.60 |
| Al | **1.00** | 0.90 | 0.76 | 0.76 |
| Fe | **1.00** | 0.44 | 0.91 | 0.40 |
| Li | **1.00** | 1.00 | 1.00 | 1.00 |
| Co | **1.00** | 1.00 | 0.33 | 1.00 |
| Ni | **1.00** | 0.67 | 0.50 | 1.00 |
| K | 0.67 | 0.60 | 0.54 | 0.70 |
| Ca | 0.62 | 0.40 | 0.46 | 0.30 |
| Ti | 0.50 | 1.00 | 0.67 | 1.00 |
| Mg | 0.39 | 0.81 | 0.50 | 0.94 |
| Pb | 0.40 | 1.00 | 0.33 | 1.00 |
| Na | 0.14 | 0.80 | 0.17 | 0.80 |
| Mn | 0.03 | 1.00 | 0.03 | 1.00 |
| Zn | 0.00 | 0.00 | 0.00 | 0.00 |
| Zr | 0.00 | 0.00 | 0.00 | 0.00 |

> **[Fig. 3]** *Per-element precision heatmap across four identification pathways (ALIAS, NNLS, Hybrid, Forward Model). Elements are ordered by hybrid precision (descending). The hybrid method achieves P = 1.00 for Si, Al, Fe, Li, Co, and Ni. Mn and Na are pathological false-positive sources across all methods.*

The hybrid method achieves **perfect precision (*P* = 1.00)** for Si, Al, Fe, Li, Co, and Ni — together accounting for 48% of all true positive detections. The NNLS stage provides global context to suppress false positives, while the ALIAS stage provides line-level confirmation.

Mn and Na remain pathological false-positive sources. Mn has > 500 transitions in 200–900 nm, creating a near-continuous pseudo-emission spectrum. Na's D-lines (589.0/589.6 nm) are ubiquitous contaminants, and its sparse line list means any chance peak match near 589 nm triggers detection.

### C. Intersection vs. Union Mode

**TABLE IV.** Hybrid identifier behavior in intersection vs. union mode.

| Mode | *P* | *R* | *F*₁ | FPR | Exact |
|------|------|------|-------|-------|-------|
| Intersect | 0.604 | 0.713 | 0.654 | 0.053 | 16/74 |
| Union | 0.308 | 0.952 | 0.466 | 0.244 | 1/74 |

> **[Fig. 2]** *Precision–recall scatter plot for 18 hybrid configurations. Intersection-mode configurations (circles) cluster at high precision/moderate recall; union-mode configurations (triangles) cluster at high recall/low precision. The Pareto front is marked, with the optimal F₁ point (star) at P = 0.604, R = 0.713. Iso-F₁ contours at F₁ = 0.4, 0.5, 0.6, and 0.7 are shown as dashed curves.*

Intersection mode sacrifices 24 percentage points of recall for 30 percentage points of precision. For CF-LIBS applications, precision is generally more important: a false positive element corrupts the Boltzmann plot, introduces errors in the closure equation, and cascades into incorrect concentrations for all elements.

### D. Sensitivity to Basis Library Temperature

**TABLE V.** NNLS performance vs. fallback temperature.

| *T* (K) | Best *P* | Best *R* | Best *F*₁ |
|---------|----------|----------|-----------|
| 6,000 | 0.272 | 0.868 | 0.414 |
| 8,000 | 0.282 | 0.922 | 0.431 |
| 10,000 | 0.293 | 0.940 | 0.447 |

Higher fallback temperature consistently improves performance, suggesting that the Aalto spectra correspond to relatively hot plasmas (*T* > 8000 K), consistent with high laser fluences used in mineral analysis.

---

## IV. Discussion

### A. The RP < 1000 Barrier

Our results quantitatively confirm the literature consensus that peak-matching approaches are fundamentally limited at RP < 1000. The best ALIAS configuration achieves *P* = 0.505, consistent with the expectation that at RP ≈ 600, a 200–900 nm window contains ~100 detectable peaks against ~5000 catalogued transitions, giving a chance coincidence rate of ~10% per element-line. Elements with only 2–3 detectable lines (Na, K) are particularly vulnerable.

The hybrid approach partially overcomes this barrier by requiring corroborating evidence from two orthogonal sources. However, the improvement is bounded by the weaker channel — ALIAS's limitations propagate into the hybrid result via the intersection operation. The 17% F₁ improvement (0.559 → 0.654) represents the maximum gain achievable by adding a global fitting stage to peak-matching confirmation, without improving the peak-matcher itself.

### B. Comparison with Mars LIBS Instruments

ChemCam operates at RP ≈ 2000–4000 across three channels (240–342, 382–469, 474–906 nm) [5]. Its PLS regression pipeline trained on 69–332 standards achieves robust quantification of SiO₂, TiO₂, Al₂O₃, FeOₜ, MgO, CaO, Na₂O, and K₂O [7]. Our CF-LIBS approach, operating at lower RP without calibration standards, cannot match ChemCam's quantitative performance. However, our hybrid identifier achieves *P* = 1.00 for Si and Al — the two most abundant rock-forming elements — essential for downstream concentration calculation.

MarSCoDe (RP ≈ 500–700) operates at resolving power comparable to our benchmark [9]. Its element identification relies on pre-selected spectral windows rather than blind identification. Our work establishes quantitative baselines for blind identification at comparable resolving power.

### C. The Mn and Na Problem

Manganese and sodium represent the dominant failure mode across all algorithms. **Mn**: With > 500 transitions in 200–900 nm, any sample with ≥ 5 detected peaks will have Mn "matches" at RP < 1000. The false positive rate is determined by peak count, not Mn presence. **Na**: The D-lines (589.0/589.6 nm) are ubiquitous contaminants, and Na's sparse line list means a single chance match suffices for detection. Both elements require RP-dependent treatment: exclusion from blind search or contaminant flagging at RP < 1000.

### D. Prospects for Machine Learning Enhancement

The per-element precision/recall trade-offs suggest that simple threshold rules (SNR, CL, concentration) are suboptimal decision boundaries. The hybrid identifier naturally produces a rich feature vector per element: NNLS coefficient, NNLS SNR, ALIAS confidence level, ALIAS match count, Boltzmann *R*², and spectral residual norm. These features could train an element-specific classifier (e.g., SVM, random forest, or gradient-boosted trees) to learn nonlinear decision boundaries that adapt to each element's spectral characteristics.

The basis library provides a natural data augmentation strategy: synthetic spectra with known compositions can be generated at arbitrary (*T*, *n*ₑ) conditions and noise levels, producing effectively unlimited labeled training data. A classifier trained on 10⁴–10⁵ synthetic spectra and fine-tuned on the 74 Aalto spectra could plausibly achieve *P* > 0.80 by learning, for example, that Mn detections at RP < 1000 should be suppressed unless the NNLS coefficient exceeds 3× the median, or that Na requires confirmation from both D-lines rather than either alone.

This approach preserves the physics-based interpretability of the CF-LIBS pipeline (the NNLS and ALIAS features are physically meaningful) while adding the discriminative power of learned classifiers. It represents the most promising path to exceeding the P ≈ 0.60 ceiling established in this work.

### E. Implications for CF-LIBS Pipeline Design

The hybrid identifier's performance suggests a recommended pipeline: (1) hybrid NNLS+ALIAS identification in intersection mode; (2) RP-dependent post-filtering to suppress Mn/Na at RP < 1000; (3) prioritization of high-confidence elements (Si, Al, Fe, Ca, Mg, K) for Boltzmann plot construction; (4) iterative refinement using CF-LIBS concentration results to prune elements below physical thresholds.

### F. Limitations and Future Work

**Higher FWHM basis libraries**: FWHM = 0.5 nm may not be optimal for the lowest-RP spectra (RP ≈ 300–500). **FAISS-indexed (*T*, *n*ₑ) estimation**: Data-driven parameter estimation could improve NNLS precision. **Mechelle 5000 benchmark**: At RP ≈ 5000, ALIAS should approach > 95% accuracy, establishing the upper bound on peak-matching. **Improved deconvolution**: Shoulder-detection and residual-completion approaches [16] may outperform standard curve fitting in severely blended regimes.

---

## V. Conclusions

We have presented the first systematic benchmark of five element identification algorithms for CF-LIBS at low resolving power (RP = 300–1100) on 74 mineral and elemental spectra. The principal findings are:

1. The hybrid NNLS+ALIAS identifier achieves the best overall performance (*P* = 0.604, *R* = 0.713, *F*₁ = 0.654, 16/74 exact matches), representing a 17% *F*₁ improvement over ALIAS.

2. Full-spectrum NNLS has excellent recall (*R* = 0.94) but poor precision (*P* = 0.29), insufficient for reliable identification alone.

3. Peak-matching precision is fundamentally limited to *P* ≈ 0.50 at RP < 1000 by chance coincidence rates.

4. Voigt deconvolution provides no net benefit at RP < 1000 because the deconvolution problem is underdetermined.

5. The hybrid method achieves *P* = 1.00 for Si, Al, Fe, Li, Co, and Ni — nearly half of all true positive detections.

6. Reaching *P* > 80% at RP < 1000 requires machine learning classifiers or higher-RP instrumentation (RP > 3000).

---

## Appendix A: Full Hybrid Sweep

**TABLE VI.** Complete 18-configuration hybrid sweep results.

| # | NNLS SNR | ALIAS *d*_t | Mode | *P* | *R* | *F*₁ | FPR | Exact |
|---|----------|------------|------|------|------|-------|-------|-------|
| 1 | 1.0 | 0.03 | ∩ | 0.557 | 0.737 | 0.634 | 0.067 | 14/74 |
| 2 | 1.0 | 0.03 | ∪ | 0.274 | 0.952 | 0.425 | 0.289 | 0/74 |
| 3 | 1.0 | 0.05 | ∩ | 0.594 | 0.719 | 0.650 | 0.056 | 16/74 |
| 4 | 1.0 | 0.05 | ∪ | 0.283 | 0.952 | 0.437 | 0.275 | 0/74 |
| 5 | 1.0 | 0.10 | ∩ | 0.649 | 0.521 | 0.578 | 0.032 | 17/74 |
| 6 | 1.0 | 0.10 | ∪ | 0.292 | 0.952 | 0.447 | 0.264 | 2/74 |
| 7 | 1.5 | 0.03 | ∩ | 0.565 | 0.731 | 0.637 | 0.064 | 16/74 |
| 8 | 1.5 | 0.03 | ∪ | 0.296 | 0.952 | 0.451 | 0.259 | 1/74 |
| **9** | **1.5** | **0.05** | **∩** | **0.604** | **0.713** | **0.654** | **0.053** | **16/74** |
| 10 | 1.5 | 0.05 | ∪ | 0.308 | 0.952 | 0.466 | 0.244 | 1/74 |
| 11 | 1.5 | 0.10 | ∩ | 0.659 | 0.521 | 0.582 | 0.031 | 17/74 |
| 12 | 1.5 | 0.10 | ∪ | 0.319 | 0.952 | 0.477 | 0.233 | 3/74 |
| 13 | 2.0 | 0.03 | ∩ | 0.568 | 0.725 | 0.637 | 0.063 | 15/74 |
| 14 | 2.0 | 0.03 | ∪ | 0.310 | 0.940 | 0.467 | 0.239 | 1/74 |
| 15 | 2.0 | 0.05 | ∩ | 0.602 | 0.707 | 0.650 | 0.053 | 15/74 |
| 16 | 2.0 | 0.05 | ∪ | 0.326 | 0.940 | 0.484 | 0.222 | 1/74 |
| 17 | 2.0 | 0.10 | ∩ | 0.656 | 0.515 | 0.577 | 0.031 | 17/74 |
| 18 | 2.0 | 0.10 | ∪ | 0.339 | 0.940 | 0.498 | 0.209 | 3/74 |

∩ = intersection mode; ∪ = union mode. **Bold** = optimal *F*₁.

## Appendix B: Computational Performance

**TABLE VII.** Wall-clock timing on Apple M-series (single core).

| Pathway | Time/spectrum (s) | Total, 74 spectra (s) |
|---------|------------------|-----------------------|
| ALIAS | 0.14 | 10.7 |
| Spectral NNLS | 0.28 | 20.5 |
| Hybrid (∩) | 0.39 | 28.6 |
| Forward model | 0.27 | 20.3 |
| Voigt+ALIAS | 6.8 | 500 |

Basis library generation: 244 s for 76 elements × 300 grid points × 4096 pixels.

---

## References

[1] D. A. Cremers and L. J. Radziemski, *Handbook of Laser-Induced Breakdown Spectroscopy*. Hoboken, NJ, USA: Wiley, 2006.

[2] T. Kim and C. Lin, "Laser-induced breakdown spectroscopy," *Nature Reviews Methods Primers*, vol. 5, 2025.

[3] A. Ciucci, M. Corsi, V. Palleschi, S. Rastelli, A. Salvetti, and E. Tognoni, "New procedure for quantitative elemental analysis by laser-induced plasma spectroscopy," *Appl. Spectrosc.*, vol. 53, no. 8, pp. 960–964, 1999.

[4] E. Tognoni, G. Cristoforetti, S. Legnaioli, and V. Palleschi, "Calibration-free laser-induced breakdown spectroscopy: State of the art," *Spectrochim. Acta Part B*, vol. 65, no. 1, pp. 1–14, 2010.

[5] R. C. Wiens, S. Maurice, J. Lasue *et al.*, "Pre-flight calibration and initial data processing for the ChemCam laser-induced breakdown spectroscopy instrument on the Mars Science Laboratory rover," *Spectrochim. Acta Part B*, vol. 82, pp. 1–27, 2013.

[6] R. C. Wiens, S. Maurice, S. H. Robinson *et al.*, "The SuperCam instrument suite on the NASA Mars 2020 rover: Body unit and combined system tests," *Space Sci. Rev.*, vol. 217, 2020.

[7] S. M. Clegg, R. C. Wiens, R. B. Anderson *et al.*, "Recalibration of the Mars Science Laboratory ChemCam instrument with an expanded geochemical database," *Spectrochim. Acta Part B*, vol. 129, pp. 64–85, 2017.

[8] T. A. Labutin, S. M. Zaytsev, and A. M. Popov, "Automatic identification of emission lines in laser-induced plasma by correlation of model and experimental spectra," *Anal. Chem.*, vol. 85, no. 4, pp. 1985–1990, 2013.

[9] R. C. Wiens, A. Cousin, S. M. Clegg *et al.*, "Geochemistry of Mars with laser-induced breakdown spectroscopy (LIBS): ChemCam, SuperCam, and MarSCoDe," *Minerals*, 2025.

[10] I. Drozdovskiy, G. Ligeza, P. Jahoda *et al.*, "The PANGAEA mineralogical database," *Data in Brief*, vol. 31, 2020.

[11] T. Fujimoto and R. W. P. McWhirter, "Validity criteria for local thermodynamic equilibrium in plasma spectroscopy," *Phys. Rev. A*, vol. 42, no. 11, pp. 6588–6601, 1990.

[12] W. Wang, B. Ayhan, C. Kwan, H. Qi, and S. Vance, "A novel and effective multivariate method for compositional analysis using laser induced breakdown spectroscopy," *IOP Conf. Ser.: Earth Environ. Sci.*, vol. 17, p. 012208, 2014.

[13] C. L. Lawson and R. J. Hanson, *Solving Least Squares Problems*. Englewood Cliffs, NJ, USA: Prentice-Hall, 1974. (Reprinted: SIAM Classics Appl. Math., vol. 15, 1995.)

[14] C. Eum, D. Jang, S. Lee, K. Cha, and H. Chung, "Alternative selection of Raman or LIBS spectral information in hierarchical discrimination of raw sapphires according to geographical origin," *Talanta*, vol. 221, p. 121555, 2021.

[15] M. Al-Jalali, I. F. Aljghami, and Y. Mahzia, "Voigt deconvolution method and its applications to pure oxygen absorption spectrum at 1270 nm band," *Spectrochim. Acta Part A*, vol. 157, pp. 34–40, 2016.

[16] P. Dai, P. Zheng, J. Wang, G. Chen, L. Li, and L. Guo, "From spectral interference to overlapped features: A full-spectrum LIBS Voigt decomposition strategy with shoulder detection and residual completion," *Anal. Chem.*, 2026.

[17] G. Cristoforetti, A. De Giacomo, M. Dell'Aglio *et al.*, "Local thermodynamic equilibrium in laser-induced breakdown spectroscopy: Beyond the McWhirter criterion," *Spectrochim. Acta Part B*, vol. 65, no. 1, pp. 86–95, 2010.

[18] H. R. Griem, *Principles of Plasma Spectroscopy*. Cambridge, U.K.: Cambridge Univ. Press, 1997.

[19] R. S. Harmon, R. E. Russo, and R. R. Hark, "Applications of laser-induced breakdown spectroscopy for geochemical and environmental analysis: A comprehensive review," *Spectrochim. Acta Part B*, vol. 87, pp. 11–26, 2013.

[20] L. C. L. Borduchi, D. Milori, and P. Villas-Boas, "One-point calibration of Saha-Boltzmann plot to improve accuracy and precision of quantitative analysis using laser-induced breakdown spectroscopy," *Spectrochim. Acta Part B*, 2019.

[21] S. Michelena, M. Ferreira Da Costa, and J. Picheral, "Convergence guarantees for unmixing PSFs over a manifold with non-convex optimization," in *Proc. IEEE Stat. Signal Process. Workshop (SSP)*, 2025, pp. 161–165.

[22] H. Xu, X. Huang, X. Le, X. Zhong, Z. Lihua, and S. Jia, "Spectral lines overlap in vacuum arc plasmas of CuCr electrodes," in *Proc. 31st Int. Symp. Discharges Elect. Insulation Vacuum (ISDEIV)*, 2025, pp. 1–3.

[23] K. Shameem, K. S. Choudhari, A. Bankapur *et al.*, "A hybrid LIBS-Raman system combined with chemometrics: An efficient tool for plastic identification and sorting," *Anal. Bioanal. Chem.*, vol. 409, pp. 3299–3308, 2017.

[24] V. Palleschi, S. Legnaioli, F. Poggialini *et al.*, "Laser-induced breakdown spectroscopy," *Nature Reviews Methods Primers*, vol. 5, 2025.
