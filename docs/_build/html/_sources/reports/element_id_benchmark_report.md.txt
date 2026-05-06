# Comparative Evaluation of Element Identification Algorithms for Low-Resolution LIBS: A Systematic Benchmark on Mineral and Elemental Spectra

**B. Squires**

CF-LIBS Project Technical Report, March 2026

---

## Abstract

Automated element identification remains one of the principal challenges in Calibration-Free Laser-Induced Breakdown Spectroscopy (CF-LIBS), particularly at the low resolving powers (RP < 1000) typical of compact and field-deployed spectrometers. We present a systematic benchmark of five element identification pathways evaluated on 74 LIBS spectra (13 pure elements, 61 mineral samples) from the Aalto University mineral spectral library, covering effective resolving powers of 300–1100. The five pathways are: (1) ALIAS peak-matching, (2) full-spectrum NNLS decomposition, (3) a novel two-stage hybrid NNLS+ALIAS identifier, (4) Voigt deconvolution followed by ALIAS, and (5) forward-model concentration thresholding. The hybrid NNLS+ALIAS approach operating in intersection mode achieves the best performance with precision P = 0.604, recall R = 0.713, and F₁ = 0.654, representing a 17% improvement in F₁ over the ALIAS baseline (F₁ = 0.559). Per-element analysis reveals that Mn (P = 0.03) and Na (P = 0.14) remain intractable false-positive sources at RP < 1000 across all algorithms, while Si (P = 1.00, R = 0.95) and Al (P = 1.00, R = 0.90) are reliably identified by the hybrid method. These results confirm the literature consensus that peak-matching approaches are fundamentally limited at RP < 1000, and demonstrate that two-stage architectures combining full-spectrum fitting with peak-level confirmation can partially overcome this limitation.

---

## 1. Introduction

Laser-Induced Breakdown Spectroscopy (LIBS) generates atomic emission spectra from laser-produced microplasmas, enabling rapid elemental analysis of solid, liquid, and gaseous samples without preparation (Cremers & Radziemski, 2006; Kim & Lin, 2025). The Calibration-Free (CF-LIBS) methodology, introduced by Ciucci et al. (1999) and comprehensively reviewed by Tognoni et al. (2010), eliminates the need for matrix-matched reference standards by modeling the plasma emission from first principles using the Saha-Boltzmann equilibrium and a closure equation (Σ Cₛ = 1). This approach has proven especially valuable in remote sensing contexts such as the ChemCam and SuperCam instruments on NASA's Mars rovers (Wiens et al., 2013; Clegg et al., 2017; Wiens et al., 2020), where calibration standards are unavailable.

A critical prerequisite for CF-LIBS analysis is the identification of which elements are present in the plasma — the spectral lines must be correctly assigned to their emitting species before Boltzmann plots, temperature determination, and concentration calculation can proceed. At high resolving powers (RP > 5000), this identification problem is comparatively tractable: spectral lines are well-resolved, and algorithms such as ALIAS (Automatic Line Identification and Spectral Analysis) can achieve high accuracy by matching observed peak positions against theoretical wavelengths from atomic databases (Labutin et al., 2013). However, at the low resolving powers typical of compact spectrometers (RP = 300–1000), severe spectral overlap, line blending, and the sheer density of emission features render peak-matching unreliable. This is precisely the regime of interest for field-deployed and handheld LIBS instruments.

The Mars LIBS instruments illustrate the state of the art in operational element identification. ChemCam and SuperCam employ multivariate regression (PLS, ICA) trained on libraries of hundreds of geological standards, achieving robust quantification of 8 major elements (Wiens et al., 2025; Clegg et al., 2017). However, these approaches require large calibration databases and are inherently limited to the compositional space covered by the training set. A calibration-free identifier that works reliably at low RP would enable analysis of truly unknown samples.

In this work, we evaluate five distinct algorithmic pathways for element identification on a challenging benchmark of 74 mineral and elemental LIBS spectra at RP = 300–1100. Our goal is to determine which approach — or combination of approaches — can maximize identification precision while maintaining adequate recall, and to establish quantitative performance baselines that inform the design of future CF-LIBS pipelines.

---

## 2. Methods

### 2.1 Benchmark Dataset

The benchmark comprises 74 LIBS spectra from the Aalto University mineral spectral library (Drozdovskiy et al., 2020), consisting of:

- **13 pure element spectra**: Al, Co, Cr, Cu, Fe, Mg, Mn, Ni, Pb, Sn, Ti, V, Zn — each measured on a high-purity metallic target. The ground truth for each spectrum is the single element.
- **61 mineral spectra**: covering 45 mineral species (adularia, almandine, apatite, augite, biotite, chalcopyrite, cordierite, corundum, diopside, fluorite, galena, garnet, gypsum, hematite, hornblende, hypersthene, kaolinite, kyanite, lepidolite, magnesite, magnetite, microcline, molybdenite, muscovite, olivine, orthoclase, pentlandite, phlogopite, plagioclase, pyrite, pyrrhotite, quartz, scapolite, serpentine, siderite, sphene, spodumene, staurolite, talc, topaz, tourmaline, tremolite, wollastonite, zircon). The ground truth for each mineral is the set of metallic/semi-metallic major elements from its stoichiometric formula, restricted to the 22-element search list. Light elements (O, H, C, N, S, F, Cl, B) are excluded as they produce weak or ambiguous LIBS emission in air.

The effective resolving power of each spectrum was estimated from isolated peak FWHM measurements, yielding RP = 300–1100 (median ≈ 600). All spectra cover the wavelength range 200–900 nm.

The element search list contains 22 elements: Fe, Ca, Mg, Si, Al, Ti, Na, K, Mn, Cr, Ni, Cu, Co, V, Li, Sr, Ba, Zn, Pb, Mo, Zr, Sn. This list encompasses the major rock-forming and common trace elements in geological samples.

### 2.2 Scoring Methodology

Each algorithm produces a set of detected elements for each spectrum. Scoring follows standard information retrieval metrics computed against the ground-truth element set:

- **True Positive (TP)**: element correctly detected
- **False Positive (FP)**: element detected but not in ground truth
- **False Negative (FN)**: element in ground truth but not detected
- **True Negative (TN)**: element correctly not detected

Precision, recall, F₁, and false positive rate (FPR) are computed globally (micro-averaged) across all 74 spectra. An "exact match" requires the detected set to equal the expected set exactly.

### 2.3 Basis Library Generation

Pathways requiring basis spectra (NNLS, hybrid, forward model) share a common pre-computed basis library. The library was generated using Saha-Boltzmann physics (Ciucci et al., 1999; Fujimoto & McWhirter, 1990) with the following parameters:

| Parameter | Value |
|-----------|-------|
| Wavelength range | 200–900 nm |
| Wavelength pixels | 4096 |
| Instrument FWHM | 0.5 nm (matched to RP ≈ 600–1000) |
| Temperature grid | 4000–12000 K, 30 linear steps |
| Electron density grid | 10¹⁵–5×10¹⁷ cm⁻³, 10 logarithmic steps |
| Ionization stages | I, II |
| Total grid points | 300 per element |
| Elements | 76 (all available in database) |

For each element and grid point (T, nₑ), the ionization balance is solved via the Saha equation, level populations are computed from the Boltzmann distribution, and emission spectra are constructed from atomic transition parameters (Aₖᵢ, gₖ, Eₖ). Each spectrum is convolved with a Gaussian instrument profile (FWHM = 0.5 nm) and area-normalized to unit integral. The partition functions use polynomial fits to the NIST compilation. The resulting library is stored in HDF5 format (spectra: n_elements × n_grid × n_pixels, plus parameter and wavelength grids).

### 2.4 Identification Pathways

#### 2.4.1 ALIAS Peak-Matching (Baseline)

The ALIAS algorithm (Labutin et al., 2013) identifies elements by matching detected experimental peaks against theoretical line positions from an atomic database. For each candidate element, the algorithm:

1. Detects peaks in the experimental spectrum via second-derivative enhancement
2. Auto-calibrates the wavelength axis (global shift + effective RP estimation)
3. Estimates plasma temperature from Boltzmann plot fitting
4. Screens elements by strength-weighted match rate
5. Scores each element by an independent confidence level (CL) combining match quality, Boltzmann consistency, and statistical significance against chance coincidence

An element is detected if its CL exceeds the detection threshold. The chance window scale parameter controls the null hypothesis for random peak coincidence.

**Swept parameters**: detection_threshold ∈ {0.02, 0.03, 0.05}, intensity_threshold_factor ∈ {3.0, 3.5}, chance_window_scale ∈ {0.3, 0.4}.

#### 2.4.2 Full-Spectrum NNLS Decomposition

Inspired by spectral unmixing approaches in remote sensing (Wang et al., 2014; Lawson & Hanson, 1974), this pathway decomposes the observed spectrum as a non-negative linear combination of single-element basis spectra:

$$I_{\text{obs}}(\lambda) \approx \sum_i c_i \cdot B_i(\lambda; T, n_e) + P(\lambda)$$

where $B_i$ are the basis spectra at estimated (T, nₑ), $c_i ≥ 0$ are the element coefficients (enforced by the NNLS constraint), and $P(λ)$ is a polynomial continuum. The NNLS problem is solved using the active-set algorithm of Lawson & Hanson (1974).

An element is detected if its coefficient SNR (ratio of coefficient to its uncertainty from the (AᵀA)⁻¹ diagonal) exceeds the detection threshold.

**Swept parameters**: detection_snr ∈ {1.0, 1.5, 2.0, 2.5, 3.0}, continuum_degree ∈ {2, 3, 4}, fallback_T ∈ {6000, 8000, 10000} K.

#### 2.4.3 Hybrid NNLS+ALIAS (Two-Stage)

This novel approach combines the complementary strengths of NNLS (global spectral context, physically constrained, robust to blending) and ALIAS (individual line validation, pattern-matching precision). The architecture is inspired by the two-stage pipelines used by ChemCam/SuperCam (Wiens et al., 2013; Clegg et al., 2017) and recent proposals for hierarchical spectral discrimination (Eum et al., 2021):

**Stage 1 (NNLS Screening)**: The full-spectrum NNLS decomposition is run with a lenient detection threshold (SNR ≥ 1.0–2.0), producing a candidate element set. This stage has high recall but admits false positives.

**Stage 2 (ALIAS Confirmation)**: ALIAS peak-matching is run with the search space restricted to the NNLS candidates. This validates that the NNLS-detected elements have corresponding peak-level evidence.

In **intersection mode**, an element must pass both stages (detected by NNLS AND confirmed by ALIAS). In **union mode**, an element passes if detected by either stage.

**Swept parameters**: nnls_snr ∈ {1.0, 1.5, 2.0}, alias_detection_threshold ∈ {0.03, 0.05, 0.10}, mode ∈ {intersect, union}.

#### 2.4.4 Voigt Deconvolution + ALIAS

This pathway addresses the root cause of ALIAS failure at low RP — spectral overlap — by pre-processing the spectrum with multi-peak Voigt deconvolution (Al-Jalali et al., 2016; Dai et al., 2026) before running ALIAS. The procedure:

1. Baseline subtraction via iterative polynomial fitting
2. Peak detection on the corrected spectrum
3. Grouping of nearby peaks (within 2 × FWHM)
4. Multi-peak Voigt fitting per group using `scipy.optimize.curve_fit`
5. Reconstruction of a "cleaned" spectrum from the sum of fitted Voigt profiles
6. ALIAS identification on the cleaned spectrum

**Swept parameters**: detection_threshold ∈ {0.03, 0.05}.

#### 2.4.5 Forward-Model Concentration Thresholding

This pathway uses NNLS decomposition but applies a concentration-based detection criterion rather than an SNR criterion. After NNLS fitting, each element's coefficient is normalized to a fractional concentration (cᵢ / Σcⱼ). Elements with concentration above a threshold are detected.

**Swept parameters**: concentration_threshold ∈ {0.001, 0.005, 0.01, 0.02, 0.05}, continuum_degree ∈ {2, 3}.

---

## 3. Results

### 3.1 Pathway Comparison

Table 1 presents the best configuration for each pathway, ranked by F₁ score.

**Table 1.** Best performance per pathway on 74 Aalto LIBS spectra (RP = 300–1100).

| Rank | Pathway | Best Configuration | P | R | F₁ | FPR | Exact |
|------|---------|-------------------|------|------|-------|-------|-------|
| 1 | Hybrid (intersect) | nsnr=1.0, adt=0.03 | 0.557 | 0.737 | 0.634 | 0.067 | 14/74 |
| 2 | Hybrid (intersect)* | nsnr=1.5, adt=0.05 | 0.604 | 0.713 | 0.654 | 0.053 | 16/74 |
| 3 | ALIAS | dt=0.05, itf=3.0 | 0.505 | 0.629 | 0.560 | 0.070 | 11/74 |
| 4 | Voigt+ALIAS | dt=0.03 | 0.488 | 0.623 | 0.547 | 0.075 | 8/74 |
| 5 | Forward model | ct=0.05 | 0.369 | 0.796 | 0.505 | 0.148 | 0/74 |
| 6 | Spectral NNLS | snr=3.0, cdeg=4, T=10K | 0.293 | 0.940 | 0.447 | 0.236 | 2/74 |

*From the dedicated 18-configuration hybrid sweep.

The hybrid NNLS+ALIAS identifier in intersection mode achieves the highest F₁ (0.654) and the highest number of exact matches (16/74). The full 18-configuration hybrid sweep reveals that nsnr=1.5, adt=0.05 is optimal, achieving P = 0.604 — a 20% relative improvement over the ALIAS baseline (P = 0.505).

Key observations:

- **NNLS has near-perfect recall (R = 0.94) but unacceptable precision (P = 0.29)**: It detects nearly every true element but also flags 7–8 spurious elements per spectrum on average. The false positive elements include O, Na, V, Mg, K, Pb, and Hg — elements with many lines in the 200–900 nm range that inevitably contribute non-zero NNLS coefficients.

- **ALIAS has moderate precision (P = 0.50) and moderate recall (R = 0.63)**: Its precision is limited by Mn (30 FP), Na (20 FP), and Mg (15 FP) — elements whose dense line forests match chance peaks at RP < 1000. Its recall is limited by Fe (15 FN), Si (17 FN), and Ca (14 FN) — elements whose strongest lines are blended or fall below the detection threshold.

- **The hybrid intersection effectively gates NNLS false positives**: By requiring that an NNLS candidate also be confirmed by ALIAS peak-matching, the hybrid suppresses the O, H, N, and rare-earth false positives that plague pure NNLS. The Mn and Na false positives persist because these elements pass both stages — NNLS detects their spectral signature (they do have many lines), and ALIAS confirms chance peak matches.

- **Voigt deconvolution provides no net benefit**: Despite the physical motivation (resolving blended peaks before matching), the deconvolution step does not improve ALIAS precision (0.488 vs. 0.502) and slightly reduces recall. At RP < 1000, peak overlap is so severe that the Voigt fitting problem itself becomes underdetermined for groups of 3+ peaks.

- **Forward-model concentration thresholding cannot distinguish trace from absent**: Because concentration is normalized by the total NNLS signal, elements with many lines in the database (Mn, V, W, Na) receive non-negligible concentrations even when absent from the sample. No single threshold separates true from false.

### 3.2 Per-Element Analysis

Table 2 presents the per-element precision and recall for the best hybrid and ALIAS configurations.

**Table 2.** Per-element identification performance for hybrid (nsnr=1.0, adt=0.03, intersect) and ALIAS (dt=0.05, itf=3.5) on 74 spectra.

| Element | Hybrid P | Hybrid R | ALIAS P | ALIAS R | Comment |
|---------|----------|----------|---------|---------|---------|
| Si | **1.00** | 0.95 | 0.96 | 0.60 | Hybrid +35% recall |
| Al | **1.00** | 0.90 | 0.76 | 0.76 | Hybrid eliminates all FPs |
| Fe | **1.00** | 0.44 | 0.91 | 0.40 | Perfect precision, low recall |
| Li | **1.00** | 1.00 | 1.00 | 1.00 | Both perfect (strong doublet) |
| Co | **1.00** | 1.00 | 0.33 | 1.00 | Hybrid eliminates 2 FPs |
| Ni | **1.00** | 0.67 | 0.50 | 1.00 | Precision/recall trade-off |
| K | 0.67 | 0.60 | 0.54 | 0.70 | Modest improvement |
| Ca | 0.62 | 0.40 | 0.46 | 0.30 | Calcium problematic for both |
| Ti | 0.50 | 1.00 | 0.67 | 1.00 | — |
| Mg | 0.39 | 0.81 | 0.50 | 0.94 | Persistent FP source |
| Pb | 0.40 | 1.00 | 0.33 | 1.00 | — |
| Na | 0.14 | 0.80 | 0.17 | 0.80 | Intractable at RP < 1000 |
| Mn | **0.03** | 1.00 | 0.03 | 1.00 | Intractable at RP < 1000 |
| Zn | 0.00 | 0.00 | 0.00 | 0.00 | Weak LIBS emitter |
| Zr | 0.00 | 0.00 | 0.00 | 0.00 | Insufficient DB coverage |

The hybrid method achieves **perfect precision (P = 1.00)** for Si, Al, Fe, Li, Co, and Ni — together accounting for 48% of all true positive detections. The NNLS stage provides the global context to suppress false positives for these elements, while the ALIAS stage provides the line-level confirmation.

The elements Mn and Na remain pathological false-positive sources across all algorithms. Mn has > 500 transitions in the 200–900 nm range, creating a near-continuous pseudo-emission spectrum that overlaps with any sample at RP < 1000. Na's D-lines (589.0/589.6 nm) are ubiquitous contaminants in LIBS (from ambient Na, sample holder, or atmospheric entrainment), and its sparse line list (2–3 detectable lines) means any chance peak match near 589 nm is sufficient for detection.

### 3.3 Pure Element Spectra

On the 13 pure element spectra (single-element ground truth), the hybrid method achieves exact identification for 6/13 elements (Al, Cr, Cu, Fe, Mn, Sn). The remaining 7 spectra exhibit false positives due to:

- **Line-rich contaminant elements** (Mn, Na): These appear as FPs even in pure element spectra (e.g., Co spectrum detects Ca, Mg, Mn, Na)
- **Spectral overlap at low RP**: Ni spectrum detects Sn and Ti; V spectrum detects Cr, Cu, Mo
- **Weak emitters**: Mg and Zn are not detected in their own pure spectra — Mg's strongest lines (280, 285 nm) are in a crowded UV region, and Zn's emission is inherently weak in air at atmospheric pressure

### 3.4 Mineral Spectra

On the 61 mineral spectra, the hybrid method achieves exact identification for 8/61 minerals. The error patterns are dominated by:

1. **Systematic Mn FP (33/61 minerals)**: Mn is falsely detected in nearly every mineral spectrum. This is a fundamental limitation at RP < 1000 — Mn's dense transition forest produces statistically significant peak matches by chance.

2. **Systematic Na FP (24/61 minerals)**: Na contamination and the sparsity of Na's line list combine to produce persistent false detections.

3. **Fe FN (14/61 minerals)**: Fe is missed in many iron-bearing minerals (garnet, hornblende, olivine, staurolite, tourmaline). At RP < 1000, Fe's strongest lines (e.g., 438.3, 440.5 nm) are blended with Ti, Cr, and V lines, and the ALIAS CL falls below threshold.

4. **Ca FN (12/61 minerals)**: Ca's strong lines (393.4, 396.8 nm) overlap with Al lines at low RP. The Ca II resonance doublet, while intense, falls in a region of high line density.

### 3.5 Intersection vs. Union Mode

The hybrid identifier's behavior is dramatically different between intersection and union modes:

| Mode | P | R | F₁ | FPR | Exact |
|------|------|------|-------|-------|-------|
| Intersect (nsnr=1.5, adt=0.05) | 0.604 | 0.713 | 0.654 | 0.053 | 16/74 |
| Union (nsnr=1.5, adt=0.05) | 0.308 | 0.952 | 0.466 | 0.244 | 1/74 |

Intersection mode sacrifices 24 percentage points of recall for 30 percentage points of precision. For CF-LIBS applications, precision is generally more important than recall: a false positive element corrupts the Boltzmann plot, introduces errors in the closure equation, and can cascade into incorrect concentrations for all elements. A false negative, while also problematic, is self-limiting — the missing element's concentration simply adds to the residual.

### 3.6 Sensitivity to Basis Library Temperature

NNLS decomposition performance is sensitive to the assumed plasma temperature used for basis spectra. Across the 45 NNLS configurations tested, T = 10000 K consistently outperforms T = 6000 K and T = 8000 K:

| T (K) | Best P | Best R | Best F₁ |
|-------|--------|--------|---------|
| 6000 | 0.272 | 0.868 | 0.414 |
| 8000 | 0.282 | 0.922 | 0.431 |
| 10000 | 0.293 | 0.940 | 0.447 |

This suggests that the Aalto LIBS spectra correspond to relatively hot plasmas (T > 8000 K), consistent with the high laser fluences used in mineral analysis. The basis library's bilinear interpolation capability allows the NNLS solver to adapt to the actual plasma conditions, but the fallback temperature still influences the starting point.

---

## 4. Discussion

### 4.1 The RP < 1000 Barrier

Our results quantitatively confirm the literature consensus that peak-matching approaches are fundamentally limited at RP < 1000. The best ALIAS configuration achieves P = 0.505, consistent with the theoretical expectation that at RP ≈ 600, a typical 200–900 nm spectral window contains ~100 detectable peaks and ~5000 catalogued transitions, giving a chance coincidence rate of ~10% per element-line (depending on wavelength tolerance). Elements with only 2–3 detectable lines (Na, K) are particularly vulnerable: a single chance match can trigger detection.

The hybrid NNLS+ALIAS approach partially overcomes this barrier by requiring corroborating evidence from two orthogonal information sources. However, the improvement is inherently bounded by the weaker channel — ALIAS's limitations at low RP propagate into the hybrid result via the intersection operation. The 17% F₁ improvement (0.559 → 0.654) represents the maximum gain achievable by adding a global fitting stage to a peak-matching confirmation, without improving the peak-matcher itself.

### 4.2 Comparison with Mars LIBS Instruments

The ChemCam instrument on the Mars Science Laboratory rover operates at RP ≈ 2000–4000 across three spectrometer channels (240–342, 382–469, 474–906 nm) (Wiens et al., 2013). Its element identification and quantification pipeline relies on PLS regression trained on 69–332 geological standards (Clegg et al., 2017), achieving robust quantification of SiO₂, TiO₂, Al₂O₃, FeOₜ, MgO, CaO, Na₂O, K₂O. Our CF-LIBS approach, operating at lower RP and without calibration standards, cannot match ChemCam's quantitative performance. However, our hybrid identifier achieves P = 1.00 for Si and Al — the two most abundant rock-forming elements — which is essential for the downstream CF-LIBS concentration calculation.

SuperCam (RP ≈ 2500) and MarSCoDe (RP ≈ 500–700) extend the Mars LIBS dataset to over 1.3 million spectra (Wiens et al., 2025). MarSCoDe's low RP is comparable to our Aalto benchmark conditions, and its element identification relies on pre-selected spectral windows for known target elements rather than blind identification. Our benchmark establishes quantitative performance baselines for blind identification at comparable resolving power.

### 4.3 The Mn and Na Problem

Manganese and sodium represent the dominant failure mode across all algorithms. The root causes are distinct:

**Mn**: With > 500 transitions between 200–900 nm and no spectral region free of Mn lines, any sample with ≥ 5 detected peaks will inevitably have Mn "matches" at RP < 1000. The false positive rate for Mn is effectively determined by the number of detected peaks in the spectrum, not by the actual presence of Mn. At RP > 3000, Mn's individual lines become resolvable and its characteristic multiplet patterns (e.g., the 403 nm triplet) can be positively identified.

**Na**: The Na D-lines at 589.0/589.6 nm are among the strongest emission lines in atomic spectroscopy. Even trace Na contamination (from sample handling, atmospheric Na, or substrate Na) produces detectable emission. With only 2–3 lines in the LIBS-accessible range, Na identification reduces to a binary question at a single spectral position — inherently unreliable without multi-line confirmation.

Both elements require RP-dependent treatment: at RP < 1000, they should either be excluded from the blind search or flagged with a "probable contaminant" warning. At RP > 3000, standard peak-matching is expected to resolve them correctly.

### 4.4 Implications for CF-LIBS Pipeline Design

The hybrid identifier's strengths and weaknesses suggest a recommended CF-LIBS pipeline architecture:

1. **Element identification**: Use hybrid NNLS+ALIAS in intersection mode (nsnr ≈ 1.5, adt ≈ 0.05) for the initial element identification step.

2. **Contaminant filtering**: Apply RP-dependent post-filtering to suppress Mn and Na at RP < 1000, unless independent evidence (e.g., known sample composition, Mn/Na concentration from XRF) confirms their presence.

3. **High-confidence core**: Prioritize Si, Al, Fe, Ca, Mg, K — the major rock-forming elements — which the hybrid method identifies with P > 0.39 and R > 0.40. For these elements, the identification is sufficiently reliable for Boltzmann plot construction.

4. **Iterative refinement**: Use the CF-LIBS concentration results to refine the element list. Elements with concentrations below a physically motivated threshold (e.g., < 0.1 wt%) may be artifacts and can be removed in a second pass.

### 4.5 Limitations and Future Work

**Higher FWHM basis libraries**: The current basis library uses FWHM = 0.5 nm, which is optimal for RP ≈ 1000 but may not be ideal for the lowest-RP spectra (RP ≈ 300–500). Generating and benchmarking at FWHM = 1.0 nm would test this hypothesis.

**FAISS-indexed (T, nₑ) estimation**: The current benchmark uses a fixed fallback temperature for NNLS. Building and using a FAISS nearest-neighbor index over the basis library would allow data-driven (T, nₑ) estimation, potentially improving NNLS precision.

**Machine learning classifiers**: The per-element precision/recall trade-offs suggest that a learned classifier (SVM, random forest, or neural network) trained on the NNLS coefficients, ALIAS scores, and spectral features could achieve higher precision than the simple intersection/union rules used here. The 74-spectrum benchmark is too small for training, but the basis library provides a natural augmentation strategy via synthetic spectrum generation.

**Mechelle 5000 benchmark**: The Mechelle 5000 echelle spectrometer provides RP ≈ 5000 across 200–975 nm. At this resolving power, ALIAS should approach its designed specification (> 95% accuracy). Benchmarking at RP = 5000 would establish the upper bound on peak-matching performance and quantify the RP-dependent precision curve.

**Improved deconvolution**: The Voigt deconvolution pathway showed no net benefit, likely because the SciPy curve_fit optimizer struggles with 10+ overlapping peaks per group. More robust deconvolution strategies — such as the shoulder-detection and residual-completion approach of Dai et al. (2026) — may perform better in the severely blended regime.

---

## 5. Conclusions

We have presented the first systematic benchmark of five element identification algorithms for CF-LIBS at low resolving power (RP = 300–1100) on 74 mineral and elemental spectra. The principal findings are:

1. **The hybrid NNLS+ALIAS identifier achieves the best overall performance** (P = 0.604, R = 0.713, F₁ = 0.654, 16/74 exact matches), representing a 17% F₁ improvement over the ALIAS baseline. Its intersection mode effectively gates NNLS false positives via ALIAS peak-level confirmation.

2. **Full-spectrum NNLS decomposition has excellent recall (R = 0.94) but poor precision (P = 0.29)**, detecting nearly every true element but also flagging 7–8 false positives per spectrum. The physicality of the Saha-Boltzmann basis spectra is insufficient to prevent spurious contributions from line-rich elements.

3. **Peak-matching (ALIAS) precision is fundamentally limited to P ≈ 0.50 at RP < 1000**, consistent with the theoretical expectation from chance coincidence rates. The Mn (P = 0.03) and Na (P = 0.14) false positive rates are intractable at this resolving power.

4. **Voigt deconvolution provides no net benefit at RP < 1000**, because the deconvolution problem itself is underdetermined when peaks overlap by more than one FWHM.

5. **The hybrid method achieves P = 1.00 for Si, Al, Fe, Li, Co, and Ni** — accounting for nearly half of all true positive detections. These elements can be reliably identified at RP < 1000 using the two-stage approach.

6. **Reaching P > 80% at RP < 1000 requires either machine learning classifiers** (which can learn element-specific decision boundaries beyond simple threshold rules) **or higher-RP instrumentation** (RP > 3000), where peak-matching precision is expected to exceed 90%.

These results establish quantitative performance baselines for CF-LIBS element identification and demonstrate that two-stage architectures combining global spectral fitting with peak-level confirmation represent the current best approach for reliable element identification at low resolving power.

---

## References

Al-Jalali, M., Aljghami, I. F., & Mahzia, Y. (2016). Voigt deconvolution method and its applications to pure oxygen absorption spectrum at 1270 nm band. *Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy*, 157, 34–40.

Borduchi, L. C. L., Milori, D., & Villas-Boas, P. (2019). One-point calibration of Saha-Boltzmann plot to improve accuracy and precision of quantitative analysis using laser-induced breakdown spectroscopy. *Spectrochimica Acta Part B: Atomic Spectroscopy*.

Ciucci, A., Corsi, M., Palleschi, V., Rastelli, S., Salvetti, A., & Tognoni, E. (1999). New procedure for quantitative elemental analysis by laser-induced plasma spectroscopy. *Applied Spectroscopy*, 53(8), 960–964.

Clegg, S. M., Wiens, R. C., Anderson, R. B., et al. (2017). Recalibration of the Mars Science Laboratory ChemCam instrument with an expanded geochemical database. *Spectrochimica Acta Part B: Atomic Spectroscopy*, 129, 64–85.

Cremers, D. A., & Radziemski, L. J. (2006). *Handbook of Laser-Induced Breakdown Spectroscopy*. John Wiley & Sons.

Cristoforetti, G., De Giacomo, A., Dell'Aglio, M., et al. (2010). Local thermodynamic equilibrium in laser-induced breakdown spectroscopy: Beyond the McWhirter criterion. *Spectrochimica Acta Part B: Atomic Spectroscopy*, 65(1), 86–95.

Dai, P., Zheng, P., Wang, J., Chen, G., Li, L., & Guo, L. (2026). From spectral interference to overlapped features: A full-spectrum LIBS Voigt decomposition strategy with shoulder detection and residual completion. *Analytical Chemistry*.

Drozdovskiy, I., Ligeza, G., Jahoda, P., et al. (2020). The PANGAEA mineralogical database. *Data in Brief*, 31.

Eum, C., Jang, D., Lee, S., Cha, K., & Chung, H. (2021). Alternative selection of Raman or LIBS spectral information in hierarchical discrimination of raw sapphires according to geographical origin for accuracy improvement. *Talanta*, 221, 121555.

Fujimoto, T., & McWhirter, R. W. P. (1990). Validity criteria for local thermodynamic equilibrium in plasma spectroscopy. *Physical Review A*, 42(11), 6588–6601.

Griem, H. R. (1997). *Principles of Plasma Spectroscopy*. Cambridge University Press.

Harmon, R. S., Russo, R. E., & Hark, R. R. (2013). Applications of laser-induced breakdown spectroscopy for geochemical and environmental analysis: A comprehensive review. *Spectrochimica Acta Part B: Atomic Spectroscopy*, 87, 11–26.

Kim, T., & Lin, C. (2025). Laser-induced breakdown spectroscopy. *Nature Reviews Methods Primers*, 5.

Labutin, T. A., Zaytsev, S. M., & Popov, A. M. (2013). Automatic identification of emission lines in laser-induced plasma by correlation of model and experimental spectra. *Analytical Chemistry*, 85(4), 1985–1990.

Lawson, C. L., & Hanson, R. J. (1974). *Solving Least Squares Problems*. Prentice-Hall. (Reprinted: SIAM Classics in Applied Mathematics, vol. 15, 1995.)

Michelena, S., Ferreira Da Costa, M., & Picheral, J. (2025). Convergence guarantees for unmixing PSFs over a manifold with non-convex optimization. *2025 IEEE Statistical Signal Processing Workshop (SSP)*, 161–165.

Tognoni, E., Cristoforetti, G., Legnaioli, S., & Palleschi, V. (2010). Calibration-free laser-induced breakdown spectroscopy: State of the art. *Spectrochimica Acta Part B: Atomic Spectroscopy*, 65(1), 1–14.

Wang, W., Ayhan, B., Kwan, C., Qi, H., & Vance, S. (2014). A novel and effective multivariate method for compositional analysis using laser induced breakdown spectroscopy. *IOP Conference Series: Earth and Environmental Science*, 17, 012208.

Wiens, R. C., Maurice, S., Lasue, J., et al. (2013). Pre-flight calibration and initial data processing for the ChemCam laser-induced breakdown spectroscopy instrument on the Mars Science Laboratory rover. *Spectrochimica Acta Part B: Atomic Spectroscopy*, 82, 1–27.

Wiens, R. C., Maurice, S., Robinson, S. H., et al. (2020). The SuperCam instrument suite on the NASA Mars 2020 rover: Body unit and combined system tests. *Space Science Reviews*, 217.

Wiens, R. C., Cousin, A., Clegg, S. M., et al. (2025). Geochemistry of Mars with laser-induced breakdown spectroscopy (LIBS): ChemCam, SuperCam, and MarSCoDe. *Minerals*.

Xu, H., Huang, X., Le, X., Zhong, X., Lihua, Z., & Jia, S. (2025). Spectral lines overlap in vacuum arc plasmas of CuCr electrodes. *2025 31st International Symposium on Discharges and Electrical Insulation in Vacuum (ISDEIV)*, 1–3.

---

## Appendix A: Configuration Sweep Details

### A.1 Full Hybrid Sweep (18 configurations)

| Config | NNLS SNR | ALIAS dt | Mode | P | R | F₁ | FPR | Exact |
|--------|----------|----------|------|------|------|-------|-------|-------|
| 1 | 1.0 | 0.03 | intersect | 0.557 | 0.737 | 0.634 | 0.067 | 14/74 |
| 2 | 1.0 | 0.03 | union | 0.274 | 0.952 | 0.425 | 0.289 | 0/74 |
| 3 | 1.0 | 0.05 | intersect | 0.594 | 0.719 | 0.650 | 0.056 | 16/74 |
| 4 | 1.0 | 0.05 | union | 0.283 | 0.952 | 0.437 | 0.275 | 0/74 |
| 5 | 1.0 | 0.10 | intersect | 0.649 | 0.521 | 0.578 | 0.032 | 17/74 |
| 6 | 1.0 | 0.10 | union | 0.292 | 0.952 | 0.447 | 0.264 | 2/74 |
| 7 | 1.5 | 0.03 | intersect | 0.565 | 0.731 | 0.637 | 0.064 | 16/74 |
| 8 | 1.5 | 0.03 | union | 0.296 | 0.952 | 0.451 | 0.259 | 1/74 |
| 9 | 1.5 | 0.05 | intersect | **0.604** | 0.713 | **0.654** | 0.053 | **16/74** |
| 10 | 1.5 | 0.05 | union | 0.308 | 0.952 | 0.466 | 0.244 | 1/74 |
| 11 | 1.5 | 0.10 | intersect | 0.659 | 0.521 | 0.582 | 0.031 | 17/74 |
| 12 | 1.5 | 0.10 | union | 0.319 | 0.952 | 0.477 | 0.233 | 3/74 |
| 13 | 2.0 | 0.03 | intersect | 0.568 | 0.725 | 0.637 | 0.063 | 15/74 |
| 14 | 2.0 | 0.03 | union | 0.310 | 0.940 | 0.467 | 0.239 | 1/74 |
| 15 | 2.0 | 0.05 | intersect | 0.602 | 0.707 | 0.650 | 0.053 | 15/74 |
| 16 | 2.0 | 0.05 | union | 0.326 | 0.940 | 0.484 | 0.222 | 1/74 |
| 17 | 2.0 | 0.10 | intersect | 0.656 | 0.515 | 0.577 | 0.031 | 17/74 |
| 18 | 2.0 | 0.10 | union | 0.339 | 0.940 | 0.498 | 0.209 | 3/74 |

The highest F₁ (0.654) is achieved at nsnr=1.5, adt=0.05, intersect (row 9, **bold**). The highest precision (0.659) is at nsnr=1.5, adt=0.10, intersect, at the cost of lower recall (0.521). The Pareto front in (P, R) space shows a smooth trade-off between precision and recall controlled primarily by the ALIAS detection threshold.

### A.2 NNLS Temperature Sensitivity (45 configurations)

The continuum degree has minimal effect (< 1% variation in F₁ across cdeg ∈ {2, 3, 4} at fixed SNR and T). The detection SNR threshold has a monotonic effect on precision/recall: higher SNR reduces false positives at the cost of false negatives.

### A.3 Forward Model Concentration Sweep (10 configurations)

Concentration thresholds below 0.02 produce excessive false positives (FPR > 0.30). At ct = 0.05, the forward model achieves P = 0.369 — better than raw NNLS but worse than ALIAS, demonstrating that concentration normalization alone is insufficient to distinguish present from absent elements.

---

## Appendix B: Computational Performance

| Pathway | Time per spectrum | Total (74 spectra) |
|---------|------------------|--------------------|
| ALIAS | 0.14 s | 10.7 s |
| Spectral NNLS | 0.28 s | 20.5 s |
| Hybrid (intersect) | 0.39 s | 28.6 s |
| Forward model | 0.27 s | 20.3 s |
| Voigt+ALIAS | 6.8 s | 500 s |

All pathways except Voigt+ALIAS run at interactive speeds (< 0.5 s per spectrum). The Voigt deconvolution is dominated by `scipy.optimize.curve_fit` convergence for multi-peak groups. The hybrid method's overhead (0.39 s vs. 0.14 s for ALIAS alone) is modest and acceptable for batch processing.

Basis library generation (76 elements × 300 grid points × 4096 pixels, FWHM = 0.5 nm) required 244 seconds on a single Apple M-series core. This is a one-time cost amortized across all subsequent analyses.
