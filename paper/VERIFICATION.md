# Paper Verification Report: JQSRT GPU-Accelerated CF-LIBS

**Verified:** 2026-03-25
**Status:** PASSED with minor issues
**Confidence:** HIGH

---

## 1. Convention Consistency

### Units and Notation

The Methods section (Sec. 2, line 5) establishes explicit notation conventions:

| Symbol | Meaning | Units |
|--------|---------|-------|
| T | Electron temperature | eV (unless subscripted T_K for Kelvin) |
| n_e | Electron density | cm^{-3} |
| gamma | Lorentzian HWHM | nm |
| sigma | Gaussian std dev | nm |
| lambda | Wavelength | nm |
| E_k | Upper-level energy | eV |
| k_B | Boltzmann constant | 8.617333e-5 eV/K |

**Verdict: CONSISTENT.** These conventions are used correctly throughout all sections. The T vs T_K distinction is maintained properly (T in eV in physics equations, T_K in Kelvin for Boltzmann slope inversion). The k_B value is standard CODATA. Units of line emissivity (W/m^3/sr) and spectrum (W/m^3/sr/nm) are stated and dimensionally verified (see Section 4 below).

### F-score notation

Two metrics are used:
- **F_1**: Standard F1-score (harmonic mean of P and R). Used in pathway comparison table and hybrid sweep.
- **F_{beta=0.5}**: Precision-weighted F-beta with beta=0.5. Used for ensemble optimization.

**Verdict: CONSISTENT.** The paper always specifies which metric is being used. F_1 appears in pathway/manual comparison contexts; F_{beta=0.5} appears in ensemble optimization contexts. No instances of ambiguous "F-score" without subscript.

### RP notation

Resolving power is consistently written as "RP" (without subscript) throughout all sections. The range RP = 300--1100 is stated consistently in the abstract, introduction, methods, and results.

---

## 2. Numerical Consistency

### Key Numbers Cross-Section Check

| Number | Abstract | Introduction | Methods | Results | Discussion | Conclusion | Verdict |
|--------|----------|-------------|---------|---------|------------|------------|---------|
| F_{beta}=0.914 | 0.914 | 0.914 | -- | 0.914 | 0.914 | 0.914 | MATCH |
| 121% improvement | 121% | 121% | -- | 121% | -- | 121% | MATCH |
| 101 parameters | 101 | 101 | 101 | 101 | 101 | 101 | MATCH |
| 8 groups | -- | -- | 8 | -- | -- | -- | SINGLE (ok) |
| 2,048,000 evals | 2,048,000 | 2,048,000 | 2,048,000 | 2,048,000 | 2x10^6 | 2x10^6 | MATCH |
| 58 seconds | 58s | 58s | 58s | 58s | 58s | 58s | MATCH |
| NNLS R=0.922 | 0.922 | 0.922 | -- | 0.922 | 0.922 | -- | MATCH |
| NNLS P=0.336 | 0.336 | 0.336 | -- | 0.336 | -- | -- | MATCH |
| Hybrid P=0.578 | 0.578 | 0.578 | -- | 0.578 | 0.578 | -- | MATCH |
| Hybrid R=0.731 | 0.731 | -- | -- | 0.731 | -- | -- | MATCH |
| 9,626 energy levels | 9,626 | -- | 9,626 | -- | -- | -- | MATCH |
| 83 elements | 83 | -- | 83 | -- | -- | -- | MATCH |
| 3.1 min GPU gen | 3.1 | -- | 3.1 | -- | -- | -- | MATCH |
| Voigt 76x speedup | 76x | 76x | -- | 76.4x | -- | 76x | see note |
| Boltzmann 8.8x | 8.8x | 8.8x | -- | 8.8x | -- | 8.8x | MATCH |
| Anderson 1.6x | 1.6x | 1.6x | -- | 1.6x | 1.6x | 1.6x | MATCH |
| 10,708 spec/s | 10,708 | -- | -- | 10,708 | 10,708 | 10,708 | MATCH |
| E2E 13.6x | 13.6x | -- | -- | 13.6x | -- | 13.6x | MATCH |

**Voigt speedup rounding:** Abstract/intro/conclusion say "76x" while results says "76.4x" (source: 76.38). The abstract rounds to the nearest integer, the results use one decimal. This is acceptable rounding but slightly inconsistent. Consider using "76x" everywhere or ">76x" for uniformity.

### Voigt Error Numbers -- Two Different Metrics

| Context | Value | Comparison | Section |
|---------|-------|------------|---------|
| GPU-CPU parity | 6.81 x 10^{-8} | Same algorithm, GPU vs CPU backends | Abstract, Results Tab 3 |
| Algorithm accuracy | 5.07 x 10^{-14} | Weideman N=36 vs SciPy wofz | Results text |
| Algorithm accuracy (rounded) | 5.1 x 10^{-14} | Same as above | Conclusion |

**ISSUE (MINOR):** The abstract says "the maximum relative error is 6.8 x 10^{-8} for Voigt profiles" which is the GPU-CPU parity number. The conclusion says "maximum relative error of 5.1 x 10^{-14} against the SciPy reference" which is the algorithm accuracy number. A reader comparing abstract and conclusion sees a 6-order-of-magnitude discrepancy in "Voigt error." Both numbers are correct but measure different things. **Recommendation:** Clarify in the abstract that 6.8e-8 is the GPU-CPU parity error, or use the algorithm accuracy number (5.1e-14) in the abstract instead.

### 121% Improvement Baseline

The paper claims "121% improvement over the hand-tuned baseline" in multiple places but does not explicitly state the baseline F_{beta} value. Computational verification:

- Best manual hybrid intersect (P=0.604, R=0.713): F_{beta}(0.5) = 0.623 -> improvement = 46.7%
- 41-parameter CMA-ES: F_{beta} = 0.551 -> improvement = 65.9%
- Forward-model best config (P=0.369, R=0.796): F_{beta}(0.5) = 0.413 -> improvement = 121.1%

The 121% matches the forward-model F_{beta}(0.5) = 0.413 exactly. However, calling this the "hand-tuned baseline" is ambiguous -- a reader might expect the baseline to be the best single-method result (hybrid, F_{beta} = 0.623) rather than the weakest method.

**ISSUE (MODERATE):** The baseline F_{beta} value for the "121% improvement" claim is not explicitly stated. The number 0.413 or 0.414 does not appear anywhere in the paper. **Recommendation:** Add a sentence like "relative to the hand-tuned baseline ($F_{\beta=0.5} = 0.414$, corresponding to the default multi-algorithm configuration before optimization)" to make the comparison reproducible. Alternatively, clarify which configuration constitutes the "hand-tuned baseline."

### Standalone vs. Pathway Table Numbers

The paper presents two different sets of P/R numbers for the same algorithms:
- **Standalone table (Tab. 5):** Default/standard configuration per algorithm
- **Pathway comparison table (Tab. 2):** Best configuration per pathway after parameter sweep

These correctly differ because different configurations yield different P/R tradeoffs. The abstract explicitly labels its numbers as "Per-algorithm standalone," matching Tab. 5. **No inconsistency.**

### Element Counts: 76 vs 83

- **76 elements:** Basis library coverage for identification (Methods Sec. 2.6.3)
- **83 elements:** Energy level / partition function database coverage (Methods Sec. 2.6.3, Abstract)

These are different things (the basis library uses a subset of the database). Both numbers appear only in appropriate contexts. **No inconsistency.**

---

## 3. Cross-Reference Integrity

### Labels and References

All 28 `\Cref` / `\cref` targets resolve to defined `\label` entries. No dangling references.

| Used References | Defined Labels | Unresolved | Orphaned Labels |
|-----------------|---------------|------------|-----------------|
| 28 | 48 | **0** | 20 (subsection labels not all cref'd -- normal) |

### Citations

All 34 citation keys used in the text have corresponding entries in `refs.bib`. No missing citations.

**Uncited bib entries:** `Kim2025` is defined in refs.bib but never cited. Consider removing or citing it.

### Citation Key vs. Year Mismatches

Several bib entry keys do not match their year fields (common when keys are based on preprint year):

| Key | Bib Year | Renders As | Issue |
|-----|----------|------------|-------|
| Evans2018 | 2020 | Evans et al. (2020) | arXiv 2018, published 2020 |
| Wiens2013 | 2012 | Wiens et al. (2012) | Key uses 2013, bib says 2012 |
| Labutin2013 | 2016 | Labutin et al. (2016) | Key uses 2013, bib says 2016 |
| Johnson2019 | 2021 | Johnson et al. (2021) | Key uses 2019, bib says 2021 |

These key names are cosmetic and do not affect rendered output. Not a bug, but could confuse collaborators reading the .tex source.

---

## 4. Physics Claims Verification

### 4.1 Voigt Profile (Eq. 1-3)

**Equation:** V(lambda) = Re[w(z)] / (sigma * sqrt(2*pi)), z = (lambda - lambda_0 + i*gamma) / (sigma*sqrt(2))

- **Normalization:** Integrates to unity over wavelength. CORRECT (standard result).
- **Dimensions:** [V] = 1/nm (inverse wavelength). The denominator sigma*sqrt(2*pi) has dimensions [nm], and Re[w(z)] is dimensionless. CORRECT.
- **Faddeeva function:** w(z) = exp(-z^2) * erfc(-iz). Standard definition. CORRECT.
- **Weideman N=36:** Branch-free rational approximation. The 236-FLOP count is consistent with Horner evaluation of a degree-35 polynomial in complex arithmetic (35 complex multiply-adds ~ 210 FLOPs + overhead ~ 26 FLOPs). PLAUSIBLE.
- **Error bound < 10^{-13}:** Consistent with Weideman (1994) theoretical bound. CORRECT.

**Verdict: INDEPENDENTLY CONFIRMED** (dimensional analysis and normalization verified by computation).

### 4.2 Boltzmann Plot (Eq. 4-6)

**Equation:** y = ln(I*lambda / (g_k * A_ki)) vs x = E_k, slope = -1/(k_B * T)

- **Physics:** Standard CF-LIBS Boltzmann plot formulation (Tognoni 2010). CORRECT.
- **WLS formulas (Eq. 6):** Standard closed-form weighted least squares slope and intercept. Verified algebraically. CORRECT.
- **Temperature from slope:** T_K = -1/(b * k_B). Dimensional check: b has units [1/eV], k_B has units [eV/K], so T_K = 1/([1/eV]*[eV/K]) = [K]. CORRECT.
- **Uncertainty:** sigma_T = T_K^2 * k_B * sigma_b. This follows from error propagation of T = -1/(b*k_B). CORRECT.
- **Numerical spot-check:** At b = -1.0 eV^{-1}: T = -1/(-1.0 * 8.617e-5) = 11605 K = 1.000 eV. CORRECT.

**Verdict: INDEPENDENTLY CONFIRMED.**

### 4.3 Anderson Acceleration (Eq. 7-8)

- **Fixed-point formulation:** n_e = g(n_e) for charge balance. Standard. CORRECT.
- **Anderson update (Eq. 7):** Standard form from Walker & Ni (2011). CORRECT.
- **Tikhonov regularization (Eq. 8):** Standard least-squares with lambda regularization. CORRECT.
- **Convergence claims:** 1.6x average reduction at tol=10^{-6}, m=1 optimal. Consistent with the physics: LIBS charge balance has contraction rate 0.3-0.7, so Picard already converges in 3-8 steps. Anderson provides modest improvement in this well-conditioned regime. Physically plausible.
- **Depth m=5 at tighter tolerance gives 2.9x:** Consistent with theory (deeper history helps more when more iterations are needed).

**Verdict: STRUCTURALLY PRESENT** (cannot re-run the code, but formulation is standard and claims are physically reasonable).

### 4.4 Softmax Closure (Eq. 9-10)

**Jacobian:** J_ij = C_i * (delta_ij - C_j)

- **Verified computationally** for C = [0.5, 0.3, 0.2]:
  - Row sums = [0, 0, 0] (kernel is the all-ones vector). CORRECT.
  - Rank = D-1 = 2. CORRECT.
- **Shift invariance claim:** C(theta) = C(theta + c*1) follows from softmax properties. CORRECT.
- **Condition number kappa=1 at uniform composition:** At C_i = 1/D, J = (1/D)(I - 1/D * 11^T). Eigenvalues are 1/D (multiplicity D-1) and 0 (multiplicity 1). Condition number of the nonzero eigenvalues is 1. CORRECT.

**Verdict: INDEPENDENTLY CONFIRMED.**

### 4.5 Line Emissivity and Spectrum Assembly (Eq. 11)

**Emissivity:** eps = (hc / 4*pi*lambda) * A_ki * n_k

- **Dimensional analysis:**
  - [hc/lambda] = [J*m / m] = [J] (photon energy)
  - [A_ki] = [s^{-1}] (transition rate)
  - [n_k] = [m^{-3}] (number density after 10^6 conversion)
  - [eps] = [J * s^{-1} * m^{-3}] = [W/m^3]
  - With 1/(4*pi): [W/(m^3*sr)]
  - CORRECT.

**Spectrum:** S(lambda) = sum eps_l * V_l(lambda)
  - [S] = [W/(m^3*sr)] * [nm^{-1}] = [W/(m^3*sr*nm)]
  - Paper states units as [W/m^3/sr/nm]. CORRECT.

**Verdict: INDEPENDENTLY CONFIRMED.**

### 4.6 Memory Budget Formula (Eq. 13)

B_max = floor((M_GPU - M_overhead) / (N_wl * N_lines * d_bytes))

- For B=5000, N_wl=4096, N_lines=50, d_bytes=8: memory = 5000 * 4096 * 50 * 8 = 8.192 GB.
- Paper claims 8.2 GB estimated at B=5000. MATCHES (to rounding).
- OOM at B=5000 on 32GB V100S: profile matrix is 8.2 GB, but total memory (intermediates + JAX runtime + overhead) likely exceeds 32GB. PLAUSIBLE.

**Verdict: INDEPENDENTLY CONFIRMED.**

### 4.7 F_{beta} as Precision-Weighted Metric

F_{beta} = (1 + beta^2) * P * R / (beta^2 * P + R)

- With beta = 0.5: precision is weighted 4x more than recall (since 1/beta^2 = 4).
- Paper describes it as "precision-weighted, emphasizing false-positive suppression over recall." CORRECT.
- F_{beta=0.5} penalizes false positives more than false negatives, appropriate for CF-LIBS where false positives corrupt Boltzmann plots. Physics motivation is sound.

**Verdict: INDEPENDENTLY CONFIRMED.**

### 4.8 Partition Functions from Energy Levels

The paper claims partition functions are "computed from 9,626 energy levels" rather than polynomial fits. This approach:
- Computes U(T) = sum_i g_i * exp(-E_i / (k_B * T)) directly
- Is physically exact (up to completeness of the level database)
- Follows Barklem & Collet (2016) methodology
- Avoids polynomial fit extrapolation errors

**Verdict: STRUCTURALLY PRESENT** (cannot verify the 9,626 count against NIST ASD, but the methodology is standard and well-referenced).

### 4.9 Batch Forward Model Bit-Identical Claim

The paper claims vmap produces bit-identical results to sequential computation. This is correct: JAX's vmap is a purely semantic transformation that reindexes the computation graph without changing floating-point operations. The same XLA kernels execute with the same operand values in the same order.

**Verdict: INDEPENDENTLY CONFIRMED** (follows from JAX vmap semantics).

---

## 5. Structural Issues

### 5.1 Conclusion Enumerate Structure

The conclusion says "five core algorithms were optimized for GPU execution" and then lists 7 enumerated items. Items 1-5 are the five GPU kernels. Items 6 (benchmark) and 7 (ensemble optimization) break the framing -- they are contributions, not GPU kernels.

**Severity: MINOR.** The enumerate works as a list of all contributions but the introductory sentence promises only the five GPU algorithms. Consider restructuring: end the enumerate after item 5, then describe items 6-7 in separate paragraphs.

### 5.2 Unused Bibliography Entry

`Kim2025` (Kim & Lin, "Recent Advances in LIBS") is defined in refs.bib but never cited. Remove or cite.

---

## 6. Summary of Issues

| # | Severity | Issue | Location | Recommendation |
|---|----------|-------|----------|----------------|
| 1 | **MODERATE** | 121% improvement baseline F_{beta} not explicitly stated | Abstract, Intro, Results, Conclusion | State the baseline value (likely ~0.414) |
| 2 | MINOR | Voigt error: abstract (6.8e-8) vs conclusion (5.1e-14) measure different things | Abstract vs Conclusion | Clarify in abstract that 6.8e-8 is GPU-CPU parity |
| 3 | MINOR | Voigt speedup: "76x" (abstract/conclusion) vs "76.4x" (results) | Multiple | Harmonize to "76x" everywhere or ">76x" |
| 4 | MINOR | Conclusion enumerate: 7 items after "five core algorithms" framing | Conclusion | Restructure to separate GPU kernels from other contributions |
| 5 | MINOR | Unused bib entry Kim2025 | refs.bib | Remove or cite |
| 6 | INFO | Citation key/year mismatches (Evans2018->2020, Wiens2013->2012, etc.) | refs.bib | Cosmetic only; optionally rename keys |

---

## 7. Overall Assessment

The paper is numerically consistent, physically sound, and well-referenced. All cross-references resolve, all citations have bib entries, and key numbers match across sections where they appear multiple times. The physics equations (Voigt profile, Boltzmann plot, softmax Jacobian, line emissivity, memory formula) are correct and dimensionally consistent. The F_{beta=0.5} metric is appropriate for the stated goal of false-positive suppression.

The one moderate issue is the undocumented baseline for the "121% improvement" claim. All other issues are minor or informational.
