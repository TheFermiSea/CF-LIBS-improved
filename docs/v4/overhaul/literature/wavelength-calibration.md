# Wavelength Calibration: Literature Reference

**Topic:** Automated spectrograph wavelength calibration — RANSAC line matching, Hough-transform (RASCAL), cross-correlation comb, differentiable/joint dispersion fitting; when pre-calibrated data needs no recalibration.

---

## 1. Canonical Formulation and Governing Equations

### 1.1 Dispersion Model (Pixel → Wavelength)

The fundamental mapping from detector pixel position `p` (0-indexed or 1-indexed, must be consistent) to wavelength `λ` is expressed as a polynomial:

```
λ(p) = Σ_{k=0}^{N} c_k · P_k(p̃)
```

where:
- `p̃ = 2(p − p_min)/(p_max − p_min) − 1`  is the **normalized pixel coordinate**, mapped to [−1, +1]
- `P_k` is either a **monomial basis** (`p̃^k`), **Legendre polynomial** `L_k(p̃)`, or **Chebyshev polynomial** `T_k(p̃)`
- `c_k` are the polynomial coefficients determined by fitting
- Typical order: N = 3–6 for long-slit spectrographs; N = 4–7 for echelle orders

**Normalization is mandatory**: fitting raw pixel numbers (e.g., 0–2048) in a monomial basis produces a wildly ill-conditioned Vandermonde matrix. Always normalize to [−1, +1] or use Legendre/Chebyshev basis.

For echelle spectrographs a 2D dispersion model is used:

```
λ(p, m) = Σ_{i,j} c_{ij} · L_i(p̃) · L_j(m̃)
```

where `m` is the diffraction order number, normalized similarly. Polynomial degree is typically `(N_i, N_x) ≈ (5, 4)` in the order and cross-dispersion directions.

### 1.2 Line Identification: The Correspondence Problem

Given:
- `{p_k}` — detected peak positions in the observed spectrum (in pixels)
- `{λ_j}` — known arc-lamp or emission-line wavelengths from an atlas

The goal is to find the bijection `σ: p_k → λ_j` that satisfies the dispersion model. The difficulty is the combinatorial explosion: for M peaks and N atlas lines, there are O(N^M) candidate assignments.

#### RANSAC-Based Approach (RASCAL / Veitch-Michaelis & Lam 2019)

**Phase 1 — Hough Transform pre-filter:**

Each (pixel, wavelength) pair `(p_i, λ_j)` defines a family of lines in `(m, b)` space (slope, intercept of a linear approximation `λ ≈ m·p + b`). The Hough accumulator votes for each `(m, b)` bin:

```
For each candidate pair (p_i, λ_j):
    for each m in [m_min, m_max] with step δm:
        b = λ_j − m·p_i
        accumulator[m_bin, b_bin] += 1
```

The accumulator peak identifies the approximate linear dispersion `(m*, b*)`. Gradient bounds are:

```
m_max ≈ (λ_max + Δλ_tol − λ_min + Δλ_tol) / (N_pix / 3)
m_min ≈ (λ_max − Δλ_tol − λ_min − Δλ_tol) / (N_pix × 3)
```

Default `Δλ_tol ≈ 500 Å`, `num_slopes ≈ 1000` bins.

**Phase 2 — RANSAC polynomial fit:**

From the Hough-filtered candidate pairs, RANSAC iteratively:
1. Draw a minimal random sample (size = polynomial order + 1, e.g. 5 for a 4th-order polynomial)
2. Fit the polynomial via least squares to the sample
3. Count inliers: `|λ_j − λ(p_k)| < ε` (residual tolerance, e.g. ε = 1–2 Å)
4. After `T` iterations, refit on all inliers and report the polynomial with the most inliers

**RANSAC number of iterations** (Fischler & Bolles 1981):
```
T = log(1 − p_success) / log(1 − w^s)
```
where:
- `p_success` = desired probability of finding an outlier-free sample (typical: 0.99)
- `w` = expected fraction of inliers (typical: 0.5–0.8 for spectral calibration)
- `s` = minimum sample size (= poly order + 1)

Final solution: least-squares refit on all RANSAC inliers, yielding polynomial coefficients `{c_k}`.

### 1.3 Cross-Correlation Shift Estimation

For tracking small **wavelength drift** of a pre-calibrated spectrometer against a reference spectrum:

```
Δp = argmax_{τ} [ Σ_i  f(i) · g(i − τ) ]
```

in the Fourier domain (O(N log N) via FFT):
```
Δp = IFFT( FFT(f)* · FFT(g) )   → peak location gives integer shift
```

For **sub-pixel accuracy**, fit a Gaussian or parabola to the CCF peak:
```
Δp_subpixel ≈ (CCF[k−1] − CCF[k+1]) / (2·CCF[k−1] − 4·CCF[k] + 2·CCF[k+1])
```

For instruments with **nonlinear dispersion**, the pixel shift is wavelength-dependent. The wavelength correction at position `p` is:
```
Δλ(p) = Δp(p) · (dλ/dp)|_p
```
where `dλ/dp` is the local dispersion (derivative of the polynomial) evaluated at `p`. A single global shift is only valid when the dispersion is nearly linear over the shift range.

After finding `Δp`, the wavelength axis is corrected by interpolation (linear or cubic spline) of the spectrum onto the shifted pixel grid, NOT by shifting the wavelength array directly.

### 1.4 Laser Frequency Comb (LFC) Calibration

LFC lines are uniformly spaced in frequency:
```
f_n = f_0 + n · f_rep
```
where `f_rep` is the repetition rate (typically 10–25 GHz), `f_0` the offset frequency, and `n` is an integer mode number. Converting to vacuum wavelength:
```
λ_n = c / f_n  (vacuum)
```

The extremely dense, precisely known line positions permit a **nonparametric** (non-polynomial) dispersion solution (Excalibur/Zhao et al. 2021): the model is simply a smoothly varying function evaluated on the dense LFC grid, rather than a fixed polynomial.

### 1.5 Dynamic Time Warping (DTW) Approach

DTW aligns an observed spectrum `O` to a template `T` by finding the optimal nonlinear warping path `W`:
```
W = argmin_{W} Σ_{(i,j)∈W} d(O_i, T_j)
```
where `d(·,·)` is a local distance (e.g., squared difference of intensities). The path satisfies monotonicity and continuity constraints. The result is a pixel-to-pixel correspondence from which emission line matches are extracted, followed by a polynomial/spline fit as in the standard pipeline.

---

## 2. Common Implementation Pitfalls and Correct Treatments

### 2.1 Pixel Indexing Convention
- **Pitfall:** mixing 0-indexed and 1-indexed pixel coordinates when computing the normalized coordinate `p̃`. Off-by-one errors shift the entire wavelength solution.
- **Correct:** fix the convention at the start (0-indexed in Python); apply the same convention throughout arc extraction, peak finding, and dispersion fitting.

### 2.2 Polynomial Basis / Normalization
- **Pitfall:** fitting raw pixel numbers (0–4096) in a monomial basis. The Vandermonde matrix condition number grows as `~(N_pix/2)^N` — for N=5 and 2048 pixels this is ~10^17, numerically singular.
- **Correct:** always normalize pixels to [−1, +1] before fitting, or use `numpy.polynomial.legendre.legfit` / `chebyshev.chebfit` which handle this implicitly. Store the normalization bounds (p_min, p_max) with the solution.

### 2.3 Polynomial Order Selection
- **Pitfall:** too-high order overfits noise (wavy dispersion); too-low order leaves systematic residuals.
- **Correct:** use N = 3–5 for single-order spectrographs; inspect residual plots; use cross-validation or Bayesian information criterion (BIC) if automated. The RMS residual should match the expected centroiding precision (~0.05–0.2 px). A residual significantly above the centroiding floor indicates too-low order; significant wavy structure in residuals indicates too-high order.

### 2.4 Line Centroiding
- **Pitfall:** using the raw peak pixel index instead of a sub-pixel centroid. This introduces ~0.5 pixel systematic error.
- **Correct:** fit a Gaussian (or Lorentzian for Voigt-broadened lines) to each arc peak to obtain a sub-pixel center. Centroid precision is typically 1/20–1/40 pixel for well-exposed arcs.

### 2.5 Atlas Line Selection
- **Pitfall:** including blended or self-absorbed atlas lines as calibration anchors. These have systematically offset centroids.
- **Correct:** filter the atlas to unblended, unsaturated lines; verify centroids against their expected positions; reject outliers with iterative sigma-clipping (3–5σ).

### 2.6 RANSAC Inlier Tolerance
- **Pitfall:** using too tight a tolerance (ε) causes the algorithm to find no inlier set; too loose includes many wrong correspondences.
- **Correct:** set ε ≈ 2–5 × (expected centroid precision in wavelength units). For a 0.1 nm/pixel spectrometer and 0.1 pixel centroid precision, ε ≈ 0.05–0.25 nm is appropriate.

### 2.7 Cross-Correlation Shift on Nonlinear Dispersions
- **Pitfall:** applying a single global pixel shift (or wavelength shift) to a nonlinear spectrometer, then calling the spectrum "calibrated." This is wrong for instruments with significant second-order dispersion curvature.
- **Correct:** for nonlinear dispersions, cross-correlation gives only a local shift estimate. Either (a) re-run the full polynomial fit with updated anchors, or (b) apply the shift in linearized pixel space and multiply by the local `dλ/dp` at each wavelength.

### 2.8 Pre-Calibrated Spectrometers — When No Recalibration Is Needed
- Fixed-optics, no-moving-parts fiber spectrometers (e.g., Ocean Optics, Avantes) are stable at room temperature; factory polynomial coefficients remain valid for months to years.
- Recalibration **is** needed when: (a) temperature deviates significantly from calibration temperature (thermal expansion shifts pixel-λ relation); (b) the physical bench is subjected to mechanical shock; (c) optical components (slits, filters) are changed; (d) sub-nm accuracy is required for a LIBS measurement where line identification depends on precise position.
- **LIBS-specific guidance:** for CF-LIBS, the relevant precision is ~0.05–0.2 nm (sufficient to distinguish closely-spaced elemental lines). A factory calibration good to ~1 nm is marginal; drift of 0.5–1 nm can swap line identities. Monitor with at least one strong known emission line per session.

### 2.9 Hough Accumulator Resolution
- **Pitfall:** too-coarse a slope grid causes the Hough peak to be smeared across multiple bins, producing multiple candidate solutions.
- **Correct:** `num_slopes ≈ 1000` is the empirically validated default in RASCAL; the intercept bin count is typically half that. Increasing resolution helps but costs memory quadratically.

### 2.10 Vacuum vs. Air Wavelengths
- **Pitfall:** mixing vacuum and air wavelengths. NIST arc-lamp databases list wavelengths in air (for λ > 200 nm), while laser frequency comb values are in vacuum.
- **Correct:** convert air→vacuum using the Edlén (1953/1966) formula:
  ```
  n_air(λ_air) = 1 + 6432.8×10⁻⁸ + 2949810/(146×10⁸ − 1/λ_air²) + 25540/(41×10⁸ − 1/λ_air²)
  λ_vac = n_air · λ_air
  ```
  Use a consistent convention throughout (NIST LIBS lines are typically in air).

---

## 3. Key References

1. **Fischler, M. A. & Bolles, R. C. (1981).** "Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography." *Communications of the ACM*, 24(6), 381–395. DOI: 10.1145/358669.358692
   — Foundational RANSAC algorithm. Canonical reference for the inlier/outlier model, iteration count formula, and minimum-sample-size argument.

2. **Veitch-Michaelis, J. & Lam, M. C. W. (2019/2020).** "RASCAL: Towards automated spectral wavelength calibration." *ADASS 2019*, arXiv:1912.05883.
   — Introduces RASCAL (RANSAC-Assisted Spectral CALibration): combines Hough transform pre-filter in (slope, intercept) space with RANSAC polynomial fitting. Python library: `pip install rascal`. Key RASCAL parameters: `num_slopes=1000`, `range_tolerance=500 Å`, residual tolerance ε.

3. **Lam, M. C. W., & Veitch-Michaelis, J. (2021).** "Automated SpectroPhotometric Image REDuction (ASPIRED)." *arXiv:2111.02127* (JOSS submission).
   — Integrates RASCAL into a full spectral reduction pipeline; provides worked examples, parameter guidance, and shows Hough→RANSAC two-stage workflow.

4. **Brandt, G. M., et al. (2020).** "Automatic Echelle Spectrograph Wavelength Calibration." *AJ*, 160, 25. DOI: 10.3847/1538-3881/ab929c. arXiv:1910.08079.
   — xwavecal algorithm: uses scale-invariant features (lines that appear in multiple adjacent echelle orders) to establish the dispersion solution without prior knowledge of order numbers or anchor pixels. Polynomial order `(N_i, N_x) ≈ (5, 4)`. Reaches 10 m/s (NRES) / 1 m/s (HARPS) precision.

5. **Zhao, L. L., Hogg, D. W., Bedell, M., & Fischer, D. A. (2021).** "Excalibur: A Nonparametric, Hierarchical Wavelength Calibration Method for a Precision Spectrograph." *AJ*, 161, 80. DOI: 10.3847/1538-3881/abd105. arXiv:2010.13786.
   — Excalibur: nonparametric (Gaussian Process / hierarchical) wavelength calibration exploiting laser frequency combs; 5× lower residual RMS vs. polynomial solutions; reduces radial velocity noise from 1.17→1.05 m/s. Key insight: polynomial fits each exposure independently, discarding cross-exposure information.

6. **Murphy, M. T., et al. (2007).** "High-precision wavelength calibration of astronomical spectrographs with laser frequency combs." *MNRAS*, 380, 839. DOI: 10.1111/j.1365-2966.2007.12147.x.
   — First demonstration of LFC calibration for astronomy; establishes LFC formalism `f_n = f_0 + n·f_rep` and sub-pm wavelength accuracy.

7. **Cersullo, F., et al. (2019).** "New wavelength calibration of the HARPS spectrograph." *A&A*, 624, A122. DOI: 10.1051/0004-6361/201833852.
   — Best practice for high-resolution echelle calibration: Fabry–Pérot etalon + ThAr arc; Legendre polynomial dispersion model; discussion of vacuum/air wavelength conversion (Edlén formula) and instrumental drift.

8. **Wang, K., et al. (2022).** "Initial Drift Correction and Spectral Calibration of MarSCoDe LIBS on the Zhurong Rover." *Remote Sensing*, 14, 5964. DOI: 10.3390/rs14235964.
   — LIBS-specific wavelength drift correction using cross-correlation between acquired spectra and calibration reference; demonstrates pixel-shift → wavelength-shift correction with cubic spline interpolation. Real-world evidence that even space-qualified spectrometers experience drift requiring in-situ correction.

9. **Willett, P., et al. (2025).** "Automated Spectroscopic Wavelength Calibration using Dynamic Time Warping." *arXiv:2508.05862*.
   — DTW as an alternative to RANSAC/Hough for wavelength calibration; handles nonlinear dispersion without initial guesses; follows the same final polynomial/spline fitting step as IRAF IDENTIFY. Best for complex or poorly characterized instruments.

10. **Song, Y.-H., et al. (2018).** "Automated wavelength calibration using arc-lamp spectra." *(internal RASCAL citation, cited as Song 2018 by Veitch-Michaelis; full citation in RASCAL source).*
    — Originator of the Hough-accumulator approach for spectral line matching; RASCAL extends it with RANSAC and higher-order polynomial fitting.

---

## 4. What Correct Code MUST Do — Checklist

### Dispersion Model
- [ ] **Normalize pixel coordinates** to [−1, +1] before polynomial fitting; store `p_min`, `p_max` with the solution so it is reproducible.
- [ ] **Choose polynomial basis** (Legendre/Chebyshev preferred over monomial for numerical stability); validate condition number of the design matrix.
- [ ] **Select polynomial order via residuals**: target RMS ≈ centroiding precision (typically 0.05–0.2 pixel); use cross-validation or BIC if automated.

### Line Identification and Matching
- [ ] **Sub-pixel centroiding**: fit each detected arc peak with a Gaussian to obtain fractional-pixel center before building the calibration pairs.
- [ ] **Atlas filtering**: exclude blended, saturated, and telluric-contaminated lines from the reference atlas used for calibration.
- [ ] **Set RANSAC tolerance ε** as a multiple (~3–5×) of the expected centroid precision in wavelength units, not as a fixed absolute value.
- [ ] **Verify vacuum/air convention**: NIST arc-line databases give air wavelengths for λ > 200 nm; convert consistently using the Edlén formula if mixing with LFC vacuum wavelengths.

### Drift Correction (for pre-calibrated instruments)
- [ ] **Check whether recalibration is warranted** at each measurement session: temperature excursion, mechanical disturbance, or optical component change all invalidate the factory calibration.
- [ ] **Cross-correlation shift for drift monitoring**: compute CCF of a known reference spectrum against the measured calibration line(s); use Gaussian/parabolic sub-pixel interpolation of the CCF peak for fractional-pixel accuracy.
- [ ] **Account for nonlinear dispersion in cross-correlation corrections**: multiply pixel shift by local `dλ/dp` at the reference line position; do NOT apply a single global wavelength offset to the entire nonlinear array.
- [ ] **Apply shift by interpolating the spectrum**, not by shifting the wavelength array. The spectrum is an intensity vs. pixel array; the wavelength axis is derived by evaluating the polynomial.

### LIBS-Specific
- [ ] **Monitor calibration per session** with at least one strong, unblended known emission line (e.g., Hg 253.65 nm, Ne 585.25 nm, or an internal element line with known wavelength).
- [ ] **Report calibration residual (RMS in nm)** alongside each CF-LIBS result so downstream physics (line identification, Boltzmann fitting) can be weighted appropriately.
- [ ] **For CF-LIBS line identification**: require calibration accuracy < 0.1 nm (1σ) to reliably distinguish closely-spaced lines; flag results if calibration RMS > 0.2 nm.
- [ ] **RANSAC on LIBS spectra**: if running fresh wavelength calibration from a laser or arc source co-acquired with LIBS, use RASCAL's default `fit_deg=4` and `num_slopes=1000`; validate by checking all strong known lines fall within ε of their atlas values.

---

*Sources: Based on peer-reviewed literature from 2018–2025 plus knowledge of established methods (Fischler & Bolles 1981, Murphy et al. 2007). Web search conducted 2026-06-25.*
