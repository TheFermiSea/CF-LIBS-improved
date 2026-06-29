# RASCAL Reference — Wavelength Calibration via Hough Transform + RANSAC

**Repo:** https://github.com/jveitchmichaelis/rascal  
**Docs:** https://rascal.readthedocs.io/en/latest/  
**Paper:** arXiv:1912.05883 (Veitch-Michaelis & Lam, ADASS 2019)  
**Source studied:** `src/rascal/` — `calibrator.py`, `houghtransform.py`, `models.py`, `util.py`, `atlas.py`

---

## 1. How RASCAL Structures the Computation

### 1.1 Pipeline Overview

```
peaks (pixel positions)
    │
    ├─ refine_peaks()       ← Gaussian subpixel refinement per peak
    │
    ├─ Atlas (line list)    ← NIST arc lines, filtered by intensity + min_distance
    │
    ├─ _generate_pairs()    ← Cartesian product of every peak × every atlas line
    │     (optionally masked by a Delaunay convex polygon for linearity constraint)
    │
    ├─ HoughTransform       ← maps each (pixel, wavelength) pair → (slope, intercept)
    │     generate_hough_points()  → dense (gradient, intercept) point cloud
    │     bin_hough_points()       → 2D histogram (xbins × ybins)
    │     sorted by count → hough_lines (ordered list of most-likely (m, c) lines)
    │
    ├─ _get_candidate_points_linear()
    │     for each hough_line: predict wavelength = m*pixel + c
    │     keep pairs within candidate_tolerance (Å)
    │     weight by Gaussian probability ∝ exp(−Δλ² / (2·σ²))
    │
    ├─ _get_most_common_candidates()
    │     aggregate weighted vote per peak × wavelength pair
    │     keep top_n_candidate wavelengths per peak
    │
    └─ _solve_candidate_ransac()
          random draw of sample_size (peak, wavelength) pairs  (inversely weighted
          by local peak density to spread samples across the spectrum)
          → polyfit on the sample (exact fit)
          → monotonicity check (derivative > 0 everywhere)
          → intercept bounds check
          → _match_bijective(): one-to-one peak↔line assignment by nearest match
          → M-SAC cost: errors clamped at ransac_tolerance; then sum / (n − dof)
             optionally weighted by Hough-space density via 2D spline interpolation
          → if cost improves: robust_polyfit (Huber loss) on inliers → store best
          → early exit if all peaks matched
```

### 1.2 Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `calibrator.py` | Top-level orchestrator; holds all state; calls Hough → RANSAC → fit |
| `houghtransform.py` | Generates and bins the 2D (slope, intercept) accumulator |
| `models.py` | `robust_polyfit` — Huber-loss LSQ via `scipy.optimize.least_squares` |
| `atlas.py` | Line list management; NIST data; vacuum↔air conversion (Modified Edlén) |
| `util.py` | `refine_peaks` (Gaussian fit); `gauss`; `vacuum_to_air_wavelength`; `_derivative` |

### 1.3 Two-Stage Polynomial Fitting

1. **Exact fit** on the RANSAC mini-sample (`np.polynomial.*.polyfit`): used only to test the hypothesis — no robustness required.
2. **Robust refit** on inliers (`robust_polyfit` with Huber loss): the accepted best fit. Inputs are z-score normalised before fitting (divided by std) and rescaled back after. This avoids numerical conditioning problems with high-degree polynomials over large pixel ranges.

### 1.4 Bijective Matching

`_match_bijective` ensures a one-to-one mapping between detected peaks and atlas lines: first pick the closest atlas wavelength for each peak, then resolve any remaining wavelength duplicates by keeping the peak with the smallest error. The result is strictly injective in both directions before the robust refit is run.

---

## 2. Specific APIs / Patterns / Numerical Techniques Worth Adopting

### 2.1 Hough-Space Prefilter Before RANSAC

RASCAL does **not** run blind RANSAC over all (peak, line) pairs. Instead it uses the Hough accumulator to identify the most densely populated (slope, intercept) regions first, then restricts RANSAC sampling to candidate pairs that are consistent with those regions. This collapses the combinatorial space from O(P·L) pairs to a small set of likely correspondences before any polynomial is attempted.

```python
# From houghtransform.py: vectorised intercept computation
slopes = np.linspace(min_slope, max_slope, num_slopes)
intercepts = np.concatenate(y - np.outer(slopes, x))  # broadcast outer product
gradients  = np.concatenate(np.column_stack([slopes] * len(x)))
# Apply bounds, bin into 2D histogram, sort by count
```

**cflibs relevance:** The existing RANSAC calibration does not have a Hough prefilter. Adding one would collapse the search from O(N_peaks × N_lines × max_tries) to a focused set of candidate line pairs.

### 2.2 Candidate Weighted Voting

Each candidate (peak, wavelength) pair is weighted by a Gaussian centred on the Hough-predicted wavelength:

```python
weight = gauss(actual[mask], 1.0, predicted[mask],
               (range_tolerance + linearity_tolerance) * 1.1775)
```

The factor 1.1775 maps FWHM → 1σ (σ = FWHM / 2.355). When multiple Hough lines vote for the same pair, votes are summed. The top-N candidates per peak by aggregated weight are forwarded to RANSAC.

### 2.3 M-SAC Cost Function (Torr & Zisserman 1996)

Standard RANSAC counts inliers (binary). M-SAC weights inliers by their residual:

```python
# From _solve_candidate_ransac:
err[err > ransac_tolerance] = ransac_tolerance   # clamp outliers
cost = sum(err) / (len(err) - len(fit_coeffs) + 1)
```

The denominator `(n_matched − n_coeffs + 1)` normalises for degrees of freedom — a fit with more inliers is preferred over an equally-tight fit with fewer. This distinguishes fits that have identical inlier counts but different residual distributions.

### 2.4 Hough-Density Weighting of the Cost

When `hough_weight` is set, the M-SAC cost is further divided by the Hough-space density evaluated along the current polynomial solution:

```python
wave     = polyval(pixel_list, fit_coeffs)
gradient = polyval(pixel_list, _derivative(fit_coeffs))
intercept = wave - gradient * pixel_list
# Interpolate from the 2D Hough histogram at each (gradient, intercept):
weight = hough_weight * sum(twoditp(intercept, gradient, grid=False))
cost /= (weight + 1e-9)
```

This biases the search toward solutions that pass through dense Hough regions — a principled way to incorporate the global consistency evidence into the per-sample hypothesis evaluation.

### 2.5 Monotonicity Enforcement

After every RANSAC hypothesis, the polynomial is evaluated on the full pixel range ± 20% and rejected if it is not strictly monotonically increasing:

```python
if not np.all(np.diff(polyval(np.arange(pix_min, pix_max, 1), fit_coeffs)) > 0):
    continue
```

This eliminates physically impossible solutions (wavelength going backwards) before the expensive bijective matching step.

### 2.6 Intercept Bounds Check

Before any further computation, the fitted intercept (polynomial value at pixel 0) is compared against `[min_intercept, max_intercept]` derived from the user-supplied wavelength range ± range_tolerance. Fast reject before the O(N) matching step.

### 2.7 Subpixel Peak Refinement via Gaussian Fit

```python
# util.py refine_peaks:
popt, _ = curve_fit(gauss, x[mask], y[mask], p0=[1, mean, sigma])
# returns centre offset from window left edge → sub-pixel position
```

Window of ±10 pixels around each `find_peaks` hit; each peak is fitted independently. Peaks with negative heights or out-of-range centres are silently dropped. Peaks that collapse to the same location (to floating-point resolution) are deduplicated.

### 2.8 Inverse-Density Sampling

To avoid over-sampling peaks in crowded spectral regions, RANSAC draws peaks with probability inversely proportional to local density:

```python
h = np.histogram(peaks, bins=10)
prob = 1.0 / h[0][np.digitize(peaks, h[1], right=True) - 1]
prob /= np.sum(prob)
x_hat = np.random.choice(peaks, sample_size, replace=False, p=prob)
```

This ensures well-spread samples spanning the full spectral range, which improves polynomial conditioning.

### 2.9 Polynomial Basis Choice

`fit_type` selects among `'poly'` (standard monomial), `'legendre'`, and `'chebyshev'`, all via `numpy.polynomial.*`. Legendre/Chebyshev have better conditioning for high-degree fits over large domains. The robust refit uses the same basis as the Hough/RANSAC stage.

### 2.10 Atlas Filtering Pipeline

Atlas lines are filtered in order:
1. Wavelength range (with added `range_tolerance` padding)
2. Minimum NIST relative intensity (per-element, can be a list)
3. Minimum separation between neighbouring lines (`filter_separation`)

Intensity is filtered before separation so that a bright line near a faint line is never removed in favour of the faint one.

### 2.11 Robust Polyfit (Huber Loss with Input Normalisation)

```python
# models.py robust_polyfit:
x_n, y_n = x / x.std(), y / y.std()   # unit-variance normalisation
res = scipy.optimize.least_squares(poly_cost_function, p_init,
                                   args=(x_n, y_n, degree),
                                   loss='huber', diff_step=1e-5)
# rescale back: p *= y.std(); p[i] /= x.std()**(degree - i)
```

This is the correct way to do a robust fit on a high-dynamic-range polynomial problem. Standard `np.polyfit` uses QR decomposition with no outlier protection; the Huber loss makes the final inlier fit resistant to the few remaining bad matches.

### 2.12 Vacuum↔Air Wavelength Conversion

Full Modified Edlén equation (NIST documentation) is implemented in `util.py`, accounting for temperature, pressure, and relative humidity. The Atlas defaults to air wavelengths (vacuum=False) because observatory line lists are usually already in air.

### 2.13 Known-Pair Pinning

`set_known_pairs(pix, wave)` injects user-supplied (pixel, wavelength) ground-truth points into every RANSAC sample unconditionally. Useful for anchoring edge-of-spectrum lines that are too faint to be detected but whose identity is known. The docs warn this can bias the fit — use at most a couple of pairs.

### 2.14 Overfit Guard

After RANSAC completes, validity is checked: `n_inliers > fit_deg + 1`. A "perfect" fit with zero residual is also rejected (`minimum_fit_error = 1e-4` by default) because it typically indicates an overfitted degenerate solution.

---

## 3. Pitfalls Documented by RASCAL

### 3.1 num_pix / pixel_list Must Be Set Correctly

For multi-chip detectors with gaps, `pixel_list` must explicitly encode the physical pixel coordinates. The default assumes `[0, 1, 2, ..., num_pix-1]`. Getting this wrong silently produces an incorrect wavelength solution with no error because the polynomial simply fits the wrong domain.

### 3.2 Tolerance Cascade: Three Different Tolerances

RASCAL uses three distinct tolerances that are easy to confuse:

| Parameter | Role | Typical value |
|-----------|------|---------------|
| `range_tolerance` | error on the provided spectral range | 500 Å |
| `linearity_tolerance` | maximum non-linearity excursion from a linear dispersion | 50–100 Å |
| `candidate_tolerance` | Hough-stage inlier threshold (linear step) | 10 Å |
| `ransac_tolerance` | RANSAC inlier threshold (polynomial step) | 1–5 Å |
| `fit_tolerance` | post-fit RMS warning threshold | 5–10 Å |

Setting `candidate_tolerance` too tight starves RANSAC of candidates. Setting `ransac_tolerance` too loose accepts spurious inliers. The Hough stage tolerances should be wider (they handle non-linearity) while RANSAC tolerance should be close to the expected final residual.

### 3.3 min_intensity Filtering Is Instrument-Dependent

NIST relative intensities do not account for detector quantum efficiency or filter transmission. The `min_intensity` parameter needs tuning per instrument; the default of 10 is conservative and may exclude usable lines, or retain unusable ones for a different instrument configuration.

### 3.4 `minimum_fit_error` Anti-Degenerate Guard Has a False Positive Risk

If the spectrograph is extremely well characterised (sub-0.01 Å residuals), the default `minimum_fit_error=1e-4` will reject valid perfect fits. The parameter must be reduced in high-precision regimes.

### 3.5 `filter_close=True` Can Exclude Nearby Doublets

The close-line filter (`3 * ransac_tolerance` separation threshold) removes pairs with wavelengths closer than that distance. For plasma spectroscopy (LIBS), many physically important lines form close doublets (e.g. Fe lines separated by 0.3–1 Å). This filter should default to `False` for LIBS.

### 3.6 Hough Accumulator Resolution Matters

`num_slopes` (the number of trial gradients) drives both the Hough space density and the computational cost. Too few slopes → coarse accumulator → wrong hough_lines are forwarded to RANSAC. The quickstart uses `num_slopes=10000` with `xbins=ybins=1000` for a wide-range instrument. Under-resolution causes RANSAC to spend iterations on wrong candidate sets.

### 3.7 `sample_size` Floor

If `sample_size <= fit_deg`, RASCAL silently raises it to `fit_deg + 1`. Users who set `sample_size=5` but then change `fit_deg=7` will get a silent size adjustment. The polynomial is underdetermined with fewer samples than `fit_deg + 1`, so this guard is correct — but it means `max_tries` iterations may still be needed even with an "exact" sample.

### 3.8 `add_atlas` Is Deprecated

The `add_atlas()` convenience method on `Calibrator` is deprecated; the correct approach is to construct an `Atlas` object separately and call `calibrator.set_atlas(atlas)`. Code using the old API still works but logs warnings.

### 3.9 Monotonicity Check Range Is 20% Extrapolated

The monotonicity test evaluates the polynomial at `[pix_min - 0.2·ptp, pix_max + 0.2·ptp]`. This can reject valid fits for spectrographs with strong non-linearity near the chip edges. The 20% factor is hard-coded and not user-configurable.

### 3.10 Brute Force Mode Is Implicitly Triggered

If `sample_size >= len(unique candidate peaks)`, RASCAL silently switches to brute-force combination enumeration (`itertools.combinations`). For narrow spectra with few detected peaks this can cause unexpectedly long runtimes.

---

## 4. Concrete "cflibs Should Consider X" Recommendations

### R1. Add a Hough-Transform Pre-Stage to the RANSAC Calibrator

**What:** Before drawing RANSAC samples, run a vectorised Hough transform on (peak_pixel, atlas_wavelength) pairs to build a 2D (slope, intercept) accumulator. Extract the top-K densest bins as candidate linear models; restrict RANSAC sampling to pairs consistent with those models.

**Why:** cflibs's RANSAC calibration runs blind over all peak×line pairs, which scales as O(N_peaks × N_lines × max_iterations). The Hough prefilter collapses this to O(K × candidate_pool_per_bin) where K ≈ 20–50 bins. For LIBS spectra with many Fe/Ca/Mg/Si lines this difference is substantial.

**Link:** `houghtransform.py::generate_hough_points` + `bin_hough_points` — the entire prefilter is ~20 lines of vectorised NumPy.

### R2. Use M-SAC Instead of Binary RANSAC Inlier Count

**What:** Replace the binary inlier/outlier split with M-SAC: clamp errors at `ransac_tolerance`, sum the clamped errors, normalise by degrees of freedom. Use this as the cost rather than counting inliers.

**Why:** With LIBS spectra, multiple plausible polynomial fits may have the same number of inliers but very different residual distributions. M-SAC with DOF normalisation selects the fit that best explains the data globally.

**Link:** `calibrator.py::_solve_candidate_ransac` lines ~530–560.

### R3. Enforce Monotonicity Early in the RANSAC Loop

**What:** After fitting a polynomial hypothesis, immediately check that `np.diff(polyval(pixel_grid, p)) > 0` before running the expensive bijective matching step.

**Why:** The current cflibs RANSAC does not have an explicit monotonicity guard. Physically impossible solutions (wavelength reversals) waste iterations on bijective matching.

**Link:** `calibrator.py::_solve_candidate_ransac` monotonicity check block.

### R4. Replace numpy.polyfit with Huber-Loss robust_polyfit for the Final Fit

**What:** On the RANSAC inlier set, use `scipy.optimize.least_squares(loss='huber')` instead of plain `np.polyfit`. Normalise inputs to unit variance before fitting, rescale after.

**Why:** Even after RANSAC inlier selection, the inlier set for a LIBS spectrum may contain a few misidentified lines (the plasma has many overlapping weak lines). Huber loss suppresses their influence on the final polynomial coefficients. Unit-variance normalisation avoids the ill-conditioning of standard monomial basis over wide pixel ranges.

**Link:** `models.py::robust_polyfit` — 30 lines, self-contained.

### R5. Use Gaussian-Weighted Candidate Voting Before RANSAC

**What:** For each candidate (peak, line) pair accepted by the Hough-stage tolerance, weight it by `gauss(actual_wavelength, 1.0, predicted_wavelength, sigma)` where sigma encodes the range + linearity tolerance. Aggregate weights across all Hough lines voting for the same pair; forward only the top-N weighted candidates per peak to RANSAC.

**Why:** This converts the Hough accumulator's global information into a per-pair prior probability, biasing RANSAC toward the most globally consistent assignments and reducing the effective sample space.

**Link:** `calibrator.py::_get_candidate_points_linear` and `_get_most_common_candidates`.

### R6. Add Inverse-Density Sampling in RANSAC to Improve Polynomial Conditioning

**What:** Weight the RANSAC sampling distribution inversely to local peak density (10-bin histogram over peak positions). Draw peaks with `np.random.choice(..., p=prob)` rather than uniformly.

**Why:** For LIBS spectra with element-specific line clusters (e.g. dense Fe forest at 370–380 nm, sparse at 220–250 nm), uniform sampling over-represents the dense region, producing poorly conditioned polynomial fits. Spread sampling improves condition number of the Vandermonde matrix.

**Link:** `calibrator.py::_solve_candidate_ransac` lines ~480–490.

### R7. Separate Atlas Filtering: Intensity First, Then Separation

**What:** When constructing the atlas of reference lines, apply intensity threshold before minimum-distance filtering (not simultaneously or in reverse order).

**Why:** If a bright line is adjacent to a faint line, filtering by separation first may remove the bright line. RASCAL's `Atlas.add()` applies intensity then separation — this is the correct order. The current cflibs line-selection logic in `cflibs/inversion/identify/` may not respect this ordering.

**Link:** `atlas.py::Atlas.add` docstring ("Lines are filtered first by relative intensity, then by separation").

### R8. Subpixel Peak Refinement via Gaussian Window Fit

**What:** After initial peak detection (`scipy.signal.find_peaks`), fit a 1D Gaussian in a ±10-pixel window around each detected peak to get a sub-pixel centroid. Drop peaks with negative heights or centroids outside the spectrum.

**Why:** cflibs peak detection is used for wavelength calibration. Sub-pixel accuracy in the calibration peaks directly reduces residuals. RASCAL achieves this with a single `curve_fit` call per peak — cheap and effective.

**Link:** `util.py::refine_peaks` — 60 lines, zero dependencies beyond scipy.

### R9. Add `minimum_matches` and `minimum_peak_utilisation` Quality Gates

**What:** After RANSAC, reject solutions that match fewer than `minimum_matches` lines absolute, or fewer than `minimum_peak_utilisation` fraction of detected peaks. Also reject solutions with RMS below `minimum_fit_error` (degenerate/overfitted solutions).

**Why:** cflibs wavelength calibration currently has no post-RANSAC quality gates on the *number* of matched lines — a 3-line polynomial fit over 40 peaks is not physically meaningful. These gates are separate from the RMS tolerance and catch different failure modes.

**Link:** `calibrator.py::_solve_candidate_ransac` acceptance block and `set_ransac_properties` docstring.

### R10. Expose Hough Transform State for Debugging / Caching

**What:** Save/load the 2D Hough histogram to disk (`.npy` or `.json`). When re-running calibration with the same atlas and peaks but different RANSAC settings, reload the Hough state to skip the transform.

**Why:** For batch LIBS calibration (many spectra from the same instrument session), the Hough histogram for the instrument model is constant. Caching it avoids recomputation. RASCAL's `save_hough_transform` / `load_hough_transform` implements this pattern.

**Link:** `houghtransform.py::save` and `load`; `calibrator.py::save_hough_transform`.

---

## Sources

- [GitHub — jveitchmichaelis/rascal](https://github.com/jveitchmichaelis/rascal)
- [RASCAL readthedocs](https://rascal.readthedocs.io/en/latest/)
- [arXiv:1912.05883 — RASCAL: Towards automated spectral wavelength calibration](https://arxiv.org/abs/1912.05883)
- [Calibrator API reference](https://rascal.readthedocs.io/en/latest/modules/calibrator.html)
- [Quickstart tutorial](https://rascal.readthedocs.io/en/latest/tutorial/quickstart.html)
- [Hough transform background doc](https://github.com/jveitchmichaelis/rascal/blob/main/docs/source/background/houghtransform.rst)
- [RANSAC background doc](https://github.com/jveitchmichaelis/rascal/blob/main/docs/source/background/ransac.rst)
