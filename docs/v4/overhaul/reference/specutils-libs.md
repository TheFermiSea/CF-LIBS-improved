# Reference: specutils + Python LIBS Ecosystem
## How They Structure the Relevant Computation

---

## 1. astropy/specutils

**Current version:** v2.4.x (June 2026). Docs: https://specutils.readthedocs.io/en/stable/

### 1.1 Core Data Model

The central class is `Spectrum` (renamed from `Spectrum1D` in v2.0; the old name raises a deprecation warning). It subclasses `NDCube` and `NDData`, giving it:

| Attribute      | Type / notes |
|----------------|--------------|
| `spectral_axis`| `SpectralAxis` (unit-aware `Quantity`); must be monotonic; can be ascending or descending |
| `flux`         | n-dimensional `Quantity` — the "1D" refers only to the spectral axis, not the whole flux array |
| `uncertainty`  | `StdDevUncertainty`, `VarianceUncertainty`, or `InverseVariance` (all `NDUncertainty` subclasses) |
| `mask`         | boolean ndarray; `True` = bad pixel; avoids NaN "infection" in calculations |
| `wcs`          | FITS WCS or GWCS; auto-created as a GWCS if only `spectral_axis` is given |
| `meta`         | dict; file headers stored as `meta['header']` by convention |
| `radial_velocity` / `redshift` | first-class attributes; `shift_spectrum_to()` shifts axis, `set_radial_velocity_to()` updates frame only |

**v2.0 breaking changes relevant to cflibs:**
- Spectral axis is **no longer forced to be last** in multi-dimensional flux arrays (`spectral_axis_index` kwarg added for disambiguation when auto-detection fails).
- Arithmetic now **validates spectral-axis equality** (not just length equality); spectra on different grids will hard-fail, not silently mismatch.
- Creating a `Spectrum` from a bare spectral axis generates a GWCS that matches the **full dimensionality** of the flux array.

**Unit handling:**
`SpectralAxis` uses astropy's spectral equivalencies by default, enabling transparent conversion between nm, Å, cm⁻¹, eV, GHz, km/s. Only vacuum wavelengths are supported. Unit conversion produces a new `SpectralAxis` backed by a lookup-table WCS (the original pixel→world mapping is discarded). For cflibs this means keeping the native nm/Å grid and converting display-only, not in-place.

**Multi-spectrum containers:**
- `SpectrumCollection` — multiple flux arrays with matching spectral axis shapes; stored as unified arrays for performance.
- `SpectrumList` — `list` subclass with Astropy IO registry integration; for heterogeneous spectra of different lengths. Users don't normally need it directly.
- For ragged collections (different spectral lengths), a plain Python `list[Spectrum]` is recommended.

---

### 1.2 Uncertainty Hierarchy (astropy.nddata)

Three concrete types under `NDUncertainty`:

```
NDUncertainty (ABC)
├── StdDevUncertainty     → stores σ; converts to variance for propagation
├── VarianceUncertainty   → stores σ²; most efficient internally (no double-conversion)
└── InverseVariance       → stores 1/σ²; converts to variance for propagation
```

**Arithmetic propagation rules:**
- `+` / `-`: add variances → √(σ₁² + σ₂²) in result
- `*` / `/`: relative-error propagation
- All three types convert internally to variance for computation, then convert back. `VarianceUncertainty` is ~10–15% more efficient than the others for repeated operations.
- Correlation between operands is possible but requires the correlation coefficient as explicit input; the class cannot derive it.

**How fitting uses uncertainties (`fit_lines`):**
When `weights='unc'` is passed, the code converts the spectrum's uncertainty to `StdDevUncertainty` representation, then computes `weights = 1 / sigma` (inverse-sigma weighting) for the underlying `TRFLSQFitter`.

---

### 1.3 Continuum / Baseline Fitting

Two public functions:

#### `fit_generic_continuum(spectrum, ...)`
1. Median-smooths the spectrum with configurable window (default: 3 pixels) to suppress sharp features before fitting.
2. Delegates to `fit_continuum()` with a `Chebyshev1D(3)` default model.

#### `fit_continuum(spectrum, model, fitter=TRFLSQFitter, exclude_regions=None, window=None)`
- Accepts single model or list of models; multiple models are summed into a compound model.
- `exclude_regions` takes a `SpectralRegion`; the `exclude_region_upper_bounds` flag controls whether the upper boundary is exclusive or inclusive.
- Delegates to `fit_lines()` internally.
- **No iterative sigma-clipping** is implemented; median pre-smoothing is the only outlier mitigation.
- `weights` parameter is **not yet implemented** (`NotImplementedError` raised).
- Default fitter is `TRFLSQFitter` (Trust Region Reflective least squares).

**Pitfall:** Continuum fitting fails on pixel-axis spectra (no physical spectral units) — Issue #453. cflibs must always attach units to its spectral axis before passing to these routines.

**Pitfall:** `fit_generic_continuum` and `fit_continuum` do not sigma-clip. For LIBS baselines (where large emission lines sit on a slowly-varying continuum), the recommended pattern is to first identify and mask emission peaks, then fit the continuum only over the masked-out baseline channels.

---

### 1.4 Line Detection

Two detection strategies:

#### `find_lines_threshold(spectrum, noise_factor)`
- Requires **continuum-subtracted** spectrum with uncertainty defined.
- Logic: `abs(flux) > noise_factor * stddev_uncertainty` → candidate pixels.
- Groups consecutive above-threshold pixels; extracts peak per group.
- `@warn_continuum_below_threshold(threshold=0.01)` decorator warns if continuum residuals are still large.
- **Known pitfall (Issue #504):** Setting a *higher* `noise_factor` can paradoxically detect *more* lines in some spectra due to how consecutive-index grouping interacts with noise shapes. Not a monotone threshold in practice.

#### `find_lines_derivative(spectrum, flux_threshold=None)`
- Discrete derivative via `[1, 0, -1]` convolution.
- Emission peaks: `dY > 0` AND `ddS == -2` (sign-flip detection).
- Absorption troughs: `dY < 0` AND `ddS == 2`.
- Optional post-hoc `flux_threshold` filter.
- No uncertainty required; works on non-continuum-subtracted spectra.

Both return a `QTable` with columns: `line_center` (wavelength Quantity), `line_type` (`'emission'`/`'absorption'`), `line_center_index` (pixel index).

---

### 1.5 Line Fitting

#### `fit_lines(spectrum, model, fitter=TRFLSQFitter, window=None, exclude_regions=None, weights=None, get_fit_info=False)`
- Accepts single or list of models + matching list of windows.
- Window formats: single `Quantity` (symmetric region around model center), tuple of bounds, `SpectralRegion`.
- If `weights='unc'`: uses `1/sigma` weighting.
- If model lacks unit support (`_supports_unit_fitting=False`): strips units, fits, re-wraps in `QuantityModel`.
- Compound model reconstruction via reverse-Polish notation traversal.
- **Edge case:** "The whole spectrum is windowed out!" error if window produces an empty slice.
- Returns fitted model(s); covariance/fit info available with `get_fit_info=True`.

#### `estimate_line_parameters(spectrum, model)`
Pre-fills initial parameter guesses for built-in models:
- `Gaussian1D`: amplitude = max flux, mean = centroid, stddev = Gaussian sigma width.
- `Lorentz1D`: amplitude, x_0, fwhm.
- `Voigt1D`: x_0, amplitude_L, fwhm_L, fwhm_G (split 50/50 if not known a priori).

Custom models require user-supplied estimator dict mapping parameter names to callables.

---

### 1.6 Analysis Functions

All accept optional `SpectralRegion` to operate on a sub-range.

| Function | Key behavior | Pitfall |
|----------|-------------|---------|
| `snr(spectrum)` | Mean flux/uncertainty within region | Requires uncertainty on spectrum |
| `snr_derived(spectrum)` | DER_SNR estimator (MAST/ECF) from flux alone | Valid only if noise is pixel-uncorrelated and signal varies slowly |
| `centroid(spectrum)` | First moment; uncertainty via analytic or MC (`analytic=False`) | Fails without continuum subtraction |
| `line_flux(spectrum)` | Integrated flux; propagates uncertainty | Assumes continuum already subtracted |
| `equivalent_width(spectrum, continuum=1)` | Continuum-normalized | Assumes continuum = 1 unless `continuum` kwarg set |
| `moment(spectrum, order=N)` | Spectral moments of any order | Non-spectral-axis moments use pixel values → meaningless units |
| `fwhm(spectrum)` | True FWHM (no Gaussian assumption) | Requires continuum-subtracted spectrum |
| `gaussian_fwhm(spectrum)` | Gaussian FWHM via second-moment approximation; MC support | Assumes Gaussian profiles |
| `gaussian_sigma_width(spectrum)` | Second-moment width | Same |
| `fwzi(spectrum)` | Full width at zero intensity | Requires continuum subtraction |
| `template_match(obs, templates, redshift_grid=None)` | chi-square minimization; handles redshift sweep | Template normalized to observed spectrum |
| `template_correlate(obs, template)` | Cross-correlation; returns lags in km/s | obs must have rest wavelength set |

**Critical mask issue (Issue #516, partially resolved):**
As of ~2025, several analysis functions (`centroid`, `equivalent_width`, `fwhm`, `gaussian_fwhm`, `gaussian_sigma_width`, `line_flux`, `snr`) historically ignored the spectrum's `mask` attribute. The fix (`flux[~mask]`) was proposed with a `without_mask()` convenience method rather than per-function `use_mask` parameters. cflibs should not assume these functions honor masks automatically — always extract `spectrum[mask].flux` manually or check current specutils version behavior.

---

### 1.7 Manipulation

#### Smoothing
- `box_smooth(spectrum, width)`, `gaussian_smooth(spectrum, stddev)`, `trapezoid_smooth(spectrum, width)` — convolution-based; uncertainty propagated via standard error rules; **kernel widths in pixel units, not physical units**.
- `median_smooth(spectrum, width)` — not flux-conserving; **errors NOT propagated**.

#### Resampling
| Class | Flux-conserving | Uncertainty | Notes |
|-------|----------------|-------------|-------|
| `FluxConservingResampler` | Yes | Propagated | Appropriate when integrating over bins (e.g., degrading resolution) |
| `LinearInterpolatedResampler` | No | Not propagated | Simple interpolation; faster |
| `SplineInterpolatedResampler` | No | Not propagated | Higher-order but no error tracking |

All accept `extrapolation_treatment`: `'nan_fill'`, `'zero_fill'`, or `'truncate'`.

**Pitfall:** Only `FluxConservingResampler` is appropriate when absolute flux matters (e.g., template matching for CF-LIBS composition estimates). Linear/spline resamplers silently discard uncertainty.

#### `noise_region_uncertainty(spectrum, spectral_region, noise_func=np.std)`
Estimates a **constant scalar uncertainty** from a user-specified line-free region and applies it uniformly across the entire spectrum. Returns the spectrum with `uncertainty` set. Appropriate as a first-order estimate when the detector read-noise is approximately flat. Not appropriate when shot noise dominates (where uncertainty ∝ √flux). cflibs currently uses noise estimation from an explicit baseline region; this is the same pattern.

---

## 2. PyHAT (USGS Python Hyperspectral Analysis Tool)

**GitHub:** https://github.com/USGS-Astrogeology/PyHAT (v0.1.2, Aug 2025). Docs: https://pubs.usgs.gov/publication/ofr20251038/full

PyHAT was originally designed specifically for ChemCam/SuperCam LIBS data on Mars Science Laboratory and Perseverance.

### 2.1 Data Model
- **CSV-based tabular format** with two-level column headers: `('wvl', <wavelength>)` for spectra, `('meta', <field>)` for metadata, `('comp', <element>)` for ground-truth compositions.
- Core object is `SpectralData`, a pandas DataFrame wrapper.
- Operations are vectorized row-by-row across the DataFrame.
- Heterogeneous multi-spectrometer spectra are stored as separate column blocks; `Combine Datasets` concatenates via pandas outer join.
- Workflows saved as JSON for reproducibility.

### 2.2 LIBS-Specific Preprocessing
- **Baseline/continuum removal:** multiple algorithms including polynomial fitting, ALS (Asymmetric Least Squares), airPLS, wavelet-based methods, and the Dietrich algorithm (recommended for LIBS). The Wavelet à Trous Plus Spline method approximates the ChemCam pipeline baseline algorithm.
- **Normalization:** divides spectral sub-ranges by their sums (so each sub-range sums to 1.0). Mitigates plasma emission fluctuations across shots. cflibs does not have this explicit sub-range normalization.
- **Masking:** excludes wavelength ranges via a CSV mask file.
- **Resampling:** linear interpolation onto a reference grid via SciPy.
- **Peak area integration:** bins adjacent channels based on local minima/maxima (Clegg et al.).

### 2.3 ML-heavy design
PyHAT leans heavily on scikit-learn/PySptools for regression (PLS, ridge, LASSO, etc.) and dimensionality reduction (PCA, ICA, t-SNE). For cflibs purposes the preprocessing modules (not the ML modules) are the relevant reference.

---

## 3. LIBSsa

**GitHub:** https://github.com/kstenio/libssa. Paper: J. Open Source Software 2024.

A Python 3.9+ GUI + library specifically for LIBS. Key capabilities:
- Saha-Boltzmann temperature/electron density estimation.
- Boltzmann plot output.
- Gaussian/Lorentzian/Voigt peak fitting.
- SAM and MAD outlier removal.
- Full spectrum normalization (FSN).
- PCA / PLSR multivariate modeling.

LIBSsa is less architecturally sophisticated than cflibs but provides a useful public point of comparison for validating Saha-Boltzmann results against an independent open-source implementation.

---

## 4. SpectroscoPyx (PlasmaPy)

**GitHub:** https://github.com/PlasmaPy/SpectroscoPyx — **ARCHIVED March 2023**, read-only.

Intended features (never fully implemented): NIST/CHIANTI database interfaces, atomic energy level descriptors, multi-database identifier tracking. Not a viable reference.

---

## 5. Specific APIs / Patterns Worth Adopting in cflibs

### 5.1 Uncertainty class hierarchy (HIGH VALUE)
Adopt the `StdDevUncertainty / VarianceUncertainty / InverseVariance` three-way hierarchy with a common ABC. Currently cflibs tracks uncertainty as bare NumPy arrays. Using typed uncertainty objects would:
- Make uncertainty representation explicit at API boundaries (avoids "is this σ or σ² or 1/σ²?").
- Enable automatic propagation through arithmetic (`spectrum_a + spectrum_b`).
- Support the pattern where fitting weights are `1/sigma` with a safe type-check rather than a silent squaring error.

Reference: https://docs.astropy.org/en/stable/nddata/index.html

**VarianceUncertainty is internally most efficient** (no double-conversion). cflibs should consider storing variance internally and converting to stddev for display.

### 5.2 Immutable spectrum objects with arithmetic returning new objects (HIGH VALUE)
specutils creates new `Spectrum` objects rather than mutating in place. cflibs's `SingleZoneLTEPlasma` and `SpectrumModel` should follow this — especially for the manifold builder where a pipeline builds many spectra from one base state.

### 5.3 SpectralRegion as first-class API primitive (MEDIUM VALUE)
A lightweight `SpectralRegion(lower, upper)` dataclass that can be composed (`region1 | region2`), inverted, and passed to analysis/fitting functions. cflibs currently uses raw tuples/arrays for wavelength windows. A `SpectralRegion` abstraction would:
- Make `exclude_regions` semantics explicit in continuum fitting.
- Enable the `noise_region_uncertainty` pattern (estimate noise from a specific line-free window).
- Unify the window parameter across `find_lines`, `fit_lines`, `line_flux`, etc.

Reference: https://specutils.readthedocs.io/en/stable/

### 5.4 `noise_region_uncertainty` pattern (MEDIUM VALUE)
When per-pixel uncertainty is unavailable (common in real LIBS measurements), estimate a scalar `σ_noise` from a line-free region of the spectrum. cflibs's `preprocess/` module does noise estimation; formalizing it as a function with the specutils signature would improve composability with downstream fitting.

### 5.5 `estimate_line_parameters` before `fit_lines` (MEDIUM VALUE)
For the Boltzmann plot fitting in `cflibs/inversion/physics/`, pre-estimating line amplitude, center, and width from the measured spectrum (rather than from the atomic database nominal wavelength) reduces convergence failures. Specifically:
- Use spectral centroid (first moment) as `mean` initial guess rather than database λ.
- Use second-moment σ as `stddev` initial guess rather than instrument FWHM.

Reference: https://specutils.readthedocs.io/en/stable/fitting.html

### 5.6 `find_lines_derivative` for candidate detection (MEDIUM VALUE)
The `[1, 0, -1]` discrete-derivative approach followed by sign-flip detection is numerically simpler and more robust than threshold-based detection when SNR is moderate. Unlike `find_lines_threshold`, it does not require an uncertainty estimate a priori. cflibs's `identify/` module's line detection could use this as a cheap initial candidate list before the expensive ALIAS/comb correlation step.

### 5.7 PyHAT sub-range normalization (MEDIUM VALUE)
PyHAT normalizes each spectral sub-range (e.g., one per spectrometer detector arm) to unit sum, mitigating shot-to-shot plasma emission fluctuations. cflibs normalizes by a single global factor. Implementing per-arm normalization before the Boltzmann plot would reduce one source of systematic error when using multi-detector setups (UV/VIS/NIR arms).

Reference: https://pubs.usgs.gov/publication/ofr20251038/full

### 5.8 `FluxConservingResampler` for instrument convolution (LOW VALUE, flag)
When degrading synthetic spectra to instrument resolution for template matching, specutils uses `FluxConservingResampler` (which preserves integrated flux) rather than spline/linear interpolation. cflibs's `profiles.py` uses Gaussian broadening in pixel space; this is correct for the forward model but worth noting that if cflibs ever needs to resample onto a new grid (e.g., for cross-instrument comparison), only flux-conserving resampling is appropriate.

### 5.9 Mask-aware arithmetic (HIGH VALUE — negative example)
specutils Issue #516 documents that many analysis functions ignore `mask` arrays; this was a latent bug for years. cflibs should audit all analysis entry points to ensure `spectrum.mask` (or its equivalent `bad_pixel_mask`) is applied before computing integrals, centroids, and SNR estimates. Do NOT assume that passing a masked array to `np.sum` or `np.mean` will honor the mask without explicit `np.ma.sum` or explicit indexing.

---

## 6. Documented Pitfalls

| Pitfall | Source | cflibs impact |
|---------|--------|---------------|
| `find_lines_threshold` is not monotone in `noise_factor` — higher factor can detect *more* lines due to grouping logic | specutils Issue #504 | Use `find_lines_derivative` as a sanity check; do not rely on threshold tuning alone |
| `fit_continuum` weights parameter is `NotImplementedError` | specutils source | Cannot weight continuum fit by uncertainty; use explicit masking instead |
| Median smoothing in `fit_generic_continuum` does not propagate uncertainties | specutils docs | If uncertainty tracking through baseline removal matters, do not use this; implement explicit ALS/airPLS with propagation |
| Analysis functions may silently ignore `mask` (Issue #516) | specutils | Always apply mask manually before analysis; do not rely on framework propagation |
| `noise_region_uncertainty` returns constant σ — invalid when shot noise dominates | specutils docs | For high-signal LIBS lines, use per-pixel Poisson σ = √(flux × gain); constant σ underestimates uncertainty in bright regions |
| Arithmetic requires spectral axis equality in v2.x (not just length equality) | specutils CHANGES | When comparing synthetic vs. measured spectrum, must resample to identical grids first; subtraction on misaligned grids raises an error |
| Kernel widths in `gaussian_smooth` are in **pixels**, not physical units | specutils docs | A 3-pixel FWHM smooth at 220 nm is different in Å than at 800 nm; convert to pixels before smoothing |
| `SplineInterpolatedResampler` and `LinearInterpolatedResampler` do NOT propagate uncertainty | specutils docs | Do not use these when fitting will rely on propagated uncertainties |
| PyHAT `Combine Datasets` uses outer join → introduces NaN columns for unmatched wavelengths | PyHAT user guide | When merging multi-arm spectra, handle NaN columns explicitly; do not assume concatenation gives dense coverage |
| `VarianceUncertainty` is ~15% more efficient for arithmetic than `StdDevUncertainty` | astropy nddata | Store variance internally in cflibs uncertainty objects; convert to σ only for output |

---

## 7. Concrete cflibs Recommendations

1. **Define a `SpectrumUncertainty` typed hierarchy** in `cflibs/radiation/` (or `cflibs/inversion/common/`) with `StdDev`, `Variance`, `InverseVariance` variants, mirroring the astropy NDData design. This cleanly resolves the "is this σ or σ²?" ambiguity that currently exists at the `IterativeCFLIBSSolver` → `BoltzmannFitResult` boundary.  
   See: https://docs.astropy.org/en/stable/nddata/index.html

2. **Implement a `SpectralRegion` dataclass** (wavelength lower/upper bounds, unit-aware) and plumb it through `find_lines`, `line_flux`, `noise_region_uncertainty`, and `fit_continuum` equivalents. Replace raw tuple/array windows throughout `cflibs/inversion/preprocess/` and `cflibs/inversion/identify/`.  
   See: https://specutils.readthedocs.io/en/stable/

3. **Add a `noise_region_uncertainty` utility** to `cflibs/inversion/preprocess/` that accepts a `SpectralRegion` and returns a constant σ estimate. Flag it clearly as a flat-noise approximation; add a separate Poisson-noise variant for bright-line regimes.  
   See: https://specutils.readthedocs.io/en/stable/api/specutils.manipulation.noise_region_uncertainty.html

4. **Implement `find_lines_derivative`** as a second line detection path in `cflibs/inversion/identify/`. The `[1, 0, -1]` kernel approach is a zero-cost sanity check that requires no uncertainty estimate; compare its output with the current threshold-based detector to catch grouping artifacts.  
   See: https://specutils.readthedocs.io/en/stable/fitting.html

5. **Adopt per-arm normalization** (PyHAT pattern): before Boltzmann plot construction in `cflibs/inversion/physics/`, normalize each spectrometer arm's sub-range to unit sum. This mitigates shot-to-shot intensity jitter, which currently propagates directly into temperature estimates.  
   See: https://pubs.usgs.gov/publication/ofr20251038/full

6. **Use `estimate_line_parameters` logic in Boltzmann fitting**: derive initial `(amplitude, center, width)` from spectral moments of the measured peak rather than database values. Reduces convergence failures when instrument calibration shifts peaks by 0.1–0.3 nm.  
   See: https://specutils.readthedocs.io/en/stable/fitting.html

7. **Audit all mask usage in `cflibs/inversion/`** for the Issue #516 anti-pattern. Grep for `np.sum(flux)`, `np.mean(flux)`, `np.trapz(flux, wl)` without mask application — these are silent bugs if a `bad_pixel_mask` exists on the spectrum.

8. **Store resampled/comparison spectra on identical grids with an equality check**: when computing residuals between synthetic and measured spectra, enforce spectral-axis equality (or explicit interpolation to a shared grid) before subtraction. The v2.x specutils behavior (hard-fail on axis mismatch) is the correct behavior for cflibs too.

---

*Sources consulted:*
- https://specutils.readthedocs.io/en/stable/
- https://specutils.readthedocs.io/en/latest/spectrum.html
- https://specutils.readthedocs.io/en/stable/fitting.html
- https://specutils.readthedocs.io/en/stable/manipulation.html
- https://specutils.readthedocs.io/en/stable/analysis.html
- https://specutils.readthedocs.io/en/stable/_modules/specutils/fitting/fitmodels.html
- https://docs.astropy.org/en/stable/nddata/index.html
- https://github.com/astropy/specutils/issues/504
- https://github.com/astropy/specutils/issues/516
- https://pubs.usgs.gov/publication/ofr20251038/full (PyHAT user guide)
- https://github.com/kstenio/libssa (LIBSsa)
- https://github.com/PlasmaPy/SpectroscoPyx (archived)
