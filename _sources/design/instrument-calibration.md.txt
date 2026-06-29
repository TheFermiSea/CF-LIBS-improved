# Design — `InstrumentCalibration`: a first-class, provenance-tracked instrument seam

**Status:** design (implements [ADR-0006](../adr/ADR-0006-instrument-calibration-first-class.md))
**Date:** 2026-06-20
**Owner module (to create):** `cflibs/instrument/calibration.py`

This document specifies the `InstrumentCalibration` abstraction: its data model, its three builders
(one per acquisition mode), the lamp-fitting math, how it enters the forward model, and how every
pipeline consumes it. The *why* (and the decision to require it) lives in ADR-0006; this is the *how*.

---

## 1. The physical model the object must represent

A spectrometer transforms a true plasma radiance `I_true(λ)` into a recorded, sampled spectrum
`y[pixel]` through three composable operations:

```
            wavelength axis            line-spread fn            response
  y[p]  ≈   sample at λ(p)   ∘   convolve with LSF_σ(λ)   ∘   multiply by E(λ)   { I_true(λ) }   (+ noise)
```

Written as the forward operator the inversion actually fits (profile space):

```
  S_meas(θ; p) = E( λ(p) ) · [ LSF_σ(λ)  *  Σ_s C_s · ε_s(T, n_e; λ) ] evaluated on the λ(p) grid
```

with `θ = (T, log₁₀ n_e, {C_s})` and `ε_s` the per-species emissivity (Saha-Boltzmann × line data).
The instrument contributes exactly three terms, and each enters the fit differently:

| Term | Symbol | Physical origin | How it enters the fit |
|---|---|---|---|
| Wavelength solution | `λ(pixel)` | grating dispersion, pixel geometry | resample / sub-pixel shift — **nonlinear** if unknown |
| Line-spread function | `LSF_σ(λ)` | slit width, optical aberrations, pixel sampling | convolution width — **nonlinear** if unknown; **confounds Stark `n_e`** |
| Spectral response | `E(λ)` | grating efficiency, optics, detector QE | multiplicative — **linear**; rotates the Boltzmann plot if unknown |
| (Scalar efficiency) | `F` | volume, solid angle, gain | **cancels via closure `Σ C_s = 1`** — *not* represented |

The object represents the first three. `F` is deliberately absent (ADR-0006 §4.1).

---

## 2. Data model

All types are **frozen** dataclasses (calibration is a measurement, not mutable state) so they are
hashable, snapshot-safe, and jit-static.

```python
# cflibs/instrument/calibration.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional
import numpy as np

CalibrationMode = Literal["A_calibrated", "B_vendor", "C_self"]


@dataclass(frozen=True)
class WavelengthSolution:
    """λ(pixel): detector pixel index → wavelength (nm)."""
    coeffs: np.ndarray              # polynomial coeffs (pixel → nm), highest-order first
    domain_pixels: tuple[int, int] # valid pixel range the fit covers
    rms_residual_nm: float         # lamp-line fit residual (quality)
    n_anchor_lines: int            # number of lamp lines that anchored the fit

    def to_wavelength(self, pixel: np.ndarray) -> np.ndarray:
        return np.polyval(self.coeffs, np.asarray(pixel, dtype=float))


@dataclass(frozen=True)
class LineSpreadFunction:
    """The instrumental LSF: how a δ-line at λ is smeared on the detector."""
    mode: Literal["resolving_power", "fwhm", "empirical"]
    shape: Literal["gaussian", "voigt", "empirical"]
    resolving_power: Optional[float] = None      # R = λ/FWHM (mode="resolving_power")
    fwhm_nm: Optional[float] = None              # fixed FWHM (mode="fwhm")
    sigma_grid_nm: Optional[np.ndarray] = None   # σ(λ) sampled on wavelength_grid_nm (mode="empirical")
    wavelength_grid_nm: Optional[np.ndarray] = None
    lorentz_fwhm_nm: float = 0.0                 # Voigt Lorentzian part, if shape="voigt"
    kernel: Optional[np.ndarray] = None          # explicit normalized kernel(λ) if shape="empirical"

    def sigma_at(self, wl: np.ndarray) -> np.ndarray:
        """Gaussian σ (nm) of the LSF at the given wavelength(s)."""
        wl = np.asarray(wl, dtype=float)
        if self.mode == "resolving_power":
            return (wl / self.resolving_power) / 2.355
        if self.mode == "fwhm":
            return np.full_like(wl, self.fwhm_nm / 2.355)
        return np.interp(wl, self.wavelength_grid_nm, self.sigma_grid_nm)


@dataclass(frozen=True)
class SpectralResponse:
    """E(λ): relative radiometric efficiency (up to the closure-cancelled scalar F)."""
    wavelength_nm: np.ndarray
    response: np.ndarray           # E(λ); relative (normalized to max=1) unless is_absolute
    is_absolute: bool = False      # True only if traceable absolute radiance response (future F-mode)

    def at(self, wl: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(wl, dtype=float), self.wavelength_nm, self.response,
                         left=0.0, right=0.0)


@dataclass(frozen=True)
class CalibrationProvenance:
    """The audit record — who measured what, when, and how much to trust it."""
    mode: CalibrationMode
    instrument_id: str
    wavelength_source: str         # e.g. "HgAr lamp 2026-06-20, 14 lines, rms 0.011 nm"
    lsf_source: str                # e.g. "HgAr 435.8 nm line, low-T glow discharge"
    response_source: str           # e.g. "D2-halogen + NIST cal cert #SN-1234"
    acquired: Optional[str] = None # ISO date of the calibration acquisition (instrument-supplied)
    quality_flags: tuple[str, ...] = ()   # LOUD flags (Mode-C terms, extrapolation, low n_anchor, ...)
    notes: str = ""


@dataclass(frozen=True)
class InstrumentCalibration:
    """The single seam for all instrument physics (ADR-0006).

    Any term may be None *only* in Mode C, where the missing term is fit from the science
    spectrum and a quality flag is raised. In Modes A/B all supplied terms are measured.
    """
    provenance: CalibrationProvenance
    wavelength_solution: Optional[WavelengthSolution] = None
    lsf: Optional[LineSpreadFunction] = None
    response: Optional[SpectralResponse] = None

    # ---- properties -------------------------------------------------------
    @property
    def mode(self) -> CalibrationMode:
        return self.provenance.mode

    @property
    def is_quantitative_grade(self) -> bool:
        """ADR-0006 D3: only Mode A or B may back a quantitative/accuracy-claimed result."""
        return self.mode in ("A_calibrated", "B_vendor")

    @property
    def fitted_terms(self) -> tuple[str, ...]:
        """Terms that will be fit from the science spectrum (Mode C). Empty for A/B."""
        if self.mode != "C_self":
            return ()
        missing = []
        if self.response is None:
            missing.append("response")
        if self.lsf is None:
            missing.append("lsf")
        if self.wavelength_solution is None:
            missing.append("wavelength_shift")
        return tuple(missing)
```

---

## 3. Builders — one per mode

### 3.1 Mode A — `from_lamps` (the calibrated path)

```python
@classmethod
def from_lamps(
    cls,
    *,
    instrument_id: str,
    wl_lamp_spectrum: tuple[np.ndarray, np.ndarray],     # (pixel_or_wl, intensity) of HgAr/Ne/ThAr/Ar
    lamp_line_list: np.ndarray,                          # certified lamp line wavelengths (nm)
    radiance_lamp_spectrum: Optional[tuple] = None,      # (wl, intensity) of D2-halogen lamp
    lamp_certified_radiance: Optional[tuple] = None,     # (wl, L_cert(λ)) from the cal certificate
    lsf_shape: Literal["gaussian", "voigt"] = "gaussian",
    acquired: Optional[str] = None,
) -> "InstrumentCalibration":
    ...
```

**Step 1 — wavelength solution (from the line lamp).** Detect the lamp emission peaks, match each to its
certified wavelength in `lamp_line_list` (nearest within a tolerance, Hungarian/greedy one-to-one to
avoid double-assignment), and fit a low-order polynomial `λ(pixel)` (typically degree 2–3, or per-CCD-
segment for echelle). The fit residual RMS and the number of matched lines become quality metrics.
*Reuse:* `cflibs/inversion/preprocess/` segmented wavelength calibration already does the peak→known-line
matching; `from_lamps` calls it rather than reimplementing.

**Step 2 — LSF (from the same line lamp).** A glow-discharge lamp runs cold (low `T`) and thin (low `n_e`),
so the *intrinsic* line widths (Doppler ≈ a few pm, Stark ≈ 0) are far below the spectrometer width. The
**measured lamp line profile is therefore the instrumental LSF**. Fit a Gaussian (or Voigt) to several
isolated lamp lines across the range to recover `FWHM(λ)`; convert to either a constant FWHM, a resolving
power `R = λ/FWHM`, or an empirical `σ(λ)` grid if the width varies. Store the shape and, for `voigt`, the
Lorentzian part.

**Step 3 — spectral response `E(λ)` (from the radiance lamp).** If a deuterium–halogen lamp and its
certified spectral radiance `L_cert(λ)` are supplied, measure the lamp through the instrument to get
`I_meas_lamp(λ)`, then

```
E(λ) = I_meas_lamp(λ) / L_cert(λ)        (smoothed; normalized to max = 1 — F is absorbed and cancels)
```

Divide pixel-wise on a common grid (after applying Step-1's `λ(pixel)`), smooth (Savitzky–Golay or a
low-order spline; *not* an ML smoother — physics-only), and normalize. If no radiance lamp is supplied but
the line lamp exists, build the calibration with `response=None` and `mode="C_self"` for the response term
only (a *mixed* calibration — wavelength+LSF measured, response fit — which is common and honestly flagged).

Provenance records each source string; `quality_flags` notes any term that fell back to fitting.

### 3.2 Mode B — `vendor_precorrected` (flight/commercial instruments)

```python
@classmethod
def vendor_precorrected(
    cls,
    *,
    instrument_id: str,                    # "ChemCam", "SuperCam", ...
    source: str,                           # citation / cal-doc id for the published correction
    response: Optional[SpectralResponse] = None,   # published R(λ); None if delivered already flat
    lsf: Optional[LineSpreadFunction] = None,      # published LSF / resolving power
    wavelength_solution: Optional[WavelengthSolution] = None,
    acquired: Optional[str] = None,
) -> "InstrumentCalibration":
    ...
```

For instruments that ship radiance-corrected data (ChemCam, SuperCam): we *declare* the vendor provenance
and use the published terms. If the delivered spectrum is already response-flattened, `response` may be a
flat `E(λ) ≡ 1` with `is_absolute=False` and a provenance note ("vendor radiance-corrected; E(λ)≈1 by
construction"). This mode is quantitative-grade (`is_quantitative_grade is True`) because the correction is
a documented, traceable measurement — just not one *we* performed.

### 3.3 Mode C — `self_calibrating` (the flagged fallback)

```python
@classmethod
def self_calibrating(
    cls,
    *,
    instrument_id: str,
    lsf_prior: Optional[LineSpreadFunction] = None,   # optional nominal width to seed the fit
    response_prior: Optional[SpectralResponse] = None,
    wavelength_solution: Optional[WavelengthSolution] = None,
    reason: str = "no lamp data available",
) -> "InstrumentCalibration":
    ...
```

Builds a calibration whose missing terms (`fitted_terms`) are fit from the science spectrum during
inversion: `E(λ)` as a low-order Chebyshev (reusing the `BayesianForwardModel` Chebyshev-baseline prior
art), the LSF width as a nonlinear parameter, and a sub-pixel wavelength shift if no solution is supplied.
The constructor **always** sets `quality_flags` to include `"MODE_C_NONQUANTITATIVE"` and one flag per
fitted term. Per ADR-0006 D3, results from a Mode-C calibration are reported non-quantitative and can never
flip a scoreboard default.

---

## 4. How it enters each pipeline (the two bridges)

The object exposes exactly two adapters, mirroring the partition-function provider's CPU-scalar / JAX-
batched split (CONTEXT.md "the seam"):

### 4.1 `to_instrument_model()` → legacy + NumPy paths

```python
def to_instrument_model(self) -> "InstrumentModel":
    """Bridge to the existing InstrumentModel seam (legacy integrated-intensity solver,
    SpectrumModel.apply_response / apply_instrument_function)."""
```

Produces an `InstrumentModel` (or `InstrumentModelJax`) with `resolving_power`/`resolution_fwhm_nm` from
`self.lsf`, `response_curve` from `self.response`, and `wavelength_calibration` from
`self.wavelength_solution`. The legacy `cflibs/inversion/solve/iterative.py` path, `SpectrumModel`, and the
existing `apply_response`/`apply_instrument_function` consumers keep working unchanged — they just receive a
model that was *derived from a provenance-tracked source* instead of assembled ad hoc.

### 4.2 `as_snapshot_arrays()` → hot path + jitpipe + Bayesian + manifold

```python
def as_snapshot_arrays(self, wavelength_grid_nm: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed-shape static arrays for the jit forward operator, sampled on the model grid:
       { 'response_grid':  E(λ) on wavelength_grid_nm,        # linear block
         'sigma_grid_nm':  σ_LSF(λ) on wavelength_grid_nm,    # folded into the Voigt Gaussian core
         'dispersion_coeffs': λ(pixel) coeffs,                # resample/shift
         'fit_response': bool, 'fit_lsf': bool, 'fit_shift': bool }  # Mode-C switches
    """
```

These arrays drop into the `PipelineSnapshot` (ADR-0004 D2/D4) and the real-time `structured_jacobian`
kernel (plan v3 §5.1). The mapping to the structured-Jacobian / VarPro decomposition is:

- **`response_grid` (`E(λ)`)** → multiplies the per-species emissivity columns → folds into the **linear
  block** `B` of the structured Jacobian (no extra column when known). When `fit_response`, an extra
  low-order Chebyshev **linear** block is appended.
- **`sigma_grid_nm` (LSF)** → folded **exactly** into the Voigt Gaussian core (`Gaussian ⊗ Voigt = Voigt`,
  the `wphrxvuyj` mechanism — also dodges the V100S cuDNN `convForward` autotune failure). No separate
  convolution kernel. When `fit_lsf`, the LSF width becomes one extra **nonlinear** column beside
  `(T, log n_e)` (~+0.4 ms, +1 forward-difference column measured in `wphrxvuyj`).
- **`dispersion_coeffs` (`λ(pixel)`)** → a fixed resample when known; when `fit_shift`, a single nonlinear
  sub-pixel shift parameter.

So **Mode A bakes all three as fixed arrays** → the operating point stays the plan-v3
structured-Jacobian-K=1 (354 µs @ ROI-2000), best-conditioned. **Mode C adds 1 linear block + up to 2
nonlinear columns** → measurably slower and worse-conditioned, and it makes the Stark `n_e` non-honest —
which is exactly what the quality flag warns about.

### 4.3 Consumers (the call sites to wire — ADR-0006 §4.5 follow-up 2)

| Pipeline | Entry point | Bridge used |
|---|---|---|
| Forward model | `cflibs/radiation/SpectrumModel` | `to_instrument_model()` (NumPy) / `as_snapshot_arrays()` (jit) |
| Legacy CF-LIBS solver | `cflibs/inversion/solve/iterative.py` (via `pipeline.run_pipeline`) | `to_instrument_model()` |
| jitpipe | `cflibs/jitpipe/` (`PipelineSnapshot`) | `as_snapshot_arrays()` |
| Bayesian | `cflibs/inversion/solve/bayesian.py::BayesianForwardModel` | `as_snapshot_arrays()` (Mode-C reuses its Chebyshev baseline) |
| Real-time hot path | `cflibs/inversion/runtime/rt_kernel.py` (plan v3) | `as_snapshot_arrays()` |
| Manifold / ExoJAX reference | `cflibs/manifold/`, offline generator | `as_snapshot_arrays()` |

Each entry point takes an `InstrumentCalibration` (or `None`, which is resolved to an explicit
`self_calibrating(reason="not supplied")` so the flag is always set — never a silent default).

---

## 5. The lamp-fitting math (reference)

**Wavelength solution.** Given matched pairs `(p_i, λ_i)` (lamp peak pixel, certified wavelength),
least-squares fit `λ(p) = Σ_n a_n pⁿ` (degree 2–3, per-segment for echelle). Report
`rms = sqrt(mean((λ(p_i) − λ_i)²))`. Flag if `rms > 0.5·(pixel dispersion)` or `n_anchor_lines < degree+3`.

**LSF.** For each isolated lamp line, fit `A·exp(−(λ−λ₀)²/2σ²)` (+ a Lorentzian wing for Voigt). The lamp's
intrinsic width contribution is `σ_intrinsic² = σ_Doppler,lamp² + σ_Stark,lamp²`; for a glow discharge both
are ≪ `σ_instrument`, so `σ_measured ≈ σ_instrument` (subtract in quadrature if a width estimate exists).
Aggregate `FWHM(λ)` across lines → constant FWHM, `R = λ/FWHM`, or an empirical `σ(λ)` grid.

**Response.** `E(λ) = smooth( I_meas_lamp(λ) / L_cert(λ) )`, normalized to max = 1. The smoothing window must
be wide vs the LSF but narrow vs the response's true wavelength structure (Savitzky–Golay, physics-only).
`is_absolute=True` only if `L_cert` is an absolute spectral radiance with matched geometry (future F-mode).

---

## 6. Testing & validation

- **Round-trip (Mode A):** synthesize a spectrum with a known `(λ(pixel), LSF, E(λ))`, build a calibration
  from synthetic lamp spectra, confirm recovery of all three terms within tolerance, and confirm that
  inverting the calibrated spectrum recovers the input composition better than the same spectrum inverted
  Mode C. *(This is the gate that proves calibration helps, in the benchmark-gate spirit of
  `docs/benchmarks/GATE-FINDINGS-2026-06-19.md`.)*
- **Bridge parity:** `to_instrument_model()` and `as_snapshot_arrays()` must produce numerically consistent
  response/σ on a shared grid (rtol 1e-5, the existing `InstrumentModelJax` parity tolerance).
- **Provenance/flag invariants:** Mode C always carries `MODE_C_NONQUANTITATIVE`; `is_quantitative_grade`
  is True iff Mode A/B; a `None` instrument input resolves to a flagged Mode-C construction, never a silent
  default.
- **Scoreboard:** add a per-dataset calibration-mode column; assert Mode C cannot flip a default
  (ADR-0006 D3).
- **Physics-only:** the import-hygiene test covers `cflibs/instrument/calibration.py` (no banned ML imports;
  smoothing is Savitzky–Golay/spline/Chebyshev, not a learned model).

---

## 7. Summary

`InstrumentCalibration` makes the spectrometer an explicit, measured, provenance-tracked input with three
honest acquisition modes. It removes the LSF, `E(λ)`, and `λ(pixel)` from the inversion when they are
known (Mode A/B — fastest, best-conditioned, honest Stark `n_e`) and fits them with a loud flag when they
are not (Mode C). One object, two bridges (`to_instrument_model`, `as_snapshot_arrays`), every pipeline.
The scalar `F` still cancels via closure, so CF-LIBS stays calibration-free in the one sense it always
legitimately was — and stops pretending to be in the senses it never was.
