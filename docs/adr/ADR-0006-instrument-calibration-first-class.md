# ADR-0006 — Instrument calibration is a first-class, provenance-tracked input

- **Status:** PROPOSED (2026-06-20)
- **Authors:** Brian Squires + Opus 4.8 design session (2026-06-20)
- **Scope:** the decision to model the spectrometer as an explicit, provenance-tracked
  `InstrumentCalibration` object consumed identically by every forward and inversion path —
  the legacy integrated-intensity CF-LIBS solver, the differentiable real-time hot path
  (ADR plan v3, `docs/research/realtime/2026-06-20-realtime-plan-v3-structured-jacobian.md`),
  the `jitpipe` pipeline (ADR-0004/0005), the Bayesian model, and the manifold/ExoJAX reference
  generator.
- **Out of scope / non-goals:** the numerical fitting algorithm for each instrument term (that is
  the design doc, [`docs/design/instrument-calibration.md`](../design/instrument-calibration.md));
  absolute (radiometric) composition — we calibrate *relative* response only, see §4.1; any ML
  (the physics-only constraint of CLAUDE.md applies unchanged — `InstrumentCalibration` lives under
  `cflibs/` so ruff TID251 covers it automatically).
- **Related:** ADR-0004 (jittable pipeline; `PipelineSnapshot`/`InstrumentModelJax`), ADR-0005
  (jit promotion), `cflibs/instrument/` (existing `InstrumentModel`, `apply_response`,
  `apply_instrument_function`), `cflibs/inversion/solve/bayesian.py`
  (`BayesianForwardModel` resolving-power mode + Chebyshev baseline — the prior art for fittable
  instrument terms), the 2026-06-20 ExoJAX-reference validation run (workflow `wphrxvuyj`).

---

## 1. Context

### 1.1 The classical "calibration-free" claim, made precise

CF-LIBS gets its name from the cancellation of one — and only one — instrumental term. The integrated
intensity of a line `k→i` of species `s` is

```
I_ki = F · (h c / 4π λ_ki) · A_ki · g_k · (n_s / U_s(T)) · exp(−E_k / kT)
```

where `F` bundles the *wavelength-independent* experimental factors: plasma volume in the field of
view, collection solid angle, the overall optical-train throughput, and detector gain. The Boltzmann
plot linearises this:

```
ln( I_ki λ_ki / (g_k A_ki) ) = −E_k/(kT) + ln( F · (h c/4π) · n_s/U_s(T) )
```

The **slope** gives `T`; the **intercept** gives `n_s/U_s(T)` up to the common factor `F`. Because `F`
is identical for every species, the closure constraint `Σ_s C_s = 1` divides it out exactly. **That
is the entire content of "calibration-free": the scalar `F` cancels.** Nothing else does.

### 1.2 The two instrumental effects that "calibration-free" does *not* remove

1. **Wavelength-dependent spectral response `E(λ)`.** Grating efficiency, optics transmission, and
   detector quantum efficiency all vary with wavelength, so the measured spectrum is
   `I_meas(λ) = E(λ) · I_true(λ)`. In the Boltzmann plot this contributes `ln E(λ_ki)`. Because lines
   with different upper energies `E_k` sit at different wavelengths `λ_ki`, `E(λ)` injects an
   **`E_k`-correlated perturbation that rotates and scatters the Boltzmann plot** — biasing the slope
   (→ `T`) *and* the intercept (→ composition). It does **not** cancel under closure, because it is
   wavelength-dependent and different species populate different wavelength regions. **`E(λ)` is the
   dominant un-removed instrumental systematic in quantitative CF-LIBS.**

2. **The line-spread function (LSF).** The instrument convolves every line with its LSF. Convolution
   conserves area, so in the *integrated-intensity* formulation the LSF leaves the intercept invariant
   *in principle* — which is the textbook reason the classical method "lumps broadening into the
   intercept and ignores it." In practice the LSF still matters even classically (it sets which lines
   are resolved and drives blend contamination of integrated intensities), and it is **confounded with
   the physical broadening**: Stark width → `n_e` and Doppler width → `T`. An unknown LSF makes the
   Stark `n_e` diagnostic dishonest — width gets over-attributed to the plasma.

### 1.3 Why this becomes structurally mandatory now (the new method)

The legacy solver fits **integrated intensities** (areas under peaks), so it can mostly ignore the LSF
for relative composition and only optionally apply a response curve. The real-time method this project
is now building fits the **full line profile** — the residual is computed in spectrum space, not in
integrated-intensity space (plan v3 §2; the structured-Jacobian forward
`S(θ) = response ⊙ broaden(Σ_s C_s · emissivity_s(T, n_e))`). **In profile space all three instrumental
effects — LSF, `E(λ)`, and the wavelength axis `λ(pixel)` — directly shape the residual.** The
instrument operator is now *part of the forward model*; it can no longer be waved away by an area
argument. The new method therefore **cannot be calibration-free** in the LSF/`E(λ)`/axis sense, and the
instrument must be a real, declared input rather than an implicit assumption.

The 2026-06-20 ExoJAX-reference validation (workflow `wphrxvuyj`) made this concrete and gave the
numerical mechanism: the instrument FWHM is folded **exactly** into the Voigt Gaussian core
(`Gaussian ⊗ Voigt = Voigt`), both because it is the physically correct convolution and because the
naive separate `jnp.convolve` lowers to a cuDNN `convForward` that fails to autotune on the V100S. The
LSF is thus a *parameter of the forward operator*, not a post-hoc filter — exactly the seam this ADR
formalises.

### 1.4 What already exists (assets to build on, not replace)

| Asset | Location | Role |
|---|---|---|
| `InstrumentModel` (FWHM / resolving-power, `response_curve`, `wavelength_calibration`) | `cflibs/instrument/model.py` | Today's instrument carrier; `InstrumentCalibration` *produces* one of these for the legacy path |
| `InstrumentModelJax` + `apply_response` JAX kernels | `cflibs/instrument/{model,kernels}.py` | Hot-path response application; consumes calibration arrays |
| `apply_instrument_function` (Gaussian convolution) | `cflibs/instrument/convolution.py` | Legacy LSF application |
| `InstrumentModelProtocol` (structural typing) | `cflibs/core/abc.py:132` | The existing instrument seam — `InstrumentCalibration` complements, does not break it |
| Resolving-power mode + Chebyshev baseline as *fittable* terms | `cflibs/inversion/solve/bayesian.py` | Prior art for Mode-C self-calibration |
| Segmented wavelength calibration | `cflibs/inversion/preprocess/` (`pipeline.py:488`) | The measurement that produces `λ(pixel)` from lamp lines |

The seam is an *aggregation and elevation* of capabilities the repo already has, not new physics.

---

## 2. Decision

**Model the spectrometer as a single, immutable, provenance-tracked `InstrumentCalibration` object,
required as an explicit input to every quantitative forward/inversion path, with three declared
acquisition modes and a hard rule that quantitative results require Mode A or Mode B.** Six binding
sub-decisions:

- **D1 — One object, three terms, one provenance record.** `InstrumentCalibration` carries exactly
  three physical terms — `wavelength_solution` (`λ(pixel)`), `lsf` (the LSF, `σ(λ)` + shape), and
  `response` (`E(λ)`) — plus a mandatory `provenance` record (mode, per-term source, instrument id,
  acquisition date, quality flags). The object is the *single seam* for all instrument physics; no
  pipeline reaches around it to a bare FWHM or a loose response array. (API in the design doc.)

- **D2 — Three explicit acquisition modes, never implicit.**
  - **Mode A — Calibrated.** Built from the user's own lamp spectra
    (`InstrumentCalibration.from_lamps(...)`): a wavelength/LSF lamp (HgAr, Ne, ThAr, Ar) fixes
    `λ(pixel)` and the LSF; a radiance lamp (deuterium–halogen, or NIST-traceable tungsten-halogen)
    fixes `E(λ)`. All three terms are *measured*; the fit removes them from the inversion.
  - **Mode B — Vendor pre-corrected.** Flight/commercial instruments (ChemCam, SuperCam) deliver
    radiance-corrected spectra with a published response/LSF. We *declare* that provenance and use the
    published `R(λ)`/LSF; we do not re-fit them.
  - **Mode C — Self-calibrating fallback.** No lamp data and no vendor correction: the instrument terms
    that cannot be supplied are *fit from the science spectrum itself* (smooth `E(λ)` as a low-order
    Chebyshev, LSF width as a nonlinear parameter, sub-pixel shift), and the result carries a **loud
    quality flag**.

- **D3 — Mode A or B is required for any quantitative / accuracy-claimed result.** The benchmark
  scoreboard, NIST-parity, and any reported composition RMSE run in Mode A or B. Mode C is permitted
  for best-effort/exploratory use and for the existing real-data test sets where no lamp exists, but
  its outputs are flagged non-quantitative and are never used to *flip a default* or claim a SOTA
  number. (This is the discipline that prevents an uncalibrated regression from masquerading as an
  algorithm result — the same failure class ADR-0004 §1 and the 2026-06-19 gate findings guard against.)

- **D4 — The calibration enters the forward model as a typed operator, decomposed by how it enters the
  fit.** `S_meas(θ) = E(λ) ⊙ LSF_σ(λ){ Σ_s C_s · ε_s(T, n_e; λ on the λ(pixel) grid) }`. The three
  terms map cleanly onto the structured-Jacobian / VarPro decomposition (plan v3 §3–4):
  - `F` (the scalar) — **cancels via closure**; needs no calibration for relative composition.
  - `E(λ)` (smooth, multiplicative) — **linear** in the forward; when known it folds into the
    per-species basis columns (the linear block); when unknown (Mode C) it is fit as a smooth
    low-order function (added linear DOF + degeneracy).
  - LSF width (the Voigt Gaussian core) — **nonlinear**; when known it is a fixed kernel folded exactly
    into the Voigt core; when unknown (Mode C) it becomes an extra nonlinear column alongside
    `(T, log n_e)` (~+0.4 ms / +1 column measured in `wphrxvuyj`) **and confounds with Stark `n_e`**.
  - `λ(pixel)` — a resample/sub-pixel alignment; fixed when known, a small nonlinear shift when not.
  Mode A therefore yields the best-conditioned, fastest, most honest fit; Mode C pays in conditioning,
  latency, and Stark-`n_e` honesty — which is *why* it is flagged.

- **D5 — Consumed identically by all pipelines via two bridges.** `InstrumentCalibration` exposes
  (a) `to_instrument_model()` → an `InstrumentModel`/`InstrumentModelJax` for the legacy integrated-
  intensity path and the existing `apply_response`/`apply_instrument_function` consumers, and
  (b) `as_snapshot_arrays()` → fixed-shape static arrays (`σ(λ)` samples, `E(λ)` on the grid, dispersion
  coeffs) that drop into the `PipelineSnapshot` (ADR-0004 D2/D4) and the real-time hot kernel. One
  object, two adapters — mirroring the partition-function provider pattern (CONTEXT.md "the seam").

- **D6 — Physics-only and immutable.** `InstrumentCalibration` and its builders are pure physics + NumPy/
  JAX (no banned ML imports; Mode-C smoothing is explicit Chebyshev/`jnp`, reusing
  `cflibs.inversion.physics.softmax_closure`-style explicit forms, never `jax.nn`). The object is a
  frozen dataclass — calibration is a measurement, not mutable state — so it is hashable/snapshot-safe
  and jit-static.

---

## 3. Why (the deletion test and the alternatives rejected)

**Deletion test.** Delete `InstrumentCalibration` and the three instrument terms scatter: the wavelength
solution lives in `preprocess`, the LSF in `InstrumentModel.resolution_fwhm_nm` *and* the Bayesian
resolving-power knob *and* the hot kernel's folded σ, the response in a loose `response_curve` array *and*
ChemCam's vendor assumption *and* the Bayesian Chebyshev baseline — and *nothing* records which of these
were measured vs assumed. Complexity (and the silent-miscalibration risk) reappears across ≥5 call sites.
The object concentrates it: one place defines what the instrument is, one place records its provenance,
one rule decides whether a result is quantitative. That is depth, not a pass-through.

**Alternatives considered and rejected:**

- **Keep it implicit (status quo).** Works for the legacy integrated-intensity path *only because* of the
  `F`-cancellation and area-conservation arguments (§1.1–1.2). It breaks structurally for the profile-space
  method (§1.3), and it has already let an uncalibrated assumption ride silently — the exact failure the
  gate discipline exists to stop. Rejected.
- **Three separate inputs (a wavelength solution, an `InstrumentModel`, a response array) with no
  aggregate.** This is essentially today's scatter. It cannot carry a single provenance record or enforce
  the Mode-A/B-required rule, and it forces every pipeline to re-assemble the instrument from parts.
  Rejected on the deletion test.
- **Fold everything into a fit knob (always Mode C).** "Just fit the instrument with the science spectrum"
  is seductive but conflates instrumental and plasma physics — the LSF/Stark-`n_e` confound (§1.2) and the
  `E(λ)`/slope-`T` confound (§1.2) make this systematically biased. Calibration data, when available, is
  *more information*; throwing it away to fit more parameters is strictly worse. Mode C is the documented
  fallback, not the design center. Rejected as the default.

---

## 4. Consequences

### 4.1 Scope boundary: relative, not absolute

We calibrate the *relative* response `E(λ)` (shape vs wavelength) and the LSF and axis. The absolute scalar
`F` is deliberately left to closure — CF-LIBS stays calibration-free in the one sense it legitimately is.
Absolute number densities (which *would* need `F` from a radiance standard with known geometry) are an
explicit non-goal here; `provenance.response.is_absolute` records whether a traceable absolute response was
supplied, for a future absolute-radiance mode.

### 4.2 Test-data handling (honest, not aspirational)

- **ChemCam / SuperCam** datasets → **Mode B** (vendor radiance-corrected; published `R(λ)`/LSF declared in
  provenance). No re-fit.
- **Lab datasets with lamp spectra available** → **Mode A**.
- **Lab datasets without lamps** (most of the current real-data suite) → **Mode C**, results flagged
  non-quantitative. We do not pretend these are calibrated; the flag is the honesty.
- This means some current benchmark numbers are Mode-C-grade and must be re-labelled accordingly; the
  scoreboard gains a per-dataset calibration-mode column.

### 4.3 Engineering consequences

- New module `cflibs/instrument/calibration.py` (the object + builders); `cflibs/instrument/__init__.py`
  exports `InstrumentCalibration`. Existing `InstrumentModel`/`apply_response`/`apply_instrument_function`
  are unchanged and become consumers via `to_instrument_model()`.
- The forward (`SpectrumModel`), the legacy solver path, `jitpipe`, the Bayesian model, and the real-time
  kernel each take an `InstrumentCalibration` (or `None` → an explicit Mode-C construction with the flag).
- The hot path gains `as_snapshot_arrays()` → `PipelineSnapshot`; Mode A bakes `E(λ)`/`σ(λ)` as fixed
  arrays (best conditioning, no extra fit columns), Mode C adds the nonlinear LSF column + linear Chebyshev
  `E(λ)` block.
- The benchmark scoreboard records and reports the calibration mode per dataset; Mode C cannot set a
  default flip.

### 4.4 What this does *not* change

- The closure constraint and the `F`-cancellation are untouched — relative composition stays calibration-
  free in the legitimate sense.
- `InstrumentModelProtocol` (`abc.py:132`) stays the structural seam for "an object that applies a response
  and a convolution"; `InstrumentCalibration` is the richer, provenance-carrying source those objects are
  derived from.
- No change to the physics-only constraint, the partition-function provider, or the jit parity contracts.

### 4.5 Follow-ups (tracked, not blocking this ADR)

1. Implement `cflibs/instrument/calibration.py` + `from_lamps` / `vendor_precorrected` / `self_calibrating`
   builders (design doc §3).
2. Wire `InstrumentCalibration` through `SpectrumModel`, the solver path, `jitpipe`, Bayesian, and the
   real-time kernel; add `to_instrument_model()` / `as_snapshot_arrays()` bridges.
3. Add the scoreboard calibration-mode column + the Mode-A/B-required gate rule.
4. Acquire/locate lamp spectra for the lab datasets to lift them from Mode C → Mode A where possible.
5. Optional absolute-radiance mode (uses `F`) — deferred.

---

## 5. Status / acceptance

PROPOSED. Accepted by starting follow-up (1). The design doc
[`docs/design/instrument-calibration.md`](../design/instrument-calibration.md) specifies the API, the
lamp-fitting math, and the per-pipeline wiring. `InstrumentCalibration` and its three modes are added to
`CONTEXT.md` as canonical vocabulary on acceptance.
