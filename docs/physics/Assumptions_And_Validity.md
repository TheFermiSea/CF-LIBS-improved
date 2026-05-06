# Physics: Assumptions and Validity

CF-LIBS is a physics-based inversion. Its outputs are only meaningful when
its physical assumptions hold. This document enumerates every assumption
the shipped algorithm makes, the regime in which each holds, the
diagnostics CF-LIBS exposes for detecting violations, and the practical
options when an assumption fails.

For the equations these assumptions support, see
[Equations](Equations.md). For the algorithm flow, see
[Inversion Algorithm](Inversion_Algorithm.md).

---

## Summary Table

| # | Assumption | Where it appears | Diagnostic | Failure mode if violated |
|---|------------|------------------|------------|--------------------------|
| 1 | Local thermodynamic equilibrium (LTE) | Saha + Boltzmann population statistics | McWhirter criterion check (`lte_mcwhirter_satisfied`) | Inferred `T` is meaningless; ionization stages disagree |
| 2 | Optically thin plasma | Forward model `I = ε L`, Boltzmann plot uses `ln(Iλ/gA)` | Curve-of-growth check; per-line `τ` estimate | Strong lines saturate, Boltzmann plot bends, composition biased toward weak-line elements |
| 3 | Single-zone uniformity (homogeneous `T`, `n_e`, composition along LOS) | `SingleZoneLTEPlasma` model | Inter-element `T` consistency, neutral/ion `T` consistency | Apparent `T` is line-of-sight average, not a real plasma temperature |
| 4 | Temporal stationarity over the gate window | Time-integrated forward model | Compare different gate windows | Mixing of recombination-stage and emission-stage plasma |
| 5 | All major (>1 % mass) elements included in the candidate list | Standard closure equation | Residual after closure; refit with extra candidates | Closure mis-normalizes; reported `C_s` are too high |
| 6 | Atomic data accuracy | Every line uses NIST `A_ki, g, E_k` | NIST parity validation script | Per-line bias propagates to per-element intercept |
| 7 | Adequate energy spread of selected lines | Boltzmann plot slope | `min_energy_spread_ev` enforced; `R²` of fit reported | Slope (and therefore `T`) ill-conditioned |
| 8 | Adequate spectral resolution to isolate lines | Line profile model | `isolation_score` per line | Blended lines bias intensities |
| 9 | Gaussian (or Voigt) instrument line-spread function | Convolution stage | Use diffraction-line calibration | Skewed instrument response biases line widths |
| 10 | No significant continuum/background | Peak detection, intensity integration | Baseline-removal step in preprocessing | Background bias raises all intensities |

The remainder of this document expands on each assumption.

---

## 1. Local Thermodynamic Equilibrium (LTE)

### What it states

A single temperature `T` describes the population of bound electronic
levels via the Boltzmann distribution and the relative populations of
ionization stages via the Saha equation. Populations are determined by
collisions with electrons, not by the photon field.

### What it requires

Electron-impact rates `n_e ⟨σv⟩_collision` between a level and any other
level must dominate radiative decay rates `A`. The standard sufficiency
criterion (McWhirter) is

```
n_e ≥ 1.6 × 10¹² × √T × (ΔE)³   [cm⁻³, T in K, ΔE in eV]
```

with `ΔE` the largest energy gap among the levels feeding the lines used
in the analysis. CF-LIBS evaluates this at convergence and reports
`lte_mcwhirter_satisfied: bool` and `lte_n_e_ratio` (=
`n_e / n_e_required`) in `result.quality_metrics`.

### When LTE typically holds in LIBS

- Atmospheric-pressure plumes a few hundred ns to a few µs after the
  laser pulse, when `n_e` is in the 10¹⁶–10¹⁸ cm⁻³ range and `T` is
  8000–15000 K.
- Continuum-dominated early times *before* lines are visible.

### When LTE typically fails

- **Late gate windows** (recombination phase): `n_e` drops below the
  McWhirter threshold; high-`E` levels depopulate faster than they can
  refill.
- **Very low ambient pressure / vacuum** plumes: free expansion lowers
  `n_e` quickly.
- **Large energy gaps** (resonance lines from ground state to first
  excited level): the cubed `ΔE` term makes McWhirter harder to satisfy
  for high-`E` transitions.
- **Hydrogen and helium** in cool plasmas: LTE often fails for these even
  when McWhirter is satisfied because of the dominance of resonance
  radiative rates. The McWhirter criterion is necessary but not always
  sufficient — see ref. [9] in [Equations § References](Equations.md#references).

### When CF-LIBS reports a violation

`cflibs analyze --uncertainty analytical` prints

```
WARNING: McWhirter criterion NOT satisfied (n_e ratio = 0.34)
```

if `lte_n_e_ratio < 1`. Treat the temperature, electron density, and
composition as physically suspect even if the solver converged.

### Options when LTE fails

1. **Move the gate window earlier** if you can. The transition to
   non-LTE is usually time-dependent.
2. **Restrict to lines with smaller `ΔE`**. Lower the energy spread cut
   `min_energy_spread_ev` and accept the cost in temperature precision.
3. **Use Bayesian inference** with a model that includes per-stage
   temperature offsets (`T_neutral ≠ T_ion`). The shipped Bayesian model
   does not yet expose this; it is a documented extension point.
4. Accept that this spectrum is not amenable to CF-LIBS. Calibration-free
   methods rely on LTE; without it, calibration with matrix-matched
   standards is more reliable.

### Code references

- McWhirter check: `cflibs/inversion/quality.py`
  (`QualityAssessor.check_lte_validity`).
- Quality metrics surfaced in `CFLIBSResult.quality_metrics`.
- Constant: `cflibs.core.constants.MCWHIRTER_CONST = 1.6e12`.

---

## 2. Optically Thin Plasma

### What it states

The line emission emerging from the plasma equals `ε(λ) · L`. Photons do
not re-absorb significantly along the integration path.

### What it requires

The optical depth at line center,

```
τ₀ = (πe² / m_e c) f_ki n_i L / Δν_D,
```

is small (`τ₀ ≪ 1`) for every line used in the Boltzmann fit. `n_i` is
the *lower-level* population, `f_ki` is the oscillator strength.

### When optical thinness fails in LIBS

- **Resonance lines** (lower level = ground state) of major elements: Fe
  I 248.33 nm, Mg II 279.55 nm, Na I 589 nm, K I 766/770 nm, Ca II 393/397 nm.
  These are routinely self-absorbed at major-element concentrations.
- **High concentrations** of any element with strong oscillator strength.
- **Long path lengths** (`L > a few mm`) at standard LIBS densities.
- **Cold plasma cores** with hot outer shells (curve-of-growth deviates
  from the simple form because of source-function non-uniformity).

### Diagnostic signatures

- The Boltzmann plot bends at high intensity (low `E_k`) — strong lines
  fall below the linear trend.
- The neutral-only and ion-only Boltzmann plots give different
  temperatures.
- The same element measured at multiple resonance/non-resonance lines
  gives different intercepts.

### Options

1. **Exclude resonance lines** (`exclude_resonance: true` — default).
   This is the cheapest defense.
2. **Use the iterative `SelfAbsorptionCorrector`** for known offenders.
   Requires absolute intensity calibration or a known-fraction reference
   line.
3. **Use CDSB (Column-Density Self-Absorption correction)** for
   moderate-`τ` regimes (`cflibs/inversion/cdsb.py`). Implements the
   full Voigt curve of growth.
4. **Mask lines whose `τ > max_optical_depth`** (the corrector does this
   automatically).

### Code references

- `cflibs/inversion/self_absorption.py` (iterative).
- `cflibs/inversion/cdsb.py` (curve-of-growth).
- `quality_metrics["self_absorption_warnings"]` in the result.

---

## 3. Single-Zone Uniformity

### What it states

`T`, `n_e`, and elemental composition are constant in the volume sampled
by the spectrometer. The forward model is `SingleZoneLTEPlasma`.

### What it requires

The integration column along the line of sight has uniform plasma
parameters. In practice, real LIBS plumes are stratified: hot core,
cooler periphery, sharp boundaries.

### Failure mode

A single-zone fit returns a *line-of-sight emission-weighted* `T`,
biased toward the high-density emitting region. Lines whose emission is
peaked in different zones (e.g. neutral lines in the cool periphery vs.
ionic lines in the hot core) report different "temperatures", and the
common-slope fit averages them with weights set by signal strength.

### Diagnostic signatures

- Per-element temperatures (`fit.temperature_eV` per element) disagree
  by more than 10–15 % even after Saha correction.
- Neutral-only and ion-only Boltzmann fits for the same element disagree.
- The Saha-Boltzmann consistency metric in `quality_metrics` is poor.

### Options

1. Restrict the analysis to lines from a single zone (e.g. drop weakest
   neutral lines that emit predominantly in the periphery).
2. Use spatially resolved spectroscopy (Abel inversion of imaging data)
   if the experimental setup supports it. CF-LIBS does not currently
   ship a multi-zone solver; the model is `SingleZoneLTEPlasma` by name.
3. Use temporal gating to suppress one zone preferentially (e.g. early
   gates favor the hot core).

### Code references

- `cflibs/plasma/state.py:SingleZoneLTEPlasma`.
- `quality_metrics["temperature_consistency"]` (per-element agreement).
- `quality_metrics["saha_boltzmann_consistency"]` (neutral vs ion).

---

## 4. Temporal Stationarity

### What it states

Plasma parameters do not evolve significantly over the spectrometer gate
window. The forward model uses a single `(T, n_e, {C_s})`.

### What it requires

The plume cooling timescale should be long compared with the gate. In
practice, `n_e` halves on a microsecond timescale, so a 1 µs gate after
a 1 µs delay sees significant evolution.

### Failure mode

The recovered parameters are time-averages weighted by emission. If `T`
drops by 30 % across the gate, the inferred `T` is some weighted mean
with the high-`T` early times getting larger weight (their lines are
brighter).

### Options

- **Shorter gates**, accepting the SNR cost.
- **Compare multiple gate delays** to verify the recovered parameters
  are stable. Large drift means the measurement is gate-dependent.
- The runtime sub-package (`cflibs/inversion/runtime/`) includes a
  `temporal_gate_optimization` helper that suggests gate windows for a
  given LTE budget.

---

## 5. Closure Completeness

### What it states

In `closure_mode: standard`, all elements with `C_s > 1 %` are in the
candidate list. The unmodeled mass is small enough that
`Σ C_s ≈ 1` is a good normalization.

### Failure mode

If you forget an element that contributes 10 % mass, the standard
closure normalizes the remaining elements so their fractions sum to
unity — silently inflating each by ~10 %. The recovered ratios are still
correct *relative to each other*, but absolute fractions are biased.

### Diagnostic and recovery

Check the per-element residual (forward-model spectrum vs. observed
spectrum) over wavelength bands: a missed major element shows up as a
forest of unmodeled lines. Re-run with the missing elements added.

For samples where you cannot enumerate elements a priori, use:

- `closure_mode: matrix` with a known matrix-element fraction, anchoring
  the absolute scale.
- Hybrid identification (`cflibs.inversion.identify.hybrid`) to propose
  the candidate list before inversion.

---

## 6. Atomic Data Accuracy

### What it states

The Einstein `A_ki`, statistical weights `g`, energies `E_k`, and
ionization potentials `χ_z` in the database are correct. CF-LIBS uses
NIST ASD as primary source.

### Failure mode

A 20 % error in `A_ki` for one line shows up as a 20 % bias in that
line's intensity prediction, contaminating the Boltzmann plot intercept
for that element. Errors >2× exist in NIST for some transitions of
heavy/poorly-studied elements.

### Diagnostic and mitigation

- Lines with `A_ki` rated `D`/`E` in NIST should be filtered. The
  database does not currently do this automatically; users can filter
  via the `min_relative_intensity` knob, which acts as a proxy.
- The validation suite (`scripts/validate_nist_parity.py`) cross-checks
  forward-model predictions against NIST reference spectra.
- For high-precision work, supplement with element-specific compilations
  (Kurucz, OP) where they exist.

---

## 7. Energy Spread of Selected Lines

### What it states

The Boltzmann plot slope `−1/(k_B T)` is well-conditioned only when the
selected lines span a meaningful range in `E_k`. The configured minimum
is `min_energy_spread_ev: 2.0` eV.

### Failure mode

If lines cluster in a narrow `E_k` window, the slope is determined by
noise rather than signal. Temperature uncertainty balloons; the solver
may fail to converge. With `min_lines_per_element: 3` and
`min_energy_spread_ev: 2.0`, the solver enforces a minimum lever arm.

### Mitigation

- Widen the spectrometer wavelength range to capture lines from higher
  excited states.
- Lower `min_lines_per_element` only as a last resort and inspect the
  Boltzmann plot R² afterwards.

---

## 8. Spectral Isolation

### What it states

Each measured line peak corresponds to one transition of one element.
The line-detection step is a peak finder with a tolerance window
(`wavelength_tolerance_nm`). Two close transitions inside the same
window are treated as a single line, biasing the integrated intensity.

### Failure mode

Blended lines inflate intensities. If the blend partner is from the same
element and ionization stage, the bias mostly cancels in the Boltzmann
plot intercept; if not, both elements' intercepts are corrupted.

### Diagnostic

The `isolation_wavelength_nm` parameter (default 0.1 nm) excludes lines
whose nearest neighbor in the database is closer than this. Lower this
threshold when you have higher resolution; raise it for low-resolution
spectrometers.

---

## 9. Instrument Line-Spread Function

### What it states

The instrument response convolves the true spectrum with a Gaussian
(or Voigt). CF-LIBS uses a Gaussian by default in two modes:
fixed-FWHM or constant resolving-power.

### Failure mode

Real spectrometers can have asymmetric or boxy line shapes (slit
diffraction, detector pixel response). Treating an asymmetric profile
as Gaussian biases the recovered widths and can leak intensity to
neighboring channels.

### Mitigation

- Calibrate with a known atomic line lamp (e.g. Hg, Ar) and verify the
  measured FWHM matches the configured `resolution_fwhm_nm`.
- For echelle spectrometers, use `from_resolving_power(R)` instead of a
  fixed FWHM so the wavelength dependence of the slit function is
  modeled.

---

## 10. Continuum and Background Subtraction

### What it states

The line-detection and intensity-integration steps assume the baseline
has been removed. Otherwise, integrated intensities pick up the local
continuum bias.

### Failure mode

Bremsstrahlung continuum and detector dark current both add a wavelength-
dependent floor. Lines on top of a rising background appear stronger
than they are.

### Mitigation

- Use `cflibs.inversion.preprocess` baseline-removal utilities (rolling
  ball, asymmetric least squares, polynomial detrending) before passing
  the spectrum to the analyzer.
- For very high-quality work, gate selection is the bigger lever:
  bremsstrahlung continuum decays faster than line emission, so a delay
  of 1 µs typically buys a clean baseline at the cost of overall SNR.

---

## What CF-LIBS Does NOT Currently Model

The shipped algorithm is single-zone LTE optically thin with optional
self-absorption correction. It does not model:

- **Non-LTE level kinetics** (no collisional-radiative model with
  rate equations).
- **Multi-zone plasmas** (no Abel inversion, no spatial zone fitting).
- **Self-reversal** (cool absorbing periphery in front of a hot emitting
  core).
- **Stark shift of line centers** (only Stark widths are used).
- **Pressure broadening from neutral collisions** (van der Waals,
  resonance broadening) — usually negligible at LIBS densities, but
  not always for hydrogen and the alkalis.
- **Molecular emission bands** (CN, C₂, OH, etc.).
- **Continuum emission** (bremsstrahlung, recombination) — assumed
  removed in preprocessing.
- **Polarization** of emission.

If your problem needs any of the above, the calibration-free framework
is not the right tool, or the shipped model needs to be extended. The
extension points are documented in
[Codebase Architecture](../reference/Codebase_Architecture.md).

---

## How to Audit a CF-LIBS Result

A CF-LIBS measurement that you would defend in a paper should pass:

1. **McWhirter** satisfied (`lte_mcwhirter_satisfied: true`).
2. **Common-slope Boltzmann fit** has `R² ≥ 0.95` per element.
3. **Per-element temperatures** agree within 10 %.
4. **Neutral / ion temperatures** agree within 10 % after Saha
   correction (`saha_boltzmann_consistency`).
5. **Reduced χ² of forward-model fit** is `≈ 1` (in Bayesian or hybrid
   inversion mode).
6. **Composition** is stable across two distinct gate windows.
7. **NIST parity validator** passes for the wavelength range and elements
   you used (`scripts/run_nist_validation.py`).

The default analysis output already prints McWhirter and reports the
quality metrics object. The remaining checks are a few lines of Python
on the `CFLIBSResult` and are documented in the
[User Guide § Diagnostics](../user/User_Guide.md#inversion-diagnostics).

---

## See Also

- [Equations](Equations.md) — equations CF-LIBS evaluates.
- [Inversion Algorithm](Inversion_Algorithm.md) — solver step-by-step.
- [Quick Start: Real Data](../user/Quick_Start_Real_Data.md) §
  "When it does not converge".
- [User Guide § Diagnostics](../user/User_Guide.md#inversion-diagnostics).
