# Stark Broadening as Electron-Density Diagnostic in LIBS

## 1. Governing Equations — Canonical Form

### 1.1 Physical Basis

In a dense plasma, free electrons and ions exert quasi-static electric microfields on emitting atoms, shifting and splitting energy levels (Stark effect). For most atoms heavier than hydrogen (non-hydrogenic, quadratic Stark effect) the perturbation is proportional to E², giving a Lorentzian line shape dominated by electron-impact broadening, which scales **linearly with electron density** n_e. Hydrogen and hydrogen-like ions experience the linear Stark effect (proportional to E) and require separate treatment.

### 1.2 Non-Hydrogenic Lines: Griem Semiclassical Formula

**Canonical FWHM expression (Griem 1964/1974):**

```
Δλ_S [Å] = 2w(T) · (n_e / n_ref)
           + 3.5 · A(T) · (n_e / n_ref)^(1/4) · [1 − 3/(4 N_D^(1/3))] · w(T)
```

where:

| Symbol | Definition | Units |
|--------|-----------|-------|
| Δλ_S   | Stark FWHM (full width at half maximum) | Å |
| w(T)   | electron-impact half-half-width at reference density; a.k.a. electron impact width parameter | Å |
| A(T)   | ion-broadening parameter (dimensionless, tabulated alongside w) | — |
| n_e    | plasma electron number density | cm⁻³ |
| n_ref  | reference electron density | cm⁻³ |
| N_D    | number of particles in the Debye sphere = (4π/3) λ_D³ n_e | — |

**Reference densities used in standard tables:**
- Neutral atom lines (e.g., Mg I, Al I, Si I, Ca I): **n_ref = 10¹⁶ cm⁻³**
- Singly ionized lines (e.g., Ca II, Mg II): **n_ref = 10¹⁷ cm⁻³**

This convention is used by the GRIEM database (griem.obspm.fr), Griem (1974), and most subsequent tabulations. The STARK-B database (stark-b.obspm.fr) uses a wider density range and lists w directly in Å.

**Term anatomy:**
- First term `2w(n_e / n_ref)`: electron-impact (collisional) broadening — always present, Lorentzian.
- Second term `3.5A(...)w(...)`: quasi-static ion broadening correction — can often be neglected when `A ≪ w` or at moderate density. The factor `[1 − 3/(4 N_D^(1/3))]` is the Debye-shielding correction; as N_D → ∞ (dilute plasma) this approaches 1.0.

**Temperature dependence of w(T) and A(T):**

w(T) varies weakly with T (often as T^(−1/4) to T^(−1/2) for impact broadening in the semiclassical limit, but direction depends on transition). Tabulated values are provided at multiple T in the GRIEM and STARK-B databases (typically 5,000–80,000 K). Always interpolate from the table rather than assuming T-independence; errors of 10–30% result from using w at a wrong temperature.

**Scaling to arbitrary n_e:**

Since w is tabulated at n_ref, to find n_e from a measured Δλ_S:

```
n_e ≈ (Δλ_S / (2w)) · n_ref          [dominant, when ion term is small]
```

When the ion term is non-negligible, solve iteratively: compute N_D from an initial n_e estimate, evaluate the full expression, update n_e.

### 1.3 Hydrogen / Hydrogen-Like Lines (Linear Stark Effect)

Hydrogen lines (Balmer series) undergo linear Stark effect; the profile shape is more complex (not purely Lorentzian) and depends on both n_e and T. Three theoretical frameworks are in use:

#### Vidal–Cooper–Smith (VCS) tables (1973)
Provides tabulated reduced-half-widths α₁/₂ for Balmer lines. The NIST compendium form:

```
Δλ_{1/2}^{S,H} [Å] = (2.50 × 10⁻⁹) · α₁/₂(n_e, T) · n_e^(2/3)
```

where n_e is in cm⁻³ and α₁/₂ is tabulated for each line over n_e = 10¹⁵–10¹⁸ cm⁻³ and T = 5,000–30,000 K.

Note: this n_e^(2/3) scaling is approximate; the true dependence encoded in α₁/₂ varies somewhat with n_e and T.

#### Gigosos & Cardeñoso (1996) / Gigosos et al. (2003) computer-simulation tables

Preferred for modern LIBS work because they incorporate ion-dynamics effects naturally. They provide FWHM as a function of n_e and T including reduced mass μ of the perturber–emitter system.

Power-law fits to Gigosos simulation data (from the ar5iv.labs review of arXiv:2201.08783):

```
Hα (656.28 nm):  Δλ_{Hα} [nm] ≈ 1.3 · (n_e / 10¹⁷)^(0.64 ± 0.03)
Hβ (486.13 nm):  Δλ_{Hβ} [nm] ≈ 4.5 · (n_e / 10¹⁷)^(0.71 ± 0.03)
```

n_e in cm⁻³. Valid ranges: Hα: 10¹⁵–10¹⁹ cm⁻³; Hβ: 3×10¹⁵–8×10¹⁸ cm⁻³.

**Inversion for n_e from Hβ FWHM (Gigosos-based):**

```
n_e [cm⁻³] = 10¹⁷ · (Δλ_{Hβ,Stark} [nm] / 4.5)^(1/0.71)
```

#### Konjevic et al. empirical formula (recommended for LIBS Hβ)

Multiple experimental calibrations (e.g., Konjevic & Roberts 1976; Pérez et al. 1991; Dems & Konjevic) have produced:

```
n_e [cm⁻³] ≈ C · (Δλ_{Hβ,Stark} [Å])^1.4
```

with C ≈ 3.85 × 10¹⁵ (calibrated against Thomson scattering at low–moderate densities).

**Hα vs. Hβ trade-offs in LIBS:**
- Hα (656.28 nm): stronger signal, but saturates at high n_e and suffers severe self-absorption in H-rich targets.
- Hβ (486.13 nm): weaker but more linear; preferred for quantitative n_e above ~10¹⁶ cm⁻³.

### 1.4 Line Shift (Stark Shift)

The center wavelength of Stark-broadened lines also shifts:

```
Δλ_shift [Å] = D(T) · (n_e / n_ref) ∓ 2A · (n_e / n_ref)^(1/4) · [1 − 3/(4N_D^(1/3))] · w
```

where D(T) is the electron-impact shift parameter (tabulated alongside w). The shift must be subtracted from the peak position before using the line for spectral identification or wavelength calibration. Typical Stark shifts in LIBS plasmas (n_e ~ 10¹⁷ cm⁻³) are 0.01–0.1 Å, comparable to or larger than the line center accuracy needed for elemental identification.

---

## 2. Deconvolving Stark Lorentzian from Doppler + Instrument Voigt

The observed spectral profile is a convolution of multiple broadening mechanisms:

| Mechanism | Profile type | FWHM |
|-----------|-------------|------|
| Instrumental | Gaussian (usually) | Δλ_inst (measured from calibration lamp) |
| Doppler (thermal) | Gaussian | Δλ_D = λ_0 · (8 ln 2 · k_B T / (M c²))^(1/2) |
| Natural linewidth | Lorentzian | Δλ_nat (negligible in LIBS, ~fm) |
| Stark | Lorentzian | Δλ_S (dominant Lorentzian term in LIBS) |
| van der Waals / resonance | Lorentzian | Small at low pressures; often neglected |

Since the convolution of two Gaussians is Gaussian and the convolution of two Lorentzians is Lorentzian, the combined observed profile is approximately **Voigt**:

```
Δλ_G_total = sqrt(Δλ_inst² + Δλ_D²)       [Gaussian component]
Δλ_L_total = Δλ_S + Δλ_nat + Δλ_vdW       [Lorentzian component, ≈ Δλ_S]
```

### 2.1 Voigt FWHM — Thompson–Cox–Hastings (1987) Approximation

The Voigt total FWHM f_V is related to Gaussian f_G and Lorentzian f_L by the Thompson 5th-power formula:

```
f_V^5 = f_G^5 + 2.69269·f_G^4·f_L + 2.42843·f_G^3·f_L²
       + 4.47163·f_G^2·f_L³ + 0.07842·f_G·f_L^4 + f_L^5
```

Accuracy: ±0.02%.

**Simpler Olivero–Longbothum (1977) approximation:**

```
f_V ≈ 0.5343·f_L + sqrt(0.2169·f_L² + f_G²)
```

Accuracy: ±0.02%. This is the formula commonly used for LIBS line-shape inversion.

### 2.2 Extracting Stark FWHM from Measured Voigt

**Procedure:**

1. Measure instrumental FWHM Δλ_inst from a narrow calibration source (e.g., pencil lamp, Hg lamp line) at the wavelength of interest.
2. Estimate Doppler FWHM Δλ_D from temperature T (from Boltzmann plot or simultaneous fit).
3. Form total Gaussian width: `f_G = sqrt(Δλ_inst² + Δλ_D²)`.
4. Fit the observed line with a Voigt profile to extract f_V (or fit pseudo-Voigt to get both f_G and f_L simultaneously).
5. Solve for f_L (= Δλ_S) using the Olivero–Longbothum inversion or the Thompson formula numerically.
6. Apply n_e formula for the chosen line.

**Important: Doppler width in LIBS is usually small.** For metal lines (M ~ 50 amu) at T ~ 10,000 K:

```
Δλ_D [nm] ≈ 7.16 × 10⁻⁷ · λ_0[nm] · sqrt(T[K] / M[amu])
           ≈ 0.001–0.005 nm for λ ~ 300–500 nm
```

This is typically smaller than the instrument function (~0.02–0.1 nm for most grating spectrometers) and much smaller than Stark broadening at LIBS densities (~0.05–1 nm at n_e ~ 10¹⁷ cm⁻³). The Doppler contribution may be ignored for heavy species; it matters for H and light elements.

---

## 3. Key References

1. **Griem, H.R. (1964).** *Plasma Spectroscopy.* McGraw-Hill, New York.
   — Original semiclassical theory; defines w, A, D parameters and the FWHM formula for non-hydrogenic lines.

2. **Griem, H.R. (1974).** *Spectral Line Broadening by Plasmas.* Academic Press, New York.
   — Extended tables of w(T), A(T), D(T) at n_ref = 10¹⁶ (neutrals) and 10¹⁷ cm⁻³ (ions); still primary reference for parameter tables.

3. **Vidal, C.R., Cooper, J., & Smith, E.W. (1973).** Hydrogen Stark-broadening tables. *Astrophysical Journal Supplement Series*, **25**, 37–136. https://tf.nist.gov/general/pdf/407.pdf
   — VCS tables providing α₁/₂ parameters for H Balmer lines; basis for NIST hydrogen Stark-broadening formulas.

4. **Gigosos, M.A., & Cardeñoso, V. (1996).** New plasma diagnosis tables of hydrogen Stark broadening including ion dynamics. *Journal of Physics B*, **29**, 4795–4838. DOI: 10.1088/0953-4075/29/20/029
   — Computer simulation tables for Hα, Hβ, Hγ; accounts for ion dynamics; the modern standard for hydrogen LIBS diagnostics.

5. **Gigosos, M.A., González, M.Á., & Cardeñoso, V. (2003).** Computer simulated Balmer-alpha, -beta and -gamma Stark line profiles for non-equilibrium plasmas diagnostics. *Spectrochimica Acta Part B*, **58**, 1489–1504. DOI: 10.1016/S0584-8547(03)00097-1
   — Extended simulations including non-equilibrium (two-temperature) cases; provides diagnostic maps for n_e vs. T.

6. **Konjevic, N. (1999).** Plasma broadening and shifting of non-hydrogenic spectral lines: present status and applications. *Physics Reports*, **316**, 339–401. DOI: 10.1016/S0370-1573(98)00132-X
   — Comprehensive review of experimental measurements and theory for non-hydrogenic lines; recommended procedure for LIBS n_e diagnostics; covers accuracy and error sources.

7. **Olivero, J.J., & Longbothum, R.L. (1977).** Empirical fits to the Voigt line width: a brief review. *Journal of Quantitative Spectroscopy and Radiative Transfer*, **17**, 233–236. DOI: 10.1016/0022-4073(77)90161-3
   — Canonical reference for the f_V ≈ 0.5343·f_L + sqrt(0.2169·f_L² + f_G²) approximation.

8. **Thompson, P., Cox, D.E., & Hastings, J.B. (1987).** Rietveld refinement of Debye–Scherrer synchrotron X-ray data from Al₂O₃. *Journal of Applied Crystallography*, **20**, 79–83.
   — Defines the Thompson–Cox–Hastings 5th-power Voigt FWHM approximation; ±0.02% accuracy.

9. **Surmick, D.M., & Parigger, C.G. (2014).** Empirical formulae for electron density diagnostics from H_α and H_β Stark broadening. *International Review of Atomic and Molecular Physics*, **5(2)**, 73–81.
   — Practical power-law fitting of Gigosos tables to get n_e directly from measured Hα or Hβ FWHM; recommended for LIBS.

10. **Kunze, H.-J. (2009).** *Introduction to Plasma Spectroscopy.* Springer. ISBN 978-3-642-02232-9.
    — Graduate textbook; Chapter 5 derives Stark broadening theory from first principles; Chapter 8 covers diagnostic applications including LIBS; excellent on error analysis and deconvolution.

---

## 4. Common Implementation Pitfalls and Correct Treatment

### Pitfall 1: Wrong reference density
**Error:** Using n_ref = 10¹⁷ cm⁻³ for a neutral atom line whose table uses n_ref = 10¹⁶ cm⁻³, or vice versa.
**Effect:** Factor-of-10 error in n_e.
**Fix:** Check the database entry header. GRIEM database and Griem (1974): n_ref = 10¹⁶ for neutrals (I), 10¹⁷ for singly ionized (II).

### Pitfall 2: HWHM vs. FWHM confusion with w
**Error:** The tabulated w(T) is a **half-half-width** (i.e., half of the Lorentzian HWHM, or quarter-width). In Griem's notation, the Lorentzian FWHM from electron impact alone is **2w** (the factor of 2 is explicit in the formula).
**Effect:** Factor-of-2 error in n_e.
**Fix:** Always write Δλ_S = 2w·(n_e/n_ref) + ion_term; never omit the factor of 2.

### Pitfall 3: Ignoring temperature dependence of w(T)
**Error:** Using w at one temperature for a plasma where T may differ by 2–3×.
**Effect:** Errors of 20–50% in n_e.
**Fix:** Use the T from a simultaneous Boltzmann plot to select the correct w(T) from the table, interpolating between tabulated temperatures.

### Pitfall 4: Not subtracting Stark shift before fitting
**Error:** Fitting the Lorentzian center without correcting for Stark shift, then using the shifted center for wavelength identification.
**Effect:** Mistaken line identification; systematic wavelength offset.
**Fix:** Fit the full profile (center + width); note the shift D(T); subtract D from the peak wavelength before cross-matching with the atomic database.

### Pitfall 5: Ignoring instrumental broadening
**Error:** Taking the measured FWHM directly as Δλ_S.
**Effect:** Overestimates n_e, especially when Δλ_inst ≳ Δλ_S.
**Fix:** Always subtract instrumental (and Doppler) Gaussian contribution via the Voigt deconvolution (Olivero–Longbothum or Thompson formula) before applying the Stark formula.

### Pitfall 6: Applying the non-hydrogenic formula to H lines (or vice versa)
**Error:** Using the linear `2w·(n_e/n_ref)` formula for Hα or Hβ, which follow a sublinear power law (~n_e^0.65–0.71).
**Effect:** Highly inaccurate n_e (errors >50% at n_e outside calibration point).
**Fix:** For hydrogen lines, use Gigosos tables or the empirical power-law fits; for non-hydrogenic lines, use the Griem/Konjevic tables.

### Pitfall 7: Self-absorption distorting Hα Stark width
**Error:** Using Hα width for n_e in H-rich or dense plasmas where self-absorption narrows the apparent profile.
**Effect:** Underestimated Stark width → underestimated n_e.
**Fix:** Check Hα/Hβ intensity ratio against optically thin prediction. Prefer Hβ when self-absorption is suspected; it is typically optically thinner than Hα.

### Pitfall 8: Ion broadening term applied with wrong sign / Debye factor
**Error:** Applying the ion broadening correction with incorrect sign or without the Debye factor [1 − 3/(4 N_D^(1/3))].
**Effect:** Ion broadening term is overestimated at very high n_e (when N_D is small).
**Fix:** Compute N_D = (4π/3)·λ_D³·n_e with λ_D = sqrt(ε_0 k_B T / (n_e e²)) in CGS; apply the correction faithfully. In practice, for LIBS at n_e ~ 10¹⁷ cm⁻³, T ~ 10,000 K, N_D ~ 10–100 and the correction is ~10–30%.

### Pitfall 9: Using lines with significant van der Waals / resonance broadening
**Error:** Applying Stark formula to low-lying resonance lines (e.g., Ca II 393 nm, Mg II 280 nm) at late LIBS times when neutral gas pressure is high.
**Effect:** Non-Stark Lorentzian broadening mimics electron-density signal.
**Fix:** Use lines insensitive to pressure broadening (high-excitation non-resonance lines) or account for van der Waals width separately. Apply at early gating times when plasma is still hot and Stark term dominates.

---

## 5. What Correct Code MUST Do (Checklist)

- [ ] **w(T) interpolation:** Look up electron-impact width w at the measured plasma temperature T by interpolating from the reference table (e.g., GRIEM database, Griem 1974, STARK-B). Do NOT use a single fixed T.
- [ ] **Correct factor of 2:** Compute Stark FWHM as `Δλ_S = 2 * w * (n_e / n_ref)` (plus optional ion term). w is a half-half-width; the factor of 2 is mandatory.
- [ ] **Correct n_ref per line type:** Use n_ref = 1e16 cm⁻³ for neutral lines, 1e17 cm⁻³ for singly ionized lines (match whichever convention the table uses).
- [ ] **Ion broadening correction:** When A is available and n_e is known, apply the `3.5·A·(n_e/n_ref)^(1/4)·[1 − 3/(4·N_D^(1/3))]·w` correction term. Compute N_D consistently.
- [ ] **Voigt deconvolution before using FWHM:** Subtract instrumental (Gaussian) broadening via `f_G = sqrt(Δλ_inst² + Δλ_D²)` and then solve Olivero–Longbothum or Thompson formula to isolate Lorentzian f_L = Δλ_S from measured Voigt f_V.
- [ ] **Doppler width computation:** Compute Δλ_D = (λ_0/c)·sqrt(8·ln(2)·k_B·T/M) for completeness even if small; include in total Gaussian.
- [ ] **Hydrogen lines use Gigosos/VCS tables, not linear formula:** For Hα/Hβ, call into the simulation tables (or use calibrated power-law fits); do not apply the non-hydrogenic linear formula.
- [ ] **Stark shift correction:** Retrieve D(T) from the same table as w(T); subtract the predicted Stark shift from the line center before wavelength matching.
- [ ] **Self-absorption check:** Flag Hα/Hβ ratio against expected optically thin value; warn if self-absorption may compromise Hα width.
- [ ] **Uncertainty reporting:** Propagate w-table uncertainty (~10–30% from theory/experiment scatter per Konjevic 1999) plus T uncertainty (via dΔλ/dT) into the final n_e uncertainty.
