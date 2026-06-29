# CF-LIBS Method: Canonical Formulation and Key References

**Topic:** Calibration-Free LIBS (Ciucci 1999 / Tognoni 2010 / Aragón-Aguilera): Boltzmann plot,
Saha-Boltzmann plane, one-point/closure, internal reference; common-slope multi-element fits;
matrix/oxide closure.

**Note on sources:** Web searches and HTML full-texts were used. PDFs were binary-only and
unreadable by WebFetch; the equations below are drawn from accessible HTML versions of primary
papers and reviews (Frontiers 2022 review, RSC 2023 review, RSC 2024 Völker, A&A 2018 Ferus,
RSC Advances 2026 self-absorption paper) plus search-result abstracts of the Ciucci 1999,
Tognoni 2010, Aguilera/Aragón 2007, and Cristoforetti 2010 papers. All are cross-checked against
multiple independent sources.

---

## 1. Governing Equations — Canonical Forms

### 1.1 Fundamental Line Intensity (Optically Thin, LTE, Homogeneous Plasma)

The integrated intensity of emission line k→i for species s (neutral or ion) is:

```
I_{ki} = F · (hc / 4π λ_{ki}) · C_s · (A_{ki} g_k) / U_s(T) · exp(-E_k / k_B T)
```

or equivalently (using frequency ν_{ki}):

```
I_{ki} = F · C_s · A_{ki} g_k · (h ν_{ki} / 4π) / U_s(T) · exp(-E_k / k_B T)
```

**Symbol definitions:**

| Symbol | Meaning | Canonical units |
|--------|---------|-----------------|
| I_{ki} | Integrated spectral line intensity | W sr⁻¹ m⁻³ (or arbitrary, internally consistent) |
| F | Experimental factor: optical collection efficiency × plasma column density × 1/(4π sr) geometric factor | depends on normalization |
| h | Planck constant | J·s |
| c | Speed of light | m/s |
| λ_{ki} | Transition wavelength (vacuum) | m (or nm consistently) |
| C_s | Number fraction (mole fraction) of emitting species s | dimensionless, Σ C_s = 1 |
| A_{ki} | Einstein spontaneous-emission coefficient (transition probability) | s⁻¹ |
| g_k | Statistical weight (degeneracy) of upper level k: g_k = 2J_k + 1 | dimensionless |
| U_s(T) | Partition function of species s at temperature T: U_s(T) = Σ_i g_i exp(-E_i/k_B T) | dimensionless |
| E_k | Energy of upper level k (from ground state of same species) | eV or cm⁻¹ (be consistent with k_B T units) |
| k_B | Boltzmann constant | eV/K if E in eV, or J/K if E in J |
| T | Plasma (excitation) temperature | K |

**Critical notes on F:**
- F absorbs: detector quantum efficiency, spectrometer throughput, fiber/lens solid angle, plasma
  volume, and the 1/(4π) steradians factor from isotropic emission.
- F is the SAME for all lines in a single spectrum acquisition (same optics, same plasma event).
- F is determined by the closure equation (Sec. 1.4); it is NOT measured independently.

**Alternative notation used in some papers:** Some papers write this as
`I_{ki} = F_s · N_s · (A_{ki} g_k / U_s(T)) · exp(-E_k / k_B T)` where F_s absorbs hc/4πλ and
optical factors, and N_s is species number density. In practice, F_s = F (same for all s) in an
optically thin, stoichiometric-ablation plasma. Verify which formulation your code uses.

---

### 1.2 Boltzmann Plot Linearization

Taking the natural log of eq. (1.1):

```
ln( I_{ki} λ_{ki} / (A_{ki} g_k) ) = -E_k / (k_B T)  +  ln( F C_s h c / (4π U_s(T)) )
```

**Y-axis:**  `y = ln( I_{ki} λ_{ki} / (A_{ki} g_k) )`

**X-axis:**  `x = E_k`   (upper-level energy, eV or cm⁻¹)

**Slope:**   `m = -1 / (k_B T)`  →  `T = -1 / (k_B m)`

**Intercept (y-intercept, at E_k = 0):**
```
b_s = ln( F C_s h c / (4π U_s(T)) )
```

From which: `C_s = (U_s(T) / (F h c / 4π)) · exp(b_s)`

**Key properties under ideal LTE:**
- All lines from the SAME species (same ionization stage) lie on a SINGLE straight line.
- All species present yield the SAME slope (same temperature T).
- Different species (different s) have parallel lines, offset by their different ln(F C_s / U_s(T)).

**Common notation variant:** some papers write y = ln(I/gA) (absorbing λ into F or using
wavenumber units where ν̃ = 1/λ). Always check what the y-axis actually contains.

---

### 1.3 Saha-Eggert Equation and the Saha-Boltzmann Plane

The Saha-Eggert equation gives the ratio of singly-ionized (z=1) to neutral (z=0) number density:

```
N_{s,1} / N_{s,0} = (2 / n_e) · (U_{s,1}(T) / U_{s,0}(T)) · (2π m_e k_B T / h²)^(3/2) · exp(-χ_s / k_B T)
```

Where:
- N_{s,1}, N_{s,0}: number densities of ion and neutral of element s
- n_e: electron number density (cm⁻³ or m⁻³; be consistent)
- U_{s,1}(T), U_{s,0}(T): partition functions of ion and neutral
- m_e: electron mass
- χ_s: first ionization energy of element s (eV; reduced by Δχ for plasma screening, usually small)
- Factor of 2: electron spin degeneracy (from electron partition function = 2)

**The (2πm_e k_B T/h²)^(3/2) factor in CGS:** = 4.829 × 10¹⁵ T^(3/2) cm⁻³ (Griem 1964)

**Mapping ionic lines onto the Saha-Boltzmann (neutral) plane:**

For an ionic line (z=1) of element s with upper-level energy E_k^(ion) above the ion ground state,
the combined Boltzmann + Saha treatment gives:

```
ln( I_{ki}^(ion) λ_{ki} / (A_{ki} g_k) )  +  correction_term
     = -(E_k^(ion) + χ_s) / (k_B T)  +  intercept_term
```

where `correction_term = ln( n_e / 2 · (h²/(2π m_e k_B T))^(3/2) )` captures the Saha factor.

The effective x-axis energy for ionic lines on the combined Saha-Boltzmann plot is:
```
E_k^(eff) = E_k^(ion) + χ_s
```

This "shifts" ionic line data to higher energies so they can be plotted on the same axes as neutral
lines, still yielding the same temperature slope (−1/k_B T).

**Aguilera & Aragón (2007) multi-element Saha-Boltzmann plot:** All lines (neutrals and ions, all
elements) appear on a single plot. Temperature is determined from the common slope. Concentrations
of each element are determined from the parallel intercepts. This is the canonical ME-SB approach.

---

### 1.4 Closure Equation and the F Factor

The sum of mole fractions of ALL emitting species must equal unity:

```
Σ_{s} C_s = 1
```

where the sum runs over ALL elements detected (each summed over ionization stages via Saha if both
are measured). Since `C_s = (U_s(T) · exp(b_s)) / (F · h c / 4π)`, setting the sum to 1 gives:

```
F = (h c / 4π) · Σ_s (U_s(T) · exp(b_s)) / 1
```

More compactly: once the Boltzmann-plot intercepts b_s are fit for each species, F is determined
by:
```
F = Σ_s [ U_s(T) · exp(b_s) / (h c / 4π) ]
```

and each concentration follows:
```
C_s = U_s(T) · exp(b_s) / (F · h c / 4π)
```

**Mole fraction vs. mass fraction:** CF-LIBS yields MOLE fractions (number fractions). Conversion
to mass (weight) fractions requires:
```
w_s = (C_s · M_s) / Σ_j (C_j · M_j)
```
where M_s is the molar mass (g/mol). Omitting this conversion causes systematic errors proportional
to the deviation from average molar mass — errors up to 353% reported for carbon in steel
(Völker 2024). This is one of the most frequently omitted steps in published CF-LIBS code.

---

### 1.5 Iterative CF-LIBS Algorithm (Ciucci 1999 / Tognoni 2010 standard)

1. **Peak identification and integration:** Measure I_{ki} for a set of spectral lines; identify
   species (s, k→i), extract A_{ki}, g_k, E_k from atomic databases (NIST ASD or Kurucz).

2. **Initial temperature estimate:** Construct a Boltzmann plot for a single element with many
   lines spanning a wide E_k range (often Fe I or Fe II). Fit slope → T_0.

3. **Electron density n_e:** Measure Stark broadening of a suitable line (e.g., Hα 656 nm FWHM,
   or a known Stark-broadened metal line). Or estimate from Saha self-consistency. Typical LIBS
   values: 10^16–10^17 cm⁻³.

4. **Multi-element Saha-Boltzmann plot:** For all detected lines (neutral + ion), plot y_i vs
   x_i with ionic lines shifted by χ_s. Fit a common slope → refined T. Fit element intercepts
   b_s.

5. **Concentrations from intercepts:** Use closure (Sec. 1.4) to determine F and all C_s.

6. **Saha correction loop:** Update ionic/neutral concentration ratio using new T and n_e. Check
   charge balance (n_e = Σ_s C_s z_s N_total). Refit. Repeat until T and n_e converge (typically
   3–10 iterations).

7. **Output:** T, n_e, {C_s} as mole fractions → convert to mass fractions if needed.

---

### 1.6 One-Point Recalibration (Internal Reference / OPC Method)

When only one element has a known concentration C_ref (internal reference or one-point calibration):

1. Determine T from Boltzmann plot of the reference element.
2. Use known C_ref to fix F from the reference element intercept alone: F = U_ref(T)·exp(b_ref) / (C_ref · hc/4π).
3. All other concentrations follow from their Boltzmann-plot intercepts and this F.

This relaxes the full-closure requirement but still requires LTE and optically-thin conditions.
The "one-point Saha-Boltzmann" variant (e.g., Yalcin et al. 1999) uses one reference line of
each ionization stage to cross-calibrate T and n_e simultaneously.

---

### 1.7 Oxide / Matrix Closure for Non-Metal Samples

For geological/mineral samples (rocks, soils, ceramics), oxygen is ubiquitous but LIBS lines
for O are weak or absent from the main spectrum window. Common approaches:

1. **Oxide closure model:** Assume all detected metals M exist as their most stable oxide (SiO₂,
   Al₂O₃, Fe₂O₃, MgO, CaO, TiO₂, etc.). Mass balance: Σ_{oxides} w_oxide = 1, where
   w_oxide = w_M · (M_oxide / M_M). This over-constrains the system but enables oxygen-inclusive
   normalization.

2. **Stoichiometric correction:** Known oxide stoichiometry fixes n_O / n_M ratios; the closure
   then sums elemental + calculated oxygen. This was used in ChemCam / SuperCam (Mars) CF-LIBS
   (Wiens et al. 2013, Clegg et al. 2017).

3. **Mole-fraction oxide closure variant:** Σ C_s = 1 includes C_O calculated from oxide
   stoichiometries. This changes the denominator of the closure and must be handled consistently.

---

### 1.8 McWhirter Criterion for LTE

**Necessary condition (McWhirter 1962):**
```
n_e ≥ 1.6 × 10¹² · T^(1/2) · (ΔE)³   [cm⁻³]
```
where T is in K and ΔE is the energy gap of the most sensitive transition (eV), typically the
resonance-line energy of the dominant species.

**Important:** This criterion is necessary but NOT sufficient for transient, inhomogeneous LIBS
plasmas. Cristoforetti et al. (2010) showed that an additional relaxation time condition must
hold: the plasma relaxation time must be short relative to the observation time (τ_relax << t_obs).

Typical LIBS plasma at ~10,000–20,000 K requires n_e ≥ 10^16–10^17 cm⁻³ for LTE validity;
this is usually satisfied at early delays (< 2 µs) but the plasma may be still optically thick.
The trade-off: early times → LTE but self-absorption; late times → optically thin but non-LTE.

---

## 2. Common Implementation Pitfalls and Correct Treatment

### P1: Wrong y-axis in Boltzmann plot
- **Wrong:** ln(I / (g_k A_{ki})) — omits λ_{ki}
- **Correct:** ln(I λ_{ki} / (g_k A_{ki})) when using the hc/λ form of the energy.
- **Why it matters:** Omitting λ introduces a systematic energy-dependent bias in both the slope
  (temperature error) and intercepts (concentration error), especially if lines span a wide
  wavelength range.

### P2: Energy units inconsistency
- **Common error:** Mixing eV for E_k with k_B in J/K (or cm⁻¹ with wrong conversion).
- **Correct:** If E_k in eV, use k_B = 8.617333 × 10⁻⁵ eV/K. If E_k in cm⁻¹, use
  k_B T in cm⁻¹ units: k_B = 0.6950356 cm⁻¹/K.
- **Check:** The Boltzmann plot slope should be ~−1/(0.862 eV) × (1/T) at T = 10,000 K ≈ −1.16.

### P3: Degeneracy g_k confusion
- g_k = 2J_k + 1 where J_k is the total angular momentum quantum number of the UPPER level.
- **NOT** the lower level g_i; **NOT** 2l+1 or 2S+1 alone (those are for LS-coupled sub-levels).
- Many databases list g_k directly; always verify against the NIST level entry.

### P4: Partition function at wrong temperature / truncated sum
- U_s(T) must be evaluated at the current iteration's temperature T, not at a fixed value.
- The sum U_s(T) = Σ_i g_i exp(-E_i/k_B T) should include all energy levels up to the
  ionization limit. Truncating at a fixed maximum energy causes systematic underestimation,
  especially at high T.
- **Best practice:** Use tabulated partition functions from Irwin (1981), NIST, or Kurucz, fitted
  to a polynomial: log U = Σ a_n (log T)^n (e.g., Griem, or the 5th-order polynomial from NIST).

### P5: Mole fractions reported as mass fractions
- CF-LIBS closure gives C_s = mole (number) fractions. All reported comparisons to nominal
  composition must use the SAME basis.
- Völker (2024) quantified errors up to 353% from omitting the M_s-weighted conversion.

### P6: Saha equation factor-of-2 error
- The factor of 2 in the Saha equation `N_{s,1}/N_{s,0} = (2/n_e) · ...` comes from the electron
  partition function (spin degeneracy = 2). Some texts write `(1/n_e)` and absorb the 2 into U;
  verify which convention your source uses.
- Some formulations use CGS (n_e in cm⁻³) with the constant 4.829 × 10¹⁵ T^(3/2); others use
  SI (n_e in m⁻³). A factor-of-10⁶ error in n_e (unit mismatch) shifts all ionic/neutral ratios
  dramatically.

### P7: Ionic line energy shift on the Saha-Boltzmann plot
- Ionic lines must use E_k^(eff) = E_k^(ion) + χ_s (ionization energy added to all ionic
  upper-level energies).
- Failure to add χ_s places ionic lines at wrong x-axis positions → wrong slope → wrong T.
- χ_s should be the first-ionization potential in the SAME energy units as E_k^(ion).
- For higher ions (z=2), add χ_s,1 + χ_s,2 to the upper-level energy.

### P8: Stoichiometric ablation assumption
- CF-LIBS assumes laser ablation produces a plasma whose elemental composition equals the bulk
  sample. This is violated for multi-phase, highly volatile, or heterogeneous samples.
- No algorithmic correction exists for non-stoichiometric ablation; it is a fundamental limit.

### P9: Self-absorption in Boltzmann plot lines
- Optically thick lines appear weaker than their true intensity → Boltzmann plot points scatter
  below the true line → temperature overestimated (apparent slope less steep).
- Preferred fix: use lines with low transition probability and high excitation energy (less prone
  to self-absorption). Validate with doublet-ratio test: I(λ₁)/I(λ₂) should match g₁A₁/g₂A₂.

### P10: F factor sign convention / intercept extraction
- The Boltzmann plot intercept b_s involves ln(F C_s / U_s(T)) (plus constant terms from hc/4π).
  When F is extracted from the closure sum, the same constant must be consistently included or
  excluded from b_s. Any reformulation of the y-axis must propagate the constant correctly.

---

## 3. Key References

1. **Ciucci A., Corsi M., Palleschi V., Rastelli S., Salvetti A., Tognoni E. (1999)**
   "New Procedure for Quantitative Elemental Analysis by Laser-Induced Plasma Spectroscopy."
   *Applied Spectroscopy* 53(8), 960–964. DOI: 10.1366/0003702991947612
   — **The original CF-LIBS paper.** Introduces the method, three core assumptions
   (stoichiometric ablation, LTE, optical thinness), and the closure normalization.

2. **Tognoni E., Cristoforetti G., Legnaioli S., Palleschi V. (2010)**
   "Calibration-Free Laser-Induced Breakdown Spectroscopy: State of the art."
   *Spectrochimica Acta Part B* 65(1), 1–14. DOI: 10.1016/j.sab.2009.11.006
   — **The definitive review.** Critical assessment of accuracy, error sources, and limits of
   CF-LIBS; comprehensive equation set; summary of ~50 experimental results in the literature.

3. **Aguilera J.A., Aragón C. (2007)**
   "Multi-element Saha–Boltzmann and Boltzmann plots in laser-induced plasmas."
   *Spectrochimica Acta Part B* 62, 378–385. DOI: 10.1016/j.sab.2007.03.003
   — **Canonical multi-element Saha-Boltzmann plot method.** Common-slope fit across neutral
   and ionic lines of multiple elements; intercepts give relative concentrations.

4. **Aragón C., Aguilera J.A. (2008)**
   "Characterization of laser induced plasmas by optical emission spectroscopy: A review of
   experiments and methods."
   *Spectrochimica Acta Part B* 63(9), 893–916. DOI: 10.1016/j.sab.2008.05.010
   — Comprehensive treatment of plasma diagnostics including Boltzmann/Saha-Boltzmann
   methodology, electron density measurement, and spatially resolved measurements.

5. **Cristoforetti G., De Giacomo A., Dell'Aglio M., Legnaioli S., Tognoni E., Palleschi V.,
   Omenetto N. (2010)**
   "Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy: Beyond the
   McWhirter Criterion."
   *Spectrochimica Acta Part B* 65, 86–95. DOI: 10.1016/j.sab.2009.11.005
   — **Authoritative LTE validity paper.** Shows McWhirter alone is insufficient; gives the
   relaxation-time criterion; quantifies when LIBS plasmas satisfy LTE.

6. **Tognoni E., Cristoforetti G., Legnaioli S., Palleschi V., Salvetti A., Müller M.,
   Panne U., Gornushkin I. (2007)**
   "A numerical study of expected accuracy and precision in Calibration-Free LIBS in the
   assumption of ideal analytical plasma."
   *Spectrochimica Acta Part B* 62(12), 1287–1302. DOI: 10.1016/j.sab.2007.10.005
   — Systematic error analysis of CF-LIBS; quantifies sensitivity to temperature error,
   partition function errors, n_e errors, and line selection.

7. **Völker T. (2024)**
   "Mass and mole fractions in calibration-free LIBS."
   *J. Analytical Atomic Spectrometry* (RSC). DOI: 10.1039/D4JA00028E
   — **Essential correction paper.** Proves CF-LIBS closure gives mole fractions, not mass
   fractions; shows errors up to 353% from omitting molar-mass conversion; gives explicit
   conversion formulas.

8. **Cristoforetti G., Tognoni E. (2013)**
   "Calculation of elemental columnar density from self-absorbed lines in laser-induced breakdown
   spectroscopy: a resource for quantitative analysis."
   *Spectrochimica Acta Part B* 79–80, 63–71. DOI: 10.1016/j.sab.2012.11.010
   — Column-density Saha-Boltzmann (CD-SB) method for self-absorbed lines; allows CF-LIBS
   without requiring optically-thin lines.

9. **Ferus M. et al. (2018)**
   "Calibration-free quantitative elemental analysis of meteor plasma using reference LIBS of
   meteorite samples."
   *Astronomy & Astrophysics* 610, A73. DOI: 10.1051/0004-6361/201629950
   — Good worked example of full CF-LIBS pipeline with explicit equations (in accessible HTML);
   demonstrates the Saha-Boltzmann ionic line shift; uses meteor plasma as test case.

10. **Völker (RSC Adv. 2023 review) — "Catching up on calibration-free LIBS"**
    *J. Analytical Atomic Spectrometry* (RSC). DOI: 10.1039/D3JA00130J
    — Up-to-date critical review of CF-LIBS variants: CD-SB, C-Sigma, one-point calibration,
    self-absorption correction methods; best current synthesis of the field.

---

## 4. What Correct Code MUST Do — Checklist

### Data preparation
- [ ] Integrate spectral lines to get I_{ki} (area under peak, not peak height, unless Gaussian
      profile and FWHM is constant — peak height is acceptable if all lines have same lineshape
      and FWHM across the spectrum, rarely true).
- [ ] Apply detector spectral response / throughput calibration before using line intensities.
- [ ] Verify optical thinness for each line used (doublet ratio test or self-absorption coefficient).

### Boltzmann plot
- [ ] Y-axis: `ln(I_{ki} * lambda_{ki} / (A_{ki} * g_k))` — must include λ.
- [ ] X-axis: `E_k` in eV (or wavenumbers) — upper level energy above GROUND STATE of that species.
- [ ] Energy units: check k_B units match E_k units (eV/K vs J/K; NOT mixed).
- [ ] g_k = 2J_k + 1 for the UPPER level; fetch from atomic database, not derived from scratch.

### Saha-Boltzmann for ionic lines
- [ ] Shift ionic lines to x-axis value `E_k^(ion) + chi_s` (ionization energy in same units).
- [ ] Apply the Saha correction term to the y-axis (involves n_e, T, m_e, h).
- [ ] Use n_e from Stark broadening measurement or iterative Saha self-consistency.
- [ ] Factor of 2 in Saha: `(2/n_e) * (2 pi m_e k_B T / h^2)^(3/2)` — verify your source's convention.
- [ ] Ionization energy chi_s: use NIST value; apply ionization-potential lowering Δchi only if n_e > ~10^17 cm^-3 (usually < 0.1 eV in typical LIBS, often neglected).

### Partition functions
- [ ] U_s(T) evaluated at current T in the iteration, not at a fixed temperature.
- [ ] Include sufficient energy levels up to the ionization limit (truncation causes error at high T).
- [ ] Use polynomial fits (e.g., 5th-order in log T) from validated tables, not on-the-fly sums
      unless the full NIST level list is available.

### Closure and concentration
- [ ] Sum C_s = 1 over ALL detected elements (both ionization stages combined if both measured).
- [ ] Account for all elements: missing elements lead to over-concentration of detected ones.
- [ ] F is determined PURELY from the closure — never from external calibration in CF-LIBS.
- [ ] Convert mole fractions to mass fractions via w_s = (C_s * M_s) / sum_j(C_j * M_j) before
      comparing to nominal wt% compositions.

### Iteration
- [ ] Iterate until both T and n_e converge (|ΔT/T| < 0.1% and |Δn_e/n_e| < 1% typical tolerance).
- [ ] Check charge neutrality: n_e ≈ Σ_s C_s * z_s * N_total (verify within ~factor of 2).
- [ ] Do NOT assume n_e = 0 or n_e = constant; both introduce systematic errors in Saha balance.

### LTE validity
- [ ] Verify McWhirter: n_e >= 1.6e12 * T^0.5 * (ΔE)^3 [cm^-3] where ΔE is the resonance-line
      energy of the dominant species (NOT adjacent-gap; NOT max(E_k)).
- [ ] Check observation time window: LTE is most reliable at 0.5–2 µs after pulse (before plasma
      cools below LTE threshold; after optically-thick early phase).

### Oxide/geological closure
- [ ] For mineral/geological samples: decide whether to apply oxide closure model or standard
      elemental closure. Document the choice and which elements are included.
- [ ] If oxygen is inferred from oxide stoichiometry, include its contribution to the closure sum
      and clearly label output as oxide-normalized.

---

*Sources accessed: Frontiers Physics 2022 (doi 10.3389/fphy.2022.887171), RSC JAAS 2023
(doi 10.1039/D3JA00130J), RSC JAAS 2024 (doi 10.1039/D4JA00028E), A&A 2018 Ferus et al.
(doi 10.1051/0004-6361/201629950), RSC Advances 2026 (doi 10.1039/D6RA01889K), ADS abstract
for Tognoni 2010 (Spec. Acta B 65:1), Aguilera/Aragón 2007 IAEA INIS record, LibreTexts Saha
equation (Tatum stellar atmospheres), Cristoforetti 2010 ADS abstract.*
