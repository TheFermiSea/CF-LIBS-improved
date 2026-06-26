# Self-Absorption, Optical Depth, and Curve-of-Growth Corrections in LIBS

## 1. Governing Equations — Canonical Form

### 1.1 Radiative Transfer Through a Homogeneous LTE Plasma Slab

For a homogeneous, isothermal slab of thickness *l* in local thermodynamic equilibrium (LTE),
the emerging spectral radiance satisfies the equation of transfer

```
dI_λ/dl = ε_λ − κ_λ I_λ
```

with the source function S_λ = ε_λ / κ_λ = B_λ(T) (Kirchhoff's law), giving the
closed-form solution

```
I_λ = B_λ(T) [1 − exp(−τ_λ)]          ... (1)
```

where

```
τ_λ = κ_λ l                             ... (2)
```

is the **optical depth** at wavelength λ.

**Symbols and units:**
| Symbol | Meaning | SI unit |
|--------|---------|---------|
| I_λ    | Spectral radiance (observed) | W m⁻² sr⁻¹ m⁻¹ |
| B_λ(T) | Planck function at plasma temperature T | W m⁻² sr⁻¹ m⁻¹ |
| κ_λ    | Absorption coefficient (spectrally resolved) | m⁻¹ |
| l      | Effective plasma column length (line-of-sight) | m |
| τ_λ    | Optical depth (dimensionless) | — |

**Optically thin limit** (τ ≪ 1): 1 − exp(−τ) ≈ τ, so I_λ → B_λ κ_λ l ∝ N_s.
**Optically thick limit** (τ ≫ 1): I_λ → B_λ(T) (approaches blackbody at T).

### 1.2 Absorption Coefficient

For a spectral line u→l (upper level u, lower level l), under LTE, including stimulated
emission (usually small at LIBS temperatures but formally required):

```
κ(ν) = (hν/4π) · (N_l B_lu − N_u B_ul) · φ(ν)

     = (A_ul g_u λ² / 8π) · (N_l/g_l) · [1 − exp(−hν/k_B T)] · g_l/U_s(T) · exp(−E_l/k_B T) · N_s · φ(ν)
```

In the common LIBS approximation (hν ≫ k_B T for UV/visible, stimulated emission ≈ 0)
this simplifies to

```
κ(ν) ≈ (A_ul g_u λ²)/(8π) · (N_s/U_s(T)) · exp(−E_l/k_B T) · φ(ν)    ... (3)
```

**Symbols:**
| Symbol | Meaning |
|--------|---------|
| A_ul   | Einstein A-coefficient (s⁻¹); note "ki" and "ul" subscript conventions differ by author |
| g_u    | Upper-level statistical weight = 2J_u + 1 |
| g_l    | Lower-level statistical weight = 2J_l + 1 |
| E_l    | Lower-level energy (J or eV) |
| E_u    | Upper-level energy (J or eV) |
| N_s    | Total number density of the emitting species (m⁻³) |
| U_s(T) | Partition function of the species at temperature T |
| φ(ν)  | Normalised line profile (Hz⁻¹); ∫φ dν = 1 |
| λ      | Transition wavelength (m) |

> **Sign/degeneracy trap:** Some papers absorb the (g_u/g_l) ratio into the oscillator
> strength f via f_lu = (m_e c)/(π e²) · (g_u/g_l) · A_ul · λ²/(8π). If you use
> the absorption oscillator strength f_lu, the formula becomes κ(ν) = (π e²/m_e c) · (N_l/g_l) · g_l · f_lu · φ(ν).
> Using the wrong g ratio here is a common off-by-(g_u/g_l) error.

### 1.3 Self-Absorption Coefficient SA

The **SA coefficient** is the ratio of the observed peak intensity to the intensity
the line would have in an optically thin plasma of the same temperature and density:

```
SA = I(λ₀) / I₀(λ₀)                    ... (4)
```

For the homogeneous slab, using Eq. (1) and the thin-limit formula:

```
SA = [1 − exp(−τ₀)] / τ₀               ... (5)
```

where τ₀ = κ(λ₀)·l is the peak optical depth.

**Range:** SA ∈ (0, 1].
- SA = 1 : optically thin (no self-absorption).
- SA → 0 : severely self-absorbed (saturated).

**Equivalent width / integrated intensity:** For a Voigt line profile the integrated
intensity scales as SA^0.5 (to first approximation), while the FWHM scales as SA^(−0.54):

```
Δλ_obs = Δλ₀ · (SA)^(−0.54)           ... (6)
```

This is the Konjevic–Wiese empirical exponent; some texts use −0.5.

### 1.4 The Optical Depth Parameter q (Ladenburg–Reiche)

After inserting the Boltzmann population and Eq. (3), the **peak optical depth** at
line centre λ₀ for a Voigt profile becomes

```
τ₀ = κ(λ₀) · l ≡ q                     ... (7)
```

with

```
q = (2 e²)/(m_e c²) · n_i · f_lu · λ₀² · l    (Gaussian cgs form)

  ≈ (λ₀² l g_u A_ul N_s)/(8π · Δλ_D · U_s(T)) · exp(−E_l / k_B T)    ... (8)
```

where Δλ_D is the Doppler (inhomogeneous) width. Note q ≡ τ₀ in this context; authors
sometimes call q the "Ladenburg factor."

**Pitfall:** Equation (8) is written for Doppler-dominated profiles. For Voigt profiles
one must use the peak of φ(ν), which requires the Voigt damping ratio η = Γ_L / Γ_D.
The (1-exp(-q)) / q formula still holds; only the q formula changes.

### 1.5 Curve of Growth (COG)

The **curve of growth** relates the integrated emission intensity W (or equivalent width
in absorption) to the column density N_s · l:

```
W ≡ ∫ I(λ) dλ = I_peak^(thin) · ∫ [1 − exp(−τ(λ))] dλ    ... (9)
```

Three regimes:
1. **Linear regime** (τ₀ ≪ 1): W ∝ N_s · l.  Boltzmann plot is valid.
2. **Flat-of-the-curve** (τ₀ ~ 1–10): W grows as (N_s · l)^0.5 for Doppler profiles
   (Voigt: slower than linear). Self-absorption causes slope change in Boltzmann plot.
3. **Square-root regime** (τ₀ ≫ 1, Doppler-dominated): W ∝ (N_s · l)^0.5.

In LIBS the COG iteration (Bulajic et al. 2002) fits T, n_e, Gaussian width, Lorentzian
width, and l simultaneously by comparing modelled and measured COGs for multiple lines,
then corrects all line intensities to their optically thin equivalents.

### 1.6 Planck-Function Correction (BRR-SAC / Poggialini method)

If plasma temperature T is known, Eq. (1) can be inverted directly:

```
I_Thin_λ = B_λ(T) · κ_λ · l = −B_λ(T) · ln[1 − I_λ / B_λ(T)]    ... (10)
```

This requires only T (not n_e, not l separately), making it the simplest single-parameter
correction. It fails when I_λ / B_λ(T) approaches 1 (τ > ~3) due to numerical
instability.

### 1.7 CSigma (Cσ) Method — Aragón & Aguilera 2014

The Cσ method generalises the COG to multiple species and ionisation states.
Define for each observed line the **line cross section**:

```
σ_l = (λ² A_ul g_u)/(8π U_s(T)) · exp(−E_u / k_B T)
      × [1 + (N_{s+1}/N_s)]^(−1)    ... (11)
```

where the last factor accounts for ionisation balance via Saha.

The **Cσ graph** plots C = I_measured / (β_A · σ_l)  vs  σ_l · N_l,
where N_l = N_s · l (columnar density, m⁻²) and β_A is an instrumental constant.

- For optically thin lines: C ≈ 1 (flat region).
- For self-absorbed lines: C < 1, following the universal COG curve.

Four parameters are fitted globally: {β_A, N_l, T, N_e}.
The method can **exploit** self-absorption rather than avoid it.

**Limitation:** Strongly reversed lines (SA < ~0.3) lie outside the model's validity and
must be excluded from the fit.

### 1.8 Duplicating Mirror Method

A spherical mirror placed behind the plasma doubles the effective path length. For
emission I₁ (direct) and I₂ (direct + reflected), if G = solid-angle collection factor:

```
Continuum ratio:  R_c = I₂^cont / I₁^cont = 1 + G         ... (12)
Line ratio:       R_λ = I₂(λ) / I₁(λ)  = 1 + G · exp(−τ_λ)  ... (13)
```

Solving for τ:

```
τ_λ = ln[(R_c − 1) / (R_λ − 1)]                           ... (14)
```

And the correction factor:

```
K_corr = τ_λ / (1 − exp(−τ_λ))  = 1/SA                   ... (15)
```

This gives τ without needing T, n_e, or l independently — useful as a diagnostic.
Requires simultaneous measurement with and without mirror; requires smooth continuum.

### 1.9 Internal Reference Self-Absorption Correction (IRSAC)

Select a reference line (ref) of the same species and ionisation state with negligible
self-absorption (SA_ref ≈ 1). The SA-corrected intensity of any other line k is:

```
I_k^(IRSAC) = I_ref^(meas) · (A_k g_k / A_ref g_ref) · exp[−(E_k − E_ref) / k_B T]  ... (16)
```

This is simply the Boltzmann ratio relative to the reference line: it reconstructs what
the intensity *would be* if the line were optically thin, using T from an initial
(possibly biased) Boltzmann plot. An iterative loop updates T until the Boltzmann plot
correlation coefficient converges.

**Reference-line selection criteria:**
- High upper-level energy E_k (low self-absorption probability).
- Not a resonance line (E_l ≠ 0).
- Isolated, symmetric, unblended profile.
- Verify optical thinness: doublet ratio test (observed / theoretical ≤ ±4%).

---

## 2. Common Implementation Pitfalls and Correct Treatment

### 2.1 Including Resonance Lines in Boltzmann Plots Without SA Check
**Pitfall:** Resonance lines (lower level = ground state, E_l = 0) have the highest
lower-level population and therefore the largest optical depth for a given N_s · l.
Including them uncorrected in a Boltzmann plot artificially **raises the apparent
temperature** (the self-absorbed low-E points fall below the line, increasing the slope).

**Correct treatment:** Screen all lines with a doublet-ratio test or COG/Cσ check before
inclusion. Exclude or correct lines with SA < ~0.9 before fitting T.

### 2.2 Using FWHM Alone as SA Diagnostic
**Pitfall:** Equation (6) assumes a homogeneous plasma. In inhomogeneous plasmas FWHM
can be broadened by self-reversal without the same SA vs. FWHM relationship.

**Correct treatment:** Distinguish self-absorption (smooth profile flattening) from
self-reversal (dip at line centre). Self-reversal requires a two-layer inhomogeneous
model; its presence invalidates the homogeneous SA formula entirely.

### 2.3 Applying SA Correction to Reversed Lines
**Pitfall:** If a line shows a central dip (self-reversal), formula (5) does not apply.
Attempting correction with Eq. (5) or (16) gives wrong results.

**Correct treatment:** Amamou et al. (2003) derived a correction for reversed lines using
the two-layer model. Alternatively, exclude reversed lines from CF-LIBS analysis.

### 2.4 Neglecting Stimulated Emission Term
**Pitfall:** Dropping the [1 − exp(−hν/k_B T)] term in κ(ν) is valid for UV/visible at
T ~ 5000–20000 K, but introduces a systematic overestimate of κ for far-infrared or very
high-T plasmas.

**Correct treatment:** Include the correction factor. For T = 10000 K and λ = 500 nm,
hν/k_B T ≈ 2.9, so exp(−hν/k_B T) ≈ 0.055 — a ~5% error if omitted.

### 2.5 Confusing κ (per-unit-length) with τ (integrated)
**Pitfall:** Some LIBS papers define τ = κ(λ₀) · l only at line centre, while others
integrate over the profile. The SA = (1-exp(-τ))/τ formula applies strictly to the
**peak** optical depth; the integrated correction requires the full COG integral.

**Correct treatment:** Be explicit about whether τ refers to peak or integrated optical
depth. For peak-corrected intensities use Eq. (5); for integrated intensities use Eq. (9).

### 2.6 Circular Dependence of T on SA
**Pitfall:** SA depends on T (via partition function and Boltzmann factor); T is measured
from the Boltzmann plot, which depends on SA. Failure to iterate produces a biased T.

**Correct treatment:** Use an iterative loop: initial T from uncorrected Boltzmann plot →
compute SA for all lines → correct intensities → re-fit T → repeat until convergence
(typically 3–5 iterations). IRSAC and BRR-SAC both require this loop.

### 2.7 Wrong g_u vs g_l in the Absorption Coefficient
**Pitfall:** The absorption coefficient κ uses the *lower*-level population (N_l = N_s
g_l / U_s · exp(-E_l/k_B T)), not the upper level. Some implementations accidentally
substitute g_k (upper) where g_l (lower) is required.

**Correct treatment:** Eq. (3) uses E_l (lower level energy) and N_s normalised by U_s.
The A_ul already carries the (g_u) factor through Einstein's relation; do not
double-count it.

### 2.8 Identifiability Limits
When self-absorption is severe (SA < 0.3, τ₀ > ~3):
- The Boltzmann plot temperature is unreliable.
- Equation (10) becomes ill-conditioned (I_λ / B_λ → 1).
- The COG flat portion degenerates: multiple (T, N_l) pairs may fit the data equally
  well.
- In the limiting case τ → ∞, the line intensity → B_λ(T) regardless of N_s, making
  concentration unidentifiable from that line alone.

Practical consequence: **do not attempt single-line SA correction when τ₀ > 3**; rely on
lines from the weak-line multiplet members or non-resonance transitions, or apply the
full COG/Cσ with sufficient lines to constrain T and N_l simultaneously.

---

## 3. Key References

1. **Bulajic, D., Corsi, M., Cristoforetti, G., Legnaioli, S., Salvetti, A., Sezgin, F., &
   Tognoni, E. (2002).** "A procedure for correcting self-absorption in calibration
   free-laser induced breakdown spectroscopy." *Spectrochim. Acta Part B*, 57(2), 339–353.
   DOI: 10.1016/S0584-8547(01)00398-6.
   — *The foundational COG iterative algorithm for CF-LIBS; introduces the iterative
   fitting of T, n_e, Gaussian/Lorentzian widths, and optical path length.*

2. **Amamou, H., Bois, A., Ferhat, B., Redon, R., Rossetto, B., & Matheron, P. (2002).**
   "Correction of self-absorption spectral line and ratios of transition probabilities for
   homogeneous and LTE plasma." *J. Quant. Spectrosc. Radiat. Transf.*, 75(6), 747–763.
   DOI: 10.1016/S0022-4073(02)00040-7.
   — *Derives correction factors for homogeneous LTE plasma; discusses SA for resonance
   and non-resonance lines; gives the κ formula with stimulated-emission correction.*

3. **Amamou, H., Bois, A., Ferhat, B., Redon, R., Rossetto, B., & Matheron, P. (2003).**
   "Correction of the self-absorption for reversed spectral lines: application to two
   resonance lines of neutral aluminium." *J. Quant. Spectrosc. Radiat. Transf.*, 77(4),
   365–372.  DOI: 10.1016/S0022-4073(02)00163-2.
   — *Two-layer model correction for self-reversed lines; Al I resonance doublet case study.*

4. **Cristoforetti, G., De Giacomo, A., Dell'Aglio, M., Legnaioli, S., Tognoni, E.,
   Palleschi, V., & Omenetto, N. (2010).** "Local Thermodynamic Equilibrium in
   Laser-Induced Breakdown Spectroscopy: Beyond the McWhirter Criterion."
   *Spectrochim. Acta Part B*, 65(1), 86–95.  DOI: 10.1016/j.sab.2009.11.005.
   — *Quantifies when LTE holds and when self-absorption invalidates plasma diagnostics.*

5. **Aragón, C. & Aguilera, J. A. (2014).** "CSigma graphs: A new approach for plasma
   characterization in laser-induced breakdown spectroscopy." *J. Quant. Spectrosc. Radiat.
   Transf.*, 149, 90–102.  DOI: 10.1016/j.jqsrt.2014.07.026.
   — *Introduces Cσ graphs; defines line cross-section σ_l; provides the four-parameter
   fit {β_A, N_l, T, N_e}; shows how to exploit self-absorbed lines.*

6. **El Sherbini, A. M., El Sherbini, Th. M., Hegazy, H., Cristoforetti, G., Legnaioli,
   S., Palleschi, V., Pardini, L., Salvetti, A., & Tognoni, E. (2005).** "Evaluation of
   self-absorption coefficients of aluminum emission lines in laser-induced breakdown
   spectroscopy measurements." *Spectrochim. Acta Part B*, 60(12), 1573–1579.
   DOI: 10.1016/j.sab.2005.10.011.
   — *Systematic measurement of SA coefficients for Al I lines; doublet-ratio test;
   correction methodology including Stark broadening.*

7. **Sun, L. & Yu, H. (2009).** "Correction of self-absorption effect in calibration-free
   laser-induced breakdown spectroscopy by an internal reference method." *Talanta*, 79(2),
   388–395.  DOI: 10.1016/j.talanta.2009.04.003.
   — *Introduces IRSAC; describes the reference-line selection criteria and iterative
   convergence algorithm.*

8. **Poggialini, F., Campanella, B., Legnaioli, S., Raneri, S., & Palleschi, V. (2023).**
   "Investigation of a method for the correction of self-absorption by Planck function in
   laser induced breakdown spectroscopy." *J. Anal. At. Spectrom.*, 38, 571–578.
   DOI: 10.1039/D2JA00352J.
   — *Planck-function (BRR-SAC) method; requires only T; shows correction accuracy
   limit at τ ~ 2–3 and temperature-error sensitivity.*

9. **Rezaei, F., Cristoforetti, G., Tognoni, E., Legnaioli, S., Palleschi, V., &
   Safi, A. (2020).** "A review of the current analytical approaches for evaluating,
   compensating and exploiting self-absorption in Laser Induced Breakdown Spectroscopy."
   *Spectrochim. Acta Part B*, 169, 105878.  DOI: 10.1016/j.sab.2020.105878.
   — *Comprehensive review of all SA correction methods; detailed comparison; identifiability
   discussion.*

10. **Safi, A., Messaoud Aberkane, S., Botto, A., Campanella, B., Legnaioli, S.,
    Poggialini, F., Raneri, S., Rezaei, F., & Palleschi, V. (2021).** "Determination of
    Spectroscopic Parameters of Ag(I) and Ag(II) Emission Lines Using Time-Independent
    Extended C-Sigma Method." *Appl. Spectrosc.*, 75(8), 1039–1051.
    DOI: 10.1177/0003702821999425.
    — *Extended Cσ method; application to multi-species plasma; shows iterative convergence
    criteria.*

---

## 4. What Correct Code MUST Do — Checklist

### Pre-Analysis (before fitting T / concentrations)
- [ ] **Doublet-ratio screen:** For each species with known doublet/multiplet, compute
  R_obs / R_theory. Flag any line with R_obs / R_theory < 0.95 as potentially
  self-absorbed (SA < 1).
- [ ] **Resonance-line flag:** Any line with E_l = 0 (or E_l < k_B T ~ 0.1 eV at 10 kK)
  is a resonance/near-resonance line and should be treated with extra suspicion.
- [ ] **Estimate q (Eq. 8):** Using initial T, n_e, and an estimated N_s · l, compute
  q for all candidate lines. Lines with q > 0.5 should be corrected or excluded.

### Self-Absorption Coefficient Computation (Eqs. 3–5)
- [ ] Use **lower-level energy E_l** (NOT E_u) in the Boltzmann factor for κ.
- [ ] Include **g_u** (statistical weight of upper level = 2J_u + 1) — do not
  substitute g_l or the total degeneracy 2L+1.
- [ ] Apply the **stimulated-emission correction** [1 − exp(−hν / k_B T)] if T < 20 000 K
  and λ < 600 nm (correction is ~5% at T = 10 000 K, λ = 500 nm; larger for higher λ or
  lower T).
- [ ] For peak intensities, SA = (1 − exp(−τ₀)) / τ₀ where τ₀ = κ(λ₀) · l.
  For integrated intensities, evaluate the full integral Eq. (9) — **do not** simply
  apply the peak SA to the integrated intensity.

### SA Correction Algorithm
- [ ] **Iterate:** compute SA → correct intensities → re-fit T from corrected Boltzmann
  plot → recompute SA → repeat until |ΔT/T| < 0.5% (typically 3–5 iterations).
- [ ] **Exclude reversed lines:** If a line shows a central dip, the homogeneous formula
  does not apply. Flag and exclude or apply the Amamou (2003) two-layer correction.

### Identifiability Guards
- [ ] If τ₀ > 3 for a line, log a warning: "Line {line_id} severely self-absorbed
  (τ₀ = {value}); correction unreliable, excluding from quantitative analysis."
- [ ] If all lines of a species are self-absorbed, concentration is **not identifiable**
  from that species alone; set concentration uncertainty to +∞ or mark as unbounded.
- [ ] When the Planck-function method (Eq. 10) is used, guard against I_λ / B_λ > 0.95
  (numerical divergence); clamp and report.

### CSigma / COG Path
- [ ] The four-parameter Cσ fit requires sufficient spectral lines spanning the COG
  (both optically thin and optically thick lines) to constrain {β_A, N_l, T, N_e}.
  With < 4 usable lines per species the fit is underdetermined.
- [ ] Lines with SA < 0.3 should be excluded from Cσ fitting (they lie outside the
  model's validity per Aragón & Aguilera 2014).

### Doublet-Ratio Diagnostic (recommended continuous check)
- [ ] For Mg II 279.6 / 280.3 nm: R_theory = 2.03; for Ca II 393.4 / 396.8 nm:
  R_theory = 2.0; for Al I 394.4 / 396.2 nm: R_theory ≈ 2.0.
  Report SA_empirical = R_obs / R_theory for monitoring purposes.

### Documentation
- [ ] Record which lines were corrected, which were excluded, and the final τ₀ and SA
  values alongside any T / n_e / concentration outputs.
- [ ] If optical path length *l* is used in the correction, note that *l* is a fitted
  or assumed parameter — uncertainty in *l* propagates directly to uncertainty in SA.
