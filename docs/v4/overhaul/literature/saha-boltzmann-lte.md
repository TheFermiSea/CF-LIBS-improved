# Literature Reference: Saha Equation, Boltzmann Distribution, and LTE in LIBS Plasmas

Topic key: `saha-boltzmann-lte`
Date compiled: 2026-06-25

---

## 1. Governing Equations in Canonical Form

### 1.1 Boltzmann Distribution — Level Populations

For a single ionization stage of an element in LTE, the population density of
level k with statistical weight g_k and excitation energy E_k (measured from
the ground state of that stage) is:

```
n_k = n_s · (g_k / U(T)) · exp(-E_k / k_B T)
```

where:
- n_s   = total number density of the ionization stage [cm^-3]
- g_k   = statistical weight (degeneracy) of level k = 2J+1 for LS coupling
- E_k   = energy of the upper level above the ground state of this stage [eV]
- U(T)  = partition function [dimensionless] (see §1.3)
- k_B   = Boltzmann constant = 8.617333×10^-5 eV/K
- T     = plasma temperature [K]

The Boltzmann factor uses k_B T in eV when E_k is in eV (equivalently T in K
with k_B in eV/K). Energy MUST be measured from the ground state of the SAME
ionization stage, NOT from an absolute zero or from another stage.

### 1.2 Line Emissivity

The spontaneous-emission volume emissivity of a spectral line (power per unit
volume per unit solid angle) for transition k → i is:

```
ε_ki = (h c / 4π λ_ki) · A_ki · n_k         [erg s^-1 cm^-3 sr^-1]
```

or equivalently, substituting n_k:

```
ε_ki = (h c / 4π λ_ki) · A_ki · n_s · (g_k / U(T)) · exp(-E_k / k_B T)
```

where:
- h     = Planck constant = 6.626×10^-27 erg·s
- c     = speed of light = 2.998×10^10 cm/s
- λ_ki  = transition wavelength [cm]
- A_ki  = Einstein A-coefficient for spontaneous emission [s^-1]
- (upper level = k, lower level = i, consistent with NIST notation)

CRITICAL: the 1/(4π) arises from integrating over a full sphere of solid angle
(isotropic emission). The emissivity divided by integrated-sphere solid angle 4π
gives the emissivity per steradian. Some references omit 4π and quote power per
unit volume — check which convention applies before comparing.

For observed integrated line intensity I_ki (measured at detector), optically thin:

```
I_ki ∝ ε_ki · (path length or plasma volume) · F_instrument
```

In the CF-LIBS linearized form (Boltzmann plot):

```
ln(I_ki · λ_ki / (g_k · A_ki)) = -E_k / (k_B T) + const
```

The slope of ln(I λ / g A) vs. E_k gives -1/(k_B T).
Equivalently written as ln(I / (A g)) vs. E_k (the λ dependence sometimes
absorbed into the "const" if λ is approximately constant over the set of lines).

### 1.3 Partition Function

The canonical direct-sum definition:

```
U(T) = Σ_i  g_i · exp(-E_i / k_B T)
```

where the sum runs over ALL bound energy levels below the ionization potential
(or the IPD-lowered ionization potential — see §1.5).

- E_i measured from the ground state of the species (E_ground ≡ 0)
- g_i = 2J_i + 1 for the i-th level in LS coupling
- The ground-level term (i=0) contributes g_0 · exp(0) = g_0 to U(T)
- U(T) → g_0 at T → 0; U(T) grows with T as higher levels become populated

Polynomial fit form (Irwin 1981; used as a fallback in cflibs):

```
ln U(T) = a_0 + a_1·(ln T) + a_2·(ln T)^2 + a_3·(ln T)^3 + a_4·(ln T)^4
```

(4th-order polynomial in ln T, value is ln U — natural-log basis throughout.)
Irwin (1981, ApJS 45, 621) originally tabulated in log_10 T / log_10 U; when
ingesting those coefficients, apply the basis-change conversion.

Alternative 5th/6th-order polynomial fits in log_10 T are also used in the
literature (HITRAN, ExoMol conventions) — be explicit about which log base.

### 1.4 Saha Equation (Saha-Eggert Equation)

The Saha equation gives the ratio of number densities of two successive
ionization stages z+1 (ion) and z (neutral or lower ion):

```
n_{z+1} · n_e / n_z = (2 / h^3) · (2π m_e k_B T)^{3/2} · (U_{z+1} / U_z)
                       · exp(-χ_z / k_B T)
```

where:
- n_{z+1}  = number density of the (z+1)-th stage [cm^-3]
- n_z      = number density of the z-th stage [cm^-3]
- n_e      = electron number density [cm^-3]
- m_e      = electron mass = 9.109×10^-28 g
- h        = Planck constant = 6.626×10^-27 erg·s  (CGS)
- k_B      = Boltzmann constant = 1.381×10^-16 erg/K  (CGS)
- χ_z      = ionization potential of the z-th stage [eV] (energy to remove one electron)
- U_{z+1}  = partition function of the (z+1)-th (higher) stage
- U_z      = partition function of the z-th stage
- The factor "2" in the numerator is the electron spin degeneracy (g_e = 2·(1/2)+1 = 2)

CRITICAL FACTOR: The factor of 2 (electron spin degeneracy) appears in the
numerator because the free electron has two spin states. This is NOT a nuclear
spin factor. It is often written as 2·U_e where U_e = 1 (there is no excitation
for a free electron), so the total electron partition function contribution is
just 2.

CGS pre-factor evaluation (T in K, n in cm^-3):
```
(2 / h^3) · (2π m_e k_B)^{3/2} = 4.829×10^15  [cm^-3 K^{-3/2}]
```

SI pre-factor (T in K, n in m^-3):
```
(2 / h^3) · (2π m_e k_B)^{3/2} = 4.829×10^21  [m^-3 K^{-3/2}]
```

LIBS-convenient form (T in eV, n_e in cm^-3):
```
n_{z+1} · n_e / n_z = SAHA_CONST · T_eV^{3/2} · (U_{z+1} / U_z)
                       · exp(-χ_z / T_eV)
```
where SAHA_CONST ≈ 6.042×10^21 cm^-3 eV^{-3/2}

Derivation of SAHA_CONST:
  (2π m_e k_B)^{3/2} / h^3 × 2 × (eV→K conversion)^{3/2} × (m^-3 → cm^-3 factor)
  = (2π × 9.109e-31 × 1.381e-23)^{3/2} / (6.626e-34)^3 × 2 × (11604.52)^{3/2} × 1e-6
  ≈ 6.04×10^21  cm^-3 eV^{-3/2}

The Saha ratio S_z ≡ n_{z+1}/n_z is:
```
S_z = (SAHA_CONST / n_e) · T_eV^{3/2} · (U_{z+1}/U_z) · exp(-χ_eff,z / T_eV)
```

Solving the multi-stage balance (up to three stages, as in cflibs):
```
n_I + n_II + n_III = n_total
n_II / n_I = S_1
n_III / n_II = S_2
=> n_I = n_total / (1 + S_1 + S_1·S_2)
   n_II = S_1 · n_I
   n_III = S_2 · n_II
```

### 1.5 Ionization Potential Depression (IPD / Continuum Lowering)

In dense LIBS plasmas (n_e ~ 10^16–10^18 cm^-3), the effective ionization
potential is reduced by the electrostatic screening of the plasma. This enters
the Saha equation as χ_eff = χ - Δχ.

**Debye-Hückel model** (weak-coupling limit, simplest; default in cflibs):

Debye length:
```
λ_D = sqrt(k_B T / (4π n_e e^2))   [Gaussian-CGS]
    = sqrt(ε_0 k_B T / (n_e e^2))  [SI]
```

IPD:
```
Δχ_DH = e^2 / λ_D = e^2 · sqrt(4π n_e / (k_B T))   [Gaussian-CGS, result in erg]
```

Numerically at n_e=10^17 cm^-3, T=10^4 K:  Δχ_DH ≈ 0.066 eV

**Stewart-Pyatt (1966) model** (interpolates Debye-Hückel and ion-sphere limits):

Ion-sphere radius:  R_0 = (3 / (4π n_i))^{1/3}
```
Δχ_SP = (3(z+1)e^2 / (2R_0)) · [(1 + (λ_D/R_0)^3)^{2/3} - (λ_D/R_0)^2]
```

Limits:
- λ_D >> R_0 (weak coupling): Δχ_SP → (z+1)e^2/λ_D = Δχ_DH
- λ_D << R_0 (strong coupling, ion-sphere): Δχ_SP → 3(z+1)e^2/(2R_0)

Note: Stewart-Pyatt (1966) disagrees with some XFEL experiments at high density
(Ciricosta et al. 2012, PRL 109, 065002); the Ecker-Kröll model gives larger
depression but was not reproduced by other XFEL experiments. For LIBS conditions
(moderate density), both models give similar results; Stewart-Pyatt is standard.

CRITICAL: The same Δχ MUST be used consistently for BOTH the Saha equation
exponent (χ_eff = χ - Δχ) AND the partition function truncation cutoff
(levels above χ_eff are merged into the continuum and excluded from the sum).
Using different Δχ values in these two places breaks self-consistency.

### 1.6 McWhirter Criterion for LTE Validity

The McWhirter (1965) criterion gives the MINIMUM electron density for LTE:

```
n_e >= 1.6×10^12 · T^{1/2} · (ΔE)^3   [cm^-3]
```

where:
- T    = plasma temperature [K]
- ΔE   = relevant energy gap [eV]
- n_e  = electron density [cm^-3]

**CRITICAL — Definition of ΔE:**
ΔE is the energy gap of the HARDEST transition to collisionally thermalise —
which is the largest energy gap between adjacent levels in the term scheme that
is accessed by a dipole-allowed (resonance) transition.

Cristoforetti et al. (2010, Spectrochim. Acta B 65, 86-95) clarify:
"ΔE values to be inserted in the McWhirter criterion are determined by
considering the FIRST RESONANCE TRANSITION, neglecting forbidden and
intercombination lines."

The first resonance transition energy is the ground-state → first
allowed-excited-state gap (i.e., the shortest-wavelength resonance line).
This is NOT:
- The gap between consecutive adjacent observed upper levels (which would be
  tiny and badly underestimate the required density)
- The maximum upper-level energy of observed lines (which overestimates ΔE
  by treating the top of the Boltzmann ladder as a single step)

Typical values for LIBS-relevant species:
- Fe I:  ΔE ≈ 2.48 eV  (transition at ~500 nm, ground 3d^6 4s^2 a^5D)
- Cu I:  ΔE ≈ 3.82 eV  (4s^2S_{1/2} → 4p^2P° transition, ~325 nm)
- Mg I:  ΔE ≈ 2.71 eV  (3s^2 → 3s3p, ~457 nm)
- Ca II: ΔE ≈ 1.69 eV  (4s → 4p, ~393 nm K-line)

**Limitations:** McWhirter is a NECESSARY but NOT SUFFICIENT condition.
It applies to homogeneous, stationary plasmas. LIBS plasmas are inhomogeneous
and rapidly evolving, so additional criteria are required.

### 1.7 Cristoforetti Relaxation-Time Criterion (Beyond McWhirter)

For transient LIBS plasmas, the relaxation time to reach excitation equilibrium
(Cristoforetti et al. 2010, Eq. 4; Griem 1964) must be shorter than the
plasma evolution timescale:

```
τ_rel ≈ 6.3×10^4 / (n_e · <g> · f_nm) · ΔE_nm · sqrt(k_B T)
         · exp(ΔE_nm / k_B T)         [seconds]
```

where:
- n_e       = electron density [cm^-3]
- <g>       = effective EEDF-averaged Gaunt factor (~1 for order-of-magnitude)
- f_nm      = absorption oscillator strength of the transition
- ΔE_nm     = energy gap [eV]
- k_B T     = thermal energy [eV]

LTE criterion: τ_rel << τ_evol (the plasma's characteristic evolution time).
Conventionally, require τ_rel < τ_evol / 10.

Note: τ_rel grows exponentially with ΔE/kT and inversely with n_e, so a
rarefying plasma at early/late times can satisfy McWhirter (steady-state density
floor) yet FAIL the transient criterion.

---

## 2. Common Implementation Pitfalls and Correct Treatment

### P1: Energy Reference Inconsistency (CRITICAL)
- All energies E_i in the Boltzmann/partition-function sum must be from the
  GROUND STATE of that specific ionization stage.
- The Saha equation exponent χ_z uses the ionization potential of stage z —
  the energy to remove an electron from the ground state of stage z.
- DO NOT mix energies from different references. NIST ASD reports all levels
  relative to the ground state of each ionization stage — use that convention.

### P2: Partition Function Truncation Without IPD Consistency (CRITICAL)
- The partition function sum MUST be truncated at the same energy as the Saha
  equation ionization threshold. If IPD lowers χ by Δχ, both the Saha exponent
  (uses χ_eff = χ - Δχ) AND the partition function cutoff (E_max = χ_eff) must
  use the SAME Δχ.
- Previously identified in cflibs as "audit Family J" — two divergent IPD
  formulas (~1.44× difference) were fixed in 2026. The fix routes all IPD
  through a single canonical function.

### P3: The Factor of 2 in the Saha Equation
- The factor of 2 in "2 U_{z+1}/U_z" comes from the spin degeneracy of the
  free electron (g_e = 2). It is NOT the factor of 2 from the thermal de Broglie
  factor (that contributes to the T^{3/2} term).
- U_e = 1 for the free electron (no electronic excitation), so the full electron
  contribution is 2×1 = 2.
- Some texts write the Saha equation with "2 g_{z+1}/g_z" using only the ground
  state statistical weights instead of full partition functions — this is an
  approximation valid only at low T where higher levels are not populated.

### P4: McWhirter ΔE Definition (MAJOR COMMON ERROR)
- Using the gap between adjacent OBSERVED upper-level energies (which are
  typically 0.05-0.5 eV for closely-spaced levels) instead of the resonance
  transition energy (typically 2-5 eV) underestimates the required n_e by
  (2-5/0.05)^3 ≈ 10^4 to 10^6 — completely invalidating the LTE check.
- The code in cflibs.plasma.lte_validator correctly documents this and warns
  about using max(E_k) as a proxy for the resonance gap when only upper levels
  are available — but max(E_k) overestimates ΔE and should also be avoided.
- Best practice: look up the resonance transition energy from NIST for each
  element and pass delta_E_eV explicitly.

### P5: Partition Function Convergence / Divergence
- The partition function sum formally diverges for isolated atoms (infinitely
  many Rydberg states). Physical divergence is avoided by plasma broadening and
  merging into the continuum — implemented via the IPD energy cutoff.
- Using too many Rydberg levels (E_i close to IP) inflates U(T) and
  underestimates concentrations via the Saha equation.
- Correct approach: truncate at E_max = IP - Δχ. Alimohamadi & Ferland (2022,
  PASP 134) discuss this in detail for astrophysical/LIBS conditions.
- The 98% of IP shortcut (no n_e available) is a common approximation that
  excludes autoionizing levels; self-consistent IPD cutoff is preferred when n_e
  is known.

### P6: Polynomial Fit Basis (ln vs log_10)
- Historical confusion between log_10 and natural log polynomial fits for U(T).
- Irwin (1981, ApJS 45, 621) uses log_10 T and log_10 U.
- cflibs partition.py uses natural log (ln T, ln U) for the stored polynomial
  coefficients — converting from Irwin requires a basis change.
- The 30-60% errors reported in the cflibs 2026-05-09 audit were caused by
  stale fit data (coefficients fit against an older energy_levels snapshot),
  NOT a math/convention mismatch. Always re-fit polynomial coefficients from
  the CURRENT energy_levels table if levels are updated.
- Direct summation (preferred when energy levels are available) avoids all
  polynomial-fit ambiguities.

### P7: SAHA_CONST Units
- SAHA_CONST_CM3 = 6.042×10^21 cm^-3 eV^{-3/2} is valid ONLY when:
  - T is in eV (T_eV, not K)
  - n_e is in cm^-3
  - ionization potential χ is in eV
- If T is in K, multiply by (k_B in eV/K)^{3/2} before using.
- Do NOT mix SI and CGS in the same expression.

### P8: Saha Self-Consistency (Charge Balance)
- The Saha equation is implicitly coupled: the ratio n_{II}/n_I depends on n_e,
  but n_e is the sum of electrons contributed by ALL ionization stages of ALL
  elements.
- Correct approach: iterate n_e until charge balance
  n_e = Σ_elements Σ_stages (stage_number × n_stage) converges.
- The CF-LIBS inversion loop (IterativeCFLIBSSolver) handles this; the forward
  model takes n_e as an external parameter.

### P9: McWhirter Criterion is Necessary but Not Sufficient
- Satisfying McWhirter does NOT guarantee LTE in a transient LIBS plasma.
- Additional requirements: (1) Cristoforetti relaxation-time criterion for
  excitation equilibrium; (2) Spitzer thermal equilibration for electron-ion
  temperature equality; (3) spatial homogeneity assumption.
- LIBS plasma early after the laser pulse (< 100 ns) typically violates
  the temporal criterion.

### P10: Optical Thickness / Self-Absorption
- The Boltzmann plot and CF-LIBS equations assume OPTICAL THINNESS (each photon
  escapes without reabsorption). Self-absorption distorts line intensities and
  invalidates the linear relationship.
- Verify optical thinness by checking that doublet line intensity ratios match
  the theoretical A_ki × g_k ratio for lines sharing the same upper level.
- Strong resonance lines and lines of abundant elements are most prone to
  self-absorption.

---

## 3. Key References

1. **McWhirter, R. W. P. (1965)**
   "Spectral Intensities"
   In: *Plasma Diagnostic Techniques*, eds. Huddlestone & Leonard, Chapter 5, pp. 201-264
   Academic Press, New York.
   [Canonical source for the 1.6×10^12 criterion and the ΔE definition.]

2. **Ciucci, A., Corsi, M., Palleschi, V., Rastelli, S., Salvetti, A., & Tognoni, E. (1999)**
   "New Procedure for Quantitative Elemental Analysis by Laser-Induced Plasma Spectroscopy"
   *Applied Spectroscopy* 53(8), 960-964.
   DOI: 10.1366/0003702991947612
   [Seminal CF-LIBS paper. Establishes the Saha-Boltzmann + closure equation framework.]

3. **Stewart, J. C. & Pyatt, K. D. (1966)**
   "Lowering of Ionization Potentials in Plasmas"
   *The Astrophysical Journal* 144, 1203-1211.
   ADS: 1966ApJ...144.1203S
   DOI: 10.1086/148714
   [Canonical Stewart-Pyatt IPD model. Interpolates Debye-Hückel and ion-sphere limits.]

4. **Cristoforetti, G., De Giacomo, A., Dell'Aglio, M., Legnaioli, S., Tognoni, E.,
   Palleschi, V., & Omenetto, N. (2010)**
   "Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy:
   Beyond the McWhirter criterion"
   *Spectrochimica Acta Part B* 65, 86-95.
   DOI: 10.1016/j.sab.2009.11.005
   [Critical extension of LTE criteria; defines relaxation-time criterion Eq. 4;
    clarifies that ΔE = first resonance transition, not adjacent gap.]

5. **Griem, H. R. (1964)**
   "Plasma Spectroscopy"
   McGraw-Hill, New York. xii + 580 pp.
   [Foundational text. Relaxation times (used in Cristoforetti criterion).
    Partition function truncation (Inglis-Teller criterion). LTE conditions.]

6. **Tognoni, E., Palleschi, V., Corsi, M., & Cristoforetti, G. (2002)**
   "Quantitative Micro-Analysis by Laser-Induced Breakdown Spectroscopy:
   A Review of the Experimental Approaches"
   *Spectrochimica Acta Part B* 57(7), 1115-1130.
   DOI: 10.1016/S0584-8547(02)00053-8
   [Comprehensive review of LIBS quantification. Derives the Saha-Boltzmann plot
    linearization and multi-element closure equation in usable form.]

7. **Irwin, A. W. (1981)**
   "Polynomial Partition Function Approximations of 344 Atomic and Molecular Species"
   *The Astrophysical Journal Supplement Series* 45, 621-633.
   DOI: 10.1086/190730
   [Standard polynomial partition function fits (log10 T, log10 U basis).
    Used in stellar atmospheres and as fallback in spectroscopy codes.]

8. **Alimohamadi, M. & Ferland, G. J. (2022)**
   "A Practical Guide to the Partition Function of Atoms and Ions"
   arXiv: 2203.02188 [astro-ph.SR]
   [Discusses partition function divergence for complex atoms; IPD-based
    truncation at the continuum lowering limit; plasma vs. isolated-atom regime.]

9. **Griem, H. R. (1997)**
   "Principles of Plasma Spectroscopy"
   Cambridge University Press.
   ISBN: 978-0521455046
   [Updated reference for LTE criteria, Stark broadening, and partition functions.]

10. **Cremers, D. A. & Radziemski, L. J. (2013)**
    "Handbook of Laser-Induced Breakdown Spectroscopy" (2nd ed.)
    Wiley-Blackwell.
    ISBN: 978-0470092996
    [Comprehensive LIBS reference. Chapters on plasma diagnostics, temperature
     measurement via Boltzmann plot, LTE validity, self-absorption correction.]

---

## 4. What Correct Code MUST Do

### Boltzmann Distribution
- [ ] Compute n_k = n_stage × (g_k / U) × exp(-E_k / T_eV), with E_k from the
      GROUND STATE of the same ionization stage
- [ ] Use U(T) computed over the SAME energy range as the level populations
      (same IPD cutoff; never use U from one cutoff and populations from another)
- [ ] Report the emissivity as (h c / 4π λ) × A_ki × n_k with the 1/(4π) factor

### Saha Equation
- [ ] Include the factor of 2 (electron spin degeneracy) in the numerator —
      S = (2 U_{z+1} / U_z) × (const × T^{3/2} / n_e) × exp(-χ_eff / T_eV)
- [ ] Use consistent units throughout: (SAHA_CONST_CM3 ≈ 6.042e21) is valid
      only for T_eV and n_e in cm^-3 and χ in eV
- [ ] Apply IPD: replace χ with χ_eff = max(χ - Δχ, 0) in the Saha exponent
- [ ] Iterate charge balance: n_e = Σ_elem Σ_stage (stage_index × n_stage)

### Partition Function
- [ ] Prefer direct summation over energy levels when levels are available
- [ ] Truncate the sum at E_max = IP - Δχ (IPD-lowered continuum) to avoid
      Rydberg divergence; the SAME Δχ used in the Saha exponent
- [ ] If using polynomial fits: be explicit about log base (ln vs log_10);
      store and validate coefficients in the basis the code uses
- [ ] Warn (log, don't silently return 0) if no energy levels are available
      for a species; fall back to ground-state g (U ≈ g_0) only as last resort

### IPD
- [ ] Use a SINGLE canonical IPD function everywhere (Saha exponent AND
      partition function cutoff AND level population filter — all use same Δχ)
- [ ] Default to Debye-Hückel for typical LIBS conditions; make Stewart-Pyatt
      available as opt-in for higher-density or more accurate work
- [ ] The Debye-Hückel formula in Gaussian-CGS:
      λ_D = sqrt(k_B T [erg/K] / (4π n_e e^2 [esu^2/cm]))
      Δχ = e^2 / λ_D = e [esu] × sqrt(4π n_e / (k_B T))   [result in erg → convert to eV]

### McWhirter LTE Check
- [ ] Use ΔE = resonance-line energy of the element being checked (energy of
      the ground → first-allowed-excited-state transition), NOT the adjacent
      gap between observed Boltzmann-plot upper levels
- [ ] Apply n_e >= 1.6×10^12 × sqrt(T[K]) × (ΔE[eV])^3
- [ ] Flag McWhirter as necessary-not-sufficient; warn users when the criterion
      is satisfied but plasma lifetime is short (< 1 µs) — add Cristoforetti
      relaxation-time check as a diagnostic
- [ ] For the relaxation-time check: τ_rel = 6.3e4 / (n_e × g_eff × f_nm)
      × ΔE × sqrt(T_eV) × exp(ΔE / T_eV)  [seconds]; require τ_rel < τ_evol/10

### Self-Absorption / Optical Thickness
- [ ] Verify optical thinness assumption before applying CF-LIBS; flag or
      correct for self-absorption on strong lines (especially neutral resonance
      lines and lines of major constituents)
- [ ] The forward model emissivity formula only applies in the optically thin
      limit; optically thick radiative transfer requires separate treatment

---

*Note: all web search tools were available and used for this report. The equations
and constants above are well-established and cross-checked against multiple
peer-reviewed sources. The cflibs codebase itself was audited directly via
Read/rg to confirm the actual implementation against these canonical formulas.*
