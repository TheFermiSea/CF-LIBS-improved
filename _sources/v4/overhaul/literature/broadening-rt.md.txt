# Broadening and Radiative Transfer: Canonical Reference
## Topic: broadening-rt

Line emissivity, Doppler/Stark/instrument broadening, Voigt profile, optically-thin
radiative transfer, and spectral normalization for CF-LIBS forward modeling.

---

## 1. Governing Equations — Canonical Form

### 1.1 Line Emissivity (power per unit volume per steradian)

The volumetric emission coefficient (emissivity) for a single spectral line in an
optically-thin LTE plasma is:

```
epsilon_ki  [W m^-3 sr^-1]  =  (h c) / (4 pi lambda_ki)  *  A_ki  *  n_k
```

Equivalently in frequency units:

```
j_nu  [W m^-3 sr^-1 Hz^-1]  =  (h nu_ki) / (4 pi)  *  A_ki  *  n_k  *  phi(nu)
```

where phi(nu) is the normalised line-shape function [Hz^-1], integral = 1.

Symbol table:
- h        = 6.626 070 15e-34 J s           (Planck's constant)
- c        = 2.997 924 58e8 m s^-1           (speed of light)
- lambda_ki = transition wavelength [m]       (air or vacuum — be consistent)
- nu_ki    = c / lambda_ki [Hz]
- A_ki     = Einstein spontaneous emission coefficient [s^-1]   (upper→lower)
- n_k      = population number density of upper level k [m^-3]
- 4 pi     = full solid angle [sr] — the factor accounts for isotropic emission

**Important convention:** subscript k = upper level (emitting), i = lower level
(destination). A_ki goes from k DOWN to i. The factor (hc/lambda)/4pi is energy
per photon divided by 4pi sr; NOT (hc/4pi)/lambda, i.e. c and 4pi are NOT grouped.

Integrated line intensity (power per unit area reaching detector, optically thin,
slab of path length L):

```
I_ki  [W m^-2 sr^-1]  =  epsilon_ki * L
                       =  (h c) / (4 pi lambda_ki)  *  A_ki  *  n_k  *  L
```

### 1.2 Boltzmann Population of Upper Level

Under LTE (local thermodynamic equilibrium):

```
n_k  =  N_s  *  (g_k / Z_s(T))  *  exp(-E_k / k_B T)
```

where:
- N_s   = total number density of species s (neutral or ion) [m^-3]
- g_k   = statistical weight (degeneracy) of upper level k = 2J_k + 1
- Z_s(T) = partition function of species s = sum_i g_i exp(-E_i / k_B T)
- E_k   = excitation energy of upper level above ground [J] or [eV]
           (if E_k in eV, use k_B = 8.617 333e-5 eV K^-1)
- T     = plasma temperature [K]

Substituting into the emissivity:

```
epsilon_ki  =  (h c) / (4 pi lambda_ki)  *  A_ki  *  (g_k / Z_s(T))
               *  N_s  *  exp(-E_k / k_B T)
```

This is the CF-LIBS "line emission coefficient" (Tognoni et al. 2010, eq. 1).

**Pitfall:** Z_s(T) must be evaluated at the same temperature T and must sum over
ALL levels (including continuum threshold), not just those in a truncated table.
Using partition functions tabulated at wrong reference energy zero gives a constant
offset error in Boltzmann plots.

### 1.3 Doppler (Thermal) Broadening — Gaussian Profile

Maxwellian velocity distribution along line-of-sight gives a Gaussian lineshape.

FWHM in wavelength (SI):

```
Delta_lambda_D  =  lambda_ki  *  sqrt(8 k_B T ln2 / (m_s c^2))
               =  lambda_ki  *  (2 / c)  *  sqrt(2 k_B T ln2 / m_s)
```

NIST numerical form (with lambda in Angstroms, T in K, M = atomic weight in amu):

```
Delta_lambda_D [Angstrom]  =  7.16e-7  *  lambda [Angstrom]  *  sqrt(T / M)
```

FWHM in frequency:

```
Delta_nu_D  =  (nu_ki / c)  *  sqrt(8 k_B T ln2 / m_s)
```

Gaussian 1-sigma parameter sigma_D (not FWHM):

```
sigma_D  =  Delta_lambda_D / (2 sqrt(2 ln 2))  =  Delta_lambda_D / 2.3548
```

The normalized Gaussian profile (area = 1):

```
phi_G(lambda)  =  1 / (sigma_D sqrt(2 pi))  *  exp(-(lambda - lambda_ki)^2
                                                      / (2 sigma_D^2))
```

where m_s is the mass of the emitting atom/ion [kg] = M_amu * 1.660 539e-27 kg.

**Pitfall:** The factor under the sqrt is 8 ln2 for FWHM but 8 (without ln2) for
the 1/e^2 width. Confusing FWHM and 1/e half-width is a factor-of-2.355 error
in sigma. Always derive sigma from FWHM via sigma = FWHM / 2.3548.

### 1.4 Stark Broadening — Lorentzian Profile

Electron-impact (linear/quadratic Stark) broadening in the impact approximation
produces a Lorentzian lineshape with FWHM (Griem 1974):

```
Delta_lambda_S  =  2 w  *  (n_e / 10^16)        [cm^-3 reference density]
```

More precisely (Griem 1964, impact approximation):

```
Delta_lambda_S  =  2 [1 + 1.75 A (1 - 0.75 R)] w  *  (n_e / n_ref)
```

where:
- w      = electron-impact half-width parameter at reference temperature T_ref [Angstrom]
           (tabulated in Griem 1974; STARK-B database provides modern values)
- n_e    = electron number density [cm^-3]
- n_ref  = reference density = 10^16 cm^-3 (conventional)
- A      = ion-broadening parameter (usually small for non-H lines; often A ~ 0)
- R      = ratio of mean inter-ion distance to Debye length

For most non-hydrogen lines in LIBS, the simplified form suffices:

```
Delta_lambda_S  ≈  2 w  *  (n_e / 10^16)    [Angstrom, n_e in cm^-3]
```

FWHM is LINEARLY proportional to n_e (electron density diagnostic).

The Lorentzian profile (normalized, area = 1):

```
phi_L(lambda)  =  (1/pi)  *  (gamma_L/2)
                  / ((lambda - lambda_ki)^2  +  (gamma_L/2)^2)
```

where gamma_L = Delta_lambda_S is the Lorentzian FWHM.

### 1.5 Natural Linewidth — Lorentzian (usually negligible in LIBS)

```
gamma_N  =  (1 / 2pi)  *  sum_j A_kj     [Hz]   (HWHM in frequency)
FWHM_N   =  (1 / pi)   *  sum_j A_kj     [Hz]
```

For LIBS plasmas (T ~ 10^4 K, n_e ~ 10^17 cm^-3), Stark broadening dominates by
orders of magnitude; natural linewidth is negligible.

### 1.6 Voigt Profile — Convolution of Gaussian and Lorentzian

The total line profile in a plasma is the convolution of Doppler (Gaussian) and
Stark (Lorentzian) broadening mechanisms:

```
V(x; sigma, gamma)  =  Re[w(z)] / (sigma sqrt(2 pi))
```

where:
```
z = (x + i gamma) / (sigma sqrt(2))
```
and w(z) is the Faddeeva function (scaled complex error function):

```
w(z) = exp(-z^2) * erfc(-i z)
```

Here x = lambda - lambda_ki (detuning from line center), sigma = Gaussian 1-sigma
(from Doppler + instrument combined in quadrature), gamma = Lorentzian HWHM
(from Stark broadening; gamma = Delta_lambda_S / 2).

**Python implementation:**
```python
from scipy.special import wofz
import numpy as np

def voigt(x, sigma, gamma):
    """Normalized Voigt profile. integral = 1."""
    z = (x + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
```

FWHM approximation (error < 0.02%, Thompson et al. 1987 / Olivero & Longbothum 1977):

```
f_V  ≈  0.5343 f_L  +  sqrt(0.2169 f_L^2  +  f_G^2)
```

where f_L = 2*gamma (Lorentzian FWHM), f_G = 2*sigma*sqrt(2 ln 2) (Gaussian FWHM).

Pseudo-Voigt approximation (Thompson et al. 1987):

```
V_pV(x, f)  =  eta * L(x, f)  +  (1 - eta) * G(x, f)
eta  =  1.36603 (f_L/f_V)  -  0.47719 (f_L/f_V)^2  +  0.11116 (f_L/f_V)^3
```

### 1.7 Instrument Broadening

A real spectrometer has a finite resolving power R = lambda / delta_lambda, where
delta_lambda is the minimum resolvable wavelength difference (FWHM of the
instrument line-spread function, LSF). The LSF is typically approximated as
Gaussian.

Instrument FWHM at wavelength lambda:

```
Delta_lambda_inst  =  lambda / R
```

Instrument Gaussian sigma:

```
sigma_inst  =  Delta_lambda_inst / (2 sqrt(2 ln 2))
             =  lambda / (R * 2.3548)
```

The observed profile is the intrinsic physical profile convolved with the instrument
LSF. Since convolution of Gaussians adds variances:

```
sigma_total_Gaussian^2  =  sigma_Doppler^2  +  sigma_inst^2
```

The total Gaussian FWHM:

```
Delta_lambda_G^2  =  Delta_lambda_D^2  +  Delta_lambda_inst^2
```

The full observed profile is then a Voigt with:
- Gaussian component: sigma = sqrt(sigma_Doppler^2 + sigma_inst^2)
- Lorentzian component: gamma = Delta_lambda_S / 2

### 1.8 Optically-Thin Radiative Transfer

The equation of radiative transfer along a ray (path coordinate s):

```
dI_nu / ds  =  -kappa_nu * I_nu  +  j_nu
```

where I_nu = specific intensity [W m^-2 sr^-1 Hz^-1], kappa_nu = absorption
coefficient [m^-1], j_nu = emission coefficient [W m^-3 sr^-1 Hz^-1].

Defining optical depth tau_nu = integral kappa_nu ds (from 0 to L):

```
I_nu(L)  =  I_nu(0) exp(-tau_nu)  +  integral_0^L j_nu(s) exp(-(tau_nu(L) - tau_nu(s))) ds
```

In the OPTICALLY THIN limit (tau_nu << 1 everywhere, exp(-tau) ≈ 1, no
self-absorption):

```
I_nu  =  integral_0^L j_nu(s) ds
```

For a uniform plasma slab of depth L:

```
I_nu  =  j_nu * L  =  (h nu / 4 pi) * A_ki * n_k * phi(nu) * L
```

Spectral radiance integrated over the full line (in wavelength):

```
I_line  =  integral I_lambda d_lambda
         =  (h c / 4 pi lambda_ki) * A_ki * n_k * L
```

This is what is measured by a spectrometer as "line intensity" (minus background).

**Optical thickness criterion:** A line is optically thin when:

```
tau_nu = kappa_nu * L << 1
```

where kappa_nu = (g_i / g_k) * (lambda^2 / 8 pi) * A_ki * (n_i - (g_i/g_k)*n_k) * phi(nu)
(stimulated emission term is usually negligible in LIBS at LTE).

Self-absorption becomes significant when tau_nu approaches or exceeds 1,
particularly for strong resonance lines or high ground-state populations.

### 1.9 Spectral Profile Normalization

A key pitfall: the profile function phi(lambda) or phi(nu) must be normalised:

```
integral_{-inf}^{+inf} phi(lambda) d_lambda  =  1     [m^-1 or Angstrom^-1]
integral_{-inf}^{+inf} phi(nu)     d_nu      =  1     [Hz^-1]
```

The wavelength and frequency normalised profiles are related by:
phi(lambda) = phi(nu) * (c / lambda^2).

For a Gaussian:
```
phi_G(lambda)  =  1/(sigma sqrt(2pi)) * exp(-(lambda-lambda0)^2 / (2 sigma^2))
```
integral = 1 by construction.

For a Voigt via Faddeeva (see 1.6):
```
integral Re[w(z)] / (sigma sqrt(2pi)) d_x  =  1
```
This is guaranteed because w(z) has integral Re[w] dx = sqrt(pi) * sigma * sqrt(2),
so the prefactor 1/(sigma sqrt(2pi)) gives exactly 1.

---

## 2. Common Implementation Pitfalls

### P1: Wrong sign/order in subscript convention
- A_ki: k=upper → i=lower. E_k > E_i. Using A_ik (wrong direction) with the
  Boltzmann factor exp(-E_k/kT) gives wrong population weighting.

### P2: Missing 1/lambda factor in emissivity
- The emissivity is epsilon = (hc/4pi/lambda) * A_ki * n_k, NOT hc * A_ki * n_k / 4pi.
- The lambda dependence comes from converting photon energy h*nu = h*c/lambda.
- Omitting it causes a systematic wavelength-dependent error in relative line intensities
  (affects Boltzmann-plot slopes → wrong temperature).

### P3: FWHM vs sigma confusion for Doppler width
- FWHM = 2.3548 * sigma (for Gaussian).
- The Doppler FWHM formula contains sqrt(8 k_B T ln2 / m), which includes ln2.
- A common bug: using sqrt(2 k_B T / m) as FWHM (correct only for 1/e half-width).
- Correct: sigma_D = (lambda/c) * sqrt(k_B T / m).
  Then FWHM_D = 2.3548 * sigma_D.

### P4: Instrument sigma from resolving power
- FWHM_inst = lambda / R.
- sigma_inst = lambda / (R * 2.3548), NOT lambda / R.
- Off by factor 2.3548 if sigma and FWHM are confused.

### P5: Adding widths in quadrature (Gaussian components only)
- FWHM^2 of composite Gaussian = sum of FWHM^2 (not FWHM = sum FWHM).
- This applies ONLY to Gaussian components (Doppler + instrument).
- Lorentzian widths (Stark, natural) add LINEARLY: gamma_total = gamma_Stark + gamma_natural.
- Voigt combines them: compute sigma_total then convolve (Faddeeva).

### P6: Voigt profile normalization
- scipy.special.wofz returns the Faddeeva function w(z).
- The normalised Voigt is Re[w(z)] / (sigma * sqrt(2*pi)), NOT just Re[w(z)].
- Without the prefactor, the peak height is correct but the profile is not normalized,
  causing errors when integrating for line flux.

### P7: Optical depth / self-absorption check
- Always verify tau_max << 1 for every line used in analysis.
- Strong resonance lines (e.g., Ca II 393 nm, Mg II 280 nm) are frequently optically
  thick in LIBS plasmas; never use them in CF-LIBS without a self-absorption correction.
- Neglecting self-absorption → underestimated abundance of that element.

### P8: Wavelength vs frequency normalization
- phi_G(lambda) [1/m] ≠ phi_G(nu) [1/Hz]. They are related by phi(lambda) = phi(nu) * c/lambda^2.
- If computing spectrum on a wavelength grid, use phi(lambda); on a frequency grid,
  use phi(nu). Mixing them introduces c/lambda^2 errors.

### P9: Partition function reference energy
- Z(T) = sum_i g_i exp(-E_i / kT) with energies measured from ground state (E_0 = 0).
- Using energies measured from a non-zero reference (e.g., ionization limit) gives
  Z' = Z * exp(-E_ref / kT), which cancels in relative intensities but NOT in absolute ones.

### P10: Air vs vacuum wavelengths
- NIST ASD provides both air and vacuum wavelengths. Doppler formula uses the rest
  frame wavelength. Ensure consistency: either convert all to vacuum or all to air.
- Difference is ~0.03% at 500 nm (refractive index of air n ≈ 1.00028).

---

## 3. Key References

1. **Griem, H. R. (1964).** *Plasma Spectroscopy.* McGraw-Hill, New York.
   — Foundational: Stark broadening theory in the impact approximation; electron
     impact width/shift parameters w and d; Lorentzian profile formula.

2. **Griem, H. R. (1974).** *Spectral Line Broadening by Plasmas.* Academic Press, New York.
   — ISBN 978-0-12-302850-1. Extended tables of w and d for many transitions;
     the standard reference for Stark parameters in plasma diagnostics.

3. **Cowan, R. D. (1981).** *The Theory of Atomic Structure and Spectra.* University of
   California Press, Berkeley.
   — Canonical treatment of oscillator strengths, A_ki, g-values, and line strengths;
     the standard reference for these atomic data.

4. **Tognoni, E., Cristoforetti, G., Legnaioli, S., & Palleschi, V. (2010).**
   "Calibration-Free Laser-Induced Breakdown Spectroscopy: State of the art."
   *Spectrochimica Acta Part B*, 65(1), 1–14. DOI: 10.1016/j.sab.2009.11.006.
   — The definitive CF-LIBS review; Eq. 1 gives the line emission coefficient in the
     form epsilon_ki = (hc/4pi/lambda) A_ki (g_k/Z) N_s exp(-E_k/kT); discusses
     optically-thin assumption, LTE validity, partition functions.

5. **Ciucci, A., Corsi, M., Palleschi, V., Salvetti, A., & Tognoni, E. (1999).**
   "New procedure for quantitative elemental analysis by laser-induced plasma spectroscopy."
   *Applied Spectroscopy*, 53(8), 960–964. DOI: 10.1366/0003702991947612.
   — Original CF-LIBS paper; line intensity formula; closure constraint sum C_s = 1.

6. **Cremers, D. A. & Radziemski, L. J. (2013).** *Handbook of Laser-Induced Breakdown
   Spectroscopy*, 2nd ed. Wiley, Chichester. ISBN 978-1-119-97112-2.
   — Practical reference for LIBS: Boltzmann plot, Saha equation, optically-thin
     conditions, broadening mechanisms, typical LIBS plasma parameters.

7. **Olivero, J. J. & Longbothum, R. L. (1977).**
   "Empirical fits to the Voigt line width: A brief review."
   *Journal of Quantitative Spectroscopy and Radiative Transfer*, 17(2), 233–236.
   DOI: 10.1016/0022-4073(77)90161-3.
   — The fV ≈ 0.5343 fL + sqrt(0.2169 fL^2 + fG^2) FWHM approximation formula.

8. **Thompson, P., Cox, D. E., & Hastings, J. B. (1987).**
   "Rietveld refinement of Debye–Scherrer synchrotron X-ray data from Al2O3."
   *Journal of Applied Crystallography*, 20(2), 79–83. DOI: 10.1107/S0021889887087090.
   — Pseudo-Voigt mixing parameter eta formula; widely cited for spectroscopy.

9. **Rybicki, G. B. & Lightman, A. P. (1979).** *Radiative Processes in Astrophysics.*
   Wiley, New York. ISBN 0-471-82759-2.
   — Standard graduate text for radiative transfer equation; emission coefficient j_nu,
     absorption coefficient kappa_nu; optically-thin limit derivation (Chapter 1).

10. **Mihalas, D. (1978).** *Stellar Atmospheres*, 2nd ed. W. H. Freeman, San Francisco.
    — Radiative transfer equation; specific intensity I_nu; source function S = j/kappa;
      formal solution; detailed balance; stimulated emission treatment.

11. **NIST Atomic Spectra Database (ASD):** https://physics.nist.gov/asd
    — Primary source for A_ki, g_k, E_k, lambda values; Doppler FWHM formula cited
      at https://www.nist.gov/pml/atomic-spectroscopy-compendium-basic-ideas-notation-data-and-formulas/atomic-spectroscopy-6

12. **Harilal, S. S., Phillips, M. C., Froula, D. H., Anoop, K. K., Issac, R. C., & Beg, F. N. (2022).**
    "Optical diagnostics of laser-produced plasmas."
    *Reviews of Modern Physics*, 94, 035002. DOI: 10.1103/RevModPhys.94.035002.
    — Comprehensive review of LPP diagnostics; Doppler + Stark broadening measurement
      techniques; Voigt fitting; temperature and density measurements.

---

## 4. What Correct Code MUST Do — Checklist

- [ ] **Emissivity formula:** Use `(h*c) / (4*pi*lambda_ki) * A_ki * n_k`.
       Never omit the 1/lambda factor; never use photon energy hnu without dividing
       by 4*pi for isotropic emission.

- [ ] **Upper level population:** `n_k = N_s * g_k / Z_s(T) * exp(-E_k / (k_B*T))`
       with E_k measured from ground state (E=0), g_k = 2J_k+1, Z_s summed from ground.

- [ ] **Doppler sigma (Gaussian):** `sigma_D = lambda / c * sqrt(k_B * T / m_s)`.
       FWHM_D = 2.3548 * sigma_D. Do NOT use FWHM formula directly as sigma.

- [ ] **Instrument sigma:** `sigma_inst = lambda / (R * 2.3548)` where R = resolving power.
       FWHM_inst = lambda/R, then convert to sigma by dividing by 2.3548.

- [ ] **Combined Gaussian sigma:** `sigma_G = sqrt(sigma_D^2 + sigma_inst^2)`.
       This is the correct quadrature sum for Gaussian widths.

- [ ] **Stark broadening (Lorentzian HWHM):** `gamma_S = w * (n_e / 1e16) [cm^-3]`
       using tabulated w from Griem 1974 or STARK-B database.

- [ ] **Voigt profile via Faddeeva:** `V(x) = Re[wofz((x + 1j*gamma_S) / (sigma_G*sqrt(2)))] / (sigma_G * sqrt(2*pi))`.
       Profile must be normalised: integral = 1.

- [ ] **Spectrum assembly:** `I(lambda) = sum_over_lines [emissivity_ki * V(lambda - lambda_ki)]`.
       Each line is an independently scaled normalised profile. Sum gives total
       spectral radiance per unit wavelength [W m^-2 sr^-1 m^-1].

- [ ] **Optically-thin check:** Verify tau_max < 0.1 for all lines used in inversion.
       Flag or discard lines with tau > 0.3. Never use strong resonance lines in
       CF-LIBS without self-absorption correction.

- [ ] **Wavelength consistency:** Use the same wavelength convention (air or vacuum)
       throughout. NIST ASD provides both; choose one and document it.

- [ ] **Partition function:** Evaluate Z_s(T) at the relevant T; use polynomial
       approximations (Irwin 1981 or similar) for speed; verify against NIST at
       test temperatures.

- [ ] **Normalised line profile integral test:** Unit test that
       `np.trapz(voigt(x, sigma, gamma), x) ≈ 1.0` for a fine x grid.
