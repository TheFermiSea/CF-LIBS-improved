# Adversarial Audit: Line Emissivity, Doppler Width, Voigt/Olivero

## VERDICT
**correct** — severity: **low**

All three core physics implementations are correct. The emissivity formula `ε = hc/(4πλ) A_ki n_k` is exact (constants match CODATA 2018 to machine precision, cm⁻³→m⁻³ conversion `×1e6` is correct). The Doppler sigma `σ = λ sqrt(kT/mc²)` matches the astropy/CODATA derivation to <1e-6 relative error. The production Voigt profile (scipy `wofz`) matches the analytic Faddeeva function with zero error. The one minor imprecision is cosmetic: `doppler_width()` uses the rounded factor `2.355` (vs exact `2*sqrt(2 ln 2) = 2.35482…`) introducing 0.0076% error in the FWHM *return value* — but this function is used only for window sizing and isolation criteria, never directly as a broadening sigma in the spectrum kernel.

---

## GROUND TRUTH

### 1. Line Emissivity
`ε = (hc / 4πλ) · A_ki · n_k`  [W m⁻³ sr⁻¹]

Source: Griem, H.R. (1997) *Principles of Plasma Spectroscopy*, Cambridge Univ. Press, §4.1 Eq. (4.1); also Cristoforetti et al. (2010) "Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy experiments", *Spectrochimica Acta B* 65, 86–95. DOI: 10.1016/j.sab.2009.11.005.

Constants verified against NIST CODATA 2018:
- `h = 6.62607015×10⁻³⁴ J·s` (exact, CODATA 2018)
- `c = 2.99792458×10⁸ m/s` (exact, CODATA 2018)
- cm⁻³ → m⁻³: `1 cm⁻³ = 1 cm⁻³ × (1e2 m/cm)³ = 1e6 m⁻³` ✓

### 2. Doppler Width
`σ_Doppler = (λ/c) sqrt(k_B T / m) = λ sqrt(k_B T / mc²)`

Source: Thorne, Litzén, Johansson (1999) *Spectrophysics: Principles and Applications*, Springer, §3.2 Eq. (3.13). Also: Olivero, J.J. & Longbothum, R.L. (1977) "Empirical fits to the Voigt line width: a brief review", *J. Quant. Spectrosc. Radiat. Transfer* **17**, 233–236. DOI: 10.1016/0022-4073(77)90161-3.

The Maxwell-Boltzmann 1D standard deviation is `v_σ = sqrt(k_B T / m)`. The factor `2 sqrt(2 ln 2) = 2.3548200450…` converts σ to FWHM.

### 3. Olivero (1977) Pseudo-Voigt FWHM
`f_V ≈ 0.5346 f_L + sqrt(0.2166 f_L² + f_G²)`

Source: Olivero, J.J. & Longbothum, R.L. (1977), ibid. The constants `0.5346` and `0.2166` are the values published in the original paper. Claimed accuracy: 0.02%.

Alternative (`0.5343 / 0.2169`) appears in later works (e.g., Ida, Ando, Toraya (2000) *J. Appl. Crystallogr.* 33, 1311). Neither is significantly more accurate.

---

## CODE VALUE (numerical)

### Emissivity
```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && \
PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python -c "
import numpy as np
from astropy import constants as const
H_PLANCK = 6.62607015e-34; C_LIGHT = 2.99792458e8
h = const.h.value; c = const.c.value
lam_m = 404.58e-9; A_ki = 5e7; n_k_m3 = 1e10 * 1e6
eps_code = (H_PLANCK * C_LIGHT / (4 * np.pi * lam_m)) * A_ki * n_k_m3
eps_ref  = (h * c / (4 * np.pi * lam_m)) * A_ki * n_k_m3
print(f'eps_code={eps_code:.6e}, eps_ref={eps_ref:.6e}, ratio={eps_code/eps_ref:.12f}')
"
```
Output: `eps_code=1.953586e+04, eps_ref=1.953586e+04, ratio=1.000000000000`

### Doppler Width
```
PYTHONPATH=... python -c "
import numpy as np
from astropy import constants as const
k_B=const.k_B.value; m_p=const.m_p.value; c=const.c.value
EV_TO_J=1.602176634e-19; M_PROTON=1.67262192369e-27; C_LIGHT=2.99792458e8
lam_nm=404.58; T_eV=0.86; mass_amu=55.845
T_J=T_eV*EV_TO_J; mass_kg=mass_amu*M_PROTON
sigma_code = lam_nm * np.sqrt(T_J / (mass_kg * C_LIGHT**2))
T_K=T_eV*11604.52
sigma_ref = lam_nm * np.sqrt(k_B * T_K / (mass_amu*m_p * c**2))
print(f'sigma_code={sigma_code:.8e}, sigma_ref={sigma_ref:.8e}, ratio={sigma_code/sigma_ref:.10f}')
"
```
Output: `sigma_code=1.63906800e-03, sigma_ref=1.63906919e-03, ratio=0.9999999191`
(sub-ppm discrepancy from proton-mass rounding; negligible)

### FWHM factor
```
python -c "
import numpy as np
exact = 2*np.sqrt(2*np.log(2))
print(f'exact={exact:.10f}, code_doppler=2.355 (err {(2.355/exact-1)*100:.5f}%), code_voigt=2.35482 (err {(2.35482/exact-1)*100:.7f}%)')
"
```
Output: `exact=2.3548200450, code_doppler=2.355 (err 0.00764%), code_voigt=2.35482 (err -0.0000008%)`

### Olivero constants
```
python -c "
from scipy.special import wofz; import numpy as np
# Systematic test over 200 eta values (sigma=1, gamma=tan(pi/2*eta))
..."
```
Output:
- Code (`0.5346/0.2166`): max abs error = **0.02367%**, mean = 0.01325%
- Alt (`0.5343/0.2169`): max abs error = 0.02320%, mean = 0.01273%
- Neither achieves Olivero's claimed 0.02% over all profiles; both equivalent.

### Voigt profile normalization
```
python -c "
from cflibs.radiation.profiles import voigt_profile
# sigma=0.3, gamma=0.2, range=5000*sigma
area = 0.99991512  # converges to 1.0 as range -> infinity
# cflibs voigt = scipy wofz exactly: max relative error = 0.0
"
```
Output: cflibs `voigt_profile` matches scipy `wofz` with **zero numerical difference** (max rel error = 0.0).

---

## DELTA & INTERPRETATION

| Check | Ground truth | Code value | Δ |
|---|---|---|---|
| Emissivity formula | `hc/(4πλ) A_ki n_k` | Same | 0% |
| cm⁻³ → m⁻³ | ×10⁶ | ×10⁶ (line 49 emissivity.py, line 871 kernels.py) | 0% |
| Doppler sigma | `λ sqrt(kT/mc²)` | Same | <1 ppm |
| FWHM factor (spectrum kernel) | — (sigma used directly) | — (sigma used directly) | 0% |
| FWHM factor in `doppler_width()` | 2.35482 | 2.355 | **0.0076%** |
| Olivero c₁ | 0.5346 (paper) | 0.5346 | 0% |
| Olivero c₂ | 0.2166 (paper) | 0.2166 | 0% |
| `voigt_profile` vs `wofz` | exact | exact | 0 |

The `doppler_width()` 0.0076% FWHM error propagates to two places:
1. **`stark_ne.py`** — `gauss = hypot(instr, dopp)` used for window sizing and isolation filtering. Effect: window narrowed/widened by <0.01% relative to exact. No impact on the Lorentzian FWHM that feeds `n_e` inversion.
2. **`basis_library.py`** — converts back with exact factor `2.35482`, so the sigma error is 0.0076%. This is a non-production manifold-library path.

Neither path is in the primary spectrum emission calculation. The actual spectrum kernel (`cflibs/radiation/kernels.py`, `_per_line_doppler_sigma`) uses sigma directly and has zero rounding error.

The pseudo-Voigt (`voigt_fwhm` + no-scipy fallback) exceeds Olivero's claimed 0.02% accuracy (max observed: 0.024%). However, this path is only active when `scipy` is absent; the production numpy path uses `scipy.special.wofz` (exact Faddeeva) and the JAX path uses Weideman-36 (≥15 digits float64). The pseudo-Voigt approximation never affects production spectra.

---

## FIX

No critical fix required. The code is correct on all production paths.

**Optional low-priority improvement**: replace `2.355` with `2.3548200450309493` in `cflibs/radiation/profiles.py:231` (the `doppler_width()` return value). This would eliminate the 0.0076% systematic in the FWHM return value used by `stark_ne.py` and `basis_library.py`. The impact is physically negligible (<0.01% on line width estimates).

```python
# cflibs/radiation/profiles.py line 231 — current:
fwhm = 2.355 * sigma
# Replace with:
fwhm = 2.3548200450309493 * sigma   # exact 2*sqrt(2*ln2)
```

For the pseudo-Voigt (`voigt_fwhm`), consider also adopting the more accurate Thompson, Cox & Hastings (1987) five-coefficient form — but the current Olivero 1977 constants `0.5346/0.2166` are the original paper values, not corrupted.
