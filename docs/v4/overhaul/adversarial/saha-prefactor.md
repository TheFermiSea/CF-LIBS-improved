# Adversarial Audit: Saha Equation Prefactor, Units, and eV/K Convention

**Domain:** `SAHA_CONST_CM3` in `cflibs/core/constants.py` + full Saha implementation in `cflibs/plasma/saha_boltzmann.py`
**Auditor:** Adversarial physics audit — assumes code is WRONG unless proven otherwise.

---

## VERDICT

**correct** — severity: **low**

The Saha equation structure (exponent form, eV/K convention, spin-degeneracy factor of 2, partition function ratio U_II/U_I) is internally consistent and physically correct. The prefactor `SAHA_CONST_CM3 = 6.042e21` is **0.0806% above** the CODATA-exact value of `6.03713370e21 cm^-3 eV^-3/2`. This is a stale rounded value from pre-2014 constants, not a structural error. The induced temperature error at LIBS conditions (T ≈ 2 eV, IP_Fe = 7.9 eV) is **~5 K out of ~23,000 K** (~0.02%), three orders of magnitude below the 10–20% Boltzmann-plot temperature uncertainty. The eV-temperature convention is applied consistently throughout.

---

## GROUND TRUTH

### Governing Equation

The Saha equation in the form used by the code (McWhirter / CF-LIBS convention):

```
n_{z+1} * n_e / n_z = C_Saha * T_eV^{3/2} * (U_{z+1} / U_z) * exp(-chi_eV / T_eV)
```

where:
- `n` in cm^-3, `T_eV` in eV (= k_B T / e), `chi_eV` in eV
- `C_Saha = 2 * (2π m_e k_B / h²)^{3/2} * (eV/k_B)^{3/2} * 1e-6` [cm^-3 eV^-3/2]
- The factor of 2 is the electron spin degeneracy g_e = 2

### Derivation from CODATA 2018 (astropy)

```
m_e = 9.1093837015e-31 kg   (CODATA 2018, exact)
k_B = 1.380649e-23 J/K     (CODATA 2018, exact by definition)
h   = 6.62607015e-34 J*s   (CODATA 2018, exact by definition)
eV  = 1.602176634e-19 J    (exact by SI definition)

C0_SI  = 2 * (2π m_e k_B / h²)^{3/2}  =  4.82936608e+21 m^-3 K^-3/2
C0_cm3 = C0_SI * 1e-6                   =  4.82936608e+15 cm^-3 K^-3/2
C_eV   = C0_cm3 * (eV/k_B)^{3/2}       =  6.03713370e+21 cm^-3 eV^-3/2
```

### Citations

1. **McWhirter, R.W.P. (1965).** "Spectral intensities." In *Plasma Diagnostic Techniques*, Ch. 5. Academic Press. [Canonical CF-LIBS Saha form with spin degeneracy factor 2 included in prefactor.]

2. **Cristoforetti, G., De Giacomo, A., Dell'Aglio, M., Legnaioli, S., Tognoni, E., Salvetti, A., & Palleschi, V. (2010).** "Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy: Beyond the McWhirter criterion." *Spectrochimica Acta Part B*, 65(1), 86–95. DOI: 10.1016/j.sab.2009.11.005. [Cites K-form prefactor 4.829e21 cm^-3 K^-3/2 — consistent with CODATA within 0.008%.]

3. **NIST CODATA 2018.** Physical constants: m_e, k_B, h, e. https://physics.nist.gov/cuu/Constants/

4. **Griem, H.R. (1964).** *Plasma Spectroscopy.* McGraw-Hill. [Derivation of Saha equation in CGS/mixed units, spin factor 2.]

5. **Tognoni, E., Cristoforetti, G., Legnaioli, S., & Palleschi, V. (2010).** "Calibration-free laser-induced breakdown spectroscopy: State of the art." *Spectrochimica Acta Part B*, 65(1), 1–14. DOI: 10.1016/j.sab.2009.07.006. [Standard CF-LIBS Saha formulation.]

---

## CODE VALUE (numerical)

### Command 1 — Verify import path:
```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python -c "import cflibs; print(cflibs.__file__)"
```
Output: `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/cflibs/__init__.py`

### Command 2 — Read constant from code:
File: `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/cflibs/core/constants.py` line 76:
```python
SAHA_CONST_CM3 = 6.042e21  # cm^-3 eV^-1.5 (pre-factor for T in eV, n_e in cm^-3)
```

### Command 3 — Independent derivation from astropy/CODATA:
```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=... python -c "
from astropy import constants as const
import numpy as np
m_e = const.m_e.value; k_B = const.k_B.value; h = const.h.value; eV = 1.602176634e-19
C_SI = 2 * (2 * np.pi * m_e * k_B / h**2)**1.5
C_eV_cm3 = C_SI * 1e-6 * (eV/k_B)**1.5
print(f'CODATA derivation: {C_eV_cm3:.8e}')
print(f'Code value:        6.042e21')
print(f'Error:             {(6.042e21 - C_eV_cm3)/C_eV_cm3*100:.4f}%')
"
```
Output:
```
CODATA derivation: 6.03713370e+21
Code value:        6.042e21
Error:             0.0806%
```

### Command 4 — Verify eV/K convention in Boltzmann exponent:
Code at `saha_boltzmann.py` line 224:
```python
S1 = (SAHA_CONST_CM3 / n_e_cm3) * (T_e_eV**1.5) * (U_II / U_I) * np.exp(-eff_ip_I / T_e_eV)
```
Exponent `exp(-chi[eV] / T[eV])` = `exp(-chi / k_B T)` — **dimensionally correct** since T_e_eV = k_B*T / eV.

Level population Boltzmann factor at `saha_boltzmann.py` line 355:
```python
U += level.g * np.exp(-level.energy_ev / T_e_eV)
```
Same convention: `exp(-E[eV] / T[eV])` — **correct**.

### Command 5 — Verify EV_TO_K vs CODATA:
```
CODATA k_B/e = 8.617333262145e-05 eV/K
Code  KB_EV  = 8.617333262e-05 eV/K
Difference   = 1.45e-15  (sub-femto eV/K, negligible)
```

### Command 6 — Verify JAX kernel uses same constant:
`cflibs/plasma/kernels.py` line 30 imports `SAHA_CONST_CM3` from `cflibs.core.constants`, line 76 applies it identically. Both NumPy and JAX paths use the same value.

---

## DELTA & INTERPRETATION

| Quantity | CODATA-exact | Code | Error |
|----------|-------------|------|-------|
| `SAHA_CONST_CM3` (cm^-3 eV^-3/2) | 6.03713370e21 | 6.042e21 | +0.0806% |
| EV_TO_K (K/eV) | 11604.5182 | 11604.5 (from KB_EV) | ~1 ppm |
| M_E (kg) | 9.1093837015e-31 | 9.1093837015e-31 | 0 |

**Physical impact of 0.0806% prefactor error:**

The Saha ratio S1 = C * T^1.5 * (U_II/U_I) * exp(-chi/T) is proportional to C. An 0.0806% high C means all ionization ratios are 0.0806% high. For inversion (where T is inferred from Saha), the induced temperature error is:

```
dT/T = (dC/C) * T_eV/chi_eV   (dominant term at chi >> T)
```

At T = 2 eV, IP_Fe = 7.9024 eV:
```
dT/T = 0.0806% * 2/7.9 = 0.0204%
dT   = 0.000204 * 2 eV * 11604 K/eV ≈ 4.7 K  out of  23,208 K
```

For comparison, Boltzmann-plot temperature uncertainties in LIBS are typically 10–20% (2,000–4,000 K). The prefactor error is **~500× smaller** than measurement noise.

**Impact on composition:** For n_II/n_I ratio used as Saha correction in CF-LIBS inversion, a 0.0806% error in S1 propagates to an identical fractional error in composition correction — well below any experimental accuracy threshold.

**Structural correctness:**
- Saha exponent: `exp(-chi_eV / T_eV)` is correct (dimensionless, equivalent to exp(-chi/k_B T))
- Partition function ratio: `U_II / U_I` is correctly used WITHOUT an additional factor of 2 (the spin degeneracy is already baked into SAHA_CONST_CM3)
- Three-stage balance: `n_III / n_II = S2` computed correctly with the second ionization potential
- JAX kernel and NumPy solver are consistent (both import same SAHA_CONST_CM3)

---

## FIX

The code is **structurally correct**. The only finding is a stale prefactor value rounded from older (pre-CODATA-2014) constants. If tightened:

**File:** `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/cflibs/core/constants.py` line 76

```python
# Current (stale, 0.08% high):
SAHA_CONST_CM3 = 6.042e21  # cm^-3 eV^-1.5

# Corrected to CODATA 2018 (m_e, k_B, h exact by SI definition):
SAHA_CONST_CM3 = 6.03713370e21  # cm^-3 eV^-1.5 (CODATA 2018: 2*(2π m_e k_B/h²)^1.5 * (eV/k_B)^1.5 * 1e-6)
```

**Severity assessment:** Low — the physical impact (4.7 K temperature error at LIBS conditions) is three orders of magnitude below the experimental noise floor. This is a cosmetic tightening, not a physics bug. Leaving it unchanged would be defensible; updating it eliminates any future confusion.

**What would have to be true for this to be wrong:** If the code had accidentally omitted the factor of 2 (spin degeneracy), the constant would be 3.019e21 — half the CODATA value — and the ionization balance would be systematically off by 2×. That would induce ~40–70% errors in ionic-to-neutral ratios and would cause wildly incorrect Saha corrections to the Boltzmann temperature. But the code's comment at line 72–74 correctly documents that the spin factor is included, and the numerical value (6.042 vs 6.037 CODATA) confirms the factor of 2 is present (half would be 3.018e21).
