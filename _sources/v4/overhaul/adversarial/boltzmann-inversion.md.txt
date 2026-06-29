# Adversarial Audit: Boltzmann-Plot & Saha-Boltzmann-Plane Inversion Math

Audited files:
- `cflibs/inversion/physics/boltzmann.py` — `BoltzmannPlotFitter`, `LineObservation.y_value`
- `cflibs/inversion/common/data_structures.py` — `LineObservation` dataclass
- `cflibs/inversion/solve/iterative.py` — `_apply_saha_correction`, `_fit_common_boltzmann_plane`, `_create_result`
- `cflibs/inversion/physics/closure.py` — `ClosureEquation.apply_standard`
- `cflibs/core/constants.py` — `KB_EV`, `SAHA_CONST_CM3`

---

## VERDICT

**correct** — severity **none** (one cosmetic comment typo noted, zero physics errors)

The Boltzmann-plot linearization, Saha-Boltzmann y-shift, multi-element common-slope fit, temperature extraction, and closure equation are all mathematically correct. A prior audit claim that the `ln(U_II/U_I)` term is MISSING from the ionic-line y-shift is **FALSE** — the derivation below shows it cancels exactly via the Saha equation, and the code correctly omits it.

---

## GROUND TRUTH

### 1. Boltzmann plot y-axis (Ciucci 1999, eq. 1; Tognoni 2010, eq. 2)

```
y = ln(I_ki * lambda_ki / (g_k * A_ki)) = -E_k / (k_B T) + q_s
```

Slope = -1/(k_B T), intercept q_s = ln(n_s / U_s) + C_instrument.

**Citations:**
- Ciucci, A. et al. (1999). "New Procedure for Quantitative Elemental Analysis by LIBS." *Applied Spectroscopy* 53(8), 960–964. DOI: 10.1366/0003702991947612
- Tognoni, E. et al. (2010). "Calibration-free laser-induced breakdown spectroscopy: State of the art." *Spectrochimica Acta B* 65, 1-14. DOI: 10.1016/j.sab.2009.11.006
- Aragon, C. & Aguilera, J.A. (2010). "Characterization of laser induced plasmas." *Spectrochim. Acta B* 65, 395–408. DOI: 10.1016/j.sab.2010.05.004

### 2. Saha-Boltzmann ionic→neutral y-shift (derivation)

For a stage-II (ionic) line, the raw y-value is:
```
y_II = -E_k^II/(kT) + ln(n_II / U_II) + C_inst
```
To map onto the neutral plane (same q_s = ln(n_I/U_I) + C_inst), we need:
```
y_mapped = -(E_k^II + IP_I)/(kT) + q_s
shift = y_mapped - y_II = -IP_I/(kT) + ln(n_I/U_I) - ln(n_II/U_II)
      = -IP_I/(kT) + ln(n_I/n_II) + ln(U_II/U_I)
```
Substituting the Saha equation:
```
n_II/n_I = SAHA_CONST * T_eV^{3/2} * (U_II/U_I) * exp(-IP_I/kT) / n_e
=> ln(n_I/n_II) = ln(n_e) - ln(SAHA_CONST) - 1.5 ln(T_eV) - ln(U_II/U_I) + IP_I/(kT)
```
So:
```
shift = -IP_I/(kT)
        + [ln(n_e) - ln(SAHA_CONST) - 1.5 ln(T_eV) - ln(U_II/U_I) + IP_I/(kT)]
        + ln(U_II/U_I)
      = ln(n_e) - ln(SAHA_CONST) - 1.5 ln(T_eV)
      = ln(n_e / (SAHA_CONST * T_eV^{3/2}))
```

**The ln(U_II/U_I) term cancels exactly.** The correct y-shift is partition-function-independent.

### 3. SAHA_CONST_CM3 from CODATA (astropy)

Computed from CODATA 2018 via `astropy.constants`:
```
h = 6.626070e-27 erg·s, k_B = 1.380649e-16 erg/K, m_e = 9.109384e-28 g
CONST_K = 2 * (2π m_e k_B)^{3/2} / h^3 = 4.829366e15 cm^-3 K^{-3/2}
CONST_eV = CONST_K * EV_TO_K^{3/2} = 6.037134e21 cm^-3 eV^{-3/2}
```
Reference: Griem, H.R. (1997). *Principles of Plasma Spectroscopy.* Cambridge. ISBN: 978-0521455046.

---

## CODE VALUE (numerical)

### Test 1: y_value property

```bash
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python -c "
from cflibs.inversion.common.data_structures import LineObservation
import numpy as np
obs = LineObservation(wavelength_nm=400.0, intensity=1000.0, intensity_uncertainty=10.0,
    element='Fe', ionization_stage=1, E_k_ev=3.5, g_k=5, A_ki=1e7)
print(obs.y_value, np.log(1000.0 * 400.0 / (5 * 1e7)))
"
```
Output: `-4.828314 -4.828314` — code matches literature formula exactly.

### Test 2: Saha-Boltzmann y-shift correctness

Ran a forward+inverse round-trip with T=8000 K, n_e=1e17 cm^-3, synthetic Fe I and Fe II lines with U_I=25, U_II=50, IP=7.9024 eV:

```bash
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=... JAX_PLATFORMS=cpu python -c "<...>"
```

**Output:**
```
correction_term = ln(SAHA_CONST * T_eV^1.5 / n_e) = 10.4511
After Saha correction:
  Expected y on neutral plane: [-8.067, -9.518, -10.969, -12.419, -13.870]
  Actual corrected y:          [-8.067, -9.518, -10.969, -12.419, -13.870]
Recovered slope: -1.4506 (true: -1.4506)
Recovered T: 8000 K (true: 8000 K), error = 0 K
```

### Test 3: SAHA_CONST_CM3

Code value: `6.042e21 cm^-3 eV^-3/2`  
CODATA derivation: `6.037134e21 cm^-3 eV^-3/2`  
Relative error: **0.08%** (sub-milliradian)

### Test 4: KB_EV

Code: `8.617333262e-5 eV/K` vs CODATA `k_B = 8.617333262e-5 eV/K` — **exact match**.

### Test 5: Temperature formula

`T = -1.0 / (KB_EV * slope)` — round-trip verified, recovers true T=8000 K from slope=-1.4506 eV^-1 within floating-point precision.

### Test 6: Closure equation

`ClosureEquation.apply_standard` with q_Fe=10, U_Fe=25, mult_Fe=1.5, q_Si=12, U_Si=4, mult_Si=1.1 gives C_Fe=0.5356, C_Si=0.4644, sum=1.0. Manual calculation agrees to 1e-10.

### Test 7: BoltzmannPlotFitter integration

```bash
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=... python -c "..."
```
On 8-point synthetic data with 5% log-space noise:
- True T=8500 K → Fitted T=8450 K (0.6% error, within noise)
- R^2=0.9993, n_points=8, slope=-1.3734 eV^-1

---

## DELTA & INTERPRETATION

| Item | Code | Literature/CODATA | Delta |
|------|------|-------------------|-------|
| y-axis formula | `ln(I*λ/(g*A))` | Ciucci 1999 Eq. 1 | **0 (exact)** |
| Slope → T | `-1/(KB_EV*slope)` | -1/(k_B slope) | **0 (exact)** |
| Ionic x-shift | `E_k + IP` | E_k^II + IP_I | **0 (exact)** |
| Ionic y-shift | `-ln(SAHA_CONST*T_eV^1.5/n_e)` | derived above | **0 (exact)** |
| U_II/U_I in shift | absent | absent (cancels) | **0 (correct)** |
| SAHA_CONST_CM3 | 6.042e21 | 6.037e21 (CODATA) | **0.08%** |
| KB_EV | 8.617333262e-5 | 8.617333262e-5 | **0** |
| Closure: C_s ∝ U_I·exp(q_s)·(1+S_s) | ✓ | Ciucci 1999 | **0** |

The 0.08% error in `SAHA_CONST_CM3` introduces an `~0.05%` systematic shift in the Saha ratio, which propagates as a second-order correction on the Boltzmann slope and is negligible for any practical LIBS measurement (typical temperature uncertainties are 2–10%).

The comment in `constants.py` documenting `SAHA_CONST_CM3` contains a typographic error: it writes `h^2` in the denominator when the correct formula has `h^3`. However, the actual numerical constant 6.042e21 was clearly derived from the correct formula (the `h^2` expression would yield a totally different number); this is a documentation typo, not a physics bug.

---

## FIX

No physics fix needed. The code is correct.

**Optional cosmetic fix** (comment typo in `cflibs/core/constants.py`):

```python
# WRONG (current comment):
# (2π × 9.109e-31 × 1.381e-23 / (6.626e-34)^2)^{3/2} × 2 × 1e-6 ≈ 6.04e21

# CORRECT comment should read:
# Derivation: 2 × (2π × m_e × k_B)^{3/2} / h^3, then convert T from K to eV
# = 4.829e15 cm^-3 K^{-3/2} × EV_TO_K^{3/2} ≈ 6.037e21 cm^-3 eV^{-3/2}
```

The value itself could also be updated from 6.042e21 to 6.0371e21 (CODATA 2018 precision), but this is well within typical experimental uncertainty.

**What would have to be true for the shift to be wrong:** if U_II/U_I did NOT appear in the Saha equation (i.e., the Saha equation omitted partition functions). Since the Saha equation necessarily includes U_II/U_I, the cancellation is algebraically exact and the prior audit claim that this term was "MISSING" was incorrect. The 0.08% constant error in SAHA_CONST_CM3 is the only quantifiable deviation, and it is below the level of significance for any LIBS application.

---

*Audit performed: 2026-06-25*
*All derivations independently verified via sympy and numpy.*
*Code provenance: cflibs imported from `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/cflibs/__init__.py` (confirmed).*
