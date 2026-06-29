# Stark Broadening Audit: cflibs/radiation/stark.py + cflibs/inversion/physics/stark_ne.py

## VERDICT

**flawed** — severity: **critical**

The non-hydrogenic electron-impact width law, Olivero–Longbothum deconvolution, and n_e inversion are mathematically correct and consistent with Griem 1974. However, the H-alpha reference width stored in the database is the **electron-impact ONLY** component (49 pm from Konjevic 2002 Tab 1), while the physically observable Stark FWHM for H-alpha is ~1300 pm at the same conditions (Gigosos 2003). Since H-alpha is simultaneously stored as `stark_b` (literature-grade) provenance AND listed as the top-priority diagnostic line in `PREFERRED_DIAGNOSTIC_LINES`, when H-alpha appears in a spectrum it will be preferentially selected for the n_e inversion, producing a **12–60× overestimate of n_e** depending on the true density. The code acknowledges the H Balmer n_e^0.7 issue as a "known limitation" but does not guard against selecting H-alpha with the wrong w_ref for inversion.

---

## GROUND TRUTH

### Non-hydrogenic lines: Griem semiclassical formula

**Griem, H.R. (1974). *Spectral Line Broadening by Plasmas.* Academic Press, New York. (ISBN 0-12-302850-7)**

Stark FWHM:
```
Δλ_S = 2·w(T)·(n_e/n_ref) + 3.5·A(T)·(n_e/n_ref)^(1/4)·[1 − 3/(4·N_D^(1/3))]·w(T)
```
where w(T) is the electron-impact HALF-half-width at n_ref = 1×10¹⁷ cm⁻³ (singly-ionized lines). Stored FWHM convention in the code = 2w, so the linear factor of 2 is absorbed.

**Olivero, J.J. & Longbothum, R.L. (1977).** *J. Quant. Spectrosc. Radiat. Transfer*, **17**, 233–236. DOI: 10.1016/0022-4073(77)90161-3
```
f_V = 0.5346·f_L + √(0.2166·f_L² + f_G²)
```
Error < 0.02%.

### H-alpha reference: Gigosos et al. (full ion+electron simulation)

**Gigosos, M.A., González, M.Á., & Cardeñoso, V. (2003).** Computer-simulated Balmer-alpha, -beta and -gamma Stark line profiles for non-equilibrium plasma diagnostics. *Spectrochimica Acta Part B*, **58**, 1489–1504. DOI: 10.1016/S0584-8547(03)00097-1

Power-law fit at T ~ 10,000 K:
```
Hα FWHM [nm] ≈ 1.3 · (n_e / 10¹⁷ cm⁻³)^0.64
```
At n_e = 1×10¹⁷ cm⁻³, T = 10,000 K: **FWHM = 1.30 nm**

**Konjevic, N. et al. (2002).** *J. Phys. Chem. Ref. Data*, **31**, 819–927. DOI: 10.1063/1.1486456
Table 1 lists H I 656.279 nm electron-impact width: **W_e = 0.490 Å = 49 pm = 0.049 nm** at n_e = 1×10¹⁷ cm⁻³, T = 10,000 K. This is the **electron-impact contribution alone**; the total Stark width (including dominant ion broadening) is ~1.3 nm.

---

## CODE VALUE (numerical)

```bash
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && \
PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python3 -c "
import cflibs; print('cflibs:', cflibs.__file__)
from cflibs.radiation.stark import stark_width, deconvolve_stark_fwhm, estimate_ne_from_stark, REF_NE

# Test 1: Non-hydrogenic law at reference conditions
w = stark_width(1e17, 10000.0, 0.045, 0.5)
print('stark_width at REF conditions (expect 0.045):', w)

# Test 2: Olivero-Longbothum round-trip
import numpy as np
f_L, f_G = 0.1, 0.05
f_V = 0.5346 * f_L + np.sqrt(0.2166 * f_L**2 + f_G**2)
f_L_rec = deconvolve_stark_fwhm(f_V, f_G, 0.0)
print('OL round-trip error:', abs(f_L_rec - f_L)/f_L*100, '%')

# Test 3: H-alpha inversion with stored 49 pm vs Gigosos 1300 pm
w_stored = 0.049  # nm (stored in DB)
for ne_true in [1e16, 1e17, 1e18]:
    w_meas = 1.3 * (ne_true/1e17)**0.64  # Gigosos measured width
    ne_inv = estimate_ne_from_stark(w_meas, 10000.0, w_stored, 0.5)
    print(f'n_e_true={ne_true:.0e}: w_meas={w_meas:.3f}nm, n_e_inverted={ne_inv:.2e}, factor={ne_inv/ne_true:.0f}x')
"
```

**Output:**
```
cflibs: /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/cflibs/__init__.py
stark_width at REF conditions (expect 0.045): 0.045
OL round-trip error: 0.0000 %
n_e_true=1e+16: w_meas=0.298nm, n_e_inverted=6.08e+17, factor=61x
n_e_true=1e+17: w_meas=1.300nm, n_e_inverted=2.65e+18, factor=27x
n_e_true=1e+18: w_meas=5.675nm, n_e_inverted=1.16e+19, factor=12x
```

**Stored H-alpha DB value (from `scripts/archive/migrations/ingest_stark_b.py` line 144):**
```python
("H", 1, 656.279, 49.0, 0.20, 0.0, "Konjevic 2002 H-alpha; Griem 1974 Tab 4-5"),
```
= 49 pm = 0.049 nm (electron-impact FWHM only).

**Literature (Gigosos 2003) at n_e = 1×10¹⁷ cm⁻³, T = 10,000 K:** total FWHM = 1.30 nm.

---

## DELTA & INTERPRETATION

### What is correct

| Component | Code value | Literature value | Status |
|-----------|-----------|-----------------|--------|
| Linear n_e scaling | `w_ref*(n_e/REF_NE)` | Griem 1974: correct for non-H | ✓ CORRECT |
| T-scaling exponent | `(T/T_ref)^(-0.5)` default | T^(-0.25) to T^(-0.5) per line | ✓ ACCEPTABLE |
| OL constants | a=0.5346, b=0.2166 | Olivero & Longbothum 1977 exact values | ✓ CORRECT |
| n_e inversion formula | `n_e = REF_NE*(w/w_ref)*(T/T_ref)^alpha` | Griem: correct inversion | ✓ CORRECT |
| Ion broadening factor 1.75 | Correct (Griem 3.5/2) | ✓ CORRECT |
| Ion R_D hardcoded 0.5 | Varies 0.22–0.65 by conditions | MINOR ERROR ~1-2% total width |

### Critical flaw: H-alpha w_ref is 27× too small

The value stored for H I 656.28 nm (`stark_w` = 49 pm) is **exclusively the electron-impact contribution** extracted from Konjevic 2002, Table 1. For hydrogen Balmer lines, ion quasi-static broadening is **comparable to or larger than** the electron-impact term — the Gigosos 2003 simulations (the modern standard) give FWHM = 1300 pm at the same conditions, i.e. **ions contribute ~96% of the total width**.

Because `H I 656.28 nm` is:
1. stored with `stark_w_source = 'stark_b'` (= literature-grade), so it passes the `LITERATURE_STARK_SOURCES` gate in `measure_stark_ne()`, and  
2. listed as `PREFERRED_DIAGNOSTIC_LINES[0]` with a 2× ranking bonus,

any spectrum containing H-alpha will preferentially select it for the n_e diagnostic and invert with the wrong 49 pm reference instead of the correct ~1300 pm total, overestimating n_e by **12–60×** across the typical LIBS density range 10¹⁶–10¹⁸ cm⁻³.

The code comment in `stark_ne.py` line 46–50 acknowledges the H Balmer n_e^0.7 issue as a "known limitation / left for follow-up", but this misidentifies the problem. The exponent correction is secondary; the primary bug is that the stored reference width is ~27× too small because it omits ion broadening. A correct Gigosos-based n_e^0.64 law anchored at w_ref = 1.3 nm would be self-consistent; using the electron-impact 49 pm width with any exponent gives catastrophically wrong n_e.

For **non-hydrogenic lines** (Ca II, Mg II, Fe II, etc.), the code is internally self-consistent and numerically correct: the stored FWHM is the total Stark width (electron impact dominates for non-H), the inversion formula correctly recovers n_e, and round-trip tests pass to floating-point precision.

---

## FIX

**File:** `scripts/archive/migrations/ingest_stark_b.py`, line 144  
**File:** `cflibs/inversion/physics/stark_ne.py`, lines 46–50 and 82–88

**Option A (minimum guard — recommended now):** Remove H-alpha from `PREFERRED_DIAGNOSTIC_LINES` and exclude `H` from the diagnostic path until a dedicated Gigosos lookup is implemented. This prevents catastrophic misuse while keeping non-H diagnostics unaffected:

```python
# In cflibs/inversion/physics/stark_ne.py, change:
PREFERRED_DIAGNOSTIC_LINES: Tuple[Tuple[str, int, float], ...] = (
    ("H", 1, 656.28),    # <-- REMOVE until Gigosos-based path exists
    ("Ca", 2, 393.37),
    ...
)

# And in measure_stark_ne(), add element exclusion:
if obs.element == "H":
    _reject("hydrogen_requires_gigosos_tables")
    continue
```

**Option B (correct fix):** Replace the stored H-alpha `stark_w` with the Gigosos-anchored value (1.3 nm at 1×10¹⁷ cm⁻³, T = 10,000 K) and update the inversion to use the Gigosos power-law exponent (0.64) stored as `stark_alpha` for H I lines. This requires a new `stark_w_source = 'gigosos'` provenance class and corresponding handling in `estimate_ne_from_stark` to use the non-unity exponent for both forward and inverse.

**Option B DB entry:**
```python
("H", 1, 656.279, 1300.0, 0.64, 0.0, "Gigosos 2003 FWHM total at 1e17/10kK; DOI:10.1016/S0584-8547(03)00097-1"),
```

---

## CITATIONS

1. Griem, H.R. (1974). *Spectral Line Broadening by Plasmas.* Academic Press.
2. Gigosos, M.A., González, M.Á., & Cardeñoso, V. (2003). *Spectrochim. Acta B*, 58, 1489–1504. DOI:10.1016/S0584-8547(03)00097-1
3. Konjevic, N., Lesage, A., Fuhr, J.R., & Wiese, W.L. (2002). *J. Phys. Chem. Ref. Data*, 31(3), 819–927. DOI:10.1063/1.1486456
4. Olivero, J.J. & Longbothum, R.L. (1977). *J. Quant. Spectrosc. Radiat. Transfer*, 17, 233–236. DOI:10.1016/0022-4073(77)90161-3
5. Gigosos, M.A. (2014). Stark broadening models for plasma diagnostics. *J. Phys. D: Appl. Phys.*, 47(34), 343001. DOI:10.1088/0022-3727/47/34/343001
6. Vidal, C.R., Cooper, J., & Smith, E.W. (1973). Hydrogen Stark-broadening tables. *Astrophys. J. Suppl.*, 25, 37–136.
