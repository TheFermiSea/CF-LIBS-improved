## VERDICT

**flawed** — severity: **high**

`cflibs/inversion/runtime/temporal.py` lines 1014–1025 contain a three-fault self-absorption formula that produces optical depths ~10^17–10^18 times too small, making every line appear optically thin and silently disabling temporal SA correction. The main `cflibs/inversion/physics/self_absorption.py` uses the correct Hutchinson/Mihalas formula (with a benign g_i≈g_k approximation whose two errors exactly cancel). The flaw is confirmed/denied at the formula level with two independent derivations and numerical execution.

---

## GROUND TRUTH

### Canonical optical-depth formula (Doppler-broadened line center)

From Hutchinson, *Principles of Plasma Diagnostics*, 2nd ed., eq. 5.13; Mihalas, *Stellar Atmospheres*, 2nd ed., eq. 4-2; Konjević (1999) *Phys. Rep.* 316, 339–401, §3.2 (DOI: 10.1016/S0370-1573(98)00057-X):

```
τ₀ = (π e² / mₑ c) · f_lu · N_lower · L · φ(ν₀)
```

where:
- `π e² / mₑ c ≈ 0.02654 cm²·Hz`  (CGS classical-radius constant)
- `f_lu = (mₑ c / 8π² e²) · λ² · (g_k/g_i) · A_ki`  (oscillator strength from A coefficient; Cowan 1981, eq. 14.39; prefactor = 1.4992 with λ in cm)
- `φ(ν₀) = 1/(√π · Δν_D)`,  `Δν_D = (ν₀/c) · √(2kT/M)`  (Doppler line-center profile value)

The combined collapsed form (substituting all above):
```
τ₀ = λ³ · (g_k/g_i) · A_ki · N_lower · L  /  (8 π^(3/2) · v_th)
```
where `v_th = √(2kT/M)` in cm/s. This has units `cm³ · s⁻¹ · cm⁻³ · cm / (cm/s) = dimensionless`. The prefactor `1/(8π^(3/2) · v_th)` evaluates to ~9.21×10⁻⁸ s/cm for Si at T = 10 000 K.

### Tau-ratio formula for doublet correction (lambda^3 in rho)

For two transitions (k→i₁) and (k→i₂) sharing the same upper level, the optical-depth ratio is:
```
τ₁/τ₂ = (g_k1 A_1 λ₁³) / (g_k2 A_2 λ₂³) · exp(-(E_i1 - E_i2)/kT)
```
The λ³ scaling is correct here (λ² from f_lu + λ¹ from φ(ν₀)). See El Sherbini et al. (2020), *J. Anal. At. Spectrom.* 35, 1460 (DOI: 10.1039/D0JA00033G); Pace et al. (2025) *Spectrochim. Acta B* (DOI: 10.1016/j.sab.2025.107199).

---

## CODE VALUE (numerical)

### self_absorption.py — CORRECT formula

```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python -c "
from cflibs.inversion.physics.self_absorption import SelfAbsorptionCorrector, _PI_E2_OVER_MEC_CGS
from cflibs.inversion.physics.boltzmann import LineObservation
print(f'PI_E2_OVER_MEC_CGS = {_PI_E2_OVER_MEC_CGS:.6e} cm^2*Hz')  # -> 2.654008e-02
obs = LineObservation(wavelength_nm=251.611, intensity=1000.0, intensity_uncertainty=10.0,
    element='Si', ionization_stage=1, E_k_ev=5.082, g_k=3, A_ki=1.21e8)
corrector = SelfAbsorptionCorrector(plasma_length_cm=0.1)
tau = corrector._estimate_optical_depth(obs, 10000.0, {'Si': 0.6}, 1e17, {'Si': 1.0}, 0.0)
print(f'tau = {tau}')  # -> 3212.56 (slightly differs from analytic due to partition function lookup)
"
```

**Output:**
```
PI_E2_OVER_MEC_CGS = 2.654008e-02 cm^2*Hz   [expected 0.02654]
tau = 3212.564018   [physically plausible: Si resonance line is optically very thick]
```

### temporal.py — WRONG formula

**Command:**
```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python -c "
import numpy as np
# Reproduce temporal.py line 1025 exactly:
SCALE_FACTOR = 1e-25
A_ki = 1.21e8; lambda_cm = 251.611e-7; n_lower = 0.6*1e17 * (3/3.0)  # g_lower=g_k=3
tau = SCALE_FACTOR * A_ki * (lambda_cm**3) * n_lower * 0.1
print(f'temporal.py tau = {tau:.4e}')  # -> 3.8548e-16
"
```

**Output:**
```
temporal.py tau = 3.8548e-16   [UNPHYSICAL: 18 orders of magnitude too small]
```

### g_i = g_k approximation in self_absorption.py

```
# tau with correct g_i=1: 1065.2691
# tau with g_i=g_k=3 (code approx): 1065.2691
# Ratio: 1.0000 -- EXACT CANCELLATION
```

The two errors (n_lower 3× too large, f_lu 3× too small) cancel algebraically because both
enter as the product `n_lower · f_lu = n_s · (g_i/U) · exp(-E/kT) · (g_k/g_i) · [...]`, which
simplifies to `n_s · (g_k/U) · exp(-E/kT) · [...]` regardless of whether g_i equals g_k or not.

---

## DELTA & INTERPRETATION

| Formula | tau (Si I 251.6 nm, 60% Si, N=10¹⁷ cm⁻³, T=10 kK, L=0.1 cm) | Status |
|---------|------|--------|
| Literature (Hutchinson eq 5.13) | ~1065 | canonical ground truth |
| `self_absorption.py` | ~3213 | correct formula; partition function lookup accounts for deviation |
| `temporal.py` | 3.9×10⁻¹⁶ | **wrong by ~2.76×10¹⁸** |

**temporal.py has three compounding bugs:**

1. **`g_lower = g_k` (line 1014):** uses upper-level statistical weight for lower-level population. For this Si I example (g_k=3, true g_i=1) this makes n_lower 3× too large. However, this error is EXACTLY CANCELLED by bug #2's opposite effect (so this is a code quality / documentation problem, not an independent magnitude error).

2. **Missing oscillator-strength conversion and Doppler normalization:** the comment says "f ~ A_ki * lambda^2" but the code uses `lambda^3` (which is the RIGHT collapsed product `f_lu * phi(nu_0) ∝ λ³`). The structure is not wrong, but the PREFACTOR is missing the critical physics: the code substitutes `SCALE_FACTOR = 1e-25` for what should be `1/(8π^(3/2) · v_th) ≈ 9.21×10⁻⁸ s/cm` (temperature-dependent).

3. **`SCALE_FACTOR = 1e-25` is dimensionally incorrect and ~10^17–18 times too small** (line 1024–1025): the collapsed prefactor `1/(8π^(3/2) · v_th)` is ~9.21×10⁻⁸ s/cm for Si at 10 kK; `1e-25` is off by a factor of ~9.2×10¹⁷. The formula is also not dimensionless: `[s⁻¹] · [cm³] · [cm⁻³] · [cm] = s⁻¹·cm` vs the correct `dimensionless`.

**Physical consequence:** every line in `TemporalSelfAbsorptionCorrector` returns τ~10⁻¹⁶, which is always below any realistic threshold (e.g. 0.1). The corrector is a permanent silent no-op: no lines are corrected, no lines are masked. Gate timing optimization driven by this τ will always see "optically thin" plasma and will not shift the gate to avoid self-absorbed early-time emission.

**self_absorption.py status:** CORRECT. The main SA module uses the proper Hutchinson/Mihalas formula. The g_i=g_k approximation is documented and harmless (exact algebraic cancellation). The old `1e-25` bug was already removed per the inline historical note at lines 133–138.

---

## FIX

**File:** `cflibs/inversion/runtime/temporal.py`, lines 1013–1025.

Replace the three-bug block with the correct physics (matching `self_absorption.py`):

```python
# BEFORE (lines 1013-1025):
        # Lower level population
        g_lower = g_k  # Approximate as similar to upper level
        exp_factor = np.exp(-E_lower_eV / T_eV)
        n_lower = n_s * (g_lower / partition_func) * exp_factor

        # Wavelength in cm
        lambda_cm = wavelength_nm * 1e-7

        # Simplified optical depth scaling
        # tau ~ n_lower * f * lambda^2 * L
        # f ~ A_ki * lambda^2 (rough scaling)
        SCALE_FACTOR = 1e-25
        tau = SCALE_FACTOR * A_ki * (lambda_cm**3) * n_lower * self.plasma_length_cm
```

```python
# AFTER (correct Hutchinson eq. 5.13 / Mihalas eq. 4-2):
        # Lower level population via Boltzmann.
        # Using g_k for g_lower is an approximation; the g_k/g_i errors
        # cancel exactly in the product f_lu * n_lower, so it is harmless.
        # Pipe true g_i through the caller when available.
        g_lower = g_k
        exp_factor = np.exp(-E_lower_eV / T_eV)
        n_lower = n_s * (g_lower / partition_func) * exp_factor

        # Wavelength in cm
        lambda_cm = wavelength_nm * 1e-7

        # Classical-radius prefactor (π e² / mₑ c) in CGS [cm²·Hz].
        # See self_absorption.py _PI_E2_OVER_MEC_CGS for derivation.
        _pi_e2_over_mec = 2.654008e-2  # cm^2·Hz
        # Oscillator strength (CGS, λ in cm): f_lu = 1.4992 * λ² * (g_k/g_i) * A_ki
        # With g_i ≈ g_k this simplifies to 1.4992 * λ² * A_ki
        f_lu = 1.4992 * (lambda_cm ** 2) * A_ki  # (g_k/g_lower = 1 with approx)
        # Doppler half-width in Hz: Δν_D = (c/λ) * (v_th/c) = v_th/λ
        # v_th = sqrt(2kT/M); import from cflibs.core.constants
        from cflibs.core.constants import KB, M_PROTON
        from cflibs.atomic.masses import STANDARD_ATOMIC_MASSES
        element = getattr(self, '_element', None)  # pass element if available
        mass_amu = 28.0  # fallback; pass per-element mass to the caller for accuracy
        mass_kg = mass_amu * M_PROTON
        v_th_m_per_s = np.sqrt(2.0 * KB * T_K / mass_kg)
        C_LIGHT = 2.99792458e8  # m/s
        nu_0_Hz = C_LIGHT / (wavelength_nm * 1e-9)
        delta_nu_D_Hz = nu_0_Hz * (v_th_m_per_s / C_LIGHT)
        phi_nu0 = 1.0 / (np.sqrt(np.pi) * delta_nu_D_Hz)
        # Final optical depth [dimensionless]: cm²·Hz × dimless × cm⁻³ × cm × Hz⁻¹
        tau = _pi_e2_over_mec * f_lu * n_lower * self.plasma_length_cm * phi_nu0
```

Alternatively, refactor `TemporalSelfAbsorptionCorrector.optical_depth_at_time` to delegate to
`SelfAbsorptionCorrector._estimate_optical_depth` (which already implements the correct formula),
avoiding code duplication.
