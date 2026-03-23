# Research State

## Project Reference

See: .gpd/PROJECT.md

**Core research question:** What is the optimal computational architecture for real-time CF-LIBS multi-element plasma diagnostics, and how do GPU-accelerated implementations compare to existing tools?
**Current focus:** Physics formalization of 5 GPU optimization targets

## Current Position

**Current Phase:** 1
**Current Phase Name:** Physics Formalization
**Total Phases:** 7
**Current Plan:** --
**Total Plans in Phase:** --
**Status:** Ready to plan
**Last Activity:** 2026-03-23

**Progress:** [#.........] 0%

## Active Calculations

None yet.

## Intermediate Results

None yet.

## Open Questions

1. Should the paper include a roofline model analysis?
2. What is the minimum batch size where GPU overtakes CPU?
3. Is mixed-precision (float16/bfloat16) worth exploring for Voigt profiles?

## Performance Metrics

| Label | Duration | Tasks | Files |
| ----- | -------- | ----- | ----- |
| -     | -        | -     | -     |

## Accumulated Context

### Decisions

| ID | Decision | Date |
|----|----------|------|
| DEC-01 | JAX as sole GPU framework | 2026-03-23 |
| DEC-02 | V100S as benchmark target | 2026-03-23 |
| DEC-03 | JQSRT as target venue | 2026-03-23 |
| DEC-04 | 5 optimization targets spanning full pipeline | 2026-03-23 |
| DEC-05 | Zaghloul 2024 as Voigt reference implementation | 2026-03-23 |

### Active Approximations

None yet.

**Convention Lock:**

*Custom conventions:*
- Unit System: SI with eV for temperature and energy, cm^-3 for number densities
- Temperature Unit: eV (electron-volt), T_e_eV; Kelvin used as T_K when explicitly noted
- Density Unit: cm^-3 for number densities (n_e, n_i), g/cm^3 for mass density
- Wavelength Unit: nm (nanometers) for spectral wavelengths
- Boltzmann Constant: KB_EV = 8.617333e-5 eV/K (cflibs.core.constants)
- Saha Equation: n_{i+1}*n_e/n_i = (2*U_{i+1}/U_i) * (2*pi*m_e*kT/h^2)^{3/2} * exp(-chi/kT)
- Boltzmann Distribution: n_k/n_total = g_k * exp(-E_k / kT) / U(T) where n_total is total population of the SAME species and ionization stage, NOT total plasma density. E_k measured from ground state of that stage.
- Emissivity Formula: Line-integrated volumetric emissivity: epsilon = hc/(4*pi*lambda) * A_ki * n_k [W/cm^3/sr]. For spectral emissivity, multiply by normalized line profile phi(lambda). The 4*pi is for isotropic emission per steradian.
- Line Profile: Voigt = Gaussian(Doppler) convolved Lorentzian(Stark+natural); sigma for Gaussian, gamma (HWHM) for Lorentzian
- Ionization Potential Depression: Debye-Huckel: delta_chi = e^2/lambda_D where lambda_D = sqrt(kT/(4*pi*n_e*e^2)) in Gaussian CGS. Equivalently delta_chi = e^2 * sqrt(4*pi*n_e*e^2/kT). Implementation: saha_boltzmann.py:440-443
- Partition Function: U(T) = sum_k g_k * exp(-E_k/kT), truncated at the LOWERED ionization potential (IP - delta_chi_DH) when IPD is applied. Implementation: saha_boltzmann.py:294-295
- Closure Equation: CF-LIBS closure uses NUMBER/MOLE fractions internally (sum C_s = 1). Mass fractions are computed for output via C_mass = C_mole * AW / sum(C_mole * AW). See closure.py for standard/matrix/oxide modes.
- Energy Reference: All energies (E_k, E_i) are positive above ground state of each ionization stage. E_k >= E_i >= 0 always.
- Spectral Domain: Wavelength-domain in nm throughout. Line profiles phi_lambda are normalized over wavelength (integral phi_lambda dlambda = 1). NOT frequency-domain.
- Stark Width: Transition.stark_w is HWHM at reference n_e = 1e16 cm^-3. stark.py scales linearly: gamma_stark = stark_w * (n_e / 1e16). Full-width = 2 * gamma.
- Self Absorption Model: Two models: (1) escape-factor with empirical tau ~ 1e-25 * A_ki * lambda_cm^3 * n_i * L (NOT using f_ik), (2) curve-of-growth in CDSBPlotter. Note: oscillator_strength property has a latent 1e16 prefactor error (Angstrom constant with cm lambda) but it only affects log_gf deltas where the error cancels.
- Saha Form: 3-stage closed-form: explicit S1=n_II*n_e/n_I, optional S2=n_III*n_e/n_II, then n_total=n_I+n_II+n_III. NOT a general iterative all-stage solver. Inversion uses only S1 (2-stage).
- Emissivity Units: SI output: W/m^3/sr (line-integrated). NOT W/cm^3/sr. Populations converted cm^-3 -> m^-3 at emissivity.py:44. Docstring at emissivity.py:33 claims W/m^3/nm which is WRONG for line-integrated form.
- Partition Function Warning: DESIGN CHOICE: polynomial PF path uses pre-tabulated coefficients with natural IP truncation built in. IPD correction is NOT applied to polynomial path (would require non-trivial correction factor). Level-summation fallback correctly truncates at lowered IP. This is an acceptable approximation when delta_chi << IP.

### Propagated Uncertainties

None yet.

### Pending Todos

None yet.

### Blockers/Concerns

None

## Session Continuity

**Last session:** 2026-03-23
**Stopped at:** Project initialization complete. Ready to begin Phase 1 planning.
**Resume file:** --
