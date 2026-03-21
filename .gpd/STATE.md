# Research State

## Project Reference

See: .gpd/PROJECT.md

**Core research question:** [Not set]
**Current focus:** [Not set]

## Current Position

**Current Phase:** —
**Current Phase Name:** —
**Total Phases:** —
**Current Plan:** —
**Total Plans in Phase:** —
**Status:** —
**Last Activity:** —

**Progress:** [░░░░░░░░░░] 0%

## Active Calculations

None yet.

## Intermediate Results

None yet.

## Open Questions

None yet.

## Performance Metrics

| Label | Duration | Tasks | Files |
| ----- | -------- | ----- | ----- |
| -     | -        | -     | -     |

## Accumulated Context

### Decisions

None yet.

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

### Propagated Uncertainties

None yet.

### Pending Todos

None yet.

### Blockers/Concerns

None

## Session Continuity

**Last session:** —
**Stopped at:** —
**Resume file:** —
