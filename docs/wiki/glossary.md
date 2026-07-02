---
slug: glossary
title: "Glossary of Terms & Symbols"
chapter: glossary
order: 910
status: stable
register: reference
summary: >
  The plain-language index of CF-LIBS domain terms plus the canonical physics-symbol table.
  Symbols are DEFINED once in formal-spec.md (pinned to cflibs-formal Lean ids); this page is the
  quick lookup and never redefines a canonical symbol. Air-vacuum and Boltzmann-ordinate
  conventions are stated where they bite.
tags: [glossary, terms, symbols, notation, air-vacuum, boltzmann, saha]
updated: 2026-07-02
related: [formal-spec, libs-physics, cf-libs-family, orientation, bibliography]
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Glossary of Terms & Symbols

This is the plain-language index of CF-LIBS terms and the quick-lookup symbol table. **Symbols are
defined canonically in [formal-spec.md](formal-spec.md)** — the notation authority — where each is
pinned to a `cflibs-formal` Lean identifier. This page mirrors those definitions for fast reference
and never introduces a new canonical symbol. For citations see [bibliography.md](bibliography.md).

> [!NOTE] Two load-bearing conventions. **Wavelengths** in the atomic database are stored in **air**
> (NIST/ASD), converted through a single utility in `cflibs/core/`; any page comparing wavelengths
> must state which frame it uses. The canonical **Boltzmann ordinate** is
> $y=\ln\!\big(I_{ki}\lambda/(g_kA_{ki})\big)$ vs $x=E_k$, and $\lambda$ is load-bearing whenever it
> varies across the lines in a single fit.

## Symbols {#symbols}

| Symbol | Name | Meaning / units | Authority |
|--------|------|-----------------|-----------|
| $T$ | temperature | excitation/plasma temperature (K, or eV where noted); Boltzmann-plot slope $=-1/(k_BT)$ | [formal-spec.md](formal-spec.md) |
| $n_e$ | electron density | free-electron number density (cm⁻³); a Saha input, ideally measured by Stark broadening | [formal-spec.md](formal-spec.md) |
| $C_s$ | mass/number fraction | fractional abundance of species $s$; closure imposes $\sum_s C_s = 1$ | [cf-libs-family.md](cf-libs-family.md) |
| $I_{ki}$ | line intensity | integrated intensity of the $k\!\to\!i$ transition | [libs-physics.md](libs-physics.md) |
| $A_{ki}$ | Einstein coefficient | spontaneous-emission transition probability (s⁻¹) | [libs-physics.md](libs-physics.md) |
| $g_k$ | statistical weight | upper-level degeneracy, $2J+1$ | [libs-physics.md](libs-physics.md) |
| $E_k$ | upper-level energy | eV (stored as cm⁻¹ in the DB); Boltzmann-plot abscissa | [libs-physics.md](libs-physics.md) |
| $g\!\cdot\!A$ | line strength product | carries a line's intrinsic strength; source-correlated $g\!\cdot\!A$ bias is the dominant real-data error | [error-budget-and-falsification.md](error-budget-and-falsification.md) |
| $U_s(T)$ | partition function | $\sum g\exp(-E/k_BT)$ over a species' levels; normalizes Boltzmann populations | [atomic-data-and-datasets.md](atomic-data-and-datasets.md) |
| $\lambda$ | wavelength | air unless stated (NIST/ASD); load-bearing in the Boltzmann ordinate | [libs-physics.md](libs-physics.md) |
| $\tau$ | optical depth | line opacity; escape factor $SA(\tau)=(1-e^{-\tau})/\tau$ | [libs-physics.md](libs-physics.md) |
| $F_{\text{cal}}$ | calibration factor | scalar experimental/collection factor; cancels via closure and in ratios | [cf-libs-family.md](cf-libs-family.md) |
| $k_B$ | Boltzmann constant | SI plasma-unit constant | [core constants](architecture/target-architecture.md) |

## Terms {#terms}

| Term | Meaning |
|------|---------|
| **LIBS** | Laser-Induced Breakdown Spectroscopy: a laser pulse ablates a sample into a microplasma whose optical emission is spectrally resolved. |
| **CF-LIBS** | Calibration-Free LIBS: standardless quantification from the plasma's own physics (Saha-Boltzmann), no matrix-matched standards [@ciucci1999]. |
| **LTE** | Local Thermodynamic Equilibrium: collisions dominate radiation so a single $T$ fixes both ionization (Saha) and excitation (Boltzmann). |
| **pLTE** | partial LTE: upper levels equilibrated but ground/near-ground not — an LTE-validity failure mode [@cristoforetti2010]. |
| **McWhirter criterion** | The classic lower bound on $n_e$ for LTE to hold; necessary but not sufficient [@cristoforetti2010]. |
| **Saha equation** | Fixes the population ratio between adjacent ionization stages as a function of $T$, $n_e$, and partition functions. |
| **Boltzmann distribution** | Fixes level populations within a species: $n_k \propto g_k \exp(-E_k/k_B T)$. |
| **Boltzmann plot** | $y=\ln(I_{ki}\lambda/(g_k A_{ki}))$ vs $x=E_k$; slope $=-1/(k_B T)$ gives temperature. |
| **Saha-Boltzmann plot** | Multi-stage Boltzmann plot with a Saha correction mapping ion lines onto the neutral plane [@aguilera2007]. |
| **Partition function** $U_s(T)$ | Sum of $g\exp(-E/k_B T)$ over a species' levels; normalizes Boltzmann populations. |
| **IP / ionization potential** | Energy to remove an electron from a species; the Saha ladder rung. |
| **IPD** | Ionization-Potential Depression: plasma-density lowering of the effective IP; the PF cutoff must share the Saha IPD. |
| **Stage I / II / III** | Neutral / singly-ionized / doubly-ionized species (e.g. Fe I, Fe II, Fe III). |
| **Closure** | The constraint $\sum_s C_s = 1$ used to turn relative species densities into absolute fractions — a projection, not the estimator. |
| **Log-ratio / Aitchison ratio** | $\ln(N_i/N_j)$; matrix- and detected-set-invariant, the preferred DED deliverable [@aitchison1982]. |
| **ILR** | Isometric log-ratio transform: an orthonormal coordinate system for compositional data [@egozcue2003]. |
| **OPC** | One-Point Calibration: a single reference point removes the CF-LIBS scale bias [@cavalcanti2013]. |
| **C-sigma / CD-SB** | C-sigma graphs [@aragon2014] and Columnar-Density Saha-Boltzmann [@cdsb2013]: self-absorption-tolerant curve-of-growth CF variants. |
| **Self-absorption** | Re-absorption of line emission along the line of sight; saturates strong lines ($I \propto 1-e^{-\tau}$). |
| **Optical depth $\tau$** | Line opacity; escape factor $SA(\tau)=(1-e^{-\tau})/\tau$. |
| **Optically thin / thick** | Regimes where emission is (thin) linear in column density or (thick) saturated. |
| **Curve of growth (COG)** | Line intensity vs column density; the thick branch grows as $\sqrt{\tau}$. |
| **Stark broadening** | Pressure broadening by charged particles; the preferred (physical) $n_e$ diagnostic. |
| **Forward model** | Parameters → synthetic spectrum ($T,n_e,\text{composition} \to$ spectrum). |
| **Inversion** | Spectrum → parameters ($T,n_e,\text{composition}$ from a measured spectrum). |
| **Line identification** | Assigning detected peaks to species/transitions (ALIAS, comb, correlation, NNLS). |
| **ALIAS** | An identification algorithm scoring matched vs expected emissivity; mis-tuned for the dense post-reset catalog [@noel2025]. |
| **Comb matching** | High-recall identification by matching a species' line "comb" against the spectrum [@gajarska2024]. |
| **Manifold** | A precomputed grid of synthetic spectra (HDF5/Zarr) enabling fast nearest-neighbor inference — a fast-inference option, not the pipeline spine. |
| **DED** | Directed-Energy Deposition additive manufacturing; the drift-tracking deployment target (Goal a). |
| **Ti-6Al-4V** | The canonical DED titanium alloy ({Ti, Al, V}); the constrained-element tracking target. |
| **Refuse-to-report** | The reliability gate that withholds a composition when the fit is untrustworthy (the adoption gate). |
| **cflibs-formal** | The separate Lean 4 project proving the CF-LIBS physics; the notation authority. |
| **ASD59 / reset line** | NIST Atomic Spectra Database v5.9 [@kramida2024nist]; the rebuilt-DB baseline all numbers are measured against. |
| **Air vs vacuum wavelength** | The DB stores **air** wavelengths (NIST/ASD); any wavelength-handling page must state which it uses. |

## See also {#see-also}

- [formal-spec.md](formal-spec.md) — the authoritative symbol definitions and Lean pins.
- [libs-physics.md](libs-physics.md) — where the physics symbols are used in anger.
- [bibliography.md](bibliography.md) — every `@key` cited above.
- [orientation.md](orientation.md) — goals, reset line, error-budget thesis, authoring contract.
