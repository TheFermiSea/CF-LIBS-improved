---
slug: libs-physics
title: "Physics of Laser-Induced Plasmas and Their Emission"
chapter: libs-physics
order: 0
status: stable
register: review
summary: >
  The physics conscience of the wiki: the full LTE laser-plasma emission chain
  from ablation through Saha-Boltzmann populations, line emissivity, broadening,
  self-absorption, continuum, and temporal/spatial averaging, to the LTE-validity
  gate. Every quantitative claim is DOI-cited; the canonical Boltzmann ordinate is
  ln(I*lambda/(g*A)) vs E_k and lambda is load-bearing whenever it varies across a fit.
tags: [saha, boltzmann, ionization, lte, electron-density, forward-model, self-absorption, stark, broadening, mcwhirter, air-vacuum]
updated: 2026-07-02
benchmarks_pre_reset: false
sources:
  - "@aragon2008"
  - "@tognoni2010"
  - "@ciucci1999"
  - "@cristoforetti2010"
  - "@harilal2022"
  - docs/physics/Equations.md
  - docs/physics/Assumptions_And_Validity.md
  - docs/v4/overhaul/literature/saha-boltzmann-lte.md
  - docs/v4/overhaul/literature/broadening-rt.md
  - docs/v4/overhaul/literature/self-absorption-cog.md
  - docs/M-spec-boltzmann-convention-literature-verdict.md
  - cflibs/plasma/saha_boltzmann.py
  - cflibs/plasma/partition.py
  - cflibs/plasma/lte_validator.py
  - cflibs/radiation/emissivity.py
  - cflibs/radiation/profiles.py
  - cflibs/radiation/stark.py
code_refs:
  - cflibs/plasma/saha_boltzmann.py::SahaBoltzmannSolver
  - cflibs/plasma/partition.py
  - cflibs/plasma/lte_validator.py::LTEValidator
  - cflibs/plasma/state.py::SingleZoneLTEPlasma
  - cflibs/radiation/emissivity.py::calculate_line_emissivity
  - cflibs/radiation/profiles.py
  - cflibs/radiation/stark.py
  - cflibs/inversion/physics/self_absorption.py
  - cflibs/inversion/physics/self_absorption_observable.py
  - cflibs/inversion/physics/stark_ne.py
  - cflibs/atomic/wavelength_conversion.py
lean_refs:
  - CflibsFormal/Boltzmann.lean#boltzmann_plot
  - CflibsFormal/Boltzmann.lean#temperature_from_two_levels
  - CflibsFormal/Saha.lean#saha_relation
  - CflibsFormal/ForwardMap.lean#boltzmann_plot_intensity
  - CflibsFormal/LineBroadening.lean#dopplerFWHM_strictMono_T
  - CflibsFormal/StarkBroadening.lean#starkFWHM_isLinear
  - CflibsFormal/VoigtWidth.lean#voigt_gaussian_limit
  - CflibsFormal/CurveOfGrowth.lean#cogRatio_strictAntiOn
  - CflibsFormal/SelfAbsorption.lean#slabIntensity_eq_thin_mul_SA
  - CflibsFormal/PartialLTE.lean#mcwhirter_iff_thermalizationLimit
  - CflibsFormal/TemporalEvolution.lean#temporal_composition_invariant
related: [formal-spec, classical-quantification, cf-libs-family, error-budget-and-falsification, atomic-data-and-datasets, impl-literature-methods]
supersedes:
  - docs/physics/Equations.md
  - docs/physics/Assumptions_And_Validity.md
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Physics of Laser-Induced Plasmas and Their Emission

This chapter is the physics reference for the whole wiki: it derives, cites, and
bounds the LTE laser-plasma emission chain that the CF-LIBS forward model
computes and the inversion inverts. It is deliberately equation-forward and
densely cited — where a downstream chapter needs a physical fact, it links here
rather than re-deriving it, and where the code or a Lean theorem realizes an
equation, this chapter names the path or theorem id.

> [!NOTE] NOTATION AUTHORITY
> Every symbol used below is defined once in [formal-spec](formal-spec.md)
> and pinned to its `cflibs-formal` Lean id. This chapter uses those symbols
> verbatim and never redefines a canonical one. States units on first use.

> [!IMPORTANT] WAVELENGTH CONVENTION (cross-cutting invariant)
> The shipped atomic database stores **air** wavelengths for $\lambda > 200$ nm,
> because it is sourced from NIST ASD, whose observed wavelengths above 200 nm are
> air values [@morton2000]. All air-vacuum conversion goes through a single utility,
> `cflibs/atomic/wavelength_conversion.py`. Any equation below that stores or
> compares a wavelength is in **air** unless flagged vacuum. The Doppler and Stark
> width formulas use the rest-frame wavelength; mixing air and vacuum introduces a
> systematic $\approx 114$ pm offset at 400 nm (283 ppm) that exceeds a typical
> 50-100 pm line-match window and silently breaks identification. See
> [§9 Air-vacuum invariant](#air-vacuum-invariant).

---

## 0. The plasma lifecycle in one page {#lifecycle}

A nanosecond LIBS pulse deposits $\sim 10^{9}$-$10^{11}$ W cm$^{-2}$ onto the
sample; the emission we quantify comes from a short, well-chosen slice of a
rapidly evolving plume. The canonical review of laser-produced-plasma
diagnostics is [@harilal2022]; the CF-LIBS-specific synthesis is [@tognoni2010]
and [@aragon2008].

| Phase | Approx. time | State | What emits | CF-LIBS relevance |
|-------|-------------|-------|-----------|-------------------|
| Ablation / breakdown | 0-10 ns | Dense, opaque | Continuum only | Not used; plasma optically thick |
| Early plume | 10-200 ns | $n_e \gtrsim 10^{18}$ cm$^{-3}$, $T \gtrsim 15{,}000$ K | Strong continuum + broad lines | Continuum-dominated; often non-stationary (fails temporal LTE) |
| **Analytical window** | $\sim$0.3-3 $\mu$s | $n_e \sim 10^{16}$-$10^{18}$ cm$^{-3}$, $T \sim$ 8{,}000-15{,}000 K | Resolved atomic + ionic lines | **LTE typically holds here**; the gate CF-LIBS targets |
| Recombination / decay | $\gtrsim 3$ $\mu$s | $n_e < 10^{16}$ cm$^{-3}$ | Weak lines, molecular bands | McWhirter often violated; LTE fails |

The forward model assembles a spectrum for a single, uniform, stationary,
optically thin slice of the analytical window:

$$
\text{PlasmaState}(T, n_e, \{C_s\}) \;\to\; \text{Saha-Boltzmann} \;\to\; n_k
\;\to\; \varepsilon(\lambda) \;\to\; I=\varepsilon L \;\to\; \text{broaden} \;\to\;
\text{convolve LSF} \;\to\; I(\lambda).
$$

Each arrow is an assumption with a validity domain. The sections below take them
in order; each ends with a **What correct code MUST do** checklist and its
implementation + Lean cross-links.

---

## 1. Plasma formation, the plume, and when LTE holds {#formation-lte}

### 1.1 From ablation to a quantifiable plume

The laser vaporizes and ionizes a microgram-scale mass; the ejecta expand,
entrain ambient gas, and thermalize by electron collisions. CF-LIBS never models
the hydrodynamics — it assumes we sample the plume during the window in which a
**single electron temperature** $T$ governs both the bound-level populations
(Boltzmann) and the ionization balance (Saha), and in which collisional rates
dominate radiative ones so the photon field does not control level populations
[@tognoni2010; @aragon2008]. This is **local thermodynamic equilibrium (LTE)**.

### 1.2 What LTE asserts

LTE means each species $s$ in ionization stage $z$ has:

- level populations set by the Boltzmann distribution at $T$ (§2), and
- stage-to-stage ratios set by the Saha equation at $T$ and $n_e$ (§3),

with a **Maxwellian** electron energy distribution at the same $T$. It does **not**
require thermodynamic equilibrium with the radiation field (that would be a
blackbody), only *local* kinetic equilibrium maintained by collisions
[@cristoforetti2010].

### 1.3 The necessary density floor (McWhirter)

Collisional de-excitation must outrun radiative decay. The classical sufficiency
condition for the *steady-state* electron density (McWhirter, reproduced and
critiqued in [@cristoforetti2010]) is

$$
n_e \;\ge\; 1.6\times10^{12}\,\sqrt{T}\,(\Delta E)^{3}\quad[\text{cm}^{-3}],
$$

with $T$ in K and $\Delta E$ in eV. The cubed $\Delta E$ makes this the single
most abused formula in LIBS: **$\Delta E$ is the first resonance transition
energy** (ground $\to$ first dipole-allowed excited level), *not* the gap between
adjacent observed Boltzmann-plot upper levels and *not* $\max(E_k)$
[@cristoforetti2010]. Using an adjacent-level gap ($\sim$0.05 eV) instead of the
resonance gap ($\sim$2-5 eV) underestimates the required $n_e$ by $10^4$-$10^6$
and turns the check into a rubber stamp. Representative resonance gaps: Fe I
$\approx 2.48$ eV, Cu I $\approx 3.82$ eV, Mg I $\approx 2.71$ eV, Ca II
$\approx 1.69$ eV. This is discussed in depth in [§9.1 McWhirter floor](#lte-validity).

### 1.4 Where LTE fails in practice

- **Late gates / recombination**: $n_e$ drops below the McWhirter floor.
- **Low ambient pressure / vacuum**: free expansion rarefies the plume fast; the
  SuperCam-on-Mars diagnostics quantify this envelope for standoff LIBS under
  low-pressure CO$_2$ [@manelski2024].
- **Large $\Delta E$ transitions** (high-lying levels, H, He): the $(\Delta E)^3$
  term makes thermalization hard even at LIBS densities.
- **Transient (early) times**: McWhirter can be satisfied while the *relaxation
  time* criterion is not — see [§9.2](#lte-validity).

> [!NOTE] FORMAL
> The equivalence of the McWhirter density floor and the partial-LTE
> thermalization limit is proven: `lean:CflibsFormal/PartialLTE.lean#mcwhirter_iff_thermalizationLimit`.

### 1.5 What correct code MUST do — formation/LTE

- [ ] Treat LTE as a **gate**, not a default: evaluate McWhirter at converged
  $(T, n_e)$ and surface `lte_mcwhirter_satisfied` + `lte_n_e_ratio`.
- [ ] Use $\Delta E =$ resonance-transition energy per element, passed explicitly;
  never `max(E_k)` or an adjacent-level gap.
- [ ] Flag McWhirter as **necessary, not sufficient**; pair with the relaxation-time
  check for short-lived plasmas.

Implementation: `cflibs/plasma/lte_validator.py::LTEValidator`
(`MCWHIRTER_CONST = 1.6e12` in `cflibs/core/constants.py`).
See also: [lte-validity §9](#lte-validity), [error-budget-and-falsification](error-budget-and-falsification.md).

---

## 2. Boltzmann level populations — the excitation ladder {#boltzmann}

Within one ionization stage, the population of upper level $k$ (statistical weight
$g_k = 2J_k+1$, energy $E_k$ measured **from the ground state of that stage**) is

$$
\frac{n_k}{N_s^{(z)}} = \frac{g_k}{U_s^{(z)}(T)}\,\exp\!\left(-\frac{E_k}{k_B T}\right),
\qquad
U_s^{(z)}(T)=\sum_i g_i\,\exp\!\left(-\frac{E_i}{k_B T}\right),
$$

with $N_s^{(z)}$ the stage number density [cm$^{-3}$], $k_B=8.617333\times10^{-5}$
eV/K [@tognoni2010]. Energies **must** share one reference (NIST ASD reports levels
from each stage's own ground state — use that). The ground term contributes
$g_0$, so $U \to g_0$ as $T\to0$.

### 2.1 The Boltzmann plot ordinate (load-bearing $\lambda$)

Substituting into the integrated line intensity (§4) and linearizing gives the
canonical CF-LIBS ordinate:

$$
\underbrace{\ln\!\left(\frac{I_{ki}\,\lambda_{ki}}{g_k A_{ki}}\right)}_{y}
= -\frac{1}{k_B T}\,E_k + \ln\!\left(\frac{F\,C_s}{U_s(T)}\right),
$$

a straight line in $E_k$ with slope $-1/(k_B T)$. This ordinate — $\ln(I\lambda/gA)$
vs $E_k$ — is the literature standard, printed verbatim across foundational and
modern sources [@aragon2008; @tognoni2010; @ciucci1999], and it is the
*measurement* convention because a spectrometer integrates an energy/radiance
quantity carrying a $hc/\lambda$ per-photon factor. The wiki's method-verdict
document establishes this unambiguously.

> [!WARNING] $\lambda$ IS LOAD-BEARING
> $\lambda$ may be dropped from the ordinate **only** when it is constant across the
> fitted line set (or when intensity is a true photon rate). For a real
> multi-line fit spanning, e.g., 240-660 nm, $\ln\lambda$ varies by
> $\ln(660/240)\approx1.01$; because $\lambda$ correlates with $E_k$, dropping it
> **tilts the slope (biases $T$) and the intercept (biases composition)**, not just
> the intercept. A numerical bridge experiment measured slope $-0.689$ (dropped)
> vs the correct $-0.419$. Never silently drop $\lambda$.

### 2.2 Multi-element common-slope fit

All species in one LTE plasma share $T$, hence the **common slope**: pool lines
across species with per-species intercepts $q_s = \ln(FC_s/U_s)$ but one slope
$-1/(k_BT)$ [@aguilera2007]. This widens the $E_k$ lever arm and is a strict
improvement over per-species plots. Robust estimators (sigma-clip / RANSAC /
Huber) guard against outliers; see [classical-quantification](classical-quantification.md).

> [!NOTE] FORMAL
> The slope identity is machine-checked: the two-level temperature recovery
> `lean:CflibsFormal/Boltzmann.lean#temperature_from_two_levels` and the
> plot affinity `lean:CflibsFormal/Boltzmann.lean#boltzmann_plot`; the
> energy-intensity (explicit-$\lambda$) form is
> `lean:CflibsFormal/ForwardMap.lean#boltzmann_plot_intensity`.

### 2.3 Partition function — truncation is a physics choice {#pf-ipd-invariant}

CF-LIBS evaluates $U_s(T)$ by direct summation over levels when available, else the
Irwin (1981) polynomial $\log_{10}U = \sum_n a_n(\log_{10}T)^n$ [@irwin1981]. Two
non-negotiable rules:

1. **Basis discipline.** Irwin tabulated in $\log_{10}$; the code stores natural-log
   coefficients — a basis change is required. Reported 30-60% $U(T)$ errors in a
   2026 audit were stale fit data, *not* a math bug: always re-fit against the
   current `energy_levels` snapshot. Vetted references: [@barklem2016].
2. **IPD-consistent cutoff.** The sum formally diverges (infinite Rydberg series);
   physical truncation is at the ionization-potential-depression-lowered limit
   $E_{\max}=\chi-\Delta\chi$. The **same** $\Delta\chi$ MUST truncate $U(T)$ *and*
   enter the Saha exponent (§3). Divergent cutoffs break self-consistency — this was
   "audit Family J" (two IPD formulas differing $\sim1.44\times$, fixed 2026).

> [!CAUTION] DO-NOT — contradicted claim
> "Stage III is empty / Fe I has 425 levels" was a symptom of the pre-reset
> incomplete DB. Post-ASD59-reset (203k lines / 62k levels) this is false; do not
> design around a truncated level list. See [atomic-data-and-datasets](atomic-data-and-datasets.md).

### 2.4 What correct code MUST do — Boltzmann

- [ ] $n_k = N_s\,(g_k/U)\exp(-E_k/k_BT)$ with $E_k$ from the stage ground state,
  $g_k=2J_k+1$.
- [ ] Compute $U(T)$ over the **same** IPD cutoff as the populations; never mismatch.
- [ ] Fit $y=\ln(I\lambda/gA)$ vs $E_k$; keep $\lambda$ explicit and per-line.
- [ ] Enforce the multi-element common slope; report per-element intercept + $R^2$.
- [ ] Warn (don't silently return 0) when a species has no levels; fall back to
  $U\approx g_0$ only as a last resort.

Implementation: `cflibs/plasma/saha_boltzmann.py::SahaBoltzmannSolver`,
`cflibs/plasma/partition.py`.

---

## 3. Saha ionization balance {#saha}

Between adjacent stages $z$ and $z+1$, LTE fixes the ratio via the Saha-Eggert
equation [@tognoni2010; @stewart1966]:

$$
\frac{n_{z+1}\,n_e}{n_z}
= \frac{2\,U_{z+1}(T)}{U_z(T)}
\left(\frac{2\pi m_e k_B T}{h^2}\right)^{3/2}
\exp\!\left(-\frac{\chi_z-\Delta\chi}{k_B T}\right).
$$

- The leading **2** is the free-electron spin degeneracy ($g_e=2$, $U_e=1$) — *not*
  the thermal-de-Broglie factor and *not* a nuclear-spin term.
- $(2\pi m_e k_B T/h^2)^{3/2}$ is the inverse cube of the thermal de Broglie
  wavelength, carrying the $m^{-3}$ number-density dimension.
- The sign is correct: ionization costs energy, suppressing the higher stage.

In LIBS-convenient units ($T$ in eV, $n$ in cm$^{-3}$, $\chi$ in eV) the prefactor
collapses to a constant:

$$
\frac{n_{z+1} n_e}{n_z} = \text{SAHA\_CONST}\cdot T_{\text{eV}}^{3/2}
\frac{2U_{z+1}}{U_z}\exp\!\left(-\frac{\chi_{\text{eff}}}{T_{\text{eV}}}\right),
\quad \text{SAHA\_CONST}\approx 6.04\times10^{21}\ \text{cm}^{-3}\,\text{eV}^{-3/2}.
$$

This constant is valid **only** for $(T_{\text{eV}}, n_e[\text{cm}^{-3}], \chi[\text{eV}])$;
mixing SI/CGS here is a classic silent error. The multi-stage balance
($n_I+n_{II}+n_{III}=n_s$) solves $n_I=n_s/(1+S_1+S_1S_2)$ with $S_z\equiv n_{z+1}/n_z$.

### 3.1 Ionization-potential depression (IPD)

At $n_e\sim10^{16}$-$10^{18}$ cm$^{-3}$ the continuum is lowered by
$\Delta\chi=\chi-\chi_{\text{eff}}$. Default is Debye-Hückel
($\Delta\chi_{DH}=e^2/\lambda_D$, $\approx0.066$ eV at $n_e=10^{17}$,
$T=10^4$ K); Stewart-Pyatt interpolates Debye-Hückel and ion-sphere limits and is
the standard higher-density model [@stewart1966]. Same $\Delta\chi$ everywhere (§2.3).

### 3.2 Saha correction onto the neutral plane

The inversion maps every ionic-line intercept back to the neutral Boltzmann plane
via the Saha factor $S(T,n_e)$, so neutral and ionic lines share one plot. Because
Saha couples $T$ and $n_e$, the loop re-derives $n_e$ from charge balance
$n_e=\sum_s\sum_z z\,n_{s,z}$ and iterates — see
[classical-quantification](classical-quantification.md).

> [!NOTE] FORMAL
> Saha positivity/relation `lean:CflibsFormal/Saha.lean#saha_relation`;
> the electron-density antitone property (higher $n_e$ suppresses ionization)
> `lean:CflibsFormal/Saha.lean#electronDensity_antitone`; two-stage charge
> neutrality `lean:CflibsFormal/Saha.lean#chargeNeutrality_two_stage`.

### 3.3 What correct code MUST do — Saha

- [ ] Include the factor **2** (electron spin) in the numerator.
- [ ] Apply IPD: $\chi_{\text{eff}}=\max(\chi-\Delta\chi,0)$, same $\Delta\chi$ as the
  $U(T)$ cutoff.
- [ ] Keep units consistent (SAHA_CONST is eV/cm$^{-3}$-only).
- [ ] Iterate charge balance $n_e=\sum(\text{stage}\cdot n_{\text{stage}})$ to convergence.

Implementation: `cflibs/plasma/saha_boltzmann.py::SahaBoltzmannSolver`.

---

## 4. Emissivity, integrated intensity, and the optically-thin limit {#emissivity-rt}

The volumetric emission coefficient of transition $k\to i$ (power per volume per
steradian) is [@tognoni2010; @aragon2008]:

$$
\varepsilon_{ki}(\lambda)=\frac{hc}{4\pi\lambda_{ki}}\,A_{ki}\,n_k\,\phi_{ki}(\lambda),
\qquad \int\phi_{ki}\,d\lambda=1.
$$

Two conventions that bite:

- The prefactor is $(hc/\lambda)/4\pi$ — $c$ and $4\pi$ are **not** grouped; the
  $1/\lambda$ is the per-photon energy $h\nu=hc/\lambda$. Omitting the $1/\lambda$
  produces a wavelength-dependent slope error in the Boltzmann plot ($\Rightarrow$
  wrong $T$).
- The $1/(4\pi)$ is isotropic emission over the full sphere.

For a uniform slab of path length $L$ in the **optically-thin limit** ($\tau\ll1$
for every fitted line):

$$
I(\lambda)=\varepsilon(\lambda)\,L,\qquad
I_{ki}=\frac{hc}{4\pi\lambda_{ki}}A_{ki}\,n_k\,L
= F\,C_s\frac{g_k A_{ki}}{U_s(T)}\exp\!\left(-\frac{E_k}{k_BT}\right)\frac{1}{\lambda_{ki}},
$$

where $F$ is a single scalar experimental/calibration factor (geometry $\times$
detector $\times$ volume) that **cancels through closure** ($\sum_s C_s=1$). This
is the CF-LIBS line-emission coefficient of [@tognoni2010, Eq. 1] and the original
Ciucci procedure [@ciucci1999]. The radiative-transfer derivation of the
thin limit is textbook (Rybicki & Lightman, ch. 1).

The line opacity that decides thinness is set by the **lower-level** population:

$$
\kappa(\nu)\approx\frac{A_{ki}g_k\lambda^2}{8\pi}\,\frac{N_s}{U_s(T)}
\exp\!\left(-\frac{E_i}{k_BT}\right)\phi(\nu),\qquad \tau_\nu=\kappa_\nu L.
$$

Note $E_i$ (lower level) and $g_k$ (upper weight via Einstein's relation) — a
common off-by-$(g_i/g_k)$ trap [@amamou2002].

> [!NOTE] FORMAL
> Line-intensity positivity and the $\lambda$-form Boltzmann plot:
> `lean:CflibsFormal/ForwardMap.lean#lineIntensity_pos`,
> `lean:CflibsFormal/ForwardMap.lean#boltzmann_plot_intensity`.

### 4.1 What correct code MUST do — emissivity/RT

- [ ] Use $\varepsilon=(hc/4\pi\lambda)\,A_{ki}\,n_k$; never omit $1/\lambda$, never
  regroup as $(hc/4\pi)\,A_{ki}n_k$.
- [ ] Normalize $\phi$: $\int\phi\,d\lambda=1$ (unit-test $\texttt{trapz}\approx1$).
- [ ] Optically-thin intensity $I=\varepsilon L$; verify $\tau_{\max}<0.1$ (flag
  $\tau>0.3$) for every inversion line.
- [ ] Compute $\tau$ from the **lower-level** column density, with $E_i$ and the
  correct $g$ ratio.

Implementation: `cflibs/radiation/emissivity.py::calculate_line_emissivity`.

---

## 5. Line broadening — Doppler, Stark, Voigt, and the instrument LSF {#broadening}

The observed line shape is the convolution of a physical profile (thermal +
collisional) with the instrument line-spread function. Wavelengths here are
rest-frame air (§9). The canonical modern review is [@harilal2022]; the
impact-approximation Stark theory traces to Griem (*Spectral Line Broadening by
Plasmas*, Academic Press 1974 — a book, no DOI).

### 5.1 Doppler (Gaussian)

Maxwellian line-of-sight motion of emitters of mass $m_s$ gives a Gaussian of
1-$\sigma$ width

$$
\sigma_D=\frac{\lambda_0}{c}\sqrt{\frac{k_B T}{m_s}},\qquad
\Delta\lambda_D^{\text{FWHM}}=\lambda_0\sqrt{\frac{8k_BT\ln2}{m_s c^2}}=2.3548\,\sigma_D.
$$

The recurring bug is FWHM/$\sigma$ confusion (factor 2.3548): derive $\sigma_D$ from
$\lambda_0/c\cdot\sqrt{k_BT/m_s}$, never use the FWHM formula as $\sigma$.

### 5.2 Stark (Lorentzian) — the $n_e$ channel

Electron impacts in the impact approximation give a Lorentzian whose FWHM is
**linear** in $n_e$ [@harilal2022]:

$$
\Delta\lambda_S \approx 2\,w\,\frac{n_e}{10^{16}}\quad[\text{Å},\ n_e\ \text{in cm}^{-3}],
$$

with $w$ the electron-impact half-width at a reference $T$. This linearity is the
electron-density diagnostic. Coefficient sources, in order of preference:

| Source | Coverage | DOI |
|--------|----------|-----|
| Djurović-Blagojević-Konjević 2023 (newest critical review, 1665 lines, accuracy-graded + semiclassical) | 35 elements | [@djurovic2023] |
| Konjević et al. 2002 (critical experimental review 1989-2000) | broad | [@konjevic2002] |
| STARK-B / VAMDC (semiclassical, T-dependent, non-hydrogenic) | Ca II, Mg II, Fe, Al… | [@sahalbrechot2015] |
| Gigosos et al. 2003 (ion-dynamics H-Balmer tables) | H$\alpha$/$\beta$/$\gamma$ | [@gigosos2003] |

The current single-coefficient convention and the $\sim n_e^{0.7}$ H$\alpha$ scaling
are known simplifications — the ion-dynamics tables of [@gigosos2003] and the
T-scaled STARK-B widths [@sahalbrechot2015] are the upgrade path. CF-LIBS does
**not** model the Stark *shift* of line centers, only widths (a documented gap).

### 5.3 Voigt (convolution) and the Olivero-Longbothum width

Doppler $\otimes$ Stark is a Voigt profile,

$$
V(x;\sigma,\gamma)=\frac{\operatorname{Re}[w(z)]}{\sigma\sqrt{2\pi}},\quad
z=\frac{x+i\gamma}{\sigma\sqrt2},
$$

with $w(z)$ the Faddeeva function. The CPU path uses Humlíček W4; the JAX path uses
Weideman-32 (gradient-stable). The FWHM has the Olivero-Longbothum closed form
(error $<0.02\%$) [@olivero1977]:

$$
f_V \approx 0.5343\,f_L + \sqrt{0.2169\,f_L^2 + f_G^2},
$$

and the pseudo-Voigt mixing $\eta$ is [@thompson1987]. Width composition rules:
**Gaussian widths add in quadrature** ($\sigma_G^2=\sigma_D^2+\sigma_{\text{inst}}^2$);
**Lorentzian widths add linearly**. Mixing these up is a systematic width error.

### 5.4 Instrument LSF — fixed-FWHM vs resolving-power

$$
\text{fixed:}\ \sigma_{\text{inst}}=\frac{\text{FWHM}}{2.3548};\qquad
\text{resolving-power }R=\lambda/\Delta\lambda:\ \sigma_{\text{inst}}(\lambda)=\frac{\lambda}{R\cdot2.3548}.
$$

Resolving-power mode is mandatory for echelle spectrometers, where the slit width
scales with $\lambda$. Off-by-2.3548 (using $\lambda/R$ as $\sigma$) is the usual bug.
Real spectrometers can be non-Gaussian (slit diffraction, pixel response); a
Gaussian LSF assumption biases recovered widths — calibrate against a known lamp.

> [!NOTE] FORMAL
> Doppler width monotone in $T$ and invertible:
> `lean:CflibsFormal/LineBroadening.lean#dopplerFWHM_strictMono_T`,
> `#doppler_recovers`; Stark width linear/injective in $n_e$:
> `lean:CflibsFormal/StarkBroadening.lean#starkFWHM_isLinear`,
> `#starkDensity_recovers`; Voigt Gaussian/Lorentzian limits and monotonicity:
> `lean:CflibsFormal/VoigtWidth.lean#voigt_gaussian_limit`, `#voigt_lorentzian_limit`;
> Gaussian-quadrature deconvolution `#deconvolveGaussian_quadrature`.

### 5.5 What correct code MUST do — broadening

- [ ] $\sigma_D=\lambda/c\cdot\sqrt{k_BT/m_s}$; FWHM $=2.3548\,\sigma_D$.
- [ ] $\sigma_{\text{inst}}=\lambda/(R\cdot2.3548)$ (or fixed FWHM$/2.3548$).
- [ ] $\sigma_G=\sqrt{\sigma_D^2+\sigma_{\text{inst}}^2}$ (quadrature, Gaussian only);
  Lorentzian widths add linearly.
- [ ] Stark $\gamma_S=w\,(n_e/10^{16})$ from graded tables; carry per-line $w$
  uncertainty into $n_e$ error.
- [ ] Voigt via Faddeeva, normalized to unit area.

Implementation: `cflibs/radiation/profiles.py`, `cflibs/radiation/stark.py`,
`cflibs/inversion/physics/stark_ne.py`.

---

## 6. Self-absorption and the curve of growth {#self-absorption-cog}

When $\tau$ is not small, emitted photons are reabsorbed along the line of sight
and the linear $I\propto n L$ relation fails. For a homogeneous isothermal slab
the source function is Planckian ($S_\lambda=B_\lambda(T)$, Kirchhoff) and

$$
I_\lambda=B_\lambda(T)\,[1-e^{-\tau_\lambda}].
$$

The **self-absorption factor** — the ratio of observed to optically-thin intensity —
is the escape function

$$
SA(\tau_0)=\frac{1-e^{-\tau_0}}{\tau_0}\in(0,1],
$$

$SA=1$ thin, $SA\to0$ saturated [@bulajic2002; @rezaei2020]. The **curve of growth**
(COG) has three regimes: linear ($\tau_0\ll1$, $W\propto N_sL$, Boltzmann plot
valid); flat-of-the-curve ($\tau_0\sim1$-$10$); square-root ($\tau_0\gg1$, Doppler
core, $W\propto\sqrt{N_sL}$). Diagnostic signatures of self-absorption: the
Boltzmann plot **bends** at high intensity / low $E_k$; neutral-only and ion-only
plots disagree; doublet intensity ratios deviate from the theoretical $g_kA_{ki}$
ratio.

> [!NOTE] FORMAL
> $SA$ is *derived*, not assumed: the slab intensity equals thin $\times\,SA$
> (`lean:CflibsFormal/SelfAbsorption.lean#slabIntensity_eq_thin_mul_SA`), with
> $SA\in(0,1]$ (`#selfAbsorptionFactor_pos`, `#selfAbsorptionFactor_le_one`),
> strictly decreasing in $\tau$ (`#selfAbsorptionFactor_strictAntiOn`), and exactly
> left-invertible (`#lineIntensity_eq_selfAbsorbedIntensity_div`). The COG ratio is
> strictly antitone/injective: `lean:CflibsFormal/CurveOfGrowth.lean#cogRatio_strictAntiOn`.

### 6.1 Correction methods and their reach

| Method | Needs | Reach | Ref |
|--------|-------|-------|-----|
| Exclude resonance lines (default) | line list | cheapest defense | [@aragon2008] |
| IRSAC (internal reference) | thin reference line, $T$ | restores linearity, no Stark $w$ | [@sun2009] |
| COG iteration (Bulajić) | fit $T,n_e,\sigma,\gamma,L$ | moderate $\tau$ | [@bulajic2002] |
| Planck-function (BRR-SAC) | $T$ only | simplest; unstable as $I/B\to1$ | [@poggialini2023] |
| C-sigma graphs | $\ge4$ lines/species | exploits SA, joint $T,n_e$ | [@aragon2014] |
| Two-layer (reversed lines) | profile shape | self-reversal only | [@amamou2003] |

### 6.2 Self-reversal ≠ self-absorption

Self-absorption *flattens* a profile; **self-reversal** produces a central *dip*
when a cool absorbing periphery sits in front of a hot core — an inhomogeneous,
two-zone effect the homogeneous $SA(\tau)$ formula does **not** describe. Reversed
lines need the two-layer correction [@amamou2003] or exclusion. CF-LIBS does not
model self-reversal (documented gap). At severe depth ($\tau_0>3$, $SA<0.3$)
concentration becomes **unidentifiable** from that line ($I\to B_\lambda(T)$
independent of $N_s$) — do not attempt single-line correction there.

> [!CAUTION] FALSIFIED: Composition-derived per-line $\tau$ improves accuracy
> - **Claim:** compute $\tau$ from the composition-derived lower-level column density and correct each line for thick-line saturation.
> - **Predicted:** lower held-out RMSEP on ChemCam BHVO-2.
> - **Observed:** RMSEP *increased* — positive-feedback loop ($\tau$ feeds composition feeds $\tau$).
> - **Verdict:** REJECTED; replaced by the observable-anchored corrector.
> - **Evidence:** `docs/research/physics-first-principles-audit.md` Issue 3 (finding F4); `cflibs/inversion/physics/self_absorption_observable.py`.
> - **Date:** 2026-07-02

### 6.3 What correct code MUST do — self-absorption

- [ ] Screen with a doublet-ratio test (flag $R_{\text{obs}}/R_{\text{theory}}<0.95$)
  and flag resonance lines ($E_i=0$).
- [ ] Use $E_i$ (lower level) and $g_k$ in $\kappa$; peak $SA=(1-e^{-\tau_0})/\tau_0$.
- [ ] Iterate SA $\to$ correct $\to$ re-fit $T$ until $|\Delta T/T|<0.5\%$.
- [ ] Exclude reversed lines and $\tau_0>3$ lines; mark unbounded uncertainty when a
  species is entirely self-absorbed.
- [ ] Prefer an **observable-anchored** corrector over a composition-derived $\tau$
  (see falsification above).

Implementation: `cflibs/inversion/physics/self_absorption.py` (CDSB / iterative),
`cflibs/inversion/physics/self_absorption_observable.py`.

---

## 7. Continuum and molecular emission — forward-model gaps {#continuum-molecular}

### 7.1 Continuum

Two continuum sources underlie every LIBS spectrum:

- **Bremsstrahlung** (free-free), $\varepsilon_{\text{ff}}\propto n_e^2 T^{-1/2}$
  times a Gaunt factor;
- **Recombination** (free-bound), a series of edges as electrons recombine to bound
  states.

Both scale as $n_e^2$ and decay faster than line emission, which is *why* a
gate delay of $\sim1$ $\mu$s buys a clean baseline [@aragon2008; @harilal2022]. In
the shipped pipeline the continuum is **assumed removed in preprocessing** (rolling
ball / asymmetric least squares / polynomial detrend) rather than forward-modeled —
a deliberate scope choice. Leaving continuum in inflates every integrated intensity
(a wavelength-dependent floor).

> [!NOTE] FORMAL
> Baseline subtraction is exact under the additive-continuum model
> (`lean:CflibsFormal/Continuum.lean#baseline_subtraction_exact`); the line-to-continuum
> ratio is a monotone $T$ diagnostic (`#lineToContRatio_strictMono_T`).

### 7.2 Molecular bands

CN, C$_2$, OH, N$_2$, N$_2^+$ electronic bands appear in ambient-gas-entrained and
low-temperature plumes and are **not** LTE atomic lines. CF-LIBS has **zero
molecular code** — these bands are a forward-model gap flagged wherever they can
contaminate atomic-line intensities (they are not in HITRAN; electronic bands need
PGOPHER/LIFBASE-class tooling). Treat regions with molecular structure as
excluded, not modeled. See [frontier-methods](frontier-methods.md).

### 7.3 What correct code MUST do — continuum/molecular

- [ ] Subtract the continuum before intensity integration; document the method.
- [ ] Prefer gate delay ($\gtrsim1$ $\mu$s) to suppress $n_e^2$ continuum at source.
- [ ] Flag (do not fit) wavelength regions with molecular band structure.

Implementation: `cflibs/inversion/preprocess` (baseline);
`cflibs/radiation/spectrum_model.py` (line-only forward model).

---

## 8. Temporal and spatial evolution — the single-zone caveat {#temporal-spatial}

### 8.1 Temporal averaging

A gated spectrum integrates a *cooling, rarefying* plume: $n_e$ roughly halves on a
microsecond scale, so a 1 $\mu$s gate after a 1 $\mu$s delay sees real evolution
[@harilal2022]. The recovered $(T,n_e)$ are emission-weighted time averages, biased
toward the bright early times. Guard by comparing gate windows: composition that
drifts across gates is gate-dependent, not a property of the sample.

> [!NOTE] FORMAL
> Under the gate-averaged Saha-Boltzmann model, composition is provably
> gate-independent / temperature-in-situ:
> `lean:CflibsFormal/TemporalEvolution.lean#temporal_composition_invariant`,
> `#temporal_composition_gate_independent`, `#temporal_saha_composition_invariant`.
> This is the *idealized* invariance; real inhomogeneity (§8.2) breaks its premise.

### 8.2 Spatial (line-of-sight) averaging and the single-zone caveat

Real plumes are stratified: hot dense core, cool periphery, sharp boundaries. The
shipped model is `SingleZoneLTEPlasma` — one uniform $(T,n_e,\{C_s\})$ along the
line of sight. A single-zone fit therefore returns an **emission-weighted LOS
average** temperature, biased toward the high-density emitting region. Neutral
lines (cool periphery) and ionic lines (hot core) can report different
"temperatures" that the common-slope fit averages by signal strength. Diagnostics:
per-element $T$ disagreeing $>10$-$15\%$ after Saha correction; poor
neutral/ion consistency. CF-LIBS ships **no** multi-zone / Abel-inversion solver;
spatial resolution requires imaging + Abel inversion outside this framework.

> [!NOTE] FORMAL
> Chord/single-zone identifiability is established under the model geometry:
> `lean:CflibsFormal/SpatialForward.lean#singleZone_identifiable`,
> `#chord_profile_identifiable`.

### 8.3 What correct code MUST do — temporal/spatial

- [ ] Treat recovered $(T,n_e,\{C_s\})$ as gate- and LOS-averaged; report stability
  across $\ge2$ gate windows.
- [ ] Surface per-element and neutral/ion $T$ consistency metrics; large
  disagreement signals stratification, not a fit error to force-converge.
- [ ] Do not claim spatial resolution the single-zone model cannot deliver.

Implementation: `cflibs/plasma/state.py::SingleZoneLTEPlasma`,
`cflibs/inversion/runtime/` (temporal gate optimization).

---

## 9. LTE validity and the air-vacuum invariant {#lte-validity}

### 9.1 McWhirter floor (restated as a gate)

$n_e\ge1.6\times10^{12}\sqrt{T}(\Delta E)^3$ cm$^{-3}$ with $\Delta E$ the **first
resonance transition** energy [@cristoforetti2010]. Necessary, not sufficient;
evaluate per spectrum/gate and report `lte_n_e_ratio = n_e/n_e^{\text{req}}`.

### 9.2 Cristoforetti relaxation and diffusion criteria (beyond McWhirter)

For transient plasmas, the excitation relaxation time must beat the plume
evolution time [@cristoforetti2010]:

$$
\tau_{\text{rel}}\approx\frac{6.3\times10^4}{n_e\langle g\rangle f_{nm}}\,
\Delta E_{nm}\sqrt{k_BT}\,\exp\!\left(\frac{\Delta E_{nm}}{k_BT}\right)\ [\text{s}],
\qquad \tau_{\text{rel}}<\tau_{\text{evol}}/10.
$$

Because $\tau_{\text{rel}}$ grows exponentially in $\Delta E/k_BT$ and inversely in
$n_e$, a plume can satisfy the steady-state McWhirter floor yet fail the transient
criterion (early/late times). A **diffusion** length check (that species do not
diffuse out of the emitting volume before thermalizing) and the two-temperature /
ionizing-vs-recombining framework of [@cristoforetti2013] complete the picture;
non-LTE population departures are quantified against collisional-radiative models
[@pietanza2010] (deviations $\sim$10% at $n_e\sim10^{18}$, $\sim$27% at $10^{17}$,
$\sim$42% at $10^{15}$ cm$^{-3}$). The blackbody-limit in-spectrum LTE proof of
Hermann et al. is an independent check: thick line cores must not exceed
$B_\lambda(T)$ [@hermann2017; @hermann2018].

> [!NOTE] FORMAL
> `lean:CflibsFormal/PartialLTE.lean#mcwhirter_iff_thermalizationLimit`,
> `#lteValid_iff_thermalized`; McWhirter bound monotone in $T$ and $\Delta E$:
> `lean:CflibsFormal/StarkBroadening.lean#mcWhirterBound_mono_T`, `#mcWhirterBound_mono_dE`,
> and Stark/Saha/LTE self-consistency `#stark_saha_lte_consistent`.

### 9.3 Air-vacuum wavelength invariant {#air-vacuum-invariant}

The shipped DB stores **air** wavelengths above 200 nm (NIST ASD convention). The
air-vacuum dispersion is the IAU-adopted Morton (2000) relation [@morton2000];
below 200 nm values are vacuum. Consequences and rules:

- The offset is $\approx +114$ pm at 400 nm (283 ppm), rising toward the red —
  larger than a typical 50-100 pm line-match window, so a single air/vacuum slip
  silently collapses identification and corrupts the Boltzmann slope.
- **One** utility performs all conversion (`cflibs/atomic/wavelength_conversion.py`);
  ingestion must assert the source medium (VALD delivers *air* when extracted in air
  units despite storing vacuum internally — verify the `WL_air(A)` header).
- Doppler and Stark formulas use the rest-frame wavelength; be internally
  consistent (all air or all vacuum) within any single computation.

### 9.4 What correct code MUST do — LTE + air-vacuum

- [ ] Gate on McWhirter with the resonance $\Delta E$; add the relaxation-time check
  for short-lived plasmas; treat both as necessary-not-sufficient.
- [ ] Invalidate $T$, $n_e$, and composition when the gate fails, even if the solver
  converged (refuse-to-report over silent output).
- [ ] Route every wavelength through the single air-vacuum utility; assert the
  source medium at ingest; state the convention in any wavelength-bearing output.

Implementation: `cflibs/plasma/lte_validator.py::LTEValidator`,
`cflibs/atomic/wavelength_conversion.py`.

---

## How to audit a result (physics acceptance gate)

A CF-LIBS result defensible in a paper passes all of: McWhirter satisfied;
common-slope Boltzmann $R^2\ge0.95$ per element; per-element $T$ agree within 10%;
neutral/ion $T$ agree within 10% after Saha correction; reduced $\chi^2\approx1$
(Bayesian/hybrid); composition stable across two gate windows; NIST-parity
validator passes for the used range. The full acceptance logic and its failure
ledger live in [error-budget-and-falsification](error-budget-and-falsification.md).

## See also

- [formal-spec](formal-spec.md) — symbol/notation authority; Lean theorem ids.
- [classical-quantification](classical-quantification.md) — Boltzmann-plot inversion, Saha correction, closure.
- [cf-libs-family](cf-libs-family.md) — OPC, C-sigma, CD-SB, inverse-CF variants.
- [atomic-data-and-datasets](atomic-data-and-datasets.md) — line lists, partition functions, Stark coefficients, air-vacuum ingest.
- [impl-literature-methods](impl-literature-methods.md) — where each equation lives in the code and how the method verdicts map to it.
- [error-budget-and-falsification](error-budget-and-falsification.md) — the falsification ledger and do-not-do list.
