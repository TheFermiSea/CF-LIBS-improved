# CF‑LIBS

A **production‑grade** Python library for **computational laser‑induced breakdown spectroscopy (CF‑LIBS)**: forward modeling, inversion, and analysis of LIBS plasmas with an emphasis on **rigorous physics**, **high‑performance numerics**, and **reproducible workflows**.

CF‑LIBS is intended as a *foundation* for serious research and engineering work in:

- Elemental analysis and calibration‑free LIBS
- Plasma diagnostics (temperature, electron density, composition)
- Instrument design and optimization
- Synthetic spectral generation and forward modeling
- Bayesian and deterministic inversion of LIBS signals

> **Status**: Early design & scaffolding. This README documents the *intended* production‑grade architecture and physics model the project will grow into.

---

## Table of Contents

1. [Conceptual Overview](#conceptual-overview)  
2. [Physics Model](#physics-model)  
   - [Plasma State and Assumptions](#plasma-state-and-assumptions)  
   - [Level Populations: Boltzmann & Saha](#level-populations-boltzmann--saha)  
   - [Line Emission and Radiative Power](#line-emission-and-radiative-power)  
   - [Line Broadening & Profiles](#line-broadening--profiles)  
   - [Opacity & Radiative Transfer](#opacity--radiative-transfer)  
   - [Stark Broadening and Electron Density Diagnostics](#stark-broadening-and-electron-density-diagnostics)  
   - [Instrument Response and Detector Modeling](#instrument-response-and-detector-modeling)  
3. [Architecture](#architecture)  
   - [Core Packages](#core-packages)  
   - [Data & Configuration](#data--configuration)  
   - [Performance and Numerical Design](#performance-and-numerical-design)  
4. [Usage Examples (Planned API)](#usage-examples-planned-api)  
5. [Development Roadmap](#development-roadmap)  
   - [Phase 0 – Scaffold & Core Utilities](#phase-0--scaffold--core-utilities)  
   - [Phase 1 – Minimal Viable Physics Engine](#phase-1--minimal-viable-physics-engine)  
   - [Phase 2 – Production‑Grade CF‑LIBS Engine](#phase-2--production-grade-cf-libs-engine)  
   - [Phase 3 – Advanced Inversion & Uncertainty](#phase-3--advanced-inversion--uncertainty)  
   - [Phase 4 – Ecosystem & Integrations](#phase-4--ecosystem--integrations)  
6. [Contributing](#contributing)  
7. [License](#license)

---

## Conceptual Overview

### What CF‑LIBS Aims to Be

CF‑LIBS (Computational Framework for Laser‑Induced Breakdown Spectroscopy) is a **plasma‑physics‑grounded** toolkit for:

- **Forward modeling**: Given plasma parameters  
  $$
  \Theta = \{T_e,\, T_g,\, n_e,\, \mathbf{n}_\text{species},\, p,\, t,\, \text{geometry},\, \text{instrument}\},
  $$  
  compute the emergent spectrum
  $$
  I_\lambda(\lambda;\Theta)
  $$
  over a specified wavelength grid.

- **Inverse modeling (calibration‑free LIBS)**: Given measured spectra $I_\lambda^\text{meas}(\lambda)$, infer:
  - Plasma temperature(s): $T_e,\, T_g$
  - Electron density: $n_e$
  - Species number densities or concentrations: $\mathbf{n}_\text{species}$
  - Uncertainties on the inferred parameters.

- **Diagnostics and feature extraction**:
  - Stark broadening analysis for $n_e$
  - Boltzmann plots for $T_e$
  - Line ratio diagnostics
  - Synthetic spectrum generation for experimental design

---

## Physics Model

The long‑term goal is a **modular physics engine** that can support increasing model complexity without sacrificing clarity, testability, or performance.

### Plasma State and Assumptions

Baseline assumptions for the *core* CF‑LIBS model:

1. **Optically thin plasma** for initial implementation, with extension to partially optically thick conditions.
2. **Local Thermodynamic Equilibrium (LTE)** or partial LTE for electron and excitation distributions.
3. **Quasi‑neutrality**:
   $$
   n_e \approx \sum_j Z_j n_j^Z,
   $$
   where $n_j^Z$ is the density of ion $Z$ of species $j$.

4. **Single‑temperature electron population** characterized by $T_e$ (with possibility for separate heavy‑particle temperature $T_g$ in later phases).

The plasma state vector for a single homogeneous zone is:

$$
\mathbf{X} = \left( T_e,\ T_g,\ n_e,\ \{n_{j,z}\},\ p,\ \mathbf{u},\ t \right).
$$

where:
- $n_{j,z}$ – number density of ionization stage $z$ of element $j$
- $\mathbf{u}$ – flow velocity (for Doppler shifts in advanced modes)
- $t$ – time after laser pulse.

CF‑LIBS will be built to support **multi‑zone** models, i.e.,

$$
I_\lambda(\lambda) = \int_{\text{LOS}} \epsilon_\lambda(\lambda, \mathbf{r}) \, \mathrm{d}\ell,
$$

with discretization into zones $k = 1, \dots, N_z$.

---

### Level Populations: Boltzmann & Saha

For a given species $j$, ionization stage $z$, and level $i$, the population density in LTE is:

1. **Boltzmann distribution** for excitation within an ionization stage:

$$
\frac{n_{i,j,z}}{n_{j,z}} = \frac{g_{i,j,z}}{U_{j,z}(T_e)} 
\exp\left( -\frac{E_{i,j,z}}{k_B T_e} \right),
$$

where:
- $n_{i,j,z}$ – population of level $i$ of species $j$, ionization stage $z$
- $g_{i,j,z}$ – statistical weight
- $E_{i,j,z}$ – excitation energy above the ground state
- $U_{j,z}(T_e)$ – partition function
- $k_B$ – Boltzmann constant.

2. **Saha equation** for ionization balance:

$$
\frac{n_{j,z+1}\, n_e}{n_{j,z}} 
= \left( \frac{2 \pi m_e k_B T_e}{h^2} \right)^{3/2}
\frac{2 U_{j,z+1}(T_e)}{U_{j,z}(T_e)}
\exp \left( -\frac{\chi_{j,z}}{k_B T_e} \right),
$$

where:
- $\chi_{j,z}$ – ionization energy from stage $z \to z+1$
- $m_e$ – electron mass
- $h$ – Planck constant.

For a given total elemental abundance $n_j = \sum_z n_{j,z}$, CF‑LIBS will solve the coupled **Saha–Boltzmann system** under constraints:

$$
\sum_z n_{j,z} = n_j,\quad 
n_e = \sum_j \sum_z z\, n_{j,z}.
$$

This yields level populations $n_{i,j,z}$ used for line emissivity.

---

### Line Emission and Radiative Power

For a transition $u \to l$ (upper $u$, lower $l$), assuming spontaneous emission dominates:

$$
\epsilon_{\lambda}^{(u \to l)}(\lambda) 
= \frac{h c}{4\pi \lambda_{ul}} A_{ul} n_{u,j,z} \, \phi_{ul}(\lambda),
$$

where:

- $\epsilon_{\lambda}$ – spectral emissivity [W m$^{-3}$ nm$^{-1}$]
- $\lambda_{ul}$ – transition wavelength
- $A_{ul}$ – Einstein A coefficient
- $n_{u,j,z}$ – population of the upper level
- $\phi_{ul}(\lambda)$ – normalized line profile:
  $$
  \int_{-\infty}^{\infty} \phi_{ul}(\lambda)\, \mathrm{d}\lambda = 1.
  $$

Total emissivity is the sum over all lines:

$$
\epsilon_{\lambda}(\lambda) = \sum_{j,z,u,l} \epsilon_{\lambda}^{(u \to l)}(\lambda).
$$

For an optically thin, homogeneous plasma with length $L$ along the line of sight:

$$
I_\lambda(\lambda) = \int_0^L \epsilon_{\lambda}(\lambda) \,\mathrm{d}\ell 
\approx \epsilon_{\lambda}(\lambda) L.
$$

---

### Line Broadening & Profiles

CF‑LIBS will support **composite line profiles** built from:

- **Doppler (thermal) broadening**:
  $$
  \Delta\lambda_D = \lambda_0 \sqrt{\frac{2 k_B T_g}{m c^2}},
  $$
  with Gaussian profile:
  $$
  \phi_D(\lambda) = 
  \frac{1}{\Delta\lambda_D \sqrt{\pi}}
  \exp\left[
    -\left( \frac{\lambda - \lambda_0}{\Delta\lambda_D} \right)^2
  \right].
  $$

- **Lorentzian broadening** from:
  - Natural radiative decay
  - Collisions (van der Waals, resonance, electron impact / Stark)
  
  With total Lorentzian FWHM $\Gamma_L$, the profile:
  $$
  \phi_L(\lambda) = 
  \frac{1}{\pi} \frac{(\Gamma_L/2)}{(\lambda - \lambda_0)^2 + (\Gamma_L/2)^2 }.
  $$

- **Voigt profile** as a convolution of Gaussian and Lorentzian:
  $$
  \phi_V(\lambda) = 
  \int_{-\infty}^\infty
  \phi_D(\lambda') \phi_L(\lambda - \lambda')
  \, \mathrm{d}\lambda'.
  $$

CF‑LIBS will implement efficient Voigt calculations (rational approximations / Faddeeva function) for production use.

---

### Opacity & Radiative Transfer

Moving beyond strictly optically thin conditions, the absorption coefficient for a transition $l \to u$ is:

$$
\kappa_{\lambda}^{(l \to u)}(\lambda) =
\frac{h c}{4\pi \lambda_{ul}} B_{lu}
\left( n_{l,j,z} - \frac{g_l}{g_u} n_{u,j,z} \right) \phi_{ul}(\lambda),
$$

where $B_{lu}$ is the Einstein B coefficient.

Total opacity $\kappa_\lambda(\lambda)$ is the sum over all transitions and continua. For a homogeneous slab with **source function** $S_\lambda = \epsilon_\lambda / \kappa_\lambda$:

$$
I_\lambda(\lambda) = S_\lambda(\lambda) \left[ 1 - e^{-\kappa_\lambda(\lambda) L} \right].
$$

In the optically thin limit $\kappa_\lambda L \ll 1$:

$$
I_\lambda(\lambda) \approx \epsilon_\lambda(\lambda) L.
$$

Future versions of CF‑LIBS will support multi‑zone radiative transfer:

$$
\frac{\mathrm{d} I_\lambda}{\mathrm{d} s} = -\kappa_\lambda I_\lambda + \epsilon_\lambda.
$$

---

### Stark Broadening and Electron Density Diagnostics

For many LIBS lines (especially hydrogen and certain metals), the **Stark FWHM** can be approximated as:

$$
\Gamma_S = w \left( \frac{n_e}{10^{16}\ \text{cm}^{-3}} \right)^\alpha,
$$

where:
- $w$ – Stark width parameter at a reference density
- $\alpha$ – scaling exponent (often close to 1).

Total Lorentzian FWHM:

$$
\Gamma_L = \Gamma_S + \Gamma_{\text{other}},
$$

with $\Gamma_{\text{other}}$ including van der Waals, natural, and instrumental contributions.

CF‑LIBS will implement inversion routines that, given a measured line profile $I_\lambda(\lambda)$, fit $\Gamma_S$ and thereby infer $n_e$.

---

### Instrument Response and Detector Modeling

To bridge **theoretical spectra** and **measured counts**, CF‑LIBS will include:

1. **Spectral response** $R(\lambda)$ for optics + detector:
   $$
   I_\lambda^\text{det}(\lambda) = R(\lambda) \cdot I_\lambda^\text{plasma}(\lambda).
   $$

2. **Instrument function / spectral resolution**, modeled as a convolution:
   $$
   I_\lambda^\text{obs}(\lambda) =
   (I_\lambda^\text{det} * G_\text{instr})(\lambda),
   $$
   where $G_\text{instr}$ is typically Gaussian with FWHM $\Delta\lambda_\text{instr}$.

3. **Detector sampling and noise**:
   - Pixel integration over wavelength bins $\Delta\lambda_\text{pix}$
   - Shot noise (Poisson), read noise (Gaussian), background offset.

The final synthetic data model for pixel $k$ centered at $\lambda_k$ is:

$$
C_k \sim \text{Poisson}\big( 
    t_\text{exp} \cdot A_\text{eff}
    \int_{\lambda_k - \Delta\lambda_k/2}^{\lambda_k + \Delta\lambda_k/2}
      I_\lambda^\text{obs}(\lambda)\,\mathrm{d}\lambda
  \big)
  + \mathcal{N}(\mu_\text{read}, \sigma_\text{read}^2),
$$

where $t_\text{exp}$ is exposure time and $A_\text{eff}$ an effective collection area.

---

## Architecture

### Core Packages

Planned high‑level package layout (all in Python):

- `cflibs.core`
  - Low‑level numerical kernels (line profiles, Voigt, partition functions)
  - Physical constants, units
  - Interpolation and quadrature utilities

- `cflibs.atomic`
  - Data structures for:
    - Energy levels
    - Transition probabilities
    - Stark and broadening parameters
  - Interfaces to external databases (e.g., NIST ASD, Kurucz, etc.) via pluggable loaders

- `cflibs.plasma`
  - Plasma state definitions and validation
  - LTE / partial‑LTE solvers
  - Saha–Boltzmann solvers & constraint enforcement
  - Multi‑zone plasma models

- `cflibs.radiation`
  - Line emissivity and opacity calculations
  - Radiative transfer solvers (single zone, multi‑zone)
  - Continuum emission (Bremsstrahlung, recombination, etc. – later phases)

- `cflibs.instrument`
  - Instrument response functions
  - Detector models
  - Wavelength calibration & rebinning tools

- `cflibs.inversion`
  - Least‑squares, gradient‑based, and global optimization routines
  - Bayesian inversion (e.g., MCMC, nested sampling; via external libs)
  - Diagnostics: Boltzmann plots, Stark fits, line ratio analysis

- `cflibs.io`
  - Standardized file formats for:
    - Spectra (raw, calibrated)
    - Plasma configurations
    - Atomic data snapshots
  - YAML/JSON config loading

- `cflibs.cli`
  - Command‑line tools for:
    - Forward modeling given a config file
    - Inversion of measured spectra
    - Batch processing and pipelines

---

### Data & Configuration

CF‑LIBS will use **declarative configuration** for reproducibility:

- YAML/JSON configs describing:
  - Plasma model (zones, assumptions)
  - Atomic data sets (versioned)
  - Instrument model
  - Inversion priors and algorithm settings

Example (planned):

```yaml
plasma:
  model: single_zone_lte
  Te: 10000        # K
  ne: 1.0e17       # cm^-3
  species:
    - element: Fe
      number_density: 1.0e15  # cm^-3
    - element: H
      number_density: 1.0e16  # cm^-3

instrument:
  response_curve: response_curves/my_spectrometer.csv
  resolution_fwhm_nm: 0.05

spectrum:
  lambda_min_nm: 200.0
  lambda_max_nm: 800.0
  delta_lambda_nm: 0.005
```

---

### Performance and Numerical Design

CF‑LIBS is intended to be *production‑grade*, with:

- Vectorized numerics (NumPy / JAX / CuPy backends, depending on constraints)
- Optional **just‑in‑time compilation** for hot loops (e.g., Voigt evaluation, Saha‑Boltzmann solvers)
- Parallelization over:
  - Wavelength chunks
  - Plasma zones
  - Parameter sets (for scanning and Monte‑Carlo)

Key numerical decisions will focus on:

- Stable and fast Voigt approximations
- Robust root‑finding / nonlinear solvers for equilibrium
- Efficient convolution and rebinning on large spectral grids

---

## Usage Examples (Planned API)

> Note: These are *design sketches* of the intended API. Actual function and class names may evolve.

### Forward Model: Single‑Zone LTE Plasma

```python
from cflibs.plasma import SingleZoneLTEPlasma
from cflibs.atomic import AtomicDatabase
from cflibs.radiation import SpectrumModel
from cflibs.instrument import InstrumentModel

# 1. Load atomic data
db = AtomicDatabase.from_nist("data/nist_lines_Fe_H.sqlite")

# 2. Define plasma state
plasma = SingleZoneLTEPlasma(
    Te=10000.0,   # K
    ne=1e17,      # cm^-3
    species={
        "Fe": 1e15,
        "H":  1e16,
    },
    pressure=1.0,  # atm, optional
)

# 3. Define instrument
instr = InstrumentModel.from_file("configs/instrument_my_spectrometer.yaml")

# 4. Build spectrum model
model = SpectrumModel(
    plasma=plasma,
    atomic_db=db,
    instrument=instr,
    lambda_min=200.0,
    lambda_max=800.0,
    delta_lambda=0.01,
)

lambda_grid, intensity = model.compute_spectrum()
```

### Inversion: Estimate Te and ne from a Measured Spectrum

```python
from cflibs.inversion import LTEInversion
from cflibs.io import load_spectrum

wavelengths, counts = load_spectrum("data/shot_001.csv")

inv = LTEInversion(
    atomic_db=db,
    instrument=instr,
    # Priors / bounds
    Te_bounds=(6000.0, 25000.0),
    ne_bounds=(1e15, 1e18),
    species={"Fe": "unknown", "H": "unknown"},
)

result = inv.fit(wavelengths, counts)

print("Te =", result.Te, "+/-", result.Te_unc)
print("ne =", result.ne, "+/-", result.ne_unc)
print("species densities:", result.species)
```

---

## Development Roadmap

This section details the **stepwise path** from the current empty scaffold to a **full CF‑LIBS physics and inversion engine**.

### Phase 0 – Scaffold & Core Utilities

**Goal:** Establish a stable base for development.

- [ ] Basic package structure (`cflibs.*` namespace) and packaging
- [ ] Constants and units module
- [ ] Minimal logger and configuration system
- [ ] CI pipeline (tests, lint, type checking)
- [ ] Documentation skeleton (API docs, user guide structure)
- [ ] Simple CLI entry point stub

**Deliverable:** Importable `cflibs` package with tests and documentation infrastructure.

---

### Phase 1 – Minimal Viable Physics Engine

**Goal:** Get a **first working forward model** for a simple LTE, optically thin, single‑zone plasma.

- [ ] Implement atomic level and transition representations
- [ ] Provide a small, bundled atomic dataset for testing
- [ ] Saha–Boltzmann solver (single element, multiple ion stages)
- [ ] Line emissivity with Gaussian broadening only
- [ ] Homogeneous, optically thin slab intensity
  $$
  I_\lambda(\lambda) = \epsilon_\lambda(\lambda) L
  $$
- [ ] Simple instrument convolution (Gaussian kernel)
- [ ] Basic forward‑model API & one CLI command

**Deliverable:** Ability to generate synthetic LIBS spectra from a YAML config file.

---

### Phase 2 – Production‑Grade CF‑LIBS Engine

**Goal:** Evolve the core into a **reliable, extensible physics engine**.

- [ ] Multi‑species, multi‑ion Saha–Boltzmann equilibrium
- [ ] Partition functions with interpolation tables over $T_e$
- [ ] Full line profile model:
  - [ ] Voigt profile (Doppler + Lorentzian)
  - [ ] Line‑by‑line Stark parameters and density scaling
- [ ] Electron‑density‑dependent Stark broadening
- [ ] Instrument response curves from measured data
- [ ] Modular, pluggable atomic data loaders
- [ ] Optimization of spectrum calculations (vectorization, JIT as appropriate)
- [ ] Extensive validation against benchmark spectra from literature or experiment

**Deliverable:** Production‑ready forward model suitable for research‑grade studies.

---

### Phase 3 – Advanced Inversion & Uncertainty

**Goal:** Implement **calibration‑free LIBS inversion** and robust diagnostics.

- [ ] Boltzmann plot generation and fitting
- [ ] Stark line fitting for $n_e$
- [ ] Multi‑parameter nonlinear least‑squares inversion (e.g., Levenberg–Marquardt)
- [ ] Support for multi‑zone parametric models (e.g., two‑temperature zones)
- [ ] Bayesian inversion layer:
  - [ ] MCMC / nested sampling wrappers
  - [ ] Priors on plasma parameters
  - [ ] Posterior diagnostics and credible intervals
- [ ] Propagation of atomic data uncertainties (where available)
- [ ] Tools for sensitivity analysis (e.g., derivatives of spectra wrt. parameters)

**Deliverable:** A robust CF‑LIBS inversion suite with uncertainty quantification.

---

### Phase 4 – Ecosystem & Integrations

**Goal:** Make CF‑LIBS easy to adopt in diverse workflows.

- [ ] Jupyter‑friendly visualization utilities
- [ ] Export tools for common data formats and analysis suites
- [ ] Hooks for integration with:
  - [ ] Experimental control systems (for on‑the‑fly modeling)
  - [ ] Other spectroscopy toolkits (e.g., atomic data pipelines)
- [ ] High‑level recipes and example notebooks:
  - [ ] End‑to‑end CF‑LIBS analysis of a benchmark dataset
  - [ ] Instrument design studies (e.g., resolution vs. S/N tradeoffs)
- [ ] Performance tuning for large batch simulations and parameter scans

**Deliverable:** A mature, documented, and integrable LIBS modeling ecosystem.

---

## Contributing

CF‑LIBS is intended as a **serious, physics‑driven** open‑source project. Contributions are welcome in:

- Physics modeling (new processes, better approximations)
- Atomic data curation and validation
- Numerical methods and performance engineering
- Documentation, examples, and educational material

Planned guidelines (to be formalized):

- Use type hints and docstrings for all public APIs
- Include tests for any new core functionality
- Maintain physical units and conventions consistently
- Favor readability in physics‑heavy code; document equations and assumptions

---

## License

The exact license is not yet finalized. Until a `LICENSE` file is added, assume that this repository is **all‑rights‑reserved** and contact the maintainer for usage permissions.

Once finalized, the license will be documented here and in the root `LICENSE` file.
