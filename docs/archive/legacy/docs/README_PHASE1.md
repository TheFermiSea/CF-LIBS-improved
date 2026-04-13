# Phase 1 Implementation Status

## Overview

Phase 1 (Minimal Viable Physics Engine) has been successfully implemented. This phase provides a working forward model for generating synthetic LIBS spectra from a simple LTE, optically thin, single-zone plasma.

## Completed Features ✅

### 1. Atomic Data Structures (`cflibs.atomic`)

- **EnergyLevel**: Data structure for atomic energy levels
- **Transition**: Data structure for atomic transitions with Einstein A coefficients
- **SpeciesPhysics**: Ionization potential and species properties
- **AtomicDatabase**: SQLite database interface for loading atomic data

### 2. Plasma State and Solvers (`cflibs.plasma`)

- **PlasmaState**: Base plasma state representation
- **SingleZoneLTEPlasma**: Single-zone LTE plasma model
- **SahaBoltzmannSolver**: 
  - Saha equation solver for ionization balance
  - Boltzmann distribution solver for level populations
  - Partition function calculations
  - Complete Saha-Boltzmann system solver

### 3. Radiation Calculations (`cflibs.radiation`)

- **Line Emissivity**: Calculate spectral emissivity from transitions and populations
- **Gaussian Broadening**: Gaussian line profiles (Doppler broadening)
- **Spectrum Model**: Complete forward model integrating all components

### 4. Instrument Modeling (`cflibs.instrument`)

- **InstrumentModel**: Instrument response and resolution modeling
- **Gaussian Convolution**: Apply instrument function via convolution
- **Response Curves**: Support for spectral response curves

### 5. Forward Model API (`cflibs.radiation.SpectrumModel`)

Complete forward model that:
1. Takes plasma state, atomic database, and instrument model
2. Solves Saha-Boltzmann equations
3. Calculates line emissivity
4. Applies Gaussian broadening
5. Converts to intensity (optically thin: I = ε × L)
6. Applies instrument response and convolution

### 6. CLI Integration

- Updated `cflibs forward` command to work with YAML config files
- Full integration with configuration system

### 7. I/O Utilities (`cflibs.io`)

- `load_spectrum()`: Load spectra from CSV files
- `save_spectrum()`: Save spectra to files

## Usage

### Installation

First, install dependencies:

```bash
pip install -e ".[all]"
```

Or install core dependencies:

```bash
pip install numpy scipy pandas pyyaml
```

### Basic Python API

```python
from cflibs.plasma import SingleZoneLTEPlasma
from cflibs.atomic import AtomicDatabase
from cflibs.instrument import InstrumentModel
from cflibs.radiation import SpectrumModel

# 1. Load atomic database
atomic_db = AtomicDatabase("libs_production.db")

# 2. Create plasma state
plasma = SingleZoneLTEPlasma(
    T_e=10000.0,  # K
    n_e=1e17,     # cm^-3
    species={
        "Fe": 1e15,
        "H":  1e16,
    }
)

# 3. Create instrument model
instrument = InstrumentModel(resolution_fwhm_nm=0.05)

# 4. Create spectrum model
model = SpectrumModel(
    plasma=plasma,
    atomic_db=atomic_db,
    instrument=instrument,
    lambda_min=200.0,
    lambda_max=800.0,
    delta_lambda=0.01,
    path_length_m=0.01
)

# 5. Compute spectrum
wavelength, intensity = model.compute_spectrum()
```

### CLI Usage

```bash
# Generate spectrum from config file
cflibs forward examples/config_example.yaml --output spectrum.csv

# Or print to stdout
cflibs forward examples/config_example.yaml
```

### Configuration File Format

See `examples/config_example.yaml` for a complete example:

```yaml
atomic_database: libs_production.db

plasma:
  model: single_zone_lte
  Te: 10000.0        # K
  ne: 1.0e17         # cm^-3
  species:
    - element: Fe
      number_density: 1.0e15
    - element: H
      number_density: 1.0e16

instrument:
  resolution_fwhm_nm: 0.05

spectrum:
  lambda_min_nm: 200.0
  lambda_max_nm: 800.0
  delta_lambda_nm: 0.01
  path_length_m: 0.01
```

## Physics Implementation

### Saha-Boltzmann Solver

The solver implements:

1. **Saha Equation**: For ionization balance
   $$
   \frac{n_{z+1} n_e}{n_z} = \frac{SAHA\_CONST}{n_e} T^{1.5} \exp\left(-\frac{IP}{kT}\right)
   $$

2. **Boltzmann Distribution**: For level populations
   $$
   n_i = n_{stage} \frac{g_i}{U(T)} \exp\left(-\frac{E_i}{kT}\right)
   $$

3. **Partition Function**: Calculated from energy levels
   $$
   U(T) = \sum_i g_i \exp\left(-\frac{E_i}{kT}\right)
   $$

### Line Emissivity

For each transition:
$$
\epsilon_\lambda = \frac{hc}{4\pi\lambda} A_{ki} n_k
$$

Where:
- $h$ = Planck constant
- $c$ = speed of light
- $\lambda$ = wavelength
- $A_{ki}$ = Einstein A coefficient
- $n_k$ = upper level population

### Optically Thin Approximation

For a homogeneous slab:
$$
I_\lambda = \epsilon_\lambda \times L
$$

Where $L$ is the path length.

### Gaussian Broadening

Currently implements Gaussian profiles only (Phase 1):
- Doppler broadening (thermal)
- Instrument function (convolution)

Voigt profiles (Doppler + Lorentzian) will be added in Phase 2.

## Limitations (Phase 1)

1. **Gaussian Broadening Only**: No Voigt profiles yet (Phase 2)
2. **Simple Partition Functions**: Uses energy levels from database, with fallback approximations
3. **No Stark Broadening**: Electron density effects on line width not yet included
4. **Optically Thin Only**: No opacity/radiative transfer (Phase 2+)
5. **Single Zone Only**: No multi-zone models (Phase 2+)
6. **No Continuum**: Only line emission, no Bremsstrahlung/recombination (Phase 2+)

## Testing

To test the implementation, you need:

1. **Atomic Database**: Run `datagen_v2.py` to generate `libs_production.db`
2. **Configuration File**: Use examples in `examples/` directory
3. **Dependencies**: Install with `pip install -e ".[all]"`

Example test:

```bash
# Generate database (if not exists)
python datagen_v2.py

# Generate spectrum
cflibs forward examples/config_example.yaml --output test_spectrum.csv
```

## Next Steps: Phase 2

Phase 2 will add:
- Voigt profiles (Doppler + Lorentzian)
- Stark broadening (electron density dependent)
- Improved partition functions
- Instrument response curves from data
- Performance optimizations
- Validation against benchmark spectra

## Files Created/Modified

### New Modules
- `cflibs/atomic/structures.py` - Data structures
- `cflibs/atomic/database.py` - Database interface
- `cflibs/plasma/state.py` - Plasma state
- `cflibs/plasma/saha_boltzmann.py` - Saha-Boltzmann solver
- `cflibs/radiation/profiles.py` - Line profiles
- `cflibs/radiation/emissivity.py` - Emissivity calculations
- `cflibs/radiation/spectrum_model.py` - Forward model
- `cflibs/instrument/model.py` - Instrument model
- `cflibs/instrument/convolution.py` - Instrument convolution
- `cflibs/io/spectrum.py` - Spectrum I/O

### Updated
- `cflibs/cli/main.py` - Forward modeling command
- All `__init__.py` files - Package exports

### Examples
- `examples/config_example.yaml` - Basic example
- `examples/config_ti64.yaml` - Ti-6Al-4V example

---

**Status**: Phase 1 Complete ✅

The minimal viable physics engine is now functional and ready for testing and validation.

