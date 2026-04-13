# Phase 0 Implementation Status

This document tracks the implementation of Phase 0 - Scaffold & Core Utilities.

## Completed ✅

- [x] Basic package structure (`cflibs.*` namespace)
  - Created all core package directories:
    - `cflibs.core` - Constants, units, config, logging
    - `cflibs.atomic` - Atomic data structures (placeholder)
    - `cflibs.plasma` - Plasma state and solvers (placeholder)
    - `cflibs.radiation` - Radiation calculations (placeholder)
    - `cflibs.instrument` - Instrument modeling (placeholder)
    - `cflibs.inversion` - Inversion routines (placeholder)
    - `cflibs.io` - I/O utilities (placeholder)
    - `cflibs.cli` - Command-line interface

- [x] Constants and units module
  - Physical constants (KB, Planck, electron mass, etc.)
  - Unit conversion utilities (temperature, density, wavelength, energy)
  - Conversion factors and standard conditions

- [x] Minimal logger and configuration system
  - Logging configuration utilities
  - YAML/JSON config loading and validation
  - Configuration structure validation

- [x] Basic CLI entry point
  - Command structure for forward modeling and inversion
  - Logging integration
  - Helpful error messages

- [x] Packaging infrastructure
  - `setup.py` with proper dependencies
  - `pyproject.toml` for modern Python packaging
  - `.gitignore` for Python projects
  - `MANIFEST.in` for source distribution

- [x] Test infrastructure
  - Basic test structure
  - Tests for constants and units modules
  - Pytest configuration

## In Progress / Next Steps

- [ ] CI pipeline setup (GitHub Actions, etc.)
- [ ] Documentation skeleton (Sphinx/API docs)
- [ ] Type hints throughout (partial)
- [ ] More comprehensive test coverage

## Usage

### Installation

```bash
# Development installation
pip install -e .

# With optional dependencies
pip install -e ".[jax,hdf5]"

# Development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from cflibs.core import constants, units

# Use physical constants
kb_ev = constants.KB_EV
h_planck = constants.H_PLANCK

# Convert units
T_ev = units.convert_temperature(10000, 'K', 'eV')
n_m3 = units.convert_density(1e17, 'cm^-3', 'm^-3')
```

### CLI Usage

```bash
# Forward modeling (Phase 1)
cflibs forward config.yaml

# Inversion (Phase 3)
cflibs invert spectrum.csv --config inversion_config.yaml
```

## Next Phase: Phase 1 - Minimal Viable Physics Engine

The next phase will implement:
- Atomic level and transition representations
- Saha-Boltzmann solver
- Line emissivity with Gaussian broadening
- Basic forward-model API

