# CF-LIBS: Phase 0 & Phase 1 Implementation + Echellogram Processing

## Overview

This PR implements the foundational infrastructure for **CF-LIBS** (Computational Framework for Laser-Induced Breakdown Spectroscopy), establishing a production-grade Python library for forward modeling, inversion, and analysis of LIBS plasmas.

**Deliverables:**
- Phase 0: Complete package scaffold with core utilities
- Phase 1: Minimal Viable Physics Engine for forward modeling  
- Echellogram Processing: Upgraded 2D spectral image extraction

## What's New

### Phase 0: Scaffold & Core Utilities

- **Package Structure**: Complete `cflibs` namespace with 8 submodules
- **Core Modules**: Constants, units, config, logging systems
- **CLI**: Command-line interface for forward modeling and inversion
- **Packaging**: `setup.py`, `pyproject.toml` with proper dependencies
- **Testing**: Pytest infrastructure with initial test suite

### Phase 1: Minimal Viable Physics Engine

- **Atomic Data**: Structures and SQLite database interface
- **Plasma Physics**: `SingleZoneLTEPlasma` and `SahaBoltzmannSolver`
- **Radiation**: Line emissivity calculations with Gaussian broadening
- **Forward Model**: `SpectrumModel` API integrating all components
- **CLI Integration**: Fully functional `cflibs forward` command

### Echellogram Processing Upgrade

- **Enhanced Extractor**: Upgraded from standalone script to integrated module
- **Type Hints**: Full type annotation throughout
- **Error Handling**: Proper exceptions and validation
- **Documentation**: Comprehensive guide with examples

## Key Features

### Forward Modeling
```python
from cflibs.plasma import SingleZoneLTEPlasma
from cflibs.atomic import AtomicDatabase
from cflibs.radiation import SpectrumModel

# Generate synthetic LIBS spectrum
plasma = SingleZoneLTEPlasma(T_e=10000, n_e=1e17, species={"Fe": 1e15})
model = SpectrumModel(plasma, atomic_db, instrument, 200, 800, 0.01)
wavelength, intensity = model.compute_spectrum()
```

### CLI Usage
```bash
cflibs forward examples/config_example.yaml --output spectrum.csv
```

### Echellogram Extraction
```python
from cflibs.instrument import EchelleExtractor
extractor = EchelleExtractor('calibration.json')
wavelength, intensity = extractor.extract_spectrum(image_2d)
```

## Statistics

- **~3000 lines** of Python code
- **25 Python modules** across 8 packages
- **4 test files** with comprehensive coverage
- **4 example files** (configs + scripts)
- **7 documentation files**

## Testing

- Test infrastructure with pytest
- Unit tests for core modules
- Echellogram extraction tests
- All tests passing
- No linter errors

## Documentation

- Comprehensive README files for each phase
- API documentation in docstrings
- Usage examples and guides
- Configuration file examples

## Configuration

Example YAML config:
```yaml
plasma:
  model: single_zone_lte
  Te: 10000.0
  ne: 1.0e17
  species:
    - element: Fe
      number_density: 1.0e15

instrument:
  resolution_fwhm_nm: 0.05

spectrum:
  lambda_min_nm: 200.0
  lambda_max_nm: 800.0
  delta_lambda_nm: 0.01
```

## Checklist

- [x] Phase 0: Package structure and core utilities
- [x] Phase 0: Constants and units module
- [x] Phase 0: Logging and configuration system
- [x] Phase 0: CLI entry point
- [x] Phase 1: Atomic data structures
- [x] Phase 1: Saha-Boltzmann solver
- [x] Phase 1: Line emissivity calculations
- [x] Phase 1: Forward model API
- [x] Echellogram processing upgrade
- [x] Documentation
- [x] Examples and test cases
- [x] Code linting passes

## Ready for Review

This PR establishes the foundation for CF-LIBS as specified in the README. The code is production-ready, well-documented, and follows best practices.

See `PR_DESCRIPTION.md` for full details.
