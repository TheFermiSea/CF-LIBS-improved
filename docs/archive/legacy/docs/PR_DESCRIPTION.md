# CF-LIBS: Phase 0 & Phase 1 Implementation + Echellogram Processing Upgrade

## 🎯 Overview

This PR implements the foundational infrastructure for CF-LIBS (Computational Framework for Laser-Induced Breakdown Spectroscopy), establishing a production-grade Python library for forward modeling, inversion, and analysis of LIBS plasmas. The implementation includes:

- **Phase 0**: Complete package scaffold with core utilities
- **Phase 1**: Minimal Viable Physics Engine for forward modeling
- **Echellogram Processing**: Upgraded 2D spectral image extraction

## 📦 What's Included

### Phase 0: Scaffold & Core Utilities ✅

#### Package Structure
- Complete `cflibs` package namespace with modular architecture:
  - `cflibs.core` - Constants, units, config, logging
  - `cflibs.atomic` - Atomic data structures and database interface
  - `cflibs.plasma` - Plasma state and Saha-Boltzmann solvers
  - `cflibs.radiation` - Line emissivity and spectrum modeling
  - `cflibs.instrument` - Instrument response and echellogram extraction
  - `cflibs.inversion` - Inversion routines (placeholder for Phase 3)
  - `cflibs.io` - I/O utilities for spectra and configs
  - `cflibs.cli` - Command-line interface

#### Core Modules
- **Constants Module** (`cflibs.core.constants`):
  - Physical constants (KB, Planck, electron properties)
  - Atomic physics constants (Rydberg, Bohr radius)
  - Conversion factors (eV↔J, cm⁻¹↔eV, etc.)
  - Plasma physics constants (Saha equation, McWhirter criterion)

- **Units Module** (`cflibs.core.units`):
  - Temperature conversions (K ↔ eV ↔ Celsius)
  - Density conversions (m⁻³ ↔ cm⁻³)
  - Wavelength conversions (m ↔ nm ↔ μm ↔ Angstrom ↔ cm⁻¹)
  - Energy conversions (J ↔ eV ↔ cm⁻¹)
  - Full NumPy array support

- **Configuration System** (`cflibs.core.config`):
  - YAML/JSON config loading with validation
  - Plasma and instrument config validation
  - Graceful handling of optional dependencies

- **Logging System** (`cflibs.core.logging_config`):
  - Standardized logging setup
  - Module-level loggers
  - Configurable log levels

#### Infrastructure
- **Packaging**: `setup.py` and `pyproject.toml` with proper dependencies
- **CLI**: Basic command structure for forward modeling and inversion
- **Testing**: Pytest infrastructure with initial test suite
- **Documentation**: README files and API structure

### Phase 1: Minimal Viable Physics Engine ✅

#### Atomic Data Structures (`cflibs.atomic`)
- `EnergyLevel`: Data structure for atomic energy levels
- `Transition`: Data structure for atomic transitions with Einstein A coefficients
- `SpeciesPhysics`: Ionization potential and species properties
- `AtomicDatabase`: SQLite database interface for loading atomic data

#### Plasma Physics (`cflibs.plasma`)
- `PlasmaState`: Base plasma state representation
- `SingleZoneLTEPlasma`: Single-zone LTE plasma model with validation
- `SahaBoltzmannSolver`: Complete solver for:
  - Saha equation for ionization balance
  - Boltzmann distribution for level populations
  - Partition function calculations from energy levels
  - Multi-species, multi-ion stage support

#### Radiation Calculations (`cflibs.radiation`)
- **Line Emissivity**: Calculate spectral emissivity from transitions and populations
- **Gaussian Broadening**: Doppler broadening and line profiles
- **SpectrumModel**: Complete forward model integrating all components

#### Instrument Modeling (`cflibs.instrument`)
- `InstrumentModel`: Instrument response and resolution modeling
- `apply_instrument_function`: Gaussian convolution for instrument function
- Response curve support

#### Forward Model API
- `SpectrumModel`: High-level API that:
  1. Takes plasma state, atomic database, and instrument model
  2. Solves Saha-Boltzmann equations
  3. Calculates line emissivity
  4. Applies Gaussian broadening
  5. Converts to intensity (optically thin: I = ε × L)
  6. Applies instrument response and convolution

#### CLI Integration
- `cflibs forward` command fully functional
- YAML/JSON config file support
- Output to file or stdout

### Echellogram Processing Upgrade ✅

#### Enhanced Echelle Extractor (`cflibs.instrument.echelle`)
- **Upgraded from standalone script to integrated module**
- Type hints throughout
- Proper error handling and logging
- Multiple merge methods: `weighted_average`, `simple_average`, `max`
- Custom extraction window size
- Background subtraction option
- Individual order extraction capability
- Calibration save/load methods
- Improved mock calibration generation

#### Documentation
- Comprehensive guide in `docs/Echellogram_Processing_Guide.md`
- Algorithm explanation with equations
- Usage examples and integration guide
- Calibration procedure documentation

## 🔬 Physics Implementation

### Saha-Boltzmann Solver

Implements the complete LTE system:

1. **Saha Equation** for ionization balance:
   $$
   \frac{n_{z+1} n_e}{n_z} = \frac{SAHA\_CONST}{n_e} T^{1.5} \exp\left(-\frac{IP}{kT}\right)
   $$

2. **Boltzmann Distribution** for level populations:
   $$
   n_i = n_{stage} \frac{g_i}{U(T)} \exp\left(-\frac{E_i}{kT}\right)
   $$

3. **Partition Function** calculated from energy levels:
   $$
   U(T) = \sum_i g_i \exp\left(-\frac{E_i}{kT}\right)
   $$

### Line Emissivity

For each transition:
$$
\epsilon_\lambda = \frac{hc}{4\pi\lambda} A_{ki} n_k
$$

### Optically Thin Approximation

For homogeneous slab:
$$
I_\lambda = \epsilon_\lambda \times L
$$

## 📝 Usage Examples

### Basic Forward Modeling

```python
from cflibs.plasma import SingleZoneLTEPlasma
from cflibs.atomic import AtomicDatabase
from cflibs.instrument import InstrumentModel
from cflibs.radiation import SpectrumModel

# Load atomic database
atomic_db = AtomicDatabase("libs_production.db")

# Create plasma state
plasma = SingleZoneLTEPlasma(
    T_e=10000.0,  # K
    n_e=1e17,     # cm^-3
    species={"Fe": 1e15, "H": 1e16}
)

# Create instrument model
instrument = InstrumentModel(resolution_fwhm_nm=0.05)

# Create spectrum model
model = SpectrumModel(
    plasma=plasma,
    atomic_db=atomic_db,
    instrument=instrument,
    lambda_min=200.0,
    lambda_max=800.0,
    delta_lambda=0.01
)

# Compute spectrum
wavelength, intensity = model.compute_spectrum()
```

### CLI Usage

```bash
# Generate spectrum from config file
cflibs forward examples/config_example.yaml --output spectrum.csv
```

### Echellogram Extraction

```python
from cflibs.instrument import EchelleExtractor
from cflibs.io import save_spectrum

# Extract from 2D echellogram
extractor = EchelleExtractor('calibration.json')
wavelength, intensity = extractor.extract_spectrum(image_2d)

# Save result
save_spectrum('spectrum.csv', wavelength, intensity)
```

## 🧪 Testing

- **Test Infrastructure**: Pytest setup with test structure
- **Unit Tests**: Tests for constants, units, echelle extraction
- **Test Coverage**: Core functionality covered
- **CI Ready**: Structure ready for CI/CD integration

## 📚 Documentation

- **README.md**: Comprehensive project documentation
- **README_PHASE0.md**: Phase 0 implementation details
- **README_PHASE1.md**: Phase 1 implementation details
- **docs/Echellogram_Processing_Guide.md**: Echellogram extraction guide
- **IMPLEMENTATION_SUMMARY.md**: Overall implementation summary
- **ECHELLE_UPGRADE_SUMMARY.md**: Echellogram upgrade details
- **Example Configs**: YAML configuration examples
- **Code Documentation**: Comprehensive docstrings throughout

## 🔧 Configuration

Example configuration file (`examples/config_example.yaml`):

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

## 📊 Files Changed

### New Files Created
- `cflibs/__init__.py` - Package initialization
- `cflibs/core/` - Core utilities (constants, units, config, logging)
- `cflibs/atomic/` - Atomic data structures and database
- `cflibs/plasma/` - Plasma state and solvers
- `cflibs/radiation/` - Radiation calculations
- `cflibs/instrument/` - Instrument modeling (including echelle)
- `cflibs/inversion/` - Inversion placeholders
- `cflibs/io/` - I/O utilities
- `cflibs/cli/` - Command-line interface
- `tests/` - Test suite
- `examples/` - Example configs and scripts
- `docs/` - Documentation
- `setup.py`, `pyproject.toml`, `MANIFEST.in` - Packaging files

### Files Removed
- `cflibs/instrument/echelle-extractor.py` (replaced by `echelle.py`)
- `cflibs/Echellogram Processing Guide.md` (moved to `docs/`)

## ⚠️ Breaking Changes

**None** - This is the initial implementation. Existing scripts in the root directory are preserved and can be gradually migrated.

## 🚀 Dependencies

### Required
- `numpy >= 1.20.0`
- `scipy >= 1.7.0`
- `pandas >= 1.3.0`
- `pyyaml >= 5.4.0`

### Optional
- `jax >= 0.4.0` (for future manifold generation)
- `h5py >= 3.0.0` (for HDF5 support)
- `pytest >= 7.0.0` (for testing)

## ✅ Checklist

- [x] Phase 0: Package structure and core utilities
- [x] Phase 0: Constants and units module
- [x] Phase 0: Logging and configuration system
- [x] Phase 0: CLI entry point
- [x] Phase 0: Test infrastructure
- [x] Phase 1: Atomic data structures
- [x] Phase 1: Saha-Boltzmann solver
- [x] Phase 1: Line emissivity calculations
- [x] Phase 1: Gaussian broadening
- [x] Phase 1: Forward model API
- [x] Phase 1: CLI integration
- [x] Echellogram processing upgrade
- [x] Documentation
- [x] Examples and test cases
- [x] Code linting passes

## 🔮 Future Work (Not in This PR)

### Phase 2: Production-Grade CF-LIBS Engine
- [ ] Voigt profiles (Doppler + Lorentzian)
- [ ] Stark broadening (electron density dependent)
- [ ] Improved partition functions with interpolation tables
- [ ] Multi-zone plasma models
- [ ] Performance optimizations (JAX, vectorization)

### Phase 3: Advanced Inversion & Uncertainty
- [ ] Boltzmann plot generation and fitting
- [ ] Stark line fitting for electron density
- [ ] Nonlinear least-squares inversion
- [ ] Bayesian inversion (MCMC, nested sampling)
- [ ] Uncertainty quantification

### Phase 4: Ecosystem & Integrations
- [ ] Jupyter visualization utilities
- [ ] Export tools for common formats
- [ ] Integration hooks for experimental systems
- [ ] High-level recipes and notebooks

## 📈 Performance

- **Forward Modeling**: Can generate spectra in seconds for typical configurations
- **Echellogram Extraction**: Efficient extraction with configurable parameters
- **Memory Usage**: Reasonable for typical use cases (can be optimized in Phase 2)

## 🎓 Code Quality

- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Error handling with descriptive messages
- ✅ Logging integration
- ✅ No linter errors
- ✅ Follows Python best practices
- ✅ Modular, testable design

## 📖 References

- CF-LIBS physics model documented in `README.md`
- Industrial requirements in `DEVELOPMENT_GUIDE_V1.md`
- High-throughput framework in `HIGH_THROUGHPUT_FRAMEWORK.md`

## 👥 Contributors

This PR establishes the foundation for CF-LIBS development. Future contributions welcome for:
- Physics modeling improvements
- Atomic data curation
- Numerical methods and performance
- Documentation and examples

---

**Status**: ✅ Ready for Review

This PR implements Phase 0 and Phase 1 as specified in the README, providing a solid foundation for the CF-LIBS library. The code is production-ready, well-documented, and follows best practices.

