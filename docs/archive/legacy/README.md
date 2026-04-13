# Legacy Scripts

This directory contains legacy scripts from the early development of CF-LIBS. These scripts have been superseded by the modular package structure in `cflibs/`, but are preserved for reference and gradual migration.

## Scripts

### `cf-libs-analyzer.py`
**Status**: Legacy - Functionality migrated to `cflibs.inversion` (Phase 3)

Early implementation of CF-LIBS analysis with:
- AirPLS baseline correction
- Peak finding and element identification
- One-point calibration solver

**Migration Path**: 
- AirPLS → `cflibs.inversion.baseline` (planned)
- Element identification → `cflibs.atomic` + `cflibs.inversion` (planned)
- Calibration solver → `cflibs.inversion` (planned)

### `saha-eggert.py`
**Status**: Legacy - Functionality migrated to `cflibs.plasma.saha_boltzmann`

Early implementation of Saha-Boltzmann solver with:
- Ionization balance calculation
- Snapshot spectrum generation
- Time-integrated spectrum generation

**Migration Path**: 
- Core functionality → `cflibs.plasma.SahaBoltzmannSolver`
- Spectrum generation → `cflibs.radiation.SpectrumModel`

### `line-identifier.py`
**Status**: Legacy - Functionality partially migrated to `cflibs.instrument.echelle`

Early implementation of line identification with:
- NNLS deconvolution
- AirPLS baseline correction
- Line matching

**Migration Path**:
- NNLS deconvolution → `cflibs.inversion.deconvolution` (planned)
- Baseline correction → `cflibs.inversion.baseline` (planned)

### `datagen.py` and `datagen_v2.py`
**Status**: Active - Essential database generation scripts

**IMPORTANT**: These scripts are **required** for generating the atomic database (`libs_production.db`) that CF-LIBS depends on. They are **NOT** legacy code.

Scripts for generating atomic database from NIST:
- `datagen.py`: Original version
- `datagen_v2.py`: Enhanced version with energy levels and filtering (recommended)

**Usage**:
```bash
# Generate database
python datagen_v2.py

# Or use CLI command
cflibs generate-db
```

**Note**: These scripts have been moved back to the root directory as they are essential utilities.

### `manifold-generator.py`
**Status**: Active - Migrated to `cflibs.manifold`

**IMPORTANT**: This script has been integrated into the `cflibs.manifold` module.

The manifold generator is **essential** for high-throughput inference and should be
used before implementing inversion algorithms.

**Migration**: 
- Core functionality → `cflibs.manifold.ManifoldGenerator`
- Configuration → `cflibs.manifold.ManifoldConfig`
- Loading/querying → `cflibs.manifold.ManifoldLoader`

**Usage**:
```bash
# Generate manifold
cflibs generate-manifold examples/manifold_config_example.yaml
```

See `docs/Manifold_Generation_Guide.md` for details.

## Usage

These scripts can still be run directly, but for new development, please use the `cflibs` package:

```python
# Old way (legacy)
from saha_eggert import LibsEngine
engine = LibsEngine("libs_production.db")

# New way (recommended)
from cflibs.plasma import SingleZoneLTEPlasma
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
```

## Migration Timeline

- **Phase 0-1**: Core infrastructure (✅ Complete)
- **Phase 2**: Advanced physics (manifold generation)
- **Phase 3**: Inversion and analysis (AirPLS, NNLS, calibration)
- **Phase 4**: Ecosystem integration

Most legacy functionality will be migrated by Phase 3.

