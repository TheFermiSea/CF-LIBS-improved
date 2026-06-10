# CF-LIBS User Guide

Complete guide for using the CF-LIBS library for forward modeling and analysis of LIBS spectra.

If you are new to the repository, start with
[Quick Start for Scientists](Quick_Start_For_Scientists.md). This guide is more detailed and
assumes you are ready to edit configuration files or use the Python API.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Forward Modeling](#forward-modeling)
4. [Echellogram Extraction](#echellogram-extraction)
5. [Configuration Files](#configuration-files)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

---

## Installation

### Requirements

- Python 3.8 or higher
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- PyYAML >= 5.4.0

### Installation Methods

#### From Source (Development)

```bash
git clone https://github.com/yourusername/CF-LIBS.git
cd CF-LIBS
pip install -e .
```

#### With Optional Dependencies

```bash
# With JAX (for future manifold generation)
pip install -e ".[jax]"

# With HDF5 support
pip install -e ".[hdf5]"

# With all optional dependencies
pip install -e ".[all]"
```

### Verify Installation

```python
import cflibs
print(cflibs.__version__)  # Should print 0.1.0
```

---

## Quick Start

### Generate Your First Spectrum

1. **Create a configuration file** (`my_config.yaml`):

```yaml
atomic_database: libs_production.db

plasma:
  model: single_zone_lte
  Te: 10000.0
  ne: 1.0e17
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

2. **Generate the spectrum**:

```bash
cflibs forward my_config.yaml --output spectrum.csv
```

3. **Load and plot** (Python):

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('spectrum.csv', delimiter=',', skiprows=1)
wavelength = data[:, 0]
intensity = data[:, 1]

plt.plot(wavelength, intensity)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (W m⁻² nm⁻¹ sr⁻¹)')
plt.title('Synthetic LIBS Spectrum')
plt.show()
```

---

## Forward Modeling

### Using the Python API

#### Basic Example

```python
from cflibs.plasma import SingleZoneLTEPlasma
from cflibs.atomic import AtomicDatabase
from cflibs.instrument import InstrumentModel
from cflibs.radiation import SpectrumModel

# 1. Load atomic database
atomic_db = AtomicDatabase("libs_production.db")

# 2. Create plasma state
plasma = SingleZoneLTEPlasma(
    T_e=10000.0,  # Electron temperature in K
    n_e=1e17,     # Electron density in cm^-3
    species={
        "Fe": 1e15,  # Iron density in cm^-3
        "H":  1e16,  # Hydrogen density in cm^-3
    }
)

# 3. Create instrument model
instrument = InstrumentModel(resolution_fwhm_nm=0.05)

# 4. Create spectrum model
model = SpectrumModel(
    plasma=plasma,
    atomic_db=atomic_db,
    instrument=instrument,
    lambda_min=200.0,    # Minimum wavelength in nm
    lambda_max=800.0,    # Maximum wavelength in nm
    delta_lambda=0.01,   # Wavelength step in nm
    path_length_m=0.01   # Plasma path length in meters
)

# 5. Compute spectrum
wavelength, intensity = model.compute_spectrum()
```

### Understanding Plasma Parameters

#### Electron Temperature (T_e)

The electron temperature determines the excitation and ionization balance. Typical values:
- **Low temperature**: 5,000 - 8,000 K (weak ionization, mostly neutral lines)
- **Medium temperature**: 8,000 - 15,000 K (balanced neutral/ion lines)
- **High temperature**: 15,000 - 25,000 K (strong ionization, many ion lines)

#### Electron Density (n_e)

The electron density affects:
- Ionization balance (via Saha equation)
- Line broadening (Stark effect, Phase 2)
- LTE validity (McWhirter criterion)

Typical values: 10¹⁶ - 10¹⁸ cm⁻³

#### Species Densities

Number densities of each element in cm⁻³. The relative values determine the composition.

**Example**: For a 90% Ti, 6% Al, 4% V alloy:
```python
species = {
    "Ti": 9.0e15,  # 90% of 1e16 total
    "Al": 6.0e14,  # 6% of 1e16 total
    "V":  4.0e14,  # 4% of 1e16 total
}
```

### Instrument Parameters

#### Resolution (FWHM)

The instrument resolution determines line broadening. Typical values:
- **High resolution**: 0.01 - 0.05 nm (echelle spectrometers)
- **Medium resolution**: 0.1 - 0.5 nm (standard spectrometers)
- **Low resolution**: > 0.5 nm (simple spectrometers)

#### Response Curve

Optional spectral response curve to account for detector/optics efficiency:

```python
# Load from file
instrument = InstrumentModel(
    resolution_fwhm_nm=0.05,
    response_curve=np.loadtxt('response_curve.csv', delimiter=',')
)
```

---

## Echellogram Extraction

### Basic Usage

```python
from cflibs.instrument import EchelleExtractor
from cflibs.io import save_spectrum
import numpy as np

# Load calibration
extractor = EchelleExtractor('calibration.json')

# Load 2D image (from FITS, TIFF, or NumPy)
image_2d = np.load('echellogram.npy')  # Shape: (height, width)

# Extract 1D spectrum
wavelength, intensity = extractor.extract_spectrum(
    image_2d,
    wavelength_step_nm=0.05,
    merge_method='weighted_average'
)

# Save result
save_spectrum('extracted_spectrum.csv', wavelength, intensity)
```

### Calibration File Format

The calibration file is a JSON file with polynomial coefficients:

```json
{
  "order_50": {
    "y_coeffs": [0.0001, 0.1, 500],
    "wl_coeffs": [0.05, 300.0]
  },
  "order_49": {
    "y_coeffs": [0.0001, 0.1, 700],
    "wl_coeffs": [0.05, 350.0]
  }
}
```

Where:
- `y_coeffs`: Polynomial coefficients for trace position y(x) = a₂x² + a₁x + a₀
- `wl_coeffs`: Polynomial coefficients for wavelength λ(x) = k₁x + k₀

### Extraction Options

```python
# Custom extraction window
extractor = EchelleExtractor(
    calibration_file='calibration.json',
    extraction_window=7  # Use ±7 pixels instead of default ±5
)

# Custom merge method
wavelength, intensity = extractor.extract_spectrum(
    image_2d,
    wavelength_step_nm=0.01,      # Higher resolution
    merge_method='max',            # Use maximum in overlaps
    min_valid_pixels=20            # Require at least 20 valid pixels
)
```

---

## Configuration Files

### Structure

Configuration files use YAML or JSON format:

```yaml
# Required sections
atomic_database: <path>
plasma: <plasma_config>
instrument: <instrument_config>
spectrum: <spectrum_config>

# Optional sections
metadata:
  author: "Your Name"
  description: "Simulation description"
  date: "2025-01-01"
```

### Plasma Configuration

```yaml
plasma:
  model: single_zone_lte      # Currently only this model
  Te: 10000.0                  # Electron temperature in K
  ne: 1.0e17                   # Electron density in cm^-3
  Tg: null                     # Gas temperature (optional, defaults to Te)
  pressure: 1.0                # Pressure in atm (optional)
  
  species:
    - element: Fe
      number_density: 1.0e15
    - element: H
      number_density: 1.0e16
```

### Instrument Configuration

```yaml
instrument:
  resolution_fwhm_nm: 0.05     # Required: Resolution (FWHM) in nm
  response_curve: null         # Optional: Path to response curve CSV
```

### Spectrum Configuration

```yaml
spectrum:
  lambda_min_nm: 200.0         # Minimum wavelength in nm
  lambda_max_nm: 800.0         # Maximum wavelength in nm
  delta_lambda_nm: 0.01        # Wavelength step in nm
  path_length_m: 0.01          # Plasma path length in meters
```

### Example Configurations

See `examples/config_example.yaml` and `examples/config_ti64.yaml` for complete examples.

---

## Advanced Usage

## Inversion (Classic CF-LIBS)

The CLI supports classic CF-LIBS inversion using detected spectral lines.
Provide a spectrum file plus an inversion config.

```bash
cflibs invert spectrum.csv --elements Fe Cu --config examples/inversion_config_example.yaml \
  --output inversion_results.json
```

### Analysis Presets

`analyze`, `invert` and `batch` share one pipeline and one set of
accuracy-critical solver knobs, bundled into presets
(`--preset` flag or `analysis.preset` YAML key):

| Preset | `saha_boltzmann_graph` | `closure_mode` | Use for |
|---|---|---|---|
| `geological` (**default**) | `true` | `oxide` | rocks, soils, minerals — oxygen-bound matrices |
| `metallic` | `true` | `standard` | alloys, metals (oxide stoichiometry would be wrong physics) |
| `raw` | `false` | `standard` | legacy-default comparison runs |

The default is `geological`: on the real ChemCam BHVO-2 basalt standard the
legacy defaults scored RMSE 10.29 wt% (Fe 39 wt% vs certified 8.6) while the
geological bundle scored **4.03 wt%** — the validated-best configuration is
now what a flagless run uses. Explicit `--closure-mode` /
`--saha-boltzmann-graph` flags (and the corresponding YAML keys) override the
preset. Every run logs the resolved preset and all knobs at INFO.

Reproduce the BHVO-2 reference result (RMSE ~4.03 wt%) with no flags:

```bash
cflibs analyze data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv \
  --elements Si,Ti,Al,Fe,Mn,Mg,Ca,Na,K,P \
  --db-path ASD_da/libs_production.db
```

(equivalent to `--preset geological`, i.e. `--saha-boltzmann-graph
--closure-mode oxide`; use `--preset raw` to reproduce the legacy 10.29
baseline).

### Closure Modes

Six closure modes are available (`--closure-mode` / `analysis.closure_mode`):

- `standard` — Σ C_s = 1 over elemental fractions.
- `matrix` — fixes a known matrix-element concentration (`matrix_element`).
- `oxide` — geological closure: cations carry stoichiometric oxygen
  (default molar-oxygen stoichiometry applied automatically; override via
  `closure_kwargs.oxide_stoichiometry` / `oxide_elements`).
- `ilr` — isometric log-ratio (compositional-geometry) closure.
- `pwlr` — pairwise log-ratio closure.
- `dirichlet_residual` — Dirichlet residual-weighted closure.

### Trust Report

Every output path (`analyze` table, `invert` stdout, batch rows, JSON `trust`
block) reports: `converged`, the Boltzmann-plane R², the number of elements
fit, the n_e provenance (Stark-width diagnostic vs the 1-atm pressure-balance
fallback — the fallback prints a visible `WARNING` because n_e was ASSUMED,
not measured), the degeneracy gates (`boltzmann_degenerate`,
`closure_degenerate`), and any requested elements that were dropped before
the fit, with the dropping stage (`detection`, `selection` or `solve`).

### Spectral-Response Correction (`--response-curve`)

CF-LIBS compares line intensities across wide wavelength spans, so the
wavelength-dependent detection efficiency E(λ) of the collection optics +
spectrometer + detector chain enters the Boltzmann intercepts directly
(Δq = ln E) and biases both temperature and concentrations. Correcting the
measured spectrum for E(λ) is an experimental prerequisite of the method
(Tognoni et al., *Spectrochim. Acta B* 65 (2010) 1–14).

**When you need it:** any spectrometer whose data are *not* already
radiometrically calibrated upstream — e.g. an in-house instrument calibrated
with a deuterium/halogen lamp. **When you must not use it:** ChemCam/SuperCam
CCS spectra arrive response-corrected from the mission pipeline; applying a
curve there would corrupt them. The default is therefore *no correction*
(identity) — flagless behaviour is bit-identical to previous releases.

```bash
cflibs analyze spectrum.csv --elements Fe,Cu --response-curve lamp_response.csv
```

or in YAML (path resolved relative to the config file):

```yaml
analysis:
  response_curve: lamp_response.csv
```

File format — the same `(wavelength_nm, relative_efficiency)` representation
the forward model uses (`instrument.response_curve`), either CSV:

```csv
wavelength_nm,relative_efficiency
250.0,0.31
300.0,0.55
...
850.0,0.42
```

or YAML (`wavelength_nm:` and `relative_efficiency:` lists). Comments (`#`)
and a header row are tolerated. Only the **relative** shape matters: the
curve is normalized to max = 1 on load, and the CF-LIBS closure cancels any
absolute scale — an absolutely-calibrated irradiance curve works unchanged.

The measured intensities are divided by the interpolated E(λ) *before* line
detection and observation building, so integrated line intensities and their
uncertainties are both response-corrected. Validation: a curve that does not
cover the spectrum's wavelength range is a hard error (the message lists the
curve coverage vs the spectrum range); up to 5 nm of edge extrapolation is
allowed with a warning (edge value held).

A lamp-free alternative — internal calibration from argon branching ratios
(*J. Anal. At. Spectrom.* 29 (2014) 657–664, DOI 10.1039/C3JA50371B) — has a
designed API surface
(`cflibs.inversion.preprocess.derive_response_from_argon_branching_ratios`)
but is not yet implemented.

### Inversion Configuration

Inversion settings live under the `analysis` section. Unknown keys are a
hard error (typos cannot silently fall back to defaults). Minimal schema:

```yaml
atomic_database: libs_production.db

analysis:
  elements: ["Fe", "Cu"]
  preset: geological      # geological | metallic | raw
  # closure_mode: oxide   # explicit override of the preset (see Closure Modes)
  # saha_boltzmann_graph: true
  min_snr: 10.0
  min_energy_spread_ev: 2.0
  min_lines_per_element: 3
  exclude_resonance: false   # keep resonance lines (sole detectable lines for some majors)
  isolation_wavelength_nm: 0.1
  max_lines_per_element: 20
  wavelength_tolerance_nm: 0.1
  min_peak_height: 0.01
  peak_width_nm: 0.2
  max_iterations: 20
  t_tolerance_k: 100.0
  ne_tolerance_frac: 0.1
  pressure_pa: 101325.0
  min_relative_intensity: null
  # boltzmann_weight_cap: 5.0
  # min_boltzmann_r2: 0.3
  # top_k_per_element: 60
  # wavelength_calibration: true
  # response_curve: lamp_response.csv   # spectral-response E(lambda) correction (see above)
  # closure_kwargs:
  #   matrix_element: Fe
  #   oxide_elements: ["Si", "Al", "Ca"]
```

If you omit `--elements`, the CLI uses `analysis.elements`.

### Custom Partition Functions

The Saha-Boltzmann solver automatically calculates partition functions from energy levels. If energy levels are not available, it uses approximations.

### Multi-Species Plasmas

You can include any number of species:

```python
plasma = SingleZoneLTEPlasma(
    T_e=12000.0,
    n_e=1e17,
    species={
        "Ti": 8.0e15,
        "Al": 6.0e14,
        "V":  4.0e14,
        "Fe": 1.0e13,  # Trace element
    }
)
```

### Extracting Individual Orders

For debugging or analysis:

```python
# Extract a single order
wl_order, flux_order = extractor.extract_order(
    image_2d,
    order_name='order_50',
    background_subtract=True
)
```

### Batch Processing

```python
import glob
from cflibs.core.config import load_config
from cflibs.radiation import SpectrumModel
# ... (setup code)

configs = glob.glob('configs/*.yaml')
for config_file in configs:
    config = load_config(config_file)
    # ... (process each config)
```

---

## Troubleshooting

### Common Issues

#### "Atomic database not found"

**Problem**: The database file doesn't exist.

**Solution**: Use the bundled example database or generate a new one:
```bash
cflibs forward examples/config_example.yaml --output spectrum.csv
cflibs generate-db --db-path libs_production.db
```

#### "No orders calibrated"

**Problem**: Echelle extractor has no calibration loaded.

**Solution**: Load calibration file:
```python
extractor = EchelleExtractor('calibration.json')
```

#### "Invalid plasma state"

**Problem**: Plasma parameters are invalid (negative temperatures, etc.).

**Solution**: Check your configuration:
- T_e > 0
- n_e > 0
- All species densities > 0

#### "Wavelength grid must be evenly spaced"

**Problem**: Trying to apply instrument function to non-uniform grid.

**Solution**: Use evenly spaced wavelength grids for convolution.

### Performance Tips

1. **Filter transitions**: Use `min_relative_intensity` to reduce computation
2. **Reduce wavelength range**: Only compute the range you need
3. **Increase wavelength step**: Use larger `delta_lambda` for faster computation
4. **Cache atomic database**: Keep database connection open for multiple calculations

### Getting Help

- Check the [API Reference](API_Reference.md)
- Review [examples](../examples/)
- See [Contributing Guide](../CONTRIBUTING.md) for development questions

---

## Examples

### Example 1: Ti-6Al-4V Alloy

```python
from cflibs.plasma import SingleZoneLTEPlasma
from cflibs.atomic import AtomicDatabase
from cflibs.radiation import SpectrumModel
from cflibs.instrument import InstrumentModel

# Load database
atomic_db = AtomicDatabase("libs_production.db")

# Ti-6Al-4V composition
plasma = SingleZoneLTEPlasma(
    T_e=12000.0,
    n_e=1e17,
    species={
        "Ti": 8.0e15,  # ~90%
        "Al": 6.0e14,  # ~6%
        "V":  4.0e14,  # ~4%
    }
)

# High-resolution instrument
instrument = InstrumentModel(resolution_fwhm_nm=0.05)

# UV-NIR range
model = SpectrumModel(
    plasma, atomic_db, instrument,
    lambda_min=300.0,
    lambda_max=500.0,
    delta_lambda=0.005
)

wavelength, intensity = model.compute_spectrum()
```

### Example 2: Temperature Scan

```python
import numpy as np
import matplotlib.pyplot as plt

temperatures = np.linspace(8000, 15000, 8)
spectra = []

for T_e in temperatures:
    plasma = SingleZoneLTEPlasma(
        T_e=T_e,
        n_e=1e17,
        species={"Fe": 1e15}
    )
    model = SpectrumModel(plasma, atomic_db, instrument, 300, 400, 0.01)
    wl, I = model.compute_spectrum()
    spectra.append((T_e, I))

# Plot temperature dependence
for T_e, I in spectra:
    plt.plot(wl, I, label=f'T={T_e:.0f} K')
plt.legend()
plt.show()
```

### Example 3: Composition Scan

```python
al_concentrations = np.linspace(0.0, 0.12, 13)

for al_conc in al_concentrations:
    ti_conc = 1.0 - al_conc
    plasma = SingleZoneLTEPlasma(
        T_e=10000.0,
        n_e=1e17,
        species={
            "Ti": ti_conc * 1e16,
            "Al": al_conc * 1e16,
        }
    )
    # ... compute spectrum
```

---

## Next Steps

- Explore the [API Reference](API_Reference.md) for detailed function documentation
- Check out [examples](../examples/) for more use cases
- Read about [contributing](../CONTRIBUTING.md) if you want to help develop CF-LIBS

For questions or issues, please open an issue on GitHub.
