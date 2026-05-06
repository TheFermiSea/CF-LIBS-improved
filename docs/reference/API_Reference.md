# CF-LIBS API Reference

Complete API documentation for the CF-LIBS library.

## Table of Contents

1. [Core Module](#core-module)
2. [Atomic Module](#atomic-module)
3. [Plasma Module](#plasma-module)
4. [Radiation Module](#radiation-module)
5. [Instrument Module](#instrument-module)
6. [I/O Module](#io-module)
7. [CLI Module](#cli-module)

---

## Core Module

### Constants (`cflibs.core.constants`)

Physical constants for CF-LIBS calculations.

#### Fundamental Constants

- `KB` (float): Boltzmann constant in J/K = 1.380649e-23
- `KB_EV` (float): Boltzmann constant in eV/K = 8.617333262e-5
- `H_PLANCK` (float): Planck constant in J·s = 6.62607015e-34
- `H_PLANCK_EV` (float): Planck constant in eV·s = 4.135667696e-15
- `C_LIGHT` (float): Speed of light in m/s = 2.99792458e8
- `M_E` (float): Electron mass in kg = 9.1093837015e-31
- `E_CHARGE` (float): Elementary charge in C = 1.602176634e-19

#### Atomic Physics Constants

- `RYDBERG` (float): Rydberg constant in m⁻¹ = 10973731.568160
- `A_BOHR` (float): Bohr radius in m = 5.29177210903e-11
- `ALPHA_FS` (float): Fine structure constant = 7.2973525693e-3

#### Conversion Factors

- `EV_TO_J` (float): Electron-volt to Joule conversion
- `J_TO_EV` (float): Joule to electron-volt conversion
- `CM_TO_EV` (float): Wavenumber (cm⁻¹) to eV conversion = 1.23984193e-4
- `EV_TO_CM` (float): eV to wavenumber conversion
- `K_TO_EV` (float): Kelvin to eV conversion
- `EV_TO_K` (float): eV to Kelvin conversion

#### Plasma Physics Constants

- `SAHA_CONST_CM3` (float): Saha equation constant = 6.042e21 cm⁻³
- `MCWHIRTER_CONST` (float): McWhirter criterion constant = 1.6e12 cm⁻³

### Units (`cflibs.core.units`)

Unit conversion utilities.

#### `convert_temperature(value, from_unit, to_unit)`

Convert temperature between units.

**Parameters:**
- `value` (float or array): Temperature value(s)
- `from_unit` (str): Source unit ('K', 'eV', 'C')
- `to_unit` (str): Target unit ('K', 'eV', 'C')

**Returns:** float or array - Converted temperature

**Example:**
```python
T_ev = convert_temperature(10000, 'K', 'eV')  # ~0.86 eV
```

#### `convert_density(value, from_unit, to_unit)`

Convert number density between units.

**Parameters:**
- `value` (float or array): Density value(s)
- `from_unit` (str): Source unit ('m^-3', 'cm^-3')
- `to_unit` (str): Target unit ('m^-3', 'cm^-3')

**Returns:** float or array - Converted density

#### `convert_wavelength(value, from_unit, to_unit)`

Convert wavelength between units.

**Parameters:**
- `value` (float or array): Wavelength value(s)
- `from_unit` (str): Source unit ('m', 'nm', 'um', 'A', 'cm^-1')
- `to_unit` (str): Target unit ('m', 'nm', 'um', 'A', 'cm^-1')

**Returns:** float or array - Converted wavelength

#### `convert_energy(value, from_unit, to_unit)`

Convert energy between units.

**Parameters:**
- `value` (float or array): Energy value(s)
- `from_unit` (str): Source unit ('J', 'eV', 'cm^-1')
- `to_unit` (str): Target unit ('J', 'eV', 'cm^-1')

**Returns:** float or array - Converted energy

### Configuration (`cflibs.core.config`)

Configuration management utilities.

#### `load_config(config_path)`

Load configuration from YAML or JSON file.

**Parameters:**
- `config_path` (str or Path): Path to configuration file

**Returns:** dict - Configuration dictionary

**Raises:**
- `FileNotFoundError`: If config file does not exist
- `ValueError`: If file format is not supported

#### `validate_plasma_config(config)`

Validate plasma configuration structure.

**Parameters:**
- `config` (dict): Configuration dictionary

**Returns:** bool - True if valid

**Raises:**
- `ValueError`: If configuration is invalid

#### `validate_instrument_config(config)`

Validate instrument configuration structure.

**Parameters:**
- `config` (dict): Configuration dictionary

**Returns:** bool - True if valid

**Raises:**
- `ValueError`: If configuration is invalid

### Logging (`cflibs.core.logging_config`)

Logging configuration utilities.

#### `setup_logging(level='INFO', format_string=None, stream=None)`

Configure logging for CF-LIBS.

**Parameters:**
- `level` (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
- `format_string` (str, optional): Custom format string
- `stream` (file-like, optional): Stream to write logs to

#### `get_logger(name)`

Get a logger instance for a module.

**Parameters:**
- `name` (str): Logger name (typically `__name__`)

**Returns:** logging.Logger - Logger instance

---

## Atomic Module

### Structures (`cflibs.atomic.structures`)

#### `EnergyLevel`

Data structure for atomic energy levels.

**Attributes:**
- `element` (str): Element symbol
- `ionization_stage` (int): Ionization stage (1=neutral, 2=singly ionized)
- `energy_ev` (float): Energy above ground state in eV
- `g` (int): Statistical weight (degeneracy)
- `j` (float, optional): Total angular momentum quantum number

#### `Transition`

Data structure for atomic transitions.

**Attributes:**
- `element` (str): Element symbol
- `ionization_stage` (int): Ionization stage
- `wavelength_nm` (float): Transition wavelength in nm
- `A_ki` (float): Einstein A coefficient in s⁻¹
- `E_k_ev` (float): Upper level energy in eV
- `E_i_ev` (float): Lower level energy in eV
- `g_k` (int): Upper level statistical weight
- `g_i` (int): Lower level statistical weight
- `relative_intensity` (float, optional): Relative intensity from database

**Properties:**
- `energy_diff_ev` (float): Energy difference between upper and lower levels

#### `SpeciesPhysics`

Physical properties of an atomic species.

**Attributes:**
- `element` (str): Element symbol
- `ionization_stage` (int): Ionization stage
- `ionization_potential_ev` (float): Ionization potential in eV

### Database (`cflibs.atomic.database`)

#### `AtomicDatabase`

Interface to atomic data stored in SQLite database.

**Methods:**

##### `__init__(db_path)`

Initialize database connection.

**Parameters:**
- `db_path` (str): Path to SQLite database file

**Raises:**
- `FileNotFoundError`: If database file does not exist

##### `get_transitions(element, ionization_stage=None, wavelength_min=None, wavelength_max=None, min_relative_intensity=None)`

Get transitions for an element.

**Parameters:**
- `element` (str): Element symbol
- `ionization_stage` (int, optional): Filter by ionization stage
- `wavelength_min` (float, optional): Minimum wavelength in nm
- `wavelength_max` (float, optional): Maximum wavelength in nm
- `min_relative_intensity` (float, optional): Minimum relative intensity

**Returns:** List[Transition] - List of transition objects

##### `get_energy_levels(element, ionization_stage)`

Get energy levels for a species.

**Parameters:**
- `element` (str): Element symbol
- `ionization_stage` (int): Ionization stage

**Returns:** List[EnergyLevel] - List of energy level objects

##### `get_ionization_potential(element, ionization_stage)`

Get ionization potential for a species.

**Parameters:**
- `element` (str): Element symbol
- `ionization_stage` (int): Ionization stage

**Returns:** float or None - Ionization potential in eV

##### `get_species_physics(element, ionization_stage)`

Get physical properties for a species.

**Parameters:**
- `element` (str): Element symbol
- `ionization_stage` (int): Ionization stage

**Returns:** SpeciesPhysics or None

##### `get_available_elements()`

Get list of elements available in the database.

**Returns:** List[str] - List of element symbols

##### `close()`

Close database connection.

---

## Plasma Module

### State (`cflibs.plasma.state`)

#### `PlasmaState`

Base plasma state representation.

**Attributes:**
- `T_e` (float): Electron temperature in K
- `n_e` (float): Electron density in cm⁻³
- `species` (Dict[str, float]): Species number densities in cm⁻³
- `T_g` (float, optional): Gas temperature in K (defaults to T_e)
- `pressure` (float, optional): Pressure in atm

**Properties:**
- `T_e_eV` (float): Electron temperature in eV
- `T_g_eV` (float): Gas temperature in eV

#### `SingleZoneLTEPlasma`

Single-zone LTE plasma model.

**Inherits from:** `PlasmaState`

**Methods:**

##### `__init__(T_e, n_e, species, T_g=None, pressure=None)`

Initialize single-zone LTE plasma.

**Parameters:**
- `T_e` (float): Electron temperature in K
- `n_e` (float): Electron density in cm⁻³
- `species` (Dict[str, float]): Species number densities in cm⁻³
- `T_g` (float, optional): Gas temperature in K
- `pressure` (float, optional): Pressure in atm

##### `validate()`

Validate plasma state.

**Returns:** bool - True if valid

**Raises:**
- `ValueError`: If plasma state is invalid

### Saha-Boltzmann (`cflibs.plasma.saha_boltzmann`)

#### `SahaBoltzmannSolver`

Solves Saha-Boltzmann equations for LTE plasma.

**Methods:**

##### `__init__(atomic_db, ipd_model=None)`

Initialize solver.

**Parameters:**
- `atomic_db` (AtomicDataSource): Atomic database
- `ipd_model` (Optional[IPDModel]): Ionization-potential depression model; `None` disables IPD corrections.

##### `solve_ionization_balance(element, T_e_eV, n_e_cm3, total_density_cm3)`

Solve ionization balance using Saha equation.

**Parameters:**
- `element` (str): Element symbol
- `T_e_eV` (float): Electron temperature in eV
- `n_e_cm3` (float): Electron density in cm⁻³
- `total_density_cm3` (float): Total element density in cm⁻³

**Returns:** Dict[int, float] - Dictionary mapping ionization stage to number density

##### `calculate_partition_function(element, ionization_stage, T_e_eV, max_energy_ev=50.0)`

Calculate partition function for a species.

**Parameters:**
- `element` (str): Element symbol
- `ionization_stage` (int): Ionization stage
- `T_e_eV` (float): Electron temperature in eV
- `max_energy_ev` (float): Maximum energy level to include

**Returns:** float - Partition function

##### `solve_level_population(element, ionization_stage, stage_density_cm3, T_e_eV)`

Solve Boltzmann distribution for level populations.

**Parameters:**
- `element` (str): Element symbol
- `ionization_stage` (int): Ionization stage
- `stage_density_cm3` (float): Total density of this ionization stage
- `T_e_eV` (float): Electron temperature in eV

**Returns:** Dict[Tuple[str, int, float], float] - Level populations

##### `solve_plasma(plasma)`

Solve complete Saha-Boltzmann system for a plasma.

**Parameters:**
- `plasma` (SingleZoneLTEPlasma): Plasma state

**Returns:** Dict[Tuple[str, int, float], float] - All level populations

---

## Radiation Module

### Profiles (`cflibs.radiation.profiles`)

#### `gaussian_profile(wavelength, center, sigma, amplitude=1.0)`

Calculate Gaussian line profile.

**Parameters:**
- `wavelength` (float or array): Wavelength(s) in nm
- `center` (float): Line center wavelength in nm
- `sigma` (float): Standard deviation in nm
- `amplitude` (float): Integrated area (not peak height). Peak height equals `amplitude / (sigma * sqrt(2*pi))`.

**Returns:** float or array - Profile value(s)

#### `doppler_width(wavelength_nm, T_eV, mass_amu)`

Calculate Doppler broadening width (FWHM).

**Parameters:**
- `wavelength_nm` (float): Wavelength in nm
- `T_eV` (float): Temperature in eV
- `mass_amu` (float): Atomic mass in atomic mass units

**Returns:** float - Doppler FWHM in nm

#### `apply_gaussian_broadening(wavelength_grid, line_wavelengths, line_intensities, sigma_nm)`

Apply Gaussian broadening to spectral lines.

**Parameters:**
- `wavelength_grid` (array): Wavelength grid in nm
- `line_wavelengths` (array): Line center wavelengths in nm
- `line_intensities` (array): Line intensities
- `sigma_nm` (float): Gaussian standard deviation in nm

**Returns:** array - Broadened spectrum

### Emissivity (`cflibs.radiation.emissivity`)

#### `calculate_line_emissivity(transition, upper_level_population_cm3, wavelength_nm=None)`

Calculate spectral emissivity for a transition.

**Parameters:**
- `transition` (Transition): Atomic transition
- `upper_level_population_cm3` (float): Population of upper level in cm⁻³
- `wavelength_nm` (float, optional): Wavelength in nm

**Returns:** float - Spectral emissivity in W m⁻³ nm⁻¹

#### `calculate_spectrum_emissivity(transitions, populations, wavelength_grid, sigma_nm)`

Calculate total spectral emissivity on a wavelength grid.

**Parameters:**
- `transitions` (List[Transition]): List of transitions
- `populations` (Dict): Level populations from Saha-Boltzmann solver
- `wavelength_grid` (array): Wavelength grid in nm
- `sigma_nm` (float): Gaussian broadening width in nm

**Returns:** array - Spectral emissivity in W m⁻³ nm⁻¹

### Spectrum Model (`cflibs.radiation.spectrum_model`)

#### `SpectrumModel`

Forward model for computing synthetic LIBS spectra.

**Methods:**

##### `__init__(plasma, atomic_db, instrument, lambda_min, lambda_max, delta_lambda, path_length_m=0.01, use_jax=False, broadening_mode=BroadeningMode.LEGACY)`

Initialize spectrum model.

**Parameters:**
- `plasma` (SingleZoneLTEPlasma): Plasma state
- `atomic_db` (AtomicDatabase): Atomic database
- `instrument` (InstrumentModel): Instrument model
- `lambda_min` (float): Minimum wavelength in nm
- `lambda_max` (float): Maximum wavelength in nm
- `delta_lambda` (float): Wavelength step in nm
- `path_length_m` (float): Plasma path length in meters (default 0.01 = 1 cm)
- `use_jax` (bool): Enable JAX-accelerated evaluation path
- `broadening_mode` (BroadeningMode): Line-broadening variant (LEGACY / RESOLVING_POWER). See `cflibs.radiation.BroadeningMode`.

##### `compute_spectrum()`

Compute synthetic spectrum.

**Returns:** Tuple[array, array] - (wavelength, intensity)
- `wavelength`: Wavelength grid in nm
- `intensity`: Spectral intensity in W m⁻² nm⁻¹ sr⁻¹

---

## Instrument Module

### Model (`cflibs.instrument.model`)

#### `InstrumentModel`

Model for spectrometer instrument response. Supports two resolution modes:

- **Fixed-FWHM mode** (default): constant `resolution_fwhm_nm` across the spectrum.
- **Resolving-power mode**: `resolution_fwhm_nm` scales with wavelength as `λ / R`. Enabled when `resolving_power` is set and positive.

**Attributes:**
- `resolution_fwhm_nm` (float): Instrument FWHM in nm (fixed-FWHM mode).
- `resolving_power` (Optional[float]): Dimensionless resolving power R = λ/Δλ. When set (> 0), resolution scales with wavelength and this overrides `resolution_fwhm_nm` at query time.
- `response_curve` (array, optional): Spectral response curve.
- `wavelength_calibration` (callable, optional): Pixel-to-wavelength function.

**Properties:**
- `resolution_sigma_nm` (float): Gaussian standard deviation derived from `resolution_fwhm_nm`.
- `is_resolving_power_mode` (bool): True iff `resolving_power` is set and positive.

**Methods:**

##### `sigma_at_wavelength(wavelength_nm)`

Return Gaussian σ (nm) at the given wavelength. In resolving-power mode, FWHM = `wavelength_nm / resolving_power`; otherwise returns the fixed-FWHM σ.

##### `from_resolving_power(resolving_power, response_curve=None)` (classmethod)

Construct an `InstrumentModel` in resolving-power mode.

**Parameters:**
- `resolving_power` (float): Dimensionless R, must be positive.
- `response_curve` (Optional[array]): Optional spectral response curve.

**Raises:** `ValueError` if `resolving_power <= 0`.

##### `from_file(config_path)`

Load instrument model from configuration file.

**Parameters:**
- `config_path` (Path): Path to configuration file

**Returns:** InstrumentModel

##### `apply_response(wavelength, intensity)`

Apply spectral response curve.

**Parameters:**
- `wavelength` (array): Wavelength grid in nm
- `intensity` (array): Intensity spectrum

**Returns:** array - Intensity with response applied

### Convolution (`cflibs.instrument.convolution`)

#### `apply_instrument_function(wavelength, intensity, sigma_nm)`

Apply Gaussian instrument function via convolution.

**Parameters:**
- `wavelength` (array): Wavelength grid in nm (must be evenly spaced)
- `intensity` (array): Intensity spectrum
- `sigma_nm` (float): Gaussian standard deviation in nm

**Returns:** array - Convolved intensity spectrum

**Raises:**
- `ValueError`: If wavelength grid is not evenly spaced

### Echelle (`cflibs.instrument.echelle`)

#### `EchelleExtractor`

Extracts 1D spectra from 2D echellogram images.

**Methods:**

##### `__init__(calibration_file=None, extraction_window=5)`

Initialize echelle extractor.

**Parameters:**
- `calibration_file` (str, optional): Path to JSON calibration file
- `extraction_window` (int): Number of pixels above/below trace center

##### `load_calibration(filepath)`

Load order trace and wavelength calibration polynomials.

**Parameters:**
- `filepath` (str): Path to JSON calibration file

**Raises:**
- `FileNotFoundError`: If calibration file does not exist
- `ValueError`: If calibration file format is invalid

##### `extract_order(image_2d, order_name, background_subtract=True)`

Extract spectrum from a single order.

**Parameters:**
- `image_2d` (array): 2D image array (height, width)
- `order_name` (str): Name of the order to extract
- `background_subtract` (bool): Whether to subtract background

**Returns:** Tuple[array, array] - (wavelengths, flux)

##### `extract_spectrum(image_2d, wavelength_step_nm=0.05, merge_method='weighted_average', min_valid_pixels=10)`

Extract complete 1D spectrum from 2D echellogram.

**Parameters:**
- `image_2d` (array): 2D echellogram image
- `wavelength_step_nm` (float): Wavelength step for output grid
- `merge_method` (str): Method for merging ('weighted_average', 'simple_average', 'max')
- `min_valid_pixels` (int): Minimum valid pixels required per order

**Returns:** Tuple[array, array] - (wavelengths, intensity)

---

## I/O Module

### Spectrum (`cflibs.io.spectrum`)

#### `load_spectrum(file_path)`

Load spectrum from file.

**Parameters:**
- `file_path` (str): Path to spectrum file (CSV or similar)

**Returns:** Tuple[array, array] - (wavelength, intensity)

**Raises:**
- `ValueError`: If file format is not supported

#### `save_spectrum(file_path, wavelength, intensity, header=None)`

Save spectrum to file.

**Parameters:**
- `file_path` (str): Output file path
- `wavelength` (array): Wavelength array in nm
- `intensity` (array): Intensity array
- `header` (str, optional): Header comment

---

## Inversion Module

The inversion module is organized into 6 physics-aligned sub-packages that reflect the CF-LIBS measurement→physics→inference pipeline:

| Sub-package | Role |
|-------------|------|
| `cflibs.inversion.common` | Data structures (LineObservation, BoltzmannFitResult), PCA pipeline |
| `cflibs.inversion.preprocess` | Signal processing (baseline, noise, deconvolution, wavelength calibration) |
| `cflibs.inversion.physics` | Saha-Boltzmann solver, closure equations, CDSB, Stark broadening, line selection, uncertainties |
| `cflibs.inversion.identify` | Element identification (ALIAS, comb, correlation, spectral NNLS, BIC selection) |
| `cflibs.inversion.solve` | Plasma inference (iterative CF-LIBS, ILR solver, Bayesian MCMC, manifold coarse-to-fine) |
| `cflibs.inversion.runtime` | Real-time: DAQ streaming, temporal gate optimization, hardware interface |

**Backward compatibility:** Old flat import paths (e.g., `from cflibs.inversion.solver import X`) still work via compatibility shims.

**Physics-only constraint:** The shipped CF-LIBS algorithm is physics-only (no neural networks or trained models). See [Evolution_Framework.md](../development/Evolution_Framework.md) for the full forbidden/allowed specification and enforcement.


### Peak Identification and Line Matching

The user-facing workflow is described in
[Peak Identification and Line Matching](../user/Peak_Identification_Guide.md).
This section maps that workflow to the public APIs.

#### Preprocessing (`cflibs.inversion.preprocessing`)

- `BaselineMethod`: `MEDIAN`, `SNIP`, or `ALS` baseline strategy.
- `estimate_baseline(wavelength, intensity, window_nm=10.0)`: moving-median
  baseline estimate.
- `estimate_baseline_snip(...)`: SNIP baseline estimate for sharp peaks on a
  slowly varying continuum.
- `estimate_baseline_als(...)`: asymmetric least-squares baseline estimate.
- `estimate_noise(intensity, baseline)`: iterative sigma-clipped MAD noise
  estimate.
- `detect_peaks(...)`: baseline-subtracted, noise-thresholded peak detection.
- `detect_peaks_auto(...)`: convenience wrapper that estimates baseline/noise
  before calling `detect_peaks`.

#### Classic Line Observation Builder (`cflibs.inversion.line_detection`)

`detect_line_observations(...)` is the bridge from raw spectra to the classic
CF-LIBS solver. It loads candidate transitions, detects peaks, applies optional
`kdet` prefiltering, scans global wavelength shifts, scores comb matches, and
returns a `LineDetectionResult` with:

- `observations`: `LineObservation` inputs for the iterative solver.
- `resonance_lines`: transition keys marked as self-absorption risk.
- `total_peaks`, `matched_peaks`, `unmatched_peaks`.
- `applied_shift_nm`: selected global wavelength shift.
- `warnings`: machine-readable diagnostics such as `no_peaks_detected`,
  `comb_no_elements_passed`, or `no_peaks_matched`.

#### Element Identifiers (`cflibs.inversion.identify`)

| Class | Use case |
|-------|----------|
| `ALIASIdentifier` | Interpretable ALIAS scoring with theoretical emissivities, detection-rate, wavelength-shift, and confidence terms. |
| `CombIdentifier` | Element fingerprint matching with triangular comb teeth and local template correlation. |
| `CorrelationIdentifier` | Model-spectrum correlation over `(T, n_e)` grids or vector-indexed libraries. |
| `SpectralNNLSIdentifier` | Full-spectrum non-negative decomposition into single-element basis spectra. |

All identifiers return `ElementIdentificationResult`, which separates detected
and rejected elements and carries per-element score/confidence metadata. For
quantitative CF-LIBS, use these identifiers to produce a defensible candidate
list, then pass selected elements through `detect_line_observations` and
`LineSelector` before solving.

## CLI Module

### Main (`cflibs.cli.main`)

#### `main()`

Main CLI entry point.

**Commands:**
- `cflibs forward <config>` - Generate synthetic spectrum
- `cflibs invert <spectrum> [--config]` - Invert measured spectrum (classic CF-LIBS)

**Options:**
- `--log-level` - Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--output` - Output file path
- `--config` - Configuration file path
- `--elements` - Elements to include in inversion (overrides config)
- `--tolerance-nm` - Wavelength matching tolerance in nm
- `--min-peak-height` - Minimum peak height (fraction of max intensity)
- `--peak-width-nm` - Peak integration width in nm

**Inversion Config (analysis section):**

```yaml
atomic_database: libs_production.db

analysis:
  elements: ["Fe", "Cu"]
  closure_mode: standard  # standard | matrix | oxide
  min_snr: 10.0
  min_energy_spread_ev: 2.0
  min_lines_per_element: 3
  exclude_resonance: true
  isolation_wavelength_nm: 0.1
  max_lines_per_element: 20
  wavelength_tolerance_nm: 0.1
  min_peak_height: 0.01
  peak_width_nm: 0.2
  max_iterations: 20
  t_tolerance_k: 100.0
  ne_tolerance_frac: 0.1
  pressure_pa: 101325.0
```

---

## Error Handling

All modules use descriptive error messages and proper exception types:

- `FileNotFoundError`: Missing files
- `ValueError`: Invalid parameters or data
- `KeyError`: Missing keys in dictionaries
- `TypeError`: Type mismatches

---

## Further Reading

See the [User Guide](../user/User_Guide.md), [Contributing Guide](../../CONTRIBUTING.md), and [ROADMAP.md](../../ROADMAP.md).
