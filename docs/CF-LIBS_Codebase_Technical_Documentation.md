# CF-LIBS Codebase Technical Documentation

**Calibration-Free Laser-Induced Breakdown Spectroscopy**

A Python library for quantitative elemental analysis without calibration standards, using plasma physics to calculate elemental compositions directly from spectral line intensities. The shipped algorithm is physics-only (no neural networks or trained models) — see [Evolution_Framework.md](Evolution_Framework.md) for the constraint spec.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture and Design Patterns](#architecture-and-design-patterns)
3. [Core Module (`cflibs.core`)](#core-module)
4. [Atomic Data Module (`cflibs.atomic`)](#atomic-data-module)
5. [Plasma Physics Module (`cflibs.plasma`)](#plasma-physics-module)
6. [Radiation Module (`cflibs.radiation`)](#radiation-module)
7. [Inversion Module (`cflibs.inversion`)](#inversion-module)
8. [Bayesian Methods](#bayesian-methods)
9. [Manifold Generation (`cflibs.manifold`)](#manifold-generation)
10. [Validation Framework (`cflibs.validation`)](#validation-framework)
11. [Key Physics and Equations](#key-physics-and-equations)
12. [Data Flow and Pipelines](#data-flow-and-pipelines)
13. [JAX Acceleration](#jax-acceleration)
14. [Database Schema](#database-schema)
15. [API Quick Reference](#api-quick-reference)

---

## Project Overview

### Purpose

CF-LIBS (Calibration-Free LIBS) enables quantitative elemental analysis from laser-induced breakdown spectroscopy data **without requiring calibration standards**. This is achieved by:

1. **Forward Modeling**: Computing synthetic spectra from known plasma parameters using fundamental atomic physics
2. **Inverse Problem**: Extracting plasma temperature, electron density, and elemental concentrations from measured spectra

### Status

See [ROADMAP.md](../ROADMAP.md) for current work streams and the epic tracking in beads.

---

## Architecture and Design Patterns

### Design Philosophy

The codebase follows "Physics First, Then Inversion" - ensuring correct physical modeling before implementing inversion algorithms to avoid GIGO (Garbage In, Garbage Out) results.

### Key Design Patterns

#### 1. Abstract Base Classes and Protocols (`cflibs.core.abc`)

Core interfaces are defined as ABCs for concrete inheritance, while optional interfaces use `typing.Protocol` for structural typing:

```python
# ABCs for concrete inheritance
class AtomicDataSource(ABC):
    """Abstract interface for atomic data providers."""
    @abstractmethod
    def get_transitions(self, element: str, ...) -> List[Transition]: ...
    @abstractmethod
    def get_energy_levels(self, element: str, ion_stage: int) -> List[EnergyLevel]: ...

class SolverStrategy(ABC):
    """Interface for plasma equilibrium solvers."""
    @abstractmethod
    def solve_ionization_balance(self, element: str, ...) -> Dict[int, float]: ...

# Protocols for structural typing (no inheritance required)
@runtime_checkable
class PlasmaModel(Protocol):
    """Protocol for plasma state containers."""
    def validate(self) -> None: ...
    @property
    def T_e_eV(self) -> float: ...
    @property
    def n_e(self) -> float: ...

@runtime_checkable
class InstrumentModelProtocol(Protocol):
    """Protocol for instrument models."""
    def apply_response(self, wavelength: np.ndarray, intensity: np.ndarray) -> np.ndarray: ...
    @property
    def resolution_sigma_nm(self) -> float: ...
```

#### 2. Factory Pattern (`cflibs.core.factory`)

Centralized object creation for solvers, plasma models, and instruments:

```python
class SolverFactory:
    @staticmethod
    def create(solver_type: str, **kwargs) -> SolverStrategy: ...

class PlasmaModelFactory:
    @staticmethod
    def create(model_type: str, **kwargs) -> PlasmaModel: ...
```

#### 3. Caching Strategy (`cflibs.core.cache`)

LRU caching for expensive computations:
- Partition function evaluations
- Transition queries
- Ionization state calculations

#### 4. Connection Pooling (`cflibs.core.pool`)

Thread-safe database access for concurrent operations.

---

## Core Module

**Location**: `cflibs/core/`

### Physical Constants (`constants.py`)

Fundamental constants used throughout the codebase:

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| `KB` | k_B | 1.380649e-23 | J/K |
| `KB_EV` | k_B | 8.617333262e-5 | eV/K |
| `H_PLANCK` | h | 6.62607015e-34 | J·s |
| `C_LIGHT` | c | 299792458 | m/s |
| `M_ELECTRON` | m_e | 9.1093837015e-31 | kg |
| `E_CHARGE` | e | 1.602176634e-19 | C |
| `BOHR_RADIUS` | a_0 | 5.29177210903e-11 | m |
| `SAHA_CONST_CM3` | - | 4.8294e15 | cm⁻³ K^(-3/2) |

### Logging Configuration (`logging_config.py`)

Hierarchical logging with configurable levels per module.

### Configuration Loading (`config.py`)

YAML/JSON configuration file parsing with validation.

---

## Atomic Data Module

**Location**: `cflibs/atomic/`

### Data Structures (`structures.py`)

Core dataclasses representing atomic physics entities:

```python
@dataclass
class EnergyLevel:
    """Atomic energy level with quantum numbers."""
    element: str
    ionization_stage: int      # 0=neutral, 1=singly ionized, etc.
    energy_ev: float           # Energy above ground state (eV)
    g: int                     # Statistical weight (2J+1)
    configuration: str         # Electron configuration
    term: str                  # Spectroscopic term symbol

@dataclass
class Transition:
    """Spectral line transition between energy levels."""
    element: str
    ionization_stage: int
    wavelength_nm: float       # Wavelength in vacuum (nm)
    A_ki: float                # Einstein A coefficient (s⁻¹)
    E_k_ev: float              # Upper level energy (eV)
    E_i_ev: float              # Lower level energy (eV)
    g_k: int                   # Upper level degeneracy
    g_i: int                   # Lower level degeneracy
    stark_w: Optional[float]   # Stark width at reference conditions
    stark_alpha: Optional[float]  # Stark temperature exponent
    is_resonance: bool         # True if lower level is ground state

@dataclass
class SpeciesPhysics:
    """Complete physics data for an element/ion combination."""
    element: str
    ionization_stage: int
    ionization_energy_ev: float
    partition_function: PartitionFunction
    transitions: List[Transition]
    energy_levels: List[EnergyLevel]

@dataclass
class PartitionFunction:
    """Temperature-dependent partition function coefficients."""
    element: str
    ionization_stage: int
    coefficients: List[float]  # Irwin polynomial coefficients
    T_min: float               # Valid temperature range (K)
    T_max: float
```

### Atomic Database (`database.py`)

SQLite-based atomic data storage with optimized queries:

```python
class AtomicDatabase(AtomicDataSource):
    """SQLite interface for atomic data with caching."""

    def __init__(self, db_path: str):
        """Initialize with connection pooling."""

    def get_transitions(
        self,
        element: str,
        wavelength_min: float = None,
        wavelength_max: float = None,
        ion_stages: List[int] = None,
        min_relative_intensity: float = None
    ) -> List[Transition]:
        """Query transitions with filtering."""

    def get_energy_levels(
        self,
        element: str,
        ion_stage: int
    ) -> List[EnergyLevel]:
        """Get all energy levels for species."""

    def get_partition_function_coefficients(
        self,
        element: str,
        ion_stage: int
    ) -> Optional[PartitionFunction]:
        """Get Irwin polynomial coefficients."""

    def get_ionization_energy(
        self,
        element: str,
        ion_stage: int
    ) -> float:
        """Get ionization potential (eV)."""

    def get_stark_parameters(
        self,
        element: str,
        wavelength_nm: float,
        tolerance_nm: float = 0.1
    ) -> Optional[Tuple[float, float]]:
        """Get Stark width and alpha for a line."""
```

---

## Plasma Physics Module

**Location**: `cflibs/plasma/`

### Plasma State (`state.py`)

Container for LTE plasma conditions:

```python
class SingleZoneLTEPlasma(PlasmaModel):
    """Single-zone Local Thermodynamic Equilibrium plasma."""

    def __init__(
        self,
        T_e: float,              # Electron temperature (K)
        n_e: float,              # Electron density (cm⁻³)
        species: Dict[str, float],  # Element -> number density (cm⁻³)
        T_g: float = None,       # Gas temperature (K), defaults to T_e
        pressure: float = 1.0    # Pressure (atm)
    ):
        self.T_e = T_e
        self.T_e_eV = T_e * KB_EV  # Convert to eV
        self.n_e = n_e
        self.species = species
        self.T_g = T_g or T_e
        self.pressure = pressure

    def validate(self) -> bool:
        """Check physical validity of plasma parameters."""
        # T_e > 0, n_e > 0, all densities > 0

    @property
    def total_density(self) -> float:
        """Sum of all species densities."""
```

### Saha-Boltzmann Solver (`saha_boltzmann.py`)

Computes ionization equilibrium and level populations:

```python
class SahaBoltzmannSolver(SolverStrategy):
    """Solves Saha-Boltzmann equations for LTE plasmas."""

    def __init__(self, atomic_db: AtomicDatabase):
        self.atomic_db = atomic_db

    def solve_plasma(
        self,
        plasma: SingleZoneLTEPlasma
    ) -> Dict[str, SpeciesPopulations]:
        """Compute all species populations."""

    def compute_ionization_fractions(
        self,
        element: str,
        T_e: float,
        n_e: float,
        max_stage: int = 3
    ) -> np.ndarray:
        """
        Compute ionization stage fractions using Saha equation.

        Returns array of fractions [f_0, f_1, f_2, ...] where
        f_i = n_i / n_total for ionization stage i.
        """

    def compute_partition_function(
        self,
        element: str,
        ion_stage: int,
        T_e: float
    ) -> float:
        """
        Evaluate partition function U(T) using Irwin polynomial:

        log U(T) = Σ a_n (log T)^n

        Falls back to statistical weight sum if coefficients unavailable.
        """

    def compute_level_populations(
        self,
        T_e: float,
        n_species: float,
        energy_levels: List[EnergyLevel],
        partition_function: float
    ) -> Dict[float, float]:
        """
        Compute Boltzmann populations for each energy level:

        n_k / n_total = (g_k / U(T)) × exp(-E_k / kT)
        """
```

#### Saha Equation Implementation

The ionization balance is computed iteratively:

```
n_{z+1} × n_e / n_z = (2 U_{z+1}(T) / U_z(T)) × (2πm_e kT / h²)^(3/2) × exp(-χ_z / kT)
```

Where:
- `n_z` = number density of ionization stage z
- `χ_z` = ionization potential of stage z
- `U_z(T)` = partition function of stage z

---

## Radiation Module

**Location**: `cflibs/radiation/`

### Line Profiles (`profiles.py`)

Spectral line shape functions:

```python
def gaussian_profile(
    wavelength: np.ndarray,
    center: float,
    sigma: float
) -> np.ndarray:
    """
    Gaussian (Doppler) line profile:

    G(λ) = (1 / σ√(2π)) × exp(-(λ-λ₀)² / 2σ²)

    Parameters
    ----------
    wavelength : array
        Wavelength grid (nm)
    center : float
        Line center wavelength (nm)
    sigma : float
        Doppler width σ_D = λ₀ × √(2kT / mc²)
    """

def lorentzian_profile(
    wavelength: np.ndarray,
    center: float,
    gamma: float
) -> np.ndarray:
    """
    Lorentzian (natural + collisional) line profile:

    L(λ) = (γ/π) / ((λ-λ₀)² + γ²)

    Parameters
    ----------
    gamma : float
        Half-width at half-maximum (nm)
    """

def voigt_profile(
    wavelength: np.ndarray,
    center: float,
    sigma: float,
    gamma: float
) -> np.ndarray:
    """
    Voigt profile via Faddeeva function (Humlicek W4 approximation):

    V(λ) = Re[w(z)] / (σ√(2π))

    where z = (λ-λ₀ + iγ) / (σ√2)
    and w(z) is the Faddeeva function.

    The Humlicek W4 algorithm provides <10⁻⁴ relative error
    across all parameter regimes.
    """

def voigt_profile_jax(
    wavelength: jnp.ndarray,
    center: float,
    sigma: float,
    gamma: float
) -> jnp.ndarray:
    """
    JAX-compatible Voigt profile using Weideman rational approximation.

    Uses 32-term rational function approximation to the Faddeeva function,
    suitable for automatic differentiation in optimization.

    Gradient-stable for MCMC sampling and L-BFGS optimization.
    """
```

### Stark Broadening (`stark.py`)

Electron-impact broadening for plasma diagnostics:

```python
def stark_width(
    w_ref: float,
    n_e: float,
    T_e: float,
    T_ref: float = 10000.0,
    n_e_ref: float = 1e16,
    alpha: float = 0.0
) -> float:
    """
    Stark width scaling law:

    w(n_e, T) = w_ref × (n_e / n_e_ref) × (T / T_ref)^(-α)

    Parameters
    ----------
    w_ref : float
        Reference Stark width (nm) at n_e_ref, T_ref
    n_e : float
        Electron density (cm⁻³)
    T_e : float
        Electron temperature (K)
    alpha : float
        Temperature exponent (typically -0.5 to 0.5)
    """

def estimate_stark_parameter_jax(
    wavelength_nm: float,
    ionization_stage: int
) -> float:
    """
    Estimate Stark width when database values unavailable.

    Uses semi-empirical formula based on wavelength and charge state.
    Provides order-of-magnitude estimate for line selection.
    """
```

### Doppler Broadening

```python
def doppler_width(
    wavelength_nm: float,
    temperature_K: float,
    atomic_mass_amu: float
) -> float:
    """
    Proper Doppler width calculation:

    σ_D = λ₀ × √(2kT / mc²)

    where m is the atomic mass of the emitting species.
    """
```

### Spectrum Model (`spectrum_model.py`)

Forward model orchestrating all components:

```python
class SpectrumModel:
    """
    Forward model for computing synthetic LIBS spectra.

    Pipeline:
    1. Saha-Boltzmann equilibrium → level populations
    2. Line emissivity calculation → ε(λ)
    3. Optically thin approximation → I = ε × L
    4. Instrument response → detector efficiency
    5. Instrument function → spectral convolution
    """

    def __init__(
        self,
        plasma: SingleZoneLTEPlasma,
        atomic_db: AtomicDatabase,
        instrument: InstrumentModel,
        lambda_min: float,
        lambda_max: float,
        delta_lambda: float,
        path_length_m: float = 0.01,
        use_jax: bool = False
    ): ...

    def compute_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute synthetic spectrum.

        Returns
        -------
        wavelength : array
            Wavelength grid (nm)
        intensity : array
            Spectral intensity (W m⁻² nm⁻¹ sr⁻¹)
        """
```

### Emissivity Calculation (`emissivity.py`)

```python
def line_emissivity(
    transition: Transition,
    population: float,
    temperature: float
) -> float:
    """
    Calculate spectral line emissivity:

    ε_ki = (hc / 4πλ) × n_k × A_ki

    where n_k is the upper level population density.
    """

def calculate_spectrum_emissivity(
    transitions: List[Transition],
    populations: Dict[str, SpeciesPopulations],
    wavelength: np.ndarray,
    sigma_nm: float,
    use_jax: bool = False
) -> np.ndarray:
    """
    Sum all line contributions with broadening.

    ε(λ) = Σ_i ε_i × φ_i(λ)

    where φ_i is the line profile (Voigt if Stark available).
    """
```

---

## Inversion Module

**Location**: `cflibs/inversion/`

**Organization:** The inversion package is organized into 6 physics-aligned sub-packages:
- `cflibs.inversion.common/` — Data structures (LineObservation, BoltzmannFitResult), PCA pipeline
- `cflibs.inversion.preprocess/` — Signal processing (baseline removal, noise estimation, deconvolution)
- `cflibs.inversion.physics/` — Saha-Boltzmann solver, closure equations, CDSB, line selection, uncertainty propagation
- `cflibs.inversion.identify/` — Element identification (ALIAS, comb, correlation, spectral NNLS, BIC)
- `cflibs.inversion.solve/` — Plasma parameter inference (iterative CF-LIBS, ILR solver, Bayesian MCMC, manifold)
- `cflibs.inversion.runtime/` — Real-time systems (DAQ streaming, temporal gate optimization, hardware interface)

**Backward compatibility:** Old flat import paths (e.g., `from cflibs.inversion.solver import X`) still work.

For the CLI and configuration schema used by classic CF-LIBS inversion, see
`docs/User_Guide.md` (Inversion section) and `docs/API_Reference.md` (CLI options).

### Optional Dependency Flags

The inversion module provides availability flags for optional features:

```python
from cflibs.inversion import HAS_HYBRID, HAS_BAYESIAN, HAS_NESTED, HAS_UNCERTAINTIES

if HAS_BAYESIAN:
    from cflibs.inversion import MCMCSampler, BayesianForwardModel
if HAS_NESTED:
    from cflibs.inversion import NestedSampler
if HAS_UNCERTAINTIES:
    from cflibs.inversion import create_boltzmann_uncertainties
```

| Flag | Requires | Features |
|------|----------|----------|
| `HAS_HYBRID` | JAX | `HybridInverter`, manifold-based inversion |
| `HAS_BAYESIAN` | JAX + NumPyro | `MCMCSampler`, MCMC inference |
| `HAS_NESTED` | dynesty | `NestedSampler`, evidence calculation |
| `HAS_UNCERTAINTIES` | uncertainties | Correlation-aware error propagation |

### Result Formatting Utilities (`result_base.py`)

Shared mixins for consistent result formatting across result classes:

```python
class ResultTableMixin:
    """Shared table formatting for MCMCResult, NestedSamplingResult, etc."""
    def _format_param_row(self, label, mean, std, ci=None): ...
    def _format_concentration_table(self, conc_mean, conc_std, conc_ci=None): ...

class StatisticsMixin:
    """Shared CI/percentile calculations."""
    @staticmethod
    def compute_ci(samples, level=0.95) -> Tuple[float, float]: ...
```

### Boltzmann Plot Fitting (`boltzmann.py`)

Extract temperature from line intensities:

```python
@dataclass
class LineObservation:
    """Measured spectral line with atomic data."""
    wavelength_nm: float
    intensity: float           # Integrated intensity
    intensity_error: float     # Measurement uncertainty
    element: str
    ionization_stage: int
    A_ki: float               # Einstein A coefficient
    g_k: int                  # Upper level degeneracy
    E_k_ev: float             # Upper level energy

@dataclass
class BoltzmannPlotData:
    """Prepared data for Boltzmann plot analysis."""
    x: np.ndarray             # E_k / kT (dimensionless)
    y: np.ndarray             # ln(I × λ / (g_k × A_ki))
    weights: np.ndarray       # 1 / σ² weights
    element: str
    ionization_stage: int

class FitMethod(Enum):
    """Available fitting methods."""
    WEIGHTED_LS = "weighted_least_squares"
    SIGMA_CLIP = "sigma_clipping"
    RANSAC = "ransac"
    HUBER = "huber"

class BoltzmannPlotFitter:
    """
    Robust Boltzmann plot fitting with multiple methods.

    Boltzmann Plot Equation:
    ln(I × λ / (g_k × A_ki)) = -E_k / kT + ln(hc × n × L / (4π × U(T)))

    Slope → Temperature
    Intercept → q = ln(F × C_s) where F is normalization factor
    """

    def __init__(
        self,
        method: FitMethod = FitMethod.WEIGHTED_LS,
        sigma_threshold: float = 3.0,
        max_iterations: int = 10,
        min_inliers: int = 3
    ): ...

    def fit(
        self,
        data: BoltzmannPlotData
    ) -> BoltzmannFitResult:
        """
        Fit Boltzmann plot and extract temperature.

        Returns
        -------
        BoltzmannFitResult
            slope: -1/kT (eV⁻¹)
            intercept: q = ln(F × C_s)
            temperature_eV: Extracted temperature
            r_squared: Goodness of fit
            residuals: Fit residuals
            outlier_mask: Boolean mask of rejected points
        """

    def _fit_weighted_ls(self, data): ...
    def _fit_sigma_clip(self, data): ...
    def _fit_ransac(self, data): ...
    def _fit_huber(self, data): ...
```

### Line Detection and Matching (`line_detection.py`)

Convert a measured spectrum into CF-LIBS line observations by detecting peaks
and matching them to database transitions:

See `docs/User_Guide.md` for the `analysis` configuration fields that control
line detection and selection.

```python
from cflibs.inversion.line_detection import detect_line_observations

result = detect_line_observations(
    wavelength,
    intensity,
    atomic_db,
    elements=["Fe", "Cu"],
    wavelength_tolerance_nm=0.1,
    min_peak_height=0.01,
    peak_width_nm=0.2,
)
observations = result.observations
resonance_lines = result.resonance_lines
```

The output provides:
- `LineObservation` objects for each matched line
- A `resonance_lines` set for exclusion in line selection
- Diagnostics on total/matched/unmatched peaks

### Closure Equation (`closure.py`)

Normalize relative concentrations:

```python
@dataclass
class ClosureResult:
    """Result of closure equation application."""
    concentrations: Dict[str, float]  # Element -> mass fraction
    normalization_factor: float       # F from closure
    mode: str                         # "standard", "matrix", "oxide"

class ClosureEquation:
    """
    Closure equation for concentration normalization.

    The closure equation enforces that concentrations sum to unity:
    Σ C_s = 1

    Three modes are supported:
    1. Standard: Direct normalization assuming all elements measured
    2. Matrix: One element (matrix) has known concentration
    3. Oxide: For geological samples, oxide fractions sum to unity
    """

    @staticmethod
    def apply_standard(
        intercepts: Dict[str, float],      # q_s from Boltzmann fits
        partition_funcs: Dict[str, float]  # U_s(T) values
    ) -> ClosureResult:
        """
        Standard closure: Σ C_s = 1

        F = Σ_s U_s × exp(q_s)
        C_s = U_s × exp(q_s) / F
        """

    @staticmethod
    def apply_matrix_mode(
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        matrix_element: str,
        matrix_fraction: float
    ) -> ClosureResult:
        """
        Matrix mode: C_matrix is known, others scale accordingly.

        F = U_matrix × exp(q_matrix) / C_matrix
        C_s = U_s × exp(q_s) / F

        Useful when one major element (e.g., Fe in steel) is known.
        """

    @staticmethod
    def apply_oxide_mode(
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        oxide_stoichiometry: Dict[str, Tuple[str, int, int]]
    ) -> ClosureResult:
        """
        Oxide mode: Σ C_oxide = 1

        For geological samples where elements exist as oxides
        (SiO2, Al2O3, Fe2O3, etc.).

        Converts elemental fractions to oxide mass fractions
        accounting for oxygen stoichiometry.
        """
```

### Iterative CF-LIBS Solver (`solver.py`)

Complete CF-LIBS algorithm:

```python
@dataclass
class CFLIBSResult:
    """Complete CF-LIBS analysis result."""
    temperature_eV: float
    temperature_K: float
    electron_density: float           # cm⁻³
    concentrations: Dict[str, float]  # Element -> mass fraction
    boltzmann_fits: Dict[str, BoltzmannFitResult]
    convergence_history: List[Dict]
    iterations: int
    converged: bool
    quality_metrics: Optional[QualityMetrics]

class IterativeCFLIBSSolver:
    """
    Full iterative CF-LIBS algorithm.

    Algorithm:
    1. Initial temperature estimate from strongest element
    2. For each element: fit Boltzmann plot → T_s, q_s
    3. Average temperature (weighted by R²)
    4. Apply Saha correction to bring ionic intercepts to neutral plane
    5. Apply closure equation → concentrations
    6. Update electron density via Saha
    7. Repeat until convergence

    Convergence criterion: |T_new - T_old| / T_old < tolerance
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        fit_method: FitMethod = FitMethod.SIGMA_CLIP,
        closure_mode: str = "standard",
        max_iterations: int = 50,
        tolerance: float = 1e-3,
        min_lines_per_element: int = 3
    ): ...

    def solve(
        self,
        observations: List[LineObservation],
        initial_T_eV: float = None,
        initial_n_e: float = 1e17
    ) -> CFLIBSResult:
        """
        Run iterative CF-LIBS analysis.

        Parameters
        ----------
        observations : list
            Measured spectral lines with atomic data
        initial_T_eV : float, optional
            Initial temperature guess (auto-estimated if None)
        initial_n_e : float
            Initial electron density guess (cm⁻³)
        """

    def _saha_correction(
        self,
        q_ion: float,
        ion_stage: int,
        element: str,
        T_eV: float,
        n_e: float
    ) -> float:
        """
        Correct ionic intercept to neutral plane.

        q_0 = q_z - z × ln(S(T, n_e))

        where S is the Saha factor between stages.
        """
```

### Line Selection (`line_selection.py`)

Automatic selection of optimal lines:

```python
@dataclass
class LineScore:
    """Quality score for a spectral line."""
    wavelength_nm: float
    element: str
    ionization_stage: int
    snr_score: float           # Signal-to-noise contribution
    atomic_score: float        # Atomic data quality
    isolation_score: float     # Spectral isolation
    total_score: float         # Combined quality score
    reasons: List[str]         # Selection/rejection reasons

class LineSelector:
    """
    Automatic line selection for CF-LIBS analysis.

    Scoring algorithm:
    score = SNR × (1/σ_atomic) × isolation_factor × energy_coverage

    Selection criteria:
    - Minimum SNR threshold
    - Avoid resonance lines (high self-absorption)
    - Energy spread requirement for Boltzmann plots
    - Spectral isolation from interfering lines
    - Atomic data quality (A_ki uncertainty)
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        min_snr: float = 10.0,
        min_isolation_nm: float = 0.1,
        exclude_resonance: bool = True,
        min_energy_spread_eV: float = 1.0,
        max_lines_per_element: int = 20
    ): ...

    def select_lines(
        self,
        spectrum: np.ndarray,
        wavelength: np.ndarray,
        elements: List[str],
        noise_estimate: float = None
    ) -> Dict[str, List[LineScore]]:
        """
        Select optimal lines for each element.

        Returns dictionary mapping element -> ranked line scores.
        """

    def compute_isolation_score(
        self,
        wavelength: float,
        all_transitions: List[Transition]
    ) -> float:
        """
        Score based on distance to nearest interfering line.

        isolation = min(|λ - λ_neighbor|) / min_isolation_nm
        Capped at 1.0 for well-isolated lines.
        """
```

### Self-Absorption Correction (`self_absorption.py`)

Handle optically thick lines:

```python
@dataclass
class AbsorptionResult:
    """Self-absorption correction result."""
    original_intensity: float
    corrected_intensity: float
    optical_depth: float       # τ at line center
    correction_factor: float   # I_corrected / I_observed
    converged: bool
    iterations: int

class SelfAbsorptionCorrector:
    """
    Self-absorption correction for optically thick plasmas.

    For optically thick lines, observed intensity is reduced:
    I_obs = I_true × (1 - exp(-τ)) / τ

    The correction iteratively estimates τ and recovers I_true.

    Assumptions (documented limitations):
    - Homogeneous plasma (uniform T, n_e, composition)
    - LTE conditions
    - Gaussian line profile for opacity calculation
    - No stimulated emission correction

    Algorithm:
    1. Estimate optical depth from line parameters
    2. Apply correction factor
    3. Re-estimate τ with corrected intensity
    4. Iterate until convergence or mask line if τ > threshold
    """

    def __init__(
        self,
        max_optical_depth: float = 2.0,  # Lines with τ > this are masked
        max_iterations: int = 10,
        tolerance: float = 0.01
    ): ...

    def correct_line(
        self,
        observation: LineObservation,
        temperature_eV: float,
        column_density: float,     # n × L (cm⁻²)
        partition_function: float
    ) -> AbsorptionResult:
        """
        Apply self-absorption correction to single line.

        Returns corrected intensity or flags line for masking.
        """

    def estimate_optical_depth(
        self,
        transition: Transition,
        lower_population: float,
        column_density: float,
        line_width_nm: float
    ) -> float:
        """
        Estimate optical depth at line center:

        τ₀ = (π e² / m_e c) × f × n_i × L / Δν_D

        where f is oscillator strength and Δν_D is Doppler width.
        """
```

### Quality Metrics (`quality.py`)

Assess analysis reliability:

```python
@dataclass
class QualityMetrics:
    """Comprehensive quality assessment."""
    boltzmann_r_squared: Dict[str, float]  # Per-element R²
    mean_r_squared: float
    temperature_consistency: float  # Inter-element agreement
    saha_boltzmann_consistency: float  # Neutral/ion agreement
    chi_squared: float
    degrees_of_freedom: int
    reduced_chi_squared: float
    warnings: List[str]
    overall_grade: str  # "A", "B", "C", "D", "F"

class QualityAssessor:
    """
    Assess CF-LIBS analysis quality.

    Checks:
    1. Boltzmann plot linearity (R² > 0.95 ideal)
    2. Temperature consistency across elements
    3. Saha-Boltzmann consistency (neutral vs ionic T)
    4. Residual analysis for systematic errors
    5. LTE validity via McWhirter criterion
    """

    def assess(
        self,
        result: CFLIBSResult,
        observations: List[LineObservation]
    ) -> QualityMetrics:
        """Compute comprehensive quality metrics."""

    def check_lte_validity(
        self,
        T_e: float,
        n_e: float,
        max_delta_E_eV: float
    ) -> Tuple[bool, str]:
        """
        Check McWhirter criterion for LTE:

        n_e > 1.6 × 10¹² × T^(1/2) × (ΔE)³

        where ΔE is the largest energy gap (eV).
        """
```

### Outlier Detection (`outliers.py`)

Robust outlier identification:

```python
class SpectralAngleMapper:
    """
    SAM (Spectral Angle Mapper) for outlier detection.

    Intensity-invariant metric based on angle between spectra:
    θ = arccos(A · B / (|A| × |B|))

    Robust to shot-to-shot laser fluctuations that scale
    overall intensity without changing spectral shape.
    """

    def __init__(self, threshold_degrees: float = None):
        """
        Parameters
        ----------
        threshold_degrees : float, optional
            Angular threshold for outliers.
            If None, automatically calculated as mean + 3σ.
        """

    def detect_outliers(
        self,
        spectra: np.ndarray,      # Shape: (n_spectra, n_wavelengths)
        reference: np.ndarray = None  # Reference spectrum (default: median)
    ) -> SAMResult:
        """
        Identify outlier spectra.

        Returns
        -------
        SAMResult
            outlier_mask: Boolean array of outliers
            angles: Spectral angles to reference
            threshold: Applied threshold
            statistics: Mean, std, median angles
        """

class MADOutlierDetector:
    """
    MAD (Median Absolute Deviation) outlier detector.

    Robust statistic with 50% breakdown point:
    MAD = median(|x - median(x)|)

    Outlier threshold: |x - median| > k × MAD
    where k = 1.4826 × threshold_sigmas for normal distribution.

    Three modes:
    - "1d": Scalar data (single value per sample)
    - "spectrum": Entire spectra (distance metric)
    - "channel": Per-wavelength (for cosmic ray removal)
    """

    def detect_outliers(
        self,
        data: np.ndarray,
        mode: str = "spectrum",
        threshold_sigmas: float = 3.0
    ) -> MADResult:
        """
        Detect outliers using MAD.

        For "channel" mode, also provides interpolation/replacement
        for cosmic ray spike removal.
        """
```

---

## Bayesian Methods

**Location**: `cflibs/inversion/bayesian.py`

### Overview

Full Bayesian inference for uncertainty quantification, providing:
- Posterior distributions for T, n_e, and all concentrations
- Credible intervals with proper uncertainty propagation
- Model comparison via Bayes factors
- Convergence diagnostics (R-hat, ESS)

### Bayesian Forward Model

```python
class BayesianForwardModel:
    """
    JAX-compatible forward model for Bayesian inference.

    Computes synthetic spectra from plasma parameters with
    full physics (Saha-Boltzmann, Voigt profiles, Stark broadening).

    Designed for use with NumPyro MCMC and gradient-based optimization.
    """

    def __init__(
        self,
        wavelength: jnp.ndarray,
        transitions: List[Transition],
        atomic_db: AtomicDatabase,
        path_length_m: float = 0.01,
        use_voigt: bool = True,
        use_stark: bool = True
    ): ...

    def compute_spectrum(
        self,
        temperature_eV: float,
        electron_density: float,
        concentrations: jnp.ndarray  # Must sum to 1
    ) -> jnp.ndarray:
        """
        Compute synthetic spectrum (JAX-traced).

        Returns intensity array matching wavelength grid.
        """

    def log_likelihood(
        self,
        observed: jnp.ndarray,
        temperature_eV: float,
        electron_density: float,
        concentrations: jnp.ndarray,
        noise_model: str = "gaussian",
        noise_params: Dict = None
    ) -> float:
        """
        Compute log-likelihood of observed spectrum.

        Noise models:
        - "gaussian": σ² = σ_readout² (constant)
        - "poisson": σ² = I (shot noise)
        - "combined": σ² = σ_readout² + α×I + dark (realistic)
        """
```

### Prior Specification

```python
@dataclass
class PriorConfig:
    """Physical prior configuration."""

    # Temperature prior
    T_min_eV: float = 0.5      # Minimum temperature
    T_max_eV: float = 3.0      # Maximum temperature
    T_prior: str = "uniform"   # "uniform" or "log_uniform"

    # Electron density prior
    n_e_min: float = 1e15      # Minimum density (cm⁻³)
    n_e_max: float = 1e19      # Maximum density (cm⁻³)
    n_e_prior: str = "log_uniform"  # Jeffreys prior (scale-invariant)

    # Concentration prior
    concentration_prior: str = "dirichlet"  # Enforces simplex
    dirichlet_alpha: float = 1.0  # Uniform on simplex if α=1

def create_temperature_prior(config: PriorConfig):
    """Create NumPyro temperature distribution."""

def create_density_prior(config: PriorConfig):
    """Create NumPyro electron density distribution."""

def create_concentration_prior(n_elements: int, config: PriorConfig):
    """
    Create Dirichlet prior for concentrations.

    The Dirichlet distribution naturally enforces Σ C_s = 1.
    α = 1 gives uniform distribution on the simplex.
    α > 1 concentrates probability toward equal fractions.
    α < 1 concentrates probability toward vertices (pure elements).
    """
```

### MCMC Sampling

```python
@dataclass
class MCMCResult:
    """MCMC sampling result with diagnostics."""
    samples: Dict[str, np.ndarray]  # Parameter -> sample chains
    summary: Dict[str, Dict]        # Mean, std, quantiles per parameter
    r_hat: Dict[str, float]         # Gelman-Rubin convergence diagnostic
    ess: Dict[str, float]           # Effective sample size
    divergences: int                # Number of divergent transitions
    arviz_data: Any                 # ArviZ InferenceData object

    def summary_table(self) -> str:
        """Publication-ready summary table."""

    def credible_interval(
        self,
        parameter: str,
        level: float = 0.95
    ) -> Tuple[float, float]:
        """Get credible interval for parameter."""

    def correlation_matrix(self) -> np.ndarray:
        """Posterior correlation matrix."""

    def plot_trace(self, **kwargs):
        """ArviZ trace plot."""

    def plot_posterior(self, **kwargs):
        """ArviZ posterior distribution plot."""

    def plot_corner(self, **kwargs):
        """Corner plot with correlations."""

class MCMCSampler:
    """
    NumPyro NUTS MCMC sampler for CF-LIBS inference.

    Uses No-U-Turn Sampler (NUTS) for efficient exploration
    of posterior geometry. Includes automatic diagnostics
    and convergence checking.
    """

    def __init__(
        self,
        forward_model: BayesianForwardModel,
        prior_config: PriorConfig = None,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 4,
        target_accept_prob: float = 0.8
    ): ...

    def sample(
        self,
        observed_spectrum: np.ndarray,
        elements: List[str],
        noise_estimate: float = None,
        rng_key: Any = None
    ) -> MCMCResult:
        """
        Run MCMC sampling.

        Returns MCMCResult with posterior samples and diagnostics.
        """

    def posterior_predictive_check(
        self,
        result: MCMCResult,
        observed: np.ndarray,
        n_samples: int = 100
    ) -> Dict:
        """
        Model validation via posterior predictive checks.

        Generates synthetic spectra from posterior samples
        and compares to observed data.

        Returns chi-squared statistics and Bayesian p-value.
        """
```

### Nested Sampling

```python
@dataclass
class NestedSamplingResult:
    """Nested sampling result for model comparison."""
    log_evidence: float           # ln(Z)
    log_evidence_err: float       # Uncertainty in ln(Z)
    samples: Dict[str, np.ndarray]
    weights: np.ndarray           # Sample importance weights
    summary: Dict[str, Dict]

    @staticmethod
    def compare_models(
        result_a: 'NestedSamplingResult',
        result_b: 'NestedSamplingResult'
    ) -> Dict:
        """
        Compute Bayes factor for model comparison.

        ln(B_AB) = ln(Z_A) - ln(Z_B)

        Interpretation:
        |ln(B)| < 1: Inconclusive
        1 < |ln(B)| < 2.5: Substantial evidence
        2.5 < |ln(B)| < 5: Strong evidence
        |ln(B)| > 5: Decisive evidence
        """

class NestedSampler:
    """
    dynesty nested sampler for evidence calculation.

    Nested sampling computes the Bayesian evidence Z = P(D|M),
    enabling rigorous model comparison (e.g., number of elements,
    LTE vs non-LTE models).
    """

    def __init__(
        self,
        forward_model: BayesianForwardModel,
        prior_config: PriorConfig = None,
        nlive: int = 500,
        bound: str = 'multi',
        sample: str = 'auto'
    ): ...

    def run(
        self,
        observed_spectrum: np.ndarray,
        elements: List[str],
        noise_estimate: float = None
    ) -> NestedSamplingResult:
        """
        Run nested sampling.

        Returns evidence and posterior samples.
        """
```

---

## Manifold Generation

**Location**: `cflibs/manifold/`

### Overview

Pre-computed spectral lookup tables for fast inversion. The "manifold" is a discretized parameter space where spectra are computed once and stored for rapid nearest-neighbor lookup.

### Manifold Generator (`generator.py`)

```python
class ManifoldGenerator:
    """
    Generate pre-computed spectral manifolds for fast inversion.

    Creates a grid of synthetic spectra over:
    - Temperature range (T_min to T_max)
    - Electron density range (n_e_min to n_e_max)
    - Composition simplex (concentration combinations)

    Output stored in HDF5 format for efficient access.
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        wavelength: np.ndarray,
        elements: List[str],
        T_range: Tuple[float, float],
        n_e_range: Tuple[float, float],
        n_T_points: int = 20,
        n_ne_points: int = 10,
        n_composition_points: int = 100,
        path_length_m: float = 0.01,
        use_voigt_profile: bool = True,
        use_stark_broadening: bool = True
    ): ...

    def generate(
        self,
        output_path: str,
        show_progress: bool = True
    ) -> None:
        """
        Generate manifold and save to HDF5.

        HDF5 structure:
        /wavelength          - Wavelength grid
        /temperatures        - Temperature grid (eV)
        /electron_densities  - n_e grid (cm⁻³)
        /compositions        - Composition array (n_points, n_elements)
        /spectra            - Spectrum array (n_T, n_ne, n_comp, n_wavelengths)
        /metadata           - Physics version, configuration, elements
        """

    @jit
    def _compute_spectrum_snapshot(
        self,
        T_eV: float,
        n_e: float,
        concentrations: jnp.ndarray
    ) -> jnp.ndarray:
        """
        JAX-JIT compiled spectrum calculation.

        Full physics: Saha-Boltzmann + Voigt + Stark.
        Vectorized over wavelength grid.
        """
```

### Hybrid Inversion (`hybrid_inversion.py`)

Combines manifold lookup with gradient optimization:

```python
class HybridInverter:
    """
    Two-stage inversion: manifold coarse search + L-BFGS refinement.

    Stage 1: Find nearest neighbors in pre-computed manifold
    Stage 2: Use JAX L-BFGS to refine from best manifold match

    Benefits:
    - Fast initial guess from manifold (avoids local minima)
    - High precision from gradient optimization
    - Robust to initialization
    """

    def __init__(
        self,
        manifold_path: str,          # HDF5 manifold file
        forward_model: BayesianForwardModel,
        n_neighbors: int = 5,
        refinement_tolerance: float = 1e-6,
        max_refinement_iterations: int = 100
    ): ...

    def invert(
        self,
        observed_spectrum: np.ndarray,
        noise_estimate: float = None
    ) -> HybridInversionResult:
        """
        Perform hybrid inversion.

        Returns best-fit parameters with uncertainties from
        Hessian at optimum.
        """

class SpectralFitter:
    """
    Standalone JAX-based spectral fitting.

    Uses L-BFGS-B optimization with parameter transforms:
    - Log-transform for positivity (T, n_e)
    - Softmax for simplex constraint (concentrations)

    Provides Hessian-based uncertainty estimates.
    """

    def fit(
        self,
        observed: np.ndarray,
        initial_params: Dict,
        bounds: Dict = None
    ) -> FitResult:
        """
        Fit spectrum to observations.

        Returns optimal parameters and covariance matrix.
        """
```

---

## Validation Framework

**Location**: `cflibs/validation/`

### Overview

Tools for validating the forward-inverse pipeline using synthetic data with known ground truth.

### Golden Spectrum Generator (`golden_spectrum.py`)

```python
class GoldenSpectrumGenerator:
    """
    Generate synthetic spectra with known ground truth.

    Creates "golden" spectra where all parameters (T, n_e, concentrations)
    are exactly known, enabling validation of inversion algorithms.
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        wavelength: np.ndarray,
        elements: List[str]
    ): ...

    def generate(
        self,
        temperature_eV: float,
        electron_density: float,
        concentrations: Dict[str, float],
        noise_model: NoiseModel = None
    ) -> GoldenSpectrum:
        """
        Generate spectrum with optional realistic noise.

        Returns GoldenSpectrum with ground truth metadata.
        """
```

### Noise Models (`noise_models.py`)

```python
class NoiseModel:
    """
    Realistic noise model for synthetic spectra.

    Combines:
    - Poisson shot noise (√N statistics)
    - Gaussian readout noise (detector electronics)
    - Laser fluctuations (multiplicative noise)
    - Dark current (baseline offset)
    """

    def __init__(
        self,
        readout_noise: float = 10.0,    # Counts
        dark_current: float = 1.0,       # Counts/pixel
        laser_fluctuation: float = 0.05, # Fractional (5%)
        gain: float = 1.0                # Detector gain
    ): ...

    def apply(
        self,
        clean_spectrum: np.ndarray,
        rng: np.random.Generator = None
    ) -> np.ndarray:
        """Apply realistic noise to spectrum."""

    def estimate_variance(
        self,
        spectrum: np.ndarray
    ) -> np.ndarray:
        """Estimate per-pixel variance for weighting."""
```

### Round-Trip Validation (`round_trip.py`)

Current validator defaults target synthetic recovery of ±5% in temperature,
±20% in electron density, and ±10% in concentration. The concentration
default was tightened from 15% to 10% in March 2026; callers that need the
legacy threshold should pass `concentration_tolerance=0.15` explicitly.

```python
class RoundTripValidator:
    """
    Validate forward-inverse round-trip accuracy.

    Workflow:
    1. Generate synthetic spectrum with known parameters
    2. Add realistic noise
    3. Run inversion algorithm
    4. Compare recovered parameters to ground truth
    5. Compute accuracy metrics
    """

    def __init__(
        self,
        forward_model: BayesianForwardModel,
        inverter: Union[IterativeCFLIBSSolver, HybridInverter, MCMCSampler]
    ): ...

    def validate(
        self,
        n_spectra: int = 100,
        parameter_ranges: Dict = None,
        noise_model: NoiseModel = None
    ) -> ValidationReport:
        """
        Run round-trip validation suite.

        Returns statistics on parameter recovery accuracy:
        - Bias (systematic error)
        - Precision (random error)
        - Coverage (credible interval calibration)
        """
```

---

## Algorithm Evolution Framework

**Location**: `cflibs/evolution/`

**Purpose**: LLM-driven algorithm optimization tooling for physics-only CF-LIBS. This is **tooling only** — not part of shipped CF-LIBS algorithm.

### Overview

The evolution framework enables automated discovery of new inversion algorithms using large language models within a strict physics-only constraint:

```python
class PhysicsOnlyBlocklist:
    """
    Enforces hard constraint: evolved candidates cannot import or use
    sklearn, torch, tensorflow, keras, flax, equinox, transformers,
    jax.nn, or jax.experimental.stax.
    
    Dual enforcement:
    1. Ruff TID251 static rule in pyproject.toml
    2. AST blocklist scanner in evaluator.py rejects violating code
    """
    
    BANNED_MODULES = {
        'sklearn', 'torch', 'tensorflow', 'keras', 
        'flax', 'equinox', 'transformers', 'jax.nn'
    }
    BANNED_ATTRIBUTES = {'jax.experimental.stax'}
```

### Key Components

- **Hierarchical ES driver**: Evolutionary strategy for parameter optimization
- **Physics fitness evaluator**: Validates evolved algorithms against NIST data
- **Blocklist scanner**: Rejects any evolved code that violates physics-only constraint
- **Persistence layer**: Saves successful algorithm variants for review and integration

See `docs/Evolution_Framework.md` for full specifications.

---

## Key Physics and Equations

### Line Intensity (LTE)

The fundamental equation for spectral line intensity in LTE:

```
I_ki = F × C_s × (g_k × A_ki / U_s(T)) × exp(-E_k / kT)
```

Where:
- `I_ki` = Integrated line intensity
- `F` = Experimental factor (optical geometry, detector response)
- `C_s` = Concentration (mass fraction) of species s
- `g_k` = Statistical weight of upper level (2J+1)
- `A_ki` = Einstein spontaneous emission coefficient (s⁻¹)
- `U_s(T)` = Partition function of species s at temperature T
- `E_k` = Upper level energy (eV)
- `k` = Boltzmann constant (8.617×10⁻⁵ eV/K)
- `T` = Temperature (K)

### Boltzmann Plot

Taking the logarithm:

```
ln(I_ki × λ / (g_k × A_ki)) = -E_k / kT + ln(hc × n × L / (4π × U(T)))
```

Plotting `ln(I×λ/gA)` vs `E_k` gives:
- **Slope** = `-1/kT` → Temperature
- **Intercept** = `q = ln(F × C_s × U(T))` → Related to concentration

### Saha Equation

Ionization balance between stages z and z+1:

```
(n_{z+1} × n_e) / n_z = (2 U_{z+1} / U_z) × (2π m_e kT / h²)^(3/2) × exp(-χ_z / kT)
```

Where:
- `χ_z` = Ionization potential of stage z (eV)
- `U_z` = Partition function of stage z

### Partition Function (Irwin Form)

```
log U(T) = Σ_{n=0}^{N} a_n × (log T)^n
```

Coefficients `a_n` are stored in the database for each species.

### Stark Broadening

Lorentzian HWHM due to electron impacts:

```
γ_Stark(n_e, T) = w_ref × (n_e / 10^16) × (T / T_ref)^(-α)
```

Where:
- `w_ref` = Reference width at n_e = 10¹⁶ cm⁻³, T = T_ref
- `α` = Temperature exponent (typically -0.5 to 0.5)

### Doppler Broadening

Gaussian sigma due to thermal motion:

```
σ_Doppler = λ₀ × √(2kT / mc²)
```

Where `m` is the atomic mass of the emitting species.

### Voigt Profile

Convolution of Gaussian (Doppler) and Lorentzian (Stark):

```
V(λ) = ∫ G(λ') × L(λ - λ') dλ' = Re[w(z)] / (σ√(2π))
```

Where `w(z)` is the Faddeeva function with `z = (λ-λ₀ + iγ) / (σ√2)`.

### Closure Equation

Standard closure (sum of concentrations = 1):

```
Σ_s C_s = 1

F = Σ_s U_s(T) × exp(q_s)
C_s = U_s(T) × exp(q_s) / F
```

---

## Data Flow and Pipelines

### Forward Model Pipeline

```
┌─────────────────┐
│ Plasma State    │  T_e, n_e, {C_s}
│ (SingleZoneLTE) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Saha-Boltzmann  │  Ionization fractions, level populations
│ Solver          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Line Emissivity │  ε_ki for each transition
│ Calculation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Line Profiles   │  Voigt (Doppler + Stark)
│ (profiles.py)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Spectrum Sum    │  ε(λ) = Σ ε_ki × φ_ki(λ)
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Optically Thin  │  I = ε × L (path length)
│ Approximation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Instrument      │  Response curve, convolution
│ Model           │
└────────┬────────┘
         │
         ▼
    Synthetic Spectrum I(λ)
```

### Classic CF-LIBS Inversion Pipeline

```
┌─────────────────┐
│ Measured        │  I(λ) from experiment
│ Spectrum        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Line Detection  │  Peak detection + DB matching
│ & Matching      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Line Selection  │  Score, filter, exclude resonance
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│           Iterative Loop                     │
│  ┌─────────────────┐                        │
│  │ Boltzmann Plots │  Per-element T, q      │
│  └────────┬────────┘                        │
│           │                                  │
│           ▼                                  │
│  ┌─────────────────┐                        │
│  │ Saha Correction │  Ionic → neutral plane │
│  └────────┬────────┘                        │
│           │                                  │
│           ▼                                  │
│  ┌─────────────────┐                        │
│  │ Closure         │  {C_s} from {q_s}      │
│  │ Equation        │                        │
│  └────────┬────────┘                        │
│           │                                  │
│           ▼                                  │
│  ┌─────────────────┐                        │
│  │ Update n_e      │  Via Saha equation     │
│  └────────┬────────┘                        │
│           │                                  │
│           ▼                                  │
│    Check convergence ──────────────────────▶│
│           │ (not converged)                  │
│           └──────────────────────────────────┘
│
└─────────────────────────────────────────────┘
         │ (converged)
         ▼
┌─────────────────┐
│ Quality         │  R², consistency checks
│ Assessment      │
└────────┬────────┘
         │
         ▼
    CFLIBSResult: T, n_e, {C_s}
```

### Bayesian Inversion Pipeline

```
┌─────────────────┐
│ Observed        │
│ Spectrum        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Prior           │  T ~ Uniform(0.5, 3.0 eV)
│ Specification   │  n_e ~ LogUniform(10¹⁵, 10¹⁹)
│                 │  C ~ Dirichlet(α)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MCMC Sampling   │  NumPyro NUTS
│ (MCMCSampler)   │
└────────┬────────┘
         │
         ├──────────────────────────────────┐
         │                                   │
         ▼                                   ▼
┌─────────────────┐               ┌─────────────────┐
│ Convergence     │               │ Posterior       │
│ Diagnostics     │               │ Samples         │
│ (R-hat, ESS)    │               │                 │
└─────────────────┘               └────────┬────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │ Summary         │
                                  │ Statistics      │
                                  │ T = μ ± σ       │
                                  │ Credible        │
                                  │ Intervals       │
                                  └─────────────────┘
```

---

## JAX Acceleration

### Overview

The codebase uses JAX for GPU acceleration and automatic differentiation. Key JAX-enabled components:

- `voigt_profile_jax`: Weideman rational Faddeeva approximation
- `ManifoldGenerator._compute_spectrum_snapshot`: JIT-compiled forward model
- `BayesianForwardModel`: Full JAX-traced likelihood
- `SpectralFitter`: L-BFGS with JAX gradients

### JAX Patterns Used

```python
# JIT compilation for performance
@jax.jit
def compute_spectrum(T_eV, n_e, concentrations):
    # Pure function, no side effects
    ...

# Vectorization over wavelengths
spectrum = jax.vmap(line_profile)(wavelengths, centers, widths)

# Automatic differentiation for optimization
grad_fn = jax.grad(negative_log_likelihood)
hessian_fn = jax.hessian(negative_log_likelihood)

# Gradient-stable Faddeeva (Weideman 32-term)
# Avoids numerical instabilities in MCMC sampling
```

### CPU/GPU Compatibility

Tests automatically use CPU via `pytest.ini`:
```ini
[pytest]
env =
    JAX_PLATFORMS=cpu
```

For GPU usage, install appropriate JAX variant:
```bash
pip install -e ".[local]"    # Apple Silicon (jax-metal)
pip install -e ".[cluster]"  # NVIDIA CUDA
```

---

## Database Schema

### Tables

#### `lines` (Spectral Transitions)

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| element | TEXT | Element symbol (e.g., "Fe") |
| ionization_stage | INTEGER | 0=neutral, 1=I, 2=II, etc. |
| wavelength_nm | REAL | Wavelength in vacuum (nm) |
| A_ki | REAL | Einstein A coefficient (s⁻¹) |
| E_k_ev | REAL | Upper level energy (eV) |
| E_i_ev | REAL | Lower level energy (eV) |
| g_k | INTEGER | Upper level degeneracy |
| g_i | INTEGER | Lower level degeneracy |
| stark_w | REAL | Stark width reference (nm), nullable |
| stark_alpha | REAL | Stark temperature exponent, nullable |
| is_resonance | INTEGER | 1 if ground state transition |
| relative_intensity | REAL | Relative intensity for filtering |

#### `energy_levels`

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| element | TEXT | Element symbol |
| ionization_stage | INTEGER | Ionization stage |
| energy_ev | REAL | Energy above ground (eV) |
| g | INTEGER | Statistical weight |
| configuration | TEXT | Electron configuration |
| term | TEXT | Term symbol |

#### `partition_functions`

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| element | TEXT | Element symbol |
| ionization_stage | INTEGER | Ionization stage |
| coefficients | TEXT | JSON array of Irwin coefficients |
| T_min | REAL | Minimum valid temperature (K) |
| T_max | REAL | Maximum valid temperature (K) |

#### `ionization_energies`

| Column | Type | Description |
|--------|------|-------------|
| element | TEXT | Element symbol |
| ionization_stage | INTEGER | Stage being ionized FROM |
| energy_ev | REAL | Ionization potential (eV) |

### Indices

- `idx_lines_element_wavelength`: (element, wavelength_nm)
- `idx_lines_wavelength`: (wavelength_nm)
- `idx_levels_element_stage`: (element, ionization_stage)
- `idx_partition_element_stage`: (element, ionization_stage)

---

## API Quick Reference

### Creating a Forward Model

```python
from cflibs.plasma import SingleZoneLTEPlasma
from cflibs.atomic import AtomicDatabase
from cflibs.radiation import SpectrumModel
from cflibs.instrument import InstrumentModel

# Load database
db = AtomicDatabase("libs_production.db")

# Define plasma
plasma = SingleZoneLTEPlasma(
    T_e=10000.0,  # K
    n_e=1e17,     # cm⁻³
    species={"Fe": 1e15, "Ti": 5e14}
)

# Define instrument
instrument = InstrumentModel(resolution_fwhm_nm=0.05)

# Create model and compute
model = SpectrumModel(
    plasma, db, instrument,
    lambda_min=300, lambda_max=500, delta_lambda=0.01
)
wavelength, intensity = model.compute_spectrum()
```

### Running Classic CF-LIBS

```python
from cflibs.inversion import IterativeCFLIBSSolver, LineObservation

# Prepare observations (from peak fitting)
observations = [
    LineObservation(
        wavelength_nm=382.04,
        intensity=1000.0,
        intensity_error=50.0,
        element="Fe",
        ionization_stage=0,
        A_ki=6.7e7,
        g_k=9,
        E_k_ev=4.10
    ),
    # ... more lines
]

# Run solver
solver = IterativeCFLIBSSolver(
    atomic_db=db,
    fit_method=FitMethod.SIGMA_CLIP,
    closure_mode="standard"
)
result = solver.solve(observations)

print(f"T = {result.temperature_eV:.3f} eV")
print(f"n_e = {result.electron_density:.2e} cm⁻³")
print(f"Concentrations: {result.concentrations}")
```

### Bayesian Inference

```python
from cflibs.inversion.bayesian import (
    BayesianForwardModel, MCMCSampler, PriorConfig
)

# Create forward model
forward = BayesianForwardModel(
    wavelength=wavelength,
    transitions=transitions,
    atomic_db=db
)

# Configure priors
priors = PriorConfig(
    T_min_eV=0.5, T_max_eV=2.5,
    n_e_min=1e16, n_e_max=1e18,
    dirichlet_alpha=1.0
)

# Run MCMC
sampler = MCMCSampler(forward, priors, num_samples=2000)
result = sampler.sample(observed_spectrum, elements=["Fe", "Ti"])

# Get results
print(result.summary_table())
T_low, T_high = result.credible_interval("temperature", 0.95)
print(f"T = {result.summary['temperature']['mean']:.3f} [{T_low:.3f}, {T_high:.3f}] eV")
```

### Manifold-Based Inversion

```python
from cflibs.manifold import ManifoldGenerator
from cflibs.inversion import HybridInverter

# Generate manifold (one-time)
generator = ManifoldGenerator(
    atomic_db=db,
    wavelength=wavelength,
    elements=["Fe", "Ti"],
    T_range=(0.5, 2.5),
    n_e_range=(1e16, 1e18)
)
generator.generate("manifold.h5")

# Use for fast inversion
inverter = HybridInverter("manifold.h5", forward_model)
result = inverter.invert(observed_spectrum)
```

---

## References

1. Ciucci, A., et al. "New procedure for quantitative elemental analysis by laser-induced plasma spectroscopy." Applied Spectroscopy 53.8 (1999): 960-964.

2. Tognoni, E., et al. "Calibration-free laser-induced breakdown spectroscopy: state of the art." Spectrochimica Acta Part B 65.1 (2010): 1-14.

3. Aragón, C., and J. A. Aguilera. "Characterization of laser induced plasmas by optical emission spectroscopy: A review of experiments and methods." Spectrochimica Acta Part B 63.9 (2008): 893-916.

4. Irwin, A. W. "Polynomial partition function approximations of 344 atomic and molecular species." Astrophysical Journal Supplement Series 45 (1981): 621-633.

5. Griem, H. R. "Spectral Line Broadening by Plasmas." Academic Press (1974).

6. Humlicek, J. "Optimized computation of the Voigt and complex probability functions." Journal of Quantitative Spectroscopy and Radiative Transfer 27.4 (1982): 437-444.

---

*Generated for NotebookLM analysis - CF-LIBS Codebase v0.1.0*
