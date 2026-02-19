"""
Synthetic benchmark generation for algorithm validation.

This module provides tools for generating synthetic LIBS benchmark datasets
with known ground truth, enabling controlled validation of CF-LIBS algorithms
before testing on real CRM data.

Key Features
------------
- Physically realistic spectrum generation using forward model
- Configurable composition ranges and distributions
- Instrumental condition variations for robustness testing
- Reproducible generation with seed control

References
----------
- Tognoni et al. (2010): CF-LIBS validation methodology
- Ciucci et al. (1999): Forward model for LIBS spectra
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.benchmark.dataset import (
    BenchmarkSpectrum,
    BenchmarkDataset,
    InstrumentalConditions,
    SampleMetadata,
    SampleType,
    MatrixType,
)

logger = get_logger("benchmark.synthetic")

# Standard atomic masses used for mass-fraction -> number-density conversion [amu]
# NOTE: Keep this table aligned with candidate elements used in synthetic benchmarks.
# Unknown elements now raise to avoid silently corrupting number-fraction conversion.
STANDARD_MASSES = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.01,
    "N": 14.01,
    "O": 16.00,
    "F": 19.00,
    "Ne": 20.18,
    "Na": 22.99,
    "Mg": 24.31,
    "Al": 26.98,
    "Si": 28.09,
    "P": 30.97,
    "S": 32.07,
    "Cl": 35.45,
    "Ar": 39.95,
    "K": 39.10,
    "Ca": 40.08,
    "Sc": 44.96,
    "Ti": 47.87,
    "V": 50.94,
    "Cr": 52.00,
    "Mn": 54.94,
    "Fe": 55.85,
    "Co": 58.93,
    "Ni": 58.69,
    "Cu": 63.55,
    "Zn": 65.38,
}


@dataclass
class CompositionRange:
    """
    Specification for element composition ranges in synthetic benchmarks.

    Attributes
    ----------
    element : str
        Element symbol
    min_fraction : float
        Minimum mass fraction (0-1)
    max_fraction : float
        Maximum mass fraction (0-1)
    distribution : str
        Distribution type: "uniform", "log_uniform", "fixed"
    fixed_value : float, optional
        Fixed value if distribution is "fixed"
    required : bool
        Whether element must be present in all samples
    """

    element: str
    min_fraction: float
    max_fraction: float
    distribution: str = "uniform"
    fixed_value: Optional[float] = None
    required: bool = True

    def __post_init__(self):
        """Validate ranges."""
        if self.min_fraction < 0 or self.max_fraction > 1:
            raise ValueError("Fractions must be in [0, 1]")
        if self.min_fraction > self.max_fraction:
            raise ValueError("min_fraction must be <= max_fraction")
        if self.distribution not in ("uniform", "log_uniform", "fixed"):
            raise ValueError(f"Unknown distribution: {self.distribution}")
        if self.distribution == "fixed" and self.fixed_value is None:
            raise ValueError("fixed distribution requires fixed_value")

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a composition value."""
        if self.distribution == "fixed":
            return self.fixed_value
        elif self.distribution == "uniform":
            return rng.uniform(self.min_fraction, self.max_fraction)
        elif self.distribution == "log_uniform":
            # Log-uniform in the range
            if self.min_fraction <= 0:
                # Fall back to uniform for zero lower bound
                return rng.uniform(self.min_fraction, self.max_fraction)
            log_min = np.log(self.min_fraction)
            log_max = np.log(self.max_fraction)
            return np.exp(rng.uniform(log_min, log_max))
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class ConditionVariation:
    """
    Specification for instrumental condition variations.

    Attributes
    ----------
    temperature_range_K : Tuple[float, float]
        Plasma temperature range (K)
    electron_density_range_cm3 : Tuple[float, float]
        Electron density range (cm^-3)
    laser_energy_variation : float
        Fractional variation in laser energy (e.g., 0.1 = 10%)
    noise_level : float
        Fractional noise level (e.g., 0.02 = 2%)
    background_level : float
        Background level relative to peak intensity
    gate_delay_range_us : Tuple[float, float]
        Gate delay range (microseconds)
    """

    temperature_range_K: Tuple[float, float] = (8000.0, 15000.0)
    electron_density_range_cm3: Tuple[float, float] = (1e16, 1e18)
    laser_energy_variation: float = 0.1
    noise_level: float = 0.02
    background_level: float = 0.01
    gate_delay_range_us: Tuple[float, float] = (0.5, 2.0)

    def sample_temperature(self, rng: np.random.Generator) -> float:
        """Sample plasma temperature."""
        return rng.uniform(*self.temperature_range_K)

    def sample_electron_density(self, rng: np.random.Generator) -> float:
        """Sample electron density (log-uniform)."""
        log_min = np.log10(self.electron_density_range_cm3[0])
        log_max = np.log10(self.electron_density_range_cm3[1])
        return 10 ** rng.uniform(log_min, log_max)


class SyntheticBenchmarkGenerator:
    """
    Generate synthetic LIBS benchmark datasets with known ground truth.

    This generator creates realistic synthetic spectra using the CF-LIBS
    forward model, enabling controlled validation of inversion algorithms.

    Parameters
    ----------
    atomic_db : AtomicDatabase, optional
        Atomic database for line data. If None, uses simplified model.
    wavelength_range : Tuple[float, float]
        Wavelength range in nm (default: 200-800 nm)
    n_wavelength_points : int
        Number of wavelength points (default: 10000)
    base_conditions : InstrumentalConditions, optional
        Base instrumental conditions

    Example
    -------
    >>> generator = SyntheticBenchmarkGenerator()
    >>>
    >>> # Define composition ranges
    >>> compositions = [
    ...     CompositionRange("Fe", 0.5, 0.9),
    ...     CompositionRange("Cu", 0.05, 0.3),
    ...     CompositionRange("Mn", 0.01, 0.1),
    ... ]
    >>>
    >>> # Generate dataset
    >>> dataset = generator.generate(
    ...     n_spectra=100,
    ...     composition_ranges=compositions,
    ...     name="synthetic_steel_v1",
    ... )
    """

    def __init__(
        self,
        atomic_db=None,
        wavelength_range: Tuple[float, float] = (200.0, 800.0),
        n_wavelength_points: int = 10000,
        base_conditions: Optional[InstrumentalConditions] = None,
    ):
        self.atomic_db = atomic_db
        self.wavelength_range = wavelength_range
        self.n_wavelength_points = n_wavelength_points
        self.base_conditions = base_conditions or self._default_conditions()

        # Create wavelength grid
        self._wavelength_grid = np.linspace(
            wavelength_range[0], wavelength_range[1], n_wavelength_points
        )

    def _default_conditions(self) -> InstrumentalConditions:
        """Create default instrumental conditions."""
        return InstrumentalConditions(
            laser_wavelength_nm=1064.0,
            laser_energy_mj=50.0,
            laser_pulse_width_ns=8.0,
            repetition_rate_hz=10.0,
            spot_diameter_um=100.0,
            gate_delay_us=1.0,
            gate_width_us=10.0,
            spectrometer_type="Echelle",
            spectral_range_nm=self.wavelength_range,
            spectral_resolution_nm=0.05,
            detector_type="ICCD",
            accumulations=10,
            atmosphere="air",
        )

    def generate(
        self,
        n_spectra: int,
        composition_ranges: List[CompositionRange],
        name: str = "synthetic_benchmark",
        version: str = "1.0.0",
        condition_variation: Optional[ConditionVariation] = None,
        seed: int = 42,
        create_default_split: bool = True,
        train_fraction: float = 0.7,
        matrix_type: MatrixType = MatrixType.METAL_ALLOY,
    ) -> BenchmarkDataset:
        """
        Generate a synthetic benchmark dataset.

        Parameters
        ----------
        n_spectra : int
            Number of spectra to generate
        composition_ranges : List[CompositionRange]
            Composition specifications for each element
        name : str
            Dataset name
        version : str
            Dataset version
        condition_variation : ConditionVariation, optional
            Instrumental variation specification
        seed : int
            Random seed for reproducibility
        create_default_split : bool
            Create default train/test split
        train_fraction : float
            Fraction of data for training
        matrix_type : MatrixType
            Matrix type for metadata

        Returns
        -------
        BenchmarkDataset
            Generated synthetic benchmark dataset
        """
        rng = np.random.default_rng(seed)

        if condition_variation is None:
            condition_variation = ConditionVariation()

        elements = [cr.element for cr in composition_ranges]
        spectra = []

        for i in range(n_spectra):
            # Sample composition
            composition = self._sample_composition(composition_ranges, rng)

            # Sample plasma conditions
            temperature_K = condition_variation.sample_temperature(rng)
            electron_density_cm3 = condition_variation.sample_electron_density(rng)

            # Vary instrumental conditions
            conditions = self._vary_conditions(self.base_conditions, condition_variation, rng)

            # Generate spectrum
            spectrum = self._generate_single_spectrum(
                spectrum_id=f"synthetic_{i:04d}",
                composition=composition,
                temperature_K=temperature_K,
                electron_density_cm3=electron_density_cm3,
                conditions=conditions,
                condition_variation=condition_variation,
                matrix_type=matrix_type,
                rng=rng,
            )
            spectra.append(spectrum)

        # Create dataset
        dataset = BenchmarkDataset(
            name=name,
            version=version,
            spectra=spectra,
            elements=elements,
            description=(
                f"Synthetic benchmark with {n_spectra} spectra. "
                f"Elements: {', '.join(elements)}. "
                f"T: {condition_variation.temperature_range_K[0]:.0f}-"
                f"{condition_variation.temperature_range_K[1]:.0f} K. "
                f"Generated with seed={seed}."
            ),
            citation="Generated by CF-LIBS SyntheticBenchmarkGenerator",
            license="CC-BY-4.0",
            created_date=datetime.now().strftime("%Y-%m-%d"),
            contributors=["CF-LIBS Library"],
        )

        # Create default split
        if create_default_split:
            dataset.create_random_split(
                name="default",
                train_fraction=train_fraction,
                test_fraction=1.0 - train_fraction,
                random_seed=seed,
            )

        logger.info(f"Generated synthetic benchmark: {name} with {n_spectra} spectra")

        return dataset

    def _sample_composition(
        self,
        ranges: List[CompositionRange],
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """Sample composition from specified ranges and normalize."""
        composition = {}

        for cr in ranges:
            composition[cr.element] = cr.sample(rng)

        # Normalize to sum to 1
        total = sum(composition.values())
        if total > 0:
            composition = {el: c / total for el, c in composition.items()}

        return composition

    def _vary_conditions(
        self,
        base: InstrumentalConditions,
        variation: ConditionVariation,
        rng: np.random.Generator,
    ) -> InstrumentalConditions:
        """Apply variations to instrumental conditions."""
        # Vary laser energy
        energy_factor = 1.0 + rng.uniform(
            -variation.laser_energy_variation,
            variation.laser_energy_variation,
        )
        new_energy = base.laser_energy_mj * energy_factor

        # Vary gate delay
        new_delay = rng.uniform(*variation.gate_delay_range_us)

        return InstrumentalConditions(
            laser_wavelength_nm=base.laser_wavelength_nm,
            laser_energy_mj=new_energy,
            laser_pulse_width_ns=base.laser_pulse_width_ns,
            repetition_rate_hz=base.repetition_rate_hz,
            spot_diameter_um=base.spot_diameter_um,
            fluence_j_cm2=None,  # Will be recalculated
            gate_delay_us=new_delay,
            gate_width_us=base.gate_width_us,
            spectrometer_type=base.spectrometer_type,
            spectral_range_nm=base.spectral_range_nm,
            spectral_resolution_nm=base.spectral_resolution_nm,
            detector_type=base.detector_type,
            accumulations=base.accumulations,
            atmosphere=base.atmosphere,
            pressure_mbar=base.pressure_mbar,
        )

    def _generate_single_spectrum(
        self,
        spectrum_id: str,
        composition: Dict[str, float],
        temperature_K: float,
        electron_density_cm3: float,
        conditions: InstrumentalConditions,
        condition_variation: ConditionVariation,
        matrix_type: MatrixType,
        rng: np.random.Generator,
    ) -> BenchmarkSpectrum:
        """Generate a single synthetic spectrum."""
        # Generate intensity using simplified model or full forward model
        if self.atomic_db is not None:
            intensity = self._generate_with_forward_model(
                composition, temperature_K, electron_density_cm3, rng
            )
        else:
            intensity = self._generate_simplified(composition, temperature_K, rng)

        # Add noise
        intensity = self._add_noise(
            intensity,
            condition_variation.noise_level,
            condition_variation.background_level,
            rng,
        )

        # Create metadata
        metadata = SampleMetadata(
            sample_id=spectrum_id,
            sample_type=SampleType.SIMULATED,
            matrix_type=matrix_type,
            provenance="Synthetic benchmark spectrum generated by CF-LIBS library",
        )

        # Create composition uncertainty (assume 2% relative for synthetic)
        composition_uncertainty = {el: max(c * 0.02, 0.0001) for el, c in composition.items()}

        # Intensity uncertainty from noise model
        intensity_uncertainty = np.sqrt(
            intensity * condition_variation.noise_level
            + (condition_variation.noise_level * intensity) ** 2
        )

        return BenchmarkSpectrum(
            spectrum_id=spectrum_id,
            wavelength_nm=self._wavelength_grid.copy(),
            intensity=intensity,
            intensity_uncertainty=intensity_uncertainty,
            true_composition=composition,
            composition_uncertainty=composition_uncertainty,
            conditions=conditions,
            metadata=metadata,
            plasma_temperature_K=temperature_K,
            electron_density_cm3=electron_density_cm3,
            quality_flag=0,
        )

    def _generate_with_forward_model(
        self,
        composition: Dict[str, float],
        temperature_K: float,
        electron_density_cm3: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate spectrum using full forward model."""
        try:
            from cflibs.radiation.spectrum_model import SpectrumModel
            from cflibs.plasma.state import SingleZoneLTEPlasma
            from cflibs.instrument.model import InstrumentModel

            # Convert mass-fraction style composition to number densities.
            # For synthetic benchmarking we scale to ~charge-neutral total species density.
            number_densities = self._composition_to_number_densities(
                composition=composition,
                total_number_density_cm3=electron_density_cm3,
            )

            # Create plasma state
            plasma = SingleZoneLTEPlasma(
                T_e=temperature_K,
                n_e=electron_density_cm3,
                species=number_densities,
            )

            # Create instrument + spectrum model
            delta_lambda = float(self._wavelength_grid[1] - self._wavelength_grid[0])
            instrument = InstrumentModel(
                resolution_fwhm_nm=self.base_conditions.spectral_resolution_nm
            )
            model = SpectrumModel(
                plasma=plasma,
                atomic_db=self.atomic_db,
                instrument=instrument,
                lambda_min=float(self._wavelength_grid[0]),
                lambda_max=float(self._wavelength_grid[-1]),
                delta_lambda=delta_lambda,
            )

            # Generate spectrum
            wavelength_model, intensity_model = model.compute_spectrum()

            # SpectrumModel builds grid with np.arange; interpolate if edge differs.
            if wavelength_model.shape != self._wavelength_grid.shape or not np.allclose(
                wavelength_model, self._wavelength_grid
            ):
                return np.interp(
                    self._wavelength_grid,
                    wavelength_model,
                    intensity_model,
                    left=0.0,
                    right=0.0,
                )
            return intensity_model

        except (RuntimeError, ValueError, ImportError) as e:
            logger.warning(
                "Forward model failed (%s), using simplified model fallback",
                e,
                exc_info=True,
            )
            return self._generate_simplified(composition, temperature_K, rng)

    @staticmethod
    def _composition_to_number_densities(
        composition: Dict[str, float], total_number_density_cm3: float
    ) -> Dict[str, float]:
        """
        Convert mass-fraction-like composition to element number densities.

        Number fraction n_i is proportional to w_i / A_i, where w_i is composition
        fraction and A_i is atomic mass.
        """
        if total_number_density_cm3 <= 0:
            raise ValueError("total_number_density_cm3 must be positive")

        weighted = {}
        for element, mass_fraction in composition.items():
            if mass_fraction <= 0:
                continue
            if element not in STANDARD_MASSES:
                raise KeyError(
                    f"Missing standard atomic mass for element '{element}' in synthetic generator"
                )
            mass_amu = STANDARD_MASSES[element]
            weighted[element] = mass_fraction / max(float(mass_amu), 1e-9)

        total_weighted = sum(weighted.values())
        if total_weighted <= 0:
            raise ValueError("composition must contain at least one positive component")

        return {
            element: total_number_density_cm3 * (value / total_weighted)
            for element, value in weighted.items()
        }

    def _generate_simplified(
        self,
        composition: Dict[str, float],
        temperature_K: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate simplified synthetic spectrum without atomic database."""
        from cflibs.core.constants import KB_EV

        # Initialize with low background
        spectrum = np.ones(len(self._wavelength_grid)) * 10.0

        # Add emission lines for each element
        # Using approximate wavelength positions and relative intensities
        element_lines = {
            "Fe": [
                (274.9, 5.5, 1.0),  # (wavelength_nm, E_k_eV, rel_intensity)
                (302.1, 4.2, 0.8),
                (358.1, 3.5, 0.7),
                (371.9, 4.2, 0.9),
                (385.9, 4.3, 0.8),
                (404.6, 4.5, 0.6),
                (438.4, 4.4, 0.5),
            ],
            "Cu": [
                (324.8, 3.8, 1.0),
                (327.4, 3.8, 0.95),
                (510.6, 3.8, 0.4),
                (515.3, 6.2, 0.35),
                (521.8, 6.2, 0.3),
            ],
            "Mn": [
                (279.5, 4.4, 1.0),
                (279.8, 4.4, 0.8),
                (280.1, 4.4, 0.6),
                (403.1, 3.1, 0.7),
                (403.3, 3.1, 0.65),
                (403.5, 3.1, 0.6),
            ],
            "Al": [
                (308.2, 4.0, 1.0),
                (309.3, 4.0, 0.9),
                (394.4, 3.1, 0.8),
                (396.2, 3.1, 0.85),
            ],
            "Ti": [
                (334.9, 3.7, 1.0),
                (336.1, 3.7, 0.95),
                (337.3, 3.7, 0.9),
                (365.4, 3.4, 0.7),
                (498.2, 2.5, 0.5),
            ],
            "Ni": [
                (341.5, 3.6, 1.0),
                (352.5, 3.5, 0.9),
                (361.9, 3.8, 0.8),
            ],
            "Cr": [
                (357.9, 3.5, 1.0),
                (359.3, 3.5, 0.95),
                (360.5, 3.5, 0.9),
                (425.4, 2.9, 0.8),
                (427.5, 2.9, 0.75),
                (428.9, 2.9, 0.7),
            ],
            "Ca": [
                (393.4, 3.1, 1.0),
                (396.8, 3.1, 0.95),
                (422.7, 2.9, 0.6),
            ],
            "Mg": [
                (279.6, 4.4, 1.0),
                (280.3, 4.4, 0.9),
                (285.2, 4.3, 0.85),
                (518.4, 5.1, 0.3),
            ],
            "Si": [
                (250.7, 5.0, 0.7),
                (251.6, 5.0, 0.65),
                (252.9, 5.0, 0.6),
                (288.2, 5.1, 0.8),
            ],
        }

        # Voigt-like profile parameters
        gaussian_width = 0.03  # nm (Doppler broadening)

        for element, conc in composition.items():
            if element not in element_lines:
                continue

            lines = element_lines[element]
            for wl_center, E_k, rel_int in lines:
                # Boltzmann factor
                boltz = np.exp(-E_k / (KB_EV * temperature_K))

                # Line intensity
                intensity = conc * rel_int * boltz * 1e4

                # Add Gaussian line profile
                profile = self._gaussian_profile(self._wavelength_grid, wl_center, gaussian_width)
                spectrum += intensity * profile

        return spectrum

    def _gaussian_profile(
        self,
        wavelengths: np.ndarray,
        center: float,
        width: float,
    ) -> np.ndarray:
        """Calculate Gaussian line profile."""
        return np.exp(-((wavelengths - center) ** 2) / (2 * width**2))

    def _add_noise(
        self,
        spectrum: np.ndarray,
        noise_level: float,
        background_level: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add realistic noise to spectrum."""
        # Add background
        max_intensity = spectrum.max()
        spectrum = spectrum + background_level * max_intensity

        # Shot noise (Poisson)
        # Use sqrt(I) approximation for Poisson variance
        shot_noise = np.sqrt(np.maximum(spectrum, 1.0)) * rng.standard_normal(len(spectrum))

        # Multiplicative noise (laser fluctuation)
        mult_noise = spectrum * noise_level * rng.standard_normal(len(spectrum))

        # Add noise
        noisy_spectrum = spectrum + shot_noise + mult_noise

        # Ensure positive
        return np.maximum(noisy_spectrum, 1.0)


def create_steel_benchmark(
    n_spectra: int = 100,
    seed: int = 42,
    atomic_db=None,
) -> BenchmarkDataset:
    """
    Create a synthetic steel alloy benchmark dataset.

    Convenience function for generating a steel-like benchmark
    with typical LIBS elements (Fe, Mn, Cr, Ni, Cu, etc.).

    Parameters
    ----------
    n_spectra : int
        Number of spectra to generate
    seed : int
        Random seed
    atomic_db : AtomicDatabase, optional
        Atomic database for forward model

    Returns
    -------
    BenchmarkDataset
        Steel alloy benchmark dataset
    """
    generator = SyntheticBenchmarkGenerator(atomic_db=atomic_db)

    composition_ranges = [
        CompositionRange("Fe", 0.60, 0.90),
        CompositionRange("Mn", 0.005, 0.03),
        CompositionRange("Cr", 0.10, 0.25),
        CompositionRange("Ni", 0.02, 0.15),
        CompositionRange("Cu", 0.001, 0.02),
        CompositionRange("Si", 0.002, 0.015),
    ]

    return generator.generate(
        n_spectra=n_spectra,
        composition_ranges=composition_ranges,
        name="synthetic_steel_benchmark",
        version="1.0.0",
        seed=seed,
        matrix_type=MatrixType.METAL_ALLOY,
    )


def create_geological_benchmark(
    n_spectra: int = 100,
    seed: int = 42,
    atomic_db=None,
) -> BenchmarkDataset:
    """
    Create a synthetic geological/soil benchmark dataset.

    Convenience function for generating a geological benchmark
    with typical elements (Si, Al, Fe, Ca, Mg, etc.).

    Parameters
    ----------
    n_spectra : int
        Number of spectra to generate
    seed : int
        Random seed
    atomic_db : AtomicDatabase, optional
        Atomic database for forward model

    Returns
    -------
    BenchmarkDataset
        Geological benchmark dataset
    """
    generator = SyntheticBenchmarkGenerator(atomic_db=atomic_db)

    composition_ranges = [
        CompositionRange("Si", 0.20, 0.40),
        CompositionRange("Al", 0.05, 0.15),
        CompositionRange("Fe", 0.02, 0.10),
        CompositionRange("Ca", 0.01, 0.08),
        CompositionRange("Mg", 0.01, 0.05),
        CompositionRange("Ti", 0.001, 0.02),
    ]

    condition_variation = ConditionVariation(
        temperature_range_K=(7000.0, 12000.0),
        electron_density_range_cm3=(5e15, 5e17),
    )

    return generator.generate(
        n_spectra=n_spectra,
        composition_ranges=composition_ranges,
        name="synthetic_geological_benchmark",
        version="1.0.0",
        seed=seed,
        condition_variation=condition_variation,
        matrix_type=MatrixType.GEOLOGICAL,
    )
