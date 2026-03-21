"""
Round-trip testing framework for CF-LIBS validation.

This module provides tools for validating the complete CF-LIBS pipeline:
1. Generate synthetic spectra with known plasma parameters (forward model)
2. Add realistic noise (Poisson + Gaussian)
3. Run inversion to recover parameters
4. Verify parameter recovery within tolerance

References:
- Tognoni et al., "CF-LIBS: State of the art" (2010)
- Ciucci et al., "New procedure for quantitative elemental analysis" (1999)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from cflibs.core.constants import KB, KB_EV, EV_TO_K, SAHA_CONST_CM3
from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation

logger = get_logger("validation.round_trip")


@dataclass
class GoldenSpectrum:
    """
    A synthetic spectrum with known ground-truth parameters.

    Attributes
    ----------
    temperature_K : float
        True plasma temperature in Kelvin
    electron_density_cm3 : float
        True electron density in cm^-3
    concentrations : Dict[str, float]
        True elemental concentrations (sum to 1.0)
    line_observations : List[LineObservation]
        Synthetic line measurements (before noise)
    wavelength_grid : np.ndarray
        Wavelength grid in nm (if full spectrum generated)
    spectrum : np.ndarray
        Synthetic spectrum intensity (if full spectrum generated)
    seed : int
        Random seed used for reproducibility
    metadata : Dict
        Additional metadata about generation
    """

    temperature_K: float
    electron_density_cm3: float
    concentrations: Dict[str, float]
    line_observations: List[LineObservation]
    wavelength_grid: Optional[np.ndarray] = None
    spectrum: Optional[np.ndarray] = None
    seed: int = 42
    metadata: Dict = field(default_factory=dict)

    @property
    def temperature_eV(self) -> float:
        """Temperature in eV."""
        return self.temperature_K / EV_TO_K


@dataclass
class RoundTripResult:
    """
    Result of a round-trip validation test.

    Attributes
    ----------
    true_temperature_K : float
        Ground truth temperature
    recovered_temperature_K : float
        Temperature from inversion
    temperature_error_frac : float
        Fractional error in temperature
    true_electron_density : float
        Ground truth electron density
    recovered_electron_density : float
        Electron density from inversion
    electron_density_error_frac : float
        Fractional error in electron density
    true_concentrations : Dict[str, float]
        Ground truth concentrations
    recovered_concentrations : Dict[str, float]
        Concentrations from inversion
    concentration_errors : Dict[str, float]
        Fractional errors per element
    converged : bool
        Whether inversion converged
    iterations : int
        Number of iterations used
    passed : bool
        Whether all tolerances were met
    tolerances : Dict[str, float]
        Tolerances used for validation
    """

    true_temperature_K: float
    recovered_temperature_K: float
    temperature_error_frac: float
    true_electron_density: float
    recovered_electron_density: float
    electron_density_error_frac: float
    true_concentrations: Dict[str, float]
    recovered_concentrations: Dict[str, float]
    concentration_errors: Dict[str, float]
    converged: bool
    iterations: int
    passed: bool
    tolerances: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a human-readable summary of the result."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Round-Trip Validation: {status}",
            f"  Temperature: {self.true_temperature_K:.0f} K -> {self.recovered_temperature_K:.0f} K "
            f"(error: {self.temperature_error_frac*100:.1f}%)",
            f"  Electron density: {self.true_electron_density:.2e} -> {self.recovered_electron_density:.2e} "
            f"(error: {self.electron_density_error_frac*100:.1f}%)",
            "  Concentrations:",
        ]
        for el in self.true_concentrations:
            true_c = self.true_concentrations[el]
            rec_c = self.recovered_concentrations.get(el, 0.0)
            err = self.concentration_errors.get(el, float("inf"))
            lines.append(f"    {el}: {true_c:.3f} -> {rec_c:.3f} (error: {err*100:.1f}%)")
        lines.append(f"  Converged: {self.converged} ({self.iterations} iterations)")
        return "\n".join(lines)


class GoldenSpectrumGenerator:
    """
    Generate synthetic spectra with known ground-truth parameters.

    This class creates "golden" spectra that can be used to validate
    the CF-LIBS inversion pipeline. The spectra follow the Boltzmann
    distribution exactly (no self-absorption, LTE assumption).

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database for line data
    wavelength_range : Tuple[float, float]
        Wavelength range in nm (min, max)
    """

    def __init__(
        self,
        atomic_db,
        wavelength_range: Tuple[float, float] = (200.0, 800.0),
    ):
        self.atomic_db = atomic_db
        self.wavelength_range = wavelength_range

    def generate(
        self,
        temperature_K: float,
        electron_density_cm3: float,
        concentrations: Dict[str, float],
        n_lines_per_element: int = 10,
        seed: int = 42,
        include_ionic: bool = True,
        min_intensity: float = 0.1,
    ) -> GoldenSpectrum:
        """
        Generate a golden spectrum with known parameters.

        Parameters
        ----------
        temperature_K : float
            Plasma temperature in Kelvin
        electron_density_cm3 : float
            Electron density in cm^-3
        concentrations : Dict[str, float]
            Elemental concentrations (should sum to 1.0)
        n_lines_per_element : int
            Approximate number of lines per element
        seed : int
            Random seed for reproducibility
        include_ionic : bool
            Include singly-ionized lines
        min_intensity : float
            Minimum intensity threshold for line selection

        Returns
        -------
        GoldenSpectrum
            Synthetic spectrum with ground truth parameters
        """
        rng = np.random.default_rng(seed)

        # Normalize concentrations to sum to 1.0
        total_c = sum(concentrations.values())
        if abs(total_c - 1.0) > 0.01:
            logger.warning(f"Concentrations sum to {total_c:.3f}, normalizing to 1.0")
            concentrations = {el: c / total_c for el, c in concentrations.items()}

        all_observations = []

        T_eV = temperature_K / EV_TO_K
        for element, concentration in concentrations.items():
            # Pre-compute Saha ratios and 3-stage coupled fractions
            ip_I = self._get_ionization_potential(element, 1) or 7.0
            U_I = self._get_partition_function(element, 1, temperature_K)
            U_II = self._get_partition_function(element, 2, temperature_K)
            S1 = (
                (SAHA_CONST_CM3 / electron_density_cm3)
                * (T_eV**1.5)
                * (U_II / U_I)
                * np.exp(-ip_I / T_eV)
            )

            ip_II = self._get_ionization_potential(element, 2)
            S2 = 0.0
            if ip_II is not None:
                U_III = self._get_partition_function(element, 3, temperature_K)
                S2 = (
                    (SAHA_CONST_CM3 / electron_density_cm3)
                    * (T_eV**1.5)
                    * (U_III / U_II)
                    * np.exp(-ip_II / T_eV)
                )
            denom = 1.0 + S1 + S1 * S2
            f_I = 1.0 / denom
            f_II = S1 / denom

            for ion_stage in [1, 2] if include_ionic else [1]:
                transitions = self._get_element_transitions(element, ion_stage, n_lines_per_element)

                # Get partition function (approximate if not in database)
                U = self._get_partition_function(element, ion_stage, temperature_K)

                # Stage fraction: what fraction of this element is in this stage
                stage_fraction = f_I if ion_stage == 1 else f_II

                for trans in transitions:
                    # Calculate intensity using Boltzmann distribution
                    # I ∝ C × f_stage × (g_k × A_ki / U) × exp(-E_k / kT)
                    E_k = trans["E_k_ev"]
                    g_k = trans["g_k"]
                    A_ki = trans["A_ki"]
                    wavelength = trans["wavelength_nm"]

                    boltzmann_factor = np.exp(-E_k / (KB_EV * temperature_K))
                    base_intensity = (
                        concentration * stage_fraction * g_k * A_ki * boltzmann_factor / U
                    )

                    # Scale to reasonable intensity values
                    intensity = base_intensity * 1e-4  # Arbitrary scaling

                    if intensity < min_intensity:
                        continue

                    # Estimate uncertainty (SNR ~ 20-100 for typical LIBS)
                    snr = rng.uniform(20, 100)
                    uncertainty = intensity / snr

                    obs = LineObservation(
                        wavelength_nm=wavelength,
                        intensity=intensity,
                        intensity_uncertainty=uncertainty,
                        element=element,
                        ionization_stage=ion_stage,
                        E_k_ev=E_k,
                        g_k=g_k,
                        A_ki=A_ki,
                    )
                    all_observations.append(obs)

        logger.info(
            f"Generated golden spectrum: T={temperature_K:.0f} K, "
            f"n_e={electron_density_cm3:.2e} cm^-3, "
            f"{len(all_observations)} lines"
        )

        return GoldenSpectrum(
            temperature_K=temperature_K,
            electron_density_cm3=electron_density_cm3,
            concentrations=concentrations,
            line_observations=all_observations,
            seed=seed,
            metadata={
                "n_lines_per_element": n_lines_per_element,
                "include_ionic": include_ionic,
                "min_intensity": min_intensity,
            },
        )

    def _get_element_transitions(self, element: str, ion_stage: int, n_lines: int) -> List[Dict]:
        """Get transitions from database or generate synthetic ones."""
        try:
            transitions = self.atomic_db.get_transitions(
                element,
                wavelength_min=self.wavelength_range[0],
                wavelength_max=self.wavelength_range[1],
                ionization_stage=ion_stage,
                min_relative_intensity=10.0,
            )

            # Convert to dict format and limit number
            result = []
            for t in transitions[:n_lines]:
                result.append(
                    {
                        "wavelength_nm": t.wavelength_nm,
                        "E_k_ev": t.E_k_ev,
                        "g_k": t.g_k,
                        "A_ki": t.A_ki,
                    }
                )

            # Fall back to synthetic if database returned empty
            if not result:
                logger.debug(f"No transitions in DB for {element} {ion_stage}, using synthetic")
                return self._generate_synthetic_transitions(element, ion_stage, n_lines)

            return result

        except Exception as e:
            logger.warning(f"Could not get transitions for {element} {ion_stage}: {e}")
            return self._generate_synthetic_transitions(element, ion_stage, n_lines)

    def _generate_synthetic_transitions(
        self, element: str, ion_stage: int, n_lines: int
    ) -> List[Dict]:
        """Generate synthetic transitions when database is unavailable."""
        # Deterministic seed (avoid process-randomized Python hash()).
        seed = ion_stage + sum((idx + 1) * ord(ch) for idx, ch in enumerate(element))
        rng = np.random.default_rng(seed)

        transitions = []
        for i in range(n_lines):
            # Spread wavelengths across range
            wl = (
                self.wavelength_range[0]
                + (self.wavelength_range[1] - self.wavelength_range[0])
                * (i + rng.random())
                / n_lines
            )

            transitions.append(
                {
                    "wavelength_nm": wl,
                    "E_k_ev": rng.uniform(0.5, 6.0),
                    "g_k": rng.integers(3, 15),
                    "A_ki": 10 ** rng.uniform(6, 8),
                }
            )

        return transitions

    def _get_partition_function(self, element: str, ion_stage: int, temperature_K: float) -> float:
        """Get partition function from database or estimate."""
        try:
            coeffs = self.atomic_db.get_partition_coefficients(element, ion_stage)
            if coeffs:
                from cflibs.plasma.partition import PartitionFunctionEvaluator

                return PartitionFunctionEvaluator.evaluate(temperature_K, coeffs.coefficients)
        except Exception:
            pass

        # Fallback: rough estimates for common elements
        fallback = {
            ("Fe", 1): 25.0,
            ("Fe", 2): 15.0,
            ("Cu", 1): 2.0,
            ("Cu", 2): 1.0,
            ("Al", 1): 6.0,
            ("Al", 2): 1.0,
            ("Ti", 1): 30.0,
            ("Ti", 2): 20.0,
        }
        return fallback.get((element, ion_stage), 10.0)

    def _get_ionization_potential(self, element: str, stage: int = 1) -> float | None:
        """Get ionization potential from database or estimate.

        Parameters
        ----------
        element : str
            Element symbol
        stage : int
            Ionization stage (1 = neutral → singly ionized, 2 = singly → doubly)

        Returns
        -------
        float or None
            Ionization potential in eV, or None if unavailable for stage > 1
        """
        try:
            ip = self.atomic_db.get_ionization_potential(element, stage)
            if ip is not None:
                return ip
        except Exception:
            pass

        if stage == 1:
            # Fallback values for common elements (stage 1 only)
            fallback = {"Fe": 7.87, "Cu": 7.73, "Al": 5.99, "Ti": 6.83, "Mg": 7.65, "Ca": 6.11}
            return fallback.get(element, 7.0)
        return None

    def compute_equilibrium_ne(
        self,
        temperature_K: float,
        concentrations: Dict[str, float],
        pressure_pa: float = 101325.0,
        n_e_guess: float = 1e17,
        max_iter: int = 50,
        tol: float = 1e-3,
    ) -> float:
        """
        Compute self-consistent electron density from pressure balance.

        Iteratively solves:
            n_total = P / (kT × (1 + Z_avg))
            n_e = Z_avg × n_total

        where Z_avg = Σ C_s × S_s / (1 + S_s) depends on n_e via the Saha ratio.

        Parameters
        ----------
        temperature_K : float
            Plasma temperature in Kelvin
        concentrations : Dict[str, float]
            Elemental concentrations (number fractions, sum to 1)
        pressure_pa : float
            Ambient pressure in Pa (default: 1 atm)
        n_e_guess : float
            Initial guess for n_e in cm^-3
        max_iter : int
            Maximum iterations
        tol : float
            Fractional convergence tolerance for n_e

        Returns
        -------
        float
            Self-consistent electron density in cm^-3
        """
        T_eV = temperature_K / EV_TO_K
        n_e = n_e_guess

        # Pre-compute temperature-dependent quantities (constant across iterations)
        element_data = {}
        for el, C_s in concentrations.items():
            ip_I = self._get_ionization_potential(el, 1) or 7.0
            U_I = self._get_partition_function(el, 1, temperature_K)
            U_II = self._get_partition_function(el, 2, temperature_K)
            boltz_I = (T_eV**1.5) * (U_II / U_I) * np.exp(-ip_I / T_eV)

            ip_II = self._get_ionization_potential(el, 2)
            boltz_II = 0.0
            if ip_II is not None:
                U_III = self._get_partition_function(el, 3, temperature_K)
                boltz_II = (T_eV**1.5) * (U_III / U_II) * np.exp(-ip_II / T_eV)
            element_data[el] = (C_s, boltz_I, boltz_II)

        for _ in range(max_iter):
            avg_Z = 0.0
            for el, (C_s, boltz_I, boltz_II) in element_data.items():
                S1 = (SAHA_CONST_CM3 / n_e) * boltz_I
                S2 = (SAHA_CONST_CM3 / n_e) * boltz_II if boltz_II > 0 else 0.0

                # avg_Z = <Z> = (S1 + 2*S1*S2) / (1 + S1 + S1*S2)
                denom = 1.0 + S1 + S1 * S2
                avg_Z += C_s * (S1 + 2.0 * S1 * S2) / denom

            n_total_m3 = pressure_pa / (KB * temperature_K * (1.0 + avg_Z))
            n_total_cm3 = n_total_m3 * 1e-6
            n_e_new = avg_Z * n_total_cm3

            if n_e_new > 0 and abs(n_e_new - n_e) / n_e < tol:
                return n_e_new
            n_e = 0.5 * n_e + 0.5 * n_e_new

        logger.warning(
            "compute_equilibrium_ne did not converge after %d iterations " "(last n_e=%.3e cm^-3)",
            max_iter,
            n_e,
        )
        return n_e


class NoiseModel:
    """
    Apply realistic noise to synthetic spectra.

    LIBS noise sources:
    1. Shot noise (Poisson): √I
    2. Readout noise (Gaussian): constant
    3. Background noise: additive
    4. Multiplicative noise (laser fluctuations): proportional to I

    Parameters
    ----------
    shot_noise : bool
        Enable Poisson shot noise
    readout_noise : float
        Gaussian readout noise (standard deviation)
    background : float
        Constant background level
    multiplicative_noise : float
        Fractional laser fluctuation noise
    """

    def __init__(
        self,
        shot_noise: bool = True,
        readout_noise: float = 5.0,
        background: float = 10.0,
        multiplicative_noise: float = 0.02,
    ):
        self.shot_noise = shot_noise
        self.readout_noise = readout_noise
        self.background = background
        self.multiplicative_noise = multiplicative_noise

    def apply(
        self,
        golden: GoldenSpectrum,
        seed: Optional[int] = None,
    ) -> GoldenSpectrum:
        """
        Apply noise to a golden spectrum.

        Parameters
        ----------
        golden : GoldenSpectrum
            Clean synthetic spectrum
        seed : int, optional
            Random seed (defaults to golden.seed + 1)

        Returns
        -------
        GoldenSpectrum
            New spectrum with noisy observations
        """
        if seed is None:
            seed = golden.seed + 1

        rng = np.random.default_rng(seed)

        noisy_observations = []
        for obs in golden.line_observations:
            intensity = obs.intensity

            # Add background
            intensity += self.background

            # Apply multiplicative noise (laser fluctuations)
            if self.multiplicative_noise > 0:
                intensity *= 1.0 + rng.normal(0, self.multiplicative_noise)

            # Apply shot noise (Poisson)
            if self.shot_noise and intensity > 0:
                # Poisson variance = mean, so std = sqrt(I)
                shot_std = np.sqrt(max(intensity, 1.0))
                intensity += rng.normal(0, shot_std)

            # Apply readout noise (Gaussian)
            if self.readout_noise > 0:
                intensity += rng.normal(0, self.readout_noise)

            # Ensure positive
            intensity = max(intensity, 1.0)

            # Update uncertainty to reflect added noise
            # Total variance = shot + readout^2 + (mult*I)^2
            var_shot = max(intensity, 1.0) if self.shot_noise else 0
            var_readout = self.readout_noise**2
            var_mult = (self.multiplicative_noise * intensity) ** 2
            total_uncertainty = np.sqrt(var_shot + var_readout + var_mult)

            noisy_obs = LineObservation(
                wavelength_nm=obs.wavelength_nm,
                intensity=intensity,
                intensity_uncertainty=total_uncertainty,
                element=obs.element,
                ionization_stage=obs.ionization_stage,
                E_k_ev=obs.E_k_ev,
                g_k=obs.g_k,
                A_ki=obs.A_ki,
            )
            noisy_observations.append(noisy_obs)

        return GoldenSpectrum(
            temperature_K=golden.temperature_K,
            electron_density_cm3=golden.electron_density_cm3,
            concentrations=golden.concentrations,
            line_observations=noisy_observations,
            seed=seed,
            metadata={**golden.metadata, "noise_applied": True},
        )


class RoundTripValidator:
    """
    Validate CF-LIBS pipeline through round-trip testing.

    The validation process:
    1. Generate golden spectrum with known parameters
    2. Add realistic noise
    3. Run CF-LIBS inversion
    4. Compare recovered parameters to ground truth

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database
    temperature_tolerance : float
        Fractional tolerance for temperature (default: 0.05 = 5%)
    density_tolerance : float
        Fractional tolerance for electron density (default: 0.20 = 20%)
    concentration_tolerance : float
        Fractional tolerance for concentrations (default: 0.10 = 10%)
        This default was tightened from 0.15 in March 2026 to align the
        round-trip validator with the Phase 7 synthetic recovery target. Pass
        ``concentration_tolerance=0.15`` explicitly to preserve the earlier
        behavior in existing workflows.
    """

    def __init__(
        self,
        atomic_db,
        temperature_tolerance: float = 0.05,
        density_tolerance: float = 0.20,
        concentration_tolerance: float = 0.10,
    ):
        """
        Create a RoundTripValidator that performs end-to-end validation of CF-LIBS by generating synthetic spectra, adding noise, and running the inversion within specified tolerances.

        Parameters:
            atomic_db: Atomic database used to query transition and partition-function data for spectrum generation and inversion.
            temperature_tolerance (float): Allowed fractional error for recovered temperature (e.g., 0.05 = 5%).
            density_tolerance (float): Allowed fractional error for recovered electron density (e.g., 0.20 = 20%).
            concentration_tolerance (float): Allowed fractional error for recovered elemental concentrations (e.g., 0.10 = 10%). The default is tightened to 0.10; pass 0.15 to preserve the prior, less strict behavior.

        Behavior:
            Stores the provided tolerances and atomic database reference and initializes a GoldenSpectrumGenerator and a NoiseModel used by validation runs.
        """
        self.atomic_db = atomic_db
        self.temperature_tolerance = temperature_tolerance
        self.density_tolerance = density_tolerance
        self.concentration_tolerance = concentration_tolerance

        self.generator = GoldenSpectrumGenerator(atomic_db)
        self.noise_model = NoiseModel()

    def validate(
        self,
        temperature_K: float,
        electron_density_cm3: float,
        concentrations: Dict[str, float],
        seed: int = 42,
        n_lines_per_element: int = 10,
        closure_mode: str = "standard",
        add_noise: bool = True,
        **solver_kwargs,
    ) -> RoundTripResult:
        """
        Run a complete round-trip validation.

        Parameters
        ----------
        temperature_K : float
            True plasma temperature
        electron_density_cm3 : float
            True electron density
        concentrations : Dict[str, float]
            True elemental concentrations
        seed : int
            Random seed for reproducibility
        n_lines_per_element : int
            Lines per element to generate
        closure_mode : str
            Closure mode for solver
        add_noise : bool
            Whether to add noise to spectrum
        **solver_kwargs
            Additional arguments for solver

        Returns
        -------
        RoundTripResult
            Validation results with pass/fail status
        """
        # Generate golden spectrum
        golden = self.generator.generate(
            temperature_K=temperature_K,
            electron_density_cm3=electron_density_cm3,
            concentrations=concentrations,
            n_lines_per_element=n_lines_per_element,
            seed=seed,
        )

        # Apply noise if requested
        if add_noise:
            golden = self.noise_model.apply(golden)

        # Run inversion
        from cflibs.inversion.solver import IterativeCFLIBSSolver

        solver = IterativeCFLIBSSolver(self.atomic_db, **solver_kwargs)

        try:
            result = solver.solve(golden.line_observations, closure_mode=closure_mode)
        except Exception as e:
            logger.error(f"Inversion failed: {e}")
            # Return a failed result
            return RoundTripResult(
                true_temperature_K=temperature_K,
                recovered_temperature_K=0.0,
                temperature_error_frac=float("inf"),
                true_electron_density=electron_density_cm3,
                recovered_electron_density=0.0,
                electron_density_error_frac=float("inf"),
                true_concentrations=concentrations,
                recovered_concentrations={},
                concentration_errors={el: float("inf") for el in concentrations},
                converged=False,
                iterations=0,
                passed=False,
                tolerances={
                    "temperature": self.temperature_tolerance,
                    "electron_density": self.density_tolerance,
                    "concentration": self.concentration_tolerance,
                },
            )

        # Calculate errors
        temp_error = abs(result.temperature_K - temperature_K) / temperature_K
        ne_error = abs(result.electron_density_cm3 - electron_density_cm3) / electron_density_cm3

        conc_errors = {}
        for el, true_c in concentrations.items():
            rec_c = result.concentrations.get(el, 0.0)
            if true_c > 0.01:  # Only check significant concentrations
                conc_errors[el] = abs(rec_c - true_c) / true_c
            else:
                conc_errors[el] = 0.0

        # Check if all tolerances are met
        passed = (
            temp_error <= self.temperature_tolerance
            and ne_error <= self.density_tolerance
            and all(e <= self.concentration_tolerance for e in conc_errors.values())
            and result.converged
        )

        tolerances = {
            "temperature": self.temperature_tolerance,
            "electron_density": self.density_tolerance,
            "concentration": self.concentration_tolerance,
        }

        return RoundTripResult(
            true_temperature_K=temperature_K,
            recovered_temperature_K=result.temperature_K,
            temperature_error_frac=temp_error,
            true_electron_density=electron_density_cm3,
            recovered_electron_density=result.electron_density_cm3,
            electron_density_error_frac=ne_error,
            true_concentrations=concentrations,
            recovered_concentrations=result.concentrations,
            concentration_errors=conc_errors,
            converged=result.converged,
            iterations=result.iterations,
            passed=passed,
            tolerances=tolerances,
        )

    def run_parameter_sweep(
        self,
        temperature_range: Tuple[float, float] = (8000.0, 15000.0),
        density_range: Tuple[float, float] = (1e16, 1e18),
        n_temperatures: int = 3,
        n_densities: int = 3,
        concentrations: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> List[RoundTripResult]:
        """
        Run validation across a parameter sweep.

        Parameters
        ----------
        temperature_range : tuple
            (min, max) temperature in K
        density_range : tuple
            (min, max) electron density in cm^-3
        n_temperatures : int
            Number of temperature points
        n_densities : int
            Number of density points
        concentrations : dict
            Fixed concentrations (default: Fe=0.7, Cu=0.3)
        seed : int
            Base random seed

        Returns
        -------
        List[RoundTripResult]
            Results for each parameter combination
        """
        if concentrations is None:
            concentrations = {"Fe": 0.7, "Cu": 0.3}

        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_temperatures)
        densities = np.logspace(np.log10(density_range[0]), np.log10(density_range[1]), n_densities)

        results = []
        for i, T in enumerate(temperatures):
            for j, n_e in enumerate(densities):
                result = self.validate(
                    temperature_K=T,
                    electron_density_cm3=n_e,
                    concentrations=concentrations,
                    seed=seed + i * n_densities + j,
                )
                results.append(result)
                logger.info(
                    f"Sweep [{i*n_densities+j+1}/{n_temperatures*n_densities}]: "
                    f"T={T:.0f}K, n_e={n_e:.1e} -> {'PASS' if result.passed else 'FAIL'}"
                )

        return results
