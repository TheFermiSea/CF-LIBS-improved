"""
Pytest configuration and shared fixtures for CF-LIBS tests.

This module provides:
- Automatic JAX CPU backend configuration (Metal is abandoned, incompatible with JAX >= 0.6)
- Shared fixtures for database, plasma, and atomic data
- Factory fixtures for generating synthetic test data
"""

import os
import pytest
import numpy as np
import sqlite3
import tempfile
from pathlib import Path

# Force JAX to use CPU backend before any JAX imports.
# Metal backend is abandoned (jax-metal incompatible with JAX >= 0.6)
# and does not support float64, which CF-LIBS requires.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

from cflibs.atomic.structures import Transition, EnergyLevel
from cflibs.atomic.database import AtomicDatabase
from cflibs.plasma.state import SingleZoneLTEPlasma


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked ``requires_*`` when their optional dependency is missing
    or broken.

    The ``requires_*`` markers declared in pytest.ini are descriptive labels; they
    do not skip on their own. This hook makes them functional: for each marker
    listed below, probe the dependency once during collection and attach a
    ``pytest.mark.skip`` to every item carrying that marker when the probe fails.

    The probe has two parts: (1) the bare module imports, and (2) any cflibs
    flag the production code uses to decide whether the dependency is usable.
    Both must succeed — on Python 3.12 we have seen environments where the
    top-level ``import numpyro`` succeeds but cflibs's deeper
    ``from numpyro.infer import MCMC, NUTS, init_to_uniform`` block raises
    ``ImportError`` (so ``cflibs.inversion.solve.bayesian.HAS_NUMPYRO`` is
    ``False``). A bare-module-only probe would pass and the test would then
    crash with cflibs's own ``ImportError("NumPyro required")`` instead of
    being skipped.

    Catches any ``Exception`` (not just ``ImportError``) because real-world
    failure modes for optional native deps include broken-wheel
    ``RuntimeError``s, partial installs that surface as ``AttributeError``,
    and ``OSError`` from missing shared libraries. A failed probe must never
    abort collection.

    Closes CF-LIBS-improved-48c2.
    """
    # marker_name -> (probes, hint)
    # probes is a list of (module_name, attr) pairs:
    #   - if attr is None, just import the module
    #   - if attr is a string, also assert getattr(module, attr) is truthy
    optional_deps: dict[str, tuple[list[tuple[str, str | None]], str]] = {
        "requires_jax": ([("jax", None)], 'pip install ".[jax-cpu]"'),
        "requires_bayesian": (
            [
                ("numpyro", None),
                ("cflibs.inversion.solve.bayesian", "HAS_NUMPYRO"),
            ],
            'pip install ".[bayesian]"',
        ),
        "requires_uncertainty": ([("uncertainties", None)], 'pip install ".[uncertainty]"'),
        "requires_rust": ([("cflibs._core", None)], 'compile cflibs-core rust extension'),
    }
    missing: dict[str, tuple[str, str, str]] = {}
    for marker_name, (probes, hint) in optional_deps.items():
        for module_name, attr in probes:
            try:
                module = __import__(module_name, fromlist=["__all__"])
                if attr is not None and not getattr(module, attr, False):
                    missing[marker_name] = (module_name, hint, f"{attr}=False")
                    break
            except Exception as exc:  # noqa: BLE001 — see docstring rationale
                missing[marker_name] = (module_name, hint, type(exc).__name__)
                break
    # Probe database files
    candidates = [
        Path("libs_production.db"),
        Path("ASD_da/libs_production.db"),
        Path(__file__).parent.parent / "libs_production.db",
        Path(__file__).parent.parent / "ASD_da" / "libs_production.db",
    ]
    if not any(p.exists() for p in candidates):
        missing["requires_db"] = ("ASD_da/libs_production.db", "download atomic database", "missing_file")

    if not missing:
        return
    for item in items:
        for marker_name, (module_name, hint, exc_name) in missing.items():
            if item.get_closest_marker(marker_name) is not None:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"{marker_name}: {module_name} unavailable " f"({exc_name}); {hint}"
                        )
                    )
                )
                break


@pytest.fixture(scope="session")
def production_db():
    """Session-scoped production database fixture.

    Tries several common paths for the real atomic database.
    Skips if not found.
    """
    candidates = [
        Path("libs_production.db"),
        Path("ASD_da/libs_production.db"),
        Path(__file__).parent.parent / "libs_production.db",
        Path(__file__).parent.parent / "ASD_da" / "libs_production.db",
    ]
    for p in candidates:
        if p.exists():
            return AtomicDatabase(str(p))
    pytest.skip("Production database not found")


@pytest.fixture
def temp_db():
    """Create a temporary atomic database for testing."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)  # Close file descriptor to prevent leaks

    conn = sqlite3.connect(db_path)

    # Create tables
    conn.execute("""
        CREATE TABLE lines (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            wavelength_nm REAL,
            aki REAL,
            ei_ev REAL,
            ek_ev REAL,
            gi INTEGER,
            gk INTEGER,
            rel_int REAL
        )
    """)

    conn.execute("""
        CREATE TABLE energy_levels (
            element TEXT,
            sp_num INTEGER,
            g_level INTEGER,
            energy_ev REAL
        )
    """)

    conn.execute("""
        CREATE TABLE species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            PRIMARY KEY (element, sp_num)
        )
    """)

    # Insert test data
    # Fe I (neutral iron) - lines spanning a wide energy range for Boltzmann fitting
    conn.execute("""
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES ('Fe', 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000),
               ('Fe', 1, 373.49, 5.0e6, 0.0, 3.32, 9, 9, 500),
               ('Fe', 1, 374.95, 2.0e6, 0.0, 3.31, 9, 7, 200),
               ('Fe', 1, 382.04, 6.7e7, 0.0, 3.24, 9, 9, 800),
               ('Fe', 1, 404.58, 8.6e6, 0.86, 3.93, 7, 9, 600),
               ('Fe', 1, 427.18, 2.2e7, 1.49, 4.39, 5, 7, 700),
               ('Fe', 1, 438.35, 5.0e6, 1.61, 4.44, 9, 9, 800),
               ('Fe', 1, 495.76, 4.2e6, 2.18, 4.68, 7, 7, 300),
               ('Fe', 1, 516.75, 5.7e6, 0.05, 2.45, 11, 9, 400),
               ('Fe', 1, 532.80, 1.1e6, 0.91, 3.24, 5, 5, 250)
    """)

    # Fe II (singly ionized iron)
    conn.execute("""
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES ('Fe', 2, 238.20, 3.0e8, 0.0, 5.20, 10, 10, 900),
               ('Fe', 2, 259.94, 2.2e8, 0.05, 4.82, 8, 8, 700),
               ('Fe', 2, 273.95, 2.0e8, 0.99, 5.51, 8, 6, 500),
               ('Fe', 2, 234.35, 1.1e8, 0.0, 5.29, 10, 8, 600)
    """)

    conn.execute("""
        INSERT INTO energy_levels (element, sp_num, g_level, energy_ev)
        VALUES ('Fe', 1, 9, 0.0),
               ('Fe', 1, 11, 0.05),
               ('Fe', 1, 7, 0.86),
               ('Fe', 1, 5, 1.49),
               ('Fe', 1, 9, 1.61),
               ('Fe', 1, 7, 2.18),
               ('Fe', 1, 11, 3.33),
               ('Fe', 1, 9, 3.32),
               ('Fe', 1, 7, 3.31),
               ('Fe', 1, 9, 3.24),
               ('Fe', 1, 9, 3.93),
               ('Fe', 1, 7, 4.39),
               ('Fe', 1, 9, 4.44),
               ('Fe', 1, 7, 4.68),
               ('Fe', 1, 5, 2.45),
               ('Fe', 2, 10, 0.0),
               ('Fe', 2, 8, 0.05),
               ('Fe', 2, 8, 0.99),
               ('Fe', 2, 10, 5.20),
               ('Fe', 2, 8, 4.82),
               ('Fe', 2, 6, 5.51),
               ('Fe', 2, 8, 5.29)
    """)

    conn.execute("""
        INSERT INTO species_physics (element, sp_num, ip_ev)
        VALUES ('Fe', 1, 7.87),
               ('Fe', 2, 16.18)
    """)

    # H I (hydrogen)
    conn.execute("""
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES ('H', 1, 656.28, 4.4e7, 0.0, 12.75, 2, 8, 10000),
               ('H', 1, 486.13, 8.4e6, 0.0, 12.75, 2, 8, 2000)
    """)

    conn.execute("""
        INSERT INTO energy_levels (element, sp_num, g_level, energy_ev)
        VALUES ('H', 1, 2, 0.0),
               ('H', 1, 8, 12.75)
    """)

    conn.execute("""
        INSERT INTO species_physics (element, sp_num, ip_ev)
        VALUES ('H', 1, 13.60)
    """)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    Path(db_path).unlink()


@pytest.fixture
def atomic_db(temp_db):
    """Create AtomicDatabase instance from temp database."""
    return AtomicDatabase(temp_db)


@pytest.fixture
def sample_plasma():
    """Create a sample plasma state for testing."""
    return SingleZoneLTEPlasma(
        T_e=10000.0,
        n_e=1e17,
        species={
            "Fe": 1e15,
            "H": 1e16,
        },
    )


@pytest.fixture
def sample_transition():
    """Create a sample transition for testing."""
    return Transition(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=371.99,
        A_ki=1.0e7,
        E_k_ev=3.33,
        E_i_ev=0.0,
        g_k=11,
        g_i=9,
        relative_intensity=1000.0,
    )


@pytest.fixture
def sample_energy_level():
    """Create a sample energy level for testing."""
    return EnergyLevel(element="Fe", ionization_stage=1, energy_ev=3.33, g=11)


@pytest.fixture
def sample_wavelength_grid():
    """Create a sample wavelength grid for testing."""
    return np.linspace(200.0, 800.0, 1000)


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary."""
    return {
        "atomic_database": "libs_production.db",
        "plasma": {
            "model": "single_zone_lte",
            "Te": 10000.0,
            "ne": 1.0e17,
            "species": [
                {"element": "Fe", "number_density": 1.0e15},
                {"element": "H", "number_density": 1.0e16},
            ],
        },
        "instrument": {"resolution_fwhm_nm": 0.05},
        "spectrum": {
            "lambda_min_nm": 200.0,
            "lambda_max_nm": 800.0,
            "delta_lambda_nm": 0.01,
            "path_length_m": 0.01,
        },
    }


@pytest.fixture
def temp_config_file(sample_config_dict):
    """Create a temporary YAML config file."""
    import yaml

    config_fd, config_path = tempfile.mkstemp(suffix=".yaml")
    os.close(config_fd)  # Close file descriptor to prevent leaks

    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)

    yield config_path

    Path(config_path).unlink()


@pytest.fixture
def mock_echellogram_image():
    """Create a mock 2D echellogram image for testing."""
    image = np.random.normal(100, 10, (1024, 2048)).astype(np.float32)
    # Add some bright spectral lines
    image[595:605, 995:1005] += 5000
    image[715:725, 995:1005] += 5000
    return image


# ==============================================================================
# Phase 2c Fixtures - Quality, Self-Absorption, Line Selection
# ==============================================================================

from cflibs.inversion.boltzmann import LineObservation, BoltzmannFitResult  # noqa: E402


@pytest.fixture
def synthetic_line_observations():
    """
    Factory fixture for creating synthetic LineObservation lists.

    Returns a function that generates line observations with known parameters.
    """

    def _create(
        n_lines: int = 10,
        element: str = "Fe",
        ionization_stage: int = 1,
        temperature_K: float = 10000.0,
        base_intensity: float = 1000.0,
        snr: float = 50.0,
        seed: int = 42,
    ) -> list:
        """
        Generate synthetic line observations following Boltzmann distribution.

        Parameters
        ----------
        n_lines : int
            Number of lines to generate
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage (1=neutral, 2=singly ionized)
        temperature_K : float
            Plasma temperature in K
        base_intensity : float
            Base intensity scaling
        snr : float
            Signal-to-noise ratio
        seed : int
            Random seed for reproducibility

        Returns
        -------
        List[LineObservation]
        """
        rng = np.random.default_rng(seed)

        # Physical constants
        KB_EV = 8.617333262e-5  # eV/K

        # Generate upper level energies spread across 0-5 eV
        E_k_values = np.linspace(0.5, 5.0, n_lines)

        observations = []
        for i, E_k in enumerate(E_k_values):
            # Wavelength: spread across 300-600 nm
            wavelength = 300.0 + (300.0 * i / n_lines)

            # Statistical weight and A coefficient
            g_k = rng.integers(5, 15)
            A_ki = 10 ** rng.uniform(6, 8)  # 10^6 to 10^8 s^-1

            # Intensity from Boltzmann distribution
            # I ∝ g_k * A_ki * exp(-E_k / kT)
            boltzmann_factor = np.exp(-E_k / (KB_EV * temperature_K))
            intensity = base_intensity * g_k * A_ki * boltzmann_factor / 1e7

            # Add noise
            uncertainty = intensity / snr
            intensity += rng.normal(0, uncertainty)
            intensity = max(intensity, 1.0)  # Ensure positive

            observations.append(
                LineObservation(
                    wavelength_nm=wavelength,
                    intensity=intensity,
                    intensity_uncertainty=uncertainty,
                    element=element,
                    ionization_stage=ionization_stage,
                    E_k_ev=E_k,
                    g_k=int(g_k),
                    A_ki=A_ki,
                )
            )

        return observations

    return _create


@pytest.fixture
def mock_boltzmann_fit_result():
    """
    Factory fixture for creating mock BoltzmannFitResult objects.

    Returns a function that creates results with specified quality parameters.
    """

    def _create(
        temperature_K: float = 10000.0,
        r_squared: float = 0.95,
        n_points: int = 10,
        slope_uncertainty_frac: float = 0.05,
    ) -> BoltzmannFitResult:
        """
        Create a mock BoltzmannFitResult.

        Parameters
        ----------
        temperature_K : float
            Fitted temperature
        r_squared : float
            R² value (0-1)
        n_points : int
            Number of data points
        slope_uncertainty_frac : float
            Fractional uncertainty in slope

        Returns
        -------
        BoltzmannFitResult
        """
        KB_EV = 8.617333262e-5
        slope = -1.0 / (KB_EV * temperature_K)
        slope_uncertainty = abs(slope) * slope_uncertainty_frac
        temp_uncertainty = temperature_K * slope_uncertainty_frac

        return BoltzmannFitResult(
            temperature_K=temperature_K,
            temperature_uncertainty_K=temp_uncertainty,
            intercept=-10.0,
            intercept_uncertainty=0.5,
            r_squared=r_squared,
            n_points=n_points,
            rejected_points=[],
            slope=slope,
            slope_uncertainty=slope_uncertainty,
        )

    return _create


@pytest.fixture
def self_absorption_test_line():
    """
    Factory fixture for creating lines with known optical depth.

    Returns a function that generates absorbed and unabsorbed intensity pairs.
    """

    def _create(
        optical_depth: float = 1.0,
        original_intensity: float = 1000.0,
        wavelength_nm: float = 400.0,
        element: str = "Fe",
    ) -> dict:
        """
        Create a test line with known optical depth.

        The absorption formula is: f(τ) = (1 - exp(-τ)) / τ
        absorbed_intensity = original_intensity * f(τ)

        Parameters
        ----------
        optical_depth : float
            Optical depth τ at line center
        original_intensity : float
            True (unabsorbed) intensity
        wavelength_nm : float
            Line wavelength
        element : str
            Element symbol

        Returns
        -------
        dict with keys: original_intensity, absorbed_intensity, optical_depth,
                       correction_factor, wavelength_nm, element, observation
        """
        # Calculate correction factor f(τ)
        if optical_depth < 1e-10:
            f_tau = 1.0
        elif optical_depth > 50:
            f_tau = 1.0 / optical_depth
        else:
            f_tau = (1.0 - np.exp(-optical_depth)) / optical_depth

        absorbed_intensity = original_intensity * f_tau

        # Create LineObservation with absorbed intensity
        observation = LineObservation(
            wavelength_nm=wavelength_nm,
            intensity=absorbed_intensity,
            intensity_uncertainty=absorbed_intensity * 0.02,
            element=element,
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=9,
            A_ki=1e7,
        )

        return {
            "original_intensity": original_intensity,
            "absorbed_intensity": absorbed_intensity,
            "optical_depth": optical_depth,
            "correction_factor": f_tau,
            "wavelength_nm": wavelength_nm,
            "element": element,
            "observation": observation,
        }

    return _create


@pytest.fixture
def line_selector_test_data():
    """
    Factory fixture for creating lines with known scores.

    Returns a function that generates observations with controllable SNR,
    isolation, and atomic uncertainty.
    """

    def _create(
        n_lines: int = 5,
        snr_values: "list | None" = None,
        isolation_values: "list | None" = None,
        uncertainty_values: "list | None" = None,
        element: str = "Fe",
    ) -> list:
        """
        Create line observations with known scoring parameters.

        Parameters
        ----------
        n_lines : int
            Number of lines
        snr_values : list, optional
            SNR for each line (defaults to 50 for all)
        isolation_values : list, optional
            Wavelength separations in nm (affects isolation factor)
        uncertainty_values : list, optional
            Atomic data uncertainties (0-1)
        element : str
            Element symbol

        Returns
        -------
        List[LineObservation]
        """
        if snr_values is None:
            snr_values = [50.0] * n_lines
        if isolation_values is None:
            isolation_values = [1.0] * n_lines  # Well-separated
        if uncertainty_values is None:
            uncertainty_values = [0.10] * n_lines

        observations = []
        base_wavelength = 400.0

        for i in range(n_lines):
            snr = snr_values[i] if i < len(snr_values) else 50.0
            separation = isolation_values[i] if i < len(isolation_values) else 1.0

            intensity = 1000.0
            uncertainty = intensity / snr

            observations.append(
                LineObservation(
                    wavelength_nm=base_wavelength + i * separation,
                    intensity=intensity,
                    intensity_uncertainty=uncertainty,
                    element=element,
                    ionization_stage=1,
                    E_k_ev=1.0 + i * 0.5,
                    g_k=9,
                    A_ki=1e7,
                )
            )

        return observations

    return _create


@pytest.fixture
def quality_input_set(synthetic_line_observations, mock_boltzmann_fit_result):
    """
    Factory fixture for complete QualityAssessor input sets.

    Returns a function that creates observations and related parameters
    for different quality levels.
    """

    def _create(quality_level: str = "excellent") -> dict:
        """
        Create a complete input set for QualityAssessor.

        Parameters
        ----------
        quality_level : str
            One of "excellent", "good", "acceptable", "poor"

        Returns
        -------
        dict with all inputs needed for QualityAssessor.assess()
        """
        # Map quality levels to R² and other parameters
        params = {
            "excellent": {"r_squared": 0.98, "temperature_K": 10000.0},
            "good": {"r_squared": 0.92, "temperature_K": 10000.0},
            "acceptable": {"r_squared": 0.85, "temperature_K": 10000.0},
            "poor": {"r_squared": 0.65, "temperature_K": 10000.0},
        }

        p = params.get(quality_level, params["good"])
        T = p["temperature_K"]

        observations = synthetic_line_observations(
            n_lines=10,
            element="Fe",
            temperature_K=T,
            snr=100.0 if quality_level == "excellent" else 30.0,
        )

        # Add a second element for inter-element consistency tests
        observations.extend(
            synthetic_line_observations(
                n_lines=5,
                element="Cu",
                temperature_K=T * (1.0 if quality_level == "excellent" else 1.1),
                snr=100.0 if quality_level == "excellent" else 30.0,
                seed=123,
            )
        )

        # Concentrations that sum to ~1.0
        concentrations = {"Fe": 0.7, "Cu": 0.3}
        if quality_level == "poor":
            concentrations = {"Fe": 0.6, "Cu": 0.2}  # Sum to 0.8, not 1.0

        return {
            "observations": observations,
            "temperature_K": T,
            "electron_density_cm3": 1e17,
            "concentrations": concentrations,
            "ionization_potentials": {"Fe": 7.87, "Cu": 7.73},
            "partition_funcs_I": {"Fe": 25.0, "Cu": 2.0},
            "partition_funcs_II": {"Fe": 15.0, "Cu": 1.0},
            "expected_quality": quality_level,
        }

    return _create


@pytest.fixture
def sample_atomic_transitions():
    """
    Fixture providing realistic atomic transition data.

    Returns a list of LineObservation objects for common LIBS elements.
    """
    transitions = [
        # Fe I lines
        LineObservation(
            wavelength_nm=371.99,
            intensity=1000.0,
            intensity_uncertainty=20.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.33,
            g_k=11,
            A_ki=1.0e7,
        ),
        LineObservation(
            wavelength_nm=373.49,
            intensity=500.0,
            intensity_uncertainty=15.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.32,
            g_k=9,
            A_ki=5.0e6,
        ),
        LineObservation(
            wavelength_nm=374.95,
            intensity=200.0,
            intensity_uncertainty=10.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.31,
            g_k=7,
            A_ki=2.0e6,
        ),
        LineObservation(
            wavelength_nm=438.35,
            intensity=800.0,
            intensity_uncertainty=18.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=4.47,
            g_k=9,
            A_ki=5.0e6,
        ),
        # Fe II line
        LineObservation(
            wavelength_nm=238.20,
            intensity=600.0,
            intensity_uncertainty=25.0,
            element="Fe",
            ionization_stage=2,
            E_k_ev=5.22,
            g_k=10,
            A_ki=3.0e8,
        ),
        # Cu I lines
        LineObservation(
            wavelength_nm=324.75,
            intensity=2000.0,
            intensity_uncertainty=40.0,
            element="Cu",
            ionization_stage=1,
            E_k_ev=3.82,
            g_k=4,
            A_ki=1.4e8,
        ),
        LineObservation(
            wavelength_nm=327.40,
            intensity=1000.0,
            intensity_uncertainty=30.0,
            element="Cu",
            ionization_stage=1,
            E_k_ev=3.79,
            g_k=2,
            A_ki=1.4e8,
        ),
        # Al I lines
        LineObservation(
            wavelength_nm=394.40,
            intensity=1500.0,
            intensity_uncertainty=35.0,
            element="Al",
            ionization_stage=1,
            E_k_ev=3.14,
            g_k=2,
            A_ki=5.0e7,
        ),
        LineObservation(
            wavelength_nm=396.15,
            intensity=750.0,
            intensity_uncertainty=25.0,
            element="Al",
            ionization_stage=1,
            E_k_ev=3.14,
            g_k=4,
            A_ki=1.0e8,
        ),
    ]
    return transitions


@pytest.fixture
def synthetic_libs_spectrum():
    """Factory fixture for creating synthetic LIBS spectra with known peaks."""

    def _create(
        elements=None,
        temperature_K=10000.0,
        wavelength_range=(200.0, 800.0),
        n_points=6000,
        noise_level=0.01,
        fwhm_nm=0.15,
        seed=42,
    ):
        """
        Generate synthetic LIBS spectrum with Gaussian peaks at known wavelengths.

        Parameters
        ----------
        elements : dict, optional
            Dict mapping element symbols to list of (wavelength_nm, amplitude) tuples.
            Default: {"Fe": [(371.99, 1000.0), (373.49, 500.0), (374.95, 200.0)],
                      "H": [(656.28, 5000.0), (486.13, 1000.0)]}
        temperature_K : float
            Plasma temperature in K (metadata only)
        wavelength_range : tuple
            (min_nm, max_nm) wavelength range
        n_points : int
            Number of wavelength points
        noise_level : float
            Relative noise level (0.01 = 1%)
        fwhm_nm : float
            Full width at half maximum for Gaussian peaks in nm
        seed : int
            Random seed for reproducibility

        Returns
        -------
        dict
            Dictionary with keys:
            - "wavelength": ndarray of wavelengths
            - "intensity": ndarray of intensities
            - "elements": dict of element peaks
            - "peaks": list of (element, wavelength_nm, amplitude) tuples
            - "temperature_K": plasma temperature (metadata)
        """
        if elements is None:
            elements = {
                "Fe": [(371.99, 1000.0), (373.49, 500.0), (374.95, 200.0)],
                "H": [(656.28, 5000.0), (486.13, 1000.0)],
            }

        rng = np.random.default_rng(seed)
        wl_min, wl_max = wavelength_range
        wavelength = np.linspace(wl_min, wl_max, n_points)
        intensity = np.ones_like(wavelength) * 10.0  # baseline

        sigma = fwhm_nm / 2.3548  # FWHM to sigma
        all_peaks = []
        for element, peaks in elements.items():
            for center_nm, amplitude in peaks:
                if wl_min <= center_nm <= wl_max:
                    peak = amplitude * np.exp(-0.5 * ((wavelength - center_nm) / sigma) ** 2)
                    intensity += peak
                    all_peaks.append((element, center_nm, amplitude))

        intensity += rng.normal(0, noise_level * np.max(intensity), size=n_points)
        intensity = np.maximum(intensity, 0.0)

        return {
            "wavelength": wavelength,
            "intensity": intensity,
            "elements": elements,
            "peaks": all_peaks,
            "temperature_K": temperature_K,
        }

    return _create
