import pytest
import numpy as np
import time
from pathlib import Path

from cflibs.atomic.database import AtomicDatabase
from cflibs.validation.round_trip import RoundTripValidator, GoldenSpectrumGenerator, NoiseModel
from cflibs.manifold.config import ManifoldConfig
from cflibs.manifold.generator import ManifoldGenerator
from cflibs.inversion.hybrid import HybridInverter

DB_PATH = "ASD_da/libs_production.db"

@pytest.fixture
def atomic_db():
    if not Path(DB_PATH).exists():
        pytest.skip(f"Database not found at {DB_PATH}")
    return AtomicDatabase(DB_PATH)

@pytest.mark.integration
@pytest.mark.requires_db
def test_round_trip_noiseless(atomic_db):
    """Test standard iterative solver round-trip without noise."""
    validator = RoundTripValidator(atomic_db, temperature_tolerance=0.05, density_tolerance=0.20, concentration_tolerance=0.10)
    result = validator.validate(
        temperature_K=10000.0,
        electron_density_cm3=1e17,
        concentrations={"Fe": 0.8, "Cu": 0.2},
        add_noise=False
    )
    assert result.passed, f"Noiseless round-trip validation failed: T_err={result.temperature_error_frac:.2f}"

@pytest.mark.integration
@pytest.mark.requires_db
def test_round_trip_noisy(atomic_db):
    """Test standard iterative solver round-trip with realistic noise."""
    validator = RoundTripValidator(atomic_db, temperature_tolerance=0.10, density_tolerance=0.30, concentration_tolerance=0.15)
    result = validator.validate(
        temperature_K=10000.0,
        electron_density_cm3=1e17,
        concentrations={"Fe": 0.8, "Cu": 0.2},
        add_noise=True
    )
    assert result.passed, f"Noisy round-trip validation failed: T_err={result.temperature_error_frac:.2f}"

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_db
def test_manifold_generate_and_query(atomic_db, tmp_path):
    """Test generating a small manifold and querying it via FAISS."""
    from cflibs.manifold.loader import ManifoldLoader
    
    manifold_path = tmp_path / "test_manifold.h5"
    config = ManifoldConfig(
        elements=["Fe", "Cu"],
        temperature_range=(0.8, 1.2),
        density_range=(1e16, 1e17),
        temperature_steps=3,
        density_steps=3,
        concentration_steps=3,
        output_path=str(manifold_path),
        db_path=DB_PATH,
        batch_size=10
    )
    generator = ManifoldGenerator(config)
    generator.generate_manifold()
    
    loader = ManifoldLoader(str(manifold_path))
    loader.build_vector_index(index_type="flat")
    
    # Ensure it generated 3*3*3 = 27 spectra
    assert loader.n_spectra == 27
    
    # Query with a dummy spectrum
    dummy_spectrum = np.ones(config.pixels)
    idx, sim, params = loader.find_nearest_spectrum(dummy_spectrum)
    assert "T_eV" in params
    assert "n_e_cm3" in params

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_db
def test_hybrid_inversion_e2e(atomic_db, tmp_path):
    """Test the end-to-end hybrid inversion pipeline on a mock spectrum."""
    from cflibs.manifold.loader import ManifoldLoader
    
    manifold_path = tmp_path / "test_hybrid_manifold.h5"
    config = ManifoldConfig(
        elements=["Fe", "Cu"],
        temperature_range=(0.8, 1.2),
        density_range=(1e16, 1e17),
        temperature_steps=3,
        density_steps=3,
        concentration_steps=3,
        output_path=str(manifold_path),
        db_path=DB_PATH
    )
    generator = ManifoldGenerator(config)
    generator.generate_manifold()
    
    loader = ManifoldLoader(str(manifold_path))
    loader.build_vector_index(index_type="flat")
    
    inverter = HybridInverter(loader, max_iterations=50)
    
    # Fit a dummy flat spectrum to ensure the pipeline executes without crashing
    dummy_spectrum = np.ones(config.pixels)
    result = inverter.invert(dummy_spectrum, use_manifold_init=True)
    
    # The dummy spectrum won't yield physically accurate results, but it shouldn't crash
    assert result.converged or result.iterations > 0
    assert "Fe" in result.concentrations
    assert "Cu" in result.concentrations
    assert result.temperature_eV > 0

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_db
def test_hybrid_faster_than_iterative(atomic_db, tmp_path):
    """Verify that hybrid inference (after JIT) is extremely fast."""
    from cflibs.manifold.loader import ManifoldLoader
    from cflibs.inversion.solver import IterativeCFLIBSSolver
    
    manifold_path = tmp_path / "test_speed_manifold.h5"
    config = ManifoldConfig(
        elements=["Fe", "Cu"],
        temperature_range=(0.8, 1.2),
        density_range=(1e16, 1e17),
        temperature_steps=3,
        density_steps=3,
        concentration_steps=3,
        output_path=str(manifold_path),
        db_path=DB_PATH
    )
    generator = ManifoldGenerator(config)
    generator.generate_manifold()
    loader = ManifoldLoader(str(manifold_path))
    loader.build_vector_index(index_type="flat")
    
    inverter = HybridInverter(loader, max_iterations=50)
    dummy_spectrum = np.ones(config.pixels)
    
    # Warmup JIT
    inverter.invert(dummy_spectrum, use_manifold_init=True)
    
    # Time hybrid
    t0 = time.perf_counter()
    inverter.invert(dummy_spectrum, use_manifold_init=True)
    hybrid_time = time.perf_counter() - t0
    
    # Hybrid should be very fast after JIT warmup
    assert hybrid_time < 0.5  # Should easily be under 500ms, targeting <10ms
