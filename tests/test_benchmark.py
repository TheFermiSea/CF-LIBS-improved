"""
Tests for the benchmark module.

Tests cover:
- BenchmarkSpectrum and BenchmarkDataset data structures
- Evaluation metrics (RMSEP, MAE, bias, R-squared)
- Synthetic benchmark generation
- JSON and HDF5 I/O
- Train/test splits and k-fold CV
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cflibs.benchmark.dataset import (
    BenchmarkSpectrum,
    BenchmarkDataset,
    InstrumentalConditions,
    SampleMetadata,
    SampleType,
    MatrixType,
    DataSplit,
    SUPPORTED_ELEMENTS,
)
from cflibs.benchmark.metrics import (
    BenchmarkMetrics,
    EvaluationResult,
    ElementMetrics,
    MetricType,
    calculate_figure_of_merit,
)
from cflibs.benchmark.synthetic import (
    SyntheticBenchmarkGenerator,
    CompositionRange,
    ConditionVariation,
    create_steel_benchmark,
    create_geological_benchmark,
)
from cflibs.benchmark.loaders import (
    load_benchmark,
    save_benchmark,
    BenchmarkFormat,
    validate_benchmark_file,
)
from cflibs.plasma.state import mass_fractions_to_species_densities

# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def sample_conditions():
    """Create sample instrumental conditions."""
    return InstrumentalConditions(
        laser_wavelength_nm=1064.0,
        laser_energy_mj=50.0,
        laser_pulse_width_ns=8.0,
        gate_delay_us=1.0,
        gate_width_us=10.0,
        spectrometer_type="Echelle",
        spectral_range_nm=(200.0, 800.0),
        spectral_resolution_nm=0.05,
        detector_type="ICCD",
        accumulations=10,
        atmosphere="air",
    )


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return SampleMetadata(
        sample_id="test_sample_001",
        sample_type=SampleType.CRM,
        matrix_type=MatrixType.METAL_ALLOY,
        crm_name="NIST SRM 1261a",
        crm_source="NIST",
        preparation="polished",
        surface_condition="polished",
    )


@pytest.fixture
def sample_spectrum(sample_conditions, sample_metadata):
    """Create a sample benchmark spectrum."""
    n_points = 1000
    wavelength = np.linspace(200.0, 800.0, n_points)
    intensity = np.random.default_rng(42).exponential(100, n_points)

    return BenchmarkSpectrum(
        spectrum_id="test_spectrum_001",
        wavelength_nm=wavelength,
        intensity=intensity,
        true_composition={"Fe": 0.70, "Cu": 0.20, "Mn": 0.10},
        composition_uncertainty={"Fe": 0.01, "Cu": 0.005, "Mn": 0.002},
        conditions=sample_conditions,
        metadata=sample_metadata,
        plasma_temperature_K=10000.0,
        electron_density_cm3=1e17,
    )


@pytest.fixture
def sample_dataset(sample_conditions, sample_metadata):
    """Create a sample benchmark dataset with multiple spectra."""
    spectra = []
    rng = np.random.default_rng(42)

    for i in range(10):
        n_points = 1000
        wavelength = np.linspace(200.0, 800.0, n_points)
        intensity = rng.exponential(100, n_points)

        # Vary compositions
        fe = 0.60 + rng.uniform(0, 0.30)
        cu = 0.10 + rng.uniform(0, 0.15)
        mn = 1.0 - fe - cu  # Balance

        metadata = SampleMetadata(
            sample_id=f"sample_{i:03d}",
            sample_type=SampleType.CRM,
            matrix_type=MatrixType.METAL_ALLOY,
        )

        spec = BenchmarkSpectrum(
            spectrum_id=f"spectrum_{i:03d}",
            wavelength_nm=wavelength,
            intensity=intensity,
            true_composition={"Fe": fe, "Cu": cu, "Mn": mn},
            conditions=sample_conditions,
            metadata=metadata,
        )
        spectra.append(spec)

    return BenchmarkDataset(
        name="test_dataset",
        version="1.0.0",
        spectra=spectra,
        elements=["Fe", "Cu", "Mn"],
        description="Test dataset",
    )


# =============================================================================
# Test InstrumentalConditions
# =============================================================================


class TestInstrumentalConditions:
    """Tests for InstrumentalConditions dataclass."""

    def test_creation(self, sample_conditions):
        """Test basic creation."""
        assert sample_conditions.laser_wavelength_nm == 1064.0
        assert sample_conditions.laser_energy_mj == 50.0
        assert sample_conditions.atmosphere == "air"

    def test_fluence_calculation(self):
        """Test automatic fluence calculation."""
        cond = InstrumentalConditions(
            laser_wavelength_nm=1064.0,
            laser_energy_mj=50.0,
            spot_diameter_um=100.0,
        )
        # Fluence = 50mJ / (pi * (50um)^2) = 50e-3 / (pi * (50e-4)^2) cm
        # = 50e-3 / (pi * 25e-8) = 50e-3 / 7.85e-7 = 63.7 J/cm^2
        assert cond.fluence_j_cm2 is not None
        assert cond.fluence_j_cm2 > 0

    def test_to_dict_from_dict(self, sample_conditions):
        """Test round-trip serialization."""
        d = sample_conditions.to_dict()
        assert isinstance(d, dict)
        assert d["laser_wavelength_nm"] == 1064.0

        restored = InstrumentalConditions.from_dict(d)
        assert restored.laser_wavelength_nm == sample_conditions.laser_wavelength_nm
        assert restored.spectral_range_nm == sample_conditions.spectral_range_nm


# =============================================================================
# Test SampleMetadata
# =============================================================================


class TestSampleMetadata:
    """Tests for SampleMetadata dataclass."""

    def test_creation(self, sample_metadata):
        """Test basic creation."""
        assert sample_metadata.sample_id == "test_sample_001"
        assert sample_metadata.sample_type == SampleType.CRM
        assert sample_metadata.crm_name == "NIST SRM 1261a"

    def test_to_dict_from_dict(self, sample_metadata):
        """Test round-trip serialization."""
        d = sample_metadata.to_dict()
        assert d["sample_type"] == "crm"
        assert d["matrix_type"] == "metal_alloy"

        restored = SampleMetadata.from_dict(d)
        assert restored.sample_type == SampleType.CRM
        assert restored.matrix_type == MatrixType.METAL_ALLOY


# =============================================================================
# Test BenchmarkSpectrum
# =============================================================================


class TestBenchmarkSpectrum:
    """Tests for BenchmarkSpectrum dataclass."""

    def test_creation(self, sample_spectrum):
        """Test basic creation."""
        assert sample_spectrum.spectrum_id == "test_spectrum_001"
        assert sample_spectrum.n_points == 1000
        assert len(sample_spectrum.elements) == 3
        assert "Fe" in sample_spectrum.elements

    def test_composition_validation(self, sample_conditions, sample_metadata, caplog):
        """Test that composition validation logs warning on non-unity sum."""
        import logging

        wavelength = np.linspace(200, 800, 100)
        intensity = np.ones(100)

        # This should log a warning (composition sums to 0.5)
        with caplog.at_level(logging.WARNING):
            BenchmarkSpectrum(
                spectrum_id="test",
                wavelength_nm=wavelength,
                intensity=intensity,
                true_composition={"Fe": 0.3, "Cu": 0.2},
                conditions=sample_conditions,
                metadata=sample_metadata,
            )

        # Check that the warning was logged
        assert "composition sums to" in caplog.text

    def test_array_shape_validation(self, sample_conditions, sample_metadata):
        """Test that mismatched array shapes raise error."""
        wavelength = np.linspace(200, 800, 100)
        intensity = np.ones(50)  # Different length

        with pytest.raises(ValueError, match="same length"):
            BenchmarkSpectrum(
                spectrum_id="test",
                wavelength_nm=wavelength,
                intensity=intensity,
                true_composition={"Fe": 1.0},
                conditions=sample_conditions,
                metadata=sample_metadata,
            )

    def test_wavelength_range(self, sample_spectrum):
        """Test wavelength range property."""
        wl_min, wl_max = sample_spectrum.wavelength_range
        assert wl_min == pytest.approx(200.0)
        assert wl_max == pytest.approx(800.0)

    def test_get_composition_with_uncertainty(self, sample_spectrum):
        """Test getting composition with uncertainty."""
        value, uncertainty = sample_spectrum.get_composition_with_uncertainty("Fe")
        assert value == 0.70
        assert uncertainty == 0.01

        # Non-existent element
        value, uncertainty = sample_spectrum.get_composition_with_uncertainty("Zn")
        assert value == 0.0
        assert uncertainty == 0.0

    def test_to_dict_from_dict(self, sample_spectrum):
        """Test round-trip serialization."""
        d = sample_spectrum.to_dict()
        assert isinstance(d, dict)
        assert d["spectrum_id"] == "test_spectrum_001"
        assert len(d["wavelength_nm"]) == 1000

        restored = BenchmarkSpectrum.from_dict(d)
        assert restored.spectrum_id == sample_spectrum.spectrum_id
        assert np.allclose(restored.wavelength_nm, sample_spectrum.wavelength_nm)
        assert restored.true_composition == sample_spectrum.true_composition


# =============================================================================
# Test BenchmarkDataset
# =============================================================================


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset class."""

    def test_creation(self, sample_dataset):
        """Test basic creation."""
        assert sample_dataset.name == "test_dataset"
        assert sample_dataset.n_spectra == 10
        assert sample_dataset.elements == ["Fe", "Cu", "Mn"]

    def test_get_spectrum(self, sample_dataset):
        """Test getting spectrum by ID."""
        spec = sample_dataset.get_spectrum("spectrum_000")
        assert spec.spectrum_id == "spectrum_000"

        with pytest.raises(KeyError):
            sample_dataset.get_spectrum("nonexistent")

    def test_create_random_split(self, sample_dataset):
        """Test creating random train/test split."""
        split = sample_dataset.create_random_split(
            name="test_split",
            train_fraction=0.7,
            test_fraction=0.3,
            random_seed=42,
        )

        assert split.name == "test_split"
        assert split.train_size == 7
        assert split.test_size == 3
        assert "test_split" in sample_dataset.splits

    def test_get_split(self, sample_dataset):
        """Test getting train/test split."""
        sample_dataset.create_random_split("default", 0.7, 0.3, random_seed=42)

        train, test = sample_dataset.get_split("default")
        assert len(train) == 7
        assert len(test) == 3

        with pytest.raises(KeyError):
            sample_dataset.get_split("nonexistent")

    def test_create_kfold_splits(self, sample_dataset):
        """Test creating k-fold cross-validation splits."""
        splits = sample_dataset.create_kfold_splits(n_folds=5, random_seed=42)

        assert len(splits) == 5
        for i, split in enumerate(splits):
            assert split.name == f"fold_{i+1}"
            assert split.test_size == 2  # 10 / 5 = 2

    def test_filter_by_quality(self, sample_dataset):
        """Test filtering by quality flag."""
        # Set some quality flags
        sample_dataset.spectra[0].quality_flag = 2
        sample_dataset.spectra[1].quality_flag = 1

        filtered = sample_dataset.filter_by_quality(max_flag=0)
        assert filtered.n_spectra == 8

    def test_get_composition_matrix(self, sample_dataset):
        """Test getting composition matrix."""
        compositions, elements, ids = sample_dataset.get_composition_matrix()

        assert compositions.shape == (10, 3)
        assert len(elements) == 3
        assert len(ids) == 10

        # Check composition sums approximately to 1
        row_sums = compositions.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=0.01)

    def test_summary(self, sample_dataset):
        """Test summary generation."""
        summary = sample_dataset.summary()
        assert "test_dataset" in summary
        assert "10" in summary  # Number of spectra
        assert "Fe" in summary

    def test_to_dict_from_dict(self, sample_dataset):
        """Test round-trip serialization."""
        sample_dataset.create_random_split("default", 0.7, 0.3)

        d = sample_dataset.to_dict()
        assert d["name"] == "test_dataset"
        assert len(d["spectra"]) == 10

        restored = BenchmarkDataset.from_dict(d)
        assert restored.name == sample_dataset.name
        assert restored.n_spectra == sample_dataset.n_spectra


# =============================================================================
# Test BenchmarkMetrics
# =============================================================================


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics class."""

    def test_rmsep_calculation(self):
        """Test RMSEP calculation."""
        pred = np.array([0.10, 0.20, 0.30])
        true = np.array([0.11, 0.19, 0.31])

        rmsep = BenchmarkMetrics.rmsep(pred, true)
        expected = np.sqrt(np.mean([0.01**2, 0.01**2, 0.01**2]))
        assert rmsep == pytest.approx(expected)

    def test_mae_calculation(self):
        """Test MAE calculation."""
        pred = np.array([0.10, 0.20, 0.30])
        true = np.array([0.12, 0.18, 0.32])

        mae = BenchmarkMetrics.mae(pred, true)
        expected = np.mean([0.02, 0.02, 0.02])
        assert mae == pytest.approx(expected)

    def test_bias_calculation(self):
        """Test bias calculation."""
        pred = np.array([0.10, 0.20, 0.30])
        true = np.array([0.08, 0.18, 0.28])

        bias = BenchmarkMetrics.bias(pred, true)
        assert bias == pytest.approx(0.02)  # Positive = over-estimation

    def test_r_squared_calculation(self):
        """Test R-squared calculation."""
        # Perfect prediction
        true = np.array([0.1, 0.2, 0.3, 0.4])
        pred = true.copy()

        r2 = BenchmarkMetrics.r_squared(pred, true)
        assert r2 == pytest.approx(1.0)

        # Poor prediction
        pred = np.array([0.25, 0.25, 0.25, 0.25])  # Mean value
        r2 = BenchmarkMetrics.r_squared(pred, true)
        assert r2 == pytest.approx(0.0)

    def test_evaluate(self):
        """Test full evaluation."""
        metrics = BenchmarkMetrics()

        predictions = {
            "Fe": [0.70, 0.75, 0.65],
            "Cu": [0.20, 0.18, 0.22],
        }
        true_values = {
            "Fe": [0.72, 0.73, 0.68],
            "Cu": [0.18, 0.20, 0.20],
        }

        result = metrics.evaluate(
            predictions,
            true_values,
            dataset_name="test",
            split_name="default",
            algorithm_name="test_algo",
        )

        assert result.n_spectra == 3
        assert result.n_elements == 2
        assert "Fe" in result.elements
        assert result.overall_rmsep > 0

    def test_element_metrics(self):
        """Test ElementMetrics properties."""
        em = ElementMetrics(
            element="Fe",
            n_samples=10,
            rmsep=0.01,
            mae=0.008,
            mape=5.0,
            bias=0.002,
            r_squared=0.95,
            pearson_r=0.98,
            lod=0.001,
            loq=0.0033,
            relative_rmsep=0.015,
            true_range=(0.6, 0.9),
        )

        assert em.rmsep_pct == pytest.approx(1.0)
        assert em.mae_pct == pytest.approx(0.8)
        assert em.lod_ppm == pytest.approx(1000)

    def test_compare_algorithms(self):
        """Test algorithm comparison."""
        metrics = BenchmarkMetrics()

        # Create two results
        result1 = EvaluationResult(
            dataset_name="test",
            split_name="default",
            n_spectra=10,
            element_metrics={
                "Fe": ElementMetrics(
                    "Fe", 10, 0.01, 0.008, 5.0, 0.002, 0.95, 0.98, 0.001, 0.003, 0.015, (0.6, 0.9)
                ),
            },
            overall_rmsep=0.01,
            overall_mae=0.008,
            overall_r_squared=0.95,
            algorithm_name="algo_a",
            algorithm_version="1.0",
        )

        result2 = EvaluationResult(
            dataset_name="test",
            split_name="default",
            n_spectra=10,
            element_metrics={
                "Fe": ElementMetrics(
                    "Fe", 10, 0.02, 0.015, 8.0, 0.005, 0.85, 0.92, 0.002, 0.006, 0.030, (0.6, 0.9)
                ),
            },
            overall_rmsep=0.02,
            overall_mae=0.015,
            overall_r_squared=0.85,
            algorithm_name="algo_b",
            algorithm_version="1.0",
        )

        comparison = metrics.compare_algorithms([result1, result2], metric=MetricType.RMSEP)

        assert "algo_a v1.0" in comparison
        assert comparison["algo_a v1.0"] < comparison["algo_b v1.0"]


class TestFigureOfMerit:
    """Tests for figure of merit calculation."""

    def test_calculate_figure_of_merit(self):
        """Test FOM calculation."""
        result = EvaluationResult(
            dataset_name="test",
            split_name="default",
            n_spectra=10,
            element_metrics={},
            overall_rmsep=0.01,
            overall_mae=0.008,
            overall_r_squared=0.95,
        )

        fom = calculate_figure_of_merit(result)
        assert fom > 0

        # Better results should have lower FOM
        result2 = EvaluationResult(
            dataset_name="test",
            split_name="default",
            n_spectra=10,
            element_metrics={},
            overall_rmsep=0.005,
            overall_mae=0.004,
            overall_r_squared=0.99,
        )

        fom2 = calculate_figure_of_merit(result2)
        assert fom2 < fom


# =============================================================================
# Test SyntheticBenchmarkGenerator
# =============================================================================


class TestSyntheticBenchmarkGenerator:
    """Tests for synthetic benchmark generation."""

    def test_composition_range_sampling(self):
        """Test CompositionRange sampling."""
        rng = np.random.default_rng(42)

        # Uniform
        cr = CompositionRange("Fe", 0.5, 0.8, distribution="uniform")
        samples = [cr.sample(rng) for _ in range(100)]
        assert all(0.5 <= s <= 0.8 for s in samples)

        # Fixed
        cr = CompositionRange("Fe", 0.5, 0.8, distribution="fixed", fixed_value=0.7)
        assert cr.sample(rng) == 0.7

    def test_composition_range_validation(self):
        """Test CompositionRange validation."""
        with pytest.raises(ValueError):
            CompositionRange("Fe", 0.8, 0.5)  # min > max

        with pytest.raises(ValueError):
            CompositionRange("Fe", 0.5, 0.8, distribution="fixed")  # No fixed_value

    def test_condition_variation_sampling(self):
        """Test ConditionVariation sampling."""
        rng = np.random.default_rng(42)
        cv = ConditionVariation(
            temperature_range_K=(8000, 12000),
            electron_density_range_cm3=(1e16, 1e18),
        )

        T = cv.sample_temperature(rng)
        assert 8000 <= T <= 12000

        ne = cv.sample_electron_density(rng)
        assert 1e16 <= ne <= 1e18

    def test_generate_basic(self):
        """Test basic dataset generation."""
        generator = SyntheticBenchmarkGenerator()

        composition_ranges = [
            CompositionRange("Fe", 0.6, 0.9),
            CompositionRange("Cu", 0.05, 0.2),
        ]

        dataset = generator.generate(
            n_spectra=10,
            composition_ranges=composition_ranges,
            name="test_synthetic",
            seed=42,
        )

        assert dataset.n_spectra == 10
        assert "Fe" in dataset.elements
        assert "Cu" in dataset.elements
        assert "default" in dataset.splits

    def test_generate_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        generator = SyntheticBenchmarkGenerator()

        composition_ranges = [
            CompositionRange("Fe", 0.6, 0.9),
            CompositionRange("Cu", 0.1, 0.3),
        ]

        dataset1 = generator.generate(
            n_spectra=5,
            composition_ranges=composition_ranges,
            seed=42,
        )

        dataset2 = generator.generate(
            n_spectra=5,
            composition_ranges=composition_ranges,
            seed=42,
        )

        # Same compositions
        for i in range(5):
            assert dataset1.spectra[i].true_composition == pytest.approx(
                dataset2.spectra[i].true_composition, rel=1e-10
            )

    def test_create_steel_benchmark(self):
        """Test steel benchmark convenience function."""
        dataset = create_steel_benchmark(n_spectra=20, seed=42)

        assert dataset.n_spectra == 20
        assert "Fe" in dataset.elements
        assert "Cr" in dataset.elements
        assert dataset.name == "synthetic_steel_benchmark"

    def test_create_geological_benchmark(self):
        """Test geological benchmark convenience function."""
        dataset = create_geological_benchmark(n_spectra=20, seed=42)

        assert dataset.n_spectra == 20
        assert "Si" in dataset.elements
        assert "Al" in dataset.elements
        assert dataset.name == "synthetic_geological_benchmark"

    def test_generate_uses_forward_model_when_atomic_db_available(self, atomic_db, monkeypatch):
        """Ensure forward-model path is exercised when atomic DB is provided."""
        generator = SyntheticBenchmarkGenerator(
            atomic_db=atomic_db,
            wavelength_range=(370.0, 375.0),
            n_wavelength_points=200,
        )

        # If forward model fails and code falls back, this test should fail.
        def _fail_simplified(*_args, **_kwargs):
            raise AssertionError("Unexpected fallback to simplified synthetic model")

        monkeypatch.setattr(generator, "_generate_simplified", _fail_simplified)

        composition_ranges = [
            CompositionRange("Fe", 1.0, 1.0, distribution="fixed", fixed_value=1.0),
        ]
        dataset = generator.generate(
            n_spectra=1,
            composition_ranges=composition_ranges,
            seed=7,
            create_default_split=False,
        )

        spectrum = dataset.spectra[0]
        assert spectrum.intensity.shape == (200,)
        assert np.all(np.isfinite(spectrum.intensity))
        assert np.max(spectrum.intensity) > 0

    def test_composition_to_number_density_conversion(self):
        """Mass-fraction composition should convert to normalized number densities."""
        densities = SyntheticBenchmarkGenerator._composition_to_number_densities(
            {"Fe": 0.5, "Cu": 0.5},
            total_number_density_cm3=1.0e17,
        )
        assert np.isclose(sum(densities.values()), 1.0e17, rtol=1e-10)
        # For equal mass fraction, lighter element has larger number fraction.
        assert densities["Fe"] > densities["Cu"]

    def test_composition_to_number_density_uses_plasma_semantics_helper(self):
        """Benchmark conversion should agree with the shared plasma-state helper."""
        densities = SyntheticBenchmarkGenerator._composition_to_number_densities(
            {"Fe": 0.5, "Cu": 0.5},
            total_number_density_cm3=1.0e17,
        )
        expected = mass_fractions_to_species_densities(
            {"Fe": 0.5, "Cu": 0.5},
            total_number_density_cm3=1.0e17,
            atomic_masses_amu={"Fe": 55.85, "Cu": 63.55},
        )
        assert densities == pytest.approx(expected)


# =============================================================================
# Test I/O
# =============================================================================


class TestBenchmarkIO:
    """Tests for benchmark I/O functions."""

    def test_save_load_json(self, sample_dataset):
        """Test JSON save/load round-trip."""
        sample_dataset.create_random_split("default", 0.7, 0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dataset.json"

            save_benchmark(sample_dataset, path)
            assert path.exists()

            loaded = load_benchmark(path)
            assert loaded.name == sample_dataset.name
            assert loaded.n_spectra == sample_dataset.n_spectra
            assert "default" in loaded.splits

    def test_save_load_hdf5(self, sample_dataset):
        """Test HDF5 save/load round-trip."""
        pytest.importorskip("h5py")

        sample_dataset.create_random_split("default", 0.7, 0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dataset.h5"

            save_benchmark(sample_dataset, path, format=BenchmarkFormat.HDF5)
            assert path.exists()

            loaded = load_benchmark(path)
            assert loaded.name == sample_dataset.name
            assert loaded.n_spectra == sample_dataset.n_spectra

            # Check array data preserved
            orig_spec = sample_dataset.spectra[0]
            loaded_spec = loaded.get_spectrum(orig_spec.spectrum_id)
            assert np.allclose(orig_spec.wavelength_nm, loaded_spec.wavelength_nm)
            assert np.allclose(orig_spec.intensity, loaded_spec.intensity)

    def test_format_auto_detection(self, sample_dataset):
        """Test automatic format detection from extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test.json"
            save_benchmark(sample_dataset, json_path)

            loaded = load_benchmark(json_path)
            assert loaded.name == sample_dataset.name

    def test_validate_benchmark_file(self, sample_dataset):
        """Test benchmark file validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_benchmark(sample_dataset, path)

            results = validate_benchmark_file(path)
            assert results["file_exists"]
            assert results["readable"]
            assert results["has_spectra"]

            # Non-existent file
            results = validate_benchmark_file(Path(tmpdir) / "nonexistent.json")
            assert not results["file_exists"]


# =============================================================================
# Test DataSplit
# =============================================================================


class TestDataSplit:
    """Tests for DataSplit dataclass."""

    def test_creation(self):
        """Test basic creation."""
        split = DataSplit(
            name="test",
            train_ids=["a", "b", "c"],
            test_ids=["d", "e"],
            random_seed=42,
        )

        assert split.train_size == 3
        assert split.test_size == 2
        assert split.validation_size == 0

    def test_with_validation(self):
        """Test split with validation set."""
        split = DataSplit(
            name="test",
            train_ids=["a", "b", "c"],
            test_ids=["d", "e"],
            validation_ids=["f"],
            random_seed=42,
        )

        assert split.validation_size == 1

    def test_to_dict_from_dict(self):
        """Test round-trip serialization."""
        split = DataSplit(
            name="test",
            train_ids=["a", "b"],
            test_ids=["c"],
            description="Test split",
            random_seed=42,
        )

        d = split.to_dict()
        restored = DataSplit.from_dict(d)

        assert restored.name == split.name
        assert restored.train_ids == split.train_ids
        assert restored.random_seed == split.random_seed


# =============================================================================
# Test supported elements
# =============================================================================


class TestSupportedElements:
    """Tests for supported elements list."""

    def test_common_elements_present(self):
        """Test that common LIBS elements are in the list."""
        common = ["Fe", "Cu", "Al", "Si", "Ca", "Mg", "Na", "K", "Ti", "Mn", "Cr", "Ni"]
        for elem in common:
            assert elem in SUPPORTED_ELEMENTS

    def test_no_duplicates(self):
        """Test that there are no duplicate elements."""
        assert len(SUPPORTED_ELEMENTS) == len(set(SUPPORTED_ELEMENTS))


# =============================================================================
# Integration tests
# =============================================================================


class TestBenchmarkIntegration:
    """Integration tests for the benchmark module."""

    def test_full_workflow(self):
        """Test complete benchmark workflow: generate, split, evaluate, save."""
        # 1. Generate synthetic dataset
        generator = SyntheticBenchmarkGenerator()
        composition_ranges = [
            CompositionRange("Fe", 0.6, 0.85),
            CompositionRange("Cu", 0.05, 0.20),
            CompositionRange("Mn", 0.02, 0.10),
        ]

        dataset = generator.generate(
            n_spectra=50,
            composition_ranges=composition_ranges,
            name="integration_test",
            seed=42,
        )

        # 2. Get train/test split
        train, test = dataset.get_split("default")
        assert len(train) + len(test) == 50

        # 3. Simulate predictions (with small random error)
        rng = np.random.default_rng(42)
        predictions = {}
        true_values = {}

        for elem in dataset.elements:
            true_values[elem] = [s.true_composition[elem] for s in test]
            # Add 5% relative error
            predictions[elem] = [t * (1 + 0.05 * rng.standard_normal()) for t in true_values[elem]]

        # 4. Evaluate
        metrics = BenchmarkMetrics()
        result = metrics.evaluate(
            predictions,
            true_values,
            dataset_name=dataset.name,
            split_name="default",
            algorithm_name="simulated",
        )

        assert result.overall_rmsep < 0.1  # Reasonable for 5% error
        assert result.overall_r_squared > 0.8

        # 5. Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "integration_test.json"
            save_benchmark(dataset, path)

            loaded = load_benchmark(path)
            assert loaded.n_spectra == dataset.n_spectra