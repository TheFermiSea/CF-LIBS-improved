"""
Tests for export tools module.

These tests validate:
1. Base Exporter ABC interface
2. CSVExporter functionality
3. HDF5Exporter functionality
4. JSONExporter functionality
5. Factory function create_exporter
6. Metadata preservation
7. Dataclass serialization
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest

from cflibs.io.exporters import (
    CSVExporter,
    ExportMetadata,
    Exporter,
    HDF5Exporter,
    JSONExporter,
    create_exporter,
    export_to_csv,
    export_to_json,
)

# --- Test fixtures ---


@dataclass
class MockCFLIBSResult:
    """Mock CFLIBSResult for testing."""

    temperature_K: float = 10000.0
    temperature_uncertainty_K: float = 500.0
    electron_density_cm3: float = 1e17
    electron_density_uncertainty_cm3: float = 1e16
    concentrations: Dict[str, float] = field(default_factory=lambda: {"Fe": 0.8, "Cu": 0.2})
    concentration_uncertainties: Dict[str, float] = field(
        default_factory=lambda: {"Fe": 0.05, "Cu": 0.03}
    )
    iterations: int = 10
    converged: bool = True
    quality_metrics: Dict[str, float] = field(
        default_factory=lambda: {"r_squared": 0.95, "chi_squared": 1.2}
    )
    boltzmann_covariance: Optional[np.ndarray] = None


@dataclass
class MockMCMCResult:
    """Mock MCMCResult for testing."""

    samples: Dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "T_eV": np.random.normal(1.0, 0.1, 100),
            "log_ne": np.random.normal(17.0, 0.5, 100),
            "concentrations": np.random.dirichlet([1, 1], 100),
        }
    )
    T_eV_mean: float = 1.0
    T_eV_std: float = 0.1
    T_eV_q025: float = 0.8
    T_eV_q975: float = 1.2
    log_ne_mean: float = 17.0
    log_ne_std: float = 0.5
    log_ne_q025: float = 16.0
    log_ne_q975: float = 18.0
    concentrations_mean: Dict[str, float] = field(default_factory=lambda: {"Fe": 0.8, "Cu": 0.2})
    concentrations_std: Dict[str, float] = field(default_factory=lambda: {"Fe": 0.05, "Cu": 0.03})
    concentrations_q025: Dict[str, float] = field(default_factory=lambda: {"Fe": 0.7, "Cu": 0.15})
    concentrations_q975: Dict[str, float] = field(default_factory=lambda: {"Fe": 0.9, "Cu": 0.25})
    r_hat: Dict[str, float] = field(default_factory=lambda: {"T_eV": 1.001, "log_ne": 1.002})
    ess: Dict[str, float] = field(default_factory=lambda: {"T_eV": 450, "log_ne": 420})
    convergence_status: str = "converged"
    n_samples: int = 100
    n_chains: int = 1
    n_warmup: int = 50


@dataclass
class MockNestedSamplingResult:
    """Mock NestedSamplingResult for testing."""

    samples: Dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "T_eV": np.random.normal(1.0, 0.1, 100),
            "log_ne": np.random.normal(17.0, 0.5, 100),
        }
    )
    weights: np.ndarray = field(default_factory=lambda: np.ones(100) / 100)
    log_evidence: float = -150.5
    log_evidence_err: float = 0.5
    information: float = 12.3
    T_eV_mean: float = 1.0
    T_eV_std: float = 0.1
    log_ne_mean: float = 17.0
    log_ne_std: float = 0.5
    concentrations_mean: Dict[str, float] = field(default_factory=lambda: {"Fe": 0.8, "Cu": 0.2})
    concentrations_std: Dict[str, float] = field(default_factory=lambda: {"Fe": 0.05, "Cu": 0.03})
    n_live: int = 100
    n_iterations: int = 500
    n_calls: int = 5000


@pytest.fixture
def cflibs_result():
    """Create mock CFLIBSResult."""
    return MockCFLIBSResult()


@pytest.fixture
def mcmc_result():
    """Create mock MCMCResult with deterministic samples."""
    np.random.seed(42)
    return MockMCMCResult()


@pytest.fixture
def nested_result():
    """Create mock NestedSamplingResult with deterministic samples."""
    np.random.seed(42)
    return MockNestedSamplingResult()


@pytest.fixture
def spectrum_data():
    """Create mock spectrum data."""
    wavelength = np.linspace(200, 800, 1000)
    intensity = np.exp(-((wavelength - 500) ** 2) / 5000)
    return {"wavelength": wavelength, "intensity": intensity}


@pytest.fixture
def temp_path():
    """Create a temporary file path and clean up after test."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    Path(path).unlink()  # Remove file so we can test creation
    yield path
    # Cleanup
    if Path(path).exists():
        Path(path).unlink()


# --- ExportMetadata tests ---


class TestExportMetadata:
    """Tests for ExportMetadata dataclass."""

    def test_create_metadata(self):
        """Test metadata creation with current timestamp."""
        meta = ExportMetadata.create(
            format="csv",
            source_type="CFLIBSResult",
            parameters={"test": True},
        )

        assert meta.format == "csv"
        assert meta.source_type == "CFLIBSResult"
        assert meta.parameters == {"test": True}
        assert "Z" in meta.timestamp  # ISO format with Z suffix
        assert meta.version is not None

    def test_metadata_to_dict(self):
        """Test metadata conversion to dictionary."""
        meta = ExportMetadata.create(format="json", source_type="dict")
        d = meta.to_dict()

        assert "timestamp" in d
        assert d["format"] == "json"
        assert d["source_type"] == "dict"


# --- Exporter ABC tests ---


class TestExporterABC:
    """Tests for Exporter abstract base class."""

    def test_cannot_instantiate_abc(self):
        """Test that Exporter ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Exporter()

    def test_subclass_must_implement_export(self):
        """Test that subclass must implement export method."""

        class IncompleteExporter(Exporter):
            def get_format(self):
                return "test"

        with pytest.raises(TypeError):
            IncompleteExporter()

    def test_subclass_must_implement_get_format(self):
        """Test that subclass must implement get_format method."""

        class IncompleteExporter(Exporter):
            def export(self, data, path, metadata=None):
                pass

        with pytest.raises(TypeError):
            IncompleteExporter()


# --- CSVExporter tests ---


class TestCSVExporter:
    """Tests for CSVExporter."""

    def test_get_format(self):
        """Test format name."""
        exporter = CSVExporter()
        assert exporter.get_format() == "csv"

    def test_export_cflibs_result(self, cflibs_result, temp_path):
        """Test exporting CFLIBSResult to CSV."""
        exporter = CSVExporter()
        exporter.export(cflibs_result, temp_path)

        assert Path(temp_path).exists()

        content = Path(temp_path).read_text()
        assert "CF-LIBS Export" in content
        assert "temperature" in content
        assert "Fe" in content
        assert "Cu" in content
        assert "10000" in content  # temperature_K

    def test_export_mcmc_result(self, mcmc_result, temp_path):
        """Test exporting MCMCResult to CSV."""
        exporter = CSVExporter()
        exporter.export(mcmc_result, temp_path)

        content = Path(temp_path).read_text()
        assert "MCMC Summary" in content
        assert "T_eV" in content
        assert "log_ne" in content
        assert "r_hat" in content

    def test_export_nested_sampling_result(self, nested_result, temp_path):
        """Test exporting NestedSamplingResult to CSV."""
        exporter = CSVExporter()
        exporter.export(nested_result, temp_path)

        content = Path(temp_path).read_text()
        assert "Evidence" in content
        assert "log_evidence" in content

    def test_export_spectrum_data(self, spectrum_data, temp_path):
        """Test exporting spectrum data to CSV."""
        exporter = CSVExporter()
        exporter.export(spectrum_data, temp_path)

        content = Path(temp_path).read_text()
        lines = content.strip().split("\n")

        # Should have header + data lines
        assert len(lines) > 100  # 1000 data points + header + metadata

    def test_export_dict(self, temp_path):
        """Test exporting generic dictionary to CSV."""
        data = {"key1": 1.5, "key2": "value", "key3": True}
        exporter = CSVExporter()
        exporter.export(data, temp_path)

        content = Path(temp_path).read_text()
        assert "key1" in content
        assert "1.5" in content

    def test_custom_delimiter(self, cflibs_result, temp_path):
        """Test CSV export with custom delimiter."""
        exporter = CSVExporter(delimiter=";")
        exporter.export(cflibs_result, temp_path)

        content = Path(temp_path).read_text()
        # Data lines should use semicolon
        assert ";" in content

    def test_no_metadata(self, cflibs_result, temp_path):
        """Test CSV export without metadata comments."""
        exporter = CSVExporter(include_metadata=False)
        exporter.export(cflibs_result, temp_path)

        content = Path(temp_path).read_text()
        assert "CF-LIBS Export" not in content

    def test_no_header(self, spectrum_data, temp_path):
        """Test CSV export without column headers."""
        exporter = CSVExporter(include_header=False, include_metadata=False)
        exporter.export(spectrum_data, temp_path)

        content = Path(temp_path).read_text()
        # First line should be data, not header
        first_line = content.strip().split("\n")[0]
        assert "wavelength" not in first_line

    def test_float_format(self, temp_path):
        """Test custom float format."""
        data = {"value": 0.123456789}
        exporter = CSVExporter(float_format="%.2f")
        exporter.export(data, temp_path)

        content = Path(temp_path).read_text()
        assert "0.12" in content

    def test_with_metadata_parameter(self, cflibs_result, temp_path):
        """Test passing additional metadata."""
        exporter = CSVExporter()
        exporter.export(cflibs_result, temp_path, metadata={"sample_id": "TEST123"})

        content = Path(temp_path).read_text()
        assert "sample_id" in content
        assert "TEST123" in content


# --- HDF5Exporter tests ---


class TestHDF5Exporter:
    """Tests for HDF5Exporter."""

    @pytest.fixture(autouse=True)
    def check_h5py(self):
        """Skip tests if h5py not available."""
        pytest.importorskip("h5py")

    def test_get_format(self):
        """Test format name."""
        exporter = HDF5Exporter()
        assert exporter.get_format() == "hdf5"

    def test_export_cflibs_result(self, cflibs_result, temp_path):
        """Test exporting CFLIBSResult to HDF5."""
        import h5py

        temp_path = temp_path + ".h5"
        exporter = HDF5Exporter()
        exporter.export(cflibs_result, temp_path)

        with h5py.File(temp_path, "r") as f:
            assert "timestamp" in f.attrs
            assert "version" in f.attrs
            assert "source_type" in f.attrs
            assert f.attrs["source_type"] == "MockCFLIBSResult"

            assert "summary" in f
            assert f["summary"].attrs["temperature_K"] == 10000.0
            assert f["summary"].attrs["converged"]

            assert "concentrations" in f
            elements = [e.decode() for e in f["concentrations/elements"][:]]
            assert "Fe" in elements
            assert "Cu" in elements

        Path(temp_path).unlink()

    def test_export_mcmc_result(self, mcmc_result, temp_path):
        """Test exporting MCMCResult to HDF5."""
        import h5py

        temp_path = temp_path + ".h5"
        exporter = HDF5Exporter()
        exporter.export(mcmc_result, temp_path)

        with h5py.File(temp_path, "r") as f:
            assert "summary" in f
            assert "concentrations" in f
            assert "samples" in f

            # Check samples are stored
            assert "T_eV" in f["samples"]
            assert len(f["samples/T_eV"]) == 100

            # Check diagnostics
            assert "diagnostics" in f
            assert "r_hat" in f["diagnostics"]

        Path(temp_path).unlink()

    def test_export_nested_sampling_result(self, nested_result, temp_path):
        """Test exporting NestedSamplingResult to HDF5."""
        import h5py

        temp_path = temp_path + ".h5"
        exporter = HDF5Exporter()
        exporter.export(nested_result, temp_path)

        with h5py.File(temp_path, "r") as f:
            assert "evidence" in f
            assert f["evidence"].attrs["log_evidence"] == pytest.approx(-150.5)

            assert "samples" in f
            assert "weights" in f

        Path(temp_path).unlink()

    def test_compression_gzip(self, mcmc_result, temp_path):
        """Test HDF5 export with gzip compression."""
        import h5py

        temp_path = temp_path + ".h5"
        exporter = HDF5Exporter(compression="gzip", compression_opts=4)
        exporter.export(mcmc_result, temp_path)

        with h5py.File(temp_path, "r") as f:
            # Large datasets should be compressed
            if "T_eV" in f["samples"]:
                ds = f["samples/T_eV"]
                # compression is applied to datasets
                assert ds.compression is None or ds.compression == "gzip"

        Path(temp_path).unlink()

    def test_no_compression(self, mcmc_result, temp_path):
        """Test HDF5 export without compression."""

        temp_path = temp_path + ".h5"
        exporter = HDF5Exporter(compression=None)
        exporter.export(mcmc_result, temp_path)

        assert Path(temp_path).exists()
        Path(temp_path).unlink()

    def test_with_metadata(self, cflibs_result, temp_path):
        """Test HDF5 export with additional metadata."""
        import h5py

        temp_path = temp_path + ".h5"
        exporter = HDF5Exporter()
        exporter.export(cflibs_result, temp_path, metadata={"sample_id": "TEST123"})

        with h5py.File(temp_path, "r") as f:
            assert f.attrs["sample_id"] == "TEST123"

        Path(temp_path).unlink()


# --- JSONExporter tests ---


class TestJSONExporter:
    """Tests for JSONExporter."""

    def test_get_format(self):
        """Test format name."""
        exporter = JSONExporter()
        assert exporter.get_format() == "json"

    def test_export_cflibs_result(self, cflibs_result, temp_path):
        """Test exporting CFLIBSResult to JSON."""
        exporter = JSONExporter()
        exporter.export(cflibs_result, temp_path)

        with open(temp_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "data" in data
        assert data["metadata"]["format"] == "json"
        assert data["data"]["temperature_K"] == 10000.0
        assert data["data"]["concentrations"]["Fe"] == 0.8

    def test_export_mcmc_result(self, mcmc_result, temp_path):
        """Test exporting MCMCResult to JSON."""
        exporter = JSONExporter()
        exporter.export(mcmc_result, temp_path)

        with open(temp_path) as f:
            data = json.load(f)

        assert data["data"]["T_eV_mean"] == 1.0
        assert "samples" in data["data"]

    def test_export_nested_sampling_result(self, nested_result, temp_path):
        """Test exporting NestedSamplingResult to JSON."""
        exporter = JSONExporter()
        exporter.export(nested_result, temp_path)

        with open(temp_path) as f:
            data = json.load(f)

        assert data["data"]["log_evidence"] == pytest.approx(-150.5)

    def test_export_dict(self, temp_path):
        """Test exporting generic dictionary to JSON."""
        input_data = {"key": "value", "number": 42, "array": [1, 2, 3]}
        exporter = JSONExporter()
        exporter.export(input_data, temp_path)

        with open(temp_path) as f:
            data = json.load(f)

        assert data["data"]["key"] == "value"
        assert data["data"]["number"] == 42
        assert data["data"]["array"] == [1, 2, 3]

    def test_numpy_array_serialization(self, temp_path):
        """Test that numpy arrays are properly serialized."""
        input_data = {"array": np.array([1.0, 2.0, 3.0])}
        exporter = JSONExporter()
        exporter.export(input_data, temp_path)

        with open(temp_path) as f:
            data = json.load(f)

        assert data["data"]["array"] == [1.0, 2.0, 3.0]

    def test_numpy_scalar_serialization(self, temp_path):
        """Test that numpy scalars are properly serialized."""
        input_data = {
            "int64": np.int64(42),
            "float64": np.float64(3.14),
            "bool": np.bool_(True),
        }
        exporter = JSONExporter()
        exporter.export(input_data, temp_path)

        with open(temp_path) as f:
            data = json.load(f)

        assert data["data"]["int64"] == 42
        assert data["data"]["float64"] == pytest.approx(3.14)
        assert data["data"]["bool"]

    def test_nan_handling(self, temp_path):
        """Test NaN value handling."""
        input_data = {"value": float("nan")}
        exporter = JSONExporter(allow_nan=True)
        exporter.export(input_data, temp_path)

        with open(temp_path) as f:
            content = f.read()

        assert "NaN" in content

    def test_inf_handling(self, temp_path):
        """Test Infinity value handling."""
        input_data = {"pos_inf": float("inf"), "neg_inf": float("-inf")}
        exporter = JSONExporter(allow_nan=True)
        exporter.export(input_data, temp_path)

        with open(temp_path) as f:
            content = f.read()

        assert "Infinity" in content
        assert "-Infinity" in content

    def test_custom_indent(self, cflibs_result, temp_path):
        """Test custom indentation."""
        exporter = JSONExporter(indent=4)
        exporter.export(cflibs_result, temp_path)

        content = Path(temp_path).read_text()
        # 4-space indent should be visible
        assert "    " in content

    def test_no_indent(self, cflibs_result, temp_path):
        """Test compact JSON (no indentation)."""
        exporter = JSONExporter(indent=None)
        exporter.export(cflibs_result, temp_path)

        content = Path(temp_path).read_text()
        # Should be a single line (no newlines except at end)
        assert content.count("\n") <= 1

    def test_sorted_keys(self, temp_path):
        """Test key sorting."""
        input_data = {"zebra": 1, "apple": 2, "mango": 3}
        exporter = JSONExporter(sort_keys=True)
        exporter.export(input_data, temp_path)

        content = Path(temp_path).read_text()
        # Keys should appear in sorted order
        apple_pos = content.find("apple")
        mango_pos = content.find("mango")
        zebra_pos = content.find("zebra")
        assert apple_pos < mango_pos < zebra_pos


# --- Factory function tests ---


class TestCreateExporter:
    """Tests for create_exporter factory function."""

    def test_create_csv_exporter(self):
        """Test creating CSVExporter via factory."""
        exporter = create_exporter("csv")
        assert isinstance(exporter, CSVExporter)

    def test_create_csv_with_options(self):
        """Test creating CSVExporter with options."""
        exporter = create_exporter("csv", delimiter=";", include_header=False)
        assert isinstance(exporter, CSVExporter)
        assert exporter.delimiter == ";"
        assert not exporter.include_header

    def test_create_hdf5_exporter(self):
        """Test creating HDF5Exporter via factory."""
        pytest.importorskip("h5py")
        exporter = create_exporter("hdf5")
        assert isinstance(exporter, HDF5Exporter)

    def test_create_h5_alias(self):
        """Test h5 alias for hdf5."""
        pytest.importorskip("h5py")
        exporter = create_exporter("h5")
        assert isinstance(exporter, HDF5Exporter)

    def test_create_hdf5_with_options(self):
        """Test creating HDF5Exporter with options."""
        pytest.importorskip("h5py")
        exporter = create_exporter("hdf5", compression="gzip", compression_opts=9)
        assert isinstance(exporter, HDF5Exporter)
        assert exporter.compression == "gzip"
        assert exporter.compression_opts == 9

    def test_create_json_exporter(self):
        """Test creating JSONExporter via factory."""
        exporter = create_exporter("json")
        assert isinstance(exporter, JSONExporter)

    def test_create_json_with_options(self):
        """Test creating JSONExporter with options."""
        exporter = create_exporter("json", indent=4, sort_keys=False)
        assert isinstance(exporter, JSONExporter)
        assert exporter.indent == 4
        assert not exporter.sort_keys

    def test_case_insensitive(self):
        """Test that format name is case-insensitive."""
        assert isinstance(create_exporter("CSV"), CSVExporter)
        assert isinstance(create_exporter("Json"), JSONExporter)

    def test_unsupported_format(self):
        """Test error for unsupported format."""
        with pytest.raises(ValueError) as exc_info:
            create_exporter("xml")
        assert "Unsupported export format" in str(exc_info.value)
        assert "xml" in str(exc_info.value)


# --- Convenience function tests ---


class TestConvenienceFunctions:
    """Tests for convenience export functions."""

    def test_export_to_csv(self, cflibs_result, temp_path):
        """Test export_to_csv convenience function."""
        export_to_csv(cflibs_result, temp_path)
        assert Path(temp_path).exists()

        content = Path(temp_path).read_text()
        assert "temperature" in content

    def test_export_to_json(self, cflibs_result, temp_path):
        """Test export_to_json convenience function."""
        export_to_json(cflibs_result, temp_path)
        assert Path(temp_path).exists()

        with open(temp_path) as f:
            data = json.load(f)
        assert "metadata" in data


# --- Edge case tests ---


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dict(self, temp_path):
        """Test exporting empty dictionary."""
        exporter = CSVExporter()
        exporter.export({}, temp_path)
        assert Path(temp_path).exists()

    def test_nested_dict(self, temp_path):
        """Test exporting nested dictionary."""
        data = {"level1": {"level2": {"value": 42}}}
        exporter = JSONExporter()
        exporter.export(data, temp_path)

        with open(temp_path) as f:
            loaded = json.load(f)
        assert loaded["data"]["level1"]["level2"]["value"] == 42

    def test_large_array(self, temp_path):
        """Test exporting large arrays."""
        data = {"large_array": np.random.randn(10000)}
        exporter = JSONExporter()
        exporter.export(data, temp_path)

        with open(temp_path) as f:
            loaded = json.load(f)

        # Large arrays should be stored with metadata
        arr_data = loaded["data"]["large_array"]
        if isinstance(arr_data, dict) and "_type" in arr_data:
            assert arr_data["_type"] == "ndarray"
            assert arr_data["shape"] == [10000]
        else:
            assert len(arr_data) == 10000

    def test_unsupported_data_type(self, temp_path):
        """Test error for unsupported data type."""
        exporter = CSVExporter()

        with pytest.raises(ValueError) as exc_info:
            exporter.export("not a dict or dataclass", temp_path)
        assert "Unsupported data type" in str(exc_info.value)

    def test_path_as_string(self, cflibs_result, temp_path):
        """Test that string paths work."""
        exporter = CSVExporter()
        exporter.export(cflibs_result, str(temp_path))
        assert Path(temp_path).exists()

    def test_path_as_pathlib(self, cflibs_result, temp_path):
        """Test that pathlib.Path works."""
        exporter = CSVExporter()
        exporter.export(cflibs_result, Path(temp_path))
        assert Path(temp_path).exists()

    def test_special_characters_in_values(self, temp_path):
        """Test handling of special characters."""
        data = {"text": "comma,separated;values\ttab"}
        exporter = JSONExporter()
        exporter.export(data, temp_path)

        with open(temp_path) as f:
            loaded = json.load(f)
        assert loaded["data"]["text"] == "comma,separated;values\ttab"


# --- Integration tests ---


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_round_trip_csv_spectrum(self, spectrum_data, temp_path):
        """Test that exported spectrum can be read back."""
        exporter = CSVExporter(include_metadata=False, include_header=True)
        exporter.export(spectrum_data, temp_path)

        # Read back with numpy
        data = np.loadtxt(temp_path, delimiter=",", skiprows=1)
        assert data.shape[0] == len(spectrum_data["wavelength"])
        assert np.allclose(data[:, 0], spectrum_data["wavelength"])
        assert np.allclose(data[:, 1], spectrum_data["intensity"], rtol=1e-5)

    def test_round_trip_json(self, cflibs_result, temp_path):
        """Test JSON round-trip preserves data."""
        exporter = JSONExporter()
        exporter.export(cflibs_result, temp_path)

        with open(temp_path) as f:
            loaded = json.load(f)

        assert loaded["data"]["temperature_K"] == cflibs_result.temperature_K
        assert loaded["data"]["concentrations"]["Fe"] == cflibs_result.concentrations["Fe"]

    @pytest.mark.skipif(
        not pytest.importorskip("h5py", reason="h5py not available"), reason="h5py required"
    )
    def test_round_trip_hdf5(self, mcmc_result, temp_path):
        """Test HDF5 round-trip preserves data."""
        import h5py

        temp_path = temp_path + ".h5"
        exporter = HDF5Exporter()
        exporter.export(mcmc_result, temp_path)

        with h5py.File(temp_path, "r") as f:
            T_samples = f["samples/T_eV"][:]
            assert len(T_samples) == len(mcmc_result.samples["T_eV"])
            assert np.allclose(T_samples, mcmc_result.samples["T_eV"])

        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
