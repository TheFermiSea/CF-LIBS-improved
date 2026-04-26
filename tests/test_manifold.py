"""
Tests for manifold generation module.
"""

import pytest
import numpy as np

try:
    import h5py  # noqa: E402
except ImportError:  # pragma: no cover - exercised in minimal envs
    h5py = None

from cflibs.manifold.config import ManifoldConfig
from cflibs.core.logging_config import setup_logging

setup_logging()

requires_h5py = pytest.mark.skipif(h5py is None, reason="h5py not installed")


class TestManifoldConfig:
    """Tests for ManifoldConfig."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = ManifoldConfig(
            db_path="test.db",
            output_path="test.h5",
            elements=["Ti", "Al"],
            wavelength_range=(250.0, 550.0),
            temperature_range=(0.5, 2.0),
            temperature_steps=10,
            density_range=(1e16, 1e19),
            density_steps=5,
        )

        assert config.db_path == "test.db"
        assert config.output_path == "test.h5"
        assert config.elements == ["Ti", "Al"]
        assert config.wavelength_range == (250.0, 550.0)
        assert config.temperature_range == (0.5, 2.0)
        assert config.temperature_steps == 10
        assert config.density_range == (1e16, 1e19)
        assert config.density_steps == 5

    def test_config_validation(self):
        """Test config validation."""
        # Create dummy db file
        from pathlib import Path

        Path("test.db").touch()

        try:
            # Valid config
            config = ManifoldConfig(
                db_path="test.db",
                output_path="test.h5",
                elements=["Ti"],
                wavelength_range=(250.0, 550.0),
                temperature_range=(0.5, 2.0),
                temperature_steps=10,
                density_range=(1e16, 1e19),
                density_steps=5,
            )

            # Should raise if db doesn't exist - but we created it, so it should pass check
            # Wait, validate() checks existence.
            # config.validate()

            # If we delete it, it should fail
            Path("test.db").unlink()
            with pytest.raises(ValueError, match="Database file not found"):
                config.validate()

            # Restore it for other tests
            Path("test.db").touch()

            # Invalid wavelength range
            config.wavelength_range = (550.0, 250.0)
            with pytest.raises(ValueError, match="Invalid wavelength range"):
                config.validate()
            config.wavelength_range = (250.0, 550.0)  # Reset

            # Invalid temperature range
            config.temperature_range = (2.0, 0.5)
            with pytest.raises(ValueError, match="Invalid temperature range"):
                config.validate()
            config.temperature_range = (0.5, 2.0)  # Reset

            # Invalid density range
            config.density_range = (1e19, 1e16)
            with pytest.raises(ValueError, match="Invalid density range"):
                config.validate()
            config.density_range = (1e16, 1e19)  # Reset

            # Invalid steps
            config.temperature_steps = 1
            with pytest.raises(ValueError, match="temperature_steps must be >= 2"):
                config.validate()
            config.temperature_steps = 10

            config.density_steps = 1
            with pytest.raises(ValueError, match="density_steps must be >= 2"):
                config.validate()
            config.density_steps = 5

            # Invalid pixels
            config.pixels = 5
            with pytest.raises(ValueError, match="pixels must be >= 10"):
                config.validate()

        finally:
            # Cleanup
            if Path("test.db").exists():
                Path("test.db").unlink()

    def test_config_from_file(self, tmp_path):
        """Test loading config from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
manifold:
  db_path: test.db
  output_path: test.h5
  elements:
    - Ti
    - Al
  wavelength_range: [250.0, 550.0]
  temperature_range: [0.5, 2.0]
  temperature_steps: 10
  density_range: [1e16, 1e19]
  density_steps: 5
""")

        config = ManifoldConfig.from_file(config_file)

        assert config.db_path == "test.db"
        assert config.output_path == "test.h5"
        assert config.elements == ["Ti", "Al"]
        assert config.wavelength_range == (250.0, 550.0)
        assert config.temperature_range == (0.5, 2.0)
        assert config.temperature_steps == 10


class TestManifoldLoader:
    """Tests for ManifoldLoader."""

    @requires_h5py
    def test_loader_creation(self, tmp_path):
        """Test creating a mock manifold file and loading it."""
        manifold_path = tmp_path / "test_manifold.h5"

        # Create mock manifold
        with h5py.File(manifold_path, "w") as f:
            f.create_dataset("spectra", (10, 100), dtype="f4")
            f.create_dataset("params", (10, 4), dtype="f4")  # T, ne, C1, C2
            f.create_dataset("wavelength", data=np.linspace(250, 550, 100), dtype="f4")
            f.attrs["elements"] = ["Ti", "Al"]
            f.attrs["wavelength_range"] = [250.0, 550.0]
            f.attrs["temperature_range"] = [0.5, 2.0]
            f.attrs["density_range"] = [1e16, 1e19]

        # Load manifold
        from cflibs.manifold.loader import ManifoldLoader

        loader = ManifoldLoader(str(manifold_path))

        assert len(loader.spectra) == 10
        assert len(loader.wavelength) == 100
        assert loader.elements == ["Ti", "Al"]
        assert loader.wavelength_range == (250.0, 550.0)

        loader.close()

    @requires_h5py
    def test_find_nearest_spectrum(self, tmp_path):
        """Test finding nearest spectrum."""
        manifold_path = tmp_path / "test_manifold.h5"

        # Create mock manifold with known spectra
        n_samples = 10
        n_pixels = 100
        wavelength = np.linspace(250, 550, n_pixels)

        with h5py.File(manifold_path, "w") as f:
            # Create spectra: each is a Gaussian peak at different wavelengths
            spectra = np.zeros((n_samples, n_pixels))
            params = np.zeros((n_samples, 4))

            for i in range(n_samples):
                center = 300 + i * 20
                spectra[i] = np.exp(-0.5 * ((wavelength - center) / 10) ** 2)
                params[i] = [1.0, 1e17, 0.9, 0.1]  # T, ne, Ti, Al

            f.create_dataset("spectra", data=spectra, dtype="f4")
            f.create_dataset("params", data=params, dtype="f4")
            f.create_dataset("wavelength", data=wavelength, dtype="f4")
            f.attrs["elements"] = ["Ti", "Al"]
            f.attrs["wavelength_range"] = [250.0, 550.0]
            f.attrs["temperature_range"] = [0.5, 2.0]
            f.attrs["density_range"] = [1e16, 1e19]

        from cflibs.manifold.loader import ManifoldLoader

        loader = ManifoldLoader(str(manifold_path))

        # Create a test spectrum matching index 5
        test_spectrum = spectra[5]

        index, similarity, params = loader.find_nearest_spectrum(test_spectrum, method="cosine")

        assert index == 5
        assert similarity > 0.9  # Should be very similar
        assert params["T_eV"] == 1.0
        assert params["n_e_cm3"] == pytest.approx(1e17)

        loader.close()

    @pytest.mark.requires_jax
    @requires_h5py
    def test_find_nearest_spectrum_jax(self, tmp_path):
        """Test finding nearest spectrum with JAX."""
        try:
            import jax  # noqa: F401
        except ImportError:
            pytest.skip("JAX not installed")

        manifold_path = tmp_path / "test_manifold.h5"

        n_samples = 6
        n_pixels = 80
        wavelength = np.linspace(250, 550, n_pixels)

        with h5py.File(manifold_path, "w") as f:
            spectra = np.zeros((n_samples, n_pixels))
            params = np.zeros((n_samples, 4))

            for i in range(n_samples):
                center = 280 + i * 25
                spectra[i] = np.exp(-0.5 * ((wavelength - center) / 12) ** 2)
                params[i] = [1.2, 5e16, 0.85, 0.15]

            f.create_dataset("spectra", data=spectra, dtype="f4")
            f.create_dataset("params", data=params, dtype="f4")
            f.create_dataset("wavelength", data=wavelength, dtype="f4")
            f.attrs["elements"] = ["Ti", "Al"]
            f.attrs["wavelength_range"] = [250.0, 550.0]
            f.attrs["temperature_range"] = [0.5, 2.0]
            f.attrs["density_range"] = [1e16, 1e19]

        from cflibs.manifold.loader import ManifoldLoader

        loader = ManifoldLoader(str(manifold_path))

        test_spectrum = spectra[3]
        index, similarity, params = loader.find_nearest_spectrum(
            test_spectrum, method="cosine", use_jax=True
        )

        assert index == 3
        assert similarity > 0.9
        assert params["T_eV"] == pytest.approx(1.2, rel=1e-6)  # float32 precision
        assert params["n_e_cm3"] == pytest.approx(5e16)

        loader.close()

    @requires_h5py
    def test_get_spectrum(self, tmp_path):
        """Test getting spectrum by index."""
        manifold_path = tmp_path / "test_manifold.h5"

        with h5py.File(manifold_path, "w") as f:
            spectra = np.random.rand(10, 100).astype("f4")
            params = np.random.rand(10, 4).astype("f4")

            f.create_dataset("spectra", data=spectra, dtype="f4")
            f.create_dataset("params", data=params, dtype="f4")
            f.create_dataset("wavelength", data=np.linspace(250, 550, 100), dtype="f4")
            f.attrs["elements"] = ["Ti", "Al"]
            f.attrs["wavelength_range"] = [250.0, 550.0]
            f.attrs["temperature_range"] = [0.5, 2.0]
            f.attrs["density_range"] = [1e16, 1e19]

        from cflibs.manifold.loader import ManifoldLoader

        loader = ManifoldLoader(str(manifold_path))

        spectrum, params_dict = loader.get_spectrum(5)

        assert np.allclose(spectrum, spectra[5])
        assert params_dict["T_eV"] == float(params[5, 0])
        assert params_dict["n_e_cm3"] == float(params[5, 1])

        loader.close()

    @requires_h5py
    def test_context_manager(self, tmp_path):
        """Test using loader as context manager."""
        manifold_path = tmp_path / "test_manifold.h5"

        with h5py.File(manifold_path, "w") as f:
            f.create_dataset("spectra", (10, 100), dtype="f4")
            f.create_dataset("params", (10, 4), dtype="f4")
            f.create_dataset("wavelength", data=np.linspace(250, 550, 100), dtype="f4")
            f.attrs["elements"] = ["Ti", "Al"]
            f.attrs["wavelength_range"] = [250.0, 550.0]
            f.attrs["temperature_range"] = [0.5, 2.0]
            f.attrs["density_range"] = [1e16, 1e19]

        from cflibs.manifold.loader import ManifoldLoader

        with ManifoldLoader(str(manifold_path)) as loader:
            assert len(loader.spectra) == 10

        # File should be closed now

    def test_loader_creation_zarr(self, tmp_path):
        """Test creating and loading a Zarr-backed manifold."""
        zarr = pytest.importorskip("zarr")
        manifold_path = tmp_path / "test_manifold.zarr"

        root = zarr.open_group(str(manifold_path), mode="w")
        root.create_array(
            "spectra",
            data=np.random.rand(10, 100).astype("f4"),
            chunks=(4, 100),
            overwrite=True,
        )
        root.create_array(
            "params",
            data=np.random.rand(10, 4).astype("f4"),
            chunks=(4, 4),
            overwrite=True,
        )
        root.create_array(
            "wavelength",
            data=np.linspace(250, 550, 100, dtype=np.float32),
            overwrite=True,
        )
        root.attrs["elements"] = ["Ti", "Al"]
        root.attrs["wavelength_range"] = [250.0, 550.0]
        root.attrs["temperature_range"] = [0.5, 2.0]
        root.attrs["density_range"] = [1e16, 1e19]

        from cflibs.manifold.loader import ManifoldLoader

        loader = ManifoldLoader(str(manifold_path))

        assert loader.storage_format == "zarr"
        assert len(loader.spectra) == 10
        assert len(loader.wavelength) == 100
        assert loader.elements == ["Ti", "Al"]

        loader.close()

    def test_generator_infers_zarr_for_directory_output(self, tmp_path):
        """Test generator storage inference matches loader directory handling."""
        from cflibs.manifold.generator import _infer_storage_format

        manifold_dir = tmp_path / "manifold_store"
        manifold_dir.mkdir()

        assert _infer_storage_format(manifold_dir) == "zarr"

    def test_find_nearest_spectrum_zarr(self, tmp_path):
        """Test chunked nearest-neighbor search over a Zarr manifold."""
        zarr = pytest.importorskip("zarr")
        manifold_path = tmp_path / "test_manifold.zarr"

        n_samples = 10
        n_pixels = 100
        wavelength = np.linspace(250, 550, n_pixels, dtype=np.float32)
        spectra = np.zeros((n_samples, n_pixels), dtype=np.float32)
        params = np.zeros((n_samples, 4), dtype=np.float32)

        for i in range(n_samples):
            center = 300 + i * 20
            spectra[i] = np.exp(-0.5 * ((wavelength - center) / 10) ** 2)
            params[i] = [1.0, 1e17, 0.9, 0.1]

        root = zarr.open_group(str(manifold_path), mode="w")
        root.create_array("spectra", data=spectra, chunks=(3, n_pixels), overwrite=True)
        root.create_array("params", data=params, chunks=(3, 4), overwrite=True)
        root.create_array("wavelength", data=wavelength, overwrite=True)
        root.attrs["elements"] = ["Ti", "Al"]
        root.attrs["wavelength_range"] = [250.0, 550.0]
        root.attrs["temperature_range"] = [0.5, 2.0]
        root.attrs["density_range"] = [1e16, 1e19]

        from cflibs.manifold.loader import ManifoldLoader

        loader = ManifoldLoader(str(manifold_path))

        index, similarity, params_dict = loader.find_nearest_spectrum(
            spectra[5], method="cosine", search_batch_size=3
        )

        assert index == 5
        assert similarity > 0.9
        assert params_dict["T_eV"] == pytest.approx(1.0)
        assert params_dict["n_e_cm3"] == pytest.approx(1e17)

        loader.close()

    @requires_h5py
    def test_find_nearest_spectrum_rejects_empty_manifold(self, tmp_path):
        """Test nearest-spectrum search fails clearly for empty manifolds."""
        manifold_path = tmp_path / "empty_manifold.h5"

        with h5py.File(manifold_path, "w") as f:
            f.create_dataset("spectra", shape=(0, 100), dtype="f4")
            f.create_dataset("params", shape=(0, 4), dtype="f4")
            f.create_dataset("wavelength", data=np.linspace(250, 550, 100), dtype="f4")
            f.attrs["elements"] = ["Ti", "Al"]
            f.attrs["wavelength_range"] = [250.0, 550.0]
            f.attrs["temperature_range"] = [0.5, 2.0]
            f.attrs["density_range"] = [1e16, 1e19]

        from cflibs.manifold.loader import ManifoldLoader

        loader = ManifoldLoader(str(manifold_path))
        with pytest.raises(ValueError, match="empty manifold"):
            loader.find_nearest_spectrum(np.zeros(100, dtype=np.float32))
        loader.close()
