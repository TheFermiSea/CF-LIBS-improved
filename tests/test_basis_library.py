"""Tests for single-element basis library generator."""

import os
import tempfile

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from cflibs.manifold.basis_library import (  # noqa: E402
    BasisLibrary,
    BasisLibraryConfig,
    BasisLibraryGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_config(db_path: str, output_path: str) -> BasisLibraryConfig:
    """Return a small config suitable for fast tests."""
    return BasisLibraryConfig(
        db_path=db_path,
        output_path=output_path,
        wavelength_range=(370.0, 380.0),
        pixels=512,
        temperature_range=(4000.0, 12000.0),
        temperature_steps=3,
        density_range=(1e15, 5e17),
        density_steps=2,
        ionization_stages=(1, 2),
        instrument_fwhm_nm=0.05,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestBasisLibraryConfig:
    def test_defaults(self):
        cfg = BasisLibraryConfig(db_path="dummy.db")
        assert cfg.pixels == 4096
        assert cfg.temperature_steps == 50
        assert cfg.density_steps == 20
        assert cfg.instrument_fwhm_nm == 0.05

    def test_validate_missing_db_path(self):
        cfg = BasisLibraryConfig()
        with pytest.raises(ValueError, match="db_path"):
            cfg.validate()

    def test_validate_bad_wavelength_range(self):
        cfg = BasisLibraryConfig(db_path="x.db", wavelength_range=(500.0, 200.0))
        with pytest.raises(ValueError, match="wavelength_range"):
            cfg.validate()

    def test_validate_bad_temperature_range(self):
        cfg = BasisLibraryConfig(db_path="x.db", temperature_range=(15000.0, 4000.0))
        with pytest.raises(ValueError, match="temperature_range"):
            cfg.validate()

    def test_validate_bad_density_range(self):
        cfg = BasisLibraryConfig(db_path="x.db", density_range=(1e18, 1e15))
        with pytest.raises(ValueError, match="density_range"):
            cfg.validate()

    def test_validate_negative_wavelength(self):
        cfg = BasisLibraryConfig(db_path="x.db", wavelength_range=(-10.0, 200.0))
        with pytest.raises(ValueError, match="non-negative"):
            cfg.validate()

    def test_validate_zero_pixels(self):
        cfg = BasisLibraryConfig(db_path="x.db", pixels=0)
        with pytest.raises(ValueError, match="pixels"):
            cfg.validate()

    def test_validate_passes_for_good_config(self, temp_db):
        cfg = _small_config(temp_db, "out.h5")
        cfg.validate()  # should not raise


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@pytest.fixture
def h5_path():
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.mark.requires_db
class TestBasisLibraryGenerator:
    def test_generates_non_zero_spectra(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            # Fe should have non-zero spectra (it has lines in 370-380 nm)
            fe_spec = lib.get_element_spectrum("Fe", 8000.0, 1e17)
            assert np.any(fe_spec > 0), "Fe spectrum should be non-zero in 370-380 nm"

    def test_spectra_area_normalized(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            fe_spec = lib.get_element_spectrum("Fe", 8000.0, 1e17)
            if np.sum(fe_spec) > 0:
                np.testing.assert_allclose(np.sum(fe_spec), 1.0, atol=1e-10)

    def test_fe_peak_near_expected(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            wl = lib.wavelength
            fe_spec = lib.get_element_spectrum("Fe", 8000.0, 1e17)
            peak_wl = wl[np.argmax(fe_spec)]
            # The test DB has strong Fe I lines at 371.99 and 373.49 nm
            assert (
                371.0 < peak_wl < 374.5
            ), f"Fe peak at {peak_wl} nm, expected near 371.99 or 373.49"

    def test_hdf5_round_trip(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            assert lib.n_pixels == cfg.pixels
            assert lib.n_grid == cfg.temperature_steps * cfg.density_steps
            assert "Fe" in lib.elements
            assert "H" in lib.elements
            # Wavelength grid matches config
            wl = lib.wavelength
            np.testing.assert_allclose(wl[0], 370.0, atol=0.1)
            np.testing.assert_allclose(wl[-1], 380.0, atol=0.1)

    def test_basis_matrix_shape(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            basis = lib.get_basis_matrix(8000.0, 1e17)
            assert basis.shape == (lib.n_elements, lib.n_pixels)

    def test_interp_at_grid_point_matches_nearest(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            # Use exact grid point
            T_vals = np.unique(np.linspace(4000.0, 12000.0, 3))
            ne_vals = np.unique(np.geomspace(1e15, 5e17, 2))
            T_exact = T_vals[1]
            ne_exact = ne_vals[0]

            nearest = lib.get_basis_matrix(T_exact, ne_exact)
            interp = lib.get_basis_matrix_interp(T_exact, ne_exact)
            np.testing.assert_allclose(interp, nearest, atol=1e-10)

    def test_element_with_no_transitions_is_zero(self, temp_db, h5_path):
        # H has no lines in 370-380 nm range (H-alpha is at 656 nm)
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            h_spec = lib.get_element_spectrum("H", 8000.0, 1e17)
            assert np.allclose(h_spec, 0.0), "H should have no emission in 370-380 nm"

    def test_temperature_affects_spectrum(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        gen.generate()

        with BasisLibrary(h5_path) as lib:
            fe_low_T = lib.get_element_spectrum("Fe", 4000.0, 1e17)
            fe_high_T = lib.get_element_spectrum("Fe", 12000.0, 1e17)
            # Both should be non-zero but different shapes
            if np.sum(fe_low_T) > 0 and np.sum(fe_high_T) > 0:
                assert not np.allclose(
                    fe_low_T, fe_high_T
                ), "Spectra at different temperatures should differ"

    def test_progress_callback(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        gen = BasisLibraryGenerator(cfg)
        calls = []
        gen.generate(progress_callback=lambda i, n: calls.append((i, n)))
        assert len(calls) > 0
        # Last call should have i == n
        assert calls[-1][0] == calls[-1][1]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@pytest.mark.requires_db
class TestBasisLibrary:
    def test_context_manager(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        BasisLibraryGenerator(cfg).generate()

        lib = BasisLibrary(h5_path)
        lib.close()
        # After close, HDF5 file handle should be invalid
        assert not lib._f.id.valid

    def test_context_manager_with_block(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        BasisLibraryGenerator(cfg).generate()

        with BasisLibrary(h5_path) as lib:
            assert lib.n_pixels > 0
        # After exiting the block, file should be closed
        assert not lib._f.id.valid

    def test_unknown_element_raises(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        BasisLibraryGenerator(cfg).generate()

        with BasisLibrary(h5_path) as lib:
            with pytest.raises(KeyError):
                lib.get_element_spectrum("Zz", 8000.0, 1e17)

    def test_properties(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        BasisLibraryGenerator(cfg).generate()

        with BasisLibrary(h5_path) as lib:
            assert isinstance(lib.elements, list)
            assert lib.n_elements == len(lib.elements)
            assert lib.n_pixels == cfg.pixels
            assert lib.n_grid == cfg.temperature_steps * cfg.density_steps
            wl = lib.wavelength
            assert isinstance(wl, np.ndarray)
            assert len(wl) == lib.n_pixels

    def test_get_basis_matrix_interp_clamps(self, temp_db, h5_path):
        cfg = _small_config(temp_db, h5_path)
        BasisLibraryGenerator(cfg).generate()

        with BasisLibrary(h5_path) as lib:
            # Requesting far outside the grid should clamp without error
            basis = lib.get_basis_matrix_interp(1.0, 1.0)
            assert basis.shape == (lib.n_elements, lib.n_pixels)
            basis2 = lib.get_basis_matrix_interp(1e9, 1e30)
            assert basis2.shape == (lib.n_elements, lib.n_pixels)

    def test_single_point_grid_interp_fallback(self, temp_db, h5_path):
        # Use valid ranges but only 1 step per axis so that the grid
        # contains a single (T, ne) point, exercising the fallback path
        # in get_basis_matrix_interp().
        cfg = BasisLibraryConfig(
            db_path=temp_db,
            output_path=h5_path,
            wavelength_range=(370.0, 380.0),
            pixels=512,
            temperature_range=(4000.0, 12000.0),
            temperature_steps=1,
            density_range=(1e15, 5e17),
            density_steps=1,
            ionization_stages=(1, 2),
            instrument_fwhm_nm=0.05,
        )
        BasisLibraryGenerator(cfg).generate()

        with BasisLibrary(h5_path) as lib:
            # Single-point grid: interp should fall back to nearest
            interp = lib.get_basis_matrix_interp(8000.0, 1e16)
            nearest = lib.get_basis_matrix(8000.0, 1e16)
            np.testing.assert_array_equal(interp, nearest)
