"""
Tests for BasisIndex — FAISS-based index for plasma parameter estimation.
"""

import numpy as np
import pytest

from cflibs.manifold.basis_index import BasisIndex, _weighted_median

pytestmark = [pytest.mark.requires_db]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def small_library(atomic_db, tmp_path):
    """Generate a small basis library for testing."""
    from cflibs.manifold.basis_library import BasisLibraryConfig, BasisLibraryGenerator

    cfg = BasisLibraryConfig(
        db_path=str(atomic_db.db_path),
        output_path=str(tmp_path / "test_basis.h5"),
        wavelength_range=(370.0, 380.0),
        pixels=256,
        temperature_range=(6000.0, 10000.0),
        temperature_steps=3,
        density_range=(1e16, 1e17),
        density_steps=2,
    )
    gen = BasisLibraryGenerator(cfg)
    gen.generate()

    from cflibs.manifold.basis_library import BasisLibrary

    lib = BasisLibrary(cfg.output_path)
    yield lib
    lib.close()


@pytest.fixture
def built_index(small_library):
    """Build a BasisIndex from the small library."""
    idx = BasisIndex(n_components=5)  # Small PCA for test
    idx.build_from_library(small_library)
    return idx


# ---------------------------------------------------------------------------
# Unit tests for _weighted_median
# ---------------------------------------------------------------------------


class TestWeightedMedian:
    def test_uniform_weights(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5) / 5
        assert _weighted_median(vals, weights) == 3.0

    def test_skewed_weights(self):
        vals = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 1.0])
        assert _weighted_median(vals, weights) == 3.0

    def test_single_value(self):
        assert _weighted_median(np.array([42.0]), np.array([1.0])) == 42.0


# ---------------------------------------------------------------------------
# BasisIndex build and query tests
# ---------------------------------------------------------------------------


class TestBasisIndex:
    def test_build_from_library(self, small_library):
        idx = BasisIndex(n_components=5)
        idx.build_from_library(small_library)
        assert idx.is_built
        assert idx.n_vectors > 0

    def test_not_built_raises(self):
        idx = BasisIndex(n_components=5)
        assert not idx.is_built
        with pytest.raises(RuntimeError, match="build_from_library"):
            idx.estimate_plasma_params(np.zeros(256))

    def test_estimate_plasma_params(self, built_index, small_library):
        # Use a real basis spectrum as query — should recover its (T, ne)
        T_query = 8000.0
        ne_query = 5e16
        fe_spectrum = small_library.get_element_spectrum("Fe", T_query, ne_query)

        T_est, ne_est, details = built_index.estimate_plasma_params(fe_spectrum, k=10)

        # T estimate should be in the grid range
        assert 5000 < T_est < 11000
        # ne estimate should be in the grid range
        assert 5e15 < ne_est < 5e17
        # Details should have the right keys
        assert "neighbor_elements" in details
        assert "element_votes" in details
        assert len(details["neighbor_elements"]) == 10

    def test_element_votes(self, built_index, small_library):
        # Query with Fe spectrum — Fe should be a top vote
        fe_spectrum = small_library.get_element_spectrum("Fe", 8000.0, 5e16)
        _, _, details = built_index.estimate_plasma_params(fe_spectrum, k=20)
        votes = details["element_votes"]
        assert "Fe" in votes

    def test_save_load_roundtrip(self, built_index, tmp_path):
        path = str(tmp_path / "test_index.h5")
        built_index.save(path)

        loaded = BasisIndex.load(path)
        assert loaded.is_built
        assert loaded.n_vectors == built_index.n_vectors
        assert loaded.n_components == built_index.n_components
        assert loaded._elements == built_index._elements

    def test_save_load_query_consistency(self, built_index, small_library, tmp_path):
        """Saved and loaded index should give same results."""
        path = str(tmp_path / "test_index.h5")
        built_index.save(path)
        loaded = BasisIndex.load(path)

        fe_spectrum = small_library.get_element_spectrum("Fe", 8000.0, 5e16)
        T1, ne1, _ = built_index.estimate_plasma_params(fe_spectrum, k=10)
        T2, ne2, _ = loaded.estimate_plasma_params(fe_spectrum, k=10)

        assert T1 == pytest.approx(T2, rel=1e-6)
        assert ne1 == pytest.approx(ne2, rel=1e-6)

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BasisIndex.load(str(tmp_path / "nonexistent.h5"))

    def test_save_before_build_raises(self, tmp_path):
        idx = BasisIndex(n_components=5)
        with pytest.raises(RuntimeError, match="build_from_library"):
            idx.save(str(tmp_path / "bad.h5"))

    def test_skip_zero_spectra(self, small_library):
        """Elements with no transitions should be skipped."""
        idx = BasisIndex(n_components=5)
        idx.build_from_library(small_library, skip_zero=True)
        n_with_skip = idx.n_vectors

        idx2 = BasisIndex(n_components=5)
        idx2.build_from_library(small_library, skip_zero=False)
        n_without_skip = idx2.n_vectors

        # Without skip should have more (or equal) vectors
        assert n_without_skip >= n_with_skip
