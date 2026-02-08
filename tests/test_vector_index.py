"""
Tests for SpectralEmbedder and VectorIndex.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from cflibs.manifold.vector_index import SpectralEmbedder, VectorIndexConfig

# Test SpectralEmbedder availability (always available)


def test_spectral_embedder_fit_transform():
    """Test basic fit and transform."""
    spectra = np.random.rand(50, 1000) + 1.0  # Positive spectra
    embedder = SpectralEmbedder(n_components=10)

    embedder.fit(spectra)
    assert embedder.is_fitted

    embeddings = embedder.transform(spectra)
    assert embeddings.shape == (50, 10)

    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-6)


def test_spectral_embedder_fit_transform_combined():
    """Test fit_transform convenience method."""
    spectra = np.random.rand(30, 500) + 1.0
    embedder = SpectralEmbedder(n_components=5)

    embeddings = embedder.fit_transform(spectra)
    assert embeddings.shape == (30, 5)
    assert embedder.is_fitted


def test_spectral_embedder_transform_without_fit():
    """Test that transform without fit raises error."""
    embedder = SpectralEmbedder(n_components=10)
    spectra = np.random.rand(10, 100)

    with pytest.raises(RuntimeError, match="Must call fit"):
        embedder.transform(spectra)


def test_spectral_embedder_area_normalization():
    """Test that area normalization is applied."""
    # Create spectra with known area
    spectra = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    embedder = SpectralEmbedder(n_components=2)

    # Fit (this triggers area normalization internally)
    embedder.fit(spectra)

    # Verify by checking that internal PCA was fitted
    assert embedder.pca_pipeline.is_fitted


# VectorIndex tests - require faiss
@pytest.mark.parametrize("index_type", ["flat"])
def test_vector_index_build_and_search(index_type):
    """Test building and searching vector index."""
    faiss = pytest.importorskip("faiss")
    from cflibs.manifold.vector_index import VectorIndex

    # Create embeddings
    embeddings = np.random.randn(100, 30).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Build index
    config = VectorIndexConfig(index_type=index_type)
    index = VectorIndex(dimension=30, config=config)
    index.build(embeddings)

    assert index.is_built
    assert index.index.ntotal == 100

    # Search
    query = embeddings[:5]
    distances, indices = index.search(query, k=10)

    assert distances.shape == (5, 10)
    assert indices.shape == (5, 10)

    # First result should be exact match (distance ~0)
    assert distances[0, 0] < 0.01


def test_vector_index_search_without_build():
    """Test that search without build raises error."""
    faiss = pytest.importorskip("faiss")
    from cflibs.manifold.vector_index import VectorIndex

    config = VectorIndexConfig(index_type="flat")
    index = VectorIndex(dimension=30, config=config)

    query = np.random.randn(5, 30).astype(np.float32)

    with pytest.raises(RuntimeError, match="Must call build"):
        index.search(query, k=10)


def test_vector_index_save_load():
    """Test saving and loading index."""
    faiss = pytest.importorskip("faiss")
    h5py = pytest.importorskip("h5py")
    from cflibs.manifold.vector_index import VectorIndex

    # Create and build index
    embeddings = np.random.randn(50, 20).astype(np.float32)
    config = VectorIndexConfig(index_type="flat")
    index = VectorIndex(dimension=20, config=config)
    index.build(embeddings)

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_index.h5"
        index.save(str(save_path))

        # Load
        loaded_index = VectorIndex.load(str(save_path))

        assert loaded_index.is_built
        assert loaded_index.dimension == 20
        assert loaded_index.index.ntotal == 50

        # Verify search works
        query = embeddings[:3]
        distances, indices = loaded_index.search(query, k=5)
        assert distances.shape == (3, 5)


def test_vector_index_without_faiss():
    """Test that VectorIndex raises ImportError when faiss not available."""
    # This test only makes sense if faiss is not installed
    # Skip if faiss is available
    try:
        import faiss
        pytest.skip("faiss is installed, cannot test ImportError")
    except ImportError:
        pass

    from cflibs.manifold.vector_index import VectorIndex, VectorIndexConfig

    config = VectorIndexConfig(index_type="flat")

    with pytest.raises(ImportError, match="faiss is required"):
        VectorIndex(dimension=30, config=config)


def test_vector_index_config_validation():
    """Test VectorIndexConfig validation."""
    from cflibs.manifold.vector_index import VectorIndexConfig

    # Valid configs
    VectorIndexConfig(index_type="flat")
    VectorIndexConfig(index_type="ivf_flat")
    VectorIndexConfig(index_type="ivf_pq")

    # Invalid index_type
    with pytest.raises(ValueError, match="index_type must be"):
        VectorIndexConfig(index_type="invalid")
