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
    pytest.importorskip("faiss")
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
    pytest.importorskip("faiss")
    from cflibs.manifold.vector_index import VectorIndex

    config = VectorIndexConfig(index_type="flat")
    index = VectorIndex(dimension=30, config=config)

    query = np.random.randn(5, 30).astype(np.float32)

    with pytest.raises(RuntimeError, match="Must call build"):
        index.search(query, k=10)


def test_vector_index_save_load():
    """Test saving and loading index."""
    pytest.importorskip("faiss")
    pytest.importorskip("h5py")
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
        import faiss  # noqa: F401

        pytest.skip("faiss is installed, cannot test ImportError")
    except ImportError:  # faiss not installed; continue to test error handling
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


# ---------------------------------------------------------------------------
# IVF/PQ training-vector guard (#8d)
#
# Faiss trains the IVF coarse quantizer by k-means over n_lists centroids and
# warns (+ produces degenerate near-empty cells) when there are far fewer
# training points than centroids. The rule of thumb is >= ~39 points/centroid
# (MIN_POINTS_PER_CENTROID = 39). The guard must auto-reduce n_lists, or fall
# back to IndexFlatL2, with a logged warning — never silently train a
# pathologically under-trained IVF index.
# ---------------------------------------------------------------------------


def _normed_embeddings(n: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    e = rng.standard_normal((n, d)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e


@pytest.mark.parametrize("index_type", ["ivf_flat", "ivf_pq"])
def test_ivf_too_few_training_vectors_falls_back_to_flat(index_type, caplog):
    """With far fewer than one cell's worth of training vectors, the build
    must fall back to an exact IndexFlatL2 and log a guard warning."""
    pytest.importorskip("faiss")
    import faiss

    from cflibs.manifold.vector_index import VectorIndex

    # n_lists=100 needs >= 3900 training vectors; provide only 20 (< 39 -> can't
    # even fill one cell), forcing the flat fallback.
    config = VectorIndexConfig(index_type=index_type, n_lists=100)
    index = VectorIndex(dimension=8, config=config)
    embeddings = _normed_embeddings(20, 8)

    with caplog.at_level("WARNING", logger="manifold.vector_index"):
        index.build(embeddings)

    assert index.is_built
    assert index.index.ntotal == 20
    # Fell back to an exact flat index, not an IVF index.
    assert isinstance(index.index, faiss.IndexFlatL2)
    assert any("falling back to exact IndexFlatL2" in r.message for r in caplog.records)

    # Search still works post-fallback.
    distances, indices = index.search(embeddings[:2], k=3)
    assert distances.shape == (2, 3)


def test_ivf_flat_few_training_vectors_reduces_n_lists(caplog):
    """When there are enough vectors for >=1 well-populated cell but fewer than
    needed for the configured n_lists, n_lists is reduced (logged) and an
    ivf_flat index is still built."""
    pytest.importorskip("faiss")
    import faiss

    from cflibs.manifold.vector_index import VectorIndex

    # n_lists=10 needs >= 390 vectors; provide 120 -> safe n_lists = 120//39 = 3.
    config = VectorIndexConfig(index_type="ivf_flat", n_lists=10, n_probe=10)
    index = VectorIndex(dimension=8, config=config)
    embeddings = _normed_embeddings(120, 8)

    with caplog.at_level("WARNING", logger="manifold.vector_index"):
        index.build(embeddings)

    assert index.is_built
    assert index.index.ntotal == 120
    # An IVF index (not a flat fallback) was built with a reduced cell count.
    assert isinstance(index.index, faiss.IndexIVF)
    assert index.index.nlist == 120 // 39
    # nprobe must not exceed the reduced n_lists.
    assert index.index.nprobe <= index.index.nlist
    assert any("reducing n_lists" in r.message for r in caplog.records)

    distances, indices = index.search(embeddings[:2], k=3)
    assert distances.shape == (2, 3)


def test_ivf_pq_few_training_vectors_reduces_n_lists(caplog):
    """For ivf_pq the product-quantizer k-means ALSO needs >= 2**pq_bits
    training points (256 at the default pq_bits=8). With enough for the PQ but
    fewer than needed for n_lists, n_lists is reduced and an IVF_PQ index is
    still built."""
    pytest.importorskip("faiss")
    import faiss

    from cflibs.manifold.vector_index import VectorIndex

    # n_lists=20 needs >= 780 vectors; provide 300 (>= 256 for the PQ) ->
    # safe n_lists = 300 // 39 = 7.
    config = VectorIndexConfig(index_type="ivf_pq", n_lists=20, n_probe=10, pq_bits=8, pq_m=2)
    index = VectorIndex(dimension=8, config=config)
    embeddings = _normed_embeddings(300, 8)

    with caplog.at_level("WARNING", logger="manifold.vector_index"):
        index.build(embeddings)

    assert index.is_built
    assert index.index.ntotal == 300
    assert isinstance(index.index, faiss.IndexIVFPQ)
    assert index.index.nlist == 300 // 39
    assert any("reducing n_lists" in r.message for r in caplog.records)

    distances, indices = index.search(embeddings[:2], k=3)
    assert distances.shape == (2, 3)


def test_ivf_pq_too_few_for_product_quantizer_falls_back(caplog):
    """ivf_pq with enough vectors for the IVF coarse quantizer but fewer than
    2**pq_bits for the product quantizer must fall back to a flat index with a
    logged warning (the PQ codebook cannot be trained)."""
    pytest.importorskip("faiss")
    import faiss

    from cflibs.manifold.vector_index import VectorIndex

    # n_lists=2 needs >= 78 vectors (satisfied), but pq_bits=8 needs >= 256;
    # provide 120 -> PQ cannot train -> flat fallback.
    config = VectorIndexConfig(index_type="ivf_pq", n_lists=2, pq_bits=8, pq_m=2)
    index = VectorIndex(dimension=8, config=config)
    embeddings = _normed_embeddings(120, 8)

    with caplog.at_level("WARNING", logger="manifold.vector_index"):
        index.build(embeddings)

    assert index.is_built
    assert index.index.ntotal == 120
    assert isinstance(index.index, faiss.IndexFlatL2)
    assert any("product quantizer" in r.message for r in caplog.records)


@pytest.mark.parametrize("index_type", ["ivf_flat", "ivf_pq"])
def test_ivf_sufficient_training_vectors_no_guard(index_type, caplog):
    """With ample training vectors the configured n_lists is honored and no
    guard warning is emitted."""
    pytest.importorskip("faiss")
    import faiss

    from cflibs.manifold.vector_index import VectorIndex

    # n_lists=4 needs >= 156 vectors; provide 400 — comfortably enough.
    config = VectorIndexConfig(index_type=index_type, n_lists=4, n_probe=2)
    index = VectorIndex(dimension=8, config=config)
    embeddings = _normed_embeddings(400, 8)

    with caplog.at_level("WARNING", logger="manifold.vector_index"):
        index.build(embeddings)

    assert isinstance(index.index, faiss.IndexIVF)
    assert index.index.nlist == 4
    assert not any(
        ("reducing n_lists" in r.message or "falling back to exact" in r.message)
        for r in caplog.records
    )


def test_guarded_n_lists_helper():
    """Unit-test the rule-of-thumb math directly."""
    pytest.importorskip("faiss")
    from cflibs.manifold.vector_index import VectorIndex

    idx = VectorIndex(dimension=8, config=VectorIndexConfig(index_type="ivf_flat", n_lists=100))
    # Enough for the full request (>= 39*100).
    assert idx._guarded_n_lists(3900) == 100
    assert idx._guarded_n_lists(5000) == 100
    # Reduce: 1000 // 39 = 25.
    assert idx._guarded_n_lists(1000) == 1000 // 39
    # Fall back: fewer than 39 -> cannot fill a single cell.
    assert idx._guarded_n_lists(38) is None
    assert idx._guarded_n_lists(0) is None
