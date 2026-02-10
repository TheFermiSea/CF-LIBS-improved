"""
Vector indexing for fast spectral similarity search.

Provides SpectralEmbedder (PCA + L2 normalization) and VectorIndex (FAISS wrapper)
for efficient approximate nearest neighbor search on pre-computed model spectra.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from pathlib import Path

from cflibs.inversion.pca import PCAPipeline
from cflibs.core.logging_config import get_logger

logger = get_logger("manifold.vector_index")

# Check for FAISS availability
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

# Check for HDF5 availability
try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None


class SpectralEmbedder:
    """
    Spectral embedding pipeline: area-normalize -> PCA -> L2-normalize.

    Converts high-dimensional spectra into compact embeddings suitable for
    fast similarity search using vector indices.

    Parameters
    ----------
    n_components : int
        Number of PCA components to retain (default: 30)

    Attributes
    ----------
    pca_pipeline : PCAPipeline
        Internal PCA pipeline
    is_fitted : bool
        Whether the embedder has been fitted

    Examples
    --------
    >>> embedder = SpectralEmbedder(n_components=20)
    >>> embedder.fit(training_spectra)
    >>> embeddings = embedder.transform(test_spectra)
    >>> # embeddings are L2-normalized PCA scores
    """

    def __init__(self, n_components: int = 30):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")

        self.n_components = n_components
        self.pca_pipeline = PCAPipeline(n_components=n_components, center=True, use_jax=False)

    @property
    def is_fitted(self) -> bool:
        """Whether the embedder has been fitted."""
        return self.pca_pipeline.is_fitted

    def fit(self, spectra: np.ndarray) -> "SpectralEmbedder":
        """
        Fit embedder on training spectra.

        Parameters
        ----------
        spectra : np.ndarray
            Training spectra, shape (n_spectra, n_wavelengths)

        Returns
        -------
        SpectralEmbedder
            Self for chaining
        """
        spectra = np.asarray(spectra, dtype=np.float64)

        if spectra.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {spectra.shape}")

        # Area-normalize
        spectra_normalized = self._area_normalize(spectra)

        # Fit PCA
        self.pca_pipeline.fit(spectra_normalized)

        logger.info(
            f"SpectralEmbedder fitted: {self.n_components} components, "
            f"{len(spectra)} training spectra"
        )

        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """
        Transform spectra to embeddings.

        Parameters
        ----------
        spectra : np.ndarray
            Spectra to transform, shape (n_spectra, n_wavelengths)

        Returns
        -------
        np.ndarray
            Embeddings, shape (n_spectra, n_components), L2-normalized

        Raises
        ------
        RuntimeError
            If embedder has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        spectra = np.asarray(spectra, dtype=np.float64)

        if spectra.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {spectra.shape}")

        # Area-normalize
        spectra_normalized = self._area_normalize(spectra)

        # PCA transform
        pca_scores = self.pca_pipeline.transform(spectra_normalized)

        # L2-normalize
        embeddings = self._l2_normalize(pca_scores)

        return embeddings

    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        """
        Fit embedder and transform in one call.

        Parameters
        ----------
        spectra : np.ndarray
            Training spectra, shape (n_spectra, n_wavelengths)

        Returns
        -------
        np.ndarray
            Embeddings, shape (n_spectra, n_components), L2-normalized
        """
        self.fit(spectra)
        return self.transform(spectra)

    @staticmethod
    def _area_normalize(spectra: np.ndarray) -> np.ndarray:
        """Area-normalize spectra (sum of absolute values = 1)."""
        norms = np.sum(np.abs(spectra), axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms > 1e-10, norms, 1.0)
        return spectra / norms

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors (Euclidean norm = 1)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms > 1e-10, norms, 1.0)
        return vectors / norms


@dataclass
class VectorIndexConfig:
    """
    Configuration for FAISS vector index.

    Attributes
    ----------
    index_type : str
        Index type: "flat" (exact, <100k), "ivf_flat" (100k-10M), "ivf_pq" (>10M)
    n_lists : int
        Number of IVF cells for "ivf_flat" and "ivf_pq" (default: 100)
    n_probe : int
        Number of cells to visit during search for IVF indices (default: 10)
    pq_m : int
        Number of PQ subquantizers for "ivf_pq" (default: 8)
    pq_bits : int
        Bits per PQ code for "ivf_pq" (default: 8)
    """

    index_type: str = "flat"
    n_lists: int = 100
    n_probe: int = 10
    pq_m: int = 8
    pq_bits: int = 8

    def __post_init__(self):
        if self.index_type not in ("flat", "ivf_flat", "ivf_pq"):
            raise ValueError(
                f"index_type must be 'flat', 'ivf_flat', or 'ivf_pq', got {self.index_type}"
            )


class VectorIndex:
    """
    FAISS-based vector index for fast approximate nearest neighbor search.

    Supports three index types:
    - **flat**: Exact search, <100k vectors
    - **ivf_flat**: Inverted file with flat quantizer, 100k-10M vectors
    - **ivf_pq**: Inverted file with product quantization, >10M vectors

    Parameters
    ----------
    dimension : int
        Embedding dimension (must match SpectralEmbedder n_components)
    config : VectorIndexConfig
        Index configuration

    Attributes
    ----------
    dimension : int
        Embedding dimension
    config : VectorIndexConfig
        Index configuration
    index : faiss.Index
        Underlying FAISS index (None before build)
    is_built : bool
        Whether the index has been built

    Examples
    --------
    >>> config = VectorIndexConfig(index_type="flat")
    >>> index = VectorIndex(dimension=30, config=config)
    >>> index.build(embeddings)
    >>> distances, indices = index.search(query_embeddings, k=10)
    """

    def __init__(self, dimension: int, config=None):
        if not HAS_FAISS:
            raise ImportError(
                "faiss is required for VectorIndex. "
                "Install with: pip install faiss-cpu (or faiss-gpu for CUDA)"
            )

        self.dimension = dimension
        self.config = config or VectorIndexConfig()
        self.index = None

    @property
    def is_built(self) -> bool:
        """Whether the index has been built."""
        return self.index is not None

    def build(self, embeddings: np.ndarray):
        """
        Build index from embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to index, shape (n_vectors, dimension)

        Raises
        ------
        ValueError
            If embeddings have wrong dimension
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != expected {self.dimension}"
            )

        n_vectors = len(embeddings)
        logger.info(
            f"Building {self.config.index_type} index: "
            f"{n_vectors} vectors, dimension={self.dimension}"
        )

        # Create index based on type
        if self.config.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)

        elif self.config.index_type == "ivf_flat":
            # IVF with flat quantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.config.n_lists, faiss.METRIC_L2
            )
            # Train with sample (or all data if small)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = self.config.n_probe

        elif self.config.index_type == "ivf_pq":
            # IVF with product quantization
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                self.config.n_lists,
                self.config.pq_m,
                self.config.pq_bits,
            )
            # Train with sample (or all data if small)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = self.config.n_probe

        logger.info(f"Index built: {self.index.ntotal} vectors indexed")

    def search(self, query_embeddings: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Query embeddings, shape (n_queries, dimension)
        k : int
            Number of neighbors to return

        Returns
        -------
        distances : np.ndarray
            Distances to neighbors, shape (n_queries, k)
        indices : np.ndarray
            Indices of neighbors, shape (n_queries, k)

        Raises
        ------
        RuntimeError
            If index has not been built
        """
        if not self.is_built:
            raise RuntimeError("Must call build() before search()")

        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        if query_embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_embeddings.shape[1]} != expected {self.dimension}"
            )

        distances, indices = self.index.search(query_embeddings, k)

        return distances, indices

    def save(self, path: str):
        """
        Save index to HDF5 file.

        Parameters
        ----------
        path : str
            Path to save index

        Raises
        ------
        RuntimeError
            If index has not been built
        ImportError
            If h5py is not available
        """
        if not self.is_built:
            raise RuntimeError("Cannot save index before build()")

        if not HAS_H5PY:
            raise ImportError("h5py is required for saving. Install with: pip install h5py")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize FAISS index to bytes
        faiss_bytes = faiss.serialize_index(self.index)

        # Save to HDF5
        with h5py.File(path, "w") as f:
            f.attrs["dimension"] = self.dimension
            f.attrs["index_type"] = self.config.index_type
            f.attrs["n_lists"] = self.config.n_lists
            f.attrs["n_probe"] = self.config.n_probe
            f.attrs["pq_m"] = self.config.pq_m
            f.attrs["pq_bits"] = self.config.pq_bits

            # Store FAISS index as binary blob
            f.create_dataset("faiss_index", data=np.frombuffer(faiss_bytes, dtype=np.uint8))

        logger.info(f"Index saved to {path}")

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        """
        Load index from HDF5 file.

        Parameters
        ----------
        path : str
            Path to load index from

        Returns
        -------
        VectorIndex
            Loaded index

        Raises
        ------
        ImportError
            If h5py is not available
        FileNotFoundError
            If file does not exist
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required for loading. Install with: pip install h5py")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with h5py.File(path, "r") as f:
            dimension = int(f.attrs["dimension"])
            config = VectorIndexConfig(
                index_type=str(f.attrs["index_type"]),
                n_lists=int(f.attrs["n_lists"]),
                n_probe=int(f.attrs["n_probe"]),
                pq_m=int(f.attrs["pq_m"]),
                pq_bits=int(f.attrs["pq_bits"]),
            )

            # Deserialize FAISS index - convert to numpy array first
            faiss_array = np.array(f["faiss_index"][:], dtype=np.uint8)

        vector_index = cls(dimension=dimension, config=config)
        vector_index.index = faiss.deserialize_index(faiss_array)

        logger.info(f"Index loaded from {path}: {vector_index.index.ntotal} vectors")

        return vector_index
