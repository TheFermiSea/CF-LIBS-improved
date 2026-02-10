"""
Manifold loading and querying utilities.
"""

from typing import Tuple
import numpy as np
from pathlib import Path

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

from cflibs.core.logging_config import get_logger

logger = get_logger("manifold.loader")


class ManifoldLoader:
    """
    Loader for pre-computed spectral manifolds.

    Provides fast lookup and similarity search capabilities for the manifold.
    """

    def __init__(self, manifold_path: str):
        """
        Initialize manifold loader.

        Parameters
        ----------
        manifold_path : str
            Path to HDF5 manifold file

        Raises
        ------
        ImportError
            If h5py is not installed
        """
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for manifold loading. " "Install with: pip install h5py"
            )

        manifold_path = Path(manifold_path)
        if not manifold_path.exists():
            raise FileNotFoundError(f"Manifold file not found: {manifold_path}")

        self.manifold_path = manifold_path
        self.file = h5py.File(manifold_path, "r")

        # Load data into memory for fast access
        self.spectra = self.file["spectra"][:]
        self.params = self.file["params"][:]
        self.wavelength = self.file["wavelength"][:]

        # Load metadata
        self.elements = list(self.file.attrs.get("elements", []))
        self.wavelength_range = tuple(self.file.attrs.get("wavelength_range", []))
        self.temperature_range = tuple(self.file.attrs.get("temperature_range", []))
        self.density_range = tuple(self.file.attrs.get("density_range", []))

        logger.info(
            f"Loaded manifold: {len(self.spectra)} spectra, "
            f"{len(self.wavelength)} wavelength points"
        )

    def find_nearest_spectrum(
        self,
        measured_spectrum: np.ndarray,
        method: str = "cosine",
        use_jax: bool = False
    ) -> Tuple[int, float, dict]:
        """
        Find nearest matching spectrum in manifold.

        Parameters
        ----------
        measured_spectrum : array
            Measured spectrum (must match manifold wavelength grid)
        method : str
            Similarity method: 'cosine', 'euclidean', 'correlation'
        use_jax : bool
            Use JAX acceleration for similarity search when available

        Returns
        -------
        index : int
            Index of nearest spectrum
        similarity : float
            Similarity score
        params : dict
            Parameters for matched spectrum
        """
        if len(measured_spectrum) != len(self.wavelength):
            raise ValueError(
                f"Spectrum length {len(measured_spectrum)} does not match "
                f"manifold wavelength grid {len(self.wavelength)}"
            )

        if use_jax:
            if not HAS_JAX:
                raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")

            measured_j = jnp.asarray(measured_spectrum)
            spectra_j = jnp.asarray(self.spectra)

            measured_norm = measured_j / (jnp.linalg.norm(measured_j) + 1e-10)
            manifold_norms = spectra_j / (
                jnp.linalg.norm(spectra_j, axis=1, keepdims=True) + 1e-10
            )

            if method == "cosine":
                similarities = jnp.dot(manifold_norms, measured_norm)
            elif method == "euclidean":
                distances = jnp.linalg.norm(spectra_j - measured_j, axis=1)
                similarities = 1.0 / (1.0 + distances)
            elif method == "correlation":
                measured_centered = measured_norm - jnp.mean(measured_norm)
                manifold_centered = manifold_norms - jnp.mean(manifold_norms, axis=1, keepdims=True)
                similarities = jnp.dot(manifold_centered, measured_centered) / (
                    jnp.linalg.norm(manifold_centered, axis=1) * jnp.linalg.norm(measured_centered)
                    + 1e-10
                )
            else:
                raise ValueError(f"Unknown similarity method: {method}")

            best_idx = int(jnp.argmax(similarities))
            best_similarity = float(similarities[best_idx])
        else:
            # Normalize spectra
            measured_norm = measured_spectrum / (np.linalg.norm(measured_spectrum) + 1e-10)
            manifold_norms = self.spectra / (
                np.linalg.norm(self.spectra, axis=1, keepdims=True) + 1e-10
            )

            if method == "cosine":
                # Cosine similarity
                similarities = np.dot(manifold_norms, measured_norm)
            elif method == "euclidean":
                # Euclidean distance (inverted for similarity)
                distances = np.linalg.norm(self.spectra - measured_spectrum, axis=1)
                similarities = 1.0 / (1.0 + distances)
            elif method == "correlation":
                # Pearson correlation
                measured_centered = measured_norm - np.mean(measured_norm)
                manifold_centered = manifold_norms - np.mean(manifold_norms, axis=1, keepdims=True)
                similarities = np.dot(manifold_centered, measured_centered) / (
                    np.linalg.norm(manifold_centered, axis=1) * np.linalg.norm(measured_centered)
                    + 1e-10
                )
            else:
                raise ValueError(f"Unknown similarity method: {method}")

            # Find best match
            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])

        # Extract parameters
        param_array = self.params[best_idx]
        params = {
            "T_eV": float(param_array[0]),
            "n_e_cm3": float(param_array[1]),
        }

        # Add element concentrations
        for i, element in enumerate(self.elements):
            params[element] = float(param_array[2 + i])

        logger.debug(
            f"Best match: index={best_idx}, similarity={best_similarity:.4f}, "
            f"T={params['T_eV']:.3f} eV, ne={params['n_e_cm3']:.2e} cm^-3"
        )

        return best_idx, best_similarity, params

    def get_spectrum(self, index: int) -> Tuple[np.ndarray, dict]:
        """
        Get spectrum and parameters by index.

        Parameters
        ----------
        index : int
            Spectrum index

        Returns
        -------
        spectrum : array
            Spectral intensity
        params : dict
            Parameters
        """
        if index < 0 or index >= len(self.spectra):
            raise IndexError(f"Index {index} out of range [0, {len(self.spectra)})")

        spectrum = self.spectra[index]
        param_array = self.params[index]

        params = {
            "T_eV": float(param_array[0]),
            "n_e_cm3": float(param_array[1]),
        }

        for i, element in enumerate(self.elements):
            params[element] = float(param_array[2 + i])

        return spectrum, params

    def get_wavelength(self) -> np.ndarray:
        """Get wavelength grid."""
        return self.wavelength.copy()

    def close(self):
        """Close manifold file."""
        self.file.close()
        logger.debug("Manifold file closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def build_vector_index(self, n_components: int = 30, index_config=None):
        """
        Build vector index for fast similarity search.

        Parameters
        ----------
        n_components : int
            Number of PCA components for embedding (default: 30)
        index_config : VectorIndexConfig, optional
            FAISS index configuration (default: flat index)

        Returns
        -------
        embedder : SpectralEmbedder
            Fitted spectral embedder
        index : VectorIndex
            Built vector index

        Raises
        ------
        ImportError
            If FAISS or vector_index module not available

        Examples
        --------
        >>> loader = ManifoldLoader("manifold.h5")
        >>> embedder, index = loader.build_vector_index(n_components=20)
        >>> # Use embedder for query spectra, index for search
        """
        from cflibs.manifold.vector_index import SpectralEmbedder, VectorIndex, VectorIndexConfig

        if index_config is None:
            index_config = VectorIndexConfig(index_type="flat")

        logger.info(f"Building vector index with {n_components} components")

        # Fit embedder
        embedder = SpectralEmbedder(n_components=n_components)
        embeddings = embedder.fit_transform(self.spectra)

        # Build index
        vector_index = VectorIndex(dimension=n_components, config=index_config)
        vector_index.build(embeddings)

        logger.info(
            f"Vector index built: {vector_index.index.ntotal} vectors, "
            f"type={index_config.index_type}"
        )

        return embedder, vector_index
