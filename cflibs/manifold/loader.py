"""
Manifold loading and querying utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    zarr = None

try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None  # type: ignore[assignment]

from cflibs.core.logging_config import get_logger

logger = get_logger("manifold.loader")


def _infer_storage_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".zarr":
        return "zarr"
    if suffix in {".h5", ".hdf5", ".hdf"}:
        return "hdf5"
    return "zarr" if path.is_dir() else "hdf5"


def _normalize_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        raw = [raw]
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw]


def _normalize_numeric_tuple(raw: Any) -> tuple[float, ...]:
    if raw is None:
        return ()
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return tuple(float(value) for value in raw)


class _DatasetView:
    """Thin compatibility wrapper around HDF5/Zarr arrays."""

    def __init__(self, dataset: Any):
        self._dataset = dataset

    def __getitem__(self, item):
        return self._dataset[item]

    def __setitem__(self, item, value):
        self._dataset[item] = value

    def __len__(self) -> int:
        return int(self._dataset.shape[0])

    def __array__(self, dtype=None):
        array = np.asarray(self._dataset[:])
        if dtype is not None:
            return array.astype(dtype)
        return array

    @property
    def shape(self):
        return self._dataset.shape

    @property
    def chunks(self):
        return getattr(self._dataset, "chunks", None)


class ManifoldLoader:
    """
    Loader for pre-computed spectral manifolds.

    Provides fast lookup and similarity search capabilities for the manifold.
    Uses dataset-backed arrays for HDF5/Zarr so large manifolds can be queried
    without eagerly loading the full cube into memory.
    """

    def __init__(self, manifold_path: str):
        """
        Initialize manifold loader.

        Parameters
        ----------
        manifold_path : str
            Path to HDF5 or Zarr manifold store

        Raises
        ------
        ImportError
            If the required backend is not installed
        """
        path = Path(manifold_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifold file not found: {path}")

        self.manifold_path = path
        self.storage_format = _infer_storage_format(path)
        self.file = None
        self.root = None

        if self.storage_format == "hdf5":
            if not HAS_H5PY:
                raise ImportError(
                    "h5py is required for HDF5 manifold loading. Install with: pip install h5py"
                )
            self.file = h5py.File(path, "r")
            self.root = self.file
        elif self.storage_format == "zarr":
            if not HAS_ZARR:
                raise ImportError(
                    "zarr is required for Zarr manifold loading. Install with: pip install zarr"
                )
            self.root = zarr.open_group(str(path), mode="r")
        else:
            raise ValueError(f"Unsupported manifold storage format: {self.storage_format}")

        self.spectra = _DatasetView(self.root["spectra"])
        self.params = _DatasetView(self.root["params"])
        self.wavelength = np.asarray(self.root["wavelength"], dtype=np.float32)

        attrs = self.root.attrs
        self.elements = _normalize_string_list(attrs.get("elements", []))
        self.wavelength_range = _normalize_numeric_tuple(attrs.get("wavelength_range", ()))
        self.temperature_range = _normalize_numeric_tuple(attrs.get("temperature_range", ()))
        self.density_range = _normalize_numeric_tuple(attrs.get("density_range", ()))
        self.n_spectra = int(self.spectra.shape[0])

        logger.info(
            f"Loaded {self.storage_format.upper()} manifold: {self.n_spectra} spectra, "
            f"{len(self.wavelength)} wavelength points"
        )

    def _default_chunk_size(self) -> int:
        chunks = getattr(self.spectra, "chunks", None)
        if chunks and len(chunks) > 0 and chunks[0]:
            return max(int(chunks[0]), 1)
        return 4096

    def _params_dict(self, index: int) -> dict:
        param_array = np.asarray(self.params[index], dtype=np.float32)
        params = {
            "T_eV": float(param_array[0]),
            "n_e_cm3": float(param_array[1]),
        }
        for i, element in enumerate(self.elements):
            params[element] = float(param_array[2 + i])
        return params

    def find_nearest_spectrum(
        self,
        measured_spectrum: np.ndarray,
        method: str = "cosine",
        use_jax: bool = False,
        search_batch_size: int | None = None,
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
        search_batch_size : int, optional
            Number of manifold spectra to evaluate per chunk. Defaults to the
            dataset chunk size or 4096 when chunk metadata is unavailable.

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
        if method not in {"cosine", "euclidean", "correlation"}:
            raise ValueError(f"Unknown similarity method: {method}")
        if use_jax and not HAS_JAX:
            raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")
        if self.n_spectra == 0:
            raise ValueError("Cannot search an empty manifold")

        batch_size = search_batch_size or self._default_chunk_size()
        measured_np = np.asarray(measured_spectrum, dtype=np.float32)
        measured_norm_np = measured_np / (np.linalg.norm(measured_np) + 1e-10)
        measured_centered_np = measured_norm_np - np.mean(measured_norm_np)

        if use_jax:
            measured_j = jnp.asarray(measured_np)
            measured_norm_j = measured_j / (jnp.linalg.norm(measured_j) + 1e-10)
            measured_centered_j = measured_norm_j - jnp.mean(measured_norm_j)

        best_idx = -1
        best_similarity = -np.inf

        for start in range(0, self.n_spectra, batch_size):
            stop = min(start + batch_size, self.n_spectra)
            spectra_chunk_np = np.asarray(self.spectra[start:stop], dtype=np.float32)

            if use_jax:
                spectra_chunk = jnp.asarray(spectra_chunk_np)
                if method == "cosine":
                    chunk_norms = spectra_chunk / (
                        jnp.linalg.norm(spectra_chunk, axis=1, keepdims=True) + 1e-10
                    )
                    similarities = jnp.dot(chunk_norms, measured_norm_j)
                elif method == "euclidean":
                    distances = jnp.linalg.norm(spectra_chunk - measured_j, axis=1)
                    similarities = 1.0 / (1.0 + distances)
                else:
                    chunk_norms = spectra_chunk / (
                        jnp.linalg.norm(spectra_chunk, axis=1, keepdims=True) + 1e-10
                    )
                    chunk_centered = chunk_norms - jnp.mean(chunk_norms, axis=1, keepdims=True)
                    similarities = jnp.dot(chunk_centered, measured_centered_j) / (
                        jnp.linalg.norm(chunk_centered, axis=1)
                        * jnp.linalg.norm(measured_centered_j)
                        + 1e-10
                    )
                similarities_np = np.asarray(similarities, dtype=np.float32)
            else:
                if method == "cosine":
                    chunk_norms = spectra_chunk_np / (
                        np.linalg.norm(spectra_chunk_np, axis=1, keepdims=True) + 1e-10
                    )
                    similarities = np.dot(chunk_norms, measured_norm_np)
                elif method == "euclidean":
                    distances = np.linalg.norm(spectra_chunk_np - measured_np, axis=1)
                    similarities = 1.0 / (1.0 + distances)
                else:
                    chunk_norms = spectra_chunk_np / (
                        np.linalg.norm(spectra_chunk_np, axis=1, keepdims=True) + 1e-10
                    )
                    chunk_centered = chunk_norms - np.mean(chunk_norms, axis=1, keepdims=True)
                    similarities = np.dot(chunk_centered, measured_centered_np) / (
                        np.linalg.norm(chunk_centered, axis=1)
                        * np.linalg.norm(measured_centered_np)
                        + 1e-10
                    )
                similarities_np = np.asarray(similarities, dtype=np.float32)

            finite_mask = np.isfinite(similarities_np)
            if not np.any(finite_mask):
                continue
            masked_similarities = np.where(finite_mask, similarities_np, -np.inf)
            local_idx = int(np.argmax(masked_similarities))
            local_similarity = float(masked_similarities[local_idx])

            if local_similarity > best_similarity:
                best_similarity = local_similarity
                best_idx = start + local_idx

        if best_idx < 0:
            raise ValueError("Unable to compute a finite similarity score for the manifold search")

        params = self._params_dict(best_idx)
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
        if index < 0 or index >= self.n_spectra:
            raise IndexError(f"Index {index} out of range [0, {self.n_spectra})")

        spectrum = np.asarray(self.spectra[index], dtype=np.float32)
        return spectrum, self._params_dict(index)

    def get_wavelength(self) -> np.ndarray:
        """Get wavelength grid."""
        return self.wavelength.copy()

    def close(self):
        """Close manifold file."""
        if self.file is not None:
            self.file.close()
            self.file = None
        logger.debug("Manifold store closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def build_vector_index(self, n_components: int = 30, index_config=None):
        """
        Build vector index for fast similarity search.

        Notes
        -----
        ``build_vector_index()`` materializes ``self.spectra[:]`` with
        ``np.asarray(..., dtype=np.float32)`` before fitting
        ``SpectralEmbedder`` and ``VectorIndex``. That eager load is expected
        because the current PCA embedding path is in-memory, but it means large
        manifolds should be subsampled or preprocessed with an external
        out-of-core workflow before calling this method.

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
        """
        from cflibs.manifold.vector_index import SpectralEmbedder, VectorIndex, VectorIndexConfig

        if index_config is None:
            index_config = VectorIndexConfig(index_type="flat")

        logger.info(f"Building vector index with {n_components} components")

        embedder = SpectralEmbedder(n_components=n_components)
        embeddings = embedder.fit_transform(np.asarray(self.spectra[:], dtype=np.float32))

        vector_index = VectorIndex(dimension=n_components, config=index_config)
        vector_index.build(embeddings)

        logger.info(
            f"Vector index built: {vector_index.index.ntotal} vectors, "
            f"type={index_config.index_type}"
        )

        return embedder, vector_index
