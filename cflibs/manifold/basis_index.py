"""
FAISS-based index for fast plasma parameter estimation from basis library.

Wraps SpectralEmbedder + VectorIndex with element/T/ne metadata to enable
neighbor voting for rapid (T, ne) estimation from an observed spectrum.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.manifold.vector_index import SpectralEmbedder, VectorIndex, VectorIndexConfig

logger = get_logger("manifold.basis_index")

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import faiss  # noqa: F401

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class BasisIndex:
    """
    FAISS index over a BasisLibrary for fast plasma parameter estimation.

    Each vector in the index corresponds to one (element, T, ne) combination
    from the basis library.  Given an observed spectrum, the index finds the
    k nearest synthetic spectra and returns the weighted-median (T, ne) of
    those neighbors as the plasma parameter estimate.

    Parameters
    ----------
    n_components : int
        PCA embedding dimension (default: 30)
    index_type : str
        FAISS index type: "flat", "ivf_flat", or "ivf_pq" (default: "flat")

    Examples
    --------
    >>> from cflibs.manifold.basis_library import BasisLibrary
    >>> lib = BasisLibrary("basis_library.h5")
    >>> idx = BasisIndex(n_components=30)
    >>> idx.build_from_library(lib)
    >>> T_est, ne_est, details = idx.estimate_plasma_params(observed_spectrum)
    >>> idx.save("basis_index.h5")
    """

    def __init__(self, n_components: int = 30, index_type: str = "flat"):
        if not HAS_FAISS:
            raise ImportError(
                "faiss is required for BasisIndex. "
                "Install with: pip install faiss-cpu (or faiss-gpu)"
            )
        self.n_components = n_components
        self.index_type = index_type
        self.embedder = SpectralEmbedder(n_components=n_components)
        self.vector_index: Optional[VectorIndex] = None

        # Metadata arrays — one entry per indexed vector
        self._element_indices: Optional[np.ndarray] = None  # int, maps to elements list
        self._grid_indices: Optional[np.ndarray] = None  # int, maps to params array
        self._params: Optional[np.ndarray] = None  # (n_grid, 2) — T_K, ne_cm3
        self._elements: Optional[List[str]] = None

    @property
    def is_built(self) -> bool:
        """Whether the index has been built."""
        return self.vector_index is not None and self.vector_index.is_built

    @property
    def n_vectors(self) -> int:
        """Number of vectors in the index."""
        if self.vector_index is None or not self.vector_index.is_built:
            return 0
        return self.vector_index.index.ntotal

    def build_from_library(self, library, skip_zero: bool = True) -> None:
        """
        Build the FAISS index from a BasisLibrary.

        Parameters
        ----------
        library : BasisLibrary
            An open BasisLibrary instance.
        skip_zero : bool
            If True, skip spectra that are all zeros (elements with no
            transitions in the wavelength range).  Default True.
        """
        n_el = library.n_elements
        n_grid = library.n_grid

        self._elements = library.elements
        self._params = library._params.copy()

        logger.info(
            f"Building basis index: {n_el} elements × {n_grid} grid points "
            f"= {n_el * n_grid} candidate vectors"
        )

        # Flatten (n_el, n_grid, n_pix) → (n_el * n_grid, n_pix) with metadata
        all_spectra = []
        element_indices = []
        grid_indices = []

        for el_idx in range(n_el):
            for g_idx in range(n_grid):
                spec = library._spectra[el_idx, g_idx, :]
                if skip_zero and np.sum(np.abs(spec)) < 1e-20:
                    continue
                all_spectra.append(spec)
                element_indices.append(el_idx)
                grid_indices.append(g_idx)

        if not all_spectra:
            raise ValueError("No non-zero spectra found in library")

        spectra_flat = np.array(all_spectra, dtype=np.float64)
        self._element_indices = np.array(element_indices, dtype=np.int32)
        self._grid_indices = np.array(grid_indices, dtype=np.int32)

        logger.info(
            f"Embedding {len(spectra_flat)} spectra "
            f"({n_el * n_grid - len(spectra_flat)} zero-spectra skipped)"
        )

        # Fit embedder and transform
        embeddings = self.embedder.fit_transform(spectra_flat)

        # Build FAISS index
        config = VectorIndexConfig(index_type=self.index_type)
        self.vector_index = VectorIndex(dimension=self.n_components, config=config)
        self.vector_index.build(embeddings)

        logger.info(f"Basis index built: {self.n_vectors} vectors indexed")

    def estimate_plasma_params(
        self,
        spectrum: np.ndarray,
        k: int = 50,
    ) -> Tuple[float, float, dict]:
        """
        Estimate plasma (T, ne) from an observed spectrum via neighbor voting.

        Parameters
        ----------
        spectrum : np.ndarray
            Observed spectrum, shape (n_pixels,).  Should be baseline-corrected
            and on the same wavelength grid as the basis library.
        k : int
            Number of nearest neighbors to use for voting (default: 50).

        Returns
        -------
        T_K : float
            Estimated plasma temperature in Kelvin.
        ne_cm3 : float
            Estimated electron density in cm^-3.
        details : dict
            Diagnostic information including neighbor elements, distances,
            and per-neighbor (T, ne) values.

        Raises
        ------
        RuntimeError
            If the index has not been built.
        """
        if not self.is_built:
            raise RuntimeError("Must call build_from_library() before estimate_plasma_params()")

        # Narrow Optional types (guaranteed non-None after is_built check)
        assert self.vector_index is not None
        assert self._elements is not None
        assert self._element_indices is not None
        assert self._grid_indices is not None
        assert self._params is not None

        # Embed the observed spectrum
        spec_2d = spectrum.reshape(1, -1)
        embedding = self.embedder.transform(spec_2d)

        # Search
        distances, indices = self.vector_index.search(embedding, k=k)
        distances = distances[0]  # (k,)
        indices = indices[0]  # (k,)

        # Gather metadata for neighbors
        neighbor_elements = [self._elements[self._element_indices[i]] for i in indices]
        neighbor_grid_idx = self._grid_indices[indices]
        neighbor_T = self._params[neighbor_grid_idx, 0]
        neighbor_ne = self._params[neighbor_grid_idx, 1]

        # Weighted median (weight = 1 / (distance + epsilon))
        weights = 1.0 / (distances + 1e-10)
        weights /= np.sum(weights)

        T_est = float(_weighted_median(neighbor_T, weights))
        ne_est = float(_weighted_median(neighbor_ne, weights))

        # Element vote counts
        from collections import Counter

        element_votes = Counter(neighbor_elements)

        details = {
            "neighbor_elements": neighbor_elements,
            "neighbor_T_K": neighbor_T.tolist(),
            "neighbor_ne_cm3": neighbor_ne.tolist(),
            "distances": distances.tolist(),
            "weights": weights.tolist(),
            "element_votes": dict(element_votes.most_common()),
            "k": k,
        }

        return T_est, ne_est, details

    def save(self, path: str | Path) -> None:
        """
        Save the index, embedder, and metadata to HDF5.

        Parameters
        ----------
        path : str
            Output file path.
        """
        if not self.is_built:
            raise RuntimeError("Cannot save index before build_from_library()")
        if not HAS_H5PY:
            raise ImportError("h5py is required for saving")

        # Narrow Optional types (guaranteed non-None after is_built check)
        assert self.vector_index is not None
        assert self._element_indices is not None
        assert self._grid_indices is not None
        assert self._params is not None
        assert self._elements is not None

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as f:
            # Metadata arrays
            f.create_dataset("element_indices", data=self._element_indices)
            f.create_dataset("grid_indices", data=self._grid_indices)
            f.create_dataset("params", data=self._params)
            dt = h5py.string_dtype()
            f.create_dataset("elements", data=self._elements, dtype=dt)

            # Embedder: PCA components + mean
            pca_result = self.embedder.pca_pipeline.result_
            assert pca_result is not None
            f.create_dataset("pca_components", data=pca_result.components)
            f.create_dataset("pca_mean", data=pca_result.mean)

            # FAISS index as binary blob
            import faiss as _faiss

            faiss_bytes = _faiss.serialize_index(self.vector_index.index)
            f.create_dataset("faiss_index", data=np.frombuffer(faiss_bytes, dtype=np.uint8))

            # Config
            f.attrs["n_components"] = self.n_components
            f.attrs["index_type"] = self.index_type

        logger.info(f"Basis index saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BasisIndex":
        """
        Load a saved BasisIndex from HDF5.

        Parameters
        ----------
        path : str
            Path to the saved index file.

        Returns
        -------
        BasisIndex
            Loaded index ready for queries.
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required for loading")

        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Index file not found: {in_path}")

        with h5py.File(in_path, "r") as f:
            n_components = int(f.attrs["n_components"])
            index_type = str(f.attrs["index_type"])

            idx = cls(n_components=n_components, index_type=index_type)

            # Metadata
            idx._element_indices = f["element_indices"][:]
            idx._grid_indices = f["grid_indices"][:]
            idx._params = f["params"][:]
            idx._elements = [e.decode() if isinstance(e, bytes) else e for e in f["elements"][:]]

            # Embedder — reconstruct a minimal PCAResult so transform() works
            pca_components = f["pca_components"][:]
            pca_mean = f["pca_mean"][:]

            from cflibs.inversion.pca import PCAResult

            n_feat = pca_components.shape[1]
            minimal_result = PCAResult(
                n_components=n_components,
                components=pca_components,
                singular_values=np.ones(n_components),
                explained_variance=np.ones(n_components),
                explained_variance_ratio=np.ones(n_components) / n_components,
                mean=pca_mean,
                n_samples=1,
                n_features=n_feat,
                total_variance=1.0,
            )
            idx.embedder.pca_pipeline.result_ = minimal_result

            # FAISS index
            import faiss as _faiss

            faiss_bytes = np.array(f["faiss_index"][:], dtype=np.uint8)
            config = VectorIndexConfig(index_type=index_type)
            idx.vector_index = VectorIndex(dimension=n_components, config=config)
            idx.vector_index.index = _faiss.deserialize_index(faiss_bytes)

        logger.info(f"Basis index loaded from {path}: {idx.n_vectors} vectors")
        return idx


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median of values."""
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cum_weight = np.cumsum(sorted_weights)
    # np.searchsorted returns np.intp; coerce to int so min() returns int.
    median_idx = int(np.searchsorted(cum_weight, 0.5))
    median_idx = min(median_idx, len(sorted_vals) - 1)
    return float(sorted_vals[median_idx])
