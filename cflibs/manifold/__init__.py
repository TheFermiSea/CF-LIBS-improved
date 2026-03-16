"""
Manifold generation for high-throughput CF-LIBS analysis.

This module provides tools for generating pre-computed spectral manifolds
that enable fast inference without solving physics equations at runtime.

The manifold is a high-dimensional lookup table of synthetic spectra
generated from first principles using JAX for GPU acceleration.
"""

from cflibs.manifold.generator import ManifoldGenerator
from cflibs.manifold.config import ManifoldConfig
from cflibs.manifold.loader import ManifoldLoader

try:
    from cflibs.manifold.basis_library import (  # noqa: F401
        BasisLibrary,
        BasisLibraryConfig,
        BasisLibraryGenerator,
    )

    HAS_BASIS_LIBRARY = True
except ImportError:
    HAS_BASIS_LIBRARY = False

# SpectralEmbedder is always available (only needs numpy + pca)
from cflibs.manifold.vector_index import SpectralEmbedder

# VectorIndex and BasisIndex require faiss (optional)
try:
    from cflibs.manifold.vector_index import VectorIndex, VectorIndexConfig, HAS_FAISS  # noqa: F401
    from cflibs.manifold.basis_index import BasisIndex  # noqa: F401

    HAS_VECTOR_INDEX = HAS_FAISS
except ImportError:
    HAS_VECTOR_INDEX = False

__all__ = [
    "ManifoldGenerator",
    "ManifoldConfig",
    "ManifoldLoader",
    "SpectralEmbedder",
    "HAS_BASIS_LIBRARY",
    "HAS_VECTOR_INDEX",
]

if HAS_BASIS_LIBRARY:
    __all__.extend(["BasisLibrary", "BasisLibraryConfig", "BasisLibraryGenerator"])

if HAS_VECTOR_INDEX:
    __all__.extend(["VectorIndex", "VectorIndexConfig", "BasisIndex"])
