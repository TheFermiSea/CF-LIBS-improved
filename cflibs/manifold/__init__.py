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

# SpectralEmbedder is always available (only needs numpy + pca)
from cflibs.manifold.vector_index import SpectralEmbedder

# VectorIndex requires faiss (optional)
try:
    from cflibs.manifold.vector_index import VectorIndex, VectorIndexConfig
    HAS_VECTOR_INDEX = True
except ImportError:
    HAS_VECTOR_INDEX = False

__all__ = [
    "ManifoldGenerator",
    "ManifoldConfig",
    "ManifoldLoader",
    "SpectralEmbedder",
    "HAS_VECTOR_INDEX",
]

if HAS_VECTOR_INDEX:
    __all__.extend(["VectorIndex", "VectorIndexConfig"])
