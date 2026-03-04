"""
Core numerical kernels and utilities.

This module provides:
- Physical constants
- Units and unit conversion
- Configuration and logging
- Caching utilities
- Abstract base classes
- Factory patterns
- Connection pooling
"""

from cflibs.core import constants
from cflibs.core import units
from cflibs.core import config
from cflibs.core import logging_config
from cflibs.core.platform_config import configure_jax, AcceleratorBackend
from cflibs.core.cache import (
    LRUCache,
    cached_partition_function,
    cached_transitions,
    cached_ionization,
    get_cache_stats,
    clear_all_caches,
)
from cflibs.core.abc import AtomicDataSource, SolverStrategy, PlasmaModel, InstrumentModelInterface
from cflibs.core.factory import SolverFactory, PlasmaModelFactory, InstrumentFactory
from cflibs.core.pool import DatabaseConnectionPool, get_pool, close_all_pools

__all__ = [
    # Modules
    "constants",
    "units",
    "config",
    "logging_config",
    # Caching
    "LRUCache",
    "cached_partition_function",
    "cached_transitions",
    "cached_ionization",
    "get_cache_stats",
    "clear_all_caches",
    # Abstract base classes
    "AtomicDataSource",
    "SolverStrategy",
    "PlasmaModel",
    "InstrumentModelInterface",
    # Factories
    "SolverFactory",
    "PlasmaModelFactory",
    "InstrumentFactory",
    # Connection pooling
    "DatabaseConnectionPool",
    "get_pool",
    "close_all_pools",
    # Platform configuration
    "configure_jax",
    "AcceleratorBackend",
]
