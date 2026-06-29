"""
Caching utilities for expensive computations.
"""

from functools import wraps
from typing import Callable, Any, Optional, Dict, Tuple
import hashlib
import pickle
import time
from collections import OrderedDict

from cflibs.core.logging_config import get_logger

logger = get_logger("core.cache")


class LRUCache:
    """
    Least Recently Used (LRU) cache with size limit and TTL support.

    This cache automatically evicts least recently used items when
    the cache exceeds max_size, and expires items older than ttl_seconds.
    """

    def __init__(self, max_size: int = 128, ttl_seconds: Optional[float] = None):
        """
        Initialize LRU cache.

        Parameters
        ----------
        max_size : int
            Maximum number of items to cache
        ttl_seconds : float, optional
            Time-to-live in seconds. If None, items never expire.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        # Use pickle to handle complex objects
        key_data = pickle.dumps((args, sorted(kwargs.items())))
        return hashlib.md5(key_data).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item from cache.

        Parameters
        ----------
        key : str
            Cache key
        default : Any, optional
            Value returned on a miss/expiry (default ``None``). Pass a distinct
            sentinel to distinguish a cached ``None`` value from a cache miss —
            the decorator does this so functions that legitimately return
            ``None`` (e.g. ``get_ionization_potential`` for an absent species)
            are cached instead of re-queried on every call.

        Returns
        -------
        Any
            Cached value, or ``default`` if not found or expired.
        """
        if key not in self.cache:
            self.misses += 1
            return default

        value, timestamp = self.cache[key]

        # Check TTL
        if self.ttl_seconds is not None:
            age = time.time() - timestamp
            if age > self.ttl_seconds:
                del self.cache[key]
                self.misses += 1
                logger.debug(f"Cache entry expired: {key[:16]}...")
                return default

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set item in cache.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        timestamp = time.time()

        # Remove if exists
        if key in self.cache:
            self.cache.move_to_end(key)

        # Add new entry
        self.cache[key] = (value, timestamp)

        # Evict if over size limit
        if len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key[:16]}...")

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Cache statistics
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


# Global cache instances
_partition_function_cache = LRUCache(max_size=256, ttl_seconds=3600)  # 1 hour TTL
_transition_cache = LRUCache(max_size=512, ttl_seconds=1800)  # 30 min TTL
_ionization_cache = LRUCache(max_size=128, ttl_seconds=1800)  # 30 min TTL


# Distinct miss sentinel so a cached ``None`` (a legitimate function result,
# e.g. an absent species) is returned as a hit instead of being re-computed.
_MISS = object()


def _make_cache_decorator(cache: LRUCache) -> Callable:
    """Create a caching decorator backed by the given LRUCache instance."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache._make_key(*args, **kwargs)
            cached_value = cache.get(cache_key, _MISS)
            if cached_value is not _MISS:
                return cached_value

            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        return wrapper

    return decorator


cached_partition_function = _make_cache_decorator(_partition_function_cache)
# Cache decorator for partition function queries.

cached_transitions = _make_cache_decorator(_transition_cache)
# Cache decorator for transition queries.

cached_ionization = _make_cache_decorator(_ionization_cache)
# Cache decorator for ionization potential queries.


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for all caches.

    Returns
    -------
    dict
        Dictionary mapping cache name to statistics
    """
    return {
        "partition_function": _partition_function_cache.stats(),
        "transitions": _transition_cache.stats(),
        "ionization": _ionization_cache.stats(),
    }


def clear_all_caches() -> None:
    """Clear all caches."""
    _partition_function_cache.clear()
    _transition_cache.clear()
    _ionization_cache.clear()
    # Also clear the process-global partition spec/level caches (no TTL); lazy
    # import + guard against the partition<->cache import cycle. Without this a
    # long-lived process keeps stale U(T) after the DB's energy_levels change.
    try:
        from cflibs.plasma.partition import clear_partition_module_caches

        clear_partition_module_caches()
    except Exception:  # pragma: no cover - never let cache-clear hard-fail
        pass
    logger.info("All caches cleared")
