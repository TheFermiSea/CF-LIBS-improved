"""Tests for cflibs.core.cache LRUCache and cache decorators."""

import time

import pytest

from cflibs.core.cache import (
    LRUCache,
    cached_partition_function,
    clear_all_caches,
    get_cache_stats,
)


class TestLRUCacheBasics:
    def test_set_and_get(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_returns_none(self):
        cache = LRUCache(max_size=10)
        assert cache.get("missing") is None

    def test_hit_and_miss_counters(self):
        cache = LRUCache(max_size=10)
        cache.set("k", 1)
        cache.get("k")
        cache.get("k")
        cache.get("missing")
        assert cache.hits == 2
        assert cache.misses == 1

    def test_stats_structure(self):
        cache = LRUCache(max_size=5)
        cache.set("a", 1)
        cache.get("a")
        cache.get("missing")
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 5
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)

    def test_stats_hit_rate_when_empty(self):
        cache = LRUCache(max_size=5)
        stats = cache.stats()
        assert stats["hit_rate"] == 0.0  # NOSONAR — empty cache has hit rate literally 0.0


class TestLRUEviction:
    def test_evict_least_recently_used(self):
        cache = LRUCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.get("a")
        cache.set("d", 4)
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_set_existing_key_updates_order(self):
        cache = LRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("a", 10)
        cache.set("c", 3)
        assert cache.get("b") is None
        assert cache.get("a") == 10
        assert cache.get("c") == 3

    def test_size_does_not_exceed_max(self):
        cache = LRUCache(max_size=3)
        for i in range(10):
            cache.set(f"k{i}", i)
        assert len(cache.cache) == 3


class TestLRUTTL:
    def test_ttl_expires_entries(self):
        cache = LRUCache(max_size=10, ttl_seconds=0.05)
        cache.set("k", "v")
        assert cache.get("k") == "v"
        time.sleep(0.1)
        assert cache.get("k") is None

    def test_no_ttl_means_no_expiry(self):
        cache = LRUCache(max_size=10, ttl_seconds=None)
        cache.set("k", "v")
        time.sleep(0.05)
        assert cache.get("k") == "v"

    def test_expired_entry_counts_as_miss(self):
        cache = LRUCache(max_size=10, ttl_seconds=0.01)
        cache.set("k", "v")
        time.sleep(0.05)
        cache.get("k")
        assert cache.misses == 1
        assert cache.hits == 0


class TestLRUClear:
    def test_clear_removes_entries(self):
        cache = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert len(cache.cache) == 0
        assert cache.get("a") is None

    def test_clear_resets_counters(self):
        cache = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.get("a")
        cache.get("missing")
        cache.clear()
        assert cache.hits == 0
        assert cache.misses == 0


class TestMakeKey:
    def test_same_args_same_key(self):
        cache = LRUCache()
        k1 = cache._make_key(1, 2, x="a")
        k2 = cache._make_key(1, 2, x="a")
        assert k1 == k2

    def test_different_args_different_keys(self):
        cache = LRUCache()
        k1 = cache._make_key(1, 2)
        k2 = cache._make_key(1, 3)
        assert k1 != k2

    def test_kwargs_order_independent(self):
        cache = LRUCache()
        k1 = cache._make_key(x=1, y=2)
        k2 = cache._make_key(y=2, x=1)
        assert k1 == k2


class TestCacheDecorator:
    def test_decorator_caches_results(self):
        clear_all_caches()
        call_count = {"n": 0}

        @cached_partition_function
        def expensive(x):
            call_count["n"] += 1
            return x * 2

        assert expensive(5) == 10
        assert expensive(5) == 10
        assert call_count["n"] == 1

    def test_decorator_different_args_recompute(self):
        clear_all_caches()
        call_count = {"n": 0}

        @cached_partition_function
        def fn(x):
            call_count["n"] += 1
            return x + 1

        fn(1)
        fn(2)
        fn(1)
        assert call_count["n"] == 2

    def test_decorator_preserves_function_metadata(self):
        @cached_partition_function
        def my_func(x):
            """docstring."""
            return x

        assert my_func.__name__ == "my_func"
        assert "docstring" in (my_func.__doc__ or "")


class TestGlobalCacheStats:
    def test_get_cache_stats_returns_all(self):
        clear_all_caches()
        stats = get_cache_stats()
        assert "partition_function" in stats
        assert "transitions" in stats
        assert "ionization" in stats
        for name, s in stats.items():
            assert "size" in s
            assert "hits" in s

    def test_clear_all_caches_empties_all(self):
        @cached_partition_function
        def fn(x):
            return x

        fn(1)
        fn(2)
        clear_all_caches()
        stats = get_cache_stats()
        assert stats["partition_function"]["size"] == 0
        assert stats["partition_function"]["hits"] == 0
