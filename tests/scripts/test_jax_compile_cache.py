"""Tests for the JAX persistent compile-cache configuration.

T1.2 — CF-LIBS-improved-b5xw.

These tests verify the *configuration plumbing* (env-var defaults, cache
directory creation, no-clobber semantics), not the JAX library's actual
cache behaviour (which is exhaustively covered by JAX's own test suite
and is too slow to re-verify here — a single cold compile takes 15-45 s).

The end-to-end speedup acceptance test (cold ~30 s → warm <2 s, NFS-shared
across hosts) is documented in `docs/jax-compile-cache.md` and is meant
to be run manually after the cluster-side `mkdir -p /cluster/shared/jax-
cache; chmod 1777` setup.
"""
from __future__ import annotations

import importlib
import os
import platform
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_unified_benchmark.py"


# --------------------------------------------------------------------------- #
# run_unified_benchmark.py top-of-file env defaults
# --------------------------------------------------------------------------- #


def _scrub_jax_env(monkeypatch):
    """Drop any pre-existing JAX cache env so `setdefault` semantics fire."""
    for key in (
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_run_unified_benchmark_sets_jax_cache_dir_default(monkeypatch):
    """Importing scripts/run_unified_benchmark.py stamps the NFS cache default."""
    _scrub_jax_env(monkeypatch)
    # Block argparse from running the actual benchmark by simulating --help-ish:
    # use runpy with __name__ != "__main__" by importing as a module via path.
    spec_path = str(SCRIPT_PATH)
    # Read the source and exec the top section only (everything above the
    # `def main(...)`), so we avoid argparse / cflibs imports that pull
    # in the heavy benchmark stack and are out of scope for this test.
    src = SCRIPT_PATH.read_text()
    head_end = src.index("def main(")
    head = src[:head_end]
    ns = {"__name__": "__top__", "__file__": spec_path}
    exec(compile(head, spec_path, "exec"), ns)

    assert os.environ.get("JAX_COMPILATION_CACHE_DIR") == "/cluster/shared/jax-cache"
    assert os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS") == "0.5"


def test_run_unified_benchmark_respects_existing_env(monkeypatch, tmp_path):
    """User-provided cache dir wins over the cluster default (setdefault)."""
    custom = tmp_path / "my-jax-cache"
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(custom))
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2.0")

    src = SCRIPT_PATH.read_text()
    head_end = src.index("def main(")
    head = src[:head_end]
    ns = {"__name__": "__top__", "__file__": str(SCRIPT_PATH)}
    exec(compile(head, str(SCRIPT_PATH), "exec"), ns)

    # setdefault must NOT overwrite caller-provided values.
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == str(custom)
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "2.0"


# --------------------------------------------------------------------------- #
# cflibs.core.platform_config.configure_jax — Linux path stamps the defaults
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="JAX cache cluster default is Linux-only — Darwin path skips it.",
)
def test_configure_jax_stamps_cache_dir_on_linux(monkeypatch):
    _scrub_jax_env(monkeypatch)
    # Reload to re-execute module-level code paths in configure_jax fresh.
    from cflibs.core import platform_config

    importlib.reload(platform_config)
    platform_config.configure_jax(enable_x64=True, prefer_gpu=False)

    assert os.environ.get("JAX_COMPILATION_CACHE_DIR") == "/cluster/shared/jax-cache"
    assert os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS") == "0.5"


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="setdefault-respects-existing semantics are platform-agnostic but "
    "we only set defaults on Linux, so test only there.",
)
def test_configure_jax_respects_existing_cache_env(monkeypatch, tmp_path):
    custom = tmp_path / "test-cache"
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(custom))

    from cflibs.core import platform_config

    importlib.reload(platform_config)
    platform_config.configure_jax(enable_x64=True, prefer_gpu=False)

    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == str(custom)


def test_run_unified_benchmark_script_path_exists():
    """Trip-wire test: the entry point we're patching actually exists.

    If a refactor moves run_unified_benchmark.py, this test fails loudly
    and the cache plumbing tests above stop being silently misleading.
    """
    assert SCRIPT_PATH.exists(), f"Expected entry point at {SCRIPT_PATH}"
    src = SCRIPT_PATH.read_text()
    assert "JAX_COMPILATION_CACHE_DIR" in src
    assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" in src
