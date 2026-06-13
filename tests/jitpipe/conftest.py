"""Test-isolation fixtures for the jitpipe parity suite.

The jitpipe parity tests compare a JAX stage to the **reference** physics on
identical inputs. They are sensitive to PROCESS-GLOBAL state left behind by
upstream tests. The failure is CI-only: it needs the full accumulated suite and
passes in isolation or after any single test dir. Known vectors:

  * Atomic / partition memoization -- ``cflibs.core.cache`` (``clear_all_caches``)
    and ``cflibs.plasma.partition._spec_cache`` / ``_level_cache``. These key on
    ``(element, species, ...)`` WITHOUT the source DB's identity, so a stub/alternate
    DB in an upstream test can poison an entry.
  * ``jax_enable_x64`` flipped off, or ``cflibs.core.jax_runtime._PROCESS_POLICY``
    (``allow_32bit``) left non-default -- the parity contracts require fp64.
  * **The JAX compilation cache** -- a shared kernel (e.g. ``voigt_profile``, used
    by the reference Stark per-line fit) traced/compiled by an upstream test under
    a different global state and then REUSED here. A reused stale-traced kernel can
    change a marginal per-line fit, silently dropping diagnostic lines
    (``test_parity_j6`` ``measure_stark_ne`` n_lines 3 -> 1). ``jax.clear_caches()``
    forces a fresh trace under this suite's fp64 state.

Resetting around the jitpipe tests makes the parity suite hermetic w.r.t.
cross-test global state. ``jax.clear_caches()`` is module-scoped (once per test
file -- it forces recompiles, so per-test would be needlessly slow; the polluter
is upstream, so clearing at each jitpipe module's entry suffices).
"""

from __future__ import annotations

import pytest


def _reset_atomic_caches() -> None:
    try:
        from cflibs.core.cache import clear_all_caches

        clear_all_caches()
    except Exception:  # pragma: no cover - cache module always importable here
        pass
    try:
        from cflibs.plasma import partition as _partition

        _partition._spec_cache.clear()
        _partition._level_cache.clear()
    except Exception:  # pragma: no cover
        pass


def _force_fp64_state() -> None:
    """Re-assert fp64: ``jax_enable_x64`` on, and the process memory policy at its
    fp64 default. The parity contracts require float64; an upstream test that
    flipped either off (and failed to restore) makes the reference physics run
    fp32 here."""
    try:
        import jax

        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)
    except Exception:  # pragma: no cover
        pass
    try:
        from cflibs.core import jax_runtime as _jr

        if getattr(_jr.jax_policy(), "allow_32bit", False):
            _jr.set_jax_policy(type(_jr.jax_policy())())
    except Exception:  # pragma: no cover
        pass


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_compilation_cache():
    """Drop JAX-compiled kernels traced by upstream tests under a different global
    state so this file's parity tests trace fresh under fp64. Once per module."""
    try:
        import jax

        jax.clear_caches()
    except Exception:  # pragma: no cover
        pass
    yield


@pytest.fixture(autouse=True)
def _hermetic_jitpipe_state():
    """Reset atomic/partition caches and re-assert fp64 before & after each test."""
    _reset_atomic_caches()
    _force_fp64_state()
    yield
    _reset_atomic_caches()
