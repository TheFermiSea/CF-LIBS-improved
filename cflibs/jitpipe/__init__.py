"""``cflibs.jitpipe`` — the jittable inversion pipeline (ADR-0004).

A parallel, end-to-end jit/vmap-able re-implementation of the CF-LIBS inversion
pipeline. Per ADR-0004 D1 this is a *parallel* implementation: nothing outside
``cflibs.jitpipe`` imports it, and it never modifies the two pre-existing
snapshot types it unifies.

Public API
----------
- :class:`PipelineSnapshot` — one frozen, pytree-registered struct-of-arrays
  bundle of the whole atomic DB (host-built, ``.npz``-cached). See
  :mod:`cflibs.jitpipe.snapshot`.
- :class:`PipelineParams` — traced pytree of every continuous knob.
- :class:`StaticConfig` — hashable jit cache key (statics).
- :func:`run_one`, :func:`run_batch` — single / batched pipeline entry points
  (stage logic lands in J1-J7; the J0 skeleton wires the signatures).

JAX requirement
---------------
``jitpipe`` requires JAX by definition (ADR-0004 §5.5). Importing the package
without JAX raises a clear :class:`ImportError` with install guidance, rather
than degrading to a NumPy fallback like the rest of ``cflibs``.
"""

from __future__ import annotations

# jitpipe REQUIRES JAX — fail loudly and early with install guidance.
try:
    import jax as _jax  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "cflibs.jitpipe requires JAX, which is not installed. The jittable "
        "inversion pipeline has no NumPy fallback (ADR-0004 §5.5). Install it "
        "with one of:\n"
        "    uv pip install -e '.[local]'    # Apple Silicon (JAX Metal)\n"
        "    uv pip install -e '.[cluster]'  # NVIDIA GPU (JAX CUDA)\n"
        "    pip install 'jax[cpu]'          # CPU-only\n"
        "If you only need the reference (non-jit) pipeline, use "
        "cflibs.inversion instead."
    ) from exc

from cflibs.jitpipe.params import PipelineParams, StaticConfig
from cflibs.jitpipe.pipeline import run_batch, run_one
from cflibs.jitpipe.snapshot import PipelineSnapshot, build_snapshot

__all__ = [
    "PipelineSnapshot",
    "PipelineParams",
    "StaticConfig",
    "build_snapshot",
    "run_one",
    "run_batch",
]
