"""Runtime helpers for JAX backend capability checks.

These helpers let physics modules keep their numerically strict ``float64``
CPU/CUDA paths while providing explicit reduced-precision fallbacks on
backends such as Metal that do not support ``float64`` or complex dtypes.
"""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only when JAX is absent
    HAS_JAX = False
    jax = None
    jnp = None


def jax_active_backend() -> str | None:
    """Return the active JAX backend name, normalized to lowercase."""
    if not HAS_JAX:
        return None
    try:
        return str(jax.default_backend()).lower()
    except Exception:
        return None


def jax_backend_supports_x64() -> bool:
    """Whether the active backend supports ``float64`` execution."""
    return jax_active_backend() != "metal"


def jax_backend_supports_complex() -> bool:
    """Whether the active backend supports complex-valued arrays."""
    return jax_active_backend() != "metal"


def jax_default_real_dtype():
    """Return the preferred real dtype for the active backend."""
    if not HAS_JAX:
        raise ImportError("JAX is not installed")

    x64_enabled = bool(getattr(jax.config, "jax_enable_x64", False))
    if x64_enabled and jax_backend_supports_x64():
        return jnp.float64
    return jnp.float32


def jax_default_complex_dtype():
    """Return the preferred complex dtype for the active backend."""
    if not HAS_JAX:
        raise ImportError("JAX is not installed")
    return jnp.complex128 if jax_default_real_dtype() == jnp.float64 else jnp.complex64
