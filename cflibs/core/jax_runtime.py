"""Runtime helpers for JAX backend capability checks.

These helpers let physics modules keep their numerically strict ``float64``
CPU/CUDA paths while providing explicit reduced-precision fallbacks on
backends such as Metal that do not support ``float64`` or complex dtypes.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from jax.typing import DTypeLike
else:
    DTypeLike = Any

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only when JAX is absent
    HAS_JAX = False
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


def jax_active_backend() -> str | None:
    """
    Return the active JAX backend name.

    The reported backend name is normalized to lowercase for stable
    downstream comparisons.

    Returns
    -------
    str or None
        Lowercase backend identifier such as ``"cpu"``, ``"cuda"``, or
        ``"metal"`` when JAX is available. Returns ``None`` when JAX is not
        installed or the backend cannot be queried.
    """
    if not HAS_JAX:
        return None
    try:
        return str(jax.default_backend()).lower()
    except Exception:
        return None


def jax_backend_supports_x64() -> bool:
    """
    Report whether the active backend supports ``float64`` execution.

    Returns
    -------
    bool
        ``True`` when the active backend can execute ``float64`` kernels.
        Returns ``False`` when JAX is unavailable or when the backend is known
        to reject ``float64`` execution, such as Metal.
    """
    backend = jax_active_backend()
    if backend is None:
        return False
    return backend != "metal"


def jax_backend_supports_complex() -> bool:
    """
    Report whether the active backend supports complex-valued arrays.

    Returns
    -------
    bool
        ``True`` when the active backend supports complex dtypes. Returns
        ``False`` when JAX is unavailable or when the backend is known to
        reject complex execution, such as Metal.
    """
    backend = jax_active_backend()
    if backend is None:
        return False
    return backend != "metal"


def jax_default_real_dtype() -> DTypeLike:
    """
    Return the preferred real dtype for the active backend.

    Returns
    -------
    DTypeLike
        ``jnp.float64`` when JAX x64 mode is enabled and the active backend
        supports ``float64`` execution. Otherwise returns ``jnp.float32``.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    if not HAS_JAX:
        raise ImportError("JAX is not installed")

    x64_enabled = bool(getattr(jax.config, "jax_enable_x64", False))
    if x64_enabled and jax_backend_supports_x64():
        return jnp.float64
    return jnp.float32


def jax_default_complex_dtype() -> DTypeLike:
    """
    Return the preferred complex dtype for the active backend.

    Returns
    -------
    DTypeLike
        ``jnp.complex128`` when the preferred real dtype is
        ``jnp.float64``. Otherwise returns ``jnp.complex64``.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    if not HAS_JAX:
        raise ImportError("JAX is not installed")
    return jnp.complex128 if jax_default_real_dtype() == jnp.float64 else jnp.complex64
