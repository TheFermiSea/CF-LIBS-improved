"""Runtime helpers for JAX backend capability checks.

These helpers let physics modules keep their numerically strict ``float64``
CPU/CUDA paths while providing explicit reduced-precision fallbacks on
backends such as Metal that do not support ``float64`` or complex dtypes.

In addition to capability probes, this module is the **single approved
adapter** for optional ``jax`` consumption inside the shipped CF-LIBS
algorithm. Modules under ``cflibs/`` should decorate functions with
``@jit_if_available`` / ``@vmap_if_available`` and consume ``jnp`` from
this module rather than wiring up their own ``try: import jax`` blocks.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal, TYPE_CHECKING, TypeVar

import numpy as _np

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
    jnp = _np  # type: ignore[assignment]


F = TypeVar("F", bound=Callable[..., Any])

JAX_BACKEND: Literal["cpu", "cuda", "metal", "none"]


def _resolve_backend() -> Literal["cpu", "cuda", "metal", "none"]:
    if not HAS_JAX:
        return "none"
    try:
        name = str(jax.default_backend()).lower()
    except Exception:
        return "none"
    if name in ("cpu", "cuda", "metal"):
        return name  # type: ignore[return-value]
    return "cpu"


JAX_BACKEND = _resolve_backend()
"""Module-level snapshot of jax_active_backend() resolved at import time.

Equals ``'none'`` when ``HAS_JAX`` is False. Recompute via
``_refresh_runtime_state()`` after manipulating ``jax.config``.
"""


def _refresh_runtime_state() -> None:
    """Recompute module-level cached JAX state. Test helper."""
    global JAX_BACKEND
    JAX_BACKEND = _resolve_backend()


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


# ---------------------------------------------------------------------------
# T1-1 additions: shared jit/vmap decorators, JaxMemoryPolicy, x64 guard,
# AtomicSnapshot carrier.
# ---------------------------------------------------------------------------


def _looks_like_decorator_invocation(args: tuple) -> bool:
    """Detect bare-decorator usage ``@jit_if_available`` (a single callable arg)."""
    return len(args) == 1 and callable(args[0])


def jit_if_available(*jit_args: Any, **jit_kwargs: Any) -> Any:
    """Decorator that applies ``jax.jit`` when JAX is importable, else returns
    the wrapped function unchanged.

    Drop-in replacement for the ad-hoc ``try: import jax`` blocks scattered
    across ``cflibs/``. Supports both invocation styles:

    >>> @jit_if_available
    ... def _broaden(wl, lines, sigmas): ...

    >>> @jit_if_available(static_argnums=(2,))
    ... def _kernel(x, y, n): ...

    When JAX is missing the wrapped function still runs against
    ``cflibs.core.jax_runtime.jnp`` (aliased to ``numpy`` in the fallback).
    """
    if _looks_like_decorator_invocation(jit_args) and not jit_kwargs:
        func = jit_args[0]
        if not HAS_JAX:
            return func
        return jax.jit(func)

    def _decorator(func: F) -> F:
        if not HAS_JAX:
            return func
        return jax.jit(func, *jit_args, **jit_kwargs)  # type: ignore[return-value]

    return _decorator


def _vmap_decorator(
    func: F,
    in_axes: Any = 0,
    out_axes: Any = 0,
    axis_name: Any = None,
) -> F:
    if HAS_JAX:
        return jax.vmap(func, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name)  # type: ignore[return-value]

    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        if isinstance(in_axes, tuple):
            axes = in_axes
        else:
            axes = tuple(in_axes if i < len(args) else None for i in range(len(args)))
        n_batch: int | None = None
        for arg, axis in zip(args, axes):
            if axis is None:
                continue
            arr = _np.asarray(arg)
            if arr.ndim == 0:
                raise ValueError("vmap_if_available fallback requires array-like batched arguments")
            n_batch = arr.shape[axis]
            break
        if n_batch is None:
            return func(*args, **kwargs)

        outputs = []
        for i in range(n_batch):
            sliced = []
            for arg, axis in zip(args, axes):
                if axis is None:
                    sliced.append(arg)
                else:
                    sliced.append(_np.take(_np.asarray(arg), i, axis=axis))
            outputs.append(func(*sliced, **kwargs))
        if isinstance(outputs[0], tuple):
            return tuple(_np.stack(parts, axis=out_axes) for parts in zip(*outputs))
        return _np.stack(outputs, axis=out_axes)

    return _wrapper  # type: ignore[return-value]


def vmap_if_available(*vmap_args: Any, **vmap_kwargs: Any) -> Any:
    """``jax.vmap``-compatible wrapper.

    Supports the two ``jax.vmap`` invocation styles:

    >>> vmap_if_available(f)                      # direct call form
    >>> vmap_if_available(f, in_axes=(0, None))   # direct call with kwargs

    And the decorator-factory style used elsewhere in cflibs:

    >>> @vmap_if_available(in_axes=0)
    ... def batched_fn(x): ...

    When JAX is missing the fallback is a NumPy-loop emulation (``numpy.stack``
    along ``out_axes``). Correctness-only fallback; performance parity not
    promised. Only ``in_axes=0`` (or a flat tuple thereof) is supported.
    """
    if _looks_like_decorator_invocation(vmap_args):
        return _vmap_decorator(vmap_args[0], **vmap_kwargs)

    def _decorator(func: F) -> F:
        return _vmap_decorator(func, *vmap_args, **vmap_kwargs)

    return _decorator


@dataclass(frozen=True)
class JaxMemoryPolicy:
    """Centralised knob-set for JAX precision, chunking, OOM mitigation.

    Mirrors exojax's ``opacity/policies.py::MemoryPolicy`` (ADR-0001 §5.2 C-P9).
    Hashable so it can be keyed into a jit cache.

    Notes
    -----
    Default ``allow_32bit=False`` is safe for cluster (cpu/cuda) but
    Metal-backed callers MUST construct with ``allow_32bit=True`` or call
    ``check_jax64bit(allow_fp32_on_metal=True)`` at entry; Metal has no
    ``float64``.
    """

    allow_32bit: bool = False
    nstitch: int = 1
    cutwing: float = 50.0
    checkpoint: bool = False
    overlap_factor: float = 4.0

    @property
    def real_dtype(self) -> DTypeLike:
        """``jnp.float32`` if ``allow_32bit`` else ``jnp.float64``.

        Falls back to ``numpy`` dtypes when JAX is unavailable.
        """
        if not HAS_JAX:
            return _np.float32 if self.allow_32bit else _np.float64
        return jnp.float32 if self.allow_32bit else jnp.float64


_PROCESS_POLICY: JaxMemoryPolicy = JaxMemoryPolicy()


def jax_policy() -> JaxMemoryPolicy:
    """Return the process-default :class:`JaxMemoryPolicy`.

    Override via :func:`set_jax_policy`. T1-2/T1-5 consume this to source
    ``real_dtype`` and ``nstitch`` defaults.
    """
    return _PROCESS_POLICY


def set_jax_policy(policy: JaxMemoryPolicy) -> None:
    """Replace the process-default policy.

    Test-only — production callers should pass :class:`JaxMemoryPolicy`
    explicitly to ``SpectrumModel`` / ``ManifoldGenerator``.
    """
    global _PROCESS_POLICY
    _PROCESS_POLICY = policy


def check_jax64bit(allow_fp32_on_metal: bool = False, raise_on_violation: bool = True) -> None:
    """Runtime guard for ``float64`` consumers.

    Mirrors exojax's ``utils/jaxstatus.py``. Verifies:

    1. ``jax.config.jax_enable_x64`` is True (``conftest.py`` sets this).
    2. :func:`jax_backend_supports_x64` returns True.

    Parameters
    ----------
    allow_fp32_on_metal : bool, optional
        When True and the active backend is Metal, log a WARNING rather
        than raise.
    raise_on_violation : bool, optional
        Whether to raise ``ValueError`` on violation. When False, only log.

    Raises
    ------
    ValueError
        On x64 violation when ``raise_on_violation=True`` and the violation
        is not waived by ``allow_fp32_on_metal``.
    """
    if not HAS_JAX:
        msg = "JAX is not installed; check_jax64bit cannot enforce x64 mode"
        if raise_on_violation:
            raise ValueError(msg)
        logging.getLogger("cflibs.core.jax_runtime").warning(msg)
        return

    x64_enabled = bool(getattr(jax.config, "jax_enable_x64", False))
    backend = jax_active_backend()
    supports = jax_backend_supports_x64()

    if backend == "metal" and allow_fp32_on_metal:
        logging.getLogger("cflibs.core.jax_runtime").warning(
            "check_jax64bit: Metal backend lacks float64; running in fp32."
        )
        return

    if not x64_enabled or not supports:
        msg = (
            f"JAX x64 not active (jax_enable_x64={x64_enabled}, "
            f"backend={backend}, supports_x64={supports}). "
            "Set jax.config.update('jax_enable_x64', True) before importing CF-LIBS."
        )
        if raise_on_violation:
            raise ValueError(msg)
        logging.getLogger("cflibs.core.jax_runtime").warning(msg)


@dataclass(frozen=True)
class AtomicSnapshot:
    """Frozen, jit-friendly snapshot of an :class:`AtomicDatabase` query.

    Mirrors exojax ``database/contracts.py::MDBSnapshot`` (ADR-0001 §5.2
    C-P10). Decouples jit-traced kernels from the SQLite connection.

    All field names use the canonical ``_nm`` / ``_ev`` / ``_cm3`` suffix
    convention.

    Attributes
    ----------
    species : tuple[tuple[str, int], ...]
        ``((element, ion_stage), ...)`` mapping the species axis. Static
        metadata; NOT a pytree leaf.
    line_wavelengths_nm : ndarray, shape (N_lines,)
        Line center wavelengths in nm. Canonical name — do not drop suffix.
    line_A_ki : ndarray, shape (N_lines,)
        Einstein spontaneous-emission coefficient, s^-1.
    line_E_k_ev : ndarray, shape (N_lines,)
        Upper-level energy in eV.
    line_g_k : ndarray, shape (N_lines,)
        Upper-level statistical weight.
    line_E_i_ev : ndarray, shape (N_lines,)
        Lower-level energy in eV (used for self-absorption).
    line_g_i : ndarray, shape (N_lines,)
        Lower-level statistical weight.
    line_species_index : ndarray, shape (N_lines,)
        ``int32`` index into ``species`` for each line.
    line_stark_w : ndarray, shape (N_lines,)
        Stark width parameter; 0.0 when missing.
    line_stark_alpha : ndarray, shape (N_lines,)
        Stark width temperature-power-law exponent. Used by
        :func:`cflibs.radiation.kernels._per_line_stark_gamma` to apply
        ``factor_T = (T_eV / 0.86173) ** (-alpha)``. Lines without
        catalogued temperature dependence carry ``alpha = 0.0`` (the
        :meth:`AtomicDatabase.snapshot` default for missing DB entries),
        which collapses the factor to 1.0 — i.e. the canonical formula
        degrades gracefully to temperature-independent Stark for those
        rows without any kernel-side branching.
    line_natural_w : ndarray, shape (N_lines,)
        Natural broadening width; 0.0 when missing.
    partition_coeffs : ndarray, shape (N_species, N_poly_order)
        Polynomial coefficients for ``log U(T)``.
    ionization_potential_ev : ndarray, shape (N_species,)
        Neutral ionization potential in eV.
    level_g, level_E_ev, level_mask : optional ndarrays
        Padded level-resolved arrays for direct-sum partition functions
        (T1-3 consumer). Absent on snapshots built without
        ``include_levels=True``.
    """

    species: tuple[tuple[str, int], ...]

    line_wavelengths_nm: Any
    line_A_ki: Any
    line_E_k_ev: Any
    line_g_k: Any
    line_E_i_ev: Any
    line_g_i: Any
    line_species_index: Any
    line_stark_w: Any
    line_stark_alpha: Any
    line_natural_w: Any

    partition_coeffs: Any
    ionization_potential_ev: Any

    level_g: Any = None
    level_E_ev: Any = None
    level_mask: Any = None

    # Per-species validity-window + g0 arrays for the
    # ``BatchedPartitionFunctionProvider`` (arch candidate 4).  Shape
    # ``(N_species,)``.  Optional / nullable to keep the snapshot
    # back-compatible with callers built before the provider rollout — when
    # ``None`` the legacy ``partition_coeffs``-only path is used and the
    # polynomial extrapolates without clamping (the same behaviour shipped
    # before the s1qr.1 fix).  ``partition_g0`` defaults to 1.0 per
    # species in the snapshot builder; ``partition_t_min/max`` default to
    # the standard CF-LIBS DB window (2000–25000 K) when no row is
    # available.
    partition_t_min: Any = None
    partition_t_max: Any = None
    partition_g0: Any = None


# Register AtomicSnapshot as a pytree so it can flow through jit/vmap.
if HAS_JAX:
    _SNAPSHOT_LEAF_FIELDS: tuple[str, ...] = (
        "line_wavelengths_nm",
        "line_A_ki",
        "line_E_k_ev",
        "line_g_k",
        "line_E_i_ev",
        "line_g_i",
        "line_species_index",
        "line_stark_w",
        "line_stark_alpha",
        "line_natural_w",
        "partition_coeffs",
        "ionization_potential_ev",
        "level_g",
        "level_E_ev",
        "level_mask",
        "partition_t_min",
        "partition_t_max",
        "partition_g0",
    )

    def _snapshot_flatten(snap: AtomicSnapshot):
        children = tuple(getattr(snap, name) for name in _SNAPSHOT_LEAF_FIELDS)
        aux = (snap.species,)
        return children, aux

    def _snapshot_unflatten(aux: tuple, children: tuple) -> AtomicSnapshot:
        (species,) = aux
        kwargs = dict(zip(_SNAPSHOT_LEAF_FIELDS, children))
        return AtomicSnapshot(species=species, **kwargs)

    jax.tree_util.register_pytree_node(AtomicSnapshot, _snapshot_flatten, _snapshot_unflatten)


# ---------------------------------------------------------------------------
# Minimal pytree registration for SingleZoneLTEPlasma + InstrumentModel.
# Wired here (rather than each class's home module) to keep host modules
# free of unconditional ``jax`` imports and to centralise the leaf/aux
# convention. T1-2 consumes this so it can vmap over a plasma batch.
# ---------------------------------------------------------------------------


def _register_cflibs_pytrees() -> None:
    if not HAS_JAX:
        return

    from cflibs.plasma.state import SingleZoneLTEPlasma
    from cflibs.instrument.model import InstrumentModel

    def _plasma_flatten(plasma: "SingleZoneLTEPlasma"):
        elements = tuple(plasma.species.keys())
        dtype = jax_default_real_dtype()
        # Emit one leaf per species so tree_map(stack, plasma) cleanly batches
        # the species dict instead of collapsing it into a 2-D array.
        density_leaves = tuple(
            v if hasattr(v, "shape") else jnp.asarray(float(v), dtype=dtype)
            for v in (plasma.species[e] for e in elements)
        )
        T_e = (
            plasma.T_e
            if hasattr(plasma.T_e, "shape")
            else jnp.asarray(float(plasma.T_e), dtype=dtype)
        )
        n_e = (
            plasma.n_e
            if hasattr(plasma.n_e, "shape")
            else jnp.asarray(float(plasma.n_e), dtype=dtype)
        )
        children = (T_e, n_e, *density_leaves)
        aux = (
            elements,
            float(plasma.T_g) if plasma.T_g is not None else None,
            float(plasma.pressure) if plasma.pressure is not None else None,
        )
        return children, aux

    def _plasma_unflatten(aux: tuple, children: tuple) -> "SingleZoneLTEPlasma":
        elements, T_g, pressure = aux
        T_e = children[0]
        n_e = children[1]
        density_leaves = children[2:]
        # Bypass __init__ — it logs an f-string format on T_e which fails on
        # batched/traced arrays. Reconstruct via object.__new__ + field set.
        instance = object.__new__(SingleZoneLTEPlasma)
        species = dict(zip(elements, density_leaves))
        instance.T_e = T_e
        instance.n_e = n_e
        instance.species = species
        instance.T_g = T_g
        instance.pressure = pressure
        return instance

    def _instrument_flatten(instrument: "InstrumentModel"):
        dtype = jax_default_real_dtype()
        fwhm = (
            instrument.resolution_fwhm_nm
            if hasattr(instrument.resolution_fwhm_nm, "shape")
            else jnp.asarray(float(instrument.resolution_fwhm_nm), dtype=dtype)
        )
        if instrument.response_curve is not None:
            response = jnp.asarray(instrument.response_curve, dtype=dtype)
        else:
            response = jnp.zeros((0, 2), dtype=dtype)
        if instrument.resolving_power is not None:
            R = (
                instrument.resolving_power
                if hasattr(instrument.resolving_power, "shape")
                else jnp.asarray(float(instrument.resolving_power), dtype=dtype)
            )
        else:
            R = jnp.asarray(0.0, dtype=dtype)
        children = (fwhm, response, R)
        aux = (
            instrument.wavelength_calibration,
            instrument.response_curve is not None,
            instrument.resolving_power is not None,
        )
        return children, aux

    def _instrument_unflatten(aux: tuple, children: tuple) -> "InstrumentModel":
        fwhm, response, R = children
        calibration, has_response, has_R = aux
        instance = object.__new__(InstrumentModel)
        instance.resolution_fwhm_nm = fwhm
        instance.response_curve = response if has_response else None
        instance.wavelength_calibration = calibration
        instance.resolving_power = R if has_R else None
        return instance

    jax.tree_util.register_pytree_node(SingleZoneLTEPlasma, _plasma_flatten, _plasma_unflatten)
    jax.tree_util.register_pytree_node(InstrumentModel, _instrument_flatten, _instrument_unflatten)


# Trigger pytree registration lazily so ``from cflibs.core.jax_runtime
# import jit_if_available`` does not pull in ``cflibs.plasma`` /
# ``cflibs.instrument`` at module import time (would create import cycles).
def _ensure_pytrees_registered() -> None:
    """Public entry point — call from ``cflibs/__init__.py`` or test setup
    to wire :class:`SingleZoneLTEPlasma` and :class:`InstrumentModel` as
    pytrees. Idempotent and re-entrant (the registration logic imports
    from ``cflibs.plasma.state`` and ``cflibs.instrument.model``, both of
    which call back into this function on module load).
    """
    if not HAS_JAX:
        return
    if getattr(_ensure_pytrees_registered, "_done", False):
        return
    # Mark done BEFORE registering so the recursive import path
    # cflibs.plasma.state -> cflibs.core.jax_runtime -> cflibs.plasma.state
    # short-circuits instead of double-registering.
    _ensure_pytrees_registered._done = True  # type: ignore[attr-defined]
    _register_cflibs_pytrees()


__all__ = [
    "AtomicSnapshot",
    "DTypeLike",
    "HAS_JAX",
    "JAX_BACKEND",
    "JaxMemoryPolicy",
    "_ensure_pytrees_registered",
    "_refresh_runtime_state",
    "check_jax64bit",
    "jax",
    "jax_active_backend",
    "jax_backend_supports_complex",
    "jax_backend_supports_x64",
    "jax_default_complex_dtype",
    "jax_default_real_dtype",
    "jax_policy",
    "jit_if_available",
    "jnp",
    "set_jax_policy",
    "vmap_if_available",
]
