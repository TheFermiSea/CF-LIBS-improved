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

    from cflibs.instrument.model import InstrumentModel
    from cflibs.plasma.state import SingleZoneLTEPlasma
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


def _resolve_vmap_axes(in_axes: Any, n_args: int) -> tuple:
    """Normalise ``in_axes`` to a per-positional-argument tuple."""
    if isinstance(in_axes, tuple):
        return in_axes
    return tuple(in_axes if i < n_args else None for i in range(n_args))


def _infer_vmap_batch_size(args: tuple, axes: tuple) -> int | None:
    """Return the batch length from the first non-``None`` axis, or ``None``."""
    for arg, axis in zip(args, axes):
        if axis is None:
            continue
        arr = _np.asarray(arg)
        if arr.ndim == 0:
            raise ValueError("vmap_if_available fallback requires array-like batched arguments")
        return arr.shape[axis]
    return None


def _slice_vmap_args(args: tuple, axes: tuple, i: int) -> list:
    """Slice batched arguments at index ``i`` along their mapped axis."""
    sliced = []
    for arg, axis in zip(args, axes):
        if axis is None:
            sliced.append(arg)
        else:
            sliced.append(_np.take(_np.asarray(arg), i, axis=axis))
    return sliced


def _stack_vmap_outputs(outputs: list, out_axes: Any) -> Any:
    """Stack per-element fallback outputs along ``out_axes``."""
    if isinstance(outputs[0], tuple):
        return tuple(_np.stack(parts, axis=out_axes) for parts in zip(*outputs))
    return _np.stack(outputs, axis=out_axes)


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
        axes = _resolve_vmap_axes(in_axes, len(args))
        n_batch = _infer_vmap_batch_size(args, axes)
        if n_batch is None:
            return func(*args, **kwargs)

        outputs = [func(*_slice_vmap_args(args, axes, i), **kwargs) for i in range(n_batch)]
        return _stack_vmap_outputs(outputs, out_axes)

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


def configure_for_identifiers() -> None:
    """Enable JAX ``float64`` mode for the JAX-backed identifier path.

    Architectural seam for the contract documented in bead
    ``CF-LIBS-improved-jbfg.1``: every identifier's JAX helpers
    explicitly request ``jnp.float64`` (FISTA NNLS, Boltzmann fit,
    ``compute_p_snr_jax``) but without ``jax_enable_x64=True`` JAX
    silently demotes to ``float32`` and FISTA produces ~95% coefficient
    error on column-correlated spectra.

    This used to live as a hidden side effect of
    ``cflibs.benchmark.unified._jax_identifier_flags_for``; arch review
    #2 candidate 2 lifted it to this named function. Call once at
    session start — :class:`cflibs.benchmark.unified.UnifiedBenchmarkRunner`
    does this automatically; ad-hoc scripts constructing
    ``ALIASIdentifier(use_jax_nnls=True)`` directly should call this
    explicitly. Each identifier's ``__init__`` then verifies the contract
    via :func:`check_jax64bit` and fails fast if it was missed.

    Idempotent and a no-op when JAX is not installed.
    """
    if not HAS_JAX:
        return
    jax.config.update("jax_enable_x64", True)


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
    line_stark_d : ndarray, shape (N_lines,)
        Signed Stark **shift** of the line center at ``REF_NE = 1e17 cm^-3``
        (nm); 0.0 when missing. Scales linearly with density:
        ``delta_lambda = line_stark_d * (n_e / REF_NE)`` (see
        :func:`cflibs.radiation.stark.stark_shift` and
        :func:`cflibs.radiation.kernels._per_line_stark_shift`). Applied to
        line centers BEFORE broadening when ``apply_stark`` is on; lines
        without a catalogued shift carry ``0.0`` and are left unmoved.
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

    # Per-line signed Stark shift of the line center at REF_NE = 1e17 cm^-3
    # (nm). Optional / nullable so snapshots built before the Stark-shift
    # rollout (and the many lines lacking a catalogued shift) remain valid;
    # ``None`` is treated as "all-zero" by the forward kernel, i.e. no shift.
    line_stark_d: Any = None

    # --- Stage-III Saha extension (audit 01-F4, bead CF-LIBS-improved-rs7e) ---
    # Per-SPECIES-row arrays carrying the row's ELEMENT's stage-III
    # (doubly-ionized) partition spec, duplicated across that element's
    # stage-I and stage-II rows so the kernel gathers them with the same
    # species-axis indices it already uses for U_I / U_II.  Baked from
    # ``partition_spec_for(element, 3)`` with the canonical fallback ladder
    # (closed-shell values exact, warn-once otherwise) when no spec exists.
    # ``None`` on snapshots built before the rollout — the kernel then
    # degrades to the two-stage balance (S2 = 0).
    partition_coeffs_iii: Any = None  # (N_species, 5)
    partition_t_min_iii: Any = None  # (N_species,)
    partition_t_max_iii: Any = None  # (N_species,)
    partition_g0_iii: Any = None  # (N_species,)

    # Per-species flag (1.0/0.0): the CPU scalar adapter for this species row
    # evaluates the EXACT direct sum over its energy levels
    # (``PartitionFunctionSpec.from_direct_sum``).  When the snapshot also
    # carries the padded level arrays (``include_levels=True``) the forward
    # kernel mirrors that exact IPD-truncated sum for flagged species,
    # restoring <1 % ionization-fraction parity with the CPU solver at
    # n_e = 1e18 cm^-3 where the IPD truncation of U is no longer negligible.
    partition_from_direct_sum: Any = None  # (N_species,)


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
        "line_stark_d",
        "partition_coeffs",
        "ionization_potential_ev",
        "level_g",
        "level_E_ev",
        "level_mask",
        "partition_t_min",
        "partition_t_max",
        "partition_g0",
        "partition_coeffs_iii",
        "partition_t_min_iii",
        "partition_t_max_iii",
        "partition_g0_iii",
        "partition_from_direct_sum",
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


def _as_leaf(value: Any, dtype: Any) -> Any:
    """Pass arrays through unchanged; box Python scalars as ``jnp`` leaves."""
    return value if hasattr(value, "shape") else jnp.asarray(float(value), dtype=dtype)


def _plasma_flatten(plasma: "SingleZoneLTEPlasma"):
    elements = tuple(plasma.species.keys())
    dtype = jax_default_real_dtype()
    # Emit one leaf per species so tree_map(stack, plasma) cleanly batches
    # the species dict instead of collapsing it into a 2-D array.
    density_leaves = tuple(_as_leaf(plasma.species[e], dtype) for e in elements)
    T_e = _as_leaf(plasma.T_e, dtype)
    n_e = _as_leaf(plasma.n_e, dtype)
    children = (T_e, n_e, *density_leaves)
    aux = (
        elements,
        float(plasma.T_g) if plasma.T_g is not None else None,
        float(plasma.pressure) if plasma.pressure is not None else None,
    )
    return children, aux


def _plasma_unflatten(aux: tuple, children: tuple) -> "SingleZoneLTEPlasma":
    from cflibs.plasma.state import SingleZoneLTEPlasma

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
    fwhm = _as_leaf(instrument.resolution_fwhm_nm, dtype)
    if instrument.response_curve is not None:
        response = jnp.asarray(instrument.response_curve, dtype=dtype)
    else:
        response = jnp.zeros((0, 2), dtype=dtype)
    if instrument.resolving_power is not None:
        R = _as_leaf(instrument.resolving_power, dtype)
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
    from cflibs.instrument.model import InstrumentModel

    fwhm, response, R = children
    calibration, has_response, has_R = aux
    instance = object.__new__(InstrumentModel)
    instance.resolution_fwhm_nm = fwhm
    instance.response_curve = response if has_response else None
    instance.wavelength_calibration = calibration
    instance.resolving_power = R if has_R else None
    return instance


def _register_cflibs_pytrees() -> None:
    if not HAS_JAX:
        return

    from cflibs.plasma.state import SingleZoneLTEPlasma
    from cflibs.instrument.model import InstrumentModel

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
    "configure_for_identifiers",
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
