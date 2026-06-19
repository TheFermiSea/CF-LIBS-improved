"""Host-side helpers for radiation kernels (ADR-0001 T1-4 + T1-5).

This module hosts CPU-side bookkeeping that supports the jit-compiled
forward kernels:

* :func:`auto_nstitch` — auto-select the number of wavelength chunks for
  :func:`cflibs.radiation.kernels.forward_model_chunked` given the line
  count, wavelength grid size, and available device memory.
* :func:`build_chunk_metadata` — convenience driver that combines the
  device-memory query, ``auto_nstitch`` calculation, and the
  ``_split_wavelength_grid`` returned tuple into a single ready-to-pass
  payload for the chunked forward kernel.
* :func:`_split_wavelength_grid` — split a wavelength grid into
  ``nstitch`` overlapping chunks with their per-chunk line masks.

These helpers are **never** jit-traced; the spec (§3, §5) deliberately
keeps the chunk-metadata construction on the host so the device-side
``lax.scan`` body sees homogeneous shapes.

T1-4 added :func:`cflibs.radiation.ldm.broaden_lines_ldm` as a host-side
driver; T1-5 follows the same naming convention here.

.. note::
   Status: parity-tested but not wired into any shipped pipeline. No
   production module imports this module, and the chunked/overlap-and-add
   forward path (``kernels.forward_model_chunked`` with ``nstitch>1``,
   ``auto_nstitch``, ``build_chunk_metadata``, ``ChunkPlan``) has no
   production caller — ``SpectrumModel``, ``manifold/batch_forward``, and the
   Bayesian forward model all use the single-shot ``forward_model``. The only
   consumers of the chunked subsystem are ``tests/radiation/test_chunked_scan.py``
   and ``tests/radiation/test_memory_bench.py``. The kernels are kept as a
   tested-but-unwired capability for large-grid memory-budgeted scans.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("radiation.host")

if TYPE_CHECKING:  # pragma: no cover - import-only
    from cflibs.core.jax_runtime import JaxMemoryPolicy

__all__ = [
    "ChunkPlan",
    "auto_nstitch",
    "available_device_bytes",
    "build_chunk_metadata",
    "build_chunk_plan",
]


@dataclass(frozen=True)
class ChunkPlan:
    """Frozen plan describing the chunked-forward kernel inputs (a2m2).

    Collapses the five always-paired keyword arguments of
    :func:`cflibs.radiation.kernels.forward_model_chunked` (``nstitch``,
    ``overlap``, ``chunk_wavelength_grids``, ``line_masks``,
    ``output_length``) into a single value-typed payload. Mirrors the
    :class:`cflibs.core.jax_runtime.JaxMemoryPolicy` pattern: ``frozen=True``
    so the plan is hashable / safe to keep alive across compilations, and
    every field is a primitive int or numpy/JAX array.

    Fields
    ------
    nstitch : int
        Number of wavelength chunks.
    overlap : int
        Per-side wing padding in samples.
    chunk_wavelength_grids : array, shape (nstitch, div_length + 2·overlap)
        Padded per-chunk wavelength grids (numpy or jax array).
    line_masks : array, shape (nstitch, N_lines)
        Per-chunk line activation masks.
    output_length : int
        Original wavelength-grid length to trim to after overlap-and-add.

    Notes
    -----
    The dataclass is intentionally NOT registered as a jax pytree: callers
    pass the underlying arrays into the kernel which already accepts
    ``Array``-typed inputs, and the integer fields are static. Pytree
    registration would force a tree-flatten on every jit invocation
    without a corresponding speedup.
    """

    nstitch: int
    overlap: int
    chunk_wavelength_grids: Any
    line_masks: Any
    output_length: int

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ChunkPlan":
        """Construct a :class:`ChunkPlan` from the legacy
        :func:`build_chunk_metadata` dict.

        Convenience helper for callers transitioning from the dict-based
        API to the typed plan. The legacy dict uses ``chunks`` as the key
        for the wavelength grids; we accept both ``chunks`` and
        ``chunk_wavelength_grids`` for compatibility.
        """
        if "chunk_wavelength_grids" in metadata:
            grids = metadata["chunk_wavelength_grids"]
        else:
            grids = metadata["chunks"]
        return cls(
            nstitch=int(metadata["nstitch"]),
            overlap=int(metadata["overlap"]),
            chunk_wavelength_grids=grids,
            line_masks=metadata["line_masks"],
            output_length=int(metadata["output_length"]),
        )


# Conservative fallback when neither psutil nor jax device-stats are reachable.
_FALLBACK_AVAILABLE_BYTES = 4 * 1024**3


def _jax_free_device_bytes() -> int | None:
    """Free bytes from ``jax.devices()[0].memory_stats()`` (CUDA / TPU).

    Returns ``None`` on CPU / Metal backends (their ``memory_stats`` is not
    meaningful), when JAX is unavailable, when no device exposes
    ``memory_stats``, or when ``bytes_limit`` is missing — the caller then
    falls through to the next fallback source.
    """
    try:
        import jax

        devices = jax.devices()
        if not devices:
            return None
        device = devices[0]
        platform = getattr(device, "platform", "").lower()
        if platform in ("cpu", "metal"):
            return None
        stats_fn = getattr(device, "memory_stats", None)
        if not callable(stats_fn):
            return None
        stats = stats_fn() or {}
        bytes_limit = stats.get("bytes_limit")
        bytes_in_use = stats.get("bytes_in_use", 0)
        if bytes_limit:
            # Report actual free bytes; do NOT floor at the 4 GiB fallback —
            # a CUDA card with 2 GiB free genuinely has 2 GiB free, not 4 GiB.
            return max(int(bytes_limit) - int(bytes_in_use), 0)
    except Exception:  # noqa: BLE001
        return None
    return None


def _psutil_available_bytes() -> int | None:
    """25% of ``psutil.virtual_memory().available`` — CPU host-RAM fallback.

    Returns ``None`` when psutil is unavailable so the caller can fall back
    to the hard 4 GiB default.
    """
    try:
        import psutil

        return int(0.25 * psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


@functools.lru_cache(maxsize=1)
def available_device_bytes() -> int:
    """Return a conservative estimate of available memory (bytes).

    Order of attempts:

    1. ``jax.devices()[0].memory_stats()['bytes_limit']`` minus
       ``bytes_in_use`` — works on CUDA / TPU. Skipped on CPU and Metal
       because their backends do not implement ``memory_stats`` meaningfully.
    2. ``psutil.virtual_memory().available`` × 0.25 — CPU host-RAM fallback.
    3. Hard fallback to 4 GiB when no source is reachable.
    """
    jax_bytes = _jax_free_device_bytes()
    if jax_bytes is not None:
        return jax_bytes

    psutil_bytes = _psutil_available_bytes()
    if psutil_bytes is not None:
        return psutil_bytes

    return _FALLBACK_AVAILABLE_BYTES


def auto_nstitch(
    n_lines: int,
    n_lambda: int,
    available_bytes: int,
    *,
    dtype_bytes: int = 4,
    safety_factor: float = 0.5,
) -> int:
    """Select ``nstitch`` so the working ``(N_lines, N_λ)`` block fits.

    Spec §5: ``needed = N_lines · N_λ · dtype_bytes``; pick the smallest
    ``nstitch`` such that ``needed / nstitch ≤ safety_factor · available``.

    Parameters
    ----------
    n_lines : int
        Number of catalog lines after pre-filtering (rows of the per-line
        ``(N_λ, N_lines)`` profile matrix).
    n_lambda : int
        Wavelength grid length.
    available_bytes : int
        Device memory budget in bytes. Use :func:`available_device_bytes`
        for an auto-detected value, or pass an explicit number from
        ``JaxMemoryPolicy`` if the caller has a tighter envelope.
    dtype_bytes : int, default 4
        Per-element size of the temporary profile matrix. 4 for fp32
        (jax-metal / jax-cpu in the default ``allow_32bit=True`` policy)
        and 8 for fp64.
    safety_factor : float, default 0.5
        Fraction of ``available_bytes`` we are willing to spend on the
        single largest transient.

    Returns
    -------
    nstitch : int
        ``max(1, ceil(needed / (safety_factor · available_bytes)))``.

    Notes
    -----
    The returned value is monotone in ``n_lines · n_lambda`` and inversely
    monotone in ``available_bytes``; this is verified by
    ``tests/radiation/test_chunked_scan.py::test_auto_nstitch_logic``.
    """
    if n_lines <= 0 or n_lambda <= 0:
        return 1
    if available_bytes <= 0:
        # Defensive: a non-positive budget should not cause divide-by-zero.
        return max(1, n_lines)
    needed = n_lines * n_lambda * dtype_bytes
    budget = safety_factor * available_bytes
    if budget <= 0:
        return max(1, n_lines)
    return max(1, int(math.ceil(needed / budget)))


def _split_wavelength_grid(
    wavelength_grid: np.ndarray,
    line_wavelengths_nm: np.ndarray,
    *,
    nstitch: int,
    overlap: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Split a wavelength grid into ``nstitch`` overlapping chunks.

    Spec §3, §10: each chunk is ``div_length + 2·overlap`` samples wide,
    with ``div_length = ceil(N_λ / nstitch)`` and pad-and-mask on the last
    chunk when ``N_λ % nstitch ≠ 0``. The line mask is the host-side
    "lines within ``overlap·Δλ`` of this chunk" pre-filter.

    Parameters
    ----------
    wavelength_grid : ndarray, shape (N_λ,)
        Uniform wavelength grid in nm (used here for its dtype and Δλ).
    line_wavelengths_nm : ndarray, shape (N_lines,)
        Per-line center wavelengths from
        :attr:`AtomicSnapshot.line_wavelengths_nm`.
    nstitch : int
        Number of wavelength chunks. Must be ``>= 1``.
    overlap : int
        Per-side overlap padding in samples. ``overlap >= 0``.

    Returns
    -------
    chunks : ndarray, shape (nstitch, div_length + 2·overlap)
        Padded wavelength chunks. Tail samples beyond the original grid
        are filled with the last in-bound wavelength (the corresponding
        ``valid_lengths`` entry encodes the unpadded prefix length).
    line_masks : ndarray of bool, shape (nstitch, N_lines)
        True for every (chunk, line) pair where the line center sits
        within ``overlap`` samples of the chunk's interior.
    div_length : int
        Per-chunk inner length (``ceil(N_λ / nstitch)``).
    output_length : int
        Original ``N_λ`` — what the post-OLA spectrum should be trimmed
        to.

    Raises
    ------
    ValueError
        If ``nstitch < 1``, ``overlap < 0``, or the grid is empty / not
        uniform enough to read a single ``Δλ``.
    """
    if nstitch < 1:
        raise ValueError(f"nstitch must be >= 1; got {nstitch}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0; got {overlap}")

    wl = np.asarray(wavelength_grid).reshape(-1)
    n_lambda = wl.shape[0]
    if n_lambda < 2:
        raise ValueError(
            "wavelength_grid must have at least two samples to infer Δλ; " f"got {n_lambda}"
        )
    dlam = float(wl[1] - wl[0])

    div_length = int(math.ceil(n_lambda / nstitch))
    chunk_length = div_length + 2 * overlap

    line_wl = np.asarray(line_wavelengths_nm, dtype=np.float64).reshape(-1)
    n_lines = line_wl.shape[0]

    # Assign every line to exactly one chunk so overlap-and-add does not
    # double-count it. Lines whose wings spill into the neighbour's overlap
    # region are recombined naturally by OLA. Assigned by interior bin
    # index ``floor(j_line / div_length)``, clamped to ``[0, nstitch-1]``.
    if n_lines:
        # Index of the nearest wavelength sample on the parent grid.
        j_line = np.clip(
            np.round((line_wl - float(wl[0])) / dlam).astype(np.int64),
            0,
            n_lambda - 1,
        )
        owner_chunk = np.clip(j_line // div_length, 0, nstitch - 1)
    else:
        owner_chunk = np.zeros(0, dtype=np.int64)

    # Uniform-grid extrapolation: wl[k] = wl[0] + k·Δλ holds for ALL integer
    # k including k < 0 and k ≥ N_λ. So the entire padded grid collapses to
    # a single affine expression — no branching needed.
    c_idx = np.arange(nstitch, dtype=np.int64)[:, None]
    j_idx = np.arange(chunk_length, dtype=np.int64)[None, :]
    src = c_idx * div_length - overlap + j_idx  # (nstitch, chunk_length)
    chunks = (wl[0] + src.astype(wl.dtype) * dlam).astype(wl.dtype)

    if n_lines:
        line_masks = owner_chunk[None, :] == c_idx  # (nstitch, n_lines)
    else:
        line_masks = np.zeros((nstitch, 0), dtype=bool)

    return chunks, line_masks, div_length, n_lambda


def build_chunk_metadata(
    wavelength_grid: np.ndarray,
    line_wavelengths_nm: np.ndarray,
    *,
    nstitch: int | None = None,
    overlap_factor: float = 4.0,
    max_sigma_nm: float | None = None,
    memory_policy: "JaxMemoryPolicy | None" = None,
    available_bytes: int | None = None,
) -> dict:
    """Build a ready-to-use ``(chunks, masks, overlap, ...)`` payload.

    Convenience wrapper used by the SpectrumModel / manifold callers that
    do not want to compose :func:`available_device_bytes`,
    :func:`auto_nstitch`, and :func:`_split_wavelength_grid` themselves.

    Parameters
    ----------
    wavelength_grid : ndarray, shape (N_λ,)
        Uniform wavelength grid in nm.
    line_wavelengths_nm : ndarray, shape (N_lines,)
        Catalog line centres.
    nstitch : int, optional
        Explicit chunk count. ``None`` (default) triggers
        :func:`auto_nstitch` against ``available_bytes`` (or
        :func:`available_device_bytes` when that is also ``None``). The
        ``memory_policy.nstitch > 1`` override always wins.
    overlap_factor : float, default 4.0
        Multiplier applied to ``max_sigma_nm / Δλ`` to size the per-side
        overlap. Spec §5 default; ``test_overlap_factor`` is the canary.
    max_sigma_nm : float, optional
        Largest expected Gaussian σ in nm; used together with the
        wavelength step to size ``overlap``. ``None`` → ``overlap=0``
        (only safe for ``nstitch=1`` dispatch).
    memory_policy : JaxMemoryPolicy, optional
        When provided, ``policy.nstitch`` (if > 1) and
        ``policy.overlap_factor`` override the auto-selected values.
    available_bytes : int, optional
        Override for :func:`available_device_bytes`.

    Returns
    -------
    payload : dict
        Keys: ``nstitch``, ``overlap``, ``chunks``, ``line_masks``,
        ``div_length``, ``output_length``. Wavelength chunks are returned
        as a ``numpy`` array so the caller can ``jnp.asarray`` once at
        the device boundary.
    """
    wl = np.asarray(wavelength_grid).reshape(-1)
    line_wl = np.asarray(line_wavelengths_nm, dtype=np.float64).reshape(-1)

    if memory_policy is not None and getattr(memory_policy, "nstitch", 1) > 1:
        chosen_nstitch = int(memory_policy.nstitch)
    elif nstitch is not None:
        chosen_nstitch = int(nstitch)
    else:
        budget = available_bytes if available_bytes is not None else available_device_bytes()
        dtype_bytes = 8 if wl.dtype == np.float64 else 4
        chosen_nstitch = auto_nstitch(
            n_lines=line_wl.shape[0],
            n_lambda=wl.shape[0],
            available_bytes=budget,
            dtype_bytes=dtype_bytes,
        )

    if memory_policy is not None:
        effective_overlap_factor = float(getattr(memory_policy, "overlap_factor", overlap_factor))
    else:
        effective_overlap_factor = float(overlap_factor)

    if max_sigma_nm is not None and wl.shape[0] >= 2:
        dlam = float(wl[1] - wl[0])
        overlap = int(math.ceil(effective_overlap_factor * max_sigma_nm / max(dlam, 1e-30)))
    else:
        overlap = 0

    chunks, line_masks, div_length, output_length = _split_wavelength_grid(
        wl, line_wl, nstitch=chosen_nstitch, overlap=overlap
    )
    return {
        "nstitch": chosen_nstitch,
        "overlap": overlap,
        "chunks": chunks,
        "line_masks": line_masks,
        "div_length": div_length,
        "output_length": output_length,
    }


def build_chunk_plan(
    wavelength_grid: np.ndarray,
    line_wavelengths_nm: np.ndarray,
    *,
    nstitch: int | None = None,
    overlap_factor: float = 4.0,
    max_sigma_nm: float | None = None,
    memory_policy: "JaxMemoryPolicy | None" = None,
    available_bytes: int | None = None,
) -> ChunkPlan:
    """Same as :func:`build_chunk_metadata` but returns a :class:`ChunkPlan`.

    Convenience for the typed-API path (a2m2). Internally delegates to
    :func:`build_chunk_metadata` and wraps the returned dict via
    :meth:`ChunkPlan.from_metadata`. The dict form is retained for
    back-compat with existing callers; new callers should prefer this
    typed helper.
    """
    metadata = build_chunk_metadata(
        wavelength_grid,
        line_wavelengths_nm,
        nstitch=nstitch,
        overlap_factor=overlap_factor,
        max_sigma_nm=max_sigma_nm,
        memory_policy=memory_policy,
        available_bytes=available_bytes,
    )
    return ChunkPlan.from_metadata(metadata)
