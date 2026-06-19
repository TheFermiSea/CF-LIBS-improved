"""
Unified compositional-closure protocol and adapters.

This module defines a single :class:`ClosureStrategy` :class:`~typing.Protocol`
that wraps the three different mechanisms previously scattered across
``softmax_closure.py`` (JAX, used by :mod:`joint_optimizer` and
:mod:`coarse_to_fine`) and ``closure.py`` (NumPy, used by the iterative
solver).

This implements architecture-review *Candidate 3*, generalizing ADR-0001 T1-3
(pre-resolving closure mode into a closed-over ``closure_fn``).  In practice
the ``ClosureStrategy`` abstraction was adopted only by the two JAX solvers
(:mod:`joint_optimizer` and :mod:`coarse_to_fine`, via :class:`SoftmaxClosure`);
the iterative solver still dispatches closure modes by string through
:class:`~cflibs.inversion.physics.closure.ClosureEquation`.  The :class:`ILRClosure`
and :class:`PWLRClosure` adapters below are available for callers that want a
log-ratio ``ClosureStrategy`` but are not currently wired into any shipped solver.

The three adapters are:

* :class:`SoftmaxClosure` — JAX softmax (log-sum-exp stabilised). Wraps
  :func:`cflibs.inversion.physics.softmax_closure.softmax_closure`.
* :class:`ILRClosure` — Isometric Log-Ratio. Wraps
  :func:`cflibs.inversion.physics.closure.ilr_transform` +
  :func:`cflibs.inversion.physics.closure.ilr_inverse`.
* :class:`PWLRClosure` — Pivot Log-Ratio with adaptive ridge regularization.
  Wraps :func:`cflibs.inversion.physics.closure.optimize_pwlr_coordinates`
  + :func:`cflibs.inversion.physics.closure.plr_inverse`.

The protocol is purely additive — existing module-level functions and
classes remain untouched; the adapters delegate to them.

Convention: ``C_i [dimensionless], sum(C_i) = 1``.

References
----------
DERV-04: Softmax closure derivation with Jacobian and limiting cases.
Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations for
    Compositional Data Analysis." Mathematical Geology 35(3), 279-300.

See Also
--------
docs/architecture/2026-05-26-architecture-review.md — Candidate 3 spec.
docs/adr/specs/T1-3-lax-while-iterative.md — closure-mode pre-resolution.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from cflibs.core.jax_runtime import HAS_JAX
from cflibs.inversion.physics.closure import (
    LOGRATIO_CLIP_FLOOR,
    ilr_inverse,
    ilr_transform,
    optimize_pwlr_coordinates,
    plr_inverse,
)

# ``Array`` is used in the Protocol signature as a *structural* hint that
# accepts both ``jax.numpy.ndarray`` and ``numpy.ndarray``.  We type-alias to
# ``Any`` so the protocol stays usable whether or not JAX is installed.
Array = Any


__all__ = [
    "ClosureStrategy",
    "SoftmaxClosure",
    "ILRClosure",
    "PWLRClosure",
]


@runtime_checkable
class ClosureStrategy(Protocol):
    """Compositional-closure strategy protocol.

    Implementations map unconstrained parameters (``theta`` in softmax;
    or raw relative concentrations in log-ratio space) onto the open
    ``(D-1)``-simplex, enforcing ``sum(C_i) = 1`` and ``C_i > 0``.

    Attributes
    ----------
    name : str
        Stable identifier for the closure, used in logging and result
        metadata (e.g. ``"softmax"``, ``"ilr"``, ``"pwlr"``).
    backend : Literal["jax", "numpy"]
        Numerical backend the adapter executes on.  Solvers can use this
        to decide whether the strategy can be used inside a
        ``lax.while_loop`` body or must be lifted to a host callback.

    Methods
    -------
    apply(params)
        Forward transform: parameters → simplex composition.
    gradient_check(c)
        Diagnostic for trace-element safety.  Returns ``True`` if the
        gradient at composition ``c`` is well-conditioned (i.e. not
        catastrophically vanishing for the smallest component).
    """

    name: str
    backend: Literal["jax", "numpy"]

    def apply(self, params: Array) -> Array:  # pragma: no cover - protocol
        ...

    def gradient_check(self, c: Array) -> bool:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Softmax (JAX) adapter
# ---------------------------------------------------------------------------


class SoftmaxClosure:
    """JAX softmax closure adapter.

    Forward map ``C_i = exp(theta_i - max(theta)) / sum_j exp(theta_j - max(theta))``
    (log-sum-exp stabilised).  The map is exactly the existing
    :func:`cflibs.inversion.physics.softmax_closure.softmax_closure` —
    this adapter delegates to it so output is bit-identical with the
    pre-refactor call site.

    Notes
    -----
    The analytic Jacobian is ``J = diag(C) - C C^T``.  Gradient of
    ``C_i`` w.r.t. ``theta_j`` scales as ``C_i``, so as ``C_i -> 0`` the
    gradient also vanishes.  This is the *vanishing-gradient* failure
    mode that motivates ILR/PWLR — captured by :meth:`gradient_check`.
    """

    name: str = "softmax"
    backend: Literal["jax", "numpy"] = "jax"

    #: Threshold below which a component is considered to suffer from
    #: vanishing gradients in the softmax parameterisation.  Picked to
    #: roughly match double-precision underflow of the squared component
    #: weight (a 1e-6 component gives a 1e-12 effective gradient norm).
    _TRACE_FLOOR: float = 1e-6

    def __init__(self) -> None:
        if not HAS_JAX:
            raise ImportError("SoftmaxClosure requires JAX. Install with: pip install jax jaxlib")
        # Imported lazily so that importing this module on a JAX-less
        # platform does not fail; the ImportError above is the real gate.
        from cflibs.inversion.physics.softmax_closure import softmax_closure

        self._softmax = softmax_closure

    def apply(self, params: Array) -> Array:
        """Forward softmax: ``theta -> C`` on the simplex.

        Parameters
        ----------
        params : Array
            Unconstrained logits ``theta``, shape ``(D,)`` or ``(B, D)``.

        Returns
        -------
        Array
            Simplex composition, same shape as input.
        """
        return self._softmax(params)

    def gradient_check(self, c: Array) -> bool:
        """Return ``True`` iff the smallest component is above the
        trace-element floor.

        The softmax gradient ``dC_i/dtheta_j`` is proportional to
        ``C_i``, so a component below :attr:`_TRACE_FLOOR` is effectively
        gradient-locked.  Callers in trace-element regimes should fall
        back to :class:`ILRClosure` or :class:`PWLRClosure`.
        """
        c_arr = np.asarray(c, dtype=np.float64)
        if c_arr.size == 0:
            return True
        return bool(np.min(c_arr) > self._TRACE_FLOOR)


# ---------------------------------------------------------------------------
# ILR (numpy) adapter
# ---------------------------------------------------------------------------


class ILRClosure:
    """Isometric Log-Ratio closure adapter.

    Maps raw relative concentrations onto the simplex by round-tripping
    through ILR coordinates (Helmert basis).  Mathematically equivalent
    to standard normalization for a single pass, but provides
    well-conditioned gradients down to ``C_i -> 0`` (singularity moved
    to ``-inf`` in ILR space).

    Parameters
    ----------
    eps : float, optional
        Small floor applied to raw inputs before the log transform to
        avoid ``log(0)``.  Defaults to
        :data:`cflibs.inversion.physics.closure.LOGRATIO_CLIP_FLOOR`.
    """

    name: str = "ilr"
    backend: Literal["jax", "numpy"] = "numpy"

    def __init__(self, eps: float = LOGRATIO_CLIP_FLOOR) -> None:
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and > 0")
        self.eps = float(eps)

    def apply(self, params: Array) -> Array:
        """Project raw values onto the simplex via ILR round-trip.

        Parameters
        ----------
        params : Array
            Raw relative concentrations (positive), shape ``(D,)``.

        Returns
        -------
        np.ndarray
            Simplex composition of length ``D``.
        """
        raw = np.asarray(params, dtype=np.float64)
        if raw.ndim != 1:
            raise ValueError(f"ILRClosure.apply expects a 1-D array, got shape {raw.shape}")
        if raw.size < 2:
            # Degenerate single-element case — trivially on the 0-simplex.
            return np.ones_like(raw)
        clipped = np.clip(raw, self.eps, None)
        total = float(np.sum(clipped))
        if total <= 0.0:
            return np.full_like(clipped, 1.0 / clipped.size)
        simplex = clipped / total
        coords = ilr_transform(simplex)
        return ilr_inverse(coords, simplex.shape[-1])

    def gradient_check(self, c: Array) -> bool:
        """Return ``True`` — ILR gradients remain finite everywhere on the
        open simplex.

        The singularity is at ``C_i = 0`` and is mapped to ``-inf`` in
        ILR coordinates, so optimization never reaches it from finite
        coordinates.  As long as ``c`` lies in the open simplex this is
        always safe.
        """
        c_arr = np.asarray(c, dtype=np.float64)
        if c_arr.size == 0:
            return True
        return bool(np.all(c_arr > 0.0))


# ---------------------------------------------------------------------------
# PWLR (numpy) adapter
# ---------------------------------------------------------------------------


class PWLRClosure:
    """Pivot/Pairwise Log-Ratio closure adapter with adaptive ridge.

    Wraps :func:`optimize_pwlr_coordinates` +
    :func:`plr_inverse`.  Behaves like
    :meth:`ClosureEquation.apply_pwlr` but on bare arrays (no element
    dictionaries) — suitable both for the iterative solver's
    intercept→concentration step and as a drop-in replacement for
    Softmax inside JAX-side solvers (via ``host_callback`` if needed).

    Parameters
    ----------
    regularization_strength : float, optional
        Base ridge regularization strength for the PWLR-space inner
        optimization.  Larger values produce smoother but biased
        solutions in trace-element regimes.  Defaults to ``1e-4``,
        matching ``ClosureEquation.apply_pwlr``.
    pivot_index : int, optional
        Optional fixed pivot index.  If omitted the dominant component
        is used as pivot (the default in
        :meth:`ClosureEquation.apply_pwlr`).
    """

    name: str = "pwlr"
    backend: Literal["jax", "numpy"] = "numpy"

    def __init__(
        self,
        regularization_strength: float = 1e-4,
        pivot_index: int | None = None,
    ) -> None:
        if regularization_strength < 0.0 or not np.isfinite(regularization_strength):
            raise ValueError("regularization_strength must be finite and >= 0")
        if pivot_index is not None and pivot_index < 0:
            raise ValueError("pivot_index must be >= 0")
        self.regularization_strength = float(regularization_strength)
        self.pivot_index = pivot_index

    def apply(self, params: Array) -> Array:
        """Project raw values onto the simplex via PWLR optimization.

        Parameters
        ----------
        params : Array
            Raw relative concentrations (positive), shape ``(D,)``.

        Returns
        -------
        np.ndarray
            Simplex composition of length ``D``.
        """
        raw = np.asarray(params, dtype=np.float64)
        if raw.ndim != 1:
            raise ValueError(f"PWLRClosure.apply expects a 1-D array, got shape {raw.shape}")
        if raw.size < 2:
            return np.ones_like(raw)
        clipped = np.clip(raw, LOGRATIO_CLIP_FLOOR, None)
        total = float(np.sum(clipped))
        if total <= 0.0:
            return np.full_like(clipped, 1.0 / clipped.size)
        simplex = clipped / total
        pivot = self.pivot_index if self.pivot_index is not None else int(np.argmax(simplex))
        if pivot >= simplex.size:
            raise ValueError(f"pivot_index={pivot} out of bounds for D={simplex.size}")
        coords = optimize_pwlr_coordinates(
            simplex=simplex,
            pivot_index=pivot,
            regularization_strength=self.regularization_strength,
        )
        return plr_inverse(coords, D=simplex.size, pivot_index=pivot)

    def gradient_check(self, c: Array) -> bool:
        """Return ``True`` — PWLR gradients remain finite everywhere on
        the open simplex (same argument as :meth:`ILRClosure.gradient_check`).
        """
        c_arr = np.asarray(c, dtype=np.float64)
        if c_arr.size == 0:
            return True
        return bool(np.all(c_arr > 0.0))
