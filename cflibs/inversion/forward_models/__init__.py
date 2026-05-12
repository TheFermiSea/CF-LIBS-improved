"""Named forward-model registry (ADR-0001 T1-6).

This module exposes a string-keyed registry of CF-LIBS forward-model kernels
so downstream callers can select a model by name rather than importing a
class. The canonical kernel for single-zone LTE is the T1-2 unified kernel
:func:`cflibs.radiation.kernels.forward_model`; it is registered under the
``"single_zone_lte"`` key.

Pattern source: petitRADTRANS named radiative-transfer kernels (D-P2) +
exojax named-model registry.

Public API
----------
* :data:`FORWARD_MODELS` -- read-only mapping of ``name -> kernel callable``.
* :func:`get_forward_model` -- safe registry lookup with a helpful ``KeyError``.
* :class:`ForwardModelFn` -- structural Protocol describing the kernel
  signature.

Notes
-----
The Hermann two-region and LTE-with-self-absorption entries are thin partial
wrappers around the canonical kernel:

* ``hermann_two_region`` performs two evaluations of
  :func:`cflibs.radiation.kernels.forward_model` and combines them with the
  Hermann self-absorption form. Two-zone Bayesian inference is provided by
  :class:`cflibs.inversion.solve.bayesian.forward.TwoZoneBayesianForwardModel`;
  the registry entry is the *kernel-level* (snapshot-based) analogue.
* ``lte_with_self_absorption`` is :func:`forward_model` partially applied with
  ``apply_self_absorption=True``.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from cflibs.radiation.kernels import forward_model as _single_zone_lte_kernel


class ForwardModelFn(Protocol):
    """Structural type of every entry in :data:`FORWARD_MODELS`."""

    def __call__(
        self,
        plasma_state: Any,
        atomic_snapshot: Any,
        instrument: Any,
        wavelength_grid: Any,
        **kwargs: Any,
    ) -> Any: ...


def _hermann_two_region(
    plasma_state: Any,
    atomic_snapshot: Any,
    instrument: Any,
    wavelength_grid: Any,
    *,
    plasma_state_shell: Any = None,
    shell_fraction: float = 0.0,
    optical_depth_scale: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Hermann two-region kernel.

    Calls :func:`cflibs.radiation.kernels.forward_model` once for the hot core
    and once for the cooler shell, then combines them with a Hermann-style
    radiative-transfer form:

    .. math::

        I_{\\mathrm{obs}} = I_{\\mathrm{core}} e^{-\\tau_{\\mathrm{shell}}}
        + I_{\\mathrm{shell}} \\frac{1 - e^{-\\tau_{\\mathrm{shell}}}}{\\tau_{\\mathrm{shell}}}

    Parameters
    ----------
    plasma_state : SingleZoneLTEPlasma
        Core plasma state.
    plasma_state_shell : Optional[SingleZoneLTEPlasma]
        Shell plasma state. If ``None`` the kernel falls back to the single-zone
        kernel (the registry entry is then equivalent to ``single_zone_lte``).
    shell_fraction : float
        Geometric fraction of total plasma length occupied by the shell.
    optical_depth_scale : float
        Scale factor applied to the shell optical depth.
    **kwargs : dict
        Forwarded to :func:`cflibs.radiation.kernels.forward_model`.

    Notes
    -----
    The two-zone Bayesian path uses the dedicated, gradient-friendly
    :class:`cflibs.inversion.solve.bayesian.forward.TwoZoneBayesianForwardModel`
    on its legacy ``AtomicDataArrays`` carrier. This registry entry is the
    snapshot-based analogue intended for non-Bayesian callers.
    """
    # Lazy import to avoid pulling JAX at module import time when the kernel
    # is unused.
    from cflibs.core.jax_runtime import jnp

    I_core = _single_zone_lte_kernel(
        plasma_state, atomic_snapshot, instrument, wavelength_grid, **kwargs
    )
    if plasma_state_shell is None:
        # Degenerate two-region: no shell. Mirrors single-zone behaviour.
        return I_core

    I_shell = _single_zone_lte_kernel(
        plasma_state_shell, atomic_snapshot, instrument, wavelength_grid, **kwargs
    )
    # Shell absorption proxy: scale the shell intensity by a frequency-flat
    # optical depth derived from the supplied scale and shell_fraction. The
    # full kappa(lambda) path is available via
    # ``BayesianForwardModel.TwoZone*``; here we use a kernel-level
    # approximation consistent with the registry's snapshot contract.
    tau = jnp.asarray(optical_depth_scale * shell_fraction)
    tau_safe = jnp.maximum(tau, 1e-30)
    exp_neg_tau = jnp.exp(-tau_safe)
    source_term = (1.0 - exp_neg_tau) / tau_safe
    return I_core * exp_neg_tau + I_shell * source_term


def _lte_with_self_absorption(
    plasma_state: Any,
    atomic_snapshot: Any,
    instrument: Any,
    wavelength_grid: Any,
    **kwargs: Any,
) -> Any:
    """:func:`forward_model` with ``apply_self_absorption=True`` injected.

    Equivalent to::

        forward_model(plasma_state, atomic_snapshot, instrument, wavelength_grid,
                      apply_self_absorption=True, **kwargs)
    """
    kwargs.setdefault("apply_self_absorption", True)
    return _single_zone_lte_kernel(
        plasma_state, atomic_snapshot, instrument, wavelength_grid, **kwargs
    )


FORWARD_MODELS: Mapping[str, ForwardModelFn] = {
    "single_zone_lte": _single_zone_lte_kernel,
    "hermann_two_region": _hermann_two_region,
    "lte_with_self_absorption": _lte_with_self_absorption,
}


def get_forward_model(name: str) -> ForwardModelFn:
    """Look up a forward-model kernel by name.

    Parameters
    ----------
    name : str
        Registry key. One of :data:`FORWARD_MODELS`.

    Returns
    -------
    ForwardModelFn
        The kernel callable.

    Raises
    ------
    KeyError
        If ``name`` is not a registered key. The error message lists all
        available keys in sorted order.
    """
    if name not in FORWARD_MODELS:
        raise KeyError(f"Unknown forward model {name!r}; available: {sorted(FORWARD_MODELS)}")
    return FORWARD_MODELS[name]


__all__ = ["FORWARD_MODELS", "ForwardModelFn", "get_forward_model"]
