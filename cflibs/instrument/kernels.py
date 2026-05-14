"""JAX-accelerated instrument-model inner kernels (ADR-0001 T1-1 host/kernel split).

This module holds the pure ``@jit_if_available`` helpers that
:class:`cflibs.instrument.model.InstrumentModelJax` dispatches through.
Splitting them out of ``model.py`` mirrors the pattern established by
:mod:`cflibs.radiation.kernels` and keeps the host file focused on the
``InstrumentModel`` dataclass, config loading, and response-curve I/O.

Functions here take ``jnp.ndarray`` / Python scalars only and never call
``self`` — the host materializes results across the JAX/NumPy boundary
as needed.

* :func:`_sigma_at_wavelength_jax` -- per-wavelength Gaussian sigma,
  branchless so jit compiles either fixed-FWHM or resolving-power mode.
* :func:`_apply_response_jax` -- linear interpolation of the (normalized)
  response curve onto a wavelength grid.
"""

from __future__ import annotations

from cflibs.core.jax_runtime import HAS_JAX, jit_if_available, jnp

if HAS_JAX:

    @jit_if_available
    def _sigma_at_wavelength_jax(
        wavelength_nm: "jnp.ndarray",
        resolving_power: "jnp.ndarray",
        resolution_sigma_nm: "jnp.ndarray",
        use_resolving_power: "jnp.ndarray",
    ) -> "jnp.ndarray":
        """Per-wavelength Gaussian sigma — branchless for JIT compatibility.

        ``use_resolving_power`` is a 0/1 mask (jnp scalar) so the function
        can be jit-compiled regardless of which mode is active.
        """
        sigma_R = wavelength_nm / jnp.maximum(resolving_power, 1e-30) / 2.355
        return jnp.where(use_resolving_power > 0.5, sigma_R, resolution_sigma_nm)

    @jit_if_available
    def _apply_response_jax(
        wavelength: "jnp.ndarray",
        intensity: "jnp.ndarray",
        wl_resp: "jnp.ndarray",
        resp: "jnp.ndarray",
    ) -> "jnp.ndarray":
        """Linear interpolation of the (normalized) response curve onto ``wavelength``.

        Uses ``jnp.interp`` (which becomes a fused linear-interp kernel
        under XLA) to avoid the SciPy interp1d dependency in JAX land.
        """
        resp_norm = resp / jnp.maximum(jnp.max(resp), 1e-30)
        # jnp.interp clamps out-of-range to the boundary — match SciPy's
        # ``fill_value=0.0`` behavior by zeroing those points explicitly.
        in_range = (wavelength >= wl_resp[0]) & (wavelength <= wl_resp[-1])
        interpolated = jnp.interp(wavelength, wl_resp, resp_norm)
        response = jnp.where(in_range, interpolated, 0.0)
        return intensity * response

else:  # pragma: no cover - JAX should be installed in this repo

    def _sigma_at_wavelength_jax(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")

    def _apply_response_jax(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")


__all__ = [
    "_sigma_at_wavelength_jax",
    "_apply_response_jax",
]
