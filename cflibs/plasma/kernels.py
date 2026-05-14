"""JAX-accelerated Saha-Boltzmann inner kernels (ADR-0001 T1-1 host/kernel split).

This module holds the pure ``@jit_if_available`` helpers that the
:class:`cflibs.plasma.saha_boltzmann.SahaBoltzmannSolverJax` host class
dispatches through. Splitting them out of ``saha_boltzmann.py`` mirrors the
pattern established by :mod:`cflibs.radiation.kernels` and keeps the host
file focused on validation, atomic-data lookup, and mode dispatch.

Functions here take ``jnp.ndarray`` / Python scalars only, never call
``self``, and return ``jnp`` arrays — the host materializes them to
Python floats at the JAX/NumPy boundary.

When JAX is not available the symbols are kept as stubs that raise
``ImportError`` if called. The host code is responsible for never
invoking them on the NumPy path (it has its own scalar fallbacks).

The trio mirrors the closed-form expressions in
:meth:`SahaBoltzmannSolver.solve_ionization_balance` and
:meth:`SahaBoltzmannSolver.solve_level_population`:

* :func:`_partition_sum_jax` -- ``U(T) = Σ gᵢ exp(-Eᵢ / T_eV)`` capped at
  the IPD-lowered ionization potential.
* :func:`_saha_balance_kernel` -- closed-form 3-stage Saha ratios.
* :func:`_boltzmann_populations_kernel` -- per-level Boltzmann densities
  masked by the IPD cutoff.
"""

from __future__ import annotations

from cflibs.core.constants import SAHA_CONST_CM3
from cflibs.core.jax_runtime import HAS_JAX, jit_if_available, jnp

if HAS_JAX:

    @jit_if_available
    def _partition_sum_jax(
        T_e_eV: "jnp.ndarray",
        g_arr: "jnp.ndarray",
        E_arr: "jnp.ndarray",
        max_energy_ev: "jnp.ndarray",
    ) -> "jnp.ndarray":
        """U(T) = Σ gᵢ exp(-Eᵢ / T_eV) over levels with Eᵢ < max_energy_ev.

        All inputs are jnp arrays / scalars to keep the kernel jit-friendly
        with a fixed computation graph. The mask is applied by zeroing out
        contributions above ``max_energy_ev`` rather than boolean indexing.
        """
        kT = jnp.maximum(T_e_eV, 1e-12)
        boltzmann = g_arr * jnp.exp(-E_arr / kT)
        contrib = jnp.where(E_arr < max_energy_ev, boltzmann, 0.0)
        return jnp.maximum(jnp.sum(contrib), 1.0)

    @jit_if_available
    def _saha_balance_kernel(
        T_e_eV: "jnp.ndarray",
        n_e_cm3: "jnp.ndarray",
        eff_ip_I: "jnp.ndarray",
        eff_ip_II: "jnp.ndarray",
        U_I: "jnp.ndarray",
        U_II: "jnp.ndarray",
        U_III: "jnp.ndarray",
        has_II: "jnp.ndarray",
        total_density_cm3: "jnp.ndarray",
    ) -> "jnp.ndarray":
        """Solve the 3-stage Saha balance, returning ``[n_I, n_II, n_III]``.

        Mirrors the closed-form expression in ``solve_ionization_balance``::

            S1 = (SAHA / n_e) * T^1.5 * (U_II/U_I) * exp(-IP_I / T_eV)
            S2 = (SAHA / n_e) * T^1.5 * (U_III/U_II) * exp(-IP_II / T_eV)
            n_I = n_total / (1 + S1 + S1*S2)
            n_II = S1 * n_I
            n_III = S2 * n_II
        """
        T15 = T_e_eV**1.5
        prefactor = SAHA_CONST_CM3 / jnp.maximum(n_e_cm3, 1e-30) * T15

        S1 = prefactor * (U_II / jnp.maximum(U_I, 1e-30)) * jnp.exp(-eff_ip_I / T_e_eV)
        S2_raw = prefactor * (U_III / jnp.maximum(U_II, 1e-30)) * jnp.exp(-eff_ip_II / T_e_eV)
        # Only include S2 when stage II ionization potential is known.
        S2 = jnp.where(has_II > 0.5, S2_raw, 0.0)

        denom = 1.0 + S1 + S1 * S2
        n_I = total_density_cm3 / denom
        n_II = S1 * n_I
        n_III = S2 * n_II
        return jnp.stack([n_I, n_II, n_III])

    @jit_if_available
    def _boltzmann_populations_kernel(
        T_e_eV: "jnp.ndarray",
        stage_density_cm3: "jnp.ndarray",
        g_arr: "jnp.ndarray",
        E_arr: "jnp.ndarray",
        max_energy_ev: "jnp.ndarray",
        U: "jnp.ndarray",
    ) -> "jnp.ndarray":
        """nᵢ = n_stage * (gᵢ / U) * exp(-Eᵢ / T_eV) for Eᵢ ≤ max_energy_ev.

        Returns an array shaped like ``g_arr``; entries above the IPD cutoff
        are zeroed so callers can mask them out in NumPy land.
        """
        boltzmann = jnp.exp(-E_arr / jnp.maximum(T_e_eV, 1e-12))
        pop = stage_density_cm3 * (g_arr / jnp.maximum(U, 1e-30)) * boltzmann
        return jnp.where(E_arr <= max_energy_ev, pop, 0.0)

else:  # pragma: no cover - JAX should be installed in this repo

    def _partition_sum_jax(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")

    def _saha_balance_kernel(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")

    def _boltzmann_populations_kernel(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")


__all__ = [
    "_partition_sum_jax",
    "_saha_balance_kernel",
    "_boltzmann_populations_kernel",
]
