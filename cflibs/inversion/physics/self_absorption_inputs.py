"""Shared physics-input derivations for self-absorption analysis.

Two canonical helpers used by the solver and the observable-gated
self-absorption corrector:

* :func:`lower_level_energy_ev` — the absorbing-level energy ``E_i`` per
  line, recovered exactly from energy conservation ``E_i = E_k - hc/λ``
  (a :class:`~cflibs.inversion.physics.boltzmann.LineObservation` carries
  only the *upper*-level ``E_k`` and the wavelength). Defaulting every line
  to the ground state (0 eV) would make every line look like a maximally
  self-absorbed resonance line.
* :func:`evaluate_partition_function` — the partition function ``U(T)``
  per species, sourced from the atomic database through the single
  :meth:`AtomicDatabase.partition_function_for` policy (direct-sum
  preferred, always clamped + ``g0``-floored), with the canonical fallback
  ladder for species the factory cannot resolve.

Historical note: this module also packaged the plasma-state inputs for the
composition-fed ``SelfAbsorptionCorrector.correct()``
(``SelfAbsorptionInputs`` / ``build_self_absorption_inputs``). That
correction path was a positive feedback loop on the recovered composition
(audit 2026-06-09, 02-inversion-solver.md F4) and was deleted in bead
CF-LIBS-improved-0jvr together with its input builder; the production
correction is now
:class:`cflibs.inversion.physics.self_absorption_observable.ObservableSelfAbsorptionCorrector`.
"""

from __future__ import annotations


from cflibs.core.constants import C_LIGHT, H_PLANCK_EV
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.plasma.partition import canonical_partition_fallback


def lower_level_energy_ev(obs: LineObservation) -> float:
    """Lower-level energy ``E_i`` of a line, in eV.

    ``LineObservation`` carries only the upper-level energy ``E_k`` and the
    wavelength, but the curve-of-growth optical-depth estimate needs the
    *lower*-level energy (the absorbing level). Energy conservation for the
    transition gives ``E_i = E_k - hc/lambda`` exactly, so we recover it
    rather than defaulting every line to the ground state (which would make
    every line look like a maximally self-absorbed resonance line).

    Clamped to be non-negative to absorb small wavelength/energy rounding in
    the atomic data.
    """
    photon_ev = (H_PLANCK_EV * C_LIGHT) / (obs.wavelength_nm * 1e-9)
    return max(0.0, obs.E_k_ev - photon_ev)


def evaluate_partition_function(
    atomic_db, element: str, ionization_stage: int, T_K: float
) -> float:
    """Evaluate a partition function through the single provider factory.

    Routes U(T) through :meth:`AtomicDatabase.partition_function_for` — THE
    single source of the partition-function policy (direct-sum preferred,
    always clamped + ``g0``-floored).  For species with energy levels the
    CPU scalar provider sums the levels directly, so this path stays
    bit-for-bit identical to the historical ``evaluate_direct`` call it
    replaces; for level-less species it applies the guarded stored
    polynomial.  The hardcoded estimates remain only for species the
    factory cannot resolve at all (no levels, no stored row).

    ``partition_function_for`` is a convenience method on the concrete
    :class:`AtomicDatabase`, NOT part of the :class:`AtomicDataSource` ABC.
    Pluggable backends (the documented Key Abstraction) need only satisfy
    the ABC, so we ``hasattr``-guard the factory call and fall back to the
    ABC-level accessors (``get_energy_levels`` direct sum, then the stored
    polynomial) — the same fallback ladder this method used before the
    provider unification, and mirroring the guard in
    :meth:`SahaBoltzmannSolver.calculate_partition_function`.

    When ``atomic_db`` is ``None`` (callers with no atomic-data handle) the
    function short-circuits to the canonical fallback ladder
    (:func:`cflibs.plasma.partition.canonical_partition_fallback`: exact
    closed-shell values → ``g0`` → warned generic constant); the shipped
    solver always passes a real database, so this guard never fires on the
    solver path and the rest of this function is the solver's verbatim
    fallback-ladder logic.
    """
    if atomic_db is None:
        return canonical_partition_fallback(element, ionization_stage)

    if hasattr(atomic_db, "partition_function_for"):
        provider = atomic_db.partition_function_for(element, ionization_stage)
        if provider is not None:
            return float(provider.at(T_K))
    else:
        # ABC-only backend: reproduce the pre-unification fallback ladder
        # (direct sum over energy levels, then the stored polynomial).
        from cflibs.plasma.partition import (
            PartitionFunctionEvaluator,
            get_levels_for_species,
        )

        levels = get_levels_for_species(atomic_db, element, ionization_stage)
        if levels is not None:
            g_arr, E_arr, ip_ev = levels
            return PartitionFunctionEvaluator.evaluate_direct(T_K, g_arr, E_arr, ip_ev)

        pf = atomic_db.get_partition_coefficients(element, ionization_stage)
        if pf:
            return PartitionFunctionEvaluator.evaluate(T_K, pf.coefficients)

    return canonical_partition_fallback(element, ionization_stage, atomic_db)


__all__ = [
    "evaluate_partition_function",
    "lower_level_energy_ev",
]
