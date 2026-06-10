"""Derive the inputs to the self-absorption corrector in ONE place.

:class:`~cflibs.inversion.physics.self_absorption.SelfAbsorptionCorrector.correct`
needs four physics-derived inputs that are NOT carried directly on a
:class:`~cflibs.inversion.physics.boltzmann.LineObservation`:

* ``lower_level_energies`` — the absorbing-level energy ``E_i`` per line, which
  the curve-of-growth optical-depth estimate Boltzmann-weights. ``E_i`` is
  recovered exactly from energy conservation, ``E_i = E_k - hc/λ`` (a
  ``LineObservation`` carries only the *upper*-level ``E_k`` and the
  wavelength). Defaulting every line to the ground state (0 eV) would make
  every line look like a maximally self-absorbed resonance line.
* ``partition_funcs`` — the stage-I partition function ``U(T)`` per element,
  sourced from the atomic database through the single
  :meth:`AtomicDatabase.partition_function_for` policy (direct-sum preferred,
  always clamped + ``g0``-floored), with the canonical fallback ladder for
  species the factory cannot resolve.
* ``concentrations`` and ``total_number_density_cm3`` — pass-throughs that the
  caller already holds (the previous iteration's number fractions and the
  configurable absorbing column density), carried here so a single object
  splats straight into ``correct()``.

Before this module existed, three call sites derived these inputs
independently and drifted apart:

* the shipped solver (``IterativeCFLIBSSolver``) derived ``E_i`` exactly and
  used the full partition fallback ladder — the *correct* derivation;
* ``scripts/run_experiments.py`` omitted ``lower_level_energies`` entirely, so
  every ``E_i`` collapsed to ~0 eV (a defect), and passed an EMPTY
  ``partition_funcs``;
* ``scripts/run_accuracy_ablation.py`` looked ``E_i`` up via a fuzzy
  ±0.05 nm database query and used a non-canonical ``U(T)`` fallback of 1.0.

This module turns that leaky, duplicated seam into a DEEP module: callers
hand it the plasma state and an atomic database and receive a single
:class:`SelfAbsorptionInputs` ready to correct with. The shipped solver's
derivation helpers were moved here VERBATIM (``lower_level_energy_ev`` and
``evaluate_partition_function``), and the solver now calls back into them, so
the solver's self-absorption path stays byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

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


@dataclass(frozen=True)
class SelfAbsorptionInputs:
    """Physics-derived inputs to :meth:`SelfAbsorptionCorrector.correct`.

    A single object carrying everything ``correct()`` needs that is not
    already on the :class:`LineObservation` list. Splat it with::

        result = corrector.correct(observations, **inputs.as_correct_kwargs())

    Attributes
    ----------
    temperature_K : float
        Excitation temperature used for the Boltzmann weighting of the
        lower-level population in the optical-depth estimate.
    concentrations : Dict[str, float]
        Per-element fractions (number or mass) modulating ``tau``. A flat
        prior is the only unbiased choice when no composition is known yet.
    total_number_density_cm3 : float
        Absorbing heavy-particle column reference (``n_heavy``). The shipped
        solver anchors this to a LIBS-realistic ``1e16`` rather than the STP
        ``n_e`` (~``1e18``), which would saturate every strong major line.
    partition_funcs : Dict[str, float]
        Stage-I partition function ``U(T)`` per element, from
        :func:`evaluate_partition_function`.
    lower_level_energies : Dict[float, float]
        Absorbing-level energy ``E_i`` (eV) keyed by wavelength (nm), from
        :func:`lower_level_energy_ev`.
    """

    temperature_K: float
    concentrations: Dict[str, float]
    total_number_density_cm3: float
    partition_funcs: Dict[str, float]
    lower_level_energies: Dict[float, float]

    def as_correct_kwargs(self) -> Dict[str, object]:
        """Return the keyword arguments to splat into ``correct()``."""
        return {
            "temperature_K": self.temperature_K,
            "concentrations": self.concentrations,
            "total_number_density_cm3": self.total_number_density_cm3,
            "partition_funcs": self.partition_funcs,
            "lower_level_energies": self.lower_level_energies,
        }


def build_self_absorption_inputs(
    observations: Sequence[LineObservation],
    *,
    temperature_K: float,
    concentrations: Dict[str, float],
    total_number_density_cm3: float,
    atomic_db,
    ionization_stage: int = 1,
) -> SelfAbsorptionInputs:
    """Derive every self-absorption input from the plasma state in ONE place.

    Computes the exact lower-level energy ``E_i`` for each observation
    (:func:`lower_level_energy_ev`) and the stage-``ionization_stage``
    partition function ``U(T)`` for each distinct element
    (:func:`evaluate_partition_function`), then packages them with the
    caller-supplied pass-throughs into a :class:`SelfAbsorptionInputs`.

    Parameters
    ----------
    observations : sequence of LineObservation
        Emission lines whose intensities will be corrected. Their
        ``wavelength_nm`` / ``E_k_ev`` give ``E_i``; their ``element`` set
        drives the partition-function lookups.
    temperature_K : float
        Excitation temperature (also used to evaluate ``U(T)``).
    concentrations : Dict[str, float]
        Per-element fractions, passed through unchanged.
    total_number_density_cm3 : float
        Absorbing-column reference, passed through unchanged.
    atomic_db : AtomicDatabase or AtomicDataSource or None
        Atomic-data backend; only used to source partition functions. When
        ``None`` (callers with no DB handle, e.g. the Gaussian-fallback
        experiment harness) partition functions fall back to the canonical
        ladder (:func:`cflibs.plasma.partition.canonical_partition_fallback`:
        exact closed-shell values, then ``g0``, then a warned generic
        constant) — never an empty dict. The same helper backs the
        missing-entry substitution inside
        :meth:`SelfAbsorptionCorrector.correct`, so the DB-less path stays
        numerically equivalent to passing an empty ``partition_funcs``.
    ionization_stage : int, default 1
        Stage whose partition function is evaluated (1 = neutral). The
        self-absorption corrector keys ``partition_funcs`` by element, so a
        single stage is selected; neutral is the LIBS-resonance default.

    Returns
    -------
    SelfAbsorptionInputs
        Ready to splat into ``SelfAbsorptionCorrector.correct``.
    """
    lower_level_energies: Dict[float, float] = {}
    elements: List[str] = []
    seen: set[str] = set()
    for obs in observations:
        lower_level_energies[obs.wavelength_nm] = lower_level_energy_ev(obs)
        if obs.element not in seen:
            seen.add(obs.element)
            elements.append(obs.element)

    partition_funcs: Dict[str, float] = {
        el: evaluate_partition_function(atomic_db, el, ionization_stage, temperature_K)
        for el in elements
    }

    return SelfAbsorptionInputs(
        temperature_K=temperature_K,
        concentrations=concentrations,
        total_number_density_cm3=total_number_density_cm3,
        partition_funcs=partition_funcs,
        lower_level_energies=lower_level_energies,
    )


__all__ = [
    "SelfAbsorptionInputs",
    "build_self_absorption_inputs",
    "evaluate_partition_function",
    "lower_level_energy_ev",
]
