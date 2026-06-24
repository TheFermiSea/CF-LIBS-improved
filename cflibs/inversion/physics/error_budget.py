"""Error-budget thresholds derived from the verified ``cflibs-formal`` spec.

The empirical line-selection magic numbers — minimum energy spread and minimum SNR — are
*derived from a target temperature accuracy* via a proven error-propagation chain, rather than
tuned. This module mirrors the machine-checked Lean theorems in
``cflibs-formal/CflibsFormal/ErrorBudget.lean`` (axiom-clean, no ``sorry``); see
``tests/data/cflibs_formal_oracle/`` and ``tests/inversion/physics/test_oracle_regression.py``,
which validate the dimensionless core below against the verified ``error-budget`` fixture.

Verified chain (``|Delta beta|`` is the inverse-temperature / Boltzmann-plot slope error):

* slope sensitivity (``olsSlope_stable_l2``):
  ``|Delta beta| <= eps * sqrt(n) / sqrt(ss_e)``  with ``ss_e = sum_k (E_k - Ebar)**2``;
* energy-spread threshold (``requiredEnergySpread_sufficient``):
  ``ss_e >= eps**2 * n / tau_beta**2`` guarantees ``|Delta beta| <= tau_beta``;
* SNR threshold (``maxPerLineError_sufficient``):
  ``eps <= tau_beta * sqrt(ss_e / n)`` guarantees ``|Delta beta| <= tau_beta``;
* exact temperature relative error (``temp_rel_error_eq``):
  ``|Delta T| / T = kB * T * |Delta beta|``.

The deterministic worst case shows energy spread and SNR dominate; the line-count benefit is
statistical (Gauss-Markov), so only the noise-gain kernel ``sum w_k**2 = 1/ss_e``
(``olsSlope_noise_gain``) is proven — see ``required_min_lines_stat`` below and the Lean module
docstring.

Physics-only: this module imports nothing outside ``math`` and ``cflibs.core``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cflibs.core.constants import KB_EV  # eV/K

__all__ = [
    "slope_error_bound",
    "required_energy_spread",
    "max_per_line_error",
    "required_min_lines_stat",
    "slope_target_from_temp_rel",
    "density_budget_from_composition",
    "temp_rel_error_bound",
    "required_energy_span_ev",
    "min_snr_for_target",
    "LineSelectionThresholds",
    "derive_line_selection_thresholds",
]


# --------------------------------------------------------------------------------------------
# Dimensionless core — exact mirror of CflibsFormal/ErrorBudget.lean (validated vs the fixture)
# --------------------------------------------------------------------------------------------


def slope_error_bound(per_line_error: float, n_lines: float, energy_ss: float) -> float:
    """Worst-case inverse-temperature (slope) error ``|Delta beta|`` (l2 bound).

    Mirror of ``ErrorBudget.olsSlope_stable_l2``:
    ``per_line_error * sqrt(n_lines) / sqrt(energy_ss)`` where
    ``energy_ss = sum_k (E_k - Ebar)**2``.
    """
    return per_line_error * math.sqrt(n_lines) / math.sqrt(energy_ss)


def required_energy_spread(slope_target: float, per_line_error: float, n_lines: float) -> float:
    """Energy spread ``ss_e`` that *guarantees* a target slope error ``slope_target``.

    Mirror of ``ErrorBudget.requiredEnergySpread_sufficient``:
    ``per_line_error**2 * n_lines / slope_target**2`` (the derived ``min_energy_spread``).
    """
    return per_line_error**2 * n_lines / slope_target**2


def max_per_line_error(slope_target: float, n_lines: float, energy_ss: float) -> float:
    """Largest per-line ordinate error tolerable for a target slope error (the derived min SNR).

    Mirror of ``ErrorBudget.maxPerLineError_sufficient``:
    ``slope_target * sqrt(energy_ss / n_lines)``.
    """
    return slope_target * math.sqrt(energy_ss / n_lines)


def required_min_lines_stat(
    slope_target: float, per_line_error: float, var_per_line: float
) -> float:
    """Statistical (Gauss-Markov) minimum line count, kernel ``sum w_k**2 = 1/ss_e``.

    Mirror of the ``ErrorBudget.olsSlope_noise_gain``-backed law: under independent ordinate
    noise of variance ``per_line_error**2`` with per-line energy variance ``var_per_line``
    (so ``ss_e = n * var_per_line``), ``n >= per_line_error**2 / (var_per_line * slope_target**2)``.
    The deterministic worst case does **not** give a line-count threshold; this is the
    statistical route (see the Lean module docstring).
    """
    return per_line_error**2 / (var_per_line * slope_target**2)


def slope_target_from_temp_rel(rel_temp_target: float, kb_times_t: float) -> float:
    """Slope (inverse-temperature) accuracy needed for a target *relative* temperature error.

    Mirror of the exact identity ``ErrorBudget.temp_rel_error_eq`` (``|Delta T|/T = kB*T*|Delta beta|``):
    ``slope_target = rel_temp_target / (kB * T)``.
    """
    return rel_temp_target / kb_times_t


def density_budget_from_composition(
    comp_target: float, total_density: float, n_species: float
) -> float:
    """Per-species density-error budget guaranteeing a target composition accuracy.

    Mirror of ``ErrorBudget.composition_target_sufficient``:
    ``comp_target * total_density / (n_species + 1)``.
    """
    return comp_target * total_density / (n_species + 1.0)


# --------------------------------------------------------------------------------------------
# Physical adoption layer — maps the dimensionless chain onto the pipeline's units (Kelvin/eV)
# --------------------------------------------------------------------------------------------
#
# Boltzmann plot: ``y = ln(I*lambda/(g*A))`` vs ``E`` (eV), slope ``m = -1/(kB*T)``.
# A per-line ordinate error ``eps`` (dimensionless; ``eps ~ 1/SNR`` since ``y = ln(I) + const``)
# and an energy spread ``ss_e`` in eV**2 give a slope error ``|Delta m| = eps*sqrt(n)/sqrt(ss_e)``
# (eV^-1). By ``temp_rel_error_eq``, ``sigma_T/T = kB*T*|Delta m| = KB_EV*T_K*|Delta m|``.
#
# ``min_energy_spread_ev`` in the pipeline is an energy *span* ``R = max(E) - min(E)`` (eV), not
# ``ss_e``. Their relation depends on how the lines are distributed across the span:
#   * "uniform":   ss_e ~= n * R**2 / 12     (lines spread evenly -- the practitioner default)
#   * "endpoints": ss_e  = n * R**2 / 4      (extreme: half at each end -- maximal ss_e for a span)
# We adopt "uniform" by default (conservative for a gate).

_SS_OVER_SPAN_SQ = {"uniform": 1.0 / 12.0, "endpoints": 1.0 / 4.0}


def _ss_per_line_factor(distribution: str) -> float:
    try:
        return _SS_OVER_SPAN_SQ[distribution]
    except KeyError as exc:
        raise ValueError(
            f"distribution must be one of {sorted(_SS_OVER_SPAN_SQ)}, got {distribution!r}"
        ) from exc


def temp_rel_error_bound(
    per_line_error: float, n_lines: float, energy_ss_ev2: float, t_k: float
) -> float:
    """Worst-case *relative* temperature error ``sigma_T/T`` from the per-line ordinate error.

    ``KB_EV * t_k * slope_error_bound(per_line_error, n_lines, energy_ss_ev2)`` with the energy
    spread ``energy_ss_ev2 = sum_k (E_k - Ebar)**2`` in eV**2 (combines ``olsSlope_stable_l2``
    and the exact ``temp_rel_error_eq``).
    """
    return KB_EV * t_k * slope_error_bound(per_line_error, n_lines, energy_ss_ev2)


def required_energy_span_ev(
    rel_temp_target: float,
    snr: float,
    n_lines: float,
    t_k: float,
    distribution: str = "uniform",
) -> float:
    """Minimum energy *span* (eV) to hit a target ``sigma_T/T`` at the given SNR and line count.

    Inverts the chain: the required ``ss_e`` (``requiredEnergySpread_sufficient`` with the slope
    target ``rel_temp_target/(KB_EV*t_k)`` and per-line error ``eps = 1/snr``), converted to a
    span under the stated line ``distribution`` (default ``"uniform"``: ``ss_e = n*R**2/12``).
    """
    slope_target = slope_target_from_temp_rel(rel_temp_target, KB_EV * t_k)
    ss_e = required_energy_spread(slope_target, 1.0 / snr, n_lines)
    return math.sqrt(ss_e / (n_lines * _ss_per_line_factor(distribution)))


def min_snr_for_target(
    rel_temp_target: float,
    energy_span_ev: float,
    n_lines: float,
    t_k: float,
    distribution: str = "uniform",
) -> float:
    """Minimum per-line SNR to hit a target ``sigma_T/T`` at the given energy span and line count.

    Inverts ``maxPerLineError_sufficient``: ``eps_max = slope_target * sqrt(ss_e/n)`` with the
    span converted to ``ss_e`` under ``distribution``; ``min_snr = 1/eps_max``.
    """
    slope_target = slope_target_from_temp_rel(rel_temp_target, KB_EV * t_k)
    ss_e = n_lines * _ss_per_line_factor(distribution) * energy_span_ev**2
    eps_max = max_per_line_error(slope_target, n_lines, ss_e)
    return 1.0 / eps_max


@dataclass(frozen=True)
class LineSelectionThresholds:
    """Line-selection gates derived from a target temperature accuracy (not tuned)."""

    min_energy_spread_ev: float
    min_snr: float
    min_lines_per_element: int
    rel_temp_target: float
    t_k: float
    distribution: str


def derive_line_selection_thresholds(
    rel_temp_target: float,
    n_lines: int,
    t_k: float,
    snr: float = 10.0,
    distribution: str = "uniform",
) -> LineSelectionThresholds:
    """Derive ``LineSelector`` gates from a target relative-temperature accuracy.

    Given a target ``sigma_T/T`` (``rel_temp_target``), an expected per-line ``snr``, a line
    count ``n_lines``, and the plasma temperature ``t_k``, returns the minimum energy span and
    SNR that the verified chain guarantees will meet the target. ``min_lines_per_element`` is set
    to ``n_lines`` (the assumed count); the line-count law itself is statistical, see
    ``required_min_lines_stat``.
    """
    return LineSelectionThresholds(
        min_energy_spread_ev=required_energy_span_ev(
            rel_temp_target, snr, n_lines, t_k, distribution
        ),
        min_snr=snr,
        min_lines_per_element=n_lines,
        rel_temp_target=rel_temp_target,
        t_k=t_k,
        distribution=distribution,
    )
