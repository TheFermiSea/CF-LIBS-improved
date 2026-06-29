"""First-principles line-selection thresholds, derived from the Boltzmann-plot error
budget — replacing the pipeline's tuned "magic numbers" with formulas PROVEN in the
companion Lean spec (``CflibsFormal.ErrorBudget``).

Each function mirrors a machine-verified theorem (and the oracle Float export in
``cflibs-formal/oracle/Generate.lean``); ``tests/oracle/test_derived_thresholds.py``
conformance-tests them against the spec's emitted fixtures, so a drift from the proof
fails CI. The physical chain (proven end-to-end):

    measurement error ε  --olsSlope_stable_l2-->  slope error |Δβ| ≤ ε·√n/√ssE
    --temp_rel_error_eq-->  |ΔT|/T = kB·T·|Δβ|
    --composition_target_sufficient-->  per-species density budget for a target σ_C

Inverting that chain turns a *target accuracy* (σ_T/T or σ_C) into the required line
count, energy spread, and per-line SNR — i.e. the DERIVE-tier of the parameter scheme
(min_lines_per_element / min_energy_spread_ev / min_snr were tuned constants; here they
are consequences of a target with explicit validity hypotheses).

Symbols (dimensionless-consistent; energies in eV, ``ssE = Σ(E_k − Ē)²`` in eV²):
``tau_beta`` target on the inverse-temperature slope β=1/(kB·T); ``snr`` per-line
ordinate error (≈ 1/SNR of the log-intensity); ``n`` line count; ``v_per_line``
per-line energy variance (ssE = n·v_per_line); ``card`` number of species.

Physics-only: pure stdlib math (no ML, no DB).
"""

from __future__ import annotations

import math

from cflibs.core.constants import KB_EV

# ---- verbatim mirrors of CflibsFormal.ErrorBudget (conformance-pinned to the oracle) ----


def slope_error_bound(snr: float, n: float, ss_e: float) -> float:
    """|Δβ| ≤ snr·√n/√ssE — the worst-case inverse-temperature slope error from a per-line
    ordinate error over n lines of spread ssE (ErrorBudget.olsSlope_stable_l2)."""
    return snr * math.sqrt(n) / math.sqrt(ss_e)


def required_energy_spread(tau_beta: float, snr: float, n: float) -> float:
    """ssE ≥ snr²·n/τβ² — the energy spread that GUARANTEES slope error ≤ τβ
    (ErrorBudget.requiredEnergySpread_sufficient). The derived ``min_energy_spread``."""
    return snr * snr * n / (tau_beta * tau_beta)


def max_per_line_error(tau_beta: float, n: float, ss_e: float) -> float:
    """snr ≤ τβ·√(ssE/n) — the largest tolerable per-line error (≡ the minimum SNR) for a
    target slope error τβ (ErrorBudget.maxPerLineError_sufficient). The derived ``min_snr``."""
    return tau_beta * math.sqrt(ss_e / n)


def required_min_lines(tau_beta: float, snr: float, v_per_line: float) -> float:
    """n ≥ snr²/(v_per_line·τβ²) — the statistical (Gauss-Markov) line count for a target
    slope std τβ (ErrorBudget.olsSlope_noise_gain). The derived ``min_lines_per_element``."""
    return snr * snr / (v_per_line * tau_beta * tau_beta)


def ols_noise_gain(ss_e: float) -> float:
    """Σ wₖ² = 1/ssE — the Gauss-Markov slope-variance multiplier (ErrorBudget.olsSlope_noise_gain)."""
    return 1.0 / ss_e


def slope_target_from_temp_rel(rel_t_target: float, kB: float, T: float) -> float:
    """τβ = relTtarget/(kB·T) — the slope accuracy needed for a target RELATIVE temperature
    error, from the EXACT identity |ΔT|/T = kB·T·|Δβ| (ErrorBudget.temp_rel_error_eq)."""
    return rel_t_target / (kB * T)


def density_budget_from_composition(tau_C: float, s_hat: float, card: float) -> float:
    """δ ≤ τC·Ŝ/(card+1) — the per-species absolute density-error budget that GUARANTEES a
    target composition accuracy τC over ``card`` species at total recovered density Ŝ
    (ErrorBudget.composition_target_sufficient)."""
    return tau_C * s_hat / (card + 1.0)


# ---- pipeline-facing wrappers: target accuracy -> the actual config thresholds ----


def min_lines_per_element_for(
    rel_t_target: float,
    snr: float,
    energy_var_ev2: float,
    temperature_K: float = 11604.5,
    floor: int = 3,
) -> int:
    """DERIVED ``min_lines_per_element``: the fewest lines achieving a target relative
    temperature accuracy at the given per-line SNR + per-line energy variance. Composes
    ``slope_target_from_temp_rel`` + ``required_min_lines``; floored at ``floor`` (the
    statistical minimum for a slope+intercept fit). This replaces the tuned constant 3/20."""
    tau_beta = slope_target_from_temp_rel(rel_t_target, KB_EV, temperature_K)
    n = required_min_lines(tau_beta, snr, energy_var_ev2)
    return max(floor, math.ceil(n))


def min_energy_spread_ev_for(
    rel_t_target: float, snr: float, n_lines: float, temperature_K: float = 11604.5
) -> float:
    """DERIVED ``min_energy_spread_ev``: the per-line energy std (eV) whose ssE = n·var meets a
    target relative temperature accuracy. Replaces the tuned constant 2.0 eV."""
    tau_beta = slope_target_from_temp_rel(rel_t_target, KB_EV, temperature_K)
    ss_e = required_energy_spread(tau_beta, snr, n_lines)
    return math.sqrt(ss_e / max(n_lines, 1.0))
