"""Reliability ranking + LTE self-consistency gate, each formula a verbatim mirror of a
machine-verified theorem in the companion Lean spec (``CflibsFormal``).

This module turns the *proven robustness constants* of the CF-LIBS estimators into
practical selectors and refuse/flag gates:

* **Temperature conditioning** — the SHARP Lipschitz constant of the two-line slope
  estimate (``CflibsFormal.Robustness.twoLineBeta_stable_sharp``):
  ``|Δβ| = 2·ε/|E_i − E_j|`` for the worst-case opposite-sign ordinate perturbation.
  So ``2/|E_i − E_j|`` is the exact error-amplification factor — a line *pair* with a
  wider upper-level energy separation is strictly better conditioned for temperature.

* **Composition error bound** — the WHOLE-VECTOR ℓ¹ bound
  (``CflibsFormal.CompositionRobustness.composition_dist_vector_le``):
  ``∑ₛ |C̃ₛ − Cₛ| ≤ 2·card·δ/Ŝ`` for per-species density error ≤ δ over ``card``
  species at recovered total density Ŝ. The certified worst-case composition error.

* **McWhirter LTE bound** — ``n_e ≥ 1.6·10¹²·√T·(ΔE)³``
  (``CflibsFormal.StarkBroadening.mcWhirterBound``), proven monotone in both T
  (``mcWhirterBound_mono_T``) and ΔE (``mcWhirterBound_mono_dE``).

* **Stark↔Saha LTE cross-check** — two INDEPENDENT electron-density diagnostics
  (a line WIDTH vs a stage-intensity RATIO) are LTE-self-consistent iff they AGREE
  and their common value clears the McWhirter bound
  (``CflibsFormal.StarkBroadening.stark_saha_lte_consistent``). This is the
  reliability gate the M7 refuse-to-report path consumes.

Physics-only: pure stdlib ``math`` (no numpy state, no ML, no DB).
"""

from __future__ import annotations

import math
from typing import Sequence

from cflibs.core.constants import MCWHIRTER_CONST

__all__ = [
    "temperature_conditioning",
    "rank_line_pairs_by_conditioning",
    "best_temperature_pair",
    "composition_error_bound",
    "mcwhirter_min_ne",
    "stark_saha_lte_gate",
]


def temperature_conditioning(e_i: float, e_j: float) -> float:
    """Conditioning number of the two-line temperature (slope) estimate.

    The SHARP Lipschitz constant of the two-line inverse-temperature slope
    β = (y_j − y_i)/(E_i − E_j): under the worst-case opposite-sign ordinate
    perturbation (y_i ↦ y_i − ε, y_j ↦ y_j + ε), the slope error is EXACTLY
    ``2·ε/|E_i − E_j|`` (``CflibsFormal.Robustness.twoLineBeta_stable_sharp``).
    Hence ``2/|E_i − E_j|`` is the exact per-unit-ε error amplification.

    Lower is better: a wider upper-level energy separation ``|E_i − E_j|`` gives a
    smaller conditioning number, so the slope (and thus the temperature) is recovered
    more robustly. Returns ``+inf`` when the two energies coincide (the estimate is
    undefined — division by zero in the slope).

    Parameters
    ----------
    e_i, e_j : float
        Upper-level energies (eV) of the two transitions forming the pair.

    Returns
    -------
    float
        ``2/|E_i − E_j|`` (≥ 0), or ``+inf`` if ``E_i == E_j``.
    """
    gap = abs(e_i - e_j)
    if gap == 0.0:
        return math.inf
    return 2.0 / gap


def rank_line_pairs_by_conditioning(
    lines: Sequence[float],
) -> list[tuple[int, int, float]]:
    """Rank all line pairs best→worst by temperature conditioning.

    Enumerates every unordered pair ``(i, j)`` of the given upper-level energies and
    sorts them by ascending ``temperature_conditioning`` (= descending energy
    separation ``|E_i − E_j|``). A selector should PREFER the front of this list:
    those are the widest-energy-separation pairs, the best conditioned for the
    two-line temperature estimate (``twoLineBeta_stable_sharp``).

    Parameters
    ----------
    lines : sequence of float
        Upper-level energies (eV); ``lines[k]`` is line ``k``.

    Returns
    -------
    list of (i, j, conditioning)
        Index pairs with ``i < j`` and their conditioning number, sorted
        best (lowest conditioning) first. Empty if fewer than two lines.
    """
    n = len(lines)
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, temperature_conditioning(lines[i], lines[j])))
    pairs.sort(key=lambda p: p[2])
    return pairs


def best_temperature_pair(lines: Sequence[float]) -> tuple[int, int]:
    """Indices of the best-conditioned line pair for the two-line temperature estimate.

    The pair with the WIDEST upper-level energy separation (minimum conditioning
    number; ``twoLineBeta_stable_sharp``). This is the front element of
    ``rank_line_pairs_by_conditioning``.

    Parameters
    ----------
    lines : sequence of float
        Upper-level energies (eV).

    Returns
    -------
    (i, j) : tuple of int
        Indices (``i < j``) of the best-conditioned pair.

    Raises
    ------
    ValueError
        If fewer than two lines are supplied (no pair exists).
    """
    ranked = rank_line_pairs_by_conditioning(lines)
    if not ranked:
        raise ValueError("best_temperature_pair requires at least two lines")
    i, j, _ = ranked[0]
    return (i, j)


def composition_error_bound(card: int, delta: float, total_density: float) -> float:
    """Certified worst-case ℓ¹ error of the recovered composition vector.

    The WHOLE-VECTOR bound ``∑ₛ |C̃ₛ − Cₛ| ≤ 2·card·δ/Ŝ``
    (``CflibsFormal.CompositionRobustness.composition_dist_vector_le``): for a
    per-species absolute density error bounded by ``δ`` over ``card`` species at
    recovered total density ``Ŝ``, the total composition (mole-fraction) error in the
    ℓ¹ norm cannot exceed ``2·card·δ/Ŝ``. Smaller is better; it shrinks with a larger
    recovered total density and grows linearly in both the species count and the
    per-species density uncertainty.

    Parameters
    ----------
    card : int
        Number of species in the composition vector.
    delta : float
        Per-species absolute density-error budget (same units as ``total_density``).
    total_density : float
        Recovered total number density Ŝ (must be > 0).

    Returns
    -------
    float
        The ℓ¹ composition error bound ``2·card·δ/Ŝ``.

    Raises
    ------
    ValueError
        If ``total_density`` is not strictly positive.
    """
    if total_density <= 0.0:
        raise ValueError("total_density must be strictly positive")
    return 2.0 * card * delta / total_density


def mcwhirter_min_ne(t_k: float, d_e_ev: float) -> float:
    """McWhirter lower bound on electron density for LTE validity.

    ``n_e ≥ 1.6·10¹²·√T·(ΔE)³`` (``CflibsFormal.StarkBroadening.mcWhirterBound``):
    the classical McWhirter criterion (Cristoforetti et al. 2010) for collisional
    (LTE) processes to dominate radiative ones, where ``ΔE`` is the largest relevant
    upper-level energy gap (eV) and ``T`` is in kelvin. Proven monotone increasing in
    both ``T`` (``mcWhirterBound_mono_T``) and ``ΔE`` (``mcWhirterBound_mono_dE``):
    a hotter plasma or a larger energy gap demands a higher electron density for LTE.

    Parameters
    ----------
    t_k : float
        Temperature (K).
    d_e_ev : float
        Largest relevant upper-level energy gap ΔE (eV).

    Returns
    -------
    float
        The minimum electron density (cm⁻³) for LTE validity.
    """
    return MCWHIRTER_CONST * math.sqrt(t_k) * d_e_ev**3


def stark_saha_lte_gate(
    ne_stark: float,
    ne_saha: float,
    t_k: float,
    d_e_ev: float,
    rtol: float = 0.5,
) -> tuple[bool, str]:
    """Two-diagnostic LTE self-consistency gate (refuse/flag decision).

    Mirrors ``CflibsFormal.StarkBroadening.stark_saha_lte_consistent``: the Stark
    route recovers ``n_e`` from a measured line WIDTH and the Saha route recovers it
    from a stage-intensity RATIO — two physically INDEPENDENT diagnostics consuming
    genuinely DIFFERENT observations. LTE is certified self-consistent iff BOTH
    hypotheses of the theorem hold:

    1. **Agreement** (``hagree``): the two independent estimates agree within ``rtol``
       (relative tolerance on their mean). Their equality is empirical evidence, not a
       definitional identity, precisely because they feed different observations.
    2. **Clears McWhirter** (``hlte``): the agreed value (their mean) is at least the
       McWhirter LTE bound ``mcwhirter_min_ne(T, ΔE)``.

    On failure the reason names the violated hypothesis so the M7 refuse-to-report
    path can flag the result (``disagree`` vs ``below-mcwhirter``), and a degenerate
    non-positive density short-circuits to ``invalid-ne``.

    Parameters
    ----------
    ne_stark : float
        Electron density (cm⁻³) from the Stark-width diagnostic.
    ne_saha : float
        Electron density (cm⁻³) from the Saha stage-ratio diagnostic.
    t_k : float
        Temperature (K) for the McWhirter bound.
    d_e_ev : float
        Largest relevant upper-level energy gap ΔE (eV) for the McWhirter bound.
    rtol : float, optional
        Relative tolerance for diagnostic agreement (default 0.5, i.e. the two
        densities must be within 50% of their mean).

    Returns
    -------
    (valid, reason) : tuple of (bool, str)
        ``valid`` is True iff both LTE hypotheses hold. ``reason`` is ``"ok"`` on
        success, else ``"invalid-ne"`` / ``"disagree"`` / ``"below-mcwhirter"``.
    """
    if not (math.isfinite(ne_stark) and math.isfinite(ne_saha)):
        return (False, "invalid-ne")
    if ne_stark <= 0.0 or ne_saha <= 0.0:
        return (False, "invalid-ne")

    mean_ne = 0.5 * (ne_stark + ne_saha)
    rel_disagreement = abs(ne_stark - ne_saha) / mean_ne
    if rel_disagreement > rtol:
        return (
            False,
            f"disagree: |n_e^Stark - n_e^Saha|/mean = {rel_disagreement:.3g} > rtol={rtol:.3g}",
        )

    min_ne = mcwhirter_min_ne(t_k, d_e_ev)
    if mean_ne < min_ne:
        return (
            False,
            f"below-mcwhirter: mean n_e = {mean_ne:.3g} < McWhirter bound {min_ne:.3g} cm^-3",
        )

    return (True, "ok")
