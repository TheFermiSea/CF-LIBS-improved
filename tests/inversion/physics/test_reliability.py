"""Tests for cflibs.inversion.physics.reliability — reliability ranking + LTE gate.

Each test pins a property of a machine-verified ``CflibsFormal`` theorem:
``twoLineBeta_stable_sharp`` (temperature conditioning), ``composition_dist_vector_le``
(composition error bound), ``mcWhirterBound`` + its monotonicity lemmas, and
``stark_saha_lte_consistent`` (the two-diagnostic LTE gate).
"""

from __future__ import annotations

import math

import pytest

from cflibs.core.constants import MCWHIRTER_CONST
from cflibs.inversion.physics.reliability import (
    best_temperature_pair,
    composition_error_bound,
    mcwhirter_min_ne,
    rank_line_pairs_by_conditioning,
    stark_saha_lte_gate,
    temperature_conditioning,
)

# ---- temperature_conditioning (twoLineBeta_stable_sharp) ----


def test_temperature_conditioning_sharp_value():
    # 2 / |E_i - E_j|  — the exact sharp Lipschitz constant.
    assert temperature_conditioning(5.0, 1.0) == pytest.approx(2.0 / 4.0)
    assert temperature_conditioning(1.0, 5.0) == pytest.approx(2.0 / 4.0)  # symmetric


def test_temperature_conditioning_monotone_in_gap():
    # Wider energy separation -> strictly LOWER conditioning value (better).
    narrow = temperature_conditioning(2.0, 1.0)  # gap 1.0
    wide = temperature_conditioning(5.0, 1.0)  # gap 4.0
    assert wide < narrow


def test_temperature_conditioning_degenerate_is_inf():
    assert temperature_conditioning(3.0, 3.0) == math.inf


# ---- rank_line_pairs_by_conditioning / best_temperature_pair ----


def test_rank_prefers_widest_separation_first():
    # Energies: 0, 1, 5. Widest gap is (0, 5) -> indices (0, 2) ranked first.
    ranked = rank_line_pairs_by_conditioning([0.0, 1.0, 5.0])
    assert len(ranked) == 3  # C(3,2)
    i, j, cond = ranked[0]
    assert (i, j) == (0, 2)
    # Front pair has the smallest conditioning number.
    assert cond == pytest.approx(2.0 / 5.0)
    # Sorted best -> worst (non-decreasing conditioning).
    conds = [p[2] for p in ranked]
    assert conds == sorted(conds)


def test_best_temperature_pair_picks_widest():
    assert best_temperature_pair([0.0, 1.0, 5.0]) == (0, 2)
    assert best_temperature_pair([3.0, 3.2]) == (0, 1)


def test_rank_empty_and_single_line():
    assert rank_line_pairs_by_conditioning([]) == []
    assert rank_line_pairs_by_conditioning([1.0]) == []


def test_best_temperature_pair_requires_two_lines():
    with pytest.raises(ValueError):
        best_temperature_pair([1.0])


# ---- composition_error_bound (composition_dist_vector_le) ----


def test_composition_error_bound_value():
    # 2 * card * delta / Shat
    assert composition_error_bound(3, 0.1, 2.0) == pytest.approx(2.0 * 3 * 0.1 / 2.0)


def test_composition_error_bound_monotone():
    base = composition_error_bound(3, 0.1, 2.0)
    # grows with card and delta, shrinks with total density.
    assert composition_error_bound(4, 0.1, 2.0) > base
    assert composition_error_bound(3, 0.2, 2.0) > base
    assert composition_error_bound(3, 0.1, 4.0) < base


def test_composition_error_bound_rejects_nonpositive_density():
    with pytest.raises(ValueError):
        composition_error_bound(3, 0.1, 0.0)
    with pytest.raises(ValueError):
        composition_error_bound(3, 0.1, -1.0)


# ---- mcwhirter_min_ne (mcWhirterBound + monotonicity) ----


def test_mcwhirter_min_ne_value():
    t, de = 10000.0, 2.0
    expected = MCWHIRTER_CONST * math.sqrt(t) * de**3
    assert mcwhirter_min_ne(t, de) == pytest.approx(expected)


def test_mcwhirter_monotone_in_temperature():
    # mcWhirterBound_mono_T: hotter plasma -> higher bound (scales as sqrt(T)).
    assert mcwhirter_min_ne(20000.0, 2.0) > mcwhirter_min_ne(10000.0, 2.0)


def test_mcwhirter_monotone_in_energy_gap():
    # mcWhirterBound_mono_dE: larger gap -> higher bound (scales as dE^3).
    assert mcwhirter_min_ne(10000.0, 3.0) > mcwhirter_min_ne(10000.0, 2.0)


def test_mcwhirter_cube_scaling():
    # Doubling dE multiplies the bound by 2^3 = 8.
    assert mcwhirter_min_ne(10000.0, 2.0) == pytest.approx(8.0 * mcwhirter_min_ne(10000.0, 1.0))


# ---- stark_saha_lte_gate (stark_saha_lte_consistent) ----


def _above_mcwhirter(t=10000.0, de=2.0):
    """An n_e comfortably above the McWhirter bound for (t, de)."""
    return 10.0 * mcwhirter_min_ne(t, de)


def test_gate_passes_when_agree_and_above_mcwhirter():
    t, de = 10000.0, 2.0
    ne = _above_mcwhirter(t, de)
    valid, reason = stark_saha_lte_gate(ne, ne * 1.05, t, de, rtol=0.5)
    assert valid is True
    assert reason == "ok"


def test_gate_fails_on_disagreement():
    t, de = 10000.0, 2.0
    ne = _above_mcwhirter(t, de)
    # Stark is 3x Saha -> rel disagreement = (2)/(2.0) = 1.0 > 0.5.
    valid, reason = stark_saha_lte_gate(ne, ne * 3.0, t, de, rtol=0.5)
    assert valid is False
    assert reason.startswith("disagree")


def test_gate_fails_below_mcwhirter():
    t, de = 10000.0, 2.0
    min_ne = mcwhirter_min_ne(t, de)
    # Agreeing but BELOW the McWhirter bound.
    ne = 0.01 * min_ne
    valid, reason = stark_saha_lte_gate(ne, ne * 1.02, t, de, rtol=0.5)
    assert valid is False
    assert reason.startswith("below-mcwhirter")


def test_gate_fails_on_invalid_ne():
    t, de = 10000.0, 2.0
    for bad in (0.0, -1.0, math.nan, math.inf):
        valid, reason = stark_saha_lte_gate(bad, 1e17, t, de)
        assert valid is False
        assert reason == "invalid-ne"


def test_gate_rtol_controls_strictness():
    t, de = 10000.0, 2.0
    ne = _above_mcwhirter(t, de)
    # 40% apart: rel = 0.4/1.2 ... compute pair within a chosen tolerance.
    a, b = ne, ne * 1.4
    # mean = 1.2 ne; |a-b| = 0.4 ne; rel = 0.333
    rel = abs(a - b) / (0.5 * (a + b))
    assert rel < 0.5  # passes default
    assert stark_saha_lte_gate(a, b, t, de, rtol=0.5)[0] is True
    # but a stricter rtol rejects it.
    assert stark_saha_lte_gate(a, b, t, de, rtol=rel * 0.5)[0] is False
