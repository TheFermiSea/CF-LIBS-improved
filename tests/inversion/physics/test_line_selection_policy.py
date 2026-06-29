"""Tests for the opt-in DB+window line-selection policies.

Covers:
* the DEFAULT ``"emissivity"`` policy reproduces the established
  ``tests/benchmarks/ded_precision/line_lists.select_lines`` behavior
  byte-identically (default path unchanged), and
* the opt-in ``"neutral_anchor"`` lever-L2 policy: neutral-plane preference,
  resonance-anchor admission, and dropping ion-only elements.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from cflibs.atomic.structures import Transition
from cflibs.inversion.physics.line_selection import (
    SelectedLine,
    select_lines_by_policy,
)


class FakeDB:
    """Minimal duck-typed atomic DB returning canned transitions."""

    def __init__(self, transitions: List[Transition]):
        self._t = transitions

    def get_transitions(
        self,
        element: str,
        ionization_stage: Optional[int] = None,
        wavelength_min: Optional[float] = None,
        wavelength_max: Optional[float] = None,
        min_relative_intensity: Optional[float] = None,
    ) -> List[Transition]:
        out = []
        for t in self._t:
            if t.element != element:
                continue
            if ionization_stage is not None and t.ionization_stage != ionization_stage:
                continue
            if wavelength_min is not None and t.wavelength_nm < wavelength_min:
                continue
            if wavelength_max is not None and t.wavelength_nm > wavelength_max:
                continue
            out.append(t)
        return out


def _tr(
    element: str,
    stage: int,
    wl: float,
    e_k: float,
    a_ki: float = 1e8,
    g_k: int = 5,
    resonance: bool = False,
) -> Transition:
    return Transition(
        element=element,
        ionization_stage=stage,
        wavelength_nm=wl,
        A_ki=a_ki,
        E_k_ev=e_k,
        E_i_ev=0.0 if resonance else 1.0,
        g_k=g_k,
        g_i=3,
        is_resonance=resonance,
        aki_uncertainty=0.05,
    )


def _emissivity_transitions(element: str = "Fe") -> List[Transition]:
    """A spread of neutral + ion lines suitable for the emissivity policy."""
    trs: List[Transition] = []
    for i in range(12):
        trs.append(_tr(element, 1, 300.0 + 5.0 * i, e_k=2.0 + 0.4 * i, a_ki=1e8 - 1e6 * i))
    for i in range(4):
        trs.append(_tr(element, 2, 360.0 + 3.0 * i, e_k=4.0 + 0.5 * i, a_ki=5e7))
    return trs


def test_default_policy_byte_identical_to_ded_select_lines():
    """DEFAULT 'emissivity' policy reproduces the established DED selection."""
    from tests.benchmarks.ded_precision.line_lists import select_lines

    db = FakeDB(_emissivity_transitions("Fe"))
    window = (290.0, 400.0)
    n_lines = 5

    promoted = select_lines_by_policy(db, "Fe", window, n_lines, policy="emissivity")
    legacy = select_lines(db, "Fe", window, n_lines, prefer_spread=True)

    assert [s.wavelength_nm for s in promoted] == [s.wavelength_nm for s in legacy]
    assert [s.ionization_stage for s in promoted] == [s.ionization_stage for s in legacy]
    assert [s.E_k_ev for s in promoted] == [s.E_k_ev for s in legacy]
    assert all(isinstance(s, SelectedLine) for s in promoted)


def test_default_policy_is_the_default():
    """Omitting ``policy`` selects the emissivity policy (no behavior change)."""
    db = FakeDB(_emissivity_transitions("Fe"))
    window = (290.0, 400.0)
    explicit = select_lines_by_policy(db, "Fe", window, 5, policy="emissivity")
    implicit = select_lines_by_policy(db, "Fe", window, 5)
    assert [s.wavelength_nm for s in implicit] == [s.wavelength_nm for s in explicit]


def test_neutral_anchor_drops_ion_only_element():
    """An element with no neutral line in band is dropped (empty list)."""
    db = FakeDB([_tr("Cu", 2, 320.0, e_k=5.0), _tr("Cu", 2, 325.0, e_k=6.0)])
    out = select_lines_by_policy(db, "Cu", (310.0, 330.0), 8, policy="neutral_anchor")
    assert out == []


def test_neutral_anchor_prefers_neutral_lines():
    """Neutral lines are preferred; ion lines never appear in the output."""
    trs = [_tr("Fe", 1, 300.0 + 4.0 * i, e_k=2.0 + 0.5 * i) for i in range(8)]
    trs += [_tr("Fe", 2, 340.0 + 2.0 * i, e_k=5.0, a_ki=1e9) for i in range(4)]
    db = FakeDB(trs)
    out = select_lines_by_policy(db, "Fe", (290.0, 360.0), 6, policy="neutral_anchor")
    assert out, "expected neutral lines to be selected"
    assert all(s.ionization_stage == 1 for s in out)


def test_neutral_anchor_admits_resonance_when_neutral_resonance_only():
    """With <2 non-resonance neutrals, a strong neutral resonance is admitted."""
    db = FakeDB(
        [
            _tr("Cu", 1, 324.75, e_k=3.8, a_ki=1.4e8, resonance=True),
            _tr("Cu", 1, 327.40, e_k=3.8, a_ki=1.4e8, resonance=True),
        ]
    )
    out = select_lines_by_policy(db, "Cu", (320.0, 330.0), 8, policy="neutral_anchor")
    assert out, "resonance neutral anchor should be admitted"
    assert all(s.is_resonance for s in out)
    assert all(s.ionization_stage == 1 for s in out)


def test_neutral_anchor_resonance_ratio_branch():
    """A dominant resonance line (>= ratio) is admitted alongside weak neutrals."""
    db = FakeDB(
        [
            _tr("Mn", 1, 300.0, e_k=4.0, a_ki=1e6),  # weak non-resonance
            _tr("Mn", 1, 305.0, e_k=4.5, a_ki=1e6),  # weak non-resonance
            _tr("Mn", 1, 279.5, e_k=4.4, a_ki=1e9, resonance=True),  # dominant resonance
        ]
    )
    out = select_lines_by_policy(db, "Mn", (270.0, 310.0), 8, policy="neutral_anchor")
    wls = {round(s.wavelength_nm, 2) for s in out}
    assert 279.5 in wls, "dominant resonance anchor should be included"


def test_unknown_policy_raises():
    db = FakeDB(_emissivity_transitions("Fe"))
    with pytest.raises(ValueError, match="Unknown line-selection policy"):
        select_lines_by_policy(db, "Fe", (290.0, 400.0), 5, policy="bogus")


# ---------------------------------------------------------------------------
# Matrix-isolation filter (opt-in dominant-matrix deblending)
# ---------------------------------------------------------------------------


def _obs(element, stage, wl, e_k=4.0, a_ki=1e8, g_k=5):
    from cflibs.inversion.physics.boltzmann import LineObservation

    return LineObservation(
        wavelength_nm=wl,
        intensity=1.0,
        intensity_uncertainty=0.01,
        element=element,
        ionization_stage=stage,
        E_k_ev=e_k,
        g_k=g_k,
        A_ki=a_ki,
    )


def test_matrix_isolation_drops_blended_trace_keeps_isolated():
    """A trace line on top of a strong matrix transition is dropped; an isolated
    trace line and all matrix lines are kept."""
    from cflibs.inversion.physics.line_selection import filter_matrix_blended_lines

    # Ti matrix transitions: one near the blended V line, none near the isolated one.
    db = FakeDB([_tr("Ti", 1, 306.62, e_k=4.1, a_ki=1e8)])
    obs = [
        _obs("V", 1, 306.64, e_k=4.1),  # 0.02 nm from a strong Ti line -> blended
        _obs("V", 1, 350.00, e_k=3.5),  # no Ti nearby -> isolated, kept
        _obs("Ti", 1, 306.62, e_k=4.1),  # matrix line itself -> always kept
    ]
    kept, dropped = filter_matrix_blended_lines(
        obs, db, "Ti", resolving_power=2000.0, min_lines_per_element=0
    )
    kept_wl = {(o.element, round(o.wavelength_nm, 2)) for o in kept}
    assert ("V", 306.64) not in kept_wl  # blended V dropped
    assert ("V", 350.0) in kept_wl  # isolated V kept
    assert ("Ti", 306.62) in kept_wl  # matrix line kept
    assert [(o.element, round(o.wavelength_nm, 2)) for o in dropped] == [("V", 306.64)]


def test_matrix_isolation_min_lines_floor_restores_least_contaminated():
    """When every trace line is blended, the per-element floor restores the
    least-contaminated one so the element is never silently deleted."""
    from cflibs.inversion.physics.line_selection import filter_matrix_blended_lines

    db = FakeDB(
        [
            _tr("Ti", 1, 306.62, e_k=4.1, a_ki=5e8),  # very strong -> high contamination
            _tr("Ti", 1, 310.00, e_k=4.1, a_ki=1e8),  # weaker -> lower contamination
        ]
    )
    obs = [
        _obs("V", 1, 306.63, e_k=4.1),  # heavily blended
        _obs("V", 1, 310.01, e_k=4.1),  # less blended (weaker Ti neighbour)
    ]
    kept, dropped = filter_matrix_blended_lines(
        obs, db, "Ti", resolving_power=2000.0, min_lines_per_element=1
    )
    kept_v = [round(o.wavelength_nm, 2) for o in kept if o.element == "V"]
    assert kept_v == [310.01], "the least-contaminated V line is restored to meet the floor"
    assert len(dropped) == 1


def test_matrix_isolation_noop_when_no_matrix_lines_in_band():
    """No matrix transition in any window -> nothing dropped (default-path safe)."""
    from cflibs.inversion.physics.line_selection import filter_matrix_blended_lines

    db = FakeDB([_tr("Ti", 1, 500.0, e_k=4.1)])  # far from every trace line
    obs = [_obs("V", 1, 306.64), _obs("Al", 1, 396.15), _obs("Ti", 1, 410.0)]
    kept, dropped = filter_matrix_blended_lines(obs, db, "Ti", resolving_power=2000.0)
    assert dropped == []
    assert len(kept) == 3
