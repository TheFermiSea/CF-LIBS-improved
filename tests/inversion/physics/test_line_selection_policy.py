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
