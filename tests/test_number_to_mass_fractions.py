"""Unit test for the canonical number->mass fraction converter (blueprint M3-1)."""

import pytest

from cflibs.inversion.pipeline import _number_to_mass_fractions


def test_two_element_hand_calc():
    """50/50 mole Fe+Si -> mass fractions by atomic weight (Fe 55.845, Si 28.0855)."""
    out = _number_to_mass_fractions({"Fe": 0.5, "Si": 0.5})
    # C_mass_Fe = 55.845 / (55.845 + 28.0855) = 0.66538...
    assert out["Fe"] == pytest.approx(55.845 / (55.845 + 28.0855), rel=1e-3)
    assert out["Si"] == pytest.approx(28.0855 / (55.845 + 28.0855), rel=1e-3)
    assert sum(out.values()) == pytest.approx(1.0, rel=1e-9)


def test_single_element_is_unity():
    assert _number_to_mass_fractions({"Fe": 1.0}) == {"Fe": pytest.approx(1.0)}


def test_all_zero_returns_zeros():
    assert _number_to_mass_fractions({"Fe": 0.0, "Si": 0.0}) == {"Fe": 0.0, "Si": 0.0}


def test_negative_clamped():
    out = _number_to_mass_fractions({"Fe": 1.0, "Si": -0.3})
    assert out["Si"] == 0.0 and out["Fe"] == pytest.approx(1.0)
