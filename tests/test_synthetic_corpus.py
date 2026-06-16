"""Tests for synthetic corpus helper utilities."""

import numpy as np

from cflibs.benchmark.synthetic_corpus import (
    PerturbationAxes,
    default_recipes,
    diagnostic_recipes,
    distort_wavelength_axis,
    full_factorial_perturbations,
    mass_to_number_fractions,
)

DIAGNOSTIC_PANEL = ["Al", "Co", "Cr", "Cu", "Fe", "Mg", "Mn", "Ni", "Si", "Ti", "V"]


def test_mass_to_number_fractions_normalized_and_ordered():
    fractions = mass_to_number_fractions({"Fe": 0.5, "Cu": 0.5})
    assert np.isclose(sum(fractions.values()), 1.0, rtol=1e-12)
    # Equal mass fraction -> lighter element (Fe) has higher number fraction.
    assert fractions["Fe"] > fractions["Cu"]


def test_distort_wavelength_axis_is_monotonic():
    wl = np.linspace(224.6, 265.3, 2560)
    warped = distort_wavelength_axis(wl, shift_nm=0.8, warp_quadratic_nm=0.15)
    assert warped.shape == wl.shape
    assert np.all(np.diff(warped) > 0)
    # Check global shift remains close to requested (quadratic component is centered)
    assert np.isclose(np.mean(warped - wl), 0.8, atol=1e-6)


def test_full_factorial_perturbation_count():
    axes = PerturbationAxes(
        snr_db=[20.0, 30.0],
        continuum_level=[0.0, 0.03],
        resolving_power=[700.0],
        shift_nm=[-1.0, 0.0, 1.0],
        warp_quadratic_nm=[0.0, 0.15],
    )
    combos = list(full_factorial_perturbations(axes))
    assert len(combos) == 2 * 2 * 1 * 3 * 2


def test_default_recipes_filter_to_candidates():
    recipes = default_recipes(["Fe", "Ni"])
    assert recipes
    for recipe in recipes:
        assert set(recipe.mass_fractions.keys()).issubset({"Fe", "Ni"})
        assert np.isclose(sum(recipe.mass_fractions.values()), 1.0)


def test_default_recipes_empty_for_disjoint_candidates():
    recipes = default_recipes(["Zn", "Pb"])
    assert recipes == []


def test_diagnostic_recipes_normalized_and_within_panel():
    recipes = diagnostic_recipes(DIAGNOSTIC_PANEL)
    assert recipes
    for recipe in recipes:
        assert set(recipe.mass_fractions.keys()).issubset(set(DIAGNOSTIC_PANEL))
        assert np.isclose(sum(recipe.mass_fractions.values()), 1.0)


def test_diagnostic_recipes_full_set_has_23_recipes():
    # 11 pure controls + 1 Fe/Ni binary + 11 alloy matrices.
    recipes = diagnostic_recipes(DIAGNOSTIC_PANEL)
    assert len(recipes) == 23
    names = {r.name for r in recipes}
    for el in DIAGNOSTIC_PANEL:
        assert f"pure_{el}" in names
    assert "binary_Fe_Ni" in names
    assert "co_alloy_stellite" in names


def test_diagnostic_recipes_every_panel_element_present_in_at_least_four():
    # The whole point of the diagnostic set: no element left unexercised
    # (the ak3.1.3 hygiene confound where Al/Co/Cu/Mg/Si/Ti/V never appeared).
    recipes = diagnostic_recipes(DIAGNOSTIC_PANEL)
    counts = {el: 0 for el in DIAGNOSTIC_PANEL}
    for recipe in recipes:
        for el, w in recipe.mass_fractions.items():
            if w > 0:
                counts[el] += 1
    for el in DIAGNOSTIC_PANEL:
        assert counts[el] >= 4, f"{el} appears in only {counts[el]} recipes (expected >=4)"


def test_diagnostic_recipes_filter_and_renormalize():
    # Restricting to a sub-panel drops out-of-panel elements and renormalizes.
    recipes = diagnostic_recipes(["Fe", "Ni"])
    assert recipes
    for recipe in recipes:
        assert set(recipe.mass_fractions.keys()).issubset({"Fe", "Ni"})
        assert np.isclose(sum(recipe.mass_fractions.values()), 1.0)
    names = {r.name for r in recipes}
    # Pure Fe / Ni and the Fe/Ni binary survive; pure_Al etc. drop out.
    assert "pure_Fe" in names
    assert "pure_Ni" in names
    assert "pure_Al" not in names
