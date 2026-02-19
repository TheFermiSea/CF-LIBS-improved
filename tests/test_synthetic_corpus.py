"""Tests for synthetic corpus helper utilities."""

import numpy as np

from cflibs.benchmark.synthetic_corpus import (
    PerturbationAxes,
    default_recipes,
    distort_wavelength_axis,
    full_factorial_perturbations,
    mass_to_number_fractions,
)


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
