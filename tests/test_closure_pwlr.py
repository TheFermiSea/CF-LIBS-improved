"""Tests for PWLR (pairwise/pivot log-ratio) closure behavior."""

import numpy as np
import pytest

from cflibs.inversion.physics.closure import (
    ClosureEquation,
    LOGRATIO_CLIP_FLOOR,
    plr_inverse,
    plr_transform,
)


def _intercepts_from_composition(
    composition: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    partition_funcs = {el: 1.0 for el in composition}
    intercepts = {el: np.log(max(val, LOGRATIO_CLIP_FLOOR)) for el, val in composition.items()}
    return intercepts, partition_funcs


@pytest.mark.parametrize(
    ("comp", "pivot_index"),
    [
        (np.array([0.62, 0.23, 0.1, 0.05]), 2),
        (np.array([0.26, 0.25, 0.24, 0.25]), 0),
        (np.array([0.97, 0.02, 0.009, 0.001]), 1),
    ],
)
def test_pwlr_round_trip_identity_sanity(comp: np.ndarray, pivot_index: int) -> None:
    coords = plr_transform(comp, pivot_index=pivot_index)
    recovered = plr_inverse(coords, D=4, pivot_index=pivot_index)
    np.testing.assert_allclose(recovered, comp, atol=1e-12)


def test_pwlr_aalto_suite_matches_ilr_closely() -> None:
    # Representative Aalto-like mineral compositions (major + trace elements)
    aalto_suite = [
        {"Si": 0.47, "Al": 0.21, "Fe": 0.17, "Ca": 0.11, "Mg": 0.04},
        {"Fe": 0.55, "Si": 0.18, "Al": 0.12, "Ca": 0.09, "Mn": 0.06},
        {"Ca": 0.41, "Si": 0.24, "Al": 0.14, "Mg": 0.12, "Fe": 0.09},
        {"Mg": 0.36, "Si": 0.3, "Fe": 0.16, "Al": 0.11, "Ca": 0.07},
    ]

    for composition in aalto_suite:
        intercepts, partition_funcs = _intercepts_from_composition(composition)
        ilr = ClosureEquation.apply_ilr(intercepts, partition_funcs)
        pwlr = ClosureEquation.apply_pwlr(intercepts, partition_funcs)

        for element, ilr_value in ilr.concentrations.items():
            rel_err = abs(pwlr.concentrations[element] - ilr_value) / max(ilr_value, 1e-12)
            assert rel_err < 0.01


def test_pwlr_handles_compositional_zeros_without_offset() -> None:
    intercepts = {"Fe": np.log(0.999998), "Cu": np.log(2e-6), "Zn": np.log(1e-20)}
    partition_funcs = {"Fe": 1.0, "Cu": 1.0, "Zn": 1.0}

    result = ClosureEquation.apply_pwlr(intercepts, partition_funcs)

    assert result.mode == "pwlr"
    assert all(np.isfinite(v) and v > 0.0 for v in result.concentrations.values())
    np.testing.assert_allclose(sum(result.concentrations.values()), 1.0, atol=1e-12)
    assert result.concentrations["Zn"] < 1e-6


def test_pwlr_diverges_from_ilr_near_zero() -> None:
    composition = {"Fe": 0.999995, "Cu": 4e-6, "Zn": 1e-6}
    intercepts, partition_funcs = _intercepts_from_composition(composition)

    ilr = ClosureEquation.apply_ilr(intercepts, partition_funcs)
    pwlr = ClosureEquation.apply_pwlr(
        intercepts,
        partition_funcs,
        regularization_strength=5e-2,
    )

    deltas = [abs(pwlr.concentrations[el] - ilr.concentrations[el]) for el in ilr.concentrations]
    assert max(deltas) > 1e-4
    np.testing.assert_allclose(sum(pwlr.concentrations.values()), 1.0, atol=1e-12)
    assert all(v > 0.0 for v in pwlr.concentrations.values())
