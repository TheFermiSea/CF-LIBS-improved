"""Tests for PWLR (pivot log-ratio) closure behavior."""

import numpy as np
import pytest

from cflibs.inversion.physics.closure import (
    ClosureEquation,
    LOGRATIO_CLIP_FLOOR,
    alr_inverse,
    alr_transform,
    clr_transform,
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


# ---------------------------------------------------------------------------
# True Hron 2012 isometric pivot coordinates (audit Family 4, bug d)
# ---------------------------------------------------------------------------


def test_plr_first_coordinate_matches_hron_oracle() -> None:
    """Validation gate (d): for x=[0.7,0.2,0.1], pivot=0,
    z_1 = sqrt(2/3) * ln(0.7 / sqrt(0.2*0.1)).

    The expected value is computed from the closed-form Hron 2012 pivot
    coordinate, independent of the implementation."""
    x = np.array([0.7, 0.2, 0.1])
    z = plr_transform(x, pivot_index=0)

    # Independent oracle: z_1 = sqrt((D-1)/D) * ln(x_1 / geomean(x_2, x_3))
    oracle_z1 = np.sqrt(2.0 / 3.0) * np.log(0.7 / np.sqrt(0.2 * 0.1))
    assert z.shape == (2,)
    np.testing.assert_allclose(z[0], oracle_z1, atol=1e-12)


def test_plr_general_pivot_coordinate_formula() -> None:
    """Each PLR coordinate equals the Hron balance against the geometric mean
    of the subsequent parts (verified for all coordinates, D=4)."""
    x = np.array([0.5, 0.25, 0.15, 0.10])
    z = plr_transform(x, pivot_index=0)
    D = 4
    logx = np.log(x)
    for i in range(1, D):  # 1-based balance index
        coef = np.sqrt((D - i) / (D - i + 1))
        geo = np.mean(logx[i:])
        expected = coef * (logx[i - 1] - geo)
        np.testing.assert_allclose(z[i - 1], expected, atol=1e-12)


def test_plr_is_isometric_preserves_aitchison_distance() -> None:
    """Validation gate (d): PLR Euclidean distance == Aitchison distance."""
    c1 = np.array([0.7, 0.2, 0.1])
    c2 = np.array([0.5, 0.3, 0.2])

    # Aitchison distance via CLR.
    aitchison = np.linalg.norm(clr_transform(c1) - clr_transform(c2))

    for pivot in (0, 1, 2):
        z1 = plr_transform(c1, pivot_index=pivot)
        z2 = plr_transform(c2, pivot_index=pivot)
        np.testing.assert_allclose(np.linalg.norm(z1 - z2), aitchison, atol=1e-12)


def test_plr_inverse_round_trip_all_pivots() -> None:
    rng = np.random.default_rng(7)
    for _ in range(25):
        raw = rng.uniform(0.01, 1.0, size=5)
        comp = raw / raw.sum()
        for pivot in range(5):
            coords = plr_transform(comp, pivot_index=pivot)
            recovered = plr_inverse(coords, D=5, pivot_index=pivot)
            np.testing.assert_allclose(recovered, comp, atol=1e-12)
            np.testing.assert_allclose(recovered.sum(), 1.0, atol=1e-14)


def test_alr_distinct_from_isometric_plr() -> None:
    """ALR is the plain log-ratio (not isometric); it must differ from the
    true Hron pivot coordinates and must NOT preserve Aitchison distance."""
    c1 = np.array([0.7, 0.2, 0.1])
    c2 = np.array([0.5, 0.3, 0.2])

    # ALR is the naive ln(x_j / x_pivot); first coordinate is ln(0.2/0.7).
    a = alr_transform(c1, pivot_index=0)
    np.testing.assert_allclose(a[0], np.log(0.2 / 0.7), atol=1e-12)

    # ALR differs from isometric PLR.
    z = plr_transform(c1, pivot_index=0)
    assert not np.allclose(a, z)

    # ALR distance does not equal Aitchison distance (non-isometric).
    aitchison = np.linalg.norm(clr_transform(c1) - clr_transform(c2))
    alr_dist = np.linalg.norm(alr_transform(c1, pivot_index=0) - alr_transform(c2, pivot_index=0))
    assert abs(alr_dist - aitchison) > 1e-3


def test_alr_round_trip() -> None:
    rng = np.random.default_rng(11)
    for _ in range(10):
        raw = rng.uniform(0.01, 1.0, size=4)
        comp = raw / raw.sum()
        for pivot in range(4):
            coords = alr_transform(comp, pivot_index=pivot)
            recovered = alr_inverse(coords, D=4, pivot_index=pivot)
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
