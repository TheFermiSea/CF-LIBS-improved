"""
Tests for ILR (Isometric Log-Ratio) closure implementation.

Tests cover the CLR/ILR transforms, the Helmert basis, and the
ClosureEquation.apply_ilr() method.
"""

import numpy as np
import pytest

from cflibs.inversion.closure import (
    ClosureEquation,
    ClosureMode,
    _helmert_basis,
    clr_transform,
    ilr_transform,
    ilr_inverse,
)

# ---------------------------------------------------------------------------
# Helmert basis tests
# ---------------------------------------------------------------------------


class TestHelmertBasis:
    """Tests for the Helmert sub-composition contrast matrix."""

    def test_shape(self):
        for D in (2, 3, 5, 10):
            V = _helmert_basis(D)
            assert V.shape == (D, D - 1)

    def test_orthonormality(self):
        """V^T V should be the identity matrix I_{D-1}."""
        for D in (2, 3, 5, 8):
            V = _helmert_basis(D)
            product = V.T @ V
            np.testing.assert_allclose(product, np.eye(D - 1), atol=1e-14)

    def test_columns_unit_norm(self):
        V = _helmert_basis(4)
        for j in range(3):
            np.testing.assert_allclose(np.linalg.norm(V[:, j]), 1.0, atol=1e-14)

    def test_d_less_than_2_raises(self):
        with pytest.raises(ValueError, match="D >= 2"):
            _helmert_basis(1)


# ---------------------------------------------------------------------------
# CLR transform tests
# ---------------------------------------------------------------------------


class TestCLRTransform:
    """Tests for the Centered Log-Ratio transform."""

    def test_clr_sums_to_zero(self):
        """CLR coordinates must sum to zero (centered)."""
        comp = np.array([0.7, 0.2, 0.1])
        clr = clr_transform(comp)
        np.testing.assert_allclose(np.sum(clr), 0.0, atol=1e-14)

    def test_clr_round_trip(self):
        """exp(clr) -> normalize should recover the original composition."""
        comp = np.array([0.5, 0.3, 0.15, 0.05])
        clr = clr_transform(comp)
        recovered = np.exp(clr)
        recovered = recovered / np.sum(recovered)
        np.testing.assert_allclose(recovered, comp, atol=1e-12)

    def test_clr_batch(self):
        """CLR should work on batches (N, D)."""
        comps = np.array([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]])
        clr = clr_transform(comps)
        assert clr.shape == (2, 3)
        np.testing.assert_allclose(np.sum(clr, axis=-1), 0.0, atol=1e-14)

    def test_clr_uniform_is_zero(self):
        """Uniform composition should give CLR = 0."""
        comp = np.array([0.25, 0.25, 0.25, 0.25])
        clr = clr_transform(comp)
        np.testing.assert_allclose(clr, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# ILR transform tests
# ---------------------------------------------------------------------------


class TestILRTransform:
    """Tests for the ILR transform and its inverse."""

    def test_ilr_output_dimension(self):
        comp = np.array([0.7, 0.2, 0.1])
        coords = ilr_transform(comp)
        assert coords.shape == (2,)

    def test_ilr_round_trip(self):
        """ilr_inverse(ilr_transform(x)) should recover x."""
        comp = np.array([0.7, 0.2, 0.1])
        coords = ilr_transform(comp)
        recovered = ilr_inverse(coords, 3)
        np.testing.assert_allclose(recovered, comp, atol=1e-12)

    def test_ilr_round_trip_many_elements(self):
        comp = np.array([0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02])
        coords = ilr_transform(comp)
        assert coords.shape == (6,)
        recovered = ilr_inverse(coords, 7)
        np.testing.assert_allclose(recovered, comp, atol=1e-12)

    def test_ilr_round_trip_two_elements(self):
        comp = np.array([0.8, 0.2])
        coords = ilr_transform(comp)
        assert coords.shape == (1,)
        recovered = ilr_inverse(coords, 2)
        np.testing.assert_allclose(recovered, comp, atol=1e-12)

    def test_ilr_batch_round_trip(self):
        comps = np.array([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
        coords = ilr_transform(comps)
        assert coords.shape == (3, 2)
        recovered = ilr_inverse(coords, 3)
        np.testing.assert_allclose(recovered, comps, atol=1e-12)

    def test_ilr_uniform_gives_zero(self):
        """Uniform composition maps to the origin in ILR space."""
        comp = np.array([1 / 3, 1 / 3, 1 / 3])
        coords = ilr_transform(comp)
        np.testing.assert_allclose(coords, 0.0, atol=1e-14)

    def test_ilr_inverse_produces_valid_simplex(self):
        """Random ILR coords should always map back to a valid simplex."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            coords = rng.standard_normal(4)
            comp = ilr_inverse(coords, 5)
            assert np.all(comp > 0), f"Negative component: {comp}"
            np.testing.assert_allclose(np.sum(comp), 1.0, atol=1e-14)

    def test_ilr_isometry(self):
        """ILR should preserve Aitchison distances (isometry)."""
        c1 = np.array([0.7, 0.2, 0.1])
        c2 = np.array([0.5, 0.3, 0.2])

        # Aitchison distance via CLR
        clr1 = clr_transform(c1)
        clr2 = clr_transform(c2)
        aitchison_dist = np.linalg.norm(clr1 - clr2)

        # ILR Euclidean distance
        z1 = ilr_transform(c1)
        z2 = ilr_transform(c2)
        ilr_dist = np.linalg.norm(z1 - z2)

        np.testing.assert_allclose(ilr_dist, aitchison_dist, atol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for near-boundary and extreme compositions."""

    def test_near_zero_component(self):
        """Very small component should survive the round trip."""
        comp = np.array([0.99, 0.009, 0.001])
        coords = ilr_transform(comp)
        recovered = ilr_inverse(coords, 3)
        np.testing.assert_allclose(recovered, comp, atol=1e-10)

    def test_single_dominant_element(self):
        comp = np.array([0.999, 0.0005, 0.0005])
        coords = ilr_transform(comp)
        recovered = ilr_inverse(coords, 3)
        np.testing.assert_allclose(recovered, comp, atol=1e-9)

    def test_very_small_clipped(self):
        """Components below the clipping threshold should still produce valid output."""
        comp = np.array([0.9999999, 1e-12, 1e-12])
        coords = ilr_transform(comp)
        recovered = ilr_inverse(coords, 3)
        assert np.all(recovered > 0)
        np.testing.assert_allclose(np.sum(recovered), 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# ClosureEquation.apply_ilr tests
# ---------------------------------------------------------------------------


class TestApplyILR:
    """Tests for the ClosureEquation.apply_ilr() method."""

    def test_known_composition(self):
        """Fe=0.7, Cu=0.2, Al=0.1 should be recovered from intercepts."""
        # Build intercepts so that U_s * exp(q_s) gives the target ratios
        target = {"Fe": 0.7, "Cu": 0.2, "Al": 0.1}
        partition_funcs = {"Fe": 1.0, "Cu": 1.0, "Al": 1.0}
        intercepts = {el: np.log(c) for el, c in target.items()}

        result = ClosureEquation.apply_ilr(intercepts, partition_funcs)

        assert result.mode == "ilr"
        for el, expected in target.items():
            assert abs(result.concentrations[el] - expected) < 1e-12

    def test_sum_to_one(self):
        intercepts = {"Fe": 0.5, "Cu": -0.3, "Al": -1.2}
        partition_funcs = {"Fe": 2.0, "Cu": 1.5, "Al": 1.0}

        result = ClosureEquation.apply_ilr(intercepts, partition_funcs)

        total = sum(result.concentrations.values())
        np.testing.assert_allclose(total, 1.0, atol=1e-14)

    def test_all_positive(self):
        intercepts = {"Fe": 0.5, "Cu": -0.3, "Al": -1.2, "Si": -2.0}
        partition_funcs = {"Fe": 2.0, "Cu": 1.5, "Al": 1.0, "Si": 0.8}

        result = ClosureEquation.apply_ilr(intercepts, partition_funcs)

        for el, c in result.concentrations.items():
            assert c > 0, f"{el} has non-positive concentration {c}"

    def test_matches_standard_closure(self):
        """ILR closure should give the same result as standard for a single pass."""
        intercepts = {"Fe": 0.3, "Cu": -0.5, "Zn": -1.0}
        partition_funcs = {"Fe": 1.5, "Cu": 1.2, "Zn": 0.9}

        std = ClosureEquation.apply_standard(intercepts, partition_funcs)
        ilr = ClosureEquation.apply_ilr(intercepts, partition_funcs)

        for el in intercepts:
            np.testing.assert_allclose(
                ilr.concentrations[el],
                std.concentrations[el],
                atol=1e-12,
                err_msg=f"Mismatch for {el}",
            )

    def test_with_abundance_multipliers(self):
        intercepts = {"Fe": 0.3, "Cu": -0.5}
        partition_funcs = {"Fe": 1.5, "Cu": 1.2}
        multipliers = {"Fe": 2.0, "Cu": 0.5}

        result = ClosureEquation.apply_ilr(
            intercepts, partition_funcs, abundance_multipliers=multipliers
        )

        total = sum(result.concentrations.values())
        np.testing.assert_allclose(total, 1.0, atol=1e-14)
        assert all(c > 0 for c in result.concentrations.values())

    def test_missing_partition_function(self):
        """Elements without partition functions should be silently skipped."""
        intercepts = {"Fe": 0.3, "Cu": -0.5, "Zn": -1.0}
        partition_funcs = {"Fe": 1.5, "Cu": 1.2}

        result = ClosureEquation.apply_ilr(intercepts, partition_funcs)

        assert "Zn" not in result.concentrations
        assert len(result.concentrations) == 2
        np.testing.assert_allclose(sum(result.concentrations.values()), 1.0, atol=1e-14)

    def test_single_element_returns_empty(self):
        """ILR needs at least 2 elements."""
        intercepts = {"Fe": 0.3}
        partition_funcs = {"Fe": 1.5}

        result = ClosureEquation.apply_ilr(intercepts, partition_funcs)

        assert result.concentrations == {}
        assert result.mode == "ilr"

    def test_experimental_factor(self):
        intercepts = {"Fe": 0.3, "Cu": -0.5}
        partition_funcs = {"Fe": 1.5, "Cu": 1.2}

        result = ClosureEquation.apply_ilr(intercepts, partition_funcs)

        # F should equal the sum of raw relative concentrations
        expected_F = 1.5 * np.exp(0.3) + 1.2 * np.exp(-0.5)
        np.testing.assert_allclose(result.experimental_factor, expected_F, atol=1e-12)


# ---------------------------------------------------------------------------
# ClosureMode enum tests
# ---------------------------------------------------------------------------


class TestClosureMode:
    def test_enum_values(self):
        assert ClosureMode.STANDARD.value == "standard"
        assert ClosureMode.MATRIX.value == "matrix"
        assert ClosureMode.OXIDE.value == "oxide"
        assert ClosureMode.ILR.value == "ilr"

    def test_enum_from_string(self):
        assert ClosureMode("ilr") is ClosureMode.ILR
