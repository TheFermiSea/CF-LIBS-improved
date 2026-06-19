"""
Tests for ILR/Aitchison analytical uncertainty propagation.

Covers :func:`ilr_propagate_covariance` and
:func:`simplex_covariance_from_ilr`, which push a composition covariance
into full-rank ILR coordinates and back to the closure-consistent simplex.

References
----------
Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations for
Compositional Data Analysis." Mathematical Geology 35(3), 279-300.
Aitchison, J. (1986). "The Statistical Analysis of Compositional Data."
"""

import numpy as np
import pytest

from cflibs.inversion.physics.closure import (
    ilr_inverse,
    ilr_propagate_covariance,
    ilr_transform,
    simplex_covariance_from_ilr,
)


def _is_psd(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True if ``matrix`` is symmetric positive semidefinite."""
    sym = 0.5 * (matrix + matrix.T)
    eigvals = np.linalg.eigvalsh(sym)
    return bool(np.min(eigvals) >= -tol)


def _closed_covariance(rng: np.random.Generator, D: int) -> np.ndarray:
    """Build a symmetric, closure-consistent (rows/cols sum to 0) covariance.

    Constructed as ``G A A^T G`` with centering matrix ``G = I - (1/D) 1 1^T``,
    which is PSD of rank ``D - 1`` and annihilates the all-ones direction.
    """
    A = rng.standard_normal((D, D))
    G = np.eye(D) - np.ones((D, D)) / D
    cov = G @ (A @ A.T) @ G
    return 0.5 * (cov + cov.T)


# ---------------------------------------------------------------------------
# Round-trip identity of the underlying ILR transforms (sanity precondition)
# ---------------------------------------------------------------------------


class TestILRRoundTripPrecondition:
    """ilr(ilr_inv(z)) == z, the algebraic basis of the propagation."""

    def test_ilr_inverse_then_forward_recovers_coords(self):
        rng = np.random.default_rng(0)
        for D in (2, 3, 5, 8):
            z = rng.standard_normal(D - 1)
            comp = ilr_inverse(z, D)
            recovered = ilr_transform(comp)
            np.testing.assert_allclose(recovered, z, atol=1e-12)


# ---------------------------------------------------------------------------
# Forward propagation: simplex -> ILR coordinates
# ---------------------------------------------------------------------------


class TestForwardPropagation:
    """Tests for ilr_propagate_covariance."""

    def test_output_shape(self):
        comp = np.array([0.6, 0.3, 0.1])
        cov = _closed_covariance(np.random.default_rng(1), 3)
        sigma_z = ilr_propagate_covariance(comp, cov)
        assert sigma_z.shape == (2, 2)

    def test_ilr_covariance_is_full_rank(self):
        """A closed (rank D-1) simplex covariance maps to a full-rank (D-1)."""
        rng = np.random.default_rng(2)
        for D in (3, 5, 7):
            comp = rng.dirichlet(np.full(D, 3.0))
            cov = _closed_covariance(rng, D)
            sigma_z = ilr_propagate_covariance(comp, cov)
            assert np.linalg.matrix_rank(sigma_z, tol=1e-9) == D - 1

    def test_ilr_covariance_is_psd(self):
        rng = np.random.default_rng(3)
        comp = rng.dirichlet(np.full(5, 2.0))
        cov = _closed_covariance(rng, 5)
        sigma_z = ilr_propagate_covariance(comp, cov)
        assert _is_psd(sigma_z)

    def test_symmetric_output(self):
        rng = np.random.default_rng(4)
        comp = rng.dirichlet(np.full(4, 2.0))
        cov = _closed_covariance(rng, 4)
        sigma_z = ilr_propagate_covariance(comp, cov)
        np.testing.assert_allclose(sigma_z, sigma_z.T, atol=1e-14)

    def test_jacobian_matches_finite_difference(self):
        """Delta-method propagation must match a finite-difference Jacobian."""
        rng = np.random.default_rng(5)
        D = 4
        comp = rng.dirichlet(np.full(D, 4.0))
        cov = _closed_covariance(rng, D)

        # Numerical Jacobian of ilr() restricted to the simplex tangent space.
        eps = 1e-7
        J = np.zeros((D - 1, D))
        base = ilr_transform(comp)
        for k in range(D):
            pert = comp.copy()
            pert[k] += eps
            pert = pert / np.sum(pert)  # stay on the simplex
            J[:, k] = (ilr_transform(pert) - base) / eps
        sigma_fd = J @ cov @ J.T
        sigma_analytic = ilr_propagate_covariance(comp, cov)
        np.testing.assert_allclose(sigma_analytic, sigma_fd, rtol=1e-4, atol=1e-6)

    def test_rejects_bad_shapes(self):
        comp = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="shape"):
            ilr_propagate_covariance(comp, np.eye(3))
        with pytest.raises(ValueError, match="D >= 2"):
            ilr_propagate_covariance(np.array([1.0]), np.array([[1.0]]))


# ---------------------------------------------------------------------------
# Inverse propagation: ILR coordinates -> closure-consistent simplex
# ---------------------------------------------------------------------------


class TestInversePropagation:
    """Tests for simplex_covariance_from_ilr."""

    def test_output_shape(self):
        z = np.array([0.3, -0.5])
        sigma_z = np.array([[0.04, 0.01], [0.01, 0.09]])
        sigma_c = simplex_covariance_from_ilr(z, sigma_z)
        assert sigma_c.shape == (3, 3)

    def test_rows_and_columns_sum_to_zero(self):
        """Closure consistency: 1^T Sigma_c = 0 and Sigma_c 1 = 0."""
        rng = np.random.default_rng(6)
        for D in (3, 5, 8):
            z = rng.standard_normal(D - 1)
            B = rng.standard_normal((D - 1, D - 1))
            sigma_z = B @ B.T  # PSD, full rank
            sigma_c = simplex_covariance_from_ilr(z, sigma_z)
            np.testing.assert_allclose(sigma_c.sum(axis=0), 0.0, atol=1e-12)
            np.testing.assert_allclose(sigma_c.sum(axis=1), 0.0, atol=1e-12)

    def test_simplex_covariance_is_psd(self):
        rng = np.random.default_rng(7)
        z = rng.standard_normal(4)
        B = rng.standard_normal((4, 4))
        sigma_z = B @ B.T
        sigma_c = simplex_covariance_from_ilr(z, sigma_z)
        assert _is_psd(sigma_c)

    def test_simplex_covariance_rank_deficient(self):
        """The back-transformed simplex covariance has rank D-1 (singular)."""
        rng = np.random.default_rng(8)
        D = 6
        z = rng.standard_normal(D - 1)
        B = rng.standard_normal((D - 1, D - 1))
        sigma_z = B @ B.T
        sigma_c = simplex_covariance_from_ilr(z, sigma_z)
        assert np.linalg.matrix_rank(sigma_c, tol=1e-9) == D - 1


# ---------------------------------------------------------------------------
# Acceptance test: full round trip of the covariance through ILR coordinates
# ---------------------------------------------------------------------------


class TestAcceptanceRoundTrip:
    """End-to-end acceptance: forward then inverse covariance propagation."""

    def test_covariance_round_trip(self):
        """Sigma_z -> Sigma_c -> Sigma_z recovers the ILR covariance.

        Mapping a full-rank ILR covariance back to the simplex and forward
        again through the *same* expansion point must be the identity, because
        the two Jacobians are mutual (pseudo-)inverses on the tangent space.
        """
        rng = np.random.default_rng(9)
        for D in (3, 4, 6):
            z = rng.standard_normal(D - 1)
            B = rng.standard_normal((D - 1, D - 1))
            sigma_z = B @ B.T  # full-rank PSD ILR covariance

            comp = ilr_inverse(z, D)
            sigma_c = simplex_covariance_from_ilr(z, sigma_z)

            # Closure consistency of the back-transformed simplex covariance.
            np.testing.assert_allclose(sigma_c.sum(axis=0), 0.0, atol=1e-12)
            np.testing.assert_allclose(sigma_c.sum(axis=1), 0.0, atol=1e-12)
            assert _is_psd(sigma_c)

            # Forward propagation recovers the original full-rank ILR cov.
            sigma_z_rt = ilr_propagate_covariance(comp, sigma_c)
            assert sigma_z_rt.shape == (D - 1, D - 1)
            assert np.linalg.matrix_rank(sigma_z_rt, tol=1e-9) == D - 1
            np.testing.assert_allclose(sigma_z_rt, sigma_z, rtol=1e-9, atol=1e-9)

    def test_ilr_transform_round_trip_acceptance(self):
        """Explicit acceptance clause: ilr(ilr_inv(z)) == z."""
        rng = np.random.default_rng(10)
        z = rng.standard_normal(5)
        comp = ilr_inverse(z, 6)
        np.testing.assert_allclose(ilr_transform(comp), z, atol=1e-12)
