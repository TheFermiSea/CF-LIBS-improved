"""Numerical-agreement tests for :func:`nnls_jax` and
:func:`nnls_jax_batch` against :func:`scipy.optimize.nnls`.

The JAX path uses FISTA on the Gram form; scipy uses Lawson--Hanson
active set. On rank-deficient problems these can converge to different
NNLS minimizers (multiple feasible vertices have the same residual), so
tests assert agreement on **residual norm** with tight rtol and only
check the coefficient vector when the problem is well-conditioned.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import nnls as scipy_nnls

from cflibs.inversion.identify.spectral_nnls import (
    _HAS_JAX,
    nnls_jax,
    nnls_jax_batch,
)

pytestmark = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")


# ---------------------------------------------------------------------------
# Single-spectrum nnls_jax
# ---------------------------------------------------------------------------


@pytest.fixture
def well_conditioned_problem():
    """A well-conditioned NNLS problem with a unique minimizer."""
    rng = np.random.default_rng(42)
    m, n = 300, 20
    A = rng.normal(size=(m, n))
    x_true = np.maximum(rng.normal(size=n), 0.0)
    b = A @ x_true + 0.005 * rng.normal(size=m)
    return A, b, x_true


def test_nnls_jax_residual_norm_matches_scipy(well_conditioned_problem):
    A, b, _ = well_conditioned_problem
    x_sp, r_sp = scipy_nnls(A, b)
    x_jx, r_jx = nnls_jax(A, b, max_iter=500)
    # Residual norm is the invariant -- multiple NNLS minimizers can
    # share the same residual but differ in x.
    assert r_jx == pytest.approx(r_sp, rel=1e-5)


def test_nnls_jax_coefficients_close_well_conditioned(well_conditioned_problem):
    A, b, _ = well_conditioned_problem
    x_sp, _ = scipy_nnls(A, b)
    x_jx, _ = nnls_jax(A, b, max_iter=500)
    # Coefficient match is sensible only when the minimizer is unique
    # (well-conditioned case here).
    assert np.linalg.norm(x_jx - x_sp) / max(np.linalg.norm(x_sp), 1e-30) < 1e-4


def test_nnls_jax_solution_is_non_negative(well_conditioned_problem):
    A, b, _ = well_conditioned_problem
    x_jx, _ = nnls_jax(A, b, max_iter=500)
    assert np.all(x_jx >= -1e-12), f"Negative coefficient: min={x_jx.min()}"


def test_nnls_jax_zero_input():
    """All-zero RHS should yield all-zero solution."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 5))
    b = np.zeros(50)
    x_jx, r_jx = nnls_jax(A, b)
    assert np.all(x_jx == 0.0)
    assert r_jx == pytest.approx(0.0, abs=1e-10)


def test_nnls_jax_diagnostics_return():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(100, 10))
    b = A @ np.abs(rng.normal(size=10)) + 0.01 * rng.normal(size=100)
    out = nnls_jax(A, b, max_iter=500, return_diagnostics=True)
    assert len(out) == 3
    x, r, kkt = out
    assert kkt >= 0.0
    # KKT violation should be small at convergence
    assert kkt < 1e-3


def test_nnls_jax_shape_validation():
    with pytest.raises(ValueError, match="A must be 2-D"):
        nnls_jax(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="b must be 1-D"):
        nnls_jax(np.zeros((5, 3)), np.zeros((5, 1)))
    with pytest.raises(ValueError, match="does not match"):
        nnls_jax(np.zeros((5, 3)), np.zeros(7))


def test_nnls_jax_ill_conditioned_matches_residual():
    """Even when A is rank-deficient (cond ~1e8), residual must match
    even though coefficient vectors may differ."""
    rng = np.random.default_rng(3)
    m, n = 200, 15
    U = rng.normal(size=(m, n))
    # Inject near-collinearity: two columns nearly identical
    U[:, 1] = U[:, 0] + 1e-7 * rng.normal(size=m)
    x_true = np.abs(rng.normal(size=n))
    b = U @ x_true + 0.001 * rng.normal(size=m)
    _, r_sp = scipy_nnls(U, b)
    _, r_jx = nnls_jax(U, b, max_iter=1000)
    # Allow looser tolerance for ill-conditioned problems
    assert r_jx == pytest.approx(r_sp, rel=1e-3)


# ---------------------------------------------------------------------------
# Batched nnls_jax_batch
# ---------------------------------------------------------------------------


@pytest.fixture
def batched_problem():
    """A basis-matrix-style batched problem (CF-LIBS convention: rows =
    components, cols = pixels). Includes a leave-one-out mask family."""
    rng = np.random.default_rng(11)
    n_pixels = 400
    n_comp = 12
    A = np.abs(rng.normal(size=(n_comp, n_pixels))) * 0.01 + 1e-4
    x_true = np.zeros(n_comp)
    x_true[[0, 2, 5, 7]] = [1.0, 0.5, 0.3, 0.7]
    b = A.T @ x_true + 0.001 * rng.normal(size=n_pixels)
    # Build masks: all-on, then leave-one-out
    masks = np.vstack(
        [np.ones((1, n_comp), dtype=bool), ~np.eye(n_comp, dtype=bool)]
    )
    return A, b, masks, x_true


def test_nnls_jax_batch_shapes(batched_problem):
    A, b, masks, _ = batched_problem
    X, R = nnls_jax_batch(A, b, masks, max_iter=500)
    assert X.shape == masks.shape
    assert R.shape == (masks.shape[0],)


def test_nnls_jax_batch_masked_entries_are_zero(batched_problem):
    A, b, masks, _ = batched_problem
    X, _ = nnls_jax_batch(A, b, masks, max_iter=500)
    # Wherever the mask is False, the coefficient must be exactly 0.
    inactive = ~masks
    assert np.all(X[inactive] == 0.0)


def test_nnls_jax_batch_first_row_matches_full_solve(batched_problem):
    """The first batch row has mask=all-on; its solution should match
    a single nnls_jax call on the full system."""
    A, b, masks, _ = batched_problem
    X, R = nnls_jax_batch(A, b, masks, max_iter=500)
    x_full, r_full = nnls_jax(A.T, b, max_iter=500)
    assert np.allclose(X[0], x_full, rtol=1e-6, atol=1e-10)
    assert R[0] == pytest.approx(r_full, rel=1e-8)


def test_nnls_jax_batch_residuals_match_scipy(batched_problem):
    """For each batch entry, the residual norm should match scipy on the
    corresponding row-restricted problem."""
    A, b, masks, _ = batched_problem
    X, R = nnls_jax_batch(A, b, masks, max_iter=500)
    for k in range(masks.shape[0]):
        A_sub = A[masks[k], :].T  # (n_pix, n_active)
        x_sp, r_sp = scipy_nnls(A_sub, b)
        assert R[k] == pytest.approx(r_sp, rel=1e-4), (
            f"batch {k}: jax residual {R[k]} vs scipy {r_sp}"
        )


def test_nnls_jax_batch_shape_validation():
    A = np.zeros((5, 100))
    b = np.zeros(100)
    bad_masks = np.zeros((3, 4), dtype=bool)  # n_comp mismatch
    with pytest.raises(ValueError, match="row_masks shape"):
        nnls_jax_batch(A, b, bad_masks)

    with pytest.raises(ValueError, match="b shape"):
        nnls_jax_batch(A, np.zeros(50), np.zeros((3, 5), dtype=bool))


def test_nnls_jax_batch_single_batch_entry():
    """Edge case: batch dimension of 1."""
    rng = np.random.default_rng(0)
    A = np.abs(rng.normal(size=(8, 200))) * 0.01
    b = A.T @ np.abs(rng.normal(size=8)) + 0.001 * rng.normal(size=200)
    masks = np.ones((1, 8), dtype=bool)
    X, R = nnls_jax_batch(A, b, masks, max_iter=500)
    assert X.shape == (1, 8)
    assert R.shape == (1,)


# ---------------------------------------------------------------------------
# Determinism + jit caching
# ---------------------------------------------------------------------------


def test_nnls_jax_deterministic():
    """Two calls with the same input must return bit-identical output."""
    rng = np.random.default_rng(99)
    A = rng.normal(size=(150, 12))
    b = A @ np.abs(rng.normal(size=12)) + 0.01 * rng.normal(size=150)
    x1, r1 = nnls_jax(A, b, max_iter=500)
    x2, r2 = nnls_jax(A, b, max_iter=500)
    assert np.array_equal(x1, x2)
    assert r1 == r2
