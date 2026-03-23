"""
Tests for the JAX softmax closure module.

Tests cover sum-to-one, positivity, round-trip, limiting cases (D=1, D=2,
uniform), Jacobian correctness, numerical stability, gradient flow, and
batch consistency.

Convention: theta [dimensionless], C_i [dimensionless], sum(C_i) = 1.
"""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from cflibs.inversion.softmax_closure import (  # noqa: E402
    softmax_closure,
    inverse_softmax,
    softmax_jacobian,
)

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _random_theta(D: int, seed: int = 42) -> jnp.ndarray:
    """Generate a random theta vector of dimension D."""
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, shape=(D,), dtype=jnp.float64)


def _random_composition(D: int, seed: int = 42) -> jnp.ndarray:
    """Generate a random valid composition (positive, sums to 1)."""
    key = jax.random.PRNGKey(seed)
    raw = jax.random.uniform(key, shape=(D,), dtype=jnp.float64, minval=0.01, maxval=1.0)
    return raw / jnp.sum(raw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSumToOne:
    """test_sum_to_one: 100 random theta, verify abs(sum(C) - 1) < 1e-15."""

    def test_many_random(self):
        for seed in range(100):
            D = (seed % 14) + 2  # D from 2 to 15
            theta = _random_theta(D, seed=seed)
            C = softmax_closure(theta)
            err = abs(float(jnp.sum(C)) - 1.0)
            assert err < 1e-15, f"Sum error {err} at seed={seed}, D={D}"


class TestPositivity:
    """test_positivity: all C_i > 0 for all random theta."""

    def test_all_positive(self):
        for seed in range(100):
            D = (seed % 14) + 2
            theta = _random_theta(D, seed=seed)
            C = softmax_closure(theta)
            assert jnp.all(C > 0), f"Non-positive C at seed={seed}: {C}"


class TestRoundTrip:
    """test_round_trip: softmax(inverse_softmax(C)) recovers C to <1e-14."""

    def test_round_trip(self):
        for seed in range(50):
            D = (seed % 14) + 2
            C_orig = _random_composition(D, seed=seed)
            theta = inverse_softmax(C_orig)
            C_recovered = softmax_closure(theta)
            err = float(jnp.max(jnp.abs(C_recovered - C_orig)))
            assert err < 1e-14, f"Round-trip error {err} at seed={seed}, D={D}"


class TestUniform:
    """test_uniform: theta = [0,...,0] gives C_i = 1/D for D=2..15."""

    def test_uniform_theta(self):
        for D in range(2, 16):
            theta = jnp.zeros(D, dtype=jnp.float64)
            C = softmax_closure(theta)
            expected = jnp.full(D, 1.0 / D, dtype=jnp.float64)
            err = float(jnp.max(jnp.abs(C - expected)))
            assert err < 1e-15, f"Uniform error {err} at D={D}"


class TestD1Trivial:
    """test_d1_trivial: D=1 gives C=[1.0]."""

    def test_scalar(self):
        for val in [-10.0, 0.0, 5.0, 100.0]:
            theta = jnp.array([val], dtype=jnp.float64)
            C = softmax_closure(theta)
            assert abs(float(C[0]) - 1.0) < 1e-15, f"D=1 failed for theta={val}: C={C}"


class TestD2Sigmoid:
    """test_d2_sigmoid: D=2, compare against jax.nn.sigmoid."""

    def test_sigmoid_equivalence(self):
        for delta in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            theta = jnp.array([delta, 0.0], dtype=jnp.float64)
            C = softmax_closure(theta)
            expected_0 = jax.nn.sigmoid(jnp.float64(delta))
            err = abs(float(C[0]) - float(expected_0))
            assert err < 1e-14, f"Sigmoid mismatch for delta={delta}: C[0]={C[0]}, sig={expected_0}"


class TestJacobianAnalytical:
    """test_jacobian_analytical: analytical vs autodiff Jacobian, max error < 1e-12."""

    def test_jacobian_matches_autodiff(self):
        for seed in range(20):
            D = (seed % 9) + 2  # D from 2 to 10
            theta = _random_theta(D, seed=seed)

            J_analytical = softmax_jacobian(theta)
            J_autodiff = jax.jacobian(softmax_closure)(theta)

            err = float(jnp.max(jnp.abs(J_analytical - J_autodiff)))
            assert err < 1e-12, f"Jacobian error {err} at seed={seed}, D={D}"


class TestJacobianRank:
    """test_jacobian_rank: rank(J) = D-1 for D=2..10."""

    def test_rank(self):
        for D in range(2, 11):
            theta = _random_theta(D, seed=D)
            J = softmax_jacobian(theta)
            sv = jnp.linalg.svd(J, compute_uv=False)
            # The smallest singular value should be near zero (rank D-1)
            assert float(sv[-1]) < 1e-10 * float(sv[0]), f"Jacobian rank not D-1 at D={D}: sv={sv}"
            # Second-smallest should be well away from zero
            if D > 2:
                assert float(sv[-2]) > 1e-10 * float(
                    sv[0]
                ), f"Unexpected low rank at D={D}: sv={sv}"


class TestGradientFlow:
    """test_gradient_flow: grad of loss through softmax produces finite nonzero gradients."""

    def test_gradient_finite_nonzero(self):
        for D in [2, 5, 10]:
            theta = _random_theta(D, seed=D)
            C_target = _random_composition(D, seed=D + 100)

            def loss(theta: jnp.ndarray) -> jnp.ndarray:
                C = softmax_closure(theta)
                return jnp.sum((C - C_target) ** 2)

            grad = jax.grad(loss)(theta)
            assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradient at D={D}: {grad}"
            assert float(jnp.max(jnp.abs(grad))) > 1e-30, f"Zero gradient at D={D}: {grad}"


class TestBatchConsistency:
    """test_batch_consistency: batched (B, D) same results as sequential (D,) calls."""

    def test_batch_vs_sequential(self):
        B, D = 8, 5
        key = jax.random.PRNGKey(0)
        theta_batch = jax.random.normal(key, shape=(B, D), dtype=jnp.float64)

        C_batch = softmax_closure(theta_batch)

        for i in range(B):
            C_single = softmax_closure(theta_batch[i])
            err = float(jnp.max(jnp.abs(C_batch[i] - C_single)))
            assert err < 1e-15, f"Batch mismatch at i={i}: err={err}"


class TestNumericalStability:
    """test_numerical_stability: theta = [500, -500, 0] -- no NaN/Inf."""

    def test_extreme_theta(self):
        theta = jnp.array([500.0, -500.0, 0.0], dtype=jnp.float64)
        C = softmax_closure(theta)
        assert jnp.all(jnp.isfinite(C)), f"Non-finite C: {C}"
        assert abs(float(jnp.sum(C)) - 1.0) < 1e-15
        # First element should be ~1, others ~0
        assert float(C[0]) > 0.99


class TestIntegrationWithLoss:
    """test_integration_with_loss: full round-trip theta -> softmax -> loss -> grad."""

    def test_full_pipeline(self):
        D = 6
        theta = _random_theta(D, seed=99)
        C_target = _random_composition(D, seed=100)

        def pipeline_loss(theta: jnp.ndarray) -> jnp.ndarray:
            C = softmax_closure(theta)
            return jnp.sum((C - C_target) ** 2)

        # Forward
        loss_val = pipeline_loss(theta)
        assert jnp.isfinite(loss_val)
        assert float(loss_val) > 0

        # Backward
        grad = jax.grad(pipeline_loss)(theta)
        assert jnp.all(jnp.isfinite(grad))

        # Gradient descent step should reduce loss
        theta_new = theta - 0.1 * grad
        loss_new = pipeline_loss(theta_new)
        assert float(loss_new) < float(
            loss_val
        ), f"GD step did not reduce loss: {loss_val} -> {loss_new}"
