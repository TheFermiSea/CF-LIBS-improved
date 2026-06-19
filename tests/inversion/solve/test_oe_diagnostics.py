"""Tests for Rodgers optimal-estimation closure diagnostics.

These validate :func:`cflibs.inversion.solve.joint_optimizer.compute_oe_diagnostics`
against the analytic linear-Gaussian posterior. They are pure-NumPy and require
neither JAX nor the atomic database.

Reference
---------
Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding: Theory and
Practice*. World Scientific. Eqs. 2.27, 2.31; Sec. 2.4.
"""

import numpy as np
import pytest

from cflibs.inversion.solve.joint_optimizer import (
    OEDiagnostics,
    compute_oe_diagnostics,
)

pytestmark = pytest.mark.unit


def _random_spd(n: int, seed: int, scale: float = 1.0) -> np.ndarray:
    """Generate a random symmetric positive-definite matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return scale * (A @ A.T + n * np.eye(n))


def test_posterior_covariance_matches_analytic_with_prior():
    """S_hat equals the analytic linear-Gaussian posterior (Rodgers Eq. 2.27)."""
    rng = np.random.default_rng(0)
    m, n = 12, 4
    K = rng.standard_normal((m, n))
    S_e = _random_spd(m, seed=1, scale=0.5)
    S_a = _random_spd(n, seed=2, scale=2.0)

    diag = compute_oe_diagnostics(K, S_e, S_a)

    # Analytic posterior covariance built independently with explicit inverses.
    fisher = K.T @ np.linalg.inv(S_e) @ K
    expected_S_hat = np.linalg.inv(fisher + np.linalg.inv(S_a))

    assert isinstance(diag, OEDiagnostics)
    assert diag.has_prior is True
    assert diag.posterior_covariance.shape == (n, n)
    np.testing.assert_allclose(diag.posterior_covariance, expected_S_hat, rtol=1e-9, atol=1e-12)

    # Posterior covariance must be symmetric positive-definite.
    np.testing.assert_allclose(
        diag.posterior_covariance, diag.posterior_covariance.T, rtol=1e-10, atol=1e-12
    )
    assert np.all(np.linalg.eigvalsh(diag.posterior_covariance) > 0)


def test_averaging_kernel_and_dofs_with_prior():
    """A = S_hat K^T S_e^-1 K, DOFS = trace(A), 0 <= DOFS <= n_params."""
    rng = np.random.default_rng(3)
    m, n = 20, 5
    K = rng.standard_normal((m, n))
    S_e = _random_spd(m, seed=4, scale=1.0)
    S_a = _random_spd(n, seed=5, scale=3.0)

    diag = compute_oe_diagnostics(K, S_e, S_a)

    fisher = K.T @ np.linalg.inv(S_e) @ K
    expected_A = diag.posterior_covariance @ fisher

    np.testing.assert_allclose(diag.averaging_kernel, expected_A, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(diag.dofs, np.trace(expected_A), rtol=1e-12, atol=1e-12)

    # DOFS is bounded by the number of parameters.
    assert 0.0 <= diag.dofs <= n + 1e-9


def test_well_constrained_limit_kernel_rows_sum_toward_one():
    """In the well-constrained limit A -> I, so rows sum toward 1 and DOFS -> n."""
    rng = np.random.default_rng(6)
    m, n = 40, 3
    K = rng.standard_normal((m, n))

    # Tiny measurement noise + very broad prior => information matrix dominates
    # the prior precision, driving A toward the identity (Rodgers 2000, Sec. 2.4).
    S_e = 1e-6 * np.eye(m)
    S_a = 1e6 * np.eye(n)

    diag = compute_oe_diagnostics(K, S_e, S_a)

    # Averaging kernel approaches identity; row sums approach 1.
    np.testing.assert_allclose(diag.averaging_kernel, np.eye(n), rtol=0, atol=1e-3)
    row_sums = diag.averaging_kernel.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(n), rtol=0, atol=1e-3)

    # DOFS approaches n_params in the well-constrained limit.
    assert abs(diag.dofs - n) < 1e-3


def test_gauss_newton_fallback_without_prior():
    """Without S_a, S_hat = (K^T S_e^-1 K)^-1, A = I, DOFS = n_params."""
    rng = np.random.default_rng(7)
    m, n = 15, 4
    K = rng.standard_normal((m, n))
    S_e = _random_spd(m, seed=8, scale=0.7)

    diag = compute_oe_diagnostics(K, S_e, prior_covariance=None)

    assert diag.has_prior is False

    expected_S_hat = np.linalg.inv(K.T @ np.linalg.inv(S_e) @ K)
    np.testing.assert_allclose(diag.posterior_covariance, expected_S_hat, rtol=1e-9, atol=1e-12)

    # With no prior the retrieval is fully data-driven: A = I exactly.
    np.testing.assert_allclose(diag.averaging_kernel, np.eye(n), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(diag.dofs, float(n), rtol=0, atol=1e-9)


def test_diagonal_covariance_inputs_accepted():
    """1-D measurement/prior covariances are treated as diagonals."""
    rng = np.random.default_rng(9)
    m, n = 10, 3
    K = rng.standard_normal((m, n))
    se_diag = rng.uniform(0.1, 1.0, size=m)
    sa_diag = rng.uniform(1.0, 5.0, size=n)

    diag_input = compute_oe_diagnostics(K, se_diag, sa_diag)
    full_input = compute_oe_diagnostics(K, np.diag(se_diag), np.diag(sa_diag))

    np.testing.assert_allclose(
        diag_input.posterior_covariance, full_input.posterior_covariance, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        diag_input.averaging_kernel, full_input.averaging_kernel, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(diag_input.dofs, full_input.dofs, rtol=1e-12, atol=1e-14)


def test_shape_validation_errors():
    """Inconsistent shapes raise ValueError."""
    K = np.ones((6, 3))

    with pytest.raises(ValueError):
        compute_oe_diagnostics(np.ones(6), np.eye(6))  # 1-D jacobian

    with pytest.raises(ValueError):
        compute_oe_diagnostics(K, np.eye(5))  # S_e rows != m

    with pytest.raises(ValueError):
        compute_oe_diagnostics(K, np.ones(5))  # diagonal S_e length != m

    with pytest.raises(ValueError):
        compute_oe_diagnostics(K, np.eye(6), np.eye(4))  # S_a != (n, n)

    with pytest.raises(ValueError):
        compute_oe_diagnostics(K, np.eye(6), np.ones(4))  # diagonal S_a length != n


def test_partial_information_intermediate_dofs():
    """A rank-deficient Jacobian + prior yields 0 < DOFS < n_params."""
    # Jacobian only constrains 2 of 3 parameters (third column is zero).
    K = np.zeros((10, 3))
    K[:, 0] = np.linspace(1.0, 2.0, 10)
    K[:, 1] = np.linspace(-1.0, 1.0, 10)
    S_e = 1e-3 * np.eye(10)
    S_a = np.eye(3)  # finite prior keeps S_hat well-defined on the null direction

    diag = compute_oe_diagnostics(K, S_e, S_a)

    # Two well-measured directions ~ DOFS near 2, third direction prior-dominated.
    assert 1.5 < diag.dofs < 3.0
    assert diag.dofs <= 3.0 + 1e-9
    # The unconstrained parameter's averaging-kernel row should be ~0.
    assert abs(diag.averaging_kernel[2, 2]) < 1e-2


@pytest.mark.requires_jax
@pytest.mark.integration
def test_joint_optimizer_surfaces_oe_diagnostics():
    """JointOptimizer.optimize() surfaces OE diagnostics without altering the point estimate."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: F401

    from cflibs.inversion.solve.joint_optimizer import (
        JointOptimizationResult,
        JointOptimizer,
        LossType,
        create_simple_forward_model,
    )

    elements = ["Fe", "Cu"]
    fm = create_simple_forward_model(
        elements,
        {"Fe": [400.0, 430.0], "Cu": [500.0, 510.0]},
        {"Fe": [1.0, 0.5], "Cu": [1.0, 0.6]},
    )
    wl = np.linspace(350.0, 600.0, 120)
    opt = JointOptimizer(fm, elements, wl, loss_type=LossType.CHI_SQUARED, max_iterations=80)

    spec = np.asarray(fm(1.5, 1e17, jax.numpy.array([0.7, 0.3]), jax.numpy.array(wl)))
    sigma = np.sqrt(np.maximum(spec, 1.0))

    # A finite prior regularizes the (softmax-degenerate) information matrix.
    s_a = np.diag([0.5, 0.5] + [1.0] * opt.n_elements)
    res = opt.optimize(
        spec, uncertainties=sigma, initial_T_eV=1.4, initial_n_e=8e16, prior_covariance=s_a
    )

    assert isinstance(res, JointOptimizationResult)
    assert res.oe_diagnostics is not None
    assert res.oe_diagnostics.has_prior is True
    assert res.oe_diagnostics.posterior_covariance.shape == (opt.n_params, opt.n_params)
    assert res.oe_diagnostics.averaging_kernel.shape == (opt.n_params, opt.n_params)
    assert 0.0 <= res.oe_diagnostics.dofs <= opt.n_params + 1e-9
    # The convenience property mirrors the diagnostics object.
    assert res.dofs == res.oe_diagnostics.dofs

    # Additive guarantee: diagnostics do not change the retrieved state.
    res_no_diag = opt.optimize(spec, uncertainties=sigma, initial_T_eV=1.4, initial_n_e=8e16)
    assert abs(res.temperature_eV - res_no_diag.temperature_eV) < 1e-6
    assert abs(res.electron_density_cm3 - res_no_diag.electron_density_cm3) < 1.0
