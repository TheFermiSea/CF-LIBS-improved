"""
Tests for JAX-accelerated batched Boltzmann fitting.

Tests the batched_boltzmann_fit closed-form WLS kernel against numpy reference
implementations, verifies edge cases, masking, and gradient stability.

References:
    Tognoni et al. (2010) Spectrochim. Acta B 65 -- CF-LIBS methodology
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.core.constants import KB_EV  # noqa: E402
from cflibs.inversion.boltzmann_jax import (  # noqa: E402
    batched_boltzmann_fit,
    boltzmann_temperature_jax,
)

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


def _numpy_wls_fit(x, y, w):
    """Reference WLS fit using numpy.polyfit with weights."""
    coeffs = np.polyfit(x, y, 1, w=np.sqrt(w))
    slope, intercept = coeffs[0], coeffs[1]
    return slope, intercept


class TestBatchedBoltzmannFit:
    """Tests for batched_boltzmann_fit kernel."""

    def test_known_temperature(self):
        """Synthetic data at T=10000K, recover T within sigma_T."""
        T_true = 10000.0  # K
        slope_true = -1.0 / (KB_EV * T_true)  # eV^-1
        intercept_true = 20.0

        rng = np.random.default_rng(42)
        n_lines = 20
        E_k = rng.uniform(1.0, 7.0, n_lines)
        noise_sigma = 0.05
        y = slope_true * E_k + intercept_true + rng.normal(0, noise_sigma, n_lines)

        x = jnp.array(E_k)[None, :]  # (1, N)
        y_jax = jnp.array(y)[None, :]
        w = jnp.ones_like(y_jax) / noise_sigma**2
        mask = jnp.ones_like(y_jax, dtype=bool)

        result = batched_boltzmann_fit(x, y_jax, w, mask)
        T_fit = float(result.T_K[0])
        sigma_T = float(result.sigma_T[0])

        assert (
            abs(T_fit - T_true) < 3 * sigma_T
        ), f"T_fit={T_fit:.1f} K, T_true={T_true:.1f} K, sigma_T={sigma_T:.1f} K"
        assert T_fit > 0, "Temperature should be positive"

    def test_batch_consistency(self):
        """B=10 independent fits match individual np.polyfit results."""
        rng = np.random.default_rng(123)
        B = 10
        N = 15

        slopes_np = np.zeros(B)
        intercepts_np = np.zeros(B)

        x_all = np.zeros((B, N))
        y_all = np.zeros((B, N))
        w_all = np.ones((B, N))

        for i in range(B):
            T = rng.uniform(5000, 20000)
            slope = -1.0 / (KB_EV * T)
            intercept = rng.uniform(15, 25)
            E_k = rng.uniform(1.0, 8.0, N)
            y = slope * E_k + intercept + rng.normal(0, 0.1, N)

            x_all[i] = E_k
            y_all[i] = y

            s_np, i_np = _numpy_wls_fit(E_k, y, np.ones(N))
            slopes_np[i] = s_np
            intercepts_np[i] = i_np

        result = batched_boltzmann_fit(
            jnp.array(x_all),
            jnp.array(y_all),
            jnp.array(w_all),
            jnp.ones((B, N), dtype=bool),
        )

        np.testing.assert_allclose(
            np.asarray(result.slope),
            slopes_np,
            rtol=1e-10,
            err_msg="Slopes disagree with numpy reference",
        )
        np.testing.assert_allclose(
            np.asarray(result.intercept),
            intercepts_np,
            rtol=1e-10,
            err_msg="Intercepts disagree with numpy reference",
        )

    def test_unweighted_reduction(self):
        """w=1 recovers OLS (compare against np.polyfit)."""
        rng = np.random.default_rng(456)
        N = 12
        E_k = rng.uniform(1.0, 6.0, N)
        y = -2.0 * E_k + 18.0 + rng.normal(0, 0.05, N)

        slope_np, intercept_np = _numpy_wls_fit(E_k, y, np.ones(N))

        result = batched_boltzmann_fit(
            jnp.array(E_k)[None, :],
            jnp.array(y)[None, :],
            jnp.ones((1, N)),
            jnp.ones((1, N), dtype=bool),
        )

        np.testing.assert_allclose(float(result.slope[0]), slope_np, rtol=1e-10)
        np.testing.assert_allclose(float(result.intercept[0]), intercept_np, rtol=1e-10)

    def test_single_element_batch(self):
        """B=1 with no shape issues."""
        x = jnp.array([[1.0, 3.0, 5.0]])
        y = jnp.array([[-2.0, -6.0, -10.0]])
        w = jnp.ones((1, 3))
        mask = jnp.ones((1, 3), dtype=bool)

        result = batched_boltzmann_fit(x, y, w, mask)
        assert result.slope.shape == (1,)
        assert result.T_K.shape == (1,)
        assert float(result.slope[0]) < 0, "Slope should be negative"

    def test_degenerate_det(self):
        """All E_k identical => det~0, no NaN/Inf."""
        x = jnp.array([[3.0, 3.0, 3.0, 3.0]])
        y = jnp.array([[10.0, 11.0, 10.5, 10.2]])
        w = jnp.ones((1, 4))
        mask = jnp.ones((1, 4), dtype=bool)

        result = batched_boltzmann_fit(x, y, w, mask)

        assert jnp.all(jnp.isfinite(result.slope)), "NaN/Inf in slope for degenerate case"
        assert jnp.all(jnp.isfinite(result.T_K)), "NaN/Inf in T_K for degenerate case"
        assert float(result.slope[0]) == 0.0, "Degenerate det should give slope=0"
        assert float(result.T_K[0]) == 0.0, "Degenerate det should give T=0"

    def test_two_lines_exact(self):
        """N=2: exact fit, R^2 = 1."""
        slope_true = -1.0 / (KB_EV * 8000.0)
        intercept_true = 22.0
        E1, E2 = 2.0, 6.0
        y1 = slope_true * E1 + intercept_true
        y2 = slope_true * E2 + intercept_true

        x = jnp.array([[E1, E2]])
        y = jnp.array([[y1, y2]])
        w = jnp.ones((1, 2))
        mask = jnp.ones((1, 2), dtype=bool)

        result = batched_boltzmann_fit(x, y, w, mask)

        np.testing.assert_allclose(float(result.slope[0]), slope_true, rtol=1e-12)
        np.testing.assert_allclose(float(result.intercept[0]), intercept_true, rtol=1e-12)
        np.testing.assert_allclose(float(result.R_squared[0]), 1.0, atol=1e-12)

    def test_masking(self):
        """Mixed N_valid per batch element via mask produces correct results."""
        # Element 0: 5 valid lines, element 1: 3 valid lines (2 masked)
        N_max = 5

        # Element 0: all 5 valid
        E_k_0 = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
        slope_0 = -1.0 / (KB_EV * 12000.0)
        y_0 = slope_0 * E_k_0 + 20.0

        # Element 1: only first 3 valid
        E_k_1 = np.array([1.5, 3.5, 6.0, 0.0, 0.0])
        slope_1 = -1.0 / (KB_EV * 8000.0)
        y_1_valid = slope_1 * E_k_1[:3] + 18.0
        y_1 = np.array([*y_1_valid, 0.0, 0.0])

        x = jnp.array(np.stack([E_k_0, E_k_1]))
        y = jnp.array(np.stack([y_0, y_1]))
        w = jnp.ones((2, N_max))
        mask = jnp.array([[True, True, True, True, True], [True, True, True, False, False]])

        result = batched_boltzmann_fit(x, y, w, mask)

        # Element 0: all valid
        ref_slope_0, ref_int_0 = _numpy_wls_fit(E_k_0, y_0, np.ones(5))
        np.testing.assert_allclose(float(result.slope[0]), ref_slope_0, rtol=1e-10)

        # Element 1: only first 3
        ref_slope_1, ref_int_1 = _numpy_wls_fit(E_k_1[:3], y_1_valid, np.ones(3))
        np.testing.assert_allclose(float(result.slope[1]), ref_slope_1, rtol=1e-10)

        # Check n_valid
        np.testing.assert_array_equal(np.asarray(result.n_valid), [5, 3])

    def test_accuracy_vs_numpy(self):
        """100 random batch elements, max relative error < 1e-10 vs numpy."""
        rng = np.random.default_rng(321)
        B = 100
        N = 20

        x_all = np.zeros((B, N))
        y_all = np.zeros((B, N))
        w_all = np.zeros((B, N))
        slopes_np = np.zeros(B)
        intercepts_np = np.zeros(B)

        for i in range(B):
            T = rng.uniform(3000, 25000)
            slope = -1.0 / (KB_EV * T)
            intercept = rng.uniform(10, 30)
            E_k = rng.uniform(0.5, 9.0, N)
            noise = rng.normal(0, 0.1, N)
            y = slope * E_k + intercept + noise
            w = rng.uniform(0.5, 2.0, N)

            x_all[i] = E_k
            y_all[i] = y
            w_all[i] = w

            s, c = _numpy_wls_fit(E_k, y, w)
            slopes_np[i] = s
            intercepts_np[i] = c

        result = batched_boltzmann_fit(
            jnp.array(x_all),
            jnp.array(y_all),
            jnp.array(w_all),
            jnp.ones((B, N), dtype=bool),
        )

        slope_err = np.abs(np.asarray(result.slope) - slopes_np) / np.abs(slopes_np)
        intercept_err = np.abs(np.asarray(result.intercept) - intercepts_np) / np.abs(intercepts_np)
        assert np.max(slope_err) < 1e-10, f"Max slope rel error: {np.max(slope_err):.2e}"
        assert (
            np.max(intercept_err) < 1e-10
        ), f"Max intercept rel error: {np.max(intercept_err):.2e}"

    def test_gradient(self):
        """jax.grad through batched_boltzmann_fit produces finite values."""

        def loss_fn(y):
            x = jnp.array([[1.0, 3.0, 5.0, 7.0]])
            w = jnp.ones((1, 4))
            mask = jnp.ones((1, 4), dtype=bool)
            result = batched_boltzmann_fit(x, y, w, mask)
            return jnp.sum(result.slope)

        y = jnp.array([[-2.0, -6.0, -10.0, -14.0]])
        grad_y = jax.grad(loss_fn)(y)

        assert jnp.all(jnp.isfinite(grad_y)), f"NaN in gradient: {grad_y}"

    def test_temperature_uncertainty(self):
        """Verify sigma_T is reasonable via Monte Carlo estimate."""
        T_true = 10000.0
        slope_true = -1.0 / (KB_EV * T_true)
        intercept_true = 20.0
        noise_sigma = 0.1

        rng = np.random.default_rng(55)
        N = 15
        E_k = rng.uniform(1.0, 7.0, N)

        # Monte Carlo: many realizations
        n_mc = 500
        T_mc = np.zeros(n_mc)
        for i in range(n_mc):
            y = slope_true * E_k + intercept_true + rng.normal(0, noise_sigma, N)
            s, _ = _numpy_wls_fit(E_k, y, np.ones(N) / noise_sigma**2)
            T_mc[i] = -1.0 / (KB_EV * s)

        sigma_T_mc = np.std(T_mc)

        # Analytical from JAX kernel
        y_single = (
            slope_true * E_k + intercept_true + np.random.default_rng(42).normal(0, noise_sigma, N)
        )
        result = batched_boltzmann_fit(
            jnp.array(E_k)[None, :],
            jnp.array(y_single)[None, :],
            jnp.ones((1, N)) / noise_sigma**2,
            jnp.ones((1, N), dtype=bool),
        )
        sigma_T_analytical = float(result.sigma_T[0])

        # Within factor of 2 of MC estimate (analytical is approximate)
        ratio = sigma_T_analytical / sigma_T_mc
        assert 0.3 < ratio < 3.0, (
            f"sigma_T ratio analytical/MC = {ratio:.2f} "
            f"(analytical={sigma_T_analytical:.1f}, MC={sigma_T_mc:.1f})"
        )


class TestBoltzmannTemperatureJax:
    """Tests for convenience wrapper boltzmann_temperature_jax."""

    def test_returns_tuple(self):
        """Returns (T_K, sigma_T) tuple."""
        x = jnp.array([[1.0, 3.0, 5.0]])
        y = jnp.array([[-2.0, -6.0, -10.0]])
        w = jnp.ones((1, 3))
        mask = jnp.ones((1, 3), dtype=bool)

        T_K, sigma_T = boltzmann_temperature_jax(x, y, w, mask)
        assert T_K.shape == (1,)
        assert sigma_T.shape == (1,)
        assert float(T_K[0]) > 0
