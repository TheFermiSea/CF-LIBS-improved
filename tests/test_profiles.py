"""
Tests for line profile functions.
"""

import numpy as np
import pytest
from cflibs.radiation.profiles import (
    gaussian_profile,
    lorentzian_profile,
    voigt_profile,
    voigt_fwhm,
    doppler_width,
    HAS_JAX,
)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def test_gaussian_integral():
    """Test Gaussian profile integrates to amplitude."""
    x = np.linspace(-10, 10, 1000)
    sigma = 1.0
    amp = 5.0
    y = gaussian_profile(x, 0.0, sigma, amp)
    integral = _trapezoid(y, x)
    assert np.isclose(integral, amp, rtol=1e-3)


def test_lorentzian_integral():
    """Test Lorentzian profile integrates to amplitude."""
    # Lorentzian has heavy tails, need wide integration range
    x = np.linspace(-100, 100, 10000)
    gamma = 1.0
    amp = 5.0
    y = lorentzian_profile(x, 0.0, gamma, amp)

    integral = _trapezoid(y, x)
    # 1e-2 tolerance due to finite integration range
    assert np.isclose(integral, amp, rtol=1e-2)


def test_voigt_limits():
    """Test Voigt profile reduces to Gaussian or Lorentzian in limits."""
    x = np.linspace(-5, 5, 100)
    sigma = 1.0
    gamma = 1e-9  # limit -> 0

    # Gaussian limit
    v = voigt_profile(x, 0.0, sigma, gamma, 1.0)
    g = gaussian_profile(x, 0.0, sigma, 1.0)
    assert np.allclose(v, g, atol=1e-4)

    # Lorentzian limit
    sigma = 1e-9
    gamma = 1.0
    v = voigt_profile(x, 0.0, sigma, gamma, 1.0)
    lorentzian = lorentzian_profile(x, 0.0, gamma, 1.0)
    assert np.allclose(v, lorentzian, atol=1e-4)


def test_voigt_fwhm():
    """Test Voigt FWHM approximation."""
    sigma = 1.0
    gamma = 1.0

    # Calculate expected FWHM
    # fG = 2.355 * 1.0 = 2.355
    # fL = 2.0 * 1.0 = 2.0
    # fV approx 0.5346*2 + sqrt(0.2166*4 + 2.355^2)
    # = 1.0692 + sqrt(0.8664 + 5.546)
    # = 1.0692 + sqrt(6.4124)
    # = 1.0692 + 2.532
    # = 3.601

    fwhm = voigt_fwhm(sigma, gamma)
    assert 3.5 < fwhm < 3.7


def test_doppler_width():
    """Test Doppler width calculation."""
    # T = 1 eV (~11600 K), Mass = 1 amu (H)
    # sigma = lambda * sqrt(2kT/mc^2)
    # v_th = sqrt(2kT/m) ~ sqrt(2 * 1.6e-19 / 1.67e-27) ~ sqrt(1.9e8) ~ 1.38e4 m/s
    # c = 3e8
    # sigma/lambda = v/c ~ 1.38e4 / 3e8 ~ 4.6e-5
    # lambda = 500 nm -> sigma ~ 0.023 nm
    # FWHM ~ 2.355 * 0.023 ~ 0.054 nm

    dw = doppler_width(500.0, 1.0, 1.0)
    assert 0.04 < dw < 0.07


def test_jax_import():
    """Test if JAX-compatible function is importable (even if JAX missing)."""
    from cflibs.radiation.profiles import voigt_profile_jax

    if not HAS_JAX:
        with pytest.raises(ImportError):
            voigt_profile_jax(0.0, 0.0, 1.0, 1.0)


# --- JAX Gradient Stability Tests ---


@pytest.mark.requires_jax
@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestWidemanFaddeeva:
    """Tests for the Weideman Faddeeva function implementation."""

    def test_weideman_accuracy_vs_scipy(self):
        """Test Weideman approximation accuracy against scipy.wofz."""
        import jax.numpy as jnp
        from cflibs.radiation.profiles import _faddeeva_weideman_jax

        try:
            from scipy.special import wofz
        except ImportError:
            pytest.skip("scipy not available")

        # Test points across complex plane
        test_points = [
            0.5 + 0.5j,
            1.0 + 0.1j,
            2.0 + 2.0j,
            0.1 + 0.01j,
            5.0 + 0.5j,
            0.0 + 1.0j,
            3.0 + 0.001j,
        ]

        for z in test_points:
            z_jax = jnp.array(z)
            w_scipy = wofz(z)
            w_weideman = complex(_faddeeva_weideman_jax(z_jax))
            rel_err = abs(w_weideman - w_scipy) / abs(w_scipy)
            # Weideman N=36 gives ~1e-8 accuracy, which is excellent for spectroscopy
            assert rel_err < 1e-6, f"At z={z}: rel_err={rel_err:.2e}"

    def test_weideman_gradient_stability(self):
        """Test that Weideman Faddeeva has stable gradients across all regions."""
        import jax
        import jax.numpy as jnp
        from cflibs.radiation.profiles import voigt_profile_jax

        wavelength = jnp.linspace(400.0, 410.0, 100)
        center = 405.0
        sigma = 0.01  # Doppler width

        def loss(params):
            """Loss function for gradient testing."""
            sigma_p, gamma_p = params
            profile = voigt_profile_jax(wavelength, center, sigma_p, gamma_p, 1.0)
            return jnp.sum(profile**2)

        grad_fn = jax.grad(loss)

        # Test across range of electron densities (log_ne from 15 to 19)
        # gamma scales with n_e via Stark broadening
        for log_ne in [15.0, 16.0, 17.0, 17.5, 18.0, 18.5, 19.0]:
            n_e = 10**log_ne
            gamma = 0.001 * (n_e / 1e16)  # Simplified Stark scaling

            params = jnp.array([sigma, gamma])
            grad = grad_fn(params)

            assert jnp.isfinite(grad).all(), f"NaN/Inf gradient at log_ne={log_ne}: grad={grad}"

    def test_voigt_jax_gradient_finite(self):
        """Test Voigt profile has finite gradients for typical LIBS parameters."""
        import jax
        import jax.numpy as jnp
        from cflibs.radiation.profiles import voigt_profile_jax

        wavelength = jnp.linspace(390.0, 400.0, 200)
        center = 396.15  # Ca II

        def integrated_intensity(params):
            T_eV, log_ne = params
            # Doppler width (simplified)
            sigma = 0.001 * jnp.sqrt(T_eV)
            # Stark width
            n_e = 10**log_ne
            gamma = 0.001 * (n_e / 1e16) ** 0.5

            profile = voigt_profile_jax(wavelength, center, sigma, gamma, 1.0)
            return jnp.sum(profile)

        grad_fn = jax.grad(integrated_intensity)

        # Test at various plasma conditions
        test_params = [
            (0.5, 16.0),  # Cool, moderate density
            (1.0, 17.0),  # Typical LIBS
            (2.0, 18.0),  # Hot, high density
            (1.5, 19.0),  # Very high density
        ]

        for T_eV, log_ne in test_params:
            params = jnp.array([T_eV, log_ne])
            grad = grad_fn(params)
            assert jnp.isfinite(
                grad
            ).all(), f"NaN/Inf gradient at T={T_eV}eV, log_ne={log_ne}: grad={grad}"

    def test_voigt_jax_matches_numpy(self):
        """Test JAX Voigt profile matches NumPy version."""
        import jax.numpy as jnp
        from cflibs.radiation.profiles import voigt_profile_jax

        wavelength = np.linspace(400.0, 410.0, 100)
        center = 405.0
        sigma = 0.05
        gamma = 0.02
        amplitude = 1.0

        v_numpy = voigt_profile(wavelength, center, sigma, gamma, amplitude)
        v_jax = np.array(voigt_profile_jax(jnp.array(wavelength), center, sigma, gamma, amplitude))

        assert np.allclose(
            v_numpy, v_jax, rtol=1e-4
        ), f"Max diff: {np.max(np.abs(v_numpy - v_jax))}"
