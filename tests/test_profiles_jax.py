"""
Tests for JAX GPU Voigt profile batch API.

Tests the voigt_spectrum_jax broadcasting kernel against scipy.special.wofz
reference implementation, verifies normalization, limiting cases, output shapes,
float64 precision, and gradient stability.

References:
    Weideman (1994) SIAM J. Numer. Anal. 31(5) -- N=36 rational approximation
    Zaghloul (2024) arXiv:2411.00917 -- accuracy reference
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from scipy.special import wofz as scipy_wofz  # noqa: E402

from cflibs.radiation.profiles import voigt_spectrum_jax  # noqa: E402

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


def _scipy_voigt(wl, center, sigma, gamma):
    """Reference Voigt profile via scipy.special.wofz."""
    z = ((wl - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    w = scipy_wofz(z)
    return w.real / (sigma * np.sqrt(2.0 * np.pi))


class TestVoigtSpectrumJax:
    """Tests for voigt_spectrum_jax batch broadcasting API."""

    def test_voigt_spectrum_single_line(self):
        """Single line matches scipy.special.wofz reference."""
        wl = jnp.linspace(399.0, 401.0, 2001)
        center = jnp.array([400.0])
        intensity = jnp.array([1.0])
        sigma = jnp.array([0.05])
        gamma = jnp.array([0.02])

        spectrum = voigt_spectrum_jax(wl, center, intensity, sigma, gamma)

        ref = _scipy_voigt(np.asarray(wl), 400.0, 0.05, 0.02)
        rel_err = np.abs(np.asarray(spectrum) - ref) / np.maximum(np.abs(ref), 1e-30)
        # Only check where profile is significant (> 1e-10 of peak)
        significant = ref > 1e-10 * np.max(ref)
        assert (
            np.max(rel_err[significant]) < 1e-6
        ), f"Max relative error {np.max(rel_err[significant]):.2e} exceeds 1e-6"

    def test_voigt_spectrum_multi_line(self):
        """Sum of 10 lines matches sequential single-line computation."""
        wl = jnp.linspace(380.0, 420.0, 4001)
        rng = np.random.default_rng(42)
        n_lines = 10
        centers = jnp.array(rng.uniform(385.0, 415.0, n_lines))
        intensities = jnp.array(rng.uniform(0.5, 5.0, n_lines))
        sigmas = jnp.array(rng.uniform(0.02, 0.1, n_lines))
        gammas = jnp.array(rng.uniform(0.01, 0.05, n_lines))

        # Batch computation
        spectrum_batch = voigt_spectrum_jax(wl, centers, intensities, sigmas, gammas)

        # Sequential reference
        spectrum_seq = np.zeros(len(wl))
        for i in range(n_lines):
            v = _scipy_voigt(np.asarray(wl), float(centers[i]), float(sigmas[i]), float(gammas[i]))
            spectrum_seq += float(intensities[i]) * v

        rel_err = np.abs(np.asarray(spectrum_batch) - spectrum_seq) / np.maximum(
            np.abs(spectrum_seq), 1e-30
        )
        significant = spectrum_seq > 1e-10 * np.max(spectrum_seq)
        assert np.max(rel_err[significant]) < 1e-6

    def test_voigt_normalization(self):
        """Verify integral of Voigt profile = 1.0 for each line."""
        # Wide grid to capture Lorentzian wings; fine spacing for accuracy
        wl = jnp.linspace(200.0, 600.0, 200001)
        dlambda = float(wl[1] - wl[0])

        test_cases = [
            (0.05, 0.02),  # typical
            (0.01, 0.05),  # Lorentzian-dominated
            (0.1, 0.001),  # Gaussian-dominated
            (0.2, 0.2),  # equal widths
            (0.005, 0.005),  # narrow
        ]

        for sigma_val, gamma_val in test_cases:
            center = jnp.array([400.0])
            intensity = jnp.array([1.0])
            sigma = jnp.array([sigma_val])
            gamma = jnp.array([gamma_val])

            spectrum = voigt_spectrum_jax(wl, center, intensity, sigma, gamma)
            integral = float(jnp.sum(spectrum) * dlambda)
            # Tolerance accounts for trapezoidal quadrature truncation of
            # Lorentzian wings (decay as 1/x^2) over finite [200,600] nm domain.
            # True normalization is exact; error is purely from integration limits.
            assert abs(integral - 1.0) < 1e-3, (
                f"Normalization failed for sigma={sigma_val}, gamma={gamma_val}: "
                f"integral={integral:.8f}"
            )

    def test_voigt_limiting_gaussian(self):
        """gamma -> 0 gives Gaussian profile."""
        wl = jnp.linspace(398.0, 402.0, 4001)
        center = jnp.array([400.0])
        intensity = jnp.array([1.0])
        sigma_val = 0.05
        sigma = jnp.array([sigma_val])
        gamma = jnp.array([1e-10])  # near-zero Lorentzian

        spectrum = voigt_spectrum_jax(wl, center, intensity, sigma, gamma)

        # Gaussian reference
        x = (np.asarray(wl) - 400.0) / sigma_val
        gaussian = np.exp(-0.5 * x**2) / (sigma_val * np.sqrt(2.0 * np.pi))

        rel_err = np.abs(np.asarray(spectrum) - gaussian) / np.maximum(gaussian, 1e-30)
        significant = gaussian > 1e-6 * np.max(gaussian)
        assert (
            np.max(rel_err[significant]) < 1e-4
        ), f"Gaussian limit: max rel err = {np.max(rel_err[significant]):.2e}"

    def test_voigt_limiting_lorentzian(self):
        """sigma -> 0 gives Lorentzian profile."""
        wl = jnp.linspace(395.0, 405.0, 10001)
        center = jnp.array([400.0])
        intensity = jnp.array([1.0])
        sigma = jnp.array([1e-10])  # near-zero Gaussian
        gamma_val = 0.05
        gamma = jnp.array([gamma_val])

        spectrum = voigt_spectrum_jax(wl, center, intensity, sigma, gamma)

        # Lorentzian reference
        lorentz = (1.0 / np.pi) * gamma_val / ((np.asarray(wl) - 400.0) ** 2 + gamma_val**2)

        rel_err = np.abs(np.asarray(spectrum) - lorentz) / np.maximum(lorentz, 1e-30)
        significant = lorentz > 1e-6 * np.max(lorentz)
        assert (
            np.max(rel_err[significant]) < 1e-3
        ), f"Lorentzian limit: max rel err = {np.max(rel_err[significant]):.2e}"

    def test_voigt_accuracy_sweep(self):
        """100 (sigma, gamma) combinations, max relative error < 1e-6 vs wofz."""
        wl = jnp.linspace(395.0, 405.0, 1001)
        rng = np.random.default_rng(123)

        max_rel_err = 0.0
        for _ in range(100):
            sigma_val = 10 ** rng.uniform(-2, 0)  # 0.01 to 1.0 nm
            gamma_val = 10 ** rng.uniform(-2, 0)  # 0.01 to 1.0 nm

            center = jnp.array([400.0])
            intensity = jnp.array([1.0])
            sigma = jnp.array([sigma_val])
            gamma = jnp.array([gamma_val])

            spectrum = voigt_spectrum_jax(wl, center, intensity, sigma, gamma)
            ref = _scipy_voigt(np.asarray(wl), 400.0, sigma_val, gamma_val)

            significant = ref > 1e-10 * np.max(ref)
            if np.any(significant):
                rel_err = (
                    np.abs(np.asarray(spectrum)[significant] - ref[significant]) / ref[significant]
                )
                max_rel_err = max(max_rel_err, np.max(rel_err))

        assert max_rel_err < 1e-6, f"Max relative error across sweep: {max_rel_err:.2e}"

    def test_voigt_output_shape(self):
        """Verify output shape is (N_wl,)."""
        n_wl = 500
        n_lines = 20
        wl = jnp.linspace(380.0, 420.0, n_wl)
        centers = jnp.linspace(390.0, 410.0, n_lines)
        intensities = jnp.ones(n_lines)
        sigmas = jnp.full(n_lines, 0.05)
        gammas = jnp.full(n_lines, 0.02)

        spectrum = voigt_spectrum_jax(wl, centers, intensities, sigmas, gammas)
        assert spectrum.shape == (n_wl,), f"Expected ({n_wl},), got {spectrum.shape}"

    def test_voigt_float64_precision(self):
        """Confirm float64 dtype throughout."""
        wl = jnp.linspace(399.0, 401.0, 101, dtype=jnp.float64)
        center = jnp.array([400.0], dtype=jnp.float64)
        intensity = jnp.array([1.0], dtype=jnp.float64)
        sigma = jnp.array([0.05], dtype=jnp.float64)
        gamma = jnp.array([0.02], dtype=jnp.float64)

        spectrum = voigt_spectrum_jax(wl, center, intensity, sigma, gamma)
        assert spectrum.dtype == jnp.float64, f"Expected float64, got {spectrum.dtype}"

    def test_voigt_gradient(self):
        """jax.grad through the spectrum function produces no NaN."""

        def loss_fn(sigmas, gammas):
            wl = jnp.linspace(399.0, 401.0, 201)
            centers = jnp.array([400.0])
            intensities = jnp.array([1.0])
            return jnp.sum(voigt_spectrum_jax(wl, centers, intensities, sigmas, gammas))

        sigmas = jnp.array([0.05])
        gammas = jnp.array([0.02])

        grad_sigma, grad_gamma = jax.grad(loss_fn, argnums=(0, 1))(sigmas, gammas)

        assert jnp.all(jnp.isfinite(grad_sigma)), f"NaN in grad_sigma: {grad_sigma}"
        assert jnp.all(jnp.isfinite(grad_gamma)), f"NaN in grad_gamma: {grad_gamma}"

    def test_voigt_scalar_sigma_gamma(self):
        """Scalar sigma/gamma broadcasts correctly to all lines."""
        wl = jnp.linspace(390.0, 410.0, 2001)
        centers = jnp.array([395.0, 400.0, 405.0])
        intensities = jnp.array([1.0, 2.0, 1.5])

        # Scalar widths
        spectrum_scalar = voigt_spectrum_jax(wl, centers, intensities, 0.05, 0.02)
        # Array widths (same values)
        spectrum_array = voigt_spectrum_jax(
            wl,
            centers,
            intensities,
            jnp.array([0.05, 0.05, 0.05]),
            jnp.array([0.02, 0.02, 0.02]),
        )

        np.testing.assert_allclose(
            np.asarray(spectrum_scalar), np.asarray(spectrum_array), rtol=1e-12
        )
