"""
Tests for manifold generator physics functions.

Validates JAX-compatible physics implementations against NumPy reference.

Requirements: JAX
"""

import numpy as np
import pytest

# Mark entire module as requiring JAX
pytestmark = pytest.mark.requires_jax

# Skip all tests if JAX is not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.radiation.profiles import (  # noqa: E402
    doppler_width,
    voigt_profile,
)

# Import JAX versions
try:
    from cflibs.radiation.profiles import (
        doppler_sigma_jax,
        voigt_profile_jax,
        gaussian_profile_jax,
        lorentzian_profile_jax,
    )
    from cflibs.radiation.stark import (
        stark_hwhm,
        stark_hwhm_jax,
        estimate_stark_parameter,
        estimate_stark_parameter_jax,
    )

    HAS_JAX_FUNCS = True
except ImportError:
    HAS_JAX_FUNCS = False


def _jnp_trapezoid(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    if hasattr(jnp, "trapezoid"):
        return jnp.trapezoid(y, x)
    return jnp.trapz(y, x)


@pytest.fixture
def sample_params():
    """Sample physical parameters for testing."""
    return {
        "wavelength_nm": 500.0,
        "T_eV": 1.0,
        "mass_amu": 48.0,  # Ti
        "n_e_cm3": 1e17,
        "stark_w_ref": 0.01,  # nm
        "stark_alpha": 0.5,
    }


@pytest.mark.requires_jax
class TestDopplerWidthJAX:
    """Tests for JAX-compatible Doppler width calculation."""

    def test_doppler_sigma_jax_basic(self, sample_params):
        """Test basic Doppler sigma calculation."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        sigma = doppler_sigma_jax(
            sample_params["wavelength_nm"],
            sample_params["T_eV"],
            sample_params["mass_amu"],
        )
        # Should be positive
        assert sigma > 0
        # Typical LIBS Doppler width is ~0.01-0.1 nm sigma
        assert 0.001 < sigma < 1.0

    def test_doppler_sigma_matches_numpy(self, sample_params):
        """Test JAX Doppler sigma matches NumPy reference."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        # Calculate using JAX
        sigma_jax = float(
            doppler_sigma_jax(
                sample_params["wavelength_nm"],
                sample_params["T_eV"],
                sample_params["mass_amu"],
            )
        )

        # Calculate FWHM using NumPy and convert to sigma
        fwhm_numpy = doppler_width(
            sample_params["wavelength_nm"],
            sample_params["T_eV"],
            sample_params["mass_amu"],
        )
        sigma_numpy = fwhm_numpy / 2.355

        # Should match within 1%
        np.testing.assert_allclose(sigma_jax, sigma_numpy, rtol=0.01)

    def test_doppler_scaling_with_temperature(self):
        """Test Doppler width scales as sqrt(T)."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        sigma_1 = float(doppler_sigma_jax(500.0, 1.0, 48.0))
        sigma_4 = float(doppler_sigma_jax(500.0, 4.0, 48.0))

        # Doppler width ∝ sqrt(T), so sigma_4 / sigma_1 ≈ 2
        ratio = sigma_4 / sigma_1
        np.testing.assert_allclose(ratio, 2.0, rtol=0.01)

    def test_doppler_scaling_with_mass(self):
        """Test Doppler width scales as 1/sqrt(m)."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        sigma_light = float(doppler_sigma_jax(500.0, 1.0, 12.0))  # C
        sigma_heavy = float(doppler_sigma_jax(500.0, 1.0, 48.0))  # Ti

        # sigma ∝ 1/sqrt(m), so sigma_heavy/sigma_light = sqrt(12/48) = 0.5
        ratio = sigma_heavy / sigma_light
        expected_ratio = np.sqrt(12.0 / 48.0)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.01)


@pytest.mark.requires_jax
class TestStarkBroadeningJAX:
    """Tests for JAX-compatible Stark broadening calculations."""

    def test_stark_hwhm_jax_basic(self, sample_params):
        """Test basic Stark HWHM calculation."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        gamma = stark_hwhm_jax(
            sample_params["n_e_cm3"],
            sample_params["T_eV"],
            sample_params["stark_w_ref"],
            sample_params["stark_alpha"],
        )
        # Should be positive
        assert gamma > 0

    def test_stark_hwhm_scaling_with_density(self, sample_params):
        """Test Stark width scales linearly with electron density."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        gamma_1e17 = float(
            stark_hwhm_jax(1e17, sample_params["T_eV"], sample_params["stark_w_ref"], 0.5)
        )
        gamma_1e18 = float(
            stark_hwhm_jax(1e18, sample_params["T_eV"], sample_params["stark_w_ref"], 0.5)
        )

        # Stark width ∝ n_e, so ratio should be ~10
        ratio = gamma_1e18 / gamma_1e17
        np.testing.assert_allclose(ratio, 10.0, rtol=0.01)

    def test_stark_hwhm_matches_numpy(self, sample_params):
        """Test JAX Stark HWHM matches NumPy reference."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        # Calculate using JAX
        gamma_jax = float(
            stark_hwhm_jax(
                sample_params["n_e_cm3"],
                sample_params["T_eV"],
                sample_params["stark_w_ref"],
                sample_params["stark_alpha"],
            )
        )

        # Calculate using NumPy
        # stark_hwhm expects T_K, so convert T_eV to K
        T_K = sample_params["T_eV"] * 11604.5
        gamma_numpy = stark_hwhm(
            sample_params["n_e_cm3"],
            T_K,
            sample_params["stark_w_ref"],
            sample_params["stark_alpha"],
        )

        # Should match within 5% (different T reference handling)
        np.testing.assert_allclose(gamma_jax, gamma_numpy, rtol=0.05)

    def test_estimate_stark_parameter_jax(self):
        """Test JAX Stark parameter estimation."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        # Estimate for a typical line
        w_est = float(
            estimate_stark_parameter_jax(
                wavelength_nm=500.0,
                upper_energy_ev=3.0,
                ionization_potential_ev=6.0,
                ionization_stage=1,
            )
        )

        # Should be in reasonable range (0.0001 to 0.5 nm)
        assert 0.0001 <= w_est <= 0.5

    def test_estimate_stark_matches_numpy(self):
        """Test JAX Stark estimation matches NumPy."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        params = (500.0, 3.0, 6.0, 1)

        w_jax = float(estimate_stark_parameter_jax(*params))
        w_numpy = estimate_stark_parameter(*params)

        np.testing.assert_allclose(w_jax, w_numpy, rtol=0.01)


@pytest.mark.requires_jax
class TestVoigtProfileJAX:
    """Tests for JAX-compatible Voigt profile."""

    def test_voigt_profile_jax_basic(self):
        """Test basic Voigt profile calculation."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        wl_grid = jnp.linspace(499.0, 501.0, 100)
        center = 500.0
        sigma = 0.05
        gamma = 0.02

        profile = voigt_profile_jax(wl_grid, center, sigma, gamma, amplitude=1.0)

        # Peak should be at center
        peak_idx = jnp.argmax(profile)
        np.testing.assert_allclose(wl_grid[peak_idx], center, atol=0.03)

        # Should integrate to ~1 (amplitude)
        integral = _jnp_trapezoid(profile, wl_grid)
        np.testing.assert_allclose(float(integral), 1.0, rtol=0.1)

    def test_voigt_matches_gaussian_limit(self):
        """Test Voigt approaches Gaussian when gamma → 0."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        wl_grid = jnp.linspace(499.0, 501.0, 200)
        center = 500.0
        sigma = 0.05
        gamma = 1e-10  # Very small Lorentzian width

        voigt = voigt_profile_jax(wl_grid, center, sigma, gamma, amplitude=1.0)
        gaussian = gaussian_profile_jax(wl_grid, center, sigma, amplitude=1.0)

        # Should be nearly identical - use relaxed atol for float32 near-zero values
        # The Voigt has small residual values (~1e-8) in wings where Gaussian is exactly 0
        np.testing.assert_allclose(np.array(voigt), np.array(gaussian), rtol=0.01, atol=1e-7)

    def test_voigt_matches_lorentzian_limit(self):
        """Test Voigt approaches Lorentzian when sigma → 0."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        wl_grid = jnp.linspace(499.0, 501.0, 200)
        center = 500.0
        sigma = 1e-10  # Very small Gaussian width
        gamma = 0.05

        voigt = voigt_profile_jax(wl_grid, center, sigma, gamma, amplitude=1.0)
        lorentz = lorentzian_profile_jax(wl_grid, center, gamma, amplitude=1.0)

        # Should be nearly identical at core (wings may differ due to numerical limits)
        core_mask = np.abs(np.array(wl_grid) - center) < 0.2
        np.testing.assert_allclose(
            np.array(voigt)[core_mask], np.array(lorentz)[core_mask], rtol=0.1
        )

    def test_voigt_jax_matches_numpy(self):
        """Test JAX Voigt matches NumPy reference."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        wl_grid = np.linspace(499.0, 501.0, 100)
        center = 500.0
        sigma = 0.05
        gamma = 0.02

        voigt_jax = np.array(
            voigt_profile_jax(jnp.array(wl_grid), center, sigma, gamma, amplitude=1.0)
        )
        voigt_numpy = voigt_profile(wl_grid, center, sigma, gamma, amplitude=1.0)

        # Should match within 5%
        np.testing.assert_allclose(voigt_jax, voigt_numpy, rtol=0.05)


@pytest.mark.requires_jax
class TestProfilesVmap:
    """Test that profiles work correctly under JAX vmap."""

    def test_voigt_under_vmap(self):
        """Test Voigt profile works correctly when vmapped over lines."""
        if not HAS_JAX_FUNCS:
            pytest.skip("JAX physics functions not available")

        wl_grid = jnp.linspace(499.0, 501.0, 100)

        # Multiple lines
        centers = jnp.array([499.5, 500.0, 500.5])
        sigmas = jnp.array([0.05, 0.04, 0.06])
        gammas = jnp.array([0.02, 0.03, 0.01])
        amplitudes = jnp.array([1.0, 2.0, 0.5])

        # Vmap over lines
        profiles = jax.vmap(
            lambda c, s, g, a: voigt_profile_jax(wl_grid, c, s, g, a),
            in_axes=(0, 0, 0, 0),
        )(centers, sigmas, gammas, amplitudes)

        # Should have shape (n_lines, n_wavelengths)
        assert profiles.shape == (3, 100)

        # Sum should give combined spectrum
        spectrum = jnp.sum(profiles, axis=0)
        assert spectrum.shape == (100,)

        # Integral should be sum of amplitudes
        total_integral = float(_jnp_trapezoid(spectrum, wl_grid))
        np.testing.assert_allclose(total_integral, 3.5, rtol=0.1)
