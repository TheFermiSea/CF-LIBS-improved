import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cflibs.radiation.profiles import apply_ldm_broadening_voigt_jax, voigt_spectrum_jax

def test_ldm_parity():
    # Ensure JAX is using float64 for better parity if possible, 
    # but float32 is fine for this test.
    
    wl_grid = jnp.linspace(400, 410, 1000)
    line_centers = jnp.array([402.0, 405.0, 408.0])
    line_intensities = jnp.array([1.0, 2.0, 0.5])
    sigmas = jnp.array([0.05, 0.06, 0.04])
    gammas = jnp.array([0.02, 0.03, 0.01])
    
    # Exact calculation (direct sum of profiles)
    spec_exact = voigt_spectrum_jax(wl_grid, line_centers, line_intensities, sigmas, gammas)
    
    # LDM calculation (binned convolution)
    # Use more bins for better parity in test
    spec_ldm = apply_ldm_broadening_voigt_jax(
        wl_grid, line_centers, line_intensities, sigmas, gammas, 
        n_sigma_bins=32, n_gamma_bins=32
    )
    
    # Parity check - LDM is an approximation, so we expect some error.
    # With 32x32 bins, it should be quite accurate.
    # Peak height is ~8, so 0.1 absolute error is ~1.2% relative error.
    np.testing.assert_allclose(spec_exact, spec_ldm, atol=0.1, rtol=0.05)

def test_ldm_large_list():
    # Test with many lines to ensure performance and stability
    n_lines = 1000
    wl_grid = jnp.linspace(400, 410, 1000)
    key = jax.random.PRNGKey(42)
    line_centers = jax.random.uniform(key, (n_lines,), minval=401, maxval=409)
    line_intensities = jax.random.uniform(key, (n_lines,), minval=0.1, maxval=1.0)
    sigmas = jax.random.uniform(key, (n_lines,), minval=0.03, maxval=0.08)
    gammas = jax.random.uniform(key, (n_lines,), minval=0.01, maxval=0.05)
    
    spec_ldm = apply_ldm_broadening_voigt_jax(
        wl_grid, line_centers, line_intensities, sigmas, gammas, 
        n_sigma_bins=16, n_gamma_bins=16
    )
    
    assert spec_ldm.shape == wl_grid.shape
    assert jnp.all(jnp.isfinite(spec_ldm))
    assert jnp.sum(spec_ldm) > 0
