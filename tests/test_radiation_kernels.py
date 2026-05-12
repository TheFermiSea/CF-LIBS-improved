import jax.numpy as jnp
from cflibs.radiation.kernels import forward_model, _forward_model_per_chunk

def test_forward_model_line_mask():
    """Verify that the line_mask parameter correctly modulates the spectrum."""
    params = {'T': 10000.0, 'Ne': 1e17}
    wavelengths = jnp.linspace(290, 310, 50)
    atomic_data = None
    
    # Case 1: No mask vs All-ones mask (should be identical)
    spec_none = forward_model(params, wavelengths, atomic_data, line_mask=None)
    spec_ones = forward_model(params, wavelengths, atomic_data, line_mask=jnp.ones((10,)))
    assert jnp.allclose(spec_none, spec_ones)
    
    # Case 2: Zero mask (should result in zero intensity)
    spec_zeros = forward_model(params, wavelengths, atomic_data, line_mask=jnp.zeros((10,)))
    assert jnp.allclose(spec_zeros, 0.0)

def test_chunked_wrapper_parity():
    """Verify that _forward_model_per_chunk behaves identically to forward_model."""
    params = {'T': 15000.0, 'Ne': 2e17}
    wavelengths = jnp.linspace(300, 305, 20)
    atomic_data = None
    mask = jnp.array([0.0, 1.0, 0.5, 0.0, 1.0, 0.2, 0.8, 0.3, 0.1, 0.9])
    
    res_direct = forward_model(params, wavelengths, atomic_data, line_mask=mask)
    res_chunked = _forward_model_per_chunk(params, wavelengths, atomic_data, line_mask=mask)
    
    assert jnp.allclose(res_direct, res_chunked)
