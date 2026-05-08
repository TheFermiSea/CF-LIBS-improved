"""Numerical-equivalence tests for ``InstrumentModelJax``.

The JAX instrument model must produce identical output (within
``rtol=1e-5, atol=1e-7``) to :class:`InstrumentModel` for both
fixed-FWHM and resolving-power configurations.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from cflibs.instrument.model import InstrumentModel, InstrumentModelJax  # noqa: E402


@pytest.fixture
def wl_grid() -> np.ndarray:
    return np.linspace(200.0, 800.0, 2000)


def test_sigma_at_wavelength_fixed_fwhm(wl_grid):
    np_inst = InstrumentModel(resolution_fwhm_nm=0.05)
    jax_inst = InstrumentModelJax(resolution_fwhm_nm=0.05)

    expected = np.array([np_inst.sigma_at_wavelength(wl) for wl in wl_grid])
    actual = np.asarray(jax_inst.sigma_at_wavelength_array(wl_grid))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_sigma_at_wavelength_resolving_power(wl_grid):
    np_inst = InstrumentModel.from_resolving_power(20000.0)
    jax_inst = InstrumentModelJax.from_resolving_power(20000.0)

    expected = np.array([np_inst.sigma_at_wavelength(wl) for wl in wl_grid])
    actual = np.asarray(jax_inst.sigma_at_wavelength_array(wl_grid))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_apply_response_no_curve(wl_grid):
    intensity = np.linspace(1.0, 5.0, len(wl_grid))
    jax_inst = InstrumentModelJax(resolution_fwhm_nm=0.05)
    out = jax_inst.apply_response(wl_grid, intensity)
    np.testing.assert_allclose(out, intensity, rtol=0, atol=0)


def test_apply_response_jax_matches_numpy(wl_grid):
    # Build a synthetic response curve covering most of the grid
    rng = np.random.default_rng(1234)
    wl_resp = np.linspace(250.0, 750.0, 50)
    resp = 0.5 + 0.5 * rng.uniform(0.0, 1.0, size=wl_resp.shape)
    response_curve = np.column_stack([wl_resp, resp])

    intensity = 1.0 + 0.5 * np.sin(wl_grid / 10.0)

    np_inst = InstrumentModel(resolution_fwhm_nm=0.05, response_curve=response_curve)
    jax_inst = InstrumentModelJax(
        resolution_fwhm_nm=0.05, response_curve=response_curve
    )

    out_np = np_inst.apply_response(wl_grid, intensity)
    out_jax = np.asarray(jax_inst.apply_response(wl_grid, intensity))
    np.testing.assert_allclose(out_jax, out_np, rtol=1e-5, atol=1e-7)


def test_from_instrument_model_promote():
    base = InstrumentModel(resolution_fwhm_nm=0.04, resolving_power=15000.0)
    promoted = InstrumentModelJax.from_instrument_model(base)
    assert promoted.resolution_fwhm_nm == base.resolution_fwhm_nm
    assert promoted.resolving_power == base.resolving_power
    # Sigma helper still matches the numpy parent
    np.testing.assert_allclose(
        float(promoted.sigma_at_wavelength_array(500.0)),
        base.sigma_at_wavelength(500.0),
        rtol=1e-5,
        atol=1e-7,
    )
