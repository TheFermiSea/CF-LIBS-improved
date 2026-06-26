"""Tests for cflibs.instrument.convolution, incl. the sigma<=0 guard (M1-A2)."""

import numpy as np
import pytest

from cflibs.instrument.convolution import apply_instrument_function


def _spectrum():
    wl = np.linspace(400.0, 410.0, 201)  # evenly spaced
    inten = np.zeros_like(wl)
    inten[100] = 1.0  # a single line
    return wl, inten


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_nonpositive_sigma_returns_input_no_nan(sigma):
    """sigma<=0 means no broadening: return input unchanged, never NaN.

    Regression for instrument F10 / blueprint M1-A2: the degenerate kernel was
    exp(-0.5*(0/0)^2) = NaN, which signal.convolve propagated to the whole output.
    """
    wl, inten = _spectrum()
    out = apply_instrument_function(wl, inten, sigma)
    assert not np.any(np.isnan(out))
    np.testing.assert_array_equal(out, inten)


def test_positive_sigma_still_broadens_and_conserves_area():
    wl, inten = _spectrum()
    out = apply_instrument_function(wl, inten, 0.1)
    assert not np.any(np.isnan(out))
    assert out[100] < 1.0  # peak spread out
    assert out.sum() == pytest.approx(inten.sum(), rel=1e-6)  # normalized kernel


def test_jax_variant_guards_nonpositive_sigma():
    pytest.importorskip("jax")
    from cflibs.instrument.convolution import apply_instrument_function_jax

    wl, inten = _spectrum()
    out = apply_instrument_function_jax(wl, inten, 0.0)
    assert not np.any(np.isnan(out))
    np.testing.assert_array_equal(np.asarray(out), inten)
