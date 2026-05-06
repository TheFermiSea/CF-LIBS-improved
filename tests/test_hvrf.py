"""Tests for the HVRF intensity enhancement model."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.plasma.hvrf import HVRFParams, apply_hvrf_enhancement  # noqa: E402

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


def test_apply_hvrf_enhancement_preserves_shape_and_finite_values():
    """Enhanced intensities retain the input shape and remain finite."""
    intensities = jnp.array([1.0, 2.0, 4.0])
    params = HVRFParams(enhancement_scale=2.0, decay_constant=1.5, power_threshold=10.0)

    enhanced = apply_hvrf_enhancement(intensities, applied_power=25.0, params=params)

    assert enhanced.shape == intensities.shape
    assert np.all(np.isfinite(np.asarray(enhanced)))


def test_apply_hvrf_enhancement_is_bounded_by_scale():
    """The multiplicative factor is between one and one plus enhancement_scale."""
    intensities = jnp.array([1.0, 3.0, 9.0])
    params = HVRFParams(enhancement_scale=1.25, decay_constant=0.5, power_threshold=0.0)

    enhanced = apply_hvrf_enhancement(intensities, applied_power=1.0e6, params=params)
    factors = np.asarray(enhanced / intensities)

    assert np.all(factors >= 1.0)
    assert np.all(factors <= 1.0 + params.enhancement_scale)
    assert np.allclose(factors, 1.0 + params.enhancement_scale, rtol=1e-5)


def test_apply_hvrf_enhancement_threshold_limits_low_power_response():
    """Low power stays close to the unenhanced intensity when beta is finite."""
    intensities = jnp.array([2.0, 5.0])
    params = HVRFParams(enhancement_scale=3.0, decay_constant=2.0, power_threshold=50.0)

    enhanced = apply_hvrf_enhancement(intensities, applied_power=-50.0, params=params)

    assert np.allclose(np.asarray(enhanced), np.asarray(intensities), rtol=1e-6)
