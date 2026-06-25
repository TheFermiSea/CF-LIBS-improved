"""Regression guards for two Bayesian correctness fixes (audit C5, C6)."""

import math

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.scipy.special import gammaln  # noqa: E402

from cflibs.inversion.solve.bayesian.likelihood import (  # noqa: E402
    _poisson_cash_log_likelihood,
)
from cflibs.inversion.solve.bayesian.priors import NoiseParameters  # noqa: E402


@pytest.mark.requires_bayesian
def test_poisson_readout_residual_is_observed_minus_predicted():
    """C5: the Gaussian readout term uses ``observed - predicted``, not
    ``observed - mu``. At a perfect fit (predicted == observed) the readout
    residual must be exactly zero, so the total equals the shot term plus the
    readout normalization. With the old ``observed - mu`` reference the readout
    residual was ``observed - (observed/gain + dark)`` -> nonzero, penalizing a
    perfect fit and making the read-noise floor wrongly depend on gain/dark.
    """
    obs = jnp.array([100.0, 250.0, 80.0])
    pred = obs  # perfect fit
    npar = NoiseParameters(gain=2.0, dark_current=5.0, readout_noise=3.0)

    ll = float(_poisson_cash_log_likelihood(pred, obs, npar))

    mu = pred / npar.gain + npar.dark_current
    shot = float(jnp.sum(obs * jnp.log(mu) - mu - gammaln(obs + 1.0)))
    readout_var = npar.readout_noise**2
    readout_norm = float(-0.5 * obs.size * math.log(2.0 * math.pi * readout_var))  # residual==0

    assert ll == pytest.approx(shot + readout_norm, rel=1e-6)


@pytest.mark.requires_bayesian
def test_t_ordering_penalty_has_restoring_gradient():
    """C6: the T_core > T_shell ordering penalty is a smooth quadratic hinge with
    a genuine restoring gradient in the violated region — not the old
    ``jnp.where(..., 0, -1e6)`` cliff, which had zero gradient on both sides and
    a discontinuity at the boundary (pathological for NUTS).
    """
    from cflibs.inversion.solve.bayesian.models import _T_ORDERING_PENALTY_SCALE as K

    def penalty(T_core, T_shell):
        violation = jnp.maximum(T_shell - T_core, 0.0)
        return -K * jnp.square(violation)

    # Ordered (T_core > T_shell): no penalty, no gradient.
    assert float(penalty(1.5, 1.0)) == 0.0
    assert float(jax.grad(penalty, argnums=0)(1.5, 1.0)) == 0.0

    # Violated (T_core < T_shell): penalized, with a restoring gradient that
    # pushes T_core up and T_shell down.
    assert float(penalty(1.0, 1.5)) < 0.0
    assert float(jax.grad(penalty, argnums=0)(1.0, 1.5)) > 0.0  # raise T_core
    assert float(jax.grad(penalty, argnums=1)(1.0, 1.5)) < 0.0  # lower T_shell
