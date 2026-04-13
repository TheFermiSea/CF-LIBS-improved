"""
JAX softmax closure for compositional data on the simplex.

Provides numerically stable forward (theta -> C) and inverse (C -> theta)
transforms, plus the analytical Jacobian, for enforcing the closure
constraint sum(C_i) = 1 in gradient-based optimization.

# ASSERT_CONVENTION: theta [dimensionless], C_i [dimensionless], sum(C_i) = 1

This module is a **standalone** JAX implementation parallel to closure.py.
It does NOT modify the existing closure or joint_optimizer modules.

References
----------
DERV-04: Softmax closure derivation with Jacobian and limiting cases.
Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations for
    Compositional Data Analysis." Mathematical Geology 35(3), 279-300.
"""

from __future__ import annotations

from cflibs.core.jax_runtime import HAS_JAX

if HAS_JAX:
    import jax.numpy as jnp
    from jax import jit
else:
    raise ImportError("softmax_closure requires JAX. Install with: pip install jax jaxlib")


@jit
def softmax_closure(theta: jnp.ndarray) -> jnp.ndarray:
    """Convert unconstrained theta to compositions on the simplex.

    Uses the log-sum-exp trick for numerical stability:

        C_i = exp(theta_i - theta_max) / sum_j exp(theta_j - theta_max)

    Parameters
    ----------
    theta : jnp.ndarray
        Unconstrained parameters, shape ``(D,)`` or ``(B, D)``.

    Returns
    -------
    jnp.ndarray
        Compositions on the simplex (sum = 1, all positive), same shape
        as input.
    """
    is_1d = theta.ndim == 1
    theta = jnp.atleast_2d(theta)  # (B, D)

    # Log-sum-exp stabilization
    theta_max = jnp.max(theta, axis=-1, keepdims=True)
    exp_shifted = jnp.exp(theta - theta_max)
    C = exp_shifted / jnp.sum(exp_shifted, axis=-1, keepdims=True)

    if is_1d:
        return C.squeeze(0)
    return C


@jit
def inverse_softmax(C: jnp.ndarray) -> jnp.ndarray:
    """Convert compositions to unconstrained theta (centered representative).

    The inverse is ``theta = log(C) - mean(log(C))``, which picks the
    centered representative from the shift-invariant family.

    Parameters
    ----------
    C : jnp.ndarray
        Compositions on the simplex (all positive, sum ~ 1), shape
        ``(D,)`` or ``(B, D)``.

    Returns
    -------
    jnp.ndarray
        Unconstrained parameters, same shape as input.  Centered so that
        ``mean(theta) = 0``.
    """
    is_1d = C.ndim == 1
    C = jnp.atleast_2d(C)

    log_C = jnp.log(jnp.maximum(C, 1e-300))
    theta = log_C - jnp.mean(log_C, axis=-1, keepdims=True)

    if is_1d:
        return theta.squeeze(0)
    return theta


@jit
def softmax_jacobian(theta: jnp.ndarray) -> jnp.ndarray:
    """Analytical Jacobian of the softmax closure.

    .. math::

        J_{ij} = \\frac{\\partial C_i}{\\partial \\theta_j}
               = C_i (\\delta_{ij} - C_j)
               = [\\mathrm{diag}(C) - C C^T]_{ij}

    The Jacobian is symmetric, positive semi-definite, and has rank D-1
    (the all-ones vector is in its null space).

    Parameters
    ----------
    theta : jnp.ndarray
        Unconstrained parameters, shape ``(D,)``.

    Returns
    -------
    jnp.ndarray
        Jacobian matrix, shape ``(D, D)``.
    """
    C = softmax_closure(theta)
    return jnp.diag(C) - jnp.outer(C, C)
