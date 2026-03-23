"""
JAX-accelerated batched Boltzmann plot fitting.

Implements GPU-parallelized weighted least-squares (WLS) fitting for
Boltzmann plots using closed-form normal equations. Designed for batch
processing of multiple spectra simultaneously without Python loops.

The closed-form WLS uses 5 dot products per batch element following
DERV-02 Eq. (01-01.4). Pad-and-mask batching handles variable line counts.

# ASSERT_CONVENTION: x = E_k [eV], y = ln(I*lambda/(g_k*A_ki)) [dimensionless],
#   slope [eV^-1], T_K [K], k_B = 8.617333e-5 eV/K

References:
    Tognoni et al. (2010) Spectrochim. Acta B 65 -- CF-LIBS methodology
    Weideman (1994) -- see profiles.py for Voigt kernel companion

Notes:
    - Float64 required for numerical stability of normal equations
    - Does NOT modify or replace boltzmann.py (CPU NumPy implementation)
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None  # type: ignore[assignment]
    jnp = None

    def jit(f):  # type: ignore[misc]
        return f


from cflibs.core.constants import KB_EV


@dataclass
class BoltzmannFitResultJax:
    """
    Results of batched Boltzmann plot fitting.

    All arrays have shape (B,) where B is the batch size.

    Attributes
    ----------
    slope : array, shape (B,)
        Slope of Boltzmann plot = -1/(k_B*T) [eV^-1]
    intercept : array, shape (B,)
        Y-intercept [dimensionless]
    T_K : array, shape (B,)
        Excitation temperature [K]. Zero for degenerate fits.
    sigma_T : array, shape (B,)
        1-sigma uncertainty in temperature [K]
    sigma_slope : array, shape (B,)
        1-sigma uncertainty in slope [eV^-1]
    sigma_intercept : array, shape (B,)
        1-sigma uncertainty in intercept [dimensionless]
    R_squared : array, shape (B,)
        Coefficient of determination (goodness of fit)
    n_valid : array, shape (B,)
        Number of valid (unmasked) lines per batch element
    """

    slope: jnp.ndarray  # type: ignore[name-defined]
    intercept: jnp.ndarray  # type: ignore[name-defined]
    T_K: jnp.ndarray  # type: ignore[name-defined]
    sigma_T: jnp.ndarray  # type: ignore[name-defined]
    sigma_slope: jnp.ndarray  # type: ignore[name-defined]
    sigma_intercept: jnp.ndarray  # type: ignore[name-defined]
    R_squared: jnp.ndarray  # type: ignore[name-defined]
    n_valid: jnp.ndarray  # type: ignore[name-defined]


# Register as JAX pytree so @jit can return this dataclass
if HAS_JAX:
    _FIELDS = (
        "slope",
        "intercept",
        "T_K",
        "sigma_T",
        "sigma_slope",
        "sigma_intercept",
        "R_squared",
        "n_valid",
    )

    def _result_flatten(result):
        return [getattr(result, f) for f in _FIELDS], None

    def _result_unflatten(aux_data, children):
        return BoltzmannFitResultJax(**dict(zip(_FIELDS, children)))

    jax.tree_util.register_pytree_node(BoltzmannFitResultJax, _result_flatten, _result_unflatten)


def _raise_jax_missing() -> None:
    raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")


if HAS_JAX:

    @jit
    def batched_boltzmann_fit(
        x: jnp.ndarray,
        y: jnp.ndarray,
        w: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> BoltzmannFitResultJax:
        """
        Batched weighted least-squares Boltzmann fit via closed-form normal equations.

        Computes slope, intercept, temperature, uncertainty, and R-squared for
        B independent Boltzmann plots simultaneously using 5-sum reduction.
        No Python loops -- fully vectorized for GPU execution.

        Parameters
        ----------
        x : jnp.ndarray, shape (B, N_max)
            Upper-level energies E_k [eV].
        y : jnp.ndarray, shape (B, N_max)
            Boltzmann plot values ln(I*lambda/(g_k*A_ki)) [dimensionless].
        w : jnp.ndarray, shape (B, N_max)
            Weights (typically 1/sigma_y^2) [dimensionless].
        mask : jnp.ndarray, shape (B, N_max)
            Boolean mask for valid lines (True = valid, False = padded).

        Returns
        -------
        BoltzmannFitResultJax
            Dataclass with slope, intercept, T_K, sigma_T, sigma_slope,
            sigma_intercept, R_squared, n_valid arrays of shape (B,).
        """
        # Ensure float64 for numerical stability
        x = jnp.asarray(x, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        w = jnp.asarray(w, dtype=jnp.float64)
        mask_f = jnp.asarray(mask, dtype=jnp.float64)

        # Zero out padded entries
        w_masked = w * mask_f

        # 5-sum closed-form WLS (DERV-02 Eq. 01-01.4)
        S_w = jnp.sum(w_masked, axis=1)  # (B,)
        S_wx = jnp.sum(w_masked * x, axis=1)  # (B,)
        S_wy = jnp.sum(w_masked * y, axis=1)  # (B,)
        S_wxx = jnp.sum(w_masked * x * x, axis=1)  # (B,)
        S_wxy = jnp.sum(w_masked * x * y, axis=1)  # (B,)

        # Determinant of normal equations
        det = S_w * S_wxx - S_wx**2  # (B,)

        # Safe division for degenerate cases (det ~ 0)
        det_safe = jnp.where(jnp.abs(det) > 1e-30, det, 1.0)
        is_valid = jnp.abs(det) > 1e-30

        # Slope and intercept
        slope = jnp.where(is_valid, (S_w * S_wxy - S_wx * S_wy) / det_safe, 0.0)
        intercept = jnp.where(is_valid, (S_wxx * S_wy - S_wx * S_wxy) / det_safe, 0.0)

        # Temperature from slope: T = -1 / (slope * k_B)
        slope_safe = jnp.where(slope < -1e-30, slope, -1e-30)
        T_K = jnp.where(is_valid & (slope < -1e-30), -1.0 / (slope_safe * KB_EV), 0.0)

        # Uncertainty from covariance matrix of normal equations
        # Var(slope) = S_w / det, Var(intercept) = S_wxx / det
        sigma_slope = jnp.where(is_valid, jnp.sqrt(jnp.abs(S_w / det_safe)), 0.0)
        sigma_intercept = jnp.where(is_valid, jnp.sqrt(jnp.abs(S_wxx / det_safe)), 0.0)

        # Temperature uncertainty via error propagation:
        # sigma_T = T^2 * k_B * sigma_slope
        sigma_T = jnp.where(is_valid, T_K**2 * KB_EV * sigma_slope, 0.0)

        # R-squared (weighted)
        y_pred = intercept[:, None] + slope[:, None] * x  # (B, N_max)
        SS_res = jnp.sum(w_masked * (y - y_pred) ** 2, axis=1)  # (B,)
        y_mean = jnp.where(S_w > 0, S_wy / jnp.maximum(S_w, 1e-30), 0.0)  # (B,)
        SS_tot = jnp.sum(w_masked * (y - y_mean[:, None]) ** 2, axis=1)  # (B,)
        R_squared = jnp.where(SS_tot > 1e-30, 1.0 - SS_res / SS_tot, 0.0)

        # Count valid lines
        n_valid = jnp.sum(mask_f, axis=1)  # (B,)

        return BoltzmannFitResultJax(
            slope=slope,
            intercept=intercept,
            T_K=T_K,
            sigma_T=sigma_T,
            sigma_slope=sigma_slope,
            sigma_intercept=sigma_intercept,
            R_squared=R_squared,
            n_valid=n_valid,
        )

    @jit
    def boltzmann_temperature_jax(
        x: jnp.ndarray,
        y: jnp.ndarray,
        w: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Convenience wrapper returning only temperature and uncertainty.

        Parameters
        ----------
        x : jnp.ndarray, shape (B, N_max)
            Upper-level energies E_k [eV].
        y : jnp.ndarray, shape (B, N_max)
            Boltzmann plot values [dimensionless].
        w : jnp.ndarray, shape (B, N_max)
            Weights [dimensionless].
        mask : jnp.ndarray, shape (B, N_max)
            Boolean mask for valid lines.

        Returns
        -------
        T_K : jnp.ndarray, shape (B,)
            Temperature in Kelvin.
        sigma_T : jnp.ndarray, shape (B,)
            1-sigma uncertainty in Kelvin.
        """
        result = batched_boltzmann_fit(x, y, w, mask)
        return result.T_K, result.sigma_T

else:

    def batched_boltzmann_fit(*args, **kwargs):  # type: ignore[misc]
        _raise_jax_missing()

    def boltzmann_temperature_jax(*args, **kwargs):  # type: ignore[misc]
        _raise_jax_missing()
