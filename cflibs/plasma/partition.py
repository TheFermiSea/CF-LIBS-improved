"""
Partition function evaluation logic.
"""

from typing import Any, List, Union
import numpy as np

try:
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None


def polynomial_partition_function(T_K: float, coefficients: List[float]) -> float:
    """
    Evaluate partition function using Irwin polynomial form.

    log(U) = sum(a_n * (log T)^n)

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin
    coefficients : List[float]
        Polynomial coefficients [a0, a1, a2, a3, a4]

    Returns
    -------
    float
        Partition function value U(T)
    """
    if T_K <= 1.0:
        return 1.0

    ln_T = np.log(T_K)
    ln_U = 0.0

    for i, a in enumerate(coefficients):
        ln_U += a * (ln_T**i)

    return np.exp(ln_U)


if HAS_JAX:

    @jit
    def polynomial_partition_function_jax(
        T_K: Union[float, jnp.ndarray], coefficients: jnp.ndarray
    ) -> jnp.ndarray:
        """
        JAX-compatible evaluation of partition function.

        Parameters
        ----------
        T_K : float or array
            Temperature in Kelvin
        coefficients : array
            Coefficients [a0, a1, a2, a3, a4]
            Can be shape (5,) or (N, 5)

        Returns
        -------
        array
            Partition function value U(T)
        """
        # Ensure T_K is not too close to zero to avoid log(0)
        # In practice T_eV > 0.4 checked in generator, so T_K > 4000
        ln_T = jnp.log(jnp.maximum(T_K, 1.0))

        # Expand dimensions if necessary for broadcasting
        # If coefficients is (N, 5) and T_K is scalar, result is (N,)

        # Calculate sum a_n * (ln T)^n
        # Manual expansion for 5 coefficients is fast and clear
        # Assuming coefficients shape ends in 5

        ln_U = (
            coefficients[..., 0]
            + coefficients[..., 1] * ln_T
            + coefficients[..., 2] * (ln_T**2)
            + coefficients[..., 3] * (ln_T**3)
            + coefficients[..., 4] * (ln_T**4)
        )

        return jnp.exp(ln_U)

else:

    def polynomial_partition_function_jax(*args, **kwargs):
        raise ImportError("JAX not installed")


class PartitionFunctionEvaluator:
    """Helper class for partition function evaluation."""

    @staticmethod
    def evaluate(T_K: float, coefficients: List[float]) -> float:
        """Evaluate using NumPy implementation."""
        return polynomial_partition_function(T_K, coefficients)

    @staticmethod
    def evaluate_jax(T_K: float, coefficients: Any) -> Any:
        """Evaluate using JAX implementation."""
        return polynomial_partition_function_jax(T_K, coefficients)

    @staticmethod
    def evaluate_batch(
        coefficients: np.ndarray,
        temperatures: np.ndarray,
    ) -> np.ndarray:
        """Evaluate partition functions for all species at all temperatures.

        Uses Rust acceleration when available, falls back to NumPy.

        Parameters
        ----------
        coefficients : np.ndarray
            Shape (N_species, N_coeffs) partition function coefficients.
        temperatures : np.ndarray
            Shape (N_temps,) temperatures in Kelvin.

        Returns
        -------
        np.ndarray
            Shape (N_species, N_temps) partition function values.
        """
        try:
            from cflibs._core import batch_partition_functions
        except ImportError:
            try:
                from _core import batch_partition_functions
            except ImportError:
                batch_partition_functions = None  # type: ignore[assignment]

        if batch_partition_functions is not None:
            return np.asarray(
                batch_partition_functions(
                    np.ascontiguousarray(coefficients, dtype=np.float64),
                    np.ascontiguousarray(temperatures, dtype=np.float64),
                )
            )

        # NumPy fallback
        n_species = coefficients.shape[0]
        n_temps = temperatures.shape[0]
        result = np.zeros((n_species, n_temps))
        for s in range(n_species):
            coeffs_s = list(coefficients[s])
            for t in range(n_temps):
                result[s, t] = polynomial_partition_function(float(temperatures[t]), coeffs_s)
        return result