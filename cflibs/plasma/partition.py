"""
Partition function evaluation logic.

Two computation methods are provided:

1. **Direct summation** (recommended): U(T) = Σ gᵢ exp(-Eᵢ / kT) over energy
   levels from the atomic database, with plasma-truncated cutoff at
   E_max = IP - Δχ(nₑ, T).  Based on Alimohamadi & Ferland (2022, PASP 134).

2. **Polynomial** (legacy): log U = Σ aₙ (log T)ⁿ (Irwin 1981 form).  Retained
   for backward compatibility but NOT recommended — errors up to 66% for some
   species.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from cflibs.core.constants import KB_EV

try:
    import jax.numpy as jnp
    from jax import jit, vmap

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None


# ---------------------------------------------------------------------------
# Direct summation from energy levels
# ---------------------------------------------------------------------------


def ionization_potential_depression(n_e: float, T_K: float, Z: int = 1) -> float:
    """Debye-Hückel ionization potential depression.

    Δχ = 3×10⁻⁸ · Z · Nₑ^(1/2) · T^(-1/2)  [eV]

    From Mihalas (1978), Eq. 9-106.  See also Alimohamadi & Ferland (2022),
    Eq. 13.

    Parameters
    ----------
    n_e : float
        Electron density in cm⁻³.
    T_K : float
        Temperature in Kelvin.
    Z : int
        Ionic charge of perturbers (default 1 for singly-ionized).

    Returns
    -------
    float
        IPD in eV.  Returns 0 if n_e <= 0 or T_K <= 0.
    """
    if n_e <= 0 or T_K <= 0:
        return 0.0
    return 3.0e-8 * Z * np.sqrt(n_e) / np.sqrt(T_K)


def direct_sum_partition_function(
    T_K: float,
    g_levels: np.ndarray,
    E_levels_ev: np.ndarray,
    ip_ev: float,
    n_e: Optional[float] = None,
) -> float:
    """Compute partition function by direct summation over energy levels.

    U(T) = Σ gᵢ exp(-Eᵢ / kT)   for  Eᵢ < IP - Δχ

    This is the standard method recommended by Alimohamadi & Ferland (2022)
    and used by NIST ASD.  The sum is truncated at the plasma-lowered
    ionization potential to exclude dissolved Rydberg states.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.
    g_levels : np.ndarray
        Statistical weights (degeneracies) for each energy level.
    E_levels_ev : np.ndarray
        Energy of each level in eV (measured from ground state).
    ip_ev : float
        Ionization potential in eV (from species_physics table).
    n_e : float, optional
        Electron density in cm⁻³.  If provided, applies Debye-Hückel
        IPD to further lower the cutoff.  If None, sharp IP cutoff only.

    Returns
    -------
    float
        Partition function U(T).  Guaranteed >= 1.0 (at least the
        ground state contributes).
    """
    if T_K <= 1.0:
        return max(float(g_levels[0]) if len(g_levels) > 0 else 1.0, 1.0)

    # Plasma-truncated cutoff
    delta_chi = ionization_potential_depression(n_e, T_K) if n_e is not None else 0.0
    e_max = ip_ev - delta_chi

    # Mask: include only levels below the effective ionization limit
    mask = E_levels_ev < e_max
    if not np.any(mask):
        return max(float(g_levels[0]) if len(g_levels) > 0 else 1.0, 1.0)

    kT_ev = KB_EV * T_K
    U = float(np.sum(g_levels[mask] * np.exp(-E_levels_ev[mask] / kT_ev)))
    return max(U, 1.0)


def direct_sum_partition_function_batch(
    temperatures_K: np.ndarray,
    g_levels: np.ndarray,
    E_levels_ev: np.ndarray,
    ip_ev: float,
    n_e: Optional[float] = None,
) -> np.ndarray:
    """Vectorized direct summation over an array of temperatures.

    Parameters
    ----------
    temperatures_K : np.ndarray
        Shape (N_temps,) temperatures in Kelvin.
    g_levels : np.ndarray
        Shape (N_levels,) statistical weights.
    E_levels_ev : np.ndarray
        Shape (N_levels,) level energies in eV.
    ip_ev : float
        Ionization potential in eV.
    n_e : float, optional
        Electron density for IPD.

    Returns
    -------
    np.ndarray
        Shape (N_temps,) partition function values.
    """
    delta_chi = ionization_potential_depression(n_e, temperatures_K.mean()) if n_e else 0.0
    e_max = ip_ev - delta_chi
    mask = E_levels_ev < e_max

    g_masked = g_levels[mask]
    E_masked = E_levels_ev[mask]

    if len(g_masked) == 0:
        return np.ones_like(temperatures_K)

    # Shape: (N_temps, N_levels) via broadcasting
    kT = KB_EV * temperatures_K[:, np.newaxis]  # (N_temps, 1)
    boltzmann = np.exp(-E_masked[np.newaxis, :] / kT)  # (N_temps, N_levels)
    U = np.sum(g_masked[np.newaxis, :] * boltzmann, axis=1)  # (N_temps,)
    return np.maximum(U, 1.0)


# ---------------------------------------------------------------------------
# JAX-compiled direct summation for manifold batch computation
# ---------------------------------------------------------------------------

if HAS_JAX:

    @jit
    def _direct_sum_single_temp(
        T_K: jnp.ndarray,
        g_levels: jnp.ndarray,
        E_levels_ev: jnp.ndarray,
        ip_ev: float,
    ) -> jnp.ndarray:
        """JIT-compiled direct summation for a single temperature.

        Uses ``jnp.where`` for masking instead of boolean indexing so the
        computation graph has a fixed shape and is fully traceable by JAX.

        Parameters
        ----------
        T_K : scalar jnp.ndarray
            Temperature in Kelvin.
        g_levels : jnp.ndarray
            Shape ``(N_levels,)`` statistical weights.
        E_levels_ev : jnp.ndarray
            Shape ``(N_levels,)`` level energies in eV.
        ip_ev : float
            Ionization potential in eV.

        Returns
        -------
        jnp.ndarray
            Scalar partition function value, floored at 1.0.
        """
        kT_ev = KB_EV * jnp.maximum(T_K, 1.0)
        boltzmann = g_levels * jnp.exp(-E_levels_ev / kT_ev)
        # Zero out levels above the ionization potential (JAX-friendly mask)
        masked = jnp.where(E_levels_ev < ip_ev, boltzmann, 0.0)
        return jnp.maximum(jnp.sum(masked), 1.0)

    def direct_sum_partition_function_jax(
        T_K,
        g_levels: jnp.ndarray,
        E_levels_ev: jnp.ndarray,
        ip_ev: float,
    ) -> jnp.ndarray:
        """JAX-compiled direct summation partition function.

        Works for scalar *or* array ``T_K``.  For array inputs the
        computation is automatically vectorised with ``jax.vmap``.

        Parameters
        ----------
        T_K : float or jnp.ndarray
            Temperature(s) in Kelvin.  Scalar or 1-D array.
        g_levels : jnp.ndarray
            Shape ``(N_levels,)`` statistical weights.
        E_levels_ev : jnp.ndarray
            Shape ``(N_levels,)`` level energies in eV.
        ip_ev : float
            Ionization potential in eV.

        Returns
        -------
        jnp.ndarray
            Partition function value(s).  Same shape as *T_K*.
        """
        T_K = jnp.asarray(T_K)
        if T_K.ndim == 0:
            return _direct_sum_single_temp(T_K, g_levels, E_levels_ev, ip_ev)
        # vmap over the temperature axis; g_levels, E_levels, ip_ev are shared
        batched = vmap(lambda t: _direct_sum_single_temp(t, g_levels, E_levels_ev, ip_ev))
        return batched(T_K)

    def direct_sum_partition_function_batch_jax(
        temperatures: jnp.ndarray,
        g_levels: jnp.ndarray,
        E_levels_ev: jnp.ndarray,
        ip_ev: float,
    ) -> jnp.ndarray:
        """Batch direct summation over temperatures using ``jax.vmap``.

        Convenience wrapper that always expects a 1-D temperature array
        and returns a 1-D result array of the same length.

        Parameters
        ----------
        temperatures : jnp.ndarray
            Shape ``(N_temps,)`` temperatures in Kelvin.
        g_levels : jnp.ndarray
            Shape ``(N_levels,)`` statistical weights.
        E_levels_ev : jnp.ndarray
            Shape ``(N_levels,)`` level energies in eV.
        ip_ev : float
            Ionization potential in eV.

        Returns
        -------
        jnp.ndarray
            Shape ``(N_temps,)`` partition function values.
        """
        batched = vmap(lambda t: _direct_sum_single_temp(t, g_levels, E_levels_ev, ip_ev))
        return batched(jnp.asarray(temperatures))

else:

    def direct_sum_partition_function_jax(*args, **kwargs):
        """Stub — JAX not installed."""
        raise ImportError("JAX is required for direct_sum_partition_function_jax")

    def direct_sum_partition_function_batch_jax(*args, **kwargs):
        """Stub — JAX not installed."""
        raise ImportError("JAX is required for direct_sum_partition_function_batch_jax")


# ---------------------------------------------------------------------------
# Energy level cache for partition function evaluation
# ---------------------------------------------------------------------------

# Module-level cache: {(db_path, element, stage): (g_array, E_array, ip_ev)}
_level_cache: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray, float]] = {}


def get_levels_for_species(
    atomic_db: Any,
    element: str,
    ionization_stage: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Load and cache energy level data for a species.

    Returns (g_array, E_array, ip_ev) or None if data is unavailable.
    Levels above the ionization potential are pre-filtered.
    """
    cache_key = (str(getattr(atomic_db, "db_path", id(atomic_db))), element, ionization_stage)
    if cache_key in _level_cache:
        return _level_cache[cache_key]

    # Query energy levels via public API
    try:
        levels = atomic_db.get_energy_levels(element, ionization_stage)
    except Exception:
        return None

    if not levels:
        return None

    # Get ionization potential
    ip = atomic_db.get_ionization_potential(element, ionization_stage)
    if ip is None:
        # Fallback: use max level energy + 1 eV as rough IP
        ip = max(lev.energy_ev for lev in levels) + 1.0

    g_arr = np.array([lev.g for lev in levels], dtype=np.float64)
    E_arr = np.array([lev.energy_ev for lev in levels], dtype=np.float64)

    # Sort by energy
    sort_idx = np.argsort(E_arr)
    g_arr = g_arr[sort_idx]
    E_arr = E_arr[sort_idx]

    # Pre-filter autoionizing levels (belt-and-suspenders with DB cleanup)
    below_ip = E_arr < ip
    g_arr = g_arr[below_ip]
    E_arr = E_arr[below_ip]

    result = (g_arr, E_arr, ip)
    _level_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Legacy polynomial partition functions
# ---------------------------------------------------------------------------


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
    """Partition function evaluation with direct summation (preferred) and polynomial fallback.

    The recommended workflow is:

    1. Call :meth:`evaluate_direct` with energy-level arrays from the database.
       This gives exact results with plasma-aware truncation.
    2. If energy levels are unavailable, fall back to :meth:`evaluate` with
       polynomial coefficients (legacy path, less accurate).
    """

    @staticmethod
    def evaluate_direct(
        T_K: float,
        g_levels: np.ndarray,
        E_levels_ev: np.ndarray,
        ip_ev: float,
        n_e: Optional[float] = None,
    ) -> float:
        """Evaluate via direct summation over energy levels (recommended).

        Parameters
        ----------
        T_K : float
            Temperature in Kelvin.
        g_levels, E_levels_ev : np.ndarray
            Statistical weights and energies from energy_levels table.
        ip_ev : float
            Ionization potential in eV.
        n_e : float, optional
            Electron density for Debye-Hückel IPD truncation.
        """
        return direct_sum_partition_function(T_K, g_levels, E_levels_ev, ip_ev, n_e)

    @staticmethod
    def evaluate(T_K: float, coefficients: List[float]) -> float:
        """Evaluate using polynomial coefficients (legacy fallback)."""
        return polynomial_partition_function(T_K, coefficients)

    @staticmethod
    def evaluate_jax(T_K: float, coefficients: Any) -> Any:
        """Evaluate using JAX polynomial implementation (legacy)."""
        return polynomial_partition_function_jax(T_K, coefficients)

    @staticmethod
    def evaluate_direct_batch(
        temperatures_K: np.ndarray,
        g_levels: np.ndarray,
        E_levels_ev: np.ndarray,
        ip_ev: float,
        n_e: Optional[float] = None,
    ) -> np.ndarray:
        """Vectorized direct summation over temperature array.

        Parameters
        ----------
        temperatures_K : np.ndarray
            Shape (N_temps,) temperatures in Kelvin.
        g_levels, E_levels_ev : np.ndarray
            Shape (N_levels,) statistical weights and energies.
        ip_ev : float
            Ionization potential in eV.
        n_e : float, optional
            Electron density for IPD.

        Returns
        -------
        np.ndarray
            Shape (N_temps,) partition function values.
        """
        return direct_sum_partition_function_batch(
            temperatures_K, g_levels, E_levels_ev, ip_ev, n_e
        )

    @staticmethod
    def evaluate_direct_jax(
        T_K,
        g_levels,
        E_levels_ev,
        ip_ev: float,
    ):
        """JAX-compiled direct summation (scalar or array temperature).

        Parameters
        ----------
        T_K : float or jnp.ndarray
            Temperature(s) in Kelvin.
        g_levels : jnp.ndarray
            Shape ``(N_levels,)`` statistical weights.
        E_levels_ev : jnp.ndarray
            Shape ``(N_levels,)`` level energies in eV.
        ip_ev : float
            Ionization potential in eV.

        Returns
        -------
        jnp.ndarray
            Partition function value(s).
        """
        return direct_sum_partition_function_jax(T_K, g_levels, E_levels_ev, ip_ev)

    @staticmethod
    def evaluate_direct_batch_jax(
        temperatures,
        g_levels,
        E_levels_ev,
        ip_ev: float,
    ):
        """Batch JAX direct summation over a temperature array via ``vmap``.

        Parameters
        ----------
        temperatures : jnp.ndarray
            Shape ``(N_temps,)`` temperatures in Kelvin.
        g_levels : jnp.ndarray
            Shape ``(N_levels,)`` statistical weights.
        E_levels_ev : jnp.ndarray
            Shape ``(N_levels,)`` level energies in eV.
        ip_ev : float
            Ionization potential in eV.

        Returns
        -------
        jnp.ndarray
            Shape ``(N_temps,)`` partition function values.
        """
        return direct_sum_partition_function_batch_jax(temperatures, g_levels, E_levels_ev, ip_ev)

    @staticmethod
    def evaluate_batch(
        coefficients: np.ndarray,
        temperatures: np.ndarray,
    ) -> np.ndarray:
        """Evaluate partition functions using polynomial coefficients (legacy batch).

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
