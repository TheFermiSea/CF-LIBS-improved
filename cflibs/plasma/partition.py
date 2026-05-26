"""
Partition function evaluation logic.

Two computation methods are provided:

1. **Direct summation** (recommended): U(T) = Σ gᵢ exp(-Eᵢ / kT) over energy
   levels from the atomic database, with plasma-truncated cutoff at
   E_max = IP - Δχ(nₑ, T).  Based on Alimohamadi & Ferland (2022, PASP 134).

2. **Polynomial** (Irwin 1981 form, NATURAL-LOG basis): the implementation
   evaluates::

       ln U(T) = a0 + a1·ln T + a2·(ln T)² + a3·(ln T)³ + a4·(ln T)⁴

   i.e. a 4th-order polynomial in ``ln T`` whose value is ``ln U``.  This is
   mathematically equivalent to Irwin (1981 ApJS 45 621) once a basis change
   ``log10 ↔ ln`` is applied to the coefficients — Irwin tabulated his fits
   in ``log10 T``/``log10 U``, but the present code stores natural-log
   coefficients (per the historical NIST-ASD-fit convention used by this
   project).  When ingesting Irwin's published Table II coefficients, run
   them through :func:`irwin_log10_to_ln_coeffs` first.

   Stored in ``partition_functions`` (a0..a4, t_min, t_max, source).  Less
   accurate than direct summation when energy levels are available
   (errors up to 66 % for some species) — use only as a fallback when the
   ``energy_levels`` table is missing rows for a species.

Historical note (2026-05-09 fix): the previous docstring ambiguously said
"log U = Σ aₙ (log T)ⁿ" without specifying base.  The implementation has
always used natural log (``np.log``), and the 13 partition_functions rows
in the production DB at the time of audit were fit to that convention.
The 30–60 % poly-vs-direct-sum discrepancy noted in the audit was caused
by *stale fit data* (the polynomial coefficients were fit against an
older energy_levels snapshot than the one currently in the DB), not by a
math/convention mismatch.  Re-fitting from the current EL table — which
is what :mod:`scripts.populate_partition_functions` does — restores the
consistency.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from cflibs.core.constants import KB_EV
from cflibs.core.jax_runtime import HAS_JAX, jit_if_available, jnp, vmap_if_available

jit = jit_if_available
vmap = vmap_if_available


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
    # IPD per-temperature, not at the batch mean. The Debye-Hückel IPD
    # depends on T as ~T^{-1/2}, so applying the mean-T cutoff to all
    # temperatures incorrectly excludes (or includes) levels just below
    # the cutoff at low- and high-T edges of the batch. The scalar path
    # at `direct_sum_partition_function` and the JAX path at
    # `_direct_sum_single_temp` already compute IPD per-T; the NumPy
    # batch path here was the only one with the mean(T) bug. Surfaced
    # 2026-05-19 by AI physics review; cuts the basis-library generator
    # error for non-mean-T plasma conditions.
    if n_e:
        delta_chis = np.array(
            [ionization_potential_depression(n_e, T) for T in temperatures_K]
        )  # (N_temps,)
    else:
        delta_chis = np.zeros_like(temperatures_K, dtype=float)
    e_maxes = ip_ev - delta_chis  # (N_temps,)
    # Per-temperature mask: shape (N_temps, N_levels).
    mask = E_levels_ev[np.newaxis, :] < e_maxes[:, np.newaxis]

    if not np.any(mask):
        return np.ones_like(temperatures_K)

    # Shape: (N_temps, N_levels) via broadcasting
    kT = KB_EV * temperatures_K[:, np.newaxis]  # (N_temps, 1)
    boltzmann = np.exp(-E_levels_ev[np.newaxis, :] / kT)  # (N_temps, N_levels)
    # Apply the per-temperature level mask so excluded levels contribute 0
    # to that temperature's U, but included levels still contribute at
    # neighboring temperatures.
    U = np.sum(
        g_levels[np.newaxis, :] * boltzmann * mask.astype(boltzmann.dtype),
        axis=1,
    )  # (N_temps,)
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


def get_ground_state_g(
    atomic_db: Any,
    element: str,
    ionization_stage: int,
    default: float = 1.0,
) -> float:
    """Look up the ground-state statistical weight g0 from the energy_levels table.

    Used as the physical lower bound for polynomial-fallback partition function
    evaluation (a partition function U(T) cannot be less than the ground state's
    degeneracy because the ground state always contributes a Boltzmann weight of
    1).  Falls back to ``default`` when the DB lookup fails for any reason
    (missing species, no levels, exception).

    Parameters
    ----------
    atomic_db : Any
        Atomic data source exposing ``get_energy_levels``.
    element : str
        Element symbol.
    ionization_stage : int
        Ionization stage (1 = neutral).
    default : float
        Value returned when no levels are available (default 1.0, since
        every level has g >= 1).

    Returns
    -------
    float
        Ground-state statistical weight, or ``default`` if unavailable.
    """
    try:
        levels = atomic_db.get_energy_levels(element, ionization_stage)
    except Exception:
        return default
    if not levels:
        return default
    # Lowest-energy level — sort by energy_ev just in case the DB iteration
    # order is not energy-sorted.
    try:
        g0 = min(levels, key=lambda lev: lev.energy_ev).g
    except Exception:
        return default
    return float(g0) if g0 and g0 > 0 else default


def polynomial_partition_function(
    T_K: float,
    coefficients: List[float],
    *,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    g0: Optional[float] = None,
) -> float:
    """
    Evaluate partition function using the Irwin (1981) polynomial form,
    *natural-log basis*.

    The function evaluates::

        ln U(T) = Σ_{n=0..4} a_n · (ln T)^n

    Note: the polynomial argument is the **natural** logarithm of T, and
    the polynomial value is the **natural** logarithm of U.  Irwin's
    published Table II uses base-10 logs; convert his coefficients with
    :func:`irwin_log10_to_ln_coeffs` before storing them as ``a_n`` here.

    Extrapolation guard (bead CF-LIBS-improved-s1qr.1, 2026-05-25): outside
    the fit domain ``[t_min, t_max]`` the polynomial is unconstrained and
    can blow up exponentially (e.g. Ca I U(100 000 K) = 1.14e5 vs the
    direct-sum truth of ~200, 560× wrong) or fall below the ground-state
    degeneracy (e.g. Nb I U(500 K) = 0.31 < g0=1).  When ``t_min``/``t_max``
    are supplied, the input temperature is clamped to that interval before
    evaluation; when ``g0`` is supplied, the result is floored at ``g0``.
    All three are keyword-only so legacy callers retain bit-identical
    behaviour.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin
    coefficients : List[float]
        Polynomial coefficients [a0, a1, a2, a3, a4], natural-log basis.
    t_min, t_max : float, optional
        Validity range of the polynomial fit (from the ``partition_functions``
        table).  When both are supplied, ``T_K`` is clamped to
        ``[t_min, t_max]`` prior to polynomial evaluation.  Passing one
        alone clamps on that side only.
    g0 : float, optional
        Ground-state degeneracy (statistical weight of the lowest energy
        level).  The returned value is floored at ``g0`` so the partition
        function cannot drop below the ground-state contribution, which is
        a strict physical lower bound.  Defaults to no floor when omitted
        (legacy behaviour).

    Returns
    -------
    float
        Partition function value U(T) = exp(Σ a_n (ln T)^n).

    See Also
    --------
    irwin_log10_to_ln_coeffs : Convert log10-basis Irwin coefficients
        to the natural-log basis used here.
    """
    if T_K <= 1.0:
        return 1.0 if g0 is None else max(1.0, float(g0))

    # Clamp T to the polynomial's validity window (if supplied) before
    # evaluating — extrapolating exp(quartic in ln T) outside [t_min, t_max]
    # is the root cause of the 560× Ca I error documented above.
    T_eval = float(T_K)
    if t_min is not None:
        T_eval = max(T_eval, float(t_min))
    if t_max is not None:
        T_eval = min(T_eval, float(t_max))

    ln_T = np.log(T_eval)
    ln_U = 0.0

    for i, a in enumerate(coefficients):
        ln_U += a * (ln_T**i)

    U = float(np.exp(ln_U))
    if g0 is not None:
        U = max(U, float(g0))
    return U


def irwin_log10_to_ln_coeffs(b: List[float]) -> List[float]:
    """Convert Irwin-style (log10/log10) coefficients to natural-log basis.

    If Irwin tabulates::

        log10 U(T) = Σ_{n=0..4} b_n · (log10 T)^n

    then because ln x = ln(10)·log10 x, the equivalent natural-log
    expansion ``ln U = Σ a_n (ln T)^n`` has::

        a_n = ln(10)^(1-n) · b_n

    Apply this once per published Irwin row before storing into the
    ``partition_functions`` table (whose stored a_n are natural-log).

    Parameters
    ----------
    b : list of float
        Irwin (1981) Table II coefficients [b0, b1, b2, b3, b4]
        in the log10 basis.

    Returns
    -------
    list of float
        Equivalent coefficients [a0, a1, a2, a3, a4] in the natural-log
        basis, suitable for direct insertion into ``partition_functions``.
    """
    ln10 = float(np.log(10.0))
    return [bn * ln10 ** (1 - n) for n, bn in enumerate(b)]


if HAS_JAX:

    @jit
    def _polynomial_partition_function_jax_core(
        T_K: Union[float, jnp.ndarray], coefficients: jnp.ndarray
    ) -> jnp.ndarray:
        """Inner JIT-compiled polynomial evaluation (no clamping)."""
        ln_T = jnp.log(jnp.maximum(T_K, 1.0))

        ln_U = (
            coefficients[..., 0]
            + coefficients[..., 1] * ln_T
            + coefficients[..., 2] * (ln_T**2)
            + coefficients[..., 3] * (ln_T**3)
            + coefficients[..., 4] * (ln_T**4)
        )

        return jnp.exp(ln_U)

    def polynomial_partition_function_jax(
        T_K: Union[float, jnp.ndarray],
        coefficients: jnp.ndarray,
        *,
        t_min: Optional[Union[float, jnp.ndarray]] = None,
        t_max: Optional[Union[float, jnp.ndarray]] = None,
        g0: Optional[Union[float, jnp.ndarray]] = None,
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
        t_min, t_max : float or array, optional
            Validity range of the polynomial fit.  When supplied, ``T_K`` is
            clamped to ``[t_min, t_max]`` before evaluation (broadcasts
            against ``T_K``).  See the NumPy twin for the rationale.
        g0 : float or array, optional
            Ground-state degeneracy.  Result is floored at ``g0`` (broadcasts
            against the output).

        Returns
        -------
        array
            Partition function value U(T)
        """
        # Clamp T to validity window (if provided).  Kwargs are evaluated
        # eagerly in Python so this wrapper stays jit-friendly without
        # needing static_argnames.
        T_clamped = T_K
        if t_min is not None and t_max is not None:
            T_clamped = jnp.clip(T_K, t_min, t_max)
        elif t_min is not None:
            T_clamped = jnp.maximum(T_K, t_min)
        elif t_max is not None:
            T_clamped = jnp.minimum(T_K, t_max)

        U = _polynomial_partition_function_jax_core(T_clamped, coefficients)
        if g0 is not None:
            U = jnp.maximum(U, g0)
        return U

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
    def evaluate(
        T_K: float,
        coefficients: List[float],
        *,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        g0: Optional[float] = None,
    ) -> float:
        """Evaluate using polynomial coefficients (legacy fallback).

        Forwards optional ``t_min``/``t_max``/``g0`` extrapolation guards to
        :func:`polynomial_partition_function`.
        """
        return polynomial_partition_function(T_K, coefficients, t_min=t_min, t_max=t_max, g0=g0)

    @staticmethod
    def evaluate_jax(
        T_K: float,
        coefficients: Any,
        *,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        g0: Optional[float] = None,
    ) -> Any:
        """Evaluate using JAX polynomial implementation (legacy).

        Forwards optional ``t_min``/``t_max``/``g0`` extrapolation guards to
        :func:`polynomial_partition_function_jax`.
        """
        return polynomial_partition_function_jax(T_K, coefficients, t_min=t_min, t_max=t_max, g0=g0)

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
