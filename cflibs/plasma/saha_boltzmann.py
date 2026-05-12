"""
Saha-Boltzmann solver for LTE plasma.
"""

import threading
from typing import Dict, Optional, Protocol, Tuple

import numpy as np

from cflibs.core.constants import C_LIGHT, E_CHARGE, EV_TO_K, J_TO_EV, KB, SAHA_CONST_CM3
from cflibs.core.jax_runtime import HAS_JAX, jit_if_available, jnp
from cflibs.plasma.state import SingleZoneLTEPlasma
from cflibs.core.abc import SolverStrategy, AtomicDataSource
from cflibs.core.cache import cached_partition_function
from cflibs.core.logging_config import get_logger
from cflibs.plasma.partition import PartitionFunctionEvaluator

jit = jit_if_available


class IPDModel(Protocol):
    """Protocol for Ionization Potential Depression (IPD) models."""

    def calculate_lowering(self, n_e_cm3: float, T_K: float) -> float: ...


class DebyeHuckelIPD:
    """Debye-Hückel model for IPD."""

    def calculate_lowering(self, n_e_cm3: float, T_K: float) -> float:
        return ionization_potential_lowering(n_e_cm3, T_K)


logger = get_logger("plasma.saha_boltzmann")
_MISSING_LEVEL_WARNED: set[tuple[str, int]] = set()
_MISSING_LEVEL_WARNED_LOCK = threading.Lock()


class SahaBoltzmannSolver(SolverStrategy):
    """
    Solves Saha-Boltzmann equations for LTE plasma.

    For each element, solves:
    1. Saha equation for ionization balance
    2. Boltzmann distribution for level populations
    """

    def __init__(self, atomic_db: AtomicDataSource, ipd_model: Optional[IPDModel] = None):
        """
        Initialize solver.

        Parameters
        ----------
        atomic_db : AtomicDatabase
            Atomic database for ionization potentials and energy levels
        ipd_model : IPDModel, optional
            Strategy for calculating Ionization Potential Depression.
            Defaults to Debye-Hückel if None.
        """
        self.atomic_db = atomic_db
        self.ipd_model = ipd_model if ipd_model is not None else DebyeHuckelIPD()

    def solve_ionization_balance(
        self, element: str, T_e_eV: float, n_e_cm3: float, total_density_cm3: float
    ) -> Dict[int, float]:
        """
        Solve ionization balance using Saha equation.

        Parameters
        ----------
        element : str
            Element symbol
        T_e_eV : float
            Electron temperature in eV
        n_e_cm3 : float
            Electron density in cm^-3
        total_density_cm3 : float
            Total element density (sum over all ionization stages) in cm^-3

        Returns
        -------
        Dict[int, float]
            Dictionary mapping ionization stage to number density in cm^-3
        """
        # Get ionization potentials
        ip_I = self.atomic_db.get_ionization_potential(element, 1)
        ip_II = self.atomic_db.get_ionization_potential(element, 2)

        if ip_I is None:
            logger.warning(f"No ionization potential for {element} I, assuming neutral only")
            return {1: total_density_cm3}

        # Calculate Ionization Potential Depression (IPD)
        # In high-density LIBS plasmas, IPD lowers the effective ionization energy.
        T_K = T_e_eV * EV_TO_K
        delta_chi = self.ipd_model.calculate_lowering(n_e_cm3, T_K)

        eff_ip_I = max(ip_I - delta_chi, 0.0)

        # Saha equation: n_{z+1} * n_e / n_z = const * T^1.5 * (U_{z+1}/U_z) * exp(-IP/kT)
        # For neutral to singly ionized:
        # n_II * n_e / n_I = SAHA_CONST * T^1.5 * (U_II/U_I) * exp(-IP_I/kT)

        U_I = self.calculate_partition_function(element, 1, T_e_eV, max_energy_ev=eff_ip_I)

        eff_ip_II = max(ip_II - delta_chi, 0.0) if ip_II is not None else None
        U_II = self.calculate_partition_function(element, 2, T_e_eV, max_energy_ev=eff_ip_II)

        if U_I <= 0.0 or U_II <= 0.0:
            logger.warning(
                "Non-positive partition function for %s (U_I=%g, U_II=%g); "
                "falling back to neutral-only.",
                element,
                U_I,
                U_II,
            )
            return {1: total_density_cm3}

        S1 = (SAHA_CONST_CM3 / n_e_cm3) * (T_e_eV**1.5) * (U_II / U_I) * np.exp(-eff_ip_I / T_e_eV)

        # Solve the coupled system:
        #   n_II / n_I = S1
        #   n_III / n_II = S2  (if available)
        #   n_total = n_I + n_II + n_III
        #
        # => n_I = n_total / (1 + S1 + S1*S2)
        # => n_II = S1 * n_I
        # => n_III = S2 * n_II

        S2 = 0.0
        if ip_II is not None:
            ip_III = self.atomic_db.get_ionization_potential(element, 3)
            eff_ip_III = max(ip_III - delta_chi, 0.0) if ip_III is not None else None
            U_III = self.calculate_partition_function(element, 3, T_e_eV, max_energy_ev=eff_ip_III)
            S2 = (
                (SAHA_CONST_CM3 / n_e_cm3)
                * (T_e_eV**1.5)
                * (U_III / U_II)
                * np.exp(-eff_ip_II / T_e_eV)
            )

        denom = 1.0 + S1 + S1 * S2
        n_I = total_density_cm3 / denom
        n_II = S1 * n_I
        n_III = S2 * n_II

        result = {1: n_I, 2: n_II}
        if n_III > total_density_cm3 * 1e-6:
            result[3] = n_III

        return result

    @cached_partition_function
    def calculate_partition_function(
        self,
        element: str,
        ionization_stage: int,
        T_e_eV: float,
        max_energy_ev: float | None = None,
    ) -> float:
        """
        Calculate partition function for a species.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage
        T_e_eV : float
            Electron temperature in eV
        max_energy_ev : float or None
            Maximum energy level to include.  When ``None`` (default), the
            ionization potential of the species is used as the physical cap
            (scaled to 98 % of IP to exclude autoionizing levels).

        Returns
        -------
        float
            Partition function
        """
        # Preferred: direct summation from cached energy-level arrays with IP cutoff
        from cflibs.plasma.partition import get_levels_for_species

        T_K = T_e_eV * EV_TO_K
        level_data = get_levels_for_species(self.atomic_db, element, ionization_stage)
        if level_data is not None:
            g_arr, E_arr, ip_ev = level_data
            return PartitionFunctionEvaluator.evaluate_direct(T_K, g_arr, E_arr, ip_ev)

        # Fallback 1: polynomial coefficients
        if hasattr(self.atomic_db, "get_partition_coefficients"):
            pf = self.atomic_db.get_partition_coefficients(element, ionization_stage)
            if pf is not None:
                return PartitionFunctionEvaluator.evaluate(T_K, pf.coefficients)

        # Fallback 2: manual summation over EnergyLevel objects
        levels = self.atomic_db.get_energy_levels(element, ionization_stage)

        if not levels:
            key = (element, ionization_stage)
            with _MISSING_LEVEL_WARNED_LOCK:
                should_warn = key not in _MISSING_LEVEL_WARNED
                if should_warn:
                    _MISSING_LEVEL_WARNED.add(key)
            if should_warn:
                logger.warning(
                    "No energy levels for %s %s, using approximation", element, ionization_stage
                )
            return 2.0

        if max_energy_ev is None:
            ip = self.atomic_db.get_ionization_potential(element, ionization_stage)
            max_energy_ev = ip * 0.98 if ip else 50.0

        U = 0.0
        for level in levels:
            if level.energy_ev <= max_energy_ev:
                U += level.g * np.exp(-level.energy_ev / T_e_eV)

        return U

    def get_ionization_fractions(
        self, element: str, T_e_eV: float, n_e_cm3: float
    ) -> Dict[int, float]:
        """
        Compute ionization stage fractions for an element.

        Returns fractional populations (summing to 1.0) for each ionization
        stage, useful for comparison with NIST LIBS simulation values.

        Parameters
        ----------
        element : str
            Element symbol (e.g. "Fe")
        T_e_eV : float
            Electron temperature in eV
        n_e_cm3 : float
            Electron density in cm^-3

        Returns
        -------
        Dict[int, float]
            Mapping from ionization stage (1=neutral, 2=singly ionized, ...)
            to fractional population (0-1).
        Raises
        ------
        ValueError
            If T_e_eV <= 0, n_e_cm3 <= 0, or solver returns non-positive total.
        """
        if T_e_eV <= 0.0:
            raise ValueError(f"T_e_eV must be positive; got {T_e_eV!r}")
        if n_e_cm3 <= 0.0:
            raise ValueError(f"n_e_cm3 must be positive; got {n_e_cm3!r}")

        # Use arbitrary total density; fractions are independent of it
        total_density = 1.0
        stage_densities = self.solve_ionization_balance(element, T_e_eV, n_e_cm3, total_density)
        total = sum(stage_densities.values())
        if total <= 0.0:
            raise ValueError(
                f"Ionization balance returned non-positive total density "
                f"({total!r}) for element {element!r}"
            )
        return {stage: n / total for stage, n in stage_densities.items()}

    def solve_level_population(
        self,
        element: str,
        ionization_stage: int,
        stage_density_cm3: float,
        T_e_eV: float,
        n_e_cm3: Optional[float] = None,
    ) -> Dict[Tuple[str, int, float], float]:
        """
        Solve Boltzmann distribution for level populations.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage
        stage_density_cm3 : float
            Total density of this ionization stage in cm^-3
        T_e_eV : float
            Electron temperature in eV
        n_e_cm3 : float, optional
            Electron density in cm^-3 (used to compute IPD cutoff)

        Returns
        -------
        Dict[Tuple[str, int, float], float]
            Dictionary mapping (element, stage, energy_ev) to population density
        """
        # Determine max energy based on IPD if electron density is provided
        ip = self.atomic_db.get_ionization_potential(element, ionization_stage)
        if n_e_cm3 is not None and ip is not None:
            T_K = T_e_eV * EV_TO_K
            delta_chi = self.ipd_model.calculate_lowering(n_e_cm3, T_K)
            max_energy_ev = max(ip - delta_chi, 0.0)
        else:
            max_energy_ev = ip * 0.98 if ip else 50.0

        U = self.calculate_partition_function(
            element, ionization_stage, T_e_eV, max_energy_ev=max_energy_ev
        )

        if U <= 0.0:
            logger.warning(
                "Non-positive partition function for %s stage %d at T_e_eV=%g; "
                "returning empty level populations.",
                element,
                ionization_stage,
                T_e_eV,
            )
            return {}

        levels = self.atomic_db.get_energy_levels(element, ionization_stage)

        populations = {}
        for level in levels:
            # Exclude levels above the lowered IP as they merged into the continuum
            if level.energy_ev > max_energy_ev:
                continue
            # Boltzmann: n_i = n_total * (g_i / U) * exp(-E_i / kT)
            n_i = stage_density_cm3 * (level.g / U) * np.exp(-level.energy_ev / T_e_eV)
            key = (element, ionization_stage, level.energy_ev)
            populations[key] = n_i

        return populations

    def solve_plasma(self, plasma: SingleZoneLTEPlasma) -> Dict[Tuple[str, int, float], float]:
        """
        Solve complete Saha-Boltzmann system for a plasma.

        Parameters
        ----------
        plasma : SingleZoneLTEPlasma
            Plasma state

        Returns
        -------
        Dict[Tuple[str, int, float], float]
            Dictionary mapping (element, stage, energy_ev) to population density in cm^-3
        """
        T_e_eV = plasma.T_e_eV
        n_e_cm3 = plasma.n_e

        all_populations = {}

        for element, total_density in plasma.species.items():
            # Solve ionization balance
            stage_densities = self.solve_ionization_balance(element, T_e_eV, n_e_cm3, total_density)

            # Solve level populations for each stage
            for stage, stage_density in stage_densities.items():
                if stage_density > 0:
                    populations = self.solve_level_population(
                        element, stage, stage_density, T_e_eV, n_e_cm3
                    )
                    all_populations.update(populations)

        logger.debug(f"Solved Saha-Boltzmann for {len(plasma.species)} species")
        return all_populations


# ---------------------------------------------------------------------------
# Ionization Potential Depression (IPD)
# ---------------------------------------------------------------------------

# Derived CGS helpers built from the canonical SI constants in cflibs.core.constants
_E_ESU = E_CHARGE * C_LIGHT * 10.0  # electron charge [esu = statcoulomb]
_KB_ERG = KB * 1.0e7  # Boltzmann constant [erg/K]
_ERG_TO_EV = J_TO_EV * 1.0e-7  # conversion factor erg -> eV


def ionization_potential_lowering(
    n_e_cm3: float,
    T_K: float,
    model: str = "debye_huckel",
) -> float:
    """
    Compute the reduction in effective ionization potential due to plasma screening.

    At high electron densities (n_e ~ 10^17 cm^-3), the Coulomb potential is
    screened by the plasma, effectively lowering the ionization threshold.
    Ignoring this effect leads to a small but systematic bias in the Saha
    equation.

    **Debye-Hückel model** (``model='debye_huckel'``):

    .. math::

        \\Delta\\chi = \\frac{e^2}{4\\pi\\varepsilon_0 \\lambda_D}

    where the Debye length is:

    .. math::

        \\lambda_D = \\sqrt{\\frac{k_B T}{4\\pi n_e e^2}}

    In Gaussian CGS this simplifies to:

    .. math::

        \\Delta\\chi = e^2 \\sqrt{\\frac{4\\pi n_e}{k_B T}}

    Parameters
    ----------
    n_e_cm3 : float
        Electron density [cm^-3]
    T_K : float
        Plasma temperature [K]
    model : str
        IPD model. Currently only ``'debye_huckel'`` is supported.

    Returns
    -------
    float
        Ionization potential depression [eV]. Always >= 0.

    Raises
    ------
    ValueError
        If an unsupported model name is given.

    Examples
    --------
    >>> delta_chi = ionization_potential_lowering(1e17, 10000)
    >>> 0.03 <= delta_chi <= 0.06  # ~0.04 eV at canonical LIBS conditions
    True

    References
    ----------
    - Stewart, J.C. & Pyatt, K.D. (1966) ApJ 144, 1203
    - Kramida, A. et al. NIST Atomic Spectra Database
    """
    if model != "debye_huckel":
        raise ValueError(f"Unsupported IPD model: {model!r}. Use 'debye_huckel'.")

    if n_e_cm3 <= 0.0 or T_K <= 0.0:
        return 0.0

    # Debye length: lambda_D = sqrt(kT / (4*pi*n_e*e^2))  [cm]
    lambda_D = np.sqrt(_KB_ERG * T_K / (4.0 * np.pi * n_e_cm3 * _E_ESU**2))

    # IPD = e^2 / lambda_D  [erg]
    delta_chi_erg = _E_ESU**2 / lambda_D

    # Convert to eV
    return float(delta_chi_erg * _ERG_TO_EV)


# ---------------------------------------------------------------------------
# JAX-accelerated Saha-Boltzmann solver
# ---------------------------------------------------------------------------
#
# This is a drop-in companion to ``SahaBoltzmannSolver`` that pushes the hot
# path (partition-function direct summation, ionization-balance arithmetic
# and Boltzmann level populations) onto ``jax.numpy`` arrays so JAX's XLA
# backend can dispatch the work to the active accelerator (GPU/TPU/CPU).
#
# Design:
#  - Atomic data (energy levels, ionization potentials, transitions) is
#    fetched from the existing ``AtomicDataSource`` once per element/stage
#    and cached as ``jnp.ndarray`` lookup tables outside any ``jit``
#    boundary. This matches the convention used by
#    ``cflibs/inversion/physics/boltzmann_jax.py`` and
#    ``cflibs/plasma/partition.py::_direct_sum_single_temp``.
#  - The number-crunching kernels (``_saha_balance_kernel``,
#    ``_boltzmann_populations_kernel``) are pure functions of jnp arrays and
#    Python scalars only, so they can be jit-compiled once and re-used.
#  - The public ``solve_*`` API mirrors ``SahaBoltzmannSolver`` and returns
#    plain Python floats / dicts so callers (e.g. ``SpectrumModelJax``) can
#    feed the results into NumPy emissivity routines without leaking traced
#    values across the JAX/NumPy boundary.

# Pre-compute the IPD prefactor so we never call ``np.sqrt`` inside jit.
# delta_chi_eV = e^3 * sqrt(4*pi*n_e/(k_B*T)) (Gaussian CGS) * erg->eV
_IPD_PREFACTOR_EV_CM_K = float(_E_ESU**3 * np.sqrt(4.0 * np.pi / _KB_ERG) * _ERG_TO_EV)


def _ipd_eV(n_e_cm3: float, T_K: float) -> float:
    """Pure-Python (numpy) Debye-Hückel IPD in eV — kept out of jit boundary.

    Mirrors :func:`ionization_potential_lowering` but inlined as a fast
    numpy expression for use outside jit boundaries.
    """
    if n_e_cm3 <= 0.0 or T_K <= 0.0:
        return 0.0
    return float(_IPD_PREFACTOR_EV_CM_K * np.sqrt(n_e_cm3 / T_K))


if HAS_JAX:

    @jit
    def _partition_sum_jax(
        T_e_eV: jnp.ndarray,
        g_arr: jnp.ndarray,
        E_arr: jnp.ndarray,
        max_energy_ev: jnp.ndarray,
    ) -> jnp.ndarray:
        """U(T) = Σ gᵢ exp(-Eᵢ / T_eV) over levels with Eᵢ < max_energy_ev.

        All inputs are jnp arrays / scalars to keep the kernel jit-friendly
        with a fixed computation graph. The mask is applied by zeroing out
        contributions above ``max_energy_ev`` rather than boolean indexing.
        """
        kT = jnp.maximum(T_e_eV, 1e-12)
        boltzmann = g_arr * jnp.exp(-E_arr / kT)
        contrib = jnp.where(E_arr < max_energy_ev, boltzmann, 0.0)
        return jnp.maximum(jnp.sum(contrib), 1.0)

    @jit
    def _saha_balance_kernel(
        T_e_eV: jnp.ndarray,
        n_e_cm3: jnp.ndarray,
        eff_ip_I: jnp.ndarray,
        eff_ip_II: jnp.ndarray,
        U_I: jnp.ndarray,
        U_II: jnp.ndarray,
        U_III: jnp.ndarray,
        has_II: jnp.ndarray,
        total_density_cm3: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve the 3-stage Saha balance, returning (n_I, n_II, n_III).

        Mirrors the closed-form expression in ``solve_ionization_balance``:

            S1 = (SAHA / n_e) * T^1.5 * (U_II/U_I) * exp(-IP_I / T_eV)
            S2 = (SAHA / n_e) * T^1.5 * (U_III/U_II) * exp(-IP_II / T_eV)
            n_I = n_total / (1 + S1 + S1*S2)
            n_II = S1 * n_I
            n_III = S2 * n_II
        """
        T15 = T_e_eV**1.5
        prefactor = SAHA_CONST_CM3 / jnp.maximum(n_e_cm3, 1e-30) * T15

        S1 = prefactor * (U_II / jnp.maximum(U_I, 1e-30)) * jnp.exp(-eff_ip_I / T_e_eV)
        S2_raw = prefactor * (U_III / jnp.maximum(U_II, 1e-30)) * jnp.exp(-eff_ip_II / T_e_eV)
        # Only include S2 when stage II ionization potential is known.
        S2 = jnp.where(has_II > 0.5, S2_raw, 0.0)

        denom = 1.0 + S1 + S1 * S2
        n_I = total_density_cm3 / denom
        n_II = S1 * n_I
        n_III = S2 * n_II
        return jnp.stack([n_I, n_II, n_III])

    @jit
    def _boltzmann_populations_kernel(
        T_e_eV: jnp.ndarray,
        stage_density_cm3: jnp.ndarray,
        g_arr: jnp.ndarray,
        E_arr: jnp.ndarray,
        max_energy_ev: jnp.ndarray,
        U: jnp.ndarray,
    ) -> jnp.ndarray:
        """nᵢ = n_stage * (gᵢ / U) * exp(-Eᵢ / T_eV) for Eᵢ ≤ max_energy_ev.

        Returns an array shaped like ``g_arr``; entries above the IPD cutoff
        are zeroed so callers can mask them out in NumPy land.
        """
        boltzmann = jnp.exp(-E_arr / jnp.maximum(T_e_eV, 1e-12))
        pop = stage_density_cm3 * (g_arr / jnp.maximum(U, 1e-30)) * boltzmann
        return jnp.where(E_arr <= max_energy_ev, pop, 0.0)

else:  # pragma: no cover - JAX should be installed in this repo

    def _partition_sum_jax(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")

    def _saha_balance_kernel(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")

    def _boltzmann_populations_kernel(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("JAX is not installed; install jax + jaxlib")


class SahaBoltzmannSolverJax(SolverStrategy):
    """JAX-accelerated drop-in companion to :class:`SahaBoltzmannSolver`.

    The public method signatures match the NumPy version so existing callers
    can swap implementations without changes. The hot inner kernels run on
    ``jax.numpy`` and are jit-compiled the first time each shape is seen,
    so subsequent calls hit the GPU directly when ``JAX_PLATFORMS=cuda``.

    Numerical equivalence with :class:`SahaBoltzmannSolver` is asserted by
    the test suite (``tests/plasma/test_saha_boltzmann_jax.py``) within
    ``rtol=1e-5, atol=1e-7``.
    """

    def __init__(
        self,
        atomic_db: AtomicDataSource,
        ipd_model: Optional[IPDModel] = None,
    ):
        if not HAS_JAX:  # pragma: no cover - defensive
            raise ImportError(
                "SahaBoltzmannSolverJax requires JAX. Install with `pip install jax`."
            )
        self.atomic_db = atomic_db
        self.ipd_model = ipd_model if ipd_model is not None else DebyeHuckelIPD()
        # Cache of (element, stage) -> (g_jnp, E_jnp, ip_ev_float)
        self._level_cache: Dict[Tuple[str, int], Tuple[jnp.ndarray, jnp.ndarray, float]] = {}

    # -- atomic data lookup (NumPy land) --------------------------------------

    def _get_levels_jax(
        self, element: str, ionization_stage: int
    ) -> Optional[Tuple["jnp.ndarray", "jnp.ndarray", float]]:
        """Fetch (g, E, ip_ev) as jnp arrays, cached per species."""
        key = (element, ionization_stage)
        if key in self._level_cache:
            return self._level_cache[key]

        from cflibs.plasma.partition import get_levels_for_species

        level_data = get_levels_for_species(self.atomic_db, element, ionization_stage)
        if level_data is None:
            return None
        g_arr, E_arr, ip_ev = level_data
        g_jnp = jnp.asarray(np.asarray(g_arr, dtype=np.float64))
        E_jnp = jnp.asarray(np.asarray(E_arr, dtype=np.float64))
        cached = (g_jnp, E_jnp, float(ip_ev))
        self._level_cache[key] = cached
        return cached

    def calculate_partition_function(
        self,
        element: str,
        ionization_stage: int,
        T_e_eV: float,
        max_energy_ev: float | None = None,
    ) -> float:
        """Direct-sum partition function evaluated with JAX.

        The call returns a plain Python ``float`` — the JAX device array is
        materialized via ``float()`` at the boundary so caller-side caches
        (``cached_partition_function``) and downstream NumPy code see a
        regular scalar.
        """
        levels = self._get_levels_jax(element, ionization_stage)
        if levels is None:
            # Fall back to the polynomial / energy-level NumPy path used by
            # the original solver. This is intentionally NOT jit-compiled
            # because it depends on optional database fields.
            T_K = T_e_eV * EV_TO_K
            if hasattr(self.atomic_db, "get_partition_coefficients"):
                pf = self.atomic_db.get_partition_coefficients(element, ionization_stage)
                if pf is not None:
                    return PartitionFunctionEvaluator.evaluate(T_K, pf.coefficients)
            energy_levels = self.atomic_db.get_energy_levels(element, ionization_stage)
            if not energy_levels:
                return 2.0
            if max_energy_ev is None:
                ip = self.atomic_db.get_ionization_potential(element, ionization_stage)
                max_energy_ev = ip * 0.98 if ip else 50.0
            U = 0.0
            for level in energy_levels:
                if level.energy_ev <= max_energy_ev:
                    U += level.g * np.exp(-level.energy_ev / T_e_eV)
            return float(U)

        g_jnp, E_jnp, ip_ev = levels
        if max_energy_ev is None:
            max_energy_ev = ip_ev * 0.98
        U = _partition_sum_jax(
            jnp.asarray(float(T_e_eV)),
            g_jnp,
            E_jnp,
            jnp.asarray(float(max_energy_ev)),
        )
        return float(U)

    # -- ionization balance ---------------------------------------------------

    def solve_ionization_balance(
        self, element: str, T_e_eV: float, n_e_cm3: float, total_density_cm3: float
    ) -> Dict[int, float]:
        ip_I = self.atomic_db.get_ionization_potential(element, 1)
        if ip_I is None:
            logger.warning(f"No ionization potential for {element} I, assuming neutral only")
            return {1: total_density_cm3}

        T_K = T_e_eV * EV_TO_K
        delta_chi = self.ipd_model.calculate_lowering(n_e_cm3, T_K)
        eff_ip_I = max(ip_I - delta_chi, 0.0)

        ip_II = self.atomic_db.get_ionization_potential(element, 2)
        eff_ip_II = max(ip_II - delta_chi, 0.0) if ip_II is not None else 0.0
        has_II = ip_II is not None

        U_I = self.calculate_partition_function(element, 1, T_e_eV, max_energy_ev=eff_ip_I)
        U_II = self.calculate_partition_function(
            element, 2, T_e_eV, max_energy_ev=(eff_ip_II if has_II else None)
        )
        U_III = 1.0
        if has_II:
            ip_III = self.atomic_db.get_ionization_potential(element, 3)
            eff_ip_III = max(ip_III - delta_chi, 0.0) if ip_III is not None else None
            U_III = self.calculate_partition_function(
                element, 3, T_e_eV, max_energy_ev=eff_ip_III
            )

        if U_I <= 0.0 or U_II <= 0.0:
            logger.warning(
                "Non-positive partition function for %s (U_I=%g, U_II=%g); "
                "falling back to neutral-only.",
                element,
                U_I,
                U_II,
            )
            return {1: total_density_cm3}

        densities = _saha_balance_kernel(
            jnp.asarray(float(T_e_eV)),
            jnp.asarray(float(n_e_cm3)),
            jnp.asarray(float(eff_ip_I)),
            jnp.asarray(float(eff_ip_II)),
            jnp.asarray(float(U_I)),
            jnp.asarray(float(U_II)),
            jnp.asarray(float(U_III)),
            jnp.asarray(1.0 if has_II else 0.0),
            jnp.asarray(float(total_density_cm3)),
        )
        n_I, n_II, n_III = (float(x) for x in np.asarray(densities))

        result = {1: n_I, 2: n_II}
        if n_III > total_density_cm3 * 1e-6:
            result[3] = n_III
        return result

    def get_ionization_fractions(
        self, element: str, T_e_eV: float, n_e_cm3: float
    ) -> Dict[int, float]:
        if T_e_eV <= 0.0:
            raise ValueError(f"T_e_eV must be positive; got {T_e_eV!r}")
        if n_e_cm3 <= 0.0:
            raise ValueError(f"n_e_cm3 must be positive; got {n_e_cm3!r}")
        total_density = 1.0
        stage_densities = self.solve_ionization_balance(element, T_e_eV, n_e_cm3, total_density)
        total = sum(stage_densities.values())
        if total <= 0.0:
            raise ValueError(
                f"Ionization balance returned non-positive total density "
                f"({total!r}) for element {element!r}"
            )
        return {stage: n / total for stage, n in stage_densities.items()}

    # -- level populations ----------------------------------------------------

    def solve_level_population(
        self,
        element: str,
        ionization_stage: int,
        stage_density_cm3: float,
        T_e_eV: float,
        n_e_cm3: Optional[float] = None,
    ) -> Dict[Tuple[str, int, float], float]:
        ip = self.atomic_db.get_ionization_potential(element, ionization_stage)
        if n_e_cm3 is not None and ip is not None:
            T_K = T_e_eV * EV_TO_K
            delta_chi = self.ipd_model.calculate_lowering(n_e_cm3, T_K)
            max_energy_ev = max(ip - delta_chi, 0.0)
        else:
            max_energy_ev = ip * 0.98 if ip else 50.0

        U = self.calculate_partition_function(
            element, ionization_stage, T_e_eV, max_energy_ev=max_energy_ev
        )
        if U <= 0.0:
            logger.warning(
                "Non-positive partition function for %s stage %d at T_e_eV=%g; "
                "returning empty level populations.",
                element,
                ionization_stage,
                T_e_eV,
            )
            return {}

        # Use the database-side energy_levels list (preserves ordering and
        # energy-bucket key conventions used by the spectrum model).
        energy_levels = self.atomic_db.get_energy_levels(element, ionization_stage)
        if not energy_levels:
            return {}

        g_np = np.asarray([lev.g for lev in energy_levels], dtype=np.float64)
        E_np = np.asarray([lev.energy_ev for lev in energy_levels], dtype=np.float64)

        pops = _boltzmann_populations_kernel(
            jnp.asarray(float(T_e_eV)),
            jnp.asarray(float(stage_density_cm3)),
            jnp.asarray(g_np),
            jnp.asarray(E_np),
            jnp.asarray(float(max_energy_ev)),
            jnp.asarray(float(U)),
        )
        pops_np = np.asarray(pops)

        populations: Dict[Tuple[str, int, float], float] = {}
        for level, n_i in zip(energy_levels, pops_np):
            # Skip masked-out levels (kernel zeroed them); spectrum_model's
            # key bucketing relies on them being absent from the dict.
            if level.energy_ev > max_energy_ev:
                continue
            populations[(element, ionization_stage, level.energy_ev)] = float(n_i)
        return populations

    def solve_plasma(self, plasma: SingleZoneLTEPlasma) -> Dict[Tuple[str, int, float], float]:
        T_e_eV = plasma.T_e_eV
        n_e_cm3 = plasma.n_e
        all_populations: Dict[Tuple[str, int, float], float] = {}
        for element, total_density in plasma.species.items():
            stage_densities = self.solve_ionization_balance(
                element, T_e_eV, n_e_cm3, total_density
            )
            for stage, stage_density in stage_densities.items():
                if stage_density > 0:
                    populations = self.solve_level_population(
                        element, stage, stage_density, T_e_eV, n_e_cm3
                    )
                    all_populations.update(populations)
        logger.debug(f"Solved Saha-Boltzmann (JAX) for {len(plasma.species)} species")
        return all_populations
