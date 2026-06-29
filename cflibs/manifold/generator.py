"""
JAX-based manifold generator for high-throughput CF-LIBS.

This module implements GPU-accelerated generation of pre-computed spectral
manifolds using JAX. The manifold enables fast inference by pre-calculating
spectra for all parameter combinations of interest.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
import numpy as np
import time
from pathlib import Path

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    zarr = None

from cflibs.core.jax_runtime import HAS_JAX, jax, jit_if_available, jnp, vmap_if_available
from cflibs.core.constants import SAHA_CONST_CM3, C_LIGHT, EV_TO_J, EV_TO_K, H_PLANCK, M_PROTON
from cflibs.atomic.database import EMITTING_LINE_PREDICATE, AtomicDatabase
from cflibs.atomic.masses import DEFAULT_ATOMIC_MASS_AMU, STANDARD_ATOMIC_MASSES
from cflibs.manifold.config import ManifoldConfig
from cflibs.core.logging_config import get_logger
from cflibs.plasma.partition import (
    ionization_potential_depression_jax,
    polynomial_partition_function_jax,
)
from cflibs.radiation.ldm import DEFAULT_N_SIGMA, build_sigma_grid
from cflibs.radiation.profiles import BroadeningMode, doppler_sigma_jax

jit = jit_if_available
vmap = vmap_if_available

# Conditional imports for JAX physics functions
if HAS_JAX:
    pass  # JAX physics helpers available in cflibs.radiation if needed

logger = get_logger("manifold.generator")


def _infer_storage_format(output_path: Path) -> str:
    if output_path.is_dir():
        return "zarr"
    suffix = output_path.suffix.lower()
    if suffix == ".zarr":
        return "zarr"
    if suffix in {".h5", ".hdf5", ".hdf"}:
        return "hdf5"
    return "hdf5"


class ManifoldGenerator:
    """
    Generator for pre-computed spectral manifolds.

    This class generates a high-dimensional lookup table of synthetic spectra
    using JAX for GPU acceleration. The manifold covers a parameter space
    defined by temperature, electron density, and element concentrations.

    The generated manifold can be used for fast inference by finding the
    nearest matching spectrum rather than solving physics equations at runtime.
    """

    def __init__(self, config: ManifoldConfig):
        """
        Initialize manifold generator.

        Parameters
        ----------
        config : ManifoldConfig
            Configuration for manifold generation

        Raises
        ------
        ImportError
            If JAX is not installed
        """
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for manifold generation. Install with: pip install jax jaxlib"
            )

        config.validate()
        self.config = config

        # Load atomic database
        self.atomic_db = AtomicDatabase(config.db_path)

        # Load atomic data into JAX arrays
        self.atomic_data = self._load_atomic_data()

        logger.info(
            f"Initialized ManifoldGenerator: {len(config.elements)} elements, "
            f"λ=[{config.wavelength_range[0]:.1f}, {config.wavelength_range[1]:.1f}] nm"
        )

    def _load_atomic_data(self) -> Tuple:
        """
        Load atomic data from database and convert to JAX arrays.

        Returns
        -------
        Tuple
            Atomic data as JAX arrays:
            (lines_wl, lines_aki, lines_ek, lines_gk, lines_ip, lines_z, lines_el_idx,
             partition_coeffs, ionization_potentials, lines_stark_w, lines_stark_alpha,
             lines_mass_amu)
        """
        import pandas as pd

        logger.info("Loading atomic data from database...")

        # Build query for all elements (including Stark parameters). The emitting-
        # line filter (EMITTING_LINE_PREDICATE) is the canonical
        # AtomicDatabase.get_transitions cut: without it a single NaN aki from an
        # observation-only transition poisons the whole summed snapshot (every
        # pixel becomes NaN).
        placeholders = ",".join(["?"] * len(self.config.elements))
        query = f"""
            SELECT
                l.element, l.sp_num, l.wavelength_nm, l.aki, l.ek_ev, l.gk,
                sp.ip_ev, l.stark_w, l.stark_alpha
            FROM lines l
            JOIN species_physics sp ON l.element = sp.element AND l.sp_num = sp.sp_num
            WHERE l.wavelength_nm BETWEEN ? AND ?
            AND l.element IN ({placeholders})
            AND {EMITTING_LINE_PREDICATE.format(p="l.")}
            ORDER BY l.wavelength_nm
        """
        params = [
            self.config.wavelength_range[0],
            self.config.wavelength_range[1],
        ] + self.config.elements

        with self.atomic_db._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            raise ValueError(
                f"No atomic data found for elements {self.config.elements} "
                f"in wavelength range {self.config.wavelength_range}"
            )

        # Map element names to indices
        el_map = {el: i for i, el in enumerate(self.config.elements)}
        df["el_idx"] = df["element"].map(el_map)

        # Load atomic masses per element from database (with fallback)
        element_masses = self._resolve_element_masses()

        # Map masses to each line based on element
        df["mass_amu"] = df["element"].map(element_masses)

        # Convert to JAX arrays
        lines_wl = jnp.array(df["wavelength_nm"].values, dtype=jnp.float32)
        lines_aki = jnp.array(df["aki"].values, dtype=jnp.float32)
        lines_ek = jnp.array(df["ek_ev"].values, dtype=jnp.float32)
        lines_gk = jnp.array(df["gk"].values, dtype=jnp.float32)
        lines_ip = jnp.array(df["ip_ev"].values, dtype=jnp.float32)
        lines_z = jnp.array(df["sp_num"].values - 1, dtype=jnp.int32)  # 0=neutral, 1=ion
        lines_el_idx = jnp.array(df["el_idx"].values, dtype=jnp.int32)

        # Stark parameters - use NaN for missing (will use estimation in compute)
        stark_w_raw = df["stark_w"].fillna(float("nan")).values
        stark_alpha_raw = df["stark_alpha"].fillna(0.5).values  # Default alpha=0.5
        lines_stark_w = jnp.array(stark_w_raw, dtype=jnp.float32)
        lines_stark_alpha = jnp.array(stark_alpha_raw, dtype=jnp.float32)
        lines_mass_amu = jnp.array(df["mass_amu"].values, dtype=jnp.float32)

        # Count Stark data coverage
        stark_count = df["stark_w"].notna().sum()
        logger.info(f"Loaded {len(df)} spectral lines ({stark_count} with Stark data)")

        # --- Load Partition Function Coefficients & Ionization Potentials ---
        # Shapes:
        # partition_coeffs: (num_elements, max_stages, 5)
        # ionization_potentials: (num_elements, max_stages)
        max_stages = 3
        num_elements = len(self.config.elements)

        coeffs = np.zeros((num_elements, max_stages, 5), dtype=np.float32)
        ips = np.zeros((num_elements, max_stages), dtype=np.float32)
        # Per-species t_min / t_max / g0 for the encapsulated partition
        # provider (arch candidate 4).  Default window 2000–25000 K matches
        # the canonical CF-LIBS fit range; default g0 = 1.0 is the
        # conservative physical lower bound.  Without these the polynomial
        # in :meth:`_calculate_partition_functions` would extrapolate
        # outside the fit window — the Ca I @ 100 000 K, 560× error case
        # documented in CF-LIBS-improved-s1qr.1.
        tmin = np.full((num_elements, max_stages), 2000.0, dtype=np.float32)
        tmax = np.full((num_elements, max_stages), 25000.0, dtype=np.float32)
        g0 = np.ones((num_elements, max_stages), dtype=np.float32)

        # Seed defaults from the canonical fallback ladder; species the
        # factory resolves below overwrite these.  Closed-shell ions keep
        # their exact U (e.g. Na II -> 1.0) instead of the old generic
        # ln(25)/ln(15)/ln(10) constants.  warn=False: a warning here would
        # be misleading because most entries are overwritten.
        from cflibs.plasma.partition import canonical_partition_fallback

        for el_idx, el in enumerate(self.config.elements):
            for stage_idx in range(max_stages):
                coeffs[el_idx, stage_idx, 0] = np.log(
                    canonical_partition_fallback(el, stage_idx + 1, warn=False)
                )

        self._load_partition_physics(el_map, placeholders, max_stages, coeffs, ips, tmin, tmax, g0)

        partition_coeffs = jnp.array(coeffs, dtype=jnp.float32)
        ionization_potentials = jnp.array(ips, dtype=jnp.float32)
        partition_t_min = jnp.array(tmin, dtype=jnp.float32)
        partition_t_max = jnp.array(tmax, dtype=jnp.float32)
        partition_g0 = jnp.array(g0, dtype=jnp.float32)

        return (
            lines_wl,
            lines_aki,
            lines_ek,
            lines_gk,
            lines_ip,
            lines_z,
            lines_el_idx,
            partition_coeffs,
            ionization_potentials,
            lines_stark_w,
            lines_stark_alpha,
            lines_mass_amu,
            partition_t_min,
            partition_t_max,
            partition_g0,
        )

    def _resolve_element_masses(self) -> dict:
        """Resolve per-element atomic masses (amu) from the database with fallback.

        Prefers ``AtomicDatabase.get_atomic_mass``; falls back to the canonical
        :data:`~cflibs.atomic.masses.STANDARD_ATOMIC_MASSES` table, then to the
        generic :data:`~cflibs.atomic.masses.DEFAULT_ATOMIC_MASS_AMU` placeholder.

        The ``database -> table -> 50.0`` ladder now lives in
        :mod:`cflibs.atomic.masses`; this method retains the per-branch logging
        but produces masses identical to the previous in-module table.
        """
        element_masses = {}
        for el in self.config.elements:
            db_mass = self.atomic_db.get_atomic_mass(el)
            if db_mass is not None:
                element_masses[el] = db_mass
            elif el in STANDARD_ATOMIC_MASSES:
                element_masses[el] = STANDARD_ATOMIC_MASSES[el]
                logger.debug(f"Using standard mass for {el}: {STANDARD_ATOMIC_MASSES[el]} amu")
            else:
                element_masses[el] = DEFAULT_ATOMIC_MASS_AMU  # Generic fallback
                logger.warning(
                    f"No mass found for {el}, using fallback {DEFAULT_ATOMIC_MASS_AMU} amu"
                )
        return element_masses

    def _load_partition_physics(
        self,
        el_map: dict,
        placeholders: str,
        max_stages: int,
        coeffs: np.ndarray,
        ips: np.ndarray,
        tmin: np.ndarray,
        tmax: np.ndarray,
        g0: np.ndarray,
    ) -> None:
        """Fill IPs and partition-function spec arrays in place.

        Ionization potentials are read directly (they feed the Saha factor at
        index 8).  Partition coefficients + ``[t_min, t_max]`` + ``g0`` come
        from the SINGLE factory ``AtomicDatabase.partition_spec_for`` — the
        one place the partition-function policy lives (*prefer the direct-sum
        FIT over energy levels when the species has tabulated levels; else the
        stored polynomial; always carry the bounds + g0*).  This is the JAX
        batched adapter from the locked design: the manifold's static snapshot
        arrays bake the SAME spec the CPU scalar adapter
        (``partition_function_for``) consumes, so the two paths provably agree
        (the PF-3/PF-4 unification, diagnosis § 2.1; CONTEXT.md § "The
        partition-function provider").

        "Prefer direct-sum" is a BUILD-TIME choice here: vmap needs static,
        fixed-shape arrays and cannot hold a per-species variable-length level
        sum, so the factory fits the direct-sum to an ln-polynomial once
        (cached in ``partition._spec_cache``, compute-once per (db, element,
        stage) — invariant 5) and we bake those coefficients.  The kernel then
        evaluates the same guarded ``polynomial_partition_function_jax`` form,
        so jit / vmap are unaffected.

        Extracted verbatim from :meth:`_load_atomic_data` (behavior-preserving);
        mutates ``coeffs``/``ips``/``tmin``/``tmax``/``g0`` in place.
        """
        try:
            with self.atomic_db._get_connection() as conn:
                cursor = conn.cursor()

                # Load IPs
                ip_query = f"""
                    SELECT element, sp_num, ip_ev
                    FROM species_physics
                    WHERE element IN ({placeholders})
                """
                cursor.execute(ip_query, self.config.elements)
                self._fill_ionization_potentials(cursor, el_map, max_stages, ips)

            # Partition coefficients + bounds + g0 from the single factory.
            n_ds, n_poly = self._fill_partition_specs(el_map, max_stages, coeffs, tmin, tmax, g0)
            logger.info(
                "Loaded partition specs via factory for %d species "
                "(%d direct-sum-fit, %d stored-polynomial)",
                n_ds + n_poly,
                n_ds,
                n_poly,
            )

        except Exception as e:
            logger.warning(f"Failed to load physics data: {e}")

    @staticmethod
    def _fill_ionization_potentials(cursor, el_map: dict, max_stages: int, ips: np.ndarray) -> None:
        """Populate the ionization-potential array from an executed IP query.

        Extracted verbatim from :meth:`_load_atomic_data` (behavior-preserving).
        """
        for row in cursor.fetchall():
            el, sp_num, ip_ev = row
            if el in el_map and ip_ev is not None:
                el_idx = el_map[el]
                stage_idx = sp_num - 1
                if 0 <= stage_idx < max_stages:
                    ips[el_idx, stage_idx] = ip_ev

    def _fill_partition_specs(
        self,
        el_map: dict,
        max_stages: int,
        coeffs: np.ndarray,
        tmin: np.ndarray,
        tmax: np.ndarray,
        g0: np.ndarray,
    ) -> Tuple[int, int]:
        """Bake partition specs (coeffs/bounds/g0) from the single factory.

        Returns ``(n_direct_sum, n_polynomial)`` counts. Extracted verbatim from
        :meth:`_load_atomic_data` (behavior-preserving); mutates the supplied
        arrays in place.
        """
        n_ds = 0
        n_poly = 0
        for el, el_idx in el_map.items():
            for stage in range(1, max_stages + 1):
                stage_idx = stage - 1
                if stage_idx >= max_stages:
                    continue
                spec = self.atomic_db.partition_spec_for(el, stage)
                if spec is None:
                    # Neither energy levels nor a stored polynomial row:
                    # keep the conservative ln(U) defaults / [2000, 25000] K
                    # window / g0 = 1.0 already filled above so the static
                    # arrays stay well-formed and the guarded evaluator
                    # always receives concrete bounds.
                    continue
                spec_coeffs = list(spec.coefficients)
                spec_coeffs += [0.0] * (5 - len(spec_coeffs))
                coeffs[el_idx, stage_idx] = spec_coeffs[:5]
                tmin[el_idx, stage_idx] = float(spec.t_min)
                tmax[el_idx, stage_idx] = float(spec.t_max)
                g0[el_idx, stage_idx] = float(spec.g0)
                if spec.from_direct_sum:
                    n_ds += 1
                else:
                    n_poly += 1
        return n_ds, n_poly

    def _build_ldm_sigma_grid(self, sigma_inst: float) -> np.ndarray:
        """Build the LDM σ-layer grid that brackets the manifold sweep.

        The Line Distribution Method projects each line onto a fixed log-σ
        grid; lines whose σ falls outside the grid are clipped to the
        boundary layers. We pick the extreme σ values the manifold can ever
        produce — coldest plasma + heaviest line (smallest σ_doppler) and
        hottest plasma + lightest line (largest σ_doppler) — then convolve
        with the instrument floor and let :func:`build_sigma_grid` apply its
        standard ``[0.5×, 2×]`` bracket factors on top.

        Parameters
        ----------
        sigma_inst : float
            Instrument Gaussian σ in nm. Threaded in from
            ``ManifoldConfig.instrument_fwhm_nm`` by ``generate_manifold``
            (D3 fix; previously hardcoded 0.05/2.355).

        Returns
        -------
        ndarray, shape (DEFAULT_N_SIGMA,)
            Log-spaced σ grid in nm.
        """
        (
            l_wl,
            *_,
            l_mass_amu,
        ) = self.atomic_data

        lines_wl_np = np.asarray(l_wl, dtype=np.float64)
        lines_mass_np = np.asarray(l_mass_amu, dtype=np.float64)

        if lines_wl_np.size == 0:
            raise ValueError("Cannot build LDM sigma grid: atomic line catalog is empty.")

        T_lo, T_hi = self.config.temperature_range
        # Doppler σ is the 1-D Maxwell std: λ * sqrt(k T / m c^2). This is the
        # numpy twin of profiles.doppler_sigma_jax (the single source of truth);
        # evaluate at the extremes of the (λ, m, T) Cartesian product to bracket
        # the sweep. The previous spurious factor of 2 under the sqrt computed
        # the most-probable speed, not the standard deviation (~1.41x too wide).
        wl_lo, wl_hi = lines_wl_np.min(), lines_wl_np.max()
        m_lo, m_hi = lines_mass_np.min(), lines_mass_np.max()

        def _doppler_sigma_nm(wl_nm: float, mass_amu: float, T_eV: float) -> float:
            mass_kg = mass_amu * M_PROTON
            return float(wl_nm * np.sqrt(T_eV * EV_TO_J / (mass_kg * C_LIGHT**2)))

        sigma_dop_min = _doppler_sigma_nm(wl_lo, m_hi, T_lo)
        sigma_dop_max = _doppler_sigma_nm(wl_hi, m_lo, T_hi)

        # Instrument floor adds in quadrature. ``sigma_inst`` is threaded
        # in from ``ManifoldConfig.instrument_fwhm_nm`` via
        # ``generate_manifold`` (D3 fix; matches the floor used in
        # ``_compute_spectrum_snapshot`` / ``_compute_spectrum_snapshot_ldm``).
        sigma_min_total = float(np.sqrt(sigma_dop_min**2 + sigma_inst**2))
        sigma_max_total = float(np.sqrt(sigma_dop_max**2 + sigma_inst**2))

        # Pass the two endpoints; build_sigma_grid applies bracket factors.
        return build_sigma_grid(
            np.array([sigma_min_total, sigma_max_total], dtype=np.float64),
            n_sigma=DEFAULT_N_SIGMA,
        )

    @staticmethod
    def _calculate_partition_functions(t_k, atomic_data):
        """Calculates neutral and ion partition functions for all lines' elements.

        Every manifold U(T) evaluation goes through the ONE shared *guarded*
        :func:`polynomial_partition_function_jax`: the per-species
        ``t_min`` / ``t_max`` / ``g0`` arrays (atomic_data indices 12/13/14,
        always present — :meth:`_load_atomic_data` returns the full 15-tuple)
        are threaded through so the polynomial is clamped to its fit window and
        floored at the ground-state degeneracy.  This is the JAX batched adapter
        of the partition-function provider (CONTEXT.md § "The
        partition-function provider"); the coefficients themselves are the
        direct-sum-preferred spec baked at snapshot-build time by the single
        factory ``partition_spec_for`` (see :meth:`_load_atomic_data`).

        There is deliberately NO unguarded fallback: clamping prevents the
        extrapolation defect (manifold grids sweeping T outside the fit window
        previously produced 100×+ partition-function errors — the Ca I @
        100 000 K, 560× regression case), and the direct-sum-preferred
        coefficients fix the bias defect (the stored polynomial running below
        the direct-sum floor for iron-group workhorse species at LIBS T —
        diagnosis § 2.1).
        """
        lines_el_idx = atomic_data[6]
        partition_coeffs = atomic_data[7]
        coeffs_0 = partition_coeffs[lines_el_idx, 0]
        coeffs_1 = partition_coeffs[lines_el_idx, 1]
        coeffs_2 = partition_coeffs[lines_el_idx, 2]
        partition_t_min = atomic_data[12]
        partition_t_max = atomic_data[13]
        partition_g0 = atomic_data[14]
        tmin_0 = partition_t_min[lines_el_idx, 0]
        tmin_1 = partition_t_min[lines_el_idx, 1]
        tmin_2 = partition_t_min[lines_el_idx, 2]
        tmax_0 = partition_t_max[lines_el_idx, 0]
        tmax_1 = partition_t_max[lines_el_idx, 1]
        tmax_2 = partition_t_max[lines_el_idx, 2]
        g0_0 = partition_g0[lines_el_idx, 0]
        g0_1 = partition_g0[lines_el_idx, 1]
        g0_2 = partition_g0[lines_el_idx, 2]
        u0 = polynomial_partition_function_jax(t_k, coeffs_0, t_min=tmin_0, t_max=tmax_0, g0=g0_0)
        u1 = polynomial_partition_function_jax(t_k, coeffs_1, t_min=tmin_1, t_max=tmax_1, g0=g0_1)
        u2 = polynomial_partition_function_jax(t_k, coeffs_2, t_min=tmin_2, t_max=tmax_2, g0=g0_2)
        return u0, u1, u2

    @staticmethod
    def _calculate_saha_fractions(t_ev, n_e, u0, u1, u2, atomic_data):
        """Calculates the three-stage Saha ionization population fractions.

        Solves the same Δχ-lowered system as
        :meth:`SahaBoltzmannSolver.solve_ionization_balance` (audit 01-F4,
        bead CF-LIBS-improved-rs7e)::

            S1 = (C/n_e) T^1.5 (U_II /U_I ) exp(-(ip_I  - Δχ)/kT)
            S2 = (C/n_e) T^1.5 (U_III/U_II) exp(-(ip_II - Δχ)/kT)
            f0 = 1/(1 + S1 + S1·S2),  f1 = S1·f0,  f2 = S2·f1

        with the canonical Gaussian-CGS Debye-Hückel Δχ
        (:func:`cflibs.plasma.partition.ionization_potential_depression_jax`).
        Pre-rs7e this was the raw-IP two-stage balance, which under-ionized
        by ~9 % at 0.8 eV / 1e17 cm^-3 and re-assigned the doubly-ionized
        population to stage II at the hot manifold edge (Ca II ×2.9 at
        1.3 eV).  Elements without a catalogued stage-II ionization potential
        (``ips[el, 1] == 0`` builder default) get ``S2 = 0`` via
        ``jnp.where``, matching the CPU solver's ``ip_II is None`` branch.

        Returns ``(frac0, frac1, frac2, delta_chi)``.
        """
        lines_el_idx = atomic_data[6]
        ionization_potentials = atomic_data[8]
        ip_i = ionization_potentials[lines_el_idx, 0]
        ip_ii = ionization_potentials[lines_el_idx, 1]
        t_k = t_ev * EV_TO_K
        delta_chi = ionization_potential_depression_jax(n_e, t_k)
        eff_ip_i = jnp.maximum(ip_i - delta_chi, 0.0)
        eff_ip_ii = jnp.maximum(ip_ii - delta_chi, 0.0)
        saha_factor = (SAHA_CONST_CM3 / n_e) * (t_ev**1.5)
        s1 = saha_factor * (u1 / u0) * jnp.exp(-eff_ip_i / t_ev)
        s2 = jnp.where(
            ip_ii > 0.0,
            saha_factor * (u2 / u1) * jnp.exp(-eff_ip_ii / t_ev),
            0.0,
        )
        denom = 1.0 + s1 + s1 * s2
        frac0 = 1.0 / denom
        frac1 = s1 / denom
        frac2 = s1 * s2 / denom
        return frac0, frac1, frac2, delta_chi

    @staticmethod
    def _calculate_boltzmann_populations(
        plasma_state,
        saha_state,
        atomic_data,
    ):
        """Calculates upper level populations using the Boltzmann equation.

        Stage dispatch: ``z == 0`` (neutral) → ``(frac0, u0)``, ``z == 1``
        (singly ionized) → ``(frac1, u1)``, ``z >= 2`` → ``(frac2, u2)``
        (the handful of catalogued stage-IV lines are approximated as
        stage III; pre-rs7e they were silently treated as stage II).
        Upper levels above the Δχ-lowered ionization potential of the
        line's own species have merged into the continuum and carry zero
        population (the CPU ``max_energy_ev`` cutoff).
        """
        t_ev, n_e, concentration_map = plasma_state
        u0, u1, u2, frac0, frac1, frac2, delta_chi = saha_state

        lines_ek = atomic_data[2]
        lines_gk = atomic_data[3]
        lines_ip = atomic_data[4]
        lines_z = atomic_data[5]
        lines_el_idx = atomic_data[6]

        pop_fraction = jnp.where(lines_z == 0, frac0, jnp.where(lines_z == 1, frac1, frac2))
        u_val = jnp.where(lines_z == 0, u0, jnp.where(lines_z == 1, u1, u2))
        element_conc = concentration_map[lines_el_idx]
        n_species_total = element_conc * n_e
        n_species = n_species_total * pop_fraction
        n_upper = n_species * (lines_gk / u_val) * jnp.exp(-lines_ek / t_ev)
        # IPD level cutoff: zero population for upper levels above the
        # lowered ionization potential of the line's own species.
        level_bound = lines_ek <= jnp.maximum(lines_ip - delta_chi, 0.0)
        return n_upper * level_bound

    @staticmethod
    @jit
    def _saha_eggert_solver(
        T_eV: float,
        n_e: float,
        concentration_map: jnp.ndarray,
        atomic_data: Tuple,
    ) -> jnp.ndarray:
        """
        Vectorized Saha-Eggert solver for JAX.

        Calculates upper level populations for all lines simultaneously.

        Parameters
        ----------
        T_eV : float
            Electron temperature in eV
        n_e : float
            Electron density in cm^-3
        concentration_map : array
            Element concentrations
        atomic_data : Tuple
            Atomic data arrays from ManifoldGenerator._load_atomic_data

        Returns
        -------
        array
            Upper level populations
        """
        t_k = T_eV * EV_TO_K

        u0, u1, u2 = ManifoldGenerator._calculate_partition_functions(t_k, atomic_data)

        frac0, frac1, frac2, delta_chi = ManifoldGenerator._calculate_saha_fractions(
            T_eV, n_e, u0, u1, u2, atomic_data
        )

        n_upper = ManifoldGenerator._calculate_boltzmann_populations(
            (T_eV, n_e, concentration_map),
            (u0, u1, u2, frac0, frac1, frac2, delta_chi),
            atomic_data,
        )

        return n_upper

    @staticmethod
    def _calculate_stark_hwhm(
        T_eV: float,
        n_e: float,
        l_wl: jnp.ndarray,
        l_ek: jnp.ndarray,
        l_ip: jnp.ndarray,
        l_z: jnp.ndarray,
        l_stark_w: jnp.ndarray,
        l_stark_alpha: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the Stark Lorentzian HWHM (gamma) for every line.

        ``l_stark_w`` (and the ``w_est`` fallback) is the stored
        electron-impact FWHM at REF_NE = 1e17 cm^-3, T = 10000 K (see the
        convention note in cflibs/radiation/stark.py). The Voigt profile needs
        a HWHM, so scale to live n_e and halve::

            gamma_hwhm = 0.5 * w_fwhm * (n_e / 1e17) * (T / T_ref)^(-alpha)

        The earlier (n_e/1e16) with no 0.5 over-broadened by x20 (A4-CONV-2).
        Lines without a database value use a binding-energy estimate
        (binding energy = IP - E_upper).
        """
        REF_NE = 1.0e17
        REF_T_EV = 0.86173  # 10000 K in eV

        # Estimate Stark w_ref for lines without database values
        # Use binding energy = IP - E_upper
        binding_energy = jnp.maximum(l_ip - l_ek, 0.1)
        n_eff = (l_z + 1) * jnp.sqrt(13.605 / binding_energy)
        w_est = 2.0e-5 * (l_wl / 500.0) ** 2 * (n_eff**4)
        w_est = jnp.clip(w_est, 0.0001, 0.5)

        # Use database value if available, else estimate
        w_ref = jnp.where(jnp.isnan(l_stark_w), w_est, l_stark_w)

        # Stark HWHM calculation (w_ref is a FWHM at REF_NE; 0.5 -> HWHM)
        factor_ne = n_e / REF_NE
        factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -l_stark_alpha)
        return 0.5 * w_ref * factor_ne * factor_T

    @staticmethod
    def _humlicek_w4(z: jnp.ndarray) -> jnp.ndarray:
        """Humlicek W4 Faddeeva approximation w(z) (no complex erfc needed).

        Evaluates the complex error function via the four-region Humlicek W4
        rational approximation, selecting per element with nested
        :func:`jnp.where` so the whole evaluation stays jit/vmap-friendly.
        """
        x_h = jnp.real(z)
        y_h = jnp.abs(jnp.imag(z))
        s = jnp.abs(x_h) + y_h
        t = y_h - 1j * x_h

        # Region 1: s >= 15 (asymptotic)
        w_r1 = t * 0.5641896 / (0.5 + t * t)

        # Region 2: 5.5 <= s < 15
        u = t * t
        w_r2 = t * (1.410474 + u * 0.5641896) / (0.75 + u * (3.0 + u))

        # Region 3: s < 5.5 and y >= 0.195 * |x| - 0.176
        w_r3 = (16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + t * 0.5642236)))) / (
            16.4955 + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t))))
        )

        # Region 4: s < 5.5 and y < 0.195 * |x| - 0.176
        w_r4 = jnp.exp(u) - t * (
            36183.31
            - u
            * (
                3321.9905
                - u * (1540.787 - u * (219.0313 - u * (35.76683 - u * (1.320522 - u * 0.56419))))
            )
        ) / (
            32066.6
            - u
            * (
                24322.84
                - u
                * (9022.228 - u * (2186.181 - u * (364.2191 - u * (61.57037 - u * (1.841439 - u)))))
            )
        )

        # Select region based on conditions
        return jnp.where(
            s >= 15.0,
            w_r1,
            jnp.where(
                s >= 5.5,
                w_r2,
                jnp.where(
                    y_h >= 0.195 * jnp.abs(x_h) - 0.176,
                    w_r3,
                    w_r4,
                ),
            ),
        )

    @staticmethod
    @jit
    def _compute_spectrum_snapshot(
        wl_grid: jnp.ndarray,
        T_eV: float,
        n_e: float,
        concentrations: jnp.ndarray,
        atomic_data: Tuple,
        sigma_inst: float,
    ) -> jnp.ndarray:
        """

        Compute spectrum for a single time snapshot.



        Parameters

        ----------

        wl_grid : array

            Wavelength grid

        T_eV : float

            Temperature in eV

        n_e : float

            Electron density

        concentrations : array

            Element concentrations

        atomic_data : Tuple

            Atomic data arrays



        Returns

        -------

        array

            Spectral intensity

        """

        # ``atomic_data`` is the full 15-tuple from :meth:`_load_atomic_data`;
        # the trailing partition ``t_min`` / ``t_max`` / ``g0`` arrays (indices
        # 12-14) are consumed inside :meth:`_calculate_partition_functions` via
        # the full tuple, so we only name the per-line arrays this kernel uses
        # directly and absorb the rest with ``*_partition_arrays``.
        (
            l_wl,
            l_aki,
            l_ek,
            l_gk,
            l_ip,
            l_z,
            l_el_idx,
            partition_coeffs,
            ionization_potentials,
            l_stark_w,
            l_stark_alpha,
            l_mass_amu,
            *_partition_arrays,
        ) = atomic_data

        # Solve populations

        n_upper = ManifoldGenerator._saha_eggert_solver(
            T_eV,
            n_e,
            concentrations,
            atomic_data,
        )

        # n_upper from the Saha-Boltzmann solver is in cm⁻³ (project convention,
        # SAHA_CONST_CM3). The emissivity uses SI constants (H_PLANCK J·s,
        # C_LIGHT m/s, l_wl in m), so n_upper must be in m⁻³ — the same * 1.0e6
        # conversion applied in kernels.forward_model and the two-zone Bayesian
        # forward (forward.py:519). Omitting it underscaled every emissivity
        # (and thus every stored manifold) by 1e6.
        n_upper_m3 = n_upper * 1.0e6

        # Line emissivity: epsilon = (hc / 4pi lambda) * A * n_upper

        epsilon = (H_PLANCK * C_LIGHT / (4 * jnp.pi * l_wl * 1e-9)) * l_aki * n_upper_m3

        # --- Proper Voigt Broadening (Phase 2) ---

        # Doppler width: 1-D Maxwell std sigma = lambda/c * sqrt(kT/m).
        # Delegate to the canonical profiles.doppler_sigma_jax so there is a
        # single source of truth (it does mass_amu -> kg internally). The
        # previous open-coded form carried a spurious factor of 2 under the
        # sqrt (most-probable speed, not std dev), inflating widths ~1.41x.
        sigma_doppler = doppler_sigma_jax(l_wl, T_eV, l_mass_amu)

        # Instrument broadening (Gaussian sigma). Threaded in from
        # ``ManifoldConfig.instrument_fwhm_nm`` via ``generate_manifold``
        # (D3 fix; previously hardcoded 0.05/2.355 silently dropped the
        # configured value).
        # Total Gaussian width
        sigma_total = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)

        # Stark broadening: Lorentzian gamma (HWHM).
        gamma_stark = ManifoldGenerator._calculate_stark_hwhm(
            T_eV,
            n_e,
            l_wl,
            l_ek,
            l_ip,
            l_z,
            l_stark_w,
            l_stark_alpha,
        )

        # --- Voigt Profile Rendering (Humlicek W4 approximation) ---
        # For each wavelength point, compute Voigt profile contribution from all lines
        # z = (x + i*gamma) / (sigma * sqrt(2))
        # V(x) = Re(w(z)) / (sigma * sqrt(2*pi))

        diff = wl_grid[:, None] - l_wl[None, :]  # (n_wl, n_lines)

        # Compute Voigt profile using Humlicek W4 approximation
        z = (diff + 1j * gamma_stark) / (sigma_total * jnp.sqrt(2.0))
        w_z = ManifoldGenerator._humlicek_w4(z)

        profile = jnp.real(w_z) / (sigma_total * jnp.sqrt(2.0 * jnp.pi))

        # Sum contributions weighted by emissivity

        intensity = jnp.sum(epsilon * profile, axis=1)

        return intensity

    @staticmethod
    def _compute_spectrum_snapshot_ldm(
        wl_grid: jnp.ndarray,
        T_eV: float,
        n_e: float,
        concentrations: jnp.ndarray,
        atomic_data: Tuple,
        sigma_grid: jnp.ndarray,
        sigma_inst: float,
    ) -> jnp.ndarray:
        """Compute a single-snapshot spectrum via the LDM Gaussian path.

        This is the manifold-sweep specialisation of
        :func:`cflibs.radiation.ldm.ldm_broaden`. The σ-grid layout is
        passed as a closed-over array (built once at manifold-init in
        :meth:`generate_manifold`) so it stays jit-static across all
        grid points — only the per-line intensities recompute per (T, n_e,
        concentration) sample. Stark broadening is currently NOT modelled
        on this path (LDM 1-D is Gaussian-only; the 2-D Voigt extension is
        scope for a follow-up bead).

        See spec ``docs/adr/specs/T1-4-ldm-broadening.md`` §7.

        Parameters
        ----------
        wl_grid : array
            Uniform wavelength grid in nm.
        T_eV, n_e : float
            Plasma temperature / electron density.
        concentrations : array
            Element concentrations.
        atomic_data : Tuple
            Atomic data from :meth:`_load_atomic_data`.
        sigma_grid : array
            Pre-built log-σ grid; constructed via
            :func:`cflibs.radiation.ldm.build_sigma_grid` at manifold init.

        Returns
        -------
        array
            Spectral intensity on ``wl_grid`` in arbitrary CF-LIBS units.
        """
        from cflibs.radiation.ldm import ldm_broaden

        # ``atomic_data`` is the full 15-tuple from :meth:`_load_atomic_data`;
        # the partition arrays (indices 7-8, 12-14) are consumed inside
        # :meth:`_calculate_partition_functions` via the full tuple, so we only
        # name the per-line arrays this kernel uses directly and absorb the
        # rest with ``*_partition_arrays``.
        (
            l_wl,
            l_aki,
            _l_ek,
            _l_gk,
            _l_ip,
            _l_z,
            _l_el_idx,
            _partition_coeffs,
            _ionization_potentials,
            _l_stark_w,
            _l_stark_alpha,
            l_mass_amu,
            *_partition_arrays,
        ) = atomic_data

        n_upper = ManifoldGenerator._saha_eggert_solver(
            T_eV,
            n_e,
            concentrations,
            atomic_data,
        )

        # cm⁻³ -> m⁻³ before the SI-constant emissivity (see _compute_spectrum_snapshot).
        n_upper_m3 = n_upper * 1.0e6
        # Line emissivity: epsilon = (hc / 4pi lambda) * A * n_upper
        epsilon = (H_PLANCK * C_LIGHT / (4 * jnp.pi * l_wl * 1e-9)) * l_aki * n_upper_m3

        # Doppler sigma + instrument floor. ``sigma_inst`` is threaded in
        # from ``ManifoldConfig.instrument_fwhm_nm`` (D3 fix; previously the
        # FWHM 0.05 nm was hardcoded and silently dropped configured values).
        # Doppler is the 1-D Maxwell std via the canonical doppler_sigma_jax
        # (single source of truth); the previous open-coded form carried a
        # spurious factor of 2 under the sqrt (~1.41x too wide).
        sigma_doppler = doppler_sigma_jax(l_wl, T_eV, l_mass_amu)
        sigma_total = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)

        return ldm_broaden(
            line_wavelengths=l_wl,
            line_intensities=epsilon,
            line_sigmas=sigma_total,
            wavelength_grid=wl_grid,
            sigma_grid=sigma_grid,
        )

    @staticmethod
    @jit_if_available(static_argnames=("time_steps",))
    def _time_integrated_spectrum_ldm(
        wl_grid: jnp.ndarray,
        params: jnp.ndarray,
        atomic_data: Tuple,
        sigma_grid: jnp.ndarray,
        gate_width_s: float,
        time_steps: int,
        sigma_inst: float,
        cooling_t0_s: float,
        cooling_T_exponent: float,
        cooling_ne_exponent: float,
    ) -> jnp.ndarray:
        """Time-integrated spectrum via the LDM Gaussian broadening path.

        Mirrors :meth:`_time_integrated_spectrum` but dispatches each cooling
        snapshot through :meth:`_compute_spectrum_snapshot_ldm`, which uses
        the Line Distribution Method (van den Bekerom & Pannier 2021) instead
        of per-line Voigt broadcasting. The ``sigma_grid`` is closed-over
        (built once at manifold init) so it stays jit-static.

        The cooling-trail laws ``T(t) = T_max*(1 + t/t0)**T_exp`` and
        ``n_e(t) = ne_max*(1 + t/t0)**ne_exp`` are parameterised via
        ``ManifoldConfig`` (defaults reproduce the historical ns-ICCD
        constants) so ps-LIBS regimes can be configured.
        """
        T_max = params[0]
        ne_max = params[1]
        concs = params[2:]

        times = jnp.linspace(0, gate_width_s, time_steps)
        dt = times[1] - times[0]

        T_trail = T_max * (1 + times / cooling_t0_s) ** cooling_T_exponent
        ne_trail = ne_max * (1 + times / cooling_t0_s) ** cooling_ne_exponent

        def step_fn(carry, inputs):
            T, ne = inputs
            intensity = jnp.where(
                T > 0.4,
                ManifoldGenerator._compute_spectrum_snapshot_ldm(
                    wl_grid, T, ne, concs, atomic_data, sigma_grid, sigma_inst
                ),
                jnp.zeros_like(wl_grid),
            )
            return carry + intensity * dt, None

        spectrum_accum = jnp.zeros_like(wl_grid)
        spectrum_accum, _ = jax.lax.scan(step_fn, spectrum_accum, (T_trail, ne_trail))

        return spectrum_accum

    @staticmethod
    @jit_if_available(static_argnames=("time_steps",))
    def _time_integrated_spectrum(
        wl_grid: jnp.ndarray,
        params: jnp.ndarray,
        atomic_data: Tuple,
        gate_width_s: float,
        time_steps: int,
        sigma_inst: float,
        cooling_t0_s: float,
        cooling_T_exponent: float,
        cooling_ne_exponent: float,
    ) -> jnp.ndarray:
        """
        Compute time-integrated spectrum for cooling plasma.

        Parameters
        ----------
        wl_grid : array
            Wavelength grid
        params : array
            [T_max, ne_max, C_el1, C_el2, ...]
        atomic_data : Tuple
            Atomic data arrays
        gate_width_s : float
            Gate width in seconds
        time_steps : int
            Number of integration steps
        sigma_inst : float
            Instrument Gaussian sigma (nm).
        cooling_t0_s : float
            Cooling-trail reference timescale ``t0`` in seconds.
        cooling_T_exponent : float
            Power-law exponent for the temperature cooling trail.
        cooling_ne_exponent : float
            Power-law exponent for the electron-density cooling trail.

        Returns
        -------
        array
            Time-integrated spectral intensity
        """
        T_max = params[0]
        ne_max = params[1]
        concs = params[2:]

        # Time grid
        times = jnp.linspace(0, gate_width_s, time_steps)
        dt = times[1] - times[0]

        # Cooling laws (power-law decay). Parameterised via ManifoldConfig;
        # the defaults (t0=1e-6 s, T_exp=-0.5, ne_exp=-1.0) reproduce the
        # historical hardcoded ns-ICCD values so existing manifolds are
        # unchanged, while ps-LIBS regimes can configure a smaller t0.
        T_trail = T_max * (1 + times / cooling_t0_s) ** cooling_T_exponent
        ne_trail = ne_max * (1 + times / cooling_t0_s) ** cooling_ne_exponent

        # Integrate over time
        def step_fn(carry, inputs):
            T, ne = inputs
            intensity = jnp.where(
                T > 0.4,  # Only if T > 0.4 eV
                ManifoldGenerator._compute_spectrum_snapshot(
                    wl_grid, T, ne, concs, atomic_data, sigma_inst
                ),
                jnp.zeros_like(wl_grid),
            )
            return carry + intensity * dt, None

        spectrum_accum = jnp.zeros_like(wl_grid)
        spectrum_accum, _ = jax.lax.scan(step_fn, spectrum_accum, (T_trail, ne_trail))

        return spectrum_accum

    @staticmethod
    def _build_composition_grid(n_elements: int, concentration_steps: int) -> np.ndarray:
        """Build a D-dimensional simplex of valid composition rows.

        Every returned row satisfies the CF-LIBS closure constraint
        ``Σ C_s = 1`` exactly (to float64 precision) and contains **no
        identically-zero element**, so the manifold sweep never emits a
        composition that violates Saha-Boltzmann mass conservation (the
        previous ``[c1] + [0.0]*(D-1)`` generic branch summed to ``c1``,
        not 1, for any element count ≠ 4).

        The sampler is a deterministic, seeded symmetric-Dirichlet draw
        (one fixed seed → reproducible grids across runs). A small positive
        floor is added before renormalisation to guarantee strict positivity
        even on the (measure-zero) chance of a zero draw, then each row is
        renormalised so it sums to exactly 1.0.

        ``concentration_steps`` keeps its "resolution of the concentration
        grid" semantics: it is the number of composition rows sampled.

        Parameters
        ----------
        n_elements : int
            Number of elements ``D`` (simplex dimension).
        concentration_steps : int
            Number of composition rows to sample (>= 1).

        Returns
        -------
        ndarray, shape (n_rows, n_elements)
            Composition rows, each summing to 1.0 with all entries > 0.
        """
        n_rows = max(1, int(concentration_steps))

        if n_elements <= 1:
            # Single element: the only valid closed composition is [1.0].
            return np.ones((n_rows, 1), dtype=np.float64)

        # Deterministic symmetric-Dirichlet draws. alpha=1.0 gives a uniform
        # distribution over the simplex interior; the fixed seed makes the
        # grid reproducible. The first row is pinned to the barycentre
        # (1/D, ..., 1/D) so the grid always contains the maximally-mixed
        # composition regardless of seed/draw count.
        rng = np.random.default_rng(20240601)
        alpha = np.ones(n_elements, dtype=np.float64)
        draws = rng.dirichlet(alpha, size=n_rows)
        draws[0, :] = 1.0 / n_elements

        # Floor away exact zeros (Dirichlet is a.s. positive, but float
        # underflow at large D can produce exact 0.0) and renormalise so
        # each row sums to exactly 1.0.
        floor = 1e-6
        draws = draws + floor
        draws = draws / draws.sum(axis=1, keepdims=True)
        return draws

    def _build_param_grid(self) -> np.ndarray:
        """Build the ``[T, ne, C_el1, ...]`` parameter grid for the sweep.

        Sweeps temperature linearly and electron density geometrically, then
        takes the Cartesian product with a D-dimensional simplex of
        compositions from :meth:`_build_composition_grid`. Every composition
        row sums to 1.0 (CF-LIBS closure) with no identically-zero element,
        for *any* element count — the same sampler now drives the formerly
        special-cased 4-element Ti-Al-V-Fe branch.

        Returns
        -------
        ndarray, shape (n_samples, n_elements + 2)
            Float32 parameter rows consumed by the batch loop.
        """
        T_grid = np.linspace(
            self.config.temperature_range[0],
            self.config.temperature_range[1],
            self.config.temperature_steps,
        )
        ne_grid = np.geomspace(
            self.config.density_range[0], self.config.density_range[1], self.config.density_steps
        )

        # Single D-dimensional simplex sampler for all element counts. Every
        # row sums to 1.0 with no zero element (CF-LIBS closure Σ C_s = 1).
        comp_grid = self._build_composition_grid(
            len(self.config.elements), self.config.concentration_steps
        )

        params_list = []
        for T in T_grid:
            for ne in ne_grid:
                for comp in comp_grid:
                    params_list.append([T, ne, *comp.tolist()])

        params_arr = np.array(params_list, dtype=np.float32)
        n_samples = len(params_arr)

        logger.info(f"Parameter grid: {n_samples} spectra to generate")
        logger.info(f"  Temperature: {len(T_grid)} points")
        logger.info(f"  Density: {len(ne_grid)} points")
        logger.info(f"  Concentrations: {len(comp_grid)} combinations")
        return params_arr

    def _compute_instrument_sigma(self) -> Tuple[float, float]:
        """Resolve the instrument Gaussian σ (nm) from the configured FWHM.

        Uses the exact ``2*sqrt(2 ln 2)`` factor (not the legacy 2.355
        approximation). Falls back to the historical 0.05 nm FWHM with a
        one-time WARN if the field is missing or non-positive, so the
        silent-drop bug stays visible to anyone running malformed configs
        (D3 fix).

        Returns
        -------
        tuple of float
            ``(fwhm_nm, sigma_inst)`` — the (possibly defaulted) FWHM and the
            derived Gaussian σ in nm.
        """
        fwhm_nm = self.config.instrument_fwhm_nm
        if fwhm_nm is None or not (fwhm_nm > 0):
            logger.warning(
                "ManifoldConfig.instrument_fwhm_nm is missing or non-positive "
                "(got %r); falling back to 0.05 nm FWHM. This used to be the "
                "silent hardcoded default — set instrument_fwhm_nm explicitly "
                "to silence this warning.",
                fwhm_nm,
            )
            fwhm_nm = 0.05
        sigma_inst = float(fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0))))
        return fwhm_nm, sigma_inst

    def _create_storage_backend(
        self,
        storage_format: str,
        output_path: Path,
        n_samples: int,
        chunk_rows: int,
        wl_grid: jnp.ndarray,
    ) -> Tuple:
        """Open the output store and create its spectra/params/wavelength sets.

        Dispatches on ``storage_format`` to the HDF5 or Zarr backend, raising
        :class:`ImportError` if the required dependency is unavailable and
        :class:`ValueError` for unknown formats.

        Returns
        -------
        tuple
            ``(output_root, dset_spec, dset_param, attrs)`` — the open root
            handle, the two writable datasets, and the attribute mapping.
        """
        if storage_format == "hdf5":
            if not HAS_H5PY:
                raise ImportError(
                    "h5py is required for HDF5 manifold generation. Install with: pip install h5py"
                )
            output_root = h5py.File(output_path, "w")
            dset_spec = output_root.create_dataset(
                "spectra",
                (n_samples, self.config.pixels),
                dtype="f4",
                chunks=(chunk_rows, self.config.pixels),
                compression="gzip",
                compression_opts=4,
            )
            dset_param = output_root.create_dataset(
                "params",
                (n_samples, len(self.config.elements) + 2),
                dtype="f4",
                chunks=(chunk_rows, len(self.config.elements) + 2),
            )
            output_root.create_dataset("wavelength", data=np.asarray(wl_grid, dtype=np.float32))
            attrs = output_root.attrs
        elif storage_format == "zarr":
            if not HAS_ZARR:
                raise ImportError(
                    "zarr is required for Zarr manifold generation. Install with: pip install zarr"
                )
            output_root = zarr.open_group(str(output_path), mode="w")
            dset_spec = output_root.create_array(
                "spectra",
                shape=(n_samples, self.config.pixels),
                chunks=(chunk_rows, self.config.pixels),
                dtype="f4",
                overwrite=True,
            )
            dset_param = output_root.create_array(
                "params",
                shape=(n_samples, len(self.config.elements) + 2),
                chunks=(chunk_rows, len(self.config.elements) + 2),
                dtype="f4",
                overwrite=True,
            )
            output_root.create_array(
                "wavelength",
                data=np.asarray(wl_grid, dtype=np.float32),
                dtype="f4",
                overwrite=True,
            )
            attrs = output_root.attrs
        else:
            raise ValueError(f"Unsupported manifold storage format: {storage_format}")
        return output_root, dset_spec, dset_param, attrs

    def _write_batches(
        self,
        params_arr: np.ndarray,
        n_samples: int,
        batch_spectrum: Callable[[jnp.ndarray], jnp.ndarray],
        dset_spec,
        dset_param,
        progress_callback: Optional[Callable[[int, int, float], None]],
    ) -> None:
        """Generate spectra batch-by-batch and write them to the datasets.

        Iterates the parameter grid in ``config.batch_size`` chunks, evaluates
        ``batch_spectrum`` per chunk, writes spectra + params into the open
        datasets, and reports progress via the callback (or periodic logging).
        """
        for i in range(0, n_samples, self.config.batch_size):
            batch = params_arr[i : i + self.config.batch_size]
            batch_jax = jnp.array(batch)

            spectra = batch_spectrum(batch_jax)
            spectra_np = np.array(spectra, dtype=np.float32)

            end_idx = min(i + self.config.batch_size, n_samples)
            dset_spec[i:end_idx] = spectra_np
            dset_param[i:end_idx] = batch

            if progress_callback:
                progress_callback(i + len(batch), n_samples, (i + len(batch)) / n_samples)
            elif i % (self.config.batch_size * 10) == 0:
                logger.info(f"Generated {i}/{n_samples} ({i / n_samples:.1%})")

    def generate_manifold(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> None:
        """
        Generate the complete spectral manifold.

        Parameters
        ----------
        progress_callback : callable, optional
            Callback function(completed, total, percentage) for progress updates
        """
        logger.info("Starting manifold generation...")

        # Build parameter grid
        params_arr = self._build_param_grid()
        n_samples = len(params_arr)

        # Create wavelength grid
        wl_grid = jnp.linspace(
            self.config.wavelength_range[0], self.config.wavelength_range[1], self.config.pixels
        )

        # Move atomic data to device
        atomic_data = tuple(jax.device_put(x) for x in self.atomic_data)

        # Broadening dispatch (ADR-0001 T1-4 / bead 8n4i):
        # The LDM path needs a pre-built log-σ grid that brackets the full
        # range of per-line widths the manifold sweep can produce. We bound
        # σ_total from below (cold plasma, heaviest line) and above (hot
        # plasma, lightest line) and pass build_sigma_grid those two
        # endpoints; LDM clips out-of-grid lines to the boundary layers.
        broadening_mode = self.config.broadening_mode

        # Instrument Gaussian σ from configured FWHM (D3 fix).
        fwhm_nm, sigma_inst = self._compute_instrument_sigma()

        if broadening_mode is BroadeningMode.LDM_GAUSSIAN:
            sigma_grid_arr = self._build_ldm_sigma_grid(sigma_inst)
            sigma_grid_device = jax.device_put(sigma_grid_arr)
            logger.info(
                "Manifold broadening: LDM_GAUSSIAN "
                f"(N_sigma={sigma_grid_arr.shape[0]}, "
                f"sigma=[{sigma_grid_arr.min():.4g}, {sigma_grid_arr.max():.4g}] nm, "
                f"instrument_fwhm={fwhm_nm:.4g} nm)"
            )

            @jit
            def batch_spectrum(batch_params):
                return vmap(
                    lambda p: ManifoldGenerator._time_integrated_spectrum_ldm(
                        wl_grid,
                        p,
                        atomic_data,
                        sigma_grid_device,
                        self.config.gate_width_s,
                        self.config.time_steps,
                        sigma_inst,
                        self.config.cooling_t0_s,
                        self.config.cooling_temperature_exponent,
                        self.config.cooling_density_exponent,
                    ),
                    in_axes=0,
                )(batch_params)

        else:
            logger.info(
                f"Manifold broadening: {broadening_mode.value} "
                f"(per-line Voigt with Stark, instrument_fwhm={fwhm_nm:.4g} nm)"
            )

            @jit
            def batch_spectrum(batch_params):
                return vmap(
                    lambda p: ManifoldGenerator._time_integrated_spectrum(
                        wl_grid,
                        p,
                        atomic_data,
                        self.config.gate_width_s,
                        self.config.time_steps,
                        sigma_inst,
                        self.config.cooling_t0_s,
                        self.config.cooling_temperature_exponent,
                        self.config.cooling_density_exponent,
                    ),
                    in_axes=0,
                )(batch_params)

        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        storage_format = _infer_storage_format(output_path)
        chunk_rows = max(1, min(self.config.batch_size, n_samples))

        start_time = time.time()

        output_root, dset_spec, dset_param, attrs = self._create_storage_backend(
            storage_format, output_path, n_samples, chunk_rows, wl_grid
        )

        try:
            attrs["elements"] = list(self.config.elements)
            attrs["wavelength_range"] = list(self.config.wavelength_range)
            attrs["temperature_range"] = list(self.config.temperature_range)
            attrs["density_range"] = list(self.config.density_range)
            attrs["physics_version"] = self.config.physics_version
            attrs["use_voigt_profile"] = self.config.use_voigt_profile
            attrs["use_stark_broadening"] = self.config.use_stark_broadening
            attrs["instrument_fwhm_nm"] = self.config.instrument_fwhm_nm

            self._write_batches(
                params_arr,
                n_samples,
                batch_spectrum,
                dset_spec,
                dset_param,
                progress_callback,
            )
        finally:
            if storage_format == "hdf5":
                output_root.close()

        total_time = time.time() - start_time
        logger.info(
            f"Manifold generation complete: {n_samples} spectra in {total_time:.2f}s "
            f"({n_samples / total_time:.0f} spectra/sec)"
        )
        logger.info(f"Output saved to: {output_path} ({storage_format})")
