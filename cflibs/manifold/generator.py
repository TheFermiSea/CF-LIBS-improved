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

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]

    # Define dummy decorators to allow class definition
    def jit(func):
        return func

    def vmap(func, *args, **kwargs):
        return func


from cflibs.core.constants import SAHA_CONST_CM3, C_LIGHT, EV_TO_J, EV_TO_K, H_PLANCK, M_PROTON
from cflibs.atomic.database import AtomicDatabase
from cflibs.manifold.config import ManifoldConfig
from cflibs.core.logging_config import get_logger
from cflibs.plasma.partition import polynomial_partition_function_jax

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

        # Standard atomic masses (amu) for fallback
        STANDARD_MASSES = {
            "H": 1.008,
            "He": 4.003,
            "Li": 6.941,
            "Be": 9.012,
            "B": 10.81,
            "C": 12.01,
            "N": 14.01,
            "O": 16.00,
            "F": 19.00,
            "Ne": 20.18,
            "Na": 22.99,
            "Mg": 24.31,
            "Al": 26.98,
            "Si": 28.09,
            "P": 30.97,
            "S": 32.07,
            "Cl": 35.45,
            "Ar": 39.95,
            "K": 39.10,
            "Ca": 40.08,
            "Sc": 44.96,
            "Ti": 47.87,
            "V": 50.94,
            "Cr": 52.00,
            "Mn": 54.94,
            "Fe": 55.85,
            "Co": 58.93,
            "Ni": 58.69,
            "Cu": 63.55,
            "Zn": 65.38,
            "Ga": 69.72,
            "Ge": 72.63,
            "As": 74.92,
            "Se": 78.97,
            "Br": 79.90,
            "Kr": 83.80,
            "Rb": 85.47,
            "Sr": 87.62,
            "Y": 88.91,
            "Zr": 91.22,
            "Nb": 92.91,
            "Mo": 95.95,
            "Ru": 101.1,
            "Rh": 102.9,
            "Pd": 106.4,
            "Ag": 107.9,
            "Cd": 112.4,
            "In": 114.8,
            "Sn": 118.7,
            "Sb": 121.8,
            "Te": 127.6,
            "I": 126.9,
            "Xe": 131.3,
            "Cs": 132.9,
            "Ba": 137.3,
            "La": 138.9,
            "Ce": 140.1,
            "Pr": 140.9,
            "Nd": 144.2,
            "Sm": 150.4,
            "Eu": 152.0,
            "Gd": 157.3,
            "Tb": 158.9,
            "Dy": 162.5,
            "Ho": 164.9,
            "Er": 167.3,
            "Tm": 168.9,
            "Yb": 173.0,
            "Lu": 175.0,
            "Hf": 178.5,
            "Ta": 180.9,
            "W": 183.8,
            "Re": 186.2,
            "Os": 190.2,
            "Ir": 192.2,
            "Pt": 195.1,
            "Au": 197.0,
            "Hg": 200.6,
            "Tl": 204.4,
            "Pb": 207.2,
            "Bi": 209.0,
            "U": 238.0,
        }

        # Build query for all elements (including Stark parameters)
        placeholders = ",".join(["?"] * len(self.config.elements))
        query = f"""
            SELECT
                l.element, l.sp_num, l.wavelength_nm, l.aki, l.ek_ev, l.gk,
                sp.ip_ev, l.stark_w, l.stark_alpha
            FROM lines l
            JOIN species_physics sp ON l.element = sp.element AND l.sp_num = sp.sp_num
            WHERE l.wavelength_nm BETWEEN ? AND ?
            AND l.element IN ({placeholders})
            ORDER BY l.wavelength_nm
        """
        params = [
            self.config.wavelength_range[0],
            self.config.wavelength_range[1],
        ] + self.config.elements

        df = pd.read_sql_query(query, self.atomic_db.conn, params=params)

        if df.empty:
            raise ValueError(
                f"No atomic data found for elements {self.config.elements} "
                f"in wavelength range {self.config.wavelength_range}"
            )

        # Map element names to indices
        el_map = {el: i for i, el in enumerate(self.config.elements)}
        df["el_idx"] = df["element"].map(el_map)

        # Load atomic masses per element from database (with fallback)
        element_masses = {}
        for el in self.config.elements:
            db_mass = self.atomic_db.get_atomic_mass(el)
            if db_mass is not None:
                element_masses[el] = db_mass
            elif el in STANDARD_MASSES:
                element_masses[el] = STANDARD_MASSES[el]
                logger.debug(f"Using standard mass for {el}: {STANDARD_MASSES[el]} amu")
            else:
                element_masses[el] = 50.0  # Generic fallback
                logger.warning(f"No mass found for {el}, using fallback 50.0 amu")

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

        # Set defaults for coeffs (approximate log(U))
        coeffs[:, 0, 0] = np.log(25.0)
        coeffs[:, 1, 0] = np.log(15.0)
        coeffs[:, 2, 0] = np.log(10.0)

        # Load Physics Data (IPs and Coeffs)
        try:
            cursor = self.atomic_db.conn.cursor()

            # Load IPs
            ip_query = f"""
                SELECT element, sp_num, ip_ev
                FROM species_physics
                WHERE element IN ({placeholders})
            """
            cursor.execute(ip_query, self.config.elements)
            for row in cursor.fetchall():
                el, sp_num, ip_ev = row
                if el in el_map and ip_ev is not None:
                    el_idx = el_map[el]
                    stage_idx = sp_num - 1
                    if 0 <= stage_idx < max_stages:
                        ips[el_idx, stage_idx] = ip_ev

            # Load Partition Coeffs
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='partition_functions'"
            )
            if cursor.fetchone():
                pf_query = f"""
                    SELECT element, sp_num, a0, a1, a2, a3, a4
                    FROM partition_functions
                    WHERE element IN ({placeholders})
                """
                cursor.execute(pf_query, self.config.elements)
                count = 0
                for row in cursor.fetchall():
                    el, sp_num, a0, a1, a2, a3, a4 = row
                    if el in el_map:
                        el_idx = el_map[el]
                        stage_idx = sp_num - 1
                        if 0 <= stage_idx < max_stages:
                            coeffs[el_idx, stage_idx] = [a0, a1, a2, a3, a4]
                            count += 1
                logger.info(f"Loaded partition coefficients for {count} species")

        except Exception as e:
            logger.warning(f"Failed to load physics data: {e}")

        partition_coeffs = jnp.array(coeffs, dtype=jnp.float32)
        ionization_potentials = jnp.array(ips, dtype=jnp.float32)

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
        )

    @staticmethod
    def _calculate_partition_functions(t_k, atomic_data):
        """Calculates neutral and ion partition functions for all lines' elements."""
        lines_el_idx = atomic_data[6]
        partition_coeffs = atomic_data[7]
        coeffs_0 = partition_coeffs[lines_el_idx, 0]
        coeffs_1 = partition_coeffs[lines_el_idx, 1]
        u0 = polynomial_partition_function_jax(t_k, coeffs_0)
        u1 = polynomial_partition_function_jax(t_k, coeffs_1)
        return u0, u1

    @staticmethod
    def _calculate_saha_fractions(t_ev, n_e, u0, u1, atomic_data):
        """Calculates the Saha ionization population fractions."""
        lines_el_idx = atomic_data[6]
        ionization_potentials = atomic_data[8]
        ip_i = ionization_potentials[lines_el_idx, 0]
        saha_factor = (SAHA_CONST_CM3 / n_e) * (t_ev**1.5)
        ratio_n1_n0 = saha_factor * (u1 / u0) * jnp.exp(-ip_i / t_ev)
        frac0 = 1.0 / (1.0 + ratio_n1_n0)
        frac1 = ratio_n1_n0 / (1.0 + ratio_n1_n0)
        return frac0, frac1

    @staticmethod
    def _calculate_boltzmann_populations(
        plasma_state,
        saha_state,
        atomic_data,
    ):
        """Calculates upper level populations using the Boltzmann equation."""
        t_ev, n_e, concentration_map = plasma_state
        u0, u1, frac0, frac1 = saha_state

        lines_ek = atomic_data[2]
        lines_gk = atomic_data[3]
        lines_z = atomic_data[5]
        lines_el_idx = atomic_data[6]

        pop_fraction = jnp.where(lines_z == 0, frac0, frac1)
        u_val = jnp.where(lines_z == 0, u0, u1)
        element_conc = concentration_map[lines_el_idx]
        n_species_total = element_conc * n_e
        n_species = n_species_total * pop_fraction
        n_upper = n_species * (lines_gk / u_val) * jnp.exp(-lines_ek / t_ev)
        return n_upper

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

        u0, u1 = ManifoldGenerator._calculate_partition_functions(t_k, atomic_data)

        frac0, frac1 = ManifoldGenerator._calculate_saha_fractions(T_eV, n_e, u0, u1, atomic_data)

        n_upper = ManifoldGenerator._calculate_boltzmann_populations(
            (T_eV, n_e, concentration_map),
            (u0, u1, frac0, frac1),
            atomic_data,
        )

        return n_upper

    @staticmethod
    @jit
    def _compute_spectrum_snapshot(
        wl_grid: jnp.ndarray,
        T_eV: float,
        n_e: float,
        concentrations: jnp.ndarray,
        atomic_data: Tuple,
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
        ) = atomic_data

        # Solve populations

        n_upper = ManifoldGenerator._saha_eggert_solver(
            T_eV,
            n_e,
            concentrations,
            atomic_data,
        )

        # Line emissivity: epsilon = (hc / 4pi lambda) * A * n_upper

        epsilon = (H_PLANCK * C_LIGHT / (4 * jnp.pi * l_wl * 1e-9)) * l_aki * n_upper

        # --- Proper Voigt Broadening (Phase 2) ---

        # Doppler width: sigma = lambda/c * sqrt(2kT/m)
        # Using JAX-compatible doppler_sigma_jax
        mass_kg = l_mass_amu * M_PROTON
        sigma_doppler = l_wl * jnp.sqrt(2.0 * T_eV * EV_TO_J / (mass_kg * C_LIGHT**2))

        # Instrument broadening (Gaussian sigma)
        # Uses default 0.05 nm FWHM; configurable via ManifoldConfig.instrument_fwhm_nm
        sigma_inst = 0.05 / 2.355  # FWHM -> sigma

        # Total Gaussian width
        sigma_total = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)

        # Stark broadening: HWHM (Lorentzian gamma)
        # Use estimate_stark_parameter_jax for missing values (NaN)
        # w_stark = w_ref * (n_e / 1e16) * (T / T_ref)^(-alpha)
        REF_NE = 1.0e16
        REF_T_EV = 0.86173  # 10000 K in eV

        # Estimate Stark w_ref for lines without database values
        # Use binding energy = IP - E_upper
        binding_energy = jnp.maximum(l_ip - l_ek, 0.1)
        n_eff = (l_z + 1) * jnp.sqrt(13.605 / binding_energy)
        w_est = 2.0e-5 * (l_wl / 500.0) ** 2 * (n_eff**4)
        w_est = jnp.clip(w_est, 0.0001, 0.5)

        # Use database value if available, else estimate
        w_ref = jnp.where(jnp.isnan(l_stark_w), w_est, l_stark_w)

        # Stark HWHM calculation
        factor_ne = n_e / REF_NE
        factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -l_stark_alpha)
        gamma_stark = w_ref * factor_ne * factor_T

        # --- Voigt Profile Rendering (Humlicek W4 approximation) ---
        # For each wavelength point, compute Voigt profile contribution from all lines
        # z = (x + i*gamma) / (sigma * sqrt(2))
        # V(x) = Re(w(z)) / (sigma * sqrt(2*pi))

        diff = wl_grid[:, None] - l_wl[None, :]  # (n_wl, n_lines)

        # Compute Voigt profile using Humlicek W4 approximation
        z = (diff + 1j * gamma_stark) / (sigma_total * jnp.sqrt(2.0))

        # Humlicek W4 Faddeeva approximation (no complex erfc needed)
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
        w_z = jnp.where(
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

        profile = jnp.real(w_z) / (sigma_total * jnp.sqrt(2.0 * jnp.pi))

        # Sum contributions weighted by emissivity

        intensity = jnp.sum(epsilon * profile, axis=1)

        return intensity

    @staticmethod
    @jit
    def _time_integrated_spectrum(
        wl_grid: jnp.ndarray,
        params: jnp.ndarray,
        atomic_data: Tuple,
        gate_width_s: float,
        time_steps: int,
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

        # Cooling laws (power law decay)
        t0 = 1e-6
        T_trail = T_max * (1 + times / t0) ** (-0.5)
        ne_trail = ne_max * (1 + times / t0) ** (-1.0)

        # Integrate over time
        def step_fn(carry, inputs):
            T, ne = inputs
            intensity = jnp.where(
                T > 0.4,  # Only if T > 0.4 eV
                ManifoldGenerator._compute_spectrum_snapshot(wl_grid, T, ne, concs, atomic_data),
                jnp.zeros_like(wl_grid),
            )
            return carry + intensity * dt, None

        spectrum_accum = jnp.zeros_like(wl_grid)
        spectrum_accum, _ = jax.lax.scan(step_fn, spectrum_accum, (T_trail, ne_trail))

        return spectrum_accum

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
        T_grid = np.linspace(
            self.config.temperature_range[0],
            self.config.temperature_range[1],
            self.config.temperature_steps,
        )
        ne_grid = np.geomspace(
            self.config.density_range[0], self.config.density_range[1], self.config.density_steps
        )

        # Build concentration grid (simplex for multi-element)
        # For now, simple grid for Ti-Al-V system
        params_list = []

        if len(self.config.elements) == 4:  # Ti-Al-V-Fe system
            al_range = np.linspace(0, 0.12, self.config.concentration_steps)
            v_range = np.linspace(0, 0.12, self.config.concentration_steps)

            for T in T_grid:
                for ne in ne_grid:
                    for al in al_range:
                        for v in v_range:
                            ti = 1.0 - (al + v)
                            if ti < 0:
                                continue
                            # [T, ne, Ti, Al, V, Fe]
                            params_list.append([T, ne, ti, al, v, 0.002])
        else:
            # Generic: vary first element concentration
            conc_range = np.linspace(0.5, 1.0, self.config.concentration_steps)
            for T in T_grid:
                for ne in ne_grid:
                    for c1 in conc_range:
                        # Simple case: one varying element
                        concs = [c1] + [0.0] * (len(self.config.elements) - 1)
                        params_list.append([T, ne] + concs)

        params_arr = np.array(params_list, dtype=np.float32)
        n_samples = len(params_arr)

        logger.info(f"Parameter grid: {n_samples} spectra to generate")
        logger.info(f"  Temperature: {len(T_grid)} points")
        logger.info(f"  Density: {len(ne_grid)} points")
        logger.info(
            f"  Concentrations: {len(params_list) // (len(T_grid) * len(ne_grid))} combinations"
        )

        # Create wavelength grid
        wl_grid = jnp.linspace(
            self.config.wavelength_range[0], self.config.wavelength_range[1], self.config.pixels
        )

        # Move atomic data to device
        atomic_data = tuple(jax.device_put(x) for x in self.atomic_data)

        # Vectorized function
        @jit
        def batch_spectrum(batch_params):
            return vmap(
                lambda p: ManifoldGenerator._time_integrated_spectrum(
                    wl_grid, p, atomic_data, self.config.gate_width_s, self.config.time_steps
                ),
                in_axes=0,
            )(batch_params)

        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        storage_format = _infer_storage_format(output_path)
        chunk_rows = max(1, min(self.config.batch_size, n_samples))

        start_time = time.time()

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

        try:
            attrs["elements"] = list(self.config.elements)
            attrs["wavelength_range"] = list(self.config.wavelength_range)
            attrs["temperature_range"] = list(self.config.temperature_range)
            attrs["density_range"] = list(self.config.density_range)
            attrs["physics_version"] = self.config.physics_version
            attrs["use_voigt_profile"] = self.config.use_voigt_profile
            attrs["use_stark_broadening"] = self.config.use_stark_broadening
            attrs["instrument_fwhm_nm"] = self.config.instrument_fwhm_nm

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
        finally:
            if storage_format == "hdf5":
                output_root.close()

        total_time = time.time() - start_time
        logger.info(
            f"Manifold generation complete: {n_samples} spectra in {total_time:.2f}s "
            f"({n_samples / total_time:.0f} spectra/sec)"
        )
        logger.info(f"Output saved to: {output_path} ({storage_format})")
