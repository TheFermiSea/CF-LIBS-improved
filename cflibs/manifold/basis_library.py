"""
Single-element basis library generator for CF-LIBS.

Generates a pre-computed HDF5 library of per-element emission spectra across
a grid of (T, n_e) conditions.  Each element's spectrum is computed from
first-principles Saha-Boltzmann physics and area-normalised so that a
measured spectrum can be decomposed as a linear combination of basis vectors.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import Transition
from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.radiation.profiles import doppler_width, voigt_profile
from cflibs.radiation.stark import estimate_stark_parameter, stark_hwhm

logger = get_logger("manifold.basis_library")

# FWHM -> Gaussian sigma conversion factor: 2 * sqrt(2 * ln 2).
_FWHM_TO_SIGMA = 2.3548200450309493

# Modest atomic-mass fallback (amu) for the Doppler width when the database
# carries no ``atomic_mass`` column for the element. Values are NIST standard
# atomic weights. Doppler sigma scales as 1/sqrt(M), so a coarse mass is
# adequate; the generic 50 amu fallback matches ManifoldGenerator's default.
_STANDARD_MASSES_AMU = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.01,
    "N": 14.01,
    "O": 16.00,
    "Na": 22.99,
    "Mg": 24.31,
    "Al": 26.98,
    "Si": 28.09,
    "P": 30.97,
    "S": 32.07,
    "Cl": 35.45,
    "K": 39.10,
    "Ca": 40.08,
    "Ti": 47.87,
    "V": 50.94,
    "Cr": 52.00,
    "Mn": 54.94,
    "Fe": 55.85,
    "Co": 58.93,
    "Ni": 58.69,
    "Cu": 63.55,
    "Zn": 65.38,
    "Sr": 87.62,
    "Ba": 137.3,
    "W": 183.8,
    "Au": 197.0,
    "Pb": 207.2,
}
_DEFAULT_MASS_AMU = 50.0  # generic fallback (matches ManifoldGenerator)


@dataclass
class BasisLibraryConfig:
    """Configuration for basis library generation."""

    db_path: str = ""
    output_path: str = "basis_library.h5"
    wavelength_range: Tuple[float, float] = (200.0, 900.0)
    pixels: int = 4096
    temperature_range: Tuple[float, float] = (4000.0, 12000.0)
    temperature_steps: int = 50
    density_range: Tuple[float, float] = (1e15, 5e17)
    density_steps: int = 20
    ionization_stages: Tuple[int, ...] = (1, 2)
    instrument_fwhm_nm: float = 0.05
    # Total density used for ionization balance.  Only affects intermediate
    # magnitudes; final spectra are area-normalised so the value cancels.
    total_density_cm3: float = 1.0

    def validate(self) -> None:
        """Validate configuration ranges."""
        if not self.db_path:
            raise ValueError("db_path must be specified")
        wl_min, wl_max = self.wavelength_range
        if wl_min >= wl_max:
            raise ValueError(f"wavelength_range min ({wl_min}) must be less than max ({wl_max})")
        if wl_min < 0:
            raise ValueError(f"wavelength_range min ({wl_min}) must be non-negative")
        T_min, T_max = self.temperature_range
        if T_min >= T_max:
            raise ValueError(f"temperature_range min ({T_min}) must be less than max ({T_max})")
        if T_min <= 0:
            raise ValueError(f"temperature_range min ({T_min}) must be positive")
        ne_min, ne_max = self.density_range
        if ne_min >= ne_max:
            raise ValueError(f"density_range min ({ne_min}) must be less than max ({ne_max})")
        if ne_min <= 0:
            raise ValueError(f"density_range min ({ne_min}) must be positive")
        if self.pixels <= 0:
            raise ValueError(f"pixels ({self.pixels}) must be positive")
        if self.temperature_steps <= 0:
            raise ValueError(f"temperature_steps ({self.temperature_steps}) must be positive")
        if self.density_steps <= 0:
            raise ValueError(f"density_steps ({self.density_steps}) must be positive")
        if self.instrument_fwhm_nm <= 0:
            raise ValueError(f"instrument_fwhm_nm ({self.instrument_fwhm_nm}) must be positive")
        if not self.ionization_stages:
            raise ValueError("ionization_stages must be non-empty")
        if any(s < 1 for s in self.ionization_stages):
            raise ValueError("ionization_stages must contain positive integers")


class BasisLibraryGenerator:
    """Generates a single-element basis library over a (T, n_e) grid."""

    def __init__(self, config: BasisLibraryConfig):
        self.config = config
        self.atomic_db = AtomicDatabase(config.db_path)
        self.solver = SahaBoltzmannSolver(self.atomic_db)

    @staticmethod
    def _build_grids(
        cfg: BasisLibraryConfig,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Build wavelength grid, instrument Gaussian sigma, and parameter grid.

        Returns
        -------
        wl_grid : ndarray of shape (pixels,)
        sigma_inst : float
            Instrument-only Gaussian sigma in nm. The per-line Doppler width
            (wavelength- and temperature-dependent) is added in quadrature at
            render time, so this is no longer the total per-line sigma.
        params : ndarray of shape (n_grid, 2)
        """
        wl_min, wl_max = cfg.wavelength_range
        wl_grid = np.linspace(wl_min, wl_max, cfg.pixels)

        T_grid = np.linspace(*cfg.temperature_range, cfg.temperature_steps)
        ne_grid = np.geomspace(*cfg.density_range, cfg.density_steps)

        n_grid = cfg.temperature_steps * cfg.density_steps
        params = np.empty((n_grid, 2), dtype=np.float64)
        idx = 0
        for T_K in T_grid:
            for ne in ne_grid:
                params[idx, 0] = T_K
                params[idx, 1] = ne
                idx += 1

        sigma_inst = cfg.instrument_fwhm_nm / _FWHM_TO_SIGMA  # FWHM -> sigma
        return wl_grid, sigma_inst, params

    def _element_mass_amu(self, element: str) -> float:
        """Resolve the atomic mass (amu) for *element* for the Doppler width.

        Prefers the database ``atomic_mass`` column, then a NIST standard-weight
        fallback table, then a generic 50 amu default (matching
        :class:`~cflibs.manifold.generator.ManifoldGenerator`).
        """
        db_mass = self.atomic_db.get_atomic_mass(element)
        if db_mass is not None and db_mass > 0.0:
            return float(db_mass)
        if element in _STANDARD_MASSES_AMU:
            return _STANDARD_MASSES_AMU[element]
        logger.debug("No atomic mass for %s; using fallback %.1f amu", element, _DEFAULT_MASS_AMU)
        return _DEFAULT_MASS_AMU

    def _stark_w_ref(self, trans: Transition, ip_cache: dict) -> float:
        """Return the Stark reference FWHM (nm) at REF_NE for *trans*.

        Uses the database value when present, otherwise the same binding-energy
        estimate the manifold generator uses
        (:func:`cflibs.radiation.stark.estimate_stark_parameter`), so the basis
        fingerprints carry a Stark Lorentzian even for lines lacking tabulated
        widths — matching the forward model they approximate.
        """
        if trans.stark_w is not None and trans.stark_w > 0.0:
            return float(trans.stark_w)
        key = (trans.element, trans.ionization_stage)
        if key not in ip_cache:
            ip_cache[key] = self.atomic_db.get_ionization_potential(
                trans.element, trans.ionization_stage
            )
        return estimate_stark_parameter(
            trans.wavelength_nm,
            trans.E_k_ev,
            ip_cache[key],
            trans.ionization_stage,
        )

    def _compute_element_spectra(
        self,
        element: str,
        wl_grid: np.ndarray,
        sigma_inst: float,
        params: np.ndarray,
    ) -> np.ndarray:
        """Compute area-normalised spectra for *element* at every grid point.

        Line shapes are rendered with the SAME Voigt model the full manifold
        generator uses (single source of truth: :func:`doppler_width`,
        :func:`stark_hwhm`, :func:`voigt_profile`), so the basis fingerprints
        match the forward model they approximate:

        * Doppler Gaussian sigma is per-line and wavelength/temperature
          dependent — canonical ``sigma = (lambda / c) * sqrt(kT / m)`` (the
          factor inside :func:`doppler_width`), NOT the spurious ``sqrt(2kT/m)``.
        * The instrument Gaussian sigma adds in quadrature.
        * An ``n_e``-dependent Stark Lorentzian HWHM is folded in, scaled as
          ``0.5 * stark_w_ref * (n_e / REF_NE) * (T / T_ref)^(-alpha)`` with
          ``REF_NE = 1e17`` (:func:`stark_hwhm`).

        Returns
        -------
        ndarray of shape (n_grid, n_pix)
        """
        cfg = self.config
        wl_min, wl_max = cfg.wavelength_range
        n_grid = params.shape[0]
        n_pix = len(wl_grid)

        transitions = []
        for stage in cfg.ionization_stages:
            transitions.extend(
                self.atomic_db.get_transitions(
                    element,
                    ionization_stage=stage,
                    wavelength_min=wl_min,
                    wavelength_max=wl_max,
                )
            )

        out = np.zeros((n_grid, n_pix), dtype=np.float64)
        if not transitions:
            logger.debug("No transitions for %s in %.1f-%.1f nm", element, wl_min, wl_max)
            return out

        mass_amu = self._element_mass_amu(element)
        # Per-line static quantities (mass/IP do not vary across the grid).
        ip_cache: dict = {}
        stark_w_refs = np.array(
            [self._stark_w_ref(t, ip_cache) for t in transitions], dtype=np.float64
        )
        stark_alphas = np.array(
            [t.stark_alpha if t.stark_alpha is not None else 0.5 for t in transitions],
            dtype=np.float64,
        )

        for grid_idx in range(n_grid):
            T_K = params[grid_idx, 0]
            ne = params[grid_idx, 1]
            T_eV = T_K * KB_EV

            stage_densities = self.solver.solve_ionization_balance(
                element, T_eV, ne, total_density_cm3=cfg.total_density_cm3
            )

            spectrum = np.zeros(n_pix, dtype=np.float64)

            for li, trans in enumerate(transitions):
                stage_density = stage_densities.get(trans.ionization_stage, 0.0)
                if stage_density <= 0.0:
                    continue

                U = self.solver.calculate_partition_function(element, trans.ionization_stage, T_eV)
                if U <= 0.0:
                    continue

                n_k = stage_density * (trans.g_k / U) * np.exp(-trans.E_k_ev / T_eV)

                # Emissivity ∝ A_ki · n_k / λ.  The 1/λ factor is kept because
                # it varies per line and does not cancel in area normalization.
                eps = trans.A_ki * n_k / trans.wavelength_nm

                # Doppler Gaussian sigma (canonical λ·sqrt(kT/m)/c form, via the
                # shared doppler_width FWHM helper) ⊕ instrument sigma.
                sigma_dopp = doppler_width(trans.wavelength_nm, T_eV, mass_amu) / _FWHM_TO_SIGMA
                sigma_total = float(np.hypot(sigma_dopp, sigma_inst))

                # n_e-dependent Stark Lorentzian HWHM (REF_NE = 1e17), same law
                # as the generator's _calculate_stark_hwhm.
                gamma_stark = stark_hwhm(
                    ne,
                    T_K,
                    float(stark_w_refs[li]),
                    float(stark_alphas[li]),
                )

                spectrum += voigt_profile(
                    wl_grid,
                    trans.wavelength_nm,
                    sigma_total,
                    gamma_stark,
                    eps,
                )

            area = np.sum(spectrum)
            if area > 1e-100:
                spectrum /= area

            out[grid_idx, :] = spectrum

        return out

    def generate(self, progress_callback: Optional[Callable] = None) -> str:
        """Generate basis library and save to HDF5.

        Parameters
        ----------
        progress_callback : callable, optional
            Called with (element_index, total_elements) for progress reporting.

        Returns
        -------
        str
            Path to the generated HDF5 file.
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required for basis library generation")

        cfg = self.config
        cfg.validate()

        wl_grid, sigma_inst, params = self._build_grids(cfg)
        elements = self.atomic_db.get_available_elements()
        n_el = len(elements)
        n_grid = params.shape[0]
        spectra = np.zeros((n_el, n_grid, cfg.pixels), dtype=np.float32)

        for el_idx, element in enumerate(elements):
            logger.info("Generating basis for %s (%d/%d)", element, el_idx + 1, n_el)
            spectra[el_idx, :, :] = self._compute_element_spectra(
                element, wl_grid, sigma_inst, params
            )
            if progress_callback is not None:
                progress_callback(el_idx + 1, n_el)

        # Save to HDF5
        output_path = cfg.output_path
        with h5py.File(output_path, "w") as f:
            f.create_dataset("spectra", data=spectra, compression="gzip", compression_opts=4)
            f.create_dataset("params", data=params)
            f.create_dataset("wavelength", data=wl_grid)
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("elements", data=np.array(elements, dtype=object), dtype=dt)

        logger.info(
            "Basis library saved to %s (%d elements, %d grid points)", output_path, n_el, n_grid
        )
        return output_path


class BasisLibrary:
    """Loader for a pre-computed basis library stored in HDF5."""

    def __init__(self, path: str):
        if not HAS_H5PY:
            raise ImportError("h5py is required to load a basis library")
        self._f = h5py.File(path, "r")
        self._spectra = self._f["spectra"]  # lazy (n_el, n_grid, n_pix)
        self._params = self._f["params"][:]  # (n_grid, 2)
        self._wavelength = self._f["wavelength"][:]  # (n_pix,)
        self._elements = [e.decode() if isinstance(e, bytes) else e for e in self._f["elements"][:]]
        self._element_to_idx = {el: i for i, el in enumerate(self._elements)}
        self._T_vals = np.unique(self._params[:, 0])
        self._ne_vals = np.unique(self._params[:, 1])

    def get_basis_matrix(self, T_K: float, ne_cm3: float) -> np.ndarray:
        """Return (n_elements, n_pixels) basis matrix at nearest grid point."""
        if ne_cm3 <= 0:
            raise ValueError(f"ne_cm3 must be positive, got {ne_cm3}")
        grid_idx = self._nearest_grid_idx(T_K, ne_cm3)
        return np.array(self._spectra[:, grid_idx, :])

    def _nearest_grid_idx(self, T_K: float, ne_cm3: float) -> int:
        """Return the index of the nearest grid point.

        Both axes are normalised by their own *span* (T by ``(T_max - T_min)``,
        log10(n_e) by ``(log10 n_e_max - log10 n_e_min)``) so the distance is
        dimensionless and symmetric: equal fractional offsets along T and n_e
        contribute equally. The previous form normalised T by the absolute
        T_max (not the span) while n_e used the log10 range, biasing the
        nearest-neighbour pick toward the temperature axis.
        """
        # Span guards: a single grid value along an axis carries no information,
        # so that axis contributes zero distance (avoids divide-by-zero).
        T_span = self._T_vals[-1] - self._T_vals[0]
        log_ne_span = np.log10(self._ne_vals[-1]) - np.log10(self._ne_vals[0])

        if T_span > 0.0:
            t_term = ((self._params[:, 0] - T_K) / T_span) ** 2
        else:
            t_term = np.zeros(self._params.shape[0], dtype=np.float64)

        if log_ne_span > 0.0:
            ne_term = ((np.log10(self._params[:, 1]) - np.log10(ne_cm3)) / log_ne_span) ** 2
        else:
            ne_term = np.zeros(self._params.shape[0], dtype=np.float64)

        dists = t_term + ne_term
        return int(np.argmin(dists))

    def get_basis_matrix_interp(self, T_K: float, ne_cm3: float) -> np.ndarray:
        """Bilinear interpolation in (T, log10(ne)) space."""
        if len(self._T_vals) < 2 or len(self._ne_vals) < 2:
            return self.get_basis_matrix(T_K, ne_cm3)
        T_K = np.clip(T_K, self._T_vals[0], self._T_vals[-1])
        ne_cm3 = np.clip(ne_cm3, self._ne_vals[0], self._ne_vals[-1])

        T_idx = int(np.searchsorted(self._T_vals, T_K) - 1)
        T_idx = np.clip(T_idx, 0, len(self._T_vals) - 2)

        log_ne = np.log10(ne_cm3)
        log_ne_vals = np.log10(self._ne_vals)
        ne_idx = int(np.searchsorted(log_ne_vals, log_ne) - 1)
        ne_idx = np.clip(ne_idx, 0, len(self._ne_vals) - 2)

        # Interpolation weights
        t_frac = (T_K - self._T_vals[T_idx]) / (self._T_vals[T_idx + 1] - self._T_vals[T_idx])
        n_frac = (log_ne - log_ne_vals[ne_idx]) / (log_ne_vals[ne_idx + 1] - log_ne_vals[ne_idx])

        # Four corner spectra
        n_ne = len(self._ne_vals)

        def _grid_idx(ti: int, ni: int) -> int:
            return ti * n_ne + ni

        s00 = self._spectra[:, _grid_idx(T_idx, ne_idx), :]
        s10 = self._spectra[:, _grid_idx(T_idx + 1, ne_idx), :]
        s01 = self._spectra[:, _grid_idx(T_idx, ne_idx + 1), :]
        s11 = self._spectra[:, _grid_idx(T_idx + 1, ne_idx + 1), :]

        return (
            (1 - t_frac) * (1 - n_frac) * s00
            + t_frac * (1 - n_frac) * s10
            + (1 - t_frac) * n_frac * s01
            + t_frac * n_frac * s11
        )

    def get_element_spectrum(self, element: str, T_K: float, ne_cm3: float) -> np.ndarray:
        """Return single-element spectrum at nearest grid point.

        Raises
        ------
        KeyError
            If *element* is not in the library.
        """
        if ne_cm3 <= 0:
            raise ValueError(f"ne_cm3 must be positive, got {ne_cm3}")
        idx = self._element_to_idx[element]
        grid_idx = self._nearest_grid_idx(T_K, ne_cm3)
        return np.array(self._spectra[idx, grid_idx, :])

    @property
    def elements(self) -> list:
        return list(self._elements)

    @property
    def wavelength(self) -> np.ndarray:
        return self._wavelength.copy()

    @property
    def n_elements(self) -> int:
        return len(self._elements)

    @property
    def n_grid(self) -> int:
        return self._params.shape[0]

    @property
    def n_pixels(self) -> int:
        return len(self._wavelength)

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
