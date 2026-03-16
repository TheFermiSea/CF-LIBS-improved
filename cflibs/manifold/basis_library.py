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
from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver

logger = get_logger("manifold.basis_library")


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
        """Build wavelength grid, Gaussian sigma, and parameter grid.

        Returns
        -------
        wl_grid : ndarray of shape (pixels,)
        sigma : float
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

        sigma = cfg.instrument_fwhm_nm / 2.3548200450309493  # FWHM -> sigma
        return wl_grid, sigma, params

    def _compute_element_spectra(
        self,
        element: str,
        wl_grid: np.ndarray,
        sigma: float,
        params: np.ndarray,
    ) -> np.ndarray:
        """Compute area-normalised spectra for *element* at every grid point.

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

        for grid_idx in range(n_grid):
            T_K = params[grid_idx, 0]
            ne = params[grid_idx, 1]
            T_eV = T_K * KB_EV

            stage_densities = self.solver.solve_ionization_balance(
                element, T_eV, ne, total_density_cm3=cfg.total_density_cm3
            )

            spectrum = np.zeros(n_pix, dtype=np.float64)

            for trans in transitions:
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

                spectrum += eps * np.exp(-0.5 * ((wl_grid - trans.wavelength_nm) / sigma) ** 2)

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

        wl_grid, sigma, params = self._build_grids(cfg)
        elements = self.atomic_db.get_available_elements()
        n_el = len(elements)
        n_grid = params.shape[0]
        spectra = np.zeros((n_el, n_grid, cfg.pixels), dtype=np.float32)

        for el_idx, element in enumerate(elements):
            logger.info("Generating basis for %s (%d/%d)", element, el_idx + 1, n_el)
            spectra[el_idx, :, :] = self._compute_element_spectra(element, wl_grid, sigma, params)
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
        """Return the index of the nearest grid point."""
        dists = (self._params[:, 0] - T_K) ** 2 / self._T_vals[-1] ** 2 + (
            np.log10(self._params[:, 1]) - np.log10(ne_cm3)
        ) ** 2 / np.log10(self._ne_vals[-1] / self._ne_vals[0]) ** 2
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
