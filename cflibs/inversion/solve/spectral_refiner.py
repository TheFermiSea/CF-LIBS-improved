"""
Gradient-based refinement of NNLS element identification results.

Given initial (T, ne, concentrations) from the NNLS decomposition step,
jointly optimizes all parameters to minimize the chi-squared residual
between the observed spectrum and a forward-model synthetic spectrum
constructed from the pre-computed basis library.

Uses scipy.optimize.minimize (L-BFGS-B) with bounded parameters.
JAX acceleration is not required -- works with pure NumPy/SciPy.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from cflibs.core.logging_config import get_logger
from cflibs.manifold.basis_library import BasisLibrary

logger = get_logger("inversion.spectral_refiner")

# Default parameter bounds
_T_BOUNDS_K = (3000.0, 30000.0)
_LOG_NE_BOUNDS = (14.0, 19.0)
_CONC_BOUNDS = (0.0, 1.0)


@dataclass
class RefinementResult:
    """Result of gradient-based refinement of plasma parameters.

    Attributes
    ----------
    T_K : float
        Optimized temperature in Kelvin.
    ne_cm3 : float
        Optimized electron density in cm^-3.
    concentrations : Dict[str, float]
        Optimized element concentrations (keyed by element symbol).
    residual_norm : float
        L2 norm of the final residual vector.
    n_iterations : int
        Number of optimizer iterations performed.
    converged : bool
        Whether the optimizer reported successful convergence.
    chi_squared : float
        Sum of squared weighted residuals at the optimum.
    chi_squared_reduced : float
        chi_squared / (n_pixels - n_params).
    """

    T_K: float
    ne_cm3: float
    concentrations: Dict[str, float]
    residual_norm: float
    n_iterations: int
    converged: bool
    chi_squared: float
    chi_squared_reduced: float


class SpectralRefiner:
    """Refine NNLS element identification via gradient-based optimization.

    Given initial (T, ne, concentrations) from the NNLS step, optimizes
    these parameters to minimize the chi-squared residual between the
    observed spectrum and a forward-model synthetic spectrum interpolated
    from the pre-computed basis library.

    Uses ``scipy.optimize.minimize`` with method ``L-BFGS-B`` and bounded
    parameters.  No JAX dependency is required.

    Parameters
    ----------
    basis_library : BasisLibrary
        Pre-computed single-element basis library.
    max_iterations : int
        Maximum L-BFGS-B iterations (default: 20).
    """

    def __init__(
        self,
        basis_library: BasisLibrary,
        max_iterations: int = 20,
    ):
        self.basis_library = basis_library
        self.max_iterations = max_iterations

    def refine(
        self,
        wavelength: np.ndarray,
        observed: np.ndarray,
        detected_elements: List[str],
        T_init_K: float,
        ne_init_cm3: float,
        concentrations_init: Dict[str, float],
        noise: Optional[np.ndarray] = None,
    ) -> RefinementResult:
        """Refine plasma parameters from an NNLS starting point.

        Parameters
        ----------
        wavelength : np.ndarray
            Observed wavelength grid (nm).  If it differs from the basis
            library grid the observed spectrum is resampled.
        observed : np.ndarray
            Observed (baseline-corrected, area-normalised) spectrum.
        detected_elements : List[str]
            Elements to include in the fit.
        T_init_K : float
            Initial temperature in Kelvin.
        ne_init_K : float
            Initial electron density in cm^-3.
        concentrations_init : Dict[str, float]
            Initial concentration per element.
        noise : np.ndarray, optional
            Per-pixel noise estimate.  If *None*, a uniform noise floor
            is estimated from the data.

        Returns
        -------
        RefinementResult
        """
        # --- Edge cases -------------------------------------------------
        if not detected_elements:
            return RefinementResult(
                T_K=T_init_K,
                ne_cm3=ne_init_cm3,
                concentrations={},
                residual_norm=float(np.linalg.norm(observed)),
                n_iterations=0,
                converged=True,
                chi_squared=0.0,
                chi_squared_reduced=0.0,
            )

        # --- Resample observed to basis library wavelength grid ----------
        lib_wl = self.basis_library.wavelength
        if len(wavelength) != len(lib_wl) or not np.allclose(wavelength, lib_wl):
            obs = np.interp(lib_wl, wavelength, observed)
        else:
            obs = np.asarray(observed, dtype=np.float64)

        n_pixels = len(obs)

        # --- Noise estimate ---------------------------------------------
        if noise is not None:
            if len(noise) != len(wavelength):
                sigma = np.interp(lib_wl, wavelength, noise)
            elif len(noise) != n_pixels:
                sigma = np.interp(lib_wl, wavelength, noise)
            else:
                sigma = np.asarray(noise, dtype=np.float64)
        else:
            # MAD-based estimate of baseline noise
            sigma = np.full(n_pixels, max(np.median(np.abs(obs)) * 0.05, 1e-30))

        sigma = np.maximum(sigma, 1e-30)  # prevent division by zero

        # --- Filter to elements present in the library -------------------
        lib_elements = self.basis_library.elements
        element_indices: List[int] = []
        elements_used: List[str] = []
        for el in detected_elements:
            if el in lib_elements:
                element_indices.append(lib_elements.index(el))
                elements_used.append(el)

        if not elements_used:
            return RefinementResult(
                T_K=T_init_K,
                ne_cm3=ne_init_cm3,
                concentrations={el: concentrations_init.get(el, 0.0) for el in detected_elements},
                residual_norm=float(np.linalg.norm(obs)),
                n_iterations=0,
                converged=True,
                chi_squared=0.0,
                chi_squared_reduced=0.0,
            )

        n_elements = len(elements_used)
        n_params = 2 + n_elements  # T, log10(ne), c_1..c_n

        # --- Pack initial parameter vector --------------------------------
        x0 = np.empty(n_params, dtype=np.float64)
        x0[0] = np.clip(T_init_K, *_T_BOUNDS_K)
        x0[1] = np.clip(np.log10(max(ne_init_cm3, 1.0)), *_LOG_NE_BOUNDS)
        for i, el in enumerate(elements_used):
            x0[2 + i] = np.clip(concentrations_init.get(el, 1.0 / n_elements), *_CONC_BOUNDS)

        # --- Bounds -------------------------------------------------------
        bounds = [_T_BOUNDS_K, _LOG_NE_BOUNDS] + [_CONC_BOUNDS] * n_elements

        # --- Objective function -------------------------------------------
        inv_sigma2 = 1.0 / sigma**2

        def objective(x: np.ndarray) -> float:
            T_K = x[0]
            ne_cm3 = 10.0 ** x[1]
            conc = x[2:]

            basis = self.basis_library.get_basis_matrix_interp(T_K, ne_cm3)
            # basis shape: (n_all_elements, n_pixels)
            # Select only the detected-element rows
            selected = np.array(basis[element_indices, :])  # (n_el, n_pix)

            model = conc @ selected  # (n_pix,)
            residual = obs - model
            chi2 = float(np.sum(residual**2 * inv_sigma2))
            return chi2

        # --- Optimise -----------------------------------------------------
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iterations, "ftol": 1e-12, "gtol": 1e-8},
        )

        # --- Unpack -------------------------------------------------------
        xopt = result.x
        T_opt = float(xopt[0])
        ne_opt = 10.0 ** float(xopt[1])
        conc_opt = {el: float(xopt[2 + i]) for i, el in enumerate(elements_used)}

        chi2 = float(result.fun)
        dof = max(n_pixels - n_params, 1)
        chi2_red = chi2 / dof

        # Residual norm
        basis = self.basis_library.get_basis_matrix_interp(T_opt, ne_opt)
        selected = np.array(basis[element_indices, :])
        model = np.array([conc_opt[el] for el in elements_used]) @ selected
        residual_norm = float(np.linalg.norm(obs - model))

        n_iter = result.nit if hasattr(result, "nit") else 0

        logger.info(
            "Refinement %s in %d iterations: T=%.0f K, ne=%.2e cm^-3, chi2_red=%.3f",
            "converged" if result.success else "did not converge",
            n_iter,
            T_opt,
            ne_opt,
            chi2_red,
        )

        return RefinementResult(
            T_K=T_opt,
            ne_cm3=ne_opt,
            concentrations=conc_opt,
            residual_norm=residual_norm,
            n_iterations=n_iter,
            converged=bool(result.success),
            chi_squared=chi2,
            chi_squared_reduced=chi2_red,
        )
