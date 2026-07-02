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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from cflibs.core.logging_config import get_logger
from cflibs.inversion.common.strict import (
    NonIdentifiable,
    NonPhysicalResult,
    NotConverged,
    SolverFailure,
    resolve_strict,
)
from cflibs.manifold.basis_library import BasisLibrary

logger = get_logger("inversion.spectral_refiner")

# Default parameter bounds
_T_BOUNDS_K = (3000.0, 30000.0)
_LOG_NE_BOUNDS = (14.0, 19.0)
_CONC_BOUNDS = (0.0, 1.0)
# log10 bounds for the global amplitude (intensity scale) parameter.  The
# amplitude carries spectral intensity so the composition parameters can be
# normalized to the closure simplex (sum = 1) independently of intensity.
_LOG_AMP_BOUNDS = (-12.0, 12.0)

#: Closure constraint tolerance.  Returned concentrations satisfy
#: ``|sum(C_s) - 1| <= CLOSURE_TOLERANCE``.
CLOSURE_TOLERANCE = 1e-6


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
        Optimized element concentrations (keyed by element symbol).  These
        satisfy the CF-LIBS closure condition ``sum(C_s) = 1`` to within
        :data:`CLOSURE_TOLERANCE`; the overall spectral intensity is carried
        by the separate ``amplitude`` parameter so composition is never
        conflated with intensity scale.
    amplitude : float
        Fitted global intensity scale (the eliminated experimental factor).
        The forward model is ``amplitude * (C_s @ basis)`` with ``sum(C_s)=1``.
    closure_residual : float
        Closure diagnostic ``|sum(C_s) - 1|`` of the returned (normalized)
        concentrations.  Always ``<= CLOSURE_TOLERANCE`` for a non-empty fit.
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
    amplitude: float = 1.0
    closure_residual: float = 0.0


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
        strict: Optional[bool] = None,
    ):
        self.basis_library = basis_library
        self.max_iterations = max_iterations
        # Strict / no-fallback mode (default off -> byte-identical production
        # path). When on, the converged=True no-ops, the silent element drops,
        # the MAD noise fallback, the zero-sum->uniform substitution, and
        # non-convergence all raise typed failures instead of returning a
        # fabricated "successful" refinement.
        self.strict = resolve_strict(strict)

    def _prepare_observed_spectrum(
        self, wavelength: np.ndarray, observed: np.ndarray
    ) -> np.ndarray:
        lib_wl = self.basis_library.wavelength
        if len(wavelength) != len(lib_wl) or not np.allclose(wavelength, lib_wl):
            return np.interp(lib_wl, wavelength, observed)
        return np.asarray(observed, dtype=np.float64)

    def _estimate_noise(
        self, wavelength: np.ndarray, noise: Optional[np.ndarray], obs: np.ndarray, n_pixels: int
    ) -> np.ndarray:
        lib_wl = self.basis_library.wavelength
        if noise is not None:
            if len(noise) != len(wavelength):
                sigma = np.interp(lib_wl, wavelength, noise)
            elif len(noise) != n_pixels:
                sigma = np.interp(lib_wl, wavelength, noise)
            else:
                sigma = np.asarray(noise, dtype=np.float64)
        else:
            if self.strict:
                raise SolverFailure(
                    "SpectralRefiner: noise=None; refusing the MAD 5%-of-median "
                    "noise fallback in strict mode (chi^2 / chi^2_reduced would be "
                    "computed against a fabricated floor and reported as meaningful)"
                )
            # MAD-based estimate of baseline noise
            sigma = np.full(n_pixels, max(np.median(np.abs(obs)) * 0.05, 1e-30))

        return np.maximum(sigma, 1e-30)  # prevent division by zero

    def _filter_elements(self, detected_elements: List[str]) -> Tuple[List[int], List[str]]:
        lib_elements = self.basis_library.elements
        element_indices: List[int] = []
        elements_used: List[str] = []
        for el in detected_elements:
            if el in lib_elements:
                element_indices.append(lib_elements.index(el))
                elements_used.append(el)
        return element_indices, elements_used

    def _pack_initial_vector(
        self,
        T_init_K: float,
        ne_init_cm3: float,
        amplitude_init: float,
        concentrations_init: Dict[str, float],
        elements_used: List[str],
        n_elements: int,
        n_params: int,
    ) -> Tuple[np.ndarray, List[tuple]]:
        x0 = np.empty(n_params, dtype=np.float64)
        x0[0] = np.clip(T_init_K, *_T_BOUNDS_K)
        x0[1] = np.clip(np.log10(max(ne_init_cm3, 1.0)), *_LOG_NE_BOUNDS)
        x0[2] = np.clip(np.log10(max(amplitude_init, 1e-30)), *_LOG_AMP_BOUNDS)
        # Seed normalized initial concentrations (closure simplex).
        raw_init = np.array(
            [max(concentrations_init.get(el, 1.0 / n_elements), 0.0) for el in elements_used],
            dtype=np.float64,
        )
        total = float(np.sum(raw_init))
        if total > 0.0:
            raw_init = raw_init / total
        else:
            raw_init = np.full(n_elements, 1.0 / n_elements)
        for i in range(n_elements):
            x0[3 + i] = np.clip(raw_init[i], *_CONC_BOUNDS)

        bounds = [_T_BOUNDS_K, _LOG_NE_BOUNDS, _LOG_AMP_BOUNDS] + [_CONC_BOUNDS] * n_elements
        return x0, bounds

    def _estimate_initial_amplitude(
        self,
        obs: np.ndarray,
        element_indices: List[int],
        elements_used: List[str],
        concentrations_init: Dict[str, float],
        T_init_K: float,
        ne_init_cm3: float,
    ) -> float:
        """Least-squares estimate of the initial intensity amplitude.

        Given the normalized initial composition, the optimal scalar
        amplitude that fits the observed spectrum is the closed-form 1-D WLS
        solution ``a* = <obs, m> / <m, m>`` where ``m`` is the unit-amplitude
        model.  Falls back to ``1.0`` for a degenerate (zero-norm) model.
        """
        raw = np.array(
            [
                max(concentrations_init.get(el, 1.0 / len(elements_used)), 0.0)
                for el in elements_used
            ]
        )
        total = float(np.sum(raw))
        if total <= 0.0:
            raw = np.full(len(elements_used), 1.0 / len(elements_used))
            total = 1.0
        norm_conc = raw / total

        basis = self.basis_library.get_basis_matrix_interp(T_init_K, ne_init_cm3)
        selected = np.array(basis[element_indices, :])
        model = norm_conc @ selected
        denom = float(np.dot(model, model))
        if denom <= 0.0:
            return 1.0
        amp = float(np.dot(obs, model) / denom)
        return amp if amp > 0.0 else 1.0

    def _unpack_result(
        self,
        result: Any,
        obs: np.ndarray,
        elements_used: List[str],
        element_indices: List[int],
        n_pixels: int,
        n_params: int,
    ) -> RefinementResult:
        xopt = result.x
        T_opt = float(xopt[0])
        ne_opt = 10.0 ** float(xopt[1])
        amp_raw = 10.0 ** float(xopt[2])

        # Enforce closure: normalize the box-bounded concentration parameters
        # onto the simplex (sum = 1) and fold their total into the amplitude so
        # the forward model intensity is preserved.  Composition is thereby
        # decoupled from the global intensity scale.
        raw_conc = np.array([float(xopt[3 + i]) for i in range(len(elements_used))])
        conc_sum = float(np.sum(raw_conc))
        if conc_sum > 0.0:
            norm_conc = raw_conc / conc_sum
            amplitude = amp_raw * conc_sum
        else:
            if self.strict:
                raise NonPhysicalResult(
                    "SpectralRefiner: optimizer drove all concentrations to ~0 "
                    "(degenerate zero-total optimum); refusing to substitute a "
                    "uniform composition for a failed fit"
                )
            norm_conc = np.full(len(elements_used), 1.0 / len(elements_used))
            amplitude = amp_raw
        conc_opt = {el: float(norm_conc[i]) for i, el in enumerate(elements_used)}
        closure_residual = abs(float(np.sum(norm_conc)) - 1.0)

        chi2 = float(result.fun)
        dof = max(n_pixels - n_params, 1)
        chi2_red = chi2 / dof

        # Residual norm (model = amplitude * normalized-composition @ basis)
        basis = self.basis_library.get_basis_matrix_interp(T_opt, ne_opt)
        selected = np.array(basis[element_indices, :])
        model = amplitude * (norm_conc @ selected)
        residual_norm = float(np.linalg.norm(obs - model))

        n_iter = result.nit if hasattr(result, "nit") else 0

        logger.info(
            "Refinement %s in %d iterations: T=%.0f K, ne=%.2e cm^-3, "
            "amp=%.3e, chi2_red=%.3f, closure_residual=%.2e",
            "converged" if result.success else "did not converge",
            n_iter,
            T_opt,
            ne_opt,
            amplitude,
            chi2_red,
            closure_residual,
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
            amplitude=amplitude,
            closure_residual=closure_residual,
        )

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
        ne_init_cm3 : float
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
        if not detected_elements:
            if self.strict:
                raise NonIdentifiable(
                    "SpectralRefiner: no detected elements -> nothing to refine; "
                    "refusing to report converged=True for a no-op"
                )
            return RefinementResult(
                T_K=T_init_K,
                ne_cm3=ne_init_cm3,
                concentrations={},
                residual_norm=float(np.linalg.norm(observed)),
                n_iterations=0,
                converged=True,
                chi_squared=0.0,
                chi_squared_reduced=0.0,
                amplitude=1.0,
                closure_residual=0.0,
            )

        obs = self._prepare_observed_spectrum(wavelength, observed)
        n_pixels = len(obs)
        sigma = self._estimate_noise(wavelength, noise, obs, n_pixels)

        element_indices, elements_used = self._filter_elements(detected_elements)
        if self.strict:
            missing = [el for el in detected_elements if el not in self.basis_library.elements]
            if missing:
                # Covers both the total mismatch (all absent) and the partial
                # silent drop: refuse rather than quietly shrink the fit and
                # report converged=True on the un-refined initial composition.
                raise NonIdentifiable(
                    f"SpectralRefiner: detected elements {missing} absent from the basis "
                    f"library {list(self.basis_library.elements)}; refusing to silently "
                    f"drop them"
                )
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
                amplitude=1.0,
                closure_residual=0.0,
            )

        n_elements = len(elements_used)
        # Parameters: [T, log10(ne), log10(amplitude), C_0 ... C_{n-1}]
        n_params = 3 + n_elements

        amplitude_init = self._estimate_initial_amplitude(
            obs, element_indices, elements_used, concentrations_init, T_init_K, ne_init_cm3
        )

        x0, bounds = self._pack_initial_vector(
            T_init_K,
            ne_init_cm3,
            amplitude_init,
            concentrations_init,
            elements_used,
            n_elements,
            n_params,
        )

        inv_sigma2 = 1.0 / sigma**2

        def objective(x: np.ndarray) -> float:
            T_K = x[0]
            ne_cm3 = 10.0 ** x[1]
            amplitude = 10.0 ** x[2]
            conc = x[3:]

            # Enforce the closure constraint inside the objective: the
            # composition is the normalized simplex projection of the
            # box-bounded parameters, and the global intensity is carried by
            # the separate amplitude.  This prevents the optimizer from
            # trading composition against intensity scale.
            conc_total = np.sum(conc)
            if conc_total <= 0.0:
                return float(np.sum(obs**2 * inv_sigma2))
            norm_conc = conc / conc_total

            basis = self.basis_library.get_basis_matrix_interp(T_K, ne_cm3)
            selected = np.array(basis[element_indices, :])

            model = amplitude * (norm_conc @ selected)
            residual = obs - model
            return float(np.sum(residual**2 * inv_sigma2))

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iterations, "ftol": 1e-12, "gtol": 1e-8},
        )

        if self.strict and not bool(result.success):
            raise NotConverged(
                f"SpectralRefiner: L-BFGS-B did not converge "
                f"(success={result.success}, status={getattr(result, 'status', None)}, "
                f"nit={getattr(result, 'nit', None)}, message={getattr(result, 'message', '')!r}); "
                f"default maxiter={self.max_iterations} truncation is common"
            )

        return self._unpack_result(result, obs, elements_used, element_indices, n_pixels, n_params)
