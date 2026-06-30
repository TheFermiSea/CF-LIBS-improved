"""
Hybrid inversion combining manifold lookup with gradient descent.

This module implements a two-stage inversion strategy:
1. Coarse search: Manifold nearest-neighbor (cosine similarity) for initial guess
2. Fine tuning: JAX autodiff + L-BFGS optimization from coarse guess

This approach combines the global search capability of the manifold lookup
(avoiding local minima) with the precision of gradient-based optimization.

References:
- Tognoni et al., "CF-LIBS: State of the art" (2010)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from cflibs.core.constants import KB_EV, EV_TO_J, EV_TO_K, C_LIGHT
from cflibs.core.logging_config import get_logger
from cflibs.inversion.common.strict import (
    NonIdentifiable,
    NotConverged,
    OptimizerFailure,
    SolveDiagnostics,
    resolve_strict,
)
from cflibs.inversion.physics.closure_strategy import ClosureStrategy, SoftmaxClosure

logger = get_logger("inversion.hybrid")

try:
    from scipy.optimize import minimize as scipy_minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_minimize = None

try:
    import jax
    import jax.numpy as jnp

    # jit, grad, value_and_grad available via jax module
    from jax.scipy.optimize import minimize as jax_minimize

    from cflibs.inversion.physics.softmax_closure import softmax_closure

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None
    softmax_closure = None  # type: ignore[assignment]


def _normalize_optimizer_method(method: str) -> str:
    normalized = method.upper()
    aliases = {
        "LBFGSB": "L-BFGS-B",
        "L_BFGS_B": "L-BFGS-B",
    }
    return aliases.get(normalized.replace("-", "_"), normalized)


def _hybrid_packed_bounds(bounds: Optional[Dict[str, Tuple[float, float]]], n_elements: int):
    if bounds is None:
        return None

    packed_bounds = []

    t_bounds = bounds.get("T_eV")
    if t_bounds is None:
        packed_bounds.append((None, None))
    else:
        packed_bounds.append((float(np.log(t_bounds[0])), float(np.log(t_bounds[1]))))

    ne_bounds = bounds.get("n_e")
    if ne_bounds is None:
        packed_bounds.append((None, None))
    else:
        packed_bounds.append((float(np.log(ne_bounds[0])), float(np.log(ne_bounds[1]))))

    packed_bounds.extend([(None, None)] * n_elements)
    return packed_bounds


def _optimizer_status(result, backend: str) -> Dict:
    """Capture the optimizer's *own* status/finiteness for strict diagnostics.

    The production path re-derives convergence from heuristics and never consults
    ``result.success``/``result.status``/``result.message``; this surfaces them so
    strict (no_fallback) mode can refuse on a real optimizer failure instead of
    laundering it into the warm seed.
    """
    try:
        x_finite = bool(np.all(np.isfinite(np.asarray(result.x, dtype=float))))
    except Exception:
        x_finite = False
    try:
        fun_finite = bool(np.isfinite(float(result.fun)))
    except Exception:
        fun_finite = False
    message = getattr(result, "message", None)
    return {
        "backend": backend,
        "success": bool(getattr(result, "success", False)),
        "status": getattr(result, "status", None),
        "message": str(message) if message is not None else None,
        "nit": int(getattr(result, "nit", 0) or 0),
        "x_finite": x_finite,
        "fun_finite": fun_finite,
    }


def _run_optimizer(
    loss_fn: Callable,
    x0: jnp.ndarray,
    method: str,
    max_iterations: int,
    bounds=None,
    *,
    return_status: bool = False,
) -> Tuple[jnp.ndarray, float, bool, int, str]:
    normalized_method = _normalize_optimizer_method(method)

    if normalized_method == "BFGS":
        result = jax_minimize(
            loss_fn,
            x0,
            method="BFGS",
            options={"maxiter": max_iterations},
        )
        iterations = int(getattr(result, "nit", max_iterations))
        out = (result.x, float(result.fun), bool(result.success), iterations, "jax")
        if return_status:
            return out + (_optimizer_status(result, "jax"),)
        return out

    if not HAS_SCIPY:
        raise ValueError(
            f"Optimizer '{normalized_method}' requires SciPy. Install scipy or use method='BFGS'."
        )

    value_and_grad = jax.value_and_grad(loss_fn)

    def scipy_objective(x_np: np.ndarray):
        x_jax = jnp.asarray(x_np)
        value, grad = value_and_grad(x_jax)
        return float(value), np.asarray(grad, dtype=np.float64)

    scipy_bounds = bounds if normalized_method == "L-BFGS-B" else None
    result = scipy_minimize(
        scipy_objective,
        np.asarray(x0, dtype=np.float64),
        method=normalized_method,
        jac=True,
        bounds=scipy_bounds,
        options={"maxiter": max_iterations},
    )

    iterations = int(getattr(result, "nit", 0) or 0)
    out = (
        jnp.asarray(result.x),
        float(result.fun),
        bool(result.success),
        iterations,
        "scipy",
    )
    if return_status:
        return out + (_optimizer_status(result, "scipy"),)
    return out


def _raise_optimizer_failure(
    exc: Exception,
    loss_fn: Callable,
    x0,
    method: str,
    *,
    stage: str,
    diagnostics: SolveDiagnostics,
) -> None:
    """Strict-mode replacement for ``except Exception: return warm seed``.

    Re-raises the swallowed optimizer crash as a typed :class:`OptimizerFailure`
    carrying the real provenance (seed loss, seed finiteness, exception type) so
    the failure is honest instead of laundered into "the fit equals the seed".
    """
    try:
        loss_at_seed = float(loss_fn(x0))
    except Exception:  # pragma: no cover - seed itself is degenerate
        loss_at_seed = float("nan")
    seed_finite = bool(np.all(np.isfinite(np.asarray(x0, dtype=float))))
    backend = "jax" if _normalize_optimizer_method(method) == "BFGS" else "scipy"
    diagnostics.optimizer_success = False
    diagnostics.objective_initial = loss_at_seed
    diagnostics.extra.update(
        {
            "stage": stage,
            "backend": backend,
            "method": method,
            "loss_at_seed": loss_at_seed,
            "seed_finite": seed_finite,
            "exc_type": type(exc).__name__,
            "exc_repr": repr(exc),
        }
    )
    diagnostics.failure_reason = f"optimizer raised {type(exc).__name__}: {exc}"
    raise OptimizerFailure(
        f"strict mode: {stage}-stage optimizer ({backend}) crashed "
        f"({type(exc).__name__}: {exc}); refusing to substitute the unrefined seed "
        f"(loss_at_seed={loss_at_seed:.6g}, seed_finite={seed_finite}).",
        diagnostics,
    ) from exc


def _check_optimizer_result(
    final_x,
    final_loss: float,
    converged: bool,
    opt_status: Optional[Dict],
    *,
    stage: str,
    diagnostics: SolveDiagnostics,
) -> None:
    """Strict-mode finite/convergence guard on the optimizer's own result.

    The production path returns ``result.x``/``result.fun`` verbatim (NaN included)
    and re-derives "converged" from heuristics. Here we refuse a non-finite result
    (``OptimizerFailure``) or a genuine non-success (``NotConverged``).
    """
    diagnostics.optimizer_status = opt_status
    diagnostics.optimizer_success = bool(converged)
    x_finite = bool(np.all(np.isfinite(np.asarray(final_x, dtype=float))))
    fun_finite = bool(np.isfinite(final_loss))
    diagnostics.extra.update({"stage": stage, "x_finite": x_finite, "fun_finite": fun_finite})
    if not (x_finite and fun_finite):
        diagnostics.failure_reason = (
            f"optimizer returned non-finite result (x_finite={x_finite}, "
            f"fun_finite={fun_finite})"
        )
        raise OptimizerFailure(
            f"strict mode: {stage}-stage optimizer returned a non-finite result "
            f"(x_finite={x_finite}, fun_finite={fun_finite}); a diverged run must not "
            f"be reported as a fit.",
            diagnostics,
        )
    if not converged:
        msg = opt_status.get("message") if isinstance(opt_status, dict) else None
        diagnostics.failure_reason = f"optimizer did not converge (success=False); {msg}"
        raise NotConverged(
            f"strict mode: {stage}-stage optimizer reported success=False "
            f"(message={msg!r}); refusing to present an unconverged fit.",
            diagnostics,
        )


@dataclass
class HybridInversionResult:
    """
    Result of hybrid inversion.

    Attributes
    ----------
    temperature_eV : float
        Recovered temperature in eV
    electron_density_cm3 : float
        Recovered electron density in cm^-3
    concentrations : Dict[str, float]
        Recovered elemental concentrations
    coarse_temperature_eV : float
        Initial temperature from manifold lookup
    coarse_electron_density_cm3 : float
        Initial electron density from manifold lookup
    coarse_concentrations : Dict[str, float]
        Initial concentrations from manifold lookup
    coarse_similarity : float
        Similarity score from manifold lookup
    final_residual : float
        Final chi-squared residual
    converged : bool
        Whether optimization converged
    iterations : int
        Number of optimization iterations
    method : str
        Optimization method used
    """

    temperature_eV: float
    electron_density_cm3: float
    concentrations: Dict[str, float]
    coarse_temperature_eV: float
    coarse_electron_density_cm3: float
    coarse_concentrations: Dict[str, float]
    coarse_similarity: float
    final_residual: float
    converged: bool
    iterations: int
    method: str = "L-BFGS-B"
    metadata: Dict = field(default_factory=dict)

    @property
    def temperature_K(self) -> float:
        """Temperature in Kelvin."""
        return self.temperature_eV * EV_TO_K

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Hybrid Inversion Result",
            f"  Coarse (manifold): T={self.coarse_temperature_eV:.3f} eV, "
            f"n_e={self.coarse_electron_density_cm3:.2e} cm^-3, "
            f"similarity={self.coarse_similarity:.4f}",
            f"  Fine (optimized):  T={self.temperature_eV:.3f} eV, "
            f"n_e={self.electron_density_cm3:.2e} cm^-3",
            "  Concentrations:",
        ]
        for el, c in self.concentrations.items():
            lines.append(f"    {el}: {c:.4f}")
        lines.append(
            f"  Converged: {self.converged} ({self.iterations} iterations, "
            f"residual={self.final_residual:.2e})"
        )
        return "\n".join(lines)


def _pack_params(T_eV, n_e, concentrations, elements):
    """Pack plasma parameters into an optimization vector (log-space)."""
    log_T = jnp.log(T_eV)
    log_ne = jnp.log(n_e)
    conc_arr = jnp.array([concentrations.get(el, 0.01) for el in elements])
    conc_arr = jnp.maximum(conc_arr, 1e-6)
    log_conc = jnp.log(conc_arr)
    return jnp.concatenate([jnp.array([log_T, log_ne]), log_conc])


def _unpack_params(x):
    """Unpack optimization vector to (T_eV, n_e, concentrations).

    Parameters
    ----------
    x : array-like
        Packed parameter vector: [log(T_eV), log(n_e), softmax_logits].
        Concentrations are enforced on the simplex via softmax_closure
        from cflibs.inversion.physics.softmax_closure.

    Returns
    -------
    Tuple[float, float, array-like]
        (T_eV, n_e, concentrations) where concentrations sum to 1.
    """
    T_eV = jnp.exp(x[0])
    n_e = jnp.exp(x[1])
    conc = softmax_closure(x[2:])
    return T_eV, n_e, conc


class HybridInverter:
    """
    Hybrid inversion combining manifold lookup with gradient descent.

    Algorithm:
    1. Use manifold cosine similarity to find initial guess (T0, ne0, C0)
    2. Define loss function: L = sum((measured - forward(T, ne, C))^2 / sigma^2)
    3. Use JAX autodiff + L-BFGS to minimize loss from initial guess

    This approach:
    - Avoids local minima (manifold provides global search)
    - Achieves high precision (gradient descent refines)
    - Is faster than pure optimization (good starting point)
    - Is more robust than pure manifold (not limited to grid points)

    Parameters
    ----------
    manifold : ManifoldLoader
        Pre-computed spectral manifold
    forward_model : callable, optional
        Forward model function: (T_eV, n_e, concentrations) -> spectrum
        If not provided, uses simple Gaussian model
    """

    def __init__(
        self,
        manifold,
        forward_model: Optional[Callable] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        closure: Optional[ClosureStrategy] = None,
        strict: Optional[bool] = None,
        allow_toy_forward_model: bool = False,
    ):
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for hybrid inversion. Install with: pip install jax jaxlib"
            )

        # Strict / no-fallback mode (default off -> production behaviour byte-identical).
        self.strict = resolve_strict(strict)

        self.manifold = manifold
        # In strict mode, refuse to silently bind the toy `_default_forward_model`
        # (5 fake Gaussian lines/element, invented upper energies, fixed 50 amu mass):
        # it fabricates the physics and lets the optimizer "converge" to meaningless
        # numbers. Tests can opt back in via allow_toy_forward_model=True.
        if self.strict and forward_model is None and not allow_toy_forward_model:
            raise ValueError(
                "HybridInverter requires an explicit physics forward_model; the built-in "
                "_default_forward_model is a demo placeholder and is disabled under strict "
                "(no_fallback) mode. Pass forward_model=... or allow_toy_forward_model=True."
            )
        self.forward_model = forward_model or self._default_forward_model
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Closure strategy — defaults to softmax for backward compatibility.
        if closure is None:
            closure = SoftmaxClosure()
        # HybridInverter traces the loss through jax.value_and_grad, so the closure
        # must be JAX-native. NumPy closures (ILR/PWLR) call np.asarray(params) which
        # either ConcretizationTypeErrors on a tracer or silently produces wrong
        # gradients on a concrete array. Reject early with a clear message.
        if closure.backend != "jax":
            raise ValueError(
                f"HybridInverter requires a JAX-backend closure for autodiff, "
                f"got {closure.name!r} (backend={closure.backend!r}). "
                f"Use SoftmaxClosure() or another JAX-compatible strategy."
            )
        self.closure: ClosureStrategy = closure

        # Extract manifold info
        self.wavelength = jnp.array(manifold.wavelength)
        self.elements = list(manifold.elements)
        self.n_elements = len(self.elements)

        logger.info(
            f"HybridInverter initialized: {self.n_elements} elements, "
            f"{len(self.wavelength)} wavelengths, closure={self.closure.name}"
        )

    def invert(
        self,
        measured_spectrum: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        method: str = "L-BFGS-B",
        use_manifold_init: bool = True,
        initial_guess: Optional[Dict] = None,
        bounds: Optional[Dict] = None,
        strict: Optional[bool] = None,
        min_coarse_similarity: Optional[float] = None,
    ) -> HybridInversionResult:
        """
        Perform hybrid inversion on measured spectrum.

        Parameters
        ----------
        measured_spectrum : array
            Measured spectrum (must match manifold wavelength grid)
        uncertainties : array, optional
            Spectral uncertainties (defaults to sqrt(spectrum))
        method : str
            Optimization method: 'L-BFGS-B', 'BFGS', 'CG'
        use_manifold_init : bool
            Use manifold lookup for initial guess
        initial_guess : dict, optional
            Override initial guess (T_eV, n_e, concentrations)
        bounds : dict, optional
            Parameter bounds
        strict : bool, optional
            Per-call override of the instance's no_fallback setting (``None`` =
            inherit). When strict: refuse a fabricated default seed, refuse an
            out-of-manifold seed below ``min_coarse_similarity``, and refuse a
            crashed / non-finite / non-converged optimizer instead of silently
            returning the coarse seed.
        min_coarse_similarity : float, optional
            Strict-only floor on the manifold cosine similarity; below it the
            measurement is treated as outside the manifold and refused.

        Returns
        -------
        HybridInversionResult
            Inversion results
        """
        eff_strict = self.strict if strict is None else bool(strict)
        diagnostics = SolveDiagnostics(solver="coarse_to_fine.HybridInverter", strict=eff_strict)

        measured = jnp.array(measured_spectrum)

        # Set default uncertainties
        if uncertainties is None:
            uncertainties = jnp.sqrt(jnp.maximum(measured, 1.0))
        else:
            uncertainties = jnp.array(uncertainties)

        # Set default bounds
        if bounds is None:
            bounds = {
                "T_eV": (0.3, 3.0),  # Typical LIBS range
                "n_e": (1e15, 1e19),
                "concentration": (0.0, 1.0),
            }

        # Stage 1: Coarse search via manifold
        if use_manifold_init and initial_guess is None:
            coarse_idx, coarse_similarity, coarse_params = self.manifold.find_nearest_spectrum(
                np.array(measured), method="cosine", use_jax=True
            )

            coarse_T = coarse_params["T_eV"]
            coarse_ne = coarse_params["n_e_cm3"]
            coarse_conc = {el: coarse_params.get(el, 0.0) for el in self.elements}
            init_source = "manifold"

            logger.info(
                f"Coarse search: T={coarse_T:.3f} eV, n_e={coarse_ne:.2e}, "
                f"similarity={coarse_similarity:.4f}"
            )

            # Strict: a near-orthogonal best match means the measurement is outside
            # the manifold's parameter coverage; the seed (and any optimum near it)
            # is untrustworthy. coarse_similarity is recorded but never gated in the
            # production path -- gate it here.
            if (
                eff_strict
                and min_coarse_similarity is not None
                and (
                    not np.isfinite(coarse_similarity) or coarse_similarity < min_coarse_similarity
                )
            ):
                diagnostics.extra.update(
                    {
                        "init_source": init_source,
                        "coarse_similarity": float(coarse_similarity),
                        "min_coarse_similarity": float(min_coarse_similarity),
                        "best_idx": int(coarse_idx),
                    }
                )
                diagnostics.failure_reason = (
                    f"manifold match similarity {coarse_similarity:.4g} < floor "
                    f"{min_coarse_similarity:.4g}; measurement likely outside manifold coverage"
                )
                raise NonIdentifiable(
                    f"strict mode: best manifold similarity {coarse_similarity:.4g} below "
                    f"min_coarse_similarity={min_coarse_similarity:.4g}; the coarse seed is "
                    f"out-of-manifold and not a trustworthy starting point.",
                    diagnostics,
                )
        elif initial_guess is not None:
            coarse_T = initial_guess.get("T_eV", 1.0)
            coarse_ne = initial_guess.get("n_e", 1e17)
            coarse_conc = {el: initial_guess.get(el, 1.0 / self.n_elements) for el in self.elements}
            coarse_similarity = 0.0
            init_source = "user"
        else:
            # Strict: refuse the fabricated zero-provenance seed (T=1.0 eV,
            # n_e=1e17, uniform C) -- there was no coarse search and no user guess,
            # so we have no idea where to start.
            if eff_strict:
                diagnostics.extra["init_source"] = "default"
                diagnostics.failure_reason = (
                    "no manifold init and no initial_guess -> fabricated default seed refused"
                )
                raise ValueError(
                    "strict mode: HybridInverter.invert requires a manifold init "
                    "(use_manifold_init=True) or an explicit initial_guess; refusing the "
                    "fabricated default seed (T=1.0 eV, n_e=1e17, uniform composition)."
                )
            # Default initial guess
            coarse_T = 1.0
            coarse_ne = 1e17
            coarse_conc = {el: 1.0 / self.n_elements for el in self.elements}
            coarse_similarity = 0.0
            init_source = "default"

        # Stage 2: Fine tuning via gradient descent
        # Pack parameters into array for optimization
        # [log(T), log(n_e), softmax(concentrations)]
        x0 = self._pack_params(coarse_T, coarse_ne, coarse_conc)
        packed_bounds = _hybrid_packed_bounds(bounds, self.n_elements)

        # Define loss function
        def loss_fn(x):
            T_eV, n_e, conc_arr = self._unpack_params(x)
            predicted = self.forward_model(T_eV, n_e, conc_arr, self.wavelength)
            residuals = (measured - predicted) / uncertainties
            return jnp.sum(residuals**2)

        # Run optimization
        opt_status = None
        try:
            opt_out = _run_optimizer(
                loss_fn,
                x0,
                method=method,
                max_iterations=self.max_iterations,
                bounds=packed_bounds,
                return_status=eff_strict,
            )
            if eff_strict:
                final_x, final_loss, converged, iterations, backend, opt_status = opt_out
            else:
                final_x, final_loss, converged, iterations, backend = opt_out
        except Exception as e:
            if eff_strict:
                # Do NOT launder the crash into "the fit equals the coarse seed".
                _raise_optimizer_failure(
                    e, loss_fn, x0, method, stage="fine", diagnostics=diagnostics
                )
            logger.warning(f"Optimization failed: {e}, using coarse result")
            final_x = x0
            final_loss = float(loss_fn(x0))
            converged = False
            iterations = 0
            backend = "fallback"

        # Strict: the production path never consults the optimizer's own success/
        # finiteness; a diverged BFGS run silently yields NaN T/n_e. Refuse here.
        if eff_strict:
            _check_optimizer_result(
                final_x,
                final_loss,
                converged,
                opt_status,
                stage="fine",
                diagnostics=diagnostics,
            )

        # Unpack final parameters
        final_T, final_ne, final_conc_arr = self._unpack_params(final_x)
        final_concentrations = {el: float(final_conc_arr[i]) for i, el in enumerate(self.elements)}

        logger.info(
            f"Fine tuning: T={final_T:.3f} eV, n_e={final_ne:.2e}, "
            f"residual={final_loss:.2e}, converged={converged}"
        )

        metadata: Dict = {"optimizer_backend": backend}
        if eff_strict:
            diagnostics.converged = bool(converged)
            diagnostics.objective_final = float(final_loss)
            metadata["init_source"] = init_source
            metadata["coarse_similarity"] = float(coarse_similarity)
            metadata["diagnostics"] = diagnostics.to_dict()

        return HybridInversionResult(
            temperature_eV=float(final_T),
            electron_density_cm3=float(final_ne),
            concentrations=final_concentrations,
            coarse_temperature_eV=coarse_T,
            coarse_electron_density_cm3=coarse_ne,
            coarse_concentrations=coarse_conc,
            coarse_similarity=coarse_similarity,
            final_residual=final_loss,
            converged=converged,
            iterations=iterations,
            method=method,
            metadata=metadata,
        )

    def _pack_params(
        self, T_eV: float, n_e: float, concentrations: Dict[str, float]
    ) -> jnp.ndarray:
        """Pack parameters into optimization vector."""
        return _pack_params(T_eV, n_e, concentrations, self.elements)

    def _unpack_params(self, x: jnp.ndarray) -> Tuple[float, float, jnp.ndarray]:
        """Unpack optimization vector to parameters using the configured closure."""
        T_eV = jnp.exp(x[0])
        n_e = jnp.exp(x[1])
        conc = self.closure.apply(x[2:])
        return T_eV, n_e, conc

    def _default_forward_model(
        self,
        T_eV: float,
        n_e: float,
        concentrations: jnp.ndarray,
        wavelength: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Simple forward model for testing.

        This is a simplified Gaussian emission model. For production use,
        provide a full physics-based forward model.

        Parameters
        ----------
        T_eV : float
            Temperature in eV
        n_e : float
            Electron density
        concentrations : array
            Element concentrations
        wavelength : array
            Wavelength grid

        Returns
        -------
        array
            Predicted spectrum
        """
        # This is a placeholder - in production, use the ManifoldGenerator's
        # _compute_spectrum_snapshot or a similar physics-based model

        # Generate synthetic lines for demonstration
        n_wl = len(wavelength)
        wl_min = float(wavelength[0])
        wl_max = float(wavelength[-1])

        spectrum = jnp.zeros(n_wl)

        # Add Gaussian emission lines for each element
        for i, el in enumerate(self.elements):
            c = concentrations[i]

            # Generate some synthetic lines per element
            n_lines = 5
            for j in range(n_lines):
                # Line position depends on element index
                center = wl_min + (wl_max - wl_min) * (i * n_lines + j + 0.5) / (
                    len(self.elements) * n_lines
                )

                # Intensity from Boltzmann distribution
                E_k = 2.0 + j * 0.5  # Fake upper energy
                boltzmann = jnp.exp(-E_k / (KB_EV * T_eV * EV_TO_K))
                intensity = c * boltzmann * 1000.0

                # Doppler width: 1-D Maxwell std sigma = lambda/c * sqrt(kT/m).
                # NB: this is a PLACEHOLDER/demo forward model, not the
                # production path (see note above) — but kept consistent with
                # the canonical profiles.doppler_sigma_jax form. The previous
                # spurious factor of 2 under the sqrt computed the
                # most-probable speed, not the standard deviation (~1.41x wide).
                sigma = center * jnp.sqrt(T_eV * EV_TO_J / (50.0 * 1.67e-27 * C_LIGHT**2))
                sigma = jnp.maximum(sigma, 0.01)

                # Add Gaussian
                profile = jnp.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
                spectrum = spectrum + intensity * profile / (sigma * jnp.sqrt(2 * jnp.pi))

        return spectrum


class SpectralFitter:
    """
    Gradient-based spectral fitting using JAX autodiff.

    This class provides fine-grained control over spectral fitting without
    requiring a pre-computed manifold. Use when you have a forward model
    and want to fit plasma parameters directly.

    Parameters
    ----------
    forward_model : callable
        Forward model: (T_eV, n_e, concentrations, wavelength) -> spectrum
    elements : List[str]
        Element names
    wavelength : array
        Wavelength grid
    """

    def __init__(
        self,
        forward_model: Callable,
        elements: List[str],
        wavelength: np.ndarray,
        closure: Optional[ClosureStrategy] = None,
        strict: Optional[bool] = None,
    ):
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for spectral fitting. Install with: pip install jax jaxlib"
            )

        # Strict / no-fallback mode (default off -> production behaviour byte-identical).
        self.strict = resolve_strict(strict)
        self.forward_model = forward_model
        self.elements = elements
        self.wavelength = jnp.array(wavelength)
        self.n_elements = len(elements)
        if closure is None:
            closure = SoftmaxClosure()
        # SpectralFitter traces the loss through jax.value_and_grad — see the
        # matching check in HybridInverter for full rationale.
        if closure.backend != "jax":
            raise ValueError(
                f"SpectralFitter requires a JAX-backend closure for autodiff, "
                f"got {closure.name!r} (backend={closure.backend!r}). "
                f"Use SoftmaxClosure() or another JAX-compatible strategy."
            )
        self.closure: ClosureStrategy = closure

    def fit(
        self,
        measured_spectrum: np.ndarray,
        initial_T_eV: float = 1.0,
        initial_n_e: float = 1e17,
        initial_concentrations: Optional[Dict[str, float]] = None,
        uncertainties: Optional[np.ndarray] = None,
        method: str = "BFGS",
        max_iterations: int = 100,
        strict: Optional[bool] = None,
    ) -> HybridInversionResult:
        """
        Fit spectrum to data using gradient descent.

        Parameters
        ----------
        measured_spectrum : array
            Measured spectrum
        initial_T_eV : float
            Initial temperature guess
        initial_n_e : float
            Initial electron density guess
        initial_concentrations : dict, optional
            Initial concentration guess per element
        uncertainties : array, optional
            Spectral uncertainties
        method : str
            Optimization method
        max_iterations : int
            Maximum iterations
        strict : bool, optional
            Per-call override of the instance's no_fallback setting (``None`` =
            inherit). When strict: refuse a crashed / non-finite / non-converged
            optimizer instead of silently returning the initial guess as the "fit".

        Returns
        -------
        HybridInversionResult
            Fitting results
        """
        eff_strict = self.strict if strict is None else bool(strict)
        diagnostics = SolveDiagnostics(solver="coarse_to_fine.SpectralFitter", strict=eff_strict)

        measured = jnp.array(measured_spectrum)

        if uncertainties is None:
            uncertainties = jnp.sqrt(jnp.maximum(measured, 1.0))
        else:
            uncertainties = jnp.array(uncertainties)

        if initial_concentrations is None:
            initial_concentrations = {el: 1.0 / self.n_elements for el in self.elements}

        # Pack initial guess
        x0 = self._pack(initial_T_eV, initial_n_e, initial_concentrations)

        def loss_fn(x):
            T, ne, conc = self._unpack(x)
            predicted = self.forward_model(T, ne, conc, self.wavelength)
            residuals = (measured - predicted) / uncertainties
            return jnp.sum(residuals**2)

        opt_status = None
        try:
            opt_out = _run_optimizer(
                loss_fn,
                x0,
                method=method,
                max_iterations=max_iterations,
                return_status=eff_strict,
            )
            if eff_strict:
                final_x, final_loss, converged, iterations, backend, opt_status = opt_out
            else:
                final_x, final_loss, converged, iterations, backend = opt_out
            final_T, final_ne, final_conc = self._unpack(final_x)
        except Exception as e:
            if eff_strict:
                # Do NOT launder the crash into "the fit equals your initial guess".
                _raise_optimizer_failure(
                    e, loss_fn, x0, method, stage="fit", diagnostics=diagnostics
                )
            logger.warning(f"Fitting failed: {e}")
            final_T = initial_T_eV
            final_ne = initial_n_e
            final_conc = jnp.array(list(initial_concentrations.values()))
            converged = False
            iterations = 0
            final_loss = float(loss_fn(x0))
            backend = "fallback"

        if eff_strict:
            _check_optimizer_result(
                final_x,
                final_loss,
                converged,
                opt_status,
                stage="fit",
                diagnostics=diagnostics,
            )

        metadata: Dict = {"optimizer_backend": backend}
        if eff_strict:
            diagnostics.converged = bool(converged)
            diagnostics.objective_final = float(final_loss)
            metadata["init_source"] = "user"
            metadata["diagnostics"] = diagnostics.to_dict()

        return HybridInversionResult(
            temperature_eV=float(final_T),
            electron_density_cm3=float(final_ne),
            concentrations={el: float(final_conc[i]) for i, el in enumerate(self.elements)},
            coarse_temperature_eV=initial_T_eV,
            coarse_electron_density_cm3=initial_n_e,
            coarse_concentrations=initial_concentrations,
            coarse_similarity=0.0,
            final_residual=final_loss,
            converged=converged,
            iterations=iterations,
            method=method,
            metadata=metadata,
        )

    def _pack(self, T_eV: float, n_e: float, concentrations: Dict[str, float]) -> jnp.ndarray:
        """Pack parameters."""
        return _pack_params(T_eV, n_e, concentrations, self.elements)

    def _unpack(self, x: jnp.ndarray) -> Tuple[float, float, jnp.ndarray]:
        """Unpack parameters using the configured closure strategy."""
        T_eV = jnp.exp(x[0])
        n_e = jnp.exp(x[1])
        conc = self.closure.apply(x[2:])
        return T_eV, n_e, conc
