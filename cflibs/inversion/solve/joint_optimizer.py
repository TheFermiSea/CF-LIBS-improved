"""
Multi-element joint optimization for CF-LIBS analysis.

This module implements simultaneous optimization of temperature, electron density,
and all element concentrations. Unlike sequential single-element analysis, joint
optimization properly accounts for parameter correlations and ensures consistency
with the closure equation.

Key features:
- Simultaneous T, n_e, and concentration optimization
- Closure constraint satisfaction via softmax parameterization
- Efficient optimization for 10+ element systems using JAX
- Multiple loss functions (chi-squared, weighted least squares)
- Optional regularization for ill-posed problems

The optimization parameterization:
- Temperature: log(T_eV) for positivity and scale invariance
- Electron density: log10(n_e) for wide dynamic range
- Concentrations: softmax(theta) to enforce sum-to-one simplex constraint

References:
- Tognoni et al., "CF-LIBS: State of the art" (2010) - limitations of sequential analysis
- Ciucci et al., "New procedure for quantitative elemental analysis" (1999)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from cflibs.core.constants import EV_TO_K
from cflibs.core.logging_config import get_logger
from cflibs.inversion.result_base import ResultTableMixin, StatisticsMixin

logger = get_logger("inversion.joint_optimizer")

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    from jax.scipy.optimize import minimize as jax_minimize

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

    def jit(f):
        return f


class LossType(Enum):
    """Loss function types for joint optimization."""

    CHI_SQUARED = "chi_squared"  # Standard chi-squared (weighted residuals)
    LEAST_SQUARES = "least_squares"  # Unweighted sum of squares
    HUBER = "huber"  # Robust Huber loss


class ConvergenceStatus(Enum):
    """Convergence status for optimization."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    FAILED = "failed"
    NOT_STARTED = "not_started"


@dataclass
class JointOptimizationResult(ResultTableMixin, StatisticsMixin):
    """
    Result of multi-element joint optimization.

    This dataclass stores the optimized plasma parameters and concentrations
    from simultaneous optimization of all parameters.

    Attributes
    ----------
    temperature_eV : float
        Optimized temperature in eV
    electron_density_cm3 : float
        Optimized electron density in cm^-3
    concentrations : Dict[str, float]
        Optimized element concentrations (sum to 1)
    initial_temperature_eV : float
        Initial temperature guess
    initial_electron_density_cm3 : float
        Initial electron density guess
    initial_concentrations : Dict[str, float]
        Initial concentration guesses
    final_loss : float
        Final loss function value
    chi_squared : float
        Chi-squared statistic (if applicable)
    reduced_chi_squared : float
        Reduced chi-squared (chi^2 / dof)
    degrees_of_freedom : int
        Degrees of freedom (n_observations - n_parameters)
    convergence_status : ConvergenceStatus
        Optimization convergence status
    iterations : int
        Number of optimization iterations
    gradient_norm : float
        Final gradient norm (indicates convergence quality)
    hessian_condition : float
        Condition number of Hessian (indicates parameter identifiability)
    correlation_matrix : np.ndarray, optional
        Parameter correlation matrix from Hessian inverse
    parameter_uncertainties : Dict[str, float], optional
        Parameter uncertainties from Hessian diagonal
    method : str
        Optimization method used
    metadata : Dict
        Additional metadata
    """

    # Optimized parameters
    temperature_eV: float
    electron_density_cm3: float
    concentrations: Dict[str, float]

    # Initial guesses (for diagnostics)
    initial_temperature_eV: float
    initial_electron_density_cm3: float
    initial_concentrations: Dict[str, float]

    # Fit quality metrics
    final_loss: float
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    degrees_of_freedom: int = 0

    # Convergence information
    convergence_status: ConvergenceStatus = ConvergenceStatus.NOT_STARTED
    iterations: int = 0
    gradient_norm: float = float("inf")
    hessian_condition: float = float("inf")

    # Uncertainty estimates (from Hessian)
    correlation_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    parameter_uncertainties: Dict[str, float] = field(default_factory=dict)

    # Metadata
    method: str = "L-BFGS-B"
    loss_type: str = "chi_squared"
    metadata: Dict = field(default_factory=dict)

    @property
    def temperature_K(self) -> float:
        """Temperature in Kelvin."""
        return self.temperature_eV * EV_TO_K

    @property
    def log_ne(self) -> float:
        """Log10 of electron density."""
        return np.log10(self.electron_density_cm3)

    @property
    def is_converged(self) -> bool:
        """Check if optimization converged successfully."""
        return self.convergence_status == ConvergenceStatus.CONVERGED

    @property
    def goodness_of_fit(self) -> str:
        """Interpret reduced chi-squared."""
        if self.reduced_chi_squared < 0.5:
            return "overfitting (uncertainties too large or model too flexible)"
        elif self.reduced_chi_squared < 1.5:
            return "good fit"
        elif self.reduced_chi_squared < 3.0:
            return "acceptable fit"
        else:
            return "poor fit (model inadequate or uncertainties underestimated)"

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "Joint Optimization Result",
            "=" * 70,
            f"Method: {self.method} | Loss: {self.loss_type}",
            f"Status: {self.convergence_status.value} ({self.iterations} iterations)",
            "-" * 70,
            f"{'Parameter':<20} {'Optimized':>15} {'Initial':>15}",
            "-" * 70,
            f"{'T [eV]':<20} {self.temperature_eV:>15.4f} {self.initial_temperature_eV:>15.4f}",
            f"{'T [K]':<20} {self.temperature_K:>15.0f} {self.initial_temperature_eV * EV_TO_K:>15.0f}",
            f"{'log10(n_e)':<20} {self.log_ne:>15.4f} {np.log10(self.initial_electron_density_cm3):>15.4f}",
            f"{'n_e [cm^-3]':<20} {self.electron_density_cm3:>15.2e} {self.initial_electron_density_cm3:>15.2e}",
            "-" * 70,
            f"{'Element':<20} {'Optimized':>15} {'Initial':>15}",
            "-" * 70,
        ]

        for el in sorted(self.concentrations.keys()):
            c_opt = self.concentrations[el]
            c_init = self.initial_concentrations.get(el, 0.0)
            unc = self.parameter_uncertainties.get(f"C_{el}", 0.0)
            if unc > 0:
                lines.append(f"{el:<20} {c_opt:>11.4f} +/- {unc:.4f} {c_init:>15.4f}")
            else:
                lines.append(f"{el:<20} {c_opt:>15.4f} {c_init:>15.4f}")

        lines.extend(
            [
                "-" * 70,
                f"Sum of concentrations: {sum(self.concentrations.values()):.6f}",
                "-" * 70,
                "Fit Quality:",
                f"  Final loss: {self.final_loss:.4e}",
                f"  Chi-squared: {self.chi_squared:.2f}",
                f"  Reduced chi-squared: {self.reduced_chi_squared:.3f} ({self.goodness_of_fit})",
                f"  Degrees of freedom: {self.degrees_of_freedom}",
                f"  Gradient norm: {self.gradient_norm:.2e}",
                "=" * 70,
            ]
        )

        return "\n".join(lines)


class JointOptimizer:
    """
    Multi-element joint optimizer for CF-LIBS analysis.

    Simultaneously optimizes temperature, electron density, and all element
    concentrations using JAX autodiff and gradient-based optimization.

    The closure constraint (concentrations sum to 1) is enforced via softmax
    parameterization, ensuring the optimization remains on the probability simplex.

    Parameters
    ----------
    forward_model : callable
        Forward model function: (T_eV, n_e, concentrations, wavelength) -> spectrum
        Must be JAX-compatible (differentiable).
    elements : List[str]
        List of element symbols in the analysis
    wavelength : np.ndarray
        Wavelength grid for spectra
    loss_type : LossType or str
        Loss function type (default: chi_squared)
    regularization : float
        L2 regularization strength for concentrations (default: 0.0)
    max_iterations : int
        Maximum optimization iterations (default: 200)
    tolerance : float
        Convergence tolerance for loss function (default: 1e-8)
    gradient_tolerance : float
        Convergence tolerance for gradient norm (default: 1e-6)

    Example
    -------
    >>> optimizer = JointOptimizer(forward_model, ["Fe", "Cu", "Zn"], wavelength)
    >>> result = optimizer.optimize(
    ...     measured_spectrum,
    ...     uncertainties=sigma,
    ...     initial_T_eV=1.0,
    ...     initial_n_e=1e17,
    ... )
    >>> print(result.summary())
    """

    def __init__(
        self,
        forward_model: Callable,
        elements: List[str],
        wavelength: np.ndarray,
        loss_type: LossType = LossType.CHI_SQUARED,
        regularization: float = 0.0,
        max_iterations: int = 200,
        tolerance: float = 1e-8,
        gradient_tolerance: float = 1e-6,
    ):
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for joint optimization. Install with: pip install jax jaxlib"
            )

        self.forward_model = forward_model
        self.elements = list(elements)
        self.n_elements = len(elements)
        self.wavelength = jnp.array(wavelength)
        self.n_wavelength = len(wavelength)

        # Convert string to enum if needed
        if isinstance(loss_type, str):
            loss_type = LossType(loss_type)
        self.loss_type = loss_type

        self.regularization = regularization
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.gradient_tolerance = gradient_tolerance

        # Parameter dimension: log(T) + log10(n_e) + n_elements (softmax params)
        self.n_params = 2 + self.n_elements

        logger.info(
            f"JointOptimizer initialized: {self.n_elements} elements, "
            f"{self.n_wavelength} wavelengths, loss={loss_type.value}"
        )

    def optimize(
        self,
        measured_spectrum: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        initial_T_eV: float = 1.0,
        initial_n_e: float = 1e17,
        initial_concentrations: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = "BFGS",
    ) -> JointOptimizationResult:
        """
        Perform joint optimization of all plasma parameters.

        Parameters
        ----------
        measured_spectrum : np.ndarray
            Measured spectrum (counts or intensity)
        uncertainties : np.ndarray, optional
            Spectral uncertainties; if None, uses sqrt(max(spectrum, 1))
        initial_T_eV : float
            Initial temperature guess in eV (default: 1.0)
        initial_n_e : float
            Initial electron density guess in cm^-3 (default: 1e17)
        initial_concentrations : Dict[str, float], optional
            Initial concentration guesses; if None, uses uniform
        bounds : Dict[str, Tuple[float, float]], optional
            Parameter bounds (T_eV, n_e range)
        method : str
            Optimization method: 'BFGS', 'L-BFGS-B', 'CG' (default: 'BFGS')

        Returns
        -------
        JointOptimizationResult
            Optimization results with uncertainties
        """
        # Validate inputs
        measured = jnp.array(measured_spectrum)
        if len(measured) != self.n_wavelength:
            raise ValueError(
                f"Spectrum length {len(measured)} does not match "
                f"wavelength grid {self.n_wavelength}"
            )

        # Set default uncertainties
        if uncertainties is None:
            uncertainties = jnp.sqrt(jnp.maximum(measured, 1.0))
        else:
            uncertainties = jnp.array(uncertainties)
            uncertainties = jnp.maximum(uncertainties, 1e-10)  # Avoid division by zero

        # Set default bounds
        if bounds is None:
            bounds = {
                "T_eV": (0.3, 5.0),  # Typical LIBS temperature range
                "log_ne": (14.0, 20.0),  # 10^14 to 10^20 cm^-3
            }

        # Set default initial concentrations
        if initial_concentrations is None:
            initial_concentrations = {el: 1.0 / self.n_elements for el in self.elements}

        # Ensure all elements have initial concentrations
        for el in self.elements:
            if el not in initial_concentrations:
                initial_concentrations[el] = 0.01

        # Normalize initial concentrations to sum to 1
        total = sum(initial_concentrations.values())
        if total > 0:
            initial_concentrations = {el: c / total for el, c in initial_concentrations.items()}

        # Pack initial parameters
        x0 = self._pack_params(initial_T_eV, initial_n_e, initial_concentrations)

        # Create loss function
        loss_fn = self._create_loss_function(measured, uncertainties)

        # Run optimization
        logger.info(
            f"Starting optimization: T0={initial_T_eV:.3f} eV, "
            f"n_e0={initial_n_e:.2e} cm^-3, method={method}"
        )

        try:
            # Use JAX minimize
            result = jax_minimize(
                loss_fn,
                x0,
                method=method.lower().replace("-", ""),
                options={"maxiter": self.max_iterations},
            )

            final_x = result.x
            final_loss = float(result.fun)
            converged = result.success
            iterations = result.nit if hasattr(result, "nit") else self.max_iterations

            # Compute gradient at solution
            grad_fn = jax.grad(loss_fn)
            final_grad = grad_fn(final_x)
            gradient_norm = float(jnp.linalg.norm(final_grad))

            # Determine convergence status
            if converged and gradient_norm < self.gradient_tolerance:
                status = ConvergenceStatus.CONVERGED
            elif iterations >= self.max_iterations:
                status = ConvergenceStatus.MAX_ITERATIONS
            else:
                status = ConvergenceStatus.FAILED

        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            final_x = x0
            final_loss = float(loss_fn(x0))
            converged = False
            iterations = 0
            gradient_norm = float("inf")
            status = ConvergenceStatus.FAILED

        # Unpack final parameters
        final_T, final_ne, final_conc_arr = self._unpack_params(final_x)
        final_concentrations = {el: float(final_conc_arr[i]) for i, el in enumerate(self.elements)}

        # Compute fit statistics
        n_obs = self.n_wavelength
        n_params = self.n_params
        dof = max(n_obs - n_params, 1)

        # Chi-squared from loss
        chi_squared = final_loss * n_obs if self.loss_type == LossType.CHI_SQUARED else 0.0
        reduced_chi_squared = chi_squared / dof if dof > 0 else 0.0

        # Estimate parameter uncertainties from Hessian
        param_uncertainties = {}
        correlation_matrix = None
        hessian_condition = float("inf")

        try:
            hessian_fn = jax.hessian(loss_fn)
            hessian = np.array(hessian_fn(final_x))

            # Condition number
            hessian_condition = float(np.linalg.cond(hessian))

            if hessian_condition < 1e12:  # Reasonably conditioned
                # Covariance matrix from inverse Hessian
                # Scale by chi^2 / dof for proper uncertainties
                cov = np.linalg.inv(hessian) * (
                    reduced_chi_squared if reduced_chi_squared > 0 else 1.0
                )

                # Standard errors from diagonal
                std_errors = np.sqrt(np.abs(np.diag(cov)))

                # Extract uncertainties
                param_uncertainties["T_eV"] = float(final_T * std_errors[0])  # log(T) -> T
                param_uncertainties["log_ne"] = float(std_errors[1])

                # Concentration uncertainties (need Jacobian of softmax)
                for i, el in enumerate(self.elements):
                    # Approximate uncertainty
                    param_uncertainties[f"C_{el}"] = float(
                        final_conc_arr[i] * (1 - final_conc_arr[i]) * std_errors[2 + i]
                    )

                # Correlation matrix
                std_diag = np.diag(1.0 / (std_errors + 1e-10))
                correlation_matrix = std_diag @ cov @ std_diag

        except Exception as e:
            logger.debug(f"Hessian computation failed: {e}")

        logger.info(
            f"Optimization complete: T={final_T:.3f} eV, n_e={final_ne:.2e} cm^-3, "
            f"chi^2_red={reduced_chi_squared:.3f}, status={status.value}"
        )

        return JointOptimizationResult(
            temperature_eV=float(final_T),
            electron_density_cm3=float(final_ne),
            concentrations=final_concentrations,
            initial_temperature_eV=initial_T_eV,
            initial_electron_density_cm3=initial_n_e,
            initial_concentrations=initial_concentrations,
            final_loss=final_loss,
            chi_squared=chi_squared,
            reduced_chi_squared=reduced_chi_squared,
            degrees_of_freedom=dof,
            convergence_status=status,
            iterations=iterations,
            gradient_norm=gradient_norm,
            hessian_condition=hessian_condition,
            correlation_matrix=correlation_matrix,
            parameter_uncertainties=param_uncertainties,
            method=method,
            loss_type=self.loss_type.value,
        )

    def _pack_params(
        self, T_eV: float, n_e: float, concentrations: Dict[str, float]
    ) -> jnp.ndarray:
        """
        Pack parameters into optimization vector.

        Parameterization:
        - x[0] = log(T_eV) for positivity
        - x[1] = log10(n_e) for wide dynamic range
        - x[2:] = softmax logits for concentrations (sum to 1)
        """
        log_T = jnp.log(max(T_eV, 0.1))
        log_ne = jnp.log10(max(n_e, 1e10))

        # Convert concentrations to logits
        # softmax(theta) = exp(theta) / sum(exp(theta))
        # To invert: theta = log(c) + constant (shift doesn't affect softmax)
        conc_arr = jnp.array([concentrations.get(el, 1e-6) for el in self.elements])
        conc_arr = jnp.maximum(conc_arr, 1e-10)  # Avoid log(0)
        theta = jnp.log(conc_arr)

        return jnp.concatenate([jnp.array([log_T, log_ne]), theta])

    def _unpack_params(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Unpack optimization vector to physical parameters.

        Returns
        -------
        T_eV : float
            Temperature in eV
        n_e : float
            Electron density in cm^-3
        concentrations : array
            Element concentrations (sum to 1 via softmax)
        """
        log_T = x[0]
        log_ne = x[1]
        theta = x[2:]

        T_eV = jnp.exp(log_T)
        n_e = jnp.power(10.0, log_ne)

        # Softmax ensures concentrations sum to 1
        concentrations = jax.nn.softmax(theta)

        return T_eV, n_e, concentrations

    def _create_loss_function(
        self,
        measured: jnp.ndarray,
        uncertainties: jnp.ndarray,
    ) -> Callable[[jnp.ndarray], float]:
        """
        Create the loss function for optimization.

        The loss function depends on loss_type:
        - CHI_SQUARED: sum((y - f(x))^2 / sigma^2) / n
        - LEAST_SQUARES: sum((y - f(x))^2)
        - HUBER: Huber loss for robustness to outliers
        """

        def _apply_reg(loss, conc):
            if self.regularization > 0:
                entropy = -jnp.sum(conc * jnp.log(conc + 1e-10))
                loss = loss - self.regularization * entropy
            return loss

        @jit
        def chi_squared_loss(x: jnp.ndarray) -> float:
            T_eV, n_e, conc = self._unpack_params(x)
            predicted = self.forward_model(T_eV, n_e, conc, self.wavelength)

            residuals = (measured - predicted) / uncertainties
            loss = jnp.mean(residuals**2)
            loss = _apply_reg(loss, conc)
            return loss

        @jit
        def least_squares_loss(x: jnp.ndarray) -> float:
            T_eV, n_e, conc = self._unpack_params(x)
            predicted = self.forward_model(T_eV, n_e, conc, self.wavelength)

            residuals = measured - predicted
            loss = jnp.sum(residuals**2)
            loss = _apply_reg(loss, conc)
            return loss

        @jit
        def huber_loss(x: jnp.ndarray) -> float:
            T_eV, n_e, conc = self._unpack_params(x)
            predicted = self.forward_model(T_eV, n_e, conc, self.wavelength)

            residuals = (measured - predicted) / uncertainties
            delta = 1.0  # Huber threshold

            # Huber loss: quadratic for |r| < delta, linear for |r| >= delta
            abs_r = jnp.abs(residuals)
            quadratic = 0.5 * residuals**2
            linear = delta * abs_r - 0.5 * delta**2
            loss = jnp.mean(jnp.where(abs_r <= delta, quadratic, linear))
            loss = _apply_reg(loss, conc)
            return loss

        if self.loss_type == LossType.CHI_SQUARED:
            return chi_squared_loss
        elif self.loss_type == LossType.LEAST_SQUARES:
            return least_squares_loss
        elif self.loss_type == LossType.HUBER:
            return huber_loss

    def profile_likelihood(
        self,
        result: JointOptimizationResult,
        parameter: str,
        measured: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        n_points: int = 50,
        sigma_range: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute profile likelihood for a parameter.

        Profile likelihood fixes one parameter and re-optimizes the others,
        giving a more accurate uncertainty estimate for correlated parameters.

        Parameters
        ----------
        result : JointOptimizationResult
            Previous optimization result
        parameter : str
            Parameter name: 'T_eV', 'log_ne', or 'C_<element>'
        measured : np.ndarray
            Measured spectrum
        uncertainties : np.ndarray, optional
            Spectral uncertainties
        n_points : int
            Number of profile points
        sigma_range : float
            Range in standard deviations around optimum

        Returns
        -------
        param_values : np.ndarray
            Parameter values
        profile_loss : np.ndarray
            Profile likelihood values (delta chi-squared)
        """
        measured = jnp.array(measured)
        if uncertainties is None:
            uncertainties = jnp.sqrt(jnp.maximum(measured, 1.0))
        else:
            uncertainties = jnp.array(uncertainties)

        # Get optimum and uncertainty
        if parameter == "T_eV":
            opt_val = result.temperature_eV
            unc = result.parameter_uncertainties.get("T_eV", opt_val * 0.1)
        elif parameter == "log_ne":
            opt_val = result.log_ne
            unc = result.parameter_uncertainties.get("log_ne", 0.5)
        elif parameter.startswith("C_"):
            el = parameter[2:]
            opt_val = result.concentrations.get(el, 0.5)
            unc = result.parameter_uncertainties.get(parameter, 0.05)
        else:
            raise ValueError(f"Unknown parameter: {parameter}")

        # Generate parameter range
        param_values = np.linspace(
            opt_val - sigma_range * max(unc, 0.01),
            opt_val + sigma_range * max(unc, 0.01),
            n_points,
        )

        # Compute profile likelihood
        loss_fn = self._create_loss_function(measured, uncertainties)
        opt_loss = result.final_loss

        # Get initial free parameters based on current optimum
        x_opt = self._pack_params(
            result.temperature_eV,
            result.electron_density_cm3,
            result.concentrations,
        )
        opt_log_T = x_opt[0]
        opt_log_ne = x_opt[1]
        opt_theta = x_opt[2:]

        if parameter == "T_eV":
            x_free_opt = jnp.concatenate([jnp.array([opt_log_ne]), opt_theta])
        elif parameter == "log_ne":
            x_free_opt = jnp.concatenate([jnp.array([opt_log_T]), opt_theta])
        elif parameter.startswith("C_"):
            el = parameter[2:]
            if el not in self.elements:
                raise ValueError(f"Unknown element: {el}")
            el_idx = self.elements.index(el)
            x_free_opt = jnp.concatenate(
                [jnp.array([opt_log_T, opt_log_ne]), jnp.delete(opt_theta, el_idx)]
            )

        profile_loss = []
        for pval in param_values:
            if parameter == "T_eV":

                def sub_loss(x_free):
                    # x_free: [log_ne, theta...]
                    # pval: T_eV
                    safe_T = jnp.maximum(jnp.array(pval), 0.1)
                    x_full = jnp.concatenate([jnp.array([jnp.log(safe_T)]), x_free])
                    return loss_fn(x_full)
            elif parameter == "log_ne":

                def sub_loss(x_free):
                    # x_free: [log_T, theta...]
                    # pval: log_ne
                    x_full = jnp.concatenate([jnp.array([x_free[0], pval]), x_free[1:]])
                    return loss_fn(x_full)
            elif parameter.startswith("C_"):
                el = parameter[2:]
                el_idx = self.elements.index(el)

                def sub_loss(x_free):
                    # x_free: [log_T, log_ne, theta_free...]
                    # pval: C_el
                    log_T, log_ne = x_free[0], x_free[1]
                    theta_free = x_free[2:]

                    # Compute theta_fixed such that softmax(theta)[el_idx] == pval
                    safe_pval = jnp.clip(pval, 1e-10, 1.0 - 1e-10)
                    sum_exp_free = jnp.sum(jnp.exp(theta_free))
                    theta_fixed = (
                        jnp.log(safe_pval) - jnp.log(1.0 - safe_pval) + jnp.log(sum_exp_free)
                    )

                    theta_full = jnp.insert(theta_free, el_idx, theta_fixed)
                    x_full = jnp.concatenate([jnp.array([log_T, log_ne]), theta_full])
                    return loss_fn(x_full)
            else:
                raise ValueError(f"Unknown parameter: {parameter}")

            # Re-optimize other parameters starting from the overall optimum
            res = jax_minimize(sub_loss, x_free_opt, method="bfgs", options={"maxiter": 50})

            loss = float(res.fun)
            # You could do x_free_opt = res.x here for warm start, but starting from the overall
            # optimum is more robust against local minima for small deviations.

            profile_loss.append((loss - opt_loss) * self.n_wavelength)  # Delta chi^2

        return param_values, np.array(profile_loss)


class MultiStartJointOptimizer:
    """
    Multi-start wrapper for JointOptimizer to avoid local minima.

    Runs multiple optimizations from different starting points and returns
    the best result. This is important for CF-LIBS where the loss landscape
    may have multiple local minima.

    Parameters
    ----------
    optimizer : JointOptimizer
        Base optimizer instance
    n_starts : int
        Number of random starting points (default: 5)
    seed : int
        Random seed for reproducibility

    Example
    -------
    >>> base_optimizer = JointOptimizer(forward_model, elements, wavelength)
    >>> multi_optimizer = MultiStartJointOptimizer(base_optimizer, n_starts=10)
    >>> result = multi_optimizer.optimize(spectrum, uncertainties=sigma)
    """

    def __init__(
        self,
        optimizer: JointOptimizer,
        n_starts: int = 5,
        seed: int = 42,
    ):
        self.optimizer = optimizer
        self.n_starts = n_starts
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        measured_spectrum: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        T_eV_range: Tuple[float, float] = (0.5, 2.5),
        n_e_range: Tuple[float, float] = (1e15, 1e19),
        **kwargs,
    ) -> JointOptimizationResult:
        """
        Run multi-start optimization.

        Parameters
        ----------
        measured_spectrum : np.ndarray
            Measured spectrum
        uncertainties : np.ndarray, optional
            Spectral uncertainties
        T_eV_range : Tuple[float, float]
            Temperature range for random starts
        n_e_range : Tuple[float, float]
            Electron density range for random starts
        **kwargs
            Additional arguments passed to optimizer.optimize()

        Returns
        -------
        JointOptimizationResult
            Best result across all starts
        """
        best_result = None
        best_loss = float("inf")
        all_results = []

        for i in range(self.n_starts):
            # Generate random starting point
            T_init = self.rng.uniform(T_eV_range[0], T_eV_range[1])
            log_ne_init = self.rng.uniform(np.log10(n_e_range[0]), np.log10(n_e_range[1]))
            n_e_init = 10**log_ne_init

            # Random concentrations (Dirichlet-like)
            alpha = np.ones(self.optimizer.n_elements)
            conc_init = self.rng.dirichlet(alpha)
            conc_dict = {el: conc_init[j] for j, el in enumerate(self.optimizer.elements)}

            logger.debug(
                f"Multi-start {i + 1}/{self.n_starts}: T0={T_init:.3f} eV, n_e0={n_e_init:.2e}"
            )

            try:
                result = self.optimizer.optimize(
                    measured_spectrum,
                    uncertainties=uncertainties,
                    initial_T_eV=T_init,
                    initial_n_e=n_e_init,
                    initial_concentrations=conc_dict,
                    **kwargs,
                )

                all_results.append(result)

                if result.final_loss < best_loss:
                    best_loss = result.final_loss
                    best_result = result

            except Exception as e:
                logger.warning(f"Start {i + 1} failed: {e}")

        if best_result is None:
            raise RuntimeError("All optimization starts failed")

        # Store all results in metadata
        best_result.metadata["all_results"] = all_results
        best_result.metadata["n_starts"] = self.n_starts
        best_result.metadata["n_successful"] = len(all_results)

        logger.info(
            f"Multi-start complete: {len(all_results)}/{self.n_starts} successful, "
            f"best loss={best_loss:.4e}"
        )

        return best_result


def create_simple_forward_model(
    elements: List[str],
    line_centers: Dict[str, List[float]],
    line_strengths: Dict[str, List[float]],
) -> Callable:
    """
    Create a simple Gaussian emission forward model for testing.

    This is a simplified model for testing and demonstration. For production
    use, employ a full physics-based forward model.

    Parameters
    ----------
    elements : List[str]
        Element symbols
    line_centers : Dict[str, List[float]]
        Emission line centers [nm] by element
    line_strengths : Dict[str, List[float]]
        Relative line strengths by element

    Returns
    -------
    callable
        Forward model function (T_eV, n_e, conc, wavelength) -> spectrum
    """
    if not HAS_JAX:
        raise ImportError("JAX required for forward model")

    # Convert to JAX arrays
    all_centers = []
    all_strengths = []
    all_element_idx = []

    for i, el in enumerate(elements):
        centers = line_centers.get(el, [500.0])
        strengths = line_strengths.get(el, [1.0])
        for c, s in zip(centers, strengths):
            all_centers.append(c)
            all_strengths.append(s)
            all_element_idx.append(i)

    centers_arr = jnp.array(all_centers)
    strengths_arr = jnp.array(all_strengths)
    element_idx_arr = jnp.array(all_element_idx)

    @jit
    def forward_model(
        T_eV: float,
        n_e: float,
        concentrations: jnp.ndarray,
        wavelength: jnp.ndarray,
    ) -> jnp.ndarray:
        """Simple Gaussian emission model."""
        # Temperature-dependent Doppler width
        sigma = 0.1 * jnp.sqrt(T_eV)  # Simplified

        # Boltzmann intensity factor
        boltzmann = jnp.exp(-2.0 / T_eV)  # Simplified E_k ~ 2 eV

        spectrum = jnp.zeros_like(wavelength)

        for k in range(len(centers_arr)):
            center = centers_arr[k]
            strength = strengths_arr[k]
            el_idx = element_idx_arr[k]

            # Line intensity = concentration * strength * Boltzmann
            intensity = concentrations[el_idx] * strength * boltzmann * 1000.0

            # Gaussian profile
            profile = jnp.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
            profile = profile / (sigma * jnp.sqrt(2 * jnp.pi))

            spectrum = spectrum + intensity * profile

        return spectrum

    return forward_model
