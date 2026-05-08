"""
Iterative solver for Classic CF-LIBS.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

from cflibs.core.constants import KB, KB_EV, SAHA_CONST_CM3, STP_PRESSURE, EV_TO_K
from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.boltzmann import LineObservation, BoltzmannPlotFitter
from cflibs.inversion.closure import ClosureEquation
from cflibs.plasma.partition import PartitionFunctionEvaluator
from cflibs.core.logging_config import get_logger

# Optional JAX imports — IterativeCFLIBSSolverJax raises ImportError at
# instantiation time if JAX is missing, so the rest of the module is unaffected.
try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised when JAX absent
    HAS_JAX = False
    jax = None  # type: ignore[assignment]
    jnp = None
    jit = None  # type: ignore[assignment]

logger = get_logger("inversion.solver")


@dataclass
class CFLIBSResult:
    """
    Result of the iterative CF-LIBS inversion.

    Attributes
    ----------
    temperature_K : float
        Plasma temperature in Kelvin
    temperature_uncertainty_K : float
        1-sigma uncertainty in temperature
    electron_density_cm3 : float
        Electron density in cm^-3
    electron_density_uncertainty_cm3 : float
        1-sigma uncertainty in electron density (0 if not computed)
    concentrations : Dict[str, float]
        Element concentrations (number/mole fractions, sum to 1).
        These are the internal CF-LIBS closure fractions used in
        Saha-Boltzmann algebra. Convert to mass fractions via
        C_mass_i = C_i * AW_i / sum(C_j * AW_j).
    concentration_uncertainties : Dict[str, float]
        1-sigma uncertainties in concentrations
    iterations : int
        Number of iterations performed
    converged : bool
        Whether solver converged within tolerance
    temperature_corona_K : float, optional
        Estimated corona temperature in Kelvin (for two-region fits)
    quality_metrics : Dict[str, float]
        Quality metrics (R², chi², etc.)
    boltzmann_covariance : np.ndarray, optional
        2x2 covariance matrix of a representative pooled Boltzmann fit
        (slope, intercept). For multi-element uncertainty solves this stores
        the covariance for the selected reference element noted in
        ``quality_metrics["boltzmann_covariance_element"]``.
    """

    temperature_K: float
    temperature_uncertainty_K: float
    electron_density_cm3: float
    concentrations: Dict[str, float]
    concentration_uncertainties: Dict[str, float]
    iterations: int
    converged: bool
    temperature_corona_K: Optional[float] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    electron_density_uncertainty_cm3: float = 0.0
    boltzmann_covariance: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class _CommonSlopeElementStats:
    """Weighted per-element statistics for the pooled Boltzmann fit."""

    x_values: np.ndarray = field(repr=False)
    y_values: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    x_mean: float
    y_mean: float


@dataclass
class _CommonSlopeFit:
    """Result of the pooled common-slope Boltzmann regression."""

    slope: float
    slope_variance: float
    intercepts: Dict[str, float]
    element_stats: Dict[str, _CommonSlopeElementStats] = field(repr=False)
    r_squared: float = 0.0


class IterativeCFLIBSSolver:
    """
    Implements the iterative self-consistent CF-LIBS algorithm.

    Algorithm:
    1. Guess T, ne
    2. Saha-Boltzmann correction to map ionic lines to neutral plane
    3. Multi-species Boltzmann fit to find common T and species intercepts
    4. Closure equation to find relative concentrations
    5. Enforce Pressure/Charge balance to update ne
    6. Iterate until convergence
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        max_iterations: int = 20,
        t_tolerance_k: float = 100.0,
        ne_tolerance_frac: float = 0.1,
        pressure_pa: float = STP_PRESSURE,
        apply_ipd: bool = False,
        aki_uncertainty_weighting: bool = True,
        two_region: bool = False,
    ):
        self.atomic_db = atomic_db
        self.max_iterations = max_iterations
        self.t_tolerance_k = t_tolerance_k
        self.ne_tolerance_frac = ne_tolerance_frac
        self.pressure_pa = pressure_pa
        self.apply_ipd = apply_ipd
        self.aki_uncertainty_weighting = aki_uncertainty_weighting
        self.two_region = two_region
        self.boltzmann_fitter = BoltzmannPlotFitter(outlier_sigma=2.5)

    def _line_y_uncertainty(self, obs: LineObservation) -> float:
        """Return fit-space uncertainty with optional A_ki contribution."""
        sigma_y = obs.y_uncertainty
        unc = obs.aki_uncertainty
        if self.aki_uncertainty_weighting and unc is not None and np.isfinite(unc) and unc > 0:
            sigma_y = float(np.sqrt(sigma_y**2 + float(unc) ** 2))
        return sigma_y

    def _evaluate_partition_function(
        self, element: str, ionization_stage: int, T_K: float
    ) -> float:
        """Evaluate a partition function via direct summation, with fallbacks.

        Preferred path: direct sum over energy levels from the database.
        Fallback 1: polynomial coefficients from partition_functions table.
        Fallback 2: hardcoded statistical weight estimates.
        """
        from cflibs.plasma.partition import get_levels_for_species

        levels = get_levels_for_species(self.atomic_db, element, ionization_stage)
        if levels is not None:
            g_arr, E_arr, ip_ev = levels
            return PartitionFunctionEvaluator.evaluate_direct(T_K, g_arr, E_arr, ip_ev)

        pf = self.atomic_db.get_partition_coefficients(element, ionization_stage)
        if pf:
            return PartitionFunctionEvaluator.evaluate(T_K, pf.coefficients)
        if ionization_stage == 1:
            return 25.0
        if ionization_stage == 2:
            return 15.0
        return 2.0

    def _compute_saha_ratio(
        self,
        element: str,
        T_K: float,
        n_e_cm3: float,
        U_I: float,
        U_II: float,
        ip_ev: float,
    ) -> float:
        """Compute n_II / n_I using the first Saha ratio."""
        safe_ne = max(float(n_e_cm3), 1e10)
        T_eV = max(T_K / EV_TO_K, 0.1)
        return (SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5) * (U_II / U_I) * np.exp(-ip_ev / T_eV)

    def _compute_abundance_multipliers(
        self,
        elements: List[str],
        T_K: float,
        n_e_cm3: float,
        partition_funcs_I: Dict[str, float],
        partition_funcs_II: Dict[str, float],
        ips: Dict[str, float],
        T_corona: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Map the neutral-plane intercept back to total elemental abundance.

        The pooled Saha-Boltzmann fit returns q_s proportional to N_I / U_I.
        Closure must scale by (1 + n_II / n_I) to recover total elemental
        abundance before normalization.
        """
        multipliers: Dict[str, float] = {}
        # Per Hermann (2017), high-Z elements (Si, Fe, Ca, Al, Mg) in Aalto
        # are sensitive to the corona. We use T_corona for their neutral-plane
        # scaling if provided.
        corona_sensitive = {"Si", "Fe", "Ca", "Al", "Mg"}
        for el in elements:
            U_I = partition_funcs_I.get(el, 25.0)
            U_II = partition_funcs_II.get(el, 15.0)

            T_saha = T_K
            if T_corona is not None and el in corona_sensitive:
                # Weighted temperature for Saha-Boltzmann scaling
                T_saha = 0.3 * T_K + 0.7 * T_corona

            S = self._compute_saha_ratio(el, T_saha, n_e_cm3, U_I, U_II, ips[el])
            multipliers[el] = 1.0 + max(S, 0.0)
        return multipliers

    def _compute_effective_ips(
        self,
        ips: Dict[str, float],
        n_e: float,
        T_K: float,
    ) -> Dict[str, float]:
        """Compute ionization potentials with optional plasma screening applied."""
        if not self.apply_ipd:
            return ips

        from cflibs.plasma.saha_boltzmann import ionization_potential_lowering

        delta_chi = ionization_potential_lowering(n_e, T_K)
        return {el: max(ip - delta_chi, 0.0) for el, ip in ips.items()}

    def _apply_saha_correction(
        self,
        obs_by_element: Dict[str, List[LineObservation]],
        T_K: float,
        n_e: float,
        ips: Dict[str, float],
    ) -> Dict[str, List[LineObservation]]:
        """
        Map ionic lines to the neutral energy plane via the Saha-Boltzmann transform.

        For each ionic (stage-2) line, applies:
          y* = y - log(SAHA_CONST * T^1.5 / n_e)
          x* = E_k + IP

        Neutral lines are passed through unchanged.

        Parameters
        ----------
        obs_by_element : dict
            Raw observations grouped by element symbol
        T_K : float
            Plasma temperature [K]
        n_e : float
            Electron density [cm^-3]
        ips : dict
            First ionization potentials by element [eV]

        Returns
        -------
        dict
            Corrected observations grouped by element symbol
        """
        T_eV = max(T_K / EV_TO_K, 0.1)
        safe_ne = max(n_e, 1e10)
        correction_term = np.log((SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5))
        scale = np.exp(-correction_term)
        corrected: Dict[str, List[LineObservation]] = defaultdict(list)

        for el, obs_list in obs_by_element.items():
            ip = ips.get(el, 15.0)
            for obs in obs_list:
                if obs.ionization_stage == 2:
                    new_obs = LineObservation(
                        wavelength_nm=obs.wavelength_nm,
                        intensity=obs.intensity * scale,
                        intensity_uncertainty=obs.intensity_uncertainty * scale,
                        element=obs.element,
                        ionization_stage=obs.ionization_stage,
                        E_k_ev=obs.E_k_ev + ip,
                        g_k=obs.g_k,
                        A_ki=obs.A_ki,
                    )
                else:
                    new_obs = LineObservation(
                        wavelength_nm=obs.wavelength_nm,
                        intensity=obs.intensity,
                        intensity_uncertainty=obs.intensity_uncertainty,
                        element=obs.element,
                        ionization_stage=obs.ionization_stage,
                        E_k_ev=obs.E_k_ev,
                        g_k=obs.g_k,
                        A_ki=obs.A_ki,
                    )
                corrected[el].append(new_obs)

        return dict(corrected)

    def _fit_common_boltzmann_plane(
        self,
        corrected_obs_map: Dict[str, List[LineObservation]],
    ) -> Optional[_CommonSlopeFit]:
        """
        Compute a pooled Boltzmann slope common to multiple elements by fitting a single weighted linear slope to per-element centered Boltzmann data.

        For each element with at least two valid corrected observations, this routine computes weighted means in the Boltzmann plane, centers the element's points by those means, and pools the centered points across elements to fit a single slope. The result includes the fitted slope, its variance (accounting for one common slope plus one intercept per contributing element), per-element intercepts, per-element statistics (original values, weights, and means), and an R² goodness-of-fit metric.

        Returns:
            _CommonSlopeFit | None: A _CommonSlopeFit with fields:
                - slope: fitted common slope
                - slope_variance: estimated variance of the slope
                - intercepts: mapping from element to fitted intercept on the original (uncentered) scale
                - element_stats: per-element _CommonSlopeElementStats containing x/y values, weights, and means
                - r_squared: weighted R² of the pooled centered fit
            Returns None if there is insufficient valid data to perform the pooled fit.
        """
        pooled_x_parts: List[np.ndarray] = []
        pooled_y_parts: List[np.ndarray] = []
        pooled_w_parts: List[np.ndarray] = []
        element_stats: Dict[str, _CommonSlopeElementStats] = {}

        for el, obs_list in corrected_obs_map.items():
            if len(obs_list) < 2:
                continue

            xs = np.array([o.E_k_ev for o in obs_list], dtype=float)
            ys = np.array([o.y_value for o in obs_list], dtype=float)
            y_uncertainties = np.array([self._line_y_uncertainty(o) for o in obs_list])
            ws = np.array(
                [1.0 / sigma_y**2 if sigma_y > 0 else 1.0 for sigma_y in y_uncertainties],
                dtype=float,
            )

            valid_mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ws) & (ws > 0.0)
            xs = xs[valid_mask]
            ys = ys[valid_mask]
            ws = ws[valid_mask]

            if xs.size < 2:
                continue

            x_mean = float(np.average(xs, weights=ws))
            y_mean = float(np.average(ys, weights=ws))

            element_stats[el] = _CommonSlopeElementStats(
                x_values=xs,
                y_values=ys,
                weights=ws,
                x_mean=x_mean,
                y_mean=y_mean,
            )
            pooled_x_parts.append(xs - x_mean)
            pooled_y_parts.append(ys - y_mean)
            pooled_w_parts.append(ws)

        if not pooled_x_parts:
            return None

        pooled_x = np.concatenate(pooled_x_parts)
        pooled_y = np.concatenate(pooled_y_parts)
        pooled_w = np.concatenate(pooled_w_parts)

        if pooled_x.size < 3:
            return None

        denom = float(np.sum(pooled_w * pooled_x**2))
        if not np.isfinite(denom) or denom <= 0.0:
            return None

        slope = float(np.sum(pooled_w * pooled_x * pooled_y) / denom)
        residuals = pooled_y - slope * pooled_x
        ss_res = float(np.sum(pooled_w * residuals**2))
        ss_tot = float(np.sum(pooled_w * pooled_y**2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 1.0

        # The centered pooled fit is equivalent to y = a_element + m x, so the
        # residual variance must account for one common slope plus one intercept
        # per contributing element.
        dof = max(int(pooled_x.size) - (1 + len(element_stats)), 1)
        slope_variance = ss_res / (dof * denom)
        if not np.isfinite(slope_variance) or slope_variance <= 0.0:
            slope_variance = 1.0 / denom

        intercepts = {
            el: stats.y_mean - slope * stats.x_mean for el, stats in element_stats.items()
        }

        return _CommonSlopeFit(
            slope=slope,
            slope_variance=slope_variance,
            intercepts=intercepts,
            element_stats=element_stats,
            r_squared=r_squared,
        )

    def solve(
        self, observations: List[LineObservation], closure_mode: str = "standard", **closure_kwargs
    ) -> CFLIBSResult:
        """
        Estimate plasma temperature, electron density, and elemental concentrations from spectral line observations using the iterative CF-LIBS algorithm.

        Parameters:
            observations (List[LineObservation]): Spectral line observations to invert; lines are grouped by element.
            closure_mode (str): Closure method for converting Boltzmann intercepts to concentrations. One of "standard", "matrix", "oxide", "ilr", "pwlr", or "dirichlet_residual".
            **closure_kwargs: Additional keyword arguments forwarded to the chosen closure method (for example, a matrix_element for "matrix" mode).

        Returns:
            CFLIBSResult: Final inversion result containing:
                - temperature_K: Estimated plasma temperature (Kelvin).
                - temperature_uncertainty_K: Set to 0.0 in this routine (see solve_with_uncertainty for propagated uncertainties).
                - electron_density_cm3: Estimated electron density (cm^-3).
                - concentrations: Dictionary of elemental concentrations (relative units returned by the chosen closure).
                - concentration_uncertainties: Empty in this routine (see solve_with_uncertainty).
                - iterations: Number of iterations performed.
                - converged: Whether the iterative solver met convergence criteria.
                - quality_metrics: Diagnostics including the last Boltzmann fit R^2 and LTE validation metrics.
                - electron_density_uncertainty_cm3: Set to 0.0 here.
                - boltzmann_covariance: None in this routine; covariance information is produced by solve_with_uncertainty.
        """
        # 1. Initialization
        T_K = 10000.0
        T_corona = None
        n_e = 1.0e17

        # Cache static data (IPs, atomic data)
        # Group observations by element
        obs_by_element = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        elements = list(obs_by_element.keys())

        # Pre-fetch Ionization Potentials
        ips = {}
        for el in elements:
            # Need IP of neutral (I -> II)
            ip = self.atomic_db.get_ionization_potential(el, 1)
            if ip is None:
                logger.warning(f"No IP for {el} I, assuming high")
                ip = 15.0  # Fallback
            ips[el] = ip

        # Iteration loop
        converged = False
        history = []
        concentrations: Dict[str, float] = {}  # Initialize before loop
        last_common_fit: Optional[_CommonSlopeFit] = None

        for _ in range(1, self.max_iterations + 1):
            T_prev = T_K
            ne_prev = n_e

            T_eV = T_K / EV_TO_K
            if T_eV < 0.1:
                T_eV = 0.1  # clamp

            # 2. Calculate Partition Functions & Saha Corrections
            partition_funcs = {}  # U_I for each element
            partition_funcs_II = {}

            for el in elements:
                partition_funcs[el] = self._evaluate_partition_function(el, 1, T_K)
                partition_funcs_II[el] = self._evaluate_partition_function(el, 2, T_K)

            effective_ips = self._compute_effective_ips(ips, n_e, T_K)

            # Map ionic lines to the neutral energy plane
            corrected_obs_map = self._apply_saha_correction(obs_by_element, T_K, n_e, effective_ips)

            # 3. Multi-species Boltzmann Fit
            common_fit = self._fit_common_boltzmann_plane(corrected_obs_map)
            if common_fit is None:
                logger.warning("Insufficient points for fit")
                break

            last_common_fit = common_fit
            slope = common_fit.slope

            # Update T
            if slope >= 0:
                T_new = 50000.0  # Clamp max
            else:
                T_new = -1.0 / (slope * KB_EV)

            # Damping
            T_K = 0.5 * T_prev + 0.5 * T_new

            if self.two_region:
                # Per Hermann (2017), corona temperature is typically 70-90% of core.
                # We use a fixed ratio of 0.8 for the iterative update.
                T_corona = 0.8 * T_K

            # Calculate Intercepts
            intercepts = common_fit.intercepts

            abundance_multipliers = self._compute_abundance_multipliers(
                list(intercepts.keys()),
                T_K,
                n_e,
                partition_funcs,
                partition_funcs_II,
                effective_ips,
                T_corona=T_corona,
            )

            # 4. Closure
            if closure_mode == "matrix":
                closure_res = ClosureEquation.apply_matrix_mode(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                    **closure_kwargs,
                )
            elif closure_mode == "oxide":
                closure_res = ClosureEquation.apply_oxide_mode(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                    **closure_kwargs,
                )
            elif closure_mode == "ilr":
                closure_res = ClosureEquation.apply_ilr(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                )
            elif closure_mode == "pwlr":
                closure_res = ClosureEquation.apply_pwlr(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                    **closure_kwargs,
                )
            elif closure_mode == "dirichlet_residual":
                closure_res = ClosureEquation.apply_dirichlet_residual(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                    **closure_kwargs,
                )
            else:
                closure_res = ClosureEquation.apply_standard(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                )

            concentrations = closure_res.concentrations

            # 5. Update electron density via pressure balance

            # Calculate avg_Z based on Saha ratios
            # n_II / n_I = S(T, ne)
            # Z_bar_s = (1*n_I + 2*n_II) / (n_I + n_II) - 1  (since Z=0 for neutral? No, Z=1 neutral in our code)
            # In code sp_num=1 is neutral (charge 0), sp_num=2 is ion (charge +1)
            # So electron contribution: Neutral=0, Ion=1
            # Avg electrons per atom of species s:
            # eps_s = (n_II) / (n_I + n_II) = S / (1+S)

            total_eps = 0.0
            for el, C_s in concentrations.items():
                U_I = partition_funcs.get(el, 25.0)
                U_II = partition_funcs_II.get(el, 15.0)
                S = self._compute_saha_ratio(el, T_K, n_e, U_I, U_II, effective_ips[el])
                eps_s = S / (1.0 + S)
                total_eps += C_s * eps_s

            avg_Z = total_eps

            # Total particle density (nuclei)
            # n_tot = P / (k T (1 + avg_Z))
            # n_e = avg_Z * n_tot

            n_tot = self.pressure_pa / (KB * T_K * (1.0 + avg_Z))
            # Convert to cm^-3
            n_tot_cm3 = n_tot * 1e-6

            ne_new = avg_Z * n_tot_cm3

            # Damping
            n_e = 0.5 * ne_prev + 0.5 * ne_new

            history.append((T_K, n_e))

            # Check convergence
            if (
                abs(T_K - T_prev) < self.t_tolerance_k
                and abs(n_e - ne_prev) / ne_prev < self.ne_tolerance_frac
            ):
                converged = True
                break

        # LTE validity check
        from cflibs.plasma.lte_validator import LTEValidator

        lte_validator = LTEValidator()
        lte_report = lte_validator.validate(
            T_K=T_K,
            n_e_cm3=n_e,
            observations=observations,
        )
        quality_metrics = {
            "r_squared_last": last_common_fit.r_squared if last_common_fit is not None else 0.0
        }
        quality_metrics.update(lte_report.quality_metrics)

        if self.two_region and T_corona is None:
            T_corona = 0.8 * T_K

        return CFLIBSResult(
            temperature_K=T_K,
            temperature_uncertainty_K=0.0,  # See solve_with_uncertainty for propagation
            electron_density_cm3=n_e,
            concentrations=concentrations,
            concentration_uncertainties={},  # See solve_with_uncertainty for propagation
            iterations=len(history),
            converged=converged,
            temperature_corona_K=T_corona,
            quality_metrics=quality_metrics,
            electron_density_uncertainty_cm3=0.0,
            boltzmann_covariance=None,
        )

    def solve_with_uncertainty(
        self,
        observations: List[LineObservation],
        closure_mode: str = "standard",
        **closure_kwargs,
    ) -> CFLIBSResult:
        """
        Compute plasma parameters while propagating measurement and fit uncertainties.

        Performs uncertainty propagation through the pooled Boltzmann fit, Saha
        correction, and the chosen closure equation, returning the same result
        structure as solve() augmented with uncertainty fields.

        Parameters:
            observations (List[LineObservation]): Spectral lines with intensity uncertainties.
            closure_mode (str): Closure algorithm to use ('standard', 'matrix', 'oxide', 'ilr', 'pwlr', or 'dirichlet_residual').
            **closure_kwargs: Arguments passed to the chosen closure routine (e.g. 'matrix_element',
                'matrix_fraction', or 'oxide_stoichiometry').

        Returns:
            CFLIBSResult: Solver result including populated uncertainty fields:
                - temperature_uncertainty_K: estimated standard deviation of temperature (K)
                - concentration_uncertainties: per-element concentration uncertainties
                - boltzmann_covariance: selected 2x2 covariance matrix for slope/intercept (or None)

        Raises:
            ImportError: If the external `uncertainties`-based utilities are not available.
        """
        # First run the standard solver to convergence
        result = self.solve(observations, closure_mode, **closure_kwargs)

        # Import uncertainty utilities (will raise ImportError if not available)
        from cflibs.inversion.uncertainty import (
            propagate_through_closure_oxide,
            propagate_through_closure_standard,
            propagate_through_closure_matrix,
            extract_values_and_uncertainties,
        )
        from uncertainties import ufloat

        # Group observations by element
        obs_by_element: Dict[str, list] = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        elements = list(obs_by_element.keys())

        # Use converged plasma state
        T_K = result.temperature_K
        n_e = result.electron_density_cm3

        # Pre-fetch ionization potentials (same as solve())
        ips = {}
        for el in elements:
            ip = self.atomic_db.get_ionization_potential(el, 1)
            ips[el] = ip if ip is not None else 15.0

        effective_ips = self._compute_effective_ips(ips, n_e, T_K)

        # Apply Saha correction so intercepts match those from solve()
        corrected_obs_map = self._apply_saha_correction(obs_by_element, T_K, n_e, effective_ips)

        # Get partition functions at converged T
        partition_funcs = {}
        partition_funcs_II = {}
        for el in elements:
            partition_funcs[el] = self._evaluate_partition_function(el, 1, T_K)
            partition_funcs_II[el] = self._evaluate_partition_function(el, 2, T_K)

        abundance_multipliers = self._compute_abundance_multipliers(
            elements,
            T_K,
            n_e,
            partition_funcs,
            partition_funcs_II,
            effective_ips,
            T_corona=result.temperature_corona_K,
        )

        common_fit = self._fit_common_boltzmann_plane(corrected_obs_map)
        if common_fit is None:
            return result

        # Propagate the same common-slope model used by solve()
        intercepts_u = {}
        covariances = {}
        slope_err = (
            float(np.sqrt(common_fit.slope_variance))
            if np.isfinite(common_fit.slope_variance) and common_fit.slope_variance > 0.0
            else 0.0
        )
        slope_u = ufloat(common_fit.slope, slope_err)

        for el, stats in common_fit.element_stats.items():
            weight_sum = float(np.sum(stats.weights))
            y_mean_err = np.sqrt(1.0 / weight_sum) if weight_sum > 0.0 else 0.0
            y_mean_u = ufloat(stats.y_mean, y_mean_err)
            intercept_u = y_mean_u - slope_u * stats.x_mean
            intercepts_u[el] = intercept_u

            intercept_var = y_mean_err**2 + (stats.x_mean**2) * common_fit.slope_variance
            covariances[el] = np.array(
                [
                    [common_fit.slope_variance, -stats.x_mean * common_fit.slope_variance],
                    [-stats.x_mean * common_fit.slope_variance, intercept_var],
                ],
                dtype=float,
            )

        # Propagate through closure
        if closure_mode == "matrix" and "matrix_element" in closure_kwargs:
            concentrations_u = propagate_through_closure_matrix(
                intercepts_u,
                partition_funcs,
                closure_kwargs["matrix_element"],
                closure_kwargs.get("matrix_fraction", 0.9),
                abundance_multipliers=abundance_multipliers,
            )
        elif closure_mode == "oxide":
            concentrations_u = propagate_through_closure_oxide(
                intercepts_u,
                partition_funcs,
                closure_kwargs.get("oxide_stoichiometry", {}),
                abundance_multipliers=abundance_multipliers,
            )
        elif closure_mode in {"ilr", "pwlr", "dirichlet_residual"}:
            concentrations_u = propagate_through_closure_standard(
                intercepts_u,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )
        else:
            concentrations_u = propagate_through_closure_standard(
                intercepts_u,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )

        # Extract nominal values and uncertainties
        conc_nominal, conc_uncert = extract_values_and_uncertainties(concentrations_u)

        # Temperature uncertainty from pooled slope estimate
        from cflibs.inversion.uncertainty import temperature_from_slope

        T_err = 0.0
        if slope_err > 0.0:
            T_K_u = temperature_from_slope(slope_u)
            T_err = float(T_K_u.std_dev) if np.isfinite(T_K_u.std_dev) else 0.0

        selected_covariance = None
        covariance_element = None
        if covariances:
            preferred_element = (
                closure_kwargs.get("matrix_element") if closure_mode == "matrix" else None
            )
            if preferred_element in covariances:
                covariance_element = preferred_element
            else:
                covariance_element = sorted(covariances)[0]
            selected_covariance = covariances[covariance_element]

        quality_metrics = dict(result.quality_metrics)
        if covariance_element is not None:
            quality_metrics["boltzmann_covariance_element"] = covariance_element

        return CFLIBSResult(
            temperature_K=result.temperature_K,
            temperature_uncertainty_K=T_err,
            electron_density_cm3=result.electron_density_cm3,
            concentrations=conc_nominal if conc_nominal else result.concentrations,
            concentration_uncertainties=conc_uncert if conc_uncert else {},
            iterations=result.iterations,
            converged=result.converged,
            temperature_corona_K=result.temperature_corona_K,
            quality_metrics=quality_metrics,
            electron_density_uncertainty_cm3=0.0,  # Would need iterative uncertainty
            boltzmann_covariance=selected_covariance,
        )


# ---------------------------------------------------------------------------
# JAX-accelerated iterative solver
# ---------------------------------------------------------------------------


if HAS_JAX:

    @jit
    def _saha_correct_kernel(
        x: jnp.ndarray,
        y: jnp.ndarray,
        stage: jnp.ndarray,
        ip: jnp.ndarray,
        T_eV: jnp.ndarray,
        log_correction: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Vectorized Saha correction kernel.

        For each line: if stage==2 (ionic), shift y by ``-log_correction`` and
        x by ``+ip``. Neutral lines are passed through unchanged.

        All arrays are shape (B, N_max). ``stage`` is 1 or 2.
        ``ip`` and ``T_eV`` are scalars or shape (B, N_max) broadcastable.
        """
        is_ionic = stage == 2
        y_corr = jnp.where(is_ionic, y - log_correction, y)
        x_corr = jnp.where(is_ionic, x + ip, x)
        return x_corr, y_corr

    @jit
    def _common_slope_kernel(
        x: jnp.ndarray,  # (E, N_max) - per-element padded x
        y: jnp.ndarray,  # (E, N_max) - per-element padded y
        w: jnp.ndarray,  # (E, N_max) - per-element padded weights
        mask: jnp.ndarray,  # (E, N_max) - bool mask
    ) -> Dict[str, jnp.ndarray]:
        """
        JAX kernel for pooled common-slope Boltzmann fit across E elements.

        Mirrors ``IterativeCFLIBSSolver._fit_common_boltzmann_plane`` math
        exactly: per-element weighted means, centered points, single pooled
        weighted slope, per-element intercepts on uncentered scale.

        Returns dict with keys: slope, slope_variance, r_squared,
        intercepts (E,), x_means (E,), y_means (E,), n_valid_per_el (E,).
        """
        mf = mask.astype(jnp.float64)
        w_eff = w * mf  # zero out padded entries

        # Per-element weight totals
        S_w = jnp.sum(w_eff, axis=1)  # (E,)
        # Per-element weighted means; guard zero-weight elements
        denom_safe = jnp.where(S_w > 0.0, S_w, 1.0)
        x_means = jnp.sum(w_eff * x, axis=1) / denom_safe  # (E,)
        y_means = jnp.sum(w_eff * y, axis=1) / denom_safe

        # Centered points (still padded; padded entries get w_eff=0 so they
        # don't contribute regardless of the centering values)
        xc = x - x_means[:, None]
        yc = y - y_means[:, None]

        # Pooled sums (across E and N_max simultaneously)
        sum_wxx = jnp.sum(w_eff * xc * xc)  # scalar
        sum_wxy = jnp.sum(w_eff * xc * yc)  # scalar

        # Single common slope
        denom = sum_wxx
        slope = jnp.where(denom > 0.0, sum_wxy / jnp.where(denom > 0.0, denom, 1.0), 0.0)

        # Pooled residuals and r_squared
        y_pred_centered = slope * xc
        residuals = yc - y_pred_centered
        ss_res = jnp.sum(w_eff * residuals * residuals)
        ss_tot = jnp.sum(w_eff * yc * yc)
        r_squared = jnp.where(ss_tot > 0.0, 1.0 - ss_res / jnp.where(ss_tot > 0.0, ss_tot, 1.0), 1.0)

        # Per-element valid counts (for DOF)
        n_valid_per_el = jnp.sum(mf, axis=1)
        # DOF accounts for one slope plus one intercept per contributing element
        n_total = jnp.sum(n_valid_per_el)
        n_elements_active = jnp.sum(n_valid_per_el >= 2.0)
        dof = jnp.maximum(n_total - (1.0 + n_elements_active), 1.0)
        slope_variance = jnp.where(denom > 0.0, ss_res / (dof * jnp.where(denom > 0.0, denom, 1.0)), 1.0)
        # Fall back to inverse Fisher information when residual variance is degenerate
        slope_variance = jnp.where(
            (slope_variance > 0.0) & jnp.isfinite(slope_variance),
            slope_variance,
            jnp.where(denom > 0.0, 1.0 / jnp.where(denom > 0.0, denom, 1.0), 1.0),
        )

        intercepts = y_means - slope * x_means

        return {
            "slope": slope,
            "slope_variance": slope_variance,
            "r_squared": r_squared,
            "intercepts": intercepts,
            "x_means": x_means,
            "y_means": y_means,
            "n_valid_per_el": n_valid_per_el,
        }


def _build_padded_arrays_from_obs(
    obs_by_element: Dict[str, List[LineObservation]],
):
    """
    Build padded (E, N_max) numpy arrays from per-element observation lists.

    Returns:
        elements: list of element symbols (length E, in dict-iteration order)
        x: (E, N_max) E_k_ev padded with 0
        y: (E, N_max) y_value padded with 0
        w: (E, N_max) inverse-variance weights padded with 0
        stage: (E, N_max) ionization stage (1 or 2) padded with 1
        mask: (E, N_max) bool mask
    """
    elements = list(obs_by_element.keys())
    if not elements:
        return [], None, None, None, None, None
    counts = [len(obs_by_element[el]) for el in elements]
    n_max = max(counts) if counts else 0
    E = len(elements)
    if n_max == 0:
        return elements, None, None, None, None, None
    x = np.zeros((E, n_max), dtype=np.float64)
    y = np.zeros((E, n_max), dtype=np.float64)
    w = np.zeros((E, n_max), dtype=np.float64)
    stage = np.ones((E, n_max), dtype=np.int32)
    mask = np.zeros((E, n_max), dtype=bool)
    for i, el in enumerate(elements):
        for j, obs in enumerate(obs_by_element[el]):
            x[i, j] = obs.E_k_ev
            y[i, j] = obs.y_value
            sigma = obs.y_uncertainty
            w[i, j] = 1.0 / (sigma**2) if sigma > 0 else 1.0
            stage[i, j] = obs.ionization_stage
            mask[i, j] = (
                np.isfinite(obs.y_value)
                and np.isfinite(obs.E_k_ev)
                and np.isfinite(w[i, j])
                and w[i, j] > 0.0
            )
    # Zero-out invalid entries in x/y/w so masked sums are clean
    x = np.where(mask, x, 0.0)
    y = np.where(mask, y, 0.0)
    w = np.where(mask, w, 0.0)
    return elements, x, y, w, stage, mask


class IterativeCFLIBSSolverJax(IterativeCFLIBSSolver):
    """
    JAX-accelerated iterative CF-LIBS solver.

    Drop-in replacement for :class:`IterativeCFLIBSSolver` with the same
    constructor signature and ``solve()``/``solve_with_uncertainty()`` API.
    The hot path (Saha correction + pooled Boltzmann fit) is dispatched to
    JAX kernels; the partition-function evaluation, closure equation, and
    pressure-balance update remain in numpy/Python because they require
    ``AtomicDatabase`` lookups and per-element dictionary plumbing.

    The solver falls back gracefully to the numpy parent implementation if
    JAX is not available at instantiation time -- callers can always use
    ``IterativeCFLIBSSolverJax`` without paying the import-error cost.

    Numerical equivalence: the same closed-form pooled WLS algebra is used
    here and in the numpy parent; composition outputs match to ``rtol=1e-3``
    on representative multi-element fixtures (see
    ``tests/inversion/test_solver_jax_parity.py``).
    """

    backend: str  # "jax" or "numpy_fallback"

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        max_iterations: int = 20,
        t_tolerance_k: float = 100.0,
        ne_tolerance_frac: float = 0.1,
        pressure_pa: float = STP_PRESSURE,
        apply_ipd: bool = False,
    ):
        super().__init__(
            atomic_db=atomic_db,
            max_iterations=max_iterations,
            t_tolerance_k=t_tolerance_k,
            ne_tolerance_frac=ne_tolerance_frac,
            pressure_pa=pressure_pa,
            apply_ipd=apply_ipd,
        )
        if HAS_JAX:
            try:
                self._jax_backend = jax.default_backend()
            except Exception:  # pragma: no cover
                self._jax_backend = "cpu"
            self.backend = "jax"
        else:
            logger.info(
                "JAX unavailable; IterativeCFLIBSSolverJax will fall back to the "
                "numpy parent implementation."
            )
            self._jax_backend = None
            self.backend = "numpy_fallback"

    @property
    def jax_backend(self) -> Optional[str]:
        """Active JAX backend ('cpu', 'gpu', 'tpu') or None when JAX is absent."""
        return self._jax_backend

    # -- Hot path: vectorized Saha + common-slope Boltzmann fit ---------------

    def _saha_and_fit_jax(
        self,
        elements: List[str],
        x_raw: np.ndarray,
        y_raw: np.ndarray,
        w_raw: np.ndarray,
        stage_arr: np.ndarray,
        mask_arr: np.ndarray,
        T_K: float,
        n_e: float,
        ips: Dict[str, float],
    ) -> Optional[_CommonSlopeFit]:
        """
        Apply Saha correction + pooled common-slope fit using JAX kernels.

        Returns a ``_CommonSlopeFit`` populated from JAX-side reductions, or
        None when there is insufficient valid data.
        """
        if not HAS_JAX:
            return None

        # Saha correction inputs
        T_eV = max(T_K / EV_TO_K, 0.1)
        safe_ne = max(float(n_e), 1e10)
        log_correction = float(np.log((SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5)))

        # Build per-element ip array broadcast to (E, N_max)
        ip_per_el = np.array([ips.get(el, 15.0) for el in elements], dtype=np.float64)
        ip_arr = np.broadcast_to(ip_per_el[:, None], x_raw.shape).astype(np.float64)

        # Move to device
        x_d = jnp.asarray(x_raw, dtype=jnp.float64)
        y_d = jnp.asarray(y_raw, dtype=jnp.float64)
        w_d = jnp.asarray(w_raw, dtype=jnp.float64)
        stage_d = jnp.asarray(stage_arr, dtype=jnp.int32)
        ip_d = jnp.asarray(ip_arr, dtype=jnp.float64)
        mask_d = jnp.asarray(mask_arr, dtype=bool)
        log_correction_d = jnp.asarray(log_correction, dtype=jnp.float64)

        # Kernel 1: Saha correction
        x_c, y_c = _saha_correct_kernel(
            x_d, y_d, stage_d, ip_d, jnp.asarray(T_eV, dtype=jnp.float64), log_correction_d
        )

        # Kernel 2: pooled common-slope fit
        fit = _common_slope_kernel(x_c, y_c, w_d, mask_d)

        # Pull scalars/arrays back to host (single sync point per iteration)
        slope = float(fit["slope"])
        slope_variance = float(fit["slope_variance"])
        r_squared = float(fit["r_squared"])
        intercepts_arr = np.asarray(fit["intercepts"])
        x_means_arr = np.asarray(fit["x_means"])
        y_means_arr = np.asarray(fit["y_means"])
        n_valid_per_el = np.asarray(fit["n_valid_per_el"])

        # Validity check: need at least one element with >=2 points and >=3 total
        active_elements = [el for el, n in zip(elements, n_valid_per_el) if n >= 2]
        if not active_elements:
            return None
        if int(n_valid_per_el.sum()) < 3:
            return None
        if not np.isfinite(slope) or not np.isfinite(slope_variance):
            return None

        # Build _CommonSlopeFit dataclass with per-element stats so downstream
        # consumers (e.g. solve_with_uncertainty) keep working.
        element_stats: Dict[str, _CommonSlopeElementStats] = {}
        intercepts_dict: Dict[str, float] = {}
        for i, el in enumerate(elements):
            if n_valid_per_el[i] < 2:
                continue
            row_mask = mask_arr[i]
            xs = x_raw[i][row_mask].astype(np.float64)
            # NOTE: x_raw stores pre-Saha E_k; for ionic lines apply the
            # IP shift on the host side so element_stats reflects the
            # corrected (neutral-plane) coordinates the downstream
            # uncertainty propagation expects.
            xs_corrected = xs.copy()
            stage_row = stage_arr[i][row_mask]
            ip_el = ips.get(el, 15.0)
            xs_corrected[stage_row == 2] += ip_el
            ys_raw = y_raw[i][row_mask].astype(np.float64)
            ys_corrected = ys_raw.copy()
            ys_corrected[stage_row == 2] -= log_correction
            ws = w_raw[i][row_mask].astype(np.float64)
            element_stats[el] = _CommonSlopeElementStats(
                x_values=xs_corrected,
                y_values=ys_corrected,
                weights=ws,
                x_mean=float(x_means_arr[i]),
                y_mean=float(y_means_arr[i]),
            )
            intercepts_dict[el] = float(intercepts_arr[i])

        if not element_stats:
            return None

        return _CommonSlopeFit(
            slope=slope,
            slope_variance=slope_variance,
            intercepts=intercepts_dict,
            element_stats=element_stats,
            r_squared=r_squared,
        )

    # -- Public solve() -------------------------------------------------------

    def solve(
        self,
        observations: List[LineObservation],
        closure_mode: str = "standard",
        **closure_kwargs,
    ) -> CFLIBSResult:
        """
        JAX-accelerated equivalent of :meth:`IterativeCFLIBSSolver.solve`.

        Falls back to the parent numpy implementation if JAX is not available.
        """
        if not HAS_JAX:
            return super().solve(observations, closure_mode, **closure_kwargs)

        # Initialization (mirrors numpy solver)
        T_K = 10000.0
        n_e = 1.0e17

        obs_by_element: Dict[str, List[LineObservation]] = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        elements = list(obs_by_element.keys())

        ips: Dict[str, float] = {}
        for el in elements:
            ip = self.atomic_db.get_ionization_potential(el, 1)
            if ip is None:
                logger.warning(f"No IP for {el} I, assuming high")
                ip = 15.0
            ips[el] = ip

        # Pre-build padded arrays once (raw, pre-Saha)
        elements_ord, x_raw, y_raw, w_raw, stage_arr, mask_arr = _build_padded_arrays_from_obs(
            dict(obs_by_element)
        )
        if x_raw is None:
            logger.warning("No usable observations for JAX solver; falling back to numpy")
            return super().solve(observations, closure_mode, **closure_kwargs)

        converged = False
        history: List[Tuple[float, float]] = []
        concentrations: Dict[str, float] = {}
        last_common_fit: Optional[_CommonSlopeFit] = None

        for _ in range(1, self.max_iterations + 1):
            T_prev = T_K
            ne_prev = n_e

            partition_funcs: Dict[str, float] = {}
            partition_funcs_II: Dict[str, float] = {}
            for el in elements_ord:
                partition_funcs[el] = self._evaluate_partition_function(el, 1, T_K)
                partition_funcs_II[el] = self._evaluate_partition_function(el, 2, T_K)

            effective_ips = self._compute_effective_ips(ips, n_e, T_K)

            # JAX hot path: Saha correction + pooled common-slope fit
            common_fit = self._saha_and_fit_jax(
                elements_ord,
                x_raw,
                y_raw,
                w_raw,
                stage_arr,
                mask_arr,
                T_K,
                n_e,
                effective_ips,
            )
            if common_fit is None:
                logger.warning("Insufficient points for JAX fit")
                break

            last_common_fit = common_fit
            slope = common_fit.slope

            if slope >= 0:
                T_new = 50000.0
            else:
                T_new = -1.0 / (slope * KB_EV)

            T_K = 0.5 * T_prev + 0.5 * T_new

            intercepts = common_fit.intercepts

            abundance_multipliers = self._compute_abundance_multipliers(
                list(intercepts.keys()),
                T_K,
                n_e,
                partition_funcs,
                partition_funcs_II,
                effective_ips,
            )

            if closure_mode == "matrix":
                closure_res = ClosureEquation.apply_matrix_mode(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                    **closure_kwargs,
                )
            elif closure_mode == "oxide":
                closure_res = ClosureEquation.apply_oxide_mode(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                    **closure_kwargs,
                )
            elif closure_mode == "ilr":
                closure_res = ClosureEquation.apply_ilr(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                )
            elif closure_mode == "pwlr":
                closure_res = ClosureEquation.apply_pwlr(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                    **closure_kwargs,
                )
            else:
                closure_res = ClosureEquation.apply_standard(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                )

            concentrations = closure_res.concentrations

            # Pressure balance n_e update (numpy)
            total_eps = 0.0
            for el, C_s in concentrations.items():
                U_I = partition_funcs.get(el, 25.0)
                U_II = partition_funcs_II.get(el, 15.0)
                S = self._compute_saha_ratio(el, T_K, n_e, U_I, U_II, effective_ips[el])
                eps_s = S / (1.0 + S)
                total_eps += C_s * eps_s
            avg_Z = total_eps

            n_tot = self.pressure_pa / (KB * T_K * (1.0 + avg_Z))
            n_tot_cm3 = n_tot * 1e-6
            ne_new = avg_Z * n_tot_cm3
            n_e = 0.5 * ne_prev + 0.5 * ne_new

            history.append((T_K, n_e))

            if (
                abs(T_K - T_prev) < self.t_tolerance_k
                and abs(n_e - ne_prev) / ne_prev < self.ne_tolerance_frac
            ):
                converged = True
                break

        # LTE quality metrics (same as numpy)
        from cflibs.plasma.lte_validator import LTEValidator

        lte_validator = LTEValidator()
        lte_report = lte_validator.validate(
            T_K=T_K,
            n_e_cm3=n_e,
            observations=observations,
        )
        quality_metrics = {
            "r_squared_last": last_common_fit.r_squared if last_common_fit is not None else 0.0,
            "backend": self.backend,
            "jax_backend": self._jax_backend or "n/a",
        }
        quality_metrics.update(lte_report.quality_metrics)

        return CFLIBSResult(
            temperature_K=T_K,
            temperature_uncertainty_K=0.0,
            electron_density_cm3=n_e,
            concentrations=concentrations,
            concentration_uncertainties={},
            iterations=len(history),
            converged=converged,
            quality_metrics=quality_metrics,
            electron_density_uncertainty_cm3=0.0,
            boltzmann_covariance=None,
        )
