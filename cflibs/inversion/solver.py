"""
Iterative solver for Classic CF-LIBS.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict

from cflibs.core.constants import KB, KB_EV, SAHA_CONST_CM3, STP_PRESSURE, EV_TO_K
from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.boltzmann import LineObservation, BoltzmannPlotFitter
from cflibs.inversion.closure import ClosureEquation
from cflibs.plasma.partition import PartitionFunctionEvaluator
from cflibs.core.logging_config import get_logger

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
        Element concentrations (mass fractions, sum to 1)
    concentration_uncertainties : Dict[str, float]
        1-sigma uncertainties in concentrations
    iterations : int
        Number of iterations performed
    converged : bool
        Whether solver converged within tolerance
    quality_metrics : Dict[str, float]
        Quality metrics (R², chi², etc.)
    boltzmann_covariance : np.ndarray, optional
        2x2 covariance matrix of final Boltzmann fit (slope, intercept)
    """

    temperature_K: float
    temperature_uncertainty_K: float
    electron_density_cm3: float
    concentrations: Dict[str, float]
    concentration_uncertainties: Dict[str, float]
    iterations: int
    converged: bool
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    electron_density_uncertainty_cm3: float = 0.0
    boltzmann_covariance: Optional[np.ndarray] = field(default=None, repr=False)


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
    ):
        self.atomic_db = atomic_db
        self.max_iterations = max_iterations
        self.t_tolerance_k = t_tolerance_k
        self.ne_tolerance_frac = ne_tolerance_frac
        self.pressure_pa = pressure_pa
        self.boltzmann_fitter = BoltzmannPlotFitter(outlier_sigma=2.5)

    def _evaluate_partition_function(
        self, element: str, ionization_stage: int, T_K: float
    ) -> float:
        """Evaluate a partition function with simple production fallbacks."""
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
    ) -> Dict[str, float]:
        """
        Map the neutral-plane intercept back to total elemental abundance.

        The pooled Saha-Boltzmann fit returns q_s proportional to N_I / U_I.
        Closure must scale by (1 + n_II / n_I) to recover total elemental
        abundance before normalization.
        """
        multipliers: Dict[str, float] = {}
        for el in elements:
            U_I = partition_funcs_I.get(el, 25.0)
            U_II = partition_funcs_II.get(el, 15.0)
            S = self._compute_saha_ratio(el, T_K, n_e_cm3, U_I, U_II, ips[el])
            multipliers[el] = 1.0 + max(S, 0.0)
        return multipliers

    def solve(
        self, observations: List[LineObservation], closure_mode: str = "standard", **closure_kwargs
    ) -> CFLIBSResult:
        """
        Solve for plasma parameters.

        Parameters
        ----------
        observations : List[LineObservation]
            Spectral lines
        closure_mode : str
            'standard', 'matrix', or 'oxide'
        closure_kwargs : dict
            Arguments for closure equation (e.g. matrix_element)

        Returns
        -------
        CFLIBSResult
        """
        # 1. Initialization
        T_K = 10000.0
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

        for _ in range(1, self.max_iterations + 1):
            T_prev = T_K
            ne_prev = n_e

            T_eV = T_K / EV_TO_K
            if T_eV < 0.1:
                T_eV = 0.1  # clamp

            # 2. Calculate Partition Functions & Saha Corrections
            partition_funcs = {}  # U_I for each element
            partition_funcs_II = {}
            corrected_obs_map = defaultdict(list)

            for el in elements:
                U_I = self._evaluate_partition_function(el, 1, T_K)
                U_II = self._evaluate_partition_function(el, 2, T_K)
                partition_funcs[el] = U_I
                partition_funcs_II[el] = U_II

                # Saha correction: map ionic lines to neutral energy plane

                # Standard Saha-Boltzmann linearization for ionic lines:
                # y* = ln(I λ / gA) - ln(SahaConst * T^(3/2) / n_e)
                # x* = E_k + IP
                #
                # The ionization energy belongs only on the transformed
                # x-axis. Including exp(-IP/T) in the logarithmic y-shift and
                # then also adding IP to x double-counts the ionization energy,
                # which biases the common slope hot and skews composition.
                correction_term = np.log((SAHA_CONST_CM3 / n_e) * (T_eV**1.5))

                # Apply to observations
                for obs in obs_by_element[el]:
                    # Clone obs to avoid modifying original
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

                    # Apply correction if Ion (Stage 2)
                    if obs.ionization_stage == 2:
                        # We hack the intensity to apply the logarithmic shift
                        # y_new = y_old - correction
                        # ln(I_new) = ln(I_old) - correction
                        # I_new = I_old * exp(-correction)
                        new_obs.intensity = obs.intensity * np.exp(-correction_term)
                        # Ionic transition energies in the database are
                        # excitation energies relative to the ionic ground
                        # state. Move them onto the Saha-Boltzmann x-axis by
                        # adding the first ionization potential once.
                        new_obs.E_k_ev = obs.E_k_ev + ips[el]

                    corrected_obs_map[el].append(new_obs)

            # 3. Multi-species Boltzmann Fit
            # Fit common slope

            # Calculate centroids
            centroids = {}
            pooled_x = []
            pooled_y = []
            pooled_w = []

            for el, obs_list in corrected_obs_map.items():
                if len(obs_list) < 2:
                    continue

                # Get valid points
                xs = np.array([o.E_k_ev for o in obs_list])
                ys = np.array([o.y_value for o in obs_list])
                ws = np.array(
                    [1.0 / o.y_uncertainty**2 if o.y_uncertainty > 0 else 1.0 for o in obs_list]
                )

                # Filter invalid
                mask = np.isfinite(ys)
                xs = xs[mask]
                ys = ys[mask]
                ws = ws[mask]

                if len(xs) == 0:
                    continue

                # Centroids
                x_bar = np.average(xs, weights=ws)
                y_bar = np.average(ys, weights=ws)
                centroids[el] = (x_bar, y_bar)

                # Center data
                pooled_x.extend(xs - x_bar)
                pooled_y.extend(ys - y_bar)
                pooled_w.extend(ws)

            if len(pooled_x) < 3:
                logger.warning("Insufficient points for fit")
                break

            # Fit slope through origin
            pooled_x = np.array(pooled_x)
            pooled_y = np.array(pooled_y)
            pooled_w = np.array(pooled_w)

            # Simple linear regression through origin: m = sum(wxy) / sum(wx^2)
            slope = np.sum(pooled_w * pooled_x * pooled_y) / np.sum(pooled_w * pooled_x**2)

            # Update T
            if slope >= 0:
                T_new = 50000.0  # Clamp max
            else:
                T_new = -1.0 / (slope * KB_EV)

            # Damping
            T_K = 0.5 * T_prev + 0.5 * T_new

            # Calculate Intercepts
            intercepts = {}
            for el in centroids:
                x_bar, y_bar = centroids[el]
                q_s = y_bar - slope * x_bar
                intercepts[el] = q_s

            abundance_multipliers = self._compute_abundance_multipliers(
                list(intercepts.keys()),
                T_K,
                n_e,
                partition_funcs,
                partition_funcs_II,
                ips,
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
                S = self._compute_saha_ratio(el, T_K, n_e, U_I, U_II, ips[el])
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

        return CFLIBSResult(
            temperature_K=T_K,
            temperature_uncertainty_K=0.0,  # See solve_with_uncertainty for propagation
            electron_density_cm3=n_e,
            concentrations=concentrations,
            concentration_uncertainties={},  # See solve_with_uncertainty for propagation
            iterations=len(history),
            converged=converged,
            quality_metrics={"r_squared_last": 0.0},
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
        Solve for plasma parameters with full uncertainty propagation.

        Uses the `uncertainties` package to propagate uncertainties through:
        1. Boltzmann fit (with slope-intercept correlation)
        2. Saha correction
        3. Closure equation

        Requires: `pip install uncertainties>=3.2.0` or `pip install cflibs[uncertainty]`

        Parameters
        ----------
        observations : List[LineObservation]
            Spectral lines with intensity uncertainties
        closure_mode : str
            'standard' or 'matrix' (oxide not yet supported)
        closure_kwargs : dict
            Arguments for closure equation (e.g. matrix_element, matrix_fraction)

        Returns
        -------
        CFLIBSResult
            With populated uncertainty fields

        Raises
        ------
        ImportError
            If uncertainties package not installed
        """
        # First run the standard solver to convergence
        result = self.solve(observations, closure_mode, **closure_kwargs)

        # Import uncertainty utilities (will raise ImportError if not available)
        from cflibs.inversion.uncertainty import (
            create_boltzmann_uncertainties,
            propagate_through_closure_standard,
            propagate_through_closure_matrix,
            extract_values_and_uncertainties,
        )
        from uncertainties import ufloat

        # Re-run final Boltzmann fit to get covariance
        # Group observations by element
        obs_by_element = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        elements = list(obs_by_element.keys())

        # Use converged T to get partition functions
        T_K = result.temperature_K

        # Get intercepts with uncertainties by fitting each element
        intercepts_u = {}
        covariances = {}

        for el in elements:
            obs_list = obs_by_element[el]
            if len(obs_list) < 3:
                # Not enough points for covariance
                continue

            fit_result = self.boltzmann_fitter.fit(obs_list)

            if fit_result.covariance_matrix is not None:
                slope_u, intercept_u = create_boltzmann_uncertainties(
                    fit_result.slope,
                    fit_result.intercept,
                    fit_result.covariance_matrix,
                )
                intercepts_u[el] = intercept_u
                covariances[el] = fit_result.covariance_matrix
            else:
                # Fallback to independent uncertainties
                intercepts_u[el] = ufloat(
                    fit_result.intercept,
                    (
                        max(fit_result.intercept_uncertainty, 0.0)
                        if np.isfinite(fit_result.intercept_uncertainty)
                        else 0.0
                    ),
                )

        # Get partition functions at converged T
        partition_funcs = {}
        partition_funcs_II = {}
        ips = {}
        for el in elements:
            partition_funcs[el] = self._evaluate_partition_function(el, 1, T_K)
            partition_funcs_II[el] = self._evaluate_partition_function(el, 2, T_K)
            ip = self.atomic_db.get_ionization_potential(el, 1)
            ips[el] = ip if ip is not None else 15.0

        abundance_multipliers = self._compute_abundance_multipliers(
            list(intercepts_u.keys()),
            T_K,
            result.electron_density_cm3,
            partition_funcs,
            partition_funcs_II,
            ips,
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
        else:
            concentrations_u = propagate_through_closure_standard(
                intercepts_u,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )

        # Extract nominal values and uncertainties
        conc_nominal, conc_uncert = extract_values_and_uncertainties(concentrations_u)

        # Use temperature from first element's fit
        T_err = 0.0
        for el in elements:
            obs_list = obs_by_element[el]
            if len(obs_list) >= 3:
                fit_result = self.boltzmann_fitter.fit(obs_list)
                T_err = fit_result.temperature_uncertainty_K
                break

        return CFLIBSResult(
            temperature_K=result.temperature_K,
            temperature_uncertainty_K=T_err,
            electron_density_cm3=result.electron_density_cm3,
            concentrations=conc_nominal if conc_nominal else result.concentrations,
            concentration_uncertainties=conc_uncert if conc_uncert else {},
            iterations=result.iterations,
            converged=result.converged,
            quality_metrics=result.quality_metrics,
            electron_density_uncertainty_cm3=0.0,  # Would need iterative uncertainty
            boltzmann_covariance=covariances.get(elements[0]) if covariances else None,
        )
