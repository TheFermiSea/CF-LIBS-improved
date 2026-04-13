"""
Column Density Saha-Boltzmann (CD-SB) Plotting for CF-LIBS.

This module implements the CD-SB method for handling resonance lines with
significant self-absorption, which cause standard Boltzmann plots to fail.

Theory
------
Standard Boltzmann plot:
    ln(I*lambda/gA) = ln(F*C/U) - E_k/(k*T)

For optically thin lines, this gives a linear relationship with slope -1/(kT).
However, for optically thick lines (especially resonance lines), the observed
intensity is reduced by self-absorption:

    I_obs = I_true * (1 - exp(-tau)) / tau

where tau is the optical depth at line center.

The CD-SB method modifies the Boltzmann plot ordinate to account for this:
    y_corrected = ln(I*lambda*tau/(gA*(1-exp(-tau)))) = ln(N*L/U) - E_k/(k*T)

where N*L is the column density (number density times path length).

Key Physics
-----------
- Optical depth: tau = alpha * N_i * L
  where alpha is the absorption cross-section and N_i is the lower level population

- For resonance lines (E_i = 0), the lower level is the ground state,
  maximizing self-absorption

- The CD-SB iteration:
  1. Start with initial tau estimate (from line intensities or plasma parameters)
  2. Calculate corrected y-values
  3. Fit Boltzmann plot to get T
  4. Update tau estimates using new T
  5. Repeat until convergence

References
----------
- Aragon, C. & Aguilera, J.A. (2008): CSigma graphs for quantitative LIBS
- El Sherbini et al. (2020): Self-absorption correction methods
- Cowan, R.D. (1981): Theory of Atomic Structure and Spectra
- Aguilera, J.A. & Aragon, C. (2004): Multi-element LIBS analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import numpy as np

from cflibs.core.constants import EV_TO_K
from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import (
    LineObservation,
    BoltzmannFitResult,
    BoltzmannPlotFitter,
    FitMethod,
)

logger = get_logger("inversion.cdsb")


class CDSBConvergenceStatus(Enum):
    """Status of CD-SB iteration convergence."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    UNSTABLE = "unstable"  # tau diverging
    INSUFFICIENT_LINES = "insufficient_lines"


@dataclass
class LineOpticalDepth:
    """Optical depth information for a single line."""

    wavelength_nm: float
    tau_initial: float
    tau_final: float
    is_resonance: bool
    correction_factor: float  # tau / (1 - exp(-tau))
    y_original: float
    y_corrected: float


@dataclass
class CDSBResult:
    """
    Result of Column Density Saha-Boltzmann fitting.

    Attributes
    ----------
    temperature_K : float
        Excitation temperature from CD-SB fit (K)
    temperature_uncertainty_K : float
        1-sigma uncertainty in temperature (K)
    corrected_intercept : float
        Y-intercept after self-absorption correction (related to column density)
    corrected_intercept_uncertainty : float
        Uncertainty in corrected intercept
    column_density : float
        Estimated column density N*L (cm^-2)
    column_density_uncertainty : float
        Uncertainty in column density (cm^-2)
    iterations : int
        Number of CD-SB iterations performed
    convergence_status : CDSBConvergenceStatus
        Status of convergence
    r_squared : float
        Coefficient of determination of final fit
    n_points : int
        Number of points used in final fit
    optical_depths : List[LineOpticalDepth]
        Optical depth information for each line
    uncorrected_result : BoltzmannFitResult
        Result from standard Boltzmann fit (for comparison)
    tau_convergence_history : List[float]
        Mean tau at each iteration (for diagnostics)
    temperature_history : List[float]
        Temperature at each iteration (for diagnostics)
    warnings : List[str]
        Any warnings generated during fitting
    """

    temperature_K: float
    temperature_uncertainty_K: float
    corrected_intercept: float
    corrected_intercept_uncertainty: float
    column_density: float
    column_density_uncertainty: float
    iterations: int
    convergence_status: CDSBConvergenceStatus
    r_squared: float
    n_points: int
    optical_depths: List[LineOpticalDepth] = field(default_factory=list)
    uncorrected_result: Optional[BoltzmannFitResult] = None
    tau_convergence_history: List[float] = field(default_factory=list)
    temperature_history: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CDSBLineObservation(LineObservation):
    """
    Extended line observation with lower level information for CD-SB.

    Additional Attributes
    ---------------------
    E_i_ev : float
        Lower level energy in eV (0 for resonance lines)
    g_i : int
        Lower level statistical weight
    is_resonance : bool
        True if this is a resonance line (ground state transition)
    f_ik : float, optional
        Absorption oscillator strength (calculated from A_ki if not provided)
    """

    E_i_ev: float = 0.0
    g_i: int = 1
    is_resonance: bool = False
    f_ik: Optional[float] = None

    def __post_init__(self):
        """Calculate oscillator strength if not provided."""
        if self.f_ik is None:
            # f_ik = (m_e * c / 8 * pi^2 * e^2) * (g_k / g_i) * lambda^2 * A_ki
            # Simplified formula: f_ik ~ 1.499e-14 * (g_k/g_i) * lambda_nm^2 * A_ki
            self.f_ik = 1.499e-14 * (self.g_k / self.g_i) * self.wavelength_nm**2 * self.A_ki


class CDSBPlotter:
    """
    Column Density Saha-Boltzmann plotter for optically thick plasmas.

    The CD-SB method corrects Boltzmann plots for self-absorption by
    iteratively estimating optical depths and adjusting line intensities.

    This is particularly important for:
    - Resonance lines (ground state transitions)
    - Strong lines at high concentrations
    - Low-temperature plasmas with high ground-state populations

    Examples
    --------
    >>> plotter = CDSBPlotter(plasma_length_cm=0.1)
    >>> observations = [...]  # CDSBLineObservation list
    >>> result = plotter.fit(observations, n_e=1e17, initial_T_K=10000)
    >>> print(f"Temperature: {result.temperature_K:.0f} K")
    >>> print(f"Column density: {result.column_density:.2e} cm^-2")

    Notes
    -----
    The algorithm assumes:
    - Homogeneous plasma along line of sight
    - LTE (local thermodynamic equilibrium)
    - Gaussian line profiles for optical depth calculation
    """

    def __init__(
        self,
        plasma_length_cm: float = 0.1,
        max_iterations: int = 20,
        convergence_tolerance: float = 0.01,
        tau_max: float = 10.0,
        tau_min_correction: float = 0.05,
        fit_method: FitMethod = FitMethod.SIGMA_CLIP,
        outlier_sigma: float = 3.0,
        resonance_weight: float = 0.5,
    ):
        """
        Initialize CD-SB plotter.

        Parameters
        ----------
        plasma_length_cm : float
            Estimated plasma path length in cm (default: 0.1)
        max_iterations : int
            Maximum CD-SB iterations (default: 20)
        convergence_tolerance : float
            Relative change in mean tau for convergence (default: 0.01)
        tau_max : float
            Maximum optical depth to consider (lines with tau > tau_max are masked)
        tau_min_correction : float
            Minimum tau to apply correction (below this, line is optically thin)
        fit_method : FitMethod
            Underlying Boltzmann fit method
        outlier_sigma : float
            Sigma threshold for outlier rejection in underlying fit
        resonance_weight : float
            Weight factor for resonance lines in fit (0-1, lower = downweight).
            Resonance lines have highest self-absorption and may be less reliable.
        """
        self.plasma_length_cm = plasma_length_cm
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.tau_max = tau_max
        self.tau_min_correction = tau_min_correction
        self.fit_method = fit_method
        self.outlier_sigma = outlier_sigma
        self.resonance_weight = resonance_weight

        # Internal Boltzmann fitter
        self._boltzmann_fitter = BoltzmannPlotFitter(
            method=fit_method,
            outlier_sigma=outlier_sigma,
        )

    def fit(
        self,
        observations: List[CDSBLineObservation],
        n_e: float,
        initial_T_K: Optional[float] = None,
        partition_funcs: Optional[Dict[str, float]] = None,
        stark_widths_nm: Optional[Dict[float, float]] = None,
    ) -> CDSBResult:
        """
        Perform CD-SB fitting with iterative self-absorption correction.

        Parameters
        ----------
        observations : List[CDSBLineObservation]
            Line observations with lower level information
        n_e : float
            Electron density in cm^-3 (needed for Stark broadening / optical depth)
        initial_T_K : float, optional
            Initial temperature estimate (K). If None, uses standard Boltzmann fit.
        partition_funcs : Dict[str, float], optional
            Partition functions U(T) by element. If None, uses default U=25.
        stark_widths_nm : Dict[float, float], optional
            Stark widths by wavelength (nm). If None, estimates from n_e.

        Returns
        -------
        CDSBResult
            Fitting result with corrected temperature and column density
        """
        warnings = []

        if len(observations) < 3:
            logger.warning("Fewer than 3 lines provided for CD-SB fit")
            return self._empty_result(
                CDSBConvergenceStatus.INSUFFICIENT_LINES,
                warnings=["Fewer than 3 lines provided"],
            )

        # Convert to base LineObservation for uncorrected fit
        base_observations = [self._to_base_observation(obs) for obs in observations]

        # Uncorrected Boltzmann fit
        uncorrected_result = self._boltzmann_fitter.fit(base_observations)

        if initial_T_K is None:
            if uncorrected_result.temperature_K <= 0 or not np.isfinite(
                uncorrected_result.temperature_K
            ):
                initial_T_K = 10000.0  # Default fallback
                warnings.append("Initial T from standard fit invalid, using 10000 K")
            else:
                initial_T_K = uncorrected_result.temperature_K

        # Initialize partition functions
        if partition_funcs is None:
            partition_funcs = {obs.element: 25.0 for obs in observations}

        # Initialize tau estimates
        current_T_K = initial_T_K
        tau_values = self._estimate_initial_tau(
            observations, current_T_K, n_e, partition_funcs, stark_widths_nm
        )

        # Track convergence history
        tau_history = [np.mean(list(tau_values.values()))]
        temp_history = [current_T_K]

        # Iterative CD-SB correction
        convergence_status = CDSBConvergenceStatus.MAX_ITERATIONS
        n_iterations = 0

        for iteration in range(self.max_iterations):
            n_iterations = iteration + 1

            # Apply tau corrections to get modified y-values
            corrected_obs = self._apply_tau_correction(observations, tau_values)

            if len(corrected_obs) < 2:
                warnings.append(f"Iteration {iteration}: Too few lines after tau masking")
                convergence_status = CDSBConvergenceStatus.INSUFFICIENT_LINES
                break

            # Fit corrected Boltzmann plot
            corrected_result = self._boltzmann_fitter.fit(corrected_obs)

            if corrected_result.temperature_K <= 0 or not np.isfinite(
                corrected_result.temperature_K
            ):
                warnings.append(f"Iteration {iteration}: Invalid temperature from fit")
                convergence_status = CDSBConvergenceStatus.UNSTABLE
                break

            new_T_K = corrected_result.temperature_K
            temp_history.append(new_T_K)

            # Update tau estimates with new temperature
            new_tau_values = self._update_tau_estimates(
                observations, tau_values, current_T_K, new_T_K, n_e, partition_funcs
            )

            # Check for divergence
            mean_new_tau = np.mean(list(new_tau_values.values()))
            if mean_new_tau > 100 * tau_history[0]:
                warnings.append(f"Iteration {iteration}: tau diverging")
                convergence_status = CDSBConvergenceStatus.UNSTABLE
                break

            tau_history.append(mean_new_tau)

            # Check convergence
            if tau_history[-2] > 0:
                rel_change = abs(mean_new_tau - tau_history[-2]) / tau_history[-2]
            else:
                rel_change = abs(mean_new_tau - tau_history[-2])

            if rel_change < self.convergence_tolerance:
                convergence_status = CDSBConvergenceStatus.CONVERGED
                tau_values = new_tau_values
                current_T_K = new_T_K
                break

            tau_values = new_tau_values
            current_T_K = new_T_K

        # Final fit with converged tau values
        final_corrected_obs = self._apply_tau_correction(observations, tau_values)
        if len(final_corrected_obs) < 2:
            return self._empty_result(
                CDSBConvergenceStatus.INSUFFICIENT_LINES,
                uncorrected_result=uncorrected_result,
                warnings=warnings + ["No valid lines after final correction"],
            )

        final_result = self._boltzmann_fitter.fit(final_corrected_obs)

        # Build optical depth information
        optical_depths = self._build_optical_depth_info(
            observations, tau_values, final_corrected_obs
        )

        # Estimate column density from intercept
        # ln(N*L/U) = intercept, so N*L = U * exp(intercept)
        # This is a rough estimate - actual column density depends on element
        mean_U = np.mean(list(partition_funcs.values()))
        column_density = mean_U * np.exp(final_result.intercept)
        column_density_unc = (
            column_density * final_result.intercept_uncertainty
            if np.isfinite(final_result.intercept_uncertainty)
            else 0.0
        )

        return CDSBResult(
            temperature_K=final_result.temperature_K,
            temperature_uncertainty_K=final_result.temperature_uncertainty_K,
            corrected_intercept=final_result.intercept,
            corrected_intercept_uncertainty=final_result.intercept_uncertainty,
            column_density=column_density,
            column_density_uncertainty=column_density_unc,
            iterations=n_iterations,
            convergence_status=convergence_status,
            r_squared=final_result.r_squared,
            n_points=final_result.n_points,
            optical_depths=optical_depths,
            uncorrected_result=uncorrected_result,
            tau_convergence_history=tau_history,
            temperature_history=temp_history,
            warnings=warnings,
        )

    def _estimate_initial_tau(
        self,
        observations: List[CDSBLineObservation],
        T_K: float,
        n_e: float,
        partition_funcs: Dict[str, float],
        stark_widths_nm: Optional[Dict[float, float]],
    ) -> Dict[float, float]:
        """
        Estimate initial optical depths for all lines.

        Uses a simplified empirical formula that scales tau based on:
        - Line strength (gA product)
        - Lower level population
        - Electron density
        - Plasma path length

        The actual optical depth calculation is complex and depends on
        detailed atomic physics. This provides a reasonable order-of-magnitude
        estimate suitable for iterative correction.

        Typical tau values in LIBS plasmas:
        - Optically thin: tau < 0.1
        - Moderate self-absorption: 0.1 < tau < 1
        - Strong self-absorption: tau > 1
        - Very thick (resonance lines): tau ~ 2-5
        """
        tau_values = {}
        T_eV = T_K / EV_TO_K

        # Reference values for scaling
        n_e_ref = 1e17  # Reference electron density (cm^-3)
        L_ref = 0.1  # Reference path length (cm)
        gA_ref = 1e8  # Reference gA value (strong line)

        for obs in observations:
            # Lower level population (Boltzmann)
            U = partition_funcs.get(obs.element, 25.0)
            if U <= 0 or T_eV <= 0:
                tau_values[obs.wavelength_nm] = 0.0
                continue

            # Population factor: (g_i / U) * exp(-E_i / kT)
            # This is largest for resonance lines (E_i = 0)
            exp_factor = np.exp(-obs.E_i_ev / T_eV) if T_eV > 0 else 0.0
            population_factor = (obs.g_i / U) * exp_factor

            # Line strength factor: proportional to gA
            # Strong lines (high gA) have higher absorption
            gA_product = obs.g_i * obs.A_ki  # Use g_i for absorption
            line_strength = gA_product / gA_ref

            # Density scaling (linear with n_e at moderate densities)
            density_factor = n_e / n_e_ref

            # Path length scaling (linear)
            length_factor = self.plasma_length_cm / L_ref

            # Empirical tau estimate
            # Base tau ~ 0.5 for a strong resonance line at reference conditions
            base_tau = 0.5

            tau = base_tau * population_factor * line_strength * density_factor * length_factor

            # Resonance lines (E_i ~ 0) naturally get higher tau through population_factor
            # Add modest boost for explicitly flagged resonance lines
            if obs.is_resonance:
                tau *= 1.5

            # Clamp to physically reasonable range
            tau_values[obs.wavelength_nm] = max(0.0, min(tau, 10.0))

        return tau_values

    def _update_tau_estimates(
        self,
        observations: List[CDSBLineObservation],
        old_tau: Dict[float, float],
        old_T_K: float,
        new_T_K: float,
        n_e: float,
        partition_funcs: Dict[str, float],
    ) -> Dict[float, float]:
        """
        Update tau estimates based on new temperature.

        Tau scales with lower level population:
        tau_new / tau_old = n_i(T_new) / n_i(T_old)
                         = (U(T_old)/U(T_new)) * exp((E_i/k) * (1/T_new - 1/T_old))

        For ground state (E_i=0), this simplifies to just the partition function ratio.
        """
        new_tau = {}
        old_T_eV = old_T_K / EV_TO_K
        new_T_eV = new_T_K / EV_TO_K

        for obs in observations:
            wl = obs.wavelength_nm
            if wl not in old_tau:
                new_tau[wl] = 0.0
                continue

            # Population ratio scaling
            # For partition function: approximate as constant or use power law
            # U(T) ~ T^0.5 for many atoms, but let's use provided values
            U_old = partition_funcs.get(obs.element, 25.0)
            U_new = partition_funcs.get(obs.element, 25.0)  # Same, could interpolate

            if old_T_eV > 0 and new_T_eV > 0:
                # Boltzmann factor ratio
                delta_E_factor = np.exp(obs.E_i_ev * (1.0 / old_T_eV - 1.0 / new_T_eV))
                U_ratio = U_old / U_new if U_new > 0 else 1.0
                scale_factor = U_ratio * delta_E_factor
            else:
                scale_factor = 1.0

            new_tau[wl] = old_tau[wl] * scale_factor

        return new_tau

    def _apply_tau_correction(
        self,
        observations: List[CDSBLineObservation],
        tau_values: Dict[float, float],
    ) -> List[LineObservation]:
        """
        Apply optical depth correction to line observations.

        The corrected y-value becomes:
        y_corrected = ln(I * lambda * tau / (g * A * (1 - exp(-tau))))
                    = y_original + ln(tau / (1 - exp(-tau)))

        For tau -> 0: correction -> 0 (optically thin limit)
        For tau >> 1: correction -> ln(tau) (optically thick limit)
        """
        corrected_obs = []

        for obs in observations:
            tau = tau_values.get(obs.wavelength_nm, 0.0)

            # Skip lines that are too optically thick
            if tau > self.tau_max:
                logger.debug(
                    f"Line {obs.wavelength_nm:.2f} nm masked: tau={tau:.2f} > {self.tau_max}"
                )
                continue

            # Calculate correction factor
            if tau < self.tau_min_correction:
                # Optically thin - no correction needed
                correction_factor = 1.0
            elif tau < 1e-6:
                # Very small tau: (1 - exp(-tau))/tau -> 1 - tau/2
                correction_factor = 1.0 / (1.0 - tau / 2.0)
            elif tau > 50:
                # Large tau: (1 - exp(-tau))/tau -> 1/tau
                correction_factor = tau
            else:
                # General case
                correction_factor = tau / (1.0 - np.exp(-tau))

            # Create corrected observation
            # Effectively multiply intensity by correction factor
            corrected_intensity = obs.intensity * correction_factor
            corrected_uncertainty = obs.intensity_uncertainty * correction_factor

            # Weight resonance lines less if configured
            if obs.is_resonance and self.resonance_weight < 1.0:
                # Increase uncertainty to downweight in fit
                corrected_uncertainty /= self.resonance_weight

            corrected_obs.append(
                LineObservation(
                    wavelength_nm=obs.wavelength_nm,
                    intensity=corrected_intensity,
                    intensity_uncertainty=corrected_uncertainty,
                    element=obs.element,
                    ionization_stage=obs.ionization_stage,
                    E_k_ev=obs.E_k_ev,
                    g_k=obs.g_k,
                    A_ki=obs.A_ki,
                )
            )

        return corrected_obs

    def _build_optical_depth_info(
        self,
        observations: List[CDSBLineObservation],
        tau_values: Dict[float, float],
        corrected_obs: List[LineObservation],
    ) -> List[LineOpticalDepth]:
        """Build optical depth information for result."""
        optical_depths = []

        # Create lookup for corrected observations
        corrected_lookup = {obs.wavelength_nm: obs for obs in corrected_obs}

        for obs in observations:
            wl = obs.wavelength_nm
            tau = tau_values.get(wl, 0.0)

            # Original y-value
            base_obs = self._to_base_observation(obs)
            y_original = base_obs.y_value

            # Corrected y-value (if line was included)
            if wl in corrected_lookup:
                y_corrected = corrected_lookup[wl].y_value
            else:
                y_corrected = np.nan  # Masked

            # Correction factor
            if tau < 1e-6:
                correction_factor = 1.0
            elif tau > 50:
                correction_factor = tau
            else:
                correction_factor = tau / (1.0 - np.exp(-tau))

            optical_depths.append(
                LineOpticalDepth(
                    wavelength_nm=wl,
                    tau_initial=tau,  # We don't track initial vs final separately here
                    tau_final=tau,
                    is_resonance=obs.is_resonance,
                    correction_factor=correction_factor,
                    y_original=y_original,
                    y_corrected=y_corrected,
                )
            )

        return optical_depths

    def _to_base_observation(self, obs: CDSBLineObservation) -> LineObservation:
        """Convert CDSBLineObservation to base LineObservation."""
        return LineObservation(
            wavelength_nm=obs.wavelength_nm,
            intensity=obs.intensity,
            intensity_uncertainty=obs.intensity_uncertainty,
            element=obs.element,
            ionization_stage=obs.ionization_stage,
            E_k_ev=obs.E_k_ev,
            g_k=obs.g_k,
            A_ki=obs.A_ki,
        )

    def _empty_result(
        self,
        status: CDSBConvergenceStatus,
        uncorrected_result: Optional[BoltzmannFitResult] = None,
        warnings: Optional[List[str]] = None,
    ) -> CDSBResult:
        """Return empty result for failed fits."""
        return CDSBResult(
            temperature_K=0.0,
            temperature_uncertainty_K=0.0,
            corrected_intercept=0.0,
            corrected_intercept_uncertainty=0.0,
            column_density=0.0,
            column_density_uncertainty=0.0,
            iterations=0,
            convergence_status=status,
            r_squared=0.0,
            n_points=0,
            optical_depths=[],
            uncorrected_result=uncorrected_result,
            tau_convergence_history=[],
            temperature_history=[],
            warnings=warnings or [],
        )


def create_cdsb_observation(
    wavelength_nm: float,
    intensity: float,
    intensity_uncertainty: float,
    element: str,
    ionization_stage: int,
    E_k_ev: float,
    E_i_ev: float,
    g_k: int,
    g_i: int,
    A_ki: float,
    is_resonance: Optional[bool] = None,
) -> CDSBLineObservation:
    """
    Factory function to create a CDSBLineObservation.

    Parameters
    ----------
    wavelength_nm : float
        Transition wavelength in nm
    intensity : float
        Measured line intensity (integrated area)
    intensity_uncertainty : float
        Uncertainty in intensity measurement
    element : str
        Element symbol (e.g., 'Fe', 'Ca')
    ionization_stage : int
        Ionization stage (1=neutral, 2=singly ionized)
    E_k_ev : float
        Upper level energy in eV
    E_i_ev : float
        Lower level energy in eV
    g_k : int
        Upper level statistical weight
    g_i : int
        Lower level statistical weight
    A_ki : float
        Einstein A coefficient (spontaneous emission rate) in s^-1
    is_resonance : bool, optional
        Whether this is a resonance line. If None, auto-detected from E_i_ev.

    Returns
    -------
    CDSBLineObservation
    """
    if is_resonance is None:
        # Auto-detect: resonance if lower level is near ground state
        is_resonance = E_i_ev < 0.1  # Within 0.1 eV of ground

    return CDSBLineObservation(
        wavelength_nm=wavelength_nm,
        intensity=intensity,
        intensity_uncertainty=intensity_uncertainty,
        element=element,
        ionization_stage=ionization_stage,
        E_k_ev=E_k_ev,
        g_k=g_k,
        A_ki=A_ki,
        E_i_ev=E_i_ev,
        g_i=g_i,
        is_resonance=is_resonance,
    )


def from_transition(
    transition,  # Transition dataclass from cflibs.atomic.structures
    intensity: float,
    intensity_uncertainty: float,
) -> CDSBLineObservation:
    """
    Create CDSBLineObservation from a Transition and measured intensity.

    Parameters
    ----------
    transition : Transition
        Transition dataclass from atomic database
    intensity : float
        Measured line intensity
    intensity_uncertainty : float
        Intensity uncertainty

    Returns
    -------
    CDSBLineObservation
    """
    is_resonance = getattr(transition, "is_resonance", None)
    if is_resonance is None:
        is_resonance = transition.E_i_ev < 0.1

    return CDSBLineObservation(
        wavelength_nm=transition.wavelength_nm,
        intensity=intensity,
        intensity_uncertainty=intensity_uncertainty,
        element=transition.element,
        ionization_stage=transition.ionization_stage,
        E_k_ev=transition.E_k_ev,
        g_k=transition.g_k,
        A_ki=transition.A_ki,
        E_i_ev=transition.E_i_ev,
        g_i=transition.g_i,
        is_resonance=is_resonance,
    )
