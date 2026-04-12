"""
Self-absorption correction for CF-LIBS analysis.

Self-absorption occurs when strong emission lines are partially reabsorbed
by cooler atoms in the outer plasma regions, leading to intensity underestimation.
This module implements correction algorithms based on curve-of-growth methods.

Theory
------
For an optically thick plasma, the observed line intensity is reduced by a
factor f(τ) that depends on the optical depth at line center (τ₀):

    I_observed = I_true × f(τ₀)

For a Gaussian line profile:
    f(τ) = (1 - exp(-τ)) / τ

Limiting behaviors:
    - Optically thin (τ << 1): f(τ) ≈ 1 - τ/2 → 1
    - Moderate (τ = 1): f(τ) ≈ 0.632
    - Optically thick (τ >> 1): f(τ) ≈ 1/τ

Correction Process
------------------
1. Estimate τ₀ from plasma parameters (T, n_e, concentrations) or intensity ratios
2. Calculate correction factor: C = 1/f(τ₀)
3. Apply: I_true = I_observed × C

Literature References
---------------------
- El Sherbini et al. (2020): Curve-of-growth self-absorption correction
- Sun & Yu (2021): Automatic self-absorption correction algorithm
- Bredice et al. (2006): Self-absorption evaluation methods
- Aragon & Aguilera (2008): Review of plasma characterization methods

Algorithm Limitations
---------------------
1. **Homogeneous plasma assumption**: Algorithm assumes uniform temperature and
   density along the line of sight. Real LIBS plasmas have gradients.

2. **LTE assumption**: Optical depth calculation assumes LTE populations.
   Non-LTE effects may cause errors for early-time or low-density plasmas.

3. **Gaussian profile assumption**: The f(τ) formula assumes Gaussian lineshapes.
   Stark-dominated profiles (Lorentzian) have different saturation behavior.

4. **Single-zone model**: Does not account for inhomogeneous absorption where
   cooler outer regions absorb emission from hotter inner regions.

5. **Estimation uncertainty**: Optical depth estimation requires accurate
   partition functions, oscillator strengths, and plasma parameters.

6. **High optical depth limit**: For τ > 3, corrections become very large
   (>3x) and increasingly uncertain. Masking is recommended.

Recommended Usage
-----------------
1. Use primarily for resonance lines and strong transitions
2. Validate with doublet/multiplet intensity ratio methods when possible
3. Set mask_threshold appropriately (default 3.0) to exclude highly absorbed lines
4. Compare results with and without correction to assess impact
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np

from cflibs.core.constants import C_LIGHT, EV_TO_K, KB, KB_EV, M_PROTON
from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation

logger = get_logger("inversion.self_absorption")


def _escape_factor(tau: float) -> float:
    """Photon escape factor f(tau) = (1 - exp(-tau)) / tau."""
    if tau < 1e-10:
        return 1.0
    elif tau > 50:
        return 1.0 / tau
    else:
        return (1.0 - np.exp(-tau)) / tau


class COGRegime(Enum):
    """
    Regime of the curve-of-growth.

    The COG has three distinct regimes based on optical depth:

    - LINEAR: Optically thin, W proportional to Nf (tau << 1)
    - SATURATION: Doppler core saturated, W proportional to sqrt(ln(Nf)) (tau ~ 1-10)
    - DAMPING: Lorentzian wings dominate, W proportional to sqrt(Nf) (tau >> 10)
    """

    LINEAR = "linear"
    SATURATION = "saturation"
    DAMPING = "damping"
    UNKNOWN = "unknown"


@dataclass
class AbsorptionCorrectionResult:
    """Result of self-absorption correction for a single line."""

    original_intensity: float
    corrected_intensity: float
    optical_depth: float
    correction_factor: float
    is_optically_thick: bool
    iterations: int = 1


@dataclass
class SelfAbsorptionResult:
    """Result of self-absorption correction for all lines."""

    corrected_observations: List[LineObservation]
    masked_observations: List[LineObservation]
    corrections: Dict[float, AbsorptionCorrectionResult]  # wavelength -> result
    n_corrected: int
    n_masked: int
    max_optical_depth: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class COGLineData:
    """
    Data for a single line in curve-of-growth analysis.

    This stores the measured and derived quantities needed to place
    a line on the COG diagram.

    Attributes
    ----------
    wavelength_nm : float
        Transition wavelength in nm
    equivalent_width_nm : float
        Measured equivalent width W in nm
    log_gf : float
        Log of oscillator strength times statistical weight: log10(g*f)
    reduced_width : float
        Reduced equivalent width: W / lambda
    log_reduced_width : float
        Log10 of reduced equivalent width
    E_i_ev : float
        Lower level energy in eV (for excitation correction)
    element : str
        Element symbol
    ionization_stage : int
        Ionization stage (1=neutral, 2=singly ionized)
    """

    wavelength_nm: float
    equivalent_width_nm: float
    log_gf: float
    reduced_width: float
    log_reduced_width: float
    E_i_ev: float
    element: str
    ionization_stage: int


@dataclass
class COGResult:
    """
    Result of curve-of-growth analysis for a multiplet or set of lines.

    The COG method plots reduced equivalent width (W/lambda) against
    oscillator strength (gf) to diagnose optical depth effects.

    Physics Background
    ------------------
    The curve-of-growth relates equivalent width W to column density N:

    Linear regime (tau << 1):
        W/lambda = (pi * e^2 / m_e * c^2) * N * f * lambda

    Saturation regime (tau ~ 1-10):
        W/lambda propto sqrt(ln(N * f))

    Damping regime (tau >> 10):
        W/lambda propto sqrt(N * f * gamma)

    where gamma is the damping constant (natural + collisional).

    Attributes
    ----------
    regime : COGRegime
        Identified regime (LINEAR, SATURATION, DAMPING, or UNKNOWN)
    optical_depth_estimate : float
        Estimated optical depth at line center for the strongest line
    column_density_cm2 : float
        Derived column density N in cm^-2 (if determinable)
    doppler_width_nm : float
        Fitted or assumed Doppler width in nm
    damping_parameter : float
        Fitted damping parameter a = gamma / (4 * pi * delta_nu_D)
    fit_slope : float
        Slope of log(W/lambda) vs log(gf) fit
    fit_intercept : float
        Intercept of the fit
    fit_r_squared : float
        Coefficient of determination for the fit
    fit_residuals : np.ndarray
        Residuals from the fit for each line
    n_lines_used : int
        Number of lines used in the analysis
    lines_data : List[COGLineData]
        Individual line data used in the analysis
    warnings : List[str]
        Any warnings generated during analysis
    """

    regime: COGRegime
    optical_depth_estimate: float
    column_density_cm2: float
    doppler_width_nm: float
    damping_parameter: float
    fit_slope: float
    fit_intercept: float
    fit_r_squared: float
    fit_residuals: np.ndarray
    n_lines_used: int
    lines_data: List[COGLineData]
    warnings: List[str] = field(default_factory=list)


class SelfAbsorptionCorrector:
    """
    Corrects for self-absorption in optically thick plasmas.

    Uses the curve-of-growth approach:
    1. Estimate optical depth τ₀ at line center
    2. Apply correction factor: I_true = I_measured / f(τ)
    3. Where f(τ) = (1 - exp(-τ)) / τ (for Gaussian profile)

    For very high optical depth, lines can be masked instead of corrected.
    """

    def __init__(
        self,
        optical_depth_threshold: float = 0.1,
        mask_threshold: float = 3.0,
        max_iterations: int = 5,
        convergence_tolerance: float = 0.01,
        plasma_length_cm: float = 0.1,
    ):
        """
        Initialize corrector.

        Parameters
        ----------
        optical_depth_threshold : float
            Minimum τ₀ to apply correction (below this, line is optically thin)
        mask_threshold : float
            τ₀ above which to mask line instead of correct
        max_iterations : int
            Maximum iterations for recursive correction
        convergence_tolerance : float
            Relative change threshold for convergence
        plasma_length_cm : float
            Estimated plasma depth (path length) in cm
        """
        self.optical_depth_threshold = optical_depth_threshold
        self.mask_threshold = mask_threshold
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.plasma_length_cm = plasma_length_cm

    def correct(
        self,
        observations: List[LineObservation],
        temperature_K: float,
        concentrations: Dict[str, float],
        total_number_density_cm3: float,
        partition_funcs: Dict[str, float],
        lower_level_energies: Optional[Dict[float, float]] = None,
    ) -> SelfAbsorptionResult:
        """
        Apply self-absorption correction to line observations.

        Parameters
        ----------
        observations : List[LineObservation]
            Line observations to correct
        temperature_K : float
            Plasma temperature
        concentrations : Dict[str, float]
            Elemental concentrations (mass fractions)
        total_number_density_cm3 : float
            Total number density of plasma
        partition_funcs : Dict[str, float]
            Partition functions U(T) for each element
        lower_level_energies : Dict[float, float], optional
            Lower level energies E_i by wavelength (nm -> eV)

        Returns
        -------
        SelfAbsorptionResult
        """
        if lower_level_energies is None:
            lower_level_energies = {}

        warnings = []
        corrected_obs = []
        masked_obs = []
        corrections = {}
        max_tau = 0.0

        for obs in observations:
            # Get lower level energy (default to 0 = ground state, worst case)
            E_i_ev = lower_level_energies.get(obs.wavelength_nm, 0.0)

            # Estimate optical depth
            tau = self._estimate_optical_depth(
                obs,
                temperature_K,
                concentrations,
                total_number_density_cm3,
                partition_funcs,
                E_i_ev,
            )

            max_tau = max(max_tau, tau)

            if tau > self.mask_threshold:
                # Too optically thick - mask this line
                masked_obs.append(obs)
                corrections[obs.wavelength_nm] = AbsorptionCorrectionResult(
                    original_intensity=obs.intensity,
                    corrected_intensity=0.0,
                    optical_depth=tau,
                    correction_factor=0.0,
                    is_optically_thick=True,
                )
                warnings.append(
                    f"Line {obs.wavelength_nm:.2f} nm masked: τ={tau:.2f} > {self.mask_threshold}"
                )

            elif tau > self.optical_depth_threshold:
                # Apply correction
                result = self._apply_recursive_correction(obs, tau)
                corrected_obs.append(self._create_corrected_observation(obs, result))
                corrections[obs.wavelength_nm] = result

            else:
                # Optically thin - no correction needed
                corrected_obs.append(obs)
                corrections[obs.wavelength_nm] = AbsorptionCorrectionResult(
                    original_intensity=obs.intensity,
                    corrected_intensity=obs.intensity,
                    optical_depth=tau,
                    correction_factor=1.0,
                    is_optically_thick=False,
                )

        return SelfAbsorptionResult(
            corrected_observations=corrected_obs,
            masked_observations=masked_obs,
            corrections=corrections,
            n_corrected=len([c for c in corrections.values() if c.correction_factor != 1.0]),
            n_masked=len(masked_obs),
            max_optical_depth=max_tau,
            warnings=warnings,
        )

    def _estimate_optical_depth(
        self,
        obs: LineObservation,
        temperature_K: float,
        concentrations: Dict[str, float],
        total_n_cm3: float,
        partition_funcs: Dict[str, float],
        E_i_ev: float,
    ) -> float:
        """
        Estimate optical depth at line center.

        Uses:
        τ₀ = (π e² / m_e c) × f_ik × n_i × L × φ(ν₀)

        Simplified to:
        τ₀ ≈ B × (g_k A_ki λ³ / 8π) × (n_s / U(T)) × exp(-E_i/kT) × L

        Where B is a constant including physical constants.
        """
        element = obs.element
        C_s = concentrations.get(element, 0.0)
        U_T = partition_funcs.get(element, 25.0)

        if C_s <= 0 or U_T <= 0:
            return 0.0

        # Species number density
        n_s = C_s * total_n_cm3

        # Lower level population (Boltzmann)
        T_eV = temperature_K / EV_TO_K
        if T_eV <= 0:
            return 0.0

        # Statistical weight of lower level (approximate as g_k for now)
        g_i = obs.g_k  # Should be lower level g, but often similar order

        exp_factor = np.exp(-E_i_ev / T_eV)
        n_i = n_s * (g_i / U_T) * exp_factor

        # Wavelength in cm
        lambda_cm = obs.wavelength_nm * 1e-7

        # Absorption oscillator strength from A_ki
        # f_ik ≈ (m_e c / 8π² e²) × (g_k/g_i) × λ² × A_ki
        # Simplified: use A_ki directly with scaling

        # Optical depth estimate (order of magnitude)
        # τ ≈ σ × n_i × L
        # σ ≈ (π e² / m_e c) × f × φ(ν) ≈ 10^-12 cm² (typical)

        # Using simpler scaling based on Einstein A coefficient:
        # τ ∝ A_ki × λ³ × n_i × L
        SCALE_FACTOR = 1e-25  # Empirical scaling to get reasonable τ values

        tau = SCALE_FACTOR * obs.A_ki * (lambda_cm**3) * n_i * self.plasma_length_cm

        return max(0.0, tau)

    def _apply_recursive_correction(
        self,
        obs: LineObservation,
        tau_initial: float,
    ) -> AbsorptionCorrectionResult:
        """
        Apply recursive self-absorption correction.

        The correction factor for a Gaussian line profile is:
        f(τ) = (1 - exp(-τ)) / τ

        I_true = I_measured / f(τ)
        """
        tau = tau_initial
        I_corrected = obs.intensity
        iteration = 0

        for iteration in range(self.max_iterations):
            # Correction factor
            f_tau = _escape_factor(tau)

            # Corrected intensity
            I_new = obs.intensity / f_tau

            # Check convergence
            if I_corrected > 0:
                rel_change = abs(I_new - I_corrected) / I_corrected
                if rel_change < self.convergence_tolerance:
                    I_corrected = I_new
                    break

            I_corrected = I_new

            # Update tau for next iteration (intensity affects population estimate)
            # This is a simplification - full iteration would recalculate τ
            # based on updated concentrations from the solver
            tau = tau * (I_new / obs.intensity) if obs.intensity > 0 else tau

        return AbsorptionCorrectionResult(
            original_intensity=obs.intensity,
            corrected_intensity=I_corrected,
            optical_depth=tau_initial,
            correction_factor=obs.intensity / I_corrected if I_corrected > 0 else 0.0,
            is_optically_thick=tau_initial > 1.0,
            iterations=iteration + 1,
        )

    def _create_corrected_observation(
        self,
        obs: LineObservation,
        result: AbsorptionCorrectionResult,
    ) -> LineObservation:
        """Create a new observation with corrected intensity."""
        return LineObservation(
            wavelength_nm=obs.wavelength_nm,
            intensity=result.corrected_intensity,
            intensity_uncertainty=(
                obs.intensity_uncertainty / result.correction_factor
                if result.correction_factor > 0
                else obs.intensity_uncertainty
            ),
            element=obs.element,
            ionization_stage=obs.ionization_stage,
            E_k_ev=obs.E_k_ev,
            g_k=obs.g_k,
            A_ki=obs.A_ki,
        )

    def correct_with_cog(
        self,
        observations: List[LineObservation],
        equivalent_widths: Dict[float, float],
        lower_level_g: Dict[float, int],
        lower_level_energies: Dict[float, float],
        temperature_K: float,
        mass_amu: float = 56.0,
    ) -> SelfAbsorptionResult:
        """
        Apply self-absorption correction using Curve-of-Growth analysis.

        This method uses the COG approach to diagnose and correct for
        self-absorption across a multiplet. It is more physically rigorous
        than the simple optical depth estimation for related lines.

        Parameters
        ----------
        observations : List[LineObservation]
            Line observations to correct (must have at least 3 lines)
        equivalent_widths : Dict[float, float]
            Equivalent widths by wavelength (nm -> nm)
        lower_level_g : Dict[float, int]
            Lower level statistical weights by wavelength
        lower_level_energies : Dict[float, float]
            Lower level energies by wavelength (nm -> eV)
        temperature_K : float
            Plasma temperature for excitation correction
        mass_amu : float
            Atomic mass in amu (default 56 for Fe)

        Returns
        -------
        SelfAbsorptionResult
            Corrected observations with COG-derived optical depths

        Notes
        -----
        The COG method works best when:
        - Multiple lines from the same species are available
        - Lines span a range of oscillator strengths
        - Equivalent widths are accurately measured

        For isolated lines or single-line analysis, use the standard
        correct() method instead.
        """
        warnings_list: List[str] = []

        # Build MultipletLine objects from observations
        multiplet_lines: List["MultipletLine"] = []

        for obs in observations:
            wl = obs.wavelength_nm
            if wl not in equivalent_widths:
                warnings_list.append(f"No EW for {wl:.2f} nm, skipping")
                continue

            ew = equivalent_widths[wl]
            g_i = lower_level_g.get(wl, obs.g_k)  # Default to upper if not provided
            E_i = lower_level_energies.get(wl, 0.0)

            line = MultipletLine(
                wavelength_nm=wl,
                equivalent_width_nm=ew,
                g_i=g_i,
                g_k=obs.g_k,
                A_ki=obs.A_ki,
                E_i_ev=E_i,
                E_k_ev=obs.E_k_ev,
                element=obs.element,
                ionization_stage=obs.ionization_stage,
            )
            multiplet_lines.append(line)

        # Need at least 3 lines for COG analysis
        if len(multiplet_lines) < 3:
            fallback_warning = (
                f"Only {len(multiplet_lines)} lines available, falling back to standard correction"
            )
            warnings_list.append(fallback_warning)
            result = self.correct(
                observations,
                temperature_K,
                concentrations={},
                total_number_density_cm3=1e18,
                partition_funcs={},
                lower_level_energies=lower_level_energies,
            )
            # Merge warnings from fallback with our warnings
            return SelfAbsorptionResult(
                corrected_observations=result.corrected_observations,
                masked_observations=result.masked_observations,
                corrections=result.corrections,
                n_corrected=result.n_corrected,
                n_masked=result.n_masked,
                max_optical_depth=result.max_optical_depth,
                warnings=warnings_list + result.warnings,
            )

        # Perform COG analysis
        analyzer = CurveOfGrowthAnalyzer(
            temperature_K=temperature_K,
            mass_amu=mass_amu,
            min_lines=3,
        )

        try:
            cog_result = analyzer.fit(multiplet_lines, excitation_correct=True)
        except ValueError as e:
            warnings_list.append(f"COG analysis failed: {e}")
            return self.correct(
                observations,
                temperature_K,
                concentrations={},
                total_number_density_cm3=1e18,
                partition_funcs={},
                lower_level_energies=lower_level_energies,
            )

        # Get correction factors from COG
        cog_corrections = analyzer.get_correction_factors(cog_result)

        # Apply corrections
        corrected_obs = []
        masked_obs = []
        corrections = {}
        max_tau = cog_result.optical_depth_estimate

        # Map wavelength to observation
        obs_by_wl = {obs.wavelength_nm: obs for obs in observations}

        for line_data in cog_result.lines_data:
            wl = line_data.wavelength_nm
            if wl not in obs_by_wl:
                continue

            obs = obs_by_wl[wl]
            correction_factor = cog_corrections.get(wl, 1.0)

            # Estimate optical depth for this line
            max_log_gf = max(ld.log_gf for ld in cog_result.lines_data)
            delta_log_gf = line_data.log_gf - max_log_gf
            tau_i = max_tau * (10**delta_log_gf)

            if tau_i > self.mask_threshold:
                masked_obs.append(obs)
                corrections[wl] = AbsorptionCorrectionResult(
                    original_intensity=obs.intensity,
                    corrected_intensity=0.0,
                    optical_depth=tau_i,
                    correction_factor=0.0,
                    is_optically_thick=True,
                )
                warnings_list.append(f"Line {wl:.2f} nm masked (COG): tau={tau_i:.2f}")
            elif tau_i > self.optical_depth_threshold:
                corrected_intensity = obs.intensity * correction_factor
                corrected_obs.append(
                    LineObservation(
                        wavelength_nm=wl,
                        intensity=corrected_intensity,
                        intensity_uncertainty=obs.intensity_uncertainty * correction_factor,
                        element=obs.element,
                        ionization_stage=obs.ionization_stage,
                        E_k_ev=obs.E_k_ev,
                        g_k=obs.g_k,
                        A_ki=obs.A_ki,
                    )
                )
                corrections[wl] = AbsorptionCorrectionResult(
                    original_intensity=obs.intensity,
                    corrected_intensity=corrected_intensity,
                    optical_depth=tau_i,
                    correction_factor=1.0 / correction_factor,
                    is_optically_thick=tau_i > 1.0,
                )
            else:
                corrected_obs.append(obs)
                corrections[wl] = AbsorptionCorrectionResult(
                    original_intensity=obs.intensity,
                    corrected_intensity=obs.intensity,
                    optical_depth=tau_i,
                    correction_factor=1.0,
                    is_optically_thick=False,
                )

        # Add warnings from COG analysis
        warnings_list.extend(cog_result.warnings)
        warnings_list.append(f"COG regime: {cog_result.regime.value}")
        warnings_list.append(f"COG fit R^2: {cog_result.fit_r_squared:.3f}")

        return SelfAbsorptionResult(
            corrected_observations=corrected_obs,
            masked_observations=masked_obs,
            corrections=corrections,
            n_corrected=len([c for c in corrections.values() if c.correction_factor != 1.0]),
            n_masked=len(masked_obs),
            max_optical_depth=max_tau,
            warnings=warnings_list,
        )


def estimate_optical_depth_from_intensity_ratio(
    intensity_strong: float,
    intensity_weak: float,
    theoretical_ratio: float,
) -> float:
    """
    Estimate optical depth from intensity ratio of doublet lines.

    For two lines from the same upper level with theoretical ratio R_0:
    R_observed / R_0 = f(τ_strong) / f(τ_weak)

    This is commonly used with doublets where the stronger line
    may be self-absorbed while the weaker line remains optically thin.

    Parameters
    ----------
    intensity_strong : float
        Measured intensity of stronger line
    intensity_weak : float
        Measured intensity of weaker line
    theoretical_ratio : float
        Theoretical intensity ratio (from A*g values)

    Returns
    -------
    float
        Estimated optical depth of stronger line
    """
    if intensity_weak <= 0 or theoretical_ratio <= 0:
        return 0.0

    observed_ratio = intensity_strong / intensity_weak
    ratio_reduction = observed_ratio / theoretical_ratio

    if ratio_reduction >= 1.0:
        # No absorption detected
        return 0.0

    # Solve (1 - exp(-τ))/τ = ratio_reduction
    # Use Newton-Raphson or bisection

    def f_tau(tau: float) -> float:
        if tau < 1e-10:
            return 1.0
        return (1.0 - np.exp(-tau)) / tau

    # Bisection search
    tau_low, tau_high = 0.0, 10.0
    tau_mid = (tau_low + tau_high) / 2

    for _ in range(50):
        tau_mid = (tau_low + tau_high) / 2
        f_mid = f_tau(tau_mid)

        if f_mid > ratio_reduction:
            tau_low = tau_mid
        else:
            tau_high = tau_mid

        if tau_high - tau_low < 0.01:
            break

    return tau_mid


@dataclass
class MultipletLine:
    """
    Represents a line within a multiplet for COG analysis.

    This is used as input to the CurveOfGrowthAnalyzer. All lines in
    a multiplet should share the same element and ionization stage.

    Attributes
    ----------
    wavelength_nm : float
        Transition wavelength in nm
    equivalent_width_nm : float
        Measured equivalent width (integrated line intensity / continuum)
    g_i : int
        Statistical weight of lower level
    g_k : int
        Statistical weight of upper level
    A_ki : float
        Einstein A coefficient (spontaneous emission rate) in s^-1
    E_i_ev : float
        Lower level energy in eV
    E_k_ev : float
        Upper level energy in eV
    element : str
        Element symbol
    ionization_stage : int
        Ionization stage (1=neutral, 2=singly ionized)
    intensity_uncertainty : float, optional
        Uncertainty in equivalent width measurement
    """

    wavelength_nm: float
    equivalent_width_nm: float
    g_i: int
    g_k: int
    A_ki: float
    E_i_ev: float
    E_k_ev: float
    element: str
    ionization_stage: int
    intensity_uncertainty: float = 0.0

    @property
    def oscillator_strength(self) -> float:
        """
        Calculate absorption oscillator strength f_ik from Einstein A coefficient.

        The relation is:
            A_ki = (8 pi^2 e^2 / m_e c lambda^2) * (g_i / g_k) * f_ik

        Rearranging:
            f_ik = (m_e c / 8 pi^2 e^2) * (g_k / g_i) * lambda^2 * A_ki

        In Gaussian CGS (lambda in cm):
            f_ik = 1.4992 * (g_k / g_i) * lambda[cm]^2 * A_ki[s^-1]

        Note: the prefactor m_e*c/(8*pi^2*e^2) evaluates to 1.4992 in CGS
        with lambda in cm. The value 1.4992e-16 sometimes seen in literature
        is for lambda in Angstroms (1 cm = 1e8 Angstrom, squared = 1e16).
        """
        lambda_cm = self.wavelength_nm * 1e-7
        # Prefactor: m_e * c / (8 * pi^2 * e^2) = 1.4992 [CGS, lambda in cm]
        # Verified: m_e=9.109e-28g, c=2.998e10cm/s, e=4.803e-10esu
        f_ik = 1.4992 * (lambda_cm**2) * self.A_ki * (self.g_k / self.g_i)
        return f_ik

    @property
    def log_gf(self) -> float:
        """Log10 of g_i * f_ik (oscillator strength weighted by lower level degeneracy)."""
        gf = self.g_i * self.oscillator_strength
        if gf <= 0:
            return -np.inf
        return np.log10(gf)


class CurveOfGrowthAnalyzer:
    """
    Analyzes multiplet lines using the Curve-of-Growth method.

    The curve-of-growth (COG) is a fundamental diagnostic tool in spectroscopy
    that relates the equivalent width of absorption/emission lines to the
    column density of absorbing atoms. By plotting W/lambda vs gf for multiple
    lines from the same species, one can:

    1. Diagnose optical depth effects
    2. Determine the COG regime (linear, saturation, damping)
    3. Extract column density when combined with Doppler width
    4. Identify self-absorbed lines that deviate from the linear regime

    Theory
    ------
    For a homogeneous slab with Voigt profile:

    **Linear regime** (tau_0 << 1):
        W/lambda = (pi * e^2 / m_e * c^2) * N * lambda * f
        => log(W/lambda) = log(N * f) + const
        => slope = 1 in log-log plot

    **Saturation (flat) regime** (tau_0 ~ 1-10):
        W/lambda propto sqrt(ln(tau_0))
        => slope < 1 (typically 0-0.3)

    **Damping (square-root) regime** (tau_0 >> 10):
        W/lambda propto sqrt(gamma * N * f)
        => slope = 0.5 in log-log plot

    The transition between regimes depends on the damping parameter:
        a = gamma / (4 * pi * delta_nu_D)

    where gamma is the total damping width (natural + Stark + van der Waals)
    and delta_nu_D is the Doppler width.

    Usage
    -----
    >>> analyzer = CurveOfGrowthAnalyzer(temperature_K=10000, mass_amu=56)
    >>> lines = [MultipletLine(...), MultipletLine(...), ...]
    >>> result = analyzer.fit(lines)
    >>> print(f"Regime: {result.regime}, tau_max: {result.optical_depth_estimate}")

    References
    ----------
    - Mihalas, D. (1978): Stellar Atmospheres, Chapter 9
    - Gray, D.F. (2005): The Observation and Analysis of Stellar Photospheres
    - El Sherbini et al. (2020): Self-absorption correction in LIBS
    """

    def __init__(
        self,
        temperature_K: float = 10000.0,
        mass_amu: float = 56.0,  # Default Fe
        damping_constant: Optional[float] = None,
        min_lines: int = 3,
    ):
        """
        Initialize the COG analyzer.

        Parameters
        ----------
        temperature_K : float
            Plasma/gas temperature in Kelvin (for Doppler width calculation)
        mass_amu : float
            Atomic mass in atomic mass units (default 56 for Fe)
        damping_constant : float, optional
            Damping parameter 'a'. If None, estimated from Stark broadening.
        min_lines : int
            Minimum number of lines required for analysis (default 3)
        """
        self.temperature_K = temperature_K
        self.mass_amu = mass_amu
        self.damping_constant = damping_constant
        self.min_lines = min_lines

        # Pre-calculate Doppler velocity
        # v_D = sqrt(2 * k * T / m)
        mass_kg = mass_amu * M_PROTON
        self._v_thermal = np.sqrt(2 * KB * temperature_K / mass_kg)

    def _compute_doppler_width(self, wavelength_nm: float) -> float:
        """
        Compute Doppler width (FWHM) in nm for a given wavelength.

        delta_lambda_D = lambda * v_th / c
        FWHM = 2 * sqrt(ln 2) * delta_lambda_D

        Parameters
        ----------
        wavelength_nm : float
            Transition wavelength in nm

        Returns
        -------
        float
            Doppler FWHM in nm
        """
        # Doppler width (1/e width)
        delta_lambda = wavelength_nm * self._v_thermal / C_LIGHT
        # FWHM = 2 * sqrt(ln 2) * sigma for Gaussian
        fwhm = 2.0 * np.sqrt(np.log(2)) * delta_lambda
        return fwhm

    def _compute_theoretical_cog(
        self,
        log_tau_0: np.ndarray,
        damping_param: float = 0.01,
    ) -> np.ndarray:
        """
        Compute theoretical curve-of-growth W/delta_lambda_D vs tau_0.

        Uses the Voigt function approximation for the COG.

        For small a (Doppler-dominated):
            W/delta_lambda_D = 2 * sqrt(pi) * integral[1 - exp(-tau * H(a,v))] dv

        Parameters
        ----------
        log_tau_0 : array
            Log10 of optical depth at line center
        damping_param : float
            Voigt damping parameter a = gamma / (4 * pi * delta_nu_D)

        Returns
        -------
        array
            Log10 of reduced equivalent width W/delta_lambda_D
        """
        # Ensure float array to avoid integer power issues
        log_tau_0 = np.asarray(log_tau_0, dtype=float)
        tau_0 = 10.0**log_tau_0

        # Use analytic approximations for different regimes
        W_over_delta = np.zeros_like(tau_0)

        for i, tau in enumerate(tau_0):
            if tau < 0.1:
                # Linear regime: W ≈ sqrt(pi) * tau_0
                W_over_delta[i] = np.sqrt(np.pi) * tau
            elif tau < 1.0:
                # Transition from linear to saturation
                # Smooth interpolation using the saturation formula
                # but ensuring continuity
                log_term = np.log(tau)
                # For tau between 0.1 and 1, log(tau) is negative
                # Use linear approximation to bridge
                W_linear = np.sqrt(np.pi) * tau
                W_sat = 2.0 * np.sqrt(max(0.01, log_term + damping_param * np.sqrt(np.pi * tau)))
                # Weight towards saturation as tau increases
                weight = (tau - 0.1) / 0.9
                W_over_delta[i] = (1 - weight) * W_linear + weight * W_sat
            else:
                # Saturation/damping regime
                log_term = np.log(tau)
                W_over_delta[i] = 2.0 * np.sqrt(log_term + damping_param * np.sqrt(np.pi * tau))

        return np.log10(W_over_delta)

    def _estimate_oscillator_strength_from_A(
        self, A_ki: float, wavelength_nm: float, g_k: int, g_i: int
    ) -> float:
        """
        Calculate absorption oscillator strength from Einstein A coefficient.

        f_ik = 1.4992 * lambda[cm]^2 * A_ki[s^-1] * (g_k / g_i)

        Prefactor is m_e*c/(8*pi^2*e^2) = 1.4992 in CGS with lambda in cm.
        """
        lambda_cm = wavelength_nm * 1e-7
        return 1.4992 * (lambda_cm**2) * A_ki * (g_k / g_i)

    def fit(
        self,
        lines: List[MultipletLine],
        excitation_correct: bool = True,
    ) -> COGResult:
        """
        Fit the curve-of-growth to a set of multiplet lines.

        This analyzes the relationship between reduced equivalent width (W/lambda)
        and oscillator strength (gf) to determine the optical depth regime and
        extract column density.

        Parameters
        ----------
        lines : List[MultipletLine]
            List of lines from the same multiplet or species
        excitation_correct : bool
            If True, correct for different lower level populations using
            Boltzmann factors at the specified temperature

        Returns
        -------
        COGResult
            Analysis results including regime, optical depth, and fit parameters

        Raises
        ------
        ValueError
            If fewer than min_lines are provided
        """
        warnings_list: List[str] = []

        if len(lines) < self.min_lines:
            raise ValueError(
                f"Need at least {self.min_lines} lines for COG analysis, got {len(lines)}"
            )

        # Validate all lines are from the same species
        elements = set(line.element for line in lines)
        stages = set(line.ionization_stage for line in lines)
        if len(elements) > 1:
            warnings_list.append(
                f"Multiple elements detected: {elements}. COG best for single species."
            )
        if len(stages) > 1:
            warnings_list.append(f"Multiple ionization stages: {stages}. Consider separating.")

        # Prepare data for COG plot
        lines_data: List[COGLineData] = []
        log_gf_values = []
        log_W_lambda_values = []

        # Reference energy for excitation correction (lowest E_i)
        E_ref = min(line.E_i_ev for line in lines)

        for line in lines:
            if line.equivalent_width_nm <= 0:
                warnings_list.append(
                    f"Skipping line at {line.wavelength_nm:.2f} nm: non-positive EW"
                )
                continue

            # Calculate reduced equivalent width
            reduced_W = line.equivalent_width_nm / line.wavelength_nm
            log_reduced_W = np.log10(reduced_W)

            # Get log(gf)
            log_gf = line.log_gf

            # Excitation correction: adjust for different lower level populations
            # N_i/N_0 = (g_i/g_0) * exp(-(E_i - E_0) / kT)
            # We correct W to a common reference energy E_ref
            if excitation_correct and self.temperature_K > 0:
                T_eV = self.temperature_K * KB_EV
                delta_E = line.E_i_ev - E_ref
                # W_corrected = W * exp(+delta_E / kT) to bring to E_ref level
                excitation_factor = np.exp(delta_E / T_eV)
                reduced_W_corrected = reduced_W * excitation_factor
                log_reduced_W = np.log10(reduced_W_corrected)

            # Store line data
            line_data = COGLineData(
                wavelength_nm=line.wavelength_nm,
                equivalent_width_nm=line.equivalent_width_nm,
                log_gf=log_gf,
                reduced_width=reduced_W,
                log_reduced_width=log_reduced_W,
                E_i_ev=line.E_i_ev,
                element=line.element,
                ionization_stage=line.ionization_stage,
            )
            lines_data.append(line_data)
            log_gf_values.append(log_gf)
            log_W_lambda_values.append(log_reduced_W)

        if len(lines_data) < self.min_lines:
            raise ValueError(
                f"Only {len(lines_data)} valid lines after filtering, need {self.min_lines}"
            )

        # Convert to arrays
        x = np.array(log_gf_values)  # log(gf)
        y = np.array(log_W_lambda_values)  # log(W/lambda)

        # Sort by x for consistent analysis
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        lines_data = [lines_data[i] for i in sort_idx]

        # Perform linear fit: log(W/lambda) = slope * log(gf) + intercept
        # In linear regime, slope = 1
        # In saturation, slope -> 0
        # In damping, slope -> 0.5
        slope, intercept, r_squared, residuals = self._fit_linear(x, y)

        # Determine regime from slope
        regime = self._classify_regime(slope, r_squared)

        # Estimate mean Doppler width
        mean_wavelength = np.mean([ld.wavelength_nm for ld in lines_data])
        doppler_width = self._compute_doppler_width(mean_wavelength)

        # Estimate damping parameter if not provided
        if self.damping_constant is not None:
            damping_param = self.damping_constant
        else:
            # Typical LIBS plasmas have significant Stark broadening
            # Estimate a ~ 0.01 - 0.1 for moderate electron densities
            damping_param = 0.05  # Conservative default

        # Estimate column density from intercept
        # In linear regime: log(W/lambda) = log(const * N * f) where const includes physics
        # const = pi * e^2 / (m_e * c^2) = 8.85e-13 cm^2 / s (CGS)
        # W/lambda = const * N * f * lambda => at f=1, W/lambda = const * N * lambda
        column_density = self._estimate_column_density(intercept, slope, mean_wavelength, regime)

        # Estimate optical depth at line center for strongest line
        # tau_0 = (pi * e^2 / m_e * c) * f * N * phi(0)
        # where phi(0) = 1 / (sqrt(pi) * delta_nu_D) for Doppler profile
        max_log_gf = max(log_gf_values)
        optical_depth = self._estimate_optical_depth(
            column_density, max_log_gf, doppler_width, mean_wavelength
        )

        return COGResult(
            regime=regime,
            optical_depth_estimate=optical_depth,
            column_density_cm2=column_density,
            doppler_width_nm=doppler_width,
            damping_parameter=damping_param,
            fit_slope=slope,
            fit_intercept=intercept,
            fit_r_squared=r_squared,
            fit_residuals=residuals,
            n_lines_used=len(lines_data),
            lines_data=lines_data,
            warnings=warnings_list,
        )

    def _fit_linear(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        """
        Perform weighted linear fit to log-log data.

        Parameters
        ----------
        x : array
            Log10(gf) values
        y : array
            Log10(W/lambda) values

        Returns
        -------
        slope : float
        intercept : float
        r_squared : float
        residuals : array
        """
        n = len(x)
        if n < 2:
            return 0.0, 0.0, 0.0, np.array([])

        # Simple least squares fit
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_yy = np.sum((y - y_mean) ** 2)

        if ss_xx < 1e-10:
            # All x values are the same
            return 0.0, y_mean, 0.0, y - y_mean

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean

        # Calculate R^2
        y_pred = slope * x + intercept
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)

        r_squared = 1.0 - ss_res / ss_yy if ss_yy > 0 else 0.0

        return slope, intercept, r_squared, residuals

    def _classify_regime(self, slope: float, r_squared: float) -> COGRegime:
        """
        Classify the COG regime based on the fit slope.

        Parameters
        ----------
        slope : float
            Slope of log(W/lambda) vs log(gf)
        r_squared : float
            Coefficient of determination

        Returns
        -------
        COGRegime
            Identified regime
        """
        # Poor fit indicates transitional or mixed regime
        if r_squared < 0.5:
            return COGRegime.UNKNOWN

        # Classify by slope:
        # Linear regime: slope ~ 1.0
        # Saturation: slope ~ 0-0.3
        # Damping: slope ~ 0.5
        if slope > 0.8:
            return COGRegime.LINEAR
        elif slope < 0.3:
            return COGRegime.SATURATION
        elif 0.4 <= slope <= 0.6:
            return COGRegime.DAMPING
        else:
            # Transitional region
            return COGRegime.UNKNOWN

    def _estimate_column_density(
        self,
        intercept: float,
        slope: float,
        mean_wavelength_nm: float,
        regime: COGRegime,
    ) -> float:
        """
        Estimate column density from COG intercept.

        In the linear regime:
        W/lambda = (pi * e^2 / m_e * c^2) * N * f * lambda

        At log(gf) = 0 (gf = 1):
        log(W/lambda) = intercept = log(const * N * lambda)

        Parameters
        ----------
        intercept : float
            Y-intercept of log-log fit
        slope : float
            Slope of fit (used to validate regime)
        mean_wavelength_nm : float
            Representative wavelength
        regime : COGRegime
            Identified regime

        Returns
        -------
        float
            Column density in cm^-2
        """
        # Physical constant: pi * e^2 / (m_e * c^2) in CGS
        # = pi * (4.803e-10)^2 / (9.109e-28 * (3e10)^2)
        # = 8.85e-13 cm^2
        const = 8.85e-13  # cm^2

        lambda_cm = mean_wavelength_nm * 1e-7

        # In linear regime: W/lambda = const * N * lambda * gf
        # At gf = 1: W/lambda = const * N * lambda
        # intercept = log10(const * N * lambda)
        # N = 10^intercept / (const * lambda)

        if regime == COGRegime.LINEAR or slope > 0.7:
            # Linear regime estimate
            N = (10**intercept) / (const * lambda_cm)
        else:
            # Saturation regime - relationship is more complex
            # Use rough scaling: W/lambda ~ 2 * delta_lambda_D * sqrt(ln(tau))
            # This is less accurate but gives order of magnitude
            N = (10**intercept) / (const * lambda_cm)
            # Note: this overestimates N in saturation regime

        return max(N, 0.0)

    def _estimate_optical_depth(
        self,
        column_density: float,
        max_log_gf: float,
        doppler_width_nm: float,
        wavelength_nm: float,
    ) -> float:
        """
        Estimate optical depth at line center for the strongest line.

        tau_0 = (pi * e^2 / m_e * c) * f * N / (sqrt(pi) * delta_nu_D)

        For wavelength units:
        tau_0 = (pi * e^2 * lambda^2 / m_e * c^2) * f * N / (sqrt(pi) * delta_lambda_D)

        Parameters
        ----------
        column_density : float
            Column density in cm^-2
        max_log_gf : float
            Log10(gf) of strongest line
        doppler_width_nm : float
            Doppler width in nm
        wavelength_nm : float
            Line wavelength in nm

        Returns
        -------
        float
            Optical depth at line center
        """
        # Convert to CGS
        lambda_cm = wavelength_nm * 1e-7
        delta_lambda_cm = doppler_width_nm * 1e-7

        # Oscillator strength (approximate: gf ~ f for g=1)
        gf = 10**max_log_gf
        f_approx = gf  # Assume g ~ 1 for order of magnitude

        # Cross section at line center for Doppler profile
        # sigma_0 = (pi * e^2 / m_e * c) * f * lambda / (sqrt(pi) * delta_lambda_D)
        # = 8.85e-13 * f * lambda / (sqrt(pi) * delta_lambda_D)
        const = 8.85e-13 / np.sqrt(np.pi)  # cm^2

        if delta_lambda_cm > 0:
            sigma_0 = const * f_approx * lambda_cm / delta_lambda_cm
            tau_0 = sigma_0 * column_density
        else:
            tau_0 = 0.0

        return max(tau_0, 0.0)

    def get_correction_factors(self, result: COGResult) -> Dict[float, float]:
        """
        Get self-absorption correction factors for each line based on COG analysis.

        Uses the COG result to estimate optical depth for each line and
        returns the corresponding correction factors.

        Parameters
        ----------
        result : COGResult
            Result from fit() method

        Returns
        -------
        Dict[float, float]
            Wavelength (nm) -> correction factor mapping
            Apply as: I_true = I_measured * correction_factor
        """
        corrections = {}

        # Reference: strongest line optical depth
        tau_ref = result.optical_depth_estimate
        max_log_gf = max(ld.log_gf for ld in result.lines_data)

        for line_data in result.lines_data:
            # Scale optical depth by relative gf
            # tau_i / tau_ref = gf_i / gf_ref (in linear regime)
            delta_log_gf = line_data.log_gf - max_log_gf
            tau_i = tau_ref * (10**delta_log_gf)

            # Calculate correction factor
            f_tau = _escape_factor(tau_i)

            # Correction factor: multiply measured intensity by this to get true
            correction = 1.0 / f_tau if f_tau > 0 else 1.0
            corrections[line_data.wavelength_nm] = correction

        return corrections

    def diagnose_self_absorption(self, lines: List[MultipletLine]) -> Dict[float, bool]:
        """
        Quick diagnostic to identify which lines are likely self-absorbed.

        A line is flagged as self-absorbed if it falls below the expected
        linear COG curve, indicating saturation effects.

        Parameters
        ----------
        lines : List[MultipletLine]
            Lines to analyze

        Returns
        -------
        Dict[float, bool]
            Wavelength -> is_self_absorbed mapping
        """
        if len(lines) < self.min_lines:
            return {line.wavelength_nm: False for line in lines}

        try:
            result = self.fit(lines, excitation_correct=True)
        except ValueError:
            return {line.wavelength_nm: False for line in lines}

        diagnosed = {}

        for line_data in result.lines_data:
            # Expected y from linear fit
            y_expected = result.fit_slope * line_data.log_gf + result.fit_intercept

            # Actual y
            y_actual = line_data.log_reduced_width

            # If actual < expected by more than ~0.1 dex, likely self-absorbed
            # (self-absorption reduces equivalent width)
            # Actually, in emission, self-absorption reduces apparent intensity
            # but for COG analysis of absorption, saturation makes W grow slower
            deviation = y_actual - y_expected

            # In saturation, lines fall below the linear extrapolation
            is_absorbed = bool(deviation < -0.1 and result.regime != COGRegime.LINEAR)

            diagnosed[line_data.wavelength_nm] = is_absorbed

        return diagnosed
