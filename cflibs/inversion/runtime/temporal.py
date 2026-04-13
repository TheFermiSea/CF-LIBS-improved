"""
Temporal dynamics and time-resolved optimization for LIBS analysis.

This module provides algorithms for exploiting time-resolved LIBS data to improve
analysis accuracy. LIBS plasma conditions evolve over microsecond timescales, and
gate timing significantly affects which plasma region is sampled, impacting LTE
validity and self-absorption.

Key Components
--------------
- PlasmaEvolutionModel: Models T(t) and n_e(t) evolution during plasma decay
- GateTimingOptimizer: Finds optimal gate delay/width for CF-LIBS accuracy
- TemporalSelfAbsorptionCorrector: Corrects for time-varying optical depth
- TimeResolvedSolver: Multi-gate CF-LIBS analysis combining temporal data

Physics Background
------------------
After laser ablation, the LIBS plasma undergoes several phases:

1. **Early phase** (0-100 ns): High continuum, non-LTE, strong self-absorption
2. **Intermediate phase** (100 ns - 2 us): Favorable for analysis, good LTE,
   moderate self-absorption, strong line emission
3. **Late phase** (>2 us): Weak emission, good LTE but low SNR, minimal
   self-absorption

The optimal gate timing balances:
- LTE validity (favors later times)
- Signal strength (favors earlier times)
- Self-absorption (favors later times)
- Continuum background (favors later times)

Plasma Evolution
----------------
Temperature typically follows an exponential decay:
    T(t) = T_0 * exp(-t/tau_T) + T_ambient

Electron density follows a faster decay due to recombination:
    n_e(t) = n_e0 * exp(-t/tau_ne)

Where tau_T > tau_ne typically (temperature decays slower than density).

Literature References
---------------------
- Harilal et al. (2003): Temporal and spatial evolution of LIBS plasmas
- Aragón & Aguilera (2008): Plasma characterization methods review
- De Giacomo et al. (2011): Spatio-temporal dynamics of laser-induced plasmas
- Cristoforetti et al. (2010): Influence of timing on LTE validity
- Tognoni et al. (2010): CF-LIBS state of the art
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation

logger = get_logger("inversion.temporal")


# =============================================================================
# Data Structures
# =============================================================================


class PlasmaPhase(Enum):
    """
    Classification of plasma evolution phases.

    Each phase has distinct characteristics affecting CF-LIBS analysis:
    - EARLY: High continuum, non-LTE, unsuitable for CF-LIBS
    - INTERMEDIATE: Optimal for analysis, good LTE, strong lines
    - LATE: Weak emission, excellent LTE but low SNR
    """

    EARLY = "early"
    INTERMEDIATE = "intermediate"
    LATE = "late"


@dataclass
class TemporalGateConfig:
    """
    Configuration for a spectral acquisition gate.

    In time-resolved LIBS, the spectrometer gate controls which portion
    of the plasma emission is recorded. The gate is characterized by:
    - delay: Time from laser pulse to gate opening
    - width: Duration the gate remains open

    Attributes
    ----------
    delay_ns : float
        Gate delay after laser pulse in nanoseconds
    width_ns : float
        Gate width (integration time) in nanoseconds
    label : str
        Optional label for this gate configuration
    """

    delay_ns: float
    width_ns: float
    label: str = ""

    def __post_init__(self):
        if self.delay_ns < 0:
            raise ValueError("Gate delay must be non-negative")
        if self.width_ns <= 0:
            raise ValueError("Gate width must be positive")

    @property
    def center_ns(self) -> float:
        """Center time of the gate window."""
        return self.delay_ns + self.width_ns / 2

    @property
    def end_ns(self) -> float:
        """End time of the gate window."""
        return self.delay_ns + self.width_ns


@dataclass
class TimeResolvedSpectrum:
    """
    A spectrum acquired at a specific gate timing.

    Attributes
    ----------
    gate : TemporalGateConfig
        Gate timing configuration
    observations : List[LineObservation]
        Line observations at this gate timing
    continuum_level : float
        Background continuum intensity (arbitrary units)
    snr_estimate : float
        Estimated signal-to-noise ratio
    metadata : Dict
        Additional metadata (laser energy, spot size, etc.)
    """

    gate: TemporalGateConfig
    observations: List[LineObservation]
    continuum_level: float = 0.0
    snr_estimate: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class PlasmaEvolutionPoint:
    """
    Plasma state at a single point in time.

    Attributes
    ----------
    time_ns : float
        Time after laser pulse in nanoseconds
    temperature_K : float
        Plasma temperature in Kelvin
    electron_density_cm3 : float
        Electron density in cm^-3
    lte_validity : float
        LTE validity score (0-1, 1 = perfect LTE)
    phase : PlasmaPhase
        Classification of the plasma phase
    """

    time_ns: float
    temperature_K: float
    electron_density_cm3: float
    lte_validity: float = 1.0
    phase: PlasmaPhase = PlasmaPhase.INTERMEDIATE


@dataclass
class PlasmaEvolutionProfile:
    """
    Complete temporal profile of plasma evolution.

    Attributes
    ----------
    points : List[PlasmaEvolutionPoint]
        Time series of plasma states
    tau_T_ns : float
        Temperature decay time constant in ns
    tau_ne_ns : float
        Electron density decay time constant in ns
    T_initial_K : float
        Initial temperature in Kelvin
    ne_initial_cm3 : float
        Initial electron density in cm^-3
    lte_threshold_time_ns : float
        Time after which LTE is considered valid
    """

    points: List[PlasmaEvolutionPoint]
    tau_T_ns: float
    tau_ne_ns: float
    T_initial_K: float
    ne_initial_cm3: float
    lte_threshold_time_ns: float = 100.0

    def temperature_at(self, time_ns: float) -> float:
        """Interpolate temperature at arbitrary time."""
        times = np.array([p.time_ns for p in self.points])
        temps = np.array([p.temperature_K for p in self.points])
        return float(np.interp(time_ns, times, temps))

    def density_at(self, time_ns: float) -> float:
        """Interpolate electron density at arbitrary time."""
        times = np.array([p.time_ns for p in self.points])
        densities = np.array([p.electron_density_cm3 for p in self.points])
        return float(np.interp(time_ns, times, densities))


@dataclass
class GateOptimizationResult:
    """
    Result of gate timing optimization.

    Attributes
    ----------
    optimal_delay_ns : float
        Recommended gate delay in nanoseconds
    optimal_width_ns : float
        Recommended gate width in nanoseconds
    score : float
        Optimization score (higher is better)
    lte_validity : float
        LTE validity score at optimal timing
    expected_temperature_K : float
        Expected plasma temperature at optimal gate
    expected_density_cm3 : float
        Expected electron density at optimal gate
    self_absorption_factor : float
        Expected self-absorption severity (1 = none, >1 = significant)
    continuum_ratio : float
        Line-to-continuum ratio estimate
    analysis : Dict
        Detailed analysis of different timing options
    """

    optimal_delay_ns: float
    optimal_width_ns: float
    score: float
    lte_validity: float
    expected_temperature_K: float
    expected_density_cm3: float
    self_absorption_factor: float = 1.0
    continuum_ratio: float = 100.0
    analysis: Dict = field(default_factory=dict)


@dataclass
class TemporalSelfAbsorptionResult:
    """
    Result of temporal self-absorption correction.

    Attributes
    ----------
    corrected_observations : List[LineObservation]
        Observations with time-dependent correction applied
    optical_depths : Dict[float, Dict[float, float]]
        Optical depth by time and wavelength: {time_ns: {wavelength: tau}}
    correction_factors : Dict[float, Dict[float, float]]
        Correction factors by time and wavelength
    time_averaged_tau : Dict[float, float]
        Gate-averaged optical depth by wavelength
    warnings : List[str]
        Any warnings generated during correction
    """

    corrected_observations: List[LineObservation]
    optical_depths: Dict[float, Dict[float, float]]
    correction_factors: Dict[float, Dict[float, float]]
    time_averaged_tau: Dict[float, float]
    warnings: List[str] = field(default_factory=list)


@dataclass
class TimeResolvedCFLIBSResult:
    """
    Result of time-resolved CF-LIBS analysis.

    Attributes
    ----------
    concentrations : Dict[str, float]
        Final element concentrations (mass fractions)
    concentration_uncertainties : Dict[str, float]
        Uncertainties in concentrations
    temperature_profile : List[Tuple[float, float]]
        Temperature vs time: [(time_ns, T_K), ...]
    density_profile : List[Tuple[float, float]]
        Electron density vs time: [(time_ns, n_e), ...]
    gate_weights : Dict[float, float]
        Weight assigned to each gate in final analysis
    per_gate_results : Dict[float, Dict]
        CF-LIBS results for each individual gate
    combined_quality : float
        Overall quality metric for the combined analysis
    optimal_gate : TemporalGateConfig
        The gate that contributed most to the final result
    """

    concentrations: Dict[str, float]
    concentration_uncertainties: Dict[str, float]
    temperature_profile: List[Tuple[float, float]]
    density_profile: List[Tuple[float, float]]
    gate_weights: Dict[float, float]
    per_gate_results: Dict[float, Dict]
    combined_quality: float
    optimal_gate: Optional[TemporalGateConfig] = None


# =============================================================================
# Plasma Evolution Model
# =============================================================================


class PlasmaEvolutionModel:
    """
    Model for LIBS plasma temporal evolution.

    This class models how plasma temperature and electron density evolve
    after the laser pulse. The default model uses exponential decay with
    different time constants for T and n_e.

    Physics
    -------
    After laser ablation, the plasma expands and cools. The evolution follows:

    Temperature:
        T(t) = (T_0 - T_ambient) * exp(-t/tau_T) + T_ambient

    Electron density (faster decay due to recombination):
        n_e(t) = n_e0 * (1 + t/t_0)^(-alpha)

    Or simplified exponential:
        n_e(t) = n_e0 * exp(-t/tau_ne)

    LTE validity improves as the plasma cools and density decreases, but
    the McWhirter criterion n_e > 1.6e12 * T^0.5 * (delta_E)^3 must be satisfied.

    Parameters
    ----------
    T_initial_K : float
        Initial plasma temperature in K (default: 20000 K)
    ne_initial_cm3 : float
        Initial electron density in cm^-3 (default: 1e18)
    tau_T_ns : float
        Temperature decay time constant in ns (default: 1000 ns)
    tau_ne_ns : float
        Electron density decay time constant in ns (default: 500 ns)
    T_ambient_K : float
        Ambient temperature (default: 300 K)
    """

    def __init__(
        self,
        T_initial_K: float = 20000.0,
        ne_initial_cm3: float = 1e18,
        tau_T_ns: float = 1000.0,
        tau_ne_ns: float = 500.0,
        T_ambient_K: float = 300.0,
    ):
        self.T_initial_K = T_initial_K
        self.ne_initial_cm3 = ne_initial_cm3
        self.tau_T_ns = tau_T_ns
        self.tau_ne_ns = tau_ne_ns
        self.T_ambient_K = T_ambient_K

    def temperature(self, time_ns: float) -> float:
        """
        Calculate plasma temperature at given time.

        Parameters
        ----------
        time_ns : float
            Time after laser pulse in nanoseconds

        Returns
        -------
        float
            Temperature in Kelvin
        """
        if time_ns < 0:
            return self.T_initial_K

        delta_T = self.T_initial_K - self.T_ambient_K
        return delta_T * np.exp(-time_ns / self.tau_T_ns) + self.T_ambient_K

    def electron_density(self, time_ns: float) -> float:
        """
        Calculate electron density at given time.

        Parameters
        ----------
        time_ns : float
            Time after laser pulse in nanoseconds

        Returns
        -------
        float
            Electron density in cm^-3
        """
        if time_ns < 0:
            return self.ne_initial_cm3

        return self.ne_initial_cm3 * np.exp(-time_ns / self.tau_ne_ns)

    def lte_validity(self, time_ns: float, max_delta_E_eV: float = 3.0) -> float:
        """
        Estimate LTE validity at given time using McWhirter criterion.

        The McWhirter criterion for LTE requires:
            n_e >= 1.6e12 * T^0.5 * (delta_E)^3

        where delta_E is the largest energy gap of interest.

        Parameters
        ----------
        time_ns : float
            Time after laser pulse in nanoseconds
        max_delta_E_eV : float
            Maximum energy gap of interest in eV

        Returns
        -------
        float
            LTE validity score (0-1, 1 = perfect LTE)
        """
        T_K = self.temperature(time_ns)
        n_e = self.electron_density(time_ns)

        # McWhirter criterion threshold
        n_e_threshold = 1.6e12 * np.sqrt(T_K) * (max_delta_E_eV**3)

        # LTE score: how much n_e exceeds threshold
        if n_e <= 0:
            return 0.0

        ratio = n_e / n_e_threshold

        # Score: 1 if ratio >= 10, decreasing below that
        if ratio >= 10:
            return 1.0
        elif ratio >= 1:
            return 0.5 + 0.5 * np.log10(ratio)
        else:
            return 0.5 * ratio

    def classify_phase(self, time_ns: float) -> PlasmaPhase:
        """
        Classify the plasma phase at given time.

        Parameters
        ----------
        time_ns : float
            Time after laser pulse in nanoseconds

        Returns
        -------
        PlasmaPhase
            Phase classification
        """
        # Typical phase boundaries (can be adjusted based on experimental conditions)
        if time_ns < 100:
            return PlasmaPhase.EARLY
        elif time_ns < 2000:
            return PlasmaPhase.INTERMEDIATE
        else:
            return PlasmaPhase.LATE

    def generate_profile(
        self,
        time_points: Optional[np.ndarray] = None,
        n_points: int = 100,
        t_max_ns: float = 5000.0,
        max_delta_E_eV: float = 3.0,
    ) -> PlasmaEvolutionProfile:
        """
        Generate a complete temporal profile.

        Parameters
        ----------
        time_points : np.ndarray, optional
            Specific time points to evaluate (ns)
        n_points : int
            Number of points if time_points not specified
        t_max_ns : float
            Maximum time if time_points not specified
        max_delta_E_eV : float
            Energy gap for LTE calculation

        Returns
        -------
        PlasmaEvolutionProfile
            Complete evolution profile
        """
        if time_points is None:
            time_points = np.linspace(0, t_max_ns, n_points)

        points = []
        lte_threshold_time = None

        for t in time_points:
            T = self.temperature(t)
            n_e = self.electron_density(t)
            lte = self.lte_validity(t, max_delta_E_eV)
            phase = self.classify_phase(t)

            points.append(
                PlasmaEvolutionPoint(
                    time_ns=float(t),
                    temperature_K=T,
                    electron_density_cm3=n_e,
                    lte_validity=lte,
                    phase=phase,
                )
            )

            # Find when LTE first becomes valid
            if lte_threshold_time is None and lte > 0.9:
                lte_threshold_time = float(t)

        return PlasmaEvolutionProfile(
            points=points,
            tau_T_ns=self.tau_T_ns,
            tau_ne_ns=self.tau_ne_ns,
            T_initial_K=self.T_initial_K,
            ne_initial_cm3=self.ne_initial_cm3,
            lte_threshold_time_ns=lte_threshold_time or 100.0,
        )

    @classmethod
    def from_measurements(
        cls,
        times_ns: np.ndarray,
        temperatures_K: np.ndarray,
        densities_cm3: np.ndarray,
    ) -> "PlasmaEvolutionModel":
        """
        Fit evolution model to experimental measurements.

        Parameters
        ----------
        times_ns : np.ndarray
            Measurement times in nanoseconds
        temperatures_K : np.ndarray
            Measured temperatures in Kelvin
        densities_cm3 : np.ndarray
            Measured electron densities in cm^-3

        Returns
        -------
        PlasmaEvolutionModel
            Fitted model
        """
        # Fit exponential decay to temperature
        # ln(T - T_amb) = ln(T_0 - T_amb) - t/tau_T
        T_ambient = 300.0
        T_shifted = np.maximum(temperatures_K - T_ambient, 1.0)
        log_T = np.log(T_shifted)

        # Linear fit for temperature
        # y = a + b*x where y = log(T-T_amb), x = time, b = -1/tau_T
        mask_T = np.isfinite(log_T)
        if np.sum(mask_T) >= 2:
            coeff_T = np.polyfit(times_ns[mask_T], log_T[mask_T], 1)
            tau_T = -1.0 / coeff_T[0] if coeff_T[0] != 0 else 1000.0
            T_initial = np.exp(coeff_T[1]) + T_ambient
        else:
            tau_T = 1000.0
            T_initial = temperatures_K[0] if len(temperatures_K) > 0 else 20000.0

        # Fit exponential decay to density
        log_ne = np.log(np.maximum(densities_cm3, 1.0))
        mask_ne = np.isfinite(log_ne)
        if np.sum(mask_ne) >= 2:
            coeff_ne = np.polyfit(times_ns[mask_ne], log_ne[mask_ne], 1)
            tau_ne = -1.0 / coeff_ne[0] if coeff_ne[0] != 0 else 500.0
            ne_initial = np.exp(coeff_ne[1])
        else:
            tau_ne = 500.0
            ne_initial = densities_cm3[0] if len(densities_cm3) > 0 else 1e18

        # Ensure reasonable values
        tau_T = np.clip(tau_T, 100.0, 10000.0)
        tau_ne = np.clip(tau_ne, 50.0, 5000.0)
        T_initial = np.clip(T_initial, 5000.0, 50000.0)
        ne_initial = np.clip(ne_initial, 1e15, 1e20)

        return cls(
            T_initial_K=float(T_initial),
            ne_initial_cm3=float(ne_initial),
            tau_T_ns=float(tau_T),
            tau_ne_ns=float(tau_ne),
            T_ambient_K=T_ambient,
        )


# =============================================================================
# Gate Timing Optimizer
# =============================================================================


class GateTimingOptimizer:
    """
    Optimize LIBS gate timing for CF-LIBS analysis accuracy.

    This optimizer balances multiple factors to find the optimal gate delay
    and width:
    - LTE validity (needs adequate n_e at temperature T)
    - Signal strength (emission decays with time)
    - Self-absorption (decreases with time as n_e drops)
    - Continuum-to-line ratio (decreases with time)

    The optimization uses a scoring function that weights these factors
    according to user priorities.

    Parameters
    ----------
    evolution_model : PlasmaEvolutionModel
        Model of plasma evolution
    lte_weight : float
        Weight for LTE validity in scoring (default: 1.0)
    signal_weight : float
        Weight for signal strength (default: 0.5)
    self_absorption_weight : float
        Weight for self-absorption avoidance (default: 0.8)
    continuum_weight : float
        Weight for low continuum (default: 0.3)
    """

    def __init__(
        self,
        evolution_model: PlasmaEvolutionModel,
        lte_weight: float = 1.0,
        signal_weight: float = 0.5,
        self_absorption_weight: float = 0.8,
        continuum_weight: float = 0.3,
    ):
        self.model = evolution_model
        self.lte_weight = lte_weight
        self.signal_weight = signal_weight
        self.self_absorption_weight = self_absorption_weight
        self.continuum_weight = continuum_weight

    def _signal_score(self, time_ns: float) -> float:
        """
        Score for signal strength (higher is better).

        Signal intensity scales approximately as:
            I ~ n_e * exp(-E/kT)

        For typical emission lines, this decreases with time.
        """
        T = self.model.temperature(time_ns)
        n_e = self.model.electron_density(time_ns)

        # Approximate signal strength (normalized)
        T_ref = 10000.0
        ne_ref = 1e17

        # Simple model: signal ~ n_e * (T/T_ref)^0.5
        signal = (n_e / ne_ref) * np.sqrt(T / T_ref)

        # Convert to 0-1 score
        return min(1.0, signal)

    def _self_absorption_score(self, time_ns: float) -> float:
        """
        Score for self-absorption severity (higher = less absorption, better).

        Self-absorption scales with optical depth:
            tau ~ n * sigma * L ~ n_e * (column density)

        As density decreases, self-absorption decreases.
        """
        n_e = self.model.electron_density(time_ns)
        ne_ref = 1e17

        # Lower density = less self-absorption = higher score
        ratio = ne_ref / max(n_e, 1e10)
        return min(1.0, ratio)

    def _continuum_score(self, time_ns: float) -> float:
        """
        Score for continuum background (higher = less continuum, better).

        Continuum emission (Bremsstrahlung) scales as:
            I_cont ~ n_e^2 * T^0.5

        This decreases rapidly with time.
        """
        T = self.model.temperature(time_ns)
        n_e = self.model.electron_density(time_ns)

        T_ref = 10000.0
        ne_ref = 1e17

        # Continuum intensity (lower is better)
        continuum = (n_e / ne_ref) ** 2 * np.sqrt(T / T_ref)

        # Invert to score (less continuum = higher score)
        return 1.0 / (1.0 + continuum)

    def score_gate(
        self,
        delay_ns: float,
        width_ns: float,
        max_delta_E_eV: float = 3.0,
    ) -> Tuple[float, Dict]:
        """
        Score a gate configuration.

        Parameters
        ----------
        delay_ns : float
            Gate delay in nanoseconds
        width_ns : float
            Gate width in nanoseconds
        max_delta_E_eV : float
            Maximum energy gap for LTE criterion

        Returns
        -------
        score : float
            Overall score (higher is better)
        details : Dict
            Breakdown of individual scores
        """
        # Evaluate at gate center
        center_time = delay_ns + width_ns / 2

        # Individual scores
        lte_score = self.model.lte_validity(center_time, max_delta_E_eV)
        signal_score = self._signal_score(center_time)
        sa_score = self._self_absorption_score(center_time)
        cont_score = self._continuum_score(center_time)

        # Weighted combination
        total_weight = (
            self.lte_weight
            + self.signal_weight
            + self.self_absorption_weight
            + self.continuum_weight
        )

        score = (
            self.lte_weight * lte_score
            + self.signal_weight * signal_score
            + self.self_absorption_weight * sa_score
            + self.continuum_weight * cont_score
        ) / total_weight

        details = {
            "lte_score": lte_score,
            "signal_score": signal_score,
            "self_absorption_score": sa_score,
            "continuum_score": cont_score,
            "center_time_ns": center_time,
            "temperature_K": self.model.temperature(center_time),
            "electron_density_cm3": self.model.electron_density(center_time),
        }

        return score, details

    def optimize(
        self,
        delay_range: Tuple[float, float] = (100.0, 3000.0),
        width_range: Tuple[float, float] = (100.0, 2000.0),
        n_delay_points: int = 30,
        n_width_points: int = 10,
        max_delta_E_eV: float = 3.0,
        min_lte_validity: float = 0.7,
    ) -> GateOptimizationResult:
        """
        Find optimal gate timing.

        Parameters
        ----------
        delay_range : Tuple[float, float]
            Range of gate delays to search (ns)
        width_range : Tuple[float, float]
            Range of gate widths to search (ns)
        n_delay_points : int
            Number of delay points in grid search
        n_width_points : int
            Number of width points in grid search
        max_delta_E_eV : float
            Maximum energy gap for LTE criterion
        min_lte_validity : float
            Minimum acceptable LTE validity

        Returns
        -------
        GateOptimizationResult
            Optimization result with recommended timing
        """
        delays = np.linspace(delay_range[0], delay_range[1], n_delay_points)
        widths = np.linspace(width_range[0], width_range[1], n_width_points)

        best_score = -np.inf
        best_delay = delays[0]
        best_width = widths[0]
        best_details: Dict = {}
        all_results = []

        for delay in delays:
            for width in widths:
                score, details = self.score_gate(delay, width, max_delta_E_eV)

                # Apply LTE constraint
                if details["lte_score"] < min_lte_validity:
                    score *= 0.1  # Heavy penalty for poor LTE

                all_results.append(
                    {
                        "delay_ns": delay,
                        "width_ns": width,
                        "score": score,
                        **details,
                    }
                )

                if score > best_score:
                    best_score = score
                    best_delay = delay
                    best_width = width
                    best_details = details

        # Estimate self-absorption factor from electron density
        # Higher density = more self-absorption
        ne = best_details.get("electron_density_cm3", 1e17)
        sa_factor = 1.0 + (ne / 1e18)  # Rough estimate

        logger.info(
            f"Optimal gate: delay={best_delay:.0f} ns, width={best_width:.0f} ns, "
            f"score={best_score:.3f}"
        )

        return GateOptimizationResult(
            optimal_delay_ns=best_delay,
            optimal_width_ns=best_width,
            score=best_score,
            lte_validity=best_details.get("lte_score", 0.0),
            expected_temperature_K=best_details.get("temperature_K", 10000.0),
            expected_density_cm3=best_details.get("electron_density_cm3", 1e17),
            self_absorption_factor=sa_factor,
            continuum_ratio=1.0 / max(best_details.get("continuum_score", 0.01), 0.01),
            analysis={
                "best_details": best_details,
                "n_evaluated": len(all_results),
                "delay_range": delay_range,
                "width_range": width_range,
            },
        )

    def sweep_delays(
        self,
        delays: np.ndarray,
        fixed_width_ns: float = 500.0,
        max_delta_E_eV: float = 3.0,
    ) -> List[Dict]:
        """
        Sweep through delay values with fixed width.

        Useful for plotting optimization landscape.

        Parameters
        ----------
        delays : np.ndarray
            Array of delay values to test (ns)
        fixed_width_ns : float
            Fixed gate width (ns)
        max_delta_E_eV : float
            Energy gap for LTE criterion

        Returns
        -------
        List[Dict]
            Results for each delay value
        """
        results = []
        for delay in delays:
            score, details = self.score_gate(delay, fixed_width_ns, max_delta_E_eV)
            results.append(
                {
                    "delay_ns": delay,
                    "width_ns": fixed_width_ns,
                    "score": score,
                    **details,
                }
            )
        return results


# =============================================================================
# Temporal Self-Absorption Corrector
# =============================================================================


class TemporalSelfAbsorptionCorrector:
    """
    Correct for time-varying self-absorption during plasma evolution.

    Self-absorption varies with time because optical depth depends on
    population densities which change as T and n_e evolve. This corrector
    integrates over the gate window to compute effective correction factors.

    Physics
    -------
    Optical depth at line center:
        tau_0(t) ~ n_lower(t) * sigma_0 * L

    where n_lower(t) is the lower level population that evolves with T(t).

    For a Boltzmann distribution:
        n_lower(t) / n_total(t) = g_lower / U(T(t)) * exp(-E_lower / kT(t))

    The observed intensity integrates over the gate:
        I_obs = integral[I_emit(t) * f(tau(t))] dt

    where f(tau) = (1 - exp(-tau)) / tau is the escape factor.

    Parameters
    ----------
    evolution_model : PlasmaEvolutionModel
        Model for T(t) and n_e(t)
    plasma_length_cm : float
        Plasma column length (path length)
    integration_points : int
        Number of time points for gate integration
    """

    def __init__(
        self,
        evolution_model: PlasmaEvolutionModel,
        plasma_length_cm: float = 0.1,
        integration_points: int = 50,
    ):
        self.model = evolution_model
        self.plasma_length_cm = plasma_length_cm
        self.integration_points = integration_points

    def optical_depth_at_time(
        self,
        time_ns: float,
        wavelength_nm: float,
        A_ki: float,
        g_k: int,
        E_lower_eV: float,
        concentration: float,
        total_number_density_cm3: float,
        partition_func: float,
    ) -> float:
        """
        Calculate optical depth at a specific time.

        Parameters
        ----------
        time_ns : float
            Time after laser pulse (ns)
        wavelength_nm : float
            Line wavelength (nm)
        A_ki : float
            Einstein A coefficient (s^-1)
        g_k : int
            Upper level statistical weight
        E_lower_eV : float
            Lower level energy (eV)
        concentration : float
            Species concentration (mass fraction)
        total_number_density_cm3 : float
            Total number density (cm^-3)
        partition_func : float
            Partition function at this time

        Returns
        -------
        float
            Optical depth at line center
        """
        T_K = self.model.temperature(time_ns)
        T_eV = T_K * KB_EV

        if T_eV <= 0 or partition_func <= 0:
            return 0.0

        # Species number density
        n_s = concentration * total_number_density_cm3

        # Lower level population
        g_lower = g_k  # Approximate as similar to upper level
        exp_factor = np.exp(-E_lower_eV / T_eV)
        n_lower = n_s * (g_lower / partition_func) * exp_factor

        # Wavelength in cm
        lambda_cm = wavelength_nm * 1e-7

        # Simplified optical depth scaling
        # tau ~ n_lower * f * lambda^2 * L
        # f ~ A_ki * lambda^2 (rough scaling)
        SCALE_FACTOR = 1e-25
        tau = SCALE_FACTOR * A_ki * (lambda_cm**3) * n_lower * self.plasma_length_cm

        return max(0.0, tau)

    def gate_averaged_optical_depth(
        self,
        gate: TemporalGateConfig,
        wavelength_nm: float,
        A_ki: float,
        g_k: int,
        E_lower_eV: float,
        concentration: float,
        total_number_density_cm3: float,
        partition_func_callable: Callable[[float], float],
    ) -> float:
        """
        Calculate gate-averaged optical depth.

        Integrates optical depth over the gate window, weighted by
        emission intensity.

        Parameters
        ----------
        gate : TemporalGateConfig
            Gate timing configuration
        wavelength_nm : float
            Line wavelength (nm)
        A_ki : float
            Einstein A coefficient
        g_k : int
            Upper level statistical weight
        E_lower_eV : float
            Lower level energy (eV)
        concentration : float
            Species concentration
        total_number_density_cm3 : float
            Total number density
        partition_func_callable : Callable
            Function U(T) that returns partition function

        Returns
        -------
        float
            Effective (averaged) optical depth
        """
        times = np.linspace(gate.delay_ns, gate.end_ns, self.integration_points)

        tau_values = []
        weights = []  # Weight by emission intensity

        for t in times:
            T_K = self.model.temperature(t)
            U_T = partition_func_callable(T_K)

            tau = self.optical_depth_at_time(
                t,
                wavelength_nm,
                A_ki,
                g_k,
                E_lower_eV,
                concentration,
                total_number_density_cm3,
                U_T,
            )
            tau_values.append(tau)

            # Emission intensity weight (approx)
            n_e = self.model.electron_density(t)
            T_eV = T_K * KB_EV
            emission_weight = n_e * np.exp(-E_lower_eV / T_eV) if T_eV > 0 else 0
            weights.append(emission_weight)

        tau_values = np.array(tau_values)
        weights = np.array(weights)

        # Weighted average
        if np.sum(weights) > 0:
            return float(np.average(tau_values, weights=weights))
        else:
            return float(np.mean(tau_values))

    def correct_observations(
        self,
        observations: List[LineObservation],
        gate: TemporalGateConfig,
        concentrations: Dict[str, float],
        total_number_density_cm3: float,
        partition_func_callable: Callable[[str, int, float], float],
        lower_level_energies: Optional[Dict[float, float]] = None,
        mask_threshold: float = 3.0,
    ) -> TemporalSelfAbsorptionResult:
        """
        Apply temporal self-absorption correction to observations.

        Parameters
        ----------
        observations : List[LineObservation]
            Line observations to correct
        gate : TemporalGateConfig
            Gate timing used for acquisition
        concentrations : Dict[str, float]
            Element concentrations
        total_number_density_cm3 : float
            Total number density
        partition_func_callable : Callable
            Function(element, ion_stage, T_K) -> U(T)
        lower_level_energies : Dict[float, float], optional
            Lower level energies by wavelength (nm -> eV)
        mask_threshold : float
            Optical depth above which to mask line

        Returns
        -------
        TemporalSelfAbsorptionResult
            Corrected observations with temporal analysis
        """
        if lower_level_energies is None:
            lower_level_energies = {}

        corrected_obs = []
        optical_depths: Dict[float, Dict[float, float]] = {}
        correction_factors: Dict[float, Dict[float, float]] = {}
        time_averaged_tau: Dict[float, float] = {}
        warnings = []

        # Sample times for detailed analysis
        sample_times = np.linspace(gate.delay_ns, gate.end_ns, 5)

        for obs in observations:
            E_lower = lower_level_energies.get(obs.wavelength_nm, 0.0)
            C_s = concentrations.get(obs.element, 0.0)

            # Create partition function wrapper for this element
            def U_func(T_K: float, el=obs.element, stage=obs.ionization_stage) -> float:
                return partition_func_callable(el, stage, T_K)

            # Calculate gate-averaged optical depth
            tau_avg = self.gate_averaged_optical_depth(
                gate,
                obs.wavelength_nm,
                obs.A_ki,
                obs.g_k,
                E_lower,
                C_s,
                total_number_density_cm3,
                U_func,
            )

            time_averaged_tau[obs.wavelength_nm] = tau_avg

            # Calculate tau at each sample time for detailed analysis
            tau_at_times: Dict[float, float] = {}
            for t in sample_times:
                T_K = self.model.temperature(t)
                U_T = U_func(T_K)
                tau_t = self.optical_depth_at_time(
                    t,
                    obs.wavelength_nm,
                    obs.A_ki,
                    obs.g_k,
                    E_lower,
                    C_s,
                    total_number_density_cm3,
                    U_T,
                )
                tau_at_times[float(t)] = tau_t

            optical_depths[obs.wavelength_nm] = tau_at_times

            # Calculate correction factor
            if tau_avg > mask_threshold:
                warnings.append(
                    f"Line {obs.wavelength_nm:.2f} nm masked: " f"time-averaged tau={tau_avg:.2f}"
                )
                correction_factors[obs.wavelength_nm] = {t: 0.0 for t in sample_times}
                continue

            if tau_avg < 0.1:
                # Optically thin - no correction needed
                corrected_obs.append(obs)
                correction_factors[obs.wavelength_nm] = {t: 1.0 for t in sample_times}
            else:
                # Apply correction: f(tau) = (1 - exp(-tau)) / tau
                if tau_avg > 50:
                    f_tau = 1.0 / tau_avg
                else:
                    f_tau = (1.0 - np.exp(-tau_avg)) / tau_avg

                correction = 1.0 / f_tau
                corrected_intensity = obs.intensity * correction

                corrected_obs.append(
                    LineObservation(
                        wavelength_nm=obs.wavelength_nm,
                        intensity=corrected_intensity,
                        intensity_uncertainty=obs.intensity_uncertainty * correction,
                        element=obs.element,
                        ionization_stage=obs.ionization_stage,
                        E_k_ev=obs.E_k_ev,
                        g_k=obs.g_k,
                        A_ki=obs.A_ki,
                    )
                )

                # Calculate correction factor at each time
                factors_at_times: Dict[float, float] = {}
                for t, tau_t in tau_at_times.items():
                    if tau_t < 0.1:
                        factors_at_times[t] = 1.0
                    elif tau_t > 50:
                        factors_at_times[t] = tau_t
                    else:
                        f_t = (1.0 - np.exp(-tau_t)) / tau_t
                        factors_at_times[t] = 1.0 / f_t

                correction_factors[obs.wavelength_nm] = factors_at_times

        return TemporalSelfAbsorptionResult(
            corrected_observations=corrected_obs,
            optical_depths=optical_depths,
            correction_factors=correction_factors,
            time_averaged_tau=time_averaged_tau,
            warnings=warnings,
        )


# =============================================================================
# Time-Resolved CF-LIBS Solver
# =============================================================================


class TimeResolvedCFLIBSSolver:
    """
    Multi-gate CF-LIBS solver combining time-resolved measurements.

    This solver uses multiple gate timings to improve CF-LIBS accuracy by:
    1. Extracting temperature from multiple time points
    2. Weighting gates by their quality (LTE, SNR, self-absorption)
    3. Cross-validating results across different plasma conditions

    The approach is particularly useful when single-gate measurements
    suffer from self-absorption at early times or low SNR at late times.

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database for line data
    evolution_model : PlasmaEvolutionModel, optional
        Plasma evolution model (will be estimated if not provided)
    base_solver : IterativeCFLIBSSolver, optional
        Base solver for single-gate analysis
    """

    def __init__(
        self,
        atomic_db,
        evolution_model: Optional[PlasmaEvolutionModel] = None,
        base_solver=None,
    ):
        self.atomic_db = atomic_db
        self.evolution_model = evolution_model
        self.base_solver = base_solver

        # Import here to avoid circular imports
        from cflibs.inversion.solver import IterativeCFLIBSSolver

        if self.base_solver is None:
            self.base_solver = IterativeCFLIBSSolver(atomic_db)

    def solve_single_gate(
        self,
        spectrum: TimeResolvedSpectrum,
        closure_mode: str = "standard",
        **solver_kwargs,
    ) -> Dict:
        """
        Solve CF-LIBS for a single gate timing.

        Parameters
        ----------
        spectrum : TimeResolvedSpectrum
            Time-resolved spectrum at one gate
        closure_mode : str
            Closure mode for solver
        solver_kwargs : dict
            Additional solver arguments

        Returns
        -------
        Dict
            Results including concentrations, temperature, quality metrics
        """
        result = self.base_solver.solve(
            spectrum.observations, closure_mode=closure_mode, **solver_kwargs
        )

        return {
            "gate_delay_ns": spectrum.gate.delay_ns,
            "gate_width_ns": spectrum.gate.width_ns,
            "temperature_K": result.temperature_K,
            "electron_density_cm3": result.electron_density_cm3,
            "concentrations": result.concentrations,
            "converged": result.converged,
            "iterations": result.iterations,
            "snr": spectrum.snr_estimate,
            "continuum": spectrum.continuum_level,
        }

    def _compute_gate_weight(
        self,
        gate_result: Dict,
        expected_T: float,
        expected_ne: float,
    ) -> float:
        """
        Compute quality weight for a gate's results.

        Weights based on:
        - Convergence (binary)
        - Consistency with expected T, n_e from evolution model
        - SNR
        """
        if not gate_result["converged"]:
            return 0.0

        weight = 1.0

        # SNR contribution
        snr = gate_result.get("snr", 50.0)
        weight *= min(1.0, snr / 100.0)

        # Temperature consistency
        T_measured = gate_result["temperature_K"]
        T_error = abs(T_measured - expected_T) / expected_T
        weight *= np.exp(-T_error * 2)  # Penalize large deviations

        # Density consistency
        ne_measured = gate_result["electron_density_cm3"]
        ne_error = abs(ne_measured - expected_ne) / max(expected_ne, 1e10)
        weight *= np.exp(-ne_error)

        return max(0.0, weight)

    def solve_multi_gate(
        self,
        spectra: List[TimeResolvedSpectrum],
        closure_mode: str = "standard",
        weighting: str = "quality",
        **solver_kwargs,
    ) -> TimeResolvedCFLIBSResult:
        """
        Solve CF-LIBS combining multiple gate timings.

        Parameters
        ----------
        spectra : List[TimeResolvedSpectrum]
            Time-resolved spectra at different gates
        closure_mode : str
            Closure mode for solver
        weighting : str
            Gate weighting method: "quality", "uniform", or "snr"
        solver_kwargs : dict
            Additional solver arguments

        Returns
        -------
        TimeResolvedCFLIBSResult
            Combined results from all gates
        """
        if not spectra:
            raise ValueError("No spectra provided")

        # Sort by gate delay
        spectra = sorted(spectra, key=lambda s: s.gate.delay_ns)

        # Solve each gate
        per_gate_results = {}
        for spec in spectra:
            result = self.solve_single_gate(spec, closure_mode, **solver_kwargs)
            per_gate_results[spec.gate.delay_ns] = result

        # Extract temperature and density profiles
        temp_profile = [
            (r["gate_delay_ns"], r["temperature_K"])
            for r in per_gate_results.values()
            if r["converged"]
        ]
        density_profile = [
            (r["gate_delay_ns"], r["electron_density_cm3"])
            for r in per_gate_results.values()
            if r["converged"]
        ]

        # Fit evolution model if not provided
        if self.evolution_model is None and len(temp_profile) >= 2:
            times = np.array([t[0] for t in temp_profile])
            temps = np.array([t[1] for t in temp_profile])
            densities = np.array([d[1] for d in density_profile])
            self.evolution_model = PlasmaEvolutionModel.from_measurements(times, temps, densities)

        # Compute weights
        gate_weights: Dict[float, float] = {}
        for delay, result in per_gate_results.items():
            if weighting == "uniform":
                gate_weights[delay] = 1.0 if result["converged"] else 0.0
            elif weighting == "snr":
                gate_weights[delay] = result.get("snr", 50.0) if result["converged"] else 0.0
            else:  # quality weighting
                if self.evolution_model is not None:
                    expected_T = self.evolution_model.temperature(delay)
                    expected_ne = self.evolution_model.electron_density(delay)
                else:
                    expected_T = result["temperature_K"]
                    expected_ne = result["electron_density_cm3"]
                gate_weights[delay] = self._compute_gate_weight(result, expected_T, expected_ne)

        # Normalize weights
        total_weight = sum(gate_weights.values())
        if total_weight > 0:
            gate_weights = {k: v / total_weight for k, v in gate_weights.items()}
        else:
            # Fallback to uniform weighting
            n_converged = sum(1 for r in per_gate_results.values() if r["converged"])
            if n_converged > 0:
                gate_weights = {
                    k: 1.0 / n_converged if r["converged"] else 0.0
                    for k, r in per_gate_results.items()
                }

        # Weighted average of concentrations
        elements = set()
        for result in per_gate_results.values():
            if result["converged"]:
                elements.update(result["concentrations"].keys())

        concentrations: Dict[str, float] = {}
        concentration_variances: Dict[str, float] = {}

        for el in elements:
            weighted_sum = 0.0
            weighted_sq_sum = 0.0
            total_w = 0.0

            for delay, result in per_gate_results.items():
                if not result["converged"]:
                    continue
                w = gate_weights[delay]
                c = result["concentrations"].get(el, 0.0)
                weighted_sum += w * c
                weighted_sq_sum += w * c * c
                total_w += w

            if total_w > 0:
                mean = weighted_sum / total_w
                concentrations[el] = mean
                # Weighted variance
                variance = max(0.0, weighted_sq_sum / total_w - mean * mean)
                concentration_variances[el] = variance

        # Uncertainties from spread
        concentration_uncertainties = {el: np.sqrt(v) for el, v in concentration_variances.items()}

        # Normalize concentrations
        total_c = sum(concentrations.values())
        if total_c > 0 and abs(total_c - 1.0) > 0.01:
            concentrations = {el: c / total_c for el, c in concentrations.items()}

        # Find best gate
        best_delay = max(gate_weights.keys(), key=lambda k: gate_weights[k])
        best_spec = next(s for s in spectra if s.gate.delay_ns == best_delay)

        # Quality metric: average weight
        combined_quality = np.mean(list(gate_weights.values()))

        logger.info(
            f"Multi-gate CF-LIBS: {len(spectra)} gates, "
            f"best={best_delay:.0f} ns, quality={combined_quality:.3f}"
        )

        return TimeResolvedCFLIBSResult(
            concentrations=concentrations,
            concentration_uncertainties=concentration_uncertainties,
            temperature_profile=temp_profile,
            density_profile=density_profile,
            gate_weights=gate_weights,
            per_gate_results=per_gate_results,
            combined_quality=combined_quality,
            optimal_gate=best_spec.gate,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_default_evolution_model(
    material_type: str = "metal",
    laser_energy_mJ: float = 50.0,
) -> PlasmaEvolutionModel:
    """
    Create a default plasma evolution model based on material type.

    Parameters
    ----------
    material_type : str
        Type of material: "metal", "ceramic", "polymer", "soil"
    laser_energy_mJ : float
        Laser pulse energy in mJ

    Returns
    -------
    PlasmaEvolutionModel
        Configured evolution model
    """
    # Energy scaling: higher energy -> higher initial T and n_e
    energy_factor = laser_energy_mJ / 50.0

    defaults = {
        "metal": {
            "T_initial_K": 20000.0 * energy_factor**0.3,
            "ne_initial_cm3": 1e18 * energy_factor**0.5,
            "tau_T_ns": 1000.0,
            "tau_ne_ns": 500.0,
        },
        "ceramic": {
            "T_initial_K": 15000.0 * energy_factor**0.3,
            "ne_initial_cm3": 5e17 * energy_factor**0.5,
            "tau_T_ns": 800.0,
            "tau_ne_ns": 400.0,
        },
        "polymer": {
            "T_initial_K": 12000.0 * energy_factor**0.3,
            "ne_initial_cm3": 2e17 * energy_factor**0.5,
            "tau_T_ns": 600.0,
            "tau_ne_ns": 300.0,
        },
        "soil": {
            "T_initial_K": 18000.0 * energy_factor**0.3,
            "ne_initial_cm3": 8e17 * energy_factor**0.5,
            "tau_T_ns": 900.0,
            "tau_ne_ns": 450.0,
        },
    }

    params = defaults.get(material_type.lower(), defaults["metal"])

    return PlasmaEvolutionModel(**params)


def recommend_gate_timing(
    material_type: str = "metal",
    laser_energy_mJ: float = 50.0,
    priority: str = "balanced",
) -> TemporalGateConfig:
    """
    Recommend gate timing based on material and priorities.

    Parameters
    ----------
    material_type : str
        Type of material
    laser_energy_mJ : float
        Laser pulse energy in mJ
    priority : str
        Analysis priority: "lte" (maximize LTE), "signal" (maximize signal),
        "balanced" (balance all factors)

    Returns
    -------
    TemporalGateConfig
        Recommended gate configuration
    """
    model = create_default_evolution_model(material_type, laser_energy_mJ)

    # Set weights based on priority
    if priority == "lte":
        optimizer = GateTimingOptimizer(
            model, lte_weight=2.0, signal_weight=0.3, self_absorption_weight=1.0
        )
    elif priority == "signal":
        optimizer = GateTimingOptimizer(
            model, lte_weight=0.8, signal_weight=1.5, self_absorption_weight=0.5
        )
    else:  # balanced
        optimizer = GateTimingOptimizer(model)

    result = optimizer.optimize()

    return TemporalGateConfig(
        delay_ns=result.optimal_delay_ns,
        width_ns=result.optimal_width_ns,
        label=f"optimal_{priority}",
    )
