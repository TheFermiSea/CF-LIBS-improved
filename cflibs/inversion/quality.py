"""
Quality metrics for CF-LIBS analysis.

Provides objective measures to assess analysis quality and flag unreliable results.
"""

from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from cflibs.core.constants import KB_EV, EV_TO_K, SAHA_CONST_CM3
from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation, BoltzmannPlotFitter

logger = get_logger("inversion.quality")


@dataclass
class QualityMetrics:
    """
    Quality metrics for a CF-LIBS analysis.

    Thresholds:
    - r_squared_boltzmann: >0.95 excellent, >0.90 good, >0.80 acceptable, <0.80 poor
    - saha_boltzmann_consistency: <0.10 excellent, <0.20 good, <0.30 acceptable
    - inter_element_t_std_frac: <0.05 excellent, <0.10 good, <0.15 acceptable
    - closure_residual: <0.01 excellent, <0.05 good, <0.10 acceptable
    """

    # Boltzmann fit quality
    r_squared_boltzmann: float
    r_squared_by_element: Dict[str, float] = field(default_factory=dict)

    # Temperature consistency
    temperature_by_element: Dict[str, float] = field(default_factory=dict)
    inter_element_t_std_K: float = 0.0
    inter_element_t_std_frac: float = 0.0  # std/mean

    # Saha-Boltzmann consistency (relative difference)
    saha_boltzmann_consistency: float = 0.0
    t_boltzmann_K: float = 0.0
    t_saha_K: float = 0.0

    # Closure quality
    closure_residual: float = 0.0  # |sum(C) - 1.0|

    # Reconstruction quality
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    n_degrees_freedom: int = 0

    # Overall assessment
    quality_flag: str = "unknown"  # "excellent", "good", "acceptable", "poor", "reject"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "r_squared_boltzmann": self.r_squared_boltzmann,
            "inter_element_t_std_frac": self.inter_element_t_std_frac,
            "saha_boltzmann_consistency": self.saha_boltzmann_consistency,
            "closure_residual": self.closure_residual,
            "reduced_chi_squared": self.reduced_chi_squared,
            "quality_flag": self.quality_flag,
        }


class QualityAssessor:
    """
    Assesses quality of CF-LIBS analysis results.
    """

    # Thresholds for quality flags
    THRESHOLDS = {
        "r_squared": {"excellent": 0.95, "good": 0.90, "acceptable": 0.80},
        "saha_consistency": {"excellent": 0.10, "good": 0.20, "acceptable": 0.30},
        "t_std_frac": {"excellent": 0.05, "good": 0.10, "acceptable": 0.15},
        "closure": {"excellent": 0.01, "good": 0.05, "acceptable": 0.10},
        "reduced_chi2": {"excellent": 1.5, "good": 2.0, "acceptable": 3.0},
    }

    def __init__(
        self,
        r_squared_weight: float = 1.0,
        consistency_weight: float = 1.0,
        closure_weight: float = 1.0,
    ):
        """
        Initialize quality assessor.

        Parameters
        ----------
        r_squared_weight : float
            Weight for R² in overall assessment
        consistency_weight : float
            Weight for Saha-Boltzmann consistency
        closure_weight : float
            Weight for closure residual
        """
        self.r_squared_weight = r_squared_weight
        self.consistency_weight = consistency_weight
        self.closure_weight = closure_weight
        self.fitter = BoltzmannPlotFitter(outlier_sigma=2.5)

    @staticmethod
    def _resolve_neutral_ion_stages(stage_keys: List[int]) -> Optional[Tuple[int, int]]:
        """
        Resolve neutral/ion stage pair from observed stage keys.

        Supports both common conventions:
        - 0/1 (neutral/ion)
        - 1/2 (neutral/ion)
        """
        stage_set = set(stage_keys)
        if 0 in stage_set and 1 in stage_set:
            return (0, 1)
        if 1 in stage_set and 2 in stage_set:
            return (1, 2)
        if not stage_set:
            return None
        base = min(stage_set)
        if base + 1 in stage_set:
            return (base, base + 1)
        return None

    def assess(
        self,
        observations: List[LineObservation],
        temperature_K: float,
        electron_density_cm3: float,
        concentrations: Dict[str, float],
        ionization_potentials: Dict[str, float],
        partition_funcs_I: Dict[str, float],
        partition_funcs_II: Dict[str, float],
    ) -> QualityMetrics:
        """
        Compute quality metrics for a CF-LIBS result.

        Parameters
        ----------
        observations : List[LineObservation]
            Input line observations
        temperature_K : float
            Fitted temperature
        electron_density_cm3 : float
            Fitted electron density
        concentrations : Dict[str, float]
            Fitted concentrations (should sum to ~1.0)
        ionization_potentials : Dict[str, float]
            IP for each element (eV)
        partition_funcs_I : Dict[str, float]
            Neutral partition functions U_I(T)
        partition_funcs_II : Dict[str, float]
            Ion partition functions U_II(T)

        Returns
        -------
        QualityMetrics
        """
        warnings = []

        # 1. R² of pooled Boltzmann fit
        r_squared_pooled = self._compute_pooled_r_squared(
            observations,
            temperature_K,
            electron_density_cm3,
            ionization_potentials,
            partition_funcs_I,
            partition_funcs_II,
        )

        # 2. R² per element and inter-element T consistency
        r_squared_by_element, temp_by_element = self._compute_per_element_fits(observations)

        t_values = np.asarray(list(temp_by_element.values()))
        if len(t_values) > 1:
            t_std = float(np.std(t_values))
            t_mean = float(np.mean(t_values))
            t_std_frac = t_std / t_mean if t_mean > 0 else 0.0
        else:
            t_std = 0.0
            t_std_frac = 0.0

        # 3. Saha-Boltzmann consistency
        # Compare T from slope to T implied by Saha ratios
        saha_consistency, t_saha = self._compute_saha_consistency(
            observations,
            temperature_K,
            electron_density_cm3,
            ionization_potentials,
            partition_funcs_I,
            partition_funcs_II,
        )

        # 4. Closure residual
        total_conc = sum(concentrations.values())
        closure_residual = abs(total_conc - 1.0)

        if closure_residual > 0.05:
            warnings.append(f"Closure residual {closure_residual:.3f} > 0.05")

        # 5. Determine overall quality flag
        quality_flag = self._determine_quality_flag(
            r_squared_pooled, saha_consistency, t_std_frac, closure_residual
        )

        return QualityMetrics(
            r_squared_boltzmann=r_squared_pooled,
            r_squared_by_element=r_squared_by_element,
            temperature_by_element=temp_by_element,
            inter_element_t_std_K=t_std,
            inter_element_t_std_frac=t_std_frac,
            saha_boltzmann_consistency=saha_consistency,
            t_boltzmann_K=temperature_K,
            t_saha_K=t_saha,
            closure_residual=closure_residual,
            quality_flag=quality_flag,
            warnings=warnings,
        )

    def _compute_pooled_r_squared(
        self,
        observations: List[LineObservation],
        temperature_K: float,
        electron_density_cm3: float,
        ionization_potentials: Dict[str, float],
        partition_funcs_I: Dict[str, float],
        partition_funcs_II: Dict[str, float],
    ) -> float:
        """Compute R² for pooled Boltzmann fit with Saha correction."""
        if len(observations) < 3:
            return 0.0

        T_eV = temperature_K / EV_TO_K
        safe_ne_cm3 = max(float(electron_density_cm3), 1e10)
        obs_by_element_stage: DefaultDict[str, DefaultDict[int, List[LineObservation]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for obs in observations:
            obs_by_element_stage[obs.element][obs.ionization_stage].append(obs)

        stage_pairs: Dict[str, Tuple[int, int]] = {}
        for element, stages in obs_by_element_stage.items():
            pair = self._resolve_neutral_ion_stages(list(stages.keys()))
            if pair is not None:
                stage_pairs[element] = pair

        # Apply Saha corrections and collect points
        x_all = []
        y_all = []

        for obs in observations:
            el = obs.element
            ip = ionization_potentials.get(el, 15.0)

            # Match the classic solver's Saha-Boltzmann transform:
            # move the ionization energy onto x while subtracting only the
            # ionic partition/electron-density prefactor from y.
            saha_offset = np.log((SAHA_CONST_CM3 / safe_ne_cm3) * (T_eV**1.5))
            stage_pair = stage_pairs.get(el)
            ion_stage = stage_pair[1] if stage_pair is not None else None
            correction = (
                saha_offset if ion_stage is not None and obs.ionization_stage == ion_stage else 0.0
            )

            y = obs.y_value
            if not np.isfinite(y):
                continue

            # Apply correction
            if ion_stage is not None and obs.ionization_stage == ion_stage:
                y -= correction
                x = obs.E_k_ev + ip
            else:
                x = obs.E_k_ev

            x_all.append(x)
            y_all.append(y)

        if len(x_all) < 3:
            return 0.0

        x_ary: np.ndarray = np.array(x_all)
        y_ary: np.ndarray = np.array(y_all)

        # Expected slope from fitted T
        expected_slope = -1.0 / (KB_EV * temperature_K)

        # Fit intercept with fixed slope
        # y = m*x + c => c = mean(y - m*x)
        intercept = np.mean(y_ary - expected_slope * x_ary)
        y_pred = expected_slope * x_ary + intercept

        # R² calculation
        ss_res = np.sum((y_ary - y_pred) ** 2)
        ss_tot = np.sum((y_ary - np.mean(y_ary)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return max(0.0, r_squared)

    def _compute_per_element_fits(
        self,
        observations: List[LineObservation],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute R² and T for each element separately."""

        obs_by_element = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        r_squared_by_element = {}
        temp_by_element = {}

        for element, obs_list in obs_by_element.items():
            if len(obs_list) < 2:
                continue

            try:
                result = self.fitter.fit(obs_list)
                r_squared_by_element[element] = result.r_squared
                if np.isfinite(result.temperature_K) and result.temperature_K > 0:
                    temp_by_element[element] = result.temperature_K
            except Exception as e:
                logger.debug(f"Fit failed for {element}: {e}")

        return r_squared_by_element, temp_by_element

    def _compute_saha_consistency(
        self,
        observations: List[LineObservation],
        temperature_K: float,
        electron_density_cm3: float,
        ionization_potentials: Dict[str, float],
        partition_funcs_I: Dict[str, float],
        partition_funcs_II: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Check Saha-Boltzmann consistency.

        For elements with both neutral (I) and singly-ionized (II) lines,
        estimate the temperature implied by the observed intensity ratio
        via the Saha equation and compare to the Boltzmann-fitted T.

        The Saha ratio at temperature T is:

            S(T) = (C_Saha / n_e) * T_eV^1.5 * exp(-IP / T_eV) * 2 * U_II / U_I

        The observed II/I intensity ratio is proportional to S(T), so we
        solve for T_saha by bisection on:

            R_obs = mean(I_II) / mean(I_I) ~ S(T_saha)

        Parameters
        ----------
        observations : List[LineObservation]
            Line observations
        temperature_K : float
            Boltzmann-fitted temperature in K
        electron_density_cm3 : float
            Electron density in cm^-3
        ionization_potentials : Dict[str, float]
            Ionization potentials for each element in eV
        partition_funcs_I : Dict[str, float]
            Neutral partition functions U_I(T)
        partition_funcs_II : Dict[str, float]
            Ion partition functions U_II(T)

        Returns
        -------
        Tuple[float, float]
            (relative_difference, T_saha_estimate)
        """
        # Group by element and ionization stage
        obs_by_element_stage: DefaultDict[str, DefaultDict[int, List[LineObservation]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for obs in observations:
            obs_by_element_stage[obs.element][obs.ionization_stage].append(obs)

        t_saha_estimates = []

        for element, stages in obs_by_element_stage.items():
            pair = self._resolve_neutral_ion_stages(list(stages.keys()))
            if pair is None:
                continue
            neutral_stage, ion_stage = pair

            ip = ionization_potentials.get(element)
            U_I = partition_funcs_I.get(element)
            U_II = partition_funcs_II.get(element)
            if ip is None or U_I is None or U_II is None:
                continue
            if U_I <= 0 or U_II <= 0:
                continue

            # Observed intensity ratio (ion / neutral)
            I_neutral = float(np.mean(np.asarray([obs.intensity for obs in stages[neutral_stage]])))
            I_ion = float(np.mean(np.asarray([obs.intensity for obs in stages[ion_stage]])))
            if I_neutral <= 0 or I_ion <= 0:
                continue
            R_obs = I_ion / I_neutral

            # NOTE: U_I and U_II are evaluated at the input temperature_K as an
            # approximation. For large differences between T_saha and temperature_K,
            # this introduces error. A future improvement could accept callables.
            safe_ne_cm3 = max(float(electron_density_cm3), 1e10)

            # Saha ratio as function of temperature
            def saha_ratio(T_K: float) -> float:
                T_eV_local = T_K * KB_EV
                if T_eV_local <= 0:
                    return 0.0
                S = (
                    (SAHA_CONST_CM3 / safe_ne_cm3)
                    * (T_eV_local**1.5)
                    * np.exp(-ip / T_eV_local)
                    * 2.0
                    * (U_II / U_I)
                )
                return S

            # Bisection to find T_saha where saha_ratio(T) ~ R_obs
            T_lo, T_hi = 3000.0, 50000.0
            S_lo = saha_ratio(T_lo)
            S_hi = saha_ratio(T_hi)

            # Saha ratio is monotonically increasing with T
            if S_hi <= R_obs:
                logger.debug(
                    "Saha consistency clamp high for %s: R_obs=%.3e S_hi=%.3e",
                    element,
                    R_obs,
                    S_hi,
                )
                t_saha_estimates.append(T_hi)
                continue
            if S_lo >= R_obs:
                logger.debug(
                    "Saha consistency clamp low for %s: R_obs=%.3e S_lo=%.3e",
                    element,
                    R_obs,
                    S_lo,
                )
                t_saha_estimates.append(T_lo)
                continue

            max_iterations = 50
            iteration = 0
            while iteration < max_iterations and abs(T_hi - T_lo) >= 10.0:
                T_mid = 0.5 * (T_lo + T_hi)
                S_mid = saha_ratio(T_mid)
                if S_mid < R_obs:
                    T_lo = T_mid
                else:
                    T_hi = T_mid
                iteration += 1

            t_saha_estimates.append(0.5 * (T_lo + T_hi))

        if len(t_saha_estimates) == 0:
            return 0.0, temperature_K

        t_saha = float(np.mean(t_saha_estimates))
        consistency = abs(temperature_K - t_saha) / temperature_K if temperature_K > 0 else 0.0

        return consistency, t_saha

    def _determine_quality_flag(
        self,
        r_squared: float,
        saha_consistency: float,
        t_std_frac: float,
        closure_residual: float,
    ) -> str:
        """Determine overall quality flag from individual metrics."""

        # Check each metric against thresholds
        r2_level = self._get_level("r_squared", r_squared, higher_is_better=True)
        saha_level = self._get_level("saha_consistency", saha_consistency, higher_is_better=False)
        t_level = self._get_level("t_std_frac", t_std_frac, higher_is_better=False)
        closure_level = self._get_level("closure", closure_residual, higher_is_better=False)

        levels = [r2_level, saha_level, t_level, closure_level]
        level_order = ["excellent", "good", "acceptable", "poor", "reject"]

        # Overall quality is the worst of all metrics
        worst_idx = max(level_order.index(level) for level in levels)
        return level_order[worst_idx]

    def _get_level(self, metric: str, value: float, higher_is_better: bool) -> str:
        """Get quality level for a single metric."""
        thresholds = self.THRESHOLDS.get(metric, {})

        if higher_is_better:
            if value >= thresholds.get("excellent", 0.95):
                return "excellent"
            elif value >= thresholds.get("good", 0.90):
                return "good"
            elif value >= thresholds.get("acceptable", 0.80):
                return "acceptable"
            else:
                return "poor"
        else:
            if value <= thresholds.get("excellent", 0.10):
                return "excellent"
            elif value <= thresholds.get("good", 0.20):
                return "good"
            elif value <= thresholds.get("acceptable", 0.30):
                return "acceptable"
            else:
                return "poor"


def compute_reconstruction_chi_squared(
    measured_spectrum: np.ndarray,
    modeled_spectrum: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
) -> Tuple[float, float, int]:
    """
    Compute χ² between measured and modeled spectra.

    Parameters
    ----------
    measured_spectrum : np.ndarray
        Measured spectral intensities
    modeled_spectrum : np.ndarray
        Forward-modeled intensities
    uncertainties : np.ndarray, optional
        Measurement uncertainties (defaults to Poisson: sqrt(I))

    Returns
    -------
    Tuple[float, float, int]
        (chi_squared, reduced_chi_squared, degrees_of_freedom)
    """
    if len(measured_spectrum) != len(modeled_spectrum):
        raise ValueError("Spectrum lengths must match")

    if uncertainties is None:
        # Assume Poisson statistics
        uncertainties = np.sqrt(np.maximum(measured_spectrum, 1.0))

    # Avoid division by zero
    valid = uncertainties > 0
    residuals = (measured_spectrum[valid] - modeled_spectrum[valid]) / uncertainties[valid]

    chi_squared = float(np.sum(residuals**2))
    n_dof = int(np.sum(valid)) - 3  # Subtract for T, n_e, and normalization
    n_dof = max(1, n_dof)

    reduced_chi_squared = chi_squared / n_dof

    return chi_squared, reduced_chi_squared, n_dof
