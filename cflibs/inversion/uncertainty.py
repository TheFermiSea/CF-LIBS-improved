"""
Uncertainty propagation utilities for CF-LIBS classical inversion.

This module provides two complementary approaches for uncertainty quantification:

1. **Analytical propagation** (uncertainties package): Fast, tracks correlations
   via computational graph. Best for quick estimates when noise is Gaussian.

2. **Monte Carlo propagation** (MonteCarloUQ): Full pipeline re-runs with
   perturbed inputs. Captures non-linear effects, non-Gaussian distributions,
   and provides confidence intervals. Best for publication-quality uncertainties.

Why uncertainties over alternatives?
-----------------------------------
- **Scipp**: Doesn't track correlations (critical for CF-LIBS where Boltzmann
  slope and intercept are correlated from the same regression)
- **AutoUncertainties**: No correlation tracking (x - x != 0 exactly)
- **uncertainties**: Automatic correlation tracking via computational graph

Monte Carlo UQ Usage
--------------------
```python
from cflibs.inversion.uncertainty import MonteCarloUQ

# Create Monte Carlo sampler
mc = MonteCarloUQ(solver, n_samples=500, seed=42)

# Run with spectral noise perturbation
result = mc.run(observations, noise_fraction=0.05)

# Access results
print(f"T = {result.T_mean:.0f} +/- {result.T_std:.0f} K")
print(f"95% CI: [{result.T_ci_95[0]:.0f}, {result.T_ci_95[1]:.0f}] K")

# Compare with Bayesian (if available)
comparison = result.compare_with_bayesian(bayesian_result)
```

Dependencies
------------
- Core: numpy
- Analytical propagation: `pip install uncertainties>=3.2.0`
- Parallel execution: `pip install joblib` (optional, falls back to serial)
Install with: `pip install cflibs[uncertainty]`

References
----------
- uncertainties: https://uncertainties-python-package.readthedocs.io/
- Lebigot, E. "uncertainties: a Python package for calculations with uncertainties"
- Tognoni et al. (2010): CF-LIBS state of the art
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from enum import Enum
import numpy as np

from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.uncertainty")

# Type checking imports
if TYPE_CHECKING:
    from uncertainties import UFloat

# Check for uncertainties package
try:
    from uncertainties import ufloat, correlated_values
    from uncertainties import umath

    HAS_UNCERTAINTIES = True
except ImportError:
    HAS_UNCERTAINTIES = False
    ufloat = None
    correlated_values = None
    umath = None


def check_uncertainties_available() -> None:
    """Raise ImportError if uncertainties package is not available."""
    if not HAS_UNCERTAINTIES:
        raise ImportError(
            "uncertainties package required for uncertainty propagation. "
            "Install with: pip install uncertainties>=3.2.0 or pip install cflibs[uncertainty]"
        )


def _validated_abundance_multiplier(
    abundance_multipliers: Optional[Dict[str, float]],
    element: str,
) -> float:
    """Return a finite positive abundance multiplier for an element."""
    multiplier = abundance_multipliers.get(element, 1.0) if abundance_multipliers else 1.0
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"abundance_multipliers[{element!r}] must be finite and positive")
    return float(multiplier)


def create_boltzmann_uncertainties(
    slope: float,
    intercept: float,
    covariance_matrix: Optional[np.ndarray],
    slope_err: Optional[float] = None,
    intercept_err: Optional[float] = None,
) -> Tuple["UFloat", "UFloat"]:
    """
    Create correlated ufloat objects for Boltzmann fit slope and intercept.

    Parameters
    ----------
    slope : float
        Boltzmann plot slope = -1/(kB*T)
    intercept : float
        Boltzmann plot intercept (related to ln(C*F/U))
    covariance_matrix : np.ndarray, optional
        2x2 covariance matrix [[var(slope), cov], [cov, var(intercept)]].
        If provided, creates correlated uncertainties.
    slope_err : float, optional
        Fallback slope uncertainty if covariance_matrix is None
    intercept_err : float, optional
        Fallback intercept uncertainty if covariance_matrix is None

    Returns
    -------
    tuple[UFloat, UFloat]
        Correlated (slope_u, intercept_u) ufloat objects

    Raises
    ------
    ImportError
        If uncertainties package is not installed
    """
    check_uncertainties_available()

    if covariance_matrix is not None:
        # Create correlated values from covariance matrix
        slope_u, intercept_u = correlated_values([slope, intercept], covariance_matrix)
    else:
        # Fall back to independent uncertainties
        s_err = slope_err if slope_err is not None and np.isfinite(slope_err) else 0.0
        i_err = intercept_err if intercept_err is not None and np.isfinite(intercept_err) else 0.0
        slope_u = ufloat(slope, s_err)
        intercept_u = ufloat(intercept, i_err)

    return slope_u, intercept_u


def temperature_from_slope(slope_u: "UFloat") -> "UFloat":
    """
    Calculate temperature with uncertainty from Boltzmann slope.

    For Boltzmann plots in this codebase, the fitted slope is

        m = -1 / (k_B[eV/K] * T[K])

    Parameters
    ----------
    slope_u : UFloat
        Boltzmann slope with uncertainty

    Returns
    -------
    UFloat
        Temperature in Kelvin with propagated uncertainty
    """
    check_uncertainties_available()
    return -1.0 / (KB_EV * slope_u)


def saha_factor_with_uncertainty(
    T_eV_u: "UFloat",
    n_e: float,
    ionization_potential_eV: float,
    U_I: float,
    U_II: float,
    saha_const: float,
) -> "UFloat":
    """
    Calculate Saha factor with uncertainty propagation.

    S = (SAHA_CONST / n_e) * T^1.5 * exp(-IP/T) * (U_II/U_I)

    Note: Currently n_e is treated as exact (no uncertainty).
    For full uncertainty propagation, n_e would need to be a ufloat.

    Parameters
    ----------
    T_eV_u : UFloat
        Temperature in eV with uncertainty
    n_e : float
        Electron density in cm^-3
    ionization_potential_eV : float
        Ionization potential in eV
    U_I : float
        Partition function of neutral species
    U_II : float
        Partition function of ionized species
    saha_const : float
        Saha constant (SAHA_CONST_CM3)

    Returns
    -------
    UFloat
        Saha factor with propagated uncertainty
    """
    check_uncertainties_available()

    S_raw = (saha_const / n_e) * (T_eV_u**1.5) * umath.exp(-ionization_potential_eV / T_eV_u)
    S = S_raw * (U_II / U_I)

    return S


def propagate_through_closure_standard(
    intercepts_u: Dict[str, "UFloat"],
    partition_funcs: Dict[str, float],
    abundance_multipliers: Optional[Dict[str, float]] = None,
) -> Dict[str, "UFloat"]:
    """
    Apply standard closure equation with uncertainty propagation.

    C_s = (U_s * exp(q_s)) / F
    F = sum(U_s * exp(q_s))

    Parameters
    ----------
    intercepts_u : Dict[str, UFloat]
        Boltzmann intercepts with uncertainties for each element
    partition_funcs : Dict[str, float]
        Partition function values U_s(T) for each element
    abundance_multipliers : Optional[Dict[str, float]]
        Optional per-element scaling that maps neutral-plane intercepts back
        to total elemental abundance. Defaults to ``None``, which applies a
        multiplier of 1.0 for every element.

    Returns
    -------
    Dict[str, UFloat]
        Concentrations with propagated uncertainties
    """
    check_uncertainties_available()

    # Calculate relative concentrations with uncertainty
    rel_concentrations_u: Dict[str, "UFloat"] = {}
    for element, q_s_u in intercepts_u.items():
        if element not in partition_funcs:
            logger.warning(f"Missing partition function for {element}")
            continue

        U_s = partition_funcs[element]
        multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
        rel_C_u = multiplier * U_s * umath.exp(q_s_u)
        rel_concentrations_u[element] = rel_C_u

    if not rel_concentrations_u:
        return {}

    # F = sum of relative concentrations (with correlations tracked)
    F_u = sum(rel_concentrations_u.values())

    # Normalize
    concentrations_u = {el: rel_C / F_u for el, rel_C in rel_concentrations_u.items()}

    return concentrations_u


def propagate_through_closure_matrix(
    intercepts_u: Dict[str, "UFloat"],
    partition_funcs: Dict[str, float],
    matrix_element: str,
    matrix_fraction: float,
    abundance_multipliers: Optional[Dict[str, float]] = None,
) -> Dict[str, "UFloat"]:
    """
    Apply matrix-mode closure with uncertainty propagation.

    Matrix element has known concentration, determines F.

    Parameters
    ----------
    intercepts_u : Dict[str, UFloat]
        Boltzmann intercepts with uncertainties
    partition_funcs : Dict[str, float]
        Partition functions
    matrix_element : str
        Element with known concentration
    matrix_fraction : float
        Known concentration of matrix element
    abundance_multipliers : Optional[Dict[str, float]]
        Optional per-element scaling that maps neutral-plane intercepts back
        to total elemental abundance. Defaults to ``None``, which applies a
        multiplier of 1.0 for every element.

    Returns
    -------
    Dict[str, UFloat]
        Concentrations with propagated uncertainties
    """
    check_uncertainties_available()

    if matrix_element not in intercepts_u or matrix_element not in partition_funcs:
        logger.error(f"Matrix element {matrix_element} missing from data")
        return propagate_through_closure_standard(
            intercepts_u,
            partition_funcs,
            abundance_multipliers=abundance_multipliers,
        )

    U_m = partition_funcs[matrix_element]
    q_m_u = intercepts_u[matrix_element]
    matrix_multiplier = _validated_abundance_multiplier(abundance_multipliers, matrix_element)
    rel_C_m_u = matrix_multiplier * U_m * umath.exp(q_m_u)

    # F determined by matrix element
    F_u = rel_C_m_u / matrix_fraction

    # Calculate all concentrations
    concentrations_u: Dict[str, "UFloat"] = {}
    for element, q_s_u in intercepts_u.items():
        if element in partition_funcs:
            U_s = partition_funcs[element]
            multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
            rel_C_u = multiplier * U_s * umath.exp(q_s_u)
            concentrations_u[element] = rel_C_u / F_u

    return concentrations_u


def propagate_through_closure_oxide(
    intercepts_u: Dict[str, "UFloat"],
    partition_funcs: Dict[str, float],
    oxide_stoichiometry: Dict[str, float],
    abundance_multipliers: Optional[Dict[str, float]] = None,
) -> Dict[str, "UFloat"]:
    """
    Apply oxide-mode closure with uncertainty propagation.

    Parameters
    ----------
    intercepts_u : Dict[str, UFloat]
        Boltzmann intercepts with uncertainties for each element.
    partition_funcs : Dict[str, float]
        Partition function values U_s(T) for each element.
    oxide_stoichiometry : Dict[str, float]
        Map of element to oxide conversion factor.
    abundance_multipliers : Optional[Dict[str, float]]
        Optional per-element scaling that maps neutral-plane intercepts back
        to total elemental abundance. Defaults to ``None``, which applies a
        multiplier of 1.0 for every element.

    Returns
    -------
    Dict[str, UFloat]
        Concentrations with propagated uncertainties.
    """
    check_uncertainties_available()

    rel_concentrations_u: Dict[str, "UFloat"] = {}
    total_oxide_rel_u = None

    for element, q_s_u in intercepts_u.items():
        if element not in partition_funcs:
            logger.warning(f"Missing partition function for {element}")
            continue

        U_s = partition_funcs[element]
        multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
        rel_C_u = multiplier * U_s * umath.exp(q_s_u)
        rel_concentrations_u[element] = rel_C_u

        oxide_factor = oxide_stoichiometry.get(element, 1.0)
        weighted_rel = rel_C_u * oxide_factor
        total_oxide_rel_u = (
            weighted_rel if total_oxide_rel_u is None else total_oxide_rel_u + weighted_rel
        )

    if not rel_concentrations_u or total_oxide_rel_u is None:
        return {}

    return {
        element: rel_C_u / total_oxide_rel_u for element, rel_C_u in rel_concentrations_u.items()
    }


def extract_values_and_uncertainties(
    ufloat_dict: Dict[str, "UFloat"],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract nominal values and uncertainties from a dict of ufloats.

    Parameters
    ----------
    ufloat_dict : Dict[str, UFloat]
        Dictionary of ufloat values

    Returns
    -------
    tuple[Dict[str, float], Dict[str, float]]
        (nominal_values, uncertainties) dictionaries
    """
    check_uncertainties_available()

    nominal = {k: v.nominal_value for k, v in ufloat_dict.items()}
    uncertainties = {k: v.std_dev for k, v in ufloat_dict.items()}

    return nominal, uncertainties


# ============================================================================
# Monte Carlo Uncertainty Quantification
# ============================================================================

# Check for joblib (optional for parallel execution)
try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    Parallel = None
    delayed = None


class PerturbationType(Enum):
    """Type of perturbation for Monte Carlo sampling."""

    SPECTRAL_NOISE = "spectral_noise"  # Perturb line intensities
    ATOMIC_DATA = "atomic_data"  # Perturb A_ki values
    COMBINED = "combined"  # Both spectral and atomic


@dataclass
class AtomicDataUncertainty:
    """
    Uncertainty model for atomic data (Einstein A coefficients).

    NIST provides accuracy grades for atomic data:
    - AAA: < 0.3%
    - AA: < 1%
    - A+: < 2%
    - A: < 3%
    - B+: < 7%
    - B: < 10%
    - C+: < 18%
    - C: < 25%
    - D+: < 40%
    - D: < 50%
    - E: > 50%

    Attributes
    ----------
    default_A_uncertainty : float
        Default fractional uncertainty for A_ki (default: 0.10 = 10%, grade B)
    per_line_uncertainties : Dict[float, float], optional
        Uncertainty by wavelength (nm): {wavelength: fractional_uncertainty}
    """

    default_A_uncertainty: float = 0.10
    per_line_uncertainties: Optional[Dict[float, float]] = None

    def get_uncertainty(self, wavelength_nm: float) -> float:
        """Get fractional uncertainty for A_ki at given wavelength."""
        if self.per_line_uncertainties and wavelength_nm in self.per_line_uncertainties:
            return self.per_line_uncertainties[wavelength_nm]
        return self.default_A_uncertainty


@dataclass
class MonteCarloResult:
    """
    Result container for Monte Carlo uncertainty propagation.

    Contains full sample distributions, summary statistics, and confidence
    intervals for all plasma parameters (T, n_e, concentrations).

    Attributes
    ----------
    n_samples : int
        Number of Monte Carlo samples
    n_successful : int
        Number of successful solver runs (converged)
    T_samples : np.ndarray
        Temperature samples [K] (n_successful,)
    ne_samples : np.ndarray
        Electron density samples [cm^-3] (n_successful,)
    concentration_samples : Dict[str, np.ndarray]
        Concentration samples per element {element: (n_successful,)}
    T_mean : float
        Mean temperature [K]
    T_std : float
        Standard deviation of temperature [K]
    T_ci_68 : Tuple[float, float]
        68% confidence interval for T (1-sigma equivalent)
    T_ci_95 : Tuple[float, float]
        95% confidence interval for T (2-sigma equivalent)
    ne_mean : float
        Mean electron density [cm^-3]
    ne_std : float
        Standard deviation of electron density [cm^-3]
    ne_ci_68 : Tuple[float, float]
        68% confidence interval for n_e
    ne_ci_95 : Tuple[float, float]
        95% confidence interval for n_e
    concentrations_mean : Dict[str, float]
        Mean concentrations per element
    concentrations_std : Dict[str, float]
        Standard deviations per element
    concentrations_ci_68 : Dict[str, Tuple[float, float]]
        68% CI per element
    concentrations_ci_95 : Dict[str, Tuple[float, float]]
        95% CI per element
    perturbation_type : PerturbationType
        Type of perturbation used
    seed : int
        Random seed used for reproducibility
    failed_indices : List[int]
        Indices of failed/non-converged runs
    """

    # Sample counts
    n_samples: int
    n_successful: int

    # Raw samples
    T_samples: np.ndarray
    ne_samples: np.ndarray
    concentration_samples: Dict[str, np.ndarray]

    # Temperature statistics
    T_mean: float
    T_std: float
    T_ci_68: Tuple[float, float]
    T_ci_95: Tuple[float, float]

    # Electron density statistics
    ne_mean: float
    ne_std: float
    ne_ci_68: Tuple[float, float]
    ne_ci_95: Tuple[float, float]

    # Concentration statistics
    concentrations_mean: Dict[str, float]
    concentrations_std: Dict[str, float]
    concentrations_ci_68: Dict[str, Tuple[float, float]]
    concentrations_ci_95: Dict[str, Tuple[float, float]]

    # Metadata
    perturbation_type: PerturbationType = PerturbationType.SPECTRAL_NOISE
    seed: int = 42
    failed_indices: List[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of successful Monte Carlo runs."""
        return self.n_successful / self.n_samples if self.n_samples > 0 else 0.0

    @property
    def T_relative_uncertainty(self) -> float:
        """Relative uncertainty in temperature (sigma/mean)."""
        return self.T_std / self.T_mean if self.T_mean > 0 else float("inf")

    @property
    def ne_relative_uncertainty(self) -> float:
        """Relative uncertainty in electron density (sigma/mean)."""
        return self.ne_std / self.ne_mean if self.ne_mean > 0 else float("inf")

    def summary_table(self) -> str:
        """
        Generate a publication-ready summary table.

        Returns
        -------
        str
            Formatted summary table
        """
        lines = [
            "=" * 70,
            "Monte Carlo Uncertainty Quantification Results",
            "=" * 70,
            f"Samples: {self.n_samples} | Successful: {self.n_successful} "
            f"({self.success_rate*100:.1f}%)",
            f"Perturbation: {self.perturbation_type.value}",
            "-" * 70,
            f"{'Parameter':<20} {'Mean':>12} {'Std':>12} {'95% CI':>24}",
            "-" * 70,
            f"{'T [K]':<20} {self.T_mean:>12.0f} {self.T_std:>12.0f} "
            f"[{self.T_ci_95[0]:.0f}, {self.T_ci_95[1]:.0f}]",
            f"{'n_e [cm^-3]':<20} {self.ne_mean:>12.2e} {self.ne_std:>12.2e} "
            f"[{self.ne_ci_95[0]:.2e}, {self.ne_ci_95[1]:.2e}]",
        ]

        lines.append("-" * 70)
        lines.append(f"{'Element':<20} {'Conc.':<12} {'Std':>12} {'95% CI':>24}")
        lines.append("-" * 70)

        for el in sorted(self.concentrations_mean.keys()):
            mean = self.concentrations_mean[el]
            std = self.concentrations_std[el]
            ci = self.concentrations_ci_95.get(el, (mean - 2 * std, mean + 2 * std))
            lines.append(f"{el:<20} {mean:>12.4f} {std:>12.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

        lines.append("=" * 70)
        return "\n".join(lines)

    def correlation_matrix(self) -> Dict[str, Any]:
        """
        Compute correlation matrix between T, n_e, and concentrations.

        Returns
        -------
        dict
            Contains 'matrix', 'labels', and 'T_ne_corr'
        """
        if self.n_successful < 3:
            return {"matrix": None, "labels": [], "T_ne_corr": float("nan")}

        # Build data matrix
        labels = ["T", "n_e"]
        data = [self.T_samples, self.ne_samples]

        for el in sorted(self.concentrations_mean.keys()):
            labels.append(f"C_{el}")
            data.append(self.concentration_samples[el])

        data_matrix = np.vstack(data).T
        corr = np.corrcoef(data_matrix.T)

        return {
            "matrix": corr,
            "labels": labels,
            "T_ne_corr": float(corr[0, 1]),
        }

    def compare_with_bayesian(self, bayesian_result: Any, tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Compare Monte Carlo results with Bayesian credible intervals.

        This is useful for validating that the Monte Carlo and Bayesian
        approaches give consistent results. Large discrepancies may indicate
        issues with priors, noise models, or non-convergence.

        Parameters
        ----------
        bayesian_result : MCMCResult or NestedSamplingResult
            Result from Bayesian inference
        tolerance : float
            Fractional tolerance for agreement (default: 0.1 = 10%)

        Returns
        -------
        dict
            Comparison metrics including:
            - 'T_agreement': bool
            - 'ne_agreement': bool
            - 'T_difference': float (fractional)
            - 'ne_difference': float (fractional)
            - 'ci_overlap_T': bool
            - 'ci_overlap_ne': bool
            - 'summary': str
        """
        from cflibs.core.constants import EV_TO_K

        # Get Bayesian values (handle both MCMCResult and NestedSamplingResult)
        bayes_T_mean = getattr(bayesian_result, "T_eV_mean", 1.0) * EV_TO_K
        bayes_T_std = getattr(bayesian_result, "T_eV_std", 0.1) * EV_TO_K
        bayes_ne_mean = getattr(bayesian_result, "n_e_mean", 1e17)

        # Handle log_ne for nested sampling
        if hasattr(bayesian_result, "log_ne_mean"):
            bayes_log_ne_std = getattr(bayesian_result, "log_ne_std", 0.1)
            # Propagate uncertainty from log to linear scale
            bayes_ne_std = bayes_ne_mean * bayes_log_ne_std * np.log(10)
        else:
            bayes_ne_std = bayes_ne_mean * 0.1  # Fallback

        # Compare means
        T_diff = abs(self.T_mean - bayes_T_mean) / bayes_T_mean
        ne_diff = abs(self.ne_mean - bayes_ne_mean) / bayes_ne_mean

        T_agreement = T_diff <= tolerance
        ne_agreement = ne_diff <= tolerance

        # Check CI overlap
        mc_T_lower, mc_T_upper = self.T_ci_95
        bayes_T_lower = bayes_T_mean - 2 * bayes_T_std
        bayes_T_upper = bayes_T_mean + 2 * bayes_T_std
        ci_overlap_T = not (mc_T_upper < bayes_T_lower or mc_T_lower > bayes_T_upper)

        mc_ne_lower, mc_ne_upper = self.ne_ci_95
        bayes_ne_lower = bayes_ne_mean - 2 * bayes_ne_std
        bayes_ne_upper = bayes_ne_mean + 2 * bayes_ne_std
        ci_overlap_ne = not (mc_ne_upper < bayes_ne_lower or mc_ne_lower > bayes_ne_upper)

        # Summary
        status = (
            "AGREE"
            if (T_agreement and ne_agreement and ci_overlap_T and ci_overlap_ne)
            else "DISAGREE"
        )
        summary = (
            f"Monte Carlo vs Bayesian: {status}\n"
            f"  T: MC={self.T_mean:.0f}+/-{self.T_std:.0f} K, "
            f"Bayes={bayes_T_mean:.0f}+/-{bayes_T_std:.0f} K "
            f"(diff={T_diff*100:.1f}%)\n"
            f"  n_e: MC={self.ne_mean:.2e}+/-{self.ne_std:.2e}, "
            f"Bayes={bayes_ne_mean:.2e}+/-{bayes_ne_std:.2e} "
            f"(diff={ne_diff*100:.1f}%)"
        )

        return {
            "T_agreement": T_agreement,
            "ne_agreement": ne_agreement,
            "T_difference": T_diff,
            "ne_difference": ne_diff,
            "ci_overlap_T": ci_overlap_T,
            "ci_overlap_ne": ci_overlap_ne,
            "summary": summary,
        }


class MonteCarloUQ:
    """
    Monte Carlo uncertainty propagation for CF-LIBS.

    This class performs uncertainty quantification by:
    1. Perturbing input line observations (intensities and/or atomic data)
    2. Running the CF-LIBS solver on each perturbed dataset
    3. Collecting parameter distributions
    4. Computing statistics and confidence intervals

    Advantages over analytical propagation:
    - Captures non-linear effects in the CF-LIBS algorithm
    - Handles non-Gaussian error distributions
    - Provides full parameter distributions, not just variances
    - More robust for highly correlated parameters

    Parameters
    ----------
    solver : IterativeCFLIBSSolver
        CF-LIBS solver instance
    n_samples : int
        Number of Monte Carlo samples (default: 200)
    seed : int
        Random seed for reproducibility (default: 42)
    n_jobs : int
        Number of parallel jobs (-1 for all CPUs, 1 for serial)
    verbose : bool
        Print progress information

    Example
    -------
    >>> from cflibs.inversion.solver import IterativeCFLIBSSolver
    >>> from cflibs.inversion.uncertainty import MonteCarloUQ
    >>>
    >>> solver = IterativeCFLIBSSolver(atomic_db)
    >>> mc = MonteCarloUQ(solver, n_samples=500)
    >>>
    >>> # Run with 5% spectral noise
    >>> result = mc.run(observations, noise_fraction=0.05)
    >>> print(result.summary_table())
    """

    def __init__(
        self,
        solver: Any,  # IterativeCFLIBSSolver
        n_samples: int = 200,
        seed: int = 42,
        n_jobs: int = 1,
        verbose: bool = True,
    ):
        self.solver = solver
        self.n_samples = n_samples
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose

        if n_jobs != 1 and not HAS_JOBLIB:
            logger.warning("joblib not available, falling back to serial execution")
            self.n_jobs = 1

    def run(
        self,
        observations: List[Any],  # List[LineObservation]
        noise_fraction: Optional[float] = None,
        atomic_uncertainty: Optional[AtomicDataUncertainty] = None,
        perturbation_type: PerturbationType = PerturbationType.SPECTRAL_NOISE,
        closure_mode: str = "standard",
        **closure_kwargs,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo uncertainty propagation.

        Parameters
        ----------
        observations : List[LineObservation]
            Original line observations
        noise_fraction : float, optional
            Fractional noise to add to intensities (e.g., 0.05 = 5%).
            If None, uses the intensity_uncertainty from each observation.
        atomic_uncertainty : AtomicDataUncertainty, optional
            Uncertainty model for A_ki values. Required if perturbation_type
            includes ATOMIC_DATA.
        perturbation_type : PerturbationType
            Type of perturbation (SPECTRAL_NOISE, ATOMIC_DATA, or COMBINED)
        closure_mode : str
            Closure mode for solver ('standard', 'matrix', 'oxide')
        **closure_kwargs
            Additional arguments for closure equation

        Returns
        -------
        MonteCarloResult
            Results with samples, statistics, and confidence intervals
        """
        rng = np.random.default_rng(self.seed)

        # Validate inputs
        if perturbation_type in (PerturbationType.ATOMIC_DATA, PerturbationType.COMBINED):
            if atomic_uncertainty is None:
                atomic_uncertainty = AtomicDataUncertainty()
                logger.info("Using default atomic data uncertainty (10%)")

        # Generate perturbed observations for all samples
        perturbed_sets = [
            self._perturb_observations(
                observations,
                rng,
                noise_fraction,
                atomic_uncertainty,
                perturbation_type,
            )
            for _ in range(self.n_samples)
        ]

        # Run solver on each perturbed set
        if self.n_jobs == 1:
            # Serial execution
            results = []
            for i, obs_set in enumerate(perturbed_sets):
                if self.verbose and (i + 1) % 50 == 0:
                    logger.info(f"Monte Carlo sample {i+1}/{self.n_samples}")
                results.append(self._run_single(obs_set, closure_mode, **closure_kwargs))
        else:
            # Parallel execution with joblib
            results = Parallel(n_jobs=self.n_jobs, verbose=10 if self.verbose else 0)(
                delayed(self._run_single)(obs_set, closure_mode, **closure_kwargs)
                for obs_set in perturbed_sets
            )

        # Process results
        return self._process_results(results, perturbation_type)

    def _perturb_observations(
        self,
        observations: List[Any],
        rng: np.random.Generator,
        noise_fraction: Optional[float],
        atomic_uncertainty: Optional[AtomicDataUncertainty],
        perturbation_type: PerturbationType,
    ) -> List[Any]:
        """
        Generate perturbed observations for one Monte Carlo sample.

        Parameters
        ----------
        observations : List[LineObservation]
            Original observations
        rng : np.random.Generator
            Random number generator
        noise_fraction : float, optional
            Fractional noise for intensities
        atomic_uncertainty : AtomicDataUncertainty, optional
            Model for A_ki uncertainties
        perturbation_type : PerturbationType
            Type of perturbation to apply

        Returns
        -------
        List[LineObservation]
            Perturbed observations
        """
        from cflibs.inversion.boltzmann import LineObservation

        perturbed = []
        for obs in observations:
            # Start with original values
            intensity = obs.intensity
            A_ki = obs.A_ki

            # Perturb intensity
            if perturbation_type in (PerturbationType.SPECTRAL_NOISE, PerturbationType.COMBINED):
                if noise_fraction is not None:
                    sigma = noise_fraction * intensity
                else:
                    sigma = obs.intensity_uncertainty

                if sigma > 0:
                    intensity = intensity + rng.normal(0, sigma)
                    # Ensure positive intensity
                    intensity = max(intensity, 1.0)

            # Perturb A_ki
            if perturbation_type in (PerturbationType.ATOMIC_DATA, PerturbationType.COMBINED):
                if atomic_uncertainty is not None:
                    frac_err = atomic_uncertainty.get_uncertainty(obs.wavelength_nm)
                    sigma_A = frac_err * A_ki
                    A_ki = A_ki + rng.normal(0, sigma_A)
                    # Ensure positive A_ki
                    A_ki = max(A_ki, 1.0)

            perturbed.append(
                LineObservation(
                    wavelength_nm=obs.wavelength_nm,
                    intensity=intensity,
                    intensity_uncertainty=obs.intensity_uncertainty,
                    element=obs.element,
                    ionization_stage=obs.ionization_stage,
                    E_k_ev=obs.E_k_ev,
                    g_k=obs.g_k,
                    A_ki=A_ki,
                )
            )

        return perturbed

    def _run_single(
        self,
        observations: List[Any],
        closure_mode: str,
        **closure_kwargs,
    ) -> Optional[Any]:
        """
        Run solver on single perturbed observation set.

        Returns None if solver fails or doesn't converge.
        """
        try:
            result = self.solver.solve(observations, closure_mode, **closure_kwargs)
            if result.converged:
                return result
            else:
                return None
        except Exception as e:
            logger.debug(f"MC sample failed: {e}")
            return None

    def _process_results(
        self,
        results: List[Optional[Any]],
        perturbation_type: PerturbationType,
    ) -> MonteCarloResult:
        """
        Process solver results into MonteCarloResult.

        Parameters
        ----------
        results : List[Optional[CFLIBSResult]]
            Results from each MC sample (None for failed runs)
        perturbation_type : PerturbationType
            Type of perturbation used

        Returns
        -------
        MonteCarloResult
            Aggregated results with statistics
        """
        # Filter successful results
        successful = [r for r in results if r is not None]
        failed_indices = [i for i, r in enumerate(results) if r is None]

        n_successful = len(successful)
        if n_successful < 2:
            logger.warning(
                f"Only {n_successful} successful MC samples - insufficient for statistics"
            )
            return self._empty_result(perturbation_type, failed_indices)

        # Extract samples
        T_samples = np.array([r.temperature_K for r in successful])
        ne_samples = np.array([r.electron_density_cm3 for r in successful])

        # Get elements from first result
        elements = list(successful[0].concentrations.keys())
        concentration_samples = {
            el: np.array([r.concentrations.get(el, 0.0) for r in successful]) for el in elements
        }

        # Compute statistics
        T_mean = float(np.mean(T_samples))
        T_std = float(np.std(T_samples, ddof=1))
        T_ci_68 = (float(np.percentile(T_samples, 16)), float(np.percentile(T_samples, 84)))
        T_ci_95 = (float(np.percentile(T_samples, 2.5)), float(np.percentile(T_samples, 97.5)))

        ne_mean = float(np.mean(ne_samples))
        ne_std = float(np.std(ne_samples, ddof=1))
        ne_ci_68 = (float(np.percentile(ne_samples, 16)), float(np.percentile(ne_samples, 84)))
        ne_ci_95 = (float(np.percentile(ne_samples, 2.5)), float(np.percentile(ne_samples, 97.5)))

        concentrations_mean = {
            el: float(np.mean(samples)) for el, samples in concentration_samples.items()
        }
        concentrations_std = {
            el: float(np.std(samples, ddof=1)) for el, samples in concentration_samples.items()
        }
        concentrations_ci_68 = {
            el: (float(np.percentile(samples, 16)), float(np.percentile(samples, 84)))
            for el, samples in concentration_samples.items()
        }
        concentrations_ci_95 = {
            el: (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
            for el, samples in concentration_samples.items()
        }

        return MonteCarloResult(
            n_samples=self.n_samples,
            n_successful=n_successful,
            T_samples=T_samples,
            ne_samples=ne_samples,
            concentration_samples=concentration_samples,
            T_mean=T_mean,
            T_std=T_std,
            T_ci_68=T_ci_68,
            T_ci_95=T_ci_95,
            ne_mean=ne_mean,
            ne_std=ne_std,
            ne_ci_68=ne_ci_68,
            ne_ci_95=ne_ci_95,
            concentrations_mean=concentrations_mean,
            concentrations_std=concentrations_std,
            concentrations_ci_68=concentrations_ci_68,
            concentrations_ci_95=concentrations_ci_95,
            perturbation_type=perturbation_type,
            seed=self.seed,
            failed_indices=failed_indices,
        )

    def _empty_result(
        self,
        perturbation_type: PerturbationType,
        failed_indices: List[int],
    ) -> MonteCarloResult:
        """Return empty result for when Monte Carlo fails."""
        return MonteCarloResult(
            n_samples=self.n_samples,
            n_successful=0,
            T_samples=np.array([]),
            ne_samples=np.array([]),
            concentration_samples={},
            T_mean=0.0,
            T_std=0.0,
            T_ci_68=(0.0, 0.0),
            T_ci_95=(0.0, 0.0),
            ne_mean=0.0,
            ne_std=0.0,
            ne_ci_68=(0.0, 0.0),
            ne_ci_95=(0.0, 0.0),
            concentrations_mean={},
            concentrations_std={},
            concentrations_ci_68={},
            concentrations_ci_95={},
            perturbation_type=perturbation_type,
            seed=self.seed,
            failed_indices=failed_indices,
        )


def run_monte_carlo_uq(
    solver: Any,
    observations: List[Any],
    n_samples: int = 200,
    noise_fraction: float = 0.05,
    seed: int = 42,
    closure_mode: str = "standard",
    **closure_kwargs,
) -> MonteCarloResult:
    """
    Convenience function to run Monte Carlo UQ.

    This is a simple wrapper around MonteCarloUQ for quick uncertainty
    estimation. For more control, use the MonteCarloUQ class directly.

    Parameters
    ----------
    solver : IterativeCFLIBSSolver
        CF-LIBS solver instance
    observations : List[LineObservation]
        Line observations
    n_samples : int
        Number of Monte Carlo samples (default: 200)
    noise_fraction : float
        Fractional noise for intensity perturbation (default: 0.05)
    seed : int
        Random seed (default: 42)
    closure_mode : str
        Closure mode for solver
    **closure_kwargs
        Additional closure arguments

    Returns
    -------
    MonteCarloResult
        Results with samples and statistics

    Example
    -------
    >>> result = run_monte_carlo_uq(solver, observations, n_samples=500)
    >>> print(f"T = {result.T_mean:.0f} +/- {result.T_std:.0f} K")
    """
    mc = MonteCarloUQ(solver, n_samples=n_samples, seed=seed, verbose=False)
    return mc.run(
        observations,
        noise_fraction=noise_fraction,
        closure_mode=closure_mode,
        **closure_kwargs,
    )
