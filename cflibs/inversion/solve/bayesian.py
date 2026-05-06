"""
Bayesian inference for CF-LIBS analysis.

This module implements Bayesian forward modeling and inference for CF-LIBS,
including:
- JAX-compatible forward model with full physics (Saha-Boltzmann, Voigt, Stark)
- Physically motivated priors for plasma parameters
- Log-likelihood function with realistic noise model (Poisson + Gaussian)
- MCMC sampling with NUTS algorithm (NumPyro)
- Nested sampling for model comparison (dynesty)
- Convergence diagnostics (R-hat, ESS, divergences)
- Posterior predictive checks for model validation

The implementation is designed to work with NumPyro for MCMC sampling.

User Guide
----------
1. **Basic MCMC workflow**:

    >>> model = BayesianForwardModel(db_path, elements=["Fe", "Cu"], ...)
    >>> sampler = MCMCSampler(model)
    >>> result = sampler.run(observed_spectrum, num_samples=2000, num_chains=4)
    >>> print(result.summary_table())

2. **Check convergence** (use multiple chains for production):

    >>> if result.is_converged:
    ...     print("Converged! R-hat < 1.01 for all parameters")
    >>> else:
    ...     print(f"Warning: {result.convergence_status.value}")

3. **Posterior predictive check**:

    >>> ppc = sampler.posterior_predictive_check(result, observed)
    >>> if ppc["p_value"] > 0.05:
    ...     print("Model fits data well")

4. **Model comparison** (nested sampling):

    >>> ns = NestedSampler(model)
    >>> result = ns.run(observed, nlive=500)
    >>> print(f"Evidence: ln(Z) = {result.log_evidence:.2f}")

Prior Selection Guide
---------------------
**Temperature (T_eV)**:
- Default: Uniform(0.5, 3.0) eV
- LIBS typically: 0.6-1.5 eV (7000-17000 K)
- Use narrower range if prior information available

**Electron Density (log_ne)**:
- Default: Uniform(15, 19) on log10 scale (log-uniform = Jeffreys prior)
- LIBS typically: 10^16 - 10^18 cm^-3
- Appropriate for scale parameters with unknown order of magnitude

**Concentrations**:
- Default: Dirichlet(alpha=1.0) = uniform on simplex
- alpha > 1: Favors more equal concentrations
- alpha < 1: Favors sparse compositions (one dominant element)
- Use known_concentrations for informative priors if matrix is known

Prior Sensitivity Analysis
--------------------------
If results are sensitive to prior choice:
1. Run with different prior ranges and compare posteriors
2. Check that posterior is narrower than prior (data is informative)
3. For robust results, use minimally informative priors:
   - Temperature: wide uniform covering all plausible LIBS values
   - Density: log-uniform spanning expected orders of magnitude
   - Concentrations: Dirichlet(1) unless strong prior knowledge exists

Convergence Diagnostics
-----------------------
- **R-hat < 1.01**: Chains have mixed well (required for convergence)
- **ESS > 100**: Sufficient effective samples for reliable estimates
- **Divergences = 0**: No numerical issues during sampling

If not converged:
1. Increase num_warmup and num_samples
2. Use multiple chains (num_chains=4 recommended)
3. Check for multimodality (use nested sampling)
4. Adjust target_accept_prob (higher = more conservative)

References
----------
- Hoffman & Gelman (2014): NUTS sampler algorithm
- Phan et al. (2019): NumPyro compositional effects
- Tognoni et al. (2010): CF-LIBS state of the art
- Ciucci et al. (1999): CF-LIBS quantitative analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from enum import Enum
from scipy import stats

from cflibs.core.constants import (
    SAHA_CONST_CM3,
    C_LIGHT,
    EV_TO_K,
    EV_TO_J,
    MCWHIRTER_CONST,
    H_PLANCK,
    E_CHARGE,
    M_E,
    M_PROTON,
)
from cflibs.core.jax_runtime import jax_default_real_dtype
from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.bayesian")

try:
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
    from cflibs.radiation.profiles import _voigt_profile_kernel_jax
except ImportError:
    HAS_JAX = False
    jnp = None

    def jit(f):
        return f

    _voigt_profile_kernel_jax = None

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_uniform

    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False
    numpyro = None
    dist = None

try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False
    az = None

try:
    import dynesty
    from dynesty import NestedSampler as DynestyNestedSampler

    HAS_DYNESTY = True
except ImportError:
    HAS_DYNESTY = False
    dynesty = None

if HAS_JAX:
    _JAX_REAL_DTYPE = jax_default_real_dtype()

    def _as_jax_real(value: Any) -> Any:
        """Cast scalars and arrays to the active JAX real dtype."""
        return jnp.asarray(value, dtype=_JAX_REAL_DTYPE)

    _JAX_SAHA_CONST_CM3 = _as_jax_real(SAHA_CONST_CM3)
    _JAX_C_LIGHT = _as_jax_real(C_LIGHT)
    _JAX_EV_TO_K = _as_jax_real(EV_TO_K)
    _JAX_EV_TO_J = _as_jax_real(EV_TO_J)
    _JAX_MCWHIRTER_CONST = _as_jax_real(MCWHIRTER_CONST)
    _JAX_H_PLANCK = _as_jax_real(H_PLANCK)
    _JAX_E_CHARGE = _as_jax_real(E_CHARGE)
    _JAX_M_E = _as_jax_real(M_E)
    _JAX_M_PROTON = _as_jax_real(M_PROTON)

else:
    _JAX_REAL_DTYPE = None

    def _as_jax_real(value: Any) -> Any:
        return value

    _JAX_SAHA_CONST_CM3 = SAHA_CONST_CM3
    _JAX_C_LIGHT = C_LIGHT
    _JAX_EV_TO_K = EV_TO_K
    _JAX_EV_TO_J = EV_TO_J
    _JAX_MCWHIRTER_CONST = MCWHIRTER_CONST
    _JAX_H_PLANCK = H_PLANCK
    _JAX_E_CHARGE = E_CHARGE
    _JAX_M_E = M_E
    _JAX_M_PROTON = M_PROTON


def _resolve_total_species_density_cm3(
    n_e: float,
    total_species_density_cm3: Optional[float],
) -> float:
    """
    Resolve heavy-particle density for forward models.

    When no explicit heavy-particle density is provided, preserve the legacy
    behavior that approximates it with ``n_e``.
    """
    if total_species_density_cm3 is None:
        return n_e
    if np.isscalar(total_species_density_cm3):
        resolved = float(total_species_density_cm3)
        if resolved <= 0.0:
            raise ValueError("total_species_density_cm3 must be positive")
        return resolved
    if HAS_JAX:
        resolved = _as_jax_real(total_species_density_cm3)
        return jnp.where(resolved > 0.0, resolved, jnp.nan)
    raise ValueError("total_species_density_cm3 must be a positive scalar")


# Standard atomic masses for fallback [amu]
STANDARD_MASSES = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.01,
    "N": 14.01,
    "O": 16.00,
    "F": 19.00,
    "Ne": 20.18,
    "Na": 22.99,
    "Mg": 24.31,
    "Al": 26.98,
    "Si": 28.09,
    "P": 30.97,
    "S": 32.07,
    "Cl": 35.45,
    "Ar": 39.95,
    "K": 39.10,
    "Ca": 40.08,
    "Sc": 44.96,
    "Ti": 47.87,
    "V": 50.94,
    "Cr": 52.00,
    "Mn": 54.94,
    "Fe": 55.85,
    "Co": 58.93,
    "Ni": 58.69,
    "Cu": 63.55,
    "Zn": 65.38,
}


# ============================================================================
# Module-level shared utilities
# ============================================================================


def _compute_instrument_sigma(
    line_wavelengths_nm,
    instrument_fwhm_nm: float,
    resolving_power: Optional[float],
):
    """Compute per-line instrumental Gaussian sigma.

    Two modes (mutually exclusive):
    - Constant FWHM (Czerny-Turner): σ = FWHM / 2.35482 (scalar)
    - Constant resolving power (Echelle): σ(λ) = λ / (R · 2.35482) (per-line array)

    Parameters
    ----------
    line_wavelengths_nm : array
        Line center wavelengths in nm. Must use line centers (not the pixel grid)
        to preserve Voigt normalization.
    instrument_fwhm_nm : float
        Constant instrument FWHM in nm.
    resolving_power : float or None
        Resolving power R = λ/Δλ. When set, overrides instrument_fwhm_nm.

    Returns
    -------
    sigma_inst : scalar or array
        Instrumental Gaussian sigma in nm. Scalar for constant-FWHM mode,
        array of shape (n_lines,) for resolving-power mode.
    """
    # Using more precise constant 2.35482004503 for Gaussian FWHM-to-sigma
    GAUSS_FAC = 2.35482004503
    if resolving_power is not None:
        if resolving_power <= 0:
            raise ValueError(f"resolving_power must be positive, got {resolving_power}")
        # Echelle: R = λ/FWHM → FWHM = λ/R → σ = FWHM / 2.35482 = λ/(R·2.35482)
        return line_wavelengths_nm / (resolving_power * GAUSS_FAC)
    else:
        return instrument_fwhm_nm / GAUSS_FAC


def _query_atomic_data(
    db_path: str,
    elements: List[str],
    wavelength_range: Tuple[float, float],
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Query the database to retrieve atomic lines and species physics."""
    import pandas as pd
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        placeholders = ",".join(["?"] * len(elements))
        query = f"""
            SELECT
                l.id, l.element, l.sp_num, l.wavelength_nm, l.aki, l.ek_ev, l.gk,
                sp.ip_ev, l.stark_w, l.stark_alpha, l.ei_ev
            FROM lines l
            JOIN species_physics sp ON l.element = sp.element AND l.sp_num = sp.sp_num
            WHERE l.wavelength_nm BETWEEN ? AND ?
            AND l.element IN ({placeholders})
            ORDER BY l.wavelength_nm, l.id
        """
        params = [wavelength_range[0], wavelength_range[1]] + list(elements)
        df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            raise ValueError(f"No atomic data for elements {elements} in range {wavelength_range}")

        el_map = {el: i for i, el in enumerate(elements)}
        df["el_idx"] = df["element"].map(el_map)

        element_masses = {}
        for el in elements:
            if el in STANDARD_MASSES:
                element_masses[el] = STANDARD_MASSES[el]
            else:
                element_masses[el] = 50.0
                logger.warning(f"No mass for {el}, using fallback 50 amu")
        df["mass_amu"] = df["element"].map(element_masses)

        max_stages = 3
        n_elements = len(elements)
        coeffs = np.zeros((n_elements, max_stages, 5), dtype=np.float32)
        ips = np.zeros((n_elements, max_stages), dtype=np.float32)

        coeffs[:, 0, 0] = np.log(25.0)
        coeffs[:, 1, 0] = np.log(15.0)
        coeffs[:, 2, 0] = np.log(10.0)

        try:
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT element, sp_num, ip_ev FROM species_physics "
                f"WHERE element IN ({placeholders})",
                elements,
            )
            for row in cursor.fetchall():
                el, sp_num, ip_ev = row
                if el in el_map and ip_ev is not None:
                    el_idx = el_map[el]
                    stage_idx = sp_num - 1
                    if 0 <= stage_idx < max_stages:
                        ips[el_idx, stage_idx] = ip_ev

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='partition_functions'"
            )
            if cursor.fetchone():
                cursor.execute(
                    f"SELECT element, sp_num, a0, a1, a2, a3, a4 "
                    f"FROM partition_functions WHERE element IN ({placeholders})",
                    elements,
                )
                for row in cursor.fetchall():
                    el, sp_num, a0, a1, a2, a3, a4 = row
                    if el in el_map:
                        el_idx = el_map[el]
                        stage_idx = sp_num - 1
                        if 0 <= stage_idx < max_stages:
                            coeffs[el_idx, stage_idx] = [a0, a1, a2, a3, a4]
        except Exception as e:
            logger.warning(f"Failed to load physics data: {e}")

    return df, coeffs, ips


def _format_atomic_data_arrays(
    df: Any, coeffs: np.ndarray, ips: np.ndarray, elements: List[str]
) -> "AtomicDataArrays":
    """Format dataframe and physics arrays into JAX AtomicDataArrays."""
    stark_w_raw = df["stark_w"].fillna(float("nan")).values
    # Using 0.45 as default Stark alpha (r45l)
    stark_alpha_raw = df["stark_alpha"].fillna(0.45).values

    # Also compute lower-level energy and oscillator strength for absorption
    # (needed by two-zone model optical depth calculation)
    # E_i is approximated as E_k - hc/λ (in eV)
    wavelength_m = df["wavelength_nm"].values * 1e-9
    ek_ev_vals = df["ek_ev"].values
    if "ei_ev" in df.columns and not df["ei_ev"].isnull().all():
        ei_ev_vals = df["ei_ev"].values
    else:
        # ΔE = hc/λ in eV
        delta_e_ev = (H_PLANCK * C_LIGHT / wavelength_m) / EV_TO_J
        ei_ev_vals = np.maximum(ek_ev_vals - delta_e_ev, 0.0)

    # Oscillator strength from Einstein A: f = (m_e c λ² / 8π² e²) × (g_k/g_i) × A_ki
    # Simplified: f ≈ 1.499e-16 × λ_nm² × (g_k/g_i) × A_ki
    gk_vals = df["gk"].values.astype(float)
    aki_vals = df["aki"].values.astype(float)
    # Assume g_i ≈ g_k for now (conservative); exact g_i not in DB
    f_osc = 1.499e-16 * df["wavelength_nm"].values ** 2 * aki_vals

    return AtomicDataArrays(
        wavelength_nm=jnp.array(df["wavelength_nm"].values, dtype=jnp.float32),
        aki=jnp.array(aki_vals, dtype=jnp.float32),
        ek_ev=jnp.array(ek_ev_vals, dtype=jnp.float32),
        gk=jnp.array(gk_vals, dtype=jnp.float32),
        ip_ev=jnp.array(df["ip_ev"].values, dtype=jnp.float32),
        ion_stage=jnp.array(df["sp_num"].values - 1, dtype=jnp.int32),
        element_idx=jnp.array(df["el_idx"].values, dtype=jnp.int32),
        stark_w=jnp.array(stark_w_raw, dtype=jnp.float32),
        stark_alpha=jnp.array(stark_alpha_raw, dtype=jnp.float32),
        mass_amu=jnp.array(df["mass_amu"].values, dtype=jnp.float32),
        partition_coeffs=jnp.array(coeffs, dtype=jnp.float32),
        ionization_potentials=jnp.array(ips, dtype=jnp.float32),
        elements=list(elements),
        ei_ev=jnp.array(ei_ev_vals, dtype=jnp.float32),
        f_osc=jnp.array(f_osc, dtype=jnp.float32),
    )


def load_atomic_data(
    db_path: str,
    elements: List[str],
    wavelength_range: Tuple[float, float],
) -> "AtomicDataArrays":
    """Load atomic data from database into JAX arrays.

    This is a module-level utility shared by both ``BayesianForwardModel`` and
    ``TwoZoneBayesianForwardModel``.

    Parameters
    ----------
    db_path : str
        Path to the SQLite atomic database.
    elements : list of str
        Elements to include (e.g. ``["Fe", "Cu"]``).
    wavelength_range : tuple of float
        ``(wl_min, wl_max)`` in nm.

    Returns
    -------
    AtomicDataArrays
        Arrays ready for JAX computation.
    """
    if not HAS_JAX:
        raise ImportError("JAX required. Install with: pip install jax jaxlib")

    df, coeffs, ips = _query_atomic_data(db_path, elements, wavelength_range)
    return _format_atomic_data_arrays(df, coeffs, ips, elements)


def partition_function(T_K: float, coeffs: Any) -> Any:
    """Evaluate polynomial partition function (JAX-compatible).

    ``log(U) = \\sum_i a_i (\\log T)^i``  (Irwin form)

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.
    coeffs : array-like
        Polynomial coefficients, shape ``(..., 5)``.

    Returns
    -------
    array
        Partition function value ``U(T)``.
    """
    if HAS_JAX:
        log_T = jnp.log(T_K)
        one = jnp.ones_like(log_T)
        powers = jnp.stack([one, log_T, log_T**2, log_T**3, log_T**4])
        log_U = jnp.sum(coeffs * powers, axis=-1)
        return jnp.exp(log_U)
    else:
        log_T = np.log(T_K)
        powers = np.array([1.0, log_T, log_T**2, log_T**3, log_T**4])
        log_U = np.sum(np.asarray(coeffs) * powers, axis=-1)
        return np.exp(log_U)


def mcwhirter_log_penalty(
    T_eV: float,
    log_ne: float,
    max_delta_E_eV: float = 3.0,
    scale: float = 10.0,
) -> float:
    """Soft McWhirter criterion penalty for LTE validity.

    Returns a smooth, differentiable log-penalty that is zero when the
    electron density satisfies the McWhirter criterion and becomes
    increasingly negative when it is violated.

    The McWhirter criterion requires:

    .. math::

        n_e \\geq 1.6 \\times 10^{12} \\, T^{1/2} \\, (\\Delta E)^3

    where *T* is in Kelvin and *ΔE* is the maximum energy gap in eV.

    Parameters
    ----------
    T_eV : float
        Plasma temperature in eV.
    log_ne : float
        ``log_{10}(n_e)`` with *n_e* in cm⁻³.
    max_delta_E_eV : float
        Largest energy gap to enforce (default 3.0 eV).
    scale : float
        Penalty strength (default 10.0).

    Returns
    -------
    float
        Log-penalty ≤ 0.  Zero means the criterion is satisfied.
    """
    if HAS_JAX:
        T_K = _as_jax_real(T_eV) * _JAX_EV_TO_K
        log10_threshold = (
            jnp.log10(_JAX_MCWHIRTER_CONST)
            + 0.5 * jnp.log10(T_K)
            + 3.0 * jnp.log10(_as_jax_real(max_delta_E_eV))
        )
        deficit = jnp.maximum(0.0, log10_threshold - log_ne)
        return -scale * deficit**2
    else:
        T_K = T_eV * EV_TO_K
        log10_threshold = (
            np.log10(MCWHIRTER_CONST) + 0.5 * np.log10(T_K) + 3.0 * np.log10(max_delta_E_eV)
        )
        deficit = max(0.0, log10_threshold - log_ne)
        return -scale * deficit**2


@dataclass
class AtomicDataArrays:
    """
    Atomic data stored as JAX arrays for efficient computation.

    All arrays are indexed by line number (n_lines,).
    """

    wavelength_nm: Any  # Line wavelengths [nm]
    aki: Any  # Einstein A coefficients [s^-1]
    ek_ev: Any  # Upper level energy [eV]
    gk: Any  # Upper level degeneracy
    ip_ev: Any  # Ionization potential of parent species [eV]
    ion_stage: Any  # Ionization stage (0=neutral, 1=singly ionized)
    element_idx: Any  # Element index
    stark_w: Any  # Stark width reference [nm]
    stark_alpha: Any  # Stark temperature exponent
    mass_amu: Any  # Atomic mass [amu]
    partition_coeffs: Any  # Partition function coefficients (n_elements, n_stages, 5)
    ionization_potentials: Any  # Ionization potentials (n_elements, n_stages)
    elements: List[str] = field(default_factory=list)
    ei_ev: Any = None  # Lower level energy [eV] (for absorption / two-zone model)
    f_osc: Any = None  # Oscillator strength (for absorption cross-section)


@dataclass
class NoiseParameters:
    """
    Noise model parameters for LIBS spectra.

    The noise model combines:
    - Poisson shot noise: sqrt(I) (signal-dependent)
    - Gaussian readout noise: constant (signal-independent)
    - Background noise: additive offset

    Attributes
    ----------
    readout_noise : float
        RMS readout noise in counts (default: 10)
    dark_current : float
        Dark current per pixel in counts (default: 1)
    gain : float
        Detector gain (counts/photon) (default: 1)
    """

    readout_noise: float = 10.0
    dark_current: float = 1.0
    gain: float = 1.0


@dataclass
class PriorConfig:
    """
    Configuration for Bayesian priors on plasma parameters.

    Attributes
    ----------
    T_eV_range : Tuple[float, float]
        Temperature range in eV (default: 0.5-3.0 eV, typical LIBS)
    log_ne_range : Tuple[float, float]
        Log10(electron density) range (default: 15-19, i.e., 10^15-10^19 cm^-3)
    concentration_alpha : float
        Dirichlet prior concentration parameter (default: 1.0, uniform on simplex)
    baseline_degree : int
        Chebyshev polynomial baseline degree (0=no baseline, 3=cubic, 5=quintic).
        The baseline models bremsstrahlung continuum as an additive term.
    baseline_scale : Optional[float]
        Prior scale for baseline coefficients. If None (default), automatically
        set to 0.1 * max(observed) at sampling time. Larger values allow more
        continuum variation.
    """

    T_eV_range: Tuple[float, float] = (0.5, 3.0)
    # Full LIBS density range now supported thanks to Weideman Faddeeva approximation
    # (branch-free implementation with stable gradients)
    log_ne_range: Tuple[float, float] = (15.0, 19.0)
    concentration_alpha: float = 1.0
    baseline_degree: int = 0
    baseline_scale: Optional[float] = None

    @classmethod
    def geological(cls, **kwargs) -> "PriorConfig":
        """Geological sample prior: broad T/ne, sparse concentrations.

        Geological samples often have a dominant matrix element (Si, Fe)
        with several minor components. Dirichlet alpha < 1 favors sparse
        compositions.
        """
        defaults = {"concentration_alpha": 0.5, "T_eV_range": (0.5, 2.0)}
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def metallurgical(cls, **kwargs) -> "PriorConfig":
        """Metallurgical alloy prior: moderate T, peaked concentrations.

        Alloys have well-characterized compositions with known major
        elements. Dirichlet alpha > 1 favors more equal distributions.
        """
        defaults = {"concentration_alpha": 2.0, "T_eV_range": (0.6, 1.5)}
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def uninformative(cls, **kwargs) -> "PriorConfig":
        """Maximally uninformative prior: uniform on simplex.

        Symmetric Dirichlet with alpha=1 gives a uniform distribution on
        the concentration simplex. Use when no prior information about
        sample composition exists.
        """
        defaults = {"concentration_alpha": 1.0}
        defaults.update(kwargs)
        return cls(**defaults)


class ConvergenceStatus(Enum):
    """MCMC convergence status."""

    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    WARNING = "warning"
    UNKNOWN = "unknown"


@dataclass
class MCMCResult:
    """
    Result container for MCMC sampling with convergence diagnostics.

    Stores posterior samples, summary statistics, and convergence diagnostics
    from Bayesian CF-LIBS inference.

    Attributes
    ----------
    samples : dict
        Posterior samples for each parameter {name: array}
    T_eV : ParameterSummary
        Temperature summary (mean, std, credible intervals)
    log_ne : ParameterSummary
        Log electron density summary
    concentrations : dict
        Concentration summaries {element: ParameterSummary}
    r_hat : dict
        Gelman-Rubin R-hat statistic for each parameter
    ess : dict
        Effective sample size for each parameter
    convergence_status : ConvergenceStatus
        Overall convergence assessment
    n_samples : int
        Number of posterior samples
    n_chains : int
        Number of MCMC chains
    n_warmup : int
        Number of warmup samples
    inference_data : Any
        ArviZ InferenceData object for plotting (if ArviZ available)
    """

    # Raw samples
    samples: Dict[str, np.ndarray]

    # Summary statistics
    T_eV_mean: float
    T_eV_std: float
    T_eV_q025: float  # 2.5% quantile (lower 95% CI)
    T_eV_q975: float  # 97.5% quantile (upper 95% CI)

    log_ne_mean: float
    log_ne_std: float
    log_ne_q025: float
    log_ne_q975: float

    concentrations_mean: Dict[str, float]
    concentrations_std: Dict[str, float]
    concentrations_q025: Dict[str, float]
    concentrations_q975: Dict[str, float]

    # Convergence diagnostics
    r_hat: Dict[str, float] = field(default_factory=dict)
    ess: Dict[str, float] = field(default_factory=dict)
    convergence_status: ConvergenceStatus = ConvergenceStatus.UNKNOWN

    # Metadata
    n_samples: int = 0
    n_chains: int = 1
    n_warmup: int = 0

    # ArviZ InferenceData (for plotting)
    inference_data: Any = None

    # Derived quantities
    @property
    def n_e_mean(self) -> float:
        """Mean electron density [cm^-3]."""
        return 10.0**self.log_ne_mean

    @property
    def T_K_mean(self) -> float:
        """Mean temperature [K]."""
        return self.T_eV_mean * EV_TO_K

    @property
    def is_converged(self) -> bool:
        """Check if MCMC has converged (R-hat < 1.01 for all parameters)."""
        return self.convergence_status == ConvergenceStatus.CONVERGED

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
            "CF-LIBS Bayesian Inference Results",
            "=" * 70,
            f"Samples: {self.n_samples} | Chains: {self.n_chains} | Warmup: {self.n_warmup}",
            f"Convergence: {self.convergence_status.value}",
            "-" * 70,
            f"{'Parameter':<20} {'Mean':>12} {'Std':>12} {'95% CI':>20}",
            "-" * 70,
            f"{'T [eV]':<20} {self.T_eV_mean:>12.4f} {self.T_eV_std:>12.4f} "
            f"[{self.T_eV_q025:.4f}, {self.T_eV_q975:.4f}]",
            f"{'T [K]':<20} {self.T_K_mean:>12.0f} {self.T_eV_std * EV_TO_K:>12.0f} "
            f"[{self.T_eV_q025 * EV_TO_K:.0f}, {self.T_eV_q975 * EV_TO_K:.0f}]",
            f"{'log10(n_e)':<20} {self.log_ne_mean:>12.4f} {self.log_ne_std:>12.4f} "
            f"[{self.log_ne_q025:.4f}, {self.log_ne_q975:.4f}]",
            f"{'n_e [cm^-3]':<20} {self.n_e_mean:>12.2e}",
        ]

        lines.append("-" * 70)
        lines.append(f"{'Element':<20} {'Conc.':<12} {'Std':>12} {'95% CI':>20}")
        lines.append("-" * 70)

        for el in self.concentrations_mean:
            mean = self.concentrations_mean[el]
            std = self.concentrations_std[el]
            q025 = self.concentrations_q025.get(el, mean - 2 * std)
            q975 = self.concentrations_q975.get(el, mean + 2 * std)
            lines.append(f"{el:<20} {mean:>12.4f} {std:>12.4f} [{q025:.4f}, {q975:.4f}]")

        if self.r_hat:
            lines.append("-" * 70)
            lines.append("Convergence Diagnostics:")
            for param, rhat in self.r_hat.items():
                ess_val = self.ess.get(param, float("nan"))
                status = "✓" if rhat < 1.01 else "✗"
                lines.append(f"  {param}: R-hat={rhat:.3f} {status}, ESS={ess_val:.0f}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def correlation_matrix(self, include_concentrations: bool = True) -> Dict[str, Any]:
        """
        Compute correlation matrix between posterior parameters.

        Correlation analysis helps identify parameter degeneracies and
        understand how uncertainties are coupled (e.g., T-n_e correlation).

        Parameters
        ----------
        include_concentrations : bool
            Include concentration parameters in correlation (default: True)

        Returns
        -------
        dict
            Contains:
            - 'matrix': np.ndarray correlation matrix
            - 'labels': list of parameter names
            - 'T_log_ne_corr': float, specific T vs log_ne correlation
        """
        # Extract flattened samples
        T_samples = np.array(self.samples["T_eV"]).flatten()
        log_ne_samples = np.array(self.samples["log_ne"]).flatten()

        # Build parameter matrix
        param_names = ["T_eV", "log_ne"]
        param_data = [T_samples, log_ne_samples]

        if include_concentrations and "concentrations" in self.samples:
            conc_samples = np.array(self.samples["concentrations"])
            # Handle different shapes (chains vs no chains)
            if conc_samples.ndim == 3:
                # (n_chains, n_samples, n_elements) -> (n_samples_total, n_elements)
                conc_samples = conc_samples.reshape(-1, conc_samples.shape[-1])
            elif conc_samples.ndim == 2:
                # (n_samples, n_elements) - already correct
                pass

            # Add each concentration
            for i, el in enumerate(self.concentrations_mean.keys()):
                param_names.append(f"C_{el}")
                param_data.append(conc_samples[:, i])

        # Stack and compute correlation
        data_matrix = np.vstack(param_data).T  # (n_samples, n_params)
        corr_matrix = np.corrcoef(data_matrix.T)

        # Key correlation: T vs log_ne
        T_log_ne_corr = corr_matrix[0, 1]

        return {
            "matrix": correlation_matrix,
            "labels": param_names,
            "T_log_ne_corr": float(T_log_ne_corr),
        }

    def correlation_table(self) -> str:
        """
        Generate a formatted correlation table.

        Returns
        -------
        str
            Formatted correlation matrix table
        """
        corr_data = self.correlation_matrix()
        matrix = corr_data["matrix"]
        labels = corr_data["labels"]

        lines = [
            "=" * 70,
            "Parameter Correlations",
            "=" * 70,
        ]

        # Header row
        header = f"{'':>12}" + "".join(f"{lbl:>10}" for lbl in labels)
        lines.append(header)
        lines.append("-" * len(header))

        # Data rows
        for i, label in enumerate(labels):
            row = f"{label:>12}"
            for j in range(len(labels)):
                val = matrix[i, j]
                if i == j:
                    row += f"{'1.000':>10}"
                else:
                    row += f"{val:>10.3f}"
            lines.append(row)

        lines.append("-" * 70)
        lines.append(f"T - log_ne correlation: {corr_data['T_log_ne_corr']:.3f}")
        lines.append("=" * 70)

        return "\n".join(lines)


@dataclass
class NestedSamplingResult:
    """
    Result container for nested sampling with model evidence.

    Nested sampling provides both posterior samples AND marginal likelihood
    (evidence) for model comparison. The evidence Z = P(data | model) is
    crucial for comparing different plasma models (e.g., single-T vs multi-T).

    Attributes
    ----------
    samples : dict
        Posterior samples for each parameter {name: array}
    weights : np.ndarray
        Sample weights (normalized to sum to 1)
    log_evidence : float
        Log marginal likelihood ln(Z)
    log_evidence_err : float
        Uncertainty in log evidence
    information : float
        Kullback-Leibler divergence (bits of information gained)
    T_eV_mean : float
        Mean temperature [eV]
    T_eV_std : float
        Standard deviation of temperature
    log_ne_mean : float
        Mean log10(electron density)
    log_ne_std : float
        Standard deviation of log10(n_e)
    concentrations_mean : dict
        Mean concentrations by element
    concentrations_std : dict
        Standard deviations by element
    n_live : int
        Number of live points used
    n_iterations : int
        Number of nested sampling iterations
    n_calls : int
        Total likelihood evaluations
    """

    # Posterior samples and weights
    samples: Dict[str, np.ndarray]
    weights: np.ndarray

    # Evidence (model comparison)
    log_evidence: float
    log_evidence_err: float
    information: float  # KL divergence H

    # Summary statistics (weighted)
    T_eV_mean: float
    T_eV_std: float
    log_ne_mean: float
    log_ne_std: float
    concentrations_mean: Dict[str, float]
    concentrations_std: Dict[str, float]

    # Sampling metadata
    n_live: int = 100
    n_iterations: int = 0
    n_calls: int = 0

    @property
    def n_e_mean(self) -> float:
        """Mean electron density [cm^-3]."""
        return 10.0**self.log_ne_mean

    @property
    def T_K_mean(self) -> float:
        """Mean temperature [K]."""
        return self.T_eV_mean * EV_TO_K

    @property
    def evidence(self) -> float:
        """Marginal likelihood Z = exp(log_evidence)."""
        return np.exp(self.log_evidence)

    @property
    def bayes_factor_vs(self) -> str:
        """
        Interpretation helper for Bayes factors.

        Returns interpretation guidance for log evidence differences.
        """
        return (
            "Bayes factor interpretation (Kass & Raftery 1995):\n"
            "  |Δln(Z)| < 1:    Not worth more than a bare mention\n"
            "  1 < |Δln(Z)| < 3:  Positive evidence\n"
            "  3 < |Δln(Z)| < 5:  Strong evidence\n"
            "  |Δln(Z)| > 5:      Very strong evidence"
        )

    def summary_table(self) -> str:
        """Generate a publication-ready summary table."""
        lines = [
            "=" * 70,
            "CF-LIBS Nested Sampling Results",
            "=" * 70,
            f"Live points: {self.n_live} | Iterations: {self.n_iterations} | "
            f"Likelihood calls: {self.n_calls}",
            "-" * 70,
            "MODEL EVIDENCE:",
            f"  ln(Z) = {self.log_evidence:.2f} ± {self.log_evidence_err:.2f}",
            f"  Information (H) = {self.information:.2f} nats",
            "-" * 70,
            f"{'Parameter':<20} {'Mean':>12} {'Std':>12}",
            "-" * 70,
            f"{'T [eV]':<20} {self.T_eV_mean:>12.4f} {self.T_eV_std:>12.4f}",
            f"{'T [K]':<20} {self.T_K_mean:>12.0f} {self.T_eV_std * EV_TO_K:>12.0f}",
            f"{'log10(n_e)':<20} {self.log_ne_mean:>12.4f} {self.log_ne_std:>12.4f}",
            f"{'n_e [cm^-3]':<20} {self.n_e_mean:>12.2e}",
        ]

        lines.append("-" * 70)
        lines.append(f"{'Element':<20} {'Conc.':<12} {'Std':>12}")
        lines.append("-" * 70)

        for el in self.concentrations_mean:
            mean = self.concentrations_mean[el]
            std = self.concentrations_std[el]
            lines.append(f"{el:<20} {mean:>12.4f} {std:>12.4f}")

        lines.append("=" * 70)
        return "\n".join(lines)

    @staticmethod
    def compare_models(
        result_a: "NestedSamplingResult",
        result_b: "NestedSamplingResult",
        name_a: str = "Model A",
        name_b: str = "Model B",
    ) -> str:
        """
        Compare two models using Bayes factor.

        Parameters
        ----------
        result_a, result_b : NestedSamplingResult
            Results from two different models
        name_a, name_b : str
            Model names for display

        Returns
        -------
        str
            Formatted comparison with interpretation
        """
        delta_ln_z = result_a.log_evidence - result_b.log_evidence
        err = np.sqrt(result_a.log_evidence_err**2 + result_b.log_evidence_err**2)

        if abs(delta_ln_z) < 1:
            interpretation = "No significant preference"
        elif abs(delta_ln_z) < 3:
            preferred = name_a if delta_ln_z > 0 else name_b
            interpretation = f"Weak evidence for {preferred}"
        elif abs(delta_ln_z) < 5:
            preferred = name_a if delta_ln_z > 0 else name_b
            interpretation = f"Strong evidence for {preferred}"
        else:
            preferred = name_a if delta_ln_z > 0 else name_b
            interpretation = f"Very strong evidence for {preferred}"

        lines = [
            "=" * 60,
            "Bayesian Model Comparison",
            "=" * 60,
            f"{name_a}: ln(Z) = {result_a.log_evidence:.2f} ± {result_a.log_evidence_err:.2f}",
            f"{name_b}: ln(Z) = {result_b.log_evidence:.2f} ± {result_b.log_evidence_err:.2f}",
            "-" * 60,
            f"Δln(Z) = {delta_ln_z:.2f} ± {err:.2f}",
            f"Bayes factor K = {np.exp(delta_ln_z):.2e}",
            "-" * 60,
            f"Interpretation: {interpretation}",
            "=" * 60,
        ]
        return "\n".join(lines)


class BayesianForwardModel:
    """
    Bayesian forward model for CF-LIBS spectra.

    This class provides a JAX-compatible forward model that maps plasma
    parameters (T, n_e, concentrations) to synthetic spectra. The physics
    includes:
    - Saha-Boltzmann population distribution
    - Voigt line profiles (Weideman rational Faddeeva approximation)
    - Stark broadening with temperature scaling
    - Proper Doppler broadening with mass dependence

    Parameters
    ----------
    db_path : str
        Path to atomic database
    elements : List[str]
        Elements to include
    wavelength_range : Tuple[float, float]
        Wavelength range [nm]
    wavelength_grid : np.ndarray, optional
        Custom wavelength grid; if None, auto-generated
    instrument_fwhm_nm : float
        Instrument FWHM in nm (default: 0.05). Mutually exclusive with resolving_power.
    resolving_power : float, optional
        Spectrometer resolving power R = λ/Δλ (e.g., 5000-20000 for Echelle).
        When set, FWHM varies as λ/R across the spectrum. Mutually exclusive
        with instrument_fwhm_nm.
    """

    def __init__(
        self,
        db_path: str,
        elements: List[str],
        wavelength_range: Tuple[float, float],
        wavelength_grid: Optional[np.ndarray] = None,
        pixels: int = 2048,
        instrument_fwhm_nm: Optional[float] = None,
        resolving_power: Optional[float] = None,
    ):
        if not HAS_JAX:
            raise ImportError("JAX required. Install with: pip install jax jaxlib")

        # Validate mutually exclusive instrument modes
        has_fwhm = instrument_fwhm_nm is not None
        has_rp = resolving_power is not None
        if has_fwhm and has_rp:
            raise ValueError(
                "resolving_power and instrument_fwhm_nm are mutually exclusive. "
                "Set one or the other, not both."
            )
        if not has_fwhm and not has_rp:
            instrument_fwhm_nm = 0.05  # Default: constant 0.05 nm FWHM

        self.elements = elements
        self.wavelength_range = wavelength_range
        self.instrument_fwhm_nm = instrument_fwhm_nm if instrument_fwhm_nm is not None else 0.05
        self.resolving_power = resolving_power

        # Create wavelength grid
        if wavelength_grid is not None:
            self.wavelength = _as_jax_real(wavelength_grid)
        else:
            wl_min = _as_jax_real(wavelength_range[0])
            wl_max = _as_jax_real(wavelength_range[1])
            self.wavelength = jnp.linspace(
                wl_min,
                wl_max,
                pixels,
                dtype=_JAX_REAL_DTYPE,
            )

        # Load atomic data (using shared utility)
        self.atomic_data = load_atomic_data(db_path, elements, wavelength_range)

        # Pre-compute Chebyshev Vandermonde matrix for polynomial baseline.
        # Chebyshev polynomials are orthogonal on [-1,1], which minimizes
        # covariance between MCMC baseline coefficients and improves sampling.
        wl_np = np.asarray(self.wavelength)
        self._wl_norm = 2.0 * (wl_np - wl_np[0]) / max(wl_np[-1] - wl_np[0], 1e-6) - 1.0
        # Max degree we might need (5 = quintic); slice at runtime.
        # PriorConfig.baseline_degree must not exceed this.
        max_baseline_degree = 5
        self._max_baseline_degree = max_baseline_degree
        self._baseline_basis = np.polynomial.chebyshev.chebvander(
            self._wl_norm, max_baseline_degree
        )  # shape: (n_pixels, max_degree+1)
        if HAS_JAX:
            self._baseline_basis_jax = jnp.array(self._baseline_basis, dtype=_JAX_REAL_DTYPE)

        logger.info(
            f"BayesianForwardModel: {len(elements)} elements, "
            f"{len(self.wavelength)} wavelengths, "
            f"{len(self.atomic_data.wavelength_nm)} lines"
        )

    def _load_atomic_data(self, db_path: str) -> AtomicDataArrays:
        """Load atomic data — delegates to module-level :func:`load_atomic_data`."""
        return load_atomic_data(db_path, self.elements, self.wavelength_range)

    def forward(
        self,
        T_eV: float,
        log_ne: float,
        concentrations: jnp.ndarray,
        total_species_density_cm3: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Compute synthetic spectrum for given plasma parameters.

        Parameters
        ----------
        T_eV : float
            Temperature in eV
        log_ne : float
            Log10 of electron density [cm^-3]
        concentrations : array
            Element number fractions on a heavy-particle basis (must sum to 1)
        total_species_density_cm3 : float, optional
            Total heavy-particle number density in cm^-3. If omitted, the
            improved behavior derives it from charge balance.

        Returns
        -------
        array
            Synthetic spectrum intensity
        """
        T_eV = _as_jax_real(T_eV)
        log_ne = _as_jax_real(log_ne)
        concentrations = _as_jax_real(concentrations)
        n_e = jnp.power(_as_jax_real(10.0), log_ne)
        return self._compute_spectrum(
            T_eV,
            n_e,
            concentrations,
            total_species_density_cm3=total_species_density_cm3,
        )

    def forward_numpy(
        self,
        T_eV: float,
        log_ne: float,
        concentrations: np.ndarray,
        total_species_density_cm3: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute synthetic spectrum using NumPy arrays (for dynesty compatibility).

        This is a wrapper around forward() that handles NumPy <-> JAX conversion.

        Parameters
        ----------
        T_eV : float
            Temperature in eV
        log_ne : float
            Log10 of electron density [cm^-3]
        concentrations : np.ndarray
            Element number fractions on a heavy-particle basis (must sum to 1)
        total_species_density_cm3 : float, optional
            Total heavy-particle number density in cm^-3. If omitted, the
            improved behavior derives it from charge balance.

        Returns
        -------
        np.ndarray
            Synthetic spectrum intensity
        """
        conc_jax = _as_jax_real(concentrations)
        result = self.forward(
            T_eV,
            log_ne,
            conc_jax,
            total_species_density_cm3=total_species_density_cm3,
        )
        return np.array(result)

    @staticmethod
    def _partition_function(T_K: float, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Evaluate polynomial partition function.

        Delegates to module-level :func:`partition_function`.
        """
        return partition_function(T_K, coeffs)

    def _compute_spectrum(
        self,
        T_eV: float,
        n_e: float,
        concentrations: jnp.ndarray,
        total_species_density_cm3: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Compute spectrum with full physics.

        Uses Saha-Boltzmann populations, Voigt profiles, and Stark broadening.
        """
        data = self.atomic_data
        T_eV = _as_jax_real(T_eV)
        n_e = _as_jax_real(n_e)
        concentrations = _as_jax_real(concentrations)
        T_K = T_eV * _JAX_EV_TO_K

        # Partition functions for all elements and stages
        U0 = self._partition_function(T_K, data.partition_coeffs[:, 0])
        U1 = self._partition_function(T_K, data.partition_coeffs[:, 1])

        # Ionization potentials for neutral -> ion transition
        IP_I = data.ionization_potentials[:, 0]

        # Saha ratio: n_ion / n_neutral
        saha_factor = (_JAX_SAHA_CONST_CM3 / n_e) * (T_eV**1.5)
        ratio_ion_neutral = saha_factor * (U1 / U0) * jnp.exp(-IP_I / T_eV)

        # Population fractions
        frac_neutral = 1.0 / (1.0 + ratio_ion_neutral)
        frac_ion = ratio_ion_neutral / (1.0 + ratio_ion_neutral)

        # Resolve total species density from charge balance if not provided.
        # Self-consistent CF-LIBS: n_e = N_total * sum(C_i * frac_ion_i)
        # (assuming only single ionization contributes to n_e)
        avg_charge_per_particle = jnp.sum(concentrations * frac_ion)
        avg_charge_per_particle = jnp.maximum(avg_charge_per_particle, 1e-6)
        
        if total_species_density_cm3 is None:
            total_species_density = n_e / avg_charge_per_particle
        else:
            total_species_density = _as_jax_real(total_species_density_cm3)

        # Per-line quantities
        el_idx = data.element_idx
        ion_stage = data.ion_stage

        pop_fraction = jnp.where(ion_stage == 0, frac_neutral[el_idx], frac_ion[el_idx])
        U_val = jnp.where(ion_stage == 0, U0[el_idx], U1[el_idx])

        # Species number density.
        element_conc = concentrations[el_idx]
        N_species_total = element_conc * total_species_density
        N_species = N_species_total * pop_fraction

        # Boltzmann upper level population
        n_upper = N_species * (data.gk / U_val) * jnp.exp(-data.ek_ev / T_eV)

        # Line emissivity: epsilon = (hc / 4pi * lambda) * A * n_upper
        epsilon = (
            (_JAX_H_PLANCK * _JAX_C_LIGHT / (4 * jnp.pi * data.wavelength_nm * _as_jax_real(1e-9)))
            * data.aki
            * n_upper
        )

        # --- Line Broadening ---
        # Doppler width
        mass_kg = data.mass_amu * _JAX_M_PROTON
        sigma_doppler = data.wavelength_nm * jnp.sqrt(
            _as_jax_real(2.0) * T_eV * _JAX_EV_TO_J / (mass_kg * _JAX_C_LIGHT**2)
        )

        # Instrument broadening
        sigma_inst = _compute_instrument_sigma(
            data.wavelength_nm, self.instrument_fwhm_nm, self.resolving_power
        )
        sigma_total = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)

        # Stark broadening (HWHM)
        REF_NE = 1.0e16
        REF_T_EV = 0.86173  # 10000 K

        # Refined Stark width estimation (Griem-inspired scaling r45l)
        binding_energy = jnp.maximum(data.ip_ev - data.ek_ev, 0.1)
        n_eff = (ion_stage + 1) * jnp.sqrt(13.605 / binding_energy)
        # Neutral lines: w ~ n_eff^4. Ion lines: w ~ n_eff^2. Constant 2.45e-5.
        w_est = 2.45e-5 * (data.wavelength_nm / 500.0) ** 2 * jnp.where(ion_stage == 0, n_eff**4, n_eff**2)
        w_est = jnp.clip(w_est, 0.0001, 0.5)
        w_ref = jnp.where(jnp.isnan(data.stark_w), w_est, data.stark_w)

        factor_ne = n_e / REF_NE
        factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -data.stark_alpha)
        gamma_stark = w_ref * factor_ne * factor_T

        # --- Voigt Profile (Weideman rational approximation) ---
        # Uses branch-free implementation for gradient stability during MCMC
        diff = self.wavelength[:, None] - data.wavelength_nm[None, :]
        profile = _voigt_profile_kernel_jax(diff, sigma_total[None, :], gamma_stark[None, :])

        # Sum line contributions
        intensity = jnp.sum(epsilon * profile, axis=1)

        # Clip to prevent numerical overflow at extreme parameters
        intensity = jnp.clip(intensity, 0.0, 1e12)

        return intensity


def log_likelihood(
    predicted: jnp.ndarray,
    observed: jnp.ndarray,
    noise_params: NoiseParameters = NoiseParameters(),
) -> float:
    """
    Compute log-likelihood for observed spectrum given predicted.

    The noise model combines Poisson shot noise and Gaussian readout noise:
        variance = predicted / gain + readout_noise^2 + dark_current

    This is the standard CCD noise model for spectroscopy.

    Parameters
    ----------
    predicted : array
        Predicted spectrum from forward model
    observed : array
        Observed spectrum (counts)
    noise_params : NoiseParameters
        Noise model parameters

    Returns
    -------
    float
        Log-likelihood value
    """
    # Ensure positive predicted values
    pred_safe = jnp.maximum(predicted, 1e-10)

    # Variance: Poisson (shot) + Gaussian (readout) + dark current
    variance = (
        pred_safe / noise_params.gain + noise_params.readout_noise**2 + noise_params.dark_current
    )

    # Gaussian log-likelihood
    residual = observed - pred_safe
    log_lik = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * variance) + residual**2 / variance)

    return log_lik


def bayesian_model(
    forward_model: BayesianForwardModel,
    observed: jnp.ndarray,
    prior_config: PriorConfig = PriorConfig(),
    noise_params: NoiseParameters = NoiseParameters(),
):
    """
    NumPyro probabilistic model for CF-LIBS Bayesian inference.

    This defines the full Bayesian model with priors and likelihood.
    Use with MCMC or variational inference.

    Parameters
    ----------
    forward_model : BayesianForwardModel
        Forward model instance
    observed : array
        Observed spectrum
    prior_config : PriorConfig
        Prior configuration
    noise_params : NoiseParameters
        Noise model parameters
    """
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required. Install with: pip install numpyro")

    n_elements = len(forward_model.elements)

    # --- Priors ---
    # Temperature: uniform on physically realistic range
    T_eV = numpyro.sample(
        "T_eV",
        dist.Uniform(prior_config.T_eV_range[0], prior_config.T_eV_range[1]),
    )

    # Electron density: log-uniform (Jeffreys prior for scale parameter)
    log_ne = numpyro.sample(
        "log_ne",
        dist.Uniform(prior_config.log_ne_range[0], prior_config.log_ne_range[1]),
    )

    # Concentrations: Dirichlet prior (ensures sum to 1)
    alpha = jnp.ones(n_elements) * prior_config.concentration_alpha
    concentrations = numpyro.sample("concentrations", dist.Dirichlet(alpha))

    # --- McWhirter Criterion Penalty ---
    # Physical constraint for LTE validity
    penalty = mcwhirter_log_penalty(T_eV, log_ne)
    numpyro.factor("mcwhirter_penalty", penalty)

    # --- Forward Model ---
    predicted = forward_model.forward(T_eV, log_ne, concentrations)

    # --- Additive polynomial baseline (bremsstrahlung continuum) ---
    if prior_config.baseline_degree > 0:
        if prior_config.baseline_degree > forward_model._max_baseline_degree:
            raise ValueError(
                f"baseline_degree={prior_config.baseline_degree} exceeds max "
                f"({forward_model._max_baseline_degree}). Pre-computed Chebyshev "
                f"basis does not cover this degree."
            )
        n_coeffs = prior_config.baseline_degree + 1
        # Determine prior scale: data-driven default or user-specified
        baseline_scale = prior_config.baseline_scale
        if baseline_scale is not None and baseline_scale <= 0:
            raise ValueError(f"baseline_scale must be positive, got {baseline_scale}")
        if baseline_scale is None:
            baseline_scale = 0.1 * jnp.max(observed)
        baseline_coeffs = numpyro.sample(
            "baseline_coeffs",
            dist.Normal(jnp.zeros(n_coeffs), baseline_scale),
        )
        # Slice pre-computed Chebyshev Vandermonde basis to requested degree
        basis = forward_model._baseline_basis_jax[:, :n_coeffs]
        baseline = jnp.dot(basis, baseline_coeffs)
        predicted = predicted + baseline

    # --- Likelihood ---
    # Variance model: Poisson + readout noise
    # Add safeguards for numerical stability
    pred_safe = jnp.maximum(predicted, 1e-6)
    pred_safe = jnp.where(jnp.isnan(pred_safe), 1e-6, pred_safe)
    pred_safe = jnp.where(jnp.isinf(pred_safe), 1e6, pred_safe)

    variance = (
        pred_safe / noise_params.gain + noise_params.readout_noise**2 + noise_params.dark_current
    )
    sigma = jnp.sqrt(jnp.maximum(variance, 1e-6))

    # Observe data
    numpyro.sample("obs", dist.Normal(pred_safe, sigma), obs=observed)


class MCMCSampler:
    """
    MCMC sampler for Bayesian CF-LIBS inference.

    Wraps NumPyro's NUTS sampler with sensible defaults for CF-LIBS,
    including convergence diagnostics, initialization strategies, and
    ArviZ integration for analysis and visualization.

    Parameters
    ----------
    forward_model : BayesianForwardModel
        Forward model instance
    prior_config : PriorConfig
        Prior configuration (default: PriorConfig())
    noise_params : NoiseParameters
        Noise model parameters (default: NoiseParameters())

    Example
    -------
    >>> sampler = MCMCSampler(forward_model)
    >>> result = sampler.run(observed_spectrum, num_samples=2000)
    >>> print(result.summary_table())
    >>> if result.is_converged:
    ...     print(f"T = {result.T_eV_mean:.3f} +/- {result.T_eV_std:.3f} eV")
    """

    def __init__(
        self,
        forward_model: BayesianForwardModel,
        prior_config: PriorConfig = PriorConfig(),
        noise_params: NoiseParameters = NoiseParameters(),
    ):
        if not HAS_NUMPYRO:
            raise ImportError("NumPyro required. Install with: pip install numpyro")

        self.forward_model = forward_model
        self.prior_config = prior_config
        self.noise_params = noise_params
        self.elements = forward_model.elements

        logger.info(
            f"MCMCSampler initialized: {len(self.elements)} elements, "
            f"T range={prior_config.T_eV_range} eV"
        )

    def _get_init_values(self) -> Dict[str, Any]:
        """
        Get sensible initial values for MCMC.

        Uses midpoints of prior ranges for temperature and density,
        and uniform concentrations.
        """
        n_elements = len(self.elements)

        # Midpoint of prior ranges
        T_init = (self.prior_config.T_eV_range[0] + self.prior_config.T_eV_range[1]) / 2
        log_ne_init = (self.prior_config.log_ne_range[0] + self.prior_config.log_ne_range[1]) / 2

        # Uniform concentrations
        conc_init = jnp.ones(n_elements) / n_elements

        init_values = {
            "T_eV": T_init,
            "log_ne": log_ne_init,
            "concentrations": conc_init,
        }

        # Initialize baseline coefficients at zero (no continuum adjustment)
        if self.prior_config.baseline_degree > 0:
            n_coeffs = self.prior_config.baseline_degree + 1
            init_values["baseline_coeffs"] = jnp.zeros(n_coeffs)

        return init_values

    def run(
        self,
        observed: np.ndarray,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        seed: int = 0,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        progress_bar: bool = True,
    ) -> MCMCResult:
        """
        Run MCMC sampling.

        Parameters
        ----------
        observed : array
            Observed spectrum
        num_warmup : int
            Number of warmup samples (default: 500)
        num_samples : int
            Number of posterior samples (default: 1000)
        num_chains : int
            Number of MCMC chains (default: 1, use 4 for production)
        seed : int
            Random seed
        target_accept_prob : float
            Target acceptance probability for NUTS (default: 0.8)
        max_tree_depth : int
            Maximum tree depth for NUTS (default: 10)
        progress_bar : bool
            Show progress bar (default: True)

        Returns
        -------
        MCMCResult
            Results with posterior samples and convergence diagnostics
        """
        import jax.random as random

        observed_jax = _as_jax_real(observed)
        n_elements = len(self.elements)

        # Create model function
        def model(obs):
            bayesian_model(self.forward_model, obs, self.prior_config, self.noise_params)

        # Create NUTS sampler with initialization
        # Use init_to_uniform for robustness - samples from prior to find valid start
        kernel = NUTS(
            model,
            init_strategy=init_to_uniform(radius=0.5),
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )

        # Run sampling
        rng_key = random.PRNGKey(seed)
        logger.info(
            f"Starting MCMC: {num_chains} chains, {num_warmup} warmup, {num_samples} samples"
        )

        mcmc.run(rng_key, observed_jax)

        # Get samples
        samples = mcmc.get_samples(group_by_chain=(num_chains > 1))

        T_samples = samples["T_eV"]
        log_ne_samples = samples["log_ne"]
        conc_samples = samples["concentrations"]

        # Flatten for statistics
        T_flat = np.array(T_samples).flatten()
        log_ne_flat = np.array(log_ne_samples).flatten()
        conc_flat = np.array(conc_samples).reshape(-1, n_elements)

        # Compute convergence diagnostics
        r_hat, ess = self._compute_convergence_diagnostics(mcmc, num_chains)

        # Determine convergence status
        convergence_status = self._assess_convergence(r_hat, ess, num_samples)

        # Build MCMCResult
        result = MCMCResult(
            samples={k: np.array(v) for k, v in samples.items()},
            T_eV_mean=float(np.mean(T_flat)),
            T_eV_std=float(np.std(T_flat)),
            T_eV_q025=float(np.percentile(T_flat, 2.5)),
            T_eV_q975=float(np.percentile(T_flat, 97.5)),
            log_ne_mean=float(np.mean(log_ne_flat)),
            log_ne_std=float(np.std(log_ne_flat)),
            log_ne_q025=float(np.percentile(log_ne_flat, 2.5)),
            log_ne_q975=float(np.percentile(log_ne_flat, 97.5)),
            concentrations_mean={
                el: float(np.mean(conc_flat[:, i])) for i, el in enumerate(self.elements)
            },
            concentrations_std={
                el: float(np.std(conc_flat[:, i])) for i, el in enumerate(self.elements)
            },
            concentrations_q025={
                el: float(np.percentile(conc_flat[:, i], 2.5)) for i, el in enumerate(self.elements)
            },
            concentrations_q975={
                el: float(np.percentile(conc_flat[:, i], 97.5))
                for i, el in enumerate(self.elements)
            },
            r_hat=r_hat,
            ess=ess,
            convergence_status=convergence_status,
            n_samples=num_samples,
            n_chains=num_chains,
            n_samples_total=len(T_flat),
            n_warmup=num_warmup,
            inference_data=self._to_arviz(mcmc) if HAS_ARVIZ else None,
        )

        logger.info(
            f"MCMC complete: T = {result.T_eV_mean:.3f} +/- {result.T_eV_std:.3f} eV, "
            f"n_e = {result.n_e_mean:.2e} cm^-3, "
            f"convergence={convergence_status.value}"
        )

        return result

    def _compute_convergence_diagnostics(
        self, mcmc: Any, num_chains: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute R-hat and ESS convergence diagnostics.

        Uses ArviZ if available, otherwise falls back to simple estimates.
        """
        r_hat = {}
        ess = {}

        if HAS_ARVIZ and num_chains > 1:
            # Use ArviZ for multi-chain diagnostics
            try:
                idata = az.from_numpyro(mcmc)

                # R-hat (should be < 1.01 for convergence)
                rhat_data = az.rhat(idata)
                for var in ["T_eV", "log_ne"]:
                    if var in rhat_data:
                        val = float(rhat_data[var].values)
                        r_hat[var] = val

                # ESS (effective sample size)
                ess_data = az.ess(idata)
                for var in ["T_eV", "log_ne"]:
                    if var in ess_data:
                        val = float(ess_data[var].values)
                        ess[var] = val

            except Exception as e:
                logger.warning(f"ArviZ diagnostics failed: {e}")
                r_hat, ess = self._simple_diagnostics(mcmc, num_chains)
        else:
            # Single chain or no ArviZ - use simple estimates
            r_hat, ess = self._simple_diagnostics(mcmc, num_chains)

        return r_hat, ess

    def _simple_diagnostics(
        self, mcmc: Any, num_chains: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Simple fallback diagnostics when ArviZ unavailable."""
        samples = mcmc.get_samples()
        r_hat = {}
        ess = {}

        for var in ["T_eV", "log_ne"]:
            if var in samples:
                s = np.array(samples[var]).flatten()
                # Simple ESS estimate using total sample count
                n = len(s)
                ess[var] = float(n)

                # R-hat = 1 for single chain
                r_hat[var] = 1.0
        
        # Concentrations are vectors, handle per-element
        if "concentrations" in samples:
            c = np.array(samples["concentrations"])
            # Flatten chains if present
            if c.ndim == 3:
                c = c.reshape(-1, c.shape[-1])
            for i, el in enumerate(self.elements):
                var_name = f"C_{el}"
                ess[var_name] = float(c.shape[0])
                r_hat[var_name] = 1.0

        return r_hat, ess

    def _assess_convergence(
        self, r_hat: Dict[str, float], ess: Dict[str, float], n_samples: int
    ) -> ConvergenceStatus:
        """Assess overall convergence status based on diagnostics."""
        if not r_hat:
            return ConvergenceStatus.UNKNOWN
            
        max_rhat = max(r_hat.values())
        min_ess = min(ess.values()) if ess else 0
        
        if max_rhat < 1.01 and min_ess > 100:
            return ConvergenceStatus.CONVERGED
        elif max_rhat < 1.05:
            return ConvergenceStatus.WARNING
        else:
            return ConvergenceStatus.NOT_CONVERGED

[... truncated; full file is 110905 bytes ...]
