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
1. Estimate τ₀ from an OBSERVABLE (doublet intensity ratio, measured
   equivalent widths via the COG, or — diagnostically — plasma parameters)
2. Calculate correction factor: C = 1/f(τ₀)
3. Apply: I_true = I_observed × C

The composition-fed applicator (the old ``SelfAbsorptionCorrector.correct``)
was deleted in bead CF-LIBS-improved-0jvr — see the class docstring. The
production correction is
:class:`cflibs.inversion.physics.self_absorption_observable.ObservableSelfAbsorptionCorrector`.

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
from typing import List, Dict, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import brentq

from cflibs.core.constants import (
    C_LIGHT,
    EV_TO_K,
    KB,
    KB_EV,
    M_PROTON,
)
from cflibs.core.logging_config import get_logger
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.plasma.partition import lookup_partition_function

logger = get_logger("inversion.self_absorption")


# =============================================================================
# Optical-depth prefactor — derivation
# =============================================================================
#
# The classical line-center optical depth for a Doppler-broadened line is
# (Hutchinson, *Principles of Plasma Diagnostics*, 2nd ed., eq. 5.13;
# Mihalas, *Stellar Atmospheres*, 2nd ed., eq. 4-2; Konjević 1999 review,
# Phys. Rep. 316, 339):
#
#     τ₀ = (π e² / mₑ c) · f_lu · N_lower · L · φ(ν₀)
#
# with
#
#     φ(ν₀) = 1 / (√π · Δν_D)      (Doppler line-center value, 1/e width)
#     Δν_D  = (ν₀ / c) · √(2 k T / M)        (Doppler 1/e half-width in Hz)
#
# Evaluating the classical-radius prefactor in CGS (e = 4.8032 × 10⁻¹⁰ esu,
# mₑ = 9.1094 × 10⁻²⁸ g, c = 2.9979 × 10¹⁰ cm/s) gives
#
#     π e² / (mₑ c) ≃ 0.02654 cm² · Hz
#
# In SI (e = 1.602 × 10⁻¹⁹ C, mₑ = 9.1094 × 10⁻³¹ kg, c = 2.998 × 10⁸ m/s)
# the same dimensionful constant has to be multiplied by 1/(4πε₀) to recover
# the Gaussian-unit formula; we work in CGS throughout so the prefactor
# below is the bare numerical value computed from CGS constants (the
# scipy.constants module exposes them in SI, so we convert).
#
# Magnitude for typical LIBS conditions (T ≈ 10 000 K, M ≈ 28 amu like Si):
#
#     v_th     = √(2 k T / M)          ≃ 2.43 × 10⁵ cm/s
#     Δν_D     = (ν₀/c) · v_th         ≃ 9.7 × 10⁹ Hz  for λ = 251.6 nm
#     φ(ν₀)    ≃ 5.8 × 10⁻¹¹ Hz⁻¹
#     σ_peak   = (π e²/mₑ c) · f_lu · φ(ν₀)
#              ≃ 0.0265 · 0.3 · 5.8e-11  ≃ 5 × 10⁻¹³ cm²  for f_lu ≈ 0.3
#
# So the per-line cross-section at line center is of order 10⁻¹³–10⁻¹² cm²
# (consistent with the standard textbook range quoted by Griem and by
# Hutchinson). Multiplied by N_lower L ≃ 10¹³–10¹⁶ cm⁻² for LIBS plasmas
# gives τ in the optically-thick regime (τ ≳ 1) for major-element resonance
# lines — which is exactly where the Pace-doublet and CDSB corrections in
# the rest of this module are designed to live.
#
# The historical SCALE_FACTOR = 1e-25 × A_ki × λ³ formulation (removed in
# CF-LIBS-improved-k2h7) under-estimated τ by ~12 orders of magnitude
# because it has no Doppler-width normalization and no proper oscillator-
# strength conversion. With τ ≈ 10⁻¹⁵ for every line, the
# `optical_depth_threshold = 0.1` gate in correct() never fired, so the
# corrector was a silent no-op even for spectacularly self-absorbed lines.
#
# Reference: Hutchinson eq. (5.13); Konjević (1999) Phys. Rep. 316, 339,
# §3.2.
#
# pre-compute the classical-radius prefactor once at module load so the
# inner loop stays cheap. We use CGS via direct constants — these are also
# in cflibs.core.constants but spelled in SI (M_E in kg, E_CHARGE in C),
# so convert here.

_M_E_CGS_G = 9.1093837015e-28  # electron mass in grams
_E_CGS_ESU = 4.80320425e-10  # elementary charge in statcoulombs
_C_CGS_CM_PER_S = 2.99792458e10  # speed of light in cm/s

# (π e² / mₑ c) in CGS — units of cm² · Hz. Constant of nature; depends
# only on QED. Quote: 0.026540... cm² Hz (Cowan, *Theory of Atomic
# Structure and Spectra*, 1981; CODATA-2018-consistent).
_PI_E2_OVER_MEC_CGS = np.pi * _E_CGS_ESU**2 / (_M_E_CGS_G * _C_CGS_CM_PER_S)

# Atomic masses (in amu) for elements commonly encountered in LIBS
# matrices. Used to compute Doppler width when the caller does not supply
# a mass. Values from NIST standard atomic weights (most-abundant isotope
# weighted). Fallback for elements not in this table is 28 amu (~Si),
# which is order-of-magnitude correct for major rock-forming elements
# (Si=28, Al=27, Mg=24, Ca=40, Fe=56, Ti=48). The Doppler width scales as
# 1/√M, so a 2x mass mismatch only changes τ by √2 ≈ 1.4× — within the
# uncertainty budget of optical-depth estimation in plasma diagnostics.
_ATOMIC_MASS_AMU: Dict[str, float] = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.94,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Sr": 87.62,
    "Mo": 95.95,
    "Ag": 107.87,
    "Cd": 112.41,
    "Sn": 118.71,
    "Sb": 121.76,
    "Ba": 137.33,
    "W": 183.84,
    "Pt": 195.08,
    "Au": 196.97,
    "Hg": 200.59,
    "Pb": 207.2,
    "Bi": 208.98,
    "U": 238.03,
}
_DEFAULT_ATOMIC_MASS_AMU = 28.0  # Si-like fallback for unknown elements


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
class DoubletRatioResult:
    """
    Result of a single-spectrum, closed-form self-absorption correction
    derived from a doublet (two lines from the same upper level).

    The doublet-ratio method (Pace et al., Spectrochim. Acta B 2025,
    https://www.sciencedirect.com/science/article/abs/pii/S0584854725001995)
    is a SECOND-OPINION estimator that complements the curve-of-growth
    derived self-absorption correction (CDSB). For two lines (k -> i1,
    k -> i2) sharing the same upper level k, the theoretical EMISSION
    intensity ratio in the optically-thin limit is

        r_thin = (g_k1 * A_1 / lambda_1) / (g_k2 * A_2 / lambda_2)

    (:func:`_thin_emission_ratio`; consistent with the Boltzmann ordinate
    ln(I lambda / gA)). The measured ratio r_meas = I_1 / I_2 differs from
    r_thin because each line is attenuated by a curve-of-growth escape
    factor f(tau) = (1 - exp(-tau)) / tau, with the two line-center optical
    depths linked by the line-strength ratio (:func:`_tau_ratio`)

        tau_2 = tau_1 / rho,
        rho = (g_k1 * A_1 * lambda_1**3) / (g_k2 * A_2 * lambda_2**3)

    Solving f(tau_1) / f(tau_1 / rho) = r_meas / r_thin for tau_1
    gives a 1-D nonlinear root, which is solved with `scipy.optimize.brentq`.

    Attributes
    ----------
    tau_1 : float
        Recovered optical depth of the stronger line (line 1).
    tau_2 : float
        Recovered optical depth of the weaker line (line 2).
    f_tau_1 : float
        Escape factor f(tau_1) applied to line 1.
    f_tau_2 : float
        Escape factor f(tau_2) applied to line 2.
    r_measured : float
        Measured intensity ratio I_1 / I_2.
    r_theory : float
        Theoretical optically-thin EMISSION intensity ratio ``r_thin``.
    i1_corrected : float
        Self-absorption-corrected intensity of line 1: I_1 / f(tau_1).
    i2_corrected : float
        Self-absorption-corrected intensity of line 2: I_2 / f(tau_2).
    wavelength_pair_nm : tuple[float, float]
        (lambda_1, lambda_2) of the two doublet lines in nm.
    agreement_with_cdsb_sigma : float, optional
        (tau_doublet - tau_cdsb) / max(sigma_doublet, sigma_cdsb) when
        the result has been cross-checked against a CDSB-derived tau.
        None if no cross-check was performed.
    """

    tau_1: float
    tau_2: float
    f_tau_1: float
    f_tau_2: float
    r_measured: float
    r_theory: float
    i1_corrected: float
    i2_corrected: float
    wavelength_pair_nm: Tuple[float, float]
    agreement_with_cdsb_sigma: Optional[float] = None


def _thin_emission_ratio(line1: LineObservation, line2: LineObservation) -> float:
    """
    Optically-thin integrated EMISSION intensity ratio ``I_1 / I_2`` for two
    lines sharing the same upper level.

    The integrated emission radiance of a transition ``k -> i`` is
    ``J = (hc / 4 pi lambda) A_ki n_k L``, so for two lines from the SAME
    upper level (identical ``n_k``) the thin ratio is

        r_thin = (g_k1 A_1 / lambda_1) / (g_k2 A_2 / lambda_2)

    (``g_k`` written explicitly; it cancels for a true same-level pair).
    This is the convention consistent with the Boltzmann ordinate
    ``LineObservation.y_value = ln(I lambda / g A)``.

    Historical note (bead 0jvr): the original implementation used
    ``g A lambda^3``, which is the LINE-STRENGTH / optical-depth ratio
    (see :func:`_tau_ratio`), not the emission-intensity ratio. The two were
    conflated; for nearby pairs the error is ``(lambda_1/lambda_2)^4`` (a few
    per cent) but it diverges for well-separated pairs.
    """
    return (line1.g_k * line1.A_ki / line1.wavelength_nm) / (
        line2.g_k * line2.A_ki / line2.wavelength_nm
    )


def _tau_ratio(line1: LineObservation, line2: LineObservation) -> float:
    """
    Approximate line-center optical-depth ratio ``tau_1 / tau_2`` for a
    same-upper-level pair.

    With ``tau_0 ∝ f_lu n_i L phi(nu_0)``, ``f_lu ∝ lambda^2 g_k A / g_i``,
    the Doppler line-center profile value ``phi(nu_0) ∝ lambda`` and the
    lower-level population ``n_i ∝ g_i exp(-E_i / kT)``:

        tau_1 / tau_2 = (g_k1 A_1 lambda_1^3) / (g_k2 A_2 lambda_2^3)
                        * exp(-(E_i1 - E_i2) / kT)

    The lower-level Boltzmann factor is dropped so the estimator stays
    temperature-independent; this is exact for ``lambda_1 ~ lambda_2``
    (where ``E_i1 ~ E_i2``) and degrades smoothly with pair separation.
    """
    return (line1.g_k * line1.A_ki * line1.wavelength_nm**3) / (
        line2.g_k * line2.A_ki * line2.wavelength_nm**3
    )


def correct_via_doublet_ratio(
    line1: LineObservation,
    line2: LineObservation,
) -> DoubletRatioResult:
    """
    Closed-form, single-spectrum self-absorption correction for a doublet.

    Both lines must share the same upper level (E_k_ev within 1 meV) and
    the same species (element + ionization stage). The correction solves
    the 1-D nonlinear equation

        f(tau_1) / f(tau_1 / rho) = r_meas / r_thin

    for the optical depth tau_1 of line 1 via `scipy.optimize.brentq`
    over tau_1 in [1e-4, 30], where

    * ``r_thin`` is the optically-thin EMISSION intensity ratio
      ``(g_k1 A_1 / lambda_1) / (g_k2 A_2 / lambda_2)``
      (:func:`_thin_emission_ratio`), and
    * ``rho = tau_1 / tau_2`` is the line-strength (optical-depth) ratio
      ``(g_k1 A_1 lambda_1^3) / (g_k2 A_2 lambda_2^3)``
      (:func:`_tau_ratio`).

    The corrected intensities are I_i_corr = I_i / f(tau_i), with
    tau_2 = tau_1 / rho.

    See Pace et al., Spectrochim. Acta B 2025
    (https://www.sciencedirect.com/science/article/abs/pii/S0584854725001995).
    Bead 0jvr fixed the historical conflation of ``r_thin`` and ``rho``
    (both were ``g A lambda^3``).

    Parameters
    ----------
    line1, line2 : LineObservation
        Two lines from the same upper level. The line with the smaller
        wavelength is treated as ``line1`` after internal swapping if
        necessary (matching the ordering returned by `find_doublet_pairs`).

    Returns
    -------
    DoubletRatioResult
        Closed-form recovery of (tau_1, tau_2) and the corrected
        intensities. ``agreement_with_cdsb_sigma`` is left as ``None``;
        populate it by comparing against a CDSB-derived tau in
        :py:meth:`SelfAbsorptionCorrector.cross_check_with_doublets`.

    Raises
    ------
    ValueError
        If the lines do not share the upper level or species, or if any
        intensity / oscillator strength is non-positive.
    """
    # Order so line1 has the shorter wavelength (matches find_doublet_pairs)
    if line2.wavelength_nm < line1.wavelength_nm:
        line1, line2 = line2, line1

    # Validate species
    if line1.element != line2.element:
        raise ValueError(f"Doublet lines must share element: {line1.element} != {line2.element}")
    if line1.ionization_stage != line2.ionization_stage:
        raise ValueError(
            f"Doublet lines must share ionization stage: "
            f"{line1.ionization_stage} != {line2.ionization_stage}"
        )

    # Validate same upper level (within ~1 meV)
    if abs(line1.E_k_ev - line2.E_k_ev) > 0.001:
        raise ValueError(
            f"Doublet lines must share upper level (within 1 meV): "
            f"E_k_1={line1.E_k_ev:.4f} eV, E_k_2={line2.E_k_ev:.4f} eV"
        )

    # Validate measurable intensities and atomic data
    if line1.intensity <= 0 or line2.intensity <= 0:
        raise ValueError(
            f"Doublet correction requires positive intensities: "
            f"I_1={line1.intensity}, I_2={line2.intensity}"
        )
    if line1.A_ki <= 0 or line2.A_ki <= 0 or line1.g_k <= 0 or line2.g_k <= 0:
        raise ValueError("Doublet correction requires positive g_k * A_ki for both lines")

    r_theory = _thin_emission_ratio(line1, line2)
    rho = _tau_ratio(line1, line2)
    r_meas = line1.intensity / line2.intensity
    ratio_of_ratios = r_meas / r_theory

    def residual(tau_1_guess: float) -> float:
        """f(tau_1) / f(tau_1 / rho) - r_meas / r_thin."""
        f1 = _escape_factor(tau_1_guess)
        f2 = _escape_factor(tau_1_guess / rho)
        return f1 / f2 - ratio_of_ratios

    tau_low, tau_high = 1e-4, 30.0
    res_low = residual(tau_low)
    res_high = residual(tau_high)

    # Optically thin / inconsistent-with-self-absorption case: same-sign
    # residuals at the bracket endpoints. Either r_meas == r_theory (no
    # absorption) or r_meas implies negative tau (measurement is brighter
    # than the optically-thin prediction, which is impossible under pure
    # self-absorption). Return tau ~ 0 with a debug log.
    if res_low * res_high > 0 or abs(ratio_of_ratios - 1.0) < 1e-12:
        if abs(res_low) > 0.05:
            logger.debug(
                "Doublet at (%.3f, %.3f) nm: r_meas/r_theory=%.4f implies negative "
                "or near-zero tau (residual at tau=1e-4 is %.3e); returning tau~0.",
                line1.wavelength_nm,
                line2.wavelength_nm,
                ratio_of_ratios,
                res_low,
            )
        tau_1 = 1e-4
        tau_2 = tau_1 / rho
        f_tau_1 = _escape_factor(tau_1)
        f_tau_2 = _escape_factor(tau_2)
        return DoubletRatioResult(
            tau_1=tau_1,
            tau_2=tau_2,
            f_tau_1=f_tau_1,
            f_tau_2=f_tau_2,
            r_measured=r_meas,
            r_theory=r_theory,
            i1_corrected=line1.intensity / f_tau_1,
            i2_corrected=line2.intensity / f_tau_2,
            wavelength_pair_nm=(line1.wavelength_nm, line2.wavelength_nm),
        )

    tau_1 = brentq(residual, tau_low, tau_high, xtol=1e-6, rtol=1e-8, maxiter=100)
    tau_2 = tau_1 / rho
    f_tau_1 = _escape_factor(tau_1)
    f_tau_2 = _escape_factor(tau_2)

    logger.debug(
        "correct_via_doublet_ratio: %s %s doublet (%.3f, %.3f) nm -> "
        "tau_1=%.3f tau_2=%.3f f1=%.3f f2=%.3f r_meas=%.3f r_theory=%.3f",
        line1.element,
        line1.ionization_stage,
        line1.wavelength_nm,
        line2.wavelength_nm,
        tau_1,
        tau_2,
        f_tau_1,
        f_tau_2,
        r_meas,
        r_theory,
    )

    return DoubletRatioResult(
        tau_1=tau_1,
        tau_2=tau_2,
        f_tau_1=f_tau_1,
        f_tau_2=f_tau_2,
        r_measured=r_meas,
        r_theory=r_theory,
        i1_corrected=line1.intensity / f_tau_1,
        i2_corrected=line2.intensity / f_tau_2,
        wavelength_pair_nm=(line1.wavelength_nm, line2.wavelength_nm),
    )


def _ordered_doublet_pair(
    li: LineObservation,
    lj: LineObservation,
    dE_ev_tol: float,
) -> Optional[Tuple[LineObservation, LineObservation]]:
    """
    Return ``(li, lj)`` candidates as a doublet pair ordered by
    wavelength (shorter first), or ``None`` when the two lines do not
    form a doublet.

    Two lines form a doublet when they share the same species (element +
    ionization stage) and the same upper level (``E_k_ev`` within
    ``dE_ev_tol``). Equal-wavelength matches are degenerate and rejected.
    """
    if li.element != lj.element:
        return None
    if li.ionization_stage != lj.ionization_stage:
        return None
    if abs(li.E_k_ev - lj.E_k_ev) > dE_ev_tol:
        return None
    if li.wavelength_nm < lj.wavelength_nm:
        return (li, lj)
    if lj.wavelength_nm < li.wavelength_nm:
        return (lj, li)
    # Equal-wavelength case: skip (degenerate, not a doublet)
    return None


def find_doublet_pairs(
    lines: Sequence[LineObservation],
    dE_ev_tol: float = 0.001,
) -> List[Tuple[LineObservation, LineObservation]]:
    """
    Scan a line list for all (line_i, line_j) pairs sharing the same
    upper level (matching ``E_k_ev`` within ``dE_ev_tol``) AND the same
    species (element + ionization stage), with line_i.wavelength_nm <
    line_j.wavelength_nm.

    Parameters
    ----------
    lines : sequence of LineObservation
        Input line observations.
    dE_ev_tol : float, default 0.001
        Tolerance (in eV) for matching upper-level energies. The default
        of 1 meV catches NIST quantization noise.

    Returns
    -------
    list of (LineObservation, LineObservation)
        All matched pairs, with the shorter-wavelength line first.
    """
    pairs: List[Tuple[LineObservation, LineObservation]] = []
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            pair = _ordered_doublet_pair(lines[i], lines[j], dE_ev_tol)
            if pair is not None:
                pairs.append(pair)
    return pairs


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
    Curve-of-growth self-absorption DIAGNOSTICS for optically thick plasmas.

    Provides:

    * :meth:`_estimate_optical_depth` — the classical Doppler line-center
      optical depth from a *given* plasma state (T, concentrations, column
      density). This is a PRIOR/diagnostic quantity only.
    * :meth:`correct_with_cog` — curve-of-growth multiplet analysis driven by
      *measured equivalent widths* (an observable).
    * :meth:`cross_check_with_doublets` — Pace 2025 doublet-ratio second
      opinion on COG-derived optical depths.

    .. warning::

       The composition-fed correction applicator (the old ``correct()``
       method, which divided observed intensities by ``1/f(tau)`` with tau
       computed from the *recovered* composition) was DELETED in bead
       CF-LIBS-improved-0jvr. Driven from inside the CF-LIBS solver it is a
       positive feedback loop (over-attributed element -> bigger tau ->
       bigger boost -> bigger intercept; audit 02-inversion-solver.md F4)
       and it measurably worsened intercept inflation on real ChemCam
       BHVO-2 data. Production correction lives in
       :class:`cflibs.inversion.physics.self_absorption_observable.ObservableSelfAbsorptionCorrector`,
       which derives every correction factor from observables. Plasma-state
       tau estimates from this class must only ever be used as priors or
       cross-checks, never as the sole driver of an intensity correction.
    """

    def __init__(
        self,
        optical_depth_threshold: float = 0.1,
        mask_threshold: float = 3.0,
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
        plasma_length_cm : float
            Estimated plasma depth (path length) in cm
        """
        self.optical_depth_threshold = optical_depth_threshold
        self.mask_threshold = mask_threshold
        self.plasma_length_cm = plasma_length_cm

    @staticmethod
    def _passthrough_result(
        observations: List[LineObservation], warnings_list: List[str]
    ) -> SelfAbsorptionResult:
        """Explicit no-op result: every line passed through uncorrected.

        Replaces the historical fallback of calling the deleted plasma-state
        ``correct()`` with empty concentrations, which forced ``tau = 0`` for
        every line — i.e. exactly this pass-through.
        """
        corrections = {
            obs.wavelength_nm: AbsorptionCorrectionResult(
                original_intensity=obs.intensity,
                corrected_intensity=obs.intensity,
                optical_depth=0.0,
                correction_factor=1.0,
                is_optically_thick=False,
            )
            for obs in observations
        }
        return SelfAbsorptionResult(
            corrected_observations=list(observations),
            masked_observations=[],
            corrections=corrections,
            n_corrected=0,
            n_masked=0,
            max_optical_depth=0.0,
            warnings=warnings_list,
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

        Implements the classical Doppler-broadened line-center formula
        (Hutchinson, *Principles of Plasma Diagnostics*, eq. 5.13;
        Mihalas, *Stellar Atmospheres*, eq. 4-2; Konjević 1999, Phys. Rep.
        316, §3.2):

        .. math::

            \\tau_0 = \\frac{\\pi e^2}{m_e c}\\, f_{lu}\\, N_{lower}\\, L\\, \\phi(\\nu_0)

        with the line-center Doppler profile value

        .. math::

            \\phi(\\nu_0) = \\frac{1}{\\sqrt{\\pi}\\, \\Delta\\nu_D},
            \\qquad
            \\Delta\\nu_D = \\frac{\\nu_0}{c}\\sqrt{\\frac{2 k T}{M}}

        The classical-radius prefactor evaluates to
        ``π e² / (mₑ c) ≃ 0.02654 cm² · Hz`` in CGS — a fundamental
        constant that does NOT depend on the line. ``f_lu`` (absorption
        oscillator strength) is derived from the Einstein A coefficient
        on ``obs``:

        .. math::

            f_{lu} = \\frac{m_e c}{8 \\pi^2 e^2}\\,\\lambda^2\\,\\frac{g_k}{g_i}\\, A_{ki}
                   = 1.4992\\,\\lambda[\\mathrm{cm}]^2 \\cdot A_{ki}[\\mathrm{s}^{-1}]\\,\\frac{g_k}{g_i}

        For a Si I 251.611 nm line (``A_ki = 1.21e8 s⁻¹``, g_k=3, g_i=1)
        this gives ``f_lu ≈ 0.34``, Δν_D ≈ 9.7×10⁹ Hz at T=10 000 K, and a
        cross section σ_peak ≈ 5×10⁻¹³ cm². At 60 % Si mass fraction and
        N_total = 10¹⁵ cm⁻³ with L = 0.1 cm this lands τ in the
        optically-thick regime (τ ~ a few), which is exactly where the
        Pace-doublet and CDSB corrections in the rest of this module are
        designed to live.

        Historical note: prior to CF-LIBS-improved-k2h7 this method used
        ``SCALE_FACTOR = 1e-25 × A_ki × λ³ × n_i × L``, which is
        dimensionally a 12-order-of-magnitude under-estimate (the
        per-line σ has units of cm², not the cm³·s implied by
        ``A·λ³``). Every real LIBS line returned τ ≈ 1e-15, so the
        ``optical_depth_threshold = 0.1`` gate never fired and the
        corrector was a silent no-op. See `docs/research/` for the audit.

        Parameters
        ----------
        obs : LineObservation
            Emission line — must carry ``A_ki``, ``g_k``, ``element``,
            ``wavelength_nm``.
        temperature_K : float
            Excitation temperature in K.
        concentrations : Dict[str, float]
            Element mass fractions (or number fractions; treated as a
            multiplier on ``total_n_cm3``).
        total_n_cm3 : float
            Total heavy-particle number density in cm⁻³.
        partition_funcs : Dict[str, float]
            Partition function U(T) per element.
        E_i_ev : float
            Lower-level energy in eV (0 = ground state, the worst case).

        Returns
        -------
        float
            Line-center optical depth τ₀, clamped to be non-negative.
        """
        element = obs.element
        C_s = concentrations.get(element, 0.0)
        U_T = lookup_partition_function(partition_funcs, element, 1)

        if C_s <= 0 or U_T <= 0:
            # Silent skip — but emit a DEBUG line so the cause is recoverable
            # from a -v log. Higher-severity summary is emitted by `correct`.
            logger.debug(
                "_estimate_optical_depth(%s @ %.3f nm): tau=0 because "
                "C_s=%.3e or U_T=%.3e is non-positive",
                element,
                obs.wavelength_nm,
                C_s,
                U_T,
            )
            return 0.0

        # Species number density
        n_s = C_s * total_n_cm3

        # Lower level population (Boltzmann)
        T_eV = temperature_K / EV_TO_K
        if T_eV <= 0:
            logger.debug(
                "_estimate_optical_depth(%s @ %.3f nm): tau=0 because "
                "T_eV=%.3e (T_K=%.1f) is non-positive",
                element,
                obs.wavelength_nm,
                T_eV,
                temperature_K,
            )
            return 0.0

        # Statistical weight of the lower level. Strictly we want g_i, not
        # g_k — but LineObservation only carries g_k. For the LIBS
        # resonance-line regime (where SA correction matters most) the two
        # levels typically differ by a factor of 1–3, so using g_k is an
        # O(1) bias in N_lower; not the 12-orders-of-magnitude error the
        # 1e-25 scale factor introduced. Document this honestly so future
        # work can plumb g_i through.
        g_i = obs.g_k  # TODO: pipe true g_i through LineObservation

        exp_factor = np.exp(-E_i_ev / T_eV)
        n_lower = n_s * (g_i / U_T) * exp_factor

        # Wavelength in cm (CGS throughout the prefactor calculation).
        lambda_cm = obs.wavelength_nm * 1e-7

        # Absorption oscillator strength from A_ki via the standard
        # Einstein-coefficient relation (Cowan 1981, eq. 14.39):
        #     f_lu = (m_e c / (8 π² e²)) · λ² · A_ki · (g_k / g_i)
        # The numerical prefactor in CGS with λ in cm is exactly 1.4992
        # (see ``MultipletLine.oscillator_strength`` for the cross-check).
        # Falls back to a safe 1.0 when atomic data are malformed; this
        # is the same defensive default as the historical code.
        if obs.A_ki <= 0 or g_i <= 0:
            logger.debug(
                "_estimate_optical_depth(%s @ %.3f nm): tau=0 because "
                "A_ki=%.3e or g_i=%.3e is non-positive",
                element,
                obs.wavelength_nm,
                obs.A_ki,
                g_i,
            )
            return 0.0
        f_lu = 1.4992 * (lambda_cm**2) * obs.A_ki * (obs.g_k / g_i)

        # Doppler 1/e half-width in Hz: Δν_D = (ν₀/c) · √(2 k T / M).
        # Use M_PROTON (in kg) consistent with KB (in J/K); v_th comes
        # out in m/s; convert to CGS at the end.
        atomic_mass_amu = _ATOMIC_MASS_AMU.get(element, _DEFAULT_ATOMIC_MASS_AMU)
        mass_kg = atomic_mass_amu * M_PROTON
        v_th_m_per_s = np.sqrt(2.0 * KB * temperature_K / mass_kg)
        # ν₀ in Hz from λ in cm: ν₀ = c / λ with c in cm/s.
        nu_0_Hz = _C_CGS_CM_PER_S / lambda_cm
        # v_th cancels c when going from m/s ÷ c[m/s] — keep SI inside the
        # ratio for numerical clarity.
        delta_nu_D_Hz = nu_0_Hz * (v_th_m_per_s / C_LIGHT)

        if delta_nu_D_Hz <= 0:
            logger.debug(
                "_estimate_optical_depth(%s @ %.3f nm): tau=0 because "
                "Doppler width Δν_D=%.3e Hz is non-positive (T=%.1fK, "
                "M=%.2f amu)",
                element,
                obs.wavelength_nm,
                delta_nu_D_Hz,
                temperature_K,
                atomic_mass_amu,
            )
            return 0.0

        # Line-center Doppler profile value (normalized so ∫φ dν = 1):
        #     φ(ν₀) = 1 / (√π · Δν_D)
        phi_nu0 = 1.0 / (np.sqrt(np.pi) * delta_nu_D_Hz)

        # Final optical depth — units: cm²·Hz × (dimensionless) × cm⁻³ ×
        # cm × Hz⁻¹  =  dimensionless ✓
        tau = _PI_E2_OVER_MEC_CGS * f_lu * n_lower * self.plasma_length_cm * phi_nu0

        return max(0.0, tau)

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

        For isolated lines or single-line analysis use the observable-gated
        corrector
        (:class:`cflibs.inversion.physics.self_absorption_observable.ObservableSelfAbsorptionCorrector`).
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
            # Historical behaviour: the fallback called the (now deleted)
            # plasma-state correct() with empty concentrations, which forced
            # tau = 0 for every line — an exact pass-through. Keep that no-op
            # contract explicitly.
            warnings_list.append(
                f"Only {len(multiplet_lines)} lines available; COG needs >= 3 — "
                "no correction applied"
            )
            return self._passthrough_result(observations, warnings_list)

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
            # Same no-op contract as the < 3 lines fallback above.
            return self._passthrough_result(observations, warnings_list)

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
            n_corrected=len(
                # See the note at the matching comprehension above: the
                # ``abs(...) > 1e-9`` form is semantically equivalent to
                # ``!= 1.0`` since ``correction_factor`` is a sentinel.
                [c for c in corrections.values() if abs(c.correction_factor - 1.0) > 1e-9]
            ),
            n_masked=len(masked_obs),
            max_optical_depth=max_tau,
            warnings=warnings_list,
        )

    def cross_check_with_doublets(
        self,
        lines: Sequence[LineObservation],
        cdsb_result: SelfAbsorptionResult,
        sigma_threshold: float = 2.0,
    ) -> List[DoubletRatioResult]:
        """
        Second-opinion cross-check of CDSB-derived optical depths using the
        closed-form, single-spectrum doublet-ratio method of Pace et al.,
        Spectrochim. Acta B 2025
        (https://www.sciencedirect.com/science/article/abs/pii/S0584854725001995).

        For every detected doublet pair (two lines from the same upper level
        + same species) the optical depth tau_doublet is recovered from
        f(tau_1) / f(tau_1 / r_theory) = r_meas / r_theory and compared
        against the CDSB-derived tau for the stronger line of the pair.
        Disagreements with absolute z-score > ``sigma_threshold`` are
        flagged via ``logger.warning``.

        Sigma estimate: doublet uncertainty is propagated from the line
        intensity uncertainties through the linearised root condition;
        CDSB uncertainty is taken to be 30% of tau (the "high-tau limit"
        warned about in this module's docstring) when no covariance
        estimate is available from the CDSB step itself. These are
        deliberately conservative defaults — refine when CDSB grows a
        proper covariance estimator.

        Parameters
        ----------
        lines : sequence of LineObservation
            Same line list used to drive the CDSB correction.
        cdsb_result : SelfAbsorptionResult
            Output of :py:meth:`correct` or :py:meth:`correct_with_cog`,
            which provides per-wavelength CDSB optical depths.
        sigma_threshold : float, default 2.0
            Threshold on |tau_doublet - tau_cdsb| / max(sigma) above which
            a warning is logged.

        Returns
        -------
        list of DoubletRatioResult
            One entry per detected doublet pair, with
            ``agreement_with_cdsb_sigma`` populated.
        """
        results: List[DoubletRatioResult] = []
        pairs = find_doublet_pairs(lines)

        for line1, line2 in pairs:
            try:
                doublet_res = correct_via_doublet_ratio(line1, line2)
            except ValueError as exc:
                logger.warning(
                    "Doublet cross-check skipped for pair " "(%.3f nm, %.3f nm): %s",
                    line1.wavelength_nm,
                    line2.wavelength_nm,
                    exc,
                )
                continue

            # Map the stronger line (line1, shorter wavelength by convention,
            # but the higher-tau end of the pair) to the CDSB tau bookkeeping.
            cdsb_corr = cdsb_result.corrections.get(line1.wavelength_nm)
            if cdsb_corr is None:
                # CDSB did not produce a result for this line — skip cross-check
                results.append(doublet_res)
                continue

            tau_cdsb = cdsb_corr.optical_depth
            tau_doublet = doublet_res.tau_1

            # Conservative uncertainty estimates.
            # Doublet sigma: fractional uncertainty in the measured ratio
            # propagates ~linearly into tau in the moderate-tau regime.
            sigma_r_meas = 0.0
            if line1.intensity > 0 and line2.intensity > 0:
                rel_1 = line1.intensity_uncertainty / line1.intensity
                rel_2 = line2.intensity_uncertainty / line2.intensity
                sigma_r_meas = doublet_res.r_measured * np.sqrt(rel_1**2 + rel_2**2)
            # df/dtau ~ -1/2 + tau/3 - ...  for small tau; use |df/dtau| ~ 0.5
            # to avoid divide-by-zero. This is a conservative ~2x overestimate.
            sigma_doublet = max(sigma_r_meas / max(0.1, doublet_res.r_theory), 1e-3)
            sigma_cdsb = max(0.3 * abs(tau_cdsb), 1e-3)

            denom = max(sigma_doublet, sigma_cdsb)
            agreement_sigma = (tau_doublet - tau_cdsb) / denom
            doublet_res.agreement_with_cdsb_sigma = agreement_sigma

            if abs(agreement_sigma) > sigma_threshold:
                logger.warning(
                    "Self-absorption cross-check disagreement at %.3f nm "
                    "(pair %.3f/%.3f nm, %s %s): tau_doublet=%.3f, "
                    "tau_cdsb=%.3f, |z|=%.2f > %.2f",
                    line1.wavelength_nm,
                    *doublet_res.wavelength_pair_nm,
                    line1.element,
                    line1.ionization_stage,
                    tau_doublet,
                    tau_cdsb,
                    abs(agreement_sigma),
                    sigma_threshold,
                )

            results.append(doublet_res)

        return results


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
