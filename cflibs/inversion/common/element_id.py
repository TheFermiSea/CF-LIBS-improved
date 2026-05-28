"""
Shared data structures for automated element identification algorithms.

Three-tier result hierarchy: IdentifiedLine -> ElementIdentification -> ElementIdentificationResult.
Bridge function to_line_observations() converts results for downstream Boltzmann/solver pipeline.
"""

from dataclasses import dataclass, field
import math
from typing import List, Dict, Tuple, Any, Optional
from cflibs.atomic.structures import Transition
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.radiation.stark import stark_hwhm

# Konjević reference conditions for typical LIBS plasma (Konjević et al. 2002,
# J. Phys. Chem. Ref. Data 31, 819). These match the conditions under which
# the benchmark Stark parameters in the atomic database tabulate non-trivial
# FWHM (4-40 pm) for workhorse lines — comparable to or exceeding instrument
# FWHM at R~10^4. Used as the default when callers cannot supply live (n_e, T).
_KONJEVIC_REF_NE_CM3 = 1.0e17
_KONJEVIC_REF_T_K = 10000.0


@dataclass
class IdentifiedLine:
    """
    Represents a matched experimental line with theoretical transition.

    Attributes
    ----------
    wavelength_exp_nm : float
        Experimental peak wavelength in nm
    wavelength_th_nm : float
        Theoretical (database) wavelength in nm
    element : str
        Element symbol (e.g., 'Fe', 'Ti')
    ionization_stage : int
        Ionization stage (1=neutral, 2=singly ionized)
    intensity_exp : float
        Experimental intensity at peak (arbitrary units)
    emissivity_th : float
        Theoretical emissivity
    transition : Transition
        Atomic transition from database
    correlation : float
        Match quality metric (0-1, default 0.0)
    is_interfered : bool
        Whether line overlaps with other element (default False)
    interfering_elements : List[str]
        List of interfering element symbols (default [])
    """

    wavelength_exp_nm: float
    wavelength_th_nm: float
    element: str
    ionization_stage: int
    intensity_exp: float
    emissivity_th: float
    transition: Transition
    correlation: float = 0.0
    is_interfered: bool = False
    interfering_elements: List[str] = field(default_factory=list)


@dataclass
class ElementIdentification:
    """
    Element-level identification results.

    Attributes
    ----------
    element : str
        Element symbol (e.g., 'Fe', 'Ti')
    detected : bool
        Whether element passes detection threshold
    score : float
        0-1, comparable across algorithms
    confidence : float
        0-1, with additional quality factors
    n_matched_lines : int
        Number of matched experimental peaks
    n_total_lines : int
        Total theoretical lines in wavelength range
    matched_lines : List[IdentifiedLine]
        Detailed matched lines
    unmatched_lines : List[Transition]
        Theoretical lines with no experimental match
    metadata : Dict[str, Any]
        Algorithm-specific metadata (default {})
    """

    element: str
    detected: bool
    score: float
    confidence: float
    n_matched_lines: int
    n_total_lines: int
    matched_lines: List[IdentifiedLine]
    unmatched_lines: List[Transition]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ElementIdentificationResult:
    """
    Top-level identification results for all elements.

    Attributes
    ----------
    detected_elements : List[ElementIdentification]
        Elements passing detection threshold
    rejected_elements : List[ElementIdentification]
        Elements below detection threshold
    all_elements : List[ElementIdentification]
        All analyzed elements (detected + rejected)
    experimental_peaks : List[Tuple[int, float]]
        (index, wavelength) of detected peaks
    n_peaks : int
        Total experimental peaks
    n_matched_peaks : int
        Peaks matched to at least one element
    n_unmatched_peaks : int
        Peaks with no element match
    algorithm : str
        Algorithm used: "alias", "comb", or "correlation"
    parameters : Dict[str, Any]
        Algorithm parameters used; values may be float, str, bool, or list (default {})
    warnings : List[str]
        Any warnings generated (default [])
    """

    detected_elements: List[ElementIdentification]
    rejected_elements: List[ElementIdentification]
    all_elements: List[ElementIdentification]
    experimental_peaks: List[Tuple[int, float]]
    n_peaks: int
    n_matched_peaks: int
    n_unmatched_peaks: int
    algorithm: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def get_wavelength_tolerance(
    wavelength_nm: float,
    transition: Optional[Transition],
    resolving_power: float,
    fallback: float = 0.05,
    n_e_cm3: Optional[float] = None,
    T_K: Optional[float] = None,
) -> float:
    """
    Calculate Stark-aware wavelength tolerance per protocol.yaml §identification.wavelength_tolerance.

    Formula: sqrt(fwhm_inst**2 + omega_stark**2)
    where fwhm_inst = wavelength_nm / resolving_power and omega_stark is the
    Stark FWHM (2 × HWHM) scaled to the live plasma conditions when ``n_e_cm3``
    and ``T_K`` are supplied, or to Konjević reference conditions
    (n_e = 1e17 cm^-3, T = 10000 K — see Konjević et al. 2002, J. Phys. Chem.
    Ref. Data 31, 819) when they are not.

    The transition's ``stark_w`` attribute is interpreted as HWHM at
    REF_NE = 1e16 cm^-3, T = 10000 K (the convention used throughout
    ``cflibs/radiation/stark.py``). Scaling to the requested (n_e, T) follows
    the analytic power-law w_e = w_ref * (n_e/1e16) * (T/T_ref)^(-alpha) from
    :func:`cflibs.radiation.stark.stark_hwhm`. The HWHM is doubled to FWHM
    before quadrature combination with the instrument FWHM, matching the
    protocol formula.

    Bug history: this helper previously read ``stark_width_nm`` via
    ``getattr``, but the ``Transition`` dataclass exposes the attribute as
    ``stark_w``. As a result ``omega_stark`` collapsed to 0 for every line,
    silently degrading the protocol-prescribed Stark-aware tolerance to the
    pure-instrument FWHM. Aragón, Pellé & Aguilera 2011 (Anal. Bioanal. Chem.
    400, 3331) document Stark shifts up to 130 pm that cause Al II 281.6 nm
    overlap — the missing Stark term would let those near-coincidences slip
    through identification.

    Parameters
    ----------
    wavelength_nm : float
        Theoretical (database) wavelength in nm.
    transition : Transition, optional
        Transition object which may carry ``stark_w`` (HWHM at REF_NE=1e16,
        T_ref=10000 K) and ``stark_alpha`` (scaling exponent).
    resolving_power : float
        Instrumental resolving power (R = lambda/delta_lambda).
    fallback : float, optional
        Fixed tolerance used when neither Stark width nor a meaningful
        instrument FWHM is available (default: 0.05 nm, per
        protocol.yaml §identification.wavelength_tolerance.fallback_fixed).
    n_e_cm3 : float, optional
        Electron density in cm^-3 for dynamic Stark scaling. Defaults to the
        Konjević reference (1e17 cm^-3) when not supplied.
    T_K : float, optional
        Temperature in K for dynamic Stark scaling. Defaults to the Konjević
        reference (10000 K) when not supplied.

    Returns
    -------
    float
        Calculated tolerance in nm.
    """
    # fwhm_inst = lambda / R
    fwhm_inst = wavelength_nm / max(resolving_power, 1e-6)

    # omega_stark (Stark broadening FWHM) from transition metadata
    omega_stark = 0.0
    if transition is not None:
        # The Transition dataclass exposes Stark HWHM-at-reference as ``stark_w``;
        # the historical getattr key ``stark_width_nm`` was a typo (see bug history
        # in the docstring).
        stark_w_ref = getattr(transition, "stark_w", None)
        if stark_w_ref is not None and stark_w_ref > 0:
            stark_alpha = getattr(transition, "stark_alpha", None)
            # Use live (n_e, T) when supplied, otherwise Konjević reference.
            n_e_eff = n_e_cm3 if n_e_cm3 is not None else _KONJEVIC_REF_NE_CM3
            T_eff = T_K if T_K is not None else _KONJEVIC_REF_T_K
            # stark_hwhm returns HWHM in nm; double for FWHM to match the
            # protocol formula (Lorentzian sqrt-sum-square of FWHMs).
            omega_stark = 2.0 * stark_hwhm(
                n_e_eff,
                T_eff,
                stark_w_ref,
                stark_alpha,
            )

    # Apply formula if Stark width is available and positive
    if omega_stark > 0:
        return math.sqrt(fwhm_inst**2 + omega_stark**2)

    # Fallback to fixed 0.05 nm per protocol
    return fallback


def is_element_detected(
    element: str,
    score: float,
    n_matched_lines: int,
    min_score: float,
    min_lines: int,
) -> bool:
    """
    Decision rule for element detection with Tier-2 FP protection.

    Raises the effective score floor to 0.15 and the effective line floor
    to 2 for Mn, Na, K so low-confidence detections of those Tier-2
    elements don't slip past the global gates. This is the production
    rule used by the comb identifier prior to the inversion sub-package
    reorganization; the prior canonical body in this module diverged from
    it (forced n>=2 only, did not lift the score floor) and was
    unreachable because production imports routed through the flat-path
    shim. Reconciled 2026-05-27.

    Parameters
    ----------
    element : str
        Element symbol
    score : float
        Identification score (fingerprint)
    n_matched_lines : int
        Number of active lines/teeth
    min_score : float
        Global minimum score threshold
    min_lines : int
        Global minimum line count

    Returns
    -------
    bool
        True if element should be considered detected
    """
    effective_min_score = min_score
    effective_min_lines = min_lines

    if element in ("Mn", "Na", "K"):
        effective_min_score = max(min_score, 0.15)
        effective_min_lines = max(min_lines, 2)

    return score >= effective_min_score and n_matched_lines >= effective_min_lines


def to_line_observations(result: ElementIdentificationResult) -> List[LineObservation]:
    """
    Convert ElementIdentificationResult to LineObservation list for Boltzmann/solver pipeline.

    Filters out interfered lines and deduplicates by (element, ionization_stage, wavelength_th_nm).

    Parameters
    ----------
    result : ElementIdentificationResult
        Element identification result to convert

    Returns
    -------
    List[LineObservation]
        LineObservation objects for downstream CF-LIBS pipeline
    """
    observations = []
    seen = set()

    for element_id in result.detected_elements:
        for line in element_id.matched_lines:
            # Skip interfered lines
            if line.is_interfered:
                continue

            # Deduplicate by (element, ionization_stage, wavelength_th_nm)
            key = (line.element, line.ionization_stage, line.wavelength_th_nm)
            if key in seen:
                continue
            seen.add(key)

            # Create LineObservation with 2% intensity uncertainty floor at 1e-6
            intensity_uncertainty = max(line.intensity_exp * 0.02, 1e-6)

            obs = LineObservation(
                wavelength_nm=line.wavelength_th_nm,
                intensity=line.intensity_exp,
                intensity_uncertainty=intensity_uncertainty,
                element=line.element,
                ionization_stage=line.ionization_stage,
                E_k_ev=line.transition.E_k_ev,
                g_k=line.transition.g_k,
                A_ki=line.transition.A_ki,
            )
            observations.append(obs)

    return observations
