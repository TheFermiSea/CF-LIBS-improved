"""
Shared data structures for automated element identification algorithms.

Three-tier result hierarchy: IdentifiedLine -> ElementIdentification -> ElementIdentificationResult.
Bridge function to_line_observations() converts results for downstream Boltzmann/solver pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from cflibs.atomic.structures import Transition
from cflibs.inversion.boltzmann import LineObservation


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
    parameters : Dict[str, float]
        Algorithm parameters used (default {})
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
    parameters: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


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
