"""
Two-stage hybrid element identifier: NNLS screening + ALIAS confirmation.

Inspired by ChemCam/SuperCam's two-stage pipeline: full-spectrum
decomposition narrows the candidate set, then peak-matching confirms
line-level evidence.  Combines complementary strengths:

- NNLS: handles blending, physically constrained, low false-positive rate
- ALIAS: validates individual line positions, resolves ambiguities

Stage 1 (NNLS): Decompose spectrum into element basis spectra with lenient
    threshold → candidate elements.
Stage 2 (ALIAS): Run peak-matching restricted to NNLS candidates → line
    confirmation.

An element is detected if it passes BOTH stages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set

import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.inversion.element_id import (
    ElementIdentification,
    ElementIdentificationResult,
)

if TYPE_CHECKING:
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.manifold.basis_library import BasisLibrary

logger = get_logger("inversion.hybrid_identifier")


class HybridIdentifier:
    """
    Two-stage element identifier: NNLS screening + ALIAS confirmation.

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database for ALIAS line lookup.
    basis_library : BasisLibrary
        Pre-computed basis library for NNLS decomposition.
    elements : list of str
        Elements to search for.
    resolving_power : float
        Resolving power of the spectrometer (for ALIAS).
    nnls_detection_snr : float
        NNLS SNR threshold for Stage 1 screening (lenient, default 1.5).
    nnls_continuum_degree : int
        Polynomial continuum degree for NNLS (default 3).
    alias_detection_threshold : float
        ALIAS detection threshold for Stage 2 confirmation (default 0.05).
    alias_intensity_factor : float
        ALIAS intensity threshold factor (default 3.0).
    alias_chance_window_scale : float
        ALIAS chance window scale (default 0.4).
    alias_max_lines : int
        ALIAS max lines per element (default 30).
    fallback_T_K : float
        Fallback temperature for NNLS if no index (default 8000).
    fallback_ne_cm3 : float
        Fallback electron density (default 1e17).
    require_both : bool
        If True (default), element must pass BOTH stages.
        If False, element passes if it passes EITHER stage (union mode).
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        basis_library: BasisLibrary,
        elements: List[str],
        resolving_power: float = 1000.0,
        nnls_detection_snr: float = 1.5,
        nnls_continuum_degree: int = 3,
        alias_detection_threshold: float = 0.05,
        alias_intensity_factor: float = 3.0,
        alias_chance_window_scale: float = 0.4,
        alias_max_lines: int = 30,
        fallback_T_K: float = 8000.0,
        fallback_ne_cm3: float = 1e17,
        require_both: bool = True,
    ):
        self.atomic_db = atomic_db
        self.basis_library = basis_library
        self.elements = elements
        self.resolving_power = resolving_power
        self.nnls_detection_snr = nnls_detection_snr
        self.nnls_continuum_degree = nnls_continuum_degree
        self.alias_detection_threshold = alias_detection_threshold
        self.alias_intensity_factor = alias_intensity_factor
        self.alias_chance_window_scale = alias_chance_window_scale
        self.alias_max_lines = alias_max_lines
        self.fallback_T_K = fallback_T_K
        self.fallback_ne_cm3 = fallback_ne_cm3
        self.require_both = require_both

    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
    ) -> ElementIdentificationResult:
        """
        Run two-stage identification.

        Returns
        -------
        ElementIdentificationResult
            Elements detected by both (or either) stage, with metadata
            recording per-stage decisions.
        """
        from cflibs.inversion.alias_identifier import ALIASIdentifier
        from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier

        # ---- Stage 1: NNLS screening ----
        nnls_id = SpectralNNLSIdentifier(
            basis_library=self.basis_library,
            detection_snr=self.nnls_detection_snr,
            continuum_degree=self.nnls_continuum_degree,
            fallback_T_K=self.fallback_T_K,
            fallback_ne_cm3=self.fallback_ne_cm3,
        )
        nnls_result = nnls_id.identify(wavelength, intensity)
        nnls_detected: Set[str] = {e.element for e in nnls_result.detected_elements}

        # Build a map of NNLS scores for metadata
        nnls_scores: Dict[str, float] = {}
        nnls_snrs: Dict[str, float] = {}
        for e in nnls_result.all_elements:
            nnls_scores[e.element] = e.score
            nnls_snrs[e.element] = e.metadata.get("nnls_snr", 0.0)

        # ---- Stage 2: ALIAS confirmation ----
        # Restrict ALIAS to NNLS candidates when require_both=True
        alias_elements = (
            [e for e in self.elements if e in nnls_detected] if self.require_both else self.elements
        )
        if not alias_elements:
            alias_elements = self.elements

        alias_id = ALIASIdentifier(
            atomic_db=self.atomic_db,
            elements=alias_elements,
            resolving_power=self.resolving_power,
            intensity_threshold_factor=self.alias_intensity_factor,
            detection_threshold=self.alias_detection_threshold,
            chance_window_scale=self.alias_chance_window_scale,
            max_lines_per_element=self.alias_max_lines,
        )
        alias_result = alias_id.identify(wavelength, intensity)
        alias_detected: Set[str] = {e.element for e in alias_result.detected_elements}

        # Build ALIAS score map
        alias_scores: Dict[str, float] = {}
        alias_elements_map: Dict[str, ElementIdentification] = {}
        for e in alias_result.all_elements:
            alias_scores[e.element] = e.score
            alias_elements_map[e.element] = e

        # ---- Combine stages ----
        if self.require_both:
            final_detected = nnls_detected & alias_detected
        else:
            final_detected = nnls_detected | alias_detected

        all_element_ids: List[ElementIdentification] = []
        for element in self.elements:
            detected = element in final_detected
            in_nnls = element in nnls_detected
            in_alias = element in alias_detected

            alias_eid = alias_elements_map.get(element)
            matched_lines = alias_eid.matched_lines if alias_eid else []
            unmatched_lines = alias_eid.unmatched_lines if alias_eid else []
            n_matched = alias_eid.n_matched_lines if alias_eid else 0
            n_total = alias_eid.n_total_lines if alias_eid else 0

            s_nnls = nnls_scores.get(element, 0.0)
            s_alias = alias_scores.get(element, 0.0)
            combined_score = float(np.sqrt(max(s_nnls, 0) * max(s_alias, 0)))

            eid = ElementIdentification(
                element=element,
                detected=detected,
                score=combined_score,
                confidence=combined_score,
                n_matched_lines=n_matched,
                n_total_lines=n_total,
                matched_lines=matched_lines,
                unmatched_lines=unmatched_lines,
                metadata={
                    "nnls_detected": in_nnls,
                    "alias_detected": in_alias,
                    "nnls_score": s_nnls,
                    "alias_score": s_alias,
                    "nnls_snr": nnls_snrs.get(element, 0.0),
                    "stage": (
                        "both"
                        if in_nnls and in_alias
                        else "nnls_only" if in_nnls else "alias_only" if in_alias else "neither"
                    ),
                },
            )
            all_element_ids.append(eid)

        detected_elements = [e for e in all_element_ids if e.detected]
        rejected_elements = [e for e in all_element_ids if not e.detected]

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=all_element_ids,
            experimental_peaks=alias_result.experimental_peaks,
            n_peaks=alias_result.n_peaks,
            n_matched_peaks=alias_result.n_matched_peaks,
            n_unmatched_peaks=alias_result.n_unmatched_peaks,
            algorithm="hybrid_nnls_alias",
            parameters={
                "nnls_detection_snr": self.nnls_detection_snr,
                "alias_detection_threshold": self.alias_detection_threshold,
                "require_both": float(self.require_both),
                "n_nnls_candidates": len(nnls_detected),
                "n_alias_confirmed": len(alias_detected),
                "n_final_detected": len(final_detected),
            },
        )
