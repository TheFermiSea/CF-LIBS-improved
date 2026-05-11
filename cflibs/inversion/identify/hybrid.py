"""
Multi-method hybrid element identifier: Alias, Comb, and Correlation quorum.

Implements convergent evidence identification for trace elements. An element
is reported as 'identified' if it passes a quorum of multiple independent
identification methods (default 2-of-3 for Alias, Comb, and Correlation).

Modes:
- Quorum (Intersection): Requires >= min_votes (default 2) methods to agree.
- Union: Requires >= 1 method to agree.
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
    Multi-method element identifier: Alias, Comb, and Correlation quorum.

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database for line lookup.
    basis_library : BasisLibrary
        Pre-computed basis library for NNLS screening.
    elements : list of str
        Elements to search for.
    min_votes : int
        Minimum number of methods that must agree for detection (default 2).
    resolving_power : float
        Resolving power of the spectrometer (default 1000.0).
    nnls_screening : bool
        If True, use NNLS to narrow candidate set before running other methods (default True).
    nnls_detection_snr : float
        NNLS SNR threshold for screening (default 1.5).
    nnls_continuum_degree : int
        Polynomial continuum degree for NNLS (default 3).
    alias_detection_threshold : float
        ALIAS detection threshold (default 0.05).
    comb_detection_threshold : float
        Comb-matching detection threshold (default 0.1).
    correlation_detection_threshold : float
        Spectral correlation detection threshold (default 0.2).
    alias_intensity_factor : float
        ALIAS intensity threshold factor (default 3.0).
    alias_chance_window_scale : float
        ALIAS chance window scale (default 0.4).
    alias_max_lines : int
        ALIAS max lines per element (default 30).
    fallback_T_K : float
        Fallback temperature for NNLS (default 8000).
    fallback_ne_cm3 : float
        Fallback electron density (default 1e17).
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        basis_library: BasisLibrary,
        elements: List[str],
        min_votes: int = 2,
        resolving_power: float = 1000.0,
        nnls_screening: bool = True,
        nnls_detection_snr: float = 1.5,
        nnls_continuum_degree: int = 3,
        alias_detection_threshold: float = 0.05,
        comb_detection_threshold: float = 0.1,
        correlation_detection_threshold: float = 0.2,
        alias_intensity_factor: float = 3.0,
        alias_chance_window_scale: float = 0.4,
        alias_max_lines: int = 30,
        fallback_T_K: float = 8000.0,
        fallback_ne_cm3: float = 1e17,
    ):
        self.atomic_db = atomic_db
        self.basis_library = basis_library
        self.elements = elements
        self.min_votes = min_votes
        self.resolving_power = resolving_power
        self.nnls_screening = nnls_screening
        self.nnls_detection_snr = nnls_detection_snr
        self.nnls_continuum_degree = nnls_continuum_degree
        self.alias_detection_threshold = alias_detection_threshold
        self.comb_detection_threshold = comb_detection_threshold
        self.correlation_detection_threshold = correlation_detection_threshold
        self.alias_intensity_factor = alias_intensity_factor
        self.alias_chance_window_scale = alias_chance_window_scale
        self.alias_max_lines = alias_max_lines
        self.fallback_T_K = fallback_T_K
        self.fallback_ne_cm3 = fallback_ne_cm3

    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
    ) -> ElementIdentificationResult:
        """
        Run multi-method identification with quorum logic.

        Returns
        -------
        ElementIdentificationResult
            Elements detected by quorum, with metadata recording per-method decisions.
        """
        from cflibs.inversion.alias_identifier import ALIASIdentifier
        from cflibs.inversion.comb_identifier import CombIdentifier
        from cflibs.inversion.correlation_identifier import CorrelationIdentifier
        from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier

        # ---- Stage 0: NNLS screening (optional) ----
        candidates = self.elements
        nnls_detected: Set[str] = set()
        nnls_scores: Dict[str, float] = {}
        
        if self.nnls_screening:
            nnls_id = SpectralNNLSIdentifier(
                basis_library=self.basis_library,
                detection_snr=self.nnls_detection_snr,
                continuum_degree=self.nnls_continuum_degree,
                fallback_T_K=self.fallback_T_K,
                fallback_ne_cm3=self.fallback_ne_cm3,
            )
            nnls_result = nnls_id.identify(wavelength, intensity)
            nnls_detected = {e.element for e in nnls_result.detected_elements}
            nnls_scores = {e.element: e.score for e in nnls_result.all_elements}
            candidates = [e for e in self.elements if e in nnls_detected]
            if not candidates:
                candidates = self.elements

        # ---- Stage 1: Run the three identifiers ----
        # 1. ALIAS
        alias_id = ALIASIdentifier(
            atomic_db=self.atomic_db,
            elements=candidates,
            resolving_power=self.resolving_power,
            intensity_threshold_factor=self.alias_intensity_factor,
            detection_threshold=self.alias_detection_threshold,
            chance_window_scale=self.alias_chance_window_scale,
            max_lines_per_element=self.alias_max_lines,
        )
        alias_result = alias_id.identify(wavelength, intensity)
        alias_detected = {e.element for e in alias_result.detected_elements}
        alias_map = {e.element: e for e in alias_result.all_elements}

        # 2. Comb
        comb_id = CombIdentifier(
            atomic_db=self.atomic_db,
            elements=candidates,
            resolving_power=self.resolving_power,
            detection_threshold=self.comb_detection_threshold,
        )
        comb_result = comb_id.identify(wavelength, intensity)
        comb_detected = {e.element for e in comb_result.detected_elements}
        comb_scores = {e.element: e.score for e in comb_result.all_elements}

        # 3. Correlation
        corr_id = CorrelationIdentifier(
            atomic_db=self.atomic_db,
            elements=candidates,
            resolving_power=self.resolving_power,
            detection_threshold=self.correlation_detection_threshold,
        )
        corr_result = corr_id.identify(wavelength, intensity)
        corr_detected = {e.element for e in corr_result.detected_elements}
        corr_scores = {e.element: e.score for e in corr_result.all_elements}

        # ---- Stage 2: Quorum logic ----
        all_element_ids: List[ElementIdentification] = []
        for element in self.elements:
            votes = 0
            methods_passed = []
            if element in alias_detected:
                votes += 1
                methods_passed.append("alias")
            if element in comb_detected:
                votes += 1
                methods_passed.append("comb")
            if element in corr_detected:
                votes += 1
                methods_passed.append("correlation")

            detected = votes >= self.min_votes
            
            # Combine scores (geometric mean of available scores)
            s_alias = alias_map.get(element).score if element in alias_map else 0.0
            s_comb = comb_scores.get(element, 0.0)
            s_corr = corr_scores.get(element, 0.0)
            
            scores = [s for s in [s_alias, s_comb, s_corr] if s > 0]
            combined_score = float(np.exp(np.mean(np.log(scores)))) if scores else 0.0

            alias_eid = alias_map.get(element)
            eid = ElementIdentification(
                element=element,
                detected=detected,
                score=combined_score,
                confidence=combined_score,
                n_matched_lines=alias_eid.n_matched_lines if alias_eid else 0,
                n_total_lines=alias_eid.n_total_lines if alias_eid else 0,
                matched_lines=alias_eid.matched_lines if alias_eid else [],
                unmatched_lines=alias_eid.unmatched_lines if alias_eid else [],
                metadata={
                    "alias_detected": element in alias_detected,
                    "comb_detected": element in comb_detected,
                    "correlation_detected": element in corr_detected,
                    "nnls_detected": element in nnls_detected,
                    "votes": votes,
                    "methods_passed": methods_passed,
                    "alias_score": s_alias,
                    "comb_score": s_comb,
                    "correlation_score": s_corr,
                    "nnls_score": nnls_scores.get(element, 0.0),
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
            algorithm="hybrid_quorum",
            parameters={
                "min_votes": self.min_votes,
                "nnls_screening": self.nnls_screening,
                "alias_threshold": self.alias_detection_threshold,
                "comb_threshold": self.comb_detection_threshold,
                "correlation_threshold": self.correlation_detection_threshold,
            },
        )
