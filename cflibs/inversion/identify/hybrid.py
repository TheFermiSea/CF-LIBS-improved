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
    quorum_threshold : int
        Number of identifiers that must agree for detection (default 2).
        Used when multiple identifiers are active (Alias, Comb, Correlation).
    mode : str
        Identification mode: 'quorum' (default) or 'union'.
    comb_detection_threshold : float
        Detection threshold for CombIdentifier (default 0.1).
    correlation_threshold : float
        Detection threshold for CorrelationIdentifier (default 0.7).
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
        comb_detection_threshold: float = 0.1,
        correlation_threshold: float = 0.7,
        quorum_threshold: int = 2,
        mode: str = "quorum",
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
        self.comb_detection_threshold = comb_detection_threshold
        self.correlation_threshold = correlation_threshold
        self.quorum_threshold = quorum_threshold
        self.mode = mode
        self.fallback_T_K = fallback_T_K
        self.fallback_ne_cm3 = fallback_ne_cm3
        self.require_both = require_both

    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
    ) -> ElementIdentificationResult:
        """
        Run multi-stage convergent identification.

        Returns
        -------
        ElementIdentificationResult
            Elements detected by quorum/union of methods, with metadata
            recording per-method decisions.
        """
        from cflibs.inversion.alias_identifier import ALIASIdentifier
        from cflibs.inversion.comb_identifier import CombIdentifier
        from cflibs.inversion.correlation_identifier import CorrelationIdentifier
        from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier

        # ---- Stage 1: NNLS screening (optional candidate reduction) ----
        nnls_id = SpectralNNLSIdentifier(
            basis_library=self.basis_library,
            detection_snr=self.nnls_detection_snr,
            continuum_degree=self.nnls_continuum_degree,
            fallback_T_K=self.fallback_T_K,
            fallback_ne_cm3=self.fallback_ne_cm3,
        )
        nnls_result = nnls_id.identify(wavelength, intensity)
        nnls_detected: Set[str] = {e.element for e in nnls_result.detected_elements}

        # Candidate elements for expensive secondary checks
        candidates = self.elements
        if self.require_both and nnls_detected:
            candidates = [e for e in self.elements if e in nnls_detected]
        if not candidates:
            candidates = self.elements

        # ---- Stage 2: Convergent Identification (Alias + Comb + Correlation) ----
        # 1. Alias
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
        alias_detected: Set[str] = {e.element for e in alias_result.detected_elements}

        # 2. Comb
        comb_id = CombIdentifier(
            atomic_db=self.atomic_db,
            elements=candidates,
            resolving_power=self.resolving_power,
            detection_threshold=self.comb_detection_threshold,
        )
        comb_result = comb_id.identify(wavelength, intensity)
        comb_detected: Set[str] = {e.element for e in comb_result.detected_elements}

        # 3. Correlation
        corr_id = CorrelationIdentifier(
            atomic_db=self.atomic_db,
            elements=candidates,
            detection_threshold=self.correlation_threshold,
        )
        corr_result = corr_id.identify(wavelength, intensity)
        corr_detected: Set[str] = {e.element for e in corr_result.detected_elements}

        # Collect scores and metadata
        alias_map = {e.element: e for e in alias_result.all_elements}
        comb_map = {e.element: e for e in comb_result.all_elements}
        corr_map = {e.element: e for e in corr_result.all_elements}
        nnls_map = {e.element: e for e in nnls_result.all_elements}

        all_element_ids: List[ElementIdentification] = []
        q_thresh = self.quorum_threshold if self.mode == "quorum" else 1

        for element in self.elements:
            in_alias = element in alias_detected
            in_comb = element in comb_detected
            in_corr = element in corr_detected
            in_nnls = element in nnls_detected

            vote_count = sum([in_alias, in_comb, in_corr])
            detected = vote_count >= q_thresh

            # Score calculation (geometric mean of active identifiers)
            s_alias = alias_map.get(element).score if element in alias_map else 0.0
            s_comb = comb_map.get(element).score if element in comb_map else 0.0
            s_corr = corr_map.get(element).score if element in corr_map else 0.0
            s_nnls = nnls_map.get(element).score if element in nnls_map else 0.0

            active_scores = [s for s in [s_alias, s_comb, s_corr] if s > 0]
            if active_scores:
                combined_score = float(np.power(np.prod(active_scores), 1.0 / len(active_scores)))
            else:
                combined_score = 0.0

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
                    "nnls_detected": in_nnls,
                    "alias_detected": in_alias,
                    "comb_detected": in_comb,
                    "correlation_detected": in_corr,
                    "vote_count": int(vote_count),
                    "nnls_score": s_nnls,
                    "alias_score": s_alias,
                    "comb_score": s_comb,
                    "correlation_score": s_corr,
                    "mode": self.mode,
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
            algorithm=f"hybrid_quorum_{self.mode}",
            parameters={
                "nnls_detection_snr": self.nnls_detection_snr,
                "alias_detection_threshold": self.alias_detection_threshold,
                "comb_detection_threshold": self.comb_detection_threshold,
                "correlation_threshold": self.correlation_threshold,
                "quorum_threshold": int(self.quorum_threshold),
                "mode": self.mode,
                "n_final_detected": len(detected_elements),
            },
        )
