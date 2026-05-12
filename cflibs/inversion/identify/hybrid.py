"""
Hybrid element identifier with convergent multi-method validation.

Implements a quorum-based identification strategy requiring agreement
between multiple independent algorithms: ALIAS (peak matching), 
COMB (harmonic line-set matching), and Correlation (template matching).

An element is considered 'identified' if it passes a specified quorum
(default: 2 out of 3). This significantly reduces false positives
for trace elements where single-method evidence is prone to accidental
matches in complex spectra.

Modes:
- intersection: 2-of-3 quorum (default)
- union: 1-of-3 (any method matches)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set, Any

import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.inversion.element_id import (
    ElementIdentification,
    ElementIdentificationResult,
)

if TYPE_CHECKING:
    from cflibs.atomic.database import AtomicDatabase

logger = get_logger("inversion.hybrid_identifier")


class HybridIdentifier:
    """
    Multi-method hybrid identifier: ALIAS + COMB + Correlation.

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database for line lookups.
    elements : list of str
        Elements to search for.
    resolving_power : float
        Resolving power of the spectrometer (default 1000.0).
    quorum : int
        Number of methods required for detection (default 2).
        Use 1 for 'union' mode, 2 for 'intersection/quorum' mode.
    alias_params : dict, optional
        Parameters for ALIASIdentifier.
    comb_params : dict, optional
        Parameters for CombIdentifier.
    corr_params : dict, optional
        Parameters for CorrelationIdentifier.
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        elements: List[str],
        resolving_power: float = 1000.0,
        quorum: int = 2,
        alias_params: Dict[str, Any] = None,
        comb_params: Dict[str, Any] = None,
        corr_params: Dict[str, Any] = None,
    ):
        self.atomic_db = atomic_db
        self.elements = elements
        self.resolving_power = resolving_power
        self.quorum = quorum
        self.alias_params = alias_params or {}
        self.comb_params = comb_params or {}
        self.corr_params = corr_params or {}

    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
    ) -> ElementIdentificationResult:
        """
        Run multi-method identification and apply quorum logic.

        Returns
        -------
        ElementIdentificationResult
            Elements detected by quorum of stages.
        """
        from cflibs.inversion.alias_identifier import ALIASIdentifier
        from cflibs.inversion.comb_identifier import CombIdentifier
        from cflibs.inversion.correlation_identifier import CorrelationIdentifier

        # 1. Run ALIAS
        alias_id = ALIASIdentifier(
            atomic_db=self.atomic_db,
            elements=self.elements,
            resolving_power=self.resolving_power,
            **self.alias_params,
        )
        alias_res = alias_id.identify(wavelength, intensity)
        alias_detected = {e.element for e in alias_res.detected_elements}

        # 2. Run COMB
        comb_id = CombIdentifier(
            atomic_db=self.atomic_db,
            elements=self.elements,
            resolving_power=self.resolving_power,
            **self.comb_params,
        )
        comb_res = comb_id.identify(wavelength, intensity)
        comb_detected = {e.element for e in comb_res.detected_elements}

        # 3. Run Correlation
        corr_id = CorrelationIdentifier(
            atomic_db=self.atomic_db,
            elements=self.elements,
            resolving_power=self.resolving_power,
            **self.corr_params,
        )
        corr_res = corr_id.identify(wavelength, intensity)
        corr_detected = {e.element for e in corr_res.detected_elements}

        all_results = {
            "alias": alias_res,
            "comb": comb_res,
            "correlation": corr_res,
        }

        all_element_ids: List[ElementIdentification] = []
        for element in self.elements:
            votes = [
                element in alias_detected,
                element in comb_detected,
                element in corr_detected,
            ]
            n_votes = sum(votes)
            detected = n_votes >= self.quorum

            # Get scores from all methods
            scores = [
                next((e.score for e in res.all_elements if e.element == element), 0.0)
                for res in all_results.values()
            ]
            # Combined score is the mean of active scores (or max if none)
            active_scores = [s for s, v in zip(scores, votes) if v]
            combined_score = float(np.mean(active_scores)) if active_scores else float(np.max(scores))

            # Metadata and line info from ALIAS as primary reference
            alias_eid = next((e for e in alias_res.all_elements if e.element == element), None)
            
            eid = ElementIdentification(
                element=element,
                detected=detected,
                score=combined_score,
                confidence=combined_score * (n_votes / 3.0),
                n_matched_lines=alias_eid.n_matched_lines if alias_eid else 0,
                n_total_lines=alias_eid.n_total_lines if alias_eid else 0,
                matched_lines=alias_eid.matched_lines if alias_eid else [],
                unmatched_lines=alias_eid.unmatched_lines if alias_eid else [],
                metadata={
                    "n_votes": n_votes,
                    "methods": {
                        "alias": votes[0],
                        "comb": votes[1],
                        "correlation": votes[2],
                    },
                    "scores": {
                        "alias": scores[0],
                        "comb": scores[1],
                        "correlation": scores[2],
                    }
                },
            )
            all_element_ids.append(eid)

        detected_elements = [e for e in all_element_ids if e.detected]
        rejected_elements = [e for e in all_element_ids if not e.detected]

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=all_element_ids,
            experimental_peaks=alias_res.experimental_peaks,
            n_peaks=alias_res.n_peaks,
            n_matched_peaks=alias_res.n_matched_peaks,
            n_unmatched_peaks=alias_res.n_unmatched_peaks,
            algorithm="hybrid_quorum",
            parameters={
                "quorum": self.quorum,
                "n_elements": len(self.elements),
                "n_detected": len(detected_elements),
            },
        )
