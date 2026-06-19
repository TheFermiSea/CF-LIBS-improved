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
from cflibs.inversion.common.element_id import (
    ElementIdentification,
    ElementIdentificationResult,
)
from cflibs.inversion.identify._coverage import (
    CoverageTracker,
    merge_coverage_into_parameters,
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
    nnls_min_relative_coeff : float
        Relative-magnitude detection floor forwarded to the Stage-1
        :class:`SpectralNNLSIdentifier`. Defaults to ``0.0`` (recall-favoring)
        rather than the standalone NNLS default of ``0.05``. The standalone
        5%-of-total-mass floor is calibrated for *precision* on a near-orthogonal
        small-candidate basis (blocker NNLS-GAUSS-BASIS-4); applied here it
        scales with the candidate count and silently drops legitimate
        minor/major elements (Si, Mg, Na, K) on real multi-element spectra
        where each true element holds only a few percent of the total
        coefficient mass. The hybrid_union arm's job is recall — its precision
        comes from the SNR gate plus ALIAS agreement (require_both) — so the
        floor is disabled here by default. See the PR fixing the #215
        hybrid_union recall regression. Standalone ``SpectralNNLSIdentifier``
        is unaffected and keeps its precision floor.
    alias_detection_threshold : float
        ALIAS confidence threshold ``C_th`` for Stage 2 confirmation: the
        wrapped ALIAS detects an element when ``k_det > C_th`` (Noel 2025
        sec 3.8). Default ``0.5`` (the paper strict C_th). The old ``0.05``
        was a CL-floor on the deflated metric and is "accept everything" on
        the k_det scale, which neutered this confirmation stage.
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
        nnls_min_relative_coeff: float = 0.0,
        alias_detection_threshold: float = 0.5,
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
        self.nnls_min_relative_coeff = float(nnls_min_relative_coeff)
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
        spectrum_id: "str | None" = None,
    ) -> ElementIdentificationResult:
        """
        Run two-stage identification.

        Parameters
        ----------
        spectrum_id : str, optional
            Caller-supplied identifier for this spectrum.  Forwarded
            to the wrapped ALIAS identifier so its per-element L2/L3/L4
            coverage records are correlated with this spectrum.  Hybrid
            itself emits an additional summary record at the end.

        Returns
        -------
        ElementIdentificationResult
            Elements detected by both (or either) stage, with metadata
            recording per-stage decisions.
        """
        from cflibs.inversion.identify.alias import ALIASIdentifier
        from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier

        # Detection-coverage tracker -- additive telemetry only.  Hybrid
        # delegates per-element scoring to ALIAS, which records its own
        # L2/L3/L4 telemetry under identifier="alias"; this tracker
        # surfaces the *hybrid* combined decision under
        # identifier="hybrid_nnls_alias".
        coverage = CoverageTracker(
            spectrum_id=spectrum_id if spectrum_id is not None else "<unset>",
            identifier_name="hybrid_nnls_alias",
        )

        # ---- Stage 1: NNLS screening ----
        nnls_detected, nnls_scores, nnls_snrs = self._run_nnls_stage(
            SpectralNNLSIdentifier, wavelength, intensity
        )

        # ---- Stage 2: ALIAS confirmation ----
        alias_result, alias_detected, alias_scores, alias_elements_map = self._run_alias_stage(
            ALIASIdentifier, wavelength, intensity, nnls_detected, spectrum_id
        )

        # ---- Combine stages ----
        if self.require_both:
            final_detected = nnls_detected & alias_detected
        else:
            final_detected = nnls_detected | alias_detected

        all_element_ids = self._build_all_element_ids(
            coverage,
            alias_result,
            final_detected,
            nnls_detected,
            alias_detected,
            nnls_scores,
            nnls_snrs,
            alias_scores,
            alias_elements_map,
        )

        detected_elements = [e for e in all_element_ids if e.detected]
        rejected_elements = [e for e in all_element_ids if not e.detected]

        # Detection-coverage finalisation: peak count from ALIAS (the
        # peaks reported on the result are ALIAS's), then emit summary.
        coverage.set_n_peaks(alias_result.n_peaks)
        coverage.emit_summary()

        base_parameters = {
            "nnls_detection_snr": self.nnls_detection_snr,
            "alias_detection_threshold": self.alias_detection_threshold,
            "require_both": float(self.require_both),
            "n_nnls_candidates": len(nnls_detected),
            "n_alias_confirmed": len(alias_detected),
            "n_final_detected": len(final_detected),
        }

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=all_element_ids,
            experimental_peaks=alias_result.experimental_peaks,
            n_peaks=alias_result.n_peaks,
            n_matched_peaks=alias_result.n_matched_peaks,
            n_unmatched_peaks=alias_result.n_unmatched_peaks,
            algorithm="hybrid_nnls_alias",
            parameters=merge_coverage_into_parameters(base_parameters, coverage.build_payload()),
        )

    def _run_nnls_stage(
        self,
        nnls_cls: type,
        wavelength: np.ndarray,
        intensity: np.ndarray,
    ) -> "tuple[Set[str], Dict[str, float], Dict[str, float]]":
        """Stage 1: run NNLS screening and build score/SNR maps."""
        nnls_id = nnls_cls(
            basis_library=self.basis_library,
            detection_snr=self.nnls_detection_snr,
            min_relative_coeff=self.nnls_min_relative_coeff,
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

        return nnls_detected, nnls_scores, nnls_snrs

    def _run_alias_stage(
        self,
        alias_cls: type,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        nnls_detected: Set[str],
        spectrum_id: "str | None",
    ) -> "tuple[ElementIdentificationResult, Set[str], Dict[str, float], Dict[str, ElementIdentification]]":  # noqa: E501
        """Stage 2: run ALIAS confirmation (restricted to NNLS candidates)."""
        # Restrict ALIAS to NNLS candidates when require_both=True
        alias_elements = (
            [e for e in self.elements if e in nnls_detected] if self.require_both else self.elements
        )
        if not alias_elements:
            alias_elements = self.elements

        alias_id = alias_cls(
            atomic_db=self.atomic_db,
            elements=alias_elements,
            resolving_power=self.resolving_power,
            intensity_threshold_factor=self.alias_intensity_factor,
            detection_threshold=self.alias_detection_threshold,
            chance_window_scale=self.alias_chance_window_scale,
            max_lines_per_element=self.alias_max_lines,
        )
        alias_result = alias_id.identify(wavelength, intensity, spectrum_id=spectrum_id)
        alias_detected: Set[str] = {e.element for e in alias_result.detected_elements}

        # Build ALIAS score map
        alias_scores: Dict[str, float] = {}
        alias_elements_map: Dict[str, ElementIdentification] = {}
        for e in alias_result.all_elements:
            alias_scores[e.element] = e.score
            alias_elements_map[e.element] = e

        return alias_result, alias_detected, alias_scores, alias_elements_map

    @staticmethod
    def _record_hybrid_coverage(
        coverage: CoverageTracker,
        element: str,
        detected: bool,
        in_nnls: bool,
        in_alias: bool,
        nnls_scores: Dict[str, float],
        alias_scores: Dict[str, float],
        alias_zero_db: Set[str],
        alias_zero_matches: Set[str],
    ) -> None:
        """Mirror ALIAS L2/L3 counters and record the hybrid L4 decision."""
        # L2/L3 mirroring from ALIAS's coverage payload.
        if element in alias_zero_db:
            coverage.record_db_lines(element, 0)
        if element in alias_zero_matches:
            # No record_db_lines call here -- ALIAS will have
            # already classified L2 above when applicable.
            coverage.record_peak_matches(element, 0)
        # L4 -- the hybrid combined decision (both / either stage).
        coverage.record_fingerprint(
            element,
            passed=bool(detected),
            score=(
                0.0
                if not (in_nnls or in_alias)
                else float(nnls_scores.get(element, 0.0) + alias_scores.get(element, 0.0))
            ),
        )

    @staticmethod
    def _build_element_id(
        element: str,
        detected: bool,
        in_nnls: bool,
        in_alias: bool,
        nnls_scores: Dict[str, float],
        nnls_snrs: Dict[str, float],
        alias_scores: Dict[str, float],
        alias_elements_map: Dict[str, ElementIdentification],
    ) -> ElementIdentification:
        """Construct the combined :class:`ElementIdentification` for one element."""
        alias_eid = alias_elements_map.get(element)
        matched_lines = alias_eid.matched_lines if alias_eid else []
        unmatched_lines = alias_eid.unmatched_lines if alias_eid else []
        n_matched = alias_eid.n_matched_lines if alias_eid else 0
        n_total = alias_eid.n_total_lines if alias_eid else 0

        s_nnls = nnls_scores.get(element, 0.0)
        s_alias = alias_scores.get(element, 0.0)
        combined_score = float(np.sqrt(max(s_nnls, 0) * max(s_alias, 0)))

        return ElementIdentification(
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

    def _build_all_element_ids(
        self,
        coverage: CoverageTracker,
        alias_result: ElementIdentificationResult,
        final_detected: Set[str],
        nnls_detected: Set[str],
        alias_detected: Set[str],
        nnls_scores: Dict[str, float],
        nnls_snrs: Dict[str, float],
        alias_scores: Dict[str, float],
        alias_elements_map: Dict[str, ElementIdentification],
    ) -> List[ElementIdentification]:
        """Combine per-element stage decisions and emit hybrid coverage telemetry."""
        # Reach into ALIAS's coverage payload (now stored on
        # ``alias_result.parameters``) to mirror the L2/L3 per-element
        # counters at the hybrid layer.  This is purely additive
        # telemetry -- hybrid uses ALIAS's identification output
        # unchanged.
        alias_params = alias_result.parameters or {}
        alias_zero_db = set(alias_params.get("elements_with_zero_db_lines_in_range", []) or [])
        alias_zero_matches = set(alias_params.get("elements_with_zero_peak_matches", []) or [])

        all_element_ids: List[ElementIdentification] = []
        for element in self.elements:
            detected = element in final_detected
            in_nnls = element in nnls_detected
            in_alias = element in alias_detected

            self._record_hybrid_coverage(
                coverage,
                element,
                detected,
                in_nnls,
                in_alias,
                nnls_scores,
                alias_scores,
                alias_zero_db,
                alias_zero_matches,
            )

            eid = self._build_element_id(
                element,
                detected,
                in_nnls,
                in_alias,
                nnls_scores,
                nnls_snrs,
                alias_scores,
                alias_elements_map,
            )
            all_element_ids.append(eid)

        return all_element_ids
