"""
Correlation-based element identification (modernized Labutin method).

Implements classic and vector-accelerated modes for identifying elements
from experimental spectra using model spectrum correlation matching.
"""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import math
import numpy as np
from scipy.stats import pearsonr

from cflibs.inversion.element_id import (
    IdentifiedLine,
    ElementIdentification,
    ElementIdentificationResult,
)
from cflibs.inversion.preprocessing import detect_peaks_auto
from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import Transition
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.correlation_identifier")


class CorrelationIdentifier:
    """
    Correlation-based element identification using model spectra.

    Supports two modes:
    - **classic**: Grid search over (T, n_e) with Pearson correlation
    - **vector**: Fast ANN search via FAISS VectorIndex with multi-model consensus

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database for transitions
    vector_index : VectorIndex, optional
        Pre-built vector index for vector mode
    vector_embedder : object, optional
        Fitted embedder with ``transform()`` method for vector-mode queries.
    library_metadata : List[dict], optional
        Per-spectrum metadata aligned with the vector index. Each item must
        expose element membership via ``element``, ``elements``, ``species``,
        or ``composition``.
    library_spectra : np.ndarray, optional
        Spectra aligned with ``library_metadata``. When provided, vector mode
        refines ANN scores with direct spectrum correlation.
    elements : List[str], optional
        Elements to consider (default: None, uses all elements from database)
    wavelength_tolerance_nm : float
        Wavelength matching tolerance in nm (default: 0.1)
    top_k : int
        Number of nearest neighbors for vector mode (default: 10)
    min_confidence : float
        Minimum confidence threshold for detection (default: 0.03)
    T_range_K : Tuple[float, float]
        Temperature range for classic mode in Kelvin (default: (8000, 12000))
    n_e_range_cm3 : Tuple[float, float]
        Electron density range for classic mode in cm⁻³ (default: (3e16, 3e17))
    T_steps : int
        Temperature grid steps for classic mode (default: 5)
    n_e_steps : int
        Density grid steps for classic mode (default: 3)
    resolving_power : float, optional
        Instrument resolving power (λ/Δλ). If set, per-line sigma is
        wavelength/resolving_power instead of fixed instrument_fwhm_nm (default: None).
    instrument_fwhm_nm : float
        Instrument spectral FWHM in nm (default: 0.05). Used when
        resolving_power is None to derive Gaussian sigma = FWHM / 2.355.
    max_lines_per_element : int
        Cap transitions per element by emissivity (default: 100)
    min_line_strength : float
        Minimum observable line strength A_ki * g_k (default: 1e4)
    reference_temperature : float
        Reference temperature in K for emissivity ranking (default: 10000.0)
    relative_threshold_scale : float
        Scale factor applied to median non-zero score for adaptive rejection
        (default: 1.5). Lower values increase recall; higher values reduce
        false positives.
    peak_region_threshold : float
        Normalized intensity threshold used to define peak-region masks for
        Pearson correlation in classic mode (default: 0.15).
    peak_region_min_points : int
        Minimum mask support before fallback from AND-mask to OR-mask in
        classic mode (default: 5).

    Attributes
    ----------
    atomic_db : AtomicDatabase
        Atomic database
    vector_index : VectorIndex or None
        Vector index for fast search
    saha_solver : SahaBoltzmannSolver
        Solver for partition functions

    Examples
    --------
    >>> # Classic mode
    >>> identifier = CorrelationIdentifier(atomic_db, elements=['Fe', 'Ti', 'Cr'])
    >>> result = identifier.identify(wavelength, intensity, mode="classic")
    >>> print(f"Detected: {[e.element for e in result.detected_elements]}")

    >>> # Vector mode with pre-built index
    >>> identifier = CorrelationIdentifier(atomic_db, vector_index=index)
    >>> result = identifier.identify(wavelength, intensity, mode="vector")
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        vector_index=None,
        vector_embedder=None,
        library_metadata: Optional[List[Mapping[str, Any]]] = None,
        library_spectra: Optional[np.ndarray] = None,
        elements: Optional[List[str]] = None,
        resolving_power: Optional[float] = None,
        wavelength_tolerance_nm: float = 0.1,
        top_k: int = 10,
        min_confidence: float = 0.03,
        T_range_K: Tuple[float, float] = (8000, 12000),
        n_e_range_cm3: Tuple[float, float] = (3e16, 3e17),
        T_steps: int = 5,
        n_e_steps: int = 3,
        instrument_fwhm_nm: float = 0.05,
        max_lines_per_element: int = 100,
        min_line_strength: float = 1e4,
        reference_temperature: float = 10000.0,
        relative_threshold_scale: float = 1.5,
        peak_region_threshold: float = 0.15,
        peak_region_min_points: int = 5,
    ):
        self.atomic_db = atomic_db
        self.resolving_power = resolving_power
        self.vector_index = vector_index
        self.vector_embedder = vector_embedder
        self.library_metadata = list(library_metadata) if library_metadata is not None else None
        self.library_spectra = None if library_spectra is None else np.asarray(library_spectra)
        self.elements = elements
        self.wavelength_tolerance_nm = wavelength_tolerance_nm
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.T_range_K = T_range_K
        self.n_e_range_cm3 = n_e_range_cm3
        self.T_steps = T_steps
        self.n_e_steps = n_e_steps
        self.instrument_fwhm_nm = instrument_fwhm_nm
        self.max_lines_per_element = max_lines_per_element
        self.min_line_strength = min_line_strength
        self.reference_temperature = reference_temperature
        self.relative_threshold_scale = relative_threshold_scale
        self.peak_region_threshold = peak_region_threshold
        self.peak_region_min_points = max(1, int(peak_region_min_points))

        self.saha_solver = SahaBoltzmannSolver(atomic_db)

    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        mode: str = "auto",
    ) -> ElementIdentificationResult:
        """
        Identify elements from experimental spectrum.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array
        mode : str
            Identification mode: "auto", "classic", or "vector"
            (default: "auto" uses vector if available, otherwise classic)

        Returns
        -------
        ElementIdentificationResult
            Identification results with detected and rejected elements

        Raises
        ------
        ValueError
            If mode is "vector" but the full vector workflow is not configured
        """
        # Resolve mode
        if mode == "auto":
            mode = "vector" if self._has_vector_workflow() else "classic"

        if mode == "vector" and not self._has_vector_workflow():
            raise ValueError(
                "mode='vector' requires vector_index, vector_embedder, and library_metadata"
            )

        logger.info(f"Running correlation identifier in {mode} mode")

        # Detect experimental peaks using canonical baseline-subtracted pipeline
        experimental_peaks, _, _ = detect_peaks_auto(
            wavelength,
            intensity,
            resolving_power=self.resolving_power,
        )

        logger.info(f"Detected {len(experimental_peaks)} experimental peaks")

        # Run identification
        if mode == "classic":
            element_scores = self._identify_classic(wavelength, intensity, experimental_peaks)
        elif mode == "vector":
            element_scores = self._identify_vector(wavelength, intensity, experimental_peaks)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Relative score filter: require score to stand out from median
        # Only apply when comparing 3+ elements (fewer can't form a noise floor)
        non_zero_scores = [s for _, s, _, _, _ in element_scores if s > 0]
        if len(non_zero_scores) >= 3:
            median_score = np.median(non_zero_scores)
            relative_threshold = min(1.0, self.relative_threshold_scale * median_score)
        else:
            relative_threshold = 0.0

        # Build result
        detected_elements = []
        rejected_elements = []

        for element, score, confidence, matched_lines, unmatched_lines in element_scores:
            elem_id = ElementIdentification(
                element=element,
                detected=confidence >= self.min_confidence and confidence >= relative_threshold,
                score=score,
                confidence=confidence,
                n_matched_lines=len(matched_lines),
                n_total_lines=len(matched_lines) + len(unmatched_lines),
                matched_lines=matched_lines,
                unmatched_lines=unmatched_lines,
                metadata={"correlation": score, "relative_threshold": relative_threshold},
            )

            if elem_id.detected:
                detected_elements.append(elem_id)
            else:
                rejected_elements.append(elem_id)

        # Count matched peaks
        matched_peak_wavelengths = set()
        for elem in detected_elements:
            for line in elem.matched_lines:
                matched_peak_wavelengths.add(line.wavelength_exp_nm)

        n_matched_peaks = len(matched_peak_wavelengths)
        n_unmatched_peaks = len(experimental_peaks) - n_matched_peaks

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=detected_elements + rejected_elements,
            experimental_peaks=experimental_peaks,
            n_peaks=len(experimental_peaks),
            n_matched_peaks=n_matched_peaks,
            n_unmatched_peaks=n_unmatched_peaks,
            algorithm="correlation",
            parameters={
                "mode": mode,
                "wavelength_tolerance_nm": self.wavelength_tolerance_nm,
                "min_confidence": self.min_confidence,
                "peak_region_threshold": self.peak_region_threshold,
                "peak_region_min_points": float(self.peak_region_min_points),
            },
        )

    def _identify_classic(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple],
    ) -> List[Tuple[str, float, float, List[IdentifiedLine], List[Transition]]]:
        """
        Classic mode: grid search over (T, n_e) with Pearson correlation.

        Returns
        -------
        List[Tuple[str, float, float, List[IdentifiedLine], List[Transition]]]
            List of (element, score, confidence, matched_lines, unmatched_lines)
        """
        wl_min, wl_max = wavelength.min(), wavelength.max()

        # Generate (T, n_e) grid
        T_grid = np.linspace(self.T_range_K[0], self.T_range_K[1], self.T_steps)
        n_e_grid = np.linspace(self.n_e_range_cm3[0], self.n_e_range_cm3[1], self.n_e_steps)

        element_scores: List[Tuple[str, float, float, List[Any], List[Any]]] = []

        elements_to_search = (
            self.elements if self.elements is not None else self.atomic_db.get_available_elements()
        )
        for element in elements_to_search:
            transitions = self._get_transitions_for_element(element, wl_min, wl_max)
            if not transitions:
                logger.debug(f"No transitions for {element} in wavelength range")
                element_scores.append((element, 0.0, 0.0, [], []))
                continue

            # Compute correlations for each (T, n_e) point
            # Paper (Labutin et al. 2013): correlate only in peak regions, not full spectrum
            correlations = []
            for T_K in T_grid:
                T_eV = T_K * KB_EV
                for n_e in n_e_grid:
                    model_spectrum = self._generate_model_spectrum(
                        intensity, element, transitions, wavelength, T_eV, n_e
                    )
                    # Peak-region mask: correlate only where BOTH spectra have significant
                    # signal to avoid baseline-dominated correlations.
                    i_min, i_max = intensity.min(), intensity.max()
                    m_min, m_max = model_spectrum.min(), model_spectrum.max()
                    sigma_threshold = float(self.peak_region_threshold)
                    if (i_max - i_min) > 1e-10 and (m_max - m_min) > 1e-10:
                        exp_norm = (intensity - i_min) / (i_max - i_min)
                        mod_norm = (model_spectrum - m_min) / (m_max - m_min)
                        peak_mask = (exp_norm >= sigma_threshold) & (mod_norm >= sigma_threshold)
                        # Fallback: if AND is too restrictive, use OR.
                        if np.sum(peak_mask) < self.peak_region_min_points:
                            peak_mask = (exp_norm >= sigma_threshold) | (
                                mod_norm >= sigma_threshold
                            )
                    else:
                        peak_mask = np.ones(len(intensity), dtype=bool)

                    # Pearson correlation on peak regions only
                    exp_peaks = intensity[peak_mask]
                    mod_peaks = model_spectrum[peak_mask]
                    if (
                        len(exp_peaks) > 2
                        and np.std(mod_peaks) > 1e-10
                        and np.std(exp_peaks) > 1e-10
                    ):
                        corr, _ = pearsonr(exp_peaks, mod_peaks)
                        correlations.append(corr)
                    else:
                        correlations.append(0.0)

            # Best correlation = element score
            best_corr = max(correlations) if correlations else 0.0
            score = np.clip(best_corr, 0.0, 1.0)
            confidence = score  # Simple mapping for now

            # Match lines to experimental peaks
            matched_lines, unmatched_lines = self._match_lines_to_peaks(
                element, transitions, wavelength, intensity, peaks
            )

            element_scores.append((element, score, confidence, matched_lines, unmatched_lines))

        return element_scores

    def _identify_vector(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple],
    ) -> List[Tuple[str, float, float, List[IdentifiedLine], List[Transition]]]:
        """
        Vector mode: ANN search via FAISS with multi-model consensus.

        Returns
        -------
        List[Tuple[str, float, float, List[IdentifiedLine], List[Transition]]]
            List of (element, score, confidence, matched_lines, unmatched_lines)

        Notes
        -----
        The vector workflow requires:
        - ``vector_index`` with ``search()``
        - ``vector_embedder`` with ``transform()``
        - ``library_metadata`` aligned with the index rows
        """
        if not self._has_vector_workflow():
            raise ValueError(
                "Vector mode requires vector_index, vector_embedder, and library_metadata"
            )

        query_embedding = np.asarray(self.vector_embedder.transform(np.asarray(intensity)[None, :]))
        distances, indices = self.vector_index.search(query_embedding, k=self.top_k)

        library_metadata = self.library_metadata or []
        candidate_weights: Dict[str, float] = defaultdict(float)
        candidate_counts: Dict[str, int] = defaultdict(int)
        candidate_best_distance: Dict[str, float] = {}

        for distance, neighbor_idx in zip(np.ravel(distances), np.ravel(indices)):
            idx = int(neighbor_idx)
            if idx < 0 or idx >= len(library_metadata):
                continue

            similarity = 1.0 / (1.0 + max(float(distance), 0.0))
            if self.library_spectra is not None and idx < len(self.library_spectra):
                candidate_spectrum = np.asarray(self.library_spectra[idx], dtype=np.float64)
                if candidate_spectrum.shape == intensity.shape:
                    if np.std(candidate_spectrum) > 1e-12 and np.std(intensity) > 1e-12:
                        corr, _ = pearsonr(intensity, candidate_spectrum)
                        similarity = 0.5 * similarity + 0.5 * np.clip((corr + 1.0) / 2.0, 0.0, 1.0)

            for element in self._metadata_elements(library_metadata[idx]):
                if self.elements is not None and element not in self.elements:
                    continue
                candidate_weights[element] += similarity
                candidate_counts[element] += 1
                best_distance = candidate_best_distance.get(element, float("inf"))
                candidate_best_distance[element] = min(best_distance, float(distance))

        if self.elements is not None:
            elements_to_search = list(self.elements)
        else:
            elements_to_search = sorted(candidate_weights.keys())

        if not elements_to_search:
            return []

        total_weight = sum(candidate_weights.values())
        wl_min, wl_max = wavelength.min(), wavelength.max()
        element_scores: List[Tuple[str, float, float, List[Any], List[Any]]] = []

        for element in elements_to_search:
            score = candidate_weights.get(element, 0.0) / max(total_weight, 1e-12)
            count_weight = candidate_counts.get(element, 0) / max(self.top_k, 1)
            best_similarity = 0.0
            if element in candidate_best_distance:
                best_similarity = 1.0 / (1.0 + max(candidate_best_distance[element], 0.0))
            confidence = np.clip(0.4 * score + 0.4 * count_weight + 0.2 * best_similarity, 0.0, 1.0)
            transitions = self._get_transitions_for_element(element, wl_min, wl_max)
            matched_lines, unmatched_lines = self._match_lines_to_peaks(
                element, transitions, wavelength, intensity, peaks
            )

            if score <= 0.0:
                matched_lines = []
                unmatched_lines = transitions

            element_scores.append((element, score, confidence, matched_lines, unmatched_lines))

        return element_scores

    def _has_vector_workflow(self) -> bool:
        """Return True when the full vector-mode workflow is configured."""
        if self.vector_index is None or self.vector_embedder is None:
            return False
        if self.library_metadata is None or len(self.library_metadata) == 0:
            return False
        return True

    def _metadata_elements(self, metadata: Mapping[str, Any]) -> List[str]:
        """Extract element symbols from a library metadata record."""
        if (
            "elements" in metadata
            and not isinstance(metadata["elements"], str)
            and isinstance(metadata["elements"], Iterable)
        ):
            return [str(element) for element in metadata["elements"]]
        if "element" in metadata:
            return [str(metadata["element"])]
        for key in ("species", "composition"):
            value = metadata.get(key)
            if isinstance(value, Mapping):
                return [
                    str(element) for element, fraction in value.items() if float(fraction) > 0.0
                ]
        return []

    def _get_transitions_for_element(
        self, element: str, wl_min: float, wl_max: float
    ) -> List[Transition]:
        """Load and rank transitions for a candidate element."""
        transitions = self.atomic_db.get_transitions(
            element, wavelength_min=wl_min, wavelength_max=wl_max
        )
        transitions = [t for t in transitions if t.A_ki * t.g_k >= self.min_line_strength]
        if len(transitions) > self.max_lines_per_element:
            kT = KB_EV * self.reference_temperature
            transitions = sorted(
                transitions,
                key=lambda t: t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT),
                reverse=True,
            )
            transitions = transitions[: self.max_lines_per_element]
        return transitions

    def _generate_model_spectrum(
        self,
        intensity: np.ndarray,
        element: str,
        transitions: List[Transition],
        wavelength: np.ndarray,
        T_eV: float,
        n_e: float,
    ) -> np.ndarray:
        """
        Generate model spectrum for element at (T, n_e).

        Uses Boltzmann distribution with Gaussian stick spectrum.

        Parameters
        ----------
        intensity : np.ndarray
            Experimental intensity array for normalization
        element : str
            Element symbol
        transitions : List[Transition]
            Atomic transitions
        wavelength : np.ndarray
            Wavelength grid in nm
        T_eV : float
            Temperature in eV
        n_e : float
            Electron density in cm⁻³

        Returns
        -------
        np.ndarray
            Model spectrum intensity on wavelength grid
        """

        model_spectrum = np.zeros_like(wavelength, dtype=np.float64)

        # Compute ionization fractions using Saha equation
        total_density = 1e15  # arbitrary reference density
        try:
            stage_densities = self.saha_solver.solve_ionization_balance(
                element, T_eV, n_e, total_density
            )
        except Exception:
            stage_densities = None

        default_sigma = self.instrument_fwhm_nm / 2.355

        for trans in transitions:
            # Partition function
            U = self.saha_solver.calculate_partition_function(element, trans.ionization_stage, T_eV)

            # Ion-stage population fraction from Saha balance
            if stage_densities is not None:
                W_q = stage_densities.get(trans.ionization_stage, 1.0) / max(total_density, 1e-30)
            else:
                W_q = 1.0  # Fallback: avoid zeroing model when Saha fails

            # Boltzmann factor weighted by ionization fraction
            eps = W_q * trans.A_ki * trans.g_k * np.exp(-trans.E_k_ev / T_eV) / U

            # Per-transition sigma from RP if available, else fixed default
            if self.resolving_power:
                sigma = (trans.wavelength_nm / self.resolving_power) / 2.355
            else:
                sigma = default_sigma

            gaussian = np.exp(-0.5 * ((wavelength - trans.wavelength_nm) / sigma) ** 2)
            model_spectrum += eps * gaussian

        # Robust normalization: 95th percentile instead of max (resistant to spikes)
        exp_scale = np.percentile(intensity, 95.0)
        model_scale = np.percentile(model_spectrum, 95.0)
        if model_scale > 1e-10 and exp_scale > 1e-10:
            model_spectrum = model_spectrum * (exp_scale / model_scale)

        return model_spectrum

    def _match_lines_to_peaks(
        self,
        element: str,
        transitions: List[Transition],
        wavelength: np.ndarray,
        intensity: np.ndarray,
        peaks: Optional[List[Tuple]] = None,
    ) -> Tuple[List[IdentifiedLine], List[Transition]]:
        """
        Match theoretical transitions to experimental peaks.

        Uses canonical peak detection and one-to-one greedy matching
        (closest distance first; each peak and transition used at most once).

        Parameters
        ----------
        element : str
            Element symbol
        transitions : List[Transition]
            Theoretical transitions
        wavelength : np.ndarray
            Experimental wavelength in nm
        intensity : np.ndarray
            Experimental intensity
        peaks : List[Tuple], optional
            Pre-detected peaks. If None, peaks are detected fresh.

        Returns
        -------
        matched_lines : List[IdentifiedLine]
            Matched lines
        unmatched_lines : List[Transition]
            Transitions with no experimental match
        """
        if peaks is None:
            peaks, _, _ = detect_peaks_auto(
                wavelength, intensity, resolving_power=self.resolving_power
            )
        if not peaks:
            return [], list(transitions)

        peak_wavelengths = np.array([p[1] for p in peaks])
        peak_intensities = np.array([intensity[p[0]] for p in peaks])

        # Build candidate matches: (distance, peak_idx, trans_idx)
        candidates = []
        for t_idx, trans in enumerate(transitions):
            distances = np.abs(peak_wavelengths - trans.wavelength_nm)
            for p_idx in range(len(peak_wavelengths)):
                if distances[p_idx] <= self.wavelength_tolerance_nm:
                    candidates.append((distances[p_idx], p_idx, t_idx))

        # Greedy one-to-one: sort by distance, assign first-come
        candidates.sort(key=lambda c: c[0])
        claimed_peaks: set = set()
        claimed_trans: set = set()

        matched_lines = []
        for _dist, p_idx, t_idx in candidates:
            if p_idx in claimed_peaks or t_idx in claimed_trans:
                continue
            claimed_peaks.add(p_idx)
            claimed_trans.add(t_idx)
            trans = transitions[t_idx]
            matched_lines.append(
                IdentifiedLine(
                    wavelength_exp_nm=float(peak_wavelengths[p_idx]),
                    wavelength_th_nm=trans.wavelength_nm,
                    element=element,
                    ionization_stage=trans.ionization_stage,
                    intensity_exp=float(peak_intensities[p_idx]),
                    emissivity_th=0.0,
                    transition=trans,
                    correlation=0.0,
                    is_interfered=False,
                    interfering_elements=[],
                )
            )

        unmatched_lines = [
            transitions[i] for i in range(len(transitions)) if i not in claimed_trans
        ]

        return matched_lines, unmatched_lines
