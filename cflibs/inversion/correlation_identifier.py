"""
Correlation-based element identification (modernized Labutin method).

Implements classic and vector-accelerated modes for identifying elements
from experimental spectra using model spectrum correlation matching.
"""

from typing import Any, List, Optional, Tuple
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
    ):
        self.atomic_db = atomic_db
        self.resolving_power = resolving_power
        self.vector_index = vector_index
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
            If mode is "vector" but no vector_index provided
        """
        # Resolve mode
        if mode == "auto":
            mode = "vector" if self.vector_index is not None else "classic"

        if mode == "vector" and self.vector_index is None:
            raise ValueError("mode='vector' requires vector_index to be provided")

        logger.info(f"Running correlation identifier in {mode} mode")

        # Detect experimental peaks using canonical baseline-subtracted pipeline
        # Store peaks for reuse in _match_lines_to_peaks to avoid redundant calls
        self._peaks, _, _ = detect_peaks_auto(
            wavelength,
            intensity,
            resolving_power=self.resolving_power,
        )
        experimental_peaks = self._peaks

        logger.info(f"Detected {len(experimental_peaks)} experimental peaks")

        # Run identification
        if mode == "classic":
            element_scores = self._identify_classic(wavelength, intensity)
        elif mode == "vector":
            element_scores = self._identify_vector(wavelength, intensity)
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
            },
        )

    def _identify_classic(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
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
            # Get transitions with strength filtering
            transitions = self.atomic_db.get_transitions(
                element, wavelength_min=wl_min, wavelength_max=wl_max
            )
            # Remove unobservable weak lines
            transitions = [t for t in transitions if t.A_ki * t.g_k >= self.min_line_strength]

            if not transitions:
                logger.debug(f"No transitions for {element} in wavelength range")
                element_scores.append((element, 0.0, 0.0, [], []))
                continue

            # Cap to strongest lines by estimated emissivity to avoid line-count disparity
            if len(transitions) > self.max_lines_per_element:
                kT = KB_EV * self.reference_temperature
                transitions = sorted(
                    transitions,
                    key=lambda t: t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT),
                    reverse=True,
                )
                transitions = transitions[: self.max_lines_per_element]

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
                    # signal to avoid baseline-dominated correlations. The 0.15 level and
                    # fallback minimum support (5 points) were tuned on sparse/weak-line
                    # cases to preserve recall while still suppressing continuum regions.
                    i_min, i_max = intensity.min(), intensity.max()
                    m_min, m_max = model_spectrum.min(), model_spectrum.max()
                    sigma_threshold = 0.15
                    if (i_max - i_min) > 1e-10 and (m_max - m_min) > 1e-10:
                        exp_norm = (intensity - i_min) / (i_max - i_min)
                        mod_norm = (model_spectrum - m_min) / (m_max - m_min)
                        peak_mask = (exp_norm >= sigma_threshold) & (mod_norm >= sigma_threshold)
                        # Fallback: if AND is too restrictive (< 5 pts), use OR
                        if np.sum(peak_mask) < 5:
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
                element, transitions, wavelength, intensity
            )

            element_scores.append((element, score, confidence, matched_lines, unmatched_lines))

        return element_scores

    def _identify_vector(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
    ) -> List[Tuple[str, float, float, List[IdentifiedLine], List[Transition]]]:
        """
        Vector mode: ANN search via FAISS with multi-model consensus.

        Returns
        -------
        List[Tuple[str, float, float, List[IdentifiedLine], List[Transition]]]
            List of (element, score, confidence, matched_lines, unmatched_lines)

        Raises
        ------
        NotImplementedError
            Vector mode not yet implemented
        """
        raise NotImplementedError("Vector mode not yet implemented. Use mode='classic'.")

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

        Returns
        -------
        matched_lines : List[IdentifiedLine]
            Matched lines
        unmatched_lines : List[Transition]
            Transitions with no experimental match
        """
        # Use cached peaks from identify() or detect fresh
        peaks = getattr(self, "_peaks", None)
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
