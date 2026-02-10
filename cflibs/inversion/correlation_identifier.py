"""
Correlation-based element identification (modernized Labutin method).

Implements classic and vector-accelerated modes for identifying elements
from experimental spectra using model spectrum correlation matching.
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr

from cflibs.inversion.element_id import (
    IdentifiedLine,
    ElementIdentification,
    ElementIdentificationResult,
)
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
        Minimum confidence threshold for detection (default: 0.3)
    T_range_K : Tuple[float, float]
        Temperature range for classic mode in Kelvin (default: (8000, 12000))
    n_e_range_cm3 : Tuple[float, float]
        Electron density range for classic mode in cm⁻³ (default: (3e16, 3e17))
    T_steps : int
        Temperature grid steps for classic mode (default: 5)
    n_e_steps : int
        Density grid steps for classic mode (default: 3)

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
        wavelength_tolerance_nm: float = 0.1,
        top_k: int = 10,
        min_confidence: float = 0.3,
        T_range_K: Tuple[float, float] = (8000, 12000),
        n_e_range_cm3: Tuple[float, float] = (3e16, 3e17),
        T_steps: int = 5,
        n_e_steps: int = 3,
    ):
        self.atomic_db = atomic_db
        self.vector_index = vector_index
        self.elements = elements
        self.wavelength_tolerance_nm = wavelength_tolerance_nm
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.T_range_K = T_range_K
        self.n_e_range_cm3 = n_e_range_cm3
        self.T_steps = T_steps
        self.n_e_steps = n_e_steps

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

        # Detect experimental peaks
        peak_indices, _ = find_peaks(
            intensity, height=np.max(intensity) * 0.05, distance=5
        )
        experimental_peaks = [(int(idx), float(wavelength[idx])) for idx in peak_indices]

        logger.info(f"Detected {len(experimental_peaks)} experimental peaks")

        # Run identification
        if mode == "classic":
            element_scores = self._identify_classic(wavelength, intensity)
        elif mode == "vector":
            element_scores = self._identify_vector(wavelength, intensity)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Apply relative threshold filter (only if multiple elements)
        if element_scores and len(element_scores) > 1:
            scores = [score for _, score, _, _, _ in element_scores]
            median_score = np.median(scores)
            relative_threshold = 1.5 * median_score
            logger.debug(f"Relative threshold: {relative_threshold:.3f} (1.5 × median {median_score:.3f})")
        else:
            relative_threshold = 0.0  # No relative filter for single element

        # Build result
        detected_elements = []
        rejected_elements = []

        for element, score, confidence, matched_lines, unmatched_lines in element_scores:
            # Apply both absolute (min_confidence) and relative threshold
            passes_threshold = confidence >= self.min_confidence and score >= relative_threshold
            elem_id = ElementIdentification(
                element=element,
                detected=passes_threshold,
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

        element_scores = []

        elements_to_search = self.elements if self.elements is not None else self.atomic_db.get_available_elements()
        for element in elements_to_search:
            # Get transitions
            transitions = self.atomic_db.get_transitions(
                element, wavelength_min=wl_min, wavelength_max=wl_max
            )

            if not transitions:
                logger.debug(f"No transitions for {element} in wavelength range")
                element_scores.append((element, 0.0, 0.0, [], []))
                continue

            # Compute correlations for each (T, n_e) point
            correlations = []
            for T_K in T_grid:
                T_eV = T_K * KB_EV
                for n_e in n_e_grid:
                    model_spectrum = self._generate_model_spectrum(intensity, 
                        element, transitions, wavelength, T_eV, n_e
                    )
                    # Pearson correlation
                    if np.std(model_spectrum) > 1e-10 and np.std(intensity) > 1e-10:
                        corr, _ = pearsonr(intensity, model_spectrum)
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

        for trans in transitions:
            # Partition function
            U = self.saha_solver.calculate_partition_function(
                element, trans.ionization_stage, T_eV
            )

            # Boltzmann factor: eps = A_ki * g_k * exp(-E_k/kT) / U(T)
            eps = trans.A_ki * trans.g_k * np.exp(-trans.E_k_ev / T_eV) / U

            # Add Gaussian line profile (simple approximation)
            sigma = 0.05  # nm, simple width
            gaussian = np.exp(-0.5 * ((wavelength - trans.wavelength_nm) / sigma) ** 2)
            model_spectrum += eps * gaussian

        # Normalize to match experimental scale
        if np.max(model_spectrum) > 1e-10:
            model_spectrum = model_spectrum / np.max(model_spectrum) * np.max(intensity)

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
        # Detect peaks
        peak_indices, _ = find_peaks(intensity, height=np.max(intensity) * 0.05, distance=5)
        peak_wavelengths = wavelength[peak_indices]
        peak_intensities = intensity[peak_indices]

        matched_lines = []
        unmatched_lines = []

        for trans in transitions:
            # Find nearest experimental peak
            distances = np.abs(peak_wavelengths - trans.wavelength_nm)
            if len(distances) > 0:
                nearest_idx = np.argmin(distances)
                min_distance = distances[nearest_idx]

                if min_distance <= self.wavelength_tolerance_nm:
                    # Matched
                    matched_lines.append(
                        IdentifiedLine(
                            wavelength_exp_nm=float(peak_wavelengths[nearest_idx]),
                            wavelength_th_nm=trans.wavelength_nm,
                            element=element,
                            ionization_stage=trans.ionization_stage,
                            intensity_exp=float(peak_intensities[nearest_idx]),
                            emissivity_th=0.0,  # Could compute from Boltzmann
                            transition=trans,
                            correlation=0.0,
                            is_interfered=False,
                            interfering_elements=[],
                        )
                    )
                else:
                    unmatched_lines.append(trans)
            else:
                unmatched_lines.append(trans)

        return matched_lines, unmatched_lines
