"""
Comb template correlation algorithm for element identification.

Based on Gajarska et al. (2024), "Identification of elements in LIBS spectra
using the comb template matching method." This method uses triangular templates
to correlate with spectral peaks, treating atomic spectral lines as teeth in a comb.
"""

from typing import List, Dict, Optional, Tuple
import math
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import pearsonr

from cflibs.atomic.database import AtomicDatabase
from cflibs.core.constants import KB_EV
from cflibs.atomic.structures import Transition
from cflibs.inversion.element_id import (
    IdentifiedLine,
    ElementIdentification,
    ElementIdentificationResult,
)
from cflibs.inversion.preprocessing import detect_peaks_auto
from cflibs.core.logging_config import get_logger

logger = get_logger(__name__)


class CombIdentifier:
    """
    Automated element identification using comb template correlation.

    This algorithm identifies elements by correlating triangular templates (teeth)
    with spectral peaks at known transition wavelengths. For each element, it
    computes a fingerprint score based on the correlation strength of active teeth.

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Database containing atomic transitions
    baseline_window_nm : float, optional
        Window size for moving median baseline in nm (default: 10.0)
    threshold_percentile : float, optional
        Percentile for peak detection threshold (default: 85.0)
    min_correlation : float, optional
        Minimum fingerprint score for element detection (default: 0.10)
    tooth_activation_threshold : float, optional
        Per-tooth correlation threshold to count as active (default: 0.5)
    max_shift_pts : int, optional
        Maximum shift in data points for template matching (default: 5)
    min_width_pts : int, optional
        Minimum tooth width in data points (default: 5)
    max_width_factor : float, optional
        Maximum width as fraction of resolution element (default: 1.0)
    relative_threshold_scale : float, optional
        Scale factor applied to median non-zero score for adaptive rejection
        (default: 1.5). Lower values increase recall; higher values reduce
        false positives.
    elements : List[str], optional
        List of elements to search for (default: None means all in database)
    resolving_power : float, optional
        Instrument resolving power (λ/Δλ). If set, tooth width is derived from
        center wavelength / resolving_power instead of fixed 0.1 nm (default: None).
    min_aki_gk : float, optional
        Minimum observable line strength A_ki * g_k (default: 1e4).
    fingerprint_top_k : int, optional
        Maximum number of active teeth contributing to the fingerprint
        numerator and denominator (default: 10).

    Attributes
    ----------
    atomic_db : AtomicDatabase
        Atomic database instance
    baseline_window_nm : float
        Baseline window size in nm
    threshold_percentile : float
        Peak detection threshold percentile
    min_correlation : float
        Minimum tooth correlation threshold
    max_shift_pts : int
        Maximum template shift in points
    min_width_pts : int
        Minimum tooth width in points
    max_width_factor : float
        Maximum width scaling factor
    resolving_power : Optional[float]
        Instrument resolving power (None = use fixed 0.1 nm width).
    elements : Optional[List[str]]
        Elements to search (None = all)

    References
    ----------
    Gajarska et al. (2024), "Identification of elements in LIBS spectra
    using the comb template matching method."
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        resolving_power: Optional[float] = None,
        baseline_window_nm: float = 10.0,
        threshold_percentile: float = 85.0,
        min_correlation: float = 0.05,
        tooth_activation_threshold: float = 0.5,
        min_active_teeth: int = 2,
        max_shift_pts: int = 5,
        min_width_pts: int = 5,
        max_width_factor: float = 1.0,
        relative_threshold_scale: float = 1.2,
        elements: Optional[List[str]] = None,
        max_lines_per_element: int = 50,
        reference_temperature: float = 10000.0,
        min_aki_gk: float = 5e3,
        fingerprint_top_k: int = 10,
    ):
        if fingerprint_top_k <= 0:
            raise ValueError("fingerprint_top_k must be positive")
        self.atomic_db = atomic_db
        self.resolving_power = resolving_power
        self.baseline_window_nm = baseline_window_nm
        self.threshold_percentile = threshold_percentile
        self.min_correlation = min_correlation  # fingerprint detection threshold
        self.tooth_activation_threshold = (
            tooth_activation_threshold  # per-tooth threshold (paper: 0.5)
        )
        self.min_active_teeth = min_active_teeth
        self.max_shift_pts = max_shift_pts
        self.min_width_pts = min_width_pts
        self.max_width_factor = max_width_factor
        self.relative_threshold_scale = relative_threshold_scale
        self.elements = elements
        self.max_lines_per_element = max_lines_per_element
        self.reference_temperature = reference_temperature
        self.min_aki_gk = min_aki_gk
        self.fingerprint_top_k = fingerprint_top_k

    def identify(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> ElementIdentificationResult:
        """
        Identify elements in a spectrum using comb template correlation.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array (arbitrary units)

        Returns
        -------
        ElementIdentificationResult
            Complete identification results with algorithm="comb"
        """
        # Guard against empty arrays
        if len(wavelength) == 0 or len(intensity) == 0:
            logger.warning("Empty wavelength or intensity array")
            return ElementIdentificationResult(
                detected_elements=[],
                rejected_elements=[],
                all_elements=[],
                experimental_peaks=[],
                n_peaks=0,
                n_matched_peaks=0,
                n_unmatched_peaks=0,
                algorithm="comb",
                parameters={},
            )

        # Validate input arrays
        self._validate_inputs(wavelength, intensity)

        logger.info(
            f"Starting comb identification on spectrum: "
            f"{wavelength[0]:.1f}-{wavelength[-1]:.1f} nm, {len(wavelength)} points"
        )

        # Step 1: Estimate baseline and threshold
        baseline, threshold = self._estimate_baseline_threshold(wavelength, intensity)
        logger.debug(f"Baseline estimated, threshold={threshold:.2f}")

        # Step 2: Determine elements to search
        if self.elements is None:
            elements_to_search = self.atomic_db.get_available_elements()
        else:
            elements_to_search = self.elements

        # Step 3, 4, 5: Spectral matching and relative thresholding
        element_identifications = self._match_spectra(
            wavelength, intensity, baseline, threshold, elements_to_search
        )

        # Step 6: Split into detected and rejected
        detected_elements = [e for e in element_identifications if e.detected]
        rejected_elements = [e for e in element_identifications if not e.detected]

        # Step 7: Identify experimental peaks and count matches
        experimental_peaks, n_matched_peaks = self._detect_and_count_peaks(
            wavelength, intensity, detected_elements
        )

        result = ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=element_identifications,
            experimental_peaks=experimental_peaks,
            n_peaks=len(experimental_peaks),
            n_matched_peaks=n_matched_peaks,
            n_unmatched_peaks=len(experimental_peaks) - n_matched_peaks,
            algorithm="comb",
            parameters={
                "baseline_window_nm": self.baseline_window_nm,
                "threshold_percentile": self.threshold_percentile,
                "min_correlation": self.min_correlation,
                "max_shift_pts": float(self.max_shift_pts),
                "min_width_pts": float(self.min_width_pts),
                "max_width_factor": self.max_width_factor,
                "fingerprint_top_k": float(self.fingerprint_top_k),
            },
        )

        logger.info(
            f"Comb identification complete: {len(detected_elements)} detected, "
            f"{len(rejected_elements)} rejected"
        )

        return result

    def _validate_inputs(self, wavelength: np.ndarray, intensity: np.ndarray) -> None:
        """Validate input arrays for identification."""
        if len(wavelength) != len(intensity):
            raise ValueError(
                f"Wavelength and intensity arrays must have same length: "
                f"{len(wavelength)} vs {len(intensity)}"
            )

        if not np.all(np.diff(wavelength) > 0):
            raise ValueError("Wavelength array must be monotonically increasing")

    def _match_spectra(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        baseline: np.ndarray,
        threshold: float,
        elements_to_search: List[str],
    ) -> List[ElementIdentification]:
        """Match spectral lines for given elements."""
        element_teeth: Dict[str, List[dict]] = {}
        element_identifications = []

        for element in elements_to_search:
            # Get transitions for this element in wavelength range
            transitions = self._get_element_lines(element, wavelength[0], wavelength[-1])

            if not transitions:
                logger.debug(f"No transitions found for {element}")
                continue

            # Correlate each transition (tooth)
            teeth = []
            matched_lines = []
            unmatched_lines = []

            for trans in transitions:
                omega_stark = getattr(trans, "omega_stark", 0.0) or 0.0
                tooth_result = self._correlate_tooth(
                    wavelength,
                    intensity,
                    baseline,
                    trans.wavelength_nm,
                    threshold,
                    omega_stark=omega_stark,
                )
                tooth_result["transition"] = trans
                teeth.append(tooth_result)

                if tooth_result["active"]:
                    # Create IdentifiedLine for active tooth
                    matched_lines.append(
                        IdentifiedLine(
                            wavelength_exp_nm=tooth_result["center_nm"]
                            + tooth_result["best_shift"] * np.median(np.diff(wavelength)),
                            wavelength_th_nm=trans.wavelength_nm,
                            element=element,
                            ionization_stage=trans.ionization_stage,
                            intensity_exp=intensity[
                                np.clip(
                                    np.argmin(np.abs(wavelength - tooth_result["center_nm"]))
                                    + tooth_result["best_shift"],
                                    0,
                                    len(intensity) - 1,
                                )
                            ],
                            emissivity_th=0.0,
                            transition=trans,
                            correlation=tooth_result["best_correlation"],
                            is_interfered=False,
                            interfering_elements=[],
                        )
                    )
                else:
                    unmatched_lines.append(trans)

            # Compute fingerprint for this element
            fingerprint = self._compute_fingerprint(teeth)
            element_teeth[element] = teeth

            # Create ElementIdentification
            n_active_teeth = sum(1 for t in teeth if t["active"])
            detected = (
                fingerprint >= self.min_correlation and n_active_teeth >= self.min_active_teeth
            )
            element_id = ElementIdentification(
                element=element,
                detected=detected,
                score=fingerprint,
                confidence=fingerprint,
                n_matched_lines=len(matched_lines),
                n_total_lines=len(transitions),
                matched_lines=matched_lines,
                unmatched_lines=unmatched_lines,
                metadata={
                    "fingerprint": fingerprint,
                    "n_active_teeth": n_active_teeth,
                    "n_total_teeth": len(teeth),
                },
            )
            element_identifications.append(element_id)

        # Step 4: Analyze interferences across all elements
        element_teeth = self._analyze_interferences(element_teeth)

        # Update interfered status in element identifications
        for element_id in element_identifications:
            element = element_id.element
            if element in element_teeth:
                for line in element_id.matched_lines:
                    # Check if this line's wavelength is interfered
                    for tooth in element_teeth[element]:
                        if abs(tooth["center_nm"] - line.wavelength_th_nm) < 0.01 and tooth.get(
                            "interfering_elements"
                        ):
                            line.is_interfered = True
                            line.interfering_elements = tooth["interfering_elements"]

        # Step 5: Apply relative threshold to reject elements that don't stand out
        # Uses median of ALL non-zero scores as noise baseline; requires 3+ elements
        # to form a meaningful noise floor (2 elements can't distinguish signal from noise)
        non_zero_scores = [e.score for e in element_identifications if e.score > 0]
        if len(non_zero_scores) >= 3:
            median_score = np.median(non_zero_scores)
            relative_threshold = min(1.0, self.relative_threshold_scale * median_score)
        else:
            relative_threshold = 0.0

        for i, element_id in enumerate(element_identifications):
            if element_id.detected and element_id.score < relative_threshold:
                element_identifications[i] = ElementIdentification(
                    element=element_id.element,
                    detected=False,
                    score=element_id.score,
                    confidence=element_id.confidence,
                    n_matched_lines=element_id.n_matched_lines,
                    n_total_lines=element_id.n_total_lines,
                    matched_lines=element_id.matched_lines,
                    unmatched_lines=element_id.unmatched_lines,
                    metadata=element_id.metadata,
                )

        return element_identifications

    def _detect_and_count_peaks(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        detected_elements: List[ElementIdentification],
    ) -> Tuple[List[Tuple[int, float]], int]:
        """Detect experimental peaks and count matches."""
        experimental_peaks, _, _ = detect_peaks_auto(
            wavelength,
            intensity,
            resolving_power=self.resolving_power,
            baseline_window_nm=self.baseline_window_nm,
        )

        # Count matched peaks (peaks that have at least one identified line)
        matched_peak_wavelengths = set()
        for element_id in detected_elements:
            for line in element_id.matched_lines:
                matched_peak_wavelengths.add(line.wavelength_exp_nm)

        n_matched_peaks = sum(
            1
            for _, wl in experimental_peaks
            if any(abs(wl - mwl) < 0.1 for mwl in matched_peak_wavelengths)
        )
        return experimental_peaks, n_matched_peaks

    def _get_element_lines(self, element: str, wl_min: float, wl_max: float) -> List[Transition]:
        """
        Get all transitions for an element in wavelength range.

        Parameters
        ----------
        element : str
            Element symbol
        wl_min : float
            Minimum wavelength in nm
        wl_max : float
            Maximum wavelength in nm

        Returns
        -------
        List[Transition]
            All transitions for all ionization stages
        """
        # Get all ionization stages for this element
        transitions = self.atomic_db.get_transitions(
            element, wavelength_min=wl_min, wavelength_max=wl_max
        )
        # Remove unobservable weak lines
        transitions = [t for t in transitions if t.A_ki * t.g_k >= self.min_aki_gk]
        if len(transitions) > self.max_lines_per_element:
            kT = KB_EV * self.reference_temperature
            transitions = sorted(
                transitions,
                key=lambda t: t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT),
                reverse=True,
            )
            transitions = transitions[: self.max_lines_per_element]
        return transitions

    def _estimate_baseline_threshold(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate baseline using moving median and compute peak threshold.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array

        Returns
        -------
        baseline : np.ndarray
            Estimated baseline
        threshold : float
            Peak detection threshold
        """
        # Compute window size in points
        dwl_median = np.median(np.diff(wavelength))
        window_pts = int(self.baseline_window_nm / dwl_median)
        # Ensure odd window size for median filter
        if window_pts % 2 == 0:
            window_pts += 1
        window_pts = max(3, window_pts)  # Minimum window of 3

        # Compute moving median baseline
        baseline = median_filter(intensity, size=window_pts)

        # Compute residual
        residual = intensity - baseline

        # Threshold based on percentile of positive residuals
        positive_residual = residual[residual > 0]
        if len(positive_residual) > 0:
            threshold = np.percentile(positive_residual, self.threshold_percentile)
        else:
            threshold = 0.0

        return baseline, threshold

    def _build_triangular_template(self, width_pts: int) -> np.ndarray:
        """
        Create triangular (isosceles) template of given width.

        Parameters
        ----------
        width_pts : int
            Width of template in data points (must be odd)

        Returns
        -------
        np.ndarray
            Triangular template normalized to max=1.0
        """
        if width_pts % 2 == 0:
            width_pts += 1  # Ensure odd width

        template = np.zeros(width_pts)
        center_idx = width_pts // 2

        # Build triangle
        for i in range(width_pts):
            distance = abs(i - center_idx)
            template[i] = 1.0 - (distance / (center_idx + 1))

        # Normalize to max=1.0
        template = template / np.max(template)

        return template

    def _correlate_tooth(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        baseline: np.ndarray,
        center_nm: float,
        threshold: float,
        omega_stark: float = 0.0,
    ) -> dict:
        """
        Correlate triangular template with spectral data at a given wavelength.

        Searches over shifts and widths to maximize Pearson correlation.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array
        baseline : np.ndarray
            Baseline array
        center_nm : float
            Center wavelength for tooth in nm
        threshold : float
            Peak detection threshold

        Returns
        -------
        dict
            Dictionary with keys: center_nm, best_correlation, best_shift,
            best_width, active (bool)
        """
        # Find nearest index to center_nm
        center_idx = np.argmin(np.abs(wavelength - center_nm))

        # Estimate resolution element from wavelength spacing
        dwl = np.median(np.diff(wavelength))
        # Derive resolution from resolving power if available, else fallback
        if self.resolving_power:
            fwhm_inst = center_nm / self.resolving_power
        else:
            fwhm_inst = 0.1

        # Stark-aware tolerance: sqrt(fwhm_inst**2 + omega_stark**2)
        if omega_stark > 0:
            resolution_nm = math.sqrt(fwhm_inst**2 + omega_stark**2)
        else:
            resolution_nm = 0.05  # fallback_fixed

        max_width_pts = int((resolution_nm * self.max_width_factor) / dwl)
        max_width_pts = max(self.min_width_pts, max_width_pts)

        best_correlation = -1.0
        best_shift = 0
        best_width = self.min_width_pts

        # Search over widths (odd values only)
        for width in range(self.min_width_pts, max_width_pts + 1, 2):
            # Build template once per width (cache across shifts)
            template = self._build_triangular_template(width)

            # Search over shifts
            for shift in range(-self.max_shift_pts, self.max_shift_pts + 1):
                shifted_idx = center_idx + shift

                # Extract data segment
                half_width = width // 2
                start_idx = max(0, shifted_idx - half_width)
                end_idx = min(len(intensity), shifted_idx + half_width + 1)

                if end_idx - start_idx < self.min_width_pts:
                    continue

                data_segment = intensity[start_idx:end_idx] - baseline[start_idx:end_idx]

                # Ensure same length
                if len(data_segment) != len(template):
                    # Truncate or skip
                    continue

                # Compute Pearson correlation
                if np.std(data_segment) < 1e-10 or np.std(template) < 1e-10:
                    correlation = 0.0
                else:
                    correlation, _ = pearsonr(data_segment, template)
                    if np.isnan(correlation):
                        correlation = 0.0

                if correlation > best_correlation:
                    best_correlation = correlation
                    best_shift = shift
                    best_width = width

        # Clamp negative correlations to 0 (downstream expects 0-1 metric)
        best_correlation = max(best_correlation, 0.0)

        # Check if there's actually signal above threshold at the best position
        shifted_idx = center_idx + best_shift
        half_w = best_width // 2
        start = max(0, shifted_idx - half_w)
        end = min(len(intensity), shifted_idx + half_w + 1)
        segment = intensity[start:end] - baseline[start:end]
        peak_amplitude = np.max(segment) if len(segment) > 0 else 0.0

        # Tooth is active only if BOTH correlation meets per-tooth threshold AND signal is present
        # Paper (Gajarska et al. 2024): per-tooth activation uses separate threshold (0.5)
        active = best_correlation >= self.tooth_activation_threshold and peak_amplitude > threshold

        return {
            "center_nm": center_nm,
            "best_correlation": best_correlation,
            "best_shift": best_shift,
            "best_width": best_width,
            "active": active,
        }

    def _analyze_interferences(
        self, element_teeth: Dict[str, List[dict]], wl_tolerance_nm: float = 0.1
    ) -> Dict[str, List[dict]]:
        """
        Analyze interferences between elements based on overlapping teeth.

        Parameters
        ----------
        element_teeth : Dict[str, List[dict]]
            Dictionary mapping element symbols to lists of tooth results
        wl_tolerance_nm : float, optional
            Wavelength tolerance for interference detection (default: 0.1 nm)

        Returns
        -------
        Dict[str, List[dict]]
            Updated element_teeth with interference information added
        """
        # Build list of all active teeth with element labels
        all_active_teeth = []
        for element, teeth in element_teeth.items():
            for tooth in teeth:
                if tooth["active"]:
                    all_active_teeth.append((element, tooth))

        # Sort by wavelength for efficient sliding-window comparison
        all_active_teeth.sort(key=lambda x: x[1]["center_nm"])

        # Two-pointer interference detection
        n = len(all_active_teeth)
        for i in range(n):
            element_i, tooth_i = all_active_teeth[i]
            interfering = []
            # Look forward while within tolerance
            j = i + 1
            while (
                j < n
                and (all_active_teeth[j][1]["center_nm"] - tooth_i["center_nm"]) < wl_tolerance_nm
            ):
                element_j, tooth_j = all_active_teeth[j]
                if element_i != element_j:
                    interfering.append(element_j)
                    # Also mark the other tooth (reciprocal marking)
                    if "interfering_elements" not in tooth_j:
                        tooth_j["interfering_elements"] = []
                    if element_i not in tooth_j["interfering_elements"]:
                        tooth_j["interfering_elements"].append(element_i)
                        tooth_j["is_interfered"] = True
                j += 1

            # Only update if not already set by reciprocal marking
            if interfering:
                tooth_i["is_interfered"] = True
                existing = tooth_i.get("interfering_elements", [])
                tooth_i["interfering_elements"] = sorted(set(existing + interfering))
            elif "is_interfered" not in tooth_i:
                # Only set to False if not already marked from a previous iteration
                tooth_i["is_interfered"] = False
                tooth_i["interfering_elements"] = []

        return element_teeth

    def _compute_fingerprint(self, teeth: List[dict]) -> float:
        """
        Compute fingerprint as coverage-penalized mean correlation.

        Score = sum(active correlations) / denominator.
        The denominator is capped to avoid over-penalizing elements with many
        potential lines (like trace elements in a dense database).

        Parameters
        ----------
        teeth : List[dict]
            List of tooth results from _correlate_tooth

        Returns
        -------
        float
            Fingerprint score (0-1)
        """
        if not teeth:
            return 0.0
        active_teeth = [t for t in teeth if t["active"]]
        if not active_teeth:
            return 0.0
        # Sum the best active correlations only, and divide by the same capped
        # count so dense line lists improve recall without allowing scores > 1.
        active_correlations = sorted(
            (float(t["best_correlation"]) for t in active_teeth),
            reverse=True,
        )
        denominator = min(len(teeth), self.fingerprint_top_k)
        top_correlations = active_correlations[:denominator]
        fingerprint = sum(top_correlations) / denominator
        return float(np.clip(fingerprint, 0.0, 1.0))
