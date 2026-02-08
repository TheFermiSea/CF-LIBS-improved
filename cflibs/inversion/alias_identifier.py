"""
ALIAS (Automated Line Identification Algorithm for Spectroscopy) implementation.

Based on Noel et al. (2025) arXiv:2501.01057. The ALIAS algorithm identifies elements
in LIBS spectra through a 7-step process: peak detection, theoretical emissivity
calculation, line fusion, matching, threshold determination, scoring, and decision.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.signal import find_peaks

from cflibs.atomic.database import AtomicDatabase
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.core.constants import KB_EV
from cflibs.inversion.element_id import (
    IdentifiedLine,
    ElementIdentification,
    ElementIdentificationResult,
)


class ALIASIdentifier:
    """
    ALIAS algorithm for automated element identification in LIBS spectra.

    The algorithm operates in 7 steps:
    1. Peak detection via 2nd derivative enhancement
    2. Theoretical emissivity calculation over (T, n_e) grid
    3. Line fusion within resolution element
    4. Matching theoretical lines to experimental peaks
    5. Emissivity threshold determination via detection rate
    6. Score computation (k_sim, k_rate, k_shift)
    7. Decision and confidence level calculation

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database for transitions and partition functions
    resolving_power : float, optional
        Instrument resolving power R = λ/Δλ (default: 5000.0)
    T_range_K : Tuple[float, float], optional
        Temperature grid range in K (default: (8000.0, 12000.0))
    n_e_range_cm3 : Tuple[float, float], optional
        Electron density grid range in cm^-3 (default: (3e16, 3e17))
    T_steps : int, optional
        Number of temperature grid points (default: 5)
    n_e_steps : int, optional
        Number of electron density grid points (default: 3)
    intensity_threshold_factor : float, optional
        Peak detection threshold = factor × noise_estimate (default: 10.0)
    detection_threshold : float, optional
        Minimum confidence level for element detection (default: 0.5)
    elements : Optional[List[str]], optional
        List of elements to search for. If None, searches all available (default: None)
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        resolving_power: float = 5000.0,
        T_range_K: Tuple[float, float] = (8000.0, 12000.0),
        n_e_range_cm3: Tuple[float, float] = (3e16, 3e17),
        T_steps: int = 5,
        n_e_steps: int = 3,
        intensity_threshold_factor: float = 10.0,
        detection_threshold: float = 0.5,
        elements: Optional[List[str]] = None,
    ):
        self.atomic_db = atomic_db
        self.resolving_power = resolving_power
        self.T_range_K = T_range_K
        self.n_e_range_cm3 = n_e_range_cm3
        self.T_steps = T_steps
        self.n_e_steps = n_e_steps
        self.intensity_threshold_factor = intensity_threshold_factor
        self.detection_threshold = detection_threshold
        self.elements = elements

        # Create Saha-Boltzmann solver
        self.solver = SahaBoltzmannSolver(atomic_db)

        # Create (T, n_e) grid
        self.T_grid_K = np.linspace(T_range_K[0], T_range_K[1], T_steps)
        self.n_e_grid_cm3 = np.linspace(n_e_range_cm3[0], n_e_range_cm3[1], n_e_steps)

    def identify(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> ElementIdentificationResult:
        """
        Identify elements in experimental spectrum.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array (arbitrary units)

        Returns
        -------
        ElementIdentificationResult
            Complete identification result with detected/rejected elements
        """
        # Step 1: Detect peaks
        peaks = self._detect_peaks(wavelength, intensity)

        wl_min = np.min(wavelength)
        wl_max = np.max(wavelength)

        # Get elements to search
        if self.elements is None:
            # Get all available elements from database
            # For now, use common LIBS elements as default
            search_elements = ["Fe", "H", "Cu", "Al", "Ti", "Ca", "Mg", "Si"]
        else:
            search_elements = self.elements

        all_element_ids = []

        for element in search_elements:
            # Step 2: Compute theoretical emissivities
            element_lines = self._compute_element_emissivities(element, wl_min, wl_max)

            if not element_lines:
                continue

            # Step 3: Fuse nearby lines
            fused_lines = self._fuse_lines(element_lines, wavelength)

            if not fused_lines:
                continue

            # Step 4: Match lines to experimental peaks
            matched_mask, wavelength_shifts = self._match_lines(fused_lines, peaks)

            # Step 5: Determine emissivity threshold
            if np.any(matched_mask):
                emissivity_threshold = self._determine_emissivity_threshold(
                    fused_lines, matched_mask
                )
            else:
                emissivity_threshold = -np.inf  # No matches, keep all for scoring

            # Step 6: Compute scores
            k_sim, k_rate, k_shift = self._compute_scores(
                fused_lines, matched_mask, wavelength_shifts, intensity, peaks, emissivity_threshold
            )

            # Step 7: Decision
            N_X = np.sum(
                matched_mask
                & (
                    np.array([line["avg_emissivity"] for line in fused_lines])
                    >= 10**emissivity_threshold
                )
            )
            k_det, CL = self._decide(k_sim, k_rate, k_shift, N_X, intensity, peaks)

            # Build ElementIdentification
            detected = CL >= self.detection_threshold

            # Create IdentifiedLine objects for matched lines
            matched_lines = []
            unmatched_lines = []
            for i, line_data in enumerate(fused_lines):
                trans = line_data["transition"]
                if matched_mask[i]:
                    # Find closest peak
                    peak_idx = np.argmin(
                        np.abs(np.array([p[1] for p in peaks]) - line_data["wavelength_nm"])
                    )
                    peak_wl = peaks[peak_idx][1]
                    peak_int = intensity[peaks[peak_idx][0]]

                    matched_lines.append(
                        IdentifiedLine(
                            wavelength_exp_nm=peak_wl,
                            wavelength_th_nm=line_data["wavelength_nm"],
                            element=element,
                            ionization_stage=trans.ionization_stage,
                            intensity_exp=peak_int,
                            emissivity_th=line_data["avg_emissivity"],
                            transition=trans,
                            correlation=k_sim,
                        )
                    )
                else:
                    unmatched_lines.append(trans)

            element_id = ElementIdentification(
                element=element,
                detected=detected,
                score=k_det,
                confidence=CL,
                n_matched_lines=np.sum(matched_mask),
                n_total_lines=len(fused_lines),
                matched_lines=matched_lines,
                unmatched_lines=unmatched_lines,
                metadata={
                    "k_sim": k_sim,
                    "k_rate": k_rate,
                    "k_shift": k_shift,
                    "k_det": k_det,
                    "emissivity_threshold": emissivity_threshold,
                    "N_X": int(N_X),
                },
            )

            all_element_ids.append(element_id)

        # Split into detected/rejected
        detected_elements = [e for e in all_element_ids if e.detected]
        rejected_elements = [e for e in all_element_ids if not e.detected]

        # Count matched peaks (peak matched if any element matched it)
        matched_peak_indices = set()
        for element_id in detected_elements:
            for line in element_id.matched_lines:
                # Find peak index
                peak_idx = np.argmin(
                    np.abs(np.array([p[1] for p in peaks]) - line.wavelength_exp_nm)
                )
                matched_peak_indices.add(peak_idx)

        n_matched_peaks = len(matched_peak_indices)
        n_unmatched_peaks = len(peaks) - n_matched_peaks

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=all_element_ids,
            experimental_peaks=peaks,
            n_peaks=len(peaks),
            n_matched_peaks=n_matched_peaks,
            n_unmatched_peaks=n_unmatched_peaks,
            algorithm="alias",
            parameters={
                "resolving_power": self.resolving_power,
                "T_min_K": self.T_range_K[0],
                "T_max_K": self.T_range_K[1],
                "n_e_min_cm3": self.n_e_range_cm3[0],
                "n_e_max_cm3": self.n_e_range_cm3[1],
                "intensity_threshold_factor": self.intensity_threshold_factor,
                "detection_threshold": self.detection_threshold,
            },
        )

    def _detect_peaks(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Detect peaks using 2nd derivative enhancement.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array
        intensity : np.ndarray
            Intensity array

        Returns
        -------
        List[Tuple[int, float]]
            List of (peak_index, peak_wavelength) tuples
        """
        # Estimate noise using MAD (Median Absolute Deviation)
        median_intensity = np.median(intensity)
        mad = np.median(np.abs(intensity - median_intensity))
        noise_estimate = mad * 1.4826  # Scale factor for normal distribution

        # Threshold
        threshold = noise_estimate * self.intensity_threshold_factor

        # Find peaks
        peak_indices, properties = find_peaks(intensity, height=threshold, prominence=threshold / 2)

        # Return as list of (index, wavelength) tuples
        peaks = [(int(idx), float(wavelength[idx])) for idx in peak_indices]

        return peaks

    def _compute_element_emissivities(
        self, element: str, wl_min: float, wl_max: float
    ) -> List[dict]:
        """
        Compute theoretical emissivities for element over (T, n_e) grid.

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
        List[dict]
            List of dicts with keys: transition, avg_emissivity, wavelength_nm
        """
        # Get transitions for element (try both neutral and ionized)
        transitions = []
        for ion_stage in [1, 2]:
            try:
                trans_list = self.atomic_db.get_transitions(
                    element, ion_stage, wavelength_min=wl_min, wavelength_max=wl_max
                )
                if trans_list:
                    transitions.extend(trans_list)
            except Exception:
                # No data for this ionization stage
                continue

        if not transitions:
            return []

        # Compute emissivities
        line_data = []
        total_density = 1e15  # Arbitrary reference density

        for transition in transitions:
            emissivities = []

            for T_K in self.T_grid_K:
                for n_e in self.n_e_grid_cm3:
                    T_eV = T_K * KB_EV

                    # Get ionization balance
                    try:
                        stage_densities = self.solver.solve_ionization_balance(
                            element, T_eV, n_e, total_density
                        )
                        stage_density = stage_densities.get(transition.ionization_stage, 0.0)
                        W_q = stage_density / total_density

                        # Get partition function
                        U_T = self.solver.calculate_partition_function(
                            element, transition.ionization_stage, T_eV
                        )

                        # Emissivity: eps = W^q * A_ki * g_k * exp(-E_k/kT) / U(T)
                        boltzmann_factor = np.exp(-transition.E_k_ev / T_eV)
                        eps = W_q * transition.A_ki * transition.g_k * boltzmann_factor / U_T

                        emissivities.append(eps)
                    except Exception:
                        # Failed for this grid point, skip
                        continue

            if emissivities:
                avg_emissivity = np.mean(emissivities)
                line_data.append(
                    {
                        "transition": transition,
                        "avg_emissivity": avg_emissivity,
                        "wavelength_nm": transition.wavelength_nm,
                    }
                )

        return line_data

    def _fuse_lines(self, line_data: List[dict], wavelength_nm: np.ndarray) -> List[dict]:
        """
        Fuse lines within resolution element.

        Parameters
        ----------
        line_data : List[dict]
            List of line dicts from _compute_element_emissivities
        wavelength_nm : np.ndarray
            Experimental wavelength array (for reference wavelength)

        Returns
        -------
        List[dict]
            Fused line list with combined emissivities
        """
        if not line_data:
            return []

        # Sort by wavelength
        sorted_lines = sorted(line_data, key=lambda x: x["wavelength_nm"])

        # Resolution element at mean wavelength
        mean_wl = np.mean(wavelength_nm)
        delta_lambda = mean_wl / self.resolving_power

        # Group lines within delta_lambda
        fused = []
        current_group = [sorted_lines[0]]

        for i in range(1, len(sorted_lines)):
            line = sorted_lines[i]
            prev_line = current_group[-1]

            if abs(line["wavelength_nm"] - prev_line["wavelength_nm"]) <= delta_lambda:
                # Add to current group
                current_group.append(line)
            else:
                # Finalize current group
                fused.append(self._finalize_group(current_group))
                current_group = [line]

        # Finalize last group
        if current_group:
            fused.append(self._finalize_group(current_group))

        return fused

    def _finalize_group(self, group: List[dict]) -> dict:
        """
        Finalize a group of lines by summing emissivities.

        Parameters
        ----------
        group : List[dict]
            Group of line dicts

        Returns
        -------
        dict
            Fused line dict
        """
        # Sum emissivities
        total_emissivity = sum(line["avg_emissivity"] for line in group)

        # Position = wavelength of strongest line
        strongest = max(group, key=lambda x: x["avg_emissivity"])

        return {
            "transition": strongest["transition"],
            "avg_emissivity": total_emissivity,
            "wavelength_nm": strongest["wavelength_nm"],
            "n_fused": len(group),
        }

    def _match_lines(
        self, fused_lines: List[dict], peaks: List[Tuple[int, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match theoretical lines to experimental peaks.

        Parameters
        ----------
        fused_lines : List[dict]
            Fused theoretical lines
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (matched_mask, wavelength_shifts) where matched_mask is bool array and
            wavelength_shifts is float array of shifts in nm
        """
        if not peaks or not fused_lines:
            return np.zeros(len(fused_lines), dtype=bool), np.zeros(len(fused_lines))

        peak_wavelengths = np.array([p[1] for p in peaks])

        # Resolution element for matching window
        mean_wl = np.mean([line["wavelength_nm"] for line in fused_lines])
        delta_lambda = mean_wl / self.resolving_power

        matched_mask = np.zeros(len(fused_lines), dtype=bool)
        wavelength_shifts = np.zeros(len(fused_lines))

        for i, line in enumerate(fused_lines):
            wl_th = line["wavelength_nm"]

            # Find peaks within +/- delta_lambda/2
            distances = np.abs(peak_wavelengths - wl_th)
            within_window = distances <= delta_lambda / 2

            if np.any(within_window):
                matched_mask[i] = True
                # Shift to closest peak
                closest_idx = np.argmin(distances)
                wavelength_shifts[i] = peak_wavelengths[closest_idx] - wl_th

        return matched_mask, wavelength_shifts

    def _determine_emissivity_threshold(
        self, fused_lines: List[dict], matched_mask: np.ndarray
    ) -> float:
        """
        Determine emissivity threshold where detection rate > 50%.

        Parameters
        ----------
        fused_lines : List[dict]
            Fused theoretical lines
        matched_mask : np.ndarray
            Boolean mask of matched lines

        Returns
        -------
        float
            Log10 emissivity threshold
        """
        emissivities = np.array([line["avg_emissivity"] for line in fused_lines])

        # Avoid log(0)
        emissivities = np.maximum(emissivities, 1e-100)

        log_emissivities = np.log10(emissivities)

        # Bin in log decades
        min_log = np.floor(np.min(log_emissivities))
        max_log = np.ceil(np.max(log_emissivities))
        n_bins = int(max_log - min_log) + 1

        if n_bins < 2:
            # Not enough dynamic range, return minimum
            return min_log

        bins = np.linspace(min_log, max_log, n_bins + 1)

        # Compute detection rate per bin
        bin_indices = np.digitize(log_emissivities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        detection_rates = []
        thresholds = []

        for bin_idx in range(n_bins):
            in_bin = bin_indices == bin_idx
            if np.sum(in_bin) > 0:
                detection_rate = np.sum(matched_mask & in_bin) / np.sum(in_bin)
                detection_rates.append(detection_rate)
                thresholds.append(bins[bin_idx])

        # Find threshold where detection_rate > 0.5
        detection_rates = np.array(detection_rates)
        thresholds = np.array(thresholds)

        above_50 = detection_rates > 0.5
        if np.any(above_50):
            # Return lowest threshold with >50% detection
            return thresholds[np.where(above_50)[0][0]]
        else:
            # No threshold meets criterion, return minimum
            return min_log

    def _compute_scores(
        self,
        fused_lines: List[dict],
        matched_mask: np.ndarray,
        wavelength_shifts: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
        emissivity_threshold: float,
    ) -> Tuple[float, float, float]:
        """
        Compute k_sim, k_rate, k_shift scores.

        Parameters
        ----------
        fused_lines : List[dict]
            Fused theoretical lines
        matched_mask : np.ndarray
            Boolean mask of matched lines
        wavelength_shifts : np.ndarray
            Wavelength shifts in nm
        intensity : np.ndarray
            Experimental intensity array
        peaks : List[Tuple[int, float]]
            Experimental peaks
        emissivity_threshold : float
            Log10 emissivity threshold

        Returns
        -------
        Tuple[float, float, float]
            (k_sim, k_rate, k_shift) scores in [0, 1]
        """
        if not np.any(matched_mask):
            return 0.0, 0.0, 0.0

        emissivities = np.array([line["avg_emissivity"] for line in fused_lines])
        above_threshold = emissivities >= 10**emissivity_threshold

        # Filter to lines above threshold
        matched_above = matched_mask & above_threshold
        n_matched_above = np.sum(matched_above)

        if n_matched_above == 0:
            return 0.0, 0.0, 0.0

        # k_rate: emissivity-weighted detection rate
        total_emissivity_above = np.sum(emissivities[above_threshold])
        matched_emissivity = np.sum(emissivities[matched_above])

        if total_emissivity_above > 0:
            k_rate = matched_emissivity / total_emissivity_above
        else:
            k_rate = 0.0

        # k_shift: wavelength match quality
        mean_wl = np.mean([line["wavelength_nm"] for line in fused_lines])
        delta_lambda = mean_wl / self.resolving_power

        shifts_matched = np.abs(wavelength_shifts[matched_above])
        if len(shifts_matched) > 0:
            mean_shift_frac = np.mean(shifts_matched) / delta_lambda
            k_shift = max(0.0, 1.0 - mean_shift_frac)
        else:
            k_shift = 0.0

        # k_sim: cosine similarity between theoretical and experimental intensities
        # For matched lines, compare emissivities to experimental peak heights
        theoretical_intensities = []
        experimental_intensities = []

        for i, line in enumerate(fused_lines):
            if matched_above[i]:
                theoretical_intensities.append(emissivities[i])

                # Find closest peak
                peak_idx = np.argmin(
                    np.abs(np.array([p[1] for p in peaks]) - line["wavelength_nm"])
                )
                experimental_intensities.append(intensity[peaks[peak_idx][0]])

        if len(theoretical_intensities) > 1:
            th_vec = np.array(theoretical_intensities)
            exp_vec = np.array(experimental_intensities)

            # Cosine similarity
            dot_product = np.dot(th_vec, exp_vec)
            norm_th = np.linalg.norm(th_vec)
            norm_exp = np.linalg.norm(exp_vec)

            if norm_th > 0 and norm_exp > 0:
                k_sim = dot_product / (norm_th * norm_exp)
                k_sim = max(0.0, min(1.0, k_sim))  # Clamp to [0, 1]
            else:
                k_sim = 0.0
        else:
            # Not enough points for correlation
            k_sim = 0.5  # Neutral value

        return k_sim, k_rate, k_shift

    def _decide(
        self,
        k_sim: float,
        k_rate: float,
        k_shift: float,
        N_X: int,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
    ) -> Tuple[float, float]:
        """
        Compute detection score k_det and confidence level CL.

        Parameters
        ----------
        k_sim : float
            Similarity score
        k_rate : float
            Detection rate score
        k_shift : float
            Wavelength shift score
        N_X : int
            Number of matched lines above threshold
        intensity : np.ndarray
            Experimental intensity array
        peaks : List[Tuple[int, float]]
            Experimental peaks

        Returns
        -------
        Tuple[float, float]
            (k_det, CL) detection score and confidence level
        """
        # k_det formula
        if N_X > 0:
            k_det = k_rate * ((1.0 / N_X) * k_shift + ((N_X - 1.0) / N_X) * k_sim)
        else:
            k_det = 0.0

        # P_maj: majority of strong lines matched
        if k_rate > 0.5:
            P_maj = 1.0
        else:
            P_maj = k_rate

        # P_SNR: approximate SNR quality
        if len(peaks) > 0:
            peak_intensities = [intensity[p[0]] for p in peaks]
            median_peak = np.median(peak_intensities)
            noise_estimate = np.median(np.abs(intensity - np.median(intensity))) * 1.4826
            snr_estimate = median_peak / max(noise_estimate, 1e-10)
            P_SNR = min(1.0, snr_estimate / 10.0)
        else:
            P_SNR = 0.5

        # P_ab: abundance prior (placeholder)
        P_ab = 1.0

        # Confidence level
        CL = k_det * P_maj * P_SNR * P_ab

        return k_det, CL
