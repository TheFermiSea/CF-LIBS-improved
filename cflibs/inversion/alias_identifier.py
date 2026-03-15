"""
ALIAS (Automated Line Identification Algorithm for Spectroscopy) implementation.

Based on Noel et al. (2025) arXiv:2501.01057. The ALIAS algorithm identifies elements
in LIBS spectra through a 7-step process: peak detection, theoretical emissivity
calculation, line fusion, matching, threshold determination, scoring, and decision.
"""

from typing import List, Tuple, Optional
from collections import defaultdict
import math
import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks
from scipy.special import erf
from scipy.stats import binom

from cflibs.atomic.database import AtomicDatabase
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.core.constants import KB_EV
from cflibs.inversion.element_id import (
    IdentifiedLine,
    ElementIdentification,
    ElementIdentificationResult,
)
from cflibs.inversion.preprocessing import estimate_baseline, estimate_noise


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
        Peak detection threshold = factor × noise_estimate (default: 4.0)
    detection_threshold : float, optional
        Minimum confidence level for element detection (default: 0.02)
    chance_window_scale : float, optional
        Scale factor for chance-coincidence windows used in fill-factor estimation.
        The chance half-window is `chance_window_scale * (lambda / R)`.
    elements : Optional[List[str]], optional
        List of elements to search for. If None, uses default common LIBS elements:
        ["Fe", "H", "Cu", "Al", "Ti", "Ca", "Mg", "Si"] (default: None)
    """

    # Crustal abundance in log10(ppm) — from CRC Handbook / USGS
    CRUSTAL_ABUNDANCE_LOG_PPM = {
        "O": 5.67,
        "Si": 5.44,
        "Al": 4.91,
        "Fe": 4.70,
        "Ca": 4.57,
        "Na": 4.36,
        "Mg": 4.33,
        "K": 4.32,
        "Ti": 3.75,
        "H": 3.15,
        "Mn": 2.98,
        "P": 2.97,
        "F": 2.80,
        "Ba": 2.70,
        "C": 2.30,
        "Sr": 2.57,
        "S": 2.56,
        "Zr": 2.23,
        "V": 2.10,
        "Cl": 2.20,
        "Cr": 2.00,
        "Ni": 1.88,
        "Zn": 1.88,
        "Cu": 1.78,
        "Co": 1.40,
        "Li": 1.30,
        "N": 1.30,
        "Ga": 1.28,
        "Pb": 1.15,
        "Rb": 1.95,
        "B": 1.00,
        "Sn": 0.35,
        "W": 0.18,
        "Mo": 0.18,
        "Ag": -0.62,
        "Cd": -0.82,
        "Au": -2.40,
    }

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        resolving_power: float = 5000.0,
        T_range_K: Tuple[float, float] = (5000.0, 15000.0),
        n_e_range_cm3: Tuple[float, float] = (1e16, 5e17),
        T_steps: int = 7,
        n_e_steps: int = 3,
        intensity_threshold_factor: float = 3.0,
        detection_threshold: float = 0.03,
        chance_window_scale: float = 0.4,
        elements: Optional[List[str]] = None,
        max_lines_per_element: int = 50,
        reference_temperature: float = 10000.0,
    ):
        self.atomic_db = atomic_db
        if not (np.isfinite(resolving_power) and resolving_power > 0):
            raise ValueError(f"resolving_power must be finite and > 0, got {resolving_power!r}")
        self.resolving_power = float(resolving_power)
        self.T_range_K = T_range_K
        self.n_e_range_cm3 = n_e_range_cm3
        self.T_steps = T_steps
        self.n_e_steps = n_e_steps
        self.intensity_threshold_factor = intensity_threshold_factor
        self.detection_threshold = detection_threshold
        self.chance_window_scale = chance_window_scale
        self.elements = elements
        self.max_lines_per_element = max_lines_per_element
        self.reference_temperature = reference_temperature

        # Create Saha-Boltzmann solver
        self.solver = SahaBoltzmannSolver(atomic_db)

        # Create (T, n_e) grid
        self.T_grid_K = np.linspace(T_range_K[0], T_range_K[1], T_steps)
        self.n_e_grid_cm3 = np.linspace(n_e_range_cm3[0], n_e_range_cm3[1], n_e_steps)

    def identify(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> ElementIdentificationResult:
        """
        Identify elements in experimental spectrum with cross-element peak
        competition.

        Three-phase algorithm:
        1. Score all elements independently (same as before).
        2. Global peak competition: each disputed experimental peak is
           assigned to the element with the highest initial confidence
           level (CL).  Losers have their match revoked.
        3. Rescore elements that lost peaks and build final results.

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
            # Prefer database-provided availability when possible.
            get_available = getattr(self.atomic_db, "get_available_elements", None)
            if callable(get_available):
                try:
                    available = list(get_available())
                except Exception:
                    available = []
                search_elements = available or ["Fe", "H", "Cu", "Al", "Ti", "Ca", "Mg", "Si"]
            else:
                search_elements = ["Fe", "H", "Cu", "Al", "Ti", "Ca", "Mg", "Si"]
        else:
            search_elements = self.elements

        # ── Phase 1: Independent scoring ──────────────────────────────
        global_p_snr = self._compute_p_snr(intensity, peaks)
        candidates: List[dict] = []

        for element in search_elements:
            element_lines = self._compute_element_emissivities(element, wl_min, wl_max)
            if not element_lines:
                continue

            fused_lines = self._fuse_lines(element_lines, wavelength)
            if not fused_lines:
                continue

            matched_mask, wavelength_shifts, matched_peak_idx = self._match_lines(
                fused_lines, peaks
            )

            if np.any(matched_mask):
                emissivity_threshold = self._determine_emissivity_threshold(
                    fused_lines, matched_mask
                )
            else:
                emissivity_threshold = -np.inf

            k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov = self._compute_scores(
                fused_lines,
                matched_mask,
                matched_peak_idx,
                wavelength_shifts,
                intensity,
                peaks,
                emissivity_threshold,
            )

            P_sig, fill_factor, p_chance, p_tail = self._compute_random_match_significance(
                peaks=peaks,
                wavelength=wavelength,
                N_expected=N_expected,
                N_matched=N_matched,
            )

            k_det, CL = self._decide(
                k_sim,
                k_rate,
                k_shift,
                N_expected,
                intensity,
                peaks,
                element=element,
                P_maj=P_maj,
                P_sig=P_sig,
                N_matched=N_matched,
                P_cov=P_cov,
            )

            candidates.append(
                {
                    "element": element,
                    "fused_lines": fused_lines,
                    "matched_mask": matched_mask,
                    "matched_peak_idx": matched_peak_idx,
                    "wavelength_shifts": wavelength_shifts,
                    "emissivity_threshold": emissivity_threshold,
                    "initial_CL": CL,
                    # Cache phase 1 scores for reuse when competition is skipped
                    "scores": (k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov),
                    "N_matched": N_matched,
                    "P_sig_data": (P_sig, fill_factor, p_chance, p_tail),
                    "k_det": k_det,
                    "P_SNR": global_p_snr,
                }
            )

        # ── Phase 1.5: NNLS peak-space mixture attribution ────────────
        # Non-negative least squares fit of all candidate templates against
        # observed peak intensities.  Returns three metrics:
        #   P_mix  — leave-one-out partial R^2 (global)
        #   P_local — local explanation score (what fraction of claimed
        #             peaks' intensity does this element actually explain?)
        if candidates and peaks:
            peak_intensities_arr = np.array([intensity[p[0]] for p in peaks])
            A = self._build_nnls_templates(candidates, peaks)
            P_mix_arr, P_local_arr, _ = self._compute_nnls_attribution(A, peak_intensities_arr)
            for i, cand in enumerate(candidates):
                cand["P_mix"] = float(P_mix_arr[i])
                cand["P_local"] = float(P_local_arr[i])
        else:
            for cand in candidates:
                cand["P_mix"] = 1.0
                cand["P_local"] = 1.0

        # ── Phase 2: Global peak competition ──────────────────────────
        # Only active at RP >= 2000 where peaks are narrow enough for
        # meaningful exclusivity.  At low RP (broadband spectrometers),
        # shared peaks are the norm and winner-take-all competition
        # causes false negatives for real minor elements.
        if self.resolving_power >= 2000:
            peak_claims: dict = defaultdict(list)

            for c_idx, cand in enumerate(candidates):
                mask = cand["matched_mask"]
                pidx_arr = cand["matched_peak_idx"]
                for l_idx in range(len(mask)):
                    if mask[l_idx]:
                        pidx = int(pidx_arr[l_idx])
                        peak_claims[pidx].append((cand["initial_CL"], c_idx, l_idx))

            # Resolve: highest initial CL wins; losers get unmatched
            for pidx, claims in peak_claims.items():
                if len(claims) <= 1:
                    continue
                claims.sort(key=lambda x: x[0], reverse=True)
                for i in range(1, len(claims)):
                    _, loser_c, loser_l = claims[i]
                    candidates[loser_c]["matched_mask"][loser_l] = False
                    candidates[loser_c]["matched_peak_idx"][loser_l] = -1
                    candidates[loser_c]["wavelength_shifts"][loser_l] = 0.0

        # ── Phase 3: Rescore & build results ──────────────────────────
        competition_ran = self.resolving_power >= 2000
        all_element_ids = []

        for cand in candidates:
            element = cand["element"]
            fused_lines = cand["fused_lines"]
            matched_mask = cand["matched_mask"]
            matched_peak_idx = cand["matched_peak_idx"]
            wavelength_shifts = cand["wavelength_shifts"]
            emissivity_threshold = cand["emissivity_threshold"]

            if competition_ran:
                # Rescore with post-competition matches
                k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov = self._compute_scores(
                    fused_lines,
                    matched_mask,
                    matched_peak_idx,
                    wavelength_shifts,
                    intensity,
                    peaks,
                    emissivity_threshold,
                )

                P_sig, fill_factor, p_chance, p_tail = self._compute_random_match_significance(
                    peaks=peaks,
                    wavelength=wavelength,
                    N_expected=N_expected,
                    N_matched=N_matched,
                )

                k_det, CL = self._decide(
                    k_sim,
                    k_rate,
                    k_shift,
                    N_expected,
                    intensity,
                    peaks,
                    element=element,
                    P_maj=P_maj,
                    P_sig=P_sig,
                    N_matched=N_matched,
                    P_cov=P_cov,
                )
            else:
                # No competition — reuse phase 1 scores
                k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov = cand["scores"]
                P_sig, fill_factor, p_chance, p_tail = cand["P_sig_data"]
                k_det = cand["k_det"]
                CL = cand["initial_CL"]

            P_SNR = cand["P_SNR"] if not competition_ran else global_p_snr

            # ── Post-CL discriminators ──────────────────────────────────
            # Two NNLS-derived gates suppress false positives whose peaks
            # ride on a dominant element's lines:
            #
            # 1. P_local (NNLS peak ownership): fraction of claimed peaks'
            #    intensity that this element's NNLS coefficient explains.
            #    FP elements ride on dominant-element peaks → P_local ~ 0.
            # 2. P_mix (leave-one-out partial R²): how much total spectrum
            #    energy is uniquely attributable to this element.
            #    FP elements add nothing → P_mix ~ 0.
            #
            # R_rat (intensity-ratio consistency) provides a soft additional
            # check: do observed ratios match predicted emissivity ratios?
            #
            # NOTE: P_sig (binomial significance) is deliberately excluded.
            # At low RP (high fill factor), line-rich elements like Fe have
            # match counts BELOW random expectation, giving P_sig → 0 for
            # true positives. P_sig only works at high RP / low fill factor.

            P_mix = cand.get("P_mix", 1.0)
            P_local = cand.get("P_local", 1.0)

            R_rat = self._compute_ratio_consistency(
                fused_lines,
                matched_mask,
                matched_peak_idx,
                intensity,
                peaks,
            )

            # Post-CL discriminators with raised floors to prevent
            # minor elements from being crushed when their peaks overlap
            # with dominant-element emission.

            # Gate 1: P_local — ramp from 0.25 (floor) to 1.0
            CL *= float(np.clip(2.0 * P_local, 0.25, 1.0))

            # Gate 2: P_mix — linear ramp (0.25 at P_mix=0, 1.0 at P_mix=1)
            CL *= 0.25 + 0.75 * min(P_mix, 1.0)

            # Gate 3: R_rat — soft consistency check (0.5 min, 1.0 max)
            CL *= 0.5 + 0.5 * R_rat

            detected = CL >= self.detection_threshold

            # Create IdentifiedLine objects for matched lines
            # Reuse peak indices from matching to avoid re-selection outside window
            matched_lines = []
            unmatched_lines = []
            for i, line_data in enumerate(fused_lines):
                trans = line_data["transition"]
                if matched_mask[i]:
                    pidx = matched_peak_idx[i]
                    matched_lines.append(
                        IdentifiedLine(
                            wavelength_exp_nm=peaks[pidx][1],
                            wavelength_th_nm=line_data["wavelength_nm"],
                            element=element,
                            ionization_stage=trans.ionization_stage,
                            intensity_exp=intensity[peaks[pidx][0]],
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
                n_matched_lines=int(np.sum(matched_mask)),
                n_total_lines=len(fused_lines),
                matched_lines=matched_lines,
                unmatched_lines=unmatched_lines,
                metadata={
                    "k_sim": k_sim,
                    "k_rate": k_rate,
                    "k_shift": k_shift,
                    "k_det": k_det,
                    "emissivity_threshold": emissivity_threshold,
                    "N_expected": N_expected,
                    "N_matched": N_matched,
                    "P_maj": P_maj,
                    "P_ab": self._compute_P_ab(element),
                    "P_cov": P_cov,
                    "P_mix": P_mix,
                    "P_local": P_local,
                    "R_rat": R_rat,
                    "P_SNR": P_SNR,
                    "P_sig": P_sig,
                    "p_tail": p_tail,
                    "p_chance": p_chance,
                    "fill_factor": fill_factor,
                    "N_penalty": 0.5 if N_expected == 2 else 1.0,
                },
            )

            all_element_ids.append(element_id)

        # Split into detected/rejected
        detected_elements = [e for e in all_element_ids if e.detected]
        rejected_elements = [e for e in all_element_ids if not e.detected]

        # Count matched peaks (peak matched if any element matched it, detected or rejected)
        matched_peak_indices = set()
        for element_id in all_element_ids:  # Use all_element_ids, not just detected
            for line in element_id.matched_lines:
                peak_idx = np.argmin(
                    np.abs(np.array([p[1] for p in peaks]) - line.wavelength_exp_nm)
                )
                matched_peak_indices.add(int(peak_idx))

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=all_element_ids,
            experimental_peaks=peaks,
            n_peaks=len(peaks),
            n_matched_peaks=len(matched_peak_indices),
            n_unmatched_peaks=len(peaks) - len(matched_peak_indices),
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
        Detect peaks using MAD-based noise estimation and scipy.signal.find_peaks.

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
        # Estimate baseline and noise using sigma-clipped MAD
        baseline = estimate_baseline(wavelength, intensity)
        noise_estimate = estimate_noise(intensity, baseline)

        # Threshold in intensity domain (with floor for flat spectra / zero MAD)
        threshold = max(noise_estimate * self.intensity_threshold_factor, np.finfo(float).eps)
        prominence = max(threshold / 3, np.finfo(float).eps)

        # Find peaks in baseline-corrected intensity
        corrected = intensity - baseline
        peak_indices, _ = find_peaks(corrected, height=threshold, prominence=prominence)

        # Paper (Noël et al. 2025): enhance peak detection using negative 2nd derivative
        # Compute -d²I/dλ², zero negatives — true peaks have positive curvature here
        d2 = -np.gradient(np.gradient(corrected, wavelength), wavelength)
        d2[d2 < 0] = 0.0

        # Filter: keep peaks where d2 > 0 in a ±2-point neighborhood around peak center
        # This handles discretization effects where d2 peak may be slightly offset
        confirmed = []
        for idx in peak_indices:
            lo = max(0, idx - 2)
            hi = min(len(d2), idx + 3)
            if np.max(d2[lo:hi]) > 0:
                confirmed.append(idx)
        peak_indices = np.array(confirmed, dtype=int) if confirmed else np.array([], dtype=int)

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
            except (KeyError, ValueError, AttributeError):
                # No data for this ionization stage
                continue

        # Remove unobservable weak lines before emissivity calculation
        transitions = [t for t in transitions if t.A_ki * t.g_k >= 1e4]

        if not transitions:
            return []

        # Cap to strongest lines by estimated emissivity to avoid line-count disparity
        if len(transitions) > self.max_lines_per_element:
            kT = KB_EV * self.reference_temperature
            transitions = sorted(
                transitions,
                key=lambda t: t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT),
                reverse=True,
            )
            transitions = transitions[: self.max_lines_per_element]

        # Compute emissivities
        line_data = []
        total_density = 1e15  # Arbitrary reference density

        # Precompute stage densities for all (T, n_e) grid points
        grid_stage_densities = {}
        for T_K in self.T_grid_K:
            for n_e in self.n_e_grid_cm3:
                T_eV = T_K * KB_EV
                try:
                    stage_densities = self.solver.solve_ionization_balance(
                        element, T_eV, n_e, total_density
                    )
                    grid_stage_densities[(T_K, n_e)] = stage_densities
                except (KeyError, ValueError, ZeroDivisionError):
                    # Failed for this grid point, skip
                    continue

        for transition in transitions:
            emissivities = []

            for T_K in self.T_grid_K:
                for n_e in self.n_e_grid_cm3:
                    T_eV = T_K * KB_EV

                    # Get precomputed ionization balance
                    stage_densities = grid_stage_densities.get((T_K, n_e))
                    if stage_densities is None:
                        continue

                    stage_density = stage_densities.get(transition.ionization_stage, 0.0)
                    if stage_density == 0.0:
                        continue

                    W_q = stage_density / total_density

                    try:
                        # Get partition function
                        U_T = self.solver.calculate_partition_function(
                            element, transition.ionization_stage, T_eV
                        )

                        # Emissivity: eps = W^q * A_ki * g_k * exp(-E_k/kT) / U(T)
                        boltzmann_factor = np.exp(-transition.E_k_ev / T_eV)
                        eps = W_q * transition.A_ki * transition.g_k * boltzmann_factor / U_T

                        emissivities.append(eps)
                    except (KeyError, ValueError, ZeroDivisionError):
                        # Failed to compute partition function or emissivity, skip
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (matched_mask, wavelength_shifts, matched_peak_idx) where
            matched_mask is bool array, wavelength_shifts is float array of
            shifts in nm, and matched_peak_idx is int array (-1 if unmatched)
        """
        n = len(fused_lines)
        if not peaks or not fused_lines:
            return (
                np.zeros(n, dtype=bool),
                np.zeros(n),
                np.full(n, -1, dtype=int),
            )

        peak_wavelengths = np.array([p[1] for p in peaks])

        # Resolution element for matching window
        mean_wl = np.mean([line["wavelength_nm"] for line in fused_lines])
        delta_lambda = mean_wl / self.resolving_power

        # Estimate global wavelength offset from strongest peaks
        sorted_by_emissivity = sorted(fused_lines, key=lambda x: x["avg_emissivity"], reverse=True)
        top_lines = sorted_by_emissivity[: min(5, len(sorted_by_emissivity))]

        shifts = []
        for line in top_lines:
            wl_th = line["wavelength_nm"]
            distances = np.abs(peak_wavelengths - wl_th)
            if len(distances) > 0:
                min_dist = np.min(distances)
                if min_dist <= delta_lambda:  # within one resolution element
                    closest_idx = np.argmin(distances)
                    shifts.append(peak_wavelengths[closest_idx] - wl_th)

        global_shift = np.median(shifts) if shifts else 0.0

        matched_mask = np.zeros(n, dtype=bool)
        wavelength_shifts = np.zeros(n)
        matched_peak_idx = np.full(n, -1, dtype=int)

        for i, line in enumerate(fused_lines):
            wl_th = line["wavelength_nm"] + global_shift  # Apply correction

            # Find peaks within +/- delta_lambda
            distances = np.abs(peak_wavelengths - wl_th)
            within_window = distances <= delta_lambda

            if np.any(within_window):
                matched_mask[i] = True
                closest_idx = int(np.argmin(distances))
                matched_peak_idx[i] = closest_idx
                # Shift relative to uncorrected wavelength
                wavelength_shifts[i] = peak_wavelengths[closest_idx] - line["wavelength_nm"]

        # Enforce one-to-one: each experimental peak is assigned to at most
        # one theoretical line (highest emissivity wins).  This prevents a
        # single broad peak from "confirming" multiple theoretical lines,
        # which inflates k_rate at low resolving power.
        claimed_peaks: dict = {}  # peak_idx -> (line_idx, emissivity)
        for i in range(n):
            if not matched_mask[i]:
                continue
            pidx = int(matched_peak_idx[i])
            emiss = fused_lines[i]["avg_emissivity"]
            if pidx not in claimed_peaks or emiss > claimed_peaks[pidx][1]:
                if pidx in claimed_peaks:
                    old_i = claimed_peaks[pidx][0]
                    matched_mask[old_i] = False
                    wavelength_shifts[old_i] = 0.0
                    matched_peak_idx[old_i] = -1
                claimed_peaks[pidx] = (i, emiss)
            else:
                # Peak already claimed by a stronger line
                matched_mask[i] = False
                wavelength_shifts[i] = 0.0
                matched_peak_idx[i] = -1

        return matched_mask, wavelength_shifts, matched_peak_idx

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
        matched_peak_idx: np.ndarray,
        wavelength_shifts: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
        emissivity_threshold: float,
    ) -> Tuple[float, float, float, float, int, int, float]:
        """
        Compute k_sim, k_rate, k_shift scores, P_maj, N_expected, N_matched, P_cov.

        Parameters
        ----------
        fused_lines : List[dict]
            Fused theoretical lines
        matched_mask : np.ndarray
            Boolean mask of matched lines
        matched_peak_idx : np.ndarray
            Index of matched peak per line (-1 if unmatched)
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
        Tuple[float, float, float, float, int, int, float]
            (k_sim, k_rate, k_shift, P_maj, N_expected, N_matched, P_cov)
        """
        if not np.any(matched_mask):
            return 0.0, 0.0, 0.0, 0.5, 0, 0, 0.0

        emissivities = np.array([line["avg_emissivity"] for line in fused_lines])
        above_threshold = emissivities >= 10**emissivity_threshold

        # N_expected: ALL above-threshold theoretical lines (matched or not).
        N_expected = int(np.sum(above_threshold))

        # Filter to lines above threshold that are also matched
        matched_above = matched_mask & above_threshold
        n_matched_above = int(np.sum(matched_above))

        # P_cov: emissivity-weighted coverage penalty — single channel for
        # penalizing missing lines. Missing a weak line matters less than
        # missing the resonance line.
        total_emissivity_above = float(np.sum(emissivities[above_threshold]))
        matched_emissivity = float(np.sum(emissivities[matched_above]))
        P_cov = matched_emissivity / total_emissivity_above if total_emissivity_above > 0 else 0.0

        if n_matched_above == 0:
            return 0.0, 0.0, 0.0, 0.5, N_expected, 0, P_cov

        # Soft P_maj: weighted coverage of top-k strongest above-threshold
        # lines.  Binary P_maj (strongest matched → 1.0, else 0.5) causes
        # false negatives when the major line is obscured by matrix
        # emission (e.g. V in Ti6Al4V where Ti dominates).
        top_k = min(3, N_expected)
        if top_k > 0:
            above_emissivities = emissivities * above_threshold.astype(float)
            sorted_indices = np.argsort(above_emissivities)[::-1][:top_k]
            # sqrt: softer than linear, prevents single dominant line
            # from driving P_maj to 1.0 alone
            weights = np.sqrt(emissivities[sorted_indices])
            matched_weights = float(np.sum(weights * matched_above[sorted_indices]))
            total_weights = float(np.sum(weights))
            P_maj = 0.5 + 0.5 * (matched_weights / total_weights) if total_weights > 0 else 0.5
        else:
            P_maj = 0.5

        # k_rate: emissivity-weighted detection rate.
        if total_emissivity_above > 0:
            k_rate = matched_emissivity / total_emissivity_above
        else:
            k_rate = 0.0

        # k_shift: wavelength match quality
        mean_wl = np.mean([line["wavelength_nm"] for line in fused_lines])
        delta_lambda = mean_wl / self.resolving_power

        shifts_matched = np.abs(wavelength_shifts[matched_above])
        emiss_matched = emissivities[matched_above]
        if len(shifts_matched) > 0 and np.sum(emiss_matched) > 0:
            weighted_shift = np.average(shifts_matched, weights=emiss_matched)
            k_shift = max(0.0, 1.0 - weighted_shift / delta_lambda)
        else:
            k_shift = 0.0

        # k_sim: cosine similarity between theoretical and experimental
        # intensities over MATCHED lines only (paper-faithful).
        # Coverage is handled exclusively by k_rate.
        #
        # Self-absorption correction: resonance lines (E_i < 0.1 eV) are
        # systematically weaker than optically-thin predictions. Damping
        # the theoretical emissivity avoids penalizing the cosine angle.
        SA_DAMPING = 0.3  # resonance lines ~3× weaker than thin prediction
        theoretical_intensities = []
        experimental_intensities = []
        unique_peak_set: set = set()

        for i in range(len(fused_lines)):
            if matched_above[i]:
                eps_th = emissivities[i]
                trans = fused_lines[i]["transition"]
                if getattr(trans, "E_i_ev", 1.0) < 0.1:
                    eps_th *= SA_DAMPING
                theoretical_intensities.append(eps_th)
                pidx = matched_peak_idx[i]
                experimental_intensities.append(intensity[peaks[pidx][0]])
                unique_peak_set.add(pidx)

        if len(theoretical_intensities) > 1:
            th_vec = np.array(theoretical_intensities)
            exp_vec = np.array(experimental_intensities)

            dot_product = np.dot(th_vec, exp_vec)
            norm_th = np.linalg.norm(th_vec)
            norm_exp = np.linalg.norm(exp_vec)

            if norm_th > 0 and norm_exp > 0:
                k_sim = dot_product / (norm_th * norm_exp)
                k_sim = max(0.0, min(1.0, k_sim))
            else:
                k_sim = 0.0
        else:
            # Single matched line: cosine similarity undefined.
            # Use neutral value so penalty comes from k_rate and P_cov,
            # not an outright rejection via the k_sim >= 0.15 gate.
            k_sim = 0.5

        # Uniqueness penalty: many-to-one mapping lowers k_sim
        n_unique_peaks = len(unique_peak_set)
        if n_matched_above > 0:
            uniqueness_factor = n_unique_peaks / n_matched_above
            k_sim *= uniqueness_factor

        return k_sim, k_rate, k_shift, P_maj, N_expected, n_matched_above, P_cov

    def _compute_P_ab(self, element: str) -> float:
        """
        Compute crustal-abundance prior P_ab for an element.

        3-tier weighting (Noel et al. 2025):
        - ppm >= 100    → 1.0  (common, > 0.01%)
        - ppm >= 0.001  → 0.75 (intermediate)
        - ppm < 0.001   → 0.5  (rare)

        Parameters
        ----------
        element : str
            Element symbol

        Returns
        -------
        float
            P_ab weighting factor
        """
        log_ppm = self.CRUSTAL_ABUNDANCE_LOG_PPM.get(element, 0.0)
        ppm = 10**log_ppm
        if ppm >= 100:
            return 1.0
        elif ppm >= 1e-3:
            return 0.75
        else:
            return 0.5

    def _compute_fill_factor(
        self,
        peaks: List[Tuple[int, float]],
        wavelength: np.ndarray,
    ) -> float:
        """
        Compute spectral fill factor from merged peak-match windows.

        Each peak contributes an interval centered at its wavelength with half-width:
            chance_window_scale * (lambda / resolving_power)
        Overlapping intervals are merged before computing covered span fraction.

        Parameters
        ----------
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples.
        wavelength : np.ndarray
            Full spectral wavelength axis in nm.

        Returns
        -------
        float
            Fraction of spectral span covered by merged intervals in [0, 1].
        """
        if len(peaks) == 0 or len(wavelength) < 2:
            return 0.0

        wl_min = float(np.min(wavelength))
        wl_max = float(np.max(wavelength))
        span = wl_max - wl_min
        if span <= 0:
            return 0.0

        intervals: List[Tuple[float, float]] = []
        for _, peak_wl in peaks:
            half_window = self.chance_window_scale * (peak_wl / self.resolving_power)
            if half_window <= 0:
                continue
            start = max(wl_min, peak_wl - half_window)
            end = min(wl_max, peak_wl + half_window)
            if end > start:
                intervals.append((start, end))

        if not intervals:
            return 0.0

        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        covered = sum(end - start for start, end in merged)
        return float(np.clip(covered / span, 0.0, 1.0))

    def _build_nnls_templates(
        self,
        candidates: List[dict],
        peaks: List[Tuple[int, float]],
    ) -> np.ndarray:
        """
        Build NNLS template matrix from candidate element data.

        Each column is an element's expected peak contribution based on
        Gaussian kernels at instrument resolution centered on each
        theoretical line position.  Three fixes over the original:

        1. **Per-peak sigma** — ``sigma_i = lambda_i / RP / 2.355`` varies
           across the spectral window instead of using the mean wavelength.
        2. **Per-element shift** — median wavelength shift from the matching
           phase is applied to each line position so the template aligns
           with the actual peak locations.
        3. **3-sigma proximity filter** — only peaks within 3 sigma of a
           line receive its contribution.  At 3 sigma the Gaussian is
           ``exp(-4.5) ~ 0.011``, so excluded contributions are < 1%.

        Parameters
        ----------
        candidates : List[dict]
            Candidate dicts with ``fused_lines``, ``matched_mask``, and
            ``wavelength_shifts`` keys.
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples.

        Returns
        -------
        np.ndarray
            Matrix A of shape (n_peaks, n_candidates).
        """
        n_peaks = len(peaks)
        n_cands = len(candidates)
        A = np.zeros((n_peaks, n_cands))
        peak_wls = np.array([p[1] for p in peaks])

        # Per-peak sigma (FWHM = lambda/R, sigma = FWHM/2.355)
        peak_sigmas = peak_wls / self.resolving_power / 2.355

        for j, cand in enumerate(candidates):
            # Per-element global shift from matching phase
            mm = cand["matched_mask"]
            ws = cand["wavelength_shifts"]
            shifts = ws[mm] if np.any(mm) else np.array([0.0])
            shift = float(np.median(shifts))

            for line in cand["fused_lines"]:
                wl_shifted = line["wavelength_nm"] + shift
                eps = line["avg_emissivity"]

                # 3-sigma proximity filter: only contribute to nearby peaks
                diffs = np.abs(peak_wls - wl_shifted)
                relevant = diffs < (3.0 * peak_sigmas)
                if np.any(relevant):
                    A[relevant, j] += eps * np.exp(
                        -0.5 * ((peak_wls[relevant] - wl_shifted) / peak_sigmas[relevant]) ** 2
                    )
        return A

    def _compute_nnls_attribution(
        self,
        A: np.ndarray,
        peak_intensities: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve NNLS and return per-element attribution metrics.

        Returns three arrays:

        * **P_mix** — leave-one-out partial-R^2 (absolute).
        * **P_local** — local explanation score: what fraction of the
          observed intensity at claimed peaks is explained by this
          element's NNLS contribution?  FP elements that merely ride on
          a dominant element's peaks get P_local ~ 0.
        * **c** — raw NNLS coefficients (useful for diagnostics).

        Parameters
        ----------
        A : np.ndarray
            Template matrix (n_peaks, n_candidates).
        peak_intensities : np.ndarray
            Observed peak intensities (n_peaks,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (P_mix, P_local, c) arrays of length n_candidates.
        """
        n_cands = A.shape[1]
        if n_cands == 0 or np.all(A == 0):
            return np.ones(n_cands), np.ones(n_cands), np.zeros(n_cands)

        c, _ = nnls(A, peak_intensities)
        total_rss = float(np.sum((peak_intensities - A @ c) ** 2))

        # Total signal energy (denominator for partial R^2)
        total_energy = float(np.sum(peak_intensities**2))
        if total_energy == 0:
            return np.ones(n_cands), np.ones(n_cands), c

        # ── P_mix: leave-one-out partial R^2 ──
        P_mix = np.zeros(n_cands)
        for j in range(n_cands):
            A_reduced = np.delete(A, j, axis=1)
            if A_reduced.shape[1] == 0:
                P_mix[j] = 1.0
                continue
            c_reduced, _ = nnls(A_reduced, peak_intensities)
            rss_without = float(np.sum((peak_intensities - A_reduced @ c_reduced) ** 2))
            P_mix[j] = (rss_without - total_rss) / total_energy

        # ── P_local: local explanation score ──
        # For each element, compute what fraction of the observed
        # intensity at its claimed peaks is explained by its own
        # NNLS contribution.  This discriminates FP elements
        # (tiny coefficient on dominant-element peaks) from real
        # minor elements (significant coefficient on their own peaks).
        P_local = np.zeros(n_cands)
        for j in range(n_cands):
            # Peaks where element j has meaningful template presence
            claimed = A[:, j] > 1e-6
            if not np.any(claimed):
                P_local[j] = 0.0
                continue
            obs_at_claimed = np.sum(peak_intensities[claimed])
            if obs_at_claimed <= 0:
                P_local[j] = 0.0
                continue
            elem_contribution = np.sum(A[claimed, j] * c[j])
            P_local[j] = float(np.clip(elem_contribution / obs_at_claimed, 0.0, 1.0))

        return P_mix, P_local, c

    @staticmethod
    def _compute_ratio_consistency(
        fused_lines: List[dict],
        matched_mask: np.ndarray,
        matched_peak_idx: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
    ) -> float:
        """
        Intensity-ratio consistency between matched lines.

        For real elements, the pairwise log-ratios of observed peak
        intensities should correlate with theoretical emissivity
        log-ratios (both follow the same Boltzmann distribution).
        Coincidental matches hit peaks belonging to a *different*
        element, so the ratios are uncorrelated.

        Parameters
        ----------
        fused_lines : List[dict]
            Fused lines with 'avg_emissivity' keys.
        matched_mask : np.ndarray
            Boolean mask of matched lines.
        matched_peak_idx : np.ndarray
            Peak indices for matched lines.
        intensity : np.ndarray
            Experimental intensity array.
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples.

        Returns
        -------
        float
            R_rat in [0, 1].  1.0 = perfect ratio match, 0.0 = anti-
            correlated.  Returns 0.5 (neutral) with < 3 matched lines.
        """
        matched_indices = np.where(matched_mask)[0]
        if len(matched_indices) < 3:
            return 0.5  # Too few pairs for meaningful correlation

        # Apply self-absorption damping to resonance lines so theoretical
        # log-ratios better match observed ratios for strong transitions.
        SA_DAMPING = 0.3
        raw_emiss = np.array([fused_lines[i]["avg_emissivity"] for i in matched_indices])
        damping = np.array(
            [
                SA_DAMPING if getattr(fused_lines[i]["transition"], "E_i_ev", 1.0) < 0.1 else 1.0
                for i in matched_indices
            ]
        )
        emissivities = raw_emiss * damping
        obs_intensities = np.array(
            [intensity[peaks[matched_peak_idx[i]][0]] for i in matched_indices]
        )

        # Guard against zeros
        valid = (emissivities > 0) & (obs_intensities > 0)
        if np.sum(valid) < 3:
            return 0.5

        log_th = np.log(emissivities[valid])
        log_obs = np.log(obs_intensities[valid])

        # Build all pairwise log-ratio differences
        n = len(log_th)
        th_ratios = []
        exp_ratios = []
        for i in range(n):
            for j in range(i + 1, n):
                th_ratios.append(log_th[i] - log_th[j])
                exp_ratios.append(log_obs[i] - log_obs[j])

        if len(th_ratios) < 3:
            return 0.5

        th_arr = np.array(th_ratios)
        exp_arr = np.array(exp_ratios)

        # Pearson correlation of log-ratios
        corr = np.corrcoef(th_arr, exp_arr)[0, 1]
        if np.isnan(corr):
            return 0.5

        # Map [-1, 1] → [0, 1]; negative correlation is worse than zero
        return float(max(0.0, (corr + 1.0) / 2.0))

    def _compute_random_match_significance(
        self,
        peaks: List[Tuple[int, float]],
        wavelength: np.ndarray,
        N_expected: int,
        N_matched: int,
    ) -> Tuple[float, float, float, float]:
        """
        Compute chance-coincidence significance from a binomial tail test.

        Uses per-element theoretical line occupancy as the chance probability:
        p_chance = fraction of spectral span occupied by N_expected
        above-threshold theoretical line windows (not experimental peaks).

        Parameters
        ----------
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples.
        wavelength : np.ndarray
            Full spectral wavelength axis in nm.
        N_expected : int
            Number of above-threshold theoretical lines.
        N_matched : int
            Number of above-threshold lines matched to peaks.

        Returns
        -------
        Tuple[float, float, float, float]
            (P_sig, fill_factor, p_chance, p_tail), where:
            - fill_factor is experimental peak fill factor (for metadata)
            - p_chance is theoretical-window occupancy
            - p_tail = P(X >= N_matched | n=N_expected, p=p_chance)
            - P_sig = 1 - p_tail
        """
        fill_factor = self._compute_fill_factor(peaks, wavelength)

        # Theoretical-window occupancy: per-element chance probability
        wl_min = float(np.min(wavelength))
        wl_max = float(np.max(wavelength))
        span = wl_max - wl_min
        if span <= 0 or N_expected <= 0:
            p_chance = float(np.clip(fill_factor, 1e-6, 1.0 - 1e-6))
        else:
            mean_wl = 0.5 * (wl_min + wl_max)
            line_window = mean_wl / self.resolving_power  # delta_lambda
            # Each line occupies ±line_window around its center
            theoretical_coverage = N_expected * 2 * line_window / span
            p_chance = float(np.clip(theoretical_coverage, 1e-6, 1.0 - 1e-6))

        if N_expected <= 0 or N_matched <= 0:
            return 1.0, fill_factor, p_chance, 1.0

        # Binomial test: "Given N_expected opportunities, what's the
        # probability of N_matched or more matches by chance?"
        n_trials = N_expected
        n_success = N_matched

        if n_success > n_trials:
            # More matches than theoretical lines — extremely unlikely
            # by chance.  Can happen with fused-line bookkeeping; treat
            # as maximally significant.
            return 1.0, fill_factor, p_chance, 0.0

        p_tail = float(binom.sf(n_success - 1, n_trials, p_chance))
        P_sig = float(np.clip(1.0 - p_tail, 0.0, 1.0))

        return P_sig, fill_factor, p_chance, p_tail

    @staticmethod
    def _compute_p_snr(intensity: np.ndarray, peaks: List[Tuple[int, float]]) -> float:
        """Compute erf-based SNR quality factor used in CL."""
        if len(peaks) > 0:
            peak_intensities_local = [intensity[p[0]] for p in peaks]
            median_peak = np.median(peak_intensities_local)
            noise_estimate = np.median(np.abs(intensity - np.median(intensity))) * 1.4826
            noise_estimate = max(noise_estimate, 1e-10)
            z = (median_peak - noise_estimate) / (noise_estimate * math.sqrt(2))
            return 0.5 * (1.0 + float(erf(z)))
        return 0.5

    def _decide(
        self,
        k_sim: float,
        k_rate: float,
        k_shift: float,
        N_expected: int,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
        element: str = "",
        P_maj: float = 0.5,
        P_sig: float = 1.0,
        N_matched: int = 0,
        P_cov: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Compute detection score k_det and confidence level CL.

        Parameters
        ----------
        k_sim : float
            Similarity score (matched-only cosine similarity)
        k_rate : float
            Detection rate score (emissivity-weighted)
        k_shift : float
            Wavelength shift score
        N_expected : int
            Number of above-threshold theoretical lines (for gates/penalties)
        intensity : np.ndarray
            Experimental intensity array
        peaks : List[Tuple[int, float]]
            Experimental peaks
        element : str
            Element symbol (for crustal abundance weighting)
        P_maj : float
            Major-line coverage factor (0.5–1.0), computed from top-k
            strongest theoretical lines
        P_sig : float
            Statistical significance factor against random coincidence
        N_matched : int
            Number of matched above-threshold lines (used in k_det blend)
        P_cov : float
            Emissivity-weighted coverage penalty (0–1)

        Returns
        -------
        Tuple[float, float]
            (k_det, CL) detection score and confidence level.
        """
        # k_det formula — uses N_matched (paper: N_X = matched count)
        # for the blend weighting.  Single-line elements (N_X=1) naturally
        # reduce to k_rate × k_shift via the blend formula.
        if N_matched > 0:
            N_X = N_matched
            k_det = k_rate * ((1.0 / N_X) * k_shift + ((N_X - 1.0) / N_X) * k_sim)
        else:
            k_det = 0.0

        P_SNR = self._compute_p_snr(intensity, peaks)

        # P_ab — crustal abundance prior
        P_ab = self._compute_P_ab(element)

        # Confidence level — paper formula (Noel et al. 2025):
        # CL = k_det × P_SNR × P_maj × P_ab
        # P_cov, N_penalty, P_sig are stored in metadata for diagnostics
        # but no longer multiply into CL.  NNLS P_mix is applied later.
        CL = k_det * P_SNR * P_maj * P_ab

        return k_det, CL