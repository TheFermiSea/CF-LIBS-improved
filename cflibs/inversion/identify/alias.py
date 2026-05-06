"""
ALIAS (Automated Line Identification Algorithm for Spectroscopy) implementation.

Based on Noel et al. (2025) arXiv:2501.01057. The ALIAS algorithm identifies elements
in LIBS spectra through a 7-step process: peak detection, theoretical emissivity
calculation, line fusion, matching, threshold determination, scoring, and decision.
"""

from typing import List, Tuple, Optional, Set
from collections import defaultdict
import math
import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks
from scipy.special import erf
from scipy.stats import binom, linregress

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

    Thread-safety
    -------------
    ``identify()`` mutates instance state (``_effective_R``,
    ``_global_wl_shift``, ``_estimated_T``), so a single instance is
    **not** safe for concurrent calls. Create one instance per thread or
    guard calls with an external lock.

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
    max_screening_candidates : int, optional
        Maximum number of candidates retained by fast screening (default: 12)
    relative_cl_threshold : float, optional
        CL must be >= max_CL * relative_cl_threshold to count as detected.
        Set to 0 to disable the relative threshold (default: 0.1)
    """

    # Temperature bounds for physics validation
    _T_ESTIMATE_MIN_K = 3000.0
    _T_ESTIMATE_MAX_K = 30000.0
    # Consistency check uses a wider range because its purpose is only to
    # flag grossly unphysical fits, not to narrow the estimate.
    _T_CONSISTENCY_MIN_K = 3000.0
    _T_CONSISTENCY_MAX_K = 50000.0

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
        detection_threshold: float = 0.02,
        chance_window_scale: float = 0.4,
        elements: Optional[List[str]] = None,
        max_lines_per_element: int = 20,
        reference_temperature: float = 10000.0,
        max_screening_candidates: int = 12,
        relative_cl_threshold: float = 0.1,
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
        self.max_screening_candidates = max_screening_candidates
        self.relative_cl_threshold = relative_cl_threshold

        # Create Saha-Boltzmann solver
        self.solver = SahaBoltzmannSolver(atomic_db)

        # Create (T, n_e) grid
        self.T_grid_K = np.linspace(T_range_K[0], T_range_K[1], T_steps)
        self.n_e_grid_cm3 = np.linspace(n_e_range_cm3[0], n_e_range_cm3[1], n_e_steps)

        # Set during identify() by auto-calibration
        self._effective_R: Optional[float] = None
        self._global_wl_shift: float = 0.0

        # Ubiquitous atmospheric/ablation contaminants always tested
        self._always_test: Set[str] = {"H"}

        # Estimated plasma temperature (set by _estimate_plasma_temperature)
        self._estimated_T: Optional[float] = None

    def identify(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> ElementIdentificationResult:
        """
        Identify elements in experimental spectrum with cross-element peak
        competition.

        Enhanced multi-phase algorithm:
        0. Baseline correction + peak detection
        0a. Wavelength auto-calibration (estimate global shift + effective R)
        0b. Plasma temperature estimation (for adaptive emissivities)
        0c. Fast screening (restrict candidate elements)
        1. Score screened elements independently
        2. Global peak competition
        3. Rescore, Boltzmann consistency check, build results

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
        # Step 0: Baseline correction — ALL scoring uses corrected intensities
        # so that cosine similarity, NNLS, and P_SNR measure peak heights above
        # continuum rather than absolute intensity that is dominated by the
        # Bremsstrahlung background.
        baseline = estimate_baseline(wavelength, intensity)
        corrected_intensity = np.maximum(intensity - baseline, 0.0)

        # Step 1: Detect peaks (uses its own internal baseline correction)
        peaks = self._detect_peaks(wavelength, intensity)

        wl_min = np.min(wavelength)
        wl_max = np.max(wavelength)

        # Step 0a: Auto-calibrate wavelength (estimate global shift + effective R)
        self._auto_calibrate_wavelength(peaks, wl_min, wl_max)

        # Step 0b: Estimate plasma temperature from detected peaks
        self._estimate_plasma_temperature(peaks, corrected_intensity, wl_min, wl_max)

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

        # Step 0c: Fast screening — restrict to elements with strong-line matches
        # Skip screening when user explicitly provided a short element list
        if self.elements is not None and len(self.elements) <= 10:
            screened = search_elements
        else:
            screened = self._fast_screening(search_elements, peaks, wl_min, wl_max)

        # ── Phase 1: Independent scoring ──────────────────────────────
        # Use corrected_intensity throughout scoring so continuum doesn't
        # dominate cosine similarity and NNLS attribution.
        global_p_snr = self._compute_p_snr(corrected_intensity, peaks)
        candidates: List[dict] = []

        for element in screened:
            element_lines = self._compute_element_emissivities(
                element, wl_min, wl_max, T_estimated=self._estimated_T
            )
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
                corrected_intensity,
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
                corrected_intensity,
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
        peak_intensities_arr = None
        A = None
        if candidates and peaks:
            peak_intensities_arr = np.array([corrected_intensity[p[0]] for p in peaks])
            A = self._build_nnls_templates(candidates, peaks)
            P_mix_arr, P_local_arr, _ = self._compute_nnls_attribution(A, peak_intensities_arr)

            # Sparse NNLS: L1-penalized fit suppresses diffuse FPs
            # (Black et al. 2024: standard NNLS overfits → many small
            # non-zero coefficients for absent elements)
            # Higher alpha at low RP where blending causes more false sharing
            sparse_alpha = 0.05 if self.resolving_power < 1000 else 0.01
            sparse_c, _ = self._compute_sparse_nnls_scores(
                A, peak_intensities_arr, alpha=sparse_alpha
            )
            # Noise floor: coefficient must exceed 10% of median to be
            # considered significant — elements below this are noise
            nonzero_c = sparse_c[sparse_c > 0]
            nnls_noise = float(np.median(nonzero_c) * 0.1) if len(nonzero_c) > 0 else 0.0

            for i, cand in enumerate(candidates):
                cand["P_mix"] = float(P_mix_arr[i])
                cand["P_local"] = float(P_local_arr[i])
                cand["sparse_nnls_coeff"] = float(sparse_c[i])
                cand["nnls_significant"] = float(sparse_c[i]) > nnls_noise
        else:
            for cand in candidates:
                cand["P_mix"] = 1.0
                cand["P_local"] = 1.0
                cand["sparse_nnls_coeff"] = 0.0
                cand["nnls_significant"] = False

        # ── Phase 1.75: Iron-group pre-subtraction (ChemCam-style) ────
        # At low RP, Fe/Mn/Cr/Ti create a dense pseudo-continuum that
        # inflates other elements' NNLS ownership scores.  Subtract
        # their predicted contribution from peak intensities and
        # recompute P_local for non-iron-group elements so the gate
        # discriminates on the residual, not the raw spectrum.
        _IRON_GROUP = {"Fe", "Mn", "Cr", "Ti"}
        if candidates and peaks and A is not None and self.resolving_power < 2000:
            ig_indices = [i for i, c in enumerate(candidates) if c["element"] in _IRON_GROUP]
            if ig_indices and peak_intensities_arr is not None:
                ig_contribution = np.zeros_like(peak_intensities_arr)
                c_nnls = np.zeros(len(candidates))
                # Re-solve NNLS to get coefficients
                try:
                    from scipy.optimize import nnls as _nnls

                    c_nnls, _ = _nnls(A, peak_intensities_arr)
                except Exception:
                    pass
                for idx in ig_indices:
                    ig_contribution += c_nnls[idx] * A[:, idx]

                # Compute residual peak intensities
                residual_peaks = np.maximum(peak_intensities_arr - ig_contribution, 0.0)
                residual_total = float(np.sum(residual_peaks))

                if residual_total > 0:
                    # Recompute P_local for non-iron-group elements against residual
                    for i, cand in enumerate(candidates):
                        if cand["element"] in _IRON_GROUP:
                            continue
                        claimed = A[:, i] > 1e-6
                        if not np.any(claimed):
                            continue
                        obs_residual = np.sum(residual_peaks[claimed])
                        if obs_residual <= 0:
                            cand["P_local"] = 0.0
                            continue
                        elem_contrib = np.sum(A[claimed, i] * c_nnls[i])
                        cand["P_local"] = float(np.clip(elem_contrib / obs_residual, 0.0, 1.0))

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
                    corrected_intensity,
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
                    corrected_intensity,
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
                corrected_intensity,
                peaks,
            )

            # Post-CL discriminators — suppress false positives whose peaks
            # ride on a dominant element's lines.

            # Gate 1: P_local — soft ramp with 0.25 floor
            CL *= float(np.clip(P_local + 0.25, 0.25, 1.0))

            # Hard rejection: negligible NNLS ownership means this element's
            # peaks are fully explained by other elements.
            # Adaptive threshold: line-rich elements (Fe, Ca, Mn) overlap
            # heavily at low RP, driving P_local artificially low even for
            # true positives.  Use a softer threshold for them.
            # Strong multi-line evidence (high match rate + decent k_sim)
            # can bypass P_local entirely — the element matched most of
            # its lines with consistent intensities.
            p_local_threshold = 0.01 if N_expected >= 10 else 0.05
            high_match_evidence = N_expected >= 5 and N_matched >= 0.7 * N_expected and k_sim > 0.3
            if P_local < p_local_threshold and not high_match_evidence:
                CL = 0.0

            # Gate 2: P_mix — moderate gate, 0.2 floor
            # True minor elements have P_mix ~0.02-0.10, FPs have ~0.000-0.005
            CL *= float(np.clip(0.2 + 8.0 * P_mix, 0.2, 1.0))

            # Gate 3: R_rat — soft consistency check (0.5 min, 1.0 max)
            CL *= 0.5 + 0.5 * R_rat

            # Gate 4: Boltzmann consistency — verify matched lines follow
            # ln(I·λ/gA) vs E_k with physically reasonable temperature
            boltz_factor = self._boltzmann_consistency_check(
                fused_lines,
                matched_mask,
                matched_peak_idx,
                corrected_intensity,
                peaks,
            )
            CL *= boltz_factor

            # Gate 5: Sparse NNLS significance (primary discriminator at low RP)
            # At RP<2000, peak-matching CL cannot discriminate (TP/FP overlap).
            # The sparse NNLS coefficient is the strongest false-positive
            # suppressor: elements zeroed out by L1 penalty are truly absent.
            nnls_sig = cand.get("nnls_significant", True)
            if not nnls_sig and self.resolving_power < 2000:
                CL = 0.0

            # Adaptive detection threshold: elements with few expected
            # lines have higher false-match rates at low RP and need a
            # proportionally higher CL to be considered detected.
            adaptive_dt = self.detection_threshold
            if N_expected > 0 and N_expected < 10:
                adaptive_dt *= min(3.0, math.sqrt(10.0 / N_expected))
            detected = CL >= adaptive_dt

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
                            intensity_exp=corrected_intensity[peaks[pidx][0]],
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
                    "N_penalty": min(1.0, math.sqrt(N_expected / 5.0)) if N_expected > 0 else 0.0,
                    "boltzmann_factor": boltz_factor,
                    "sparse_nnls_coeff": cand.get("sparse_nnls_coeff", 0.0),
                    "nnls_significant": cand.get("nnls_significant", True),
                    "effective_R": self._effective_R,
                    "global_wl_shift": self._global_wl_shift,
                    "estimated_T": self._estimated_T,
                },
            )

            all_element_ids.append(element_id)

        # Apply relative threshold: element CL must be >= max_CL * relative_cl_threshold
        # This prevents spurious detections when one element dominates.
        # Set self.relative_cl_threshold = 0 to disable.
        if all_element_ids and self.relative_cl_threshold > 0:
            max_CL = max(e.confidence for e in all_element_ids)
            relative_threshold = max_CL * self.relative_cl_threshold
            for e in all_element_ids:
                if e.confidence < relative_threshold:
                    e.detected = False

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
                "effective_R": self._effective_R,
                "global_wl_shift_nm": self._global_wl_shift,
                "estimated_T_K": self._estimated_T,
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

    def _auto_calibrate_wavelength(
        self,
        peaks: List[Tuple[int, float]],
        wl_min: float,
        wl_max: float,
    ) -> None:
        """
        Auto-calibrate wavelength offset and effective resolving power.

        Compares detected peak positions to the strongest NIST reference
        lines across common LIBS elements to estimate:
        1. Global wavelength shift (median offset from best matches)
        2. Effective resolving power (from distribution of offsets)

        Sets self._global_wl_shift and self._effective_R.
        """
        if not peaks:
            self._global_wl_shift = 0.0
            self._effective_R = self.resolving_power
            return

        peak_wls = np.array([p[1] for p in peaks])

        # Get strong reference lines from common LIBS elements
        reference_elements = ["Fe", "Ca", "Mg", "Ti", "Al", "Cu", "Na", "Si", "Cr", "Mn"]
        kT_ref = KB_EV * self.reference_temperature

        ref_lines = []
        for el in reference_elements:
            for ion_stage in [1, 2]:
                try:
                    trans = self.atomic_db.get_transitions(
                        el, ion_stage, wavelength_min=wl_min, wavelength_max=wl_max
                    )
                    if trans:
                        for t in trans:
                            strength = t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT_ref)
                            ref_lines.append((t.wavelength_nm, strength, el))
                except (KeyError, ValueError, AttributeError):
                    continue

        if not ref_lines:
            self._global_wl_shift = 0.0
            self._effective_R = self.resolving_power
            return

        # Take top 30 strongest reference lines
        ref_lines.sort(key=lambda x: x[1], reverse=True)
        top_refs = ref_lines[:30]

        # For each reference line, find the nearest peak using a generous
        # initial tolerance (R=1000, ~0.4nm at 400nm)
        offsets = []
        for ref_wl, _strength, _el in top_refs:
            dists = peak_wls - ref_wl
            abs_dists = np.abs(dists)
            generous_tol = ref_wl / 1000.0  # R=1000
            within = abs_dists <= generous_tol
            if np.any(within):
                best_idx = np.argmin(abs_dists)
                offsets.append(float(dists[best_idx]))

        if len(offsets) < 3:
            self._global_wl_shift = 0.0
            self._effective_R = self.resolving_power
            return

        # Global shift = median offset
        self._global_wl_shift = float(np.median(offsets))

        # Effective R: estimate from MAD of offsets after shift correction
        corrected_offsets = np.array(offsets) - self._global_wl_shift
        mad = float(np.median(np.abs(corrected_offsets)))
        # The matching tolerance delta_lambda = mean_wl / R
        # We want delta_lambda ~ 3*MAD to capture 99% of real matches
        mean_wl = 0.5 * (wl_min + wl_max)
        if mad > 0:
            estimated_R = mean_wl / (3.0 * mad)
            # Clamp to reasonable range [500, nominal R]
            self._effective_R = float(np.clip(estimated_R, 500.0, self.resolving_power))
        else:
            # Perfect calibration — use nominal R
            self._effective_R = self.resolving_power

    def _estimate_plasma_temperature(
        self,
        peaks: List[Tuple[int, float]],
        corrected_intensity: np.ndarray,
        wl_min: float,
        wl_max: float,
    ) -> None:
        """
        Estimate plasma temperature from Boltzmann slope of strong detected peaks.

        Uses Fe I lines preferentially (most common in LIBS). Falls back to any
        transition metal with enough matched lines, then a line-ratio method,
        and finally ``self.reference_temperature`` if all methods fail.

        Always sets ``self._estimated_T`` (K) to a finite value.
        """
        if len(peaks) < 3:
            self._estimated_T = self.reference_temperature
            return

        peak_wls = np.array([p[1] for p in peaks])
        peak_intensities = np.array([corrected_intensity[p[0]] for p in peaks])

        # Try to match strong lines from Fe I, Ti I, Cr I, Ca I etc.
        probe_elements = ["Fe", "Ti", "Cr", "Ca", "Mn", "Ni", "V", "Cu", "Mg", "Si", "Al"]
        delta_lambda = 0.5 * (wl_min + wl_max) / max(self._effective_R or self.resolving_power, 500)

        for probe_el in probe_elements:
            try:
                transitions = self.atomic_db.get_transitions(
                    probe_el, 1, wavelength_min=wl_min, wavelength_max=wl_max
                )
                if not transitions:
                    continue
            except (KeyError, ValueError, AttributeError):
                continue

            # Filter to lines with known A_ki and g_k
            good_trans = [t for t in transitions if t.A_ki > 0 and t.g_k > 0 and t.E_k_ev > 0]
            if len(good_trans) < 4:
                continue

            # Sort by expected strength and take top 15
            kT_ref = KB_EV * self.reference_temperature
            good_trans.sort(
                key=lambda t: t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT_ref),
                reverse=True,
            )
            good_trans = good_trans[:15]

            # Match to peaks
            E_k_vals = []
            y_vals = []
            shift = self._global_wl_shift

            for t in good_trans:
                wl_shifted = t.wavelength_nm + shift
                dists = np.abs(peak_wls - wl_shifted)
                best_idx = int(np.argmin(dists))
                if dists[best_idx] <= delta_lambda:
                    I_obs = peak_intensities[best_idx]
                    if I_obs > 0 and t.A_ki > 0 and t.g_k > 0:
                        y = math.log(I_obs * t.wavelength_nm / (t.g_k * t.A_ki))
                        E_k_vals.append(t.E_k_ev)
                        y_vals.append(y)

            if len(E_k_vals) < 4:
                continue

            # Fit Boltzmann slope: y = -1/(kT) * E_k + const
            E_k_arr = np.array(E_k_vals)
            y_arr = np.array(y_vals)

            try:
                result = linregress(E_k_arr, y_arr)
                slope = result.slope
                r_sq = result.rvalue**2

                if abs(slope) < 1e-10:
                    continue
                if slope < 0 and r_sq > 0.2:
                    T_K = -1.0 / (slope * KB_EV)
                    if self._T_ESTIMATE_MIN_K < T_K < self._T_ESTIMATE_MAX_K:
                        self._estimated_T = float(T_K)
                        return
            except (ValueError, ZeroDivisionError):
                continue

        # Pass 2: Line-ratio fallback — estimate T from best 2-line pair
        shift = self._global_wl_shift
        for probe_el in probe_elements:
            try:
                transitions = self.atomic_db.get_transitions(
                    probe_el, 1, wavelength_min=wl_min, wavelength_max=wl_max
                )
                if not transitions:
                    continue
            except (KeyError, ValueError, AttributeError):
                continue

            good_trans = [t for t in transitions if t.A_ki > 0 and t.g_k > 0 and t.E_k_ev > 0]
            if len(good_trans) < 2:
                continue

            # Match to peaks
            matched_pairs = []
            for t in good_trans:
                wl_shifted = t.wavelength_nm + shift
                dists = np.abs(peak_wls - wl_shifted)
                best_idx = int(np.argmin(dists))
                if dists[best_idx] <= delta_lambda:
                    I_obs = peak_intensities[best_idx]
                    if I_obs > 0:
                        matched_pairs.append((t, I_obs))

            if len(matched_pairs) < 2:
                continue

            # Find pair with largest E_k separation
            best_T_local = None
            best_dE = 0.0
            for i in range(len(matched_pairs)):
                for j in range(i + 1, len(matched_pairs)):
                    t1, I1 = matched_pairs[i]
                    t2, I2 = matched_pairs[j]
                    dE = abs(t1.E_k_ev - t2.E_k_ev)
                    if dE < 0.5:
                        continue
                    numer = I2 * t2.wavelength_nm * t1.g_k * t1.A_ki
                    denom = I1 * t1.wavelength_nm * t2.g_k * t2.A_ki
                    if denom <= 0 or numer <= 0:
                        continue
                    ln_ratio = math.log(numer / denom)
                    if abs(ln_ratio) < 1e-10:
                        continue
                    T_K = -(t2.E_k_ev - t1.E_k_ev) / (KB_EV * ln_ratio)
                    if self._T_ESTIMATE_MIN_K < T_K < self._T_ESTIMATE_MAX_K and dE > best_dE:
                        best_T_local = T_K
                        best_dE = dE

            if best_T_local is not None:
                self._estimated_T = float(best_T_local)
                return

        # Final fallback: use reference temperature instead of None
        self._estimated_T = self.reference_temperature

    def _fast_screening(
        self,
        all_elements: List[str],
        peaks: List[Tuple[int, float]],
        wl_min: float,
        wl_max: float,
    ) -> List[str]:
        """
        Fast screening to restrict candidate elements.

        Two-stage approach:
        1. For each element, compute a quick screening score based on how many
           of its top-10 lines match peaks, weighted by line strength.
        2. Pass the top max_screening_candidates scoring elements.

        Always-test elements bypass screening.

        Returns list of elements that passed screening.
        """
        if not peaks:
            return list(self._always_test & set(all_elements))

        peak_wls = np.array([p[1] for p in peaks])
        eff_R = self._effective_R or self.resolving_power
        mean_wl = 0.5 * (wl_min + wl_max)
        delta_lambda = mean_wl / eff_R
        screening_tol = 2.0 * delta_lambda
        shift = self._global_wl_shift

        kT_ref = KB_EV * self.reference_temperature
        element_scores = []

        for element in all_elements:
            if element in self._always_test:
                continue

            # Get all lines and compute strengths
            lines_with_strength = []
            for ion_stage in [1, 2]:
                try:
                    trans = self.atomic_db.get_transitions(
                        element, ion_stage, wavelength_min=wl_min, wavelength_max=wl_max
                    )
                    if trans:
                        for t in trans:
                            strength = t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT_ref)
                            lines_with_strength.append((t.wavelength_nm, strength))
                except (KeyError, ValueError, AttributeError):
                    continue

            if not lines_with_strength:
                continue

            lines_with_strength.sort(key=lambda x: x[1], reverse=True)
            top10 = lines_with_strength[:10]

            # Compute screening score: sum of strengths for matched lines
            # divided by total strength (strength-weighted match rate)
            total_strength = sum(s for _, s in top10)
            matched_strength = 0.0
            n_matched = 0
            for wl_th, strength in top10:
                wl_shifted = wl_th + shift
                dists = np.abs(peak_wls - wl_shifted)
                if np.min(dists) <= screening_tol:
                    matched_strength += strength
                    n_matched += 1

            # Single-line exception: elements with ≤1 line in the window
            # (e.g., Li I 670.8nm) need only 1 match to pass screening.
            min_matches = 1 if len(top10) <= 1 else 2
            if n_matched >= min_matches and total_strength > 0:
                score = matched_strength / total_strength
                if score >= 0.3:
                    element_scores.append((element, score, n_matched))

        # Sort by screening score, take top max_screening_candidates
        element_scores.sort(key=lambda x: x[1], reverse=True)
        passed = list(self._always_test & set(all_elements))
        for element, score, n_matched in element_scores[: self.max_screening_candidates]:
            if element not in passed:
                passed.append(element)

        return passed

    def _boltzmann_consistency_check(
        self,
        fused_lines: List[dict],
        matched_mask: np.ndarray,
        matched_peak_idx: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
    ) -> float:
        """
        Boltzmann consistency check for matched lines.

        For elements with >=3 matched lines, fit ln(I*lambda/(g*A)) vs E_k.
        Slope should give physical temperature (3000-50000K) with reasonable R^2.

        Returns a factor in [0.5, 1.0] to multiply into CL.
        """
        matched_indices = np.nonzero(matched_mask)[0]
        if len(matched_indices) < 3:
            return 0.5  # Penalize — not enough lines for Boltzmann check

        E_k_vals = []
        y_vals = []

        for i in matched_indices:
            trans = fused_lines[i]["transition"]
            pidx = int(matched_peak_idx[i])
            if pidx < 0 or pidx >= len(peaks):
                continue
            I_obs = intensity[peaks[pidx][0]]
            if I_obs <= 0 or trans.A_ki <= 0 or trans.g_k <= 0:
                continue

            y = math.log(I_obs * trans.wavelength_nm / (trans.g_k * trans.A_ki))
            E_k_vals.append(trans.E_k_ev)
            y_vals.append(y)

        if len(E_k_vals) < 3:
            return 0.5

        E_k_arr = np.array(E_k_vals)
        y_arr = np.array(y_vals)

        # Need some spread in E_k for meaningful fit
        if np.ptp(E_k_arr) < 0.5:
            return 1.0  # All same energy level, can't fit

        try:
            result = linregress(E_k_arr, y_arr)
            slope = result.slope
            r_sq = result.rvalue**2
        except (ValueError, ZeroDivisionError):
            return 1.0

        # Check physical validity
        if slope >= 0:
            # Positive slope = anti-Boltzmann → likely false positive
            return 0.5

        T_K = -1.0 / (slope * KB_EV)
        if T_K < self._T_CONSISTENCY_MIN_K or T_K > self._T_CONSISTENCY_MAX_K:
            # Unphysical temperature
            return 0.5

        # Scale by R^2: good fit → 1.0, poor fit → 0.7
        return float(0.7 + 0.3 * min(r_sq, 1.0))

    def _compute_element_emissivities(
        self,
        element: str,
        wl_min: float,
        wl_max: float,
        T_estimated: Optional[float] = None,
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

        # When T_estimated is available, use a narrow grid around that T
        # instead of the full T range. This makes emissivities reflect the
        # actual plasma conditions rather than averaged-out values.
        if T_estimated is not None:
            T_grid = np.array([T_estimated])
        else:
            # Always use a single reference T — averaging over the full
            # grid dilutes the reference vector and makes cosine similarity
            # meaningless.
            T_grid = np.array([self.reference_temperature])

        # Precompute stage densities for all (T, n_e) grid points
        grid_stage_densities = {}
        for T_K in T_grid:
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

            for T_K in T_grid:
                for n_e in self.n_e_grid_cm3:
                    T_eV = T_K * KB_EV

                    # Get precomputed ionization balance
                    cached_densities = grid_stage_densities.get((T_K, n_e))
                    if cached_densities is None:
                        continue

                    stage_density = cached_densities.get(transition.ionization_stage, 0.0)
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

        # Resolution element at mean wavelength — use effective R if available
        mean_wl = np.mean(wavelength_nm)
        eff_R = self._effective_R or self.resolving_power
        delta_lambda = mean_wl / eff_R

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

        Uses auto-calibrated global wavelength shift and effective resolving
        power from _auto_calibrate_wavelength(). Two-pass strategy:
        - Pass 1: tight tolerance (delta_lambda from effective R)
        - Pass 2: for unmatched strong lines, wider tolerance (2x delta_lambda)

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

        # Use auto-calibrated shift and effective R
        global_shift = self._global_wl_shift
        eff_R = self._effective_R or self.resolving_power
        mean_wl = np.mean([line["wavelength_nm"] for line in fused_lines])
        delta_lambda = mean_wl / eff_R

        # Additionally refine per-element shift from top 10 lines
        sorted_by_emissivity = sorted(fused_lines, key=lambda x: x["avg_emissivity"], reverse=True)
        top_lines = sorted_by_emissivity[: min(10, len(sorted_by_emissivity))]

        per_element_shifts = []
        for line in top_lines:
            wl_th = line["wavelength_nm"] + global_shift
            distances = np.abs(peak_wavelengths - wl_th)
            if len(distances) > 0:
                min_dist = np.min(distances)
                if min_dist <= 1.5 * delta_lambda:
                    closest_idx = int(np.argmin(distances))
                    per_element_shifts.append(peak_wavelengths[closest_idx] - line["wavelength_nm"])

        # Use per-element shift if enough matches, else fall back to global
        if len(per_element_shifts) >= 2:
            element_shift = float(np.median(per_element_shifts))
        else:
            element_shift = global_shift

        matched_mask = np.zeros(n, dtype=bool)
        wavelength_shifts = np.zeros(n)
        matched_peak_idx = np.full(n, -1, dtype=int)

        # Pass 1: tight tolerance
        for i, line in enumerate(fused_lines):
            wl_th = line["wavelength_nm"] + element_shift

            distances = np.abs(peak_wavelengths - wl_th)
            within_window = distances <= delta_lambda

            if np.any(within_window):
                matched_mask[i] = True
                closest_idx = int(np.argmin(distances))
                matched_peak_idx[i] = closest_idx
                wavelength_shifts[i] = peak_wavelengths[closest_idx] - line["wavelength_nm"]

        # Pass 2: for unmatched strong lines, try wider tolerance (2x)
        # "strong" = above median emissivity of all lines
        emissivities = np.array([line["avg_emissivity"] for line in fused_lines])
        emiss_median = np.median(emissivities) if len(emissivities) > 0 else 0.0
        wide_tol = 2.0 * delta_lambda

        for i, line in enumerate(fused_lines):
            if matched_mask[i]:
                continue  # Already matched
            if emissivities[i] < emiss_median:
                continue  # Only retry strong lines

            wl_th = line["wavelength_nm"] + element_shift
            distances = np.abs(peak_wavelengths - wl_th)
            within_wide = distances <= wide_tol

            if np.any(within_wide):
                matched_mask[i] = True
                closest_idx = int(np.argmin(distances))
                matched_peak_idx[i] = closest_idx
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
        detection_rates_arr = np.asarray(detection_rates)
        thresholds_arr = np.asarray(thresholds)

        above_50 = detection_rates_arr > 0.5
        if np.any(above_50):
            # Return lowest threshold with >50% detection
            candidate = thresholds_arr[np.where(above_50)[0][0]]
        else:
            candidate = min_log

        # Additional constraint: never count more than max_lines_per_element
        # above-threshold lines.  For elements with hundreds of DB lines (Fe,
        # V, Ti), a low threshold counts dozens of weak lines that are below
        # noise.  Raise the threshold until the above-threshold count is
        # manageable, so N_expected reflects only realistically detectable
        # lines rather than the full database catalogue.
        emiss_sorted = np.sort(emissivities)[::-1]
        if len(emiss_sorted) > self.max_lines_per_element:
            # Threshold = emissivity of the (max_lines)th strongest line
            floor = np.log10(max(emiss_sorted[self.max_lines_per_element - 1], 1e-100))
            candidate = max(candidate, floor)

        return candidate

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
            # Set to 0.0 — single-line elements are penalized via k_det
            # blend (N_X=1 means k_sim is not used) and the N_penalty.
            k_sim = 0.0

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
    def _compute_sparse_nnls_scores(
        A: np.ndarray,
        peak_intensities: np.ndarray,
        alpha: float = 0.01,
        l1_ratio: float = 0.9,
    ) -> Tuple[np.ndarray, float]:
        """
        Sparse NNLS via L-BFGS-B constrained optimization.

        Standard NNLS distributes signal across correlated endmembers,
        producing many small non-zero coefficients for absent elements
        (Black & Burnside 2024). The L1 penalty enforces sparsity, driving
        truly absent elements to zero.

        Physics-only implementation: minimizes the elastic-net objective
        with non-negativity via L-BFGS-B bounds rather than sklearn.

        Parameters
        ----------
        A : np.ndarray
            Template matrix (n_peaks, n_candidates).
        peak_intensities : np.ndarray
            Observed peak intensities (n_peaks,).
        alpha : float
            Regularization strength (higher = sparser).
        l1_ratio : float
            L1 vs L2 mix (1.0 = pure lasso, 0.0 = pure ridge).

        Returns
        -------
        Tuple[np.ndarray, float]
            (coefficients, residual_norm) — sparse non-negative coefficients
            and the norm of the fit residual.
        """
        n_cands = A.shape[1]
        if n_cands == 0 or np.all(A == 0) or np.all(peak_intensities == 0):
            return np.zeros(n_cands), 0.0

        try:
            # Physics-only non-negative elastic-net via L-BFGS-B. Mirrors
            # sklearn.linear_model.ElasticNet(positive=True, ...) but without
            # importing an ML library (see CF-LIBS-improved-3fy3): minimize
            #   0.5 * ||A_norm x - y||^2
            #   + alpha * ( l1_ratio * sum(x) + 0.5 * (1-l1_ratio) * x^T x )
            # subject to x >= 0. Under x >= 0 the L1 term reduces to a smooth
            # linear sum(x), so the full objective is differentiable and
            # L-BFGS-B handles the non-negativity via bounds.
            from scipy.optimize import minimize

            col_norms = np.linalg.norm(A, axis=0)
            col_norms[col_norms == 0] = 1.0
            A_norm = A / col_norms

            l1_weight = float(alpha * l1_ratio)
            l2_weight = float(alpha * (1.0 - l1_ratio))

            def _loss_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
                r = A_norm @ x - peak_intensities
                loss = (
                    0.5 * float(r @ r)
                    + l1_weight * float(np.sum(x))
                    + 0.5 * l2_weight * float(x @ x)
                )
                grad = A_norm.T @ r + l1_weight + l2_weight * x
                return loss, grad

            result = minimize(
                _loss_grad,
                x0=np.zeros(n_cands),
                jac=True,
                bounds=[(0.0, None)] * n_cands,
                method="L-BFGS-B",
                options={"maxiter": 2000},
            )
            sparse_c = np.asarray(result.x) / col_norms
            residual = float(np.linalg.norm(peak_intensities - A @ sparse_c))
        except Exception:
            # Fallback to standard NNLS (no sparsity regularization).
            from scipy.optimize import nnls as _nnls

            sparse_c, residual = _nnls(A, peak_intensities)
            residual = float(residual)

        return sparse_c, residual

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
            return 0.1  # Penalize — too few lines for meaningful ratio check

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
        #
        # Modified from original: blend P_cov (emissivity-weighted coverage)
        # into k_det so that elements with many weak undetected lines are
        # not excessively penalized.  P_cov weights by emissivity, so missing
        # a weak line (emissivity 1% of total) only reduces P_cov by 1%.
        if N_matched > 0:
            N_X = N_matched
            k_det_raw = k_rate * ((1.0 / N_X) * k_shift + ((N_X - 1.0) / N_X) * k_sim)
            # Blend: use geometric mean of raw k_det and P_cov to soften
            # the penalty for many unmatched weak lines
            k_det = math.sqrt(k_det_raw * max(P_cov, 0.01))
        else:
            k_det = 0.0

        # Fix 4: N_expected penalty — elements with few expected lines
        # get scaled down to prevent 2/3 matches from scoring high.
        N_penalty = min(1.0, math.sqrt(N_expected / 5.0)) if N_expected > 0 else 0.0
        k_det *= N_penalty

        P_SNR = self._compute_p_snr(intensity, peaks)

        # P_ab — crustal abundance prior
        P_ab = self._compute_P_ab(element)

        # Confidence level — paper formula (Noel et al. 2025):
        # CL = k_det × P_SNR × P_maj × P_ab
        CL = k_det * P_SNR * P_maj * P_ab

        # Hard gate — reject if too few lines matched.
        # At RP<1000, matching 2 lines by chance is trivial for elements
        # with few expected lines (Na, K). Require enough matches to be
        # statistically meaningful.
        if N_expected <= 1:
            # Single-line elements (H-alpha): 1 match is sufficient
            pass
        elif N_expected <= 4:
            # Sparse elements (Na, K, Li): require ALL lines matched
            # AND elevated CL to pass — chance matching 2/2 is too easy
            if N_matched < N_expected:
                CL = 0.0
        else:
            # Normal elements: require at least 3 matched lines
            if N_matched < 3:
                CL = 0.0

        return k_det, CL
