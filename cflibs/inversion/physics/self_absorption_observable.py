"""
Observable-gated self-absorption correction (bead CF-LIBS-improved-0jvr).

This module replaces the composition-feedback self-absorption correction that
the 2026-06-09 overhaul audit condemned (02-inversion-solver.md, finding F4):
the old path computed the line-center optical depth from the *recovered*
composition with no observable gate, which is a positive feedback loop
(over-attributed Fe -> bigger tau -> bigger intensity boost -> bigger
intercept q_Fe -> closure gives Fe even more mass). Empirically it made
intercept inflation WORSE on real ChemCam BHVO-2.

The literature requirement (audit 06-literature-sota.md, Q2) is that every
per-line correction factor must be anchored to an **observable** of the
measured spectrum, never only to the current composition iterate:

* Bulajic et al. 2002, *Spectrochim. Acta B* 57, 339
  (doi:10.1016/S0584-8547(01)00398-6) — curve-of-growth constrained by
  measured line profiles (~10x CF-LIBS accuracy gain when SA dominates).
* Sun & Yu 2009, *Talanta* 79, 388 (doi:10.1016/j.talanta.2009.03.066) —
  internal-reference (IRSAC) anchored to observed high-E_k lines.
* El Sherbini et al. 2005, *Spectrochim. Acta B* 60, 1573
  (doi:10.1016/j.sab.2005.10.011) — SA coefficient from the measured/Stark
  width ratio.
* Völker & Gornushkin 2023, *J. Anal. At. Spectrom.* 38, 1098
  (doi:10.1039/D2JA00352J) — closed-form Planck-function correction
  (see :func:`planck_ceiling_optical_depth` below).
* Pace et al. 2025, *Spectrochim. Acta B*
  (doi:10.1016/j.sab.2025.107361, PII S0584854725001995) — doublet
  intensity-ratio method (T-independent, single spectrum).

Correction ladder (per line, in order of preference)
----------------------------------------------------
(a) **Doublet/multiplet intensity ratio** — when a measured pair sharing the
    same upper level exists, the deviation of the measured intensity ratio
    from the optically-thin emission ratio ``(g_k A_ki / lambda)_1 /
    (g_k A_ki / lambda)_2`` yields the optical depth of both lines directly
    (no plasma state needed at all). Wires the machinery landed by bead
    CF-LIBS-improved-1fcg
    (:func:`cflibs.inversion.physics.self_absorption.correct_via_doublet_ratio`).
(b) **Planck-ceiling closed form** (Völker & Gornushkin 2023) — when the
    measured *peak spectral radiance* of the line is available in absolute
    units and a temperature estimate exists, the line-center optical depth
    follows from the homogeneous-slab radiative-transfer solution
    ``I_lambda = B_lambda (1 - exp(-tau_lambda))`` (their eq. 1). Valid for
    ``tau <= ~3`` (their stated 10% RSD budget).
(c) **No correction + SA-suspect flag** — lines with no usable observable
    that match the published SA-risk signature (resonance / low-E_i lower
    level, Fayyaz et al. 2023 threshold E_i < 6000 cm^-1 ~ 0.74 eV, AND high
    intensity within their element) are *down-weighted* in the Boltzmann fit
    by inflating their intensity uncertainty — consistent with the existing
    inverse-variance ``boltzmann_weight_cap`` machinery — never silently
    boosted.

The correction is applied to the observed line list ONCE, BEFORE the
Boltzmann / Saha-Boltzmann-graph fit (the audit shows the old per-iteration
post-hoc placement was part of the feedback problem). Nothing in this module
reads a recovered composition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from cflibs.core.constants import C_LIGHT, H_PLANCK, KB
from cflibs.core.logging_config import get_logger
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.physics.self_absorption import (
    DoubletRatioResult,
    correct_via_doublet_ratio,
    find_doublet_pairs,
)
from cflibs.inversion.physics.self_absorption_inputs import lower_level_energy_ev

logger = get_logger("inversion.self_absorption_observable")

#: Völker & Gornushkin (2023, JAAS, doi:10.1039/D2JA00352J) validity ceiling:
#: "if 10% RSD in the corrected integral line intensity is required, then the
#: allowable maximum optical thickness at the center of the line is increased
#: to about 3." Above this the closed-form correction is unreliable.
PLANCK_TAU_VALIDITY_MAX = 3.0

#: Doublet-ratio validity ceiling. The Pace et al. (2025) recovery was
#: validated for tau in [0.1, 5] (bead CF-LIBS-improved-1fcg acceptance);
#: beyond tau ~ 5 the escape-factor model itself loses validity (El Sherbini
#: 2005) and the implied >5x boost is no longer trustworthy.
DOUBLET_TAU_VALIDITY_MAX = 5.0

#: SA-risk lower-level energy threshold: Fayyaz et al. 2023 (*Metals*, Cu-Fe
#: alloys) drop lines with E_i < 6000 cm^-1 (~0.74 eV) as SA-prone
#: (audit 06-literature-sota.md, Q7).
SUSPECT_E_I_MAX_EV = 0.74


def planck_spectral_radiance(wavelength_nm: float, temperature_K: float) -> float:
    """Planck spectral radiance ``B_lambda(T)`` in W m^-2 nm^-1 sr^-1.

    Same formulation as the Wave-1 slab radiative-transfer path in
    :func:`cflibs.radiation.kernels._apply_self_absorption` so that
    intensities corrected here are consistent with spectra synthesised
    there.
    """
    if wavelength_nm <= 0 or temperature_K <= 0:
        return 0.0
    wl_m = wavelength_nm * 1.0e-9
    exponent = (H_PLANCK * C_LIGHT) / (wl_m * KB * temperature_K)
    exponent = min(exponent, 700.0)
    b_per_m = (2.0 * H_PLANCK * C_LIGHT**2 / wl_m**5) / math.expm1(exponent)
    return b_per_m * 1.0e-9  # per-m -> per-nm


def planck_ceiling_optical_depth(
    peak_spectral_radiance: float,
    wavelength_nm: float,
    temperature_K: float,
) -> Optional[float]:
    """Line-center optical depth from the measured peak vs the Planck ceiling.

    Völker & Gornushkin 2023 (*J. Anal. At. Spectrom.*,
    doi:10.1039/D2JA00352J) solve the homogeneous, isothermal LTE slab
    radiative-transfer equation (their eq. 1)

        I_lambda = B_lambda(T) * (1 - exp(-tau_lambda))

    for the optically-thin radiance (their eq. 4):

        I_thin_lambda = -B_lambda * ln(1 - I_lambda / B_lambda)

    Evaluated at line center this gives the line-center optical depth
    directly from two observables — the measured peak spectral radiance and
    the Planck function at the measured temperature:

        tau_0 = -ln(1 - I_peak / B_lambda(T))

    Validity: the correction degrades as the line saturates against the
    Planck ceiling; Völker & Gornushkin state a maximum line-center optical
    thickness of about 3 for a 10% RSD budget on the corrected integral
    intensity (:data:`PLANCK_TAU_VALIDITY_MAX`). Callers must treat a
    returned ``tau > 3`` (or ``None``) as "no usable correction".

    Parameters
    ----------
    peak_spectral_radiance : float
        Measured continuum-subtracted peak spectral radiance of the line in
        the SAME absolute units as ``B_lambda`` (W m^-2 nm^-1 sr^-1). This
        requires a radiometrically calibrated spectrum.
    wavelength_nm : float
        Line-center wavelength.
    temperature_K : float
        Plasma temperature estimate (Völker & Gornushkin iterate this; they
        report T to 0.4% and n_e to 0.7% relative error after four
        iterations on synthetic alloy spectra).

    Returns
    -------
    float or None
        ``tau_0 >= 0``, or ``None`` when the measurement is at/above the
        Planck ceiling (``I_peak >= B_lambda``, unphysical for a homogeneous
        LTE slab — usually a units or temperature error).
    """
    if peak_spectral_radiance <= 0:
        return 0.0
    b_lambda = planck_spectral_radiance(wavelength_nm, temperature_K)
    if b_lambda <= 0:
        return None
    ratio = peak_spectral_radiance / b_lambda
    if ratio >= 1.0:
        return None
    return -math.log1p(-ratio)


def doppler_cog_escape_factor(tau_0: float, n_terms: int = 64) -> float:
    """Profile-integrated escape factor for a Doppler (Gaussian) line.

    Ratio of the wavelength-integrated slab intensity to the integrated
    optically-thin intensity for a Gaussian optical-depth profile
    ``tau(x) = tau_0 exp(-x^2)``:

        f_G(tau_0) = (1 / (sqrt(pi) tau_0)) * Int [1 - exp(-tau_0 e^{-x^2})] dx
                   = sum_{n>=1} (-1)^{n+1} tau_0^{n-1} / (n! sqrt(n))

    This is the exact Doppler curve-of-growth attenuation (Mihalas,
    *Stellar Atmospheres*, ch. 9) — slightly milder than the line-center
    escape factor ``(1 - e^-tau)/tau`` because the line wings stay thin.
    Used by the Planck-ceiling correction so that intensities synthesised
    through the slab radiative-transfer kernel
    (:func:`cflibs.radiation.kernels._apply_self_absorption`) are recovered
    without overshoot. The alternating series converges rapidly for the
    validity range ``tau_0 <= 3``.
    """
    if tau_0 < 1.0e-10:
        return 1.0
    total = 0.0
    term = 1.0  # tau_0^{n-1} / n! at n=1
    for n in range(1, n_terms + 1):
        total += ((-1.0) ** (n + 1)) * term / math.sqrt(n)
        term *= tau_0 / (n + 1)
    return max(total, 1.0e-12)


@dataclass
class PlanckCorrection:
    """Result of the Planck-ceiling correction for one line."""

    tau_0: float
    correction_factor: float  # I_thin = I_obs * correction_factor (>= 1)
    corrected_intensity: float
    valid: bool


def correct_intensity_planck(
    integrated_intensity: float,
    peak_spectral_radiance: float,
    wavelength_nm: float,
    temperature_K: float,
    tau_max: float = PLANCK_TAU_VALIDITY_MAX,
) -> PlanckCorrection:
    """Correct an integrated line intensity via the Planck-ceiling method.

    Combines the Völker & Gornushkin (2023) line-center optical depth
    (:func:`planck_ceiling_optical_depth`) with the Doppler
    profile-integrated escape factor (:func:`doppler_cog_escape_factor`)
    to recover the optically-thin *integrated* intensity:

        I_thin = I_obs / f_G(tau_0)

    ``valid=False`` (and ``correction_factor=1``) when ``tau_0`` is
    undeterminable or exceeds ``tau_max`` (default
    :data:`PLANCK_TAU_VALIDITY_MAX`) — the published validity gate.
    """
    tau_0 = planck_ceiling_optical_depth(peak_spectral_radiance, wavelength_nm, temperature_K)
    if tau_0 is None or tau_0 > tau_max:
        return PlanckCorrection(
            tau_0=float("inf") if tau_0 is None else tau_0,
            correction_factor=1.0,
            corrected_intensity=integrated_intensity,
            valid=False,
        )
    f_g = doppler_cog_escape_factor(tau_0)
    factor = 1.0 / f_g
    return PlanckCorrection(
        tau_0=tau_0,
        correction_factor=factor,
        corrected_intensity=integrated_intensity * factor,
        valid=True,
    )


@dataclass
class ObservableLineCorrection:
    """Per-line outcome of the observable-gated correction.

    ``method`` is one of:

    * ``"doublet"`` — corrected via the doublet intensity ratio.
    * ``"doublet-thin"`` — a doublet observable existed and showed the line
      optically thin (no correction needed; cleared of suspicion).
    * ``"planck"`` — corrected via the Planck-ceiling closed form.
    * ``"suspect"`` — no usable observable; line matches the SA-risk
      signature and was down-weighted (uncertainty inflated), NOT boosted.
    * ``"none"`` — no observable and no SA-risk signature; untouched.
    """

    wavelength_nm: float
    element: str
    method: str
    tau: float
    correction_factor: float
    suspect: bool


@dataclass
class ObservableSAResult:
    """Result of :meth:`ObservableSelfAbsorptionCorrector.correct`."""

    observations: List[LineObservation]
    corrections: Dict[float, ObservableLineCorrection]
    n_corrected: int
    n_suspect: int
    max_tau: float
    warnings: List[str] = field(default_factory=list)


class ObservableSelfAbsorptionCorrector:
    """Observable-gated self-absorption correction of measured line lists.

    See the module docstring for the correction ladder and literature
    grounding. The corrector never reads a composition estimate; every
    correction factor derives from measured quantities (line intensity
    ratios, or peak radiance vs the Planck ceiling), and lines without a
    usable observable are at most *down-weighted*, never boosted.

    Parameters
    ----------
    doublet_tau_max : float
        Validity ceiling for doublet-derived optical depths
        (default :data:`DOUBLET_TAU_VALIDITY_MAX`). Pairs implying a larger
        tau are not corrected; both lines are flagged SA-suspect instead.
    planck_tau_max : float
        Validity ceiling for the Planck-ceiling correction
        (default :data:`PLANCK_TAU_VALIDITY_MAX`, Völker & Gornushkin 2023).
    min_ratio_significance_sigma : float
        Minimum significance (in propagated sigmas — intensity noise AND
        fractional A_ki uncertainty when available) of the measured
        doublet-ratio deviation from the optically-thin ratio before a
        correction is applied. Below it the pair is treated as observably
        thin (a noisy ratio must not manufacture optical depth).
    min_ratio_deviation : float
        Absolute floor on the fractional ratio deviation
        ``|1 - r_meas/r_thin|`` (default 0.10). Lines frequently carry no
        intensity uncertainty (sigma = 0 would make ANY deviation
        infinitely significant) and NIST grade-B transition probabilities
        are only accurate to ~10%, so deviations below this floor cannot be
        attributed to self-absorption rather than atomic-data error.
    suspect_E_i_max_ev : float
        Lower-level energy below which a line is SA-prone (default
        :data:`SUSPECT_E_I_MAX_EV`, the Fayyaz 2023 6000 cm^-1 cut).
    suspect_intensity_factor : float
        A low-E_i line is only flagged when its intensity is at least this
        multiple of its element's median line intensity (the bright
        resonance line is the SA risk, not every low-E_i line).
    suspect_uncertainty_inflation : float
        Factor by which a suspect line's intensity uncertainty is inflated.
        Weight in the inverse-variance Boltzmann fit drops by the square
        (default 3.0 -> w/9), consistent with the existing
        ``boltzmann_weight_cap`` weighting machinery.
    """

    def __init__(
        self,
        doublet_tau_max: float = DOUBLET_TAU_VALIDITY_MAX,
        planck_tau_max: float = PLANCK_TAU_VALIDITY_MAX,
        min_ratio_significance_sigma: float = 1.0,
        min_ratio_deviation: float = 0.10,
        suspect_E_i_max_ev: float = SUSPECT_E_I_MAX_EV,
        suspect_intensity_factor: float = 1.0,
        suspect_uncertainty_inflation: float = 3.0,
    ):
        self.doublet_tau_max = float(doublet_tau_max)
        self.planck_tau_max = float(planck_tau_max)
        self.min_ratio_significance_sigma = float(min_ratio_significance_sigma)
        self.min_ratio_deviation = float(min_ratio_deviation)
        self.suspect_E_i_max_ev = float(suspect_E_i_max_ev)
        self.suspect_intensity_factor = float(suspect_intensity_factor)
        self.suspect_uncertainty_inflation = float(suspect_uncertainty_inflation)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def correct(
        self,
        observations: Sequence[LineObservation],
        temperature_K: Optional[float] = None,
        peak_spectral_radiance: Optional[Dict[float, float]] = None,
    ) -> ObservableSAResult:
        """Apply the observable-gated correction ladder to a line list.

        Parameters
        ----------
        observations : sequence of LineObservation
            Measured lines (integrated intensities). Order is preserved in
            the returned list.
        temperature_K : float, optional
            Temperature estimate enabling the Planck-ceiling path (b).
            Without it (the common case before any fit), only the
            T-independent doublet path (a) and the suspect flagging (c) run.
        peak_spectral_radiance : dict, optional
            ``{wavelength_nm: peak spectral radiance}`` in absolute units
            (W m^-2 nm^-1 sr^-1) for lines measured on a radiometrically
            calibrated spectrum. Required for path (b).
        """
        obs_list = list(observations)
        corrections: Dict[float, ObservableLineCorrection] = {}
        warnings: List[str] = []
        corrected: Dict[int, LineObservation] = {}
        cleared: set = set()  # indices observably thin (exempt from suspicion)
        suspect_forced: set = set()  # indices forced suspect by an observable
        index_of = {id(o): i for i, o in enumerate(obs_list)}

        # --- (a) doublet / multiplet intensity ratios -------------------
        self._apply_doublet_pass(
            obs_list, index_of, corrections, corrected, cleared, suspect_forced, warnings
        )

        # --- (b) Planck-ceiling closed form ------------------------------
        if temperature_K is not None and peak_spectral_radiance:
            self._apply_planck_pass(
                obs_list,
                temperature_K,
                peak_spectral_radiance,
                corrections,
                corrected,
                cleared,
                warnings,
            )

        # --- (c) SA-suspect flagging + down-weighting --------------------
        n_suspect = self._apply_suspect_pass(
            obs_list, corrections, corrected, cleared, suspect_forced
        )

        out = [corrected.get(i, o) for i, o in enumerate(obs_list)]
        n_corrected = sum(
            1 for c in corrections.values() if not c.suspect and c.correction_factor > 1.0
        )
        finite_taus = [c.tau for c in corrections.values() if np.isfinite(c.tau)]
        max_tau = max(finite_taus) if finite_taus else 0.0

        logger.info(
            "observable self-absorption: n_lines=%d corrected=%d suspect=%d max_tau=%.3f",
            len(obs_list),
            n_corrected,
            n_suspect,
            max_tau,
        )
        return ObservableSAResult(
            observations=out,
            corrections=corrections,
            n_corrected=n_corrected,
            n_suspect=n_suspect,
            max_tau=max_tau,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # pass (a): doublets
    # ------------------------------------------------------------------

    def _ratio_deviation_significant(
        self, line1: LineObservation, line2: LineObservation, res: DoubletRatioResult
    ) -> bool:
        """Is the measured ratio deviation attributable to self-absorption?

        Two gates, both required:

        1. an absolute floor ``min_ratio_deviation`` on the fractional
           deviation ``|1 - r_meas/r_thin|`` (uncalibrated noise sources and
           ~10% grade-B A_ki accuracy put a hard floor on what a ratio can
           prove);
        2. a significance threshold in propagated sigmas, combining the
           relative intensity uncertainties of both lines with their
           fractional A_ki uncertainties when the database supplies them.

        SA shifts the measured ratio away from the thin ratio in either
        direction depending on which member of the pair is more absorbed
        (tau_2 = tau_1 / rho can exceed tau_1), so the test is two-sided.
        """
        if line1.intensity <= 0 or line2.intensity <= 0:
            return False
        deviation = abs(1.0 - res.r_measured / res.r_theory)
        if deviation < self.min_ratio_deviation:
            return False
        rel1 = line1.intensity_uncertainty / line1.intensity
        rel2 = line2.intensity_uncertainty / line2.intensity
        aki1 = line1.aki_uncertainty or 0.0
        aki2 = line2.aki_uncertainty or 0.0
        sigma_rel = math.sqrt(rel1**2 + rel2**2 + aki1**2 + aki2**2)
        if sigma_rel <= 0:
            return True  # floor already passed; no noise model to test against
        return deviation / sigma_rel >= self.min_ratio_significance_sigma

    def _apply_doublet_pass(
        self,
        obs_list: List[LineObservation],
        index_of: Dict[int, int],
        corrections: Dict[float, ObservableLineCorrection],
        corrected: Dict[int, LineObservation],
        cleared: set,
        suspect_forced: set,
        warnings: List[str],
    ) -> None:
        """Correct every usable doublet pair in place (ladder step (a))."""
        for line1, line2 in find_doublet_pairs(obs_list):
            i1 = index_of[id(line1)]
            i2 = index_of[id(line2)]
            if i1 in corrected or i2 in corrected or i1 in cleared or i2 in cleared:
                continue  # first usable pair wins per line
            try:
                res = correct_via_doublet_ratio(line1, line2)
            except ValueError as exc:
                warnings.append(
                    f"doublet ({line1.wavelength_nm:.3f}, {line2.wavelength_nm:.3f}) nm "
                    f"skipped: {exc}"
                )
                continue

            significant = self._ratio_deviation_significant(line1, line2, res)
            tau_pair_max = max(res.tau_1, res.tau_2)
            if tau_pair_max <= 1e-3 or not significant:
                # Observably thin (or deviation within noise): clear both.
                for idx, line, tau in ((i1, line1, res.tau_1), (i2, line2, res.tau_2)):
                    cleared.add(idx)
                    corrections[line.wavelength_nm] = ObservableLineCorrection(
                        wavelength_nm=line.wavelength_nm,
                        element=line.element,
                        method="doublet-thin",
                        tau=min(tau, 1e-3),
                        correction_factor=1.0,
                        suspect=False,
                    )
                continue

            if tau_pair_max > self.doublet_tau_max:
                # Observable says VERY thick — beyond the validated recovery
                # range. Do not boost; force-flag both lines as SA-suspect.
                warnings.append(
                    f"doublet ({line1.wavelength_nm:.3f}, {line2.wavelength_nm:.3f}) nm: "
                    f"tau={tau_pair_max:.2f} > {self.doublet_tau_max} validity ceiling; "
                    "flagging pair SA-suspect instead of correcting"
                )
                for idx, line, tau in ((i1, line1, res.tau_1), (i2, line2, res.tau_2)):
                    suspect_forced.add(idx)
                    corrections[line.wavelength_nm] = ObservableLineCorrection(
                        wavelength_nm=line.wavelength_nm,
                        element=line.element,
                        method="suspect",
                        tau=tau,
                        correction_factor=1.0,
                        suspect=True,
                    )
                continue

            for idx, line, tau, f_tau, i_corr in (
                (i1, line1, res.tau_1, res.f_tau_1, res.i1_corrected),
                (i2, line2, res.tau_2, res.f_tau_2, res.i2_corrected),
            ):
                factor = 1.0 / f_tau if f_tau > 0 else 1.0
                corrected[idx] = replace(
                    line,
                    intensity=i_corr,
                    intensity_uncertainty=line.intensity_uncertainty * factor,
                )
                cleared.add(idx)
                corrections[line.wavelength_nm] = ObservableLineCorrection(
                    wavelength_nm=line.wavelength_nm,
                    element=line.element,
                    method="doublet",
                    tau=tau,
                    correction_factor=factor,
                    suspect=False,
                )

    # ------------------------------------------------------------------
    # pass (b): Planck ceiling
    # ------------------------------------------------------------------

    def _apply_planck_pass(
        self,
        obs_list: List[LineObservation],
        temperature_K: float,
        peak_spectral_radiance: Dict[float, float],
        corrections: Dict[float, ObservableLineCorrection],
        corrected: Dict[int, LineObservation],
        cleared: set,
        warnings: List[str],
    ) -> None:
        """Planck-ceiling correction for lines with calibrated peak radiance."""
        for i, obs in enumerate(obs_list):
            if i in corrected or i in cleared:
                continue
            peak = peak_spectral_radiance.get(obs.wavelength_nm)
            if peak is None:
                continue
            pc = correct_intensity_planck(
                obs.intensity, peak, obs.wavelength_nm, temperature_K, self.planck_tau_max
            )
            if not pc.valid:
                warnings.append(
                    f"Planck-ceiling correction invalid at {obs.wavelength_nm:.3f} nm "
                    f"(tau_0={pc.tau_0:.2f} beyond validity); leaving uncorrected"
                )
                continue
            corrected[i] = replace(
                obs,
                intensity=pc.corrected_intensity,
                intensity_uncertainty=obs.intensity_uncertainty * pc.correction_factor,
            )
            cleared.add(i)
            corrections[obs.wavelength_nm] = ObservableLineCorrection(
                wavelength_nm=obs.wavelength_nm,
                element=obs.element,
                method="planck",
                tau=pc.tau_0,
                correction_factor=pc.correction_factor,
                suspect=False,
            )

    # ------------------------------------------------------------------
    # pass (c): suspects
    # ------------------------------------------------------------------

    def _is_sa_risk(self, obs: LineObservation, median_intensity: float) -> bool:
        """Published SA-risk signature: low-E_i lower level AND bright."""
        e_i = lower_level_energy_ev(obs)
        if e_i > self.suspect_E_i_max_ev:
            return False
        return obs.intensity >= self.suspect_intensity_factor * median_intensity

    def _down_weighted(self, obs: LineObservation) -> LineObservation:
        """Copy of ``obs`` with its fit weight reduced by ~inflation^2."""
        k = self.suspect_uncertainty_inflation
        if obs.intensity_uncertainty > 0:
            new_unc = obs.intensity_uncertainty * k
        else:
            # sigma=0 lines get unit weight in the fit; give the suspect a
            # fractional sigma of k so its weight lands at 1/k^2.
            new_unc = obs.intensity * k
        return replace(obs, intensity_uncertainty=new_unc)

    def _apply_suspect_pass(
        self,
        obs_list: List[LineObservation],
        corrections: Dict[float, ObservableLineCorrection],
        corrected: Dict[int, LineObservation],
        cleared: set,
        suspect_forced: set,
    ) -> int:
        """Down-weight SA-risk lines without observables (ladder step (c))."""
        median_by_element: Dict[str, float] = {}
        for el in {o.element for o in obs_list}:
            intensities = [o.intensity for o in obs_list if o.element == el and o.intensity > 0]
            median_by_element[el] = float(np.median(intensities)) if intensities else 0.0

        n_suspect = 0
        for i, obs in enumerate(obs_list):
            if i in cleared or i in corrected:
                continue
            forced = i in suspect_forced
            if not forced and not self._is_sa_risk(obs, median_by_element[obs.element]):
                continue
            corrected[i] = self._down_weighted(obs)
            n_suspect += 1
            existing = corrections.get(obs.wavelength_nm)
            if existing is not None and forced:
                continue  # keep the doublet-derived tau record
            corrections[obs.wavelength_nm] = ObservableLineCorrection(
                wavelength_nm=obs.wavelength_nm,
                element=obs.element,
                method="suspect",
                tau=float("nan"),
                correction_factor=1.0,
                suspect=True,
            )
        return n_suspect


@dataclass
class ThickLineMask:
    """Observable-anchored optically-thick line mask over a wavelength grid.

    Produced by :func:`build_observed_thick_line_mask` for the raw-spectrum
    (full-spectrum / joint) fit path, which — unlike the iterative Boltzmann
    path — has no line list to hand to
    :class:`ObservableSelfAbsorptionCorrector` and no saturation term in its
    optically-thin forward. The mask carries the set of wavelength windows the
    corrector's *observable* diagnostics (doublet intensity ratios; optional
    Planck-ceiling COG) flagged as optically thick, so the caller can exclude
    them from the residual on BOTH the observed and model side.

    Attributes
    ----------
    keep : np.ndarray or None
        Boolean array over the wavelength grid (``True`` = pixel retained in
        the fit). ``None`` means *no line was flagged optically thick* — the
        caller MUST then skip masking entirely so a thin spectrum reproduces
        the un-masked fit bit-for-bit (the audit's required no-op).
    flagged_wavelengths : list of float
        Line-center wavelengths (nm) flagged optically thick by an observable.
    max_tau : float
        Largest observable line-center optical depth among flagged lines.
    n_lines_measured : int
        Number of catalog lines with a measurable peak that were handed to the
        observable corrector.
    n_flagged : int
        Number of lines flagged optically thick.
    corrections : dict
        The corrector's per-line :class:`ObservableLineCorrection` records.
    warnings : list of str
        Non-fatal notes (e.g. mask disabled because it would remove too much).
    """

    keep: Optional[np.ndarray]
    flagged_wavelengths: List[float]
    max_tau: float
    n_lines_measured: int
    n_flagged: int
    corrections: Dict[float, ObservableLineCorrection]
    warnings: List[str] = field(default_factory=list)


#: numpy>=2.0 renamed ``trapz`` to ``trapezoid``; keep both spellings working.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


def _robust_noise_sigma(intensity: np.ndarray) -> float:
    """Robust per-pixel noise sigma via the DER_SNR second-difference estimator.

    ``sigma = 1.482602/sqrt(6) * median(|2 I_i - I_{i-2} - I_{i+2}|)`` (Stoehr
    et al. 2008, DER_SNR). The symmetric second difference cancels any locally
    linear signal and is insensitive to smooth curvature to first order, so a
    clean (noiseless) spectrum — even one full of sharp lines — returns ~0
    because the median is dominated by the line-free pixels. That is what lets
    the doublet significance test fall through to its atomic-data floor on
    synthetic data instead of manufacturing a huge noise budget from line
    slopes (the failure mode of a first-difference estimator). Returns ``0.0``
    for a spectrum too short to form the second difference.
    """
    arr = np.asarray(intensity, dtype=np.float64)
    if arr.size < 5:
        return 0.0
    second_diff = np.abs(2.0 * arr[2:-2] - arr[:-4] - arr[4:])
    return float(1.482602 / math.sqrt(6.0) * np.median(second_diff))


def build_observed_thick_line_mask(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    *,
    line_wavelengths_nm: Sequence[float],
    line_elements: Sequence[str],
    line_ion_stages: Sequence[int],
    line_E_k_ev: Sequence[float],
    line_g_k: Sequence[float],
    line_A_ki: Sequence[float],
    mask_wavelength_grid: Optional[np.ndarray] = None,
    corrector: Optional["ObservableSelfAbsorptionCorrector"] = None,
    temperature_K: Optional[float] = None,
    peak_spectral_radiance: Optional[Dict[float, float]] = None,
    line_sigma_nm: Optional[Sequence[float]] = None,
    measure_half_width_nm: float = 0.2,
    mask_half_width_nm: float = 0.3,
    mask_tau_min: float = 0.5,
    min_peak_snr: float = 4.0,
    baseline_percentile: float = 10.0,
    isolation_margin: float = 1.5,
    max_pair_wavelength_ratio: float = 1.4,
    max_lines: int = 400,
    max_mask_fraction: float = 0.5,
) -> ThickLineMask:
    """Flag optically-thick line windows in a raw spectrum from observables.

    This is the raw-spectrum companion to
    :meth:`ObservableSelfAbsorptionCorrector.correct` (which needs a fitted
    line list). It (1) measures each in-band catalog line's integrated
    intensity from the *observed* spectrum, (2) hands the resulting
    :class:`LineObservation` list to the SAME observable corrector, and
    (3) turns every line the corrector flagged optically thick — via a
    *measured* observable (doublet intensity ratio, or Planck-ceiling COG when
    a calibrated peak radiance is supplied) — into an exclusion window.

    Design constraints honoured (physics-first-principles audit, Issue 3,
    "Corrected scope"): the flag is anchored ONLY to spectrum observables, never
    to a composition-derived optical depth (the audited-harmful F4 feedback
    loop), and no new fitted optical-depth degree of freedom is introduced. A
    line whose *signature* looks SA-prone but has no usable observable
    (``method == "suspect"`` with ``tau`` NaN) is deliberately NOT masked — that
    is a risk heuristic, not a measurement, and masking on it would remove thin
    resonance lines and break the thin-spectrum no-op.

    Parameters
    ----------
    wavelength, intensity : ndarray
        Observed spectrum used for line MEASUREMENT (nm; arbitrary but
        self-consistent radiometric units — the doublet-ratio observable is
        intensity-scale-independent). Pass the NATIVE, high-resolution spectrum
        here: a coarsely-resampled grid under-samples narrow lines and corrupts
        the integrated-intensity ratio.
    line_wavelengths_nm, line_elements, line_ion_stages, line_E_k_ev, \
    line_g_k, line_A_ki : sequences
        Per-line catalog metadata (typically unpacked from an
        :class:`~cflibs.core.jax_runtime.AtomicSnapshot`).
    mask_wavelength_grid : ndarray, optional
        Wavelength grid the returned ``keep`` mask is defined on (e.g. the
        coarser fit grid). Defaults to ``wavelength`` (measure and mask on the
        same grid). Flagged line centers are grid-independent, so measurement
        can run on the native grid while the mask lands on the fit grid.
    corrector : ObservableSelfAbsorptionCorrector, optional
        Reuses the caller's configured corrector; a default one is built when
        omitted.
    temperature_K, peak_spectral_radiance : optional
        Enable the Planck-ceiling observable (path (b)); require a
        radiometrically calibrated spectrum, so they are ``None`` on the
        area-normalised full-spectrum path and only the T-/scale-independent
        doublet observable runs.
    measure_half_width_nm : float
        Half-width (nm) of the window used to integrate each line's intensity.
    line_sigma_nm : sequence, optional
        Per-line expected Gaussian width (nm) at each catalog wavelength (e.g.
        the instrument sigma). When supplied, each line's measurement window is
        ``max(6*sigma_i, measure_half_width_nm/2)`` — a wavelength-scaled window
        that captures the full profile at both the narrow (blue) and broad (red)
        ends of a wide spectrum, instead of one fixed window that mis-measures
        one end. Falls back to the scalar ``measure_half_width_nm`` when omitted.
    mask_half_width_nm : float
        Half-width (nm) of the excluded window centred on each flagged line, on
        the ``mask_wavelength_grid`` (typically the coarse fit grid).
    mask_tau_min : float
        Minimum observable line-center optical depth for a line to be masked
        (a tiny measured tau is not worth excluding).
    min_peak_snr : float
        A catalog line is only measured when its peak clears this multiple of
        the estimated noise sigma (skips the thousands of absent catalog lines).
    isolation_margin : float
        A measured line is only kept when its measurement window does not
        overlap another detected line's window within this margin: line ``i`` is
        dropped if a detected neighbour ``j`` has
        ``|lambda_i - lambda_j| < isolation_margin * (hw_i + hw_j)``. A blended
        line's windowed intensity is contaminated, so its doublet ratio is not a
        clean observable. This non-overlap isolation is what keeps a genuinely
        thin (but dense) spectrum a no-op.
    max_pair_wavelength_ratio : float
        Only trust a doublet correction whose two shared-upper-level members lie
        within this wavelength ratio (default 1.4). A wide (e.g. UV↔IR) branching
        pair has an extreme optical-depth ratio ``rho = (lambda_1/lambda_2)^3``
        and its thin ratio hinges on cross-band A_ki self-consistency and equal
        cross-band radiometric response — neither holds on real data — so such
        pairs manufacture spurious optical depth. Restricting to moderate
        separations keeps the mask a reliable no-op on thin data at the cost of
        not seeing resonance lines whose only branching partner is far away
        (a documented coverage limit of the observable-only approach).
    max_lines : int
        Cap on the strongest measured lines handed to the corrector (bounds the
        O(n^2) doublet scan on dense real spectra).
    max_mask_fraction : float
        Safety valve: if the mask would remove more than this fraction of
        pixels the mask is disabled (``keep=None``) and a warning recorded, so a
        pathological detection can never gut the fit.

    Returns
    -------
    ThickLineMask
    """
    wl = np.asarray(wavelength, dtype=np.float64)
    obs = np.asarray(intensity, dtype=np.float64)
    lam = np.asarray(line_wavelengths_nm, dtype=np.float64)
    empty = ThickLineMask(None, [], 0.0, 0, 0, {}, [])
    if wl.size == 0 or obs.size == 0 or lam.size == 0 or wl.size != obs.size:
        return empty

    ek = np.asarray(line_E_k_ev, dtype=np.float64)
    gk = np.asarray(line_g_k, dtype=np.float64)
    aki = np.asarray(line_A_ki, dtype=np.float64)
    n_cat = lam.size
    if not (len(line_elements) == len(line_ion_stages) == n_cat == ek.size == gk.size == aki.size):
        return empty

    wl_lo, wl_hi = float(np.min(wl)), float(np.max(wl))
    sigma_noise = _robust_noise_sigma(obs)

    # Per-line measurement half-widths: wavelength-scaled (6 sigma) when a
    # per-line width is supplied, else the scalar floor. A single fixed window
    # over a wide spectrum mis-measures one end (clips broad red lines / catches
    # neighbours at the narrow blue end).
    sigma_floor = 0.5 * measure_half_width_nm
    if line_sigma_nm is not None:
        sig = np.asarray(line_sigma_nm, dtype=np.float64)
        if sig.size != n_cat:
            sig = np.full(n_cat, sigma_floor / 6.0)
        hw_cat = np.maximum(6.0 * sig, sigma_floor)
    else:
        hw_cat = np.full(n_cat, measure_half_width_nm)

    # --- (1) measure each in-band catalog line from the observed spectrum ----
    detected: List[tuple] = []  # (idx, wl, integrated, unc, hw)
    for i in range(n_cat):
        c = float(lam[i])
        if c < wl_lo or c > wl_hi or aki[i] <= 0.0 or gk[i] <= 0.0:
            continue
        hw = float(hw_cat[i])
        sel = np.abs(wl - c) <= hw
        if not np.any(sel):
            continue
        wseg = wl[sel]
        seg = obs[sel]
        srt = np.argsort(wseg)
        wseg = wseg[srt]
        seg = seg[srt]
        baseline = float(np.percentile(seg, baseline_percentile))
        above = np.clip(seg - baseline, 0.0, None)
        peak = float(np.max(above)) if above.size else 0.0
        if peak <= 0.0:
            continue
        if sigma_noise > 0.0 and peak < min_peak_snr * sigma_noise:
            continue
        integrated = float(_trapezoid(above, wseg)) if wseg.size > 1 else peak
        if integrated <= 0.0:
            continue
        mean_dwl = float(np.mean(np.diff(wseg))) if wseg.size > 1 else 0.0
        unc = sigma_noise * math.sqrt(float(wseg.size)) * mean_dwl
        detected.append((i, c, integrated, unc, hw))

    if not detected:
        return empty

    # Non-overlap isolation among the DETECTED lines only: line i is contaminated
    # if a detected neighbour j has |lambda_i - lambda_j| < margin*(hw_i + hw_j),
    # i.e. their measurement windows overlap. Isolation is computed against the
    # measured (above-SNR) set — not the full catalog — so a strong isolated line
    # is not blocked by a weak/absent catalog neighbour. This non-overlap check
    # is what keeps a dense-but-thin spectrum a no-op while still flagging genuine
    # strong self-absorbers.
    det_wl = np.array([d[1] for d in detected], dtype=np.float64)
    det_hw = np.array([d[4] for d in detected], dtype=np.float64)
    d_order = np.argsort(det_wl)
    isolated = np.ones(len(detected), dtype=bool)
    for a in range(d_order.size):
        ia = d_order[a]
        if a > 0:
            ib = d_order[a - 1]
            if det_wl[ia] - det_wl[ib] < isolation_margin * (det_hw[ia] + det_hw[ib]):
                isolated[ia] = False
        if a < d_order.size - 1:
            ic = d_order[a + 1]
            if det_wl[ic] - det_wl[ia] < isolation_margin * (det_hw[ia] + det_hw[ic]):
                isolated[ia] = False

    measured: List[tuple] = [
        (idx, integ, unc) for k, (idx, _c, integ, unc, _hw) in enumerate(detected) if isolated[k]
    ]
    if not measured:
        return empty

    # Keep the strongest lines (self-absorption is a strong-line effect) and
    # bound the O(n^2) doublet scan on dense real catalogs.
    measured.sort(key=lambda t: t[1], reverse=True)
    if len(measured) > max_lines:
        measured = measured[:max_lines]

    observations = [
        LineObservation(
            wavelength_nm=float(lam[i]),
            intensity=integ,
            intensity_uncertainty=unc,
            element=str(line_elements[i]),
            ionization_stage=int(line_ion_stages[i]),
            E_k_ev=float(ek[i]),
            g_k=float(gk[i]),
            A_ki=float(aki[i]),
            aki_uncertainty=None,
        )
        for (i, integ, unc) in measured
    ]

    # --- (2) run the SAME observable corrector -------------------------------
    if corrector is None:
        corrector = ObservableSelfAbsorptionCorrector()
    result = corrector.correct(
        observations,
        temperature_K=temperature_K,
        peak_spectral_radiance=peak_spectral_radiance,
    )

    # Wavelength centers of doublet members within the trustworthy separation
    # ratio (cross-band UV/IR pairs are atomic-data/response fragile — excluded).
    trustworthy_doublet: set = set()
    for a, b in find_doublet_pairs(observations):
        lo, hi = sorted((a.wavelength_nm, b.wavelength_nm))
        if lo > 0 and hi / lo <= max_pair_wavelength_ratio:
            trustworthy_doublet.add(a.wavelength_nm)
            trustworthy_doublet.add(b.wavelength_nm)

    # --- (3) build the exclusion mask from OBSERVABLE thick flags only -------
    # A line is masked iff the corrector *solved* a self-absorption correction
    # for it — ``method in {doublet, planck}`` with a bounded, in-validity-range
    # optical depth at or above ``mask_tau_min``. Deliberately EXCLUDED:
    #   * ``method == "suspect"`` — either a signature-only risk heuristic
    #     (tau NaN) or a doublet-forced suspect whose partner-implied
    #     ``tau_2 = tau_1/rho`` is out of the validated range (and can blow up
    #     for very unequal line strengths); neither is a trustworthy per-line
    #     measurement to mask on.
    #   * ``method == "doublet-thin"`` (tau <= 1e-3) — observably thin.
    # Restricting to solved doublet/planck corrections both bounds tau to the
    # published validity ceiling and preserves the thin-spectrum no-op.
    mask_grid = wl if mask_wavelength_grid is None else np.asarray(mask_wavelength_grid, np.float64)
    keep = np.ones(mask_grid.shape, dtype=bool)
    flagged: List[float] = []
    max_tau = 0.0
    for w, corr in result.corrections.items():
        trustworthy = corr.method == "planck" or w in trustworthy_doublet
        if (
            corr.method in ("doublet", "planck")
            and trustworthy
            and corr.correction_factor > 1.0
            and np.isfinite(corr.tau)
            and corr.tau >= mask_tau_min
        ):
            flagged.append(float(w))
            max_tau = max(max_tau, float(corr.tau))
            keep &= np.abs(mask_grid - float(w)) > mask_half_width_nm

    if not flagged:
        return ThickLineMask(
            None, [], 0.0, len(observations), 0, result.corrections, list(result.warnings)
        )

    masked_fraction = 1.0 - float(np.count_nonzero(keep)) / float(keep.size)
    warnings = list(result.warnings)
    if masked_fraction > max_mask_fraction:
        warnings.append(
            f"observable thick-line mask would remove {masked_fraction:.0%} of pixels "
            f"(> {max_mask_fraction:.0%} guard); disabling mask (fit stays optically thin)"
        )
        return ThickLineMask(
            None, flagged, max_tau, len(observations), len(flagged), result.corrections, warnings
        )

    logger.info(
        "observable thick-line mask: measured=%d flagged=%d max_tau=%.3f masked_pixels=%.1f%%",
        len(observations),
        len(flagged),
        max_tau,
        100.0 * masked_fraction,
    )
    return ThickLineMask(
        keep, flagged, max_tau, len(observations), len(flagged), result.corrections, warnings
    )


def normalize_self_absorption_mode(value) -> str:
    """Normalize the ``apply_self_absorption`` knob to ``'off'|'observable'``.

    Accepts the legacy booleans (``False`` -> ``'off'``, ``True`` ->
    ``'observable'``) and the explicit mode strings. The old
    composition-feedback mode was deleted (audit 02-F4) and has no spelling.
    """
    if value is None or value is False:
        return "off"
    if value is True:
        return "observable"
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in ("off", "observable"):
            return mode
    raise ValueError(
        f"Invalid apply_self_absorption mode {value!r}: expected False/'off', " "True/'observable'."
    )


__all__ = [
    "DOUBLET_TAU_VALIDITY_MAX",
    "PLANCK_TAU_VALIDITY_MAX",
    "SUSPECT_E_I_MAX_EV",
    "ObservableLineCorrection",
    "ObservableSAResult",
    "ObservableSelfAbsorptionCorrector",
    "PlanckCorrection",
    "ThickLineMask",
    "build_observed_thick_line_mask",
    "correct_intensity_planck",
    "doppler_cog_escape_factor",
    "normalize_self_absorption_mode",
    "planck_ceiling_optical_depth",
    "planck_spectral_radiance",
]
