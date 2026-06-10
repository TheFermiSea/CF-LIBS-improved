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
  (see :func:`planck_thin_spectral_radiance` below).
* Pace et al. 2025, *Spectrochim. Acta B*
  (doi:10.1016/j.sab.2025.107361, PII S0584854725001995) — doublet
  intensity-ratio method (T-independent, single spectrum).

Correction ladder (per line, in order of preference)
----------------------------------------------------
(a) **Doublet/multiplet intensity ratio** — when a measured pair sharing the
    same upper level exists, the deviation of the measured intensity ratio
    from the optically-thin ``g_k A_ki lambda^3`` ratio yields the optical
    depth of both lines directly (no plasma state needed at all). Wires the
    machinery landed by bead CF-LIBS-improved-1fcg
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
        Minimum significance (in propagated intensity-uncertainty sigmas) of
        the measured doublet-ratio deviation from the optically-thin ratio
        before a correction is applied. Below it the pair is treated as
        observably thin (a noisy ratio must not manufacture optical depth).
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
        suspect_E_i_max_ev: float = SUSPECT_E_I_MAX_EV,
        suspect_intensity_factor: float = 1.0,
        suspect_uncertainty_inflation: float = 3.0,
    ):
        self.doublet_tau_max = float(doublet_tau_max)
        self.planck_tau_max = float(planck_tau_max)
        self.min_ratio_significance_sigma = float(min_ratio_significance_sigma)
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

    def _ratio_significance_sigma(
        self, line1: LineObservation, line2: LineObservation, res: DoubletRatioResult
    ) -> float:
        """Significance of the ratio deviation in propagated-noise sigmas."""
        if line1.intensity <= 0 or line2.intensity <= 0:
            return 0.0
        rel1 = line1.intensity_uncertainty / line1.intensity
        rel2 = line2.intensity_uncertainty / line2.intensity
        sigma_rel = math.sqrt(rel1**2 + rel2**2)
        deviation = 1.0 - res.r_measured / res.r_theory
        if sigma_rel <= 0:
            return float("inf") if deviation > 0 else 0.0
        return deviation / sigma_rel

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

            significance = self._ratio_significance_sigma(line1, line2, res)
            if res.tau_1 <= 1e-3 or significance < self.min_ratio_significance_sigma:
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

            if res.tau_1 > self.doublet_tau_max:
                # Observable says VERY thick — beyond the validated recovery
                # range. Do not boost; force-flag both lines as SA-suspect.
                warnings.append(
                    f"doublet ({line1.wavelength_nm:.3f}, {line2.wavelength_nm:.3f}) nm: "
                    f"tau_1={res.tau_1:.2f} > {self.doublet_tau_max} validity ceiling; "
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
    "correct_intensity_planck",
    "doppler_cog_escape_factor",
    "normalize_self_absorption_mode",
    "planck_ceiling_optical_depth",
    "planck_spectral_radiance",
]
