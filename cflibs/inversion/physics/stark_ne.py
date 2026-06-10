"""Stark-broadening electron-density diagnostic from observed line profiles.

This module turns the *measured spectrum* into an electron-density
measurement, replacing the physically-invalid 1-atm pressure-balance n_e
assumption (audit 2026-06-09, report 02 finding F2). Stark broadening of a
well-characterized, isolated emission line is the canonical n_e diagnostic in
LIBS:

* Ciucci et al. 1999 (*Appl. Spectrosc.* 53, 960) and Tognoni et al. 2010
  (*Spectrochim. Acta B* 65, 1) treat the Stark-derived n_e as an INPUT to the
  Saha terms of CF-LIBS, not a quantity iterated from closure.
* Gigosos 2014 (*J. Phys. D: Appl. Phys.* 47, 343001,
  doi:10.1088/0022-3727/47/34/343001) reviews Stark-broadening models for
  plasma diagnostics and the accuracy hierarchy of tabulated widths.
* Konjevic et al. 2002 (*J. Phys. Chem. Ref. Data* 31, 819) tabulate the
  critically-evaluated experimental widths the ``stark_b`` provenance class
  in the atomic database descends from.

Method
------
1. **Select** candidate diagnostic lines among the detected observations:
   only lines whose database Stark width carries *literature-grade*
   provenance (``lines.stark_w_source`` in :data:`LITERATURE_STARK_SOURCES`),
   with adequate SNR and no blending neighbour within the instrument
   bandwidth. Candidates are ranked by SNR x isolation, with a mild
   preference for canonical n_e diagnostics (H-alpha, Ca II H/K, Mg II
   doublet — Aragon & Aguilera 2008, *Spectrochim. Acta B* 63, 893).
2. **Fit** each candidate's observed profile with a Voigt whose Gaussian
   component is *pinned* to the known instrument + Doppler width, leaving the
   Lorentzian (Stark) FWHM free. This is more robust than measuring a raw
   FWHM and applying the Olivero-Longbothum deconvolution because the wings
   constrain the Lorentzian fraction directly.
3. **Invert** the project-wide electron-impact width law (single source of
   truth in :mod:`cflibs.radiation.stark`, FWHM at n_e = 1e17 cm^-3 /
   T = 10000 K, linear in n_e with a ``(T/T_ref)^-alpha`` factor) for a
   per-line n_e, and **combine** lines robustly (median + MAD scatter).

Van-der-Waals / resonance broadening is *not* subtracted: the forward model
(:func:`cflibs.radiation.kernels.forward_model`) includes no vdW term (the
``gamma_vdw_log`` column has no runtime consumer), so subtracting it here
would break forward/inverse symmetry; physically, the electron-impact Stark
width dominates vdW at LIBS densities >= 1e16 cm^-3 for the literature-grade
diagnostic lines (Gigosos 2014 Sec. 2). If a vdW term is ever added to the
forward kernels it must be subtracted here too.

Known limitation: hydrogen Balmer widths actually scale as ~n_e^0.7 rather
than linearly (Gigosos 2014); the database stores H-alpha under the same
linear-in-n_e convention as every other line, so this module honours that
convention (forward/inverse symmetric) and the H-alpha-specific exponent is
left for a dedicated follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cflibs.atomic.masses import resolve_element_mass
from cflibs.core.constants import EV_TO_K
from cflibs.core.logging_config import get_logger
from cflibs.inversion.common.data_structures import LineObservation
from cflibs.radiation.profiles import doppler_width, voigt_profile
from cflibs.radiation.stark import estimate_ne_from_stark

if TYPE_CHECKING:  # layering: physics/ must not import solve/ at runtime
    from cflibs.inversion.solve.iterative import StarkDiagnosticLine

logger = get_logger("inversion.stark_ne")

#: ``lines.stark_w_source`` values trusted for *measuring* n_e. Only the
#: Stark-B / critically-evaluated literature class qualifies; the
#: lambda^2-scaled (``konjevic_lambda_sq_scaled``), ``interpolated`` and
#: ``hydrogenic`` heuristics are good enough for forward broadening but not
#: for a density measurement (Konjevic 2002; Gigosos 2014).
LITERATURE_STARK_SOURCES: Tuple[str, ...] = ("stark_b",)

#: Canonical LIBS n_e diagnostic lines (nm) given a mild ranking preference:
#: H-alpha, Ca II H/K, Mg II resonance doublet (Aragon & Aguilera 2008;
#: El Sherbini et al. 2005). Selection is still data-driven (SNR x isolation).
PREFERRED_DIAGNOSTIC_LINES: Tuple[Tuple[str, int, float], ...] = (
    ("H", 1, 656.28),
    ("Ca", 2, 393.37),
    ("Ca", 2, 396.85),
    ("Mg", 2, 279.55),
    ("Mg", 2, 280.27),
)

#: Loose physical plausibility band for a per-line n_e (cm^-3). LIBS plasmas
#: in the analysis window span ~1e15-1e19; values outside indicate a failed
#: fit or a mis-attributed line, not a measurement.
NE_SANITY_MIN_CM3 = 1.0e14
NE_SANITY_MAX_CM3 = 1.0e20

_FWHM_PER_SIGMA = 2.0 * float(np.sqrt(2.0 * np.log(2.0)))  # 2.3548


@dataclass
class StarkLineMeasurement:
    """One accepted diagnostic line and its width -> n_e measurement."""

    element: str
    ionization_stage: int
    wavelength_nm: float
    stark_w_ref_nm: float
    stark_alpha: float
    stark_w_source: str
    snr: float
    isolation_nm: float
    instrument_fwhm_nm: float
    doppler_fwhm_nm: float
    lorentz_fwhm_nm: float
    fit_rel_rmse: float
    ne_cm3: float


@dataclass
class StarkNeDiagnostics:
    """Result of :func:`measure_stark_ne` — the measured-n_e trust surface."""

    measurements: List[StarkLineMeasurement] = field(default_factory=list)
    #: Solver-ready diagnostic lines (Lorentzian widths; Gaussian already
    #: removed by the pinned-sigma fit, so instrument/Doppler are 0 here).
    diagnostics: List["StarkDiagnosticLine"] = field(default_factory=list)
    ne_median_cm3: Optional[float] = None
    ne_scatter_cm3: float = 0.0
    n_lines: int = 0
    instrument_fwhm_source: str = "none"
    #: rejection-reason -> count, for the trust report / debugging.
    rejected: Dict[str, int] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.n_lines > 0 and self.ne_median_cm3 is not None


def _local_half_max_fwhm(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    center_nm: float,
    window_nm: float,
) -> Optional[float]:
    """Raw FWHM of the peak nearest ``center_nm`` via half-max crossings.

    Local baseline = the lower of the two window-edge medians. Returns ``None``
    when no usable peak is found in the window.
    """
    mask = (wavelength >= center_nm - window_nm) & (wavelength <= center_nm + window_nm)
    if int(np.count_nonzero(mask)) < 5:
        return None
    wl = wavelength[mask]
    inten = intensity[mask].astype(float)
    edge = max(3, len(inten) // 8)
    baseline = min(float(np.median(inten[:edge])), float(np.median(inten[-edge:])))
    inten = inten - baseline
    peak_idx = int(np.argmax(inten))
    peak = inten[peak_idx]
    if peak <= 0:
        return None
    half = 0.5 * peak

    def _crossing(start: int, stop: int, step: int) -> Optional[float]:
        prev = peak_idx
        for i in range(start, stop, step):
            if inten[i] <= half:
                # Linear interpolation between i and prev.
                denom = inten[prev] - inten[i]
                frac = (inten[prev] - half) / denom if denom > 0 else 0.0
                return float(wl[prev] + frac * (wl[i] - wl[prev]))
            prev = i
        return None

    left = _crossing(peak_idx - 1, -1, -1)
    right = _crossing(peak_idx + 1, len(inten), 1)
    if left is None or right is None:
        return None
    fwhm = right - left
    if not np.isfinite(fwhm) or fwhm <= 0:
        return None
    return float(fwhm)


def estimate_instrument_fwhm(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    observations: Sequence[LineObservation],
    center_nm: Optional[float] = None,
    local_window_nm: float = 50.0,
    percentile: float = 20.0,
) -> Optional[float]:
    """Data-driven instrument-FWHM floor from the narrowest observed lines.

    Every observed line width is the instrument response convolved with the
    physical (Doppler + Stark) width, so a low percentile of the raw measured
    FWHMs across many lines bounds the instrument FWHM from above. For
    moderate-resolution LIBS spectrometers (instrument FWHM >> typical
    non-hydrogenic Stark widths at <= 1e17 cm^-3) the bound is tight; at high
    resolution it *over*-estimates the instrument width, which biases the
    recovered n_e LOW — i.e. the estimator is conservative for the diagnostic.

    Stitched multi-channel spectrometers (e.g. ChemCam) have per-channel
    resolution, so when ``center_nm`` is given the floor is computed from
    lines within ``local_window_nm`` of it (falling back to the global set
    when fewer than 5 local lines are available).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if len(wavelength) < 5:
        return None
    wl_step = float(np.median(np.diff(wavelength)))

    def _collect(obs_list: Sequence[LineObservation]) -> List[float]:
        widths = []
        for obs in obs_list:
            fwhm = _local_half_max_fwhm(
                wavelength, intensity, obs.wavelength_nm, window_nm=max(20 * wl_step, 1.0)
            )
            if fwhm is not None and fwhm >= wl_step:
                widths.append(fwhm)
        return widths

    pool: Sequence[LineObservation] = observations
    if center_nm is not None:
        local = [o for o in observations if abs(o.wavelength_nm - center_nm) <= local_window_nm]
        if len(local) >= 5:
            pool = local
    widths = _collect(pool)
    if len(widths) < 3 and pool is not observations:
        widths = _collect(observations)
    if len(widths) < 3:
        return None
    return float(np.percentile(widths, percentile))


def _recenter_on_local_peak(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    center_nm: float,
    search_nm: float,
) -> float:
    """Re-centre on the local intensity maximum near ``center_nm``.

    Observations carry the DB transition wavelength; the spectrum axis can be
    offset by residual calibration error (and the line by its Stark shift),
    so the profile fit anchors on the actual local peak.
    """
    mask = (wavelength >= center_nm - search_nm) & (wavelength <= center_nm + search_nm)
    if int(np.count_nonzero(mask)) < 3:
        return center_nm
    wl = wavelength[mask]
    inten = intensity[mask]
    return float(wl[int(np.argmax(inten))])


def _fit_lorentz_fwhm(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    center_nm: float,
    gaussian_fwhm_nm: float,
    window_nm: float,
) -> Optional[Tuple[float, float]]:
    """Fit a pinned-Gaussian Voigt to the profile around ``center_nm``.

    The Gaussian sigma is fixed to the known instrument (+) Doppler width;
    free parameters are amplitude (area), centre (small shift, e.g. Stark
    shift / residual calibration), Lorentzian HWHM and a linear baseline.

    Returns ``(lorentz_fwhm_nm, relative_rms_residual)`` or ``None``.
    """
    from scipy.optimize import least_squares

    mask = (wavelength >= center_nm - window_nm) & (wavelength <= center_nm + window_nm)
    if int(np.count_nonzero(mask)) < 7:
        return None
    wl = np.asarray(wavelength[mask], dtype=float)
    inten = np.asarray(intensity[mask], dtype=float)

    sigma_g = max(gaussian_fwhm_nm / _FWHM_PER_SIGMA, 1e-6)
    edge = max(3, len(inten) // 8)
    baseline0 = min(float(np.median(inten[:edge])), float(np.median(inten[-edge:])))
    peak0 = float(np.max(inten) - baseline0)
    if peak0 <= 0:
        return None
    # Initial area guess: peak height x Gaussian width (order of magnitude).
    area0 = peak0 * gaussian_fwhm_nm
    gamma0 = 0.1 * gaussian_fwhm_nm  # Lorentzian HWHM start (small)

    def _model(params: np.ndarray) -> np.ndarray:
        area, x0, gamma, c0, c1 = params
        prof = voigt_profile(wl, x0, sigma_g, max(gamma, 0.0), amplitude=area)
        return np.asarray(prof) + c0 + c1 * (wl - center_nm)

    def _residuals(params: np.ndarray) -> np.ndarray:
        return _model(params) - inten

    x0 = np.array([area0, center_nm, gamma0, baseline0, 0.0])
    lower = [0.0, center_nm - window_nm / 2.0, 0.0, -np.inf, -np.inf]
    upper = [np.inf, center_nm + window_nm / 2.0, 10.0 * window_nm, np.inf, np.inf]
    try:
        fit = least_squares(_residuals, x0, bounds=(lower, upper), max_nfev=400)
    except Exception as exc:  # pragma: no cover - scipy internal failure
        logger.debug("Voigt fit failed at %.3f nm: %r", center_nm, exc)
        return None
    if not fit.success and fit.status <= 0:
        return None
    area, _, gamma, _, _ = fit.x
    if area <= 0 or not np.isfinite(gamma):
        return None
    scale = float(np.max(inten) - np.min(inten))
    rel_rmse = float(np.sqrt(np.mean(fit.fun**2)) / scale) if scale > 0 else 1.0
    return float(2.0 * gamma), rel_rmse


def _preference_factor(element: str, stage: int, wavelength_nm: float) -> float:
    """Mild multiplicative ranking bonus for canonical n_e diagnostics."""
    for el, sp, wl in PREFERRED_DIAGNOSTIC_LINES:
        if element == el and stage == sp and abs(wavelength_nm - wl) < 0.3:
            return 2.0
    return 1.0


def _isolation_nm(obs: LineObservation, observations: Sequence[LineObservation]) -> float:
    """Distance (nm) to the nearest *other* observed line."""
    dists = [
        abs(other.wavelength_nm - obs.wavelength_nm)
        for other in observations
        if other is not obs
    ]
    return float(min(dists)) if dists else np.inf


def measure_stark_ne(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    observations: Sequence[LineObservation],
    atomic_db,
    *,
    resolving_power: Optional[float] = None,
    instrument_fwhm_nm: Optional[float] = None,
    T_K: float = 10000.0,
    max_lines: int = 5,
    min_snr: float = 5.0,
    isolation_factor: float = 1.5,
    max_fit_rel_rmse: float = 0.25,
    allowed_sources: Tuple[str, ...] = LITERATURE_STARK_SOURCES,
    wavelength_tolerance_nm: float = 0.1,
) -> StarkNeDiagnostics:
    """Measure n_e from the Stark widths of literature-grade observed lines.

    Parameters
    ----------
    wavelength, intensity : np.ndarray
        The measured spectrum (same axes the observations were detected on).
    observations : Sequence[LineObservation]
        Detected + selected line observations (the solver's input list).
    atomic_db : AtomicDatabase
        Database with ``stark_w`` / ``stark_w_source`` columns.
    resolving_power : float, optional
        Instrument resolving power; instrument FWHM = lambda / R. Preferred
        over the data-driven floor when available.
    instrument_fwhm_nm : float, optional
        Explicit instrument FWHM (overrides everything).
    T_K : float
        Temperature estimate for the Doppler width and the weak
        ``(T/T_ref)^alpha`` Stark correction (default 10000 K, the solver's
        initialisation; the solver refines the T factor per iteration).
    max_lines : int
        Maximum diagnostic lines to fit (ranked by SNR x isolation).
    min_snr : float
        Minimum line SNR (intensity / intensity_uncertainty).
    isolation_factor : float
        Required clear distance to the nearest other observed line, in units
        of the local Gaussian (instrument+Doppler) FWHM.
    max_fit_rel_rmse : float
        Reject fits whose residual RMS exceeds this fraction of the local
        peak amplitude.
    allowed_sources : tuple of str
        ``stark_w_source`` values accepted as literature-grade.
    wavelength_tolerance_nm : float
        DB matching tolerance when looking up Stark parameters.

    Returns
    -------
    StarkNeDiagnostics
        Per-line measurements, solver-ready diagnostic lines, and the robust
        combination (median + 1.4826*MAD scatter). When no line qualifies the
        result has ``usable == False`` and the caller must keep the existing
        fallback path (with its warning) unchanged.
    """
    # Local import: keeps the physics/ layer free of a hard runtime
    # dependency on solve/ (only this constructor needs the dataclass).
    from cflibs.inversion.solve.iterative import StarkDiagnosticLine

    wavelength = np.asarray(wavelength, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    out = StarkNeDiagnostics()

    def _reject(reason: str) -> None:
        out.rejected[reason] = out.rejected.get(reason, 0) + 1

    if len(wavelength) < 5 or not observations:
        return out

    T_eV = max(T_K, 1000.0) / EV_TO_K

    # --- instrument width resolution ladder -------------------------------
    if instrument_fwhm_nm is not None and instrument_fwhm_nm > 0:
        out.instrument_fwhm_source = "explicit"
    elif resolving_power is not None and resolving_power > 0:
        out.instrument_fwhm_source = "resolving_power"
    else:
        out.instrument_fwhm_source = "narrowest_line_floor"

    def _instr_fwhm(center_nm: float) -> Optional[float]:
        if instrument_fwhm_nm is not None and instrument_fwhm_nm > 0:
            return float(instrument_fwhm_nm)
        if resolving_power is not None and resolving_power > 0:
            return center_nm / float(resolving_power)
        return estimate_instrument_fwhm(wavelength, intensity, observations, center_nm=center_nm)

    # --- candidate selection ----------------------------------------------
    candidates = []
    for obs in observations:
        w_ref, alpha, source = atomic_db.get_stark_parameters_with_source(
            obs.element, obs.ionization_stage, obs.wavelength_nm, wavelength_tolerance_nm
        )
        if source not in allowed_sources or w_ref is None or w_ref <= 0:
            _reject("not_literature_grade")
            continue
        snr = (
            obs.intensity / obs.intensity_uncertainty
            if obs.intensity_uncertainty and obs.intensity_uncertainty > 0
            else np.inf
        )
        if snr < min_snr:
            _reject("low_snr")
            continue
        instr = _instr_fwhm(obs.wavelength_nm)
        if instr is None or instr <= 0:
            _reject("no_instrument_width")
            continue
        mass = resolve_element_mass(obs.element, atomic_db)
        dopp = doppler_width(obs.wavelength_nm, T_eV, mass)
        gauss = float(np.hypot(instr, dopp))
        iso = _isolation_nm(obs, observations)
        if iso < isolation_factor * gauss:
            _reject("blended")
            continue
        score = (min(snr, 1e6)) * min(iso, 10.0 * gauss)
        score *= _preference_factor(obs.element, obs.ionization_stage, obs.wavelength_nm)
        candidates.append((score, obs, w_ref, alpha if alpha is not None else 0.5, source, snr, iso, instr, dopp, gauss))

    candidates.sort(key=lambda c: -c[0])

    # --- per-line Voigt fit + width -> n_e inversion -----------------------
    ne_values: List[float] = []
    for score, obs, w_ref, alpha, source, snr, iso, instr, dopp, gauss in candidates:
        if len(out.measurements) >= max_lines:
            break
        # Window: wide enough for the Lorentzian wings, capped at half the
        # distance to the nearest neighbour so blends stay out of the fit.
        window = min(max(4.0 * gauss, 0.3), max(iso / 2.0, 2.0 * gauss))
        center = _recenter_on_local_peak(
            wavelength, intensity, obs.wavelength_nm, search_nm=max(0.5 * gauss, 0.15)
        )
        fit = _fit_lorentz_fwhm(wavelength, intensity, center, gauss, window)
        if fit is None:
            _reject("fit_failed")
            continue
        lorentz_fwhm, rel_rmse = fit
        if rel_rmse > max_fit_rel_rmse:
            _reject("poor_fit")
            continue
        # Resolvability floor: a Lorentzian narrower than ~5% of the pinned
        # Gaussian is indistinguishable from zero at realistic SNR.
        if lorentz_fwhm < 0.05 * gauss:
            _reject("unresolved")
            continue
        # The Gaussian was pinned in the fit, so the Lorentzian width feeds the
        # inversion directly (no further deconvolution).
        ne = estimate_ne_from_stark(
            measured_fwhm_nm=lorentz_fwhm,
            T_K=T_K,
            stark_w_ref=w_ref,
            stark_alpha=alpha,
        )
        if ne is None or not (NE_SANITY_MIN_CM3 <= ne <= NE_SANITY_MAX_CM3):
            _reject("implausible_ne")
            continue
        out.measurements.append(
            StarkLineMeasurement(
                element=obs.element,
                ionization_stage=obs.ionization_stage,
                wavelength_nm=obs.wavelength_nm,
                stark_w_ref_nm=w_ref,
                stark_alpha=alpha,
                stark_w_source=source,
                snr=float(snr),
                isolation_nm=float(iso),
                instrument_fwhm_nm=float(instr),
                doppler_fwhm_nm=float(dopp),
                lorentz_fwhm_nm=float(lorentz_fwhm),
                fit_rel_rmse=float(rel_rmse),
                ne_cm3=float(ne),
            )
        )
        out.diagnostics.append(
            StarkDiagnosticLine(
                measured_fwhm_nm=float(lorentz_fwhm),
                stark_w_ref_nm=float(w_ref),
                stark_alpha=float(alpha),
                instrument_fwhm_nm=0.0,  # already removed by the pinned fit
                doppler_fwhm_nm=0.0,
                wavelength_nm=float(obs.wavelength_nm),
            )
        )
        ne_values.append(float(ne))

    if ne_values:
        arr = np.asarray(ne_values, dtype=float)
        out.ne_median_cm3 = float(np.median(arr))
        out.n_lines = len(ne_values)
        if len(arr) >= 2:
            mad = float(np.median(np.abs(arr - np.median(arr))))
            out.ne_scatter_cm3 = 1.4826 * mad if mad > 0 else float(np.std(arr))
        logger.info(
            "Stark n_e diagnostic: %d literature-grade line(s) -> n_e = %.3e cm^-3 "
            "(scatter %.2e), lines: %s",
            out.n_lines,
            out.ne_median_cm3,
            out.ne_scatter_cm3,
            ", ".join(
                f"{m.element} {'I' if m.ionization_stage == 1 else 'II'} "
                f"{m.wavelength_nm:.2f} (ne={m.ne_cm3:.2e})"
                for m in out.measurements
            ),
        )
    else:
        logger.info(
            "Stark n_e diagnostic: no qualifying literature-grade line "
            "(rejections: %s); the solver will use its fallback n_e path.",
            dict(out.rejected) or "none",
        )
    return out
