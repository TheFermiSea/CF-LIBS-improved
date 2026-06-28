"""Constrained known-feedstock force-extraction (opt-in, physics-only).

The DED real goal is *drift tracking on a known matrix*: the feedstock element
set (e.g. {Ti, Al, V} for Ti-6Al-4V, or a steel alloy's majors) is known a
priori. In that regime generic peak detection + element identification is not
just unnecessary, it is harmful: on a dense dominant-matrix forest the minor
element's analytical lines are frequently below the generic detector's gate even
when they are well above SNR (measured: generic detection drops Al in ~44% of
real SuperCam Ti-6Al-4V spectra), so the minor element silently vanishes from
the closure and the composition collapses.

This module promotes the validated constrained force-extraction (the
``--forced`` path of ``scripts/benchmark_pds_opc.py`` Part 3, which took the
real-Mars-SuperCam Ti-6Al-4V OPC held-out RMSEP from 4.88 wt% with generic
detection to 0.35 wt% averaged / 0.65 wt% per-shot) into the shipped pipeline as
an opt-in mode. Given a KNOWN element set + a measured spectrum it:

1. builds a known-position candidate line list straight from the atomic DB
   (emissivity ranking, strongest cleanest isolated lines, resonance KEPT,
   detector-gap lines dropped) -- :func:`build_constrained_line_list`;
2. **peak-locks** onto the local peak within a small search tolerance of each
   catalog wavelength (handles the per-spectrometer wavelength offset real
   instruments carry) and integrates the line over a local-baseline window,
   bypassing generic peak detection -- :func:`extract_peak_locked`.

It is pure atomic-data + signal-processing NumPy (no measured-intensity-driven
detection, no recovered composition, no banned ML API); the default pipeline
path never calls it (``AnalysisPipelineConfig.constrained_extraction`` defaults
``False``), so the calibration-free path is byte-identical.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.inversion.common import LineObservation
from cflibs.inversion.physics.line_selection import SelectedLine, _select_emissivity

logger = get_logger("inversion.constrained_extraction")

#: SuperCam 3-spectrometer detector gaps (nm): the UV|VIO and VIO|VNIR
#: boundaries. Lines falling in a gap are physically unmeasurable, so the forced
#: known-line list drops them. Measured on the SuperCam SCCT data (341.4-379.3
#: and 464.5-537.6 nm; Manrique et al. 2020). Pass as ``detector_gaps`` for a
#: SuperCam-class instrument; ``None`` (default) applies no gap filtering.
SUPERCAM_DETECTOR_GAPS: Tuple[Tuple[float, float], ...] = ((341.4, 379.3), (464.5, 537.6))

#: Default emissivity-ranking temperature (K) for the constrained line list.
#: A fixed, physically reasonable alloy-plasma value; never fit per sample.
DEFAULT_CONSTRAINED_SELECT_T_K: float = 10000.0
#: Default per-element line budget when an element is absent from ``budget``.
DEFAULT_CONSTRAINED_BUDGET: int = 8
#: FWHM -> Gaussian sigma.
_FWHM_TO_SIGMA: float = 1.0 / 2.354820045

# NumPy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def in_detector_gap(
    wavelength_nm: float, detector_gaps: Optional[Sequence[Tuple[float, float]]]
) -> bool:
    """True when ``wavelength_nm`` falls inside any ``(lo, hi)`` detector gap."""
    if not detector_gaps:
        return False
    return any(lo <= wavelength_nm <= hi for lo, hi in detector_gaps)


def build_constrained_line_list(
    db,
    elements: Sequence[str],
    window: Tuple[float, float],
    *,
    budget: Optional[Dict[str, int]] = None,
    default_budget: int = DEFAULT_CONSTRAINED_BUDGET,
    select_temperature_K: float = DEFAULT_CONSTRAINED_SELECT_T_K,
    stages: Sequence[int] = (1, 2),
    min_separation_nm: float = 0.12,
    detector_gaps: Optional[Sequence[Tuple[float, float]]] = None,
    exclude_resonance: bool = False,
) -> List[SelectedLine]:
    """Curated KNOWN-position candidate line list for constrained force-extraction.

    For each known element, rank in-band DB transitions by Boltzmann-weighted
    emissivity at ``select_temperature_K`` and take the strongest, cleanest,
    isolated lines (``prefer_spread=False``: with the joint Saha-Boltzmann graph
    T is constrained across all elements together, so forcing per-element E_k
    spread only admits weak, blended high-E_k lines that corrupt dense-spectrum
    elements). Resonance lines are KEPT by default (a minor element's only strong
    in-band lines are often its resonance doublet, e.g. Al 394.4/396.15 nm), and
    lines that fall inside a ``detector_gaps`` spectrometer gap are dropped.

    The line POSITIONS and atomic parameters come from the atomic DB and the
    KNOWN element set only -- never from a truth composition -- so this encodes
    only the legitimate DED known-feedstock assumption (honest).

    Parameters
    ----------
    db
        Atomic database exposing ``get_transitions(element, ionization_stage,
        wavelength_min, wavelength_max)``.
    elements : sequence of str
        The KNOWN element set (the feedstock species).
    window : tuple of float
        ``(wavelength_min_nm, wavelength_max_nm)`` selection window.
    budget : dict, optional
        Per-element maximum line count. Elements absent from the dict use
        ``default_budget``. ``None`` uses ``default_budget`` for every element.
    default_budget : int, default 8
        Per-element line budget when ``budget`` does not name the element.
    select_temperature_K : float, default ``DEFAULT_CONSTRAINED_SELECT_T_K``
        Boltzmann-weighting temperature for the emissivity ranking.
    stages : sequence of int, default (1, 2)
        Ionization stages searched (neutral + singly ionized).
    min_separation_nm : float, default 0.12
        Minimum wavelength separation between selected lines (isolation).
    detector_gaps : sequence of (lo, hi), optional
        Spectrometer detector gaps (nm); selected lines inside a gap are
        dropped. ``None`` (default) applies no gap filtering. Use
        :data:`SUPERCAM_DETECTOR_GAPS` for a SuperCam-class instrument.
    exclude_resonance : bool, default False
        Exclude resonance lines. Default ``False`` keeps them (see above).

    Returns
    -------
    list of SelectedLine
        The known-position candidate lines, ``budget``-capped per element.
    """
    specs: List[SelectedLine] = []
    for element in elements:
        n = (budget or {}).get(element, default_budget)
        if n <= 0:
            continue
        # Over-select (n*4) then drop detector-gap lines and cap at n, so a gap
        # does not starve an element of its budget (matches the validated lever).
        cand = _select_emissivity(
            db,
            element,
            window,
            n * 4,
            stages,
            select_temperature_K,
            min_separation_nm,
            exclude_resonance,
            prefer_spread=False,
        )
        cand = [s for s in cand if not in_detector_gap(s.wavelength_nm, detector_gaps)]
        specs.extend(cand[:n])
    return specs


def extract_peak_locked(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    line_specs: Sequence[SelectedLine],
    *,
    instrument_fwhm_nm: float = 0.18,
    search_tol_nm: float = 0.35,
) -> List[LineObservation]:
    """Peak-locked windowed extraction at known line positions -> observations.

    For each known line, lock onto the local peak within +/-``search_tol_nm`` of
    the catalog wavelength (handles the spectrometer-dependent wavelength offset
    real instruments carry, e.g. ~+0.15 nm in one SuperCam channel, ~-0.3 nm in
    another), then integrate over +/-1.5 sigma after a robust local linear
    baseline. This GUARANTEES every known line is measured -- including the
    dense-matrix-blended minor-element lines that generic peak detection drops --
    so a known feedstock element never silently vanishes from the closure.

    Parameters
    ----------
    wavelength, intensity : np.ndarray
        Measured spectrum axes (nm, intensity).
    line_specs : sequence of SelectedLine
        Known-position candidate lines (from :func:`build_constrained_line_list`).
    instrument_fwhm_nm : float, default 0.18
        Instrument line FWHM (nm); sets the integration half-window (1.5 sigma).
    search_tol_nm : float, default 0.35
        Half-width (nm) of the peak-lock search window around each catalog
        wavelength (the maximum per-spectrometer offset to tolerate).

    Returns
    -------
    list of LineObservation
        One observation per successfully measured known line (lines with no
        finite positive integrated area are skipped).
    """
    wl = np.asarray(wavelength, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    if wl.size < 5:
        return []
    step = float(np.median(np.diff(wl)))
    sigma = instrument_fwhm_nm * _FWHM_TO_SIGMA
    half = max(1.5 * sigma, 3.0 * step)
    obs: List[LineObservation] = []
    for ls in line_specs:
        sm = (wl >= ls.wavelength_nm - search_tol_nm) & (wl <= ls.wavelength_nm + search_tol_nm)
        if int(sm.sum()) < 5:
            continue
        xs, ys = wl[sm], inten[sm]
        k = max(1, xs.size // 6)
        base = np.interp(xs, [xs[0], xs[-1]], [np.median(ys[:k]), np.median(ys[-k:])])
        peak_wl = float(xs[int(np.argmax(ys - base))])
        im = (wl >= peak_wl - half) & (wl <= peak_wl + half)
        if int(im.sum()) < 4:
            continue
        x, y = wl[im], inten[im].astype(float)
        k2 = max(1, x.size // 6)
        b = np.interp(x, [x[0], x[-1]], [np.median(y[:k2]), np.median(y[-k2:])])
        area = float(_trapz(y - b, x))
        if not np.isfinite(area) or area <= 0.0:
            continue
        obs.append(
            LineObservation(
                wavelength_nm=peak_wl,
                intensity=area,
                intensity_uncertainty=max(area * 0.05, 1e-12),
                element=ls.element,
                ionization_stage=ls.ionization_stage,
                E_k_ev=ls.E_k_ev,
                g_k=ls.g_k,
                A_ki=ls.A_ki,
                aki_uncertainty=ls.aki_uncertainty,
            )
        )
    return obs


def constrained_extract(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    db,
    elements: Sequence[str],
    *,
    window: Optional[Tuple[float, float]] = None,
    budget: Optional[Dict[str, int]] = None,
    default_budget: int = DEFAULT_CONSTRAINED_BUDGET,
    select_temperature_K: float = DEFAULT_CONSTRAINED_SELECT_T_K,
    stages: Sequence[int] = (1, 2),
    min_separation_nm: float = 0.12,
    detector_gaps: Optional[Sequence[Tuple[float, float]]] = None,
    exclude_resonance: bool = False,
    instrument_fwhm_nm: float = 0.18,
    search_tol_nm: float = 0.35,
    return_specs: bool = False,
):
    """Constrained known-feedstock extraction: line list + peak-locked integrate.

    The single shipped entry point for the constrained mode: build the
    known-position line list (:func:`build_constrained_line_list`) over the
    KNOWN ``elements`` and peak-lock + integrate at each position
    (:func:`extract_peak_locked`), bypassing generic peak detection and element
    identification entirely. Pure atomic-data + signal processing; never reads a
    recovered composition.

    Parameters
    ----------
    wavelength, intensity : np.ndarray
        Measured spectrum axes (nm, intensity).
    db
        Atomic database (see :func:`build_constrained_line_list`).
    elements : sequence of str
        The KNOWN feedstock element set.
    window : tuple of float, optional
        Selection window (nm). Defaults to the spectrum's ``(min, max)``.
    budget, default_budget, select_temperature_K, stages, min_separation_nm, \
detector_gaps, exclude_resonance
        Forwarded to :func:`build_constrained_line_list`.
    instrument_fwhm_nm, search_tol_nm
        Forwarded to :func:`extract_peak_locked`.
    return_specs : bool, default False
        When True, return ``(observations, line_specs)`` instead of just the
        observations (useful for diagnostics / tests).

    Returns
    -------
    list of LineObservation, or (list, list of SelectedLine)
        The peak-locked observations (and, with ``return_specs``, the candidate
        line list they were extracted from).
    """
    wl = np.asarray(wavelength, dtype=float)
    if window is None:
        if wl.size == 0:
            window = (0.0, 0.0)
        else:
            window = (float(np.nanmin(wl)), float(np.nanmax(wl)))
    specs = build_constrained_line_list(
        db,
        elements,
        window,
        budget=budget,
        default_budget=default_budget,
        select_temperature_K=select_temperature_K,
        stages=stages,
        min_separation_nm=min_separation_nm,
        detector_gaps=detector_gaps,
        exclude_resonance=exclude_resonance,
    )
    observations = extract_peak_locked(
        wavelength,
        intensity,
        specs,
        instrument_fwhm_nm=instrument_fwhm_nm,
        search_tol_nm=search_tol_nm,
    )
    logger.info(
        "Constrained extraction over %d known element(s): %d candidate line(s) -> "
        "%d peak-locked observation(s).",
        len(list(elements)),
        len(specs),
        len(observations),
    )
    if return_specs:
        return observations, specs
    return observations


__all__ = [
    "SUPERCAM_DETECTOR_GAPS",
    "DEFAULT_CONSTRAINED_SELECT_T_K",
    "DEFAULT_CONSTRAINED_BUDGET",
    "in_detector_gap",
    "build_constrained_line_list",
    "extract_peak_locked",
    "constrained_extract",
]
