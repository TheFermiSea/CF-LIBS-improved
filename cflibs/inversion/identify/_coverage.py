"""Detection-coverage transparency helpers for CF-LIBS identifiers.

These helpers expose *why* an element is missed by the identifier
pipeline at four distinct layers:

- **L1 Peak detection** -- how many peaks were detected, and where, on
  the raw spectrum (regardless of any element-level reasoning).
- **L2 Per-element line presence in DB** -- how many of an element's
  theoretical lines fall in the spectrum's wavelength range.  Zero
  means the element *cannot* be identified, period -- the catalogue
  simply has nothing in window.
- **L3 Per-element peak match** -- how many of an element's
  in-window theoretical lines paired with any detected peak inside
  the matcher's tolerance.  Zero here while L2 > 0 is the *smoking
  gun*: the lines exist, peaks exist, but no pair survived
  tolerance / shift / Stark filtering.
- **L4 Per-element fingerprint pass** -- which elements survived the
  identifier's fingerprint / score floor and reached
  ``detected=True``.  L3 > 0 but L4 fails means the element matched
  some peaks but not strongly enough to clear the floor.

The helpers are *additive*: they only build telemetry, they never
mutate identifier state.  They are wired into the identifier
``identify()`` entry points (and into :func:`detect_peaks_auto` for
L1) and surface the four counters in ``result.parameters`` via
:func:`merge_coverage_into_parameters` so that the dogfood loop and
``id_summary.json`` preserve them for analysis.

Per task requirement: every log record carries ``spectrum_id`` and
``identifier_name`` so downstream parsers can correlate L1/L2/L3/L4
across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.detection_coverage")

# Sentinel for "no spectrum id was provided"; using a string keeps log
# output stable and grep-friendly.
_NO_SPECTRUM_ID = "<unset>"


def _summarize_peak_wavelengths(peaks: Sequence[Tuple[int, float]]) -> Dict[str, float]:
    """Return min/max/p25/p50/p75 of the detected peak wavelengths.

    Returns NaNs when the peak list is empty so the schema of the
    output dict is stable across calls.
    """
    if not peaks:
        nan = float("nan")
        return {"min_nm": nan, "max_nm": nan, "p25_nm": nan, "p50_nm": nan, "p75_nm": nan}

    wls = np.asarray([float(p[1]) for p in peaks], dtype=float)
    p25, p50, p75 = np.percentile(wls, [25.0, 50.0, 75.0])
    return {
        "min_nm": float(wls.min()),
        "max_nm": float(wls.max()),
        "p25_nm": float(p25),
        "p50_nm": float(p50),
        "p75_nm": float(p75),
    }


def log_peak_detection(
    peaks: Sequence[Tuple[int, float]],
    noise: float,
    baseline_method: Any,
    *,
    spectrum_id: Optional[str] = None,
    identifier_name: str = "<peak_detector>",
) -> Dict[str, Any]:
    """Emit the L1 peak-detection log record and return its payload.

    Parameters
    ----------
    peaks
        Output of :func:`cflibs.inversion.preprocess.preprocessing.detect_peaks`
        (list of ``(index, wavelength_nm)`` tuples).
    noise
        Estimated noise level :math:`\\sigma`.
    baseline_method
        The concrete baseline method used (e.g. ``BaselineMethod.MEDIAN``).
        Accepted as ``Any`` so the helper does not have to import the
        ``BaselineMethod`` enum.
    spectrum_id
        Optional identifier for this spectrum.  Threaded through to
        every log record so downstream parsing can correlate L1/L2/L3/L4.
    identifier_name
        Name of the caller (e.g. ``"alias"``).  Defaults to a generic
        label when invoked directly from the peak detector.

    Returns
    -------
    Dict[str, Any]
        Structured payload with keys ``n_peaks``, ``noise``,
        ``baseline_method``, and the wavelength-summary fields from
        :func:`_summarize_peak_wavelengths`.  Suitable for merging into
        ``result.parameters`` via
        :func:`merge_coverage_into_parameters`.
    """
    summary = _summarize_peak_wavelengths(peaks)
    sid = spectrum_id if spectrum_id is not None else _NO_SPECTRUM_ID
    payload: Dict[str, Any] = {
        "n_peaks": int(len(peaks)),
        "noise": float(noise) if np.isfinite(noise) else float("nan"),
        "baseline_method": (
            getattr(baseline_method, "value", None)
            or getattr(baseline_method, "name", None)
            or str(baseline_method)
        ),
        **summary,
    }
    logger.info(
        "L1 peak_detection spectrum_id=%s identifier=%s n_peaks=%d "
        "baseline_method=%s noise=%.3g wl_min=%.3f wl_max=%.3f "
        "wl_p25=%.3f wl_p50=%.3f wl_p75=%.3f",
        sid,
        identifier_name,
        payload["n_peaks"],
        payload["baseline_method"],
        payload["noise"],
        payload["min_nm"],
        payload["max_nm"],
        payload["p25_nm"],
        payload["p50_nm"],
        payload["p75_nm"],
    )
    return payload


@dataclass
class CoverageTracker:
    """Per-spectrum L2/L3/L4 coverage telemetry collector.

    Logging is additive -- the tracker never changes identifier
    behaviour.  Identifiers call ``record_db_lines`` after looking up
    an element's transitions in window (L2), ``record_peak_matches``
    after running the matcher (L3), and ``record_fingerprint`` after
    the fingerprint / detection-floor decision (L4).  The final
    payload built by :meth:`build_payload` is written to
    ``result.parameters`` so id_summary.json preserves it.
    """

    spectrum_id: str = _NO_SPECTRUM_ID
    identifier_name: str = "<unknown>"
    n_peaks_detected: int = 0
    elements_with_zero_db_lines_in_range: List[str] = field(default_factory=list)
    elements_with_zero_peak_matches: List[str] = field(default_factory=list)
    elements_below_fingerprint_floor: List[str] = field(default_factory=list)
    _per_element: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # L2 -- per-element line presence in DB
    # ------------------------------------------------------------------

    def record_db_lines(self, element: str, n_lines_in_range: int) -> None:
        """Record how many of an element's theoretical lines fall in window.

        Logs at DEBUG for the per-element row and accumulates the
        zero-line elements for the summary payload.
        """
        record = self._per_element.setdefault(element, {})
        record["n_db_lines_in_range"] = int(n_lines_in_range)
        if n_lines_in_range == 0 and element not in self.elements_with_zero_db_lines_in_range:
            self.elements_with_zero_db_lines_in_range.append(element)
        logger.debug(
            "L2 db_lines spectrum_id=%s identifier=%s element=%s n_db_lines_in_range=%d",
            self.spectrum_id,
            self.identifier_name,
            element,
            int(n_lines_in_range),
        )

    # ------------------------------------------------------------------
    # L3 -- per-element peak match
    # ------------------------------------------------------------------

    def record_peak_matches(self, element: str, n_matched_peaks: int) -> None:
        """Record how many of an element's lines matched any detected peak.

        If matched=0 and we already know L2 > 0, this is the
        smoking-gun signal: the element's lines exist in the window,
        a matcher ran, and nothing survived tolerance / shift filtering.
        """
        record = self._per_element.setdefault(element, {})
        record["n_peak_matches"] = int(n_matched_peaks)
        n_db = record.get("n_db_lines_in_range")
        if (
            n_matched_peaks == 0
            and (n_db is None or n_db > 0)
            and element not in self.elements_with_zero_peak_matches
        ):
            self.elements_with_zero_peak_matches.append(element)
        logger.debug(
            "L3 peak_matches spectrum_id=%s identifier=%s element=%s "
            "n_peak_matches=%d n_db_lines_in_range=%s",
            self.spectrum_id,
            self.identifier_name,
            element,
            int(n_matched_peaks),
            "?" if n_db is None else int(n_db),
        )

    # ------------------------------------------------------------------
    # L4 -- per-element fingerprint pass
    # ------------------------------------------------------------------

    def record_fingerprint(
        self,
        element: str,
        passed: bool,
        score: Optional[float] = None,
        floor: Optional[float] = None,
    ) -> None:
        """Record whether an element cleared the fingerprint / detection floor.

        ``passed=False`` adds the element to
        ``elements_below_fingerprint_floor`` regardless of whether L3
        matches happened -- the gate signal is what matters here.
        """
        record = self._per_element.setdefault(element, {})
        record["fingerprint_passed"] = bool(passed)
        if score is not None:
            record["fingerprint_score"] = float(score)
        if floor is not None:
            record["fingerprint_floor"] = float(floor)
        if not passed and element not in self.elements_below_fingerprint_floor:
            self.elements_below_fingerprint_floor.append(element)
        logger.debug(
            "L4 fingerprint spectrum_id=%s identifier=%s element=%s passed=%s "
            "score=%s floor=%s",
            self.spectrum_id,
            self.identifier_name,
            element,
            bool(passed),
            "?" if score is None else f"{float(score):.4g}",
            "?" if floor is None else f"{float(floor):.4g}",
        )

    # ------------------------------------------------------------------
    # Summary + finalisation
    # ------------------------------------------------------------------

    def set_n_peaks(self, n_peaks: int) -> None:
        """Record the peak count produced by L1 for the summary payload."""
        self.n_peaks_detected = int(n_peaks)

    def build_payload(self) -> Dict[str, Any]:
        """Build the public coverage payload for ``result.parameters``.

        Keys match the contract in the task description:

        - ``n_peaks_detected``
        - ``elements_with_zero_db_lines_in_range``
        - ``elements_with_zero_peak_matches``
        - ``elements_below_fingerprint_floor``
        """
        return {
            "n_peaks_detected": int(self.n_peaks_detected),
            "elements_with_zero_db_lines_in_range": sorted(
                self.elements_with_zero_db_lines_in_range
            ),
            "elements_with_zero_peak_matches": sorted(self.elements_with_zero_peak_matches),
            "elements_below_fingerprint_floor": sorted(self.elements_below_fingerprint_floor),
        }

    def emit_summary(self) -> None:
        """Emit the INFO-level summary record for this spectrum."""
        logger.info(
            "L2/L3/L4 coverage_summary spectrum_id=%s identifier=%s n_peaks=%d "
            "elements_zero_db_lines=%s elements_zero_peak_matches=%s "
            "elements_below_fingerprint=%s",
            self.spectrum_id,
            self.identifier_name,
            int(self.n_peaks_detected),
            sorted(self.elements_with_zero_db_lines_in_range),
            sorted(self.elements_with_zero_peak_matches),
            sorted(self.elements_below_fingerprint_floor),
        )


def count_lines_in_range(
    transitions: Iterable[Any],
    wl_min: float,
    wl_max: float,
) -> int:
    """Count transitions whose ``wavelength_nm`` falls in ``[wl_min, wl_max]``.

    Accepts ``Transition``-like objects (anything with a
    ``wavelength_nm`` attribute) as well as ``None`` (returns 0).
    """
    if transitions is None:
        return 0
    count = 0
    for t in transitions:
        wl = getattr(t, "wavelength_nm", None)
        if wl is None:
            continue
        try:
            wl_f = float(wl)
        except (TypeError, ValueError):
            continue
        if wl_min <= wl_f <= wl_max:
            count += 1
    return count


def merge_coverage_into_parameters(
    parameters: Mapping[str, Any] | None,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a new parameters dict with the coverage payload merged in.

    Coverage keys overwrite any pre-existing key of the same name to
    avoid silent stale telemetry, but every other parameter is kept
    intact.
    """
    merged: Dict[str, Any] = dict(parameters or {})
    for key, value in payload.items():
        merged[key] = value
    return merged
