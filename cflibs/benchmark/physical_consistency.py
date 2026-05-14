"""Tier-1 physical-consistency gate for the CF-LIBS benchmark.

Implements the four physical-consistency checks specified in
``docs/VALIDATION_METRICS.md`` §2.5 and ``validation/protocol.yaml``
``physical_consistency`` block:

1. **Multi-T LTE consistency** — Cristoforetti 2010
   ``|T_neutral - T_ion| / T_avg < 0.15``
2. **McWhirter floor** — textbook LTE
   ``n_e >= 1.6e12 * sqrt(T) * (delta_E_eV)**3``
3. **Plasma-T physicality**
   ``T_e ∈ [3000, 20000] K`` (catastrophic if any < 1000 K)
4. **Closure residual (un-normalized)**
   ``|sum(C_s) - 1| < 0.10`` *before* forced closure

Aggregation rule (matches ``validation/protocol.yaml``)::

    any 2 of {LTE consistency, McWhirter floor, T physicality,
              closure residual} tripping  → block
    any 1 trip                            → Tier-2 alarm
    catastrophic T (< 1000 K) on ANY spec → block

The module deliberately stays self-contained and side-effect-free so
the same code can be called from the unified-benchmark CLI **and** by
external parsers (e.g. ``python/benchmark_gate.py`` in beefcake-swarm).

A *check* returns either ``True`` (passed) or ``False`` (tripped).
A check is **counted as "tripped" only when it was actually evaluated
on a spectrum**; a record that lacks the inputs for a given check is
recorded as ``N/A`` and excluded from the trip count for that check.

Reference
---------
- Cristoforetti, G. et al. (2010) Spectrochim. Acta B 65, 86-95.
- McWhirter, R.W.P. (1965) in *Plasma Diagnostic Techniques*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# McWhirter constant: n_e >= 1.6e12 * sqrt(T_K) * (delta_E_eV)**3
# (cm^-3 for delta_E in eV, T in K). Mirror of MCWHIRTER_CONST in
# ``cflibs.core.constants`` so this module remains importable in
# minimal environments (e.g. the beefcake-swarm parser) without
# pulling in the full cflibs package.
_MCWHIRTER_CONST = 1.6e12

# Default physicality bounds — match validation/protocol.yaml.
T_MIN_K = 3000.0
T_MAX_K = 20000.0
T_CATASTROPHIC_BELOW_K = 1000.0

# LTE consistency threshold from Cristoforetti 2010.
LTE_CONSISTENCY_MAX = 0.15

# Closure residual threshold (un-normalized, BEFORE forced closure).
CLOSURE_RESIDUAL_MAX = 0.10


# ---------------------------------------------------------------------------
# Per-check primitives
# ---------------------------------------------------------------------------


def check_lte_consistency(t_neutral: Optional[float], t_ion: Optional[float]) -> bool:
    """Multi-T LTE consistency check (Cristoforetti 2010).

    Returns ``True`` (pass) iff ``|T_n - T_i| / T_avg < 0.15``.

    Both inputs must be positive finite Kelvin. Missing or non-finite
    inputs cause the check to *raise* :class:`ValueError`; callers that
    want "skip when data missing" should guard the call with their own
    presence check (and record N/A) rather than relying on a falsy
    return.
    """
    if t_neutral is None or t_ion is None:
        raise ValueError("LTE consistency requires both t_neutral and t_ion")
    if not (math.isfinite(t_neutral) and math.isfinite(t_ion)):
        raise ValueError("LTE consistency requires finite temperatures")
    if t_neutral <= 0 or t_ion <= 0:
        raise ValueError("LTE consistency requires positive temperatures")
    t_avg = 0.5 * (t_neutral + t_ion)
    rel_delta = abs(t_neutral - t_ion) / t_avg
    return rel_delta < LTE_CONSISTENCY_MAX


def check_mcwhirter_floor(
    n_e: Optional[float], t: Optional[float], delta_e_ev: Optional[float]
) -> bool:
    """McWhirter floor for LTE validity.

    Returns ``True`` iff ``n_e >= 1.6e12 * sqrt(T) * (delta_E_eV)**3``.

    Raises :class:`ValueError` when any input is missing / non-finite /
    non-positive.
    """
    if n_e is None or t is None or delta_e_ev is None:
        raise ValueError("McWhirter floor requires n_e, T, and delta_E_eV")
    if not (math.isfinite(n_e) and math.isfinite(t) and math.isfinite(delta_e_ev)):
        raise ValueError("McWhirter floor requires finite inputs")
    if n_e <= 0 or t <= 0 or delta_e_ev <= 0:
        raise ValueError("McWhirter floor requires positive inputs")
    floor = _MCWHIRTER_CONST * math.sqrt(t) * (delta_e_ev**3)
    return n_e >= floor


def check_t_physicality(t_e: Optional[float]) -> Tuple[bool, bool]:
    """Plasma-T physicality bound.

    Returns ``(in_bounds, catastrophic)``:

    * ``in_bounds`` is ``True`` when ``T_MIN_K <= t_e <= T_MAX_K``.
    * ``catastrophic`` is ``True`` when ``t_e < T_CATASTROPHIC_BELOW_K``.
      Catastrophic fails block on a *single* spectrum (override the
      any-2-trip rule).

    Raises :class:`ValueError` when ``t_e`` is None or non-finite.
    """
    if t_e is None or not math.isfinite(t_e):
        raise ValueError("T physicality requires a finite t_e")
    in_bounds = T_MIN_K <= t_e <= T_MAX_K
    catastrophic = t_e < T_CATASTROPHIC_BELOW_K
    return in_bounds, catastrophic


def check_closure_residual(
    composition: Optional[Mapping[str, float]],
) -> Tuple[bool, float]:
    """Closure residual on the *un-normalized* composition.

    Returns ``(passed, residual)`` where ``residual = |sum(C_s) - 1|``.
    Passed iff ``residual < CLOSURE_RESIDUAL_MAX``.

    Pass an *un-normalized* composition: callers that re-normalise to
    sum=1 destroy this signal. The magnitude of residual is itself a
    missing-element bias diagnostic — log it even when in bounds.
    """
    if composition is None or not composition:
        raise ValueError("closure residual requires a non-empty composition")
    total = sum(float(v) for v in composition.values())
    residual = abs(total - 1.0)
    return residual < CLOSURE_RESIDUAL_MAX, residual


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


@dataclass
class CheckCounts:
    """Per-check pass/fail accounting across spectra."""

    name: str
    n_evaluated: int = 0
    n_passed: int = 0
    n_failed: int = 0
    # Spectrum-IDs that tripped this check (best-effort; capped to keep
    # the JSON report small).
    failing_spectra: List[str] = field(default_factory=list)

    def record(self, passed: bool, spectrum_id: str = "") -> None:
        self.n_evaluated += 1
        if passed:
            self.n_passed += 1
        else:
            self.n_failed += 1
            if len(self.failing_spectra) < 50:
                self.failing_spectra.append(spectrum_id)

    @property
    def tripped(self) -> bool:
        """The check is "tripped" if any evaluated spectrum failed it."""
        return self.n_failed > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_evaluated": self.n_evaluated,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "tripped": self.tripped,
            "failing_spectra": list(self.failing_spectra),
        }


@dataclass
class PhysicalConsistencyReport:
    """Aggregate physical-consistency report.

    See ``aggregate_physical_consistency`` for the trip-counting and
    block-decision logic.
    """

    n_spectra: int
    lte_consistency: CheckCounts
    mcwhirter_floor: CheckCounts
    t_physicality: CheckCounts
    closure_residual: CheckCounts
    catastrophic_t_count: int
    catastrophic_t_spectra: List[str]
    # Block decision and per-rule reason — set by ``decide``.
    blocked: bool = False
    alarm: bool = False
    block_reason: str = ""
    n_tripped: int = 0
    tripped_checks: List[str] = field(default_factory=list)

    @property
    def all_checks(self) -> List[CheckCounts]:
        return [
            self.lte_consistency,
            self.mcwhirter_floor,
            self.t_physicality,
            self.closure_residual,
        ]

    def decide(self, n_trip_to_block: int = 2) -> None:
        """Populate ``blocked`` / ``alarm`` / ``block_reason``.

        Rule (matches ``validation/protocol.yaml`` §physical_consistency)::

            catastrophic T on ANY spectrum → block
            >= n_trip_to_block checks tripped → block
            >= 1 check tripped (but < n_trip_to_block) → Tier-2 alarm
        """
        tripped = [c.name for c in self.all_checks if c.tripped]
        self.tripped_checks = tripped
        self.n_tripped = len(tripped)

        if self.catastrophic_t_count > 0:
            self.blocked = True
            self.alarm = True
            self.block_reason = (
                f"catastrophic plasma-T (< {T_CATASTROPHIC_BELOW_K:.0f} K) on "
                f"{self.catastrophic_t_count} spectrum/spectra"
            )
        elif self.n_tripped >= n_trip_to_block:
            self.blocked = True
            self.alarm = True
            self.block_reason = (
                f"{self.n_tripped} of 4 physical-consistency checks tripped: "
                f"{', '.join(tripped)}"
            )
        elif self.n_tripped >= 1:
            self.alarm = True
            self.block_reason = (
                f"Tier-2 alarm: {self.n_tripped} of 4 physical-consistency "
                f"checks tripped: {', '.join(tripped)} (block requires "
                f"{n_trip_to_block})"
            )
        else:
            self.block_reason = "all physical-consistency checks passed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_spectra": self.n_spectra,
            "blocked": self.blocked,
            "alarm": self.alarm,
            "block_reason": self.block_reason,
            "n_tripped": self.n_tripped,
            "tripped_checks": list(self.tripped_checks),
            "catastrophic_t_count": self.catastrophic_t_count,
            "catastrophic_t_spectra": list(self.catastrophic_t_spectra),
            "checks": {c.name: c.to_dict() for c in self.all_checks},
            "thresholds": {
                "lte_consistency_max": LTE_CONSISTENCY_MAX,
                "t_min_k": T_MIN_K,
                "t_max_k": T_MAX_K,
                "t_catastrophic_below_k": T_CATASTROPHIC_BELOW_K,
                "closure_residual_max": CLOSURE_RESIDUAL_MAX,
                "mcwhirter_const": _MCWHIRTER_CONST,
            },
        }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _record_field(record: Any, name: str) -> Any:
    """Best-effort field access — supports both dataclasses and dicts."""
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _extract_inputs(record: Any) -> Dict[str, Any]:
    """Pull plasma-physics inputs out of a benchmark record.

    Per-spectrum records can be either (a) a
    :class:`CompositionEvaluationRecord`, (b) a plain mapping, or
    (c) any object with the same attribute names. The function looks
    for the following fields, in this order:

    * ``temperature_K`` (single-T fallback for both ``T_neutral`` and
      ``T_ion`` when only one is reported — in that case the LTE
      consistency check is *skipped* for that spectrum, recorded as
      N/A, since with one temperature it has no signal)
    * ``annotations.t_neutral_k`` / ``annotations.t_ion_k`` (preferred —
      a future PR can populate these from a multi-temperature solver)
    * ``electron_density_cm3``
    * ``annotations.delta_e_ev`` (largest gap between adjacent levels;
      conservative default 2.0 eV when absent — matches the existing
      :class:`cflibs.plasma.lte_validator.LTEValidator` behaviour)
    * ``predicted_composition`` for closure residual

    Returns a dict with keys:
    ``t_neutral``, ``t_ion``, ``t_single``, ``n_e``, ``delta_e_ev``,
    ``composition``, ``spectrum_id``.
    """
    annotations = _record_field(record, "annotations") or {}
    if not isinstance(annotations, Mapping):
        annotations = {}

    t_single = _coerce_float(_record_field(record, "temperature_K"))
    t_neutral = _coerce_float(annotations.get("t_neutral_k"))
    t_ion = _coerce_float(annotations.get("t_ion_k"))
    # Some pipelines may emit them at the top level too:
    if t_neutral is None:
        t_neutral = _coerce_float(_record_field(record, "t_neutral_k"))
    if t_ion is None:
        t_ion = _coerce_float(_record_field(record, "t_ion_k"))

    n_e = _coerce_float(_record_field(record, "electron_density_cm3"))
    if n_e is None:
        n_e = _coerce_float(annotations.get("electron_density_cm3"))
        if n_e is None:
            n_e = _coerce_float(annotations.get("n_e_cm3"))

    delta_e = _coerce_float(annotations.get("delta_e_ev"))
    if delta_e is None:
        delta_e = _coerce_float(_record_field(record, "delta_e_ev"))

    composition = _record_field(record, "predicted_composition")
    if composition is None:
        composition = _record_field(record, "concentrations")
    if not isinstance(composition, Mapping):
        composition = None

    spectrum_id = str(_record_field(record, "spectrum_id") or _record_field(record, "label") or "")
    return {
        "t_neutral": t_neutral,
        "t_ion": t_ion,
        "t_single": t_single,
        "n_e": n_e,
        "delta_e_ev": delta_e,
        "composition": composition,
        "spectrum_id": spectrum_id,
    }


def aggregate_physical_consistency(
    per_spectrum_records: Iterable[Any],
    *,
    n_trip_to_block: int = 2,
    default_delta_e_ev: float = 2.0,
) -> PhysicalConsistencyReport:
    """Aggregate per-spectrum records into a Tier-1 gate report.

    Parameters
    ----------
    per_spectrum_records:
        Iterable of records. Each is either a
        :class:`CompositionEvaluationRecord`, a plain dict, or any
        object exposing the fields enumerated in
        :func:`_extract_inputs`.
    n_trip_to_block:
        Number of distinct checks that must trip (anywhere in the
        corpus) for the gate to block. Matches
        ``validation/protocol.yaml::physical_consistency.any_n_trip_to_block``.
    default_delta_e_ev:
        Conservative default for the McWhirter ``delta_E`` gap when
        records don't supply it. Mirrors
        :class:`cflibs.plasma.lte_validator.LTEValidator`.

    Returns
    -------
    :class:`PhysicalConsistencyReport`
        Already has ``decide`` called on it.
    """
    records = list(per_spectrum_records)
    lte = CheckCounts(name="lte_consistency")
    mcw = CheckCounts(name="mcwhirter_floor")
    tphys = CheckCounts(name="t_physicality")
    closure = CheckCounts(name="closure_residual")
    catastrophic_count = 0
    catastrophic_spectra: List[str] = []

    for rec in records:
        inputs = _extract_inputs(rec)
        sid = inputs["spectrum_id"]

        # 1. LTE consistency — only when BOTH neutral and ion T are
        #    reported. Single-T records silently skip this check (with
        #    one temperature you have zero signal), which matches the
        #    intended degradation path until multi-T solvers land.
        if inputs["t_neutral"] is not None and inputs["t_ion"] is not None:
            try:
                passed = check_lte_consistency(inputs["t_neutral"], inputs["t_ion"])
                lte.record(passed, sid)
            except ValueError:
                pass

        # 2. McWhirter floor — needs n_e and at least one T. Use the
        #    average of (t_neutral, t_ion) when both are present, else
        #    the single T.
        t_for_mcw: Optional[float]
        if inputs["t_neutral"] is not None and inputs["t_ion"] is not None:
            t_for_mcw = 0.5 * (inputs["t_neutral"] + inputs["t_ion"])
        else:
            t_for_mcw = inputs["t_single"]
        if inputs["n_e"] is not None and t_for_mcw is not None:
            delta_e = inputs["delta_e_ev"] or default_delta_e_ev
            try:
                passed = check_mcwhirter_floor(inputs["n_e"], t_for_mcw, delta_e)
                mcw.record(passed, sid)
            except ValueError:
                pass

        # 3. T physicality — applied to the average T when both are
        #    present, else the single T.
        t_for_phys: Optional[float] = t_for_mcw
        if t_for_phys is not None:
            try:
                in_bounds, catastrophic = check_t_physicality(t_for_phys)
                tphys.record(in_bounds, sid)
                if catastrophic:
                    catastrophic_count += 1
                    if len(catastrophic_spectra) < 50:
                        catastrophic_spectra.append(sid)
            except ValueError:
                pass

        # 4. Closure residual — on the *un-normalized* composition. The
        #    benchmark currently normalises before recording, so this
        #    check will frequently report N/A; pipelines that wish to
        #    surface true closure residual must populate
        #    ``record.annotations['raw_concentrations']`` (or expose
        #    a ``predicted_composition`` that has not been re-closed).
        if isinstance(rec, Mapping):
            raw_comp = (
                rec.get("annotations", {}).get("raw_concentrations")
                if rec.get("annotations")
                else None
            )
        else:
            ann = getattr(rec, "annotations", None) or {}
            raw_comp = ann.get("raw_concentrations") if isinstance(ann, Mapping) else None
        comp_for_closure = raw_comp if raw_comp else inputs["composition"]
        if comp_for_closure:
            try:
                passed, _ = check_closure_residual(comp_for_closure)
                closure.record(passed, sid)
            except ValueError:
                pass

    report = PhysicalConsistencyReport(
        n_spectra=len(records),
        lte_consistency=lte,
        mcwhirter_floor=mcw,
        t_physicality=tphys,
        closure_residual=closure,
        catastrophic_t_count=catastrophic_count,
        catastrophic_t_spectra=catastrophic_spectra,
    )
    report.decide(n_trip_to_block=n_trip_to_block)
    return report


def report_to_summary_lines(report: PhysicalConsistencyReport) -> List[str]:
    """Render a human-readable summary of a report (for PR comments)."""
    lines: List[str] = []
    status = "BLOCK" if report.blocked else ("ALARM" if report.alarm else "PASS")
    lines.append(f"Physical-consistency gate: {status}")
    lines.append(f"  n_spectra={report.n_spectra}")
    lines.append(
        f"  n_tripped={report.n_tripped} of 4 checks (tripped: "
        f"{report.tripped_checks or 'none'})"
    )
    if report.catastrophic_t_count:
        lines.append(f"  CATASTROPHIC T on {report.catastrophic_t_count} spectrum/spectra")
    for c in report.all_checks:
        lines.append(
            f"  - {c.name}: pass={c.n_passed}/{c.n_evaluated} "
            f"fail={c.n_failed} tripped={c.tripped}"
        )
    lines.append(f"  reason: {report.block_reason}")
    return lines


__all__ = [
    "CheckCounts",
    "PhysicalConsistencyReport",
    "aggregate_physical_consistency",
    "check_closure_residual",
    "check_lte_consistency",
    "check_mcwhirter_floor",
    "check_t_physicality",
    "report_to_summary_lines",
    "CLOSURE_RESIDUAL_MAX",
    "LTE_CONSISTENCY_MAX",
    "T_CATASTROPHIC_BELOW_K",
    "T_MAX_K",
    "T_MIN_K",
]
