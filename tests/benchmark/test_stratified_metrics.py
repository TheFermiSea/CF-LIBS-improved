"""
Tests for stratified composition reporting (CF-LIBS-improved-e68).

Covers ``classify_stratum`` and ``stratify_per_element_errors`` from
``cflibs.benchmark.composition_metrics``, plus the integration into
``BenchmarkMetrics.evaluate`` (``EvaluationResult.per_stratum_summary``).

The acceptance criterion that motivates this file is that a "majors-only"
PR — one that improves majors at the cost of trace recovery — must NOT
silently pass the stratified gate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cflibs.benchmark.composition_metrics import (
    DEFAULT_STRATA_THRESHOLDS,
    classify_stratum,
    stratify_per_element_errors,
)
from cflibs.benchmark.metrics import BenchmarkMetrics


# ---------------------------------------------------------------------------
# classify_stratum — boundary cases at 5 wt% and 0.1 wt%
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("c", "expected"),
    [
        # Strictly above the majors floor (5 wt%) -> majors
        (0.0501, "majors"),
        (0.10, "majors"),
        (0.50, "majors"),
        # Exactly 5 wt% sits at the majors floor.  The protocol uses
        # ``> 5 wt%`` for majors; the boundary itself is minors.
        (0.05, "minors"),
        # Inside the minors band
        (0.01, "minors"),
        (0.005, "minors"),
        (0.001, "minors"),
        # Strictly below 0.1 wt% -> traces
        (0.0009, "traces"),
        (0.0001, "traces"),
        (1e-6, "traces"),
    ],
)
def test_classify_stratum_boundary_values(c: float, expected: str) -> None:
    assert classify_stratum(c) == expected


def test_classify_stratum_uses_certified_value():
    """Stratification must be by certified value, NOT predicted.

    We don't expose ``predicted`` to ``classify_stratum`` at all (by design),
    so the contract is enforced by the API itself.  This test pins the
    contract so a future refactor that accidentally introduces a predicted
    overload would have to update the test.
    """
    import inspect

    sig = inspect.signature(classify_stratum)
    assert "predicted" not in sig.parameters
    assert "certified_concentration" in sig.parameters


# ---------------------------------------------------------------------------
# stratify_per_element_errors — empty / single bucket / pass-fail
# ---------------------------------------------------------------------------


def test_stratify_empty_records_returns_vacuous_pass():
    summary = stratify_per_element_errors([])
    for stratum in ("majors", "minors", "traces"):
        assert summary[stratum]["n_spectra"] == 0
        assert summary[stratum]["pass"] is True
        assert math.isnan(summary[stratum]["mean_rd"])


def test_stratify_drops_records_with_zero_truth():
    """Records with true=0 cannot be stratified (RD undefined) -> dropped."""
    records = [
        {"element": "Fe", "true": 0.10, "predicted": 0.10},
        {"element": "Mn", "true": 0.0, "predicted": 0.001},  # dropped
    ]
    summary = stratify_per_element_errors(records)
    # Only the Fe record survives; it lands in majors with rd=0.
    assert summary["majors"]["n_spectra"] == 1
    assert summary["minors"]["n_spectra"] == 0
    assert summary["traces"]["n_spectra"] == 0


def test_stratify_majors_passes_when_under_5_pct_rd():
    records = [
        {"element": "Si", "true": 0.30, "predicted": 0.30},  # rd = 0
        {"element": "Si", "true": 0.30, "predicted": 0.31},  # rd ~ 3.3%
        {"element": "Fe", "true": 0.10, "predicted": 0.104},  # rd = 4%
    ]
    summary = stratify_per_element_errors(records)
    assert summary["majors"]["n_spectra"] == 3
    assert summary["majors"]["mean_rd"] < 0.05
    assert summary["majors"]["pass"] is True


def test_stratify_majors_fails_when_over_5_pct_rd():
    records = [
        {"element": "Si", "true": 0.30, "predicted": 0.36},  # rd = 20%
        {"element": "Fe", "true": 0.10, "predicted": 0.105},  # rd = 5%
    ]
    summary = stratify_per_element_errors(records)
    assert summary["majors"]["n_spectra"] == 2
    assert summary["majors"]["mean_rd"] > 0.05
    assert summary["majors"]["pass"] is False


def test_stratify_minors_passes_under_20_pct_rd():
    records = [
        {"element": "Mn", "true": 0.005, "predicted": 0.0055},  # rd = 10%
        {"element": "Cu", "true": 0.002, "predicted": 0.0023},  # rd = 15%
    ]
    summary = stratify_per_element_errors(records)
    assert summary["minors"]["n_spectra"] == 2
    assert summary["minors"]["pass"] is True


def test_stratify_minors_fails_over_20_pct_rd():
    records = [
        {"element": "Mn", "true": 0.005, "predicted": 0.008},  # rd = 60%
    ]
    summary = stratify_per_element_errors(records)
    assert summary["minors"]["pass"] is False


def test_stratify_traces_without_loq_table_is_informational():
    """Trace gating without an LOQ lookup must not silently fail.

    Without a per-element LOQ we cannot enforce MDL-bounded recovery.
    The default behavior is to pass (informational only) so that the
    absence of LOQ data does not block PRs that simply have no trace LOQ
    instrumentation.
    """
    records = [
        {"element": "Hg", "true": 0.0001, "predicted": 0.0005},  # 4x recovery
    ]
    summary = stratify_per_element_errors(records, loq_lookup=None)
    assert summary["traces"]["n_spectra"] == 1
    assert summary["traces"]["pass"] is True  # vacuous pass — no LOQ supplied


def test_stratify_traces_gates_on_loq_when_provided():
    """With an LOQ table, trace RD must stay within ``mdl_factor * LOQ / true``."""
    records = [
        # true 0.0001 (1 ppm), pred 0.00012 (rd=20%)
        # If LOQ=0.0001 (1 ppm), bound = 3 * (LOQ/true) = 3.0; 20% << 300% -> pass
        {"element": "Hg", "true": 0.0001, "predicted": 0.00012},
    ]
    summary = stratify_per_element_errors(
        records, loq_lookup={"Hg": 0.0001}
    )
    assert summary["traces"]["pass"] is True

    # Now make recovery wildly inflated (10x) AND set LOQ very tight
    records_bad = [
        {"element": "Hg", "true": 0.0001, "predicted": 0.001},  # rd = 9.0
    ]
    summary_bad = stratify_per_element_errors(
        records_bad, loq_lookup={"Hg": 1e-7}  # extremely low LOQ
    )
    assert summary_bad["traces"]["pass"] is False


# ---------------------------------------------------------------------------
# Boundary cases AT 5 wt% / 0.1 wt% / LOQ — required by the issue spec
# ---------------------------------------------------------------------------


def test_boundary_exactly_5_wt_pct_lands_in_minors():
    records = [
        {"element": "Cr", "true": 0.05, "predicted": 0.0500001},
    ]
    summary = stratify_per_element_errors(records)
    # 5 wt% is at the majors floor (>5%) -> minors.
    assert summary["minors"]["n_spectra"] == 1
    assert summary["majors"]["n_spectra"] == 0


def test_boundary_exactly_0_1_wt_pct_lands_in_minors():
    records = [
        {"element": "Mn", "true": 0.001, "predicted": 0.001},
    ]
    summary = stratify_per_element_errors(records)
    # 0.1 wt% is at the traces ceiling (<0.1%) -> minors (inclusive minors).
    assert summary["minors"]["n_spectra"] == 1
    assert summary["traces"]["n_spectra"] == 0


def test_boundary_just_below_loq_in_traces():
    """A value just below the trace ceiling lands in traces."""
    records = [
        {"element": "Hg", "true": 0.0009, "predicted": 0.001},
    ]
    summary = stratify_per_element_errors(records)
    assert summary["traces"]["n_spectra"] == 1


# ---------------------------------------------------------------------------
# THE acceptance criterion: a "majors-only-improved" PR must NOT pass
# when traces regress.
# ---------------------------------------------------------------------------


def test_majors_only_win_does_not_silently_pass_when_traces_regress():
    """Synthetic majors-only improvement that wrecks trace recovery.

    Baseline:
    - Majors: Si@30wt%, Fe@10wt% — both at 8% RD (FAIL: > 5%)
    - Traces: Hg@0.005 wt% (50 ppm) recovered at 5x — bad but bounded.

    PR ('majors-only-improved'):
    - Majors: Si@30wt%, Fe@10wt% — both at 1% RD (PASS).
    - Traces: Hg@0.005 wt% recovered at 100x — much worse.

    When LOQ-bounded trace gating is enabled, the PR's traces fail
    even though majors improved.  Without stratification, the aggregate
    Aitchison distance might happily decrease (Si and Fe dominate the
    composition vector) — which is exactly the silent-pass failure mode
    we're guarding against.
    """
    # PR / candidate
    candidate_records = [
        {"element": "Si", "true": 0.30, "predicted": 0.303},  # majors, rd=1%
        {"element": "Fe", "true": 0.10, "predicted": 0.101},  # majors, rd=1%
        {"element": "Hg", "true": 0.00005, "predicted": 0.005},  # traces, 100x
    ]
    summary = stratify_per_element_errors(
        candidate_records,
        loq_lookup={"Hg": 1e-6},  # 1 ppb LOQ
    )
    assert summary["majors"]["pass"] is True, "majors should pass"
    assert summary["traces"]["pass"] is False, (
        "traces must FAIL — 100x over-prediction far exceeds LOQ-bounded gate"
    )
    # Therefore: stratified gate fails overall (any tier failing -> overall fail)
    overall_pass = all(s["pass"] for s in summary.values())
    assert overall_pass is False, (
        "Stratified gate must reject the PR even though majors improved."
    )


# ---------------------------------------------------------------------------
# Integration: BenchmarkMetrics.evaluate populates per_stratum_summary
# ---------------------------------------------------------------------------


def test_benchmark_metrics_evaluate_emits_per_stratum_summary():
    metrics = BenchmarkMetrics()
    predictions = {
        "Si": [0.30, 0.30],
        "Fe": [0.10, 0.10],
        "Mn": [0.005, 0.005],
    }
    true_values = {
        "Si": [0.30, 0.30],
        "Fe": [0.10, 0.10],
        "Mn": [0.005, 0.005],
    }
    result = metrics.evaluate(predictions, true_values)
    assert "majors" in result.per_stratum_summary
    assert "minors" in result.per_stratum_summary
    assert "traces" in result.per_stratum_summary
    assert result.per_stratum_summary["majors"]["pass"] is True
    assert result.per_stratum_summary["minors"]["pass"] is True

    # Round-trip via to_dict for serialisation test
    d = result.to_dict()
    assert "per_stratum_summary" in d
    assert d["per_stratum_summary"]["majors"]["n_spectra"] == 4


def test_benchmark_metrics_default_thresholds_match_protocol_spec():
    """The library default must match the docs/VALIDATION_METRICS.md spec."""
    assert DEFAULT_STRATA_THRESHOLDS["majors"]["concentration_floor"] == pytest.approx(0.05)
    assert DEFAULT_STRATA_THRESHOLDS["majors"]["rd_max"] == pytest.approx(0.05)
    assert DEFAULT_STRATA_THRESHOLDS["minors"]["concentration_floor"] == pytest.approx(0.001)
    assert DEFAULT_STRATA_THRESHOLDS["minors"]["concentration_ceiling"] == pytest.approx(0.05)
    assert DEFAULT_STRATA_THRESHOLDS["minors"]["rd_max"] == pytest.approx(0.20)
    assert DEFAULT_STRATA_THRESHOLDS["traces"]["concentration_ceiling"] == pytest.approx(0.001)
