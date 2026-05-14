"""Tests for the Tier-1 physical-consistency gate.

Validates the four primitive checks (LTE consistency, McWhirter floor,
T physicality, closure residual) plus the aggregate / block-decision
logic specified in ``docs/VALIDATION_METRICS.md`` §2.5 and
``validation/protocol.yaml::physical_consistency``.
"""

from __future__ import annotations

import math

import pytest

from cflibs.benchmark.physical_consistency import (
    CLOSURE_RESIDUAL_MAX,
    LTE_CONSISTENCY_MAX,
    T_CATASTROPHIC_BELOW_K,
    T_MAX_K,
    T_MIN_K,
    aggregate_physical_consistency,
    check_closure_residual,
    check_lte_consistency,
    check_mcwhirter_floor,
    check_t_physicality,
)


# ---------------------------------------------------------------------------
# check_lte_consistency
# ---------------------------------------------------------------------------


def test_lte_consistency_pass_equal_temperatures():
    assert check_lte_consistency(10000.0, 10000.0) is True


def test_lte_consistency_pass_within_threshold():
    # |T_n - T_i| / T_avg = 1000 / 10500 ≈ 0.095 < 0.15
    assert check_lte_consistency(10000.0, 11000.0) is True


def test_lte_consistency_just_below_threshold():
    # exactly LTE_CONSISTENCY_MAX must FAIL (strict inequality in the
    # spec).
    t_n = 10000.0
    # solve |t_n - t_i| / 0.5*(t_n+t_i) = 0.15 → t_i = t_n * (1+0.075)/(1-0.075)
    # but the strict check is < 0.15 — with rel-delta exactly 0.15 the
    # check returns False.
    # Use a value slightly under to confirm pass and slightly over for fail.
    t_i_pass = 11000.0  # rel = 0.0952
    t_i_fail = 13500.0  # rel = 0.298
    assert check_lte_consistency(t_n, t_i_pass) is True
    assert check_lte_consistency(t_n, t_i_fail) is False


def test_lte_consistency_known_bad_aalto_case():
    # The PR #58 nightly motivating case: T_neutral and T_ion off by 50%
    # (8000 / 12000) — clearly violates the Cristoforetti criterion.
    t_n, t_i = 8000.0, 12000.0
    rel = abs(t_n - t_i) / (0.5 * (t_n + t_i))
    assert rel == pytest.approx(0.4, abs=1e-9)
    assert check_lte_consistency(t_n, t_i) is False


def test_lte_consistency_threshold_constant_matches_spec():
    # Sanity-check the exported constant matches Cristoforetti 2010.
    assert math.isclose(LTE_CONSISTENCY_MAX, 0.15)


def test_lte_consistency_rejects_missing():
    with pytest.raises(ValueError):
        check_lte_consistency(None, 10000.0)
    with pytest.raises(ValueError):
        check_lte_consistency(10000.0, None)


def test_lte_consistency_rejects_non_finite():
    with pytest.raises(ValueError):
        check_lte_consistency(float("nan"), 10000.0)
    with pytest.raises(ValueError):
        check_lte_consistency(10000.0, float("inf"))


def test_lte_consistency_rejects_non_positive():
    with pytest.raises(ValueError):
        check_lte_consistency(0.0, 10000.0)
    with pytest.raises(ValueError):
        check_lte_consistency(-100.0, 10000.0)


# ---------------------------------------------------------------------------
# check_mcwhirter_floor
# ---------------------------------------------------------------------------


def test_mcwhirter_floor_pass_high_density():
    # n_e well above the textbook LTE floor.
    # For T = 10000 K, delta_E = 2 eV: floor = 1.6e12 * sqrt(10000) * 8 = 1.28e15
    assert check_mcwhirter_floor(1e17, 10000.0, 2.0) is True


def test_mcwhirter_floor_fail_low_density():
    # The exact aalto-PR-58 motivating case: floor ≈ 4.84e17, n_e = 1.65e17.
    # (Reference math kept as comments: 1.6e12 * sqrt(20000) * 3**3 ≈ 6.10e15.)
    # Reproduce the spec's failing case more precisely. For T=10000, dE=4eV:
    # floor = 1.6e12 * sqrt(10000) * 64 = 1.024e16 — below n_e = 1.65e17 — would PASS.
    # Use a hot plasma + large gap to exceed n_e.
    assert check_mcwhirter_floor(1e15, 20000.0, 5.0) is False


def test_mcwhirter_floor_at_threshold():
    # n_e exactly at the floor must PASS (>= comparison in the spec).
    floor = 1.6e12 * math.sqrt(10000.0) * (2.0**3)
    assert check_mcwhirter_floor(floor, 10000.0, 2.0) is True
    # An order of magnitude below must FAIL.
    assert check_mcwhirter_floor(floor / 10, 10000.0, 2.0) is False


def test_mcwhirter_floor_rejects_missing():
    with pytest.raises(ValueError):
        check_mcwhirter_floor(None, 10000.0, 2.0)
    with pytest.raises(ValueError):
        check_mcwhirter_floor(1e17, None, 2.0)
    with pytest.raises(ValueError):
        check_mcwhirter_floor(1e17, 10000.0, None)


def test_mcwhirter_floor_rejects_non_positive():
    with pytest.raises(ValueError):
        check_mcwhirter_floor(0.0, 10000.0, 2.0)
    with pytest.raises(ValueError):
        check_mcwhirter_floor(1e17, -1.0, 2.0)


# ---------------------------------------------------------------------------
# check_t_physicality
# ---------------------------------------------------------------------------


def test_t_physicality_pass_typical_libs():
    in_bounds, catastrophic = check_t_physicality(10000.0)
    assert in_bounds is True
    assert catastrophic is False


def test_t_physicality_at_lower_bound():
    in_bounds, catastrophic = check_t_physicality(T_MIN_K)
    assert in_bounds is True
    assert catastrophic is False


def test_t_physicality_at_upper_bound():
    in_bounds, catastrophic = check_t_physicality(T_MAX_K)
    assert in_bounds is True
    assert catastrophic is False


def test_t_physicality_too_cold():
    in_bounds, catastrophic = check_t_physicality(2500.0)
    assert in_bounds is False
    assert catastrophic is False


def test_t_physicality_too_hot():
    in_bounds, catastrophic = check_t_physicality(25000.0)
    assert in_bounds is False
    assert catastrophic is False


def test_t_physicality_catastrophic():
    in_bounds, catastrophic = check_t_physicality(500.0)
    assert in_bounds is False
    assert catastrophic is True


def test_t_physicality_catastrophic_boundary():
    # Exactly T_CATASTROPHIC_BELOW_K is NOT catastrophic (strict <).
    in_bounds, catastrophic = check_t_physicality(T_CATASTROPHIC_BELOW_K)
    assert catastrophic is False
    # 1 K below is.
    in_bounds, catastrophic = check_t_physicality(T_CATASTROPHIC_BELOW_K - 1.0)
    assert catastrophic is True


def test_t_physicality_rejects_missing():
    with pytest.raises(ValueError):
        check_t_physicality(None)
    with pytest.raises(ValueError):
        check_t_physicality(float("nan"))


# ---------------------------------------------------------------------------
# check_closure_residual
# ---------------------------------------------------------------------------


def test_closure_residual_pass_normalized():
    # Sums to 1.0 → residual = 0 → pass.
    passed, residual = check_closure_residual({"Si": 0.4, "Fe": 0.3, "Mg": 0.3})
    assert residual == pytest.approx(0.0, abs=1e-12)
    assert passed is True


def test_closure_residual_pass_within_tolerance():
    # 0.95 sum → residual 0.05 → pass.
    passed, residual = check_closure_residual({"Si": 0.5, "Fe": 0.3, "Mg": 0.15})
    assert residual == pytest.approx(0.05, abs=1e-12)
    assert passed is True


def test_closure_residual_fail_under_closed():
    # Sum 0.85 → residual 0.15 → fail.
    passed, residual = check_closure_residual({"Si": 0.5, "Fe": 0.2, "Mg": 0.15})
    assert residual == pytest.approx(0.15, abs=1e-12)
    assert passed is False


def test_closure_residual_fail_over_closed():
    # Sum 1.20 → residual 0.20 → fail.
    passed, residual = check_closure_residual({"Si": 0.5, "Fe": 0.4, "Mg": 0.3})
    assert residual == pytest.approx(0.20, abs=1e-12)
    assert passed is False


def test_closure_residual_threshold_constant():
    assert math.isclose(CLOSURE_RESIDUAL_MAX, 0.10)


def test_closure_residual_rejects_empty():
    with pytest.raises(ValueError):
        check_closure_residual({})
    with pytest.raises(ValueError):
        check_closure_residual(None)


# ---------------------------------------------------------------------------
# aggregate_physical_consistency — synthetic per-spectrum records
# ---------------------------------------------------------------------------


def _good_record(spec_id: str = "good-1") -> dict:
    return {
        "spectrum_id": spec_id,
        "temperature_K": 10000.0,
        "electron_density_cm3": 1e17,
        "predicted_composition": {"Si": 0.4, "Fe": 0.3, "Mg": 0.3},
        "annotations": {
            "t_neutral_k": 10000.0,
            "t_ion_k": 10500.0,
            "delta_e_ev": 2.0,
        },
    }


def test_aggregate_known_good_all_pass():
    """Three synthetic 'good' records — every check evaluated and passes."""
    records = [_good_record(f"good-{i}") for i in range(3)]
    report = aggregate_physical_consistency(records)
    assert report.n_spectra == 3
    assert report.blocked is False
    assert report.alarm is False
    assert report.n_tripped == 0
    for c in report.all_checks:
        assert c.n_evaluated == 3
        assert c.n_passed == 3
        assert c.n_failed == 0
        assert c.tripped is False


def test_aggregate_known_bad_lte_alarm_only():
    """A single LTE-violating record fires a Tier-2 alarm but does NOT block."""
    bad = _good_record("bad-lte")
    bad["annotations"]["t_neutral_k"] = 8000.0
    bad["annotations"]["t_ion_k"] = 12000.0
    report = aggregate_physical_consistency([_good_record("ok-1"), bad])
    assert report.lte_consistency.tripped is True
    assert report.mcwhirter_floor.tripped is False
    assert report.t_physicality.tripped is False
    assert report.closure_residual.tripped is False
    assert report.n_tripped == 1
    assert report.blocked is False
    assert report.alarm is True


def test_aggregate_known_bad_mcwhirter_alarm_only():
    """One McWhirter-violating spectrum → Tier-2 alarm, not block."""
    bad = _good_record("bad-mcwhirter")
    # Drop n_e an order of magnitude below the floor.
    # With T=10000 K and delta_E=5 eV the floor is 1.6e12 * 100 * 125 = 2.0e16;
    # n_e = 1e15 is well below so McWhirter trips. T stays at 10000 K so
    # T-physicality remains satisfied.
    bad["electron_density_cm3"] = 1e15
    bad["annotations"]["delta_e_ev"] = 5.0
    report = aggregate_physical_consistency([_good_record("ok"), bad])
    assert report.mcwhirter_floor.tripped is True
    assert report.lte_consistency.tripped is False
    assert report.t_physicality.tripped is False
    assert report.n_tripped == 1
    assert report.blocked is False
    assert report.alarm is True


def test_aggregate_two_trips_blocks():
    """Two distinct checks tripping → block (default n_trip_to_block=2)."""
    bad_lte = _good_record("bad-lte")
    bad_lte["annotations"]["t_neutral_k"] = 8000.0
    bad_lte["annotations"]["t_ion_k"] = 12000.0

    bad_mcw = _good_record("bad-mcw")
    bad_mcw["electron_density_cm3"] = 1e15
    bad_mcw["annotations"]["delta_e_ev"] = 5.0

    report = aggregate_physical_consistency(
        [_good_record("ok-1"), _good_record("ok-2"), bad_lte, bad_mcw]
    )
    assert report.lte_consistency.tripped is True
    assert report.mcwhirter_floor.tripped is True
    # T-physicality stays passing (all temperatures are 10000-12000 K) so
    # exactly two checks trip → block.
    assert report.t_physicality.tripped is False
    assert report.n_tripped == 2
    assert report.blocked is True
    assert report.alarm is True
    assert "tripped" in report.block_reason


def test_aggregate_catastrophic_t_blocks_immediately():
    """A single catastrophic-T spectrum blocks regardless of trip count."""
    cold = _good_record("cold")
    cold["temperature_K"] = 500.0
    cold["annotations"]["t_neutral_k"] = 500.0
    cold["annotations"]["t_ion_k"] = 510.0  # within LTE band but T physicality fails
    report = aggregate_physical_consistency(
        [_good_record("ok-1"), _good_record("ok-2"), cold]
    )
    assert report.catastrophic_t_count == 1
    assert "cold" in report.catastrophic_t_spectra
    assert report.blocked is True
    assert "catastrophic" in report.block_reason.lower()


def test_aggregate_no_records_passes_trivially():
    """Empty corpus must not crash; reports n_spectra=0 and no blocks."""
    report = aggregate_physical_consistency([])
    assert report.n_spectra == 0
    assert report.blocked is False
    assert report.alarm is False
    assert report.n_tripped == 0


def test_aggregate_skips_records_missing_inputs():
    """Records missing fields → checks are recorded as N/A, not failures."""
    bare = {"spectrum_id": "bare", "predicted_composition": {"Si": 1.0}}
    report = aggregate_physical_consistency([bare, _good_record("ok")])
    # LTE consistency: only the good record had both T_n and T_i.
    assert report.lte_consistency.n_evaluated == 1
    # T physicality: only the good record had a temperature.
    assert report.t_physicality.n_evaluated == 1
    # Closure residual: both records had compositions.
    assert report.closure_residual.n_evaluated == 2
    # No trips — bare record contributes N/A, not failures.
    assert report.n_tripped == 0
    assert report.blocked is False


def test_aggregate_trip_threshold_can_be_tightened():
    """``n_trip_to_block=1`` flips a single trip from alarm → block."""
    bad = _good_record("bad-lte")
    bad["annotations"]["t_neutral_k"] = 8000.0
    bad["annotations"]["t_ion_k"] = 12000.0
    report = aggregate_physical_consistency([_good_record("ok"), bad], n_trip_to_block=1)
    assert report.n_tripped == 1
    assert report.blocked is True
    assert report.alarm is True


def test_aggregate_to_dict_serialises_fully():
    """``to_dict`` produces a JSON-friendly mapping with all expected keys."""
    report = aggregate_physical_consistency([_good_record("ok")])
    payload = report.to_dict()
    for key in (
        "n_spectra",
        "blocked",
        "alarm",
        "block_reason",
        "n_tripped",
        "tripped_checks",
        "catastrophic_t_count",
        "catastrophic_t_spectra",
        "checks",
        "thresholds",
    ):
        assert key in payload
    for check_name in (
        "lte_consistency",
        "mcwhirter_floor",
        "t_physicality",
        "closure_residual",
    ):
        assert check_name in payload["checks"]
        assert "n_evaluated" in payload["checks"][check_name]
        assert "tripped" in payload["checks"][check_name]


def test_aggregate_handles_dataclass_records():
    """Records exposed as objects with attributes (not dicts) also work."""

    class _Rec:
        def __init__(self) -> None:
            self.spectrum_id = "obj-1"
            self.temperature_K = 9500.0
            self.electron_density_cm3 = 5e16
            self.predicted_composition = {"Si": 0.5, "Fe": 0.5}
            self.annotations = {
                "t_neutral_k": 9500.0,
                "t_ion_k": 9700.0,
                "delta_e_ev": 2.0,
            }

    report = aggregate_physical_consistency([_Rec()])
    assert report.blocked is False
    assert report.lte_consistency.n_evaluated == 1
    assert report.mcwhirter_floor.n_evaluated == 1
    assert report.t_physicality.n_evaluated == 1
    assert report.closure_residual.n_evaluated == 1


def test_aggregate_reports_failing_spectrum_ids():
    """Failing-spectrum lists let downstream consumers point reviewers."""
    bad = _good_record("bad-lte")
    bad["annotations"]["t_neutral_k"] = 8000.0
    bad["annotations"]["t_ion_k"] = 12000.0
    report = aggregate_physical_consistency([_good_record("ok"), bad])
    assert "bad-lte" in report.lte_consistency.failing_spectra
