"""Unit tests for the shared confusion rule (cflibs.benchmark.scoring).

This module is the single home for the don't-care-aware confusion rule used by
the synthetic-corpus benchmark and the observability per-element aggregator, so
the rule's behavior is pinned here once rather than re-derived per consumer.
"""

import pytest

from cflibs.benchmark.scoring import (
    ScoringRow,
    classify_element,
    confusion_counts,
    per_element_tally,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# classify_element — the atomic rule
# ---------------------------------------------------------------------------
def test_classify_element_matrix():
    truth = {"Fe", "Ni"}
    pred = {"Fe", "Co"}
    assert classify_element("Fe", truth, pred) == "tp"  # truth & pred
    assert classify_element("Co", truth, pred) == "fp"  # pred only
    assert classify_element("Ni", truth, pred) == "fn"  # truth only
    assert classify_element("Cu", truth, pred) == "tn"  # neither


def test_classify_element_dont_care_returns_none():
    """An element in the don't-care band is never scored, even if predicted."""
    truth = {"Fe"}
    pred = {"Fe", "Mg"}
    ignore = {"Mg"}
    # Mg predicted-but-sub-floor: neither rewarded nor penalised.
    assert classify_element("Mg", truth, pred, ignore) is None
    # The don't-care band wins even over a would-be TP.
    assert classify_element("Fe", truth, pred, {"Fe"}) is None


# ---------------------------------------------------------------------------
# confusion_counts — panel-scoped, dict return (splat-friendly)
# ---------------------------------------------------------------------------
def test_confusion_counts_is_panel_restricted():
    # Co predicted but OFF-panel -> not an FP; Cu on-panel & absent -> TN.
    counts = confusion_counts({"Fe", "Ni"}, {"Fe", "Co"}, ["Fe", "Ni", "Cu"])
    assert counts == {"tp": 1, "fp": 0, "fn": 1, "tn": 1}


def test_confusion_counts_skips_dont_care_band():
    # Mg is predicted and on-panel, but in the don't-care band -> not an FP.
    counts = confusion_counts({"Fe"}, {"Fe", "Mg"}, ["Fe", "Mg"], ignore_elements={"Mg"})
    assert counts == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}


def test_confusion_counts_returns_dict_for_splat():
    counts = confusion_counts({"Fe"}, {"Fe"}, ["Fe"])
    assert set(counts) == {"tp", "fp", "fn", "tn"}
    # Splattable into a row record.
    row = {**counts}
    assert row["tp"] == 1


# ---------------------------------------------------------------------------
# per_element_tally — single pass over rows, tuple return
# ---------------------------------------------------------------------------
def test_per_element_tally_basic():
    rows = [
        {"true_elements": ["Fe", "Ni"], "predicted_elements": ["Fe", "Co"]},
        {"true_elements": ["Fe"], "predicted_elements": ["Fe"]},
    ]
    tally = per_element_tally(rows, ["Fe", "Ni", "Co"])
    assert tally["Fe"] == (2, 0, 0, 0)  # 2 TP
    assert tally["Ni"] == (0, 0, 1, 1)  # FN row 1, TN row 2
    assert tally["Co"] == (0, 1, 0, 1)  # FP row 1, TN row 2


def test_per_element_tally_honors_dont_care():
    rows = [
        {
            "true_elements": ["Fe"],
            "predicted_elements": ["Fe", "Mg"],
            "ignore_elements": ["Mg"],
        }
    ]
    tally = per_element_tally(rows, ["Fe", "Mg"])
    assert tally["Fe"] == (1, 0, 0, 0)
    # Mg is a don't-care for this row -> skipped entirely (not even TN).
    assert tally["Mg"] == (0, 0, 0, 0)


def test_per_element_tally_tolerates_missing_keys():
    tally = per_element_tally([{"true_elements": ["Fe"]}], ["Fe", "Ni"])
    assert tally["Fe"] == (0, 0, 1, 0)  # predicted missing -> FN
    assert tally["Ni"] == (0, 0, 0, 1)


# ---------------------------------------------------------------------------
# ScoringRow — typed handle on the synthetic_eval compute path
# ---------------------------------------------------------------------------
def test_scoring_row_from_row_and_confusion():
    row = {
        "true_elements": ["Fe", "Ni"],
        "predicted_elements": ["Fe", "Mg"],
        "ignore_elements": ["Mg"],
    }
    sr = ScoringRow.from_row(row)
    assert sr.true_elements == frozenset({"Fe", "Ni"})
    # Mg predicted but don't-care -> not an FP; Ni absent -> FN.
    assert sr.confusion(["Fe", "Ni", "Mg"]) == {"tp": 1, "fp": 0, "fn": 1, "tn": 0}
    assert sr.label_for("Fe") == "tp"
    assert sr.label_for("Mg") is None


def test_scoring_row_from_row_tolerates_none_and_missing():
    sr = ScoringRow.from_row({"true_elements": None})
    assert sr.true_elements == frozenset()
    assert sr.predicted_elements == frozenset()
    assert sr.ignore_elements == frozenset()


def test_scoring_row_is_frozen():
    sr = ScoringRow(frozenset({"Fe"}), frozenset(), frozenset())
    with pytest.raises(Exception):
        sr.true_elements = frozenset({"Ni"})  # type: ignore[misc]
