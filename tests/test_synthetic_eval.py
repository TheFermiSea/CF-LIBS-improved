"""Unit tests for synthetic benchmark evaluation helpers."""

import pytest

from cflibs.benchmark.synthetic_eval import (
    compute_binary_metrics,
    confusion_counts,
    derive_truth_elements,
    summarize_aggregate,
    summarize_by_group,
)

pytestmark = pytest.mark.unit


def test_derive_truth_elements_threshold():
    composition = {"Fe": 0.8, "Ni": 0.1999, "Cu": 1e-6}
    present = derive_truth_elements(composition, presence_threshold=1e-4)
    assert present == {"Fe", "Ni"}


def test_confusion_counts_simple():
    truth = {"Fe", "Ni"}
    pred = {"Fe", "Cu"}
    elements = ["Fe", "Ni", "Cu", "Mn"]
    counts = confusion_counts(truth, pred, elements)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_compute_binary_metrics_values():
    metrics = compute_binary_metrics(tp=8, fp=2, fn=2, tn=8)
    assert abs(metrics["precision"] - 0.8) < 1e-12
    assert abs(metrics["recall"] - 0.8) < 1e-12
    assert abs(metrics["fpr"] - 0.2) < 1e-12
    assert abs(metrics["accuracy"] - 0.8) < 1e-12
    assert abs(metrics["f1"] - 0.8) < 1e-12


def test_summarize_aggregate_handles_failed_rows():
    rows = [
        {
            "algorithm": "ALIAS",
            "failed": False,
            "tp": 2,
            "fp": 1,
            "fn": 1,
            "tn": 6,
            "peak_match_rate": 0.5,
            "n_peaks": 20,
            "n_matched_peaks": 10,
            "matched_lines_true_elements": 5,
            "total_lines_true_elements": 8,
            "matched_lines_absent_elements": 3,
        },
        {
            "algorithm": "ALIAS",
            "failed": True,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "peak_match_rate": 0.0,
            "n_peaks": 0,
            "n_matched_peaks": 0,
            "matched_lines_true_elements": 0,
            "total_lines_true_elements": 0,
            "matched_lines_absent_elements": 0,
        },
    ]
    summary = summarize_aggregate(rows, candidate_elements=["Fe", "Ni"])
    assert len(summary) == 1
    alias = summary[0]
    assert alias["algorithm"] == "ALIAS"
    assert alias["n_spectra"] == 1
    assert alias["n_failed"] == 1
    assert abs(alias["precision"] - (2 / 3)) < 1e-12
    assert abs(alias["recall"] - (2 / 3)) < 1e-12


def test_summarize_by_group_groups_by_perturbation_axis():
    rows = [
        {
            "algorithm": "Comb",
            "failed": False,
            "tp": 1,
            "fp": 0,
            "fn": 1,
            "tn": 2,
            "peak_match_rate": 0.4,
            "recipe": "pure_Fe",
            "snr_db": 25.0,
            "continuum_level": 0.02,
            "shift_nm": 0.0,
            "warp_quadratic_nm": 0.0,
        },
        {
            "algorithm": "Comb",
            "failed": False,
            "tp": 2,
            "fp": 1,
            "fn": 0,
            "tn": 1,
            "peak_match_rate": 0.8,
            "recipe": "pure_Fe",
            "snr_db": 25.0,
            "continuum_level": 0.02,
            "shift_nm": 0.0,
            "warp_quadratic_nm": 0.0,
        },
    ]
    grouped = summarize_by_group(rows, candidate_elements=["Fe", "Ni"])
    assert "recipe" in grouped
    recipe_rows = grouped["recipe"]
    assert len(recipe_rows) == 1
    record = recipe_rows[0]
    assert record["group_field"] == "recipe"
    assert record["group_value"] == "pure_Fe"
    assert record["algorithm"] == "Comb"
    assert record["n_rows"] == 2
    assert abs(record["mean_peak_match_rate"] - 0.6) < 1e-12
