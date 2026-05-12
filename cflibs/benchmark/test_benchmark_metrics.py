import numpy as np
import pytest
from cflibs.benchmark.benchmark_synthetic_identifiers import compute_sample_metrics, aggregate_metrics

def test_composition_metrics_identical():
    true_comp = {"Fe": 0.8, "Cr": 0.2}
    pred_comp = {"Fe": 0.8, "Cr": 0.2}
    all_elements = ["Fe", "Cr", "Ni"]
    
    metrics = compute_sample_metrics(true_comp, pred_comp, all_elements)
    
    # Aitchison distance of identical compositions should be 0
    assert metrics["aitchison_distance"] == pytest.approx(0.0)
    assert metrics["ilr_error"] == pytest.approx(0.0)
    # Recall should be 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["f1"] == 1.0

def test_composition_metrics_mismatch():
    true_comp = {"Fe": 1.0}
    pred_comp = {"Cr": 1.0} # Complete mismatch
    all_elements = ["Fe", "Cr"]
    
    metrics = compute_sample_metrics(true_comp, pred_comp, all_elements)
    
    # Should be non-zero
    assert metrics["aitchison_distance"] > 0
    assert metrics["recall_at_3"] == 0.0
    assert metrics["f1"] == 0.0

def test_aggregate_metrics():
    results = [
        {"aitchison_distance": 0.1, "ilr_error": 0.1, "recall_at_3": 1.0, "recall_at_5": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        {"aitchison_distance": 0.3, "ilr_error": 0.3, "recall_at_3": 0.0, "recall_at_5": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    ]
    
    summary = aggregate_metrics(results)
    
    assert summary["mean_aitchison_distance"] == pytest.approx(0.2)
    assert summary["mean_recall_at_3"] == pytest.approx(0.5)
    assert "mean_ilr_error" in summary

def test_clr_stability():
    from cflibs.benchmark.benchmark_synthetic_identifiers import clr_transform
    # Test with very small values
    comp = np.array([1.0, 0.0, 1e-12])
    transformed = clr_transform(comp)
    assert not np.any(np.isnan(transformed))
    assert not np.any(np.isinf(transformed))
