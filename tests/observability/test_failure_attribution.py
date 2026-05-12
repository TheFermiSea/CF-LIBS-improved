"""Tests for cflibs.observability.failure_attribution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cflibs.observability.failure_attribution import attribute_failures, render_markdown


def _csv(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    for c in ("annotations", "predicted_elements", "true_elements"):
        if c in df.columns:
            df[c] = df[c].apply(lambda v: json.dumps(v) if not isinstance(v, str) else v)
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def synthetic_csv(tmp_path: Path) -> Path:
    return _csv(tmp_path / "id_records.csv", [
        # clean (no modes)
        {"workflow_name": "alias", "dataset_id": "ds_a", "spectrum_id": "spec_001",
         "annotations": {"n_detected": 12}, "predicted_elements": ["Fe", "Ca"],
         "true_elements": ["Fe", "Ca"], "f1": 1.0},
        # basis_fwhm_mismatch_large + no_lines_detected + zero_f1
        {"workflow_name": "spectral_nnls", "dataset_id": "ds_a", "spectrum_id": "spec_002",
         "annotations": {"basis_fwhm_mismatch_nm": 0.12, "n_detected": 1, "residual_norm": 0.2},
         "predicted_elements": ["Mg"], "true_elements": ["Fe", "Ca"], "f1": 0.0},
        # high_residual_norm (hybrid_*) + large_overprediction
        {"workflow_name": "hybrid_nnls", "dataset_id": "ds_b", "spectrum_id": "spec_003",
         "annotations": {"residual_norm": 1.7, "n_detected": 25, "basis_fwhm_mismatch_nm": 0.01},
         "predicted_elements": ["Fe", "Ca", "Mg", "Si", "Al", "Ti", "Na"],
         "true_elements": ["Fe"], "f1": 0.2},
        # empty_predicted_elements + zero_f1
        {"workflow_name": "alias", "dataset_id": "ds_b", "spectrum_id": "spec_004",
         "annotations": {"n_detected": 8}, "predicted_elements": [],
         "true_elements": ["Fe"], "f1": 0.0},
        # spec_002 under a second workflow (cross-workflow rollup)
        {"workflow_name": "hybrid_nnls", "dataset_id": "ds_a", "spectrum_id": "spec_002",
         "annotations": {"n_detected": 0, "residual_norm": 0.1},
         "predicted_elements": ["Mg"], "true_elements": ["Fe", "Ca"], "f1": 0.0},
    ])


def test_per_mode_counts_each_category(synthetic_csv: Path) -> None:
    pm = attribute_failures([synthetic_csv])["per_mode"]
    t = {(r["workflow"], r["dataset"], r["failure_mode"]): int(r["count"]) for _, r in pm.iterrows()}
    # All 6 failure modes must appear exactly where the fixture triggers them.
    assert t[("spectral_nnls", "ds_a", "basis_fwhm_mismatch_large")] == 1
    assert t[("spectral_nnls", "ds_a", "no_lines_detected")] == 1
    assert t[("spectral_nnls", "ds_a", "zero_f1")] == 1
    assert t[("hybrid_nnls", "ds_b", "high_residual_norm")] == 1
    assert t[("hybrid_nnls", "ds_b", "large_overprediction")] == 1
    assert t[("alias", "ds_b", "empty_predicted_elements")] == 1
    assert t[("alias", "ds_b", "zero_f1")] == 1
    assert t[("hybrid_nnls", "ds_a", "no_lines_detected")] == 1
    # residual_norm threshold must NOT fire for alias workflow.
    assert ("alias", "ds_a", "high_residual_norm") not in t
    # rate = count / n_spectra (n_spectra is per (workflow, dataset) row count).
    nnls_a = pm[(pm.workflow == "spectral_nnls") & (pm.dataset == "ds_a")]
    assert (nnls_a["n_spectra"] == 1).all() and (nnls_a["rate"] == 1.0).all()


def test_per_spectrum_rollup(synthetic_csv: Path) -> None:
    ps = attribute_failures([synthetic_csv])["per_spectrum"]
    spec_002 = ps[ps.spectrum_id == "spec_002"]
    # Same (spectrum_id, dataset) across two workflows collapses to one row.
    assert len(spec_002) == 1
    assert spec_002.iloc[0].n_workflows == 2
    modes = set(spec_002.iloc[0].failure_modes.split(","))
    assert {"basis_fwhm_mismatch_large", "no_lines_detected", "zero_f1"}.issubset(modes)
    spec_004 = ps[ps.spectrum_id == "spec_004"].iloc[0]
    assert spec_004.n_workflows == 1
    assert set(spec_004.failure_modes.split(",")) == {"empty_predicted_elements", "zero_f1"}


def test_empty_inputs_return_empty_frames(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("workflow_name,dataset_id,spectrum_id,annotations,predicted_elements,true_elements,f1\n")
    r = attribute_failures([empty])
    assert r["per_mode"].empty and r["per_spectrum"].empty
    md = render_markdown(r, source_count=1)
    assert "No failures detected." in md and "No failing spectra." in md


def test_markdown_render_has_both_tables(synthetic_csv: Path) -> None:
    md = render_markdown(attribute_failures([synthetic_csv]), source_count=1)
    assert "## A) Aggregated counts per (workflow, dataset, failure_mode)" in md
    assert "## B) Top 20 failing spectra" in md
    assert "spectral_nnls" in md and "basis_fwhm_mismatch_large" in md
    assert "spec_002" in md
