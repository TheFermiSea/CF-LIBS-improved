"""Tests for cflibs.observability.element_confusion."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from cflibs.observability.element_confusion import aggregate_confusion, render_markdown


def _csv(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    for c in ("predicted_elements", "true_elements"):
        if c in df.columns:
            df[c] = df[c].apply(lambda v: json.dumps(v) if not isinstance(v, str) else v)
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def synthetic_csv(tmp_path: Path) -> Path:
    # (workflow, dataset) = (spectral_nnls, vrabel2020): 4 spectra
    #   spec_1: pred=[Fe,Ca,Pb], true=[Fe,Ca]            -> Fe TP, Ca TP, Pb FP
    #   spec_2: pred=[Fe,Pb,U],  true=[Fe,Mg]            -> Fe TP, Mg FN, Pb FP, U FP
    #   spec_3: pred=[Pb],       true=[]                 -> Pb FP
    #   spec_4: pred=[Fe,Ca],    true=[Fe,Ca]            -> Fe TP, Ca TP (no FP -> not over-pred)
    # (workflow, dataset) = (alias_jax, vrabel2020): 2 spectra
    #   spec_5: pred=[U,Pb],     true=[Fe]               -> Fe FN, U FP, Pb FP
    #   spec_6: pred=[Fe],       true=[Fe]               -> Fe TP
    return _csv(tmp_path / "id_records.csv", [
        {"workflow_name": "spectral_nnls", "dataset_id": "vrabel2020", "spectrum_id": "s1",
         "predicted_elements": ["Fe", "Ca", "Pb"], "true_elements": ["Fe", "Ca"]},
        {"workflow_name": "spectral_nnls", "dataset_id": "vrabel2020", "spectrum_id": "s2",
         "predicted_elements": ["Fe", "Pb", "U"], "true_elements": ["Fe", "Mg"]},
        {"workflow_name": "spectral_nnls", "dataset_id": "vrabel2020", "spectrum_id": "s3",
         "predicted_elements": ["Pb"], "true_elements": []},
        {"workflow_name": "spectral_nnls", "dataset_id": "vrabel2020", "spectrum_id": "s4",
         "predicted_elements": ["Fe", "Ca"], "true_elements": ["Fe", "Ca"]},
        {"workflow_name": "alias_jax", "dataset_id": "vrabel2020", "spectrum_id": "s5",
         "predicted_elements": ["U", "Pb"], "true_elements": ["Fe"]},
        {"workflow_name": "alias_jax", "dataset_id": "vrabel2020", "spectrum_id": "s6",
         "predicted_elements": ["Fe"], "true_elements": ["Fe"]},
    ])


def test_per_element_counts(synthetic_csv: Path) -> None:
    pe = aggregate_confusion([synthetic_csv])["per_element"]
    idx = {(r["workflow"], r["dataset"], r["element"]): r for _, r in pe.iterrows()}

    # spectral_nnls/vrabel2020/Fe: TP=3 (s1,s2,s4), FP=0, FN=0
    fe = idx[("spectral_nnls", "vrabel2020", "Fe")]
    assert (int(fe["tp"]), int(fe["fp"]), int(fe["fn"])) == (3, 0, 0)
    assert fe["precision"] == 1.0 and fe["recall"] == 1.0 and fe["f1"] == 1.0
    assert int(fe["support"]) == 3
    # spectral_nnls/vrabel2020/Pb: TP=0, FP=3 (s1,s2,s3), FN=0
    pb = idx[("spectral_nnls", "vrabel2020", "Pb")]
    assert (int(pb["tp"]), int(pb["fp"]), int(pb["fn"])) == (0, 3, 0)
    assert pb["precision"] == 0.0
    assert math.isnan(pb["recall"])  # no truth presence
    # spectral_nnls/vrabel2020/Mg: TP=0, FP=0, FN=1 (s2)
    mg = idx[("spectral_nnls", "vrabel2020", "Mg")]
    assert (int(mg["tp"]), int(mg["fp"]), int(mg["fn"])) == (0, 0, 1)
    assert mg["recall"] == 0.0
    # alias_jax/vrabel2020/Fe: TP=1 (s6), FN=1 (s5)
    fe_a = idx[("alias_jax", "vrabel2020", "Fe")]
    assert (int(fe_a["tp"]), int(fe_a["fp"]), int(fe_a["fn"])) == (1, 0, 1)
    assert fe_a["recall"] == 0.5


def test_overpred_rate_and_top_elements(synthetic_csv: Path) -> None:
    op = aggregate_confusion([synthetic_csv])["overpred_per_dataset"]
    nnls = op[(op.workflow == "spectral_nnls") & (op.dataset == "vrabel2020")].iloc[0]
    # 3 of 4 spectra over-predict (s1,s2,s3 have FPs; s4 does not).
    assert nnls["n_spectra"] == 4
    assert nnls["overpred_rate"] == pytest.approx(0.75)
    assert nnls["top_overpredicted_elements"].startswith("Pb (75.0%)")
    alias = op[(op.workflow == "alias_jax") & (op.dataset == "vrabel2020")].iloc[0]
    # s5 has FPs (U,Pb); s6 does not.
    assert alias["overpred_rate"] == pytest.approx(0.5)


def test_cross_workflow_pivot_and_empty_inputs(synthetic_csv: Path, tmp_path: Path) -> None:
    cw = aggregate_confusion([synthetic_csv])["cross_workflow"]
    assert "element" in cw.columns
    # Both workflow FP columns must appear.
    assert "spectral_nnls_fp" in cw.columns and "alias_jax_fp" in cw.columns
    pb_row = cw[cw.element == "Pb"].iloc[0]
    assert int(pb_row["spectral_nnls_fp"]) == 3 and int(pb_row["alias_jax_fp"]) == 1

    # Empty inputs: header-only CSV produces empty frames with stable schemas.
    empty = tmp_path / "empty.csv"
    empty.write_text("workflow_name,dataset_id,spectrum_id,predicted_elements,true_elements\n")
    r = aggregate_confusion([empty])
    assert r["per_element"].empty and r["overpred_per_dataset"].empty and r["cross_workflow"].empty


def test_markdown_render_has_all_three_tables(synthetic_csv: Path) -> None:
    md = render_markdown(aggregate_confusion([synthetic_csv]), source_count=1)
    assert "## A) Per (workflow, dataset, element) — top 50 by FP" in md
    assert "## B) Cross-workflow comparison (top-10 elements by total FP)" in md
    assert "## C) Per (workflow, dataset) over-prediction rate" in md
    # Pb is the worst offender — it should appear in table A.
    assert "Pb" in md and "spectral_nnls" in md and "vrabel2020" in md
    # Cross-workflow row has both _fp columns.
    assert "spectral_nnls_fp" in md and "alias_jax_fp" in md
