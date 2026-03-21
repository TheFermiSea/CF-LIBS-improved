"""Tests for the grouped benchmark stack."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cflibs.benchmark.dataset import (
    BenchmarkDataset,
    BenchmarkSpectrum,
    InstrumentalConditions,
    MatrixType,
    SampleMetadata,
    SampleType,
    TruthType,
)
from cflibs.benchmark.unified import (
    CompositionEvaluationRecord,
    IDEvaluationRecord,
    bootstrap_ci,
    friedman_nemenyi,
    mcnemar_test,
    summarize_composition_records,
    summarize_id_records,
)


def _make_conditions(rp_estimate: float) -> InstrumentalConditions:
    wavelength = np.linspace(200.0, 800.0, 12)
    spectral_resolution_nm = float(np.mean(wavelength) / rp_estimate)
    return InstrumentalConditions(
        laser_wavelength_nm=1064.0,
        laser_energy_mj=40.0,
        spectral_range_nm=(float(wavelength.min()), float(wavelength.max())),
        spectral_resolution_nm=spectral_resolution_nm,
        spectrometer_type="Synthetic",
        detector_type="Synthetic",
        atmosphere="argon",
    )


def _make_spectrum(
    spectrum_id: str,
    group_id: str,
    sample_id: str,
    composition: dict[str, float],
    *,
    variant_index: int,
    rp_estimate: float = 900.0,
) -> BenchmarkSpectrum:
    wavelength = np.linspace(200.0, 800.0, 12)
    intensity = np.linspace(1.0, 2.0, 12) + 0.01 * variant_index
    return BenchmarkSpectrum(
        spectrum_id=spectrum_id,
        wavelength_nm=wavelength,
        intensity=intensity,
        true_composition=composition,
        conditions=_make_conditions(rp_estimate),
        metadata=SampleMetadata(
            sample_id=sample_id,
            sample_type=SampleType.SIMULATED,
            matrix_type=MatrixType.METAL_ALLOY,
            provenance=f"recipe={group_id};variant={variant_index}",
        ),
        dataset_id="synthetic_id",
        group_id=group_id,
        specimen_id=sample_id,
        instrument_id="synthetic_forward_model",
        truth_type=TruthType.SYNTHETIC,
        rp_estimate=rp_estimate,
        label_cardinality=len(composition),
        spectrum_kind="synthetic",
        annotations={"recipe": group_id, "variant_index": variant_index},
    )


def _make_grouped_dataset() -> BenchmarkDataset:
    spectra = []
    recipes = {
        "recipe_alpha": {"Fe": 0.7, "Cu": 0.3},
        "recipe_beta": {"Fe": 0.6, "Mn": 0.4},
        "recipe_gamma": {"Cu": 0.5, "Mn": 0.5},
    }
    for recipe_index, (group_id, composition) in enumerate(recipes.items()):
        for variant_index in range(2):
            sample_id = f"{group_id}_variant_{variant_index}"
            spectra.append(
                _make_spectrum(
                    spectrum_id=sample_id,
                    group_id=group_id,
                    sample_id=group_id,
                    composition=composition,
                    variant_index=variant_index,
                    rp_estimate=800.0 + 100.0 * recipe_index,
                )
            )

    return BenchmarkDataset(
        name="synthetic_id",
        version="1.0",
        spectra=spectra,
        elements=["Fe", "Cu", "Mn"],
        description="Synthetic grouped benchmark fixture",
    )


def _assert_group_integrity(dataset: BenchmarkDataset, splits) -> None:
    group_to_ids: dict[str, set[str]] = {}
    for spectrum in dataset.spectra:
        assert spectrum.group_id is not None
        group_to_ids.setdefault(spectrum.group_id, set()).add(spectrum.spectrum_id)

    covered_test_ids: set[str] = set()
    for split in splits:
        train_groups = {
            dataset.get_spectrum(spectrum_id).group_id for spectrum_id in split.train_ids
        }
        test_groups = {
            dataset.get_spectrum(spectrum_id).group_id for spectrum_id in split.test_ids
        }
        assert train_groups.isdisjoint(test_groups)
        covered_test_ids.update(split.test_ids)

        for group_id, ids in group_to_ids.items():
            ids_in_train = ids.intersection(split.train_ids)
            ids_in_test = ids.intersection(split.test_ids)
            assert not (ids_in_train and ids_in_test), group_id

    assert covered_test_ids == {spectrum.spectrum_id for spectrum in dataset.spectra}


def _make_id_record(
    workflow_name: str,
    spectrum_id: str,
    *,
    outer_split_id: str,
    truth_type: str,
    rp_estimate: float,
    label_cardinality: int,
    spectrum_kind: str,
    true_elements: list[str],
    predicted_elements: list[str],
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    exact_match: bool,
    dataset_id: str = "synthetic_id",
    group_id: str = "recipe_alpha",
    specimen_id: str = "recipe_alpha",
    instrument_id: str = "synthetic_forward_model",
    tuning_split_id: str = "inner_1",
    config_name: str = "config_a",
    elapsed_seconds: float = 0.1,
    scored: bool = True,
) -> IDEvaluationRecord:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    true_predicted = set(true_elements) | set(predicted_elements)
    jaccard = tp / len(true_predicted) if true_predicted else 0.0
    hamming_loss = (fp + fn) / max(len(true_predicted), 1)
    return IDEvaluationRecord(
        dataset_id=dataset_id,
        spectrum_id=spectrum_id,
        group_id=group_id,
        specimen_id=specimen_id,
        instrument_id=instrument_id,
        truth_type=truth_type,
        rp_estimate=rp_estimate,
        label_cardinality=label_cardinality,
        spectrum_kind=spectrum_kind,
        workflow_name=workflow_name,
        outer_split_id=outer_split_id,
        tuning_split_id=tuning_split_id,
        config_name=config_name,
        elapsed_seconds=elapsed_seconds,
        true_elements=true_elements,
        predicted_elements=predicted_elements,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        hamming_loss=hamming_loss,
        exact_match=exact_match,
        false_positives_per_spectrum=fp,
        scored=scored,
    )


def _make_composition_record(
    id_workflow_name: str,
    composition_workflow_name: str,
    spectrum_id: str,
    *,
    outer_split_id: str,
    truth_type: str,
    rp_estimate: float,
    label_cardinality: int,
    spectrum_kind: str,
    aitchison: float,
    rmse: float,
    closure_residual: float,
    dataset_id: str = "synthetic_id",
    group_id: str = "recipe_alpha",
    specimen_id: str = "recipe_alpha",
    instrument_id: str = "synthetic_forward_model",
    tuning_split_id: str = "inner_1",
    id_config_name: str = "id_config_a",
    composition_config_name: str = "comp_config_a",
    elapsed_seconds: float = 0.2,
    scored: bool = True,
) -> CompositionEvaluationRecord:
    return CompositionEvaluationRecord(
        dataset_id=dataset_id,
        spectrum_id=spectrum_id,
        group_id=group_id,
        specimen_id=specimen_id,
        instrument_id=instrument_id,
        truth_type=truth_type,
        rp_estimate=rp_estimate,
        label_cardinality=label_cardinality,
        spectrum_kind=spectrum_kind,
        id_workflow_name=id_workflow_name,
        composition_workflow_name=composition_workflow_name,
        outer_split_id=outer_split_id,
        tuning_split_id=tuning_split_id,
        id_config_name=id_config_name,
        composition_config_name=composition_config_name,
        elapsed_seconds=elapsed_seconds,
        candidate_elements=["Fe", "Cu"],
        true_composition={"Fe": 0.7, "Cu": 0.3},
        predicted_composition={"Fe": 0.68, "Cu": 0.32},
        aitchison=aitchison,
        rmse=rmse,
        temperature_error_frac=0.05,
        ne_error_frac=0.03,
        closure_residual=closure_residual,
        scored=scored,
    )


def test_grouped_kfold_splits_keep_perturbation_variants_together():
    dataset = _make_grouped_dataset()

    splits = dataset.create_grouped_kfold_splits(
        n_folds=3,
        random_seed=7,
        group_by="group_id",
        stratify_by="label_cardinality",
        name_prefix="outer",
    )

    assert len(splits) == 3
    _assert_group_integrity(dataset, splits)
    assert {split.metadata["group_by"] for split in splits} == {"group_id"}
    assert {split.metadata["stratify_by"] for split in splits} == {"label_cardinality"}
    assert {split.metadata["n_folds"] for split in splits} == {3}
    assert {split.metadata["fold_index"] for split in splits} == {0, 1, 2}


def test_grouped_loocv_splits_hold_out_one_synthetic_recipe_at_a_time():
    dataset = _make_grouped_dataset()

    splits = dataset.create_grouped_loocv_splits(group_by="group_id", name_prefix="outer")

    assert len(splits) == 3
    _assert_group_integrity(dataset, splits)
    held_out_groups = {split.metadata["held_out_group"] for split in splits}
    assert held_out_groups == {"recipe_alpha", "recipe_beta", "recipe_gamma"}
    for split in splits:
        assert split.metadata["group_by"] == "group_id"
        assert split.test_size == 2
        assert split.train_size == 4


def test_bootstrap_ci_returns_mean_and_bounds():
    center, lower, upper = bootstrap_ci([1.0, 2.0, 3.0], n_bootstrap=64, seed=0)

    assert center == pytest.approx(2.0)
    assert lower <= center <= upper
    assert lower >= 1.0
    assert upper <= 3.0


def test_summarize_id_records_groups_metrics_and_strata():
    records = [
        _make_id_record(
            "alias",
            "spectrum_1",
            outer_split_id="outer_1",
            truth_type=TruthType.ASSAY.value,
            rp_estimate=450.0,
            label_cardinality=1,
            spectrum_kind="pure_element",
            true_elements=["Fe", "Cu"],
            predicted_elements=["Fe", "Cu"],
            tp=2,
            fp=0,
            fn=0,
            tn=3,
            exact_match=True,
            group_id="recipe_alpha",
            specimen_id="recipe_alpha",
        ),
        _make_id_record(
            "alias",
            "spectrum_2",
            outer_split_id="outer_1",
            truth_type=TruthType.FORMULA_PROXY.value,
            rp_estimate=1200.0,
            label_cardinality=2,
            spectrum_kind="mineral",
            true_elements=["Fe", "Cu"],
            predicted_elements=["Fe", "Mn"],
            tp=1,
            fp=1,
            fn=1,
            tn=2,
            exact_match=False,
            group_id="recipe_beta",
            specimen_id="recipe_beta",
        ),
        _make_id_record(
            "comb",
            "spectrum_3",
            outer_split_id="outer_1",
            truth_type=TruthType.ASSAY.value,
            rp_estimate=1200.0,
            label_cardinality=1,
            spectrum_kind="pure_element",
            true_elements=["Cu"],
            predicted_elements=["Cu"],
            tp=1,
            fp=0,
            fn=0,
            tn=2,
            exact_match=True,
            group_id="recipe_gamma",
            specimen_id="recipe_gamma",
        ),
        _make_id_record(
            "alias",
            "blind_spectrum",
            outer_split_id="outer_2",
            truth_type=TruthType.BLIND.value,
            rp_estimate=900.0,
            label_cardinality=0,
            spectrum_kind="blind_stress",
            true_elements=[],
            predicted_elements=[],
            tp=0,
            fp=0,
            fn=0,
            tn=0,
            exact_match=False,
            scored=False,
        ),
    ]

    summary = summarize_id_records(records)

    assert summary["overall"]["alias"]["n_spectra"] == 2
    assert summary["overall"]["alias"]["micro_f1"] == pytest.approx(0.75)
    assert len(summary["overall"]["alias"]["bootstrap_f1"]) == 3
    assert summary["per_element"]["alias"]["Mn"]["false_positives"] == 1
    assert summary["per_element"]["alias"]["Fe"]["support"] == 2
    assert summary["stratified"]["dataset_id"]["alias"]["synthetic_id"]["n_spectra"] == 2
    assert summary["stratified"]["truth_type"]["alias"][TruthType.ASSAY.value]["n_spectra"] == 1
    assert (
        summary["stratified"]["truth_type"]["alias"][TruthType.FORMULA_PROXY.value]["n_spectra"]
        == 1
    )
    assert summary["stratified"]["rp_bucket"]["alias"]["rp_lt_500"]["n_spectra"] == 1
    assert summary["stratified"]["rp_bucket"]["alias"]["rp_1000_2999"]["n_spectra"] == 1
    assert summary["stratified"]["spectrum_kind"]["alias"]["pure_element"]["n_spectra"] == 1
    assert summary["stratified"]["label_cardinality"]["alias"]["1"]["n_spectra"] == 1
    assert summary["stratified"]["label_cardinality"]["alias"]["2"]["n_spectra"] == 1


def test_summarize_composition_records_groups_pair_metrics():
    records = [
        _make_composition_record(
            "alias",
            "iterative",
            "spectrum_1",
            outer_split_id="outer_1",
            truth_type=TruthType.ASSAY.value,
            rp_estimate=450.0,
            label_cardinality=1,
            spectrum_kind="pure_element",
            aitchison=0.2,
            rmse=0.1,
            closure_residual=0.01,
        ),
        _make_composition_record(
            "alias",
            "iterative",
            "spectrum_2",
            outer_split_id="outer_1",
            truth_type=TruthType.FORMULA_PROXY.value,
            rp_estimate=1200.0,
            label_cardinality=2,
            spectrum_kind="mineral",
            aitchison=0.4,
            rmse=0.2,
            closure_residual=0.02,
        ),
    ]

    summary = summarize_composition_records(records)
    pair_summary = summary["overall"]["alias__iterative"]

    assert pair_summary["n_spectra"] == 2
    assert pair_summary["mean_aitchison"] == pytest.approx(0.3)
    assert pair_summary["mean_rmse"] == pytest.approx(0.15)
    assert pair_summary["mean_closure_residual"] == pytest.approx(0.015)
    assert len(pair_summary["bootstrap_aitchison"]) == 3


def test_mcnemar_test_counts_disagreements_symmetrically():
    left_records = [
        _make_id_record(
            "alias",
            "spectrum_a",
            outer_split_id="outer_1",
            truth_type=TruthType.ASSAY.value,
            rp_estimate=800.0,
            label_cardinality=1,
            spectrum_kind="pure_element",
            true_elements=["Fe"],
            predicted_elements=["Fe"],
            tp=1,
            fp=0,
            fn=0,
            tn=2,
            exact_match=True,
        ),
        _make_id_record(
            "alias",
            "spectrum_b",
            outer_split_id="outer_1",
            truth_type=TruthType.ASSAY.value,
            rp_estimate=800.0,
            label_cardinality=1,
            spectrum_kind="pure_element",
            true_elements=["Cu"],
            predicted_elements=["Mn"],
            tp=0,
            fp=1,
            fn=1,
            tn=1,
            exact_match=False,
        ),
    ]
    right_records = [
        _make_id_record(
            "comb",
            "spectrum_a",
            outer_split_id="outer_1",
            truth_type=TruthType.ASSAY.value,
            rp_estimate=800.0,
            label_cardinality=1,
            spectrum_kind="pure_element",
            true_elements=["Fe"],
            predicted_elements=["Mn"],
            tp=0,
            fp=1,
            fn=1,
            tn=1,
            exact_match=False,
        ),
        _make_id_record(
            "comb",
            "spectrum_b",
            outer_split_id="outer_1",
            truth_type=TruthType.ASSAY.value,
            rp_estimate=800.0,
            label_cardinality=1,
            spectrum_kind="pure_element",
            true_elements=["Cu"],
            predicted_elements=["Cu"],
            tp=1,
            fp=0,
            fn=0,
            tn=2,
            exact_match=True,
        ),
    ]

    result = mcnemar_test(left_records, right_records)
    if math.isnan(result["chi2"]):
        pytest.skip("McNemar requires SciPy in this environment")
    assert result["b"] == 1
    assert result["c"] == 1
    assert result["chi2"] == pytest.approx(0.5)
    assert 0.0 <= result["p_value"] <= 1.0


def test_friedman_nemenyi_orders_workflows_by_rank():
    blocks = {
        "block_1": {"alias": 0.90, "comb": 0.70, "correlation": 0.50},
        "block_2": {"alias": 0.80, "comb": 0.60, "correlation": 0.40},
        "block_3": {"alias": 0.85, "comb": 0.65, "correlation": 0.45},
    }

    result = friedman_nemenyi(blocks, higher_is_better=True, alpha=0.05)
    if not result:
        pytest.skip("Friedman/Nemenyi requires SciPy in this environment")

    assert result["friedman_statistic"] >= 0.0
    assert result["friedman_p_value"] <= 1.0
    assert result["average_ranks"]["alias"] < result["average_ranks"]["comb"]
    assert result["average_ranks"]["comb"] < result["average_ranks"]["correlation"]
    assert len(result["pairs"]) == 3
    assert any("significant" in pair for pair in result["pairs"])
