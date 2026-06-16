"""Unit tests for synthetic benchmark evaluation helpers."""

import numpy as np
import pytest

from cflibs.benchmark.dataset import (
    BenchmarkSpectrum,
    InstrumentalConditions,
    SampleMetadata,
)
from cflibs.benchmark.synthetic_eval import (
    ALGO_HYBRID_UNION,
    ALGO_SPECTRAL_NNLS,
    _BASE_ALGORITHMS,
    _derive_recipe,
    _select_spectra,
    build_identifier_runners,
    compute_binary_metrics,
    confusion_counts,
    derive_truth_elements,
    recount_rows,
    summarize_aggregate,
    summarize_by_group,
    summarize_confounders,
    summarize_per_element,
)

pytestmark = pytest.mark.unit


def _make_spectrum(
    spectrum_id: str,
    recipe: str,
    true_composition: dict,
    *,
    provenance: str | None = None,
) -> BenchmarkSpectrum:
    """Build a minimal BenchmarkSpectrum carrying a recipe in provenance.

    ``provenance`` defaults to the corpus format ``"...; recipe=<recipe>; ..."``;
    pass an explicit string (or empty) to exercise the fallback ladder.
    """
    if provenance is None:
        provenance = f"deterministic seed 42; recipe={recipe}; scenario=0"
    conditions = InstrumentalConditions(
        laser_wavelength_nm=1064.0,
        laser_energy_mj=50.0,
        spectral_resolution_nm=0.35,
    )
    metadata = SampleMetadata(sample_id=spectrum_id, provenance=provenance)
    wl = np.linspace(224.0, 265.0, 16)
    intensity = np.zeros_like(wl)
    return BenchmarkSpectrum(
        spectrum_id=spectrum_id,
        wavelength_nm=wl,
        intensity=intensity,
        true_composition=true_composition,
        conditions=conditions,
        metadata=metadata,
    )


def _build_recipe_corpus(n_per_recipe: int = 6):
    """Build 4 recipes x ``n_per_recipe`` spectra mirroring the real corpus.

    Recipes carry distinct label cardinalities so the (recipe, cardinality)
    strata are non-trivial: pure_Fe / pure_Ni are cardinality-1, binary_Fe_Ni
    is cardinality-2, steel_like is cardinality-4.
    """
    recipes = {
        "pure_Fe": {"Fe": 1.0},
        "pure_Ni": {"Ni": 1.0},
        "binary_Fe_Ni": {"Fe": 0.7, "Ni": 0.3},
        "steel_like": {"Fe": 0.7, "Ni": 0.1, "Cr": 0.1, "Mn": 0.1},
    }
    spectra = []
    for recipe, comp in recipes.items():
        for i in range(n_per_recipe):
            spectra.append(_make_spectrum(f"{recipe}_{i:04d}", recipe, dict(comp)))
    return spectra


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


def test_identifier_runners_default_is_trio():
    """Without a basis library the suite is exactly ALIAS/Comb/Correlation."""
    runners = build_identifier_runners(
        wavelength=None,
        intensity=None,
        db=None,
        elements=["Fe", "Ni"],
        resolving_power=1000.0,
        basis_library=None,
    )
    names = [name for name, _runner in runners]
    assert names == list(_BASE_ALGORITHMS)
    assert ALGO_SPECTRAL_NNLS not in names
    assert ALGO_HYBRID_UNION not in names


def test_identifier_runners_with_basis_appends_full_stack():
    """A basis library appends spectral_nnls and hybrid_union to the suite."""

    class _FakeBasis:
        elements = ["Fe", "Ni"]

    runners = build_identifier_runners(
        wavelength=None,
        intensity=None,
        db=None,
        elements=["Fe", "Ni"],
        resolving_power=1000.0,
        basis_library=_FakeBasis(),
    )
    names = [name for name, _runner in runners]
    # Trio first, then the two basis-dependent identifiers (order preserved).
    assert names[:3] == list(_BASE_ALGORITHMS)
    assert names[3:] == [ALGO_SPECTRAL_NNLS, ALGO_HYBRID_UNION]


# --------------------------------------------------------------------------- #
# Sampling-hygiene regression tests                                           #
# --------------------------------------------------------------------------- #


def test_stratified_sampling_covers_all_recipes():
    """Stratified draw covers all 4 recipes; sorted draw collapses to one."""
    spectra = _build_recipe_corpus(n_per_recipe=6)  # 24 spectra, 4 recipes

    strat = _select_spectra(
        spectra, max_spectra=8, sampling="stratified", seed=0, manifest_by_sample=None
    )
    assert len(strat) == 8
    strat_recipes = {_derive_recipe(s, None) for s in strat}
    assert strat_recipes == {"pure_Fe", "pure_Ni", "binary_Fe_Ni", "steel_like"}

    # The regression this fixes: legacy sorted+truncate returns only the
    # alphabetically-first recipe (binary_Fe_Ni) for a cap <= per-recipe count.
    legacy = _select_spectra(
        spectra, max_spectra=6, sampling="sorted", seed=0, manifest_by_sample=None
    )
    assert len(legacy) == 6
    assert {_derive_recipe(s, None) for s in legacy} == {"binary_Fe_Ni"}


def test_sampling_is_seed_reproducible():
    """Same seed => identical draw; different seeds still cover all recipes."""
    spectra = _build_recipe_corpus(n_per_recipe=6)

    draw_a = _select_spectra(
        spectra, max_spectra=8, sampling="stratified", seed=0, manifest_by_sample=None
    )
    draw_b = _select_spectra(
        spectra, max_spectra=8, sampling="stratified", seed=0, manifest_by_sample=None
    )
    ids_a = [s.spectrum_id for s in draw_a]
    ids_b = [s.spectrum_id for s in draw_b]
    assert ids_a == ids_b

    draw_c = _select_spectra(
        spectra, max_spectra=8, sampling="stratified", seed=7, manifest_by_sample=None
    )
    all_recipes = {"pure_Fe", "pure_Ni", "binary_Fe_Ni", "steel_like"}
    assert {_derive_recipe(s, None) for s in draw_c} == all_recipes


def test_select_spectra_returns_all_when_cap_exceeds_size():
    """max_spectra >= n returns the full sorted list regardless of mode."""
    spectra = _build_recipe_corpus(n_per_recipe=3)  # 12 spectra
    out = _select_spectra(
        spectra, max_spectra=100, sampling="stratified", seed=0, manifest_by_sample=None
    )
    assert len(out) == 12
    out_none = _select_spectra(
        spectra, max_spectra=None, sampling="stratified", seed=0, manifest_by_sample=None
    )
    assert len(out_none) == 12


def test_derive_recipe_from_provenance_and_sample_id():
    """Recipe resolution ladder: manifest -> provenance -> sample_id prefix."""
    # 1. Manifest entry wins.
    spec = _make_spectrum("binary_Fe_Ni_0007", "binary_Fe_Ni", {"Fe": 0.7, "Ni": 0.3})
    manifest = {"binary_Fe_Ni_0007": {"recipe": "manifest_recipe"}}
    assert _derive_recipe(spec, manifest) == "manifest_recipe"

    # 2. No manifest -> parse recipe= out of provenance.
    assert _derive_recipe(spec, {}) == "binary_Fe_Ni"
    assert _derive_recipe(spec, None) == "binary_Fe_Ni"

    # 3. Neither manifest nor provenance -> sample_id prefix (strip _<digits>).
    spec_bare = _make_spectrum(
        "binary_Fe_Ni_0007", "binary_Fe_Ni", {"Fe": 0.7, "Ni": 0.3}, provenance=""
    )
    assert _derive_recipe(spec_bare, None) == "binary_Fe_Ni"


def test_summarize_by_group_includes_cardinality():
    """summarize_by_group splits by label_cardinality with correct n_rows."""
    base = {
        "algorithm": "Comb",
        "failed": False,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "tn": 1,
        "peak_match_rate": 0.5,
        "recipe": "steel_like",
        "snr_db": 25.0,
        "continuum_level": 0.02,
        "shift_nm": 0.0,
        "warp_quadratic_nm": 0.0,
    }
    rows = [
        {**base, "label_cardinality": 1},
        {**base, "label_cardinality": 1},
        {**base, "label_cardinality": 2},
    ]
    grouped = summarize_by_group(rows, candidate_elements=["Fe", "Ni"])
    assert "label_cardinality" in grouped
    card_rows = grouped["label_cardinality"]
    by_value = {r["group_value"]: r for r in card_rows}
    assert set(by_value) == {1, 2}
    assert by_value[1]["n_rows"] == 2
    assert by_value[2]["n_rows"] == 1


def test_summarize_confounders_flags_never_truth():
    """Co predicted-but-never-truth lands in top_fp + never_truth; misses in top_fn."""
    candidates = ["Fe", "Ni", "Co", "Cu"]
    rows = [
        {
            "algorithm": "ALIAS",
            "failed": False,
            "true_elements": ["Fe", "Ni"],
            "predicted_elements": ["Fe", "Co"],  # Co = FP, Ni = FN
            "tp": 1,
            "fp": 1,
            "fn": 1,
            "tn": 1,
        },
        {
            "algorithm": "ALIAS",
            "failed": False,
            "true_elements": ["Fe", "Ni"],
            "predicted_elements": ["Fe", "Co"],  # Co = FP again, Ni = FN again
            "tp": 1,
            "fp": 1,
            "fn": 1,
            "tn": 1,
        },
    ]
    summary = summarize_confounders(rows, candidate_elements=candidates)
    alias = summary["ALIAS"]
    top_fp = dict(alias["top_fp"])
    top_fn = dict(alias["top_fn"])
    assert top_fp.get("Co") == 2
    assert top_fn.get("Ni") == 2
    # Co and Cu never appear in any truth set -> flagged as never_truth.
    assert "Co" in alias["never_truth"]
    assert "Cu" in alias["never_truth"]
    assert "Fe" not in alias["never_truth"]


def test_per_element_summaries_honor_dont_care_band():
    """A predicted sub-floor "don't-care" trace must not count as FP in the
    per-element / confounder re-aggregation — matching the scoring-panel
    semantics already baked into each row's stored confusion counts (Codex P2).

    Row predicts {Fe, Mg}; Mg is in this spectrum's don't-care band (real but
    sub-detection-floor), so detecting it is neither rewarded nor penalised.
    """
    candidates = ["Fe", "Ni", "Mg"]
    rows = [
        {
            "algorithm": "ALIAS",
            "failed": False,
            "true_elements": ["Fe"],
            "predicted_elements": ["Fe", "Mg"],
            "ignore_elements": ["Mg"],  # sub-floor trace -> don't-care
            # Stored counts already exclude Mg (computed over the scoring panel).
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "tn": 1,
        }
    ]
    per_el = {r["element"]: r for r in summarize_per_element(rows, candidates)}
    # Mg must not be scored as a false positive despite being predicted.
    assert per_el["Mg"]["fp"] == 0
    assert per_el["Mg"]["tp"] == 0
    assert per_el["Mg"]["tn"] == 0  # skipped entirely, not counted as TN either
    assert per_el["Fe"]["tp"] == 1

    confound = summarize_confounders(rows, candidate_elements=candidates)["ALIAS"]
    assert "Mg" not in dict(confound["top_fp"]), "don't-care trace leaked into top_fp"


def test_ever_present_panel_companion():
    """Restricting to ever_present={Fe,Ni} never lowers F1 vs the full panel."""
    full_panel = ["Fe", "Ni", "Co", "Cu", "Mg"]  # Co/Cu/Mg never in truth
    ever_present = ["Fe", "Ni"]
    rows = [
        {
            "algorithm": "ALIAS",
            "failed": False,
            "true_elements": ["Fe", "Ni"],
            "predicted_elements": ["Fe", "Co"],  # 1 TP (Fe), 1 FP (Co), 1 FN (Ni)
            **confusion_counts({"Fe", "Ni"}, {"Fe", "Co"}, full_panel),
            "peak_match_rate": 0.5,
            "n_peaks": 10,
            "n_matched_peaks": 5,
            "matched_lines_true_elements": 4,
            "total_lines_true_elements": 8,
            "matched_lines_absent_elements": 1,
        }
    ]
    full_agg = summarize_aggregate(rows, full_panel)[0]
    restricted = recount_rows(rows, ever_present)
    ep_agg = summarize_aggregate(restricted, ever_present)[0]

    # On the restricted panel, Co's FP and Mg/Cu's TN drop out: precision rises.
    assert ep_agg["fp"] == 0
    assert full_agg["fp"] == 1
    assert ep_agg["f1"] >= full_agg["f1"]
