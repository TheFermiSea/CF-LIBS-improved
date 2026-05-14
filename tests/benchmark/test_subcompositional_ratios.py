"""
Tests for subcompositional ratio errors (CF-LIBS-improved-4yr).

Covers ``subcompositional_ratio_errors`` and ``load_subcompositional_pairs``
from ``cflibs.benchmark.composition_metrics``.

The acceptance criterion that motivates this file is:

  Closure-distortion synthetic where a missing-element bias inflates
  detected concentrations (closure forces sum=1 over the detected
  subset) but the *ratios between detected elements* stay invariant.

This is the Aitchison subcompositional-coherence property — the metric
that surfaces ratio-only violations cleanly when the global Aitchison
distance is dominated by closure noise.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cflibs.benchmark.composition_metrics import (
    load_subcompositional_pairs,
    subcompositional_ratio_errors,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "validation" / "protocol.yaml"


# ---------------------------------------------------------------------------
# Pair list comes from validation/protocol.yaml — NOT hardcoded
# ---------------------------------------------------------------------------


def test_pair_list_loaded_from_protocol_yaml():
    """The default pair list must be read from validation/protocol.yaml."""
    pairs = load_subcompositional_pairs(PROTOCOL_PATH)
    assert ("Fe", "Si") in pairs
    assert ("Mg", "Si") in pairs
    assert ("Ca", "Si") in pairs
    assert ("Al", "Si") in pairs


def test_pair_list_default_search_finds_protocol_yaml():
    """Default invocation walks upward from the module to find the protocol."""
    pairs = load_subcompositional_pairs()
    # Either we found protocol.yaml (which has these four pairs) OR we fell
    # back to the canonical default — both yield the same canonical list.
    assert ("Fe", "Si") in pairs
    assert ("Mg", "Si") in pairs
    assert ("Ca", "Si") in pairs
    assert ("Al", "Si") in pairs


def test_pair_list_falls_back_when_path_missing(tmp_path: Path):
    """If protocol.yaml is missing, fall back to canonical pair list."""
    missing = tmp_path / "does-not-exist.yaml"
    pairs = load_subcompositional_pairs(missing)
    assert ("Fe", "Si") in pairs
    assert len(pairs) == 4


# ---------------------------------------------------------------------------
# subcompositional_ratio_errors — happy paths
# ---------------------------------------------------------------------------


def test_identical_compositions_give_zero_error():
    gt = {"Si": 0.30, "Fe": 0.10, "Mg": 0.05, "Ca": 0.02, "Al": 0.05}
    pred = dict(gt)
    errs = subcompositional_ratio_errors(pred, gt)
    for _, val in errs.items():
        assert val == pytest.approx(0.0, abs=1e-12)


def test_uniform_scaling_preserves_ratios():
    """Multiplying all concentrations by a constant must NOT change ratios."""
    gt = {"Si": 0.30, "Fe": 0.10, "Mg": 0.05, "Ca": 0.02, "Al": 0.05}
    pred = {el: 2.0 * v for el, v in gt.items()}
    errs = subcompositional_ratio_errors(pred, gt)
    for _, val in errs.items():
        assert val == pytest.approx(0.0, abs=1e-12)


def test_ratio_error_is_log_difference_magnitude():
    """For a single Fe/Si pair, the error must equal |log(predicted/true) ratio|."""
    gt = {"Si": 0.50, "Fe": 0.10}  # Fe/Si = 0.2
    pred = {"Si": 0.50, "Fe": 0.20}  # Fe/Si = 0.4 -> log(2) error
    errs = subcompositional_ratio_errors(pred, gt, [("Fe", "Si")])
    assert errs["Fe/Si"] == pytest.approx(math.log(2.0), rel=1e-10)


def test_pair_with_undefined_truth_ratio_returns_nan():
    gt = {"Si": 0.5}  # Fe missing from truth -> Fe/Si undefined
    pred = {"Si": 0.5, "Fe": 0.1}
    errs = subcompositional_ratio_errors(pred, gt, [("Fe", "Si")])
    assert math.isnan(errs["Fe/Si"])


def test_pair_with_zero_predicted_does_not_explode_to_inf():
    """Missing Fe in prediction must produce a finite (large) error, not inf."""
    gt = {"Si": 0.5, "Fe": 0.1}
    pred = {"Si": 0.5}  # Fe absent -> clamped to epsilon
    errs = subcompositional_ratio_errors(pred, gt, [("Fe", "Si")])
    assert math.isfinite(errs["Fe/Si"])
    assert errs["Fe/Si"] > 10.0  # log(0.1 / 1e-12) is much larger than 10


# ---------------------------------------------------------------------------
# THE acceptance criterion: closure-distortion synthetic
# ---------------------------------------------------------------------------


def test_closure_distortion_inflates_concentrations_but_preserves_ratios():
    """Missing-element-bias: VUV-only element undetected -> closure inflates
    detected concentrations proportionally, but ratios between *detected*
    elements stay invariant.

    Construction:
    - True composition: Si=0.30, Fe=0.10, Mg=0.05, Ca=0.02, Al=0.05, O=0.48
      (oxygen carries half the mass — VUV-only, undetectable in optical
      LIBS).
    - Predicted composition: O is missing entirely.  Closure forces the
      detected mass fractions to sum to 1, so each detected element is
      inflated by 1/(1 - 0.48) = 1/0.52 ≈ 1.923.

    The Aitchison distance d_A picks up the closure distortion (which is
    why the global gate complains).  But ALL pairwise ratios between
    detected elements are *exactly* preserved — that's the
    subcompositional-coherence property.

    This test is the reason the metric exists: it shows that a partial
    detection can be physically faithful in ratio space even when the
    closure-driven Aitchison distance suggests otherwise.
    """
    true_comp = {
        "Si": 0.30,
        "Fe": 0.10,
        "Mg": 0.05,
        "Ca": 0.02,
        "Al": 0.05,
        "O": 0.48,  # VUV-only
    }
    # Closure distortion: drop O entirely, renormalize the rest.
    detected = {el: v for el, v in true_comp.items() if el != "O"}
    total = sum(detected.values())
    pred_comp = {el: v / total for el, v in detected.items()}

    # Sanity: the inflation factor is the same for every detected element.
    inflation = pred_comp["Si"] / true_comp["Si"]
    assert inflation == pytest.approx(1 / 0.52, rel=1e-12)
    for el in ("Fe", "Mg", "Ca", "Al"):
        assert pred_comp[el] / true_comp[el] == pytest.approx(inflation, rel=1e-12)

    # The Aitchison distance for the FULL composition is non-zero (closure
    # distortion).  We don't assert its magnitude here (out of scope) — the
    # important property is the ratio invariance below.

    # Pair list loaded from protocol.yaml (NOT hardcoded).
    pairs = load_subcompositional_pairs(PROTOCOL_PATH)
    errs = subcompositional_ratio_errors(pred_comp, true_comp, pairs)

    # Every canonical pair's |log-ratio-error| must be ~0 because the
    # inflation cancels in the ratio.
    for pair_key, err in errs.items():
        assert err == pytest.approx(0.0, abs=1e-12), (
            f"Pair {pair_key} should have zero ratio error under uniform "
            f"closure distortion, got {err}"
        )


def test_ratio_errors_pass_protocol_threshold_when_uniform_inflation():
    """Under closure distortion only, all pairs pass the 0.20 threshold."""
    true_comp = {"Si": 0.30, "Fe": 0.10, "Mg": 0.05, "Ca": 0.02, "Al": 0.05, "O": 0.48}
    detected = {el: v for el, v in true_comp.items() if el != "O"}
    total = sum(detected.values())
    pred_comp = {el: v / total for el, v in detected.items()}

    errs = subcompositional_ratio_errors(pred_comp, true_comp)
    for _, err in errs.items():
        if not math.isnan(err):
            assert err < 0.20, "Closure-distorted-only ratios must pass the 0.20 gate"


def test_ratio_errors_fail_when_one_element_systematically_biased():
    """If one detected element has a true ratio bias (not just closure),
    the corresponding pair must exceed the 0.20 threshold.

    Construction: Fe is over-recovered by 50% relative to Si — a
    physically meaningful ratio failure that closure cannot explain
    away.  log(1.5) ~= 0.405 > 0.20.
    """
    true_comp = {"Si": 0.30, "Fe": 0.10, "Mg": 0.05, "Ca": 0.02, "Al": 0.05}
    pred_comp = dict(true_comp)
    pred_comp["Fe"] = 0.15  # +50% bias on Fe only
    # renormalize so prediction still sums to 1
    total = sum(pred_comp.values())
    pred_comp = {el: v / total for el, v in pred_comp.items()}

    errs = subcompositional_ratio_errors(pred_comp, true_comp)

    # Fe/Si fails (Fe inflated relative to Si)
    assert errs["Fe/Si"] > 0.20

    # Mg/Si, Ca/Si, Al/Si all share the same closure rescaling, so they
    # should be ~equal and small (renormalisation only).  The point is
    # that the metric isolates the Fe/Si failure rather than smearing it
    # across all pairs.
    for pair_key in ("Mg/Si", "Ca/Si", "Al/Si"):
        assert errs[pair_key] < errs["Fe/Si"]


# ---------------------------------------------------------------------------
# Pair list loading via the BenchmarkMetrics integration path
# ---------------------------------------------------------------------------


def test_benchmark_metrics_emits_subcompositional_ratio_errors():
    from cflibs.benchmark.metrics import BenchmarkMetrics

    metrics = BenchmarkMetrics()
    predictions = {
        "Si": [0.30, 0.30],
        "Fe": [0.10, 0.15],  # second spectrum has Fe over-prediction
        "Mg": [0.05, 0.05],
        "Ca": [0.02, 0.02],
        "Al": [0.05, 0.05],
    }
    true_values = {
        "Si": [0.30, 0.30],
        "Fe": [0.10, 0.10],
        "Mg": [0.05, 0.05],
        "Ca": [0.02, 0.02],
        "Al": [0.05, 0.05],
    }
    result = metrics.evaluate(predictions, true_values)
    sub = result.subcompositional_ratio_errors

    # All four canonical pairs must be present
    for key in ("Fe/Si", "Mg/Si", "Ca/Si", "Al/Si"):
        assert key in sub
        assert isinstance(sub[key], list)

    # Spectrum 0: zero errors; spectrum 1: Fe/Si non-zero, others zero.
    fe_si = sub["Fe/Si"]
    assert len(fe_si) == 2
    assert fe_si[0] == pytest.approx(0.0, abs=1e-12)
    assert fe_si[1] == pytest.approx(math.log(1.5), rel=1e-10)

    for key in ("Mg/Si", "Ca/Si", "Al/Si"):
        for val in sub[key]:
            assert val == pytest.approx(0.0, abs=1e-12)
