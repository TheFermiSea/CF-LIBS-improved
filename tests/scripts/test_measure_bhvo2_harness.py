"""
Harness-bridge contract for scripts/measure_bhvo2_presence.py (bead vj82).

The BHVO-2 gate script used to hand-build its own detection + solver
pipeline; for identical knobs it produced different numbers than the CLI
(script RMSE 4.03 wt% vs CLI ~5.8 wt% on ChemCam BHVO-2), making the two
harnesses incomparable. These tests pin the bridge:

1. The script builds its configuration through the SHARED builder
   (``cflibs.inversion.pipeline.build_pipeline_config``) and the resulting
   dataclass is equal to a CLI-built one for the same knobs.
2. The divergence knobs (wavelength calibration, shift-coherence veto,
   confounder element list) round-trip through explicit flags whose defaults
   are the CLI defaults.

No atomic database or spectrum is needed: everything here is configuration
resolution.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cflibs.benchmark.reference_compositions import BHVO2_BASALT_USGS
from cflibs.inversion.pipeline import AnalysisPipelineConfig, build_pipeline_config

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "measure_bhvo2_presence.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("measure_bhvo2_presence", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(harness, argv):
    return harness.build_pipeline_from_args(harness.parse_args(argv))


# ---------------------------------------------------------------------------
# 1. Shared-builder contract
# ---------------------------------------------------------------------------


def test_flagless_harness_equals_cli_builder_defaults(harness):
    """Default script knobs == a flagless CLI ``analyze`` for the same elements."""
    cfg = _build(harness, [])
    cert = list(BHVO2_BASALT_USGS)
    expected_elements = cert + [e for e in harness.CONFOUNDERS if e not in cert]
    assert isinstance(cfg, AnalysisPipelineConfig)
    # The exact dataclass the CLI would resolve for the same element list.
    assert cfg == build_pipeline_config(expected_elements)


# ---------------------------------------------------------------------------
# 2. Knob flag round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, attr, expected",
    [
        (["--no-wavelength-calibration"], "wavelength_calibration", False),
        (["--wavelength-calibration"], "wavelength_calibration", True),
        (["--no-shift-coherence-veto"], "shift_coherence_veto", False),
        (["--shift-coherence-veto"], "shift_coherence_veto", True),
        (["--no-saha-boltzmann-graph"], "saha_boltzmann_graph", False),
        (["--apply-self-absorption"], "apply_self_absorption", "observable"),
        (["--sa-mode", "observable"], "apply_self_absorption", "observable"),
        (["--exclude-resonance", "true"], "exclude_resonance", True),
        (["--exclude-resonance", "false"], "exclude_resonance", False),
        (["--exclude-resonance", "auto"], "exclude_resonance", None),
        (["--closure-mode", "standard"], "closure_mode", "standard"),
        (["--min-relative-intensity", "100.0"], "min_relative_intensity", 100.0),
        (["--min-relative-intensity", "none"], "min_relative_intensity", None),
    ],
)
def test_knob_flags_round_trip(harness, argv, attr, expected):
    assert getattr(_build(harness, argv), attr) == expected


def test_confounders_flag_controls_element_list(harness):
    cert = list(BHVO2_BASALT_USGS)
    with_conf = _build(harness, [])
    without_conf = _build(harness, ["--no-confounders"])
    assert without_conf.elements == cert
    assert with_conf.elements == cert + [e for e in harness.CONFOUNDERS if e not in cert]
    # Everything except the element list must be identical.
    assert with_conf == build_pipeline_config(with_conf.elements)
    assert without_conf == build_pipeline_config(without_conf.elements)


def test_ablation_toggles_only_change_their_knob(harness):
    """One-factor ablation flags must not perturb any other knob."""
    base = _build(harness, [])
    for argv, attr in [
        (["--no-wavelength-calibration"], "wavelength_calibration"),
        (["--no-shift-coherence-veto"], "shift_coherence_veto"),
    ]:
        toggled = _build(harness, argv)
        diffs = {
            f.name
            for f in base.__dataclass_fields__.values()
            if getattr(base, f.name) != getattr(toggled, f.name)
        }
        assert diffs == {attr}
