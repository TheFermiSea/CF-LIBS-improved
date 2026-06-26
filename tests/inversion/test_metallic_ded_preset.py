"""metallic_ded preset (DED-PLAN step 2) + preset-as-tier-4 knob resolution.

The DED constrained-known-set preset must relax the per-element selection
floors (so a faint minor element is never dropped) and arm the degeneracy
guard for small K, while leaving the geological/metallic presets untouched.
"""

from cflibs.inversion.pipeline import ANALYSIS_PRESETS, build_pipeline_config


def test_metallic_ded_preset_registered():
    assert "metallic_ded" in ANALYSIS_PRESETS
    p = ANALYSIS_PRESETS["metallic_ded"]
    assert p["closure_mode"] == "standard"  # metal sum-to-one, not oxide
    assert p["stark_ne"] is True


def test_metallic_ded_relaxes_selection_floors():
    p = build_pipeline_config(["Ti", "Al", "V"], preset="metallic_ded")
    assert p.closure_mode == "standard"
    assert p.min_lines_per_element == 1  # accept weak minor-element signal
    assert p.degeneracy_min_elements == 2  # guard armed for K>=2 (default 4)
    assert p.min_snr == 5.0
    assert p.top_k_per_element == 40
    assert p.max_lines_per_element == 30
    assert p.degeneracy_dominance_threshold == 0.95  # Ti legitimately ~90 wt%


def test_geological_and_metallic_presets_unchanged():
    g = build_pipeline_config(["Fe"], preset="geological")
    assert g.closure_mode == "oxide"
    assert g.min_lines_per_element == 3
    assert g.top_k_per_element == 60
    assert g.min_snr == 10.0
    m = build_pipeline_config(["Fe"], preset="metallic")
    assert m.closure_mode == "standard"
    assert m.min_lines_per_element == 3  # NOT relaxed (only metallic_ded relaxes)


def test_overrides_still_beat_preset():
    p = build_pipeline_config(
        ["Ti", "Al", "V"], preset="metallic_ded", overrides={"min_lines_per_element": 5}
    )
    assert p.min_lines_per_element == 5  # overrides > preset
