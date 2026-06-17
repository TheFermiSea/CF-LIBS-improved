"""Pin the ``alias_high_recall`` benchmark workflow contract.

CF-LIBS-improved-knyz: the aa1100_substrate spectra fall out of
identification under the strict alias workflow because the default
``intensity_threshold_factor=3.0`` / ``detection_threshold=0.5``
(the paper C_th, Noel 2025 sec 3.8) reject all candidates. PR #159
added ``ALIASIdentifier(high_recall=True)``
as the opt-in recall preset. This test pins the benchmark wiring that
makes the preset accessible as ``--id-workflows alias_high_recall``.
"""

from __future__ import annotations

from cflibs.benchmark.unified import (
    build_id_workflow_registry,
    _alias_high_recall_workflow_configs,
)


def test_alias_high_recall_is_registered():
    """The new workflow must appear in the registry under its canonical name."""
    registry = build_id_workflow_registry(quick=True)
    assert "alias_high_recall" in registry
    spec = registry["alias_high_recall"]
    assert spec.name == "alias_high_recall"
    assert callable(spec.build_predictor)
    assert len(spec.parameter_grid) >= 1


def test_high_recall_configs_omit_threshold_kwargs():
    """The recall preset (PR #159) is selected ONLY when the caller does
    NOT pass explicit ``intensity_threshold_factor`` / ``detection_threshold``.
    The alias_high_recall predictor constructs the identifier without
    those kwargs, so the config dicts must NOT include them — otherwise
    the explicit values would override the preset and silently restore
    strict mode.
    """
    for quick in (True, False):
        for cfg in _alias_high_recall_workflow_configs(quick=quick):
            assert "intensity_threshold_factor" not in cfg, (
                f"intensity_threshold_factor present in high_recall config "
                f"(quick={quick}): {cfg!r} — this would override "
                "ALIASIdentifier(high_recall=True)'s preset"
            )
            assert "detection_threshold" not in cfg, (
                f"detection_threshold present in high_recall config "
                f"(quick={quick}): {cfg!r} — this would override "
                "ALIASIdentifier(high_recall=True)'s preset"
            )


def test_alias_high_recall_distinct_from_alias():
    """Sanity: the two workflows must be separate registry entries. The
    bug that closed PR #134 was a silent default-change; if these two
    entries collapsed into one, that bug would re-emerge."""
    registry = build_id_workflow_registry(quick=True)
    strict = registry["alias"]
    recall = registry["alias_high_recall"]
    assert strict.build_predictor is not recall.build_predictor
    assert strict.name != recall.name
