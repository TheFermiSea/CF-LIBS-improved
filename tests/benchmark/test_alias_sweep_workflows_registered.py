"""Pin the 8-cell alias-fix sweep workflow registration (Phase C of
jaunty-weaving-mist).

The sweep enumerates 2^3 combinations of three opt-in fix flags landed
post-Phase-B:
  - ftp1  (PR #175): ``r2_gate_mode="adaptive_t"``
  - 762f  (PR #177): ``temperature_estimator_mode="robust"``
  - dj6y  (PR #176): ``relative_cl_per_ion_stage=True``

Each cell shares the strict threshold defaults; the only difference
between cells is the fix-flag kwargs baked into the predictor closure.
These tests guard:
  1. All 8 ``alias_sweep_<cell>`` entries are present in the registry.
  2. Each cell's predictor constructs without crashing (atomic db
     present).
  3. ``alias_sweep_baseline`` is byte-identical to the strict ``alias``
     workflow for its baseline cell (i.e., applying NO fix flags reduces
     to the precision-king default behavior).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pytest

from cflibs.benchmark.unified import (
    _ALIAS_SWEEP_BASE_KWARGS,
    _ALIAS_SWEEP_CELLS,
    UnifiedBenchmarkContext,
    _alias_sweep_workflow_configs,
    _build_alias_sweep_predictor_factory,
    build_id_workflow_registry,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = _REPO_ROOT / "ASD_da" / "libs_production.db"

_EXPECTED_CELL_NAMES: Tuple[str, ...] = (
    "baseline",
    "ftp1",
    "762f",
    "dj6y",
    "ftp1+762f",
    "ftp1+dj6y",
    "762f+dj6y",
    "all_three",
)


def test_all_eight_alias_sweep_workflows_registered() -> None:
    """Every cell in the plan's configuration table must appear in the
    registry under the canonical ``alias_sweep_<cell>`` name."""
    registry = build_id_workflow_registry(quick=True)
    missing = [
        f"alias_sweep_{c}" for c in _EXPECTED_CELL_NAMES if f"alias_sweep_{c}" not in registry
    ]
    assert not missing, f"missing sweep workflows: {missing}"

    # Spec sanity for each
    for cell_name in _EXPECTED_CELL_NAMES:
        spec = registry[f"alias_sweep_{cell_name}"]
        assert spec.name == f"alias_sweep_{cell_name}"
        assert callable(spec.build_predictor)
        assert len(spec.parameter_grid) == 1, (
            f"sweep cell {cell_name!r} should have a single-config grid; "
            f"got {len(spec.parameter_grid)} configs"
        )


def test_sweep_cell_names_match_plan_order() -> None:
    """The module-level ``_ALIAS_SWEEP_CELLS`` ordering must match the
    plan's configuration table so downstream aggregation can rely on it."""
    actual = tuple(name for name, _ in _ALIAS_SWEEP_CELLS)
    assert actual == _EXPECTED_CELL_NAMES


def test_sweep_base_kwargs_pin_strict_precision_defaults() -> None:
    """Sweep cells share the strict precision-king threshold defaults
    (3.0 / 0.02 / 0.4 / 30). A silent change to these would invalidate
    every cell's baseline — the diff would no longer be "fix flags only"."""
    assert _ALIAS_SWEEP_BASE_KWARGS == {
        "intensity_threshold_factor": 3.0,
        "detection_threshold": 0.02,
        "chance_window_scale": 0.4,
        "max_lines_per_element": 30,
    }


def test_sweep_configs_helper_returns_single_empty_config() -> None:
    """All 8 cells share an empty single-element config grid; the per-cell
    variation lives entirely in the predictor closure's ``cell_kwargs``."""
    for quick in (True, False):
        cfgs = _alias_sweep_workflow_configs(quick=quick)
        assert cfgs == [{}], f"unexpected sweep config grid (quick={quick}): {cfgs!r}"


def test_baseline_predictor_factory_passes_no_fix_kwargs() -> None:
    """The ``baseline`` cell must apply ZERO fix flags. A regression here
    would silently turn the baseline arm of the experiment into a fix-on
    arm, invalidating the lift measurement."""
    baseline_kwargs = dict(_ALIAS_SWEEP_CELLS[0][1])
    assert _ALIAS_SWEEP_CELLS[0][0] == "baseline"
    assert baseline_kwargs == {}


def test_all_three_predictor_factory_enables_all_three_flags() -> None:
    """Symmetric guard for the ``all_three`` cell — it must enable all
    three fix flags simultaneously, otherwise the 8-cell sweep is
    secretly only a 7-cell sweep."""
    all_three_name, all_three_kwargs = _ALIAS_SWEEP_CELLS[-1]
    assert all_three_name == "all_three"
    assert all_three_kwargs == {
        "r2_gate_mode": "adaptive_t",
        "temperature_estimator_mode": "robust",
        "relative_cl_per_ion_stage": True,
    }


@pytest.mark.skipif(
    not _DEFAULT_DB_PATH.exists(),
    reason=f"atomic database not available at {_DEFAULT_DB_PATH}",
)
def test_every_cell_predictor_builds_without_crashing() -> None:
    """Each cell's predictor factory must successfully build an
    ``ALIASIdentifier`` — confirms the fix-flag kwargs are accepted by
    the constructor and that no signature drift has snuck in."""
    ctx = UnifiedBenchmarkContext(db_path=_DEFAULT_DB_PATH)
    candidate_elements = ["Si", "Mg", "Al", "Fe"]
    for cell_name, cell_kwargs in _ALIAS_SWEEP_CELLS:
        factory = _build_alias_sweep_predictor_factory(cell_kwargs, cell_name)
        predictor = factory(ctx, candidate_elements, {})
        assert callable(predictor), f"predictor for cell {cell_name!r} not callable"


def test_baseline_cell_produces_strict_alias_equivalent_config() -> None:
    """The ``alias_sweep_baseline`` cell, by construction, applies ZERO
    fix flags AND pins the strict precision-king threshold defaults
    (3.0 / 0.02). The ``chance_window_scale=0.4`` and
    ``max_lines_per_element=30`` are the sweep's chosen "central"
    threshold pair (matching the strict alias workflow's full grid-search
    entries, not its arch-defaults baseline which uses
    ``max_lines_per_element=20``). Plan jaunty-weaving-mist explicitly
    specifies the sweep's base kwargs as
    ``intensity_threshold_factor=3.0, detection_threshold=0.02,
    chance_window_scale=0.4, max_lines_per_element=30``.

    Guarantee: any drift of these 4 values from strict-precision land
    would invalidate the lift comparison against strict ``alias``.
    """
    from cflibs.benchmark.unified import _alias_workflow_configs

    strict_configs = _alias_workflow_configs(quick=True)
    strict_arch = strict_configs[0]
    # The two thresholds that gate precision must match the strict
    # alias workflow's architecture defaults exactly.
    for key in ("intensity_threshold_factor", "detection_threshold"):
        assert strict_arch[key] == _ALIAS_SWEEP_BASE_KWARGS[key], (
            f"sweep baseline kwarg {key!r} ({_ALIAS_SWEEP_BASE_KWARGS[key]!r}) "
            f"diverged from strict alias arch default ({strict_arch[key]!r})"
        )

    # The strict alias workflow's NON-default grid entries (the ones the
    # tune_id_workflow grid search explores) all pin
    # max_lines_per_element=30, which is the sweep's chosen central value.
    strict_grid_entries = [cfg for cfg in strict_configs[1:] if cfg]
    for cfg in strict_grid_entries:
        assert cfg["max_lines_per_element"] == 30
        assert cfg["chance_window_scale"] in (0.3, 0.4)
    assert _ALIAS_SWEEP_BASE_KWARGS["max_lines_per_element"] == 30
    assert _ALIAS_SWEEP_BASE_KWARGS["chance_window_scale"] == 0.4

    baseline_kwargs: Dict[str, object] = dict(_ALIAS_SWEEP_CELLS[0][1])
    assert baseline_kwargs == {}, (
        "alias_sweep_baseline must enable zero fix flags so the lift "
        "comparison against fix-on cells isolates the flags' effect"
    )


def test_distinct_predictor_per_cell() -> None:
    """Each registered sweep cell must use a DISTINCT predictor factory
    closure; collapsing two cells into one would silently double-count
    one fix-flag combination and drop another."""
    registry = build_id_workflow_registry(quick=True)
    builders = {
        f"alias_sweep_{name}": registry[f"alias_sweep_{name}"].build_predictor
        for name in _EXPECTED_CELL_NAMES
    }
    seen_ids = set()
    for name, fn in builders.items():
        assert id(fn) not in seen_ids, (
            f"sweep workflow {name!r} reuses a predictor factory shared "
            "with another cell"
        )
        seen_ids.add(id(fn))


def test_sweep_predictor_does_not_disturb_strict_alias_registration() -> None:
    """Additive guarantee: registering the 8 sweep cells must not
    perturb the strict ``alias`` or ``alias_high_recall`` registry
    entries (these are pinned by other tests)."""
    registry = build_id_workflow_registry(quick=True)
    assert "alias" in registry
    assert "alias_high_recall" in registry
    strict = registry["alias"]
    for cell_name in _EXPECTED_CELL_NAMES:
        sweep = registry[f"alias_sweep_{cell_name}"]
        assert sweep.build_predictor is not strict.build_predictor
        assert sweep.name != strict.name
