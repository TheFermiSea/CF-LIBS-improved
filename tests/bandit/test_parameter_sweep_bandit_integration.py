"""Integration tests for the bandit-wiring in ``scripts/parameter_sweep.py`` (T2.3).

The heavy benchmark machinery is stubbed exactly like
``tests/scripts/test_parameter_sweep.py``: ``UnifiedBenchmarkRunner``,
the cflibs imports, the workflow phases and ``aggregate_physical_consistency``
are all replaced by lightweight mocks that produce a synthetic
``composition_summary.json`` file containing a controllable
``mean_aitchison`` per cell so the bandit-update path is exercised end
to end.

We focus on:

1. ``--bandit 0`` + no ``--cells`` → manifest records contain no
   ``arm_id`` / ``posterior_*`` keys (preserves T1.1 byte-shape).
2. ``--bandit 2`` + 2-cell ``--cells`` → manifest records carry
   ``arm_id``, ``cell_name``, ``phase``, ``d_a``, ``posterior_mean``,
   ``posterior_var``, ``prob_best``, and a ``bandit_summary.json`` is
   written alongside the manifest.
3. The bandit allocates more pulls to the cell with the lower synthetic
   d_A.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "parameter_sweep.py"


def _load_parameter_sweep_module():
    """Load ``scripts/parameter_sweep.py`` as an importable module."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "scripts.parameter_sweep_for_bandit_tests", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def parameter_sweep():
    return _load_parameter_sweep_module()


@pytest.fixture
def mocked_sweep(parameter_sweep, monkeypatch, tmp_path):
    """Patch the heavy cflibs surfaces with a controllable stub.

    The stub writes a ``composition_summary.json`` whose
    ``mean_aitchison`` is determined by the ``cell_name_for_iter``
    closure — so each iter's "true" d_A is a deterministic function of
    the cell that was selected for that iter.
    """
    # The stub uses this closure to know which cell name is currently
    # being executed; the parameter_sweep main loop calls
    # _run_one_iteration which calls runner.write_outputs through
    # _RUB._run_composition_phase.  We instead patch the phase to a
    # no-op and have write_outputs read the cell from a module-level
    # slot we update per-iter via a wrapper around
    # _run_one_iteration below.
    state: dict[str, Any] = {"cell_name": "default"}

    class _StubRunner:
        def __init__(self, *args, **kwargs):
            self.id_registry = {"alias": object(), "spectral_nnls": object()}
            self.composition_registry = {
                "iterative_jax": object(),
                "iterative": object(),
            }

        def write_outputs(self, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = output_dir / "composition_summary.json"
            cell = state["cell_name"]
            # Synthetic d_A schedule keyed by cell name.
            d_a = {
                "good": 0.05,
                "bad": 0.20,
                "alias_jax": 0.05,
                "alias_iter": 0.20,
                "default": 0.10,
            }.get(cell, 0.15)
            summary = {
                f"alias__iterative_jax": {
                    "mean_aitchison": d_a,
                    "n_spectra": 3,
                }
            }
            with summary_path.open("w") as f:
                json.dump(summary, f)
            return {"composition_summary_json": summary_path}

    monkeypatch.setattr(
        parameter_sweep._RUB,
        "_validate_paths",
        lambda parser, args: None,
    )
    monkeypatch.setattr(
        parameter_sweep._RUB,
        "_validate_basis_requirements",
        lambda parser, ids, basis_dir: None,
    )
    monkeypatch.setattr(
        parameter_sweep._RUB,
        "_run_identification_phase",
        lambda parser, runner, datasets, ids, max_folds: ([], []),
    )
    monkeypatch.setattr(
        parameter_sweep._RUB,
        "_run_composition_phase",
        lambda parser, runner, datasets, ids, comps, max_folds: ([], []),
    )
    monkeypatch.setattr(
        parameter_sweep._RUB,
        "_select_datasets",
        lambda datasets, truth_types: [],
    )
    monkeypatch.setattr(
        parameter_sweep._RUB,
        "_resolve_basis_dir",
        lambda cli_value: tmp_path / "basis",
    )

    # cflibs stubs.
    fake_benchmark_module = mock.MagicMock()
    fake_benchmark_module.UnifiedBenchmarkRunner = _StubRunner
    fake_benchmark_module.load_default_datasets = mock.MagicMock(
        return_value={"_stub": object()}
    )
    monkeypatch.setitem(sys.modules, "cflibs.benchmark", fake_benchmark_module)

    fake_dataset_module = mock.MagicMock()

    class _TT:
        class _V:
            def __init__(self, v):
                self.value = v

        ASSAY = _V("assay")
        FORMULA_PROXY = _V("formula_proxy")
        SYNTHETIC = _V("synthetic")
        BLIND = _V("blind")

    fake_dataset_module.TruthType = _TT
    monkeypatch.setitem(sys.modules, "cflibs.benchmark.dataset", fake_dataset_module)

    fake_pc_module = mock.MagicMock()
    pc_report = mock.MagicMock()
    pc_report.blocked = False
    pc_report.to_dict.return_value = {"blocked": False, "checks": []}
    fake_pc_module.aggregate_physical_consistency = mock.MagicMock(
        return_value=pc_report
    )
    monkeypatch.setitem(
        sys.modules, "cflibs.benchmark.physical_consistency", fake_pc_module
    )

    # Wrap _run_one_iteration so the stub-runner knows which cell name
    # is "active" for the current iter when it writes
    # composition_summary.json.
    original = parameter_sweep._run_one_iteration

    def _wrapped(*args, **kwargs):
        base_args = kwargs.get("base_args")
        # Derive the cell name from the unique composition_workflows
        # we set on each cell's argparse Namespace.  This is the only
        # piece of per-cell state propagated into _run_one_iteration.
        comp = list(kwargs.get("composition_workflows", []))
        if comp == ["iterative_jax"]:
            state["cell_name"] = "good"
        elif comp == ["iterative"]:
            state["cell_name"] = "bad"
        else:
            state["cell_name"] = "default"
        return original(*args, **kwargs)

    monkeypatch.setattr(parameter_sweep, "_run_one_iteration", _wrapped)

    return {"tmp_path": tmp_path, "state": state}


# ---------------------------------------------------------------------------
# --bandit 0 byte-shape regression
# ---------------------------------------------------------------------------


def test_bandit_zero_produces_no_arm_fields_in_manifest(
    parameter_sweep, mocked_sweep, tmp_path
):
    out = tmp_path / "sweep"
    rc = parameter_sweep.main(
        [
            "--n-iters",
            "3",
            "--seed-base",
            "1",
            "--output-dir",
            str(out),
            "--config-args",
            "--id-workflows alias --composition-workflows iterative_jax",
        ]
    )
    assert rc == 0
    manifest_lines = (out / "manifest.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 3
    for line in manifest_lines:
        record = json.loads(line)
        # T1.1 byte-shape: NO bandit fields when --bandit 0.
        forbidden = {
            "arm_id",
            "cell_id",
            "cell_name",
            "cell_config_args",
            "phase",
            "d_a",
            "posterior_mean",
            "posterior_var",
            "prob_best",
            "n_pulls",
            "arm_posteriors",
        }
        leaked = forbidden & record.keys()
        assert not leaked, (
            f"--bandit 0 leaked bandit-only fields into manifest: {leaked}"
        )
    # No bandit_summary.json should have been written either.
    assert not (out / "bandit_summary.json").exists()


# ---------------------------------------------------------------------------
# --bandit > 0 wiring
# ---------------------------------------------------------------------------


def test_bandit_enabled_emits_arm_id_and_posterior_fields(
    parameter_sweep, mocked_sweep, tmp_path
):
    cells_json = tmp_path / "cells.json"
    cells_json.write_text(
        json.dumps(
            [
                {
                    "name": "good",
                    "config_args": "--id-workflows alias --composition-workflows iterative_jax",
                },
                {
                    "name": "bad",
                    "config_args": "--id-workflows alias --composition-workflows iterative",
                },
            ]
        )
    )

    out = tmp_path / "sweep"
    rc = parameter_sweep.main(
        [
            "--n-iters",
            "10",
            "--seed-base",
            "7",
            "--output-dir",
            str(out),
            "--cells",
            str(cells_json),
            "--bandit",
            "2",
            "--bandit-seed",
            "1234",
        ]
    )
    assert rc == 0
    lines = (out / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 10
    records = [json.loads(line) for line in lines]
    # Required fields exist on every record under --bandit > 0.
    required = {
        "arm_id",
        "cell_id",
        "cell_name",
        "cell_config_args",
        "phase",
        "d_a",
        "posterior_mean",
        "posterior_var",
        "prob_best",
        "n_pulls",
        "arm_posteriors",
    }
    for rec in records:
        missing = required - rec.keys()
        assert not missing, f"missing fields {missing} in {rec}"

    # Warmup: round-robin so first 4 iters are arms [0,1,0,1].
    assert [r["arm_id"] for r in records[:4]] == [0, 1, 0, 1]
    assert all(r["phase"] == "warmup" for r in records[:4])
    assert all(r["phase"] == "bandit" for r in records[4:])

    # The good cell (lower synthetic d_A = 0.05) should be pulled more
    # than the bad cell (0.20) in the post-warmup phase.
    post = records[4:]
    good_pulls = sum(1 for r in post if r["cell_name"] == "good")
    bad_pulls = sum(1 for r in post if r["cell_name"] == "bad")
    assert good_pulls > bad_pulls, (
        f"Bandit did not concentrate on good arm: good={good_pulls} bad={bad_pulls}"
    )

    # bandit_summary.json should exist with both cells.
    summary_path = out / "bandit_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert set(summary.keys()) == {
        "cells",
        "pulls_per_cell",
        "posterior_summary",
        "warmup_per_cell",
        "total_iters",
    }
    assert summary["warmup_per_cell"] == 2
    assert summary["total_iters"] == 10
    assert sum(summary["pulls_per_cell"]) == 10


def test_bandit_d_a_is_recorded_from_composition_summary(
    parameter_sweep, mocked_sweep, tmp_path
):
    """Confirms each iter's d_a is the mean_aitchison written by the stub runner."""
    cells_json = tmp_path / "cells.json"
    cells_json.write_text(
        json.dumps(
            [
                {
                    "name": "good",
                    "config_args": "--id-workflows alias --composition-workflows iterative_jax",
                },
                {
                    "name": "bad",
                    "config_args": "--id-workflows alias --composition-workflows iterative",
                },
            ]
        )
    )

    out = tmp_path / "sweep"
    rc = parameter_sweep.main(
        [
            "--n-iters",
            "4",
            "--seed-base",
            "9",
            "--output-dir",
            str(out),
            "--cells",
            str(cells_json),
            "--bandit",
            "2",
        ]
    )
    assert rc == 0
    records = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text().splitlines()
    ]
    # Warmup [0,1,0,1] means [good, bad, good, bad].
    expected_d_a = {"good": 0.05, "bad": 0.20}
    for rec in records:
        assert rec["d_a"] == pytest.approx(expected_d_a[rec["cell_name"]])
