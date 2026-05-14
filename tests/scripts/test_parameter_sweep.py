"""Tests for ``scripts/parameter_sweep.py``.

The module under test wraps :mod:`scripts.run_unified_benchmark` and
reuses its parser + helpers verbatim.  We exercise the sweep harness
itself: argument parsing, seed propagation, output-directory layout,
manifest format, and error containment.  The actual benchmark workflows
are mocked at the ``_run_identification_phase``/``_run_composition_phase``
seam so the suite runs in seconds — the JAX cold-start cost we are
trying to amortise would otherwise dominate the wall time of these
tests.

A separate ``@pytest.mark.slow`` test exercises a real run-through to
verify the iter-0-equals-baseline determinism contract.  It is skipped
when the required dataset files are absent.
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
        "scripts.parameter_sweep", SCRIPT_PATH
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
    """Stub out the heavy cflibs imports and validate the sweep skeleton.

    The fixture installs three layers of mocks:

    1. ``_RUB._validate_paths`` / ``_validate_basis_requirements`` — turned
       into no-ops so the test does not need real data/db paths.
    2. ``cflibs.benchmark.load_default_datasets`` — returns a single empty
       dataset dict; the sweep only needs the call to succeed.
    3. ``cflibs.benchmark.UnifiedBenchmarkRunner`` — replaced by a stub
       with the minimum API surface (id_registry, composition_registry,
       run_identification, run_composition, write_outputs).
    4. ``parameter_sweep._RUB._run_identification_phase`` and
       ``_run_composition_phase`` — return canned empty lists.
    5. ``parameter_sweep._RUB._select_datasets`` — returns an empty list
       so the runner's ``parser.error("no datasets")`` branch is bypassed
       (we override the runner's phase helpers anyway).
    6. ``cflibs.benchmark.physical_consistency.aggregate_physical_consistency``
       returns a stub with ``blocked=False`` and a trivial dict.

    The stubs also record per-iter seed observations so the test can
    assert reseeding actually happened with the expected values.
    """

    seeds_observed: list[int] = []

    def _reseed_spy(seed: int):
        seeds_observed.append(int(seed))
        return parameter_sweep._reseed.__wrapped__(seed) if hasattr(
            parameter_sweep._reseed, "__wrapped__"
        ) else _real_reseed(seed)

    _real_reseed = parameter_sweep._reseed
    monkeypatch.setattr(parameter_sweep, "_reseed", _reseed_spy)

    class _StubRunner:
        def __init__(self, *args, **kwargs):
            self.id_registry = {"alias": object()}
            self.composition_registry = {"iterative_jax": object()}
            self._call_count = 0

        def write_outputs(self, output_dir, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            # Emit a small composition_summary so the merge path runs.
            summary = output_dir / "composition_summary.json"
            with summary.open("w") as f:
                json.dump({"placeholder": True}, f)
            return {
                "composition_summary_json": summary,
            }

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

    # Mock cflibs imports at the call site.  parameter_sweep does
    # ``from cflibs.benchmark import UnifiedBenchmarkRunner, load_default_datasets``
    # inside main(); we patch the cflibs module directly.
    fake_benchmark_module = mock.MagicMock()
    fake_benchmark_module.UnifiedBenchmarkRunner = _StubRunner
    fake_benchmark_module.load_default_datasets = mock.MagicMock(
        return_value={"_stub": object()}
    )
    monkeypatch.setitem(sys.modules, "cflibs.benchmark", fake_benchmark_module)

    fake_dataset_module = mock.MagicMock()

    # Build a small enum-like TruthType stand-in.
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

    return {
        "tmp_path": tmp_path,
        "seeds_observed": seeds_observed,
    }


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def test_help_smoke(parameter_sweep):
    parser = parameter_sweep._build_sweep_parser()
    help_text = parser.format_help()
    for flag in ("--config-args", "--n-iters", "--seed-base", "--output-dir"):
        assert flag in help_text


def test_n_iters_must_be_positive(parameter_sweep, tmp_path, capsys):
    rc = None
    with pytest.raises(SystemExit) as exc_info:
        parameter_sweep.main(
            [
                "--n-iters",
                "0",
                "--seed-base",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
                "--config-args",
                "",
            ]
        )
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# End-to-end (mocked) flow
# ---------------------------------------------------------------------------


def test_n_iters_produces_n_output_dirs(parameter_sweep, mocked_sweep):
    out = mocked_sweep["tmp_path"] / "sweep"
    rc = parameter_sweep.main(
        [
            "--n-iters",
            "3",
            "--seed-base",
            "100",
            "--output-dir",
            str(out),
            "--config-args",
            "--id-workflows alias --composition-workflows iterative_jax",
        ]
    )
    assert rc == 0, f"sweep returned non-zero exit code {rc}"

    iter_dirs = sorted(out.glob("iter-*"))
    assert len(iter_dirs) == 3
    assert [p.name for p in iter_dirs] == ["iter-000", "iter-001", "iter-002"]


def test_manifest_jsonl_is_valid_and_has_one_line_per_iter(
    parameter_sweep, mocked_sweep
):
    out = mocked_sweep["tmp_path"] / "sweep"
    parameter_sweep.main(
        [
            "--n-iters",
            "4",
            "--seed-base",
            "7",
            "--output-dir",
            str(out),
            "--config-args",
            "--id-workflows alias --composition-workflows iterative_jax",
        ]
    )

    manifest = out / "manifest.jsonl"
    assert manifest.exists()
    lines = [
        line for line in manifest.read_text().splitlines() if line.strip()
    ]
    assert len(lines) == 4
    records = [json.loads(line) for line in lines]

    # Every record must carry the four required fields.
    for record in records:
        for required in (
            "iter",
            "seed",
            "status",
            "wall_time_seconds",
            "output_dir",
            "config_args",
        ):
            assert required in record, f"manifest record missing {required}: {record}"

    # iter indices monotonically increase.
    assert [r["iter"] for r in records] == [0, 1, 2, 3]

    # seeds = seed_base + iter index.
    assert [r["seed"] for r in records] == [7, 8, 9, 10]


def test_seeds_are_different_per_iter(parameter_sweep, mocked_sweep):
    out = mocked_sweep["tmp_path"] / "sweep"
    parameter_sweep.main(
        [
            "--n-iters",
            "5",
            "--seed-base",
            "42",
            "--output-dir",
            str(out),
            "--config-args",
            "--id-workflows alias --composition-workflows iterative_jax",
        ]
    )

    seeds = mocked_sweep["seeds_observed"]
    assert seeds == [42, 43, 44, 45, 46]
    assert len(set(seeds)) == len(seeds), "seeds must all differ"


def test_seed_base_is_applied(parameter_sweep, mocked_sweep):
    out = mocked_sweep["tmp_path"] / "sweep"
    parameter_sweep.main(
        [
            "--n-iters",
            "2",
            "--seed-base",
            "999",
            "--output-dir",
            str(out),
            "--config-args",
            "--id-workflows alias --composition-workflows iterative_jax",
        ]
    )
    assert mocked_sweep["seeds_observed"] == [999, 1000]


def test_manifest_records_output_dirs(parameter_sweep, mocked_sweep):
    out = mocked_sweep["tmp_path"] / "sweep"
    parameter_sweep.main(
        [
            "--n-iters",
            "2",
            "--seed-base",
            "1",
            "--output-dir",
            str(out),
            "--config-args",
            "--id-workflows alias --composition-workflows iterative_jax",
        ]
    )
    records = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for i, rec in enumerate(records):
        expected_dir = out.resolve() / f"iter-{i:03d}"
        assert Path(rec["output_dir"]).resolve() == expected_dir


def test_physical_consistency_block_propagates_exit_code(
    parameter_sweep, mocked_sweep, monkeypatch
):
    """If aggregate_physical_consistency returns blocked=True we must exit 2."""
    pc_report = mock.MagicMock()
    pc_report.blocked = True
    pc_report.to_dict.return_value = {"blocked": True, "checks": []}
    sys.modules["cflibs.benchmark.physical_consistency"].aggregate_physical_consistency = (
        mock.MagicMock(return_value=pc_report)
    )

    out = mocked_sweep["tmp_path"] / "sweep"
    rc = parameter_sweep.main(
        [
            "--n-iters",
            "2",
            "--seed-base",
            "1",
            "--output-dir",
            str(out),
            "--config-args",
            "--id-workflows alias --composition-workflows iterative_jax",
        ]
    )
    assert rc == 2

    records = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert all(
        rec["status"] == "physical_consistency_blocked" for rec in records
    )


def test_iter_failures_do_not_kill_sweep(
    parameter_sweep, mocked_sweep, monkeypatch
):
    """An exception inside one iteration must be captured in the manifest but
    must NOT prevent subsequent iterations from running."""
    call_count = {"n": 0}

    def flaky_id_phase(parser, runner, datasets, ids, max_folds):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("synthetic failure in iter 1")
        return ([], [])

    monkeypatch.setattr(
        parameter_sweep._RUB, "_run_identification_phase", flaky_id_phase
    )

    out = mocked_sweep["tmp_path"] / "sweep"
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
    # Only one of three iters errored, so exit code is 0 (the
    # "all-errored" exit-1 branch is not taken).
    assert rc == 0

    records = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    statuses = [r["status"] for r in records]
    assert statuses == ["ok", "error", "ok"]
    assert "synthetic failure in iter 1" in records[1]["error"]


# ---------------------------------------------------------------------------
# Reseed function behavior
# ---------------------------------------------------------------------------


def test_reseed_makes_numpy_deterministic(parameter_sweep):
    import numpy as np

    parameter_sweep._reseed(123)
    a = np.random.random(5)
    parameter_sweep._reseed(123)
    b = np.random.random(5)
    np.testing.assert_array_equal(a, b)

    parameter_sweep._reseed(124)
    c = np.random.random(5)
    assert not np.allclose(a, c), "different seed must produce different samples"


def test_reseed_makes_python_random_deterministic(parameter_sweep):
    import random

    parameter_sweep._reseed(7)
    a = [random.random() for _ in range(3)]
    parameter_sweep._reseed(7)
    b = [random.random() for _ in range(3)]
    assert a == b

    parameter_sweep._reseed(8)
    c = [random.random() for _ in range(3)]
    assert a != c


def test_reseed_returns_jax_key_when_jax_available(parameter_sweep):
    try:
        import jax  # noqa: F401
    except Exception:
        pytest.skip("jax not available")
    key = parameter_sweep._reseed(42)
    assert key is not None
    # Re-seeding with same seed must produce the same key.
    key2 = parameter_sweep._reseed(42)
    import numpy as np

    np.testing.assert_array_equal(np.asarray(key), np.asarray(key2))


# ---------------------------------------------------------------------------
# Slow integration test — verifies determinism against a real one-shot run.
# ---------------------------------------------------------------------------


def _has_required_data_assets() -> bool:
    """Return True only if the minimum data needed for a real benchmark run
    is available.  Used by the slow integration test to self-skip."""
    candidates_db = [
        REPO_ROOT / "ASD_da" / "libs_production.db",
        REPO_ROOT / "libs_production.db",
    ]
    if not any(p.exists() for p in candidates_db):
        return False
    data_dir = REPO_ROOT / "data"
    if not data_dir.exists():
        return False
    # Need at least aalto_libs (smallest composition-capable dataset).
    if not (data_dir / "aalto_libs").exists():
        return False
    return True


@pytest.mark.slow
@pytest.mark.requires_db
@pytest.mark.requires_jax
@pytest.mark.skipif(
    not _has_required_data_assets(),
    reason="benchmark data assets missing; cannot run determinism test",
)
def test_iter_zero_matches_one_shot_baseline(tmp_path):
    """Iter 0 of the sweep with seed S must match a single
    ``run_unified_benchmark.py`` invocation seeded with S, within
    ``rtol=1e-5, atol=1e-8`` on all numeric values inside
    ``composition_summary.json``.

    This is the core reproducibility contract.  It guards against
    accidental state leakage (e.g., a workflow that lazily mutates
    its config dict) that would break the in-process equivalence.
    """
    parameter_sweep = _load_parameter_sweep_module()

    config_args = (
        "--quick --max-outer-folds 1 --sections all "
        "--id-workflows alias --composition-workflows iterative_jax "
        "--vrabel-max-shots 1"
    )

    # Baseline: one-shot invocation, run via the same module (in-process).
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    import scripts.run_unified_benchmark as rub  # noqa: WPS433

    # Re-seed exactly as the sweep does to put us on the same RNG footing.
    parameter_sweep._reseed(1)
    rc_baseline = rub.main(
        config_args.split() + ["--output-dir", str(baseline_dir)]
    )
    # rc_baseline can be 0 or 2 (PC gate); both are acceptable here as
    # long as the artifacts exist.
    assert rc_baseline in (0, 2)
    assert (baseline_dir / "composition_summary.json").exists()
    baseline_summary = json.loads(
        (baseline_dir / "composition_summary.json").read_text()
    )

    # Sweep: 1 iter, same seed.
    sweep_dir = tmp_path / "sweep"
    rc_sweep = parameter_sweep.main(
        [
            "--n-iters",
            "1",
            "--seed-base",
            "1",
            "--output-dir",
            str(sweep_dir),
            "--config-args",
            config_args,
        ]
    )
    assert rc_sweep in (0, 2)
    iter0_summary = json.loads(
        (sweep_dir / "iter-000" / "composition_summary.json").read_text()
    )

    _assert_summaries_match(baseline_summary, iter0_summary)


def _assert_summaries_match(
    baseline: Any, candidate: Any, path: str = ""
) -> None:
    """Recursively compare two JSON-loaded summary structures.

    Numeric scalars must match within ``rtol=1e-5, atol=1e-8``.  Strings
    and bools must compare equal.  Lists must have the same length and
    align elementwise.  Dicts must share the same keyset.

    The path string is included in assertion messages to aid debugging.
    """
    import math

    if isinstance(baseline, dict):
        assert isinstance(candidate, dict), f"{path}: type mismatch dict vs {type(candidate)}"
        # Ignore wall-clock-style keys that legitimately differ run-to-run.
        # `latency_*` are aggregates (np.mean / np.percentile over
        # elapsed_seconds) that inherit the per-record timing noise — the
        # raw `elapsed_seconds` was already in the ignore set; the bug was
        # that the aggregates derived from it were not (CF-LIBS-improved-5w9f).
        ignore_keys = {
            "wall_time_seconds",
            "elapsed_seconds",
            "timestamp",
            "latency_mean_s",
            "latency_p95_s",
            "latency_p99_s",
            "latency_max_s",
        }
        keys_b = set(baseline) - ignore_keys
        keys_c = set(candidate) - ignore_keys
        assert keys_b == keys_c, f"{path}: key set differs {keys_b ^ keys_c}"
        for k in keys_b:
            _assert_summaries_match(baseline[k], candidate[k], f"{path}.{k}")
    elif isinstance(baseline, list):
        assert isinstance(candidate, list), f"{path}: type mismatch list vs {type(candidate)}"
        assert len(baseline) == len(candidate), (
            f"{path}: list length differs ({len(baseline)} vs {len(candidate)})"
        )
        for i, (b, c) in enumerate(zip(baseline, candidate)):
            _assert_summaries_match(b, c, f"{path}[{i}]")
    elif isinstance(baseline, bool) or isinstance(candidate, bool):
        assert baseline == candidate, f"{path}: {baseline} != {candidate}"
    elif isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)):
        if math.isnan(baseline) and math.isnan(candidate):
            return
        assert math.isclose(
            baseline, candidate, rel_tol=1e-5, abs_tol=1e-8
        ), f"{path}: {baseline} not close to {candidate}"
    else:
        assert baseline == candidate, f"{path}: {baseline!r} != {candidate!r}"
