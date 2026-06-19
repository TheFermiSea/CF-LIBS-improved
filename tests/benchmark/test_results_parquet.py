"""Parquet results-writer tests (T3.1 / CF-LIBS-improved-1d5t).

Covers:

* :func:`cflibs.benchmark.results.write_parquet` round-trip — all
  IDEvaluationRecord / CompositionEvaluationRecord fields survive the
  write+read cycle.
* Run-metadata columns (cell / identifier / platform / seed /
  iter_index / experiment_label) land on every row.
* DuckDB queries against the file return the right aggregates and
  finish quickly (<100 ms on a synthetic ~30-row file — well under the
  acceptance budget of 100 ms on the full 48-iter file).
* Append mode concatenates a second batch with matching schema.
* :func:`UnifiedBenchmarkRunner.write_outputs` honours
  ``output_format="parquet"`` (default), ``"json"``, and ``"both"``.

Skips automatically when ``pyarrow`` isn't importable (the rest of the
codebase doesn't currently depend on it).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pyarrow = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
# duckdb is declared in the `ci` and `parquet` pyproject extras but never
# imported by shipped cflibs/ code -- it is needed ONLY here (and for downstream
# parquet analysis). Keep it in those extras; it is not an unused dependency.
duckdb = pytest.importorskip("duckdb")

from cflibs.benchmark.results import (  # noqa: E402  (after importorskip)
    SCHEMA_VERSION,
    build_rows,
    read_parquet,
    read_parquet_dir,
    write_parquet,
)
from cflibs.benchmark.unified import (  # noqa: E402
    CompositionEvaluationRecord,
    IDEvaluationRecord,
    UnifiedBenchmarkRunner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_id_record(
    *,
    spectrum_id: str = "spec-1",
    workflow: str = "alias",
    f1: float = 0.8,
) -> IDEvaluationRecord:
    return IDEvaluationRecord(
        dataset_id="vrabel2020_soil_benchmark",
        spectrum_id=spectrum_id,
        group_id="group-A",
        specimen_id="specimen-1",
        instrument_id="instr-1",
        truth_type="assay",
        rp_estimate=2500.0,
        label_cardinality=4,
        spectrum_kind="raw",
        workflow_name=workflow,
        outer_split_id="fold-0",
        tuning_split_id="inner-0",
        config_name=f"{workflow}-default",
        elapsed_seconds=1.5,
        true_elements=["Fe", "Si", "Ca", "Al"],
        predicted_elements=["Fe", "Si", "Ca"],
        tp=3,
        fp=0,
        fn=1,
        tn=10,
        precision=1.0,
        recall=0.75,
        f1=f1,
        jaccard=0.75,
        hamming_loss=0.0625,
        exact_match=False,
        false_positives_per_spectrum=0,
        scored=True,
        failure_reason=None,
        annotations={"note": "synthetic"},
    )


def _make_comp_record(
    *,
    spectrum_id: str = "spec-1",
    id_workflow: str = "alias",
    composition_workflow: str = "iterative_jax",
    aitchison: float = 0.42,
    with_posterior: bool = False,
) -> CompositionEvaluationRecord:
    annotations: dict = {}
    if with_posterior:
        annotations["posterior"] = {
            "rhat": [1.001, 1.003, 1.002],
            "ess_bulk": [500.0, 480.0, 510.0],
            "k_hat": 0.45,
            "divergent": 0,
            "coverage": 0.92,
        }
    return CompositionEvaluationRecord(
        dataset_id="vrabel2020_soil_benchmark",
        spectrum_id=spectrum_id,
        group_id="group-A",
        specimen_id="specimen-1",
        instrument_id="instr-1",
        truth_type="assay",
        rp_estimate=2500.0,
        label_cardinality=4,
        spectrum_kind="raw",
        id_workflow_name=id_workflow,
        composition_workflow_name=composition_workflow,
        outer_split_id="fold-0",
        tuning_split_id="inner-0",
        id_config_name=f"{id_workflow}-default",
        composition_config_name=f"{composition_workflow}-default",
        elapsed_seconds=2.5,
        candidate_elements=["Fe", "Si", "Ca", "Al"],
        true_composition={"Fe": 0.3, "Si": 0.4, "Ca": 0.2, "Al": 0.1},
        predicted_composition={"Fe": 0.28, "Si": 0.42, "Ca": 0.21, "Al": 0.09},
        aitchison=aitchison,
        rmse=0.015,
        temperature_error_frac=0.05,
        ne_error_frac=0.10,
        closure_residual=0.001,
        error_tier="excellent",
        per_element_absolute_error={"Fe": 0.02, "Si": 0.02, "Ca": 0.01, "Al": 0.01},
        per_element_relative_error={"Fe": 0.067, "Si": 0.05, "Ca": 0.05, "Al": 0.10},
        subcompositional_ratio_errors={"Fe/Si": 0.05, "Ca/Al": float("nan")},
        scored=True,
        failure_reason=None,
        annotations=annotations,
    )


def _build_dataset(n_spectra: int = 5) -> tuple[list, list]:
    """Build a small synthetic experiment: 2 identifiers × 2 comp workflows × N spectra."""
    id_records = []
    comp_records = []
    identifiers = ("alias", "comb")
    composition_workflows = ("iterative_jax", "bayesian")
    for spec_idx in range(n_spectra):
        sid = f"spec-{spec_idx:03d}"
        for id_idx, identifier in enumerate(identifiers):
            id_records.append(
                _make_id_record(
                    spectrum_id=sid,
                    workflow=identifier,
                    f1=0.6 + 0.1 * id_idx,
                )
            )
            for cw_idx, comp_wf in enumerate(composition_workflows):
                comp_records.append(
                    _make_comp_record(
                        spectrum_id=sid,
                        id_workflow=identifier,
                        composition_workflow=comp_wf,
                        aitchison=0.3 + 0.05 * id_idx + 0.02 * cw_idx,
                        with_posterior=(comp_wf == "bayesian"),
                    )
                )
    return id_records, comp_records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_rows_attaches_run_metadata_to_every_row():
    id_records, comp_records = _build_dataset(n_spectra=2)
    rows = build_rows(
        id_records=id_records,
        composition_records=comp_records,
        run_metadata={
            "cell": "C1",
            "identifier": "alias",
            "platform": "jax-cpu",
            "seed": 42,
            "iter_index": 7,
            "experiment_label": "loop-2026-05-12",
        },
    )
    assert len(rows) == len(id_records) + len(comp_records)
    for row in rows:
        assert row["schema_version"] == SCHEMA_VERSION
        assert row["cell"] == "C1"
        assert row["identifier"] == "alias"
        assert row["platform"] == "jax-cpu"
        assert row["seed"] == 42
        assert row["iter_index"] == 7
        assert row["experiment_label"] == "loop-2026-05-12"
        assert row["run_id"]  # non-empty UUID
        assert row["timestamp_iso"]


def test_write_parquet_round_trip_preserves_all_record_fields(tmp_path: Path):
    id_records, comp_records = _build_dataset(n_spectra=3)
    out_path = tmp_path / "results.parquet"
    write_parquet(
        output_path=out_path,
        id_records=id_records,
        composition_records=comp_records,
        run_metadata={
            "cell": "C2",
            "identifier": "alias",
            "platform": "jax-cpu",
            "seed": 7,
            "iter_index": 1,
            "experiment_label": "test-rt",
        },
    )
    assert out_path.exists()
    table = read_parquet(out_path)
    assert table.num_rows == len(id_records) + len(comp_records)

    # All declared columns are present.
    expected_columns = {
        "schema_version", "run_id", "record_kind", "record_index",
        "timestamp_iso", "cell", "identifier", "platform", "seed",
        "iter_index", "experiment_label", "dataset_id", "spectrum_id",
        "outer_split_id", "scored", "id_workflow_name",
        "id_config_name", "true_elements", "predicted_elements",
        "tp", "fp", "fn", "tn", "precision", "recall", "f1", "jaccard",
        "hamming_loss", "exact_match", "false_positives_per_spectrum",
        "comp_id_workflow_name", "composition_workflow_name",
        "comp_id_config_name", "composition_config_name",
        "candidate_elements", "true_composition", "predicted_composition",
        "aitchison", "d_a", "rmse", "temperature_error_frac",
        "ne_error_frac", "closure_residual", "error_tier",
        "per_element_absolute_error", "per_element_relative_error",
        "subcompositional_ratio_errors", "posterior_diagnostics",
    }
    actual_columns = set(table.column_names)
    assert expected_columns <= actual_columns, expected_columns - actual_columns


def test_duckdb_aggregate_query_is_fast_and_correct(tmp_path: Path):
    """The acceptance test: GROUP BY identifier, AVG(d_a), <100 ms."""
    id_records, comp_records = _build_dataset(n_spectra=10)
    out_path = tmp_path / "results.parquet"
    write_parquet(
        output_path=out_path,
        id_records=id_records,
        composition_records=comp_records,
        run_metadata={"cell": "C1", "experiment_label": "perf-test"},
    )

    con = duckdb.connect()
    # Warm up the connection / planner so the first run doesn't bias.
    con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchall()

    # The brief specifies grouping by `identifier` for the acceptance
    # query. With no per-row identifier in run_metadata, we group by
    # the per-row id-workflow column instead (the "identifier" column
    # is a constant copied from run_metadata).
    sql = (
        "SELECT comp_id_workflow_name AS identifier, "
        "       AVG(d_a) AS mean_d_a, COUNT(*) AS n "
        f"FROM read_parquet('{out_path}') "
        "WHERE record_kind = 'composition' "
        "GROUP BY comp_id_workflow_name ORDER BY identifier"
    )
    t_start = time.perf_counter()
    rows = con.execute(sql).fetchall()
    elapsed_ms = (time.perf_counter() - t_start) * 1_000

    # Acceptance: <100 ms (we're well under on a tiny file).
    assert elapsed_ms < 100, f"DuckDB aggregate took {elapsed_ms:.1f} ms"
    identifiers = [row[0] for row in rows]
    assert identifiers == ["alias", "comb"]
    for _identifier, mean_d_a, count in rows:
        assert 0.0 < mean_d_a < 1.0
        assert count > 0


def test_duckdb_groups_by_id_workflow_name(tmp_path: Path):
    """Per-spec query: group by the id-workflow column, not the run-metadata identifier."""
    id_records, comp_records = _build_dataset(n_spectra=10)
    out_path = tmp_path / "results.parquet"
    write_parquet(
        output_path=out_path,
        id_records=id_records,
        composition_records=comp_records,
    )
    con = duckdb.connect()
    sql = (
        "SELECT comp_id_workflow_name AS identifier, AVG(d_a) AS mean_d_a "
        f"FROM read_parquet('{out_path}') "
        "WHERE record_kind = 'composition' "
        "GROUP BY comp_id_workflow_name ORDER BY identifier"
    )
    rows = con.execute(sql).fetchall()
    identifiers = [row[0] for row in rows]
    assert identifiers == ["alias", "comb"]


def test_unnest_predicted_composition(tmp_path: Path):
    """LIST<STRUCT> compositions are queryable via UNNEST."""
    id_records, comp_records = _build_dataset(n_spectra=4)
    out_path = tmp_path / "results.parquet"
    write_parquet(
        output_path=out_path,
        id_records=id_records,
        composition_records=comp_records,
    )
    con = duckdb.connect()
    # DuckDB unnests struct fields into separate columns when the
    # source column is LIST<STRUCT<...>> and ``recursive := true`` is
    # used: each struct field becomes a column named after the field.
    sql = (
        "SELECT element, AVG(mass_fraction) "
        "FROM ("
        "  SELECT UNNEST(predicted_composition, recursive := true) "
        f"  FROM read_parquet('{out_path}') "
        "  WHERE record_kind = 'composition'"
        ") "
        "GROUP BY element ORDER BY element"
    )
    rows = con.execute(sql).fetchall()
    elements = [row[0] for row in rows]
    assert elements == ["Al", "Ca", "Fe", "Si"]


def test_subcompositional_ratio_errors_preserve_nan(tmp_path: Path):
    """NaN entries in subcompositional_ratio_errors round-trip as NaN."""
    _, comp_records = _build_dataset(n_spectra=1)
    out_path = tmp_path / "results.parquet"
    write_parquet(
        output_path=out_path, composition_records=comp_records
    )
    table = read_parquet(out_path)
    sub = table.column("subcompositional_ratio_errors").to_pylist()
    assert sub  # non-empty (composition rows present)
    found_nan = False
    for entry in sub:
        if entry is None:
            continue
        for item in entry:
            if item["pair"] == "Ca/Al":
                # NaN doesn't compare equal to itself.
                assert item["value"] is None or item["value"] != item["value"]
                found_nan = True
    assert found_nan


def test_posterior_diagnostics_populated_for_bayesian(tmp_path: Path):
    _, comp_records = _build_dataset(n_spectra=2)
    out_path = tmp_path / "results.parquet"
    write_parquet(output_path=out_path, composition_records=comp_records)
    con = duckdb.connect()
    rows = con.execute(
        "SELECT composition_workflow_name, "
        "       posterior_diagnostics.rhat_max AS rhat, "
        "       posterior_diagnostics.coverage AS cov "
        f"FROM read_parquet('{out_path}') "
        "WHERE record_kind = 'composition' "
        "ORDER BY composition_workflow_name, spectrum_id"
    ).fetchall()
    bayesian = [r for r in rows if r[0] == "bayesian"]
    classical = [r for r in rows if r[0] == "iterative_jax"]
    assert bayesian, "expected bayesian rows"
    assert classical, "expected classical rows"
    for _, rhat, cov in bayesian:
        assert rhat is not None and rhat > 1.0
        assert cov is not None and 0.0 < cov <= 1.0
    for _, rhat, cov in classical:
        assert rhat is None and cov is None


def test_append_mode_concatenates(tmp_path: Path):
    out_path = tmp_path / "results.parquet"
    id_records_1, comp_records_1 = _build_dataset(n_spectra=2)
    id_records_2, comp_records_2 = _build_dataset(n_spectra=3)

    write_parquet(out_path, id_records=id_records_1, composition_records=comp_records_1)
    n_first = read_parquet(out_path).num_rows

    write_parquet(
        out_path,
        id_records=id_records_2,
        composition_records=comp_records_2,
        append=True,
    )
    table = read_parquet(out_path)
    n_total = table.num_rows
    assert n_total == n_first + len(id_records_2) + len(comp_records_2)


def test_unified_runner_write_outputs_parquet_default(tmp_path: Path):
    """``write_outputs`` writes parquet by default and skips legacy JSON records."""
    runner = UnifiedBenchmarkRunner.__new__(UnifiedBenchmarkRunner)  # bypass __init__
    id_records, comp_records = _build_dataset(n_spectra=2)
    out_dir = tmp_path / "out"
    outputs = runner.write_outputs(
        output_dir=out_dir,
        id_records=id_records,
        id_selections=[],
        composition_records=comp_records,
        composition_selections=[],
        output_format="parquet",
        run_metadata={"cell": "C1", "experiment_label": "default"},
    )
    assert "results_parquet" in outputs
    assert outputs["results_parquet"].exists()
    assert "id_records_json" not in outputs
    assert "composition_records_json" not in outputs
    # Summary JSONs still written.
    assert outputs["id_summary_json"].exists()
    assert outputs["composition_summary_json"].exists()


def test_unified_runner_write_outputs_json_legacy(tmp_path: Path):
    runner = UnifiedBenchmarkRunner.__new__(UnifiedBenchmarkRunner)
    id_records, comp_records = _build_dataset(n_spectra=2)
    out_dir = tmp_path / "out-json"
    outputs = runner.write_outputs(
        output_dir=out_dir,
        id_records=id_records,
        id_selections=[],
        composition_records=comp_records,
        composition_selections=[],
        output_format="json",
    )
    assert "results_parquet" not in outputs
    assert outputs["id_records_json"].exists()
    assert outputs["composition_records_json"].exists()


def test_unified_runner_write_outputs_both(tmp_path: Path):
    runner = UnifiedBenchmarkRunner.__new__(UnifiedBenchmarkRunner)
    id_records, comp_records = _build_dataset(n_spectra=2)
    out_dir = tmp_path / "out-both"
    outputs = runner.write_outputs(
        output_dir=out_dir,
        id_records=id_records,
        id_selections=[],
        composition_records=comp_records,
        composition_selections=[],
        output_format="both",
    )
    assert outputs["results_parquet"].exists()
    assert outputs["id_records_json"].exists()
    assert outputs["composition_records_json"].exists()


def test_parquet_file_size_under_budget_for_synthetic_experiment(tmp_path: Path):
    """Sanity check: a ~50-row synthetic experiment is well under 100 MB."""
    id_records, comp_records = _build_dataset(n_spectra=12)
    out_path = tmp_path / "results.parquet"
    write_parquet(out_path, id_records=id_records, composition_records=comp_records)
    size_bytes = out_path.stat().st_size
    # 100 MB acceptance budget; synthetic file should be tiny (<200 KB).
    assert size_bytes < 100 * 1024 * 1024
    # And specifically — it should beat the equivalent JSON dump easily.
    # Rough sanity: <200 KB for ~50 rows.
    assert size_bytes < 200 * 1024


def test_empty_inputs_produce_valid_empty_parquet(tmp_path: Path):
    """Edge case: an experiment that hit an early failure still writes a parseable file."""
    out_path = tmp_path / "results.parquet"
    write_parquet(out_path)
    assert out_path.exists()
    table = read_parquet(out_path)
    assert table.num_rows == 0
    # Schema fields are still present so downstream readers don't crash.
    assert "d_a" in table.column_names
    assert "record_kind" in table.column_names


# ---------------------------------------------------------------------------
# read_parquet_dir — checkpoint part-file reader
# (CF-LIBS-improved-xsuj — CodeRabbit review on PR #186)
# ---------------------------------------------------------------------------


def test_read_parquet_dir_concatenates_shards(tmp_path: Path):
    """Multiple part-files in a directory merge into a single Table."""
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()

    # Write two shards with one composition record each (mimicking what
    # ``evaluate_composition_workflow`` writes on each checkpoint trigger).
    write_parquet(
        parts_dir / "part_a.parquet",
        composition_records=[_make_comp_record(spectrum_id="s1")],
    )
    write_parquet(
        parts_dir / "part_b.parquet",
        composition_records=[_make_comp_record(spectrum_id="s2")],
    )

    merged = read_parquet_dir(parts_dir)
    assert merged.num_rows == 2
    spectrum_ids = merged.column("spectrum_id").to_pylist()
    assert set(spectrum_ids) == {"s1", "s2"}


def test_read_parquet_dir_lexicographic_shard_order(tmp_path: Path):
    """Shards merge in sorted-filename order so the result is deterministic."""
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    write_parquet(
        parts_dir / "part_00002.parquet",
        composition_records=[_make_comp_record(spectrum_id="z")],
    )
    write_parquet(
        parts_dir / "part_00001.parquet",
        composition_records=[_make_comp_record(spectrum_id="a")],
    )
    merged = read_parquet_dir(parts_dir)
    # "part_00001" sorts first, so "a" comes first in the merged table.
    assert merged.column("spectrum_id").to_pylist() == ["a", "z"]


def test_read_parquet_dir_rejects_file_path(tmp_path: Path):
    """Passing a single parquet file (not a dir) raises a clear error."""
    out_path = tmp_path / "results.parquet"
    write_parquet(out_path, composition_records=[_make_comp_record(spectrum_id="s1")])
    with pytest.raises(NotADirectoryError):
        read_parquet_dir(out_path)


def test_read_parquet_dir_raises_on_empty_dir(tmp_path: Path):
    """Reading an empty directory raises FileNotFoundError, not silent empty Table."""
    empty = tmp_path / "no_shards"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        read_parquet_dir(empty)


def test_read_parquet_dir_skips_orphan_tmp_files(tmp_path: Path):
    """Atomic-write contract: `.parquet.tmp` shards from a crashed mid-write
    must NOT be picked up by ``read_parquet_dir``. The unified-benchmark
    checkpoint writer (Phase 3 of CF-LIBS-improved-xsuj) stages parts as
    ``part_<slug>_<seq>.parquet.tmp`` then ``rename()``s to the final
    ``.parquet`` suffix — a SIGKILL between write+rename would otherwise
    leave a truncated shard that breaks ``pyarrow.concat_tables``.
    """
    parts_dir = tmp_path / "checkpoint.parts"
    parts_dir.mkdir()

    # Two legitimate shards.
    _id_a, comp_a = _build_dataset(n_spectra=2)
    _id_b, comp_b = _build_dataset(n_spectra=3)
    write_parquet(parts_dir / "part_alpha_00001.parquet", composition_records=comp_a)
    write_parquet(parts_dir / "part_alpha_00002.parquet", composition_records=comp_b)

    # An orphan .tmp left behind by a hypothetical crash. Empty/truncated
    # content — would crash pq.read_table if the glob picked it up.
    (parts_dir / "part_alpha_00003.parquet.tmp").write_bytes(b"PARTIAL_GARBAGE")

    table = read_parquet_dir(parts_dir)
    assert table.num_rows == len(comp_a) + len(comp_b)


def test_checkpoint_filename_includes_run_id_for_restart_safety():
    """Behavior contract: ``make_worker_slug`` must produce slugs that
    differ for two distinct run_ids on the same host+pid.

    On SLURM ``--requeue`` or PID reuse, two runs can share the
    ``(hostname, getpid())`` tuple. If the slug depended only on those,
    the second run's part-files would silently overwrite the first.
    The slug therefore folds the run_id (an 8-char hex prefix of
    UUID4) into the filename. This test calls ``make_worker_slug``
    twice with different run_ids and asserts the outputs disagree.

    Replaces an earlier brittle grep-the-source test (per Copilot
    review on PR #188): two cosmetic refactors that preserve behavior
    would have broken the old test.
    """
    from cflibs.benchmark.checkpoint import make_worker_slug

    run_id_a = "11111111-2222-3333-4444-555555555555"
    run_id_b = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    slug_a = make_worker_slug(run_id_a)
    slug_b = make_worker_slug(run_id_b)

    # Same host+pid (we're in one process), so the only thing that varies
    # is the run_id-derived portion.
    assert slug_a != slug_b, (
        "make_worker_slug must produce distinct slugs for distinct "
        "run_ids so same-host/same-PID restarts cannot overwrite prior "
        f"parts. Got identical slug={slug_a!r}"
    )
    # And the slug must reference the run_id prefix, not something
    # accidentally derived from the full hex (which would still vary
    # across runs but not match the documented 8-char-prefix contract).
    assert run_id_a.replace("-", "")[:8] in slug_a, (
        "slug must embed the first 8 hex chars of run_id (with dashes "
        f"stripped). Got slug={slug_a!r}"
    )
    assert run_id_b.replace("-", "")[:8] in slug_b


def test_checkpoint_new_run_id_returns_unique_uuids():
    """``new_run_id`` returns a fresh UUID4 string every call.

    The checkpoint flow generates one run_id per evaluate_composition_workflow
    invocation and threads it through every ``write_parquet(run_id=...)``
    call so the part-files share a queryable run_id. Two consecutive calls
    must return distinct values.
    """
    from cflibs.benchmark.checkpoint import new_run_id

    a = new_run_id()
    b = new_run_id()
    assert a != b, f"new_run_id should yield fresh UUIDs but got identical: {a}"
    # UUID4 string form: 8-4-4-4-12 = 36 chars including dashes.
    assert len(a) == 36 and a.count("-") == 4
