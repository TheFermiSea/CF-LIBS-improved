"""Dataset-sharding tests (T2.2 / CF-LIBS-improved-sxt0).

Covers:

* :func:`cflibs.benchmark.loaders.apply_dataset_shard` — stride-based
  partition is a disjoint cover of the input dataset, regardless of
  K (1, 2, 3, 5, 16) and regardless of seed (sharding is
  seed-independent).
* Shard annotations land on every kept spectrum.
* ``--dataset-shard`` CLI parser validates N/K bounds.
* ``scripts/aggregate_shards`` merges multiple shard parquets into one,
  preserves shard_n/shard_k per row, and fails fast on overlap.

These tests use synthetic fixtures (no Vrabel data on disk required)
so they run in <1 s on any host.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cflibs.benchmark.dataset import (  # noqa: E402
    BenchmarkDataset,
    BenchmarkSpectrum,
    InstrumentalConditions,
    MatrixType,
    SampleMetadata,
    SampleType,
    TruthType,
)
from cflibs.benchmark.loaders import apply_dataset_shard  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(n_spectra: int = 30) -> BenchmarkDataset:
    """Build a tiny synthetic dataset large enough to exercise N/K splits.

    Composition is identical across spectra; what matters for the shard
    tests is the spectrum_id ordering, which determines which shard each
    one ends up in.
    """
    conditions = InstrumentalConditions(
        laser_wavelength_nm=1064.0,
        laser_energy_mj=50.0,
        spectral_range_nm=(200.0, 1000.0),
        spectral_resolution_nm=0.02,
        spectrometer_type="echelle",
        detector_type="ICCD",
        atmosphere="air",
    )

    spectra = []
    wl = np.linspace(200.0, 1000.0, 400)
    for i in range(n_spectra):
        intensity = np.full_like(wl, float(i + 1))
        metadata = SampleMetadata(
            sample_id=f"synth_sample_{i:03d}",
            sample_type=SampleType.CRM,
            matrix_type=MatrixType.GEOLOGICAL,
        )
        spectra.append(
            BenchmarkSpectrum(
                spectrum_id=f"synth_s{i:03d}",
                wavelength_nm=wl,
                intensity=intensity,
                true_composition={"Fe": 0.5, "Si": 0.5},
                conditions=conditions,
                metadata=metadata,
                dataset_id="synthetic",
                group_id=f"group_{i % 5}",
                truth_type=TruthType.ASSAY,
                annotations={"index": i},
            )
        )
    return BenchmarkDataset(
        name="synthetic_shard_test",
        version="v1",
        spectra=spectra,
        elements=["Fe", "Si"],
        description="synthetic dataset for shard tests",
    )


# ---------------------------------------------------------------------------
# apply_dataset_shard — partition correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 5, 7])
def test_shard_partition_is_disjoint_cover(k):
    """For every K in [1, 7], the union of all K shards = the full corpus.

    Union must equal the original spectrum_id set; pairwise intersections
    must be empty.
    """
    dataset = _make_synthetic_dataset(n_spectra=30)
    original_ids = {s.spectrum_id for s in dataset.spectra}

    seen_ids: set[str] = set()
    seen_per_shard: list[set[str]] = []
    for shard_n in range(1, k + 1):
        sub = apply_dataset_shard(dataset, shard_n, k)
        ids = {s.spectrum_id for s in sub.spectra}
        # No spectrum appears in two shards
        assert ids.isdisjoint(seen_ids), (
            f"Shard {shard_n}/{k} overlaps with previous shards: "
            f"{ids & seen_ids}"
        )
        seen_per_shard.append(ids)
        seen_ids |= ids

    assert seen_ids == original_ids, (
        f"Union of {k} shards does not equal full corpus. "
        f"Missing: {original_ids - seen_ids}; extra: {seen_ids - original_ids}"
    )


def test_shard_1_of_1_is_identity():
    """``shard_k=1`` must return the original dataset unchanged."""
    dataset = _make_synthetic_dataset(n_spectra=10)
    sub = apply_dataset_shard(dataset, 1, 1)
    # Same dataset object (cheap early-return) — see apply_dataset_shard.
    assert sub is dataset


def test_shard_sizes_are_balanced():
    """Stride partition gives shard sizes within 1 of N/K."""
    n_total = 30
    dataset = _make_synthetic_dataset(n_spectra=n_total)
    for k in (2, 3, 5, 7):
        sizes = [
            apply_dataset_shard(dataset, n, k).n_spectra
            for n in range(1, k + 1)
        ]
        assert max(sizes) - min(sizes) <= 1, (
            f"K={k}: shard sizes {sizes} unbalanced; "
            f"max-min should be <= 1 for stride partitioning"
        )
        assert sum(sizes) == n_total


def test_shard_annotations_landed():
    """Each spectrum in shard N/K gets shard_n/shard_k in annotations."""
    dataset = _make_synthetic_dataset(n_spectra=12)
    sub = apply_dataset_shard(dataset, 2, 3)
    for s in sub.spectra:
        assert s.annotations.get("shard_n") == 2
        assert s.annotations.get("shard_k") == 3


def test_shard_is_deterministic_across_runs():
    """Re-running apply_dataset_shard yields the same partition."""
    ds_a = _make_synthetic_dataset(n_spectra=15)
    ds_b = _make_synthetic_dataset(n_spectra=15)
    sub_a = apply_dataset_shard(ds_a, 2, 3)
    sub_b = apply_dataset_shard(ds_b, 2, 3)
    ids_a = [s.spectrum_id for s in sub_a.spectra]
    ids_b = [s.spectrum_id for s in sub_b.spectra]
    assert ids_a == ids_b


def test_shard_bounds_rejected():
    """N out of [1, K] and K <= 0 raise ValueError."""
    dataset = _make_synthetic_dataset(n_spectra=5)
    with pytest.raises(ValueError):
        apply_dataset_shard(dataset, 0, 3)
    with pytest.raises(ValueError):
        apply_dataset_shard(dataset, 4, 3)
    with pytest.raises(ValueError):
        apply_dataset_shard(dataset, 1, 0)


# ---------------------------------------------------------------------------
# CLI flag parser
# ---------------------------------------------------------------------------


def test_cli_dataset_shard_parser_accepts_valid():
    from scripts.run_unified_benchmark import _parse_dataset_shard

    assert _parse_dataset_shard("1/1") == (1, 1)
    assert _parse_dataset_shard("1/3") == (1, 3)
    assert _parse_dataset_shard("3/3") == (3, 3)
    assert _parse_dataset_shard("16/16") == (16, 16)


@pytest.mark.parametrize(
    "bad",
    ["", "1", "1/", "/3", "a/3", "1/b", "0/3", "4/3", "1/17", "-1/3"],
)
def test_cli_dataset_shard_parser_rejects_bad(bad):
    from scripts.run_unified_benchmark import _parse_dataset_shard

    with pytest.raises(ValueError):
        _parse_dataset_shard(bad)


# ---------------------------------------------------------------------------
# aggregate_shards.py — Parquet merger
# ---------------------------------------------------------------------------


pyarrow = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _write_shard_parquet(
    output_path: Path,
    spectrum_ids: list[str],
    shard_n: int,
    shard_k: int,
) -> Path:
    """Build a tiny results.parquet with N composition rows."""
    from cflibs.benchmark.results import write_parquet
    from cflibs.benchmark.unified import CompositionEvaluationRecord

    records = [
        CompositionEvaluationRecord(
            dataset_id="vrabel2020_soil_benchmark",
            spectrum_id=sid,
            group_id=f"group_{i % 3}",
            specimen_id=None,
            instrument_id=None,
            truth_type="assay",
            rp_estimate=None,
            label_cardinality=None,
            spectrum_kind="geostandard",
            id_workflow_name="alias",
            composition_workflow_name="iterative_jax",
            outer_split_id="fold-0",
            tuning_split_id=None,
            id_config_name="alias-default",
            composition_config_name="iterative_jax-default",
            elapsed_seconds=1.0,
            candidate_elements=["Fe"],
            true_composition={"Fe": 0.5, "Si": 0.5},
            predicted_composition={"Fe": 0.48, "Si": 0.52},
            aitchison=0.1,
            rmse=0.02,
            scored=True,
            failure_reason=None,
            temperature_error_frac=None,
            ne_error_frac=None,
            closure_residual=None,
            error_tier=None,
            per_element_absolute_error=None,
            per_element_relative_error=None,
            subcompositional_ratio_errors=None,
            annotations={"shard_n": shard_n, "shard_k": shard_k},
        )
        for i, sid in enumerate(spectrum_ids)
    ]
    return write_parquet(
        output_path=output_path,
        id_records=(),
        composition_records=records,
        run_metadata={
            "cell": f"shard-{shard_n}",
            "identifier": "alias",
            "platform": "jax-cpu",
            "seed": 42,
            "iter_index": 0,
            "experiment_label": "shard-test",
            "shard_n": shard_n,
            "shard_k": shard_k,
        },
    )


def test_aggregator_merges_disjoint_shards(tmp_path):
    """3 disjoint shards merge into one Parquet with all rows preserved."""
    from scripts.aggregate_shards import aggregate

    shard1 = _write_shard_parquet(
        tmp_path / "shard1.parquet",
        spectrum_ids=[f"vrabel2020_s001_shot{i:03d}" for i in range(0, 9, 3)],
        shard_n=1,
        shard_k=3,
    )
    shard2 = _write_shard_parquet(
        tmp_path / "shard2.parquet",
        spectrum_ids=[f"vrabel2020_s001_shot{i:03d}" for i in range(1, 9, 3)],
        shard_n=2,
        shard_k=3,
    )
    shard3 = _write_shard_parquet(
        tmp_path / "shard3.parquet",
        spectrum_ids=[f"vrabel2020_s001_shot{i:03d}" for i in range(2, 9, 3)],
        shard_n=3,
        shard_k=3,
    )

    merged = aggregate(
        [shard1, shard2, shard3],
        tmp_path / "merged.parquet",
    )
    assert merged.is_file()

    table = pq.read_table(str(merged))
    # 3 shards × 3 spectra each = 9 rows
    assert table.num_rows == 9

    # shard_n column is preserved per row
    shard_ns = sorted(set(table.column("shard_n").to_pylist()))
    assert shard_ns == [1, 2, 3]

    # Schema version is the canonical v1
    schema_versions = set(table.column("schema_version").to_pylist())
    assert schema_versions == {1}


def test_aggregator_rejects_overlapping_shards(tmp_path):
    """If two shards share a spectrum_id, the aggregator fails fast."""
    from scripts.aggregate_shards import aggregate

    overlap_id = "vrabel2020_s001_shot000"
    shard1 = _write_shard_parquet(
        tmp_path / "shard1.parquet",
        spectrum_ids=[overlap_id, "vrabel2020_s001_shot003"],
        shard_n=1,
        shard_k=3,
    )
    shard2 = _write_shard_parquet(
        tmp_path / "shard2.parquet",
        spectrum_ids=[overlap_id, "vrabel2020_s001_shot004"],
        shard_n=2,
        shard_k=3,
    )
    with pytest.raises(ValueError, match="overlap"):
        aggregate([shard1, shard2], tmp_path / "merged.parquet")


def test_aggregator_allow_overlap_bypass(tmp_path):
    """``allow_overlap=True`` bypasses the disjoint-cover check."""
    from scripts.aggregate_shards import aggregate

    shard1 = _write_shard_parquet(
        tmp_path / "shard1.parquet",
        spectrum_ids=["vrabel2020_s001_shot000"],
        shard_n=1,
        shard_k=2,
    )
    shard2 = _write_shard_parquet(
        tmp_path / "shard2.parquet",
        spectrum_ids=["vrabel2020_s001_shot000"],
        shard_n=2,
        shard_k=2,
    )
    merged = aggregate(
        [shard1, shard2],
        tmp_path / "merged.parquet",
        allow_overlap=True,
    )
    table = pq.read_table(str(merged))
    assert table.num_rows == 2


def test_aggregator_preserves_schema_version(tmp_path):
    """Merged file inherits schema_version=1."""
    from cflibs.benchmark.results import SCHEMA_VERSION
    from scripts.aggregate_shards import aggregate

    shard1 = _write_shard_parquet(
        tmp_path / "shard1.parquet",
        spectrum_ids=["s_a", "s_b"],
        shard_n=1,
        shard_k=2,
    )
    shard2 = _write_shard_parquet(
        tmp_path / "shard2.parquet",
        spectrum_ids=["s_c", "s_d"],
        shard_n=2,
        shard_k=2,
    )
    merged = aggregate([shard1, shard2], tmp_path / "merged.parquet")
    table = pq.read_table(str(merged))
    assert set(table.column("schema_version").to_pylist()) == {SCHEMA_VERSION}
