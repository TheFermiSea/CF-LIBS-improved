#!/usr/bin/env python3
"""Merge per-shard ``results.parquet`` files from a sharded benchmark run.

Each shard produced by ``scripts/run_unified_benchmark.py --dataset-shard
N/K`` writes a standalone ``results.parquet`` (schema v1, see
``docs/results-parquet-schema.md``). After all K shards finish, this
script concatenates them into a single unified Parquet so that
downstream analysis (DuckDB, polars, pandas) sees one logical table
covering the full corpus.

Behaviour
---------
* Reads every input Parquet, validates its schema matches the canonical
  v1 schema (no silent column drops).
* Verifies the disjoint-cover invariant: no two shards may emit the
  same ``(spectrum_id, id_workflow_name, composition_workflow_name,
  outer_split_id, record_kind)`` tuple. If they do, that indicates a
  partitioner bug; the script fails fast rather than silently deduping
  (silent dedupe would mask the bug and skew population statistics).
* Concatenates with pyarrow ``concat_tables`` and writes a single
  Parquet at ``--output``. The merged file inherits ``schema_version =
  1`` so downstream readers don't need to know about sharding.
* Preserves the per-row ``shard_n`` / ``shard_k`` columns from each
  input — those carry the lineage. Aggregator does not add new columns.

Usage
-----
::

    scripts/aggregate_shards.py \\
        /scratch/cf-libs-bench/shard-1/results.parquet \\
        /scratch/cf-libs-bench/shard-2/results.parquet \\
        /scratch/cf-libs-bench/shard-3/results.parquet \\
        --output /scratch/cf-libs-bench/merged/results.parquet

Exit codes
----------
* ``0`` — success
* ``1`` — schema mismatch or other validation failure
* ``2`` — disjoint-cover invariant violated (overlapping spectra)

See ``docs/dataset-sharding.md`` for the full sharding workflow and the
bootstrap-CI math that motivates this design.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-shard results.parquet files from a sharded "
            "unified-benchmark run into a single Parquet table."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One results.parquet file per shard (typically K files for K shards).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the merged results.parquet.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help=(
            "Skip the disjoint-cover invariant check. ONLY use this when "
            "intentionally merging non-shard files (e.g. independent runs "
            "with overlapping spectra)."
        ),
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression codec (default: snappy).",
    )
    return parser


def _validate_schemas_compatible(tables: Sequence["pyarrow.Table"]) -> None:
    """Ensure every input table's schema matches the first.

    pyarrow's ``concat_tables`` already enforces compatibility, but its
    error messages are dense; this gives a clearer one. We compare on
    field names + types, ignoring nullability / metadata since the
    parquet writer may downcast nullable bits on read.
    """
    if not tables:
        return
    canonical = tables[0].schema
    canonical_pairs = [(f.name, str(f.type)) for f in canonical]
    for idx, t in enumerate(tables[1:], start=1):
        pairs = [(f.name, str(f.type)) for f in t.schema]
        if pairs != canonical_pairs:
            missing = set(canonical.names) - set(t.schema.names)
            extra = set(t.schema.names) - set(canonical.names)
            raise ValueError(
                f"Shard {idx} schema does not match shard 0. "
                f"Missing columns: {sorted(missing) or 'none'}; "
                f"extra columns: {sorted(extra) or 'none'}."
            )


def _check_disjoint_cover(table: "pyarrow.Table") -> None:
    """Verify no two rows share the same shard-cover key.

    The shard-cover key is the tuple of columns that should be unique
    *within a single benchmark run* — if two shards emit the same key,
    the partitioner mis-routed at least one spectrum.

    Key = (record_kind, spectrum_id, id_workflow_name,
           composition_workflow_name, outer_split_id).

    ``composition_workflow_name`` is NULL for identification rows and
    ``id_workflow_name`` may also be NULL on some upstream-schema
    versions; we coerce nulls to empty strings before the uniqueness
    check so they participate consistently.
    """
    import pyarrow.compute as pc

    n = table.num_rows
    if n == 0:
        return

    cols = [
        "record_kind",
        "spectrum_id",
        "id_workflow_name",
        "composition_workflow_name",
        "outer_split_id",
    ]
    # Build a single string key per row by joining the columns with a
    # delimiter. NaN/None coalesce to "<null>".
    key_arrays = []
    for col_name in cols:
        col = table.column(col_name) if col_name in table.schema.names else None
        if col is None:
            key_arrays.append(["<null>"] * n)
            continue
        col_str = pc.cast(col, "string") if col.type != "string" else col
        # Replace null with "<null>" sentinel.
        filled = pc.fill_null(col_str, "<null>")
        key_arrays.append(filled.to_pylist())

    seen: dict[tuple, int] = {}
    duplicates: list[tuple] = []
    for row_idx in range(n):
        key = tuple(key_arrays[c][row_idx] for c in range(len(cols)))
        if key in seen:
            duplicates.append(key)
            if len(duplicates) >= 5:
                break
        else:
            seen[key] = row_idx

    if duplicates:
        sample = "\n  ".join(repr(d) for d in duplicates[:5])
        raise ValueError(
            "Shard overlap detected — the same (record_kind, spectrum_id, "
            "id_workflow_name, composition_workflow_name, outer_split_id) "
            "tuple appears in multiple shards. This indicates a partitioner "
            "bug. Sample colliding keys:\n  "
            f"{sample}\n"
            "Re-run with --allow-overlap to bypass (NOT recommended for "
            "real sharded runs)."
        )


def aggregate(
    inputs: Iterable[Path],
    output: Path,
    *,
    allow_overlap: bool = False,
    compression: str = "snappy",
) -> Path:
    """Merge per-shard parquet files into one. Returns ``output``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_paths = list(inputs)
    if not input_paths:
        raise ValueError("aggregate() needs at least one input file")

    tables = []
    for path in input_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Shard parquet not found: {path}")
        tables.append(pq.read_table(str(path)))

    _validate_schemas_compatible(tables)

    merged = pa.concat_tables(tables, promote_options="default")

    if not allow_overlap:
        _check_disjoint_cover(merged)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, str(output), compression=compression)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        merged_path = aggregate(
            args.inputs,
            args.output,
            allow_overlap=args.allow_overlap,
            compression=args.compression,
        )
    except FileNotFoundError as exc:
        print(f"aggregate_shards: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # Distinguish overlap (exit 2) from schema mismatch (exit 1).
        msg = str(exc)
        print(f"aggregate_shards: {msg}", file=sys.stderr)
        return 2 if "overlap" in msg.lower() else 1

    print(f"aggregate_shards: wrote {merged_path.resolve()}")
    # Print a quick summary so log scrapers can verify the merge.
    try:
        import pyarrow.parquet as pq

        meta = pq.read_metadata(str(merged_path))
        print(f"  total rows: {meta.num_rows}")
        print(f"  total columns: {len(meta.schema.names)}")
    except Exception:  # pragma: no cover — best-effort summary
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
