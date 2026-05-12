# Parquet Results — Schema Design Consultation

CF-LIBS-improved-1d5t (T3.1). Synthesised from the Apache Arrow / Parquet
documentation, DuckDB Parquet best-practices, and prior art in
`scikit-learn`'s benchmark harness + Hugging Face `datasets`. The repo
has no live cloud LLMs configured at consultation time; the trade-off
discussion below reflects the working consensus across those sources.

## Key decisions

### One file per experiment, *not* per iteration

The brief says "one Parquet file per experiment" — adopt this directly.

- **Trade-off:** sibling subagents (`feat/parameter-sweep*`) may want to
  partition by `cell` or `seed`. Parquet *supports* hive-style
  partitioning (`results.parquet/cell=C1/...`) but it adds operational
  complexity (multiple files, partition pruning behaviour varies by
  reader).
- **Decision:** single-file path is the default. Callers that want
  partitioning pass `partition_cols=[...]` to `write_parquet`, which
  pyarrow handles via `pq.write_to_dataset`.
- **Why:** the 48-iter × 50 MB JSON estimate is for raw posterior arrays.
  Once we drop arrays into Parquet's columnar Snappy compression, the
  whole experiment fits in <100 MB and pruning by `cell` doesn't buy
  much. Keep the simple case simple.

### `predicted_composition` / `true_composition`: list-of-structs, not map

Two natural encodings for `{element: mass_fraction}`:

| Encoding              | Pros                                              | Cons                                                       |
| --------------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| Single `MAP<str,float>` column | Compact; preserves JSON shape exactly      | DuckDB/polars/pandas have *very* uneven MAP support        |
| `LIST<STRUCT<element:str, mass_fraction:float>>` | Universal — every engine handles it | Slightly more verbose; explode-then-join to query an element |

- **Decision:** `LIST<STRUCT<element:STRING, mass_fraction:DOUBLE>>`.
- **DuckDB pattern** for "average d_a per identifier":
  `SELECT identifier, AVG(d_a) FROM read_parquet('results.parquet') GROUP BY identifier`
  — no composition unnest needed, fast path.
- **DuckDB pattern** for per-element:
  `SELECT t.element, AVG(t.mass_fraction) FROM read_parquet('results.parquet'), UNNEST(true_composition) AS t GROUP BY t.element`.

### Posterior diagnostics: nullable STRUCT, not row-per-diagnostic

- Workflows like `bayesian` emit per-spectrum `rhat`, `ess_bulk`, `k_hat`,
  `divergent`. Classical workflows leave these `NULL`.
- Encoding as a flat STRUCT keeps the column count low; columns with all
  NULLs cost essentially zero on disk under Parquet's dictionary +
  RLE encoding.

### Per-element error dicts (absolute / relative / sub-ratio)

Same story as compositions — `LIST<STRUCT<key:STRING, value:DOUBLE>>`. Keep
NaN values; pyarrow preserves them via `pa.float64`. NaN means "ratio
truth undefined" per the existing comment at unified.py:679-683.

### Schema versioning

Embed a `schema_version: int` column (constant per write). v1 is what
this PR lands. When a future change adds e.g. `chain_mean` arrays we
bump to v2 and the schema doc records the diff. Cheaper than building
in a sidecar `schema.json`.

### Append semantics

`write_parquet(records, path, append=True)` reads the existing file,
concatenates the new RecordBatch, and rewrites. This is the
intentionally simple option for moderate file sizes (<1 GB). For very
large experiments we can revisit `pq.ParquetWriter` with row-group
streaming — out-of-scope for T3.1.

### Fallback to JSON

Per the brief: keep JSON writes behind an env-var/CLI flag for backwards
compatibility. Default = parquet on. Implementation: a third
`output_format` knob on `write_outputs`:
- `"parquet"` (default) — writes `results.parquet` only.
- `"json"` — writes the legacy JSON+CSV bundle only.
- `"both"` — writes both (dogfood path during transition).

Driven by `--output-format` CLI flag on
`scripts/run_unified_benchmark.py`, with env override
`CFLIBS_OUTPUT_FORMAT` for the loop scripts that can't easily plumb new
CLI args.

## Open questions intentionally deferred

- Partitioning policy for the bandit allocator (T2.3) — wait for that
  PR to land and observe access patterns.
- Compression codec selection (Snappy vs Zstd) — Snappy is the pyarrow
  default and good enough; revisit only if file size becomes a problem.
- Row-group sizing — leave at pyarrow defaults (~64 MB) until we see
  scan-time perf data from a real 48-iter run.
