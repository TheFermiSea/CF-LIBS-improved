# `results.parquet` — Unified Benchmark Result Schema (v1)

Single Parquet file per unified-benchmark experiment, replacing the
legacy `composition_records.json` + `id_records.json` triplet. One row
per (id_workflow × composition_workflow × outer_split × spectrum) pair
in the composition section, plus one row per (id_workflow × spectrum)
pair in the identification section. The two sections are
distinguished by the `record_kind` column (`"identification"` or
`"composition"`); rows with `record_kind = "identification"` leave all
composition-specific columns NULL and vice-versa.

## File layout

- Default path: `<output_dir>/results.parquet`
- Compression: Snappy (pyarrow default)
- Row-group size: pyarrow default (~64 MB)
- Append semantics: `write_parquet(..., append=True)` reads the existing
  file, concatenates, and rewrites. Schema must match.
- Partitioning: optional `partition_cols=[...]` argument forwards to
  `pyarrow.parquet.write_to_dataset` for callers that need hive-style
  partitions (e.g. by `dataset_id` or `outer_split_id`). Default is a
  single file.

## Top-level columns

| Column                        | Type                              | Nullable | Source                                  |
| ----------------------------- | --------------------------------- | -------- | --------------------------------------- |
| `schema_version`              | INT32                             | no       | constant = 1                            |
| `run_id`                      | STRING                            | no       | UUID per `write_parquet` invocation     |
| `record_kind`                 | STRING                            | no       | `"identification"` or `"composition"`   |
| `record_index`                | INT64                             | no       | 0-based position within source list     |
| `timestamp_iso`               | STRING                            | no       | ISO-8601 UTC at write time              |
| `dataset_id`                  | STRING                            | no       | from `IDEvaluationRecord.dataset_id`    |
| `spectrum_id`                 | STRING                            | no       | from record                             |
| `group_id`                    | STRING                            | yes      | from record                             |
| `specimen_id`                 | STRING                            | yes      | from record                             |
| `instrument_id`               | STRING                            | yes      | from record                             |
| `truth_type`                  | STRING                            | no       | `TruthType` enum value                  |
| `rp_estimate`                 | DOUBLE                            | yes      | resolving-power estimate                |
| `label_cardinality`           | INT64                             | yes      | from record                             |
| `spectrum_kind`               | STRING                            | yes      | from record                             |
| `outer_split_id`              | STRING                            | no       | outer cross-validation fold name        |
| `tuning_split_id`             | STRING                            | yes      | inner split when applicable             |
| `elapsed_seconds`             | DOUBLE                            | no       | from record                             |
| `scored`                      | BOOL                              | no       | from record                             |
| `failure_reason`              | STRING                            | yes      | from record                             |
| `annotations_json`            | STRING                            | yes      | `json.dumps(record.annotations)`        |

### Run-metadata columns (added by writer, not on records today)

These come from the new `run_metadata` argument to `write_parquet` and
are constant across the file. They're the columns the brief explicitly
calls out for downstream slicing.

| Column                | Type   | Nullable | Notes                                   |
| --------------------- | ------ | -------- | --------------------------------------- |
| `cell`                | STRING | yes      | e.g. `"C1"` — loop driver pass-through  |
| `identifier`          | STRING | yes      | mirrors `id_workflow_name` for queries  |
| `platform`            | STRING | yes      | e.g. `"jax-cpu"`, `"numpy"`             |
| `seed`                | INT64  | yes      | RNG seed if known                       |
| `iter_index`          | INT64  | yes      | loop iteration index                    |
| `experiment_label`    | STRING | yes      | free-form, e.g. `"loop-2026-05-12"`     |

### Identification-only columns (NULL on composition rows)

| Column                          | Type                                                       | Source                            |
| ------------------------------- | ---------------------------------------------------------- | --------------------------------- |
| `id_workflow_name`              | STRING                                                     | `IDEvaluationRecord.workflow_name`|
| `id_config_name`                | STRING                                                     | `config_name`                     |
| `true_elements`                 | LIST<STRING>                                               | from record                       |
| `predicted_elements`            | LIST<STRING>                                               | from record                       |
| `tp` / `fp` / `fn` / `tn`       | INT64                                                      | from record                       |
| `precision`                     | DOUBLE                                                     | from record                       |
| `recall`                        | DOUBLE                                                     | from record                       |
| `f1`                            | DOUBLE                                                     | from record                       |
| `jaccard`                       | DOUBLE                                                     | from record                       |
| `hamming_loss`                  | DOUBLE                                                     | from record                       |
| `exact_match`                   | BOOL                                                       | from record                       |
| `false_positives_per_spectrum`  | INT64                                                      | from record                       |

### Composition-only columns (NULL on identification rows)

| Column                              | Type                                              | Source                                      |
| ----------------------------------- | ------------------------------------------------- | ------------------------------------------- |
| `comp_id_workflow_name`             | STRING                                            | `id_workflow_name` (the upstream identifier) |
| `composition_workflow_name`         | STRING                                            | from record                                 |
| `comp_id_config_name`               | STRING                                            | from record                                 |
| `composition_config_name`           | STRING                                            | from record                                 |
| `candidate_elements`                | LIST<STRING>                                      | from record                                 |
| `true_composition`                  | LIST<STRUCT<element:STRING, mass_fraction:DOUBLE>>| `dict` -> list-of-structs                   |
| `predicted_composition`             | LIST<STRUCT<element:STRING, mass_fraction:DOUBLE>>| `dict` -> list-of-structs                   |
| `aitchison`                         | DOUBLE                                            | from record (also surfaced as `d_a`)        |
| `d_a`                               | DOUBLE                                            | alias of `aitchison` for query ergonomics   |
| `rmse`                              | DOUBLE                                            | from record                                 |
| `temperature_error_frac`            | DOUBLE                                            | from record                                 |
| `ne_error_frac`                     | DOUBLE                                            | from record                                 |
| `closure_residual`                  | DOUBLE                                            | from record                                 |
| `error_tier`                        | STRING                                            | from record                                 |
| `per_element_absolute_error`        | LIST<STRUCT<element:STRING, value:DOUBLE>>        | dict -> list-of-structs (NaNs preserved)    |
| `per_element_relative_error`        | LIST<STRUCT<element:STRING, value:DOUBLE>>        | dict -> list-of-structs                     |
| `subcompositional_ratio_errors`     | LIST<STRUCT<pair:STRING, value:DOUBLE>>           | dict -> list-of-structs                     |
| `posterior_diagnostics`             | STRUCT<rhat_max:DOUBLE, ess_bulk_min:DOUBLE, k_hat_max:DOUBLE, divergent_count:INT64, coverage:DOUBLE> | derived from `annotations["posterior"]` if present, else NULL |

## Query examples (DuckDB)

```sql
-- Acceptance test: <100 ms on a 48-iter file
SELECT identifier, AVG(d_a) AS mean_d_a
FROM read_parquet('output/loop-2026-05-12/results.parquet')
WHERE record_kind = 'composition'
GROUP BY identifier
ORDER BY mean_d_a;

-- Average d_a per (cell, identifier, seed)
SELECT cell, identifier, seed, AVG(d_a) AS mean_d_a, COUNT(*) AS n
FROM read_parquet('output/loop-2026-05-12/results.parquet')
WHERE record_kind = 'composition' AND scored
GROUP BY cell, identifier, seed
ORDER BY cell, identifier, seed;

-- Per-element relative error distribution
SELECT t.element, MEDIAN(t.value) AS median_rel_err
FROM read_parquet('output/loop-2026-05-12/results.parquet'),
     UNNEST(per_element_relative_error) AS t
WHERE record_kind = 'composition'
GROUP BY t.element;
```

## Backwards compatibility

`UnifiedBenchmarkRunner.write_outputs(...)` accepts an `output_format`
argument:

- `"parquet"` (default) — writes `results.parquet` plus all
  non-records files (CSVs, summaries, plots, statistics). Skips the
  legacy `id_records.json` and `composition_records.json` payloads.
- `"json"` — writes the legacy bundle. No parquet file.
- `"both"` — writes both. Use during dogfood transition.

The CLI flag `--output-format` on
`scripts/run_unified_benchmark.py` exposes this. The env var
`CFLIBS_OUTPUT_FORMAT` provides a non-CLI escape hatch for loop
drivers.

## Schema-version policy

`schema_version = 1` is what this file documents. Future incompatible
changes bump the integer and **must** record the diff in this document.
Readers may filter by `schema_version` to multiplex across runs.
