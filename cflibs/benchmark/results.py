"""Parquet result writer for the unified benchmark (T3.1 / CF-LIBS-improved-1d5t).

Replaces the per-iteration ``composition_records.json`` +
``id_records.json`` + ``identification_records.json`` outputs with a
single columnar Parquet file per experiment. See
``docs/results-parquet-schema.md`` for the full schema reference.

The module is intentionally light on dependencies: ``pyarrow`` is the
only hard import, and it's imported lazily so that
``cflibs.benchmark.unified`` stays importable in environments that lack
it (e.g. JSON-only fallback path).

Design notes (full discussion in
``docs/parquet-results-consultation.md``):

* One file per experiment by default; optional hive partitioning via
  ``partition_cols=[...]``.
* Compositions / per-element dicts are encoded as
  ``LIST<STRUCT<key, value>>`` rather than ``MAP`` for cross-engine
  compatibility (DuckDB, polars, pandas).
* Identification + composition records share a single schema. The
  ``record_kind`` column distinguishes the two; columns specific to one
  kind are NULL on the other.
* ``schema_version = 1`` is embedded as a column so future readers can
  multiplex across runs.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCHEMA_VERSION = 1

# Per-record_kind constants used in row dicts.
KIND_IDENTIFICATION = "identification"
KIND_COMPOSITION = "composition"


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------


def _build_schema():  # type: ignore[no-untyped-def]
    """Construct the pyarrow schema for ``results.parquet`` (v1).

    Kept inside a function so callers that never write parquet don't pay
    the pyarrow import cost.
    """
    import pyarrow as pa

    kv_str_float = pa.struct(
        [pa.field("element", pa.string()), pa.field("mass_fraction", pa.float64())]
    )
    kv_str_value = pa.struct([pa.field("element", pa.string()), pa.field("value", pa.float64())])
    kv_pair_value = pa.struct([pa.field("pair", pa.string()), pa.field("value", pa.float64())])
    posterior_struct = pa.struct(
        [
            pa.field("rhat_max", pa.float64()),
            pa.field("ess_bulk_min", pa.float64()),
            pa.field("k_hat_max", pa.float64()),
            pa.field("divergent_count", pa.int64()),
            pa.field("coverage", pa.float64()),
        ]
    )

    fields = [
        # Bookkeeping
        pa.field("schema_version", pa.int32(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("record_kind", pa.string(), nullable=False),
        pa.field("record_index", pa.int64(), nullable=False),
        pa.field("timestamp_iso", pa.string(), nullable=False),
        # Run metadata (constant per write call; supplied via run_metadata)
        pa.field("cell", pa.string()),
        pa.field("identifier", pa.string()),
        pa.field("platform", pa.string()),
        pa.field("seed", pa.int64()),
        pa.field("iter_index", pa.int64()),
        pa.field("experiment_label", pa.string()),
        # Sharding columns (CF-LIBS-improved-sxt0 / T2.2). NULL on
        # unsharded runs; populated from --dataset-shard N/K.
        pa.field("shard_n", pa.int32()),
        pa.field("shard_k", pa.int32()),
        # Shared record fields
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("spectrum_id", pa.string(), nullable=False),
        pa.field("group_id", pa.string()),
        pa.field("specimen_id", pa.string()),
        pa.field("instrument_id", pa.string()),
        pa.field("truth_type", pa.string()),
        pa.field("rp_estimate", pa.float64()),
        pa.field("label_cardinality", pa.int64()),
        pa.field("spectrum_kind", pa.string()),
        pa.field("outer_split_id", pa.string()),
        pa.field("tuning_split_id", pa.string()),
        pa.field("elapsed_seconds", pa.float64()),
        pa.field("scored", pa.bool_()),
        pa.field("failure_reason", pa.string()),
        pa.field("annotations_json", pa.string()),
        # Identification-only
        pa.field("id_workflow_name", pa.string()),
        pa.field("id_config_name", pa.string()),
        pa.field("true_elements", pa.list_(pa.string())),
        pa.field("predicted_elements", pa.list_(pa.string())),
        pa.field("tp", pa.int64()),
        pa.field("fp", pa.int64()),
        pa.field("fn", pa.int64()),
        pa.field("tn", pa.int64()),
        pa.field("precision", pa.float64()),
        pa.field("recall", pa.float64()),
        pa.field("f1", pa.float64()),
        pa.field("jaccard", pa.float64()),
        pa.field("hamming_loss", pa.float64()),
        pa.field("exact_match", pa.bool_()),
        pa.field("false_positives_per_spectrum", pa.int64()),
        # Composition-only
        pa.field("comp_id_workflow_name", pa.string()),
        pa.field("composition_workflow_name", pa.string()),
        pa.field("comp_id_config_name", pa.string()),
        pa.field("composition_config_name", pa.string()),
        pa.field("candidate_elements", pa.list_(pa.string())),
        pa.field("true_composition", pa.list_(kv_str_float)),
        pa.field("predicted_composition", pa.list_(kv_str_float)),
        pa.field("aitchison", pa.float64()),
        pa.field("d_a", pa.float64()),
        pa.field("rmse", pa.float64()),
        pa.field("temperature_error_frac", pa.float64()),
        pa.field("ne_error_frac", pa.float64()),
        pa.field("closure_residual", pa.float64()),
        pa.field("error_tier", pa.string()),
        pa.field("per_element_absolute_error", pa.list_(kv_str_value)),
        pa.field("per_element_relative_error", pa.list_(kv_str_value)),
        pa.field("subcompositional_ratio_errors", pa.list_(kv_pair_value)),
        pa.field("posterior_diagnostics", posterior_struct),
    ]
    return pa.schema(fields)


# ---------------------------------------------------------------------------
# Row construction helpers
# ---------------------------------------------------------------------------


def _composition_dict_to_kv(
    data: Optional[Mapping[str, float]],
    key_name: str,
) -> Optional[List[Dict[str, Any]]]:
    """Convert ``{name: value}`` dict to ``list[dict[key_name, value]]``.

    Returns ``None`` when ``data`` is falsy so the column lands as NULL
    in parquet (cheaper on disk and unambiguous downstream).
    """
    if not data:
        return None
    rows: List[Dict[str, Any]] = []
    for raw_key, raw_value in data.items():
        if raw_value is None:
            value = None
        else:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = None
        if value is not None and math.isnan(value):
            # Preserve NaN — caller may want to distinguish "ratio
            # undefined" from "missing key". pyarrow handles NaN inside
            # float64 fields fine; just don't normalise it away.
            pass
        rows.append(
            {key_name: str(raw_key), "value" if key_name != "element" else "mass_fraction": value}
        )
    return rows


def _composition_to_struct_list(
    composition: Optional[Mapping[str, float]],
) -> Optional[List[Dict[str, Any]]]:
    if not composition:
        return None
    return [
        {
            "element": str(element),
            "mass_fraction": (float(value) if value is not None else None),
        }
        for element, value in composition.items()
    ]


def _kv_struct_list(
    data: Optional[Mapping[str, float]],
    key_field: str,
) -> Optional[List[Dict[str, Any]]]:
    if not data:
        return None
    rows: List[Dict[str, Any]] = []
    for raw_key, raw_value in data.items():
        try:
            value = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            value = None
        rows.append({key_field: str(raw_key), "value": value})
    return rows


def _posterior_from_annotations(
    annotations: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Extract posterior diagnostics from a record's ``annotations`` dict.

    The composition records emitted by the ``bayesian`` workflow attach
    posterior summaries under ``annotations["posterior"]`` (see
    ``cflibs/benchmark/posterior_metrics.py``). We tolerate both
    ``PosteriorDiagnostics``-style flat dicts and arviz-style nested
    summaries with ``rhat``/``ess_bulk``/``k_hat`` arrays.
    """
    if not annotations:
        return None
    posterior = annotations.get("posterior")
    if posterior is None:
        # Fall back to looking for flat fields directly on annotations.
        posterior = {
            k: annotations[k]
            for k in ("rhat", "ess_bulk", "k_hat", "divergent", "coverage")
            if k in annotations
        }
    if not posterior:
        return None

    def _scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            try:
                return max(float(v) for v in value if v is not None)
            except (TypeError, ValueError):
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _scalar_min(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            try:
                return min(float(v) for v in value if v is not None)
            except (TypeError, ValueError):
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _scalar_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return {
        "rhat_max": _scalar(posterior.get("rhat_max", posterior.get("rhat"))),
        "ess_bulk_min": _scalar_min(posterior.get("ess_bulk_min", posterior.get("ess_bulk"))),
        "k_hat_max": _scalar(posterior.get("k_hat_max", posterior.get("k_hat"))),
        "divergent_count": _scalar_int(
            posterior.get("divergent_count", posterior.get("divergent"))
        ),
        "coverage": _scalar(posterior.get("coverage")),
    }


def _id_record_to_row(record: Any, index: int) -> Dict[str, Any]:
    """Project an ``IDEvaluationRecord`` into the unified row schema."""
    data = asdict(record)
    annotations = data.get("annotations") or {}
    return {
        "record_kind": KIND_IDENTIFICATION,
        "record_index": int(index),
        # Shared
        "dataset_id": data.get("dataset_id"),
        "spectrum_id": data.get("spectrum_id"),
        "group_id": data.get("group_id"),
        "specimen_id": data.get("specimen_id"),
        "instrument_id": data.get("instrument_id"),
        "truth_type": data.get("truth_type"),
        "rp_estimate": data.get("rp_estimate"),
        "label_cardinality": data.get("label_cardinality"),
        "spectrum_kind": data.get("spectrum_kind"),
        "outer_split_id": data.get("outer_split_id"),
        "tuning_split_id": data.get("tuning_split_id"),
        "elapsed_seconds": data.get("elapsed_seconds"),
        "scored": data.get("scored"),
        "failure_reason": data.get("failure_reason"),
        "annotations_json": (
            json.dumps(annotations, sort_keys=True, default=str) if annotations else None
        ),
        # ID-specific
        "id_workflow_name": data.get("workflow_name"),
        "id_config_name": data.get("config_name"),
        "true_elements": list(data.get("true_elements") or []),
        "predicted_elements": list(data.get("predicted_elements") or []),
        "tp": data.get("tp"),
        "fp": data.get("fp"),
        "fn": data.get("fn"),
        "tn": data.get("tn"),
        "precision": data.get("precision"),
        "recall": data.get("recall"),
        "f1": data.get("f1"),
        "jaccard": data.get("jaccard"),
        "hamming_loss": data.get("hamming_loss"),
        "exact_match": data.get("exact_match"),
        "false_positives_per_spectrum": data.get("false_positives_per_spectrum"),
        # Composition-only fields → NULL
        "comp_id_workflow_name": None,
        "composition_workflow_name": None,
        "comp_id_config_name": None,
        "composition_config_name": None,
        "candidate_elements": None,
        "true_composition": None,
        "predicted_composition": None,
        "aitchison": None,
        "d_a": None,
        "rmse": None,
        "temperature_error_frac": None,
        "ne_error_frac": None,
        "closure_residual": None,
        "error_tier": None,
        "per_element_absolute_error": None,
        "per_element_relative_error": None,
        "subcompositional_ratio_errors": None,
        "posterior_diagnostics": None,
    }


def _composition_record_to_row(record: Any, index: int) -> Dict[str, Any]:
    """Project a ``CompositionEvaluationRecord`` into the unified row schema."""
    data = asdict(record)
    annotations = data.get("annotations") or {}
    aitchison = data.get("aitchison")
    return {
        "record_kind": KIND_COMPOSITION,
        "record_index": int(index),
        # Shared
        "dataset_id": data.get("dataset_id"),
        "spectrum_id": data.get("spectrum_id"),
        "group_id": data.get("group_id"),
        "specimen_id": data.get("specimen_id"),
        "instrument_id": data.get("instrument_id"),
        "truth_type": data.get("truth_type"),
        "rp_estimate": data.get("rp_estimate"),
        "label_cardinality": data.get("label_cardinality"),
        "spectrum_kind": data.get("spectrum_kind"),
        "outer_split_id": data.get("outer_split_id"),
        "tuning_split_id": data.get("tuning_split_id"),
        "elapsed_seconds": data.get("elapsed_seconds"),
        "scored": data.get("scored"),
        "failure_reason": data.get("failure_reason"),
        "annotations_json": (
            json.dumps(annotations, sort_keys=True, default=str) if annotations else None
        ),
        # ID-only fields → NULL
        "id_workflow_name": None,
        "id_config_name": None,
        "true_elements": None,
        "predicted_elements": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "tn": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "jaccard": None,
        "hamming_loss": None,
        "exact_match": None,
        "false_positives_per_spectrum": None,
        # Composition-specific
        "comp_id_workflow_name": data.get("id_workflow_name"),
        "composition_workflow_name": data.get("composition_workflow_name"),
        "comp_id_config_name": data.get("id_config_name"),
        "composition_config_name": data.get("composition_config_name"),
        "candidate_elements": list(data.get("candidate_elements") or []),
        "true_composition": _composition_to_struct_list(data.get("true_composition")),
        "predicted_composition": _composition_to_struct_list(data.get("predicted_composition")),
        "aitchison": aitchison,
        "d_a": aitchison,  # alias for query ergonomics
        "rmse": data.get("rmse"),
        "temperature_error_frac": data.get("temperature_error_frac"),
        "ne_error_frac": data.get("ne_error_frac"),
        "closure_residual": data.get("closure_residual"),
        "error_tier": data.get("error_tier"),
        "per_element_absolute_error": _kv_struct_list(
            data.get("per_element_absolute_error"), key_field="element"
        ),
        "per_element_relative_error": _kv_struct_list(
            data.get("per_element_relative_error"), key_field="element"
        ),
        "subcompositional_ratio_errors": _kv_struct_list(
            data.get("subcompositional_ratio_errors"), key_field="pair"
        ),
        "posterior_diagnostics": _posterior_from_annotations(annotations),
    }


def build_rows(
    id_records: Sequence[Any],
    composition_records: Sequence[Any],
    run_metadata: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Project the dataclass records into row dicts matching the unified schema.

    Exposed for testing — most callers should use :func:`write_parquet`
    directly.
    """
    run_metadata = dict(run_metadata or {})
    run_id = run_id or str(uuid.uuid4())
    timestamp = timestamp or datetime.now(timezone.utc)
    timestamp_iso = timestamp.isoformat()

    meta_block = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_iso": timestamp_iso,
        "cell": run_metadata.get("cell"),
        "identifier": run_metadata.get("identifier"),
        "platform": run_metadata.get("platform"),
        "seed": (int(run_metadata["seed"]) if run_metadata.get("seed") is not None else None),
        "iter_index": (
            int(run_metadata["iter_index"]) if run_metadata.get("iter_index") is not None else None
        ),
        "experiment_label": run_metadata.get("experiment_label"),
        "shard_n": (
            int(run_metadata["shard_n"]) if run_metadata.get("shard_n") is not None else None
        ),
        "shard_k": (
            int(run_metadata["shard_k"]) if run_metadata.get("shard_k") is not None else None
        ),
    }

    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(id_records):
        row = _id_record_to_row(record, idx)
        row.update(meta_block)
        rows.append(row)
    for idx, record in enumerate(composition_records):
        row = _composition_record_to_row(record, idx)
        row.update(meta_block)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


def write_parquet(
    output_path: Path,
    id_records: Sequence[Any] = (),
    composition_records: Sequence[Any] = (),
    run_metadata: Optional[Mapping[str, Any]] = None,
    *,
    append: bool = False,
    partition_cols: Optional[Sequence[str]] = None,
    compression: str = "snappy",
    run_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Path:
    """Write the unified parquet result file.

    Parameters
    ----------
    output_path
        Either a ``.parquet`` file (default single-file mode) or a
        directory when ``partition_cols`` is given.
    id_records, composition_records
        ``IDEvaluationRecord`` / ``CompositionEvaluationRecord`` sequences
        (or anything ``asdict``-compatible exposing the same field
        names). Either may be empty.
    run_metadata
        Optional dict of constant-per-file fields:
        ``cell``, ``identifier``, ``platform``, ``seed``,
        ``iter_index``, ``experiment_label``. Missing keys become NULL.
    append
        Single-file mode only. When ``True`` and ``output_path``
        already exists, read it, concatenate the new rows, and rewrite.
        Schema must match; if it doesn't, the call raises.
    partition_cols
        Hive-style partitioning. When set, ``output_path`` is treated
        as the root directory and pyarrow writes
        ``output_path/<col>=<val>/part-*.parquet``.
    compression
        pyarrow compression codec. Defaults to Snappy.
    run_id, timestamp
        Override the UUID / timestamp (testing).

    Returns
    -------
    Path
        The written path (file or directory).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = build_rows(
        id_records=id_records,
        composition_records=composition_records,
        run_metadata=run_metadata,
        run_id=run_id,
        timestamp=timestamp,
    )
    schema = _build_schema()

    # Build a column-oriented dict so pa.Table.from_pydict can match
    # the declared schema field-by-field (including the nested struct
    # lists). pa.Table.from_pylist works but is slower for big tables.
    if rows:
        columns: Dict[str, List[Any]] = {field.name: [] for field in schema}
        for row in rows:
            for field in schema:
                columns[field.name].append(row.get(field.name))
        table = pa.Table.from_pydict(columns, schema=schema)
    else:
        table = schema.empty_table()

    output_path = Path(output_path)

    if partition_cols:
        output_path.mkdir(parents=True, exist_ok=True)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=list(partition_cols),
            compression=compression,
        )
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append and output_path.exists():
        existing = pq.read_table(str(output_path))
        # If the existing file used an older schema, we cast onto the
        # current one. pyarrow raises on incompatible schemas.
        try:
            existing = existing.cast(schema)
        except (pa.lib.ArrowInvalid, pa.lib.ArrowNotImplementedError):
            existing = existing.select([f.name for f in schema if f.name in existing.schema.names])
            existing = existing.cast(schema, safe=False)
        table = pa.concat_tables([existing, table], promote_options="default")

    pq.write_table(table, str(output_path), compression=compression)
    return output_path


# ---------------------------------------------------------------------------
# Read helpers (thin convenience wrappers)
# ---------------------------------------------------------------------------


def read_parquet(path: Path) -> "Any":
    """Load a unified results parquet file as a pyarrow ``Table``.

    Convenience wrapper to keep callers from importing pyarrow directly
    for the common "load and inspect" case.
    """
    import pyarrow.parquet as pq

    return pq.read_table(str(Path(path)))


def parquet_available() -> bool:
    """Return ``True`` if pyarrow is importable.

    Used by callers that want to fall back to JSON output when parquet
    isn't usable in the current environment.
    """
    try:
        import pyarrow  # noqa: F401  pylint: disable=unused-import

        return True
    except ImportError:
        return False


__all__ = [
    "SCHEMA_VERSION",
    "KIND_IDENTIFICATION",
    "KIND_COMPOSITION",
    "build_rows",
    "write_parquet",
    "read_parquet",
    "parquet_available",
]
