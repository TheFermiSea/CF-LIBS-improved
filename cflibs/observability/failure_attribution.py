"""Failure-mode attribution for LIBS benchmark runs.

Parses per-iteration ``id_records.csv`` files emitted by the benchmark
harness and buckets each row into named failure categories. Categories
are evaluated from the ``annotations`` JSON column plus the standard
score columns (``f1``, ``predicted_elements``, ``true_elements``).

Failure modes
-------------
- ``basis_fwhm_mismatch_large``: ``annotations.basis_fwhm_mismatch_nm > 0.05``
- ``high_residual_norm``: ``annotations.residual_norm > 0.5``
  (applies to ``spectral_nnls`` / ``hybrid_*`` workflows)
- ``no_lines_detected``: ``annotations.n_detected < 3``
- ``empty_predicted_elements``: predicted list empty AND true list non-empty
- ``large_overprediction``: ``len(predicted) - len(true) > 5``
- ``zero_f1``: ``f1 == 0.0`` AND true list non-empty (catch-all)

A single row may match multiple failure modes; each contributes
independently to per-mode counts. The same row is counted only once for
``per_spectrum`` rollups when computing how many *distinct workflows*
flagged a given spectrum.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds (centralised so tests and the CLI agree)
THRESH_BASIS_FWHM_NM = 0.05
THRESH_RESIDUAL_NORM = 0.5
THRESH_MIN_LINES = 3
THRESH_OVERPRED_DELTA = 5

# Workflows for which residual_norm is a meaningful failure signal.
# Other workflows (e.g. ``alias``) don't fit a continuum and emit no
# residual_norm, so we skip the check there to avoid spurious flags.
RESIDUAL_NORM_WORKFLOWS = ("spectral_nnls",)
RESIDUAL_NORM_WORKFLOW_PREFIXES = ("hybrid_",)

FAILURE_MODES = (
    "basis_fwhm_mismatch_large",
    "high_residual_norm",
    "no_lines_detected",
    "empty_predicted_elements",
    "large_overprediction",
    "zero_f1",
)


def _parse_json_cell(value: object) -> dict:
    """Tolerant JSON parser for the ``annotations`` column."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fall back to literal_eval for single-quoted dict reprs.
        try:
            result = ast.literal_eval(text)
            return result if isinstance(result, dict) else {}
        except (ValueError, SyntaxError):
            return {}


def _parse_list_cell(value: object) -> list:
    """Tolerant list parser for ``predicted_elements`` / ``true_elements``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def _residual_check_applies(workflow: str) -> bool:
    if workflow in RESIDUAL_NORM_WORKFLOWS:
        return True
    return any(workflow.startswith(p) for p in RESIDUAL_NORM_WORKFLOW_PREFIXES)


def _classify_row(row: pd.Series) -> list[str]:
    """Return the list of failure modes a single row triggers."""
    modes: list[str] = []
    ann = _parse_json_cell(row.get("annotations"))
    workflow = str(row.get("workflow_name") or "")
    predicted = _parse_list_cell(row.get("predicted_elements"))
    true = _parse_list_cell(row.get("true_elements"))

    fwhm = ann.get("basis_fwhm_mismatch_nm")
    if isinstance(fwhm, (int, float)) and fwhm > THRESH_BASIS_FWHM_NM:
        modes.append("basis_fwhm_mismatch_large")

    if _residual_check_applies(workflow):
        residual = ann.get("residual_norm")
        if isinstance(residual, (int, float)) and residual > THRESH_RESIDUAL_NORM:
            modes.append("high_residual_norm")

    n_detected = ann.get("n_detected")
    if isinstance(n_detected, (int, float)) and n_detected < THRESH_MIN_LINES:
        modes.append("no_lines_detected")

    if not predicted and true:
        modes.append("empty_predicted_elements")

    if (len(predicted) - len(true)) > THRESH_OVERPRED_DELTA:
        modes.append("large_overprediction")

    f1 = row.get("f1")
    try:
        f1_val = float(f1) if f1 is not None else None
    except (TypeError, ValueError):
        f1_val = None
    if f1_val == 0.0 and true:
        modes.append("zero_f1")

    return modes


def _load_csvs(csv_paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.warning("skipping unreadable id_records.csv %s: %s", path, exc)
            continue
        if df.empty:
            continue
        df = df.copy()
        df["_source_csv"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def attribute_failures(csv_paths: Iterable[Path]) -> dict:
    """Aggregate failure attributions across the given id_records CSVs.

    Parameters
    ----------
    csv_paths
        Iterable of paths to ``id_records.csv`` files.

    Returns
    -------
    dict
        ``{"per_mode": DataFrame, "per_spectrum": DataFrame}``.

        ``per_mode`` columns: ``workflow``, ``dataset``, ``failure_mode``,
        ``count``, ``n_spectra``, ``rate``. ``rate`` is ``count /
        n_spectra`` where ``n_spectra`` is the total number of rows for
        that (workflow, dataset) cell.

        ``per_spectrum`` columns: ``spectrum_id``, ``dataset``,
        ``n_workflows``, ``failure_modes`` (comma-sorted unique list).
    """
    paths = [Path(p) for p in csv_paths]
    df = _load_csvs(paths)
    if df.empty:
        empty_mode = pd.DataFrame(
            columns=["workflow", "dataset", "failure_mode", "count", "n_spectra", "rate"]
        )
        empty_spec = pd.DataFrame(
            columns=["spectrum_id", "dataset", "n_workflows", "failure_modes"]
        )
        return {"per_mode": empty_mode, "per_spectrum": empty_spec}

    # Normalise the columns we depend on.
    for col in ("workflow_name", "dataset_id", "spectrum_id"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("string").fillna("")

    # Classify each row.
    df["_modes"] = df.apply(_classify_row, axis=1)

    # n_spectra denominator per (workflow, dataset). Rows themselves are
    # the spectrum-eval unit in id_records.csv.
    denom = df.groupby(["workflow_name", "dataset_id"]).size().rename("n_spectra").reset_index()

    # Long-form: one row per (csv-row, mode).
    exploded = df[["workflow_name", "dataset_id", "spectrum_id", "_modes"]].explode("_modes")
    exploded = exploded[exploded["_modes"].notna()]

    if exploded.empty:
        per_mode = pd.DataFrame(
            columns=["workflow", "dataset", "failure_mode", "count", "n_spectra", "rate"]
        )
    else:
        per_mode = (
            exploded.groupby(["workflow_name", "dataset_id", "_modes"])
            .size()
            .rename("count")
            .reset_index()
            .rename(
                columns={
                    "workflow_name": "workflow",
                    "dataset_id": "dataset",
                    "_modes": "failure_mode",
                }
            )
        )
        per_mode = per_mode.merge(
            denom.rename(columns={"workflow_name": "workflow", "dataset_id": "dataset"}),
            on=["workflow", "dataset"],
            how="left",
        )
        per_mode["rate"] = per_mode["count"] / per_mode["n_spectra"].where(
            per_mode["n_spectra"] > 0, other=pd.NA
        )
        per_mode = per_mode.sort_values(
            ["workflow", "dataset", "count"], ascending=[True, True, False]
        ).reset_index(drop=True)

    # Per-spectrum rollup: count distinct workflows flagging each
    # spectrum, and concat the union of failure modes.
    failing = exploded.copy()
    if failing.empty:
        per_spectrum = pd.DataFrame(
            columns=["spectrum_id", "dataset", "n_workflows", "failure_modes"]
        )
    else:
        per_spectrum = (
            failing.groupby(["spectrum_id", "dataset_id"])
            .agg(
                n_workflows=("workflow_name", lambda s: s.nunique()),
                failure_modes=(
                    "_modes",
                    lambda s: ",".join(sorted(set(s.dropna().astype(str)))),
                ),
                _failure_count=("_modes", "size"),
            )
            .reset_index()
            .rename(columns={"dataset_id": "dataset"})
        )
        per_spectrum = per_spectrum.sort_values(
            ["_failure_count", "n_workflows"], ascending=[False, False]
        ).reset_index(drop=True)
        per_spectrum = per_spectrum.drop(columns=["_failure_count"])

    return {"per_mode": per_mode, "per_spectrum": per_spectrum}


def render_markdown(
    result: dict,
    *,
    top_n_spectra: int = 20,
    source_count: int | None = None,
) -> str:
    """Render the attribution result as a Markdown report."""
    per_mode: pd.DataFrame = result["per_mode"]
    per_spectrum: pd.DataFrame = result["per_spectrum"]

    lines: list[str] = ["# Failure-mode attribution", ""]
    if source_count is not None:
        lines += [f"Parsed **{source_count}** `id_records.csv` file(s).", ""]
    lines += ["## A) Aggregated counts per (workflow, dataset, failure_mode)", ""]
    if per_mode.empty:
        lines += ["_No failures detected._", ""]
    else:
        lines += [
            "| workflow | dataset | failure_mode | count | n_spectra | rate |",
            "|---|---|---|---|---|---|",
        ]
        for _, row in per_mode.iterrows():
            rate = row["rate"]
            rate_str = f"{rate * 100:.1f}%" if pd.notna(rate) else "n/a"
            lines.append(
                f"| {row['workflow']} | {row['dataset']} | {row['failure_mode']} "
                f"| {int(row['count'])} | {int(row['n_spectra'])} | {rate_str} |"
            )
        lines.append("")

    lines += [
        f"## B) Top {top_n_spectra} failing spectra (by failure-mode count across workflows)",
        "",
    ]
    if per_spectrum.empty:
        lines += ["_No failing spectra._", ""]
    else:
        top = per_spectrum.head(top_n_spectra)
        lines += [
            "| spectrum_id | dataset | n_workflows | failure_modes |",
            "|---|---|---|---|",
        ]
        for _, row in top.iterrows():
            lines.append(
                f"| {row['spectrum_id']} | {row['dataset']} | {int(row['n_workflows'])} "
                f"| {row['failure_modes']} |"
            )
        lines.append("")

    return "\n".join(lines)
