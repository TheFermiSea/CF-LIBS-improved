"""Per-element confusion-matrix aggregation for LIBS benchmark runs.

Parses per-iteration ``id_records.csv`` files and computes per-element
precision/recall/F1 for every (workflow, dataset, element) cell observed
across all shards and iterations of an experiment.

The headline benchmark metrics (``f1``, ``false_positives_per_spectrum``)
say *how much* a workflow is over-predicting but not *which elements*
drive the error. This aggregator answers questions like:

- "Is Ti both over- and under-predicted, or systematically one way?"
- "Are Pb/U over-predicted on noisy Vrabel spectra specifically, or
  across all datasets?"
- "Which workflows agree on the same problematic elements?"

Input columns required from each CSV row:
``workflow_name``, ``dataset_id``, ``predicted_elements`` (JSON list),
``true_elements`` (JSON list). Every other column is ignored.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_list_cell(value: object) -> list[str]:
    """Tolerant list parser for ``predicted_elements`` / ``true_elements``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
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
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(x) for x in parsed]


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


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _f1(precision: float, recall: float) -> float:
    if precision != precision or recall != recall:  # NaN guard
        return float("nan")
    denom = precision + recall
    if denom <= 0:
        return float("nan")
    return 2.0 * precision * recall / denom


def aggregate_confusion(csv_paths: Iterable[Path]) -> dict[str, pd.DataFrame]:
    """Aggregate per-element confusion across the given id_records CSVs.

    Parameters
    ----------
    csv_paths
        Iterable of paths to ``id_records.csv`` files.

    Returns
    -------
    dict with three DataFrames:

    ``per_element`` — one row per (workflow, dataset, element).
        Columns: ``workflow``, ``dataset``, ``element``, ``support``,
        ``tp``, ``fp``, ``fn``, ``precision``, ``recall``, ``f1``,
        ``n_spectra``. Sorted by FP descending.

    ``cross_workflow`` — one row per element from the top-10 elements
        with most FP across all workflows. Columns: ``element`` plus
        one ``<workflow>_fp`` column per workflow.

    ``overpred_per_dataset`` — one row per (workflow, dataset).
        Columns: ``workflow``, ``dataset``, ``n_spectra``,
        ``overpred_rate`` (fraction of spectra with ≥1 FP),
        ``top_overpredicted_elements`` (comma-joined "El (xx.x%)" of the
        top-3 elements by FP-rate).
    """
    paths = [Path(p) for p in csv_paths]
    df = _load_csvs(paths)

    per_element_cols = [
        "workflow",
        "dataset",
        "element",
        "support",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "n_spectra",
    ]
    overpred_cols = [
        "workflow",
        "dataset",
        "n_spectra",
        "overpred_rate",
        "top_overpredicted_elements",
    ]

    if df.empty:
        return {
            "per_element": pd.DataFrame(columns=per_element_cols),
            "cross_workflow": pd.DataFrame(columns=["element"]),
            "overpred_per_dataset": pd.DataFrame(columns=overpred_cols),
        }

    for col in ("workflow_name", "dataset_id"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("string").fillna("")

    df["_predicted"] = df.get("predicted_elements", pd.Series([None] * len(df))).map(
        _parse_list_cell
    )
    df["_true"] = df.get("true_elements", pd.Series([None] * len(df))).map(_parse_list_cell)

    # Per (workflow, dataset) spectrum count + tracking of overprediction.
    per_wd_counts = df.groupby(["workflow_name", "dataset_id"]).size().rename("n_spectra")

    # Build per-element accumulator by exploding into a long row-per-element-per-spectrum table.
    # For each spectrum, the union of predicted+true elements is the set of elements with non-zero tp/fp/fn contribution.
    records: list[dict] = []
    overpred_rows: list[dict] = []
    for (wf, ds), group in df.groupby(["workflow_name", "dataset_id"]):
        # Aggregate counts per element.
        tp: dict[str, int] = {}
        fp: dict[str, int] = {}
        fn: dict[str, int] = {}
        n_spectra = len(group)
        n_overpred_spectra = 0
        for _, row in group.iterrows():
            pred = set(row["_predicted"])
            true = set(row["_true"])
            if pred - true:
                n_overpred_spectra += 1
            for el in pred & true:
                tp[el] = tp.get(el, 0) + 1
            for el in pred - true:
                fp[el] = fp.get(el, 0) + 1
            for el in true - pred:
                fn[el] = fn.get(el, 0) + 1

        all_elements = sorted(set(tp) | set(fp) | set(fn))
        for el in all_elements:
            tp_c = tp.get(el, 0)
            fp_c = fp.get(el, 0)
            fn_c = fn.get(el, 0)
            precision = _safe_div(tp_c, tp_c + fp_c)
            recall = _safe_div(tp_c, tp_c + fn_c)
            records.append(
                {
                    "workflow": wf,
                    "dataset": ds,
                    "element": el,
                    "support": tp_c + fn_c,
                    "tp": tp_c,
                    "fp": fp_c,
                    "fn": fn_c,
                    "precision": precision,
                    "recall": recall,
                    "f1": _f1(precision, recall),
                    "n_spectra": n_spectra,
                }
            )

        # Overprediction rollup for this (workflow, dataset).
        overpred_rate = _safe_div(n_overpred_spectra, n_spectra)
        # Top-3 elements by FP count within this cell.
        cell_fps = [(el, c) for el, c in fp.items() if c > 0]
        cell_fps.sort(key=lambda x: x[1], reverse=True)
        top3 = cell_fps[:3]
        top_str = (
            ", ".join(f"{el} ({(c / n_spectra) * 100:.1f}%)" for el, c in top3) if top3 else ""
        )
        overpred_rows.append(
            {
                "workflow": wf,
                "dataset": ds,
                "n_spectra": n_spectra,
                "overpred_rate": overpred_rate,
                "top_overpredicted_elements": top_str,
            }
        )

    per_element = pd.DataFrame(records, columns=per_element_cols)
    if not per_element.empty:
        per_element = per_element.sort_values(
            ["fp", "workflow", "dataset", "element"], ascending=[False, True, True, True]
        ).reset_index(drop=True)

    overpred_per_dataset = pd.DataFrame(overpred_rows, columns=overpred_cols)
    if not overpred_per_dataset.empty:
        overpred_per_dataset = overpred_per_dataset.sort_values(
            ["overpred_rate", "workflow", "dataset"], ascending=[False, True, True]
        ).reset_index(drop=True)

    # Cross-workflow comparison: top-10 elements by total FP across workflows.
    if per_element.empty:
        cross = pd.DataFrame(columns=["element"])
    else:
        total_fp_by_el = per_element.groupby("element")["fp"].sum().sort_values(ascending=False)
        top_elements = list(total_fp_by_el.head(10).index)
        pivot = per_element.groupby(["element", "workflow"])["fp"].sum().unstack(fill_value=0)
        cross = pivot.loc[top_elements].copy().reset_index()
        # Suffix workflow columns with _fp for clarity, preserve column ordering.
        rename_map = {c: f"{c}_fp" for c in cross.columns if c != "element"}
        cross = cross.rename(columns=rename_map)

    # Reference per_wd_counts to silence unused-variable lint; the value is already inlined.
    _ = per_wd_counts
    return {
        "per_element": per_element,
        "cross_workflow": cross,
        "overpred_per_dataset": overpred_per_dataset,
    }


def _fmt_float(v: object) -> str:
    if v is None:
        return "nan"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "nan"
    if f != f:  # NaN
        return "nan"
    return f"{f:.3f}"


def render_markdown(
    result: dict[str, pd.DataFrame],
    *,
    top_n_per_element: int = 50,
    source_count: int | None = None,
) -> str:
    """Render the confusion aggregation as a Markdown report."""
    per_element: pd.DataFrame = result["per_element"]
    cross_workflow: pd.DataFrame = result["cross_workflow"]
    overpred_per_dataset: pd.DataFrame = result["overpred_per_dataset"]

    lines: list[str] = ["# Per-element confusion matrix", ""]
    if source_count is not None:
        lines += [f"Parsed **{source_count}** `id_records.csv` file(s).", ""]

    lines += [
        f"## A) Per (workflow, dataset, element) — top {top_n_per_element} by FP",
        "",
    ]
    if per_element.empty:
        lines += ["_No element confusion data._", ""]
    else:
        lines += [
            "| workflow | dataset | element | support | tp | fp | fn | precision | recall | f1 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, row in per_element.head(top_n_per_element).iterrows():
            lines.append(
                f"| {row['workflow']} | {row['dataset']} | {row['element']} "
                f"| {int(row['support'])} | {int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} "
                f"| {_fmt_float(row['precision'])} | {_fmt_float(row['recall'])} | {_fmt_float(row['f1'])} |"
            )
        lines.append("")

    lines += ["## B) Cross-workflow comparison (top-10 elements by total FP)", ""]
    if cross_workflow.empty:
        lines += ["_No cross-workflow data._", ""]
    else:
        cols = list(cross_workflow.columns)
        lines += [
            "| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|",
        ]
        for _, row in cross_workflow.iterrows():
            cells = [str(row[c]) if c == "element" else str(int(row[c])) for c in cols]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## C) Per (workflow, dataset) over-prediction rate", ""]
    if overpred_per_dataset.empty:
        lines += ["_No over-prediction data._", ""]
    else:
        lines += [
            "| workflow | dataset | n_spectra | overpred_rate | top_overpredicted_elements |",
            "|---|---|---|---|---|",
        ]
        for _, row in overpred_per_dataset.iterrows():
            rate = row["overpred_rate"]
            rate_str = f"{rate * 100:.1f}%" if rate == rate else "n/a"  # NaN guard
            lines.append(
                f"| {row['workflow']} | {row['dataset']} | {int(row['n_spectra'])} "
                f"| {rate_str} | {row['top_overpredicted_elements']} |"
            )
        lines.append("")

    return "\n".join(lines)
