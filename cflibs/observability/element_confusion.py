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
from pathlib import Path
from typing import Iterable

import pandas as pd

from cflibs.benchmark.scoring import FN, FP, TP, classify_element
from cflibs.observability._csv_io import load_csvs as _load_csvs


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


_PER_ELEMENT_COLS = [
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
_OVERPRED_COLS = [
    "workflow",
    "dataset",
    "n_spectra",
    "overpred_rate",
    "top_overpredicted_elements",
]


def _empty_result() -> dict[str, pd.DataFrame]:
    """Result skeleton returned when no usable input rows exist."""
    return {
        "per_element": pd.DataFrame(columns=_PER_ELEMENT_COLS),
        "cross_workflow": pd.DataFrame(columns=["element"]),
        "overpred_per_dataset": pd.DataFrame(columns=_OVERPRED_COLS),
    }


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize key columns and parse the predicted/true list cells in-place."""
    for col in ("workflow_name", "dataset_id"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("string").fillna("")

    df["_predicted"] = df.get("predicted_elements", pd.Series([None] * len(df))).map(
        _parse_list_cell
    )
    df["_true"] = df.get("true_elements", pd.Series([None] * len(df))).map(_parse_list_cell)
    # Per-spectrum don't-care band (sub-detection-floor traces). Absent from
    # today's id_records.csv -> parses to empty, leaving confusion unchanged;
    # honored automatically once the campaign emits the column.
    df["_ignore"] = df.get("ignore_elements", pd.Series([None] * len(df))).map(_parse_list_cell)
    return df


def _count_cell(group: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int], int]:
    """Accumulate per-element tp/fp/fn and overpredicted-spectrum count for one cell.

    Confusion goes through the shared :func:`classify_element` rule so this
    aggregator cannot drift from the benchmark on the don't-care band: a
    predicted sub-detection-floor trace (an element in the per-spectrum
    ``_ignore`` band) is neither an FP here nor an over-prediction. Today's
    id_records.csv carries no don't-care column, so ``_ignore`` is empty and the
    counts are identical to the prior set-based form.
    """
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    n_overpred_spectra = 0
    for _, row in group.iterrows():
        pred = set(row["_predicted"])
        true = set(row["_true"])
        ignore = set(row.get("_ignore", ()) or ())
        if pred - true - ignore:
            n_overpred_spectra += 1
        for el in pred | true:
            label = classify_element(el, true, pred, ignore)
            if label == TP:
                tp[el] = tp.get(el, 0) + 1
            elif label == FP:
                fp[el] = fp.get(el, 0) + 1
            elif label == FN:
                fn[el] = fn.get(el, 0) + 1
    return tp, fp, fn, n_overpred_spectra


def _element_records(
    wf: object,
    ds: object,
    tp: dict[str, int],
    fp: dict[str, int],
    fn: dict[str, int],
    n_spectra: int,
) -> list[dict]:
    """Build one record per element observed in a (workflow, dataset) cell."""
    records: list[dict] = []
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
    return records


def _overpred_row(
    wf: object,
    ds: object,
    fp: dict[str, int],
    n_spectra: int,
    n_overpred_spectra: int,
) -> dict:
    """Build the overprediction rollup row for a (workflow, dataset) cell."""
    overpred_rate = _safe_div(n_overpred_spectra, n_spectra)
    # Top-3 elements by FP count within this cell.
    cell_fps = [(el, c) for el, c in fp.items() if c > 0]
    cell_fps.sort(key=lambda x: x[1], reverse=True)
    top3 = cell_fps[:3]
    top_str = ", ".join(f"{el} ({(c / n_spectra) * 100:.1f}%)" for el, c in top3) if top3 else ""
    return {
        "workflow": wf,
        "dataset": ds,
        "n_spectra": n_spectra,
        "overpred_rate": overpred_rate,
        "top_overpredicted_elements": top_str,
    }


def _build_records(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Walk each (workflow, dataset) cell, producing element + overprediction rows."""
    records: list[dict] = []
    overpred_rows: list[dict] = []
    for (wf, ds), group in df.groupby(["workflow_name", "dataset_id"]):
        tp, fp, fn, n_overpred_spectra = _count_cell(group)
        n_spectra = len(group)
        records.extend(_element_records(wf, ds, tp, fp, fn, n_spectra))
        overpred_rows.append(_overpred_row(wf, ds, fp, n_spectra, n_overpred_spectra))
    return records, overpred_rows


def _cross_workflow(per_element: pd.DataFrame) -> pd.DataFrame:
    """Top-10 elements by total FP, pivoted to one ``<workflow>_fp`` column each."""
    if per_element.empty:
        return pd.DataFrame(columns=["element"])
    total_fp_by_el = per_element.groupby("element")["fp"].sum().sort_values(ascending=False)
    top_elements = list(total_fp_by_el.head(10).index)
    pivot = per_element.groupby(["element", "workflow"])["fp"].sum().unstack(fill_value=0)
    cross = pivot.loc[top_elements].copy().reset_index()
    # Suffix workflow columns with _fp for clarity, preserve column ordering.
    rename_map = {c: f"{c}_fp" for c in cross.columns if c != "element"}
    return cross.rename(columns=rename_map)


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

    if df.empty:
        return _empty_result()

    df = _prepare_frame(df)

    # Build per-element accumulator by exploding into a long row-per-element-per-spectrum table.
    # For each spectrum, the union of predicted+true elements is the set of elements with non-zero tp/fp/fn contribution.
    records, overpred_rows = _build_records(df)

    per_element = pd.DataFrame(records, columns=_PER_ELEMENT_COLS)
    if not per_element.empty:
        per_element = per_element.sort_values(
            ["fp", "workflow", "dataset", "element"], ascending=[False, True, True, True]
        ).reset_index(drop=True)

    overpred_per_dataset = pd.DataFrame(overpred_rows, columns=_OVERPRED_COLS)
    if not overpred_per_dataset.empty:
        overpred_per_dataset = overpred_per_dataset.sort_values(
            ["overpred_rate", "workflow", "dataset"], ascending=[False, True, True]
        ).reset_index(drop=True)

    # Cross-workflow comparison: top-10 elements by total FP across workflows.
    cross = _cross_workflow(per_element)

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


def _render_per_element(per_element: pd.DataFrame, top_n_per_element: int) -> list[str]:
    """Markdown for section A) per (workflow, dataset, element)."""
    lines: list[str] = [
        f"## A) Per (workflow, dataset, element) — top {top_n_per_element} by FP",
        "",
    ]
    if per_element.empty:
        lines += ["_No element confusion data._", ""]
        return lines
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
    return lines


def _render_cross_workflow(cross_workflow: pd.DataFrame) -> list[str]:
    """Markdown for section B) cross-workflow comparison."""
    lines: list[str] = ["## B) Cross-workflow comparison (top-10 elements by total FP)", ""]
    if cross_workflow.empty:
        lines += ["_No cross-workflow data._", ""]
        return lines
    cols = list(cross_workflow.columns)
    lines += [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in cross_workflow.iterrows():
        cells = [str(row[c]) if c == "element" else str(int(row[c])) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _render_overpred(overpred_per_dataset: pd.DataFrame) -> list[str]:
    """Markdown for section C) per (workflow, dataset) over-prediction rate."""
    lines: list[str] = ["## C) Per (workflow, dataset) over-prediction rate", ""]
    if overpred_per_dataset.empty:
        lines += ["_No over-prediction data._", ""]
        return lines
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
    return lines


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

    lines += _render_per_element(per_element, top_n_per_element)
    lines += _render_cross_workflow(cross_workflow)
    lines += _render_overpred(overpred_per_dataset)

    return "\n".join(lines)
