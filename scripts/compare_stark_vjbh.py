#!/usr/bin/env python3
"""Compare two unified-benchmark result dirs to quantify the Stark T-factor fix impact.

Loads ``composition_records.parquet`` from a ``--before`` and an ``--after``
run, joins records on ``(dataset_id, spectrum_id, composition_workflow_name,
composition_config_name)``, and emits a markdown delta report at the
``--output`` path.

Metrics compared (paired per-spectrum so the bootstrap CI is for the *delta*,
not the absolute distributions):

* ``aitchison`` (Aitchison composition distance — primary accuracy signal)
* ``rmse`` (composition RMSE)
* ``temperature_error_frac``, ``ne_error_frac``
* Posterior diagnostics from ``posterior_diagnostics`` struct
  (``rhat_max``, ``ess_bulk_min``, ``k_hat_max``, ``divergent_count``,
  ``coverage``)
* ``elapsed_seconds`` (per-spectrum wall-time)

The ``iterative_jax`` workflow (unaffected by the Stark T-factor — it does
not call ``forward_model``) is treated as a control. Its deltas must be
zero on every metric, modulo numerical noise. A non-zero control delta
indicates RNG / environment drift between the two runs — the report
flags this prominently.

Tracked under CF-LIBS-improved-4rwe (benchmark for the vjbh fix).

Example::

    .venv/bin/python scripts/compare_stark_vjbh.py \\
        --before output/stark-fix-vjbh/before \\
        --after output/stark-fix-vjbh/after \\
        --output output/stark-fix-vjbh/delta-report.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("compare_stark_vjbh")

# Join keys uniquely identifying one paired record across before/after.
JOIN_KEYS = (
    "dataset_id",
    "spectrum_id",
    "composition_workflow_name",
    "composition_config_name",
)

# Top-level metric columns (parquet schema flat fields).
TOP_METRICS = (
    "aitchison",
    "rmse",
    "temperature_error_frac",
    "ne_error_frac",
    "elapsed_seconds",
)

# Fields under the ``posterior_diagnostics`` struct.
POSTERIOR_METRICS = (
    "rhat_max",
    "ess_bulk_min",
    "k_hat_max",
    "divergent_count",
    "coverage",
)

# Workflow used as a control — it does NOT call cflibs.radiation.kernels.forward_model
# and therefore must show exactly-zero delta between before/after on every metric.
CONTROL_WORKFLOW = "iterative_jax"


def _load_parquet(path: Path):
    """Load a unified-benchmark composition_records parquet to a pandas DataFrame.

    The parquet may not exist if the run failed; we raise a clear error
    instead of silently producing an empty diff.
    """
    import pandas as pd  # noqa: PLC0415 — local import keeps the script importable for --help

    candidate = path / "composition_records.parquet"
    if not candidate.exists():
        raise FileNotFoundError(
            f"No composition_records.parquet at {candidate}. "
            "Did the unified benchmark run write parquet output to this dir? "
            "Check --output-format on the runner."
        )
    return pd.read_parquet(candidate)


def _flatten_posterior(df):
    """Extract ``posterior_diagnostics.<field>`` into flat columns.

    pandas reads a parquet struct as a column of dicts; we unpack so the
    delta computation can treat posterior metrics like top-level numerics.
    """
    if "posterior_diagnostics" not in df.columns:
        return df
    expanded = df["posterior_diagnostics"].apply(lambda v: v if isinstance(v, dict) else {})
    for field in POSTERIOR_METRICS:
        df[field] = expanded.apply(lambda d, k=field: d.get(k) if d else None)
    return df


def _bootstrap_delta_ci(
    deltas: np.ndarray,
    *,
    iters: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the *mean* of a paired delta sample.

    Returns ``(mean, ci_lo, ci_hi)`` at ``1 - alpha`` coverage. The deltas
    are paired so this is a proper paired-sample CI on the delta
    distribution, not on either run's absolute distribution.
    """
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite))
    if iters <= 0 or finite.size < 2:
        return mean, mean, mean
    rng = rng if rng is not None else np.random.default_rng(0)
    samples = rng.choice(finite, size=(iters, finite.size), replace=True)
    means = samples.mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return mean, lo, hi


def _format_ci(mean: float, lo: float, hi: float, *, fmt: str = "{:+.4f}") -> str:
    if not np.isfinite(mean):
        return "—"
    return f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _compute_workflow_deltas(
    before, after, workflow_filter: Optional[str], iters: int
) -> List[Dict[str, object]]:
    """Per-metric paired-delta + bootstrap CI for one workflow.

    Returns rows ready for the markdown table.
    """
    import pandas as pd  # noqa: PLC0415

    if workflow_filter:
        before = before[before["composition_workflow_name"] == workflow_filter]
        after = after[after["composition_workflow_name"] == workflow_filter]

    if before.empty or after.empty:
        return []

    merged = pd.merge(
        before, after, on=list(JOIN_KEYS), how="inner", suffixes=("_before", "_after")
    )
    if merged.empty:
        return []

    rows: List[Dict[str, object]] = []
    for metric in (*TOP_METRICS, *POSTERIOR_METRICS):
        col_b = f"{metric}_before"
        col_a = f"{metric}_after"
        if col_b not in merged.columns or col_a not in merged.columns:
            continue
        deltas = (merged[col_a] - merged[col_b]).to_numpy(dtype=float)
        mean, lo, hi = _bootstrap_delta_ci(deltas, iters=iters)
        # Also report the absolute medians for context (paired n is from `merged`).
        med_b = float(np.nanmedian(merged[col_b].to_numpy(dtype=float)))
        med_a = float(np.nanmedian(merged[col_a].to_numpy(dtype=float)))
        n_paired = int(np.sum(np.isfinite(deltas)))
        rows.append(
            {
                "metric": metric,
                "median_before": med_b,
                "median_after": med_a,
                "delta_mean": mean,
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "n_paired": n_paired,
            }
        )
    return rows


def _render_table(rows: List[Dict[str, object]], heading: str) -> str:
    if not rows:
        return f"### {heading}\n\n_(no paired records)_\n\n"
    out = [f"### {heading}\n"]
    out.append("| Metric | Median before | Median after | Δ mean [95% CI] | N paired |")
    out.append("|---|---:|---:|---|---:|")
    for r in rows:
        out.append(
            f"| `{r['metric']}` "
            f"| {r['median_before']:+.4f} "
            f"| {r['median_after']:+.4f} "
            f"| {_format_ci(r['delta_mean'], r['delta_ci_lo'], r['delta_ci_hi'])} "
            f"| {r['n_paired']} |"
        )
    out.append("")
    return "\n".join(out)


def _control_summary(rows: List[Dict[str, object]]) -> str:
    """Render a verdict block on the control-workflow deltas."""
    if not rows:
        return (
            f"### Control workflow `{CONTROL_WORKFLOW}` sanity check\n\n"
            f"⚠️ **No paired records found for {CONTROL_WORKFLOW}.** Either the workflow "
            "was filtered out of one of the runs, or both before/after datasets are "
            "missing this workflow entirely. The delta report cannot certify that "
            "non-Stark drift is absent.\n\n"
        )

    drift = [
        r
        for r in rows
        if np.isfinite(r["delta_mean"])
        and abs(r["delta_mean"]) > 1e-12
        and r["metric"] != "elapsed_seconds"  # wall-time is allowed to vary
    ]
    if drift:
        bullets = "\n".join(
            f"  * `{r['metric']}`: Δ mean = "
            f"{_format_ci(r['delta_mean'], r['delta_ci_lo'], r['delta_ci_hi'])}"
            for r in drift
        )
        return (
            f"### Control workflow `{CONTROL_WORKFLOW}` sanity check\n\n"
            f"❌ **DRIFT DETECTED — non-zero delta on a workflow that does not call "
            f"the Stark kernel.** This indicates RNG, environment, or data drift "
            f"between the two runs:\n\n{bullets}\n\n"
            f"The bayesian deltas below cannot be attributed to the Stark T-factor "
            f"alone until the source of this drift is resolved.\n\n"
        )
    return (
        f"### Control workflow `{CONTROL_WORKFLOW}` sanity check\n\n"
        f"✅ All non-wallclock metrics show |Δ| < 1e-12 — the unaffected control "
        f"workflow is bit-stable across the two runs, so any deltas observed for the "
        f"`bayesian` workflow below are attributable to the Stark T-factor change.\n\n"
    )


def _render_per_dataset_aitchison(before, after) -> str:
    """Per-dataset Aitchison breakdown for the bayesian workflow."""
    import pandas as pd  # noqa: PLC0415

    bay_b = before[before["composition_workflow_name"] == "bayesian"]
    bay_a = after[after["composition_workflow_name"] == "bayesian"]
    if bay_b.empty or bay_a.empty:
        return ""
    merged = pd.merge(bay_b, bay_a, on=list(JOIN_KEYS), how="inner", suffixes=("_before", "_after"))
    if merged.empty:
        return ""
    out = ["### Per-dataset Aitchison distance (bayesian workflow)\n"]
    out.append("| Dataset | N paired | Median before | Median after | Δ mean [95% CI] |")
    out.append("|---|---:|---:|---:|---|")
    for dataset_id, sub in merged.groupby("dataset_id"):
        deltas = (sub["aitchison_after"] - sub["aitchison_before"]).to_numpy(dtype=float)
        mean, lo, hi = _bootstrap_delta_ci(deltas, iters=1000)
        med_b = float(np.nanmedian(sub["aitchison_before"].to_numpy(dtype=float)))
        med_a = float(np.nanmedian(sub["aitchison_after"].to_numpy(dtype=float)))
        n_paired = int(np.sum(np.isfinite(deltas)))
        out.append(
            f"| `{dataset_id}` | {n_paired} | {med_b:+.4f} | {med_a:+.4f} "
            f"| {_format_ci(mean, lo, hi)} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--before",
        required=True,
        type=Path,
        help="Output dir of the legacy/pre-fix run (e.g. CFLIBS_DISABLE_STARK_T_FACTOR=1).",
    )
    parser.add_argument(
        "--after",
        required=True,
        type=Path,
        help="Output dir of the post-fix run (default kernel).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the markdown delta report.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Bootstrap resamples per delta CI (default 1000).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    before = _flatten_posterior(_load_parquet(args.before))
    after = _flatten_posterior(_load_parquet(args.after))
    logger.info("Loaded %d before records, %d after records", len(before), len(after))

    control_rows = _compute_workflow_deltas(
        before, after, workflow_filter=CONTROL_WORKFLOW, iters=args.bootstrap_iters
    )
    bayesian_rows = _compute_workflow_deltas(
        before, after, workflow_filter="bayesian", iters=args.bootstrap_iters
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write("# Stark T-factor fix (CF-LIBS-improved-vjbh) — empirical delta report\n\n")
        fh.write(f"- **Before**: `{args.before}` ({len(before)} records)\n")
        fh.write(f"- **After**:  `{args.after}` ({len(after)} records)\n")
        fh.write(f"- Bootstrap CI iterations: {args.bootstrap_iters}\n")
        fh.write(
            "- Method: per-metric paired delta (join key "
            f"{JOIN_KEYS}) with {args.bootstrap_iters}-sample bootstrap 95% CI on the mean.\n\n"
        )
        fh.write(_control_summary(control_rows))
        fh.write(_render_table(bayesian_rows, "Bayesian workflow deltas"))
        fh.write(_render_per_dataset_aitchison(before, after))
        fh.write(_render_table(control_rows, f"Control `{CONTROL_WORKFLOW}` deltas (full table)"))
    logger.info("Wrote delta report to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
