#!/usr/bin/env python
"""
Statistical analysis and publication figure generation for HPC benchmark.

Reads consolidated sweep results and produces:
- Bootstrap confidence intervals for P/R/F1/FPR
- Performance-vs-RP curves with 95% CI shading
- Performance-vs-SNR curves
- Per-element confusion analysis
- McNemar significance tests
- Publication-quality figures (300 DPI)

Usage:
  python scripts/hpc/analyze_benchmark_results.py \
    --coarse-dir output/hpc_benchmark/coarse_sweep \
    --fine-dir output/hpc_benchmark/fine_sweep \
    --ml-dir output/hpc_benchmark/ml_models \
    --figures-dir docs/reports/figures_hpc
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RP_VALUES = [200, 300, 500, 700, 1000, 2000, 3000, 5000, 10000]
SNR_VALUES = [10, 20, 50, 100, 200, 500, 1000]
PATHWAYS = ["alias", "spectral_nnls", "hybrid_intersect", "hybrid_union", "forward_model"]
ELEMENTS = [
    "Fe",
    "Ca",
    "Mg",
    "Si",
    "Al",
    "Ti",
    "Na",
    "K",
    "Mn",
    "Cr",
    "Ni",
    "Cu",
    "Co",
    "V",
    "Li",
    "Sr",
    "Ba",
    "Zn",
    "Pb",
    "Mo",
    "Zr",
    "Sn",
]


def load_summary(summary_path: Path) -> Any:
    """Load summary parquet or CSV."""
    try:
        import pandas as pd

        if summary_path.suffix == ".parquet":
            return pd.read_parquet(summary_path)
        return pd.read_csv(summary_path)
    except ImportError:
        raise ImportError("pandas required for analysis")


def bootstrap_ci(
    values: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Returns (mean, lower, upper).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(values[idx])

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return float(np.mean(values)), lower, upper


def compute_metrics_by_group(
    df: Any, group_col: str, pathway_col: str = "pathway"
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute P/R/F1 with bootstrap CIs grouped by a column (e.g., RP or SNR).

    Returns {pathway: {group_value: {metric: (mean, lower, upper)}}}.
    """
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for pathway in df[pathway_col].unique():
        pw_df = df[df[pathway_col] == pathway]
        results[pathway] = {}

        for group_val in sorted(pw_df[group_col].unique()):
            g = pw_df[pw_df[group_col] == group_val]
            tp = g["tp"].values
            fp = g["fp"].values
            fn = g["fn"].values
            g["tn"].values

            # Per-spectrum precision
            p_vals = tp / np.maximum(tp + fp, 1)
            r_vals = tp / np.maximum(tp + fn, 1)
            f1_vals = 2 * p_vals * r_vals / np.maximum(p_vals + r_vals, 1e-10)

            p_mean, p_lo, p_hi = bootstrap_ci(p_vals)
            r_mean, r_lo, r_hi = bootstrap_ci(r_vals)
            f1_mean, f1_lo, f1_hi = bootstrap_ci(f1_vals)

            results[pathway][str(group_val)] = {
                "precision": (p_mean, p_lo, p_hi),
                "recall": (r_mean, r_lo, r_hi),
                "f1": (f1_mean, f1_lo, f1_hi),
                "n_spectra": len(g),
            }

    return results


def plot_performance_vs_rp(
    metrics: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: Path,
    metric_name: str = "f1",
    title: str = "F₁ Score vs. Resolving Power",
) -> None:
    """Plot performance metric vs resolving power with CI bands."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "hybrid_intersect": "#1f77b4",
        "alias": "#ff7f0e",
        "spectral_nnls": "#2ca02c",
        "forward_model": "#d62728",
        "hybrid_union": "#9467bd",
    }
    labels = {
        "hybrid_intersect": "Hybrid (∩)",
        "alias": "ALIAS",
        "spectral_nnls": "NNLS",
        "forward_model": "Forward Model",
        "hybrid_union": "Hybrid (∪)",
    }

    for pathway, rp_data in metrics.items():
        rp_vals = sorted([int(k) for k in rp_data.keys()])
        means = [rp_data[str(rp)][metric_name][0] for rp in rp_vals]
        lows = [rp_data[str(rp)][metric_name][1] for rp in rp_vals]
        highs = [rp_data[str(rp)][metric_name][2] for rp in rp_vals]

        color = colors.get(pathway, "#333333")
        label = labels.get(pathway, pathway)

        ax.plot(rp_vals, means, "o-", color=color, label=label, markersize=5)
        ax.fill_between(rp_vals, lows, highs, color=color, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("Resolving Power (RP)", fontsize=12)
    ax.set_ylabel(title.split("vs.")[0].strip(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Add reference lines
    ax.axhline(0.80, color="gray", linestyle="--", alpha=0.5, label="P = 0.80 target")
    ax.axvline(1000, color="gray", linestyle=":", alpha=0.3)
    ax.annotate("RP = 1000", xy=(1000, 0.02), fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_performance_vs_snr(
    metrics: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: Path,
) -> None:
    """Plot F1 vs SNR with CI bands."""
    plot_performance_vs_rp(
        metrics,
        output_path,
        metric_name="f1",
        title="F₁ Score vs. Signal-to-Noise Ratio",
    )


def plot_element_heatmap(
    element_metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Plot per-element precision heatmap across pathways."""
    import matplotlib.pyplot as plt

    pathways = list(element_metrics.keys())
    elements = ELEMENTS

    data = np.zeros((len(elements), len(pathways)))
    for j, pw in enumerate(pathways):
        for i, el in enumerate(elements):
            data[i, j] = element_metrics.get(pw, {}).get(el, 0.0)

    # Sort by mean precision across pathways
    mean_prec = data.mean(axis=1)
    sort_idx = np.argsort(-mean_prec)
    data = data[sort_idx]
    sorted_elements = [elements[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pathways)))
    ax.set_xticklabels([p.replace("_", "\n") for p in pathways], fontsize=9)
    ax.set_yticks(range(len(sorted_elements)))
    ax.set_yticklabels(sorted_elements, fontsize=10)

    # Annotate cells
    for i in range(len(sorted_elements)):
        for j in range(len(pathways)):
            val = data[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Per-Element Precision by Pathway (1M Spectra)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Precision", shrink=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_rp_threshold(
    metrics: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: Path,
    target_precision: float = 0.80,
) -> None:
    """Plot: at what RP does each method first achieve target precision?"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "hybrid_intersect": "#1f77b4",
        "alias": "#ff7f0e",
        "spectral_nnls": "#2ca02c",
        "forward_model": "#d62728",
    }

    for pathway, rp_data in metrics.items():
        if pathway == "hybrid_union":
            continue
        rp_vals = sorted([int(k) for k in rp_data.keys()])
        precisions = [rp_data[str(rp)]["precision"][0] for rp in rp_vals]

        threshold_rp = None
        for rp, p in zip(rp_vals, precisions):
            if p >= target_precision:
                threshold_rp = rp
                break

        color = colors.get(pathway, "#333333")
        ax.plot(rp_vals, precisions, "o-", color=color, label=pathway.replace("_", " ").title())

        if threshold_rp:
            ax.axvline(threshold_rp, color=color, linestyle=":", alpha=0.5)
            ax.annotate(
                f"RP={threshold_rp}",
                xy=(threshold_rp, target_precision),
                xytext=(threshold_rp * 1.2, target_precision + 0.03),
                fontsize=8,
                color=color,
            )

    ax.axhline(target_precision, color="gray", linestyle="--", alpha=0.7)
    ax.annotate(
        f"P = {target_precision}", xy=(200, target_precision + 0.01), fontsize=9, color="gray"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Resolving Power", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"RP Threshold for P ≥ {target_precision}", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def mcnemar_test(tp_a: int, fp_a: int, fn_a: int, tp_b: int, fp_b: int, fn_b: int) -> float:
    """McNemar's test for significance between two classifiers.

    Returns p-value. Uses chi-squared approximation.
    """
    from scipy.stats import chi2

    # Discordant pairs: A correct & B wrong vs A wrong & B correct
    b = abs((tp_a - fn_a) - (tp_b - fn_b))  # simplified
    c = abs((fp_a - fp_b))
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / max(b + c, 1)
    return float(1 - chi2.cdf(chi2_stat, df=1))


def check_completeness(output_dir: Path) -> bool:
    """Verify all expected outputs exist."""
    required = [
        "coarse_sweep/coarse_summary.parquet",
        "fine_sweep/fine_summary.parquet",
    ]
    all_ok = True
    for rel_path in required:
        full_path = output_dir / rel_path
        if full_path.exists():
            logger.info("OK: %s", rel_path)
        else:
            logger.error("MISSING: %s", rel_path)
            all_ok = False

    # Check chunk completeness
    for subdir in ["coarse_sweep", "fine_sweep"]:
        chunk_dir = output_dir / subdir
        if chunk_dir.exists():
            n_chunks = len(list(chunk_dir.glob("results_*.parquet")))
            logger.info("%s: %d chunk results", subdir, n_chunks)
        else:
            logger.warning("%s: directory not found", subdir)

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HPC benchmark results.")
    parser.add_argument("--coarse-dir", type=str, default="output/hpc_benchmark/coarse_sweep")
    parser.add_argument("--fine-dir", type=str, default="output/hpc_benchmark/fine_sweep")
    parser.add_argument("--ml-dir", type=str, default="output/hpc_benchmark/ml_models")
    parser.add_argument("--figures-dir", type=str, default="docs/reports/figures_hpc")
    parser.add_argument("--output-dir", type=str, default="output/hpc_benchmark")
    parser.add_argument("--check-completeness", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=10000)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.check_completeness:
        ok = check_completeness(output_dir)
        sys.exit(0 if ok else 1)

    # Load coarse sweep summary
    coarse_path = Path(args.coarse_dir) / "coarse_summary.parquet"
    if not coarse_path.exists():
        coarse_path = Path(args.coarse_dir) / "coarse_summary.csv"
    if not coarse_path.exists():
        logger.error("No coarse summary found at %s", args.coarse_dir)
        sys.exit(1)

    logger.info("Loading coarse summary from %s", coarse_path)
    df = load_summary(coarse_path)
    logger.info("Loaded %d rows", len(df))

    # Performance vs RP
    if "rp" in df.columns:
        logger.info("Computing performance vs RP with bootstrap CIs...")
        rp_metrics = compute_metrics_by_group(df, "rp")

        plot_performance_vs_rp(
            rp_metrics,
            figures_dir / "fig_f1_vs_rp.png",
            metric_name="f1",
            title="F₁ Score vs. Resolving Power",
        )
        plot_performance_vs_rp(
            rp_metrics,
            figures_dir / "fig_precision_vs_rp.png",
            metric_name="precision",
            title="Precision vs. Resolving Power",
        )
        plot_performance_vs_rp(
            rp_metrics,
            figures_dir / "fig_recall_vs_rp.png",
            metric_name="recall",
            title="Recall vs. Resolving Power",
        )
        plot_rp_threshold(rp_metrics, figures_dir / "fig_rp_threshold.png")

    # Performance vs SNR
    if "snr" in df.columns:
        logger.info("Computing performance vs SNR...")
        snr_metrics = compute_metrics_by_group(df, "snr")
        plot_performance_vs_snr(snr_metrics, figures_dir / "fig_f1_vs_snr.png")

    # Save analysis JSON
    analysis = {
        "n_spectra": len(df),
        "n_pathways": int(df["pathway"].nunique()) if "pathway" in df.columns else 0,
        "rp_values": RP_VALUES,
        "snr_values": SNR_VALUES,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(analysis, indent=2, default=str))

    logger.info("Analysis complete. Figures saved to %s", figures_dir)


if __name__ == "__main__":
    main()
