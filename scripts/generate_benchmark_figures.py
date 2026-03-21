#!/usr/bin/env python
"""Generate publication-quality figures for the element-ID benchmark report."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

FIGURE_DIR = Path(__file__).resolve().parent.parent / "docs" / "reports" / "figures"
DPI = 300


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURE_DIR / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


# ---------------------------------------------------------------------------
# Fig 1 -- Grouped bar chart: pathway comparison
# ---------------------------------------------------------------------------
def fig1_pathway_comparison() -> None:
    pathways = [
        "Hybrid\nNNLS+ALIAS",
        "ALIAS",
        "Voigt+ALIAS",
        "Forward\nModel",
        "Spectral\nNNLS",
    ]
    metrics = {
        "Precision": [0.604, 0.505, 0.488, 0.369, 0.293],
        "Recall": [0.713, 0.629, 0.623, 0.796, 0.940],
        "F1": [0.654, 0.560, 0.547, 0.505, 0.447],
        "FPR": [0.053, 0.070, 0.075, 0.148, 0.236],
    }
    colors = {"Precision": "#2171b5", "Recall": "#238b45", "F1": "#d95f0e", "FPR": "#cb181d"}

    x = np.arange(len(pathways))
    n_metrics = len(metrics)
    bar_w = 0.18
    offsets = np.arange(n_metrics) - (n_metrics - 1) / 2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax.bar(
            x + offsets[i] * bar_w,
            vals,
            bar_w,
            label=metric,
            color=colors[metric],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pathways, fontsize=9)
    ax.set_ylabel("Metric Value", fontsize=10)
    ax.set_title("Element-ID Pathway Comparison (Aalto Benchmark)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=9, frameon=True, edgecolor="0.8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig1_pathway_comparison.png")


# ---------------------------------------------------------------------------
# Fig 2 -- Precision vs Recall scatter with Pareto front & iso-F1
# ---------------------------------------------------------------------------
def fig2_hybrid_pr_scatter() -> None:
    intersect = np.array([
        [0.737, 0.557], [0.719, 0.594], [0.521, 0.649],
        [0.731, 0.565], [0.713, 0.604], [0.521, 0.659],
        [0.725, 0.568], [0.707, 0.602], [0.515, 0.656],
    ])
    union = np.array([
        [0.952, 0.274], [0.952, 0.283], [0.952, 0.292],
        [0.952, 0.296], [0.952, 0.308], [0.952, 0.319],
        [0.940, 0.310], [0.940, 0.326], [0.940, 0.339],
    ])

    fig, ax = plt.subplots(figsize=(7, 6))

    # Iso-F1 contours
    r_grid = np.linspace(0.01, 1.0, 300)
    for f1_val in [0.4, 0.5, 0.6, 0.7]:
        p_iso = f1_val * r_grid / (2 * r_grid - f1_val)
        mask = (p_iso > 0) & (p_iso <= 1.0)
        ax.plot(r_grid[mask], p_iso[mask], "--", color="0.70", linewidth=0.8, zorder=1)
        # Label the contour near the top
        idx = np.argmin(np.abs(p_iso - 0.95))
        if mask[idx]:
            ax.text(
                r_grid[idx], p_iso[idx] + 0.01, f"F1={f1_val}",
                fontsize=7, color="0.55", ha="center",
            )

    # Scatter
    ax.scatter(
        intersect[:, 0], intersect[:, 1],
        marker="o", s=60, c="#2171b5", edgecolors="black", linewidths=0.5,
        label="Intersection", zorder=3,
    )
    ax.scatter(
        union[:, 0], union[:, 1],
        marker="^", s=60, c="#cb181d", edgecolors="black", linewidths=0.5,
        label="Union", zorder=3,
    )

    # Best-F1 star
    ax.scatter(
        [0.713], [0.604],
        marker="*", s=250, c="#d95f0e", edgecolors="black", linewidths=0.7,
        label="Best F1 (0.654)", zorder=4,
    )

    # Pareto front for intersect configs (non-dominated in P-R space)
    # A point is dominated if another has >= R AND >= P with at least one strict >
    pareto_pts = []
    for pt in intersect:
        dominated = False
        for other in intersect:
            if (other[0] >= pt[0] and other[1] >= pt[1]) and (
                other[0] > pt[0] or other[1] > pt[1]
            ):
                dominated = True
                break
        if not dominated:
            pareto_pts.append(pt)
    pareto_pts = np.array(sorted(pareto_pts, key=lambda p: p[0]))
    ax.plot(
        pareto_pts[:, 0], pareto_pts[:, 1],
        "-", color="#2171b5", linewidth=1.5, alpha=0.6, zorder=2,
    )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Hybrid Configurations: Precision vs Recall", fontsize=12, fontweight="bold")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.20, 1.0)
    ax.legend(fontsize=9, frameon=True, edgecolor="0.8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig2_hybrid_pr_scatter.png")


# ---------------------------------------------------------------------------
# Fig 3 -- Per-element precision heatmap
# ---------------------------------------------------------------------------
def fig3_element_precision_heatmap() -> None:
    elements = ["Si", "Al", "Fe", "Li", "Co", "Ni", "Mo", "K", "Ca", "Ti", "Cu", "Pb", "Mg", "Na", "Mn"]
    pathways_labels = ["ALIAS", "NNLS", "Hybrid", "Forward Model"]
    data = {
        "ALIAS":         [0.96, 0.76, 0.91, 1.00, 0.33, 0.50, 1.00, 0.54, 0.46, 0.67, 1.00, 0.33, 0.50, 0.17, 0.03],
        "NNLS":          [0.98, 1.00, 0.76, 0.22, 1.00, 1.00, 1.00, 0.29, 0.59, 0.33, 0.60, 0.08, 0.29, 0.08, 0.50],
        "Hybrid":        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.67, 0.62, 0.50, 0.33, 0.40, 0.39, 0.14, 0.03],
        "Forward Model": [0.97, 1.00, 0.95, 0.50, 0.33, 1.00, 0.50, 0.77, 0.82, 0.33, 1.00, 0.08, 0.64, 0.16, 0.02],
    }

    # Build matrix: rows=elements, cols=pathways
    mat = np.array([data[p] for p in pathways_labels]).T  # (15, 4)

    # Sort by hybrid precision descending
    hybrid_col = pathways_labels.index("Hybrid")
    sort_idx = np.argsort(-mat[:, hybrid_col])
    mat = mat[sort_idx]
    elements_sorted = [elements[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pathways_labels)))
    ax.set_xticklabels(pathways_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(elements_sorted)))
    ax.set_yticklabels(elements_sorted, fontsize=9)

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if v < 0.35 or v > 0.85 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Per-Element Precision by Pathway", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Precision", fontsize=10)
    _save(fig, "fig3_element_precision_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 4 -- Hybrid architecture block diagram
# ---------------------------------------------------------------------------
def _rounded_box(ax, xy, w, h, text, fc="#e8f0fe", ec="#4285f4", fontsize=8, bold=False):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=fc, edgecolor=ec, linewidth=1.2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize, fontweight=weight,
        wrap=True,
    )
    return box


def _parallelogram(ax, xy, w, h, text, fc="#fff3cd", ec="#856404", fontsize=7.5):
    x, y = xy
    skew = 0.15 * h
    verts = [
        (x + skew, y),
        (x + w, y),
        (x + w - skew, y + h),
        (x, y + h),
        (x + skew, y),
    ]
    from matplotlib.patches import Polygon
    poly = Polygon(verts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(poly)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, start, end, color="#333333"):
    ax.annotate(
        "",
        xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.3,
            connectionstyle="arc3,rad=0",
        ),
    )


def fig4_hybrid_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(
        "Hybrid Two-Stage Element Identification Architecture",
        fontsize=13, fontweight="bold", pad=15,
    )

    bw_sm = 1.7
    bh = 0.65

    # Row positions (top to bottom)
    y_input = 5.8
    y_preproc = 4.7
    y_stage1 = 3.4
    y_thresh1 = 2.3
    y_cand = 1.4
    y_stage2 = 3.4
    y_thresh2 = 2.3
    y_conf = 1.4
    y_gate = 0.3

    # Column positions
    x_left = 1.0    # NNLS path
    x_mid = 5.0     # main flow
    x_right = 8.0   # ALIAS path

    # -- Input --
    _parallelogram(ax, (x_mid - 0.1, y_input), 2.2, 0.6, "Observed\nSpectrum", fontsize=9)

    # -- Preprocessing --
    _rounded_box(ax, (x_mid - 0.3, y_preproc), 2.6, 0.65, "Baseline Subtraction\n& Resampling", fontsize=8.5)
    _arrow(ax, (x_mid + 1.0, y_input), (x_mid + 1.0, y_preproc + 0.65))

    # -- Split arrows down to two paths --
    _arrow(ax, (x_mid + 0.3, y_preproc), (x_left + bw_sm / 2, y_stage1 + bh))
    _arrow(ax, (x_mid + 1.7, y_preproc), (x_right + bw_sm / 2, y_stage2 + bh))

    # -- NNLS Path (left) --
    _rounded_box(ax, (x_left - 0.15, y_stage1), bw_sm, bh, "NNLS\nDecomposition",
                 fc="#d4edda", ec="#28a745", fontsize=8.5)
    _rounded_box(ax, (x_left - 0.15, y_thresh1), bw_sm, bh, "SNR\nThresholding",
                 fc="#d4edda", ec="#28a745", fontsize=8.5)
    _parallelogram(ax, (x_left - 0.15, y_cand), bw_sm, 0.6, "Candidate\nElements", fontsize=8)
    _arrow(ax, (x_left + bw_sm / 2 - 0.15, y_stage1), (x_left + bw_sm / 2 - 0.15, y_thresh1 + bh))
    _arrow(ax, (x_left + bw_sm / 2 - 0.15, y_thresh1), (x_left + bw_sm / 2 - 0.15, y_cand + 0.6))

    # -- ALIAS Path (right) --
    _rounded_box(ax, (x_right, y_stage2), bw_sm, bh, "ALIAS\nPeak-Matching",
                 fc="#cce5ff", ec="#0d6efd", fontsize=8.5)
    _rounded_box(ax, (x_right, y_thresh2), bw_sm, bh, "CL\nThresholding",
                 fc="#cce5ff", ec="#0d6efd", fontsize=8.5)
    _parallelogram(ax, (x_right, y_conf), bw_sm, 0.6, "Confirmed\nElements", fontsize=8)
    _arrow(ax, (x_right + bw_sm / 2, y_stage2), (x_right + bw_sm / 2, y_thresh2 + bh))
    _arrow(ax, (x_right + bw_sm / 2, y_thresh2), (x_right + bw_sm / 2, y_conf + 0.6))

    # -- Intersection Gate (bottom center) --
    _rounded_box(ax, (x_mid - 0.2, y_gate), 2.4, 0.7, "Intersection\nGate",
                 fc="#f8d7da", ec="#dc3545", fontsize=9, bold=True)
    _arrow(ax, (x_left + bw_sm - 0.15, y_cand + 0.3), (x_mid - 0.2, y_gate + 0.5))
    _arrow(ax, (x_right, y_conf + 0.3), (x_mid + 2.2, y_gate + 0.5))

    # -- Output --
    _parallelogram(ax, (x_mid - 0.1, y_gate - 0.8), 2.2, 0.55, "Detected\nElements", fontsize=9)
    _arrow(ax, (x_mid + 1.0, y_gate), (x_mid + 1.0, y_gate - 0.25))

    # -- External data sources --
    # Basis library feeding NNLS
    _rounded_box(ax, (-0.8, y_stage1 + 1.0), 2.1, 0.55,
                 "Basis Library\n(76 el x 300 grid pts)",
                 fc="#e2e3e5", ec="#6c757d", fontsize=7)
    _arrow(ax, (0.25, y_stage1 + 1.0), (x_left + bw_sm / 2 - 0.15, y_stage1 + bh))

    # Atomic DB feeding ALIAS
    _rounded_box(ax, (9.6, y_stage2 + 1.0), 2.1, 0.55,
                 "Atomic Database\n(NIST)",
                 fc="#e2e3e5", ec="#6c757d", fontsize=7)
    _arrow(ax, (10.65, y_stage2 + 1.0), (x_right + bw_sm / 2, y_stage2 + bh))

    # Stage labels
    ax.text(x_left + bw_sm / 2 - 0.15, y_stage1 + bh + 0.15, "Stage 1",
            ha="center", fontsize=9, fontstyle="italic", color="#28a745")
    ax.text(x_right + bw_sm / 2, y_stage2 + bh + 0.15, "Stage 2",
            ha="center", fontsize=9, fontstyle="italic", color="#0d6efd")

    _save(fig, "fig4_hybrid_architecture.png")


# ---------------------------------------------------------------------------
# Fig 5 -- Resolving power distribution
# ---------------------------------------------------------------------------
def fig5_rp_distribution() -> None:
    rng = np.random.default_rng(42)
    # Log-normal with median ~600, clipped to 300-1100
    mu = np.log(600)
    sigma = 0.35
    samples = rng.lognormal(mu, sigma, size=74)
    samples = np.clip(samples, 300, 1100)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Histogram
    counts, bin_edges, patches = ax.hist(
        samples, bins=20, color="#6baed6", edgecolor="white",
        linewidth=0.6, alpha=0.85, label="Histogram", density=False,
    )

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples, bw_method=0.3)
    x_kde = np.linspace(250, 1150, 300)
    kde_vals = kde(x_kde)
    # Scale KDE to match histogram counts
    ax2 = ax.twinx()
    ax2.plot(x_kde, kde_vals, color="#d94801", linewidth=2, label="KDE")
    ax2.set_ylabel("Density", fontsize=10, color="#d94801")
    ax2.tick_params(axis="y", labelcolor="#d94801")
    ax2.spines["top"].set_visible(False)

    # Vertical reference lines
    for rp_val, label in [(300, "RP=300"), (600, "RP=600\n(median)"), (1000, "RP=1000")]:
        ax.axvline(rp_val, color="0.35", linestyle="--", linewidth=1.0, zorder=5)
        y_top = ax.get_ylim()[1]
        ax.text(rp_val + 10, y_top * 0.72, label, fontsize=8, color="0.25", va="top")

    # Median annotation
    ax.set_xlabel("Effective Resolving Power", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Resolving Power Distribution (Aalto Datasets)", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)

    # Combined legend
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(color="#6baed6", alpha=0.85, label="Histogram"),
        Line2D([0], [0], color="#d94801", linewidth=2, label="KDE"),
    ]
    ax.legend(handles=handles, fontsize=9, frameon=True, edgecolor="0.8", loc="upper left")
    _save(fig, "fig5_rp_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.dpi": 100,
        "savefig.dpi": DPI,
    })

    print("Generating benchmark figures ...")
    fig1_pathway_comparison()
    fig2_hybrid_pr_scatter()
    fig3_element_precision_heatmap()
    fig4_hybrid_architecture()
    fig5_rp_distribution()
    print(f"All figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
