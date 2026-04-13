#!/usr/bin/env python3
"""Generate all 7 publication-quality figures for the CF-LIBS JQSRT paper.

Reads benchmark data from docs/reports/benchmarks/figures/*.csv (V100S GPU results).
Outputs PDF figures to paper/figures/.

Usage:
    python paper/scripts/generate_figures.py

References:
    - ExoJAX (arXiv:2105.14782): JAX spectral models on GPU
    - Zaghloul (arXiv:2411.00917): Voigt profile accuracy reference
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "docs" / "reports" / "benchmarks" / "figures"
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# JQSRT publication style
# ---------------------------------------------------------------------------
# Colorblind-safe palette (Wong 2011 / Tol bright)
CPU_COLOR = "#0072B2"  # blue
GPU_COLOR = "#D55E00"  # orange
REF_COLOR = "#009E73"  # green
ACCENT_COLOR = "#CC79A7"  # pink
GRAY = "#999999"

STYLE = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": False,
    "mathtext.fontset": "cm",
}
plt.rcParams.update(STYLE)

SINGLE_COL = 3.5  # inches
DOUBLE_COL = 7.0  # inches


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  Saved {path}")


# ===================================================================
# Figure 1: Pipeline Architecture Diagram
# ===================================================================
def fig1_pipeline() -> None:
    """Pipeline architecture diagram with CPU/GPU component highlighting."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Inversion pipeline (left column, top-to-bottom)
    inv_boxes = [
        ("Measured\nSpectrum", 2.0, 7.0, False, "input"),
        ("Preprocessing\n(baseline, noise)", 2.0, 5.8, False, "cpu"),
        ("Peak Detection\n+ Line ID (ALIAS)", 2.0, 4.6, False, "cpu"),
        ("Boltzmann\nFitting", 2.0, 3.4, True, "gpu"),
        ("Saha-Boltzmann\nSolver (Anderson)", 2.0, 2.2, True, "gpu"),
        ("Closure\n(softmax)", 2.0, 1.0, True, "gpu"),
    ]

    # Forward / manifold pipeline (right column)
    fwd_boxes = [
        ("Parameter Grid\n(T, ne, C)", 9.5, 7.0, False, "input"),
        ("Batch Forward\nModel (JAX vmap)", 9.5, 5.8, True, "gpu"),
        ("Voigt Profile\nGeneration", 9.5, 4.6, True, "gpu"),
        ("Synthetic\nSpectra", 9.5, 3.4, False, "output"),
        ("FAISS Index\n(ANN lookup)", 9.5, 2.2, False, "cpu"),
    ]

    # Output box (center bottom)
    out_boxes = [
        ("T, ne, Compositions\n(with uncertainties)", 5.75, 0.2, False, "output"),
    ]

    box_w, box_h = 2.8, 0.85

    def draw_box(cx: float, cy: float, label: str, gpu: bool, kind: str) -> None:
        if kind == "input":
            fc, ec = "#E8E8E8", "#666666"
        elif kind == "output":
            fc, ec = "#D4EDDA", "#28A745"
        elif gpu:
            fc, ec = "#FFE0CC", GPU_COLOR
        else:
            fc, ec = "#CCE5FF", CPU_COLOR
        rect = FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7.5,
                fontweight="bold" if kind == "output" else "normal")

    def arrow(x1: float, y1: float, x2: float, y2: float,
              style: str = "->", color: str = "#333333") -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=1.0,
                            connectionstyle="arc3,rad=0"),
        )

    # Draw inversion pipeline
    for label, cx, cy, gpu, kind in inv_boxes:
        draw_box(cx, cy, label, gpu, kind)
    # Arrows between inversion boxes
    for i in range(len(inv_boxes) - 1):
        _, x1, y1, _, _ = inv_boxes[i]
        _, x2, y2, _, _ = inv_boxes[i + 1]
        arrow(x1, y1 - box_h / 2, x2, y2 + box_h / 2)

    # Draw forward pipeline
    for label, cx, cy, gpu, kind in fwd_boxes:
        draw_box(cx, cy, label, gpu, kind)
    for i in range(len(fwd_boxes) - 1):
        _, x1, y1, _, _ = fwd_boxes[i]
        _, x2, y2, _, _ = fwd_boxes[i + 1]
        arrow(x1, y1 - box_h / 2, x2, y2 + box_h / 2)

    # Draw output box
    for label, cx, cy, gpu, kind in out_boxes:
        draw_box(cx, cy, label, gpu, kind)

    # Connect closure -> output
    arrow(2.0, 1.0 - box_h / 2, 5.75, 0.2 + box_h / 2)
    # Connect FAISS -> output
    arrow(9.5, 2.2 - box_h / 2, 5.75, 0.2 + box_h / 2)

    # Cross-link: Saha-Boltzmann <-> Batch Forward (shared physics)
    arrow(2.0 + box_w / 2, 2.2, 9.5 - box_w / 2, 5.8,
          style="-|>", color=GRAY)
    ax.text(5.75, 4.2, "Shared physics\nkernel", ha="center", va="center",
            fontsize=6.5, color=GRAY, fontstyle="italic")

    # Column labels
    ax.text(2.0, 7.7, "Inversion Pipeline", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
    ax.text(9.5, 7.7, "Forward / Manifold Pipeline", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor="#CCE5FF", edgecolor=CPU_COLOR,
                       linewidth=1.2, label="CPU only"),
        mpatches.Patch(facecolor="#FFE0CC", edgecolor=GPU_COLOR,
                       linewidth=1.2, label="GPU accelerated"),
        mpatches.Patch(facecolor="#E8E8E8", edgecolor="#666666",
                       linewidth=1.2, label="Input"),
        mpatches.Patch(facecolor="#D4EDDA", edgecolor="#28A745",
                       linewidth=1.2, label="Output"),
    ]
    ax.legend(handles=legend_patches, loc="upper center",
              bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=7,
              frameon=False)

    fig.tight_layout()
    _save(fig, "fig1_pipeline.pdf")


# ===================================================================
# Figure 2: Voigt Throughput
# ===================================================================
def fig2_voigt() -> None:
    """Log-log Voigt profile throughput: CPU vs GPU with speedup annotation."""
    df = pd.read_csv(DATA_DIR / "fig2_voigt_throughput.csv")

    fig, ax1 = plt.subplots(figsize=(DOUBLE_COL, 3.0))

    ax1.loglog(df["grid_size"], df["cpu_throughput"], "o-",
               color=CPU_COLOR, label="CPU (NumPy)", markersize=5)
    ax1.loglog(df["grid_size"], df["gpu_throughput"], "s-",
               color=GPU_COLOR, label="GPU (JAX)", markersize=5)

    ax1.set_xlabel("Grid size (points)")
    ax1.set_ylabel("Throughput (profiles/s)")

    # Speedup on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(df["grid_size"], df["speedup"], "^--",
             color=REF_COLOR, alpha=0.7, markersize=4, label="Speedup")
    ax2.set_ylabel("Speedup (GPU/CPU)", color=REF_COLOR)
    ax2.tick_params(axis="y", labelcolor=REF_COLOR)
    ax2.set_xscale("log")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(REF_COLOR)

    # Crossover annotation
    # Find where GPU first exceeds CPU
    cross_mask = df["gpu_throughput"] > df["cpu_throughput"]
    if cross_mask.any():
        idx = cross_mask.idxmax()
        if idx > 0:
            # Interpolate between idx-1 and idx
            x0, x1 = df["grid_size"].iloc[idx - 1], df["grid_size"].iloc[idx]
            ax1.axvline(x=np.sqrt(x0 * x1), color=GRAY, linestyle=":",
                        alpha=0.5, linewidth=0.8)
            ax1.text(np.sqrt(x0 * x1) * 1.3, df["cpu_throughput"].iloc[0] * 0.6,
                     "GPU > CPU", fontsize=7, color=GRAY, rotation=0)

    # Peak speedup annotation
    max_idx = df["speedup"].idxmax()
    ax1.annotate(
        f'{df["speedup"].iloc[max_idx]:.0f}x',
        xy=(df["grid_size"].iloc[max_idx], df["gpu_throughput"].iloc[max_idx]),
        xytext=(df["grid_size"].iloc[max_idx] / 5,
                df["gpu_throughput"].iloc[max_idx] * 2),
        fontsize=8, fontweight="bold", color=GPU_COLOR,
        arrowprops=dict(arrowstyle="->", color=GPU_COLOR, lw=0.8),
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

    fig.tight_layout()
    _save(fig, "fig2_voigt.pdf")


# ===================================================================
# Figure 3: Boltzmann Speedup
# ===================================================================
def fig3_boltzmann() -> None:
    """Boltzmann fitting time vs element count with speedup."""
    df = pd.read_csv(DATA_DIR / "fig3_boltzmann_speedup.csv")

    # Aggregate: mean time across lines_per_element for each element_count
    agg = df.groupby("element_count").agg(
        cpu_mean=("cpu_time_ms", "mean"),
        gpu_mean=("gpu_time_ms", "mean"),
        speedup_max=("speedup", "max"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL, 2.8))

    x = agg["element_count"]
    ax1.plot(x, agg["cpu_mean"], "o-", color=CPU_COLOR, label="CPU (NumPy)")
    ax1.plot(x, agg["gpu_mean"], "s-", color=GPU_COLOR, label="GPU (JAX)")

    ax1.set_xlabel("Number of elements")
    ax1.set_ylabel("Fitting time (ms)")

    # Speedup on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, agg["speedup_max"], "^--", color=REF_COLOR,
             alpha=0.7, markersize=4, label="Speedup")
    ax2.set_ylabel("Max speedup", color=REF_COLOR)
    ax2.tick_params(axis="y", labelcolor=REF_COLOR)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(REF_COLOR)

    # Crossover line
    cross_idx = (agg["speedup_max"] >= 1.0).idxmax()
    cross_el = agg["element_count"].iloc[cross_idx]
    ax1.axvline(x=cross_el, color=GRAY, linestyle=":", alpha=0.5, linewidth=0.8)
    ax1.text(cross_el + 0.3, agg["cpu_mean"].max() * 0.9,
             f"GPU > CPU\n({cross_el} elem.)", fontsize=6, color=GRAY)

    # Peak speedup annotation
    peak_speedup = agg["speedup_max"].max()
    ax1.annotate(
        f"{peak_speedup:.1f}x",
        xy=(agg["element_count"].iloc[-1], agg["gpu_mean"].iloc[-1]),
        xytext=(agg["element_count"].iloc[-1] - 4, agg["gpu_mean"].iloc[-1] + 0.2),
        fontsize=8, fontweight="bold", color=GPU_COLOR,
        arrowprops=dict(arrowstyle="->", color=GPU_COLOR, lw=0.8),
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

    fig.tight_layout()
    _save(fig, "fig3_boltzmann.pdf")


# ===================================================================
# Figure 4: Anderson Convergence (dual panel)
# ===================================================================
def fig4_anderson() -> None:
    """(a) Iteration counts vs M, (b) Residual trajectories Picard vs Anderson."""
    df_a = pd.read_csv(DATA_DIR / "fig4a_anderson_iteration_counts.csv")
    df_b = pd.read_csv(DATA_DIR / "fig4b_anderson_residual_trajectories.csv")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    # --- Panel (a): Iteration counts vs M ---
    conditions = [c for c in df_a.columns if c not in ("M", "mean_iters")]
    # Use a subset of conditions for clarity
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(conditions)))
    for i, cond in enumerate(conditions):
        ax_a.plot(df_a["M"], df_a[cond], "o-", color=cmap[i],
                  markersize=3, linewidth=1.0, alpha=0.5)
    # Mean line (bold)
    ax_a.plot(df_a["M"], df_a["mean_iters"], "k-", linewidth=2.0,
              label="Mean", zorder=10)

    ax_a.set_xlabel("Anderson depth $M$")
    ax_a.set_ylabel("Iterations to convergence")
    ax_a.set_xticks(range(0, 11, 2))
    ax_a.legend(fontsize=7)
    ax_a.text(-0.15, 1.05, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold")

    # --- Panel (b): Residual trajectories ---
    # Pick two representative conditions: worst case (low-T/low-ne) and typical (mid-T/mid-ne)
    pairs = [
        ("low-T/low-ne", CPU_COLOR, "Low-T/low-$n_e$"),
        ("mid-T/mid-ne(a)", REF_COLOR, "Mid-T/mid-$n_e$"),
        ("highest-T", ACCENT_COLOR, "Highest-T"),
    ]
    for cond_base, color, label in pairs:
        col_m0 = f"{cond_base}_M0"
        col_m3 = f"{cond_base}_M3"
        if col_m0 in df_b.columns:
            vals_m0 = df_b[col_m0].dropna()
            ax_b.semilogy(vals_m0.index, vals_m0.values, "o--",
                          color=color, alpha=0.5, markersize=3, linewidth=1.0,
                          label=f"{label} (Picard)")
        if col_m3 in df_b.columns:
            vals_m3 = df_b[col_m3].dropna()
            ax_b.semilogy(vals_m3.index, vals_m3.values, "s-",
                          color=color, markersize=3, linewidth=1.2,
                          label=f"{label} (M=3)")

    ax_b.set_xlabel("Iteration")
    ax_b.set_ylabel("Residual norm")
    ax_b.legend(fontsize=5.5, loc="upper right", ncol=1)
    ax_b.text(-0.15, 1.05, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold")

    # Convergence threshold line
    ax_b.axhline(y=1e-4, color=GRAY, linestyle=":", alpha=0.5, linewidth=0.8)
    ax_b.text(6.5, 1.5e-4, "tol = $10^{-4}$", fontsize=6, color=GRAY)

    fig.tight_layout()
    _save(fig, "fig4_anderson.pdf")


# ===================================================================
# Figure 5: FAISS Latency
# ===================================================================
def fig5_faiss() -> None:
    """FAISS query latency: CPU Flat vs IVF, with recall annotation."""
    df = pd.read_csv(DATA_DIR / "fig5_faiss_latency.csv")

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL, 2.8))

    x = np.arange(len(df))
    bar_w = 0.35

    ax1.bar(x - bar_w / 2, df["cpu_flat_ms"], bar_w,
            color=CPU_COLOR, label="Flat (exact)", alpha=0.85)
    ax1.bar(x + bar_w / 2, df["cpu_ivf_ms"], bar_w,
            color=REF_COLOR, label="IVF (approximate)", alpha=0.85)

    ax1.set_xlabel("Database size")
    ax1.set_ylabel("Query latency (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(s):,}" for s in df["db_size"]])
    ax1.set_yscale("log")

    # Recall annotation on IVF bars
    for i, (_, row) in enumerate(df.iterrows()):
        recall = row.get("recall_ivf", None)
        if pd.notna(recall):
            ax1.text(i + bar_w / 2, row["cpu_ivf_ms"] * 1.3,
                     f"R={recall:.0%}", ha="center", fontsize=6,
                     color=REF_COLOR)

    # Note about GPU data
    gpu_cols = ["gpu_flat_ms", "gpu_ivf_ms"]
    has_gpu = any(df[c].notna().any() for c in gpu_cols if c in df.columns)
    if not has_gpu:
        ax1.text(0.5, 0.02, "Note: GPU FAISS unavailable on V100S test system",
                 transform=ax1.transAxes, fontsize=6, color=GRAY,
                 ha="center", fontstyle="italic")

    ax1.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "fig5_faiss.pdf")


# ===================================================================
# Figure 6: Batch Forward Scaling
# ===================================================================
def fig6_batch() -> None:
    """Batch forward model throughput: log-log CPU vs GPU scaling."""
    df = pd.read_csv(DATA_DIR / "fig6_batch_scaling.csv")

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL, 2.8))

    # CPU: all rows have data
    ax1.loglog(df["batch_size"], df["cpu_spectra_per_sec"], "o-",
               color=CPU_COLOR, label="CPU (NumPy)", markersize=5)

    # GPU: some rows may be NaN (OOM)
    gpu_valid = df.dropna(subset=["gpu_spectra_per_sec"])
    gpu_oom = df[df["gpu_spectra_per_sec"].isna()]

    ax1.loglog(gpu_valid["batch_size"], gpu_valid["gpu_spectra_per_sec"], "s-",
               color=GPU_COLOR, label="GPU (JAX)", markersize=5)

    # Mark OOM points
    if len(gpu_oom) > 0:
        for _, row in gpu_oom.iterrows():
            ax1.plot(row["batch_size"], df["cpu_spectra_per_sec"].max() * 1.5,
                     "x", color="red", markersize=8, markeredgewidth=2)
        ax1.text(gpu_oom["batch_size"].iloc[0],
                 df["cpu_spectra_per_sec"].max() * 2.2,
                 "OOM", fontsize=7, color="red", ha="center")

    # Peak throughput annotation
    if len(gpu_valid) > 0:
        peak_idx = gpu_valid["gpu_spectra_per_sec"].idxmax()
        peak_val = gpu_valid.loc[peak_idx, "gpu_spectra_per_sec"]
        peak_batch = gpu_valid.loc[peak_idx, "batch_size"]
        ax1.annotate(
            f"{peak_val:,.0f}\nspectra/s",
            xy=(peak_batch, peak_val),
            xytext=(peak_batch * 4, peak_val * 0.4),
            fontsize=7, fontweight="bold", color=GPU_COLOR,
            arrowprops=dict(arrowstyle="->", color=GPU_COLOR, lw=0.8),
        )

    # Crossover
    merged = df.dropna(subset=["gpu_spectra_per_sec"])
    cross = merged[merged["gpu_spectra_per_sec"] > merged["cpu_spectra_per_sec"]]
    if len(cross) > 0:
        cx = cross["batch_size"].iloc[0]
        ax1.axvline(x=cx, color=GRAY, linestyle=":", alpha=0.5, linewidth=0.8)

    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Throughput (spectra/s)")
    ax1.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, "fig6_batch.pdf")


# ===================================================================
# Figure 7: E2E Pipeline Breakdown
# ===================================================================
def fig7_e2e() -> None:
    """End-to-end pipeline breakdown: stacked bars + speedup curve."""
    df = pd.read_csv(DATA_DIR / "fig7_e2e_breakdown.csv")

    # Only plot rows where GPU data exists
    gpu_valid = df.dropna(subset=["gpu_total_ms"])

    fig, ax1 = plt.subplots(figsize=(DOUBLE_COL, 3.2))

    n = len(gpu_valid)
    x = np.arange(n)
    bar_w = 0.38

    # CPU components
    cpu_comps = [
        ("cpu_saha_ms", "Saha-Boltzmann"),
        ("cpu_voigt_ms", "Voigt profiles"),
        ("cpu_assembly_ms", "Assembly"),
        ("cpu_boltzmann_ms", "Boltzmann fit"),
        ("cpu_closure_ms", "Closure"),
    ]
    cpu_colors = ["#4E79A7", "#59A14F", "#76B7B2", "#F28E2B", "#B07AA1"]

    cpu_bottom = np.zeros(n)
    for (col, label), color in zip(cpu_comps, cpu_colors):
        vals = gpu_valid[col].values
        ax1.bar(x - bar_w / 2, vals, bar_w, bottom=cpu_bottom,
                color=color, alpha=0.85, label=f"CPU: {label}", edgecolor="white",
                linewidth=0.3)
        cpu_bottom += vals

    # GPU components
    gpu_comps = [
        ("gpu_transfer_ms", "H2D transfer"),
        ("gpu_forward_ms", "Forward model"),
        ("gpu_boltzmann_ms", "Boltzmann fit"),
        ("gpu_closure_ms", "Closure"),
    ]
    gpu_colors = ["#E15759", "#FF9D9A", "#F28E2B", "#FFBE7D"]

    gpu_bottom = np.zeros(n)
    for (col, label), color in zip(gpu_comps, gpu_colors):
        vals = gpu_valid[col].values
        ax1.bar(x + bar_w / 2, vals, bar_w, bottom=gpu_bottom,
                color=color, alpha=0.85, label=f"GPU: {label}", edgecolor="white",
                linewidth=0.3)
        gpu_bottom += vals

    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Pipeline time (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(b)) for b in gpu_valid["batch_size"]])
    ax1.set_yscale("log")

    # Speedup curve on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, gpu_valid["speedup"].values, "kD-", markersize=5,
             linewidth=1.5, label="Speedup", zorder=10)
    ax2.set_ylabel("Speedup (CPU/GPU)")
    ax2.spines["right"].set_visible(True)

    # Speedup = 1 reference
    ax2.axhline(y=1, color=GRAY, linestyle=":", alpha=0.4, linewidth=0.8)

    # Annotate peak speedup
    peak_idx = gpu_valid["speedup"].idxmax()
    peak_val = gpu_valid.loc[peak_idx, "speedup"]
    peak_pos = gpu_valid.index.get_loc(peak_idx)
    ax2.annotate(
        f"{peak_val:.1f}x",
        xy=(peak_pos, peak_val),
        xytext=(peak_pos - 0.8, peak_val + 2),
        fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    # Legend: two columns (CPU left, GPU right)
    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + h2, labels + l2, loc="upper left",
               fontsize=5.5, ncol=2, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig7_e2e.pdf")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    print("Generating CF-LIBS JQSRT figures...")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Output dir: {OUT_DIR}")
    print()

    fig1_pipeline()
    fig2_voigt()
    fig3_boltzmann()
    fig4_anderson()
    fig5_faiss()
    fig6_batch()
    fig7_e2e()

    print("\nAll 7 figures generated successfully.")


if __name__ == "__main__":
    main()
