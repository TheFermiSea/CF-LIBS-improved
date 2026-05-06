"""Generate manuscript figures for the low-resolution LIBS identification paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.03, 0.20, 0.18, 0.60, "Low-resolution\nLIBS spectra", OKABE_ITO["sky"]),
        (0.29, 0.20, 0.18, 0.60, "Full-spectrum\nNNLS screen", OKABE_ITO["orange"]),
        (0.55, 0.20, 0.18, 0.60, "ALIAS line\nconfirmation", OKABE_ITO["green"]),
        (0.79, 0.20, 0.18, 0.60, "Hybrid result\nF1 = 0.654", OKABE_ITO["blue"]),
    ]
    for x, y, w, h, label, color in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.015",
            linewidth=1.1,
            edgecolor="#333333",
            facecolor=color,
            alpha=0.18,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.10, label, ha="center", va="top", weight="bold")

    x = np.linspace(0.055, 0.185, 220)
    y = 0.38 + 0.18 * np.exp(-((x - 0.085) / 0.010) ** 2)
    y += 0.12 * np.exp(-((x - 0.130) / 0.015) ** 2)
    y += 0.09 * np.exp(-((x - 0.165) / 0.008) ** 2)
    ax.plot(x, y, color=OKABE_ITO["blue"], lw=1.7)

    ax.bar([0.33, 0.37, 0.41], [0.30, 0.46, 0.23], width=0.025, color=OKABE_ITO["orange"])
    ax.plot([0.60, 0.64, 0.68], [0.62, 0.40, 0.55], "o", color=OKABE_ITO["green"], ms=5)
    ax.hlines([0.62, 0.40, 0.55], [0.58, 0.62, 0.66], [0.62, 0.66, 0.70], colors=OKABE_ITO["green"], lw=1)
    ax.text(0.88, 0.52, "P = 0.604\nR = 0.713\nFPR = 0.053", ha="center", va="center")

    for start, end in [(0.22, 0.29), (0.48, 0.55), (0.74, 0.79)]:
        ax.annotate(
            "",
            xy=(end, 0.50),
            xytext=(start, 0.50),
            arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2),
        )

    ax.text(
        0.5,
        0.05,
        "Global spectral evidence gates false positives; line-level confirmation preserves physically interpretable detections.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    save(fig, "graphical_abstract")


def workflow_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    ax.bar(["Pure\nelements", "Minerals"], [13, 61], color=[OKABE_ITO["sky"], OKABE_ITO["green"]])
    ax.set_ylabel("Spectra")
    ax.set_title("A. Benchmark composition")
    ax.text(0.5, 54, "74 spectra\nRP = 300-1100\n200-900 nm", ha="center", va="center")

    ax = axes[1]
    ax.set_axis_off()
    ax.set_title("B. Identification pathways", loc="left")
    labels = ["ALIAS", "NNLS", "Hybrid", "Voigt+ALIAS", "Forward threshold"]
    ys = np.linspace(0.85, 0.15, len(labels))
    for y, label, color in zip(
        ys,
        labels,
        [OKABE_ITO["sky"], OKABE_ITO["orange"], OKABE_ITO["blue"], OKABE_ITO["purple"], OKABE_ITO["green"]],
    ):
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.08, y - 0.06),
                0.34,
                0.10,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                facecolor=color,
                alpha=0.22,
                edgecolor="#444444",
            )
        )
        ax.text(0.25, y - 0.01, label, ha="center", va="center", weight="bold")
        ax.annotate("", xy=(0.62, y - 0.01), xytext=(0.43, y - 0.01), arrowprops=dict(arrowstyle="-|>", lw=1))
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.63, 0.27),
            0.28,
            0.45,
            boxstyle="round,pad=0.02,rounding_size=0.015",
            facecolor="#f2f2f2",
            edgecolor="#444444",
        )
    )
    ax.text(0.77, 0.50, "22-element\nmulti-label\nscoring", ha="center", va="center", weight="bold")
    save(fig, "figure1_workflow")


def performance_bars() -> None:
    pathways = ["Hybrid", "ALIAS", "Voigt+\nALIAS", "Forward", "NNLS"]
    precision = np.array([0.604, 0.505, 0.488, 0.369, 0.293])
    recall = np.array([0.713, 0.629, 0.623, 0.796, 0.940])
    f1 = np.array([0.654, 0.560, 0.547, 0.505, 0.447])
    fpr = np.array([0.053, 0.070, 0.075, 0.148, 0.236])

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(pathways))
    width = 0.18
    for offset, values, label, color in [
        (-1.5, precision, "Precision", OKABE_ITO["blue"]),
        (-0.5, recall, "Recall", OKABE_ITO["orange"]),
        (0.5, f1, "F1", OKABE_ITO["green"]),
        (1.5, fpr, "FPR", OKABE_ITO["vermillion"]),
    ]:
        ax.bar(x + offset * width, values, width, label=label, color=color)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(pathways)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_title("Best configuration per identification pathway")
    save(fig, "figure2_pathway_performance")


def precision_recall() -> None:
    configs = [
        (0.557, 0.737, 0.634, "intersect"),
        (0.274, 0.952, 0.425, "union"),
        (0.594, 0.719, 0.650, "intersect"),
        (0.283, 0.952, 0.437, "union"),
        (0.649, 0.521, 0.578, "intersect"),
        (0.292, 0.952, 0.447, "union"),
        (0.565, 0.731, 0.637, "intersect"),
        (0.296, 0.952, 0.451, "union"),
        (0.604, 0.713, 0.654, "intersect"),
        (0.308, 0.952, 0.466, "union"),
        (0.659, 0.521, 0.582, "intersect"),
        (0.319, 0.952, 0.477, "union"),
        (0.568, 0.725, 0.637, "intersect"),
        (0.310, 0.940, 0.467, "union"),
        (0.602, 0.707, 0.650, "intersect"),
        (0.326, 0.940, 0.484, "union"),
        (0.656, 0.515, 0.577, "intersect"),
        (0.339, 0.940, 0.498, "union"),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for mode, marker, color in [("intersect", "o", OKABE_ITO["blue"]), ("union", "^", OKABE_ITO["orange"])]:
        pts = np.array([(p, r, f) for p, r, f, m in configs if m == mode])
        ax.scatter(pts[:, 0], pts[:, 1], s=55, marker=marker, color=color, label=mode.capitalize(), alpha=0.85)
    ax.scatter([0.604], [0.713], s=130, marker="*", color=OKABE_ITO["green"], edgecolor="black", zorder=5)
    p = np.linspace(0.20, 0.75, 300)
    for f in [0.4, 0.5, 0.6, 0.7]:
        denom = 2 * p - f
        mask = denom > 0.01
        r = np.full_like(p, np.nan)
        r[mask] = (f * p[mask]) / denom[mask]
        mask &= (r > 0) & (r <= 1)
        ax.plot(p[mask], r[mask], ls="--", lw=0.8, color="#888888")
        if mask.any():
            ax.text(p[mask][-1] + 0.005, r[mask][-1], f"F1={f:.1f}", color="#666666", fontsize=7)
    ax.set_xlim(0.20, 0.75)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.legend(frameon=False)
    ax.set_title("Hybrid precision-recall operating points")
    save(fig, "figure3_precision_recall")


def per_element_heatmap() -> None:
    elements = ["Si", "Al", "Fe", "Li", "Co", "Ni", "K", "Ca", "Ti", "Mg", "Pb", "Na", "Mn", "Zn", "Zr"]
    data = np.array(
        [
            [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.67, 0.62, 0.50, 0.39, 0.40, 0.14, 0.03, 0.00, 0.00],
            [0.95, 0.90, 0.44, 1.00, 1.00, 0.67, 0.60, 0.40, 1.00, 0.81, 1.00, 0.80, 1.00, 0.00, 0.00],
            [0.96, 0.76, 0.91, 1.00, 0.33, 0.50, 0.54, 0.46, 0.67, 0.50, 0.33, 0.17, 0.03, 0.00, 0.00],
            [0.60, 0.76, 0.40, 1.00, 1.00, 1.00, 0.70, 0.30, 1.00, 0.94, 1.00, 0.80, 1.00, 0.00, 0.00],
        ]
    )
    rows = ["Hybrid P", "Hybrid R", "ALIAS P", "ALIAS R"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(elements)))
    ax.set_xticklabels(elements)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white" if data[i, j] < 0.45 else "black", fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Score")
    ax.set_title("Per-element precision and recall")
    save(fig, "figure4_per_element")


def stress_figure() -> None:
    shifts = np.array([-1.0, 0.0, 1.0])
    alias = np.array([0.2727, 0.5000, 0.5294])
    comb = np.array([0.1600, 0.4667, 0.2069])
    corr = np.array([0.3243, 0.3333, 0.2286])
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(shifts, alias, marker="o", label="ALIAS", color=OKABE_ITO["blue"])
    ax.plot(shifts, comb, marker="s", label="Comb", color=OKABE_ITO["orange"])
    ax.plot(shifts, corr, marker="^", label="Correlation", color=OKABE_ITO["green"])
    ax.set_xlabel("Applied wavelength shift (nm)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.62)
    ax.legend(frameon=False)
    ax.set_title("Synthetic wavelength-shift stress test")
    save(fig, "figure5_shift_stress")


def main() -> None:
    graphical_abstract()
    workflow_figure()
    performance_bars()
    precision_recall()
    per_element_heatmap()
    stress_figure()


if __name__ == "__main__":
    main()
