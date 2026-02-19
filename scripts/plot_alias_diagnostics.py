#!/usr/bin/env python
"""
Diagnostic plots for ALIAS element identification showing gate effects.

Generates per-dataset plots showing:
  - Pre-gate CL vs post-gate CL (confidence) for each element
  - Discriminator values (P_local, P_mix, R_rat) breakdown
  - Detection threshold and TP/FP labeling
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.alias_identifier import ALIASIdentifier

# Reuse data loading from validate_real_data
from scripts.validate_real_data import (
    load_scipp,
    select_representative_spectrum,
    DATASET_CONFIGS,
)


def run_alias_with_diagnostics(wavelength, intensity, db, elements, rp):
    """Run ALIAS and extract per-element metadata including gate values."""
    identifier = ALIASIdentifier(db, elements=elements, resolving_power=rp)
    result = identifier.identify(wavelength, intensity)

    diag = {}
    for e in result.all_elements:
        m = e.metadata
        diag[e.element] = {
            "score": e.score,
            "confidence": e.confidence,
            "detected": e.detected,
            "n_matched": e.n_matched_lines,
            "n_total": e.n_total_lines,
            "P_local": m.get("P_local", 1.0),
            "P_mix": m.get("P_mix", 1.0),
            "R_rat": m.get("R_rat", 0.5),
            "P_sig": m.get("P_sig", 1.0),
            "k_det": m.get("k_det", 0.0),
        }
    return result, diag, float(identifier.detection_threshold)


def plot_gate_breakdown(diag, dataset_name, expected, threshold, output_path):
    """
    Two-panel diagnostic plot:
      Top: Pre-gate CL (score) vs post-gate CL (confidence) with threshold
      Bottom: Discriminator values (P_local, P_mix, R_rat) per element
    """
    elements = sorted(diag.keys())
    n = len(elements)
    x = np.arange(n)

    scores = [diag[e]["score"] for e in elements]
    confs = [diag[e]["confidence"] for e in elements]
    detected = [diag[e]["detected"] for e in elements]
    is_tp = [e in expected for e in elements]

    p_local = [diag[e]["P_local"] for e in elements]
    p_mix = [diag[e]["P_mix"] for e in elements]
    r_rat = [diag[e]["R_rat"] for e in elements]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(10, n * 0.9), 9), gridspec_kw={"height_ratios": [1.2, 1]}
    )
    fig.suptitle(
        f"ALIAS Gate Diagnostics: {dataset_name}\nExpected: {', '.join(sorted(expected))}",
        fontsize=14,
        fontweight="bold",
    )

    # ── Top panel: pre-gate vs post-gate CL ──
    bar_w = 0.35
    colors_pre = []
    colors_post = []
    for i, e in enumerate(elements):
        if is_tp[i] and detected[i]:
            colors_pre.append("#2196F3")  # blue — TP detected
            colors_post.append("#4CAF50")  # green
        elif is_tp[i] and not detected[i]:
            colors_pre.append("#FF9800")  # orange — TP missed
            colors_post.append("#FF9800")
        elif not is_tp[i] and detected[i]:
            colors_pre.append("#f44336")  # red — FP
            colors_post.append("#f44336")
        else:
            colors_pre.append("#9E9E9E")  # gray — correctly rejected
            colors_post.append("#9E9E9E")

    ax1.bar(
        x - bar_w / 2,
        scores,
        bar_w,
        color=colors_pre,
        alpha=0.5,
        edgecolor="black",
        linewidth=0.5,
        label="Pre-gate (score)",
    )
    ax1.bar(
        x + bar_w / 2,
        confs,
        bar_w,
        color=colors_post,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
        label="Post-gate (confidence)",
    )

    ax1.axhline(
        y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})"
    )
    ax1.set_ylabel("CL Value")
    ax1.set_xticks(x)
    ax1.set_xticklabels(elements, fontsize=11, fontweight="bold")

    # Annotate bars
    for i, e in enumerate(elements):
        if detected[i]:
            marker = "TP" if is_tp[i] else "FP"
            color = "#4CAF50" if is_tp[i] else "#f44336"
            ax1.text(
                x[i] + bar_w / 2,
                confs[i] + 0.01,
                marker,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=color,
            )
        elif is_tp[i]:
            ax1.text(
                x[i] + bar_w / 2,
                max(confs[i], 0.005) + 0.01,
                "MISS",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#FF9800",
            )

    legend_elements = [
        Patch(facecolor="#2196F3", alpha=0.5, edgecolor="black", label="TP detected (pre)"),
        Patch(facecolor="#4CAF50", alpha=0.9, edgecolor="black", label="TP detected (post)"),
        Patch(facecolor="#FF9800", alpha=0.7, edgecolor="black", label="TP missed"),
        Patch(facecolor="#f44336", alpha=0.7, edgecolor="black", label="FP detected"),
        Patch(facecolor="#9E9E9E", alpha=0.7, edgecolor="black", label="Correctly rejected"),
        plt.Line2D([0], [0], color="red", linestyle="--", label=f"Threshold ({threshold})"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)
    ax1.set_title("Pre-gate Score vs Post-gate Confidence", fontsize=11)

    # ── Bottom panel: discriminator values ──
    bar_w2 = 0.25
    ax2.bar(
        x - bar_w2,
        p_local,
        bar_w2,
        color="#7B1FA2",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label="P_local",
    )
    ax2.bar(
        x,
        p_mix,
        bar_w2,
        color="#00897B",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label="P_mix",
    )
    ax2.bar(
        x + bar_w2,
        r_rat,
        bar_w2,
        color="#F57C00",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label="R_rat",
    )

    ax2.set_ylabel("Discriminator Value")
    ax2.set_xlabel("Element")
    ax2.set_xticks(x)
    ax2.set_xticklabels(elements, fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title("NNLS Discriminator Values", fontsize=11)

    # Highlight expected elements
    for i, e in enumerate(elements):
        if e in expected:
            ax2.axvspan(i - 0.45, i + 0.45, alpha=0.08, color="green")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_suppression_waterfall(diag, dataset_name, expected, threshold, output_path):
    """
    Waterfall plot showing how each gate reduces CL for every element.
    """
    elements = sorted(diag.keys())
    n = len(elements)

    fig, ax = plt.subplots(figsize=(max(12, n * 1.2), 6))
    fig.suptitle(f"Gate Suppression Waterfall: {dataset_name}", fontsize=13, fontweight="bold")

    x = np.arange(n)
    bar_w = 0.7

    for i, e in enumerate(elements):
        d = diag[e]
        cl_pre = d["score"]
        p_local_gate = float(np.clip(2.0 * d["P_local"], 0.1, 1.0))
        p_mix_gate = 0.1 + 0.9 * min(d["P_mix"], 1.0)
        r_rat_gate = 0.5 + 0.5 * d["R_rat"]

        # Compute cumulative CL after each gate
        cl0 = cl_pre
        cl1 = cl0 * p_local_gate
        cl2 = cl1 * p_mix_gate
        cl3 = cl2 * r_rat_gate  # = confidence

        # Draw waterfall bars
        is_expected = e in expected
        edge_color = "#2E7D32" if is_expected else "#424242"
        edge_w = 2.0 if is_expected else 0.8

        # Pre-gate (full height, light)
        ax.bar(i, cl0, bar_w, color="#BBDEFB", edgecolor=edge_color, linewidth=edge_w, alpha=0.4)
        # After P_local
        ax.bar(i, cl1, bar_w * 0.8, color="#7B1FA2", alpha=0.35)
        # After P_mix
        ax.bar(i, cl2, bar_w * 0.6, color="#00897B", alpha=0.45)
        # After R_rat (final)
        color_final = (
            "#4CAF50"
            if (d["detected"] and is_expected)
            else ("#f44336" if d["detected"] else ("#FF9800" if is_expected else "#9E9E9E"))
        )
        ax.bar(i, cl3, bar_w * 0.4, color=color_final, edgecolor="black", linewidth=0.8, alpha=0.9)

        # Label final confidence
        if cl3 > 0.001:
            ax.text(
                i, cl3 + 0.01, f"{cl3:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold"
            )

    ax.axhline(
        y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(elements, fontsize=11, fontweight="bold")
    ax.set_ylabel("CL Value")
    ax.set_xlabel("Element")

    legend_elements = [
        Patch(facecolor="#BBDEFB", alpha=0.4, edgecolor="black", label="Pre-gate CL"),
        Patch(facecolor="#7B1FA2", alpha=0.35, label="After P_local"),
        Patch(facecolor="#00897B", alpha=0.45, label="After P_mix"),
        Patch(facecolor="#4CAF50", alpha=0.9, edgecolor="black", label="TP detected"),
        Patch(facecolor="#FF9800", alpha=0.9, edgecolor="black", label="TP missed"),
        Patch(facecolor="#9E9E9E", alpha=0.9, edgecolor="black", label="Rejected"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Threshold"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=str, default="ASD_da/libs_production.db")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="output/validation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Atomic DB not found: {db_path}")

    datasets_to_run = [
        "FeNi_380nm",
        "FeNi_480nm",
        "FeNi_single",
        "AA1100_substrate",
        "Ti6Al4V_substrate",
    ]

    with AtomicDatabase(str(db_path)) as db:
        for ds_name in datasets_to_run:
            cfg = next((d for d in DATASET_CONFIGS if d["name"] == ds_name), None)
            if cfg is None:
                print(f"  Skipping {ds_name}: no config found")
                continue

            print(f"\n{'='*60}")
            print(f"  {ds_name}")
            print(f"{'='*60}")

            # Load data
            ds_path = data_dir / cfg["filename"]
            if not ds_path.exists():
                print(f"  Skipping: {ds_path} not found")
                continue

            wavelength, data, _meta = load_scipp(str(ds_path))
            spectrum = select_representative_spectrum(data, ds_name)

            rp = cfg.get("resolving_power", 500)
            elements = cfg["elements"]
            expected = set(cfg["expected"])

            result, diag, threshold = run_alias_with_diagnostics(
                wavelength, spectrum, db, elements, rp
            )

            # Print summary table
            print(
                f"  {'Elem':<4} {'Score':>7} {'Conf':>7} {'Det':>4} "
                f"{'P_local':>8} {'P_mix':>7} {'R_rat':>6} {'n':>6}"
            )
            print(f"  {'-'*52}")
            for e in sorted(diag.keys()):
                d = diag[e]
                tp_mark = "*" if e in expected else " "
                det_mark = "YES" if d["detected"] else "no"
                print(
                    f"  {e:<3}{tp_mark} {d['score']:7.3f} {d['confidence']:7.3f} "
                    f"{det_mark:>4} {d['P_local']:8.3f} {d['P_mix']:7.3f} "
                    f"{d['R_rat']:6.3f} {d['n_matched']:>2}/{d['n_total']}"
                )

            # Generate plots
            plot_gate_breakdown(
                diag,
                ds_name,
                expected,
                threshold,
                output_dir / f"{ds_name}_gate_breakdown.png",
            )
            plot_suppression_waterfall(
                diag,
                ds_name,
                expected,
                threshold,
                output_dir / f"{ds_name}_waterfall.png",
            )


if __name__ == "__main__":
    main()
