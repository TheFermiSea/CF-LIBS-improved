#!/usr/bin/env python3
"""
Vrábel 2020 LIBS soil benchmark runner.

Two modes:
  --mode classification: predict 1..12 class label per test spectrum
                         using a nearest-class-mean baseline (or any
                         workflow registered in cflibs.benchmark).
                         Reports accuracy + macro-F1, with per-class
                         breakdown. Published baseline: 96.0% accuracy
                         (EMSLIBS contest, Vrábel 2020 Spectrochim. Acta).

  --mode composition:    for each test spectrum, predict the per-element
                         weight-percent vector (Al, Ca, Cr, Cu, Fe, K,
                         Mg, Na, Pb, Si, Ti) using cflibs' inversion
                         pipeline, then compare to the per-sample assay
                         in support_tables.xlsx. Reports MAE, bias, and
                         Aitchison distance. The published dataset does
                         NOT bind to a per-sample mapping in test split
                         (test labels are class IDs only) — so this mode
                         requires aggregating per-class predictions and
                         comparing to per-class mean composition.

Usage:
  scripts/run_vrabel2020_benchmark.py --mode classification \\
    --data-dir data/vrabel2020_soil_benchmark \\
    --out-dir benchmark_artifacts/vrabel2020-$(date +%Y%m%d-%H%M)

The classification baseline is intentionally simple — it provides a
floor for the benchmark gate and a regression detector. Subbing in
cflibs.benchmark.classification workflows is a one-line change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cflibs.core.logging_config import get_logger  # noqa: E402
from cflibs.pds.vrabel2020 import (  # noqa: E402
    ELEMENTS,
    VrabelSampleComposition,
    load_compositions,
    load_test_iter,
    load_test_labels,
    load_train,
)

logger = get_logger("vrabel2020_benchmark")
MAJOR_ELEMENTS = {"Si", "Al", "Fe", "Ca", "Mg", "K", "Na"}


# ─── Classification: nearest-class-mean baseline ─────────────────────────
def fit_class_centroids(
    train_spectra: np.ndarray,
    train_class_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-class mean spectra (centroids) and the class IDs.

    Returns
    -------
    centroids : np.ndarray of shape (n_classes_seen, n_channels)
    class_order : np.ndarray of shape (n_classes_seen,) — the class IDs
                  in the same order as the centroid rows.
    """
    classes = np.unique(train_class_ids)
    centroids = np.empty((classes.size, train_spectra.shape[1]), dtype=np.float64)
    for i, c in enumerate(classes):
        mask = train_class_ids == c
        centroids[i] = train_spectra[mask].mean(axis=0)
    return centroids, classes


def predict_nearest_centroid(
    spectra: np.ndarray,
    centroids: np.ndarray,
    class_order: np.ndarray,
) -> np.ndarray:
    """Cosine-similarity nearest-centroid classifier.

    Cosine handles overall-intensity drift (different shot energies)
    better than Euclidean for LIBS. Pre-normalize centroids to L2=1 so
    we only need one normalize pass on the test side.
    """
    cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sn = spectra / (np.linalg.norm(spectra, axis=1, keepdims=True) + 1e-12)
    sims = sn @ cn.T  # (n_spectra, n_classes)
    pred_idx = sims.argmax(axis=1)
    return class_order[pred_idx]


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, dict[int, dict[str, float]]]:
    """Macro-averaged F1 + per-class precision/recall/F1."""
    per_class: dict[int, dict[str, float]] = {}
    f1s = []
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[int(c)] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp + fn,
        }
        f1s.append(f1)
    return float(np.mean(f1s)), per_class


def run_classification(args: argparse.Namespace) -> dict:
    train_h5 = Path(args.data_dir) / "train.h5"
    test_h5 = Path(args.data_dir) / "test.h5"
    test_labels_csv = Path(args.data_dir) / "test_labels.csv"
    for p in (train_h5, test_h5, test_labels_csv):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    t0 = time.time()
    logger.info("loading train (shots_per_sample=%d)", args.shots_per_sample)
    train = load_train(train_h5, shots_per_sample=args.shots_per_sample)

    logger.info("fitting class centroids on %d spectra", train.spectra.shape[0])
    centroids, class_order = fit_class_centroids(
        train.spectra,
        train.class_ids,
    )
    fit_s = time.time() - t0

    logger.info("loading test labels")
    y_true = load_test_labels(test_labels_csv)

    logger.info("predicting in chunks (chunk_size=%d)", args.chunk_size)
    preds = np.empty(y_true.size, dtype=np.int32)
    t1 = time.time()
    for offset, chunk in load_test_iter(test_h5, chunk_size=args.chunk_size):
        end = offset + chunk.shape[0]
        if end > preds.size:
            raise ValueError(f"test stream produced {end} spectra, labels file has {preds.size}")
        preds[offset:end] = predict_nearest_centroid(
            chunk,
            centroids,
            class_order,
        )
    pred_s = time.time() - t1

    accuracy = float((preds == y_true).mean())
    f1, per_class = macro_f1(y_true, preds, class_order)
    confusion = {int(t): dict(Counter(int(p) for p in preds[y_true == t])) for t in class_order}

    # Regression floor scales with training data size since the
    # nearest-centroid baseline is sample-hungry. With shots_per_sample
    # = 500 (full) we expect ~85-90 % accuracy on the published dataset;
    # at shots = 5 the same baseline only gets ~50 %. Linear interp
    # between (5, 0.30) and (500, 0.80) gives a sane floor for any
    # invocation. Override with --floor for tighter gating.
    if args.floor is not None:
        floor = args.floor
    else:
        s = args.shots_per_sample
        floor = 0.30 + (0.50 * (max(s, 5) - 5) / (500 - 5))

    summary = {
        "mode": "classification",
        "n_train": int(train.spectra.shape[0]),
        "n_test": int(y_true.size),
        "n_classes_seen": int(class_order.size),
        "shots_per_sample": args.shots_per_sample,
        "fit_seconds": round(fit_s, 1),
        "predict_seconds": round(pred_s, 1),
        "accuracy": accuracy,
        "macro_f1": f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
        # Floor: nearest-centroid is intentionally a weak baseline — its
        # job is to detect REGRESSIONS in the data path, not match SOTA
        # (~96 % published in the EMSLIBS contest). Real cflibs
        # identification workflows should comfortably exceed this.
        "regression_floor_accuracy": float(floor),
        "passed_regression_floor": bool(accuracy >= floor),
    }
    return summary


# ─── Composition mode: per-class assay agreement ─────────────────────────
def per_class_mean_composition(
    samples: dict[int, VrabelSampleComposition],
) -> dict[int, dict[str, float]]:
    """Aggregate per-sample compositions to per-class means.

    The Vrábel test split exposes class IDs only (no sample IDs), so
    when comparing predicted compositions on test spectra we have to
    aggregate at the class level: predict per-spectrum, group by true
    class, average, compare to the per-class mean of the assay table.
    """
    by_class: dict[int, list[VrabelSampleComposition]] = defaultdict(list)
    for s in samples.values():
        by_class[s.class_id].append(s)
    means: dict[int, dict[str, float]] = {}
    for cls, members in by_class.items():
        means[cls] = {
            elt: float(np.mean([m.composition[elt] for m in members])) for elt in ELEMENTS
        }
    return means


def aitchison_distance(
    a: dict[str, float],
    b: dict[str, float],
    *,
    eps: float = 1e-3,
) -> float:
    """Compositional Aitchison distance between two element vectors.

    Both vectors are normalized to a closed simplex (sum to 1), then
    centered log-ratio transformed; Aitchison distance = Euclidean
    distance in CLR space. ``eps`` replaces zeros so the log is finite.
    """
    elts = list(a.keys())
    av = np.array([a[e] for e in elts], dtype=np.float64) + eps
    bv = np.array([b[e] for e in elts], dtype=np.float64) + eps
    av /= av.sum()
    bv /= bv.sum()
    clr_a = np.log(av) - np.log(av).mean()
    clr_b = np.log(bv) - np.log(bv).mean()
    return float(np.linalg.norm(clr_a - clr_b))


def aggregate_predicted_composition(
    predicted_classes: np.ndarray,
    class_means: dict[int, dict[str, float]],
) -> dict[str, float] | None:
    """Average assay-derived compositions for predicted classes."""
    if predicted_classes.size == 0:
        return None

    pred_class_counts = Counter(int(p) for p in predicted_classes)
    total = sum(pred_class_counts.values())
    if total <= 0:
        return None

    aggregate = {elt: 0.0 for elt in ELEMENTS}
    for predicted_class, count in pred_class_counts.items():
        if predicted_class not in class_means:
            return None
        weight = count / total
        for elt in ELEMENTS:
            aggregate[elt] += class_means[predicted_class][elt] * weight
    return aggregate


def per_class_aitchison_distances(
    class_order: np.ndarray,
    y_true: np.ndarray,
    preds: np.ndarray,
    class_means: dict[int, dict[str, float]],
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute full and majors-only Aitchison distances for each class.

    Missing class means or absent predictions produce ``inf`` so the
    regression gate fails instead of silently dropping a class.
    """
    full_distances: dict[int, float] = {}
    major_distances: dict[int, float] = {}

    for cls in class_order:
        cls_int = int(cls)
        true_comp = class_means.get(cls_int)
        predicted_classes = preds[y_true == cls_int]
        aggregate = aggregate_predicted_composition(predicted_classes, class_means)
        if true_comp is None or aggregate is None:
            full_distances[cls_int] = float("inf")
            major_distances[cls_int] = float("inf")
            continue

        full_distances[cls_int] = aitchison_distance(true_comp, aggregate)
        major_distances[cls_int] = aitchison_distance(
            {elt: true_comp[elt] for elt in MAJOR_ELEMENTS},
            {elt: aggregate[elt] for elt in MAJOR_ELEMENTS},
        )

    return full_distances, major_distances


def run_composition(args: argparse.Namespace) -> dict:
    """Composition-mode benchmark.

    Real cflibs inversion is expensive (NumPyro / Bayesian) — running it
    on 20K test spectra takes hours. As a first-cut floor that exercises
    the data path end-to-end, we use a rough surrogate: predict per-class
    mean composition by training on the LABELED training spectra' assays
    (each train spectrum's true sample_id → composition row in xlsx),
    aggregating to class centroids, then assigning each test spectrum
    its nearest-centroid class's mean composition. Quality of THIS
    approximation tells us: does the data plumbing work? Does the
    Aitchison metric land in a sane range?

    Replacement with cflibs.benchmark.composition.* workflows is a
    one-call swap (predict_per_spectrum_compositions).
    """
    data_dir = Path(args.data_dir)
    samples = load_compositions(data_dir / "support_tables.xlsx")
    class_means = per_class_mean_composition(samples)

    t0 = time.time()
    train = load_train(data_dir / "train.h5", shots_per_sample=args.shots_per_sample)
    centroids, class_order = fit_class_centroids(
        train.spectra,
        train.class_ids,
    )
    y_true = load_test_labels(data_dir / "test_labels.csv")
    preds = np.empty(y_true.size, dtype=np.int32)
    for offset, chunk in load_test_iter(data_dir / "test.h5", chunk_size=args.chunk_size):
        preds[offset : offset + chunk.shape[0]] = predict_nearest_centroid(
            chunk,
            centroids,
            class_order,
        )

    # Per-class mean composition agreement (using TRUE labels):
    # ideal = the assay-derived class mean. Predicted = the assay-derived
    # class mean of the predicted class. So error = 0 when prediction
    # is correct AND class means differ; non-zero only on misclassified
    # spectra. This is a CLASSIFIER-via-COMPOSITION view.
    per_element_mae: dict[str, float] = {}
    per_element_bias: dict[str, float] = {}
    aitchison_per_class: dict[int, float] = {}

    for elt in ELEMENTS:
        true_vals = np.array(
            [class_means.get(int(c), {}).get(elt, np.nan) for c in y_true],
            dtype=np.float64,
        )
        pred_vals = np.array(
            [class_means.get(int(c), {}).get(elt, np.nan) for c in preds],
            dtype=np.float64,
        )
        finite = np.isfinite(true_vals) & np.isfinite(pred_vals)
        if not finite.any():
            per_element_mae[elt] = float("nan")
            per_element_bias[elt] = float("nan")
            continue
        per_element_mae[elt] = float(np.abs(pred_vals[finite] - true_vals[finite]).mean())
        per_element_bias[elt] = float((pred_vals[finite] - true_vals[finite]).mean())

    aitchison_per_class, aitchison_majors_per_class = per_class_aitchison_distances(
        class_order,
        y_true,
        preds,
        class_means,
    )

    n_good_majors = sum(1 for d in aitchison_majors_per_class.values() if d < 0.10)

    duration = time.time() - t0
    accuracy = float((preds == y_true).mean())
    mean_aitchison = (
        float(np.mean(list(aitchison_per_class.values()))) if aitchison_per_class else float("nan")
    )

    # Like the classification floor, the composition Aitchison floor
    # scales with shots_per_sample. At shots=500 (full training) the
    # surrogate's class assignments are accurate enough that mean
    # Aitchison <= 1.5 is a reasonable regression floor; at shots=5
    # it's nearly random and floor at 4.0 is more honest.
    if args.floor is not None:
        ait_floor = args.floor
    else:
        s = args.shots_per_sample
        ait_floor = 4.0 - (2.5 * (max(s, 5) - 5) / (500 - 5))

    return {
        "mode": "composition",
        "n_train": int(train.spectra.shape[0]),
        "n_test": int(y_true.size),
        "n_classes": int(class_order.size),
        "duration_seconds": round(duration, 1),
        "classifier_accuracy": accuracy,
        "elements": list(ELEMENTS),
        "per_element_mae_wt_pct": per_element_mae,
        "per_element_bias_wt_pct": per_element_bias,
        "aitchison_distance_per_class": aitchison_per_class,
        "aitchison_majors_per_class": aitchison_majors_per_class,
        "n_classes_good_majors": n_good_majors,
        "mean_aitchison_distance": mean_aitchison,
        "regression_floor_aitchison_max": float(ait_floor),
        "passed_regression_floor": bool(
            np.isfinite(mean_aitchison) and mean_aitchison <= ait_floor and n_good_majors >= 8
        ),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Vrábel 2020 LIBS soil benchmark")
    ap.add_argument("--mode", choices=("classification", "composition"), required=True)
    ap.add_argument(
        "--data-dir",
        default="data/vrabel2020_soil_benchmark",
        help="Directory containing train.h5, test.h5, test_labels.csv, support_tables.xlsx",
    )
    ap.add_argument(
        "--out-dir", help="Output directory; default = benchmark_artifacts/vrabel2020-<timestamp>"
    )
    ap.add_argument(
        "--shots-per-sample",
        type=int,
        default=100,
        help="Train spectra per sample (1..500). Lower = "
        "faster, less RAM. Default 100 ≈ 10K train "
        "spectra ≈ 3.2 GB. Use 500 for full quality.",
    )
    ap.add_argument("--chunk-size", type=int, default=2000, help="Test prediction chunk size")
    ap.add_argument(
        "--floor",
        type=float,
        default=None,
        help="Override the regression floor (e.g. 0.85 for "
        "tight gating on full-shots runs). Default: "
        "scaled by shots_per_sample for classification, "
        "fixed 2.0 Aitchison for composition.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"benchmark_artifacts/vrabel2020-{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("starting %s benchmark, out_dir=%s", args.mode, out_dir)
    if args.mode == "classification":
        result = run_classification(args)
    else:
        result = run_composition(args)

    out_file = out_dir / f"{args.mode}_summary.json"
    out_file.write_text(json.dumps(result, indent=2, sort_keys=True))
    logger.info("wrote %s", out_file)

    # Print a concise summary so post-merge-benchmark.sh can grep it
    if args.mode == "classification":
        print(
            f"VRABEL2020 classification: "
            f"accuracy={result['accuracy']:.4f} "
            f"macro_f1={result['macro_f1']:.4f} "
            f"floor_passed={result['passed_regression_floor']}"
        )
    else:
        print(
            f"VRABEL2020 composition: "
            f"accuracy={result['classifier_accuracy']:.4f} "
            f"mean_aitchison={result['mean_aitchison_distance']:.4f} "
            f"floor_passed={result['passed_regression_floor']}"
        )

    return 0 if result["passed_regression_floor"] else 1


if __name__ == "__main__":
    sys.exit(main())
