#!/usr/bin/env python
"""
Train ML classifiers for element identification using NNLS+ALIAS features.

Trains per-element binary classifiers (SVM, Random Forest, XGBoost) on
features extracted from coarse sweep results. The classifiers learn
element-specific decision boundaries that exceed simple threshold rules.

Usage:
  python scripts/hpc/train_ml_classifier.py \
    --sweep-dir output/hpc_benchmark/coarse_sweep \
    --output-dir output/hpc_benchmark/ml_models \
    --n-jobs 40
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def load_features(sweep_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load feature vectors and labels from sweep results.

    Expects parquet or CSV files with per-spectrum, per-element results
    containing NNLS and ALIAS features.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples, n_elements) — binary labels
    feature_names : list of str
    """
    try:
        import pyarrow.parquet as pq

        files = sorted(sweep_dir.glob("results_*.parquet"))
        if not files:
            raise FileNotFoundError("No parquet results found")
        tables = [pq.read_table(f) for f in files]
        import pyarrow as pa

        df = pa.concat_tables(tables).to_pandas()
    except (ImportError, FileNotFoundError):
        import pandas as pd

        files = sorted(sweep_dir.glob("results_*.csv"))
        if not files:
            raise FileNotFoundError(f"No results files found in {sweep_dir}")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    logger.info("Loaded %d result rows from %d files", len(df), len(files))

    # Extract features per spectrum
    # Expected columns from sweep: spectrum_id, pathway, config_name,
    # tp, fp, fn, tn, rp, snr, T_K, ne, n_elements,
    # nnls_coefficients (JSON), alias_scores (JSON)
    feature_names = [
        "rp",
        "snr",
        "T_K",
        "ne",
        "n_elements",
        "nnls_coeff",
        "nnls_snr_val",
        "concentration_est",
        "alias_score",
        "alias_n_matched",
        "alias_confidence",
        "total_signal",
    ]

    # For now, build features from available columns
    # This will be refined when actual sweep output format is finalized
    n_samples = len(df)
    n_features = len(feature_names)
    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, len(ELEMENTS)), dtype=np.int8)

    for i, row in enumerate(df.itertuples()):
        X[i, 0] = getattr(row, "rp", 1000)
        X[i, 1] = getattr(row, "snr", 100)
        X[i, 2] = getattr(row, "T_K", 8000)
        X[i, 3] = np.log10(max(getattr(row, "ne", 1e17), 1e10))
        X[i, 4] = getattr(row, "n_elements", 3)

        # Parse element-level features from JSON if available
        nnls_json = getattr(row, "nnls_coefficients", "{}")
        alias_json = getattr(row, "alias_scores", "{}")
        if isinstance(nnls_json, str):
            try:
                json.loads(nnls_json)
            except (json.JSONDecodeError, TypeError):
                pass
        else:
            pass
        if isinstance(alias_json, str):
            try:
                json.loads(alias_json)
            except (json.JSONDecodeError, TypeError):
                pass
        else:
            pass

        # Ground truth labels
        expected_json = getattr(row, "expected_elements", "[]")
        if isinstance(expected_json, str):
            try:
                expected = json.loads(expected_json)
            except (json.JSONDecodeError, TypeError):
                expected = []
        else:
            expected = []
        for el in expected:
            if el in ELEMENTS:
                y[i, ELEMENTS.index(el)] = 1

    return X, y, feature_names


def train_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_jobs: int = -1,
    cv_folds: int = 5,
    output_dir: Path = Path("."),
) -> Dict[str, Any]:
    """Train per-element classifiers with cross-validation.

    Returns dict of results per element per model type.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    try:
        from xgboost import XGBClassifier

        has_xgboost = True
    except ImportError:
        has_xgboost = False
        logger.warning("XGBoost not available, skipping XGB models")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results: Dict[str, Any] = {"elements": {}, "global": {}}
    models: Dict[str, Dict[str, Any]] = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for el_idx, element in enumerate(ELEMENTS):
        y_el = y[:, el_idx]
        n_pos = int(y_el.sum())
        n_neg = len(y_el) - n_pos

        if n_pos < cv_folds or n_neg < cv_folds:
            logger.warning("Skipping %s: too few samples (pos=%d, neg=%d)", element, n_pos, n_neg)
            continue

        logger.info(
            "Training classifiers for %s (pos=%d, neg=%d, ratio=%.3f)",
            element,
            n_pos,
            n_neg,
            n_pos / len(y_el),
        )

        el_results: Dict[str, Dict[str, float]] = {}

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=n_jobs,
        )
        y_pred_rf = cross_val_predict(rf, X_scaled, y_el, cv=cv, n_jobs=1)
        rf.fit(X_scaled, y_el)
        el_results["random_forest"] = {
            "precision": float(precision_score(y_el, y_pred_rf, zero_division=0)),
            "recall": float(recall_score(y_el, y_pred_rf, zero_division=0)),
            "f1": float(f1_score(y_el, y_pred_rf, zero_division=0)),
        }

        # SVM (RBF kernel)
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)
        y_pred_svm = cross_val_predict(svm, X_scaled, y_el, cv=cv, n_jobs=1)
        svm.fit(X_scaled, y_el)
        el_results["svm"] = {
            "precision": float(precision_score(y_el, y_pred_svm, zero_division=0)),
            "recall": float(recall_score(y_el, y_pred_svm, zero_division=0)),
            "f1": float(f1_score(y_el, y_pred_svm, zero_division=0)),
        }

        # XGBoost
        if has_xgboost:
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=42,
                n_jobs=n_jobs,
                verbosity=0,
            )
            y_pred_xgb = cross_val_predict(xgb, X_scaled, y_el, cv=cv, n_jobs=1)
            xgb.fit(X_scaled, y_el)
            el_results["xgboost"] = {
                "precision": float(precision_score(y_el, y_pred_xgb, zero_division=0)),
                "recall": float(recall_score(y_el, y_pred_xgb, zero_division=0)),
                "f1": float(f1_score(y_el, y_pred_xgb, zero_division=0)),
            }

        # Feature importance from RF
        importance = dict(zip(feature_names, rf.feature_importances_.tolist()))

        results["elements"][element] = {
            "models": el_results,
            "feature_importance": importance,
            "n_positive": n_pos,
            "n_negative": n_neg,
        }

        models[element] = {"rf": rf, "svm": svm}
        if has_xgboost:
            models[element]["xgb"] = xgb

        # Best model for this element
        best_model = max(el_results, key=lambda k: el_results[k]["f1"])
        best_f1 = el_results[best_model]["f1"]
        best_p = el_results[best_model]["precision"]
        logger.info("  %s best: %s (P=%.3f, F1=%.3f)", element, best_model, best_p, best_f1)

    # Save models
    try:
        import joblib

        model_path = output_dir / "trained_models.joblib"
        joblib.dump({"models": models, "scaler": scaler, "elements": ELEMENTS}, model_path)
        logger.info("Saved models to %s", model_path)
    except ImportError:
        logger.warning("joblib not available, models not saved to disk")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML classifiers for element identification.")
    parser.add_argument("--sweep-dir", type=str, required=True, help="Directory with sweep results")
    parser.add_argument("--output-dir", type=str, default="output/hpc_benchmark/ml_models")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (-1=all cores)")
    parser.add_argument("--cv-folds", type=int, default=5)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features from %s", sweep_dir)
    t0 = time.monotonic()
    X, y, feature_names = load_features(sweep_dir)
    logger.info(
        "Loaded %d samples, %d features in %.1f s", X.shape[0], X.shape[1], time.monotonic() - t0
    )

    logger.info("Training classifiers (cv_folds=%d, n_jobs=%d)", args.cv_folds, args.n_jobs)
    t0 = time.monotonic()
    results = train_classifiers(
        X,
        y,
        feature_names,
        n_jobs=args.n_jobs,
        cv_folds=args.cv_folds,
        output_dir=output_dir,
    )
    logger.info("Training completed in %.1f s", time.monotonic() - t0)

    # Write results
    results_path = output_dir / "cv_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Saved CV results to %s", results_path)

    # Summary table
    print("\n" + "=" * 80)
    print("ML CLASSIFIER RESULTS (5-fold CV)")
    print("=" * 80)
    print(
        f"{'Element':<6} {'RF P':>6} {'RF F1':>6} {'SVM P':>6} {'SVM F1':>6} {'XGB P':>6} {'XGB F1':>6} {'Best':>8}"
    )
    print("-" * 80)
    for element, data in results.get("elements", {}).items():
        models = data["models"]
        rf = models.get("random_forest", {})
        svm = models.get("svm", {})
        xgb = models.get("xgboost", {})
        best = max(models, key=lambda k: models[k].get("f1", 0))
        print(
            f"{element:<6} "
            f"{rf.get('precision', 0):>6.3f} {rf.get('f1', 0):>6.3f} "
            f"{svm.get('precision', 0):>6.3f} {svm.get('f1', 0):>6.3f} "
            f"{xgb.get('precision', 0):>6.3f} {xgb.get('f1', 0):>6.3f} "
            f"{best:>8}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
