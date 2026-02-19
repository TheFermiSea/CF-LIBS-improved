"""
Synthetic corpus identifier benchmark utilities.

Runs ALIAS, Comb, and Correlation on synthetic benchmark spectra and produces
element-level + peak-level diagnostics suitable for regression tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import csv
import json
import logging

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.dataset import BenchmarkDataset, BenchmarkSpectrum
from cflibs.benchmark.loaders import load_benchmark
from cflibs.inversion.alias_identifier import ALIASIdentifier
from cflibs.inversion.comb_identifier import CombIdentifier
from cflibs.inversion.correlation_identifier import CorrelationIdentifier
from cflibs.inversion.element_id import ElementIdentification, ElementIdentificationResult
from cflibs.inversion.wavelength_calibration import calibrate_wavelength_axis

logger = logging.getLogger(__name__)


def derive_truth_elements(
    composition: Dict[str, float], presence_threshold: float = 1e-4
) -> Set[str]:
    """Return set of elements considered present in ground truth."""
    return {el for el, frac in composition.items() if float(frac) >= float(presence_threshold)}


def compute_binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Compute precision/recall/FPR/F1/accuracy from confusion counts."""
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    tn = int(tn)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def confusion_counts(
    true_elements: Set[str],
    predicted_elements: Set[str],
    candidate_elements: Sequence[str],
) -> Dict[str, int]:
    """Return TP/FP/FN/TN counts for one sample."""
    tp = fp = fn = tn = 0
    for element in candidate_elements:
        truth = element in true_elements
        pred = element in predicted_elements
        if truth and pred:
            tp += 1
        elif (not truth) and pred:
            fp += 1
        elif truth and (not pred):
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def resolving_power_from_spectrum(spec: BenchmarkSpectrum, fallback: float = 900.0) -> float:
    """
    Infer resolving power from benchmark spectrum conditions.

    Uses mean wavelength / spectral_resolution_nm when available.
    """
    resolution_nm = float(spec.conditions.spectral_resolution_nm)
    if not np.isfinite(resolution_nm) or resolution_nm <= 0:
        return float(fallback)
    mean_wl = float(np.mean(spec.wavelength_nm))
    rp = mean_wl / resolution_nm
    return float(np.clip(rp, 100.0, 20000.0))


def _element_by_name(result: ElementIdentificationResult) -> Dict[str, ElementIdentification]:
    return {e.element: e for e in result.all_elements}


def _identifier_suite(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    db: AtomicDatabase,
    elements: List[str],
    resolving_power: float,
) -> Dict[str, Optional[ElementIdentificationResult]]:
    results: Dict[str, Optional[ElementIdentificationResult]] = {}
    algorithms: List[Tuple[str, Any, Dict[str, Any]]] = [
        (
            "ALIAS",
            ALIASIdentifier,
            {
                "resolving_power": resolving_power,
                "intensity_threshold_factor": 3.0,
                "detection_threshold": 0.01,
                "chance_window_scale": 0.3,
            },
        ),
        (
            "Comb",
            CombIdentifier,
            {
                "resolving_power": resolving_power,
                "min_correlation": 0.08,
                "tooth_activation_threshold": 0.35,
                "relative_threshold_scale": 1.4,
                "min_aki_gk": 3000.0,
            },
        ),
        (
            "Correlation",
            CorrelationIdentifier,
            {
                "resolving_power": resolving_power,
                "min_confidence": 0.008,
                "relative_threshold_scale": 1.2,
                "min_line_strength": 1000.0,
                "T_range_K": (5000, 15000),
                "T_steps": 7,
                "n_e_range_cm3": (1e15, 5e17),
                "n_e_steps": 4,
            },
        ),
    ]
    for name, cls, kwargs in algorithms:
        try:
            identifier = cls(db, elements=elements, **kwargs)
            if name == "Correlation":
                results[name] = identifier.identify(wavelength, intensity, mode="classic")
            else:
                results[name] = identifier.identify(wavelength, intensity)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Identifier %s failed during synthetic benchmark run", name, exc_info=True
            )
            results[name] = None
    return results


@dataclass
class CalibrationOptions:
    """Options for optional wavelength-calibration preprocessing."""

    mode: str = "none"
    max_pair_window_nm: float = 2.0
    inlier_tolerance_nm: float = 0.08
    apply_quality_gate: bool = True
    quality_min_inliers: int = 12
    quality_min_peak_match_fraction: float = 0.35
    quality_max_rmse_nm: float = 0.10
    quality_min_inlier_span_fraction: float = 0.25
    quality_max_abs_correction_nm: float = 2.5


def evaluate_dataset(
    dataset: BenchmarkDataset,
    db: AtomicDatabase,
    candidate_elements: List[str],
    manifest_by_sample: Optional[Dict[str, Dict[str, Any]]] = None,
    presence_threshold: float = 1e-4,
    max_spectra: Optional[int] = None,
    calibration: Optional[CalibrationOptions] = None,
) -> Dict[str, Any]:
    """
    Evaluate all identifiers on synthetic benchmark dataset.

    Parameters
    ----------
    dataset : BenchmarkDataset
        Synthetic benchmark dataset to evaluate.
    db : AtomicDatabase
        Atomic database used by the identifiers.
    candidate_elements : List[str]
        Candidate elements considered for detection.
    manifest_by_sample : dict, optional
        Optional per-sample perturbation metadata.
    presence_threshold : float
        Minimum composition fraction to consider an element present.
    max_spectra : int, optional
        Optional cap on number of spectra evaluated.
    calibration : CalibrationOptions, optional
        Optional wavelength calibration settings.

    Returns
    -------
    Dict[str, Any]
        Dictionary with per-spectrum rows and aggregate summaries.
    """
    if calibration is None:
        calibration = CalibrationOptions()

    spectra = sorted(dataset.spectra, key=lambda s: s.spectrum_id)
    if max_spectra is not None:
        spectra = spectra[: int(max_spectra)]

    rows: List[Dict[str, Any]] = []
    algorithms = ["ALIAS", "Comb", "Correlation"]

    for idx, spec in enumerate(spectra, start=1):
        true_elements = derive_truth_elements(
            spec.true_composition, presence_threshold=presence_threshold
        )
        wl = np.asarray(spec.wavelength_nm, dtype=float)
        intensity = np.asarray(spec.intensity, dtype=float)

        cal_meta: Dict[str, Any] = {
            "calibration_mode": calibration.mode,
            "calibration_applied": False,
            "calibration_success": False,
            "calibration_quality_passed": False,
            "calibration_reason": "",
            "calibration_model": "none",
        }
        if calibration.mode != "none":
            cal = calibrate_wavelength_axis(
                wavelength=wl,
                intensity=intensity,
                atomic_db=db,
                elements=candidate_elements,
                mode=calibration.mode,  # type: ignore[arg-type]
                max_pair_window_nm=calibration.max_pair_window_nm,
                inlier_tolerance_nm=calibration.inlier_tolerance_nm,
                apply_quality_gate=calibration.apply_quality_gate,
                quality_min_inliers=calibration.quality_min_inliers,
                quality_min_peak_match_fraction=calibration.quality_min_peak_match_fraction,
                quality_max_rmse_nm=calibration.quality_max_rmse_nm,
                quality_min_inlier_span_fraction=calibration.quality_min_inlier_span_fraction,
                quality_max_abs_correction_nm=calibration.quality_max_abs_correction_nm,
            )
            cal_meta.update(
                {
                    "calibration_success": bool(cal.success),
                    "calibration_quality_passed": bool(cal.quality_passed),
                    "calibration_reason": cal.quality_reason or str(cal.details.get("reason", "")),
                    "calibration_model": cal.model,
                    "calibration_rmse_nm": float(cal.rmse_nm),
                    "calibration_n_inliers": int(cal.n_inliers),
                    "calibration_n_candidates": int(cal.n_candidates),
                    "calibration_peak_match_fraction": float(cal.matched_peak_fraction),
                }
            )
            if cal.success and cal.quality_passed:
                wl = cal.corrected_wavelength
                cal_meta["calibration_applied"] = True

        resolving_power = resolving_power_from_spectrum(spec)
        results = _identifier_suite(
            wavelength=wl,
            intensity=intensity,
            db=db,
            elements=candidate_elements,
            resolving_power=resolving_power,
        )

        manifest_meta = (manifest_by_sample or {}).get(spec.spectrum_id, {})
        perturb = manifest_meta.get("perturbation", {})

        for algo_name in algorithms:
            result = results.get(algo_name)
            if result is None:
                rows.append(
                    {
                        "sample_id": spec.spectrum_id,
                        "algorithm": algo_name,
                        "failed": True,
                        "true_elements": sorted(true_elements),
                        "predicted_elements": [],
                        "tp": 0,
                        "fp": 0,
                        "fn": len(true_elements),
                        "tn": max(len(candidate_elements) - len(true_elements), 0),
                        "n_peaks": 0,
                        "n_matched_peaks": 0,
                        "n_unmatched_peaks": 0,
                        "peak_match_rate": 0.0,
                        "matched_lines_true_elements": 0,
                        "total_lines_true_elements": 0,
                        "matched_lines_absent_elements": 0,
                        "resolving_power": resolving_power,
                        "recipe": manifest_meta.get("recipe", ""),
                        "snr_db": perturb.get("snr_db"),
                        "continuum_level": perturb.get("continuum_level"),
                        "shift_nm": perturb.get("shift_nm"),
                        "warp_quadratic_nm": perturb.get("warp_quadratic_nm"),
                        **cal_meta,
                    }
                )
                continue

            predicted_elements = {e.element for e in result.detected_elements}
            counts = confusion_counts(true_elements, predicted_elements, candidate_elements)
            by_element = _element_by_name(result)

            matched_true = 0
            total_true = 0
            matched_absent = 0
            for element in candidate_elements:
                elem_result = by_element.get(element)
                if elem_result is None:
                    continue
                if element in true_elements:
                    matched_true += int(elem_result.n_matched_lines)
                    total_true += int(elem_result.n_total_lines)
                else:
                    matched_absent += int(elem_result.n_matched_lines)

            rows.append(
                {
                    "sample_id": spec.spectrum_id,
                    "algorithm": algo_name,
                    "failed": False,
                    "true_elements": sorted(true_elements),
                    "predicted_elements": sorted(predicted_elements),
                    **counts,
                    "n_peaks": int(result.n_peaks),
                    "n_matched_peaks": int(result.n_matched_peaks),
                    "n_unmatched_peaks": int(result.n_unmatched_peaks),
                    "peak_match_rate": float(result.n_matched_peaks / max(result.n_peaks, 1)),
                    "matched_lines_true_elements": int(matched_true),
                    "total_lines_true_elements": int(total_true),
                    "matched_lines_absent_elements": int(matched_absent),
                    "resolving_power": float(resolving_power),
                    "recipe": manifest_meta.get("recipe", ""),
                    "snr_db": perturb.get("snr_db"),
                    "continuum_level": perturb.get("continuum_level"),
                    "shift_nm": perturb.get("shift_nm"),
                    "warp_quadratic_nm": perturb.get("warp_quadratic_nm"),
                    **cal_meta,
                }
            )

        if idx % 20 == 0 or idx == len(spectra):
            print(f"[synthetic-benchmark] processed {idx}/{len(spectra)} spectra")

    aggregate = summarize_aggregate(rows, candidate_elements)
    per_element = summarize_per_element(rows, candidate_elements)
    group_metrics = summarize_by_group(rows, candidate_elements)
    return {
        "rows": rows,
        "aggregate_metrics": aggregate,
        "per_element_metrics": per_element,
        "group_metrics": group_metrics,
    }


def summarize_aggregate(
    rows: List[Dict[str, Any]], candidate_elements: List[str]
) -> List[Dict[str, Any]]:
    """Aggregate confusion + peak metrics by algorithm."""
    out: List[Dict[str, Any]] = []
    algorithms = sorted({row["algorithm"] for row in rows})
    for algorithm in algorithms:
        subset = [row for row in rows if row["algorithm"] == algorithm and not row["failed"]]
        failed_count = sum(1 for row in rows if row["algorithm"] == algorithm and row["failed"])
        tp = sum(int(r["tp"]) for r in subset)
        fp = sum(int(r["fp"]) for r in subset)
        fn = sum(int(r["fn"]) for r in subset)
        tn = sum(int(r["tn"]) for r in subset)
        metrics = compute_binary_metrics(tp, fp, fn, tn)
        out.append(
            {
                "algorithm": algorithm,
                "n_spectra": len(subset),
                "n_failed": failed_count,
                "n_candidate_elements": len(candidate_elements),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                **metrics,
                "mean_peak_match_rate": float(
                    np.mean([float(r["peak_match_rate"]) for r in subset]) if subset else 0.0
                ),
                "mean_n_peaks": float(
                    np.mean([float(r["n_peaks"]) for r in subset]) if subset else 0.0
                ),
                "mean_n_matched_peaks": float(
                    np.mean([float(r["n_matched_peaks"]) for r in subset]) if subset else 0.0
                ),
                "mean_true_line_match_fraction": float(
                    np.mean(
                        [
                            float(r["matched_lines_true_elements"])
                            / max(float(r["total_lines_true_elements"]), 1.0)
                            for r in subset
                        ]
                    )
                    if subset
                    else 0.0
                ),
                "mean_absent_matched_lines": float(
                    np.mean([float(r["matched_lines_absent_elements"]) for r in subset])
                    if subset
                    else 0.0
                ),
            }
        )
    return out


def summarize_per_element(
    rows: List[Dict[str, Any]],
    candidate_elements: List[str],
) -> List[Dict[str, Any]]:
    """Compute per-element confusion metrics for each algorithm."""
    out: List[Dict[str, Any]] = []
    algorithms = sorted({row["algorithm"] for row in rows})
    for algorithm in algorithms:
        subset = [row for row in rows if row["algorithm"] == algorithm and not row["failed"]]
        for element in candidate_elements:
            tp = fp = fn = tn = 0
            for row in subset:
                truth = element in set(row["true_elements"])
                pred = element in set(row["predicted_elements"])
                if truth and pred:
                    tp += 1
                elif (not truth) and pred:
                    fp += 1
                elif truth and (not pred):
                    fn += 1
                else:
                    tn += 1
            metrics = compute_binary_metrics(tp, fp, fn, tn)
            out.append(
                {
                    "algorithm": algorithm,
                    "element": element,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    **metrics,
                }
            )
    return out


def summarize_by_group(
    rows: List[Dict[str, Any]],
    candidate_elements: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate metrics by recipe and perturbation axis values."""
    group_fields = ["recipe", "snr_db", "continuum_level", "shift_nm", "warp_quadratic_nm"]
    output: Dict[str, List[Dict[str, Any]]] = {}
    algorithms = sorted({row["algorithm"] for row in rows})

    for field in group_fields:
        grouped_rows: List[Dict[str, Any]] = []
        values = sorted({value for row in rows if (value := row.get(field)) is not None})
        for value in values:
            for algorithm in algorithms:
                subset = [
                    row
                    for row in rows
                    if row["algorithm"] == algorithm
                    and not row["failed"]
                    and row.get(field) == value
                ]
                if not subset:
                    continue
                tp = sum(int(r["tp"]) for r in subset)
                fp = sum(int(r["fp"]) for r in subset)
                fn = sum(int(r["fn"]) for r in subset)
                tn = sum(int(r["tn"]) for r in subset)
                metrics = compute_binary_metrics(tp, fp, fn, tn)
                grouped_rows.append(
                    {
                        "group_field": field,
                        "group_value": value,
                        "algorithm": algorithm,
                        "n_rows": len(subset),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        **metrics,
                        "mean_peak_match_rate": float(
                            np.mean([float(r["peak_match_rate"]) for r in subset])
                        ),
                    }
                )
        output[field] = grouped_rows
    return output


def _load_manifest_index(manifest_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if manifest_path is None:
        return {}
    path = Path(manifest_path)
    if not path.exists():
        return {}
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open() as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped))
    else:
        rows = json.loads(path.read_text())
    return {row["sample_id"]: row for row in rows if "sample_id" in row}


def run_synthetic_benchmark(
    dataset_path: str,
    db_path: str,
    output_dir: str,
    candidate_elements: Optional[List[str]] = None,
    manifest_path: Optional[str] = None,
    presence_threshold: float = 1e-4,
    max_spectra: Optional[int] = None,
    calibration: Optional[CalibrationOptions] = None,
) -> Dict[str, Any]:
    """Load data, run identifier benchmark, and write artifacts."""
    dataset = load_benchmark(dataset_path)
    elements = candidate_elements if candidate_elements else list(dataset.elements)
    manifest_index = _load_manifest_index(manifest_path)
    db = AtomicDatabase(db_path)

    evaluated = evaluate_dataset(
        dataset=dataset,
        db=db,
        candidate_elements=elements,
        manifest_by_sample=manifest_index,
        presence_threshold=presence_threshold,
        max_spectra=max_spectra,
        calibration=calibration,
    )

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluated["rows"]
    aggregate = evaluated["aggregate_metrics"]
    per_element = evaluated["per_element_metrics"]
    group_metrics = evaluated["group_metrics"]

    # summary json
    summary = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "db_path": str(Path(db_path).resolve()),
        "n_rows": len(rows),
        "n_spectra": len({row["sample_id"] for row in rows}),
        "candidate_elements": elements,
        "presence_threshold": float(presence_threshold),
        "max_spectra": int(max_spectra) if max_spectra is not None else None,
        "calibration": (
            calibration.__dict__ if calibration is not None else CalibrationOptions().__dict__
        ),
        "aggregate_metrics": aggregate,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "group_metrics.json").write_text(json.dumps(group_metrics, indent=2))

    with (out_dir / "per_spectrum.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    _write_csv(out_dir / "aggregate_metrics.csv", aggregate)
    _write_csv(out_dir / "per_element_metrics.csv", per_element)
    for field, metrics in group_metrics.items():
        _write_csv(out_dir / f"group_metrics_{field}.csv", metrics)

    return {
        "summary": summary,
        "output_dir": str(out_dir),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
