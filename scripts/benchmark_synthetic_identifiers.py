#!/usr/bin/env python
"""
Run ALIAS/Comb/Correlation benchmark on synthetic corpus.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Keep JAX on CPU for stable local/CI execution.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from cflibs.benchmark.synthetic_eval import CalibrationOptions, run_synthetic_benchmark

DEFAULT_MANIFEST_PATH = "output/synthetic_corpus/ak3_1_3_corpus_v1/manifest.jsonl"


def _parse_elements(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="output/synthetic_corpus/ak3_1_3_corpus_v1/corpus.json",
    )
    parser.add_argument("--db-path", type=str, default="ASD_da/libs_production.db")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/synthetic_benchmark/ak3_1_4_v1",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help=f"Optional manifest JSONL path (default if present: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--elements",
        type=str,
        default="",
        help="Comma-separated candidate element list (default: dataset elements)",
    )
    parser.add_argument("--presence-threshold", type=float, default=1e-4)
    parser.add_argument("--max-spectra", type=int, default=None)

    # Optional calibration pass before identification
    parser.add_argument(
        "--wavelength-calibration-mode",
        type=str,
        default="none",
        choices=["none", "auto", "shift", "affine", "quadratic"],
    )
    parser.add_argument("--wavelength-calibration-max-pair-window", type=float, default=2.0)
    parser.add_argument("--wavelength-calibration-inlier-tol", type=float, default=0.08)
    parser.add_argument("--wavelength-calibration-gate-disable", action="store_true")
    parser.add_argument("--wavelength-calibration-gate-min-inliers", type=int, default=12)
    parser.add_argument("--wavelength-calibration-gate-min-peak-match", type=float, default=0.35)
    parser.add_argument("--wavelength-calibration-gate-max-rmse", type=float, default=0.10)
    parser.add_argument("--wavelength-calibration-gate-min-span-frac", type=float, default=0.25)
    parser.add_argument(
        "--wavelength-calibration-gate-max-abs-correction",
        type=float,
        default=2.5,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    default_manifest = Path(DEFAULT_MANIFEST_PATH)
    manifest_path = args.manifest_path
    if manifest_path is None and default_manifest.exists():
        manifest_path = str(default_manifest)

    calibration = CalibrationOptions(
        mode=args.wavelength_calibration_mode,
        max_pair_window_nm=args.wavelength_calibration_max_pair_window,
        inlier_tolerance_nm=args.wavelength_calibration_inlier_tol,
        apply_quality_gate=not args.wavelength_calibration_gate_disable,
        quality_min_inliers=args.wavelength_calibration_gate_min_inliers,
        quality_min_peak_match_fraction=args.wavelength_calibration_gate_min_peak_match,
        quality_max_rmse_nm=args.wavelength_calibration_gate_max_rmse,
        quality_min_inlier_span_fraction=args.wavelength_calibration_gate_min_span_frac,
        quality_max_abs_correction_nm=args.wavelength_calibration_gate_max_abs_correction,
    )

    result = run_synthetic_benchmark(
        dataset_path=args.dataset_path,
        db_path=args.db_path,
        output_dir=str(output_dir),
        candidate_elements=_parse_elements(args.elements) if args.elements else None,
        manifest_path=manifest_path if manifest_path else None,
        presence_threshold=args.presence_threshold,
        max_spectra=args.max_spectra,
        calibration=calibration,
    )

    summary = result["summary"]
    print(f"Synthetic benchmark complete: {result['output_dir']}")
    print(json.dumps(summary["aggregate_metrics"], indent=2))


if __name__ == "__main__":
    main()
