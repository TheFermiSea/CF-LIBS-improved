#!/usr/bin/env python
"""
Run the identifier benchmark on a synthetic corpus.

By default the peak-matching trio (ALIAS/Comb/Correlation) is run. Pass
``--with-nnls`` and/or ``--with-hybrid`` to additionally run the
basis-dependent ``spectral_nnls`` and ``hybrid_union`` identifiers. Either of
those flags enables basis-library construction; a small per-corpus basis
library is built once at ``--basis-library-path`` (or supply an existing
one). If the basis library can't be built, those identifiers are skipped and
the run continues with the trio.

Spectrum sampling under ``--max-spectra`` defaults to ``--sampling stratified``
with a fixed ``--seed`` (reproducible): the draw is proportional over
``(recipe, label_cardinality)`` strata with >=1 per non-empty stratum, so every
recipe is represented. This replaces the old sorted-then-truncate behavior,
which (on the 4-recipe corpus) returned 100% of the alphabetically-first recipe
for any cap <= the per-recipe count. Pass ``--sampling sorted`` to reproduce the
legacy biased draw exactly. The realized recipe / cardinality counts are printed
after the run so a skewed draw is obvious at a glance.
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

    # Spectrum-selection (sampling) controls. Default is stratified + seeded
    # (reproducible) so a --max-spectra cap covers all recipes/cardinalities.
    parser.add_argument(
        "--sampling",
        type=str,
        default="stratified",
        choices=["sorted", "shuffle", "stratified"],
        help=(
            "Spectrum-selection mode when --max-spectra is set. "
            "'stratified' (default) covers all (recipe, cardinality) strata; "
            "'shuffle' is a seeded uniform draw; 'sorted' is the legacy "
            "alpha-sort+truncate (biased toward the first recipe)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the stratified/shuffle draw (reproducible).",
    )
    parser.add_argument(
        "--panel",
        type=str,
        default="full",
        choices=["full", "ever_present"],
        help=(
            "Which aggregate is the headline metric: 'full' (all candidate "
            "elements, default) or 'ever_present' (only elements present in the "
            "sampled truth). Both aggregates are always emitted."
        ),
    )
    parser.add_argument(
        "--panel-elements",
        type=str,
        default="",
        help=(
            "Optional explicit candidate element list (comma-separated) used as "
            "the scoring panel. Overrides --elements / dataset elements."
        ),
    )

    # Full-stack identifiers (basis-dependent; opt-in).
    parser.add_argument(
        "--with-nnls",
        action="store_true",
        help="Also run the spectral_nnls identifier (requires a basis library).",
    )
    parser.add_argument(
        "--with-hybrid",
        action="store_true",
        help="Also run the hybrid_union identifier (requires a basis library).",
    )
    parser.add_argument(
        "--with-forward-fit",
        action="store_true",
        help=(
            "Also run the J10 population forward-fitting identifier "
            "(requires JAX; CPU is fine but slower)."
        ),
    )
    parser.add_argument(
        "--basis-library-path",
        type=str,
        default=None,
        help=(
            "HDF5 path to build (if missing) or load the per-corpus basis "
            "library from. Defaults to <output-dir>/basis_fwhm_<fwhm>nm.h5."
        ),
    )
    parser.add_argument(
        "--basis-fwhm-nm",
        type=float,
        default=0.3,
        help="Instrument FWHM (nm) for the built basis library (default: 0.3).",
    )

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

    include_nnls = bool(args.with_nnls or args.with_hybrid)
    basis_library_path = args.basis_library_path
    if include_nnls and basis_library_path is None:
        basis_library_path = str(output_dir / f"basis_fwhm_{args.basis_fwhm_nm:g}nm.h5")

    # --panel-elements overrides --elements as the explicit scoring panel.
    if args.panel_elements:
        candidate_elements = _parse_elements(args.panel_elements)
    elif args.elements:
        candidate_elements = _parse_elements(args.elements)
    else:
        candidate_elements = None

    result = run_synthetic_benchmark(
        dataset_path=args.dataset_path,
        db_path=args.db_path,
        output_dir=str(output_dir),
        candidate_elements=candidate_elements,
        manifest_path=manifest_path if manifest_path else None,
        presence_threshold=args.presence_threshold,
        max_spectra=args.max_spectra,
        calibration=calibration,
        include_nnls=include_nnls,
        basis_library_path=basis_library_path,
        basis_instrument_fwhm_nm=args.basis_fwhm_nm,
        with_forward_fit=bool(args.with_forward_fit),
        sampling=args.sampling,
        seed=args.seed,
        panel=args.panel,
    )

    summary = result["summary"]
    print(f"Synthetic benchmark complete: {result['output_dir']}")
    print(
        f"Sampling: {summary['sampling']} (seed={summary['seed']}, panel={summary['panel']})"
    )
    print(f"Sampled recipe counts:      {json.dumps(summary['sampled_recipe_counts'])}")
    print(f"Sampled cardinality counts: {json.dumps(summary['sampled_cardinality_counts'])}")
    print(f"Ever-present elements:      {summary['ever_present_elements']}")
    print(json.dumps(summary["aggregate_metrics"], indent=2))


if __name__ == "__main__":
    main()
