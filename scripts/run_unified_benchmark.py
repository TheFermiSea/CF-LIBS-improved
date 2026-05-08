#!/usr/bin/env python3
"""Run the unified LIBS benchmark stack.

This CLI is a thin front-end over :class:`cflibs.benchmark.unified.UnifiedBenchmarkRunner`.
It loads the configured datasets, runs the selected identification and/or
composition workflows, and writes the benchmark artifacts to an output directory.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from cflibs.benchmark.dataset import BenchmarkDataset, TruthType


BASIS_REQUIRED_ID_WORKFLOWS = {
    "hybrid_intersect",
    "hybrid_union",
    "nnls_concentration_threshold",
    "spectral_nnls",
}


def _normalize_workflow_list(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item and item not in normalized:
                normalized.append(item)
    return normalized


def _has_truth(dataset: BenchmarkDataset, truth_types: Iterable[TruthType]) -> bool:
    allowed = {getattr(truth_type, "value", truth_type) for truth_type in truth_types}
    return any(spectrum.truth_type.value in allowed for spectrum in dataset.spectra)


def _select_datasets(
    datasets: dict[str, BenchmarkDataset],
    truth_types: Iterable[TruthType],
) -> list[BenchmarkDataset]:
    return [dataset for dataset in datasets.values() if _has_truth(dataset, truth_types)]


def _validate_paths(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.db_path.exists():
        parser.error(f"Atomic database not found: {args.db_path}")
    if not args.data_dir.exists():
        parser.error(f"Data directory not found: {args.data_dir}")
    if args.synthetic_corpus is not None and not args.synthetic_corpus.exists():
        parser.error(f"Synthetic corpus not found: {args.synthetic_corpus}")


def _validate_basis_requirements(
    parser: argparse.ArgumentParser,
    id_workflows: Sequence[str],
    basis_dir: Path,
) -> None:
    if not set(id_workflows) & BASIS_REQUIRED_ID_WORKFLOWS:
        return
    if not basis_dir.exists():
        parser.error(
            "Basis-driven identification workflows were selected, but the basis directory "
            f"does not exist: {basis_dir}"
        )
    if not any(basis_dir.glob("basis_fwhm_*nm.h5")):
        parser.error(
            "Basis-driven identification workflows were selected, but no basis_fwhm_*.h5 "
            f"files were found in {basis_dir}"
        )


def _run_identification_phase(
    parser: argparse.ArgumentParser,
    runner,
    identification_datasets: Sequence[BenchmarkDataset],
    id_workflows: Sequence[str],
    max_outer_folds: int | None,
):
    if not identification_datasets:
        parser.error(
            "No identification-capable datasets were found in the selected data directory."
        )
    return runner.run_identification(
        identification_datasets,
        workflow_names=id_workflows,
        max_outer_folds=max_outer_folds,
    )


def _run_composition_phase(
    parser: argparse.ArgumentParser,
    runner,
    composition_datasets: Sequence[BenchmarkDataset],
    id_workflows: Sequence[str],
    composition_workflows: Sequence[str],
    max_outer_folds: int | None,
):
    if not composition_datasets:
        parser.error("No composition-capable datasets were found in the selected data directory.")
    return runner.run_composition(
        composition_datasets,
        id_workflow_names=id_workflows,
        composition_workflow_names=composition_workflows,
        max_outer_folds=max_outer_folds,
    )


def _selected_composition_config(
    runner,
    composition_selections: Sequence[dict[str, Any]],
    workflow_name: str,
) -> dict[str, Any]:
    for selection in composition_selections:
        if selection.get("composition_workflow_name") == workflow_name:
            return dict(selection.get("config", {}))
    return dict(runner.composition_registry[workflow_name].parameter_grid[0])


def _truth_identification_result(elements: Sequence[str]):
    from cflibs.inversion.element_id import ElementIdentification, ElementIdentificationResult

    detected = [
        ElementIdentification(
            element=element,
            detected=True,
            score=1.0,
            confidence=1.0,
            n_matched_lines=0,
            n_total_lines=0,
            matched_lines=[],
            unmatched_lines=[],
            metadata={"source": "truth_for_perturbation"},
        )
        for element in elements
    ]
    return ElementIdentificationResult(
        detected_elements=detected,
        rejected_elements=[],
        all_elements=detected,
        experimental_peaks=[],
        n_peaks=0,
        n_matched_peaks=0,
        n_unmatched_peaks=0,
        algorithm="truth_for_perturbation",
        parameters={"elements": list(elements), "candidate_elements": list(elements)},
        warnings=[],
    )


def _build_perturbation_pipeline(
    runner,
    workflow_name: str,
    spectra,
    config: dict[str, Any],
):
    from cflibs.benchmark.dataset import BenchmarkSpectrum

    workflow = runner.composition_registry[workflow_name]
    predictor = workflow.fit_predictor(runner.context, spectra, config)
    templates: dict[tuple[str, ...], BenchmarkSpectrum] = {}
    for spectrum in spectra:
        elements = tuple(
            sorted(element for element, value in spectrum.true_composition.items() if value > 0)
        )
        templates.setdefault(elements, spectrum)
    fallback_template = spectra[0]

    def pipeline_fn(wavelengths, intensities, elements):
        candidate_elements = list(elements)
        template = templates.get(tuple(candidate_elements), fallback_template)
        payload = template.to_dict()
        payload.update(
            {
                "spectrum_id": f"{template.spectrum_id}_perturbation_eval",
                "wavelength_nm": np.asarray(wavelengths, dtype=np.float64),
                "intensity": np.asarray(intensities, dtype=np.float64),
            }
        )
        spectrum = BenchmarkSpectrum.from_dict(payload)
        result = predictor(
            spectrum,
            candidate_elements,
            _truth_identification_result(candidate_elements),
        )
        return result

    return pipeline_fn


def _print_perturbation_report(report, perturbation_names: Sequence[str]) -> None:
    print(f"Perturbation Report for {report.pipeline_name}:")
    for perturbation_name in perturbation_names:
        deltas = report.delta_d_a[perturbation_name].values()
        valid_deltas = [delta for delta in deltas if math.isfinite(delta)]
        if not valid_deltas:
            print(f"  {perturbation_name}: No valid delta_d_A values computed.")
            continue

        avg_delta = sum(valid_deltas) / len(valid_deltas)
        max_delta = max(valid_deltas)
        print(
            f"  {perturbation_name}: Avg delta_d_A = {avg_delta:.4f}, "
            f"Max delta_d_A = {max_delta:.4f}"
        )

        if perturbation_name == "line_dropout" and avg_delta > 0.02:
            print(
                f"  ALARM: {report.pipeline_name} {perturbation_name} "
                f"delta_d_A ({avg_delta:.4f}) exceeds threshold 0.02!"
            )
        elif perturbation_name == "outlier_injection" and avg_delta > 0.05:
            print(
                f"  ALARM: {report.pipeline_name} {perturbation_name} "
                f"delta_d_A ({avg_delta:.4f}) exceeds threshold 0.05!"
            )


def _run_perturbation_phase(
    runner,
    composition_datasets,
    composition_workflows: Sequence[str],
    composition_selections: Sequence[dict[str, Any]],
) -> None:
    from cflibs.benchmark.robustness import (
        line_dropout_perturbation,
        outlier_injection_perturbation,
        run_perturbation_battery,
    )

    print("\n--- Running Perturbation Battery ---")
    rng = np.random.default_rng(42)
    perturbations = {
        "line_dropout": lambda spec: line_dropout_perturbation(spec, top_n=3),
        "outlier_injection": lambda spec: outlier_injection_perturbation(
            spec, fraction=0.05, sigma_mult=5.0, rng=rng
        ),
    }

    all_spectra = [
        spectrum
        for dataset in composition_datasets
        for spectrum in dataset.spectra
        if spectrum.true_composition
    ]
    if not all_spectra:
        print("No composition spectra available for perturbation battery.")
        return

    for workflow_name in composition_workflows:
        print(f"Running perturbations for {workflow_name}...")
        config = _selected_composition_config(runner, composition_selections, workflow_name)
        pipeline_fn = _build_perturbation_pipeline(runner, workflow_name, all_spectra, config)
        report = run_perturbation_battery(
            pipeline_fn,
            all_spectra,
            perturbations,
            pipeline_name=workflow_name,
        )
        _print_perturbation_report(report, tuple(perturbations))


def _build_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser for the unified LIBS benchmark.

    Defines CLI options for database and data paths, basis and synthetic corpus inputs,
    output directory, which benchmark sections and workflows to run, runtime controls
    (such as quick mode and max outer folds), and an optional flag to run robustness
    perturbation tests on composition workflows.

    Returns:
        argparse.ArgumentParser: A configured parser ready to parse the benchmark CLI.
    """
    parser = argparse.ArgumentParser(description="Run the unified LIBS benchmark pipeline.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("ASD_da/libs_production.db"),
        help="Path to the atomic database used by identification workflows.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the real benchmark datasets.",
    )
    parser.add_argument(
        "--basis-dir",
        type=Path,
        default=Path("output/basis_libraries"),
        help="Directory containing basis_fwhm_*.h5 files for basis-driven workflows.",
    )
    parser.add_argument(
        "--synthetic-corpus",
        type=Path,
        default=None,
        help="Optional path to a manifest-backed synthetic corpus.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/unified_benchmark"),
        help="Directory where benchmark artifacts will be written.",
    )
    parser.add_argument(
        "--sections",
        choices=("all", "id", "composition"),
        default="all",
        help="Which benchmark sections to run.",
    )
    parser.add_argument(
        "--id-workflows",
        nargs="*",
        default=None,
        help="Identification workflows to run. Defaults to all available workflows.",
    )
    parser.add_argument(
        "--composition-workflows",
        nargs="*",
        default=None,
        help="Composition workflows to run. Defaults to all available workflows.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller parameter grids for a faster smoke run.",
    )
    parser.add_argument(
        "--max-outer-folds",
        type=int,
        default=None,
        help="Limit the number of outer folds evaluated per dataset.",
    )
    parser.add_argument(
        "--perturb",
        action="store_true",
        help="Run robustness perturbation tests on composition workflows.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the unified LIBS benchmark pipeline using command-line arguments.

    Parses CLI arguments, configures and validates resources, runs identification and/or composition benchmark phases (optionally running a robustness perturbation battery), writes output files, and prints summary status.

    Parameters:
        argv (Sequence[str] | None): Command-line arguments to parse; when None, the program's actual argv is used.

    Returns:
        int: Process exit code (0 on success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    from cflibs.benchmark import UnifiedBenchmarkRunner, load_default_datasets
    from cflibs.benchmark.dataset import TruthType

    _validate_paths(parser, args)

    runner = UnifiedBenchmarkRunner(
        db_path=args.db_path,
        basis_dir=args.basis_dir if args.basis_dir.exists() else None,
        quick=args.quick,
    )

    id_workflows = _normalize_workflow_list(args.id_workflows) or list(runner.id_registry.keys())
    composition_workflows = _normalize_workflow_list(args.composition_workflows) or list(
        runner.composition_registry.keys()
    )

    unknown_id = sorted(set(id_workflows) - set(runner.id_registry))
    if unknown_id:
        parser.error(f"Unknown identification workflow(s): {', '.join(unknown_id)}")
    unknown_composition = sorted(set(composition_workflows) - set(runner.composition_registry))
    if unknown_composition:
        parser.error(f"Unknown composition workflow(s): {', '.join(unknown_composition)}")

    _validate_basis_requirements(parser, id_workflows, args.basis_dir)

    datasets = load_default_datasets(args.data_dir, synthetic_corpus_path=args.synthetic_corpus)
    identification_datasets = _select_datasets(
        datasets,
        truth_types=(
            TruthType.ASSAY,
            TruthType.FORMULA_PROXY,
            TruthType.SYNTHETIC,
            TruthType.BLIND,
        ),
    )
    composition_datasets = _select_datasets(
        datasets,
        truth_types=(TruthType.ASSAY, TruthType.SYNTHETIC),
    )

    id_records = []
    id_selections = []
    composition_records = []
    composition_selections = []

    if args.sections in {"all", "id"}:
        id_records, id_selections = _run_identification_phase(
            parser,
            runner,
            identification_datasets,
            id_workflows,
            args.max_outer_folds,
        )

    if args.perturb and args.sections not in {"all", "composition"}:
        print(
            "--perturb is only supported with --sections all or --sections composition; skipping."
        )

    if args.sections in {"all", "composition"}:
        composition_records, composition_selections = _run_composition_phase(
            parser,
            runner,
            composition_datasets,
            id_workflows,
            composition_workflows,
            args.max_outer_folds,
        )

        if args.perturb:
            _run_perturbation_phase(
                runner,
                composition_datasets,
                composition_workflows,
                composition_selections,
            )

    outputs = runner.write_outputs(
        output_dir=args.output_dir,
        id_records=id_records,
        id_selections=id_selections,
        composition_records=composition_records,
        composition_selections=composition_selections,
    )

    print(f"Unified benchmark completed. Outputs written to {args.output_dir.resolve()}")
    for key in sorted(outputs):
        print(f"{key}: {outputs[key].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
