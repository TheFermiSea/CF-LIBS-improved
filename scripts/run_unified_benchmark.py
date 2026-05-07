#!/usr/bin/env python3
"""Run the unified LIBS benchmark stack.

This CLI is a thin front-end over :class:`cflibs.benchmark.unified.UnifiedBenchmarkRunner`.
It loads the configured datasets, runs the selected identification and/or
composition workflows, and writes the benchmark artifacts to an output directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

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
        parser.error("No identification-capable datasets were found in the selected data directory.")
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


def _build_parser() -> argparse.ArgumentParser:
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
        truth_types=(TruthType.ASSAY, TruthType.FORMULA_PROXY, TruthType.SYNTHETIC, TruthType.BLIND),
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
            from cflibs.benchmark.robustness import (
                line_dropout_perturbation,
                outlier_injection_perturbation,
                run_perturbation_battery,
            )
            print("\n--- Running Perturbation Battery ---")
            perturbations = {
                "line_dropout": lambda spec: line_dropout_perturbation(spec, top_n=3),
                "outlier_injection": lambda spec: outlier_injection_perturbation(
                    spec, fraction=0.05, sigma_mult=5.0
                ),
            }

            # Gather all spectra from composition datasets
            all_spectra = []
            for ds in composition_datasets:
                all_spectra.extend(ds.spectra)

            # We use the first composition workflow for the perturbation battery for simplicity
            # Or iterate through all of them. The requirement says "run_perturbation_battery(pipeline_fn, spectra, perturbations)"
            # Let's test the first configured workflow.
            for comp_wf in composition_workflows:
                # We need a pipeline fn that matches the signature (wavelengths, intensities, elements)
                # We can wrap the runner's composition registry
                runner.composition_registry[comp_wf]
                print(f"Running perturbations for {comp_wf}...")

                def make_pipeline_fn(wf_name):
                    def pipeline_fn(wavelengths, intensities, elements):
                        # In the real benchmark, the ID workflow provides the elements.
                        # We just pass the ground truth elements here.
                        from cflibs.inversion.element_id import ElementIdentificationResult
                        mock_id = ElementIdentificationResult(
                            identified_elements=set(elements),
                            score=1.0,
                            peak_matches=[]
                        )
                        # We also need a mock workflow instance
                        # For robustness tests, we can instantiate a fresh workflow with default params
                        wf_instance = runner.composition_registry[wf_name](db=runner.db)
                        res = wf_instance.estimate(
                            wavelengths,
                            intensities,
                            identification=mock_id,
                        )
                        return {"concentrations": res.concentrations}
                    return pipeline_fn

                report = run_perturbation_battery(
                    make_pipeline_fn(comp_wf),
                    all_spectra,
                    perturbations,
                    pipeline_name=comp_wf
                )

                print(f"Perturbation Report for {comp_wf}:")
                for p_name in perturbations:
                    deltas = report.delta_d_A[p_name].values()
                    valid_deltas = [d for d in deltas if d != float("inf")]
                    if valid_deltas:
                        avg_delta = sum(valid_deltas) / len(valid_deltas)
                        max_delta = max(valid_deltas)
                        print(f"  {p_name}: Avg delta_d_A = {avg_delta:.4f}, Max delta_d_A = {max_delta:.4f}")

                        # Tier-2 alarms
                        if p_name == "line_dropout" and avg_delta > 0.02:
                            print(f"  ALARM: {comp_wf} {p_name} delta_d_A ({avg_delta:.4f}) exceeds threshold 0.02!")
                        elif p_name == "outlier_injection" and avg_delta > 0.05:
                            print(f"  ALARM: {comp_wf} {p_name} delta_d_A ({avg_delta:.4f}) exceeds threshold 0.05!")
                    else:
                        print(f"  {p_name}: No valid delta_d_A values computed.")

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
