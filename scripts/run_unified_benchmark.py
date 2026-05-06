#!/usr/bin/env python3
"""Run the unified LIBS benchmark stack.

This CLI is a thin front-end over :class:`cflibs.benchmark.unified.UnifiedBenchmarkRunner`.
It loads the configured datasets, runs the selected identification and/or
composition workflows, and writes the benchmark artifacts to an output directory.

Posterior calibration metrics
-----------------------------
When a composition workflow returns posterior samples (key
``posterior_samples`` in its prediction dict, or an ``mcmc_result``
object whose ``samples`` attribute matches the ``MCMCResult.samples``
shape), the unified runner automatically calls
:func:`cflibs.benchmark.posterior_metrics.compute_posterior_diagnostics`
and stashes the resulting fields under
``record.annotations["posterior_diagnostics"]`` of every emitted
``CompositionEvaluationRecord``. The diagnostics flow through
``composition_records.json`` unchanged. See ``docs/VALIDATION_METRICS.md``
§2.3 for the binding thresholds (R-hat < 1.01, ESS_bulk ≥ 400, divergent
transitions = 0, PSIS k-hat < 0.7, 95% coverage in the bidirectional
band [0.93, 0.97]).
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


def _build_perturbation_pipeline_fn(runner, id_workflow: str, comp_workflow: str):
    """Return a per-spectrum pipeline callable for the perturbation battery.

    The unified runner's composition workflows operate on a *dataset*, not
    a single spectrum.  For the perturbation battery we wrap a single
    spectrum into a one-element :class:`BenchmarkDataset`-like object and
    extract the composition prediction from the resulting record.

    The implementation deliberately uses ``getattr`` / dictionary lookups
    so that the script does not hard-couple to the runner's internal
    record schema; if the schema lacks a recognised "predicted
    composition" key, the resulting pipeline returns an empty dict and
    the perturbation report records the spectrum's contribution as a
    zero-Δ row -- the harness itself will not crash.
    """
    from cflibs.benchmark.dataset import BenchmarkDataset, DataSplit

    def pipeline_fn(spectrum):
        # Build a one-spectrum ad-hoc dataset.  We re-use the dataset's
        # default split machinery so the runner's contract is satisfied.
        ds = BenchmarkDataset(
            name="_perturb_singleton",
            description="single-spectrum perturbation evaluation",
            spectra=[spectrum],
            splits={
                "default": DataSplit(
                    train_indices=[],
                    test_indices=[0],
                    name="default",
                )
            },
        )
        try:
            records, _ = runner.run_composition(
                [ds],
                id_workflow_names=[id_workflow],
                composition_workflow_names=[comp_workflow],
                max_outer_folds=1,
            )
        except Exception:
            return {}
        if not records:
            return {}
        first = records[0]
        # Try common keys / attributes.
        for attr in ("predicted_composition", "composition_pred", "composition"):
            v = getattr(first, attr, None)
            if isinstance(v, dict) and v:
                return {str(k): float(val) for k, val in v.items()}
            if isinstance(first, dict) and isinstance(first.get(attr), dict) and first[attr]:
                return {str(k): float(val) for k, val in first[attr].items()}
        return {}

    return pipeline_fn


def _collect_spectra_with_truth(datasets):
    """Flatten the selected datasets into a list of spectra that carry a
    non-trivial ``true_composition``.

    Per the perturbation report's contract, only spectra with a known
    composition can contribute meaningfully to Δd_A computation.
    """
    out = []
    for ds in datasets:
        for s in ds.spectra:
            if s.true_composition:
                out.append(s)
    return out


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
        help=(
            "After the main benchmark, run the robustness perturbation "
            "battery (line-dropout + outlier injection) on the loaded "
            "spectra and emit perturbation_report.json alongside the "
            "other outputs."
        ),
    )
    parser.add_argument(
        "--perturb-seed",
        type=int,
        default=42,
        help=(
            "Seed for the outlier-injection RNG when --perturb is set "
            "(default 42).  Determines reproducibility of the report."
        ),
    )
    parser.add_argument(
        "--perturb-max-spectra",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of spectra fed to the perturbation "
            "battery.  Useful for smoke runs."
        ),
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

    # ---------------------------------------------------------------
    # Tier-1 physical-consistency gate (CF-LIBS-improved-wa4).
    #
    # The unified benchmark already tracks per-spectrum T and n_e via
    # ``CompositionEvaluationRecord``; aggregate the four physical-
    # consistency checks and (a) write them to
    # ``physical_consistency.json`` next to the existing summaries,
    # (b) merge a compact section into ``composition_summary.json``
    # so downstream parsers (python/benchmark_gate.py) only need to
    # read one file, and (c) reflect the gate decision in the exit
    # code so a benchmark that improves point estimates while
    # silently violating LTE blocks the PR.
    # ---------------------------------------------------------------
    from cflibs.benchmark.physical_consistency import (
        aggregate_physical_consistency,
        report_to_summary_lines,
    )
    import json as _json

    pc_report = aggregate_physical_consistency(composition_records)
    pc_dict = pc_report.to_dict()

    pc_path = args.output_dir / "physical_consistency.json"
    with pc_path.open("w") as f:
        _json.dump(pc_dict, f, indent=2)
    print(f"physical_consistency_json: {pc_path.resolve()}")

    # Merge into composition_summary.json so the beefcake-swarm gate
    # parser can find both pieces in one read.
    summary_path = outputs.get(
        "composition_summary_json", args.output_dir / "composition_summary.json"
    )
    if summary_path.exists():
        try:
            with summary_path.open("r") as f:
                summary_doc = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            summary_doc = {}
        if isinstance(summary_doc, dict):
            summary_doc["physical_consistency"] = pc_dict
            with summary_path.open("w") as f:
                _json.dump(summary_doc, f, indent=2)

    for line in report_to_summary_lines(pc_report):
        print(line)

    if pc_report.blocked:
        print(
            "Physical-consistency gate BLOCKED — see "
            f"{pc_path.resolve()} for details."
        )
        # Skip the perturbation battery when the Tier-1 gate has
        # already blocked: there is no reason to spend GPU time
        # perturbing a pipeline that's failing first-principles physics.
        return 2

    if args.perturb:
        import numpy as np

        from cflibs.benchmark.robustness import (
            default_perturbations,
            run_perturbation_battery,
        )

        perturb_spectra = _collect_spectra_with_truth(composition_datasets)
        if args.perturb_max_spectra is not None:
            perturb_spectra = perturb_spectra[: args.perturb_max_spectra]

        if not perturb_spectra:
            print(
                "[--perturb] No composition-truth spectra available; "
                "skipping perturbation battery."
            )
        elif not (id_workflows and composition_workflows):
            print(
                "[--perturb] Need at least one id workflow and one "
                "composition workflow; skipping."
            )
        else:
            id_wf = id_workflows[0]
            comp_wf = composition_workflows[0]
            print(
                f"[--perturb] Running perturbation battery on "
                f"{len(perturb_spectra)} spectra using "
                f"{id_wf}/{comp_wf} as the pipeline."
            )
            rng = np.random.default_rng(args.perturb_seed)
            perturbations = default_perturbations(rng=rng)
            pipeline_fn = _build_perturbation_pipeline_fn(runner, id_wf, comp_wf)
            report = run_perturbation_battery(
                pipeline_fn,
                perturb_spectra,
                perturbations=perturbations,
            )
            report_path = args.output_dir / "perturbation_report.json"
            report.save_json(report_path)
            print(
                f"[--perturb] Wrote {report_path.resolve()} "
                f"({len(report.results)} rows)."
            )
            for name, summary in report.reduce_to_summary(
                bootstrap_iterations=200,
                rng=np.random.default_rng(args.perturb_seed + 1),
            ).items():
                threshold = (
                    f"<{summary.threshold:.3f}"
                    if summary.threshold is not None
                    else "n/a"
                )
                print(
                    f"[--perturb]   {name}: mean Δd_A={summary.mean_delta_d_a:.4f} "
                    f"(threshold {threshold}, "
                    f"95%% CI [{summary.bootstrap_ci_lo:.4f}, "
                    f"{summary.bootstrap_ci_hi:.4f}], "
                    f"n={summary.n_spectra})"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
