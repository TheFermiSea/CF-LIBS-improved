#!/usr/bin/env python3
"""Run N seed iterations of the unified LIBS benchmark in a single process.

This script is a *seed-sweep* front-end over the same machinery used by
:mod:`scripts.run_unified_benchmark`.  Where ``run_unified_benchmark.py``
pays the JAX import + JIT-compile + HDF5-load cold-start cost once per
invocation (~90 s observed locally on ai-proxy), ``parameter_sweep.py``
pays it **once** and then runs N seed iterations against the cached
in-memory state.

Wall-time savings vs. ``N × run_unified_benchmark.py`` are typically
4-6× for small per-iter work (Vrabel-max-shots=1 smoke runs).  Larger
real workloads shift the ratio toward 1× as the per-iter compute term
dominates.

Contract
--------
Iter *i* with ``--seed-base S`` reseeds ``random``, ``numpy.random``,
and (when available) ``jax.random``'s global key with ``S + i`` before
calling the benchmark workflows.  Output is written to
``<output-dir>/iter-<NNN>/`` with the same file layout
``run_unified_benchmark.py`` would produce for a single run, plus a
top-level ``manifest.jsonl`` at ``<output-dir>/manifest.jsonl``.

Iter 0 with seed *S* MUST match a standalone ``run_unified_benchmark``
invocation seeded with *S* within ``rtol=1e-5, atol=1e-8`` on all
numeric outputs (see ``tests/scripts/test_parameter_sweep.py``).

CLI
---
::

    scripts/parameter_sweep.py \\
        --config-args "--quick --max-outer-folds 1 --sections all \\
                       --id-workflows alias --composition-workflows iterative_jax \\
                       --vrabel-max-shots 10 --jax-identifier" \\
        --n-iters 8 \\
        --seed-base 1 \\
        --output-dir /tmp/sweep-out

``--config-args`` is a single quoted string passed verbatim to
``run_unified_benchmark._build_parser().parse_args(shlex.split(...))``
so every flag accepted by that CLI is accepted here.

Out of scope for T1.1
---------------------
- ``--parallel`` execution of iterations (XLA device contention risk).
- Wiring ``--perturb`` through the sweep — orthogonal feature.
- GPU determinism — covered by a separate epic.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import random as _py_random
import shlex
import sys
import time
import traceback
from copy import copy as _copy_namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

# Force JAX-CPU default before any cflibs import wakes JAX up.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-use the existing CLI parser + helpers from run_unified_benchmark
# verbatim.  These are stable module-level symbols; importing them
# avoids drift between the one-shot and sweep code paths.
_RUB = importlib.import_module("scripts.run_unified_benchmark")


def _reseed(seed: int) -> Any | None:
    """Reseed numpy, python's ``random``, and (if available) JAX.

    Returns the new ``jax.random.PRNGKey(seed)`` when JAX is importable,
    otherwise ``None``.  The returned key is informational; the workflows
    inside cflibs that need JAX randomness already accept explicit seeds
    via their config dicts (see e.g. ``IterativeCFLIBSSolverJax``).
    Reseeding the global JAX key only matters for any ad-hoc
    ``jax.random.normal(jax.random.PRNGKey(...))`` call paths.
    """
    import numpy as _np

    _py_random.seed(seed)
    _np.random.seed(seed)
    try:
        import jax  # noqa: WPS433  — local import keeps the script JAX-optional
    except Exception:  # noqa: BLE001 — JAX optional; never let this kill the sweep
        return None
    return jax.random.PRNGKey(seed)


def _build_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run N seed iterations of the unified LIBS benchmark in a single "
            "process. Loads datasets and the runner ONCE; per-iter reseeds "
            "RNGs and writes results to iter-NNN/ subdirs."
        )
    )
    parser.add_argument(
        "--config-args",
        type=str,
        default="",
        help=(
            "Quoted string of flags to forward to run_unified_benchmark.py. "
            "Parsed with shlex.split. Example: --config-args \"--quick "
            "--id-workflows alias --composition-workflows iterative_jax\"."
        ),
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=8,
        help="Number of seed iterations to run (default: 8).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1,
        help=(
            "Base seed; per-iter seed is seed_base + iter_index (default: 1)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Top-level output directory. Per-iter outputs are written to "
            "<output-dir>/iter-<NNN>/. The manifest is "
            "<output-dir>/manifest.jsonl."
        ),
    )
    return parser


def _validate_and_normalize_base_args(
    parser: argparse.ArgumentParser,
    base_args: argparse.Namespace,
) -> tuple[list[str], list[str]]:
    """Run the prelude validation that ``run_unified_benchmark.main()`` does.

    Returns the normalized (id_workflows, composition_workflows) lists.
    Calls ``parser.error`` for unrecoverable problems.  The ``parser``
    argument is the *base* (run_unified_benchmark) parser so that error
    messages match the one-shot CLI verbatim.
    """
    if base_args.jax_identifier:
        os.environ["CFLIBS_USE_JAX_IDENTIFIER"] = "1"

    _RUB._validate_paths(parser, base_args)
    return [], []  # workflow normalization happens later (after runner is built)


def _select_dataset_partitions(
    parser: argparse.ArgumentParser,
    datasets: Mapping[str, Any],
    truth_types_module,
) -> tuple[list[Any], list[Any]]:
    """Mirror the ``main()`` block that partitions datasets by truth type."""
    identification_datasets = _RUB._select_datasets(
        datasets,
        truth_types=(
            truth_types_module.ASSAY,
            truth_types_module.FORMULA_PROXY,
            truth_types_module.SYNTHETIC,
            truth_types_module.BLIND,
        ),
    )
    composition_datasets = _RUB._select_datasets(
        datasets,
        truth_types=(truth_types_module.ASSAY, truth_types_module.SYNTHETIC),
    )
    return identification_datasets, composition_datasets


def _run_one_iteration(
    *,
    base_parser: argparse.ArgumentParser,
    runner,
    id_workflows: Sequence[str],
    composition_workflows: Sequence[str],
    identification_datasets: Sequence[Any],
    composition_datasets: Sequence[Any],
    base_args: argparse.Namespace,
    iter_output_dir: Path,
    sections: str,
    max_outer_folds: int | None,
) -> Mapping[str, Path]:
    """Run ID + composition phases and write outputs for one iteration.

    Mirrors the body of ``run_unified_benchmark.main()`` from the
    ``if args.sections in {"all", "id"}:`` block through the
    physical-consistency gate.  The perturbation battery is intentionally
    NOT invoked here — it is orthogonal to seed iteration and gated on a
    flag we do not currently forward.
    """
    id_records: list[Any] = []
    id_selections: list[Any] = []
    composition_records: list[Any] = []
    composition_selections: list[Any] = []

    if sections in {"all", "id"}:
        id_records, id_selections = _RUB._run_identification_phase(
            base_parser,
            runner,
            identification_datasets,
            id_workflows,
            max_outer_folds,
        )

    if sections in {"all", "composition"}:
        composition_records, composition_selections = _RUB._run_composition_phase(
            base_parser,
            runner,
            composition_datasets,
            id_workflows,
            composition_workflows,
            max_outer_folds,
        )

    outputs = runner.write_outputs(
        output_dir=iter_output_dir,
        id_records=id_records,
        id_selections=id_selections,
        composition_records=composition_records,
        composition_selections=composition_selections,
    )

    # Physical-consistency Tier-1 gate — mirror run_unified_benchmark.main().
    from cflibs.benchmark.physical_consistency import (
        aggregate_physical_consistency,
    )

    pc_report = aggregate_physical_consistency(composition_records)
    pc_dict = pc_report.to_dict()
    pc_path = iter_output_dir / "physical_consistency.json"
    with pc_path.open("w") as f:
        json.dump(pc_dict, f, indent=2)

    summary_path = outputs.get(
        "composition_summary_json", iter_output_dir / "composition_summary.json"
    )
    if summary_path.exists():
        try:
            with summary_path.open("r") as f:
                summary_doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            summary_doc = {}
        if isinstance(summary_doc, dict):
            summary_doc["physical_consistency"] = pc_dict
            with summary_path.open("w") as f:
                json.dump(summary_doc, f, indent=2)

    return {
        **outputs,
        "physical_consistency_json": pc_path,
        "_pc_blocked": pc_report.blocked,
    }


def main(argv: Sequence[str] | None = None) -> int:
    sweep_parser = _build_sweep_parser()
    sweep_args = sweep_parser.parse_args(argv)

    if sweep_args.n_iters < 1:
        sweep_parser.error("--n-iters must be >= 1")

    # Parse the forwarded run_unified_benchmark flags.
    base_parser = _RUB._build_parser()
    base_args = base_parser.parse_args(shlex.split(sweep_args.config_args or ""))

    _validate_and_normalize_base_args(base_parser, base_args)

    sweep_args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_args.output_dir / "manifest.jsonl"

    # Import cflibs lazily so JAX_PLATFORMS / CFLIBS_USE_JAX_IDENTIFIER
    # are set before JAX boots.
    from cflibs.benchmark import UnifiedBenchmarkRunner, load_default_datasets
    from cflibs.benchmark.dataset import TruthType

    t_setup_start = time.perf_counter()

    runner = UnifiedBenchmarkRunner(
        db_path=base_args.db_path,
        basis_dir=base_args.basis_dir if base_args.basis_dir.exists() else None,
        quick=base_args.quick,
    )

    id_workflows = _RUB._normalize_workflow_list(base_args.id_workflows) or list(
        runner.id_registry.keys()
    )
    composition_workflows = _RUB._normalize_workflow_list(
        base_args.composition_workflows
    ) or list(runner.composition_registry.keys())

    unknown_id = sorted(set(id_workflows) - set(runner.id_registry))
    if unknown_id:
        base_parser.error(
            f"Unknown identification workflow(s): {', '.join(unknown_id)}"
        )
    unknown_composition = sorted(
        set(composition_workflows) - set(runner.composition_registry)
    )
    if unknown_composition:
        base_parser.error(
            f"Unknown composition workflow(s): {', '.join(unknown_composition)}"
        )

    _RUB._validate_basis_requirements(base_parser, id_workflows, base_args.basis_dir)

    _vrabel_cap = (
        None if base_args.vrabel_max_shots == 0 else base_args.vrabel_max_shots
    )
    datasets = load_default_datasets(
        base_args.data_dir,
        synthetic_corpus_path=base_args.synthetic_corpus,
        vrabel_max_shots_per_sample=_vrabel_cap,
    )
    identification_datasets, composition_datasets = _select_dataset_partitions(
        base_parser, datasets, TruthType
    )

    setup_seconds = time.perf_counter() - t_setup_start
    print(
        f"[parameter_sweep] setup complete in {setup_seconds:.2f}s "
        f"(loaded {len(datasets)} datasets, "
        f"{len(identification_datasets)} for id, "
        f"{len(composition_datasets)} for composition)",
        flush=True,
    )

    n_iters = int(sweep_args.n_iters)
    seed_base = int(sweep_args.seed_base)
    base_output_dir = sweep_args.output_dir.resolve()

    pc_blocked_iters: list[int] = []
    iter_records: list[dict[str, Any]] = []

    # Manifest is line-buffered + flushed after every write so a
    # mid-sweep crash still leaves a readable JSONL trail.
    with manifest_path.open("a", buffering=1) as manifest_fp:
        for iter_idx in range(n_iters):
            seed = seed_base + iter_idx
            iter_dir = base_output_dir / f"iter-{iter_idx:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            jax_key = _reseed(seed)

            iter_args = _copy_namespace(base_args)
            iter_args.output_dir = iter_dir

            t_iter_start = time.perf_counter()
            iter_status = "ok"
            iter_error: str | None = None
            outputs: Mapping[str, Any] = {}
            try:
                outputs = _run_one_iteration(
                    base_parser=base_parser,
                    runner=runner,
                    id_workflows=id_workflows,
                    composition_workflows=composition_workflows,
                    identification_datasets=identification_datasets,
                    composition_datasets=composition_datasets,
                    base_args=iter_args,
                    iter_output_dir=iter_dir,
                    sections=base_args.sections,
                    max_outer_folds=base_args.max_outer_folds,
                )
                if outputs.get("_pc_blocked"):
                    iter_status = "physical_consistency_blocked"
                    pc_blocked_iters.append(iter_idx)
            except SystemExit as exc:
                # base_parser.error() raises SystemExit; surface it as a
                # per-iter failure rather than killing the sweep.
                iter_status = "error"
                iter_error = f"SystemExit({exc.code}): parser error"
            except Exception as exc:  # noqa: BLE001 — iter errors must not kill the sweep
                iter_status = "error"
                iter_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

            iter_seconds = time.perf_counter() - t_iter_start

            record: dict[str, Any] = {
                "iter": iter_idx,
                "seed": seed,
                "status": iter_status,
                "wall_time_seconds": round(iter_seconds, 4),
                "output_dir": str(iter_dir),
                "config_args": sweep_args.config_args,
                "jax_key_used": jax_key is not None,
            }
            if iter_error is not None:
                record["error"] = iter_error
            if outputs:
                record["outputs"] = {
                    k: str(v)
                    for k, v in outputs.items()
                    if not k.startswith("_") and isinstance(v, Path)
                }

            manifest_fp.write(json.dumps(record) + "\n")
            manifest_fp.flush()
            with contextlib.suppress(OSError):
                os.fsync(manifest_fp.fileno())
            iter_records.append(record)

            print(
                f"[parameter_sweep] iter {iter_idx:03d} seed={seed} "
                f"status={iter_status} time={iter_seconds:.2f}s -> {iter_dir}",
                flush=True,
            )

    total_seconds = sum(rec["wall_time_seconds"] for rec in iter_records) + setup_seconds
    print(
        f"[parameter_sweep] done: {n_iters} iters, "
        f"setup={setup_seconds:.2f}s, "
        f"total≈{total_seconds:.2f}s, "
        f"manifest={manifest_path.resolve()}",
        flush=True,
    )

    # Exit code policy: 2 if ANY iteration was blocked by the
    # physical-consistency gate (matches run_unified_benchmark.py
    # exit code 2 for the same reason).
    if pc_blocked_iters:
        print(
            f"[parameter_sweep] PHYSICAL_CONSISTENCY_BLOCKED iters="
            f"{pc_blocked_iters}",
            flush=True,
        )
        return 2

    # Exit code 1 if every iteration errored (run-time failures).
    if iter_records and all(rec["status"] == "error" for rec in iter_records):
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
