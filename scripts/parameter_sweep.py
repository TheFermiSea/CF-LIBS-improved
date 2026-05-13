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

Bandit allocation (T2.3, ``--bandit``)
--------------------------------------
With ``--cells <path-to-json>`` and ``--bandit <warmup_n>`` the sweep
treats each cell as the arm of a multi-armed bandit.  After
``warmup_n × n_cells`` round-robin warmup iters, the remaining
iters are routed to the arm with the best Thompson-sampled posterior
mean of d_A (composition Aitchison distance, lower-is-better).  See
``cflibs/bandit/thompson_allocator.py`` and
``docs/bandit-allocator-consultation.md`` for the design.

When ``--bandit 0`` (the default) or ``--cells`` is unset, behavior is
preserved byte-for-byte versus the T1.1 baseline.

Out of scope for T1.1 / T2.3
----------------------------
- ``--parallel`` execution of iterations (XLA device contention risk).
- Wiring ``--perturb`` through the sweep — orthogonal feature.
- GPU determinism — covered by a separate epic.
- Best-arm identification with PAC guarantees — out of scope for T2.3.
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


def _load_cells(cells_path: Path | None, fallback_config_args: str) -> list[dict[str, str]]:
    """Load cell descriptors or fall back to a single-cell sweep.

    Returns a list of ``{"name": <str>, "config_args": <str>}`` dicts.
    When ``cells_path`` is None, returns a single cell whose name is
    ``"default"`` and whose ``config_args`` is ``fallback_config_args``
    (i.e. preserves T1.1 behavior).
    """
    if cells_path is None:
        return [{"name": "default", "config_args": fallback_config_args}]

    with cells_path.open("r") as f:
        payload = json.load(f)

    cells: list[dict[str, str]] = []
    if isinstance(payload, list):
        for i, entry in enumerate(payload):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"--cells entry {i} is not an object: {entry!r}"
                )
            name = str(entry.get("name", f"cell{i}"))
            cfg = str(entry.get("config_args", ""))
            cells.append({"name": name, "config_args": cfg})
    elif isinstance(payload, dict):
        for name, cfg in payload.items():
            cells.append({"name": str(name), "config_args": str(cfg)})
    else:
        raise ValueError(
            f"--cells must be a JSON array or object, got {type(payload).__name__}"
        )
    if not cells:
        raise ValueError(f"--cells file {cells_path} is empty")
    return cells


def _extract_d_a(outputs: Mapping[str, Any], iter_dir: Path) -> float | None:
    """Extract the iter's d_A (mean Aitchison composition distance).

    Looks for the ``composition_summary.json`` produced by
    ``UnifiedBenchmarkRunner.write_outputs``.  The summary is keyed by
    ``id_workflow__composition_workflow`` strings; we average the
    ``mean_aitchison`` field across all such pairs.  Returns ``None``
    when no scored composition record exists.
    """
    summary_path = outputs.get("composition_summary_json")
    if not summary_path:
        summary_path = iter_dir / "composition_summary.json"
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict):
        return None
    # Strip non-pair top-level keys we may have written (e.g. physical_consistency).
    means: list[float] = []
    import math as _math
    for key, value in summary.items():
        if key == "physical_consistency" or not isinstance(value, dict):
            continue
        m = value.get("mean_aitchison")
        if m is None:
            continue
        try:
            m_f = float(m)
        except (TypeError, ValueError):
            continue
        if _math.isfinite(m_f):
            means.append(m_f)
    if not means:
        return None
    return float(sum(means) / len(means))


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
    parser.add_argument(
        "--cells",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSON file describing parameter-sweep cells. "
            "The file may be either (a) a JSON array of objects "
            "[{\"name\": <str>, \"config_args\": <str>}, ...] or (b) a "
            "JSON object {<name>: <config_args>, ...}.  When absent, the "
            "sweep runs a single cell using --config-args verbatim."
        ),
    )
    parser.add_argument(
        "--bandit",
        type=int,
        default=0,
        metavar="WARMUP_N",
        dest="bandit_warmup",
        help=(
            "Enable Thompson-sampling bandit allocation. WARMUP_N is the "
            "number of round-robin warmup pulls per cell. After warmup, "
            "remaining iters are routed to cells with the most promising "
            "d_A (composition Aitchison distance) trajectories. "
            "0 (default) preserves the existing equal-allocation behavior "
            "byte-for-byte. Requires >= 1 cell, but is only meaningful "
            "when used with --cells of length >= 2."
        ),
    )
    parser.add_argument(
        "--bandit-seed",
        type=int,
        default=None,
        help=(
            "Seed for the bandit allocator's internal RNG. If unset, "
            "derived deterministically from --seed-base so a sweep with "
            "the same --seed-base produces the same arm-pull schedule."
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
    # Match run_unified_benchmark.main()'s basis-dir resolution so the
    # runner doesn't crash on the default ``--basis-dir`` (None) path.
    base_args.basis_dir = _RUB._resolve_basis_dir(base_args.basis_dir)

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

    # When the caller didn't pass --cells, base_args drives the only
    # cell, so run the basis-requirement check on the base id_workflows.
    # When --cells is set the per-cell args may narrow id_workflows
    # below the basis-required set, so defer validation to the cell
    # parse below (each cell args.basis_dir is already resolved).
    if sweep_args.cells is None:
        _RUB._validate_basis_requirements(base_parser, id_workflows, base_args.basis_dir)

    _vrabel_cap = (
        None if base_args.vrabel_max_shots == 0 else base_args.vrabel_max_shots
    )
    # Pass-through --dataset-shard from --config-args. The base parser
    # has already validated the string format; here we just convert to
    # the tuple the loader expects (or None for the unsharded default).
    try:
        _shard_tuple = _RUB._parse_dataset_shard(base_args.dataset_shard)
    except ValueError as exc:
        base_parser.error(str(exc))
    _shard_for_loader = _shard_tuple if _shard_tuple != (1, 1) else None
    datasets = load_default_datasets(
        base_args.data_dir,
        synthetic_corpus_path=base_args.synthetic_corpus,
        vrabel_max_shots_per_sample=_vrabel_cap,
        dataset_shard=_shard_for_loader,
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

    # Cells + bandit setup.  --bandit 0 (default) without --cells
    # preserves byte-identical T1.1 behavior: a single "default" cell
    # with no arm_id / posterior fields in manifest records.
    cells = _load_cells(sweep_args.cells, sweep_args.config_args)
    bandit_warmup = max(0, int(sweep_args.bandit_warmup))
    bandit_enabled = bandit_warmup > 0
    # Per-cell args + per-cell workflow lists.  When --cells is unset
    # the single "default" cell reuses the already-validated base_args
    # verbatim so the --bandit 0 path is byte-identical with T1.1.
    per_cell_args: list[argparse.Namespace] = []
    per_cell_id_workflows: list[list[str]] = []
    per_cell_composition_workflows: list[list[str]] = []
    for cell in cells:
        if cells == [{"name": "default", "config_args": sweep_args.config_args}]:
            per_cell_args.append(base_args)
            per_cell_id_workflows.append(list(id_workflows))
            per_cell_composition_workflows.append(list(composition_workflows))
            continue
        cell_argv = shlex.split(cell["config_args"] or "")
        cell_ns = base_parser.parse_args(cell_argv)
        # Mirror the global jax-identifier env propagation done at the
        # top-level once per cell so cells that toggle --jax-identifier
        # still get the env var set correctly.
        if cell_ns.jax_identifier:
            os.environ["CFLIBS_USE_JAX_IDENTIFIER"] = "1"
        _RUB._validate_paths(base_parser, cell_ns)
        cell_ns.basis_dir = _RUB._resolve_basis_dir(cell_ns.basis_dir)
        cell_id_wf = _RUB._normalize_workflow_list(cell_ns.id_workflows) or list(
            runner.id_registry.keys()
        )
        cell_comp_wf = _RUB._normalize_workflow_list(cell_ns.composition_workflows) or list(
            runner.composition_registry.keys()
        )
        # Validate cell-narrowed workflows.
        unknown = sorted(set(cell_id_wf) - set(runner.id_registry))
        if unknown:
            base_parser.error(
                f"Cell '{cell['name']}': unknown identification workflow(s): "
                f"{', '.join(unknown)}"
            )
        unknown = sorted(set(cell_comp_wf) - set(runner.composition_registry))
        if unknown:
            base_parser.error(
                f"Cell '{cell['name']}': unknown composition workflow(s): "
                f"{', '.join(unknown)}"
            )
        _RUB._validate_basis_requirements(
            base_parser, cell_id_wf, cell_ns.basis_dir
        )
        per_cell_args.append(cell_ns)
        per_cell_id_workflows.append(cell_id_wf)
        per_cell_composition_workflows.append(cell_comp_wf)

    allocator = None
    warmup_schedule: list[int] = []
    if bandit_enabled:
        from cflibs.bandit import ThompsonAllocator
        from cflibs.bandit.thompson_allocator import round_robin_warmup_schedule

        bandit_seed = (
            int(sweep_args.bandit_seed)
            if sweep_args.bandit_seed is not None
            else seed_base
        )
        allocator = ThompsonAllocator(
            n_arms=len(cells),
            lower_is_better=True,  # d_A: lower is better
            random_state=bandit_seed,
        )
        warmup_schedule = round_robin_warmup_schedule(len(cells), bandit_warmup)
        print(
            f"[parameter_sweep] bandit enabled: n_arms={len(cells)} "
            f"warmup_per_arm={bandit_warmup} "
            f"warmup_total={len(warmup_schedule)} "
            f"bandit_seed={bandit_seed}",
            flush=True,
        )
    elif len(cells) > 1:
        print(
            f"[parameter_sweep] round-robin mode: n_arms={len(cells)} "
            f"n_iters={n_iters} -> each cell gets ~{n_iters // len(cells)} iter(s)",
            flush=True,
        )
    else:
        print(
            f"[parameter_sweep] single-cell mode: {n_iters} iters of cell 0",
            flush=True,
        )

    # Manifest is line-buffered + flushed after every write so a
    # mid-sweep crash still leaves a readable JSONL trail.
    with manifest_path.open("a", buffering=1) as manifest_fp:
        for iter_idx in range(n_iters):
            seed = seed_base + iter_idx

            # Pick which cell this iter pulls.
            # * Single-cell sweep (len(cells) == 1) always picks cell 0 —
            #   byte-identical to T1.1's pre-cells behavior.
            # * Multi-cell sweep with --bandit 0 round-robins through cells
            #   (CF-LIBS-improved-yfbg). The previous "always cell 0"
            #   behavior was catastrophic: a 25-iter / 5-cell --bandit 0
            #   sweep silently ran 25 iters of cell 0 only.
            # * Multi-cell sweep with --bandit N>0 uses warmup_schedule
            #   for the first N*len(cells) iters then the Thompson
            #   allocator.
            if not bandit_enabled:
                if len(cells) == 1:
                    arm_idx = 0
                else:
                    arm_idx = iter_idx % len(cells)
            elif iter_idx < len(warmup_schedule):
                arm_idx = warmup_schedule[iter_idx]
            else:
                assert allocator is not None
                arm_idx = allocator.select_arm()

            cell_info = cells[arm_idx]
            cell_name = cell_info["name"]
            cell_args = per_cell_args[arm_idx]
            cell_id_wf = per_cell_id_workflows[arm_idx]
            cell_comp_wf = per_cell_composition_workflows[arm_idx]

            iter_dir = base_output_dir / f"iter-{iter_idx:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            jax_key = _reseed(seed)

            iter_args = _copy_namespace(cell_args)
            iter_args.output_dir = iter_dir

            t_iter_start = time.perf_counter()
            iter_status = "ok"
            iter_error: str | None = None
            outputs: Mapping[str, Any] = {}
            try:
                outputs = _run_one_iteration(
                    base_parser=base_parser,
                    runner=runner,
                    id_workflows=cell_id_wf,
                    composition_workflows=cell_comp_wf,
                    identification_datasets=identification_datasets,
                    composition_datasets=composition_datasets,
                    base_args=iter_args,
                    iter_output_dir=iter_dir,
                    sections=cell_args.sections,
                    max_outer_folds=cell_args.max_outer_folds,
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

            # Update the bandit with this iter's d_A so future
            # select_arm() calls see the observation.
            iter_d_a: float | None = None
            if bandit_enabled and allocator is not None and iter_status != "error":
                iter_d_a = _extract_d_a(outputs, iter_dir)
                if iter_d_a is not None:
                    allocator.update(arm_idx, iter_d_a)

            record: dict[str, Any] = {
                "iter": iter_idx,
                "seed": seed,
                "status": iter_status,
                "wall_time_seconds": round(iter_seconds, 4),
                "output_dir": str(iter_dir),
                "config_args": sweep_args.config_args,
                "jax_key_used": jax_key is not None,
            }
            # Bandit-specific manifest fields are added only when the
            # bandit is enabled so single-cell --bandit 0 produces
            # byte-identical manifest lines with T1.1.
            #
            # For multi-cell --bandit 0 round-robin mode
            # (CF-LIBS-improved-yfbg), we still need cell_name + cell_id
            # so downstream tooling can tell which cell each iter ran.
            if bandit_enabled:
                record["arm_id"] = arm_idx
                record["cell_id"] = arm_idx
                record["cell_name"] = cell_name
                record["cell_config_args"] = cell_info["config_args"]
                record["phase"] = (
                    "warmup" if iter_idx < len(warmup_schedule) else "bandit"
                )
                record["d_a"] = iter_d_a
                if allocator is not None:
                    summary = allocator.posterior_summary(prob_best_samples=256)
                    arm_summary = summary[arm_idx]
                    record["posterior_mean"] = arm_summary["posterior_mean"]
                    record["posterior_var"] = arm_summary["posterior_var"]
                    record["prob_best"] = arm_summary["prob_best"]
                    record["n_pulls"] = arm_summary["n_pulls"]
                    record["arm_posteriors"] = summary
            elif len(cells) > 1:
                record["cell_id"] = arm_idx
                record["cell_name"] = cell_name
                record["cell_config_args"] = cell_info["config_args"]
                record["phase"] = "round_robin"
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

            extra = ""
            if bandit_enabled:
                phase = "warmup" if iter_idx < len(warmup_schedule) else "bandit"
                extra = (
                    f" arm={arm_idx} cell={cell_name} phase={phase} "
                    f"d_a={iter_d_a if iter_d_a is not None else 'NA'}"
                )
            print(
                f"[parameter_sweep] iter {iter_idx:03d} seed={seed} "
                f"status={iter_status} time={iter_seconds:.2f}s -> {iter_dir}"
                f"{extra}",
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

    if bandit_enabled and allocator is not None:
        final_summary = allocator.posterior_summary(prob_best_samples=512)
        pulls = allocator.n_pulls()
        print(
            "[parameter_sweep] bandit final allocation:",
            flush=True,
        )
        for cell, summary in zip(cells, final_summary):
            pb = summary["prob_best"]
            pb_s = f"{pb:.2%}" if pb is not None else "NA"
            obs_mean = summary["observed_mean"]
            obs_mean_s = (
                f"{obs_mean:.4f}"
                if obs_mean == obs_mean  # NaN check
                else "NA"
            )
            print(
                f"  cell={cell['name']:<24} pulls={summary['n_pulls']:>3} "
                f"observed_mean_d_a={obs_mean_s} "
                f"posterior_mean={summary['posterior_mean']:.4f} "
                f"prob_best={pb_s}",
                flush=True,
            )
        # Emit a machine-readable allocation summary next to the manifest.
        alloc_summary_path = sweep_args.output_dir / "bandit_summary.json"
        with alloc_summary_path.open("w") as f:
            json.dump(
                {
                    "cells": cells,
                    "pulls_per_cell": pulls,
                    "posterior_summary": final_summary,
                    "warmup_per_cell": bandit_warmup,
                    "total_iters": n_iters,
                },
                f,
                indent=2,
            )
        print(
            f"[parameter_sweep] bandit summary: {alloc_summary_path}",
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
