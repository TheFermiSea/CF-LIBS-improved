"""Campaign 1 Optuna driver: TPE study over the inversion knob space.

Storage is Optuna's journal-file backend (NFS-safe via
``JournalFileOpenLock``): every SLURM array task runs ``driver.py worker``
against the same study directory and self-serves trials until the STOP file
appears, the core-hour ledger cap is reached, the study hits its trial
target, or its own ``--max-trials`` quota is exhausted. There is no central
ask-tell process to keep alive — the journal IS the coordination point —
which replaces the design doc's long-running driver job with something
preemption-proof.

Commands::

    init    create the study directory: frozen manifest, study config,
            baseline evaluation (death-penalty reference), seed trials
    worker  consume trials (the SLURM array entry point; also local smoke)
    status  one-line study summary
    stop    drop the STOP file (kill switch)

Run from the repo root, e.g.::

    JAX_PLATFORMS=cpu PYTHONPATH=$PWD python scripts/campaign1/driver.py init \
        --study-dir output/campaign1/run1 --budget-core-hours 600
    JAX_PLATFORMS=cpu PYTHONPATH=$PWD python scripts/campaign1/driver.py worker \
        --study-dir output/campaign1/run1 --max-trials 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

_CAMPAIGN_DIR = Path(__file__).resolve().parent
if str(_CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(_CAMPAIGN_DIR))

import knob_space  # noqa: E402
import objective as objective_mod  # noqa: E402
import splits  # noqa: E402

STUDY_NAME = "campaign1-phaseA"
STOP_FILENAME = "STOP"
LEDGER_FILENAME = "ledger.jsonl"
CONFIG_FILENAME = "study_config.json"
BASELINE_FILENAME = "baseline.json"
JOURNAL_FILENAME = "journal.log"

DEFAULT_SEED = 20260610
DEFAULT_BUDGET_CORE_HOURS = 600.0
DEFAULT_TARGET_TRIALS = 800
DEFAULT_DATASETS = ",".join(splits.OPTIMIZATION_DATASETS)


# ---------------------------------------------------------------------------
# Storage / ledger / kill switch
# ---------------------------------------------------------------------------


def study_storage(study_dir: Path) -> JournalStorage:
    """NFS-safe journal storage (flock does not work reliably on NFS)."""
    path = str(Path(study_dir) / JOURNAL_FILENAME)
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def stop_requested(study_dir: Path) -> bool:
    return (Path(study_dir) / STOP_FILENAME).exists()


def append_ledger(study_dir: Path, entry: dict[str, Any]) -> None:
    """Append one ledger line (O_APPEND single write: safe for short lines on NFS)."""
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with open(Path(study_dir) / LEDGER_FILENAME, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def read_ledger(study_dir: Path) -> list[dict[str, Any]]:
    path = Path(study_dir) / LEDGER_FILENAME
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def core_hours_used(study_dir: Path) -> float:
    """Total core-hours charged so far (wall_s x cpus per entry)."""
    return sum(
        float(e.get("wall_s", 0.0)) * float(e.get("cpus", 1)) / 3600.0
        for e in read_ledger(study_dir)
    )


def budget_exceeded(study_dir: Path, budget_core_hours: float) -> bool:
    """Ledger enforcement: refuse new trials past the cap (design 6.4)."""
    return core_hours_used(study_dir) >= float(budget_core_hours)


def detect_cpus() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or 1


# ---------------------------------------------------------------------------
# Study config
# ---------------------------------------------------------------------------


def load_study_config(study_dir: Path) -> dict[str, Any]:
    return json.loads((Path(study_dir) / CONFIG_FILENAME).read_text())


def build_context(cfg: dict[str, Any], n_procs: int) -> objective_mod.EvalContext:
    manifest = splits.load_manifest(cfg["splits_manifest_path"])
    return objective_mod.EvalContext(
        db_path=Path(cfg["db_path"]),
        manifest=manifest,
        datasets=tuple(cfg["datasets"]),
        spectra_per_dataset=cfg["spectra_per_dataset"],
        sample_seed=cfg["sample_seed"],
        preset=cfg.get("preset"),
        n_procs=n_procs,
        per_spectrum_timeout_s=cfg.get("per_spectrum_timeout_s", 120.0),
    )


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    import cflibs

    print(f"cflibs={Path(cflibs.__file__).resolve().parent}")
    study_dir = Path(args.study_dir)
    if (study_dir / CONFIG_FILENAME).exists():
        raise SystemExit(f"{study_dir} already initialized (found {CONFIG_FILENAME}).")
    study_dir.mkdir(parents=True, exist_ok=True)

    manifest = splits.load_manifest(args.manifest)
    datasets = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    splits.assert_optimization_only(manifest, datasets)

    cfg = {
        "study_name": STUDY_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "sample_seed": args.seed,
        "datasets": list(datasets),
        "spectra_per_dataset": args.spectra_per_dataset,
        "db_path": str(Path(args.db).resolve()),
        "splits_manifest_path": str(Path(args.manifest).resolve()),
        "budget_core_hours": args.budget_core_hours,
        "target_trials": args.target_trials,
        "preset": args.preset,
        "per_spectrum_timeout_s": args.per_spectrum_timeout,
        "knob_space_version": knob_space.KNOB_SPACE_VERSION,
    }
    (study_dir / CONFIG_FILENAME).write_text(json.dumps(cfg, indent=2))

    knob_space.write_frozen_manifest(
        study_dir,
        splits_manifest=manifest,
        splits_manifest_path=cfg["splits_manifest_path"],
        db_path=cfg["db_path"],
        seed=args.seed,
        extra={"datasets": list(datasets), "spectra_per_dataset": args.spectra_per_dataset},
    )

    # Baseline evaluation: the death-penalty reference (FP / failure counts on
    # the EXACT spectrum sample every trial will see) + the incumbent record.
    ctx = build_context(cfg, n_procs=args.n_procs)
    print("Evaluating baseline (production defaults) for the death-penalty reference ...")
    t0 = time.perf_counter()
    baseline_overrides = knob_space.params_to_overrides(knob_space.baseline_params())
    board = objective_mod.evaluate_overrides(baseline_overrides, ctx)
    wall = time.perf_counter() - t0
    baseline_ref = objective_mod.baseline_reference_from_board(board)
    (study_dir / BASELINE_FILENAME).write_text(
        json.dumps({"reference": baseline_ref, "wall_s": wall}, indent=2)
    )
    (study_dir / "baseline_board.json").write_text(json.dumps(board, indent=2, default=str))
    append_ledger(
        study_dir, {"kind": "baseline", "wall_s": wall, "cpus": detect_cpus(), "worker": "init"}
    )

    storage = study_storage(study_dir)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True),
        load_if_exists=False,
    )
    for params in knob_space.seed_trial_params():
        study.enqueue_trial(params)
    print(
        f"Initialized study at {study_dir}: {len(knob_space.seed_trial_params())} seed "
        f"trials enqueued, baseline wall {wall:.1f}s, "
        f"budget {args.budget_core_hours} core-hours."
    )
    return 0


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


def make_objective(ctx, baseline_ref, study_dir: Path, cpus: int, worker_id: str):
    def _objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        try:
            params = knob_space.suggest_params(trial)
            overrides = knob_space.params_to_overrides(params)
            report, board = objective_mod.evaluate_candidate(overrides, ctx, baseline_ref)
        finally:
            wall = time.perf_counter() - t0
            append_ledger(
                study_dir,
                {
                    "kind": "trial",
                    "trial": trial.number,
                    "wall_s": wall,
                    "cpus": cpus,
                    "worker": worker_id,
                },
            )
        trial.set_user_attr("wall_s", wall)
        trial.set_user_attr("fitness_report", report.to_dict())
        trial.set_user_attr(
            "per_dataset_metrics",
            {
                row["name"]: {
                    "id_metrics": row["id_metrics"],
                    "composition": row["composition"],
                    "runtime": row["runtime"],
                    "n_run": row["n_run"],
                    "n_failed": row["n_failed"],
                    "failures": row["failures"],
                }
                for row in board["datasets"]
            },
        )
        return report.fitness

    return _objective


def _n_finished_trials(study: optuna.Study) -> int:
    """Terminal trials only. RUNNING is deliberately NOT counted: a killed
    worker leaves zombie RUNNING trials in the journal forever, and counting
    them could make the target unreachable. Concurrent workers may overshoot
    the target by at most the worker count — acceptable."""
    return len(
        study.get_trials(
            deepcopy=False,
            states=(
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.FAIL,
                optuna.trial.TrialState.PRUNED,
            ),
        )
    )


def cmd_worker(args: argparse.Namespace) -> int:
    import cflibs

    print(f"cflibs={Path(cflibs.__file__).resolve().parent}")
    study_dir = Path(args.study_dir)
    cfg = load_study_config(study_dir)
    cpus = args.cpus or detect_cpus()
    worker_id = str(args.worker_id)
    baseline_ref = json.loads((study_dir / BASELINE_FILENAME).read_text())["reference"]
    ctx = build_context(cfg, n_procs=args.n_procs)

    # Distinct sampler seed per worker (identical seeds would propose
    # identical startup points); non-numeric ids hash deterministically.
    worker_offset = (
        int(worker_id) if worker_id.isdigit() else zlib.crc32(worker_id.encode()) % 10000
    )
    sampler = optuna.samplers.TPESampler(
        seed=cfg["seed"] + worker_offset + 1, multivariate=True, group=True
    )
    study = optuna.load_study(
        study_name=cfg["study_name"], storage=study_storage(study_dir), sampler=sampler
    )
    objective_fn = make_objective(ctx, baseline_ref, study_dir, cpus, worker_id)

    done = 0
    while True:
        if stop_requested(study_dir):
            print(f"worker {worker_id}: STOP file present — halting.")
            break
        if budget_exceeded(study_dir, cfg["budget_core_hours"]):
            print(
                f"worker {worker_id}: core-hour ledger at "
                f"{core_hours_used(study_dir):.2f} >= cap {cfg['budget_core_hours']} — "
                "refusing new trials."
            )
            break
        if args.max_trials is not None and done >= args.max_trials:
            print(f"worker {worker_id}: max-trials quota ({args.max_trials}) reached.")
            break
        if _n_finished_trials(study) >= cfg["target_trials"]:
            print(f"worker {worker_id}: study target ({cfg['target_trials']} trials) reached.")
            break
        study.optimize(objective_fn, n_trials=1, catch=(Exception,))
        done += 1
    print(f"worker {worker_id}: exiting after {done} trial(s).")
    return 0


# ---------------------------------------------------------------------------
# status / stop
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    study_dir = Path(args.study_dir)
    cfg = load_study_config(study_dir)
    study = optuna.load_study(study_name=cfg["study_name"], storage=study_storage(study_dir))
    trials = study.get_trials(deepcopy=False)
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    best: Optional[optuna.trial.FrozenTrial] = None
    if complete:
        best = max(complete, key=lambda t: t.value)
    print(f"study: {cfg['study_name']} @ {study_dir}")
    print(
        f"trials: total={len(trials)} complete={len(complete)} "
        f"failed={sum(t.state == optuna.trial.TrialState.FAIL for t in trials)} "
        f"target={cfg['target_trials']}"
    )
    print(
        f"core-hours: {core_hours_used(study_dir):.2f} / {cfg['budget_core_hours']} "
        f"(STOP={'yes' if stop_requested(study_dir) else 'no'})"
    )
    if best is not None:
        print(f"best: trial #{best.number} fitness={best.value:.4f}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    path = Path(args.study_dir) / STOP_FILENAME
    path.touch()
    print(f"Created {path} — workers will halt before their next trial.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize a study directory")
    p_init.add_argument("--study-dir", required=True)
    p_init.add_argument("--manifest", default=str(splits.DEFAULT_MANIFEST_PATH))
    p_init.add_argument("--db", default="ASD_da/libs_production.db")
    p_init.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_init.add_argument(
        "--datasets",
        default=DEFAULT_DATASETS,
        help=f"Comma-separated optimization datasets (default: {DEFAULT_DATASETS})",
    )
    p_init.add_argument(
        "--spectra-per-dataset",
        type=int,
        default=74,
        help="Per-trial seeded sample cap per dataset (design 2.4; default 74)",
    )
    p_init.add_argument("--budget-core-hours", type=float, default=DEFAULT_BUDGET_CORE_HOURS)
    p_init.add_argument("--target-trials", type=int, default=DEFAULT_TARGET_TRIALS)
    p_init.add_argument("--preset", default=None, help="Pipeline preset (default geological)")
    p_init.add_argument("--per-spectrum-timeout", type=float, default=120.0)
    p_init.add_argument("--n-procs", type=int, default=1)

    p_worker = sub.add_parser("worker", help="Consume trials until STOP/budget/target")
    p_worker.add_argument("--study-dir", required=True)
    p_worker.add_argument("--max-trials", type=int, default=None)
    p_worker.add_argument("--worker-id", default=os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    p_worker.add_argument("--n-procs", type=int, default=1)
    p_worker.add_argument("--cpus", type=int, default=None, help="Cores to charge per wall-hour")

    p_status = sub.add_parser("status", help="Study summary")
    p_status.add_argument("--study-dir", required=True)

    p_stop = sub.add_parser("stop", help="Create the STOP file (kill switch)")
    p_stop.add_argument("--study-dir", required=True)

    args = parser.parse_args(argv)
    return {"init": cmd_init, "worker": cmd_worker, "status": cmd_status, "stop": cmd_stop}[
        args.command
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
