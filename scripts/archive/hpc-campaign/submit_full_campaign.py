#!/usr/bin/env python3
"""
Master orchestrator for the HPC benchmark campaign.

Submits the entire benchmark pipeline as a SLURM dependency chain:

  Job 1  (GPU array, 63 tasks)  generate_synthetic_benchmark.py chunk
  Job 2  (GPU array, 8 tasks)   generate_basis_libraries.py
  Job 3  (CPU single)           generate_synthetic_benchmark.py consolidate   [after Job 1]
  Job 4  (CPU array, 315 tasks) run_benchmark_sweep.py worker (coarse)        [after Job 1, Job 2]
  Job 5  (CPU single)           run_benchmark_sweep.py collect                [after Job 4]
  Job 6  (CPU array, <=100)     run_benchmark_sweep.py worker --fine          [after Job 5]
  Job 7  (CPU single)           run_benchmark_sweep.py collect --fine          [after Job 6]
  Job 8  (CPU single)           train_ml_classifier.py                        [after Job 5]
  Job 9  (CPU single)           analyze_benchmark_results.py                  [after Job 7, Job 8]
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import shlex
import sys
from pathlib import Path
from typing import List, NoReturn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Campaign parameter grid
# ---------------------------------------------------------------------------
RP_VALUES = [200, 300, 500, 700, 1000, 2000, 3000, 5000, 10000]
SNR_VALUES = [10, 20, 50, 100, 200, 500, 1000]
FWHM_VALUES = [0.05, 0.10, 0.17, 0.25, 0.50, 0.71, 1.00, 1.67]
PATHWAYS = [
    "alias",
    "spectral_nnls",
    "hybrid_intersect",
    "hybrid_union",
    "forward_model",
]

N_RP_SNR = len(RP_VALUES) * len(SNR_VALUES)  # 63
N_FWHM = len(FWHM_VALUES)  # 8
N_COARSE = N_RP_SNR * len(PATHWAYS)  # 315
N_FINE = 100

SCRIPTS_DIR = Path(__file__).resolve().parent  # scripts/hpc/


def _error_exit(message: str, code: int = 1) -> NoReturn:
    print(f"ERROR: {message}")
    sys.exit(code)


def _resolve_db_path(db_path_arg: str | None) -> Path:
    if db_path_arg is not None:
        resolved = Path(db_path_arg).expanduser().resolve()
        if not resolved.exists():
            _error_exit(f"Atomic database not found: {resolved}")
        return resolved

    candidates = [
        PROJECT_ROOT / "libs_production.db",
        PROJECT_ROOT / "ASD_da" / "libs_production.db",
        Path.cwd() / "libs_production.db",
        Path.cwd() / "ASD_da" / "libs_production.db",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    _error_exit(
        "Atomic database not found. Pass --db-path explicitly or place "
        "libs_production.db in the project root or ASD_da/."
    )


def _q(path: str | Path) -> str:
    return shlex.quote(str(path))


def submit_campaign(
    db_path: Path,
    output_dir: Path,
    partition_gpu: str,
    partition_cpu: str,
    max_concurrent: int,
    dry_run: bool,
    n_spectra_per_chunk: int,
) -> None:
    from cflibs.hpc.slurm import ArrayJobConfig, SlurmJobConfig, SlurmJobManager

    manager = SlurmJobManager(dry_run=dry_run)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    db = _q(db_path)
    out = _q(output_dir)

    jobs: List[dict] = []

    def _record(name: str, job_id: str, deps: List[str]) -> None:
        jobs.append({"name": name, "job_id": job_id, "deps": deps})

    # ------------------------------------------------------------------
    # Job 1: GPU array — generate synthetic benchmark chunks
    # ------------------------------------------------------------------
    job1_config = ArrayJobConfig(
        job_name="bench_synth",
        partition=partition_gpu,
        cpus_per_task=4,
        mem_gb=32,
        time_limit="04:00:00",
        output_path=str(log_dir / "synth_%a.out"),
        error_path=str(log_dir / "synth_%a.err"),
        extra_sbatch={"gres": "gpu:1"},
        env_vars={"JAX_PLATFORMS": "gpu"},
        array_size=N_RP_SNR,
        max_concurrent=max_concurrent,
    )
    job1_script = (
        f"python {_q(SCRIPTS_DIR / 'generate_synthetic_benchmark.py')} chunk \\\n"
        f"    --db-path {db} \\\n"
        f"    --output-dir {out} \\\n"
        f"    --n-spectra-per-chunk {n_spectra_per_chunk}"
    )
    job1_id = manager.submit(job1_config, job1_script)
    _record("Job 1: generate_synthetic_benchmark chunk (GPU array)", job1_id, [])

    # ------------------------------------------------------------------
    # Job 2: GPU array — generate basis libraries (runs in parallel w/ Job 1)
    # ------------------------------------------------------------------
    job2_config = ArrayJobConfig(
        job_name="bench_basis",
        partition=partition_gpu,
        cpus_per_task=4,
        mem_gb=32,
        time_limit="04:00:00",
        output_path=str(log_dir / "basis_%a.out"),
        error_path=str(log_dir / "basis_%a.err"),
        extra_sbatch={"gres": "gpu:1"},
        env_vars={"JAX_PLATFORMS": "gpu"},
        array_size=N_FWHM,
        max_concurrent=N_FWHM,
    )
    fwhm_args = " ".join(str(f) for f in FWHM_VALUES)
    job2_script = (
        f"python {_q(SCRIPTS_DIR / 'generate_basis_libraries.py')} \\\n"
        f"    --db-path {db} \\\n"
        f"    --output-dir {_q(output_dir / 'basis_libraries')} \\\n"
        f"    --fwhm {fwhm_args}"
    )
    job2_id = manager.submit(job2_config, job2_script)
    _record("Job 2: generate_basis_libraries (GPU array)", job2_id, [])

    # ------------------------------------------------------------------
    # Job 3: CPU single — consolidate synthetic benchmark  [after Job 1]
    # ------------------------------------------------------------------
    job3_config = SlurmJobConfig(
        job_name="bench_consol",
        partition=partition_cpu,
        cpus_per_task=40,
        mem_gb=128,
        time_limit="04:00:00",
        output_path=str(log_dir / "consolidate.out"),
        error_path=str(log_dir / "consolidate.err"),
        env_vars={"JAX_PLATFORMS": "cpu"},
    )
    job3_script = (
        f"python {_q(SCRIPTS_DIR / 'generate_synthetic_benchmark.py')} consolidate \\\n"
        f"    --output-dir {out}"
    )
    job3_id = manager.submit_with_dependency(
        job3_config, job3_script, depends_on=[job1_id], dependency_type="afterok"
    )
    _record("Job 3: consolidate synthetic benchmark (CPU)", job3_id, [job1_id])

    # ------------------------------------------------------------------
    # Job 4: CPU array — coarse benchmark sweep  [after Job 1, Job 2]
    # ------------------------------------------------------------------
    job4_config = ArrayJobConfig(
        job_name="bench_coarse",
        partition=partition_cpu,
        cpus_per_task=4,
        mem_gb=16,
        time_limit="04:00:00",
        output_path=str(log_dir / "coarse_%a.out"),
        error_path=str(log_dir / "coarse_%a.err"),
        env_vars={"JAX_PLATFORMS": "cpu"},
        array_size=N_COARSE,
        max_concurrent=max_concurrent,
    )
    job4_script = (
        f"python {_q(SCRIPTS_DIR / 'run_benchmark_sweep.py')} worker \\\n"
        f"    --db-path {db} \\\n"
        f"    --output-dir {out}"
    )
    job4_id = manager.submit_with_dependency(
        job4_config, job4_script, depends_on=[job1_id, job2_id], dependency_type="afterok"
    )
    _record("Job 4: coarse benchmark sweep (CPU array)", job4_id, [job1_id, job2_id])

    # ------------------------------------------------------------------
    # Job 5: CPU single — collect coarse results  [after Job 4]
    # ------------------------------------------------------------------
    job5_config = SlurmJobConfig(
        job_name="bench_collect",
        partition=partition_cpu,
        cpus_per_task=40,
        mem_gb=128,
        time_limit="04:00:00",
        output_path=str(log_dir / "collect_coarse.out"),
        error_path=str(log_dir / "collect_coarse.err"),
        env_vars={"JAX_PLATFORMS": "cpu"},
    )
    job5_script = (
        f"python {_q(SCRIPTS_DIR / 'run_benchmark_sweep.py')} collect \\\n"
        f"    --output-dir {out}"
    )
    job5_id = manager.submit_with_dependency(
        job5_config, job5_script, depends_on=[job4_id], dependency_type="afterok"
    )
    _record("Job 5: collect coarse results (CPU)", job5_id, [job4_id])

    # ------------------------------------------------------------------
    # Job 6: CPU array — fine sweep on 100K subset  [after Job 5]
    # ------------------------------------------------------------------
    job6_config = ArrayJobConfig(
        job_name="bench_fine",
        partition=partition_cpu,
        cpus_per_task=4,
        mem_gb=16,
        time_limit="06:00:00",
        output_path=str(log_dir / "fine_%a.out"),
        error_path=str(log_dir / "fine_%a.err"),
        env_vars={"JAX_PLATFORMS": "cpu"},
        array_size=N_FINE,
        max_concurrent=max_concurrent,
    )
    job6_script = (
        f"python {_q(SCRIPTS_DIR / 'run_benchmark_sweep.py')} worker --fine \\\n"
        f"    --db-path {db} \\\n"
        f"    --output-dir {out}"
    )
    job6_id = manager.submit_with_dependency(
        job6_config, job6_script, depends_on=[job5_id], dependency_type="afterok"
    )
    _record("Job 6: fine benchmark sweep (CPU array)", job6_id, [job5_id])

    # ------------------------------------------------------------------
    # Job 7: CPU single — collect fine results  [after Job 6]
    # ------------------------------------------------------------------
    job7_config = SlurmJobConfig(
        job_name="bench_fine_col",
        partition=partition_cpu,
        cpus_per_task=40,
        mem_gb=128,
        time_limit="04:00:00",
        output_path=str(log_dir / "collect_fine.out"),
        error_path=str(log_dir / "collect_fine.err"),
        env_vars={"JAX_PLATFORMS": "cpu"},
    )
    job7_script = (
        f"python {_q(SCRIPTS_DIR / 'run_benchmark_sweep.py')} collect --fine \\\n"
        f"    --output-dir {out}"
    )
    job7_id = manager.submit_with_dependency(
        job7_config, job7_script, depends_on=[job6_id], dependency_type="afterok"
    )
    _record("Job 7: collect fine results (CPU)", job7_id, [job6_id])

    # ------------------------------------------------------------------
    # Job 8: CPU single — train ML classifier  [after Job 5]
    # ------------------------------------------------------------------
    job8_config = SlurmJobConfig(
        job_name="bench_ml",
        partition=partition_cpu,
        cpus_per_task=40,
        mem_gb=128,
        time_limit="04:00:00",
        output_path=str(log_dir / "train_ml.out"),
        error_path=str(log_dir / "train_ml.err"),
        env_vars={"JAX_PLATFORMS": "cpu"},
    )
    job8_script = (
        f"python {_q(SCRIPTS_DIR / 'train_ml_classifier.py')} \\\n"
        f"    --db-path {db} \\\n"
        f"    --output-dir {out}"
    )
    job8_id = manager.submit_with_dependency(
        job8_config, job8_script, depends_on=[job5_id], dependency_type="afterok"
    )
    _record("Job 8: train ML classifier (CPU)", job8_id, [job5_id])

    # ------------------------------------------------------------------
    # Job 9: CPU analysis — final analysis  [after Job 7, Job 8]
    # ------------------------------------------------------------------
    job9_config = SlurmJobConfig(
        job_name="bench_analyze",
        partition=partition_cpu,
        cpus_per_task=10,
        mem_gb=64,
        time_limit="02:00:00",
        output_path=str(log_dir / "analyze.out"),
        error_path=str(log_dir / "analyze.err"),
        env_vars={"JAX_PLATFORMS": "cpu"},
    )
    job9_script = (
        f"python {_q(SCRIPTS_DIR / 'analyze_benchmark_results.py')} \\\n" f"    --output-dir {out}"
    )
    job9_id = manager.submit_with_dependency(
        job9_config, job9_script, depends_on=[job7_id, job8_id], dependency_type="afterok"
    )
    _record("Job 9: analyze benchmark results (CPU)", job9_id, [job7_id, job8_id])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("HPC BENCHMARK CAMPAIGN SUBMITTED")
    print("=" * 72)
    print(f"Output directory : {output_dir}")
    print(f"Database         : {db_path}")
    print(f"GPU partition    : {partition_gpu}")
    print(f"CPU partition    : {partition_cpu}")
    print(f"Max concurrent   : {max_concurrent}")
    if dry_run:
        print("Mode             : DRY RUN (no jobs submitted)")
    print()

    header = f"{'Step':<55s} {'Job ID':<25s} {'Depends On'}"
    print(header)
    print("-" * len(header))
    for entry in jobs:
        dep_str = ", ".join(entry["deps"]) if entry["deps"] else "(none)"
        print(f"{entry['name']:<55s} {entry['job_id']:<25s} {dep_str}")

    print()
    print("Monitor with: squeue -u $USER")
    print(f"Check logs in: {log_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the full HPC benchmark campaign as a SLURM dependency chain"
    )
    parser.add_argument("--db-path", type=str, default=None, help="Atomic database path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/hpc_benchmark",
        help="Root output directory for all benchmark artifacts",
    )
    parser.add_argument(
        "--partition-gpu", type=str, default="gpu", help="SLURM partition for GPU jobs"
    )
    parser.add_argument(
        "--partition-cpu", type=str, default="compute", help="SLURM partition for CPU jobs"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=20, help="Maximum concurrent array tasks"
    )
    parser.add_argument(
        "--n-spectra-per-chunk",
        type=int,
        default=16000,
        help="Spectra per synthetic benchmark chunk",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SBATCH scripts without submitting",
    )

    args = parser.parse_args()

    db_path = _resolve_db_path(args.db_path)
    output_dir = Path(args.output_dir).resolve()

    submit_campaign(
        db_path=db_path,
        output_dir=output_dir,
        partition_gpu=args.partition_gpu,
        partition_cpu=args.partition_cpu,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
        n_spectra_per_chunk=args.n_spectra_per_chunk,
    )


if __name__ == "__main__":
    main()
