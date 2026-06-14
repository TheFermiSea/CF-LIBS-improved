#!/usr/bin/env python3
"""
Generate basis libraries at multiple FWHM values for the HPC benchmark campaign.

Each FWHM produces a separate HDF5 basis library spanning a (T, n_e) grid.
Supports local sequential execution or SLURM array job submission.
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

FWHM_VALUES = [0.05, 0.10, 0.17, 0.25, 0.50, 0.71, 1.00, 1.67]


def _error_exit(message: str, code: int = 1) -> NoReturn:
    print(f"ERROR: {message}")
    sys.exit(code)


def _resolve_db_path(db_path_arg: str | None) -> Path:
    """Resolve the atomic database path."""
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


def _output_filename(fwhm: float) -> str:
    """Consistent filename for a given FWHM value."""
    return f"basis_library_fwhm_{fwhm:.2f}.h5"


def generate_single(
    db_path: Path,
    output_dir: Path,
    fwhm: float,
    t_steps: int,
    ne_steps: int,
) -> None:
    """Generate a single basis library at the given FWHM."""
    from cflibs.manifold.basis_library import BasisLibraryConfig, BasisLibraryGenerator

    output_file = output_dir / _output_filename(fwhm)
    if output_file.exists():
        print(f"Skipping FWHM={fwhm:.2f} nm — {output_file} already exists")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating basis library: FWHM={fwhm:.2f} nm, T_steps={t_steps}, ne_steps={ne_steps}")

    config = BasisLibraryConfig(
        db_path=str(db_path),
        output_path=str(output_file),
        wavelength_range=(200.0, 900.0),
        pixels=4096,
        temperature_range=(4000.0, 15000.0),
        temperature_steps=t_steps,
        density_range=(1e15, 5e17),
        density_steps=ne_steps,
        ionization_stages=(1, 2),
        instrument_fwhm_nm=fwhm,
    )

    generator = BasisLibraryGenerator(config)
    result_path = generator.generate(
        progress_callback=lambda done, total: print(
            f"  [{done}/{total}] elements complete", flush=True
        )
    )
    print(f"Saved: {result_path}")


def run_local(
    db_path: Path,
    output_dir: Path,
    fwhm_values: List[float],
    t_steps: int,
    ne_steps: int,
) -> None:
    """Generate basis libraries sequentially for all FWHM values."""
    for i, fwhm in enumerate(fwhm_values):
        print(f"\n=== FWHM {i + 1}/{len(fwhm_values)}: {fwhm:.2f} nm ===")
        generate_single(db_path, output_dir, fwhm, t_steps, ne_steps)
    print(f"\nDone. {len(fwhm_values)} basis libraries in {output_dir}")


def run_slurm_task(
    db_path: Path,
    output_dir: Path,
    fwhm_values: List[float],
    t_steps: int,
    ne_steps: int,
) -> None:
    """Run a single SLURM array task (selected by SLURM_ARRAY_TASK_ID)."""
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_str is None:
        _error_exit("SLURM_ARRAY_TASK_ID not set. Use --submit to launch via SLURM.")

    task_id = int(task_id_str)
    if task_id < 0 or task_id >= len(fwhm_values):
        _error_exit(f"SLURM_ARRAY_TASK_ID={task_id} out of range [0, {len(fwhm_values)})")

    fwhm = fwhm_values[task_id]
    generate_single(db_path, output_dir, fwhm, t_steps, ne_steps)


def submit_slurm(
    db_path: Path,
    output_dir: Path,
    fwhm_values: List[float],
    t_steps: int,
    ne_steps: int,
    partition: str,
    dry_run: bool,
) -> str:
    """Submit a SLURM array job for parallel basis library generation."""
    from cflibs.hpc.slurm import ArrayJobConfig, SlurmJobManager

    output_dir.mkdir(parents=True, exist_ok=True)
    manager = SlurmJobManager(dry_run=dry_run)

    config = ArrayJobConfig(
        job_name="cflibs_basis",
        partition=partition,
        cpus_per_task=4,
        mem_gb=32,
        time_limit="04:00:00",
        output_path=str(output_dir / "basis_%a.out"),
        error_path=str(output_dir / "basis_%a.err"),
        env_vars={"JAX_PLATFORMS": "cpu"},
        array_size=len(fwhm_values),
        max_concurrent=len(fwhm_values),
    )

    script_path = Path(__file__).resolve()
    fwhm_args = " ".join(str(f) for f in fwhm_values)

    script_content = (
        f"python {shlex.quote(str(script_path))} \\\n"
        f"    --db-path {shlex.quote(str(db_path))} \\\n"
        f"    --output-dir {shlex.quote(str(output_dir))} \\\n"
        f"    --fwhm {fwhm_args} \\\n"
        f"    --T-steps {t_steps} \\\n"
        f"    --ne-steps {ne_steps}"
    )

    job_id = manager.submit(config, script_content)
    print(f"Submitted basis library array job: {job_id}")
    print(f"  {len(fwhm_values)} tasks, one per FWHM value")
    print(f"  Output directory: {output_dir}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate basis libraries at multiple FWHM values")
    parser.add_argument("--db-path", type=str, default=None, help="Atomic database path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/hpc_benchmark/basis_libraries",
        help="Output directory for basis library files",
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        nargs="+",
        default=FWHM_VALUES,
        help="FWHM values (nm) to generate libraries for",
    )
    parser.add_argument("--T-steps", type=int, default=50, help="Temperature grid steps")
    parser.add_argument("--ne-steps", type=int, default=20, help="Electron density grid steps")
    parser.add_argument(
        "--submit", action="store_true", help="Submit as SLURM array job instead of running locally"
    )
    parser.add_argument("--partition", type=str, default="gpu", help="SLURM partition")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print SBATCH script without submitting"
    )

    args = parser.parse_args()

    db_path = _resolve_db_path(args.db_path)
    output_dir = Path(args.output_dir).resolve()
    fwhm_values = sorted(args.fwhm)

    if args.submit or args.dry_run:
        submit_slurm(
            db_path=db_path,
            output_dir=output_dir,
            fwhm_values=fwhm_values,
            t_steps=args.T_steps,
            ne_steps=args.ne_steps,
            partition=args.partition,
            dry_run=args.dry_run,
        )
    elif os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        run_slurm_task(db_path, output_dir, fwhm_values, args.T_steps, args.ne_steps)
    else:
        run_local(db_path, output_dir, fwhm_values, args.T_steps, args.ne_steps)


if __name__ == "__main__":
    main()
