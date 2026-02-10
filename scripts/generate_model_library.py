#!/usr/bin/env python3
"""
Generate model spectrum library for CF-LIBS line identification.

Supports three modes:
- chunk: Generate spectra for a subset of parameter space
- consolidate: Combine chunk files into single library
- build-index: Build FAISS index for fast lookup
- submit: Submit SLURM array job for parallel generation
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def chunk_mode(
    chunk_id: int,
    n_chunks: int,
    output_dir: Path,
    t_min: float = 5000.0,
    t_max: float = 20000.0,
    n_e_min: float = 1e16,
    n_e_max: float = 1e18,
    n_spectra_per_chunk: int = 1000,
) -> None:
    """
    Generate model spectra for a specific chunk of parameter space.

    Parameters
    ----------
    chunk_id : int
        Chunk identifier (0 to n_chunks-1)
    n_chunks : int
        Total number of chunks
    output_dir : Path
        Output directory for chunk files
    t_min : float
        Minimum temperature (K)
    t_max : float
        Maximum temperature (K)
    n_e_min : float
        Minimum electron density (cm^-3)
    n_e_max : float
        Maximum electron density (cm^-3)
    n_spectra_per_chunk : int
        Number of spectra to generate per chunk
    """
    try:
        from cflibs.inversion.manifold import ManifoldGenerator  # noqa: F401
    except ImportError:
        print("ERROR: ManifoldGenerator not available. Install cflibs first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Divide temperature range into chunks
    t_range = np.linspace(t_min, t_max, n_chunks + 1)
    t_chunk_min = t_range[chunk_id]
    t_chunk_max = t_range[chunk_id + 1]

    print(f"Chunk {chunk_id}/{n_chunks}")
    print(f"Temperature range: {t_chunk_min:.1f} - {t_chunk_max:.1f} K")
    print(f"Generating {n_spectra_per_chunk} spectra...")

    # Generate random samples within chunk's parameter range
    rng = np.random.default_rng(seed=42 + chunk_id)
    temperatures = rng.uniform(t_chunk_min, t_chunk_max, n_spectra_per_chunk)
    n_e_values = rng.uniform(np.log10(n_e_min), np.log10(n_e_max), n_spectra_per_chunk)
    n_e_values = 10**n_e_values

    # TODO: Use ManifoldGenerator to generate spectra
    # For now, create placeholder data
    wavelengths = np.linspace(200, 800, 6000)
    spectra = np.zeros((n_spectra_per_chunk, len(wavelengths)))

    # Save chunk
    chunk_file = output_dir / f"chunk_{chunk_id:04d}.h5"
    with h5py.File(chunk_file, "w") as f:
        f.create_dataset("wavelengths", data=wavelengths)
        f.create_dataset("spectra", data=spectra)
        f.create_dataset("temperatures", data=temperatures)
        f.create_dataset("n_e", data=n_e_values)
        f.attrs["chunk_id"] = chunk_id
        f.attrs["n_chunks"] = n_chunks

    print(f"Saved: {chunk_file}")


def consolidate_mode(output_dir: Path) -> None:
    """
    Consolidate chunk files into a single model library.

    Parameters
    ----------
    output_dir : Path
        Directory containing chunk files
    """
    chunk_files = sorted(output_dir.glob("chunk_*.h5"))
    if not chunk_files:
        print(f"ERROR: No chunk files found in {output_dir}")
        sys.exit(1)

    print(f"Consolidating {len(chunk_files)} chunks...")

    # Read first chunk to get dimensions
    with h5py.File(chunk_files[0], "r") as f:
        wavelengths = f["wavelengths"][:]
        n_wavelengths = len(wavelengths)

    # Count total spectra
    total_spectra = 0
    for chunk_file in chunk_files:
        with h5py.File(chunk_file, "r") as f:
            total_spectra += f["spectra"].shape[0]

    # Allocate combined arrays
    all_spectra = np.zeros((total_spectra, n_wavelengths))
    all_temperatures = np.zeros(total_spectra)
    all_n_e = np.zeros(total_spectra)

    # Read and concatenate
    offset = 0
    for chunk_file in chunk_files:
        with h5py.File(chunk_file, "r") as f:
            n = f["spectra"].shape[0]
            all_spectra[offset : offset + n] = f["spectra"][:]
            all_temperatures[offset : offset + n] = f["temperatures"][:]
            all_n_e[offset : offset + n] = f["n_e"][:]
            offset += n

    # Save combined library
    output_file = output_dir / "model_library.h5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("wavelengths", data=wavelengths)
        f.create_dataset("spectra", data=all_spectra, compression="gzip")
        f.create_dataset("temperatures", data=all_temperatures, compression="gzip")
        f.create_dataset("n_e", data=all_n_e, compression="gzip")
        f.attrs["n_spectra"] = total_spectra
        f.attrs["n_chunks"] = len(chunk_files)

    print(f"Saved consolidated library: {output_file}")
    print(f"Total spectra: {total_spectra}")


def build_index_mode(output_dir: Path) -> None:
    """
    Build FAISS index for fast spectrum lookup.

    Parameters
    ----------
    output_dir : Path
        Directory containing model_library.h5
    """
    print("Building FAISS index...")
    print("NOTE: Index building not yet implemented.")
    print("This is a placeholder for future FAISS integration.")

    library_file = output_dir / "model_library.h5"
    if not library_file.exists():
        print(f"ERROR: Library file not found: {library_file}")
        sys.exit(1)

    # TODO: Implement FAISS index building
    # 1. Load spectra from model_library.h5
    # 2. Optionally apply dimensionality reduction (PCA, wavelength subset)
    # 3. Build FAISS index (IVF or HNSW)
    # 4. Save index to disk


def submit_mode(
    n_chunks: int,
    output_dir: Path,
    partition: str = "default",
    time_limit: str = "02:00:00",
    mem_gb: int = 8,
    max_concurrent: int = 20,
) -> None:
    """
    Submit SLURM array job for parallel chunk generation.

    Parameters
    ----------
    n_chunks : int
        Number of chunks to divide work into
    output_dir : Path
        Output directory for chunks and final library
    partition : str
        SLURM partition (default: "default")
    time_limit : str
        Time limit per job (default: "02:00:00")
    mem_gb : int
        Memory per job in GB (default: 8)
    max_concurrent : int
        Max concurrent array tasks (default: 20)
    """
    try:
        from cflibs.hpc import ArrayJobConfig, SlurmJobManager
    except ImportError:
        print("ERROR: cflibs.hpc not available. Install cflibs first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    manager = SlurmJobManager(dry_run=False)

    # Array job for chunks
    chunk_config = ArrayJobConfig(
        job_name="cflibs_chunk",
        partition=partition,
        cpus_per_task=1,
        mem_gb=mem_gb,
        time_limit=time_limit,
        output_path=str(output_dir / "chunk_%a.out"),
        error_path=str(output_dir / "chunk_%a.err"),
        array_size=n_chunks,
        max_concurrent=max_concurrent,
    )

    script_path = Path(__file__).resolve()
    chunk_script = f"""
python {script_path} chunk \\
    --chunk-id $SLURM_ARRAY_TASK_ID \\
    --n-chunks {n_chunks} \\
    --output-dir {output_dir}
"""

    print("Submitting chunk generation array job...")
    chunk_job_id = manager.submit(chunk_config, chunk_script)
    print(f"Submitted chunk job: {chunk_job_id}")

    # Consolidation job with dependency
    consolidate_config = ArrayJobConfig(
        job_name="cflibs_consolidate",
        partition=partition,
        cpus_per_task=1,
        mem_gb=mem_gb * 2,
        time_limit="01:00:00",
        output_path=str(output_dir / "consolidate.out"),
        error_path=str(output_dir / "consolidate.err"),
    )

    consolidate_script = f"""
python {script_path} consolidate --output-dir {output_dir}
"""

    print("Submitting consolidation job (depends on chunk completion)...")
    consolidate_job_id = manager.submit_with_dependency(
        consolidate_config,
        consolidate_script,
        depends_on=[chunk_job_id],
        dependency_type="afterok",
    )
    print(f"Submitted consolidate job: {consolidate_job_id}")

    print("\nJob submission complete!")
    print("Monitor with: squeue -u $USER")
    print(f"Check logs in: {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate model spectrum library for CF-LIBS")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Chunk mode
    chunk_parser = subparsers.add_parser("chunk", help="Generate spectra chunk")
    chunk_parser.add_argument(
        "--chunk-id", type=int, required=True, help="Chunk ID (0 to n-chunks-1)"
    )
    chunk_parser.add_argument("--n-chunks", type=int, required=True, help="Total number of chunks")
    chunk_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    chunk_parser.add_argument("--t-min", type=float, default=5000.0, help="Min temperature (K)")
    chunk_parser.add_argument("--t-max", type=float, default=20000.0, help="Max temperature (K)")
    chunk_parser.add_argument(
        "--n-e-min", type=float, default=1e16, help="Min electron density (cm^-3)"
    )
    chunk_parser.add_argument(
        "--n-e-max", type=float, default=1e18, help="Max electron density (cm^-3)"
    )
    chunk_parser.add_argument("--n-spectra", type=int, default=1000, help="Spectra per chunk")

    # Consolidate mode
    consolidate_parser = subparsers.add_parser("consolidate", help="Consolidate chunks")
    consolidate_parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory with chunks"
    )

    # Build index mode
    index_parser = subparsers.add_parser("build-index", help="Build FAISS index")
    index_parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory with library"
    )

    # Submit mode
    submit_parser = subparsers.add_parser("submit", help="Submit SLURM jobs")
    submit_parser.add_argument("--n-chunks", type=int, required=True, help="Number of chunks")
    submit_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    submit_parser.add_argument("--partition", type=str, default="default", help="SLURM partition")
    submit_parser.add_argument(
        "--time-limit", type=str, default="02:00:00", help="Time limit (HH:MM:SS)"
    )
    submit_parser.add_argument("--mem-gb", type=int, default=8, help="Memory per job (GB)")
    submit_parser.add_argument(
        "--max-concurrent", type=int, default=20, help="Max concurrent tasks"
    )

    args = parser.parse_args()

    if args.mode == "chunk":
        chunk_mode(
            chunk_id=args.chunk_id,
            n_chunks=args.n_chunks,
            output_dir=args.output_dir,
            t_min=args.t_min,
            t_max=args.t_max,
            n_e_min=args.n_e_min,
            n_e_max=args.n_e_max,
            n_spectra_per_chunk=args.n_spectra,
        )
    elif args.mode == "consolidate":
        consolidate_mode(output_dir=args.output_dir)
    elif args.mode == "build-index":
        build_index_mode(output_dir=args.output_dir)
    elif args.mode == "submit":
        submit_mode(
            n_chunks=args.n_chunks,
            output_dir=args.output_dir,
            partition=args.partition,
            time_limit=args.time_limit,
            mem_gb=args.mem_gb,
            max_concurrent=args.max_concurrent,
        )


if __name__ == "__main__":
    main()
