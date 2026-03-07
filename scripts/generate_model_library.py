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
import shlex
import sys
from pathlib import Path
from typing import Any, List, Mapping, NoReturn, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _error_exit(message: str, code: int = 1) -> NoReturn:
    print(f"ERROR: {message}")
    sys.exit(code)


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError:
        _error_exit("h5py not available. Install h5py: pip install h5py")
    return h5py


def _resolve_db_path(db_path: Optional[Path]) -> Path:
    if db_path is not None:
        resolved = db_path.expanduser().resolve()
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
        "Atomic database not found. Pass --db-path explicitly or place libs_production.db "
        "in the project root or ASD_da/."
    )


def _parse_elements(elements: Optional[Sequence[str]]) -> List[str]:
    if not elements:
        return []

    parsed: List[str] = []
    for token in elements:
        for element in token.split(","):
            stripped = element.strip()
            if stripped and stripped not in parsed:
                parsed.append(stripped)
    return parsed


def _validate_chunk_generation_args(
    n_chunks: int,
    chunk_id: int,
    n_spectra_per_chunk: int,
    t_min: float,
    t_max: float,
    n_e_min: float,
    n_e_max: float,
    lambda_min: float,
    lambda_max: float,
    delta_lambda: float,
) -> None:
    if n_chunks < 1:
        _error_exit(f"n_chunks must be >= 1, got {n_chunks}")
    if not (0 <= chunk_id < n_chunks):
        _error_exit(f"chunk_id must be in range [0, {n_chunks}), got {chunk_id}")
    if n_spectra_per_chunk < 1:
        _error_exit(f"n_spectra must be >= 1, got {n_spectra_per_chunk}")
    if t_min >= t_max:
        _error_exit(f"t_min must be < t_max, got {t_min} >= {t_max}")
    if n_e_min >= n_e_max:
        _error_exit(f"n_e_min must be < n_e_max, got {n_e_min} >= {n_e_max}")
    if lambda_min >= lambda_max:
        _error_exit(f"lambda_min must be < lambda_max, got {lambda_min} >= {lambda_max}")
    if delta_lambda <= 0:
        _error_exit(f"delta_lambda must be > 0, got {delta_lambda}")


def _sample_compositions(
    rng: np.random.Generator, n_spectra: int, elements: Sequence[str]
) -> np.ndarray:
    if not elements:
        _error_exit("At least one element must be provided for library generation")
    alpha = np.ones(len(elements), dtype=np.float64)
    return rng.dirichlet(alpha, size=n_spectra)


def _decode_elements(raw_elements: Any) -> List[str]:
    decoded: List[str] = []
    for raw in raw_elements:
        decoded.append(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
    return decoded


def _save_library_file(
    output_file: Path,
    wavelengths: np.ndarray,
    spectra: np.ndarray,
    temperatures: np.ndarray,
    n_e_values: np.ndarray,
    compositions: np.ndarray,
    elements: Sequence[str],
    metadata: Mapping[str, Any],
) -> None:
    h5py = _require_h5py()
    params = np.column_stack([temperatures, n_e_values, compositions])

    with h5py.File(output_file, "w") as f:
        f.create_dataset("wavelength", data=wavelengths)
        f.create_dataset("wavelengths", data=wavelengths)
        f.create_dataset("spectra", data=spectra, compression="gzip")
        f.create_dataset("temperatures", data=temperatures, compression="gzip")
        f.create_dataset("n_e", data=n_e_values, compression="gzip")
        f.create_dataset("compositions", data=compositions, compression="gzip")
        f.create_dataset("params", data=params, compression="gzip")
        f.attrs["elements"] = np.asarray(list(elements), dtype="S")
        for key, value in metadata.items():
            f.attrs[key] = value


def _save_embedder(output_path: Path, embedder) -> None:
    pca_result = embedder.pca_pipeline.result_
    if pca_result is None:
        _error_exit("Embedder has not been fitted")
    np.savez_compressed(
        output_path,
        n_components=np.asarray([embedder.n_components], dtype=np.int32),
        components=np.asarray(pca_result.components, dtype=np.float32),
        mean=np.asarray(pca_result.mean, dtype=np.float32),
        explained_variance=np.asarray(pca_result.explained_variance, dtype=np.float32),
        explained_variance_ratio=np.asarray(pca_result.explained_variance_ratio, dtype=np.float32),
    )


def chunk_mode(
    chunk_id: int,
    n_chunks: int,
    output_dir: Path,
    db_path: Optional[Path] = None,
    elements: Optional[Sequence[str]] = None,
    t_min: float = 5000.0,
    t_max: float = 20000.0,
    n_e_min: float = 1e16,
    n_e_max: float = 1e18,
    n_spectra_per_chunk: int = 1000,
    lambda_min: float = 200.0,
    lambda_max: float = 800.0,
    delta_lambda: float = 0.1,
    instrument_fwhm_nm: float = 0.15,
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
    _validate_chunk_generation_args(
        n_chunks=n_chunks,
        chunk_id=chunk_id,
        n_spectra_per_chunk=n_spectra_per_chunk,
        t_min=t_min,
        t_max=t_max,
        n_e_min=n_e_min,
        n_e_max=n_e_max,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        delta_lambda=delta_lambda,
    )

    try:
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.instrument.model import InstrumentModel
        from cflibs.plasma.state import SingleZoneLTEPlasma
        from cflibs.radiation.spectrum_model import SpectrumModel
    except ImportError:
        _error_exit("CF-LIBS runtime dependencies are not available. Install cflibs first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_db_path = _resolve_db_path(db_path)
    atomic_db = AtomicDatabase(str(resolved_db_path))
    available_elements = atomic_db.get_available_elements()
    selected_elements = _parse_elements(elements) or available_elements
    missing_elements = sorted(set(selected_elements) - set(available_elements))
    if missing_elements:
        _error_exit(f"Elements not present in database: {', '.join(missing_elements)}")
    if not selected_elements:
        _error_exit("No elements available for model-library generation")

    # Divide temperature range into chunks
    t_range = np.linspace(t_min, t_max, n_chunks + 1)
    t_chunk_min = t_range[chunk_id]
    t_chunk_max = t_range[chunk_id + 1]

    print(f"Chunk {chunk_id}/{n_chunks}")
    print(f"Temperature range: {t_chunk_min:.1f} - {t_chunk_max:.1f} K")
    print(f"Elements: {', '.join(selected_elements)}")
    print(f"Generating {n_spectra_per_chunk} spectra...")

    # Generate random samples within chunk's parameter range
    rng = np.random.default_rng(seed=42 + chunk_id)
    temperatures = rng.uniform(t_chunk_min, t_chunk_max, n_spectra_per_chunk)
    n_e_values = rng.uniform(np.log10(n_e_min), np.log10(n_e_max), n_spectra_per_chunk)
    n_e_values = 10**n_e_values
    compositions = _sample_compositions(rng, n_spectra_per_chunk, selected_elements)
    instrument = InstrumentModel(resolution_fwhm_nm=instrument_fwhm_nm)

    wavelengths = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda, dtype=np.float64)
    spectra = np.zeros((n_spectra_per_chunk, len(wavelengths)), dtype=np.float64)

    for idx, (temperature, n_e_value) in enumerate(zip(temperatures, n_e_values)):
        species = {
            element: float(fraction * n_e_value)
            for element, fraction in zip(selected_elements, compositions[idx])
        }
        plasma = SingleZoneLTEPlasma(T_e=float(temperature), n_e=float(n_e_value), species=species)
        model = SpectrumModel(
            plasma=plasma,
            atomic_db=atomic_db,
            instrument=instrument,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            delta_lambda=delta_lambda,
        )
        _, intensity = model.compute_spectrum()
        spectra[idx] = intensity

    # Save chunk
    chunk_file = output_dir / f"chunk_{chunk_id:04d}.h5"
    _save_library_file(
        chunk_file,
        wavelengths=wavelengths,
        spectra=spectra,
        temperatures=temperatures,
        n_e_values=n_e_values,
        compositions=compositions,
        elements=selected_elements,
        metadata={
            "chunk_id": chunk_id,
            "n_chunks": n_chunks,
            "db_path": str(resolved_db_path),
        },
    )

    print(f"Saved: {chunk_file}")


def consolidate_mode(output_dir: Path) -> None:
    """
    Consolidate chunk files into a single model library.

    Parameters
    ----------
    output_dir : Path
        Directory containing chunk files
    """
    h5py = _require_h5py()

    chunk_files = sorted(output_dir.glob("chunk_*.h5"))
    if not chunk_files:
        _error_exit(f"No chunk files found in {output_dir}")

    print(f"Consolidating {len(chunk_files)} chunks...")

    # Read first chunk to get dimensions
    with h5py.File(chunk_files[0], "r") as f:
        wavelengths = f["wavelength"][:]
        n_wavelengths = len(wavelengths)
        elements = _decode_elements(f.attrs.get("elements", []))
        db_path = str(f.attrs.get("db_path", ""))

    # Count total spectra
    total_spectra = 0
    for chunk_file in chunk_files:
        with h5py.File(chunk_file, "r") as f:
            total_spectra += f["spectra"].shape[0]

    # Allocate combined arrays
    all_spectra = np.zeros((total_spectra, n_wavelengths))
    all_temperatures = np.zeros(total_spectra)
    all_n_e = np.zeros(total_spectra)
    all_compositions = np.zeros((total_spectra, len(elements)))

    # Read and concatenate
    offset = 0
    for chunk_file in chunk_files:
        with h5py.File(chunk_file, "r") as f:
            chunk_elements = _decode_elements(f.attrs.get("elements", []))
            if chunk_elements != elements:
                _error_exit(f"Chunk {chunk_file} has mismatched elements: {chunk_elements}")
            n = f["spectra"].shape[0]
            all_spectra[offset : offset + n] = f["spectra"][:]
            all_temperatures[offset : offset + n] = f["temperatures"][:]
            all_n_e[offset : offset + n] = f["n_e"][:]
            all_compositions[offset : offset + n] = f["compositions"][:]
            offset += n

    # Save combined library
    output_file = output_dir / "model_library.h5"
    _save_library_file(
        output_file,
        wavelengths=wavelengths,
        spectra=all_spectra,
        temperatures=all_temperatures,
        n_e_values=all_n_e,
        compositions=all_compositions,
        elements=elements,
        metadata={
            "n_spectra": total_spectra,
            "n_chunks": len(chunk_files),
            "db_path": db_path,
        },
    )

    print(f"Saved consolidated library: {output_file}")
    print(f"Total spectra: {total_spectra}")


def build_index_mode(
    output_dir: Path,
    n_components: int = 30,
    index_type: str = "flat",
    n_lists: int = 100,
    n_probe: int = 10,
    pq_m: int = 8,
    pq_bits: int = 8,
) -> None:
    """
    Build FAISS index for fast spectrum lookup.

    Parameters
    ----------
    output_dir : Path
        Directory containing model_library.h5
    """
    h5py = _require_h5py()
    library_file = output_dir / "model_library.h5"
    if not library_file.exists():
        _error_exit(f"Library file not found: {library_file}")

    try:
        from cflibs.manifold.vector_index import SpectralEmbedder, VectorIndex, VectorIndexConfig
    except ImportError as exc:
        _error_exit(f"Vector-index dependencies unavailable: {exc}")

    with h5py.File(library_file, "r") as f:
        spectra = np.asarray(f["spectra"][:], dtype=np.float32)

    if spectra.ndim != 2 or spectra.shape[0] < 2 or spectra.shape[1] < 2:
        _error_exit(f"Library spectra must be 2D with at least 2 rows/cols, got {spectra.shape}")

    effective_components = min(n_components, spectra.shape[0], spectra.shape[1])
    if effective_components < 1:
        _error_exit(f"n_components must be >= 1 after adjustment, got {effective_components}")

    config = VectorIndexConfig(
        index_type=index_type,
        n_lists=n_lists,
        n_probe=n_probe,
        pq_m=pq_m,
        pq_bits=pq_bits,
    )

    embedder = SpectralEmbedder(n_components=effective_components)
    embeddings = embedder.fit_transform(spectra)

    vector_index = VectorIndex(dimension=effective_components, config=config)
    vector_index.build(embeddings)

    index_path = output_dir / "model_library.index.h5"
    embedder_path = output_dir / "model_library.embedder.npz"
    vector_index.save(str(index_path))
    _save_embedder(embedder_path, embedder)

    print(f"Saved vector index: {index_path}")
    print(f"Saved embedder: {embedder_path}")


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
    if n_chunks < 1:
        _error_exit(f"n_chunks must be >= 1, got {n_chunks}")
    if max_concurrent < 0:
        _error_exit(f"max_concurrent must be >= 0, got {max_concurrent}")
    if mem_gb < 1:
        _error_exit(f"mem_gb must be >= 1, got {mem_gb}")

    try:
        from cflibs.hpc import ArrayJobConfig, SlurmJobConfig, SlurmJobManager
    except ImportError:
        _error_exit("cflibs.hpc not available. Install cflibs first.")

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
    consolidate_config = SlurmJobConfig(
        job_name="cflibs_consolidate",
        partition=partition,
        cpus_per_task=1,
        mem_gb=mem_gb * 2,
        time_limit="01:00:00",
        output_path=str(output_dir / "consolidate.out"),
        error_path=str(output_dir / "consolidate.err"),
    )

    consolidate_script = f"""
python {shlex.quote(str(script_path))} consolidate --output-dir {shlex.quote(str(output_dir))}
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
    chunk_parser.add_argument("--db-path", type=Path, help="Atomic database path")
    chunk_parser.add_argument(
        "--elements",
        nargs="+",
        help="Element symbols (space or comma separated). Defaults to all elements in the DB.",
    )
    chunk_parser.add_argument("--t-min", type=float, default=5000.0, help="Min temperature (K)")
    chunk_parser.add_argument("--t-max", type=float, default=20000.0, help="Max temperature (K)")
    chunk_parser.add_argument(
        "--n-e-min", type=float, default=1e16, help="Min electron density (cm^-3)"
    )
    chunk_parser.add_argument(
        "--n-e-max", type=float, default=1e18, help="Max electron density (cm^-3)"
    )
    chunk_parser.add_argument("--n-spectra", type=int, default=1000, help="Spectra per chunk")
    chunk_parser.add_argument("--lambda-min", type=float, default=200.0, help="Min wavelength (nm)")
    chunk_parser.add_argument("--lambda-max", type=float, default=800.0, help="Max wavelength (nm)")
    chunk_parser.add_argument(
        "--delta-lambda", type=float, default=0.1, help="Wavelength step (nm)"
    )
    chunk_parser.add_argument(
        "--instrument-fwhm-nm",
        type=float,
        default=0.15,
        help="Instrument broadening FWHM (nm)",
    )

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
    index_parser.add_argument(
        "--n-components", type=int, default=30, help="PCA components for embeddings"
    )
    index_parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        help="Vector index type: flat, ivf_flat, or ivf_pq",
    )
    index_parser.add_argument("--n-lists", type=int, default=100, help="IVF cell count")
    index_parser.add_argument("--n-probe", type=int, default=10, help="IVF probe count")
    index_parser.add_argument("--pq-m", type=int, default=8, help="PQ subquantizer count")
    index_parser.add_argument("--pq-bits", type=int, default=8, help="PQ bits per code")

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
            db_path=args.db_path,
            elements=args.elements,
            t_min=args.t_min,
            t_max=args.t_max,
            n_e_min=args.n_e_min,
            n_e_max=args.n_e_max,
            n_spectra_per_chunk=args.n_spectra,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            delta_lambda=args.delta_lambda,
            instrument_fwhm_nm=args.instrument_fwhm_nm,
        )
    elif args.mode == "consolidate":
        consolidate_mode(output_dir=args.output_dir)
    elif args.mode == "build-index":
        build_index_mode(
            output_dir=args.output_dir,
            n_components=args.n_components,
            index_type=args.index_type,
            n_lists=args.n_lists,
            n_probe=args.n_probe,
            pq_m=args.pq_m,
            pq_bits=args.pq_bits,
        )
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
