#!/usr/bin/env python3
"""
Generate 1M synthetic LIBS spectra for benchmarking on an HPC cluster.

Supports three modes:
- chunk: Generate ~10K synthetic spectra for a given (RP, SNR) condition
- consolidate: Merge all chunk HDF5 files into a single benchmark file
- submit: Generate and submit SLURM array job scripts

The spectra are generated from first-principles Saha-Boltzmann physics with
realistic noise models (Poisson shot noise, Gaussian readout, laser jitter).
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import csv
import json
import logging
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("generate_synthetic_benchmark")

# ---------------------------------------------------------------------------
# Element and condition grids
# ---------------------------------------------------------------------------

AALTO_SEARCH_ELEMENTS = [
    "Fe",
    "Ca",
    "Mg",
    "Si",
    "Al",
    "Ti",
    "Na",
    "K",
    "Mn",
    "Cr",
    "Ni",
    "Cu",
    "Co",
    "V",
    "Li",
    "Sr",
    "Ba",
    "Zn",
    "Pb",
    "Mo",
    "Zr",
    "Sn",
]

RP_VALUES = [200, 300, 500, 700, 1000, 2000, 3000, 5000, 10000]
SNR_VALUES = [10, 20, 50, 100, 200, 500, 1000]

# Wavelength grid parameters
WL_MIN_NM = 200.0
WL_MAX_NM = 900.0
N_PIXELS = 4096

# Gaussian FWHM -> sigma conversion factor
FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_exit(message: str, code: int = 1) -> NoReturn:
    logger.error(message)
    print(f"ERROR: {message}", file=sys.stderr)
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
        "Atomic database not found. Pass --db-path explicitly or place "
        "libs_production.db in the project root or ASD_da/."
    )


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _condition_index(rp_idx: int, snr_idx: int) -> int:
    """Flatten (rp_idx, snr_idx) into a single array-task index."""
    return rp_idx * len(SNR_VALUES) + snr_idx


def _index_to_condition(task_id: int) -> Tuple[int, int]:
    """Inverse of _condition_index: task_id -> (rp_idx, snr_idx)."""
    rp_idx = task_id // len(SNR_VALUES)
    snr_idx = task_id % len(SNR_VALUES)
    return rp_idx, snr_idx


# ---------------------------------------------------------------------------
# Spectrum generation engine
# ---------------------------------------------------------------------------


def _compute_element_basis(
    element: str,
    wl_grid: np.ndarray,
    sigma: float,
    T_K: float,
    ne: float,
    solver,
    atomic_db,
    ionization_stages: Tuple[int, ...] = (1, 2),
) -> np.ndarray:
    """Compute area-normalised basis spectrum for one element at (T, ne).

    Follows the same physics as BasisLibraryGenerator._compute_element_spectra
    but for a single (T, ne) point.

    Returns
    -------
    ndarray of shape (n_pixels,)
        Area-normalised emission spectrum for *element*.
    """
    from cflibs.core.constants import KB_EV

    n_pix = len(wl_grid)
    wl_min = float(wl_grid[0])
    wl_max = float(wl_grid[-1])

    transitions = []
    for stage in ionization_stages:
        transitions.extend(
            atomic_db.get_transitions(
                element,
                ionization_stage=stage,
                wavelength_min=wl_min,
                wavelength_max=wl_max,
            )
        )

    if not transitions:
        return np.zeros(n_pix, dtype=np.float64)

    T_eV = T_K * KB_EV
    total_density_cm3 = 1.0  # cancels after area normalisation

    stage_densities = solver.solve_ionization_balance(
        element, T_eV, ne, total_density_cm3=total_density_cm3
    )

    spectrum = np.zeros(n_pix, dtype=np.float64)
    for trans in transitions:
        stage_density = stage_densities.get(trans.ionization_stage, 0.0)
        if stage_density <= 0.0:
            continue

        U = solver.calculate_partition_function(element, trans.ionization_stage, T_eV)
        if U <= 0.0:
            continue

        n_k = stage_density * (trans.g_k / U) * np.exp(-trans.E_k_ev / T_eV)
        eps = trans.A_ki * n_k / trans.wavelength_nm

        spectrum += eps * np.exp(-0.5 * ((wl_grid - trans.wavelength_nm) / sigma) ** 2)

    area = np.sum(spectrum)
    if area > 1e-100:
        spectrum /= area

    return spectrum


def _generate_spectra_for_condition(
    rp: int,
    snr: int,
    n_spectra: int,
    seed: int,
    wl_grid: np.ndarray,
    solver,
    atomic_db,
    available_elements: List[str],
) -> Dict[str, Any]:
    """Generate n_spectra synthetic LIBS spectra for a given (RP, SNR).

    Parameters
    ----------
    rp : int
        Resolving power (lambda / delta_lambda).
    snr : int
        Target signal-to-noise ratio.
    n_spectra : int
        Number of spectra to generate.
    seed : int
        Random seed for reproducibility.
    wl_grid : ndarray
        Wavelength grid in nm.
    solver : SahaBoltzmannSolver
        Pre-initialised solver instance.
    atomic_db : AtomicDatabase
        Atomic database instance.
    available_elements : list of str
        Subset of AALTO_SEARCH_ELEMENTS that exist in the database.

    Returns
    -------
    dict
        Keys: spectra, wavelength, ground_truth_elements, concentrations,
              temperature_K, electron_density, resolving_power, snr
    """
    rng = np.random.default_rng(seed)
    n_pix = len(wl_grid)
    n_el = len(available_elements)

    # Instrument FWHM from resolving power at median wavelength
    median_wl = np.median(wl_grid)
    fwhm_nm = median_wl / rp
    sigma = fwhm_nm * FWHM_TO_SIGMA

    # Preallocate output arrays
    spectra = np.zeros((n_spectra, n_pix), dtype=np.float64)
    temperatures = np.zeros(n_spectra, dtype=np.float64)
    electron_densities = np.zeros(n_spectra, dtype=np.float64)
    # Variable-length element lists encoded as JSON strings
    gt_elements_list: List[str] = []
    # Concentrations padded to max possible elements
    max_elements = min(8, n_el)
    concentrations = np.zeros((n_spectra, max_elements), dtype=np.float64)

    t_start = time.time()

    for idx in range(n_spectra):
        # Sample plasma parameters (log-uniform)
        T_K = np.exp(rng.uniform(np.log(4000.0), np.log(15000.0)))
        ne = np.exp(rng.uniform(np.log(1e15), np.log(5e17)))

        # Sample element subset
        n_elem = int(rng.integers(1, min(8, n_el) + 1))
        elem_indices = rng.choice(n_el, size=n_elem, replace=False)
        elem_subset = [available_elements[i] for i in sorted(elem_indices)]

        # Sample concentrations from Dirichlet(alpha=1)
        alpha = np.ones(n_elem, dtype=np.float64)
        conc = rng.dirichlet(alpha)

        temperatures[idx] = T_K
        electron_densities[idx] = ne
        gt_elements_list.append(json.dumps(elem_subset))
        concentrations[idx, :n_elem] = conc

        # Compute per-element basis spectra and linear-combine
        combined = np.zeros(n_pix, dtype=np.float64)
        for el_idx, element in enumerate(elem_subset):
            basis = _compute_element_basis(element, wl_grid, sigma, T_K, ne, solver, atomic_db)
            combined += conc[el_idx] * basis

        # Scale so that peak ~ SNR (before noise) to set meaningful SNR
        peak = np.max(combined)
        if peak > 1e-100:
            # Scale signal so peak intensity ~ SNR^2 (Poisson-limited regime)
            combined *= snr**2 / peak

        # Add realistic noise
        # 1. Poisson shot noise
        signal_positive = np.maximum(combined, 0.0)
        noisy = rng.poisson(signal_positive).astype(np.float64)

        # 2. Gaussian readout noise (sigma=5 counts)
        noisy += rng.normal(0.0, 5.0, size=n_pix)

        # 3. Multiplicative laser jitter (2% pulse-to-pulse)
        jitter = 1.0 + rng.normal(0.0, 0.02)
        noisy *= jitter

        spectra[idx] = noisy

        if (idx + 1) % 1000 == 0 or idx == n_spectra - 1:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            logger.info(
                "  [RP=%d, SNR=%d] %d/%d spectra (%.1f spec/s)",
                rp,
                snr,
                idx + 1,
                n_spectra,
                rate,
            )

    return {
        "spectra": spectra.astype(np.float32),
        "wavelength": wl_grid,
        "ground_truth_elements": gt_elements_list,
        "concentrations": concentrations.astype(np.float32),
        "temperature_K": temperatures.astype(np.float32),
        "electron_density": electron_densities.astype(np.float32),
        "resolving_power": rp,
        "snr": snr,
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def chunk_mode(
    chunk_id: int,
    n_chunks: int,
    output_dir: Path,
    db_path: Optional[Path] = None,
    rp: Optional[int] = None,
    snr: Optional[int] = None,
    n_spectra: int = 10000,
    seed: Optional[int] = None,
) -> None:
    """Generate synthetic spectra for a given (RP, SNR) condition chunk.

    If --rp and --snr are given explicitly, generates one chunk for that
    condition.  Otherwise, chunk_id indexes into the flattened
    (RP_VALUES x SNR_VALUES) grid so that each array task handles one
    condition.
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_db = _resolve_db_path(db_path)
    atomic_db = AtomicDatabase(str(resolved_db))
    solver = SahaBoltzmannSolver(atomic_db)

    # Determine which elements are available in the DB
    db_elements = set(atomic_db.get_available_elements())
    available_elements = [el for el in AALTO_SEARCH_ELEMENTS if el in db_elements]
    if not available_elements:
        _error_exit("No AALTO_SEARCH_ELEMENTS found in the database")

    # Resolve RP / SNR from chunk_id if not explicitly provided
    if rp is not None and snr is not None:
        target_rp = rp
        target_snr = snr
        rp_idx = RP_VALUES.index(rp) if rp in RP_VALUES else 0
        snr_idx = SNR_VALUES.index(snr) if snr in SNR_VALUES else 0
    else:
        n_conditions = len(RP_VALUES) * len(SNR_VALUES)
        if chunk_id >= n_conditions:
            _error_exit(
                f"chunk_id {chunk_id} >= total conditions {n_conditions} "
                f"({len(RP_VALUES)} RP x {len(SNR_VALUES)} SNR)"
            )
        rp_idx, snr_idx = _index_to_condition(chunk_id)
        target_rp = RP_VALUES[rp_idx]
        target_snr = SNR_VALUES[snr_idx]

    # Deterministic seed per condition
    if seed is None:
        seed = 42 + chunk_id * 1000 + rp_idx * 100 + snr_idx

    wl_grid = np.linspace(WL_MIN_NM, WL_MAX_NM, N_PIXELS)

    logger.info(
        "Chunk %d/%d: RP=%d, SNR=%d, n_spectra=%d, seed=%d",
        chunk_id,
        n_chunks,
        target_rp,
        target_snr,
        n_spectra,
        seed,
    )
    logger.info("Elements (%d): %s", len(available_elements), ", ".join(available_elements))

    t0 = time.time()
    result = _generate_spectra_for_condition(
        rp=target_rp,
        snr=target_snr,
        n_spectra=n_spectra,
        seed=seed,
        wl_grid=wl_grid,
        solver=solver,
        atomic_db=atomic_db,
        available_elements=available_elements,
    )
    elapsed = time.time() - t0
    logger.info("Generated %d spectra in %.1f s", n_spectra, elapsed)

    # Save to HDF5
    h5py = _require_h5py()
    chunk_file = output_dir / f"chunk_rp{target_rp}_snr{target_snr}.h5"

    with h5py.File(chunk_file, "w") as f:
        f.create_dataset("spectra", data=result["spectra"], compression="gzip")
        f.create_dataset("wavelength", data=result["wavelength"])
        f.create_dataset("temperature_K", data=result["temperature_K"], compression="gzip")
        f.create_dataset("electron_density", data=result["electron_density"], compression="gzip")
        f.create_dataset("concentrations", data=result["concentrations"], compression="gzip")

        # Variable-length element lists stored as JSON strings
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "ground_truth_elements",
            data=np.array(result["ground_truth_elements"], dtype=object),
            dtype=dt,
        )

        f.attrs["resolving_power"] = target_rp
        f.attrs["snr"] = target_snr
        f.attrs["n_spectra"] = n_spectra
        f.attrs["seed"] = seed
        f.attrs["chunk_id"] = chunk_id
        f.attrs["n_chunks"] = n_chunks
        f.attrs["db_path"] = str(resolved_db)
        f.attrs["available_elements"] = np.array(available_elements, dtype="S")

    logger.info("Saved: %s (%.1f MB)", chunk_file, chunk_file.stat().st_size / 1e6)


def consolidate_mode(output_dir: Path) -> None:
    """Merge all chunk HDF5 files into a single benchmark file with metadata."""
    h5py = _require_h5py()

    chunk_files = sorted(output_dir.glob("chunk_rp*.h5"))
    if not chunk_files:
        _error_exit(f"No chunk files (chunk_rp*.h5) found in {output_dir}")

    logger.info("Consolidating %d chunk files from %s", len(chunk_files), output_dir)

    # First pass: count total spectra and validate wavelength grids
    total_spectra = 0
    ref_wl = None
    file_info: List[Dict[str, Any]] = []

    for chunk_file in chunk_files:
        with h5py.File(chunk_file, "r") as f:
            n = f["spectra"].shape[0]
            n_pix = f["spectra"].shape[1]
            wl = f["wavelength"][:]
            rp = int(f.attrs["resolving_power"])
            snr_val = int(f.attrs["snr"])

            if ref_wl is None:
                ref_wl = wl
            elif len(wl) != len(ref_wl) or not np.allclose(wl, ref_wl):
                _error_exit(
                    f"Wavelength grid mismatch in {chunk_file.name}: "
                    f"expected {len(ref_wl)} pixels, got {len(wl)}"
                )

            total_spectra += n
            file_info.append(
                {
                    "file": chunk_file.name,
                    "rp": rp,
                    "snr": snr_val,
                    "n_spectra": n,
                    "n_pixels": n_pix,
                }
            )

    logger.info("Total spectra to consolidate: %d", total_spectra)
    n_pix = len(ref_wl)

    # Determine max_elements from concentration arrays
    max_elements = 0
    for chunk_file in chunk_files:
        with h5py.File(chunk_file, "r") as f:
            max_elements = max(max_elements, f["concentrations"].shape[1])

    # Second pass: allocate and fill
    output_file = output_dir / "synthetic_benchmark.h5"
    with h5py.File(output_file, "w") as out:
        ds_spectra = out.create_dataset(
            "spectra",
            shape=(total_spectra, n_pix),
            dtype=np.float32,
            compression="gzip",
        )
        out.create_dataset("wavelength", data=ref_wl)
        ds_temp = out.create_dataset(
            "temperature_K",
            shape=(total_spectra,),
            dtype=np.float32,
            compression="gzip",
        )
        ds_ne = out.create_dataset(
            "electron_density",
            shape=(total_spectra,),
            dtype=np.float32,
            compression="gzip",
        )
        ds_conc = out.create_dataset(
            "concentrations",
            shape=(total_spectra, max_elements),
            dtype=np.float32,
            compression="gzip",
        )

        dt = h5py.string_dtype(encoding="utf-8")
        ds_gt = out.create_dataset(
            "ground_truth_elements",
            shape=(total_spectra,),
            dtype=dt,
        )
        ds_rp = out.create_dataset(
            "resolving_power",
            shape=(total_spectra,),
            dtype=np.int32,
            compression="gzip",
        )
        ds_snr = out.create_dataset(
            "snr",
            shape=(total_spectra,),
            dtype=np.int32,
            compression="gzip",
        )

        offset = 0
        for chunk_file in chunk_files:
            with h5py.File(chunk_file, "r") as f:
                n = f["spectra"].shape[0]
                rp = int(f.attrs["resolving_power"])
                snr_val = int(f.attrs["snr"])
                n_conc = f["concentrations"].shape[1]

                ds_spectra[offset : offset + n] = f["spectra"][:]
                ds_temp[offset : offset + n] = f["temperature_K"][:]
                ds_ne[offset : offset + n] = f["electron_density"][:]
                ds_conc[offset : offset + n, :n_conc] = f["concentrations"][:]
                ds_gt[offset : offset + n] = f["ground_truth_elements"][:]
                ds_rp[offset : offset + n] = rp
                ds_snr[offset : offset + n] = snr_val

                offset += n

            logger.info("  Merged %s (RP=%d, SNR=%d, n=%d)", chunk_file.name, rp, snr_val, n)

        out.attrs["total_spectra"] = total_spectra
        out.attrs["n_conditions"] = len(chunk_files)
        out.attrs["rp_values"] = np.array(sorted({fi["rp"] for fi in file_info}))
        out.attrs["snr_values"] = np.array(sorted({fi["snr"] for fi in file_info}))

    # Write metadata CSV
    metadata_file = output_dir / "metadata.csv"
    with open(metadata_file, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["file", "rp", "snr", "n_spectra", "n_pixels"])
        writer.writeheader()
        writer.writerows(file_info)

    file_size_mb = output_file.stat().st_size / 1e6
    logger.info("Saved consolidated benchmark: %s (%.1f MB)", output_file, file_size_mb)
    logger.info("Saved metadata: %s", metadata_file)
    logger.info("Total spectra: %d across %d conditions", total_spectra, len(chunk_files))


def submit_mode(
    output_dir: Path,
    db_path: Optional[Path] = None,
    partition: str = "gpu",
    mem_gb: int = 32,
    time_limit: str = "04:00:00",
    max_concurrent: int = 20,
    dry_run: bool = False,
    n_spectra_per_chunk: int = 16000,
) -> None:
    """Generate and submit SLURM array job for all (RP, SNR) conditions."""
    from cflibs.hpc.slurm import ArrayJobConfig, SlurmJobConfig, SlurmJobManager

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_db = _resolve_db_path(db_path)

    n_conditions = len(RP_VALUES) * len(SNR_VALUES)  # 9 * 7 = 63
    logger.info(
        "Submitting %d array tasks (%d RP x %d SNR conditions)",
        n_conditions,
        len(RP_VALUES),
        len(SNR_VALUES),
    )
    logger.info("Output directory: %s", output_dir)
    logger.info("Spectra per condition: %d", n_spectra_per_chunk)
    logger.info("Total spectra: %d", n_conditions * n_spectra_per_chunk)

    manager = SlurmJobManager(dry_run=dry_run)
    script_path = Path(__file__).resolve()

    # Array job: one task per (RP, SNR) condition
    chunk_config = ArrayJobConfig(
        job_name="synth_bench_chunk",
        partition=partition,
        cpus_per_task=2,
        mem_gb=mem_gb,
        time_limit=time_limit,
        output_path=str(output_dir / "chunk_%a.out"),
        error_path=str(output_dir / "chunk_%a.err"),
        array_size=n_conditions,
        max_concurrent=max_concurrent,
        env_vars={"JAX_PLATFORMS": "cpu"},
    )

    db_arg = f"--db-path {shlex.quote(str(resolved_db))}"
    chunk_script = (
        f"python {shlex.quote(str(script_path))} chunk \\\n"
        f"    --chunk-id $SLURM_ARRAY_TASK_ID \\\n"
        f"    --n-chunks {n_conditions} \\\n"
        f"    --output-dir {shlex.quote(str(output_dir))} \\\n"
        f"    {db_arg} \\\n"
        f"    --n-spectra {n_spectra_per_chunk}"
    )

    logger.info("Submitting chunk generation array job...")
    chunk_job_id = manager.submit(chunk_config, chunk_script)
    logger.info("Submitted chunk array job: %s", chunk_job_id)

    # Consolidation job with dependency on all chunks completing
    consolidate_config = SlurmJobConfig(
        job_name="synth_bench_consolidate",
        partition=partition,
        cpus_per_task=1,
        mem_gb=mem_gb * 2,
        time_limit="02:00:00",
        output_path=str(output_dir / "consolidate.out"),
        error_path=str(output_dir / "consolidate.err"),
        env_vars={"JAX_PLATFORMS": "cpu"},
    )

    consolidate_script = (
        f"python {shlex.quote(str(script_path))} consolidate \\\n"
        f"    --output-dir {shlex.quote(str(output_dir))}"
    )

    logger.info("Submitting consolidation job (depends on chunk completion)...")
    consolidate_job_id = manager.submit_with_dependency(
        consolidate_config,
        consolidate_script,
        depends_on=[chunk_job_id],
        dependency_type="afterok",
    )
    logger.info("Submitted consolidation job: %s", consolidate_job_id)

    # Summary
    print("\nJob submission complete!")
    print(f"  Chunk array job:     {chunk_job_id} ({n_conditions} tasks)")
    print(f"  Consolidation job:   {consolidate_job_id} (afterok:{chunk_job_id})")
    print(f"  Spectra per task:    {n_spectra_per_chunk}")
    print(f"  Total spectra:       {n_conditions * n_spectra_per_chunk:,}")
    print("\nMonitor with: squeue -u $USER")
    print(f"Check logs in: {output_dir}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Generate synthetic LIBS benchmark spectra for HPC"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -- chunk --
    chunk_parser = subparsers.add_parser(
        "chunk", help="Generate spectra for one (RP, SNR) condition"
    )
    chunk_parser.add_argument(
        "--chunk-id",
        type=int,
        required=True,
        help="Chunk / array-task ID (indexes into RP x SNR grid)",
    )
    chunk_parser.add_argument(
        "--n-chunks",
        type=int,
        required=True,
        help="Total number of chunks (for metadata only)",
    )
    chunk_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for chunk files",
    )
    chunk_parser.add_argument("--db-path", type=Path, help="Atomic database path")
    chunk_parser.add_argument(
        "--rp",
        type=int,
        default=None,
        help="Resolving power (overrides chunk_id-based lookup)",
    )
    chunk_parser.add_argument(
        "--snr",
        type=int,
        default=None,
        help="Signal-to-noise ratio (overrides chunk_id-based lookup)",
    )
    chunk_parser.add_argument(
        "--n-spectra",
        type=int,
        default=10000,
        help="Number of spectra per chunk",
    )
    chunk_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: deterministic from chunk_id)",
    )

    # -- consolidate --
    consolidate_parser = subparsers.add_parser(
        "consolidate", help="Merge chunk files into single benchmark"
    )
    consolidate_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing chunk files",
    )

    # -- submit --
    submit_parser = subparsers.add_parser(
        "submit", help="Submit SLURM array jobs for all conditions"
    )
    submit_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory",
    )
    submit_parser.add_argument("--db-path", type=Path, help="Atomic database path")
    submit_parser.add_argument(
        "--partition",
        type=str,
        default="gpu",
        help="SLURM partition (default: gpu)",
    )
    submit_parser.add_argument(
        "--mem-gb",
        type=int,
        default=32,
        help="Memory per job in GB (default: 32)",
    )
    submit_parser.add_argument(
        "--time-limit",
        type=str,
        default="04:00:00",
        help="Time limit (default: 04:00:00)",
    )
    submit_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Max concurrent array tasks (default: 20)",
    )
    submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SBATCH scripts without submitting",
    )
    submit_parser.add_argument(
        "--n-spectra-per-chunk",
        type=int,
        default=16000,
        help="Spectra per (RP, SNR) condition (default: 16000)",
    )

    args = parser.parse_args()

    if args.mode == "chunk":
        chunk_mode(
            chunk_id=args.chunk_id,
            n_chunks=args.n_chunks,
            output_dir=args.output_dir,
            db_path=args.db_path,
            rp=args.rp,
            snr=args.snr,
            n_spectra=args.n_spectra,
            seed=args.seed,
        )
    elif args.mode == "consolidate":
        consolidate_mode(output_dir=args.output_dir)
    elif args.mode == "submit":
        submit_mode(
            output_dir=args.output_dir,
            db_path=args.db_path,
            partition=args.partition,
            mem_gb=args.mem_gb,
            time_limit=args.time_limit,
            max_concurrent=args.max_concurrent,
            dry_run=args.dry_run,
            n_spectra_per_chunk=args.n_spectra_per_chunk,
        )


if __name__ == "__main__":
    main()
