#!/usr/bin/env python3
"""
Run element identification benchmarks on synthetic LIBS spectra via SLURM.

Three subcommands:
  worker   – Process one HDF5 chunk through all configs for a given pathway
  submit   – Submit SLURM array jobs (one per pathway x chunk)
  collect  – Aggregate per-chunk parquet files into summary statistics

Usage:
  # Worker (called by SLURM array task):
  python scripts/hpc/run_benchmark_sweep.py worker \
    --chunk-path output/synthetic/chunk_0.h5 \
    --basis-dir output/basis_libraries \
    --output-dir output/benchmark_sweep \
    --pathway alias --db-path ASD_da/libs_production.db

  # Submit SLURM array jobs:
  python scripts/hpc/run_benchmark_sweep.py submit \
    --synthetic-dir output/synthetic \
    --basis-dir output/basis_libraries \
    --output-dir output/benchmark_sweep \
    --db-path ASD_da/libs_production.db \
    --pathways alias spectral_nnls hybrid_intersect

  # Collect results:
  python scripts/hpc/run_benchmark_sweep.py collect \
    --output-dir output/benchmark_sweep
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
import logging
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]
    HAS_PYARROW = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_PATHWAYS = [
    "alias",
    "spectral_nnls",
    "hybrid_intersect",
    "hybrid_union",
    "forward_model",
    "voigt_alias",
]

# Resolving power -> nearest FWHM (nm) mapping
RP_TO_FWHM: Dict[int, float] = {
    200: 1.67,
    300: 1.67,
    500: 1.0,
    700: 0.71,
    1000: 0.5,
    2000: 0.25,
    3000: 0.17,
    5000: 0.1,
    10000: 0.05,
}


def _error_exit(message: str, code: int = 1) -> NoReturn:
    print(f"ERROR: {message}")
    sys.exit(code)


def _safe_ratio(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# RP -> FWHM mapping
# ---------------------------------------------------------------------------


def _select_basis_fwhm(rp: float) -> float:
    """Map a resolving power to the nearest basis library FWHM."""
    best_rp = min(RP_TO_FWHM.keys(), key=lambda k: abs(k - rp))
    return RP_TO_FWHM[best_rp]


def _find_basis_path(basis_dir: Path, fwhm: float) -> Optional[Path]:
    """Find the basis library file for a given FWHM."""
    path = basis_dir / f"basis_fwhm_{fwhm:.2g}nm.h5"
    if path.exists():
        return path
    # Try with more decimal places
    for fmt in [f"{fwhm:.1f}", f"{fwhm:.2f}", f"{fwhm}"]:
        candidate = basis_dir / f"basis_fwhm_{fmt}nm.h5"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Config generators (coarse / fine grids)
# ---------------------------------------------------------------------------


def _alias_configs(fine: bool) -> List[Dict[str, Any]]:
    if fine:
        dt_vals = np.linspace(0.01, 0.15, 15).tolist()
        itf_vals = [2.5, 3.0, 3.5, 4.0, 4.5]
        cws_vals = [0.3, 0.4, 0.5]
    else:
        dt_vals = [0.02, 0.03, 0.05, 0.08]
        itf_vals = [3.0, 3.5, 4.0]
        cws_vals = [0.3, 0.4]
    configs = []
    for dt in dt_vals:
        for itf in itf_vals:
            for cws in cws_vals:
                configs.append(
                    {
                        "config_name": f"dt={dt:.3f}_itf={itf}_cws={cws}",
                        "detection_threshold": float(dt),
                        "intensity_threshold_factor": itf,
                        "chance_window_scale": cws,
                        "max_lines_per_element": 30,
                    }
                )
    return configs


def _spectral_nnls_configs(fine: bool) -> List[Dict[str, Any]]:
    if fine:
        snr_vals = np.linspace(0.5, 6.0, 12).tolist()
        cdeg_vals = [2, 3, 4]
    else:
        snr_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        cdeg_vals = [2, 3, 4]
    configs = []
    for snr in snr_vals:
        for cdeg in cdeg_vals:
            configs.append(
                {
                    "config_name": f"snr={snr:.2f}_cdeg={cdeg}",
                    "detection_snr": float(snr),
                    "continuum_degree": cdeg,
                }
            )
    return configs


def _hybrid_intersect_configs(fine: bool) -> List[Dict[str, Any]]:
    if fine:
        nsnr_vals = np.linspace(0.3, 3.0, 10).tolist()
        adt_vals = np.linspace(0.01, 0.20, 10).tolist()
    else:
        nsnr_vals = [0.5, 1.0, 1.5, 2.0]
        adt_vals = [0.03, 0.05, 0.08, 0.10]
    configs = []
    for nsnr in nsnr_vals:
        for adt in adt_vals:
            configs.append(
                {
                    "config_name": f"nsnr={nsnr:.2f}_adt={adt:.3f}",
                    "nnls_detection_snr": float(nsnr),
                    "alias_detection_threshold": float(adt),
                    "require_both": True,
                }
            )
    return configs


def _hybrid_union_configs(fine: bool) -> List[Dict[str, Any]]:
    # Same grid as intersect but with require_both=False
    if fine:
        nsnr_vals = np.linspace(0.3, 3.0, 10).tolist()
        adt_vals = np.linspace(0.01, 0.20, 10).tolist()
    else:
        nsnr_vals = [0.5, 1.0, 1.5, 2.0]
        adt_vals = [0.03, 0.05, 0.08, 0.10]
    configs = []
    for nsnr in nsnr_vals:
        for adt in adt_vals:
            configs.append(
                {
                    "config_name": f"nsnr={nsnr:.2f}_adt={adt:.3f}_union",
                    "nnls_detection_snr": float(nsnr),
                    "alias_detection_threshold": float(adt),
                    "require_both": False,
                }
            )
    return configs


def _forward_model_configs(fine: bool) -> List[Dict[str, Any]]:
    ct_vals = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    cdeg_vals = [2, 3]
    configs = []
    for ct in ct_vals:
        for cdeg in cdeg_vals:
            configs.append(
                {
                    "config_name": f"ct={ct}_cdeg={cdeg}",
                    "concentration_threshold": ct,
                    "continuum_degree": cdeg,
                }
            )
    return configs


def _voigt_alias_configs(fine: bool) -> List[Dict[str, Any]]:
    dt_vals = [0.03, 0.05, 0.08]
    grouping_vals = [1.5, 2.0, 3.0]
    configs = []
    for dt in dt_vals:
        for gf in grouping_vals:
            configs.append(
                {
                    "config_name": f"voigt_dt={dt}_gf={gf}",
                    "detection_threshold": dt,
                    "intensity_threshold_factor": 3.0,
                    "chance_window_scale": 0.4,
                    "max_lines_per_element": 30,
                    "grouping_factor": gf,
                }
            )
    return configs


def _get_configs(pathway: str, fine: bool) -> List[Dict[str, Any]]:
    dispatch = {
        "alias": _alias_configs,
        "spectral_nnls": _spectral_nnls_configs,
        "hybrid_intersect": _hybrid_intersect_configs,
        "hybrid_union": _hybrid_union_configs,
        "forward_model": _forward_model_configs,
        "voigt_alias": _voigt_alias_configs,
    }
    fn = dispatch.get(pathway)
    if fn is None:
        raise ValueError(f"Unknown pathway: {pathway}")
    return fn(fine)


# ---------------------------------------------------------------------------
# HDF5 chunk loading
# ---------------------------------------------------------------------------


def _load_chunk(chunk_path: Path) -> Dict[str, Any]:
    """Load an HDF5 chunk produced by generate_synthetic_benchmark.py."""
    import h5py

    data: Dict[str, Any] = {}
    with h5py.File(str(chunk_path), "r") as f:
        data["spectra"] = f["spectra"][:]  # (N, n_pix)
        data["wavelength"] = f["wavelength"][:]  # (n_pix,)

        # Ground truth elements per spectrum: stored as variable-length or
        # fixed-length string arrays
        if "ground_truth_elements" in f:
            gt = f["ground_truth_elements"]
            if gt.ndim == 2:
                # (N, max_el) fixed-length: each row padded with ''
                data["ground_truth_elements"] = [
                    {
                        (e.decode() if isinstance(e, bytes) else e)
                        for e in row
                        if (e.decode() if isinstance(e, bytes) else e).strip()
                    }
                    for row in gt[:]
                ]
            else:
                # 1-D list of JSON strings
                data["ground_truth_elements"] = [
                    set(json.loads(e.decode() if isinstance(e, bytes) else e)) for e in gt[:]
                ]
        elif "elements_present" in f:
            gt = f["elements_present"]
            if gt.ndim == 2:
                data["ground_truth_elements"] = [
                    {
                        (e.decode() if isinstance(e, bytes) else e)
                        for e in row
                        if (e.decode() if isinstance(e, bytes) else e).strip()
                    }
                    for row in gt[:]
                ]
            else:
                data["ground_truth_elements"] = [
                    set(json.loads(e.decode() if isinstance(e, bytes) else e)) for e in gt[:]
                ]

        # Metadata arrays (may not all be present)
        for key in ["rp", "resolving_power", "snr", "T_K", "temperature", "ne", "n_elements"]:
            if key in f:
                data[key] = f[key][:]

        # Normalise key names
        if "resolving_power" in data and "rp" not in data:
            data["rp"] = data.pop("resolving_power")
        if "temperature" in data and "T_K" not in data:
            data["T_K"] = data.pop("temperature")

        # Candidate element list
        if "candidate_elements" in f.attrs:
            raw = f.attrs["candidate_elements"]
            if isinstance(raw, np.ndarray):
                data["candidate_elements"] = [
                    e.decode() if isinstance(e, bytes) else e for e in raw
                ]
            else:
                data["candidate_elements"] = list(raw)
        elif "elements" in f.attrs:
            raw = f.attrs["elements"]
            if isinstance(raw, np.ndarray):
                data["candidate_elements"] = [
                    e.decode() if isinstance(e, bytes) else e for e in raw
                ]
            else:
                data["candidate_elements"] = list(raw)

    return data


# ---------------------------------------------------------------------------
# Single-spectrum identification runners
# ---------------------------------------------------------------------------


def _run_alias(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    config: Dict[str, Any],
    db_path: str,
    elements: List[str],
    rp: float,
) -> Dict[str, Any]:
    """Run ALIAS identifier on one spectrum."""
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.alias_identifier import ALIASIdentifier

    with AtomicDatabase(db_path) as db:
        identifier = ALIASIdentifier(
            atomic_db=db,
            elements=elements,
            resolving_power=rp,
            intensity_threshold_factor=config["intensity_threshold_factor"],
            detection_threshold=config["detection_threshold"],
            chance_window_scale=config["chance_window_scale"],
            max_lines_per_element=config["max_lines_per_element"],
        )
        result = identifier.identify(wavelength, spectrum)
    return {"detected": {e.element for e in result.detected_elements}}


def _run_spectral_nnls(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    config: Dict[str, Any],
    basis_lib_path: str,
) -> Dict[str, Any]:
    """Run SpectralNNLS identifier on one spectrum."""
    from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier
    from cflibs.manifold.basis_library import BasisLibrary

    with BasisLibrary(basis_lib_path) as basis:
        identifier = SpectralNNLSIdentifier(
            basis_library=basis,
            detection_snr=config["detection_snr"],
            continuum_degree=config["continuum_degree"],
            fallback_T_K=8000.0,
            fallback_ne_cm3=1e17,
        )
        result = identifier.identify(wavelength, spectrum)
    return {"detected": {e.element for e in result.detected_elements}}


def _run_hybrid(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    config: Dict[str, Any],
    db_path: str,
    basis_lib_path: str,
    elements: List[str],
    rp: float,
) -> Dict[str, Any]:
    """Run HybridIdentifier on one spectrum."""
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.hybrid_identifier import HybridIdentifier
    from cflibs.manifold.basis_library import BasisLibrary

    with AtomicDatabase(db_path) as db, BasisLibrary(basis_lib_path) as basis:
        identifier = HybridIdentifier(
            atomic_db=db,
            basis_library=basis,
            elements=elements,
            resolving_power=rp,
            nnls_detection_snr=config["nnls_detection_snr"],
            alias_detection_threshold=config["alias_detection_threshold"],
            require_both=config["require_both"],
        )
        result = identifier.identify(wavelength, spectrum)
    return {"detected": {e.element for e in result.detected_elements}}


def _run_forward_model(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    config: Dict[str, Any],
    basis_lib_path: str,
) -> Dict[str, Any]:
    """Run forward-model (NNLS + concentration thresholding) on one spectrum."""
    from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier
    from cflibs.manifold.basis_library import BasisLibrary

    with BasisLibrary(basis_lib_path) as basis:
        identifier = SpectralNNLSIdentifier(
            basis_library=basis,
            detection_snr=0.0,
            continuum_degree=config["continuum_degree"],
            fallback_T_K=8000.0,
            fallback_ne_cm3=1e17,
        )
        result = identifier.identify(wavelength, spectrum)

    ct = config["concentration_threshold"]
    detected = set()
    for eid in result.all_elements:
        conc = eid.metadata.get("concentration_estimate", 0.0)
        if conc >= ct:
            detected.add(eid.element)
    return {"detected": detected}


def _run_voigt_alias(
    wavelength: np.ndarray,
    spectrum: np.ndarray,
    config: Dict[str, Any],
    db_path: str,
    elements: List[str],
    rp: float,
) -> Dict[str, Any]:
    """Run Voigt deconvolution + ALIAS on one spectrum."""
    from scipy.signal import find_peaks

    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.alias_identifier import ALIASIdentifier
    from cflibs.inversion.deconvolution import deconvolve_peaks
    from cflibs.inversion.preprocessing import estimate_baseline

    baseline = estimate_baseline(wavelength, spectrum)
    corrected = np.maximum(spectrum - baseline, 0.0)

    threshold = np.percentile(corrected[corrected > 0], 70) if np.any(corrected > 0) else 0
    peak_indices, _ = find_peaks(corrected, height=threshold, distance=5)

    if len(peak_indices) > 0:
        peak_wls = wavelength[peak_indices]
        median_wl = float(np.median(wavelength))
        fwhm_est = median_wl / rp

        try:
            deconv = deconvolve_peaks(
                wavelength,
                corrected,
                peak_wls,
                fwhm_est,
                grouping_factor=config.get("grouping_factor", 2.0),
                margin_factor=3.0,
                use_jax=False,
            )
            cleaned = deconv.fitted_spectrum
        except Exception:
            cleaned = corrected
        cleaned = np.maximum(cleaned, 0.0) + np.median(spectrum) * 0.01
    else:
        cleaned = spectrum

    with AtomicDatabase(db_path) as db:
        alias_id = ALIASIdentifier(
            atomic_db=db,
            elements=elements,
            resolving_power=rp,
            intensity_threshold_factor=config["intensity_threshold_factor"],
            detection_threshold=config["detection_threshold"],
            chance_window_scale=config["chance_window_scale"],
            max_lines_per_element=config["max_lines_per_element"],
        )
        result = alias_id.identify(wavelength, cleaned)
    return {"detected": {e.element for e in result.detected_elements}}


# ---------------------------------------------------------------------------
# Per-config evaluation (unit of parallelism)
# ---------------------------------------------------------------------------


def _evaluate_config(
    config: Dict[str, Any],
    pathway: str,
    spectra: np.ndarray,
    wavelength: np.ndarray,
    ground_truth: List[set],
    rp_arr: np.ndarray,
    snr_arr: np.ndarray,
    T_K_arr: np.ndarray,
    ne_arr: np.ndarray,
    n_elements_arr: np.ndarray,
    db_path: str,
    basis_lib_path: Optional[str],
    elements: List[str],
) -> List[Dict[str, Any]]:
    """Evaluate one config across all spectra. Returns a list of row dicts."""
    config_name = config["config_name"]
    n_spectra = spectra.shape[0]
    rows = []

    for i in range(n_spectra):
        wl = wavelength
        sp = spectra[i]
        expected = ground_truth[i]
        rp_val = float(rp_arr[i]) if rp_arr is not None else 1000.0
        snr_val = float(snr_arr[i]) if snr_arr is not None else 0.0
        T_val = float(T_K_arr[i]) if T_K_arr is not None else 0.0
        ne_val = float(ne_arr[i]) if ne_arr is not None else 0.0
        nel_val = int(n_elements_arr[i]) if n_elements_arr is not None else len(expected)

        try:
            if pathway == "alias":
                out = _run_alias(wl, sp, config, db_path, elements, rp_val)
            elif pathway == "spectral_nnls":
                out = _run_spectral_nnls(wl, sp, config, basis_lib_path)
            elif pathway in ("hybrid_intersect", "hybrid_union"):
                out = _run_hybrid(wl, sp, config, db_path, basis_lib_path, elements, rp_val)
            elif pathway == "forward_model":
                out = _run_forward_model(wl, sp, config, basis_lib_path)
            elif pathway == "voigt_alias":
                out = _run_voigt_alias(wl, sp, config, db_path, elements, rp_val)
            else:
                continue
            detected = out["detected"]
        except Exception as exc:
            logger.warning("Config %s spectrum %d failed: %s", config_name, i, exc)
            detected = set()

        searched = set(elements)
        tp = len(detected & expected)
        fp = len(detected - expected)
        fn = len(expected - detected)
        tn = len((searched - expected) - detected)

        rows.append(
            {
                "spectrum_id": i,
                "pathway": pathway,
                "config_name": config_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "detected_elements": json.dumps(sorted(detected)),
                "expected_elements": json.dumps(sorted(expected)),
                "rp": rp_val,
                "snr": snr_val,
                "T_K": T_val,
                "ne": ne_val,
                "n_elements": nel_val,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Worker subcommand
# ---------------------------------------------------------------------------


def worker_main(args: argparse.Namespace) -> None:
    """Process one HDF5 chunk through all configs for a given pathway."""
    chunk_path = Path(args.chunk_path)
    basis_dir = Path(args.basis_dir)
    output_dir = Path(args.output_dir)
    pathway = args.pathway
    fine = args.fine
    db_path = args.db_path
    n_workers = args.n_workers

    if not chunk_path.exists():
        _error_exit(f"Chunk file not found: {chunk_path}")

    logger.info("Loading chunk: %s", chunk_path)
    chunk = _load_chunk(chunk_path)

    spectra = chunk["spectra"]
    wavelength = chunk["wavelength"]
    ground_truth = chunk.get("ground_truth_elements", [set()] * spectra.shape[0])
    elements = chunk.get("candidate_elements", [])

    rp_arr = chunk.get("rp")
    snr_arr = chunk.get("snr")
    T_K_arr = chunk.get("T_K")
    ne_arr = chunk.get("ne")
    n_elements_arr = chunk.get("n_elements")

    # Select basis library by mapping median RP to nearest FWHM
    if rp_arr is not None and len(rp_arr) > 0:
        median_rp = float(np.median(rp_arr))
    else:
        median_rp = 1000.0
    fwhm = _select_basis_fwhm(median_rp)
    basis_path = _find_basis_path(basis_dir, fwhm)
    basis_lib_path = str(basis_path) if basis_path is not None else None

    needs_basis = pathway in (
        "spectral_nnls",
        "hybrid_intersect",
        "hybrid_union",
        "forward_model",
    )
    if needs_basis and basis_lib_path is None:
        _error_exit(
            f"Basis library not found for FWHM={fwhm} nm in {basis_dir}. "
            f"Generate with scripts/hpc/generate_basis_libraries.py first."
        )

    configs = _get_configs(pathway, fine)
    logger.info(
        "Pathway=%s, fine=%s, %d configs, %d spectra",
        pathway,
        fine,
        len(configs),
        spectra.shape[0],
    )

    # Parallelise across configs
    all_rows: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    if n_workers <= 1:
        for cfg in configs:
            rows = _evaluate_config(
                cfg,
                pathway,
                spectra,
                wavelength,
                ground_truth,
                rp_arr,
                snr_arr,
                T_K_arr,
                ne_arr,
                n_elements_arr,
                db_path,
                basis_lib_path,
                elements,
            )
            all_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _evaluate_config,
                    cfg,
                    pathway,
                    spectra,
                    wavelength,
                    ground_truth,
                    rp_arr,
                    snr_arr,
                    T_K_arr,
                    ne_arr,
                    n_elements_arr,
                    db_path,
                    basis_lib_path,
                    elements,
                ): cfg["config_name"]
                for cfg in configs
            }
            for future in as_completed(futures):
                cfg_name = futures[future]
                try:
                    rows = future.result()
                    all_rows.extend(rows)
                except Exception as exc:
                    logger.error("Config %s raised: %s", cfg_name, exc)

    elapsed = time.monotonic() - t0
    logger.info("Evaluated %d rows in %.1f s", len(all_rows), elapsed)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_stem = chunk_path.stem
    out_name = f"results_{pathway}_{chunk_stem}"

    if HAS_PYARROW and all_rows:
        _write_parquet(all_rows, output_dir / f"{out_name}.parquet")
    elif all_rows:
        _write_csv_fallback(all_rows, output_dir / f"{out_name}.csv")
    else:
        logger.warning("No rows produced — nothing to write.")


def _write_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write rows to a Parquet file."""
    table = pa.table(
        {
            "spectrum_id": pa.array([r["spectrum_id"] for r in rows], type=pa.int32()),
            "pathway": pa.array([r["pathway"] for r in rows], type=pa.string()),
            "config_name": pa.array([r["config_name"] for r in rows], type=pa.string()),
            "tp": pa.array([r["tp"] for r in rows], type=pa.int32()),
            "fp": pa.array([r["fp"] for r in rows], type=pa.int32()),
            "fn": pa.array([r["fn"] for r in rows], type=pa.int32()),
            "tn": pa.array([r["tn"] for r in rows], type=pa.int32()),
            "detected_elements": pa.array([r["detected_elements"] for r in rows], type=pa.string()),
            "expected_elements": pa.array([r["expected_elements"] for r in rows], type=pa.string()),
            "rp": pa.array([r["rp"] for r in rows], type=pa.float64()),
            "snr": pa.array([r["snr"] for r in rows], type=pa.float64()),
            "T_K": pa.array([r["T_K"] for r in rows], type=pa.float64()),
            "ne": pa.array([r["ne"] for r in rows], type=pa.float64()),
            "n_elements": pa.array([r["n_elements"] for r in rows], type=pa.int32()),
        }
    )
    pq.write_table(table, str(path))
    logger.info("Wrote %d rows to %s", len(rows), path)


def _write_csv_fallback(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write rows to CSV (fallback when pyarrow unavailable)."""
    import csv

    fieldnames = [
        "spectrum_id",
        "pathway",
        "config_name",
        "tp",
        "fp",
        "fn",
        "tn",
        "detected_elements",
        "expected_elements",
        "rp",
        "snr",
        "T_K",
        "ne",
        "n_elements",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s (CSV fallback)", len(rows), path)


# ---------------------------------------------------------------------------
# Submit subcommand
# ---------------------------------------------------------------------------


def submit_main(args: argparse.Namespace) -> None:
    """Submit SLURM array jobs for each pathway."""
    from cflibs.hpc.slurm import ArrayJobConfig, SlurmJobConfig, SlurmJobManager

    synthetic_dir = Path(args.synthetic_dir)
    basis_dir = Path(args.basis_dir)
    output_dir = Path(args.output_dir)
    db_path = args.db_path
    pathways = args.pathways
    fine = args.fine
    dry_run = args.dry_run

    # Discover chunk files
    chunk_files = sorted(synthetic_dir.glob("chunk_*.h5"))
    if not chunk_files:
        _error_exit(f"No chunk_*.h5 files found in {synthetic_dir}")

    n_chunks = len(chunk_files)
    logger.info("Found %d chunk files in %s", n_chunks, synthetic_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    manager = SlurmJobManager(dry_run=dry_run)
    script_path = Path(__file__).resolve()

    worker_job_ids: List[str] = []

    for pathway in pathways:
        job_name = f"bench_{pathway}"

        array_config = ArrayJobConfig(
            job_name=job_name,
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            mem_gb=args.mem_gb,
            time_limit=args.time_limit,
            output_path=str(output_dir / f"{job_name}_%a.out"),
            error_path=str(output_dir / f"{job_name}_%a.err"),
            array_size=n_chunks,
            max_concurrent=args.max_concurrent if args.max_concurrent else 0,
            env_vars={"JAX_PLATFORMS": "cpu"},
        )

        # Build the worker command: each array task maps SLURM_ARRAY_TASK_ID
        # to a chunk file
        chunk_list_str = " ".join(shlex.quote(str(f)) for f in chunk_files)
        fine_flag = " --fine" if fine else ""

        script_content = f"""\
CHUNKS=({chunk_list_str})
CHUNK_PATH="${{CHUNKS[$SLURM_ARRAY_TASK_ID]}}"

python {shlex.quote(str(script_path))} worker \\
    --chunk-path "$CHUNK_PATH" \\
    --basis-dir {shlex.quote(str(basis_dir))} \\
    --output-dir {shlex.quote(str(output_dir))} \\
    --pathway {shlex.quote(pathway)} \\
    --db-path {shlex.quote(db_path)} \\
    --n-workers {args.cpus_per_task}{fine_flag}
"""

        logger.info("Submitting %s array job (%d tasks)...", pathway, n_chunks)
        job_id = manager.submit(array_config, script_content)
        worker_job_ids.append(job_id)
        print(f"Submitted {pathway}: job {job_id} ({n_chunks} array tasks)")

    # Submit a dependent collect job
    if worker_job_ids and not dry_run:
        collect_config = SlurmJobConfig(
            job_name="bench_collect",
            partition=args.partition,
            cpus_per_task=1,
            mem_gb=max(args.mem_gb, 8),
            time_limit="00:30:00",
            output_path=str(output_dir / "collect.out"),
            error_path=str(output_dir / "collect.err"),
        )
        collect_script = (
            f"python {shlex.quote(str(script_path))} collect "
            f"--output-dir {shlex.quote(str(output_dir))}\n"
        )
        collect_job_id = manager.submit_with_dependency(
            collect_config,
            collect_script,
            depends_on=worker_job_ids,
            dependency_type="afterok",
        )
        print(f"Submitted collect job: {collect_job_id} (depends on all worker jobs)")

    print("\nMonitor with: squeue -u $USER")
    print(f"Check logs in: {output_dir}")


# ---------------------------------------------------------------------------
# Collect subcommand
# ---------------------------------------------------------------------------


def collect_main(args: argparse.Namespace) -> None:
    """Aggregate per-chunk result files into summary statistics."""
    output_dir = Path(args.output_dir)

    # Read all result files (parquet or CSV)
    parquet_files = sorted(output_dir.glob("results_*.parquet"))
    csv_files = sorted(output_dir.glob("results_*.csv"))

    if not parquet_files and not csv_files:
        _error_exit(f"No results_*.parquet or results_*.csv files found in {output_dir}")

    all_rows: List[Dict[str, Any]] = []

    if parquet_files and HAS_PYARROW:
        for pf in parquet_files:
            table = pq.read_table(str(pf))
            for row_dict in table.to_pydict().items():
                pass  # handled below
            # Convert columnar to row-oriented
            col_dict = table.to_pydict()
            n_rows = len(col_dict["spectrum_id"])
            for i in range(n_rows):
                all_rows.append({k: col_dict[k][i] for k in col_dict})
        logger.info("Read %d rows from %d parquet files", len(all_rows), len(parquet_files))
    elif csv_files:
        import csv

        for cf in csv_files:
            with cf.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    for k in ("spectrum_id", "tp", "fp", "fn", "tn", "n_elements"):
                        if k in row:
                            row[k] = int(row[k])
                    for k in ("rp", "snr", "T_K", "ne"):
                        if k in row:
                            row[k] = float(row[k])
                    all_rows.append(row)
        logger.info("Read %d rows from %d CSV files", len(all_rows), len(csv_files))

    if not all_rows:
        _error_exit("No data rows found in result files.")

    # Compute micro-averaged metrics per (pathway, config_name, rp, snr)
    from collections import defaultdict

    # Bin RP and SNR for aggregation
    def _rp_bin(rp: float) -> str:
        for boundary in [200, 300, 500, 700, 1000, 2000, 3000, 5000, 10000]:
            if rp <= boundary * 1.2:
                return str(boundary)
        return "10000+"

    def _snr_bin(snr: float) -> str:
        if snr <= 0:
            return "unknown"
        for boundary in [10, 20, 30, 40, 50, 100]:
            if snr <= boundary:
                return f"<={boundary}"
        return ">100"

    groups: Dict[Tuple[str, str, str, str], Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "count": 0}
    )

    for row in all_rows:
        key = (
            row["pathway"],
            row["config_name"],
            _rp_bin(row.get("rp", 0)),
            _snr_bin(row.get("snr", 0)),
        )
        groups[key]["tp"] += row["tp"]
        groups[key]["fp"] += row["fp"]
        groups[key]["fn"] += row["fn"]
        groups[key]["tn"] += row["tn"]
        groups[key]["count"] += 1

    # Build summary rows
    summary_rows = []
    for (pathway, config_name, rp_bin, snr_bin), counts in groups.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        fpr = _safe_ratio(fp, fp + tn)

        summary_rows.append(
            {
                "pathway": pathway,
                "config_name": config_name,
                "rp_bin": rp_bin,
                "snr_bin": snr_bin,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "fpr": round(fpr, 6),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "n_spectra": counts["count"],
            }
        )

    # Determine coarse vs fine by checking config count
    has_fine = any("fine" in str(args.output_dir).lower() for _ in [1])
    # Heuristic: if more than 50 unique configs, likely fine sweep
    unique_configs = len({r["config_name"] for r in summary_rows})
    is_fine = unique_configs > 50 or has_fine

    summary_name = "fine_summary" if is_fine else "coarse_summary"

    if HAS_PYARROW:
        summary_table = pa.table({k: [r[k] for r in summary_rows] for k in summary_rows[0].keys()})
        summary_path = output_dir / f"{summary_name}.parquet"
        pq.write_table(summary_table, str(summary_path))
        print(f"Wrote summary: {summary_path} ({len(summary_rows)} rows)")
    else:
        import csv

        summary_path = output_dir / f"{summary_name}.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Wrote summary: {summary_path} ({len(summary_rows)} rows)")

    # Find optimal config per pathway per RP bin (best F1)
    optimal: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in summary_rows:
        pw = row["pathway"]
        rp = row["rp_bin"]
        if pw not in optimal:
            optimal[pw] = {}
        if rp not in optimal[pw] or row["f1"] > optimal[pw][rp].get("f1", 0):
            optimal[pw][rp] = {
                "config_name": row["config_name"],
                "f1": row["f1"],
                "precision": row["precision"],
                "recall": row["recall"],
                "fpr": row["fpr"],
                "n_spectra": row["n_spectra"],
            }

    optimal_path = output_dir / "optimal_configs.json"
    optimal_path.write_text(json.dumps(optimal, indent=2))
    print(f"Wrote optimal configs: {optimal_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print("OPTIMAL CONFIGS PER PATHWAY PER RP BIN")
    print(f"{'='*100}")
    print(
        f"{'Pathway':<20} {'RP Bin':<10} {'Config':<35} "
        f"{'P':>6} {'R':>6} {'F1':>6} {'FPR':>6} {'N':>6}"
    )
    print(f"{'-'*100}")
    for pw in sorted(optimal.keys()):
        for rp in sorted(optimal[pw].keys(), key=lambda x: int(x.rstrip("+"))):
            o = optimal[pw][rp]
            print(
                f"{pw:<20} {rp:<10} {o['config_name']:<35} "
                f"{o['precision']:>6.3f} {o['recall']:>6.3f} "
                f"{o['f1']:>6.3f} {o['fpr']:>6.3f} {o['n_spectra']:>6}"
            )
    print(f"{'='*100}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run element identification benchmarks on synthetic LIBS "
        "spectra, parallelized via SLURM array jobs.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # --- worker ---
    wp = subparsers.add_parser(
        "worker", help="Process one HDF5 chunk through all configs for a pathway."
    )
    wp.add_argument("--chunk-path", type=str, required=True, help="HDF5 chunk file path")
    wp.add_argument(
        "--basis-dir",
        type=str,
        required=True,
        help="Directory with basis_fwhm_X.Xnm.h5 files",
    )
    wp.add_argument("--output-dir", type=str, required=True, help="Output directory")
    wp.add_argument(
        "--pathway",
        type=str,
        required=True,
        choices=ALL_PATHWAYS,
        help="Identification pathway to run",
    )
    wp.add_argument(
        "--fine",
        action="store_true",
        help="Use fine sweep with dense parameter grids",
    )
    wp.add_argument(
        "--db-path",
        type=str,
        default="ASD_da/libs_production.db",
        help="Atomic database path",
    )
    wp.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Intra-node parallelism via concurrent.futures (default: 4)",
    )

    # --- submit ---
    sp = subparsers.add_parser("submit", help="Submit SLURM array jobs.")
    sp.add_argument(
        "--synthetic-dir",
        type=str,
        required=True,
        help="Directory with chunk_*.h5 files",
    )
    sp.add_argument(
        "--basis-dir",
        type=str,
        required=True,
        help="Directory with basis_fwhm_X.Xnm.h5 files",
    )
    sp.add_argument("--output-dir", type=str, required=True, help="Output directory")
    sp.add_argument(
        "--db-path",
        type=str,
        default="ASD_da/libs_production.db",
        help="Atomic database path",
    )
    sp.add_argument("--partition", type=str, default="default", help="SLURM partition")
    sp.add_argument("--mem-gb", type=int, default=8, help="Memory per task (GB)")
    sp.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per task (default: 4)")
    sp.add_argument("--time-limit", type=str, default="04:00:00", help="Time limit (HH:MM:SS)")
    sp.add_argument(
        "--max-concurrent",
        type=int,
        default=0,
        help="Max concurrent array tasks (0=unlimited)",
    )
    sp.add_argument(
        "--pathways",
        nargs="+",
        default=ALL_PATHWAYS,
        choices=ALL_PATHWAYS,
        help="Pathways to benchmark",
    )
    sp.add_argument(
        "--fine",
        action="store_true",
        help="Use fine sweep with dense parameter grids",
    )
    sp.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SLURM scripts without submitting",
    )

    # --- collect ---
    cp = subparsers.add_parser(
        "collect", help="Aggregate per-chunk results into summary statistics."
    )
    cp.add_argument("--output-dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.subcommand == "worker":
        worker_main(args)
    elif args.subcommand == "submit":
        submit_main(args)
    elif args.subcommand == "collect":
        collect_main(args)


if __name__ == "__main__":
    main()
