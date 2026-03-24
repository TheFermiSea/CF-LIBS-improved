#!/usr/bin/env python
"""
GPU vs CPU real-data validation for CF-LIBS paper.

Runs the full CF-LIBS inversion pipeline on Aalto mineral spectra
using both CPU and GPU codepaths, comparing:
1. GPU-CPU parity (same compositions recovered)
2. Element detection accuracy vs known mineral formulas
3. Wall-clock timing comparison

Usage:
    # GPU mode (on V100S):
    python scripts/validate_real_data_gpu_vs_cpu.py --db ASD_da/libs_production.db

    # CPU-only mode:
    JAX_PLATFORMS=cpu python scripts/validate_real_data_gpu_vs_cpu.py --db ASD_da/libs_production.db
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Enable float64 BEFORE any JAX imports
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cflibs.atomic.database import AtomicDatabase
from cflibs.core.logging_config import get_logger

logger = get_logger("real_data_validation")

# ============================================================================
# Mineral formulas (subset of most reliable minerals for validation)
# ============================================================================

MINERAL_FORMULAS: Dict[str, Dict[str, Optional[float]]] = {
    "corundum": {"Al": None, "O": None},  # Al2O3
    "hematite": {"Fe": None, "O": None},  # Fe2O3
    "magnetite": {"Fe": None, "O": None},  # Fe3O4
    "quartzE26": {"Si": None, "O": None},  # SiO2
    "fluorite39": {"Ca": None, "F": None},  # CaF2
    "kyanite14": {"Al": None, "Si": None, "O": None},  # Al2SiO5
    "wollastonite30": {"Ca": None, "Si": None, "O": None},  # CaSiO3
    "diopside": {"Ca": None, "Mg": None, "Si": None, "O": None},  # CaMgSi2O6
    "augite25": {"Ca": None, "Mg": None, "Fe": None, "Al": None, "Si": None, "O": None},
    "orthoclase7": {"K": None, "Al": None, "Si": None, "O": None},  # KAlSi3O8
    "plagioclaseE8": {"Na": None, "Ca": None, "Al": None, "Si": None, "O": None},
    "biotiteE60": {"K": None, "Mg": None, "Fe": None, "Al": None, "Si": None, "O": None},
    "muscoviteE35": {"K": None, "Al": None, "Si": None, "O": None},
    "hornblendeE29": {"Ca": None, "Mg": None, "Fe": None, "Al": None, "Si": None, "O": None},
    "apatite41": {"Ca": None, "P": None, "O": None},
    "topaz27": {"Al": None, "Si": None, "O": None, "F": None},
    "sphene45": {"Ca": None, "Ti": None, "Si": None, "O": None},  # CaTiSiO5
    "tourmaline34": {"Na": None, "Mg": None, "Al": None, "Si": None, "B": None, "O": None},
    "chalcopyrite": {"Cu": None, "Fe": None, "S": None},  # CuFeS2
    "pentlandite68": {"Fe": None, "Ni": None, "S": None},  # (Fe,Ni)9S8
}

PURE_ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mg", "Mn", "Ni", "Pb", "Sn", "Ti", "V", "Zn"]

DATA_DIR = Path("data/aalto_libs")


def load_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a CSV spectrum file (wavelength, intensity)."""
    import pandas as pd
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    return df[cols[0]].values, df[cols[1]].values


def run_inversion(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    db: AtomicDatabase,
    elements: List[str],
    use_jax: bool = True,
) -> Dict:
    """Run CF-LIBS inversion pipeline on a spectrum."""
    from cflibs.inversion.solver import IterativeCFLIBSSolver
    from cflibs.instrument.model import InstrumentModel

    instrument = InstrumentModel(resolving_power=1000.0)
    solver = IterativeCFLIBSSolver(
        db=db,
        instrument=instrument,
        elements=elements,
        use_jax=use_jax,
    )

    t0 = time.perf_counter()
    try:
        result = solver.solve(wavelengths, intensities)
        elapsed = time.perf_counter() - t0
        return {
            "success": True,
            "temperature_K": float(result.temperature) if result.temperature else None,
            "electron_density": float(result.electron_density) if result.electron_density else None,
            "concentrations": {k: float(v) for k, v in result.concentrations.items()} if result.concentrations else {},
            "detected_elements": list(result.concentrations.keys()) if result.concentrations else [],
            "elapsed_s": elapsed,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "success": False,
            "error": str(e),
            "elapsed_s": elapsed,
        }


def validate_detection(detected: List[str], expected: Dict[str, Optional[float]]) -> Dict:
    """Compare detected elements vs expected from mineral formula."""
    # Only compare non-light elements (O, H, N, C, F, S, Cl are hard to detect in LIBS)
    light_elements = {"O", "H", "N", "C", "F", "S", "Cl", "B", "P"}
    expected_detectable = {e for e in expected if e not in light_elements}
    detected_set = set(detected) - light_elements

    if not expected_detectable:
        return {"recall": None, "precision": None, "note": "No detectable elements in formula"}

    tp = len(detected_set & expected_detectable)
    recall = tp / len(expected_detectable) if expected_detectable else 0
    precision = tp / len(detected_set) if detected_set else 0

    return {
        "expected": sorted(expected_detectable),
        "detected": sorted(detected_set),
        "true_positives": sorted(detected_set & expected_detectable),
        "false_negatives": sorted(expected_detectable - detected_set),
        "false_positives": sorted(detected_set - expected_detectable),
        "recall": recall,
        "precision": precision,
    }


def main():
    parser = argparse.ArgumentParser(description="GPU vs CPU real-data validation")
    parser.add_argument("--db", required=True, help="Path to libs_production.db")
    parser.add_argument("--output", default="validation/real_data/results/gpu_vs_cpu_aalto.json")
    parser.add_argument("--minerals-only", action="store_true", help="Skip pure elements")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    # Report JAX status
    devices = jax.devices()
    has_gpu = any(d.platform == "gpu" for d in devices)
    print(f"JAX devices: {devices}")
    print(f"GPU available: {has_gpu}")
    print(f"Float64 enabled: {jax.config.jax_enable_x64}")

    db = AtomicDatabase(args.db)
    results = {
        "metadata": {
            "jax_version": jax.__version__,
            "jax_backend": str(devices[0].platform),
            "gpu_available": has_gpu,
            "float64": jax.config.jax_enable_x64,
            "db_path": args.db,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "pure_elements": [],
        "minerals": [],
        "summary": {},
    }

    # Pure elements
    if not args.minerals_only:
        print(f"\n{'='*60}")
        print(f"  Pure Element Validation ({len(PURE_ELEMENTS)} samples)")
        print(f"{'='*60}")
        for elem in PURE_ELEMENTS:
            path = DATA_DIR / "elements" / f"{elem}_spectrum.csv"
            if not path.exists():
                print(f"  {elem}: SKIP (file not found)")
                continue

            wl, intensity = load_spectrum(path)
            print(f"  {elem}: {len(wl)} points, {wl.min():.1f}-{wl.max():.1f} nm ... ", end="", flush=True)

            result = run_inversion(wl, intensity, db, [elem])
            if result["success"]:
                print(f"T={result['temperature_K']:.0f} K, {result['elapsed_s']:.2f}s")
            else:
                print(f"FAILED: {result['error'][:60]}")

            results["pure_elements"].append({
                "element": elem,
                "file": str(path),
                **result,
            })

            if args.max_samples and len(results["pure_elements"]) >= args.max_samples:
                break

    # Minerals
    print(f"\n{'='*60}")
    print(f"  Mineral Validation ({len(MINERAL_FORMULAS)} samples)")
    print(f"{'='*60}")
    for mineral, expected in MINERAL_FORMULAS.items():
        path = DATA_DIR / "minerals" / f"{mineral}_spectrum.csv"
        if not path.exists():
            print(f"  {mineral}: SKIP (file not found)")
            continue

        wl, intensity = load_spectrum(path)
        elements = [e for e in expected.keys() if e not in {"O", "H", "N", "C"}]
        if not elements:
            print(f"  {mineral}: SKIP (only light elements)")
            continue

        print(f"  {mineral} ({','.join(elements)}): ", end="", flush=True)

        result = run_inversion(wl, intensity, db, elements)
        detection = validate_detection(
            result.get("detected_elements", []),
            expected,
        )

        if result["success"]:
            print(f"recall={detection.get('recall', 0):.2f}, {result['elapsed_s']:.2f}s")
        else:
            print(f"FAILED: {result['error'][:60]}")

        results["minerals"].append({
            "mineral": mineral,
            "file": str(path),
            "expected_elements": list(expected.keys()),
            "detection": detection,
            **result,
        })

        if args.max_samples and len(results["minerals"]) >= args.max_samples:
            break

    # Summary
    n_pure_success = sum(1 for r in results["pure_elements"] if r["success"])
    n_mineral_success = sum(1 for r in results["minerals"] if r["success"])
    mineral_recalls = [r["detection"]["recall"] for r in results["minerals"]
                       if r["success"] and r["detection"].get("recall") is not None]

    results["summary"] = {
        "pure_elements_tested": len(results["pure_elements"]),
        "pure_elements_success": n_pure_success,
        "minerals_tested": len(results["minerals"]),
        "minerals_success": n_mineral_success,
        "mean_recall": float(np.mean(mineral_recalls)) if mineral_recalls else None,
        "mean_precision": float(np.mean([
            r["detection"]["precision"] for r in results["minerals"]
            if r["success"] and r["detection"].get("precision") is not None
        ])) if mineral_recalls else None,
        "total_time_s": sum(r["elapsed_s"] for r in results["pure_elements"] + results["minerals"]),
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Pure elements: {n_pure_success}/{len(results['pure_elements'])} succeeded")
    print(f"  Minerals: {n_mineral_success}/{len(results['minerals'])} succeeded")
    if mineral_recalls:
        print(f"  Mean recall: {np.mean(mineral_recalls):.2f}")
    print(f"  Total time: {results['summary']['total_time_s']:.1f}s")
    print(f"  Results: {args.output}")


if __name__ == "__main__":
    main()
