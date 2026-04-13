#!/usr/bin/env python
"""
Comprehensive algorithm benchmark for CF-LIBS accuracy evaluation.

Generates hundreds of physics-based synthetic spectra using the full
forward model + atomic database, and also tests against real lab spectra
(AA1100 Al alloy, Ti-6Al-4V) with manufacturer-certified compositions.

This is the definitive experiment for selecting the champion pipeline.

Usage:
    python scripts/run_comprehensive_benchmark.py --db libs_production.db
    python scripts/run_comprehensive_benchmark.py --db libs_production.db --n-compositions 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.benchmark.composition_metrics import aitchison_distance  # noqa: E402
from cflibs.core.constants import EV_TO_K  # noqa: E402
from cflibs.core.logging_config import get_logger  # noqa: E402
from cflibs.inversion.boltzmann import BoltzmannPlotFitter, FitMethod, LineObservation  # noqa: E402
from cflibs.inversion.closure import ClosureEquation  # noqa: E402
from cflibs.inversion.solver import IterativeCFLIBSSolver  # noqa: E402
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver  # noqa: E402
from cflibs.radiation.spectrum_model import SpectrumModel  # noqa: E402
from cflibs.plasma.state import SingleZoneLTEPlasma  # noqa: E402
from cflibs.instrument.model import InstrumentModel  # noqa: E402

logger = get_logger("comprehensive_benchmark")

# ============================================================================
# Known compositions for real samples
# ============================================================================

# AA1100 aluminum alloy (99.0% min Al, balance is Si+Fe+Cu+Mn+Zn)
AA1100_COMPOSITION = {
    "Al": 0.990,
    "Si": 0.004,
    "Fe": 0.004,
    "Cu": 0.001,
    "Mn": 0.001,
}

# Ti-6Al-4V (Grade 5 titanium alloy)
TI6AL4V_COMPOSITION = {
    "Ti": 0.895,
    "Al": 0.061,
    "V": 0.040,
    "Fe": 0.004,
}

# Typical steel compositions for synthetic generation
STEEL_COMPOSITIONS = [
    {"Fe": 0.700, "Cr": 0.180, "Ni": 0.080, "Mn": 0.020, "Si": 0.010, "C": 0.010},
    {"Fe": 0.960, "Mn": 0.015, "Si": 0.010, "Cu": 0.005, "Cr": 0.005, "Ni": 0.005},
    {"Fe": 0.850, "Cr": 0.100, "Ni": 0.030, "Mo": 0.010, "Mn": 0.010},
]

# Geological compositions (basalt, granite, etc.)
GEO_COMPOSITIONS = [
    {"Si": 0.250, "Al": 0.080, "Fe": 0.060, "Ca": 0.050, "Mg": 0.040, "Na": 0.020},
    {"Si": 0.330, "Al": 0.070, "K": 0.040, "Na": 0.030, "Fe": 0.020, "Ca": 0.010},
    {"Si": 0.200, "Mg": 0.120, "Fe": 0.090, "Ca": 0.060, "Al": 0.030},
]


# ============================================================================
# Forward model spectrum generation
# ============================================================================


def generate_forward_model_spectra(
    db: AtomicDatabase,
    compositions: List[Dict[str, float]],
    temperatures_K: List[float],
    electron_densities: List[float],
    wavelength_range: Tuple[float, float] = (240.0, 400.0),
    delta_lambda: float = 0.02,
    noise_snr: Optional[float] = 100.0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate physics-based synthetic spectra with known ground truth.

    Uses the full Saha-Boltzmann forward model with the atomic database
    to produce realistic synthetic spectra at various plasma conditions.
    """
    rng = np.random.default_rng(seed)
    spectra = []

    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    for comp in compositions:
        elements = list(comp.keys())
        # Normalize composition
        total = sum(comp.values())
        norm_comp = {el: c / total for el, c in comp.items()}

        for T_K in temperatures_K:
            for n_e in electron_densities:
                # Convert mass fractions to number densities (approximate)
                total_density = n_e * 10  # rough total atom density

                try:
                    species = {}
                    for el, frac in norm_comp.items():
                        species[el] = frac * total_density

                    plasma = SingleZoneLTEPlasma(T_e=T_K, n_e=n_e, species=species)

                    model = SpectrumModel(
                        plasma=plasma,
                        atomic_db=db,
                        instrument=instrument,
                        lambda_min=wavelength_range[0],
                        lambda_max=wavelength_range[1],
                        delta_lambda=delta_lambda,
                    )

                    wavelength, intensity = model.compute_spectrum()

                    # Add realistic noise
                    if noise_snr is not None and intensity.max() > 0:
                        noise_std = intensity.max() / noise_snr
                        intensity = intensity + rng.normal(0, noise_std, len(intensity))
                        intensity = np.maximum(intensity, 0)

                    spectra.append(
                        {
                            "wavelength": wavelength,
                            "intensity": intensity,
                            "ground_truth": dict(norm_comp),
                            "T_K": T_K,
                            "n_e": n_e,
                            "label": f"synth_T{T_K:.0f}_ne{n_e:.0e}_{'-'.join(elements)}",
                            "source": "forward_model",
                        }
                    )

                except Exception as e:
                    logger.warning("Failed to generate spectrum: %s", e)
                    continue

    return spectra


# ============================================================================
# Real data loading
# ============================================================================


def load_real_spectra(data_dir: Path) -> List[Dict[str, Any]]:
    """Load real experimental spectra with known compositions."""
    import h5py

    spectra = []

    # AA1100 Aluminum
    aa_path = data_dir / "AA1100_Substrate.h5"
    if aa_path.exists():
        with h5py.File(aa_path, "r") as h:
            wl = h["coords/elem_001_Wavelength/values"][:]
            data = h["data/values"][:]
            for i in range(data.shape[0]):
                spectra.append(
                    {
                        "wavelength": wl,
                        "intensity": data[i],
                        "ground_truth": AA1100_COMPOSITION,
                        "T_K": None,
                        "n_e": None,
                        "label": f"AA1100_shot{i}",
                        "source": "experimental",
                    }
                )
        logger.info("Loaded %d AA1100 spectra", data.shape[0])

    # Ti-6Al-4V
    ti_path = data_dir / "Ti6Al4V_substrate.h5"
    if ti_path.exists():
        with h5py.File(ti_path, "r") as h:
            wl = h["coords/elem_001_Wavelength/values"][:]
            data = h["data/values"][:]
            for i in range(data.shape[0]):
                spectra.append(
                    {
                        "wavelength": wl,
                        "intensity": data[i],
                        "ground_truth": TI6AL4V_COMPOSITION,
                        "T_K": None,
                        "n_e": None,
                        "label": f"Ti6Al4V_shot{i}",
                        "source": "experimental",
                    }
                )
        logger.info("Loaded %d Ti6Al4V spectra", data.shape[0])

    # 20-shot averaged
    shot_path = data_dir / "20shot.h5"
    if shot_path.exists():
        with h5py.File(shot_path, "r") as h:
            wl = h["coords/elem_002_Wavelength/values"][:]
            data = h["data/values"][:]
            # Average over shots, use each spatial position
            for x in range(data.shape[0]):
                avg = data[x, 0, :, :].mean(axis=-1)  # Average 20 shots
                spectra.append(
                    {
                        "wavelength": wl,
                        "intensity": avg,
                        "ground_truth": None,  # Unknown composition
                        "T_K": None,
                        "n_e": None,
                        "label": f"20shot_pos{x}",
                        "source": "experimental_blind",
                    }
                )
        logger.info("Loaded %d 20-shot spectra", data.shape[0])

    return spectra


# ============================================================================
# Pipeline definitions
# ============================================================================


def _detect_peaks_simple(
    wavelength: np.ndarray, intensity: np.ndarray, threshold_frac: float = 0.02
) -> List[Dict[str, float]]:
    """Simple peak detection: local maxima above threshold."""
    if intensity.max() <= 0:
        return []
    threshold = threshold_frac * intensity.max()
    peaks = []
    for i in range(1, len(intensity) - 1):
        if (
            intensity[i] > intensity[i - 1]
            and intensity[i] > intensity[i + 1]
            and intensity[i] > threshold
        ):
            peaks.append({"wavelength_nm": float(wavelength[i]), "intensity": float(intensity[i])})
    return peaks


def _match_peaks_to_db(
    peaks: List[Dict[str, float]],
    db: AtomicDatabase,
    elements: List[str],
    tolerance_nm: float = 0.05,
) -> List[LineObservation]:
    """Match detected peaks to database lines and build LineObservation list."""
    observations = []
    for el in elements:
        transitions = db.get_transitions(el, wavelength_min=230.0, wavelength_max=950.0)
        for peak in peaks:
            wl = peak["wavelength_nm"]
            best_dist = tolerance_nm
            best_trans = None
            for t in transitions:
                d = abs(wl - t.wavelength_nm)
                if d < best_dist:
                    best_dist = d
                    best_trans = t
            if best_trans is not None:
                observations.append(
                    LineObservation(
                        wavelength_nm=best_trans.wavelength_nm,
                        intensity=peak["intensity"],
                        intensity_uncertainty=max(peak["intensity"] * 0.05, 1.0),
                        element=el,
                        ionization_stage=best_trans.ionization_stage,
                        E_k_ev=best_trans.E_k_ev,
                        g_k=best_trans.g_k,
                        A_ki=best_trans.A_ki,
                    )
                )
    return observations


def run_boltzmann_pipeline(
    spectrum: Dict,
    db: AtomicDatabase,
    fit_method: FitMethod = FitMethod.SIGMA_CLIP,
    closure_mode: str = "standard",
    elements: Optional[List[str]] = None,
) -> Optional[Dict[str, float]]:
    """Run full CF-LIBS pipeline: peak detect → DB match → Boltzmann → closure.

    Returns recovered concentrations or None on failure.
    """
    wavelength = spectrum["wavelength"]
    intensity = spectrum["intensity"]
    gt = spectrum["ground_truth"]

    if gt is None:
        return None
    if elements is None:
        elements = list(gt.keys())

    try:
        # 1. Detect peaks
        peaks = _detect_peaks_simple(wavelength, intensity)
        if len(peaks) < 3:
            return None

        # 2. Match peaks to database lines
        observations = _match_peaks_to_db(peaks, db, elements)
        if len(observations) < 3:
            return None

        # 3. Run iterative solver
        solver = IterativeCFLIBSSolver(atomic_db=db)
        result = solver.solve(observations, closure_mode=closure_mode)

        if result is not None and result.concentrations:
            return result.concentrations

    except Exception as e:
        logger.debug("Iterative solver failed: %s", e)

    # Fallback: manual per-element Boltzmann + closure
    try:
        peaks = _detect_peaks_simple(wavelength, intensity)
        observations = _match_peaks_to_db(peaks, db, elements)
        if len(observations) < 2:
            return None

        fitter = BoltzmannPlotFitter(method=fit_method)

        element_intercepts: Dict[str, float] = {}
        temperatures: List[float] = []

        for el in elements:
            el_obs = [o for o in observations if o.element == el]
            if len(el_obs) < 2:
                continue
            fit = fitter.fit(el_obs)
            if fit.temperature_K > 0 and np.isfinite(fit.temperature_K):
                temperatures.append(fit.temperature_K)
                element_intercepts[el] = fit.intercept

        if not element_intercepts:
            return None

        T_K = float(np.mean(temperatures)) if temperatures else 10000.0
        T_eV = T_K / EV_TO_K

        # Get partition functions
        saha_solver = SahaBoltzmannSolver(db)
        partition_funcs = {}
        for el in element_intercepts:
            U = saha_solver.calculate_partition_function(el, 1, T_eV)
            partition_funcs[el] = U

        # Apply closure
        if closure_mode == "ilr":
            cr = ClosureEquation.apply_ilr(element_intercepts, partition_funcs)
        else:
            cr = ClosureEquation.apply_standard(element_intercepts, partition_funcs)

        if cr is None:
            return None

        conc = dict(cr.concentrations)
        # Fill missing elements
        for el in elements:
            if el not in conc:
                conc[el] = 0.0
        total = sum(conc.values())
        if total > 0:
            conc = {el: v / total for el, v in conc.items()}
        return conc

    except Exception as e:
        logger.debug("Fallback pipeline failed: %s", e)
        return None


# ============================================================================
# Benchmark runner
# ============================================================================


@dataclass
class BenchmarkResult:
    pipeline_name: str
    n_spectra: int
    n_succeeded: int
    n_failed: int
    aitchison_distances: List[float]
    rmse_values: List[float]
    per_element_errors: Dict[str, List[float]]
    elapsed_seconds: float
    details: List[Dict] = field(default_factory=list)

    @property
    def mean_aitchison(self) -> float:
        return (
            float(np.mean(self.aitchison_distances)) if self.aitchison_distances else float("inf")
        )

    @property
    def median_aitchison(self) -> float:
        return (
            float(np.median(self.aitchison_distances)) if self.aitchison_distances else float("inf")
        )

    @property
    def p95_aitchison(self) -> float:
        return (
            float(np.percentile(self.aitchison_distances, 95))
            if self.aitchison_distances
            else float("inf")
        )

    @property
    def success_rate(self) -> float:
        return self.n_succeeded / self.n_spectra if self.n_spectra > 0 else 0.0


def evaluate_pipeline(
    name: str,
    spectra: List[Dict],
    db: AtomicDatabase,
    fit_method: FitMethod = FitMethod.SIGMA_CLIP,
    closure_mode: str = "standard",
) -> BenchmarkResult:
    """Run a pipeline on all spectra and compute accuracy metrics."""
    ait_dists = []
    rmse_vals = []
    per_el_errors: Dict[str, List[float]] = {}
    n_fail = 0
    details = []

    start = time.perf_counter()

    for spec in spectra:
        gt = spec["ground_truth"]
        if gt is None:
            continue

        elements = list(gt.keys())
        recovered = run_boltzmann_pipeline(
            spec, db, fit_method=fit_method, closure_mode=closure_mode, elements=elements
        )

        if recovered is None:
            n_fail += 1
            details.append({"label": spec["label"], "status": "failed"})
            continue

        # Compute Aitchison distance (expects dicts)
        ait = float(aitchison_distance(gt, recovered))

        # RMSE
        all_els = sorted(set(gt.keys()) | set(recovered.keys()))
        gt_arr = np.array([gt.get(el, 0.0) for el in all_els])
        rec_arr = np.array([recovered.get(el, 0.0) for el in all_els])
        rmse = float(np.sqrt(np.mean((gt_arr - rec_arr) ** 2)))

        ait_dists.append(ait)
        rmse_vals.append(rmse)

        # Per-element errors
        for el in elements:
            err = abs(recovered.get(el, 0.0) - gt.get(el, 0.0))
            per_el_errors.setdefault(el, []).append(err)

        details.append(
            {
                "label": spec["label"],
                "status": "ok",
                "aitchison": ait,
                "rmse": rmse,
                "recovered": recovered,
                "ground_truth": gt,
            }
        )

    elapsed = time.perf_counter() - start
    n_with_gt = sum(1 for s in spectra if s["ground_truth"] is not None)

    return BenchmarkResult(
        pipeline_name=name,
        n_spectra=n_with_gt,
        n_succeeded=len(ait_dists),
        n_failed=n_fail,
        aitchison_distances=ait_dists,
        rmse_values=rmse_vals,
        per_element_errors=per_el_errors,
        elapsed_seconds=elapsed,
        details=details,
    )


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CF-LIBS benchmark")
    parser.add_argument("--db", default="libs_production.db")
    parser.add_argument(
        "--n-compositions", type=int, default=20, help="Number of random compositions to generate"
    )
    parser.add_argument("--output-dir", default="output/comprehensive_benchmark")
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = AtomicDatabase(args.db)

    # ---- Collect spectra ----
    all_spectra: List[Dict] = []

    # 1. Real experimental spectra
    print("Loading real experimental spectra...")
    real = load_real_spectra(Path("data"))
    real_with_gt = [s for s in real if s["ground_truth"] is not None]
    all_spectra.extend(real_with_gt)
    print(f"  {len(real_with_gt)} real spectra with known compositions")

    # 2. Physics-based synthetic spectra
    if not args.skip_synthetic:
        print("Generating forward-model synthetic spectra...")

        # Random compositions
        rng = np.random.default_rng(42)
        random_comps = []
        available_elements = [
            "Fe",
            "Cu",
            "Ni",
            "Cr",
            "Ti",
            "Al",
            "Mn",
            "Si",
            "Ca",
            "Mg",
            "Co",
            "V",
            "W",
            "Na",
            "Ba",
        ]

        for i in range(args.n_compositions):
            n_el = rng.integers(2, 6)
            els = rng.choice(available_elements, size=n_el, replace=False)
            alphas = rng.dirichlet(np.ones(n_el) * 0.5)
            random_comps.append({el: float(a) for el, a in zip(els, alphas)})

        all_compositions = STEEL_COMPOSITIONS + GEO_COMPOSITIONS + random_comps
        temperatures = [8000.0, 10000.0, 12000.0, 15000.0]
        densities = [1e16, 1e17]

        synth = generate_forward_model_spectra(
            db,
            all_compositions,
            temperatures,
            densities,
            wavelength_range=(240.0, 400.0),
            delta_lambda=0.02,
            noise_snr=50.0,
            seed=42,
        )
        all_spectra.extend(synth)
        print(f"  {len(synth)} synthetic spectra generated")

    print(f"\nTotal spectra for benchmark: {len(all_spectra)}")

    # ---- Run pipelines ----
    pipelines = [
        ("SIGMA_CLIP+standard", FitMethod.SIGMA_CLIP, "standard"),
        ("SIGMA_CLIP+ilr", FitMethod.SIGMA_CLIP, "ilr"),
        ("RANSAC+standard", FitMethod.RANSAC, "standard"),
        ("RANSAC+ilr", FitMethod.RANSAC, "ilr"),
        ("HUBER+standard", FitMethod.HUBER, "standard"),
        ("HUBER+ilr", FitMethod.HUBER, "ilr"),
    ]

    results = []
    for name, fit_method, closure_mode in pipelines:
        print(f"\nRunning pipeline: {name} ...")
        result = evaluate_pipeline(name, all_spectra, db, fit_method, closure_mode)
        results.append(result)
        print(
            f"  {result.n_succeeded}/{result.n_spectra} succeeded "
            f"({result.success_rate:.0%}), "
            f"Ait: mean={result.mean_aitchison:.4f} "
            f"med={result.median_aitchison:.4f} "
            f"p95={result.p95_aitchison:.4f} "
            f"({result.elapsed_seconds:.1f}s)"
        )

    # ---- Report ----
    print("\n" + "=" * 90)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 90)
    print(
        f"\n{'Pipeline':<25s} {'N':>4s} {'OK':>4s} {'Rate':>6s} {'Mean Ait':>10s} "
        f"{'Med Ait':>10s} {'P95 Ait':>10s} {'Mean RMSE':>10s} {'Time(s)':>8s}"
    )
    print("-" * 90)
    for r in sorted(results, key=lambda x: x.mean_aitchison):
        mean_rmse = float(np.mean(r.rmse_values)) if r.rmse_values else float("inf")
        print(
            f"{r.pipeline_name:<25s} {r.n_spectra:>4d} {r.n_succeeded:>4d} "
            f"{r.success_rate:>5.0%} {r.mean_aitchison:>10.4f} "
            f"{r.median_aitchison:>10.4f} {r.p95_aitchison:>10.4f} "
            f"{mean_rmse:>10.4f} {r.elapsed_seconds:>8.1f}"
        )

    # Per-element accuracy for best pipeline
    best = min(results, key=lambda x: x.mean_aitchison)
    print(f"\nBest pipeline: {best.pipeline_name}")
    print("\nPer-element mean absolute error:")
    for el in sorted(best.per_element_errors.keys()):
        errs = best.per_element_errors[el]
        print(
            f"  {el:3s}: {np.mean(errs):.4f} ± {np.std(errs):.4f} "
            f"(n={len(errs)}, max={np.max(errs):.4f})"
        )

    # Save results
    save_data = {
        "metadata": {
            "n_total_spectra": len(all_spectra),
            "n_real": len(real_with_gt),
            "n_synthetic": len(all_spectra) - len(real_with_gt),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "pipelines": {},
    }
    for r in results:
        save_data["pipelines"][r.pipeline_name] = {
            "n_spectra": r.n_spectra,
            "n_succeeded": r.n_succeeded,
            "success_rate": r.success_rate,
            "mean_aitchison": r.mean_aitchison,
            "median_aitchison": r.median_aitchison,
            "p95_aitchison": r.p95_aitchison,
            "mean_rmse": float(np.mean(r.rmse_values)) if r.rmse_values else None,
            "elapsed_seconds": r.elapsed_seconds,
            "per_element_mae": {
                el: float(np.mean(errs)) for el, errs in r.per_element_errors.items()
            },
            "details": r.details,
        }

    out_file = out_dir / "comprehensive_results.json"
    out_file.write_text(json.dumps(save_data, indent=2, default=str))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
