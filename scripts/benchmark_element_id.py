#!/usr/bin/env python
"""
Unified element identification benchmark.

Runs all available identification pathways on the Aalto LIBS benchmark
(74 spectra: 13 pure elements + 61 minerals) and reports P/R/F1/FPR
in a single comparison table.

Pathways:
  alias          – ALIAS peak-matching (baseline)
  spectral_nnls  – Full-spectrum NNLS decomposition
  hybrid         – Two-stage NNLS screening + ALIAS confirmation
  voigt_alias    – Voigt deconvolution + ALIAS on cleaned spectrum
  nnls_concentration_threshold
                 – NNLS concentration-threshold proxy for forward-model fitting

Usage:
  JAX_PLATFORMS=cpu python scripts/benchmark_element_id.py \
    --db-path ASD_da/libs_production.db \
    --data-dir data \
    --output-dir output/benchmark_comparison

  # Generate basis library first (slow, ~10-30 min):
  JAX_PLATFORMS=cpu python scripts/benchmark_element_id.py \
    --db-path ASD_da/libs_production.db \
    --data-dir data \
    --generate-basis --basis-fwhm 0.5 1.0

  # Run specific pathways:
  JAX_PLATFORMS=cpu python scripts/benchmark_element_id.py \
    --db-path ASD_da/libs_production.db \
    --data-dir data \
    --pathways alias spectral_nnls hybrid
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

os.environ["JAX_PLATFORMS"] = "cpu"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.inversion.alias_identifier import ALIASIdentifier  # noqa: E402
from cflibs.inversion.element_id import ElementIdentificationResult  # noqa: E402

# Reuse Aalto benchmark infrastructure from calibrate_alias
from scripts.calibrate_alias import (  # noqa: E402
    AALTO_SEARCH_ELEMENTS,
    DatasetCase,
    _estimate_effective_rp,
    _safe_ratio,
    _select_aalto_cases,
)

# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------


@dataclass
class PathwayResult:
    """Result for a single pathway + config on the full benchmark."""

    pathway: str
    config_name: str
    config: Dict[str, Any]
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fpr: float
    exact_matches: int
    datasets_evaluated: int
    wall_seconds: float
    per_case: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_result(
    result: ElementIdentificationResult,
    case: DatasetCase,
) -> Tuple[int, int, int, int, bool]:
    """Score one identification result against ground truth."""
    detected = {e.element for e in result.detected_elements}
    searched = set(case.elements)
    tp = len(detected & case.expected)
    fp = len(detected - case.expected)
    fn = len(case.expected - detected)
    tn = len((searched - case.expected) - detected)
    exact = detected == case.expected
    return tp, fp, fn, tn, exact


def run_pathway(
    identify_fn: Callable[[np.ndarray, np.ndarray], ElementIdentificationResult],
    cases: List[DatasetCase],
    pathway_name: str,
    config_name: str,
    config: Dict[str, Any],
) -> PathwayResult:
    """Run an identifier on all cases and aggregate metrics."""
    tp = fp = fn = tn = 0
    exact_matches = 0
    evaluated = 0
    per_case = []
    t0 = time.monotonic()

    for case in cases:
        try:
            result = identify_fn(case.wavelength, case.spectrum)
            ctp, cfp, cfn, ctn, exact = score_result(result, case)
        except Exception as exc:
            logger.warning("%s failed on %s: %s", pathway_name, case.name, exc)
            logger.debug("Traceback for %s on %s", pathway_name, case.name, exc_info=True)
            continue

        tp += ctp
        fp += cfp
        fn += cfn
        tn += ctn
        exact_matches += int(exact)
        evaluated += 1

        detected = {e.element for e in result.detected_elements}
        per_case.append(
            {
                "case": case.name,
                "detected": sorted(detected),
                "expected": sorted(case.expected),
                "tp": ctp,
                "fp": cfp,
                "fn": cfn,
                "tn": ctn,
                "exact": exact,
            }
        )

    wall = time.monotonic() - t0
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    fpr = _safe_ratio(fp, fp + tn)

    return PathwayResult(
        pathway=pathway_name,
        config_name=config_name,
        config=config,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        exact_matches=exact_matches,
        datasets_evaluated=evaluated,
        wall_seconds=round(wall, 1),
        per_case=per_case,
    )


# ---------------------------------------------------------------------------
# Basis library generation
# ---------------------------------------------------------------------------


def generate_basis_library(
    db_path: str,
    output_dir: Path,
    fwhm_nm: float,
    elements: List[str],
    temperature_steps: int = 30,
    density_steps: int = 10,
) -> Path:
    """Generate a basis library at the given FWHM."""
    from cflibs.manifold.basis_library import BasisLibraryConfig, BasisLibraryGenerator

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"basis_fwhm_{fwhm_nm:.1f}nm.h5"

    if out_path.exists():
        logger.info("Basis library already exists: %s", out_path)
        return out_path

    config = BasisLibraryConfig(
        db_path=db_path,
        output_path=str(out_path),
        wavelength_range=(200.0, 900.0),
        pixels=4096,
        temperature_range=(4000.0, 12000.0),
        temperature_steps=temperature_steps,
        density_range=(1e15, 5e17),
        density_steps=density_steps,
        ionization_stages=(1, 2),
        instrument_fwhm_nm=fwhm_nm,
    )

    logger.info("Generating basis library at FWHM=%.2f nm -> %s", fwhm_nm, out_path)
    t0 = time.monotonic()
    gen = BasisLibraryGenerator(config)
    gen.generate(progress_callback=lambda i, n: print(f"  [{i}/{n}] elements", end="\r"))
    print()
    logger.info("Basis library generated in %.1f s", time.monotonic() - t0)
    return out_path


# ---------------------------------------------------------------------------
# Pathway factories
# ---------------------------------------------------------------------------


def make_alias_configs(
    db: AtomicDatabase, cases: List[DatasetCase]
) -> List[Tuple[str, Dict, Callable]]:
    """Generate ALIAS pathway configurations to sweep."""
    configs = []
    for dt in [0.02, 0.03, 0.05]:
        for itf in [3.0, 3.5]:
            for cws in [0.3, 0.4]:
                config = {
                    "detection_threshold": dt,
                    "intensity_threshold_factor": itf,
                    "chance_window_scale": cws,
                    "max_lines_per_element": 30,
                }
                name = f"dt={dt}_itf={itf}_cws={cws}"

                def make_fn(cfg=config):
                    def identify(wl, sp):
                        rp = _estimate_effective_rp(wl, sp)
                        identifier = ALIASIdentifier(
                            atomic_db=db,
                            elements=AALTO_SEARCH_ELEMENTS,
                            resolving_power=rp,
                            intensity_threshold_factor=cfg["intensity_threshold_factor"],
                            detection_threshold=cfg["detection_threshold"],
                            chance_window_scale=cfg["chance_window_scale"],
                            max_lines_per_element=cfg["max_lines_per_element"],
                        )
                        return identifier.identify(wl, sp)

                    return identify

                configs.append((name, config, make_fn()))
    return configs


def make_nnls_configs(
    db: AtomicDatabase,
    basis_lib_path: Optional[Path],
) -> List[Tuple[str, Dict, Callable]]:
    """Generate SpectralNNLS pathway configurations."""
    if basis_lib_path is None or not basis_lib_path.exists():
        logger.warning("No basis library found — skipping NNLS pathway")
        return []

    from cflibs.manifold.basis_library import BasisLibrary
    from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier

    basis = BasisLibrary(str(basis_lib_path))
    configs = []

    for snr in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for cdeg in [2, 3, 4]:
            for T_K in [6000.0, 8000.0, 10000.0]:
                config = {
                    "detection_snr": snr,
                    "continuum_degree": cdeg,
                    "fallback_T_K": T_K,
                    "basis_library": str(basis_lib_path),
                }
                name = f"snr={snr}_cdeg={cdeg}_T={T_K:.0f}"

                def make_fn(cfg=config):
                    def identify(wl, sp):
                        identifier = SpectralNNLSIdentifier(
                            basis_library=basis,
                            detection_snr=cfg["detection_snr"],
                            continuum_degree=cfg["continuum_degree"],
                            fallback_T_K=cfg["fallback_T_K"],
                            fallback_ne_cm3=1e17,
                        )
                        return identifier.identify(wl, sp)

                    return identify

                configs.append((name, config, make_fn()))
    return configs


def make_hybrid_configs(
    db: AtomicDatabase,
    basis_lib_path: Optional[Path],
) -> List[Tuple[str, Dict, Callable]]:
    """Generate Hybrid NNLS+ALIAS pathway configurations."""
    if basis_lib_path is None or not basis_lib_path.exists():
        logger.warning("No basis library found — skipping hybrid pathway")
        return []

    from cflibs.manifold.basis_library import BasisLibrary
    from cflibs.inversion.hybrid_identifier import HybridIdentifier

    basis = BasisLibrary(str(basis_lib_path))
    configs = []

    for nnls_snr in [1.0, 1.5, 2.0]:
        for alias_dt in [0.03, 0.05, 0.10]:
            for require_both in [True, False]:
                mode = "intersect" if require_both else "union"
                config = {
                    "nnls_detection_snr": nnls_snr,
                    "alias_detection_threshold": alias_dt,
                    "require_both": require_both,
                }
                name = f"nsnr={nnls_snr}_adt={alias_dt}_{mode}"

                def make_fn(cfg=config):
                    def identify(wl, sp):
                        rp = _estimate_effective_rp(wl, sp)
                        identifier = HybridIdentifier(
                            atomic_db=db,
                            basis_library=basis,
                            elements=AALTO_SEARCH_ELEMENTS,
                            resolving_power=rp,
                            nnls_detection_snr=cfg["nnls_detection_snr"],
                            alias_detection_threshold=cfg["alias_detection_threshold"],
                            require_both=cfg["require_both"],
                        )
                        return identifier.identify(wl, sp)

                    return identify

                configs.append((name, config, make_fn()))
    return configs


def make_voigt_alias_configs(
    db: AtomicDatabase,
) -> List[Tuple[str, Dict, Callable]]:
    """Generate Voigt deconvolution + ALIAS pathway configurations."""
    from cflibs.inversion.deconvolution import deconvolve_peaks
    from cflibs.inversion.preprocessing import estimate_baseline
    from scipy.signal import find_peaks

    configs = []

    for dt in [0.03, 0.05]:
        config = {
            "detection_threshold": dt,
            "intensity_threshold_factor": 3.0,
            "chance_window_scale": 0.4,
            "max_lines_per_element": 30,
        }
        name = f"voigt_dt={dt}"

        def make_fn(cfg=config):
            def identify(wl, sp):
                # Step 1: Baseline subtraction
                baseline = estimate_baseline(wl, sp)
                corrected = np.maximum(sp - baseline, 0.0)

                # Step 2: Detect peaks
                threshold = (
                    np.percentile(corrected[corrected > 0], 70) if np.any(corrected > 0) else 0
                )
                peak_indices, _ = find_peaks(corrected, height=threshold, distance=5)
                if len(peak_indices) == 0:
                    # Fall back to plain ALIAS
                    rp = _estimate_effective_rp(wl, sp)
                    alias_id = ALIASIdentifier(
                        atomic_db=db,
                        elements=AALTO_SEARCH_ELEMENTS,
                        resolving_power=rp,
                        detection_threshold=cfg["detection_threshold"],
                    )
                    return alias_id.identify(wl, sp)

                peak_wls = wl[peak_indices]

                # Step 3: Estimate FWHM from resolving power
                rp = _estimate_effective_rp(wl, sp)
                median_wl = float(np.median(wl))
                fwhm_est = median_wl / rp

                # Step 4: Voigt deconvolution
                try:
                    deconv = deconvolve_peaks(
                        wl,
                        corrected,
                        peak_wls,
                        fwhm_est,
                        grouping_factor=2.0,
                        margin_factor=3.0,
                        use_jax=False,
                    )
                    # Build deconvolved spectrum from fitted peaks
                    cleaned = deconv.fitted_spectrum
                except Exception as exc:
                    logger.debug("Voigt deconvolution failed: %s, using raw", exc)
                    cleaned = corrected

                # Step 5: Run ALIAS on deconvolved spectrum
                # Add back a small baseline to avoid zeros
                cleaned = np.maximum(cleaned, 0.0) + np.median(sp) * 0.01
                alias_id = ALIASIdentifier(
                    atomic_db=db,
                    elements=AALTO_SEARCH_ELEMENTS,
                    resolving_power=rp,
                    intensity_threshold_factor=cfg["intensity_threshold_factor"],
                    detection_threshold=cfg["detection_threshold"],
                    chance_window_scale=cfg["chance_window_scale"],
                    max_lines_per_element=cfg["max_lines_per_element"],
                )
                return alias_id.identify(wl, cleaned)

            return identify

        configs.append((name, config, make_fn()))
    return configs


def make_nnls_concentration_configs(
    db: AtomicDatabase,
    basis_lib_path: Optional[Path],
) -> List[Tuple[str, Dict, Callable]]:
    """Generate NNLS concentration-threshold pathway configurations."""
    if basis_lib_path is None or not basis_lib_path.exists():
        logger.warning(
            "No basis library found — skipping nnls_concentration_threshold pathway"
        )
        return []

    from cflibs.manifold.basis_library import BasisLibrary
    from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier

    basis = BasisLibrary(str(basis_lib_path))
    configs = []

    # Forward-model proxy via NNLS with concentration thresholding.
    # (lighter-weight than full JointOptimizer but captures the key idea)
    for conc_threshold in [0.001, 0.005, 0.01, 0.02, 0.05]:
        for cdeg in [2, 3]:
            config = {
                "concentration_threshold": conc_threshold,
                "continuum_degree": cdeg,
            }
            name = f"nnls_ct={conc_threshold}_cdeg={cdeg}"

            def make_fn(cfg=config):
                def identify(wl, sp):
                    # Use NNLS decomposition as proxy for forward-model fitting.
                    identifier = SpectralNNLSIdentifier(
                        basis_library=basis,
                        detection_snr=0.0,  # Disable SNR, use concentration
                        continuum_degree=cfg["continuum_degree"],
                        fallback_T_K=8000.0,
                        fallback_ne_cm3=1e17,
                    )
                    result = identifier.identify(wl, sp)

                    # Re-threshold by concentration
                    ct = cfg["concentration_threshold"]
                    for eid in result.all_elements:
                        conc = eid.metadata.get("concentration_estimate", 0.0)
                        eid.detected = conc >= ct

                    result.detected_elements = [e for e in result.all_elements if e.detected]
                    result.rejected_elements = [e for e in result.all_elements if not e.detected]
                    result.algorithm = "nnls_concentration_threshold"
                    result.parameters["concentration_threshold"] = ct
                    return result

                return identify

            configs.append((name, config, make_fn()))
    return configs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_results(results: List[PathwayResult], output_dir: Path) -> Tuple[Path, Path]:
    """Write results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_results.json"
    csv_path = output_dir / "benchmark_summary.csv"

    # JSON with per-case details
    payload = []
    for r in results:
        d = asdict(r)
        payload.append(d)
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    # CSV summary (no per-case)
    headers = [
        "pathway",
        "config_name",
        "precision",
        "recall",
        "f1",
        "fpr",
        "tp",
        "fp",
        "fn",
        "tn",
        "exact_matches",
        "datasets_evaluated",
        "wall_seconds",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            writer.writerow(
                [
                    r.pathway,
                    r.config_name,
                    f"{r.precision:.4f}",
                    f"{r.recall:.4f}",
                    f"{r.f1:.4f}",
                    f"{r.fpr:.4f}",
                    r.tp,
                    r.fp,
                    r.fn,
                    r.tn,
                    r.exact_matches,
                    r.datasets_evaluated,
                    r.wall_seconds,
                ]
            )

    return json_path, csv_path


def print_comparison_table(results: List[PathwayResult]) -> None:
    """Print a formatted comparison table."""
    # Group by pathway, pick best F1 per pathway
    best_per_pathway: Dict[str, PathwayResult] = {}
    for r in results:
        key = r.pathway
        if key not in best_per_pathway or r.f1 > best_per_pathway[key].f1:
            best_per_pathway[key] = r

    print("\n" + "=" * 95)
    print("ELEMENT IDENTIFICATION BENCHMARK — BEST CONFIG PER PATHWAY")
    print("=" * 95)
    print(
        f"{'Pathway':<20} {'Config':<30} {'P':>6} {'R':>6} {'F1':>6} {'FPR':>6} {'Exact':>7} {'Time':>6}"
    )
    print("-" * 95)

    for pathway in [
        "alias",
        "spectral_nnls",
        "hybrid",
        "voigt_alias",
        "nnls_concentration_threshold",
    ]:
        r = best_per_pathway.get(pathway)
        if r is None:
            continue
        exact_str = f"{r.exact_matches}/{r.datasets_evaluated}"
        print(
            f"{r.pathway:<20} {r.config_name:<30} "
            f"{r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f} {r.fpr:>6.3f} "
            f"{exact_str:>7} {r.wall_seconds:>5.1f}s"
        )

    print("=" * 95)

    # Also print top 10 overall
    sorted_results = sorted(results, key=lambda r: (-r.f1, -r.precision, r.fpr))
    print("\nTOP 10 CONFIGURATIONS (by F1):")
    print(
        f"{'#':>3} {'Pathway':<20} {'Config':<30} {'P':>6} {'R':>6} {'F1':>6} {'FPR':>6} {'Exact':>7}"
    )
    print("-" * 95)
    for i, r in enumerate(sorted_results[:10], 1):
        exact_str = f"{r.exact_matches}/{r.datasets_evaluated}"
        print(
            f"{i:>3} {r.pathway:<20} {r.config_name:<30} "
            f"{r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f} {r.fpr:>6.3f} "
            f"{exact_str:>7}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LEGACY_PATHWAY_ALIASES = {"forward_model": "nnls_concentration_threshold"}
ALL_PATHWAYS = [
    "alias",
    "spectral_nnls",
    "hybrid",
    "voigt_alias",
    "nnls_concentration_threshold",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified element identification benchmark on Aalto LIBS spectra."
    )
    parser.add_argument("--db-path", type=str, default="ASD_da/libs_production.db")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="output/benchmark_comparison")
    parser.add_argument(
        "--basis-dir",
        type=str,
        default="output/basis_libraries",
        help="Directory containing pre-generated basis libraries.",
    )
    parser.add_argument(
        "--basis-fwhm",
        type=float,
        nargs="+",
        default=[0.5],
        help="FWHM values (nm) for basis library generation/use.",
    )
    parser.add_argument(
        "--generate-basis",
        action="store_true",
        help="Generate basis libraries before benchmarking.",
    )
    parser.add_argument(
        "--pathways",
        nargs="+",
        default=ALL_PATHWAYS,
        choices=ALL_PATHWAYS + list(LEGACY_PATHWAY_ALIASES),
        help=(
            "Pathways to benchmark (default: all). "
            "The legacy forward_model alias maps to nnls_concentration_threshold."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer sweep configs per pathway.",
    )
    parser.add_argument(
        "--T-steps",
        type=int,
        default=30,
        help="Temperature grid steps for basis library (default 30).",
    )
    parser.add_argument(
        "--ne-steps",
        type=int,
        default=10,
        help="Density grid steps for basis library (default 10).",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args.pathways = [LEGACY_PATHWAY_ALIASES.get(pathway, pathway) for pathway in args.pathways]

    db_path = Path(args.db_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    basis_dir = Path(args.basis_dir)

    if not db_path.exists():
        raise FileNotFoundError(f"Atomic DB not found: {db_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    # ---- Load benchmark cases ----
    cases = _select_aalto_cases(data_dir)
    if not cases:
        raise RuntimeError("No Aalto benchmark spectra found in data/aalto_libs/")

    print(f"Loaded {len(cases)} Aalto benchmark spectra")

    # ---- Generate basis libraries if requested ----
    basis_lib_paths: Dict[float, Path] = {}
    for fwhm in args.basis_fwhm:
        expected = basis_dir / f"basis_fwhm_{fwhm:.1f}nm.h5"
        if args.generate_basis or not expected.exists():
            path = generate_basis_library(
                db_path=str(db_path),
                output_dir=basis_dir,
                fwhm_nm=fwhm,
                elements=AALTO_SEARCH_ELEMENTS,
                temperature_steps=args.T_steps,
                density_steps=args.ne_steps,
            )
            basis_lib_paths[fwhm] = path
        else:
            basis_lib_paths[fwhm] = expected

    # Pick the primary basis library (first FWHM)
    primary_fwhm = args.basis_fwhm[0]
    primary_basis = basis_lib_paths.get(primary_fwhm)

    # ---- Build pathway configs ----
    all_results: List[PathwayResult] = []

    with AtomicDatabase(str(db_path)) as db:
        for pathway in args.pathways:
            print(f"\n{'='*60}")
            print(f"PATHWAY: {pathway}")
            print(f"{'='*60}")

            if pathway == "alias":
                configs = make_alias_configs(db, cases)
            elif pathway == "spectral_nnls":
                configs = make_nnls_configs(db, primary_basis)
            elif pathway == "hybrid":
                configs = make_hybrid_configs(db, primary_basis)
            elif pathway == "voigt_alias":
                configs = make_voigt_alias_configs(db)
            elif pathway == "nnls_concentration_threshold":
                configs = make_nnls_concentration_configs(db, primary_basis)
            else:
                logger.warning("Unknown pathway: %s", pathway)
                continue

            if args.quick and len(configs) > 3:
                # In quick mode, take first, middle, and last config
                indices = [0, len(configs) // 2, len(configs) - 1]
                configs = [configs[i] for i in indices]

            if not configs:
                print(f"  No configs available for {pathway} (missing basis library?)")
                continue

            print(f"  Running {len(configs)} configurations on {len(cases)} spectra...")

            for idx, (name, config, identify_fn) in enumerate(configs, 1):
                result = run_pathway(identify_fn, cases, pathway, name, config)
                all_results.append(result)
                print(
                    f"  [{idx:>3}/{len(configs)}] {name:<35} "
                    f"P={result.precision:.3f} R={result.recall:.3f} "
                    f"F1={result.f1:.3f} FPR={result.fpr:.3f} "
                    f"exact={result.exact_matches}/{result.datasets_evaluated} "
                    f"({result.wall_seconds:.1f}s)"
                )

    # ---- Output ----
    if not all_results:
        print("\nNo results produced.")
        return

    print_comparison_table(all_results)

    json_path, csv_path = write_results(all_results, output_dir)
    print("\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    # ---- Check Mechelle 5000 data ----
    mechelle_dir = data_dir / "mechelle5000"
    if mechelle_dir.exists():
        print(f"\nMechelle 5000 data found at {mechelle_dir} — run with --data-dir to benchmark")
    else:
        print(f"\nNote: No Mechelle 5000 data found at {mechelle_dir}")
        print("  At RP~5000, ALIAS should achieve >95% — acquire data for comparison.")


if __name__ == "__main__":
    main()
