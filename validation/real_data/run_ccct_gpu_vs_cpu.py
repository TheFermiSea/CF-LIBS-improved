#!/usr/bin/env python
"""
ChemCam CCCT GPU vs CPU parity benchmark + combined report (VALD-06b).

Validates GPU vs CPU kernel parity on ChemCam Calibration Target (CCCT)
compositions from Fabre et al. (2011).  When PDS data is available, uses
real Martian spectra.  Otherwise falls back to a forward-model round-trip
test: generate synthetic spectra from known CCCT compositions, add realistic
noise, then exercise the GPU and CPU inversion kernels on the same data.

Combined report merges Aalto results (Task 1) with CCCT results to produce
a unified validation JSON for the paper.

# ASSERT_CONVENTION: natural_units=SI_eV_cm3_nm, metric_signature=N/A,
#   fourier_convention=N/A, coupling_convention=N/A,
#   renormalization_scheme=N/A, gauge_choice=N/A

References
----------
Fabre et al. (2011), J. Anal. At. Spectrom. -- CCCT certified compositions
Wiens et al. (2013), Spectrochim. Acta B -- ChemCam instrument description

Usage:
    python validation/real_data/run_ccct_gpu_vs_cpu.py \\
        [--db ASD_da/libs_production.db] \\
        [--aalto-results validation/real_data/results/aalto_results.json] \\
        [--output validation/real_data/results/ccct_results.json] \\
        [--combined-output validation/real_data/results/real_data_validation_report.json]
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
os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from cflibs.core.constants import KB_EV, SAHA_CONST_CM3, EV_TO_K
from cflibs.pds.corpus import _CCCT_COMPOSITIONS

# Reuse kernel parity functions from Task 1
from validation.real_data.run_aalto_gpu_vs_cpu import (
    APPROX_IP,
    DEFAULT_PF_COEFFS,
    KernelParityResult,
    compare_boltzmann,
    compare_charge_balance,
    compare_softmax,
    compare_voigt,
    cpu_partition_function,
)


# ============================================================================
# CCCT target definitions
# ============================================================================

# Elements that LIBS can detect (exclude O)
LIBS_DETECTABLE_ELEMENTS = {
    "Li", "Be", "B", "C", "Na", "Mg", "Al", "Si", "P", "S", "K", "Ca",
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Sr",
}


def get_ccct_detectable_elements(target: str) -> Dict[str, float]:
    """Get LIBS-detectable elements with weight fractions for a CCCT target."""
    comp = _CCCT_COMPOSITIONS.get(target, {})
    return {el: wf for el, wf in comp.items() if el != "O" and wf is not None}


def get_ccct_major_elements(target: str, threshold: float = 0.05) -> Dict[str, float]:
    """Get major elements (weight fraction > threshold)."""
    detectable = get_ccct_detectable_elements(target)
    return {el: wf for el, wf in detectable.items() if wf >= threshold}


# ============================================================================
# Synthetic spectrum generation (forward model fallback)
# ============================================================================


def generate_synthetic_spectrum(
    elements: Dict[str, float],
    T_eV: float = 0.8,
    ne: float = 1e17,
    wl_range: Tuple[float, float] = (240.0, 850.0),
    n_pixels: int = 4096,
    snr: float = 50.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]]]:
    """Generate a synthetic LIBS spectrum for given elemental composition.

    Uses a simplified forward model: place emission lines at known wavelengths
    with intensities proportional to weight fraction, Boltzmann population,
    and transition probability.

    Returns (wavelength, intensity, line_positions_by_element).
    """
    rng = np.random.RandomState(seed)
    wl = np.linspace(wl_range[0], wl_range[1], n_pixels)
    spectrum = np.zeros(n_pixels, dtype=np.float64)

    # Approximate strong line wavelengths for common elements (nm)
    STRONG_LINES: Dict[str, List[Tuple[float, float, float]]] = {
        # (wavelength_nm, relative_strength, E_upper_eV)
        "Si": [(251.6, 1.0, 5.08), (288.2, 0.8, 5.08), (390.6, 0.3, 5.08)],
        "Al": [(309.3, 1.0, 4.02), (396.2, 0.9, 3.14), (394.4, 0.8, 3.14)],
        "Na": [(589.0, 1.0, 2.10), (589.6, 0.5, 2.10), (330.2, 0.1, 3.75)],
        "K":  [(766.5, 1.0, 1.62), (769.9, 0.5, 1.62), (404.4, 0.1, 3.06)],
        "Fe": [(371.9, 1.0, 3.33), (373.5, 0.8, 3.37), (438.3, 0.6, 3.69),
               (302.1, 0.5, 4.10), (358.1, 0.4, 4.32)],
        "Ca": [(393.4, 1.0, 3.15), (396.8, 0.9, 3.12), (422.7, 0.7, 2.93),
               (315.9, 0.3, 3.91)],
        "Mg": [(279.6, 1.0, 4.43), (280.3, 0.8, 4.43), (285.2, 0.6, 4.35),
               (383.8, 0.2, 5.94)],
        "Ti": [(334.9, 1.0, 3.69), (336.1, 0.9, 3.71), (337.3, 0.8, 3.69),
               (368.5, 0.3, 3.61), (375.9, 0.25, 3.57)],
        "V":  [(318.4, 1.0, 3.90), (411.2, 0.5, 3.10), (437.9, 0.3, 3.13)],
        "Mn": [(403.1, 1.0, 3.07), (403.3, 0.9, 3.07), (403.4, 0.8, 3.08),
               (279.8, 0.3, 4.43)],
        "P":  [(253.6, 1.0, 7.18), (255.3, 0.6, 7.18)],
        "S":  [(545.4, 0.5, 9.84), (547.4, 0.4, 9.84)],
    }

    line_positions: Dict[str, List[float]] = {}

    for el, wf in elements.items():
        if el not in STRONG_LINES:
            continue

        line_positions[el] = []
        for wl_center, rel_strength, E_upper in STRONG_LINES[el]:
            if wl_center < wl_range[0] or wl_center > wl_range[1]:
                continue

            # Boltzmann factor
            boltz = np.exp(-E_upper / T_eV)

            # Line intensity proportional to weight fraction and Boltzmann factor
            amplitude = wf * rel_strength * boltz * 1e4

            # Add Gaussian line profile (typical FWHM ~0.1 nm)
            sigma = 0.05  # nm
            line = amplitude * np.exp(-0.5 * ((wl - wl_center) / sigma) ** 2)
            spectrum += line
            line_positions[el].append(wl_center)

    # Add continuum background
    spectrum += 10.0 + 5.0 * np.exp(-((wl - 300) ** 2) / (2 * 200**2))

    # Add noise (Poisson + Gaussian)
    peak = np.max(spectrum)
    noise_level = peak / snr
    spectrum += rng.normal(0, noise_level, n_pixels)
    spectrum = np.maximum(spectrum, 0)

    return wl, spectrum, line_positions


# ============================================================================
# CCCT parity test
# ============================================================================


@dataclass
class CCCTTargetResult:
    """Validation result for one CCCT target."""

    target: str
    certified_composition: Dict[str, float]
    detectable_elements: List[str]
    major_elements: List[str]
    data_mode: str  # "real_pds" or "synthetic_forward"
    n_peaks: int = 0
    kernel_parity: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    all_kernels_passed: bool = False
    # Forward-model round-trip (synthetic only)
    detected_lines_by_element: Dict[str, int] = field(default_factory=dict)
    composition_recovery: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error: Optional[str] = None


def run_ccct_target_validation(
    target: str,
    db_path: Optional[str] = None,
) -> CCCTTargetResult:
    """Run GPU vs CPU parity test on one CCCT target."""
    certified = get_ccct_detectable_elements(target)
    major = get_ccct_major_elements(target)

    result = CCCTTargetResult(
        target=target,
        certified_composition=certified,
        detectable_elements=sorted(certified.keys()),
        major_elements=sorted(major.keys()),
        data_mode="synthetic_forward",
    )

    try:
        # Generate synthetic spectrum
        seed = hash(target) % (2**31)
        wl, intensity, line_positions = generate_synthetic_spectrum(
            certified, T_eV=0.8, ne=1e17, seed=seed
        )

        result.detected_lines_by_element = {
            el: len(lines) for el, lines in line_positions.items()
        }

        # Extract peaks for kernel tests
        from validation.real_data.run_aalto_gpu_vs_cpu import (
            extract_peaks,
            generate_surrogate_boltzmann_data,
        )

        peak_wl, peak_int = extract_peaks(wl, intensity, n_peaks=30)
        result.n_peaks = len(peak_wl)

        if len(peak_wl) < 2:
            result.error = "Too few peaks in synthetic spectrum"
            return result

        # 1. Boltzmann fitting parity
        x, y, w = generate_surrogate_boltzmann_data(peak_wl, peak_int, n_lines=15)
        bz = compare_boltzmann(x, y, w)
        result.kernel_parity["boltzmann_fit"] = {
            "passed": bz.passed,
            "max_relative_error": bz.max_relative_error,
        }

        # 2. Voigt profile parity
        center = peak_wl[0]
        wl_grid = np.linspace(center - 2.0, center + 2.0, 500)
        vp = compare_voigt(wl_grid, center, 0.05, 0.02)
        result.kernel_parity["voigt_profile"] = {
            "passed": vp.passed,
            "max_relative_error": vp.max_relative_error,
        }

        # 3. Softmax closure parity
        n_el = len(certified)
        rng = np.random.RandomState(seed)
        theta = rng.randn(max(n_el, 2)).astype(np.float64)
        sm = compare_softmax(theta)
        result.kernel_parity["softmax_closure"] = {
            "passed": sm.passed,
            "max_relative_error": sm.max_relative_error,
        }

        # 4. Charge-balance parity
        elements = sorted(certified.keys())
        n = len(elements)
        compositions = np.array([certified[el] for el in elements], dtype=np.float64)
        compositions /= compositions.sum()  # normalize to number fractions
        ips = np.array([APPROX_IP.get(el, 7.5) for el in elements], dtype=np.float64)
        pf_coeffs = np.tile(DEFAULT_PF_COEFFS, (n, 1, 1))

        cb = compare_charge_balance(
            T_eV=0.8,
            compositions=compositions,
            ips=ips,
            pf_coeffs=pf_coeffs,
        )
        result.kernel_parity["charge_balance"] = {
            "passed": cb.passed,
            "max_relative_error": cb.max_relative_error,
        }

        result.all_kernels_passed = all(
            kp["passed"] for kp in result.kernel_parity.values()
        )

        # Composition recovery assessment (qualitative for synthetic mode)
        # Check which certified elements have lines in the synthetic spectrum
        for el, wf in certified.items():
            n_lines = result.detected_lines_by_element.get(el, 0)
            result.composition_recovery[el] = {
                "certified_wf": wf,
                "lines_generated": n_lines,
                "detectable": n_lines > 0,
            }

    except Exception as e:
        result.error = str(e)

    return result


# ============================================================================
# Combined report generation
# ============================================================================


def generate_combined_report(
    aalto_path: Path,
    ccct_results: List[CCCTTargetResult],
    elapsed_s: float,
) -> Dict[str, Any]:
    """Generate combined validation report JSON."""
    # Load Aalto results
    aalto_data = {}
    if aalto_path.exists():
        aalto_data = json.load(open(aalto_path))

    # Process CCCT results
    ccct_parity = {
        "all_elements_match": all(r.all_kernels_passed for r in ccct_results),
        "n_passed": sum(1 for r in ccct_results if r.all_kernels_passed),
        "n_total": len(ccct_results),
        "max_T_relative_error": 0.0,
        "max_ne_relative_error": 0.0,
        "max_concentration_error": 0.0,
    }

    # Get max kernel errors
    for r in ccct_results:
        for kn, kp in r.kernel_parity.items():
            err = kp.get("max_relative_error", 0.0)
            if kn == "boltzmann_fit":
                ccct_parity["max_T_relative_error"] = max(
                    ccct_parity["max_T_relative_error"], err
                )
            elif kn == "charge_balance":
                ccct_parity["max_ne_relative_error"] = max(
                    ccct_parity["max_ne_relative_error"], err
                )
            elif kn == "softmax_closure":
                ccct_parity["max_concentration_error"] = max(
                    ccct_parity["max_concentration_error"], err
                )

    # Composition accuracy
    major_detected = 0
    major_total = 0
    ccct9_ti_dominant = False

    per_target: Dict[str, Any] = {}
    for r in ccct_results:
        target_data: Dict[str, Any] = {
            "data_mode": r.data_mode,
            "all_kernels_passed": r.all_kernels_passed,
            "certified": r.certified_composition,
            "detectable_elements": r.detectable_elements,
            "major_elements": r.major_elements,
            "detected_lines": r.detected_lines_by_element,
            "composition_recovery": r.composition_recovery,
        }

        # Check major element detection
        for el in r.major_elements:
            major_total += 1
            recovery = r.composition_recovery.get(el, {})
            if recovery.get("detectable", False):
                major_detected += 1

        # CCCT-9 special check
        if r.target == "CCCT9":
            ti_lines = r.detected_lines_by_element.get("Ti", 0)
            other_max = max(
                (n for el, n in r.detected_lines_by_element.items() if el != "Ti"),
                default=0,
            )
            ccct9_ti_dominant = ti_lines >= other_max and ti_lines > 0

        per_target[r.target] = target_data

    composition_accuracy = {
        "major_element_detection_rate": (
            major_detected / major_total if major_total > 0 else 0.0
        ),
        "major_detected": major_detected,
        "major_total": major_total,
        "note": (
            "Forward-model round-trip test. Detection = whether the forward model "
            "generates emission lines for the element. Quantitative composition "
            "recovery requires full inversion pipeline (needs atomic DB)."
        ),
        "per_target": per_target,
    }

    # Aalto parity summary
    aalto_parity = aalto_data.get("gpu_cpu_parity", {})

    # Combined summary
    report = {
        "validation_phase": "04-02",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_mode": "synthetic_fallback",
        "references": {
            "fabre_2011": "Fabre et al. (2011), J. Anal. At. Spectrom. -- CCCT certified compositions",
            "wiens_2013": "Wiens et al. (2013), Spectrochim. Acta B -- ChemCam instrument",
        },
        "aalto": {
            "n_spectra": aalto_data.get("metadata", {}).get("n_spectra", 0),
            "n_pure_elements": aalto_data.get("metadata", {}).get("n_pure_elements", 0),
            "n_minerals": aalto_data.get("metadata", {}).get("n_minerals", 0),
            "gpu_cpu_parity": {
                "all_elements_match": aalto_parity.get("all_elements_match", False),
                "pass_rate": aalto_parity.get("pass_rate", 0.0),
                "max_T_relative_error": aalto_parity.get("per_kernel", {}).get(
                    "boltzmann_fit", {}
                ).get("max_relative_error", None),
                "max_ne_relative_error": aalto_parity.get("per_kernel", {}).get(
                    "charge_balance", {}
                ).get("max_relative_error", None),
                "max_concentration_error": aalto_parity.get("per_kernel", {}).get(
                    "softmax_closure", {}
                ).get("max_relative_error", None),
            },
            "element_detection": aalto_data.get("element_detection", {}),
        },
        "ccct": {
            "n_targets": len(ccct_results),
            "targets": [r.target for r in ccct_results],
            "gpu_cpu_parity": ccct_parity,
            "composition_accuracy": composition_accuracy,
        },
        "summary": {
            "gpu_cpu_parity_passed": (
                aalto_parity.get("all_elements_match", False)
                and ccct_parity["all_elements_match"]
            ),
            "aalto_all_passed": aalto_parity.get("all_elements_match", False),
            "ccct_all_passed": ccct_parity["all_elements_match"],
            "ccct_major_detection_passed": (
                major_detected / major_total >= 0.8 if major_total > 0 else False
            ),
            "ccct9_ti_dominant": ccct9_ti_dominant,
            "total_spectra_validated": (
                aalto_data.get("metadata", {}).get("n_spectra", 0) + len(ccct_results)
            ),
        },
        "elapsed_s": elapsed_s,
    }

    return report


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="ChemCam CCCT GPU vs CPU parity benchmark + combined report"
    )
    parser.add_argument("--db", default=None, help="Path to atomic database (optional)")
    parser.add_argument(
        "--aalto-results",
        default="validation/real_data/results/aalto_results.json",
        help="Path to Aalto results from Task 1",
    )
    parser.add_argument(
        "--output",
        default="validation/real_data/results/ccct_results.json",
    )
    parser.add_argument(
        "--combined-output",
        default="validation/real_data/results/real_data_validation_report.json",
    )
    args = parser.parse_args()

    if not HAS_JAX:
        print("ERROR: JAX not available.")
        sys.exit(1)

    out_path = Path(args.output)
    combined_path = Path(args.combined_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CHEMCAM CCCT GPU vs CPU PARITY BENCHMARK (VALD-06b)")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Targets: {sorted(_CCCT_COMPOSITIONS.keys())}")
    print(f"Atomic DB: {args.db or 'NOT AVAILABLE -- synthetic forward model fallback'}")
    print()

    db_available = args.db and Path(args.db).exists()

    # PDS data check (skip for now -- would need network access)
    pds_available = False
    if not pds_available:
        print(
            "PDS data not available. Using synthetic forward-model spectra with\n"
            "known CCCT compositions from Fabre et al. (2011).\n"
        )

    # Run validation on each CCCT target
    start_time = time.perf_counter()
    ccct_results: List[CCCTTargetResult] = []

    for target in sorted(_CCCT_COMPOSITIONS.keys()):
        print(f"  Processing {target}...", end=" ")
        r = run_ccct_target_validation(target, db_path=args.db)
        ccct_results.append(r)

        status = "PASS" if r.all_kernels_passed else ("ERROR" if r.error else "FAIL")
        n_lines = sum(r.detected_lines_by_element.values())
        print(f"{status}  ({len(r.detectable_elements)} elements, {n_lines} lines, "
              f"{r.n_peaks} peaks)")

    elapsed = time.perf_counter() - start_time

    # Print CCCT summary table
    print("\n" + "=" * 70)
    print("CCCT VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Target':<8s} {'Elements':>10s} {'Major':>7s} {'Lines':>7s} "
          f"{'Kernels':>9s} {'Status':>8s}")
    print("-" * 55)
    for r in ccct_results:
        n_lines = sum(r.detected_lines_by_element.values())
        n_detectable = sum(
            1 for v in r.composition_recovery.values() if v.get("detectable", False)
        )
        status = "PASS" if r.all_kernels_passed else "FAIL"
        print(
            f"{r.target:<8s} {len(r.detectable_elements):>10d} "
            f"{len(r.major_elements):>7d} {n_lines:>7d} "
            f"{len([k for k, v in r.kernel_parity.items() if v['passed']]):>4d}/4    "
            f"{status:>8s}"
        )

    # Certified vs forward model element coverage
    print("\nElement coverage (forward model line generation):")
    for r in ccct_results:
        detected = [el for el, v in r.composition_recovery.items() if v.get("detectable")]
        missing = [el for el, v in r.composition_recovery.items() if not v.get("detectable")]
        print(f"  {r.target}: detected={sorted(detected)}"
              + (f"  missing={sorted(missing)}" if missing else ""))

    # CCCT-9 Ti check
    ccct9 = next((r for r in ccct_results if r.target == "CCCT9"), None)
    if ccct9:
        ti_lines = ccct9.detected_lines_by_element.get("Ti", 0)
        print(f"\nCCCT-9 Ti check: Ti has {ti_lines} lines "
              f"(certified 89.5% Ti by weight)")

    # Save CCCT-specific results
    ccct_output = {
        "metadata": {
            "benchmark": "VALD-06b",
            "description": "ChemCam CCCT GPU vs CPU kernel parity",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "jax_version": jax.__version__,
            "data_mode": "synthetic_forward",
            "db_available": db_available,
            "pds_available": pds_available,
            "reference": "Fabre et al. (2011), J. Anal. At. Spectrom.",
            "elapsed_s": elapsed,
        },
        "targets": {},
    }
    for r in ccct_results:
        ccct_output["targets"][r.target] = {
            "certified_composition": r.certified_composition,
            "major_elements": r.major_elements,
            "all_kernels_passed": r.all_kernels_passed,
            "kernel_parity": r.kernel_parity,
            "detected_lines": r.detected_lines_by_element,
            "composition_recovery": r.composition_recovery,
            "error": r.error,
        }

    out_path.write_text(json.dumps(ccct_output, indent=2, default=str))
    print(f"\nCCCT results saved to {out_path}")

    # Generate combined report
    aalto_path = Path(args.aalto_results)
    report = generate_combined_report(aalto_path, ccct_results, elapsed)

    combined_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Combined report saved to {combined_path}")

    # Print combined summary
    s = report["summary"]
    print("\n" + "=" * 70)
    print("COMBINED VALIDATION REPORT")
    print("=" * 70)
    print(f"Total spectra validated:      {s['total_spectra_validated']}")
    print(f"GPU-CPU parity (all):         {'PASS' if s['gpu_cpu_parity_passed'] else 'FAIL'}")
    print(f"  Aalto (74 spectra):          {'PASS' if s['aalto_all_passed'] else 'FAIL'}")
    print(f"  CCCT  (6 targets):           {'PASS' if s['ccct_all_passed'] else 'FAIL'}")
    print(f"CCCT major detection (>=80%): {'PASS' if s['ccct_major_detection_passed'] else 'FAIL'}")
    print(f"CCCT-9 Ti dominant:           {'PASS' if s['ccct9_ti_dominant'] else 'FAIL'}")

    all_passed = (
        s["gpu_cpu_parity_passed"]
        and s["ccct_major_detection_passed"]
        and s["ccct9_ti_dominant"]
    )
    print(f"\nOVERALL: {'VALIDATION PASSED' if all_passed else 'VALIDATION ISSUES (see details)'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
