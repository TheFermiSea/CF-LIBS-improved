#!/usr/bin/env python
"""
Run CF-LIBS benchmark experiments.

Experiments:
  T0.2  Baseline measurement with IterativeCFLIBSSolver (or Boltzmann fallback)
  E1    Line identification comparison (ALIAS / Comb / Correlation)
  E2    Boltzmann fitting x Self-absorption 3x3 matrix

All pipelines work with the Gaussian-fallback corpus so no atomic database
is required (though results improve if one is present).

Usage:
    python scripts/run_experiments.py [--experiments T0.2 E1 E2] [--output-dir output/experiments]
    python scripts/run_experiments.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Ensure cflibs is importable when run from repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cflibs.benchmark import BenchmarkCorpus, BenchmarkHarness  # noqa: E402
from cflibs.benchmark.corpus import _REFERENCE_LINES  # noqa: E402
from cflibs.inversion.boltzmann import (  # noqa: E402
    BoltzmannPlotFitter,
    FitMethod,
    LineObservation,
)
from cflibs.inversion.cdsb import CDSBLineObservation, CDSBPlotter  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers: peak detection on Gaussian-fallback spectra
# ---------------------------------------------------------------------------

_SIGMA_NM = 0.03  # must match corpus._gaussian_fallback


def _detect_peaks(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    threshold_frac: float = 0.02,
) -> List[Dict[str, float]]:
    """Simple peak detection: local maxima above *threshold_frac* of max."""
    if intensity.max() <= 0:
        return []
    threshold = threshold_frac * intensity.max()
    peaks: List[Dict[str, float]] = []
    for i in range(1, len(intensity) - 1):
        if (
            intensity[i] > intensity[i - 1]
            and intensity[i] > intensity[i + 1]
            and intensity[i] > threshold
        ):
            peaks.append(
                {"wavelength_nm": float(wavelength[i]), "intensity": float(intensity[i])}
            )
    return peaks


def _match_peaks_to_lines(
    peaks: List[Dict[str, float]],
    elements: List[str],
    tolerance_nm: float = 0.10,
) -> List[LineObservation]:
    """Match detected peaks to reference lines and build LineObservation list."""
    observations: List[LineObservation] = []
    for peak in peaks:
        wl = peak["wavelength_nm"]
        best_dist = tolerance_nm
        best_line = None
        best_elem = None
        for elem in elements:
            for line in _REFERENCE_LINES.get(elem, []):
                d = abs(wl - line[0])
                if d < best_dist:
                    best_dist = d
                    best_line = line
                    best_elem = elem
        if best_line is not None and best_elem is not None:
            wl_ref, E_k, g_k, log_A = best_line
            observations.append(
                LineObservation(
                    wavelength_nm=wl_ref,
                    intensity=peak["intensity"],
                    intensity_uncertainty=max(peak["intensity"] * 0.05, 1.0),
                    element=best_elem,
                    ionization_stage=1,
                    E_k_ev=E_k,
                    g_k=g_k,
                    A_ki=10.0**log_A,
                )
            )
    return observations


def _boltzmann_pipeline(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    elements: List[str],
    fit_method: FitMethod = FitMethod.SIGMA_CLIP,
    self_absorption: str = "NONE",
) -> Dict[str, Any]:
    """Core pipeline: peak detect -> match -> optional SA correction -> Boltzmann fit -> closure.

    Parameters
    ----------
    self_absorption : str
        One of "NONE", "COG", "CDSB".
    """
    peaks = _detect_peaks(wavelength, intensity)
    observations = _match_peaks_to_lines(peaks, elements)

    if len(observations) < 2:
        # Not enough lines matched — return uniform guess
        n = len(elements)
        return {"concentrations": {el: 1.0 / n for el in elements}}

    # --- Optional self-absorption correction ---
    if self_absorption == "COG" and len(observations) >= 3:
        observations = _apply_cog_correction(observations)
    elif self_absorption == "CDSB" and len(observations) >= 3:
        return _apply_cdsb_pipeline(observations, elements, fit_method)

    # --- Boltzmann fit per element ---
    fitter = BoltzmannPlotFitter(method=fit_method)
    element_intercepts: Dict[str, float] = {}
    temperatures: List[float] = []

    for elem in elements:
        elem_obs = [o for o in observations if o.element == elem]
        if len(elem_obs) < 2:
            continue
        result = fitter.fit(elem_obs)
        if result.temperature_K > 0 and np.isfinite(result.temperature_K):
            temperatures.append(result.temperature_K)
            element_intercepts[elem] = result.intercept

    if not element_intercepts:
        n = len(elements)
        return {"concentrations": {el: 1.0 / n for el in elements}}

    T_mean = float(np.mean(temperatures)) if temperatures else 10000.0

    # --- Closure: exp(intercept) proportional to concentration ---
    raw = {el: np.exp(b) for el, b in element_intercepts.items()}
    total = sum(raw.values())
    concentrations = {el: v / total for el, v in raw.items()}

    # Fill missing elements with zero then re-normalise
    for el in elements:
        if el not in concentrations:
            concentrations[el] = 0.0
    total = sum(concentrations.values())
    if total > 0:
        concentrations = {el: v / total for el, v in concentrations.items()}

    return {
        "concentrations": concentrations,
        "temperature_K": T_mean,
        "electron_density_cm3": 1e17,  # placeholder
    }


# ---------------------------------------------------------------------------
# Self-absorption helpers
# ---------------------------------------------------------------------------


def _apply_cog_correction(observations: List[LineObservation]) -> List[LineObservation]:
    """Simplified curve-of-growth correction using intensity ratios.

    For each element, if a pair of lines from the same multiplet is available,
    estimate optical depth from their intensity ratio and correct.  Falls back
    to the uncorrected observations when data is insufficient.
    """
    # Group by element
    by_elem: Dict[str, List[LineObservation]] = {}
    for obs in observations:
        by_elem.setdefault(obs.element, []).append(obs)

    corrected: List[LineObservation] = []
    for elem, obs_list in by_elem.items():
        if len(obs_list) < 2:
            corrected.extend(obs_list)
            continue

        # Sort by expected intensity (gA * Boltzmann factor proxy)
        sorted_obs = sorted(obs_list, key=lambda o: o.g_k * o.A_ki, reverse=True)
        strongest = sorted_obs[0]

        for obs in sorted_obs:
            # Estimate tau from ratio to strongest line
            if obs is strongest:
                tau_est = 0.5  # moderate self-absorption for strongest
            else:
                gA_ratio = (obs.g_k * obs.A_ki) / (strongest.g_k * strongest.A_ki)
                tau_est = 0.5 * gA_ratio  # scale

            if tau_est < 0.05:
                corrected.append(obs)
            else:
                correction = tau_est / (1.0 - np.exp(-tau_est)) if tau_est < 50 else tau_est
                corrected.append(
                    LineObservation(
                        wavelength_nm=obs.wavelength_nm,
                        intensity=obs.intensity * correction,
                        intensity_uncertainty=obs.intensity_uncertainty * correction,
                        element=obs.element,
                        ionization_stage=obs.ionization_stage,
                        E_k_ev=obs.E_k_ev,
                        g_k=obs.g_k,
                        A_ki=obs.A_ki,
                    )
                )
    return corrected


def _apply_cdsb_pipeline(
    observations: List[LineObservation],
    elements: List[str],
    fit_method: FitMethod,
) -> Dict[str, Any]:
    """Run the CD-SB plotter on observations and return pipeline result."""
    # Convert to CDSBLineObservation (assume E_i ~ 0 for simplicity)
    cdsb_obs = [
        CDSBLineObservation(
            wavelength_nm=obs.wavelength_nm,
            intensity=obs.intensity,
            intensity_uncertainty=obs.intensity_uncertainty,
            element=obs.element,
            ionization_stage=obs.ionization_stage,
            E_k_ev=obs.E_k_ev,
            g_k=obs.g_k,
            A_ki=obs.A_ki,
            E_i_ev=0.0,
            g_i=1,
            is_resonance=False,
        )
        for obs in observations
    ]

    plotter = CDSBPlotter(fit_method=fit_method)
    cdsb_result = plotter.fit(cdsb_obs, n_e=1e17)

    T = cdsb_result.temperature_K if cdsb_result.temperature_K > 0 else 10000.0

    # Use uncorrected Boltzmann intercepts for closure
    fitter = BoltzmannPlotFitter(method=fit_method)
    element_intercepts: Dict[str, float] = {}
    for elem in elements:
        elem_obs = [o for o in observations if o.element == elem]
        if len(elem_obs) < 2:
            continue
        res = fitter.fit(elem_obs)
        if res.temperature_K > 0 and np.isfinite(res.intercept):
            element_intercepts[elem] = res.intercept

    if not element_intercepts:
        n = len(elements)
        return {"concentrations": {el: 1.0 / n for el in elements}}

    raw = {el: np.exp(b) for el, b in element_intercepts.items()}
    total = sum(raw.values())
    concentrations = {el: v / total for el, v in raw.items()}
    for el in elements:
        if el not in concentrations:
            concentrations[el] = 0.0
    total = sum(concentrations.values())
    if total > 0:
        concentrations = {el: v / total for el, v in concentrations.items()}

    return {
        "concentrations": concentrations,
        "temperature_K": T,
        "electron_density_cm3": 1e17,
    }


# ============================================================================
# Experiment: T0.2 — Baseline Measurement
# ============================================================================


def register_baseline(harness: BenchmarkHarness) -> None:
    """Register baseline CF-LIBS pipeline (Boltzmann-only fallback)."""

    def baseline_pipeline(
        wavelength: np.ndarray,
        intensity: np.ndarray,
        elements: List[str],
    ) -> Dict[str, Any]:
        return _boltzmann_pipeline(wavelength, intensity, elements)

    harness.register_pipeline("T0.2_baseline", baseline_pipeline)


# ============================================================================
# Experiment: E1 — Line Identification Comparison
# ============================================================================


def _alias_score(peak_wl: float, ref_wl: float) -> float:
    """ALIAS-style scoring: Gaussian kernel centred on reference wavelength."""
    sigma = 0.04  # narrower window favours precise matches
    return np.exp(-0.5 * ((peak_wl - ref_wl) / sigma) ** 2)


def _comb_score(peak_wl: float, ref_wl: float) -> float:
    """Comb-style scoring: rectangular window (accept/reject)."""
    return 1.0 if abs(peak_wl - ref_wl) < 0.08 else 0.0


def _correlation_score(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    ref_wl: float,
) -> float:
    """Correlation-style scoring: local cross-correlation with template Gaussian."""
    mask = np.abs(wavelength - ref_wl) < 0.3
    if mask.sum() < 5:
        return 0.0
    template = np.exp(-0.5 * ((wavelength[mask] - ref_wl) / _SIGMA_NM) ** 2)
    data = intensity[mask]
    if np.std(data) < 1e-12 or np.std(template) < 1e-12:
        return 0.0
    return float(np.corrcoef(data, template)[0, 1])


def _build_id_pipeline(mode: str):
    """Factory for line-ID pipeline variants."""

    def pipeline(
        wavelength: np.ndarray,
        intensity: np.ndarray,
        elements: List[str],
    ) -> Dict[str, Any]:
        peaks = _detect_peaks(wavelength, intensity)
        observations: List[LineObservation] = []

        for peak in peaks:
            wl_p = peak["wavelength_nm"]
            best_score = -1.0
            best_line = None
            best_elem = None

            for elem in elements:
                for line in _REFERENCE_LINES.get(elem, []):
                    ref_wl = line[0]
                    if mode == "alias":
                        score = _alias_score(wl_p, ref_wl)
                    elif mode == "comb":
                        score = _comb_score(wl_p, ref_wl)
                    elif mode == "correlation":
                        score = _correlation_score(wavelength, intensity, ref_wl)
                    else:
                        score = 0.0

                    if score > best_score:
                        best_score = score
                        best_line = line
                        best_elem = elem

            # Threshold: require minimum confidence
            min_score = 0.3 if mode == "correlation" else 0.1
            if best_line is not None and best_elem is not None and best_score > min_score:
                wl_ref, E_k, g_k, log_A = best_line
                observations.append(
                    LineObservation(
                        wavelength_nm=wl_ref,
                        intensity=peak["intensity"],
                        intensity_uncertainty=max(peak["intensity"] * 0.05, 1.0),
                        element=best_elem,
                        ionization_stage=1,
                        E_k_ev=E_k,
                        g_k=g_k,
                        A_ki=10.0**log_A,
                    )
                )

        # Boltzmann fit + closure (same as baseline)
        if len(observations) < 2:
            n = len(elements)
            return {"concentrations": {el: 1.0 / n for el in elements}}

        fitter = BoltzmannPlotFitter(method=FitMethod.SIGMA_CLIP)
        element_intercepts: Dict[str, float] = {}
        temperatures: List[float] = []

        for elem in elements:
            elem_obs = [o for o in observations if o.element == elem]
            if len(elem_obs) < 2:
                continue
            result = fitter.fit(elem_obs)
            if result.temperature_K > 0 and np.isfinite(result.temperature_K):
                temperatures.append(result.temperature_K)
                element_intercepts[elem] = result.intercept

        if not element_intercepts:
            n = len(elements)
            return {"concentrations": {el: 1.0 / n for el in elements}}

        T_mean = float(np.mean(temperatures)) if temperatures else 10000.0
        raw = {el: np.exp(b) for el, b in element_intercepts.items()}
        total = sum(raw.values())
        concentrations = {el: v / total for el, v in raw.items()}
        for el in elements:
            if el not in concentrations:
                concentrations[el] = 0.0
        total = sum(concentrations.values())
        if total > 0:
            concentrations = {el: v / total for el, v in concentrations.items()}

        return {
            "concentrations": concentrations,
            "temperature_K": T_mean,
            "electron_density_cm3": 1e17,
        }

    return pipeline


def register_line_id(harness: BenchmarkHarness) -> None:
    """Register E1 line-identification pipeline variants."""
    for mode in ("alias", "comb", "correlation"):
        harness.register_pipeline(f"E1_lineID_{mode}", _build_id_pipeline(mode))


# ============================================================================
# Experiment: E2 — Boltzmann Fitting x Self-Absorption Matrix
# ============================================================================


_FIT_METHODS = {
    "SIGMA_CLIP": FitMethod.SIGMA_CLIP,
    "RANSAC": FitMethod.RANSAC,
    "HUBER": FitMethod.HUBER,
}

_SA_MODES = ("NONE", "COG", "CDSB")


def register_boltzmann_sa_matrix(harness: BenchmarkHarness) -> None:
    """Register the 3x3 grid of (Boltzmann method) x (self-absorption mode)."""
    for fm_name, fm_enum in _FIT_METHODS.items():
        for sa in _SA_MODES:

            def _make_pipeline(method: FitMethod, sa_mode: str):
                def pipeline(
                    wavelength: np.ndarray,
                    intensity: np.ndarray,
                    elements: List[str],
                ) -> Dict[str, Any]:
                    return _boltzmann_pipeline(
                        wavelength,
                        intensity,
                        elements,
                        fit_method=method,
                        self_absorption=sa_mode,
                    )

                return pipeline

            name = f"E2_{fm_name}_{sa}"
            harness.register_pipeline(name, _make_pipeline(fm_enum, sa))


# ============================================================================
# Summary printing
# ============================================================================


def _print_summary(report) -> None:
    """Pretty-print the benchmark summary table."""
    summary = report.summary()
    if not summary:
        print("No results to display.")
        return

    header = (
        f"{'Pipeline':<30s}  {'N':>3s}  {'Err':>3s}  "
        f"{'Mean Ait':>8s}  {'Med Ait':>8s}  {'P95 Ait':>8s}  "
        f"{'Mean ms':>8s}  {'P95 ms':>8s}  "
        f"{'EXC':>3s} {'GOOD':>4s} {'ACC':>3s} {'POOR':>4s}"
    )
    print()
    print(header)
    print("-" * len(header))

    for name, s in sorted(summary.items()):
        tiers = s.get("tier_distribution", {})
        print(
            f"{name:<30s}  "
            f"{s.get('n_spectra', 0):>3d}  "
            f"{s.get('n_errors', 0):>3d}  "
            f"{s.get('mean_aitchison', float('nan')):>8.4f}  "
            f"{s.get('median_aitchison', float('nan')):>8.4f}  "
            f"{s.get('p95_aitchison', float('nan')):>8.4f}  "
            f"{s.get('mean_time_ms', float('nan')):>8.2f}  "
            f"{s.get('p95_time_ms', float('nan')):>8.2f}  "
            f"{tiers.get('EXCELLENT', 0):>3d} "
            f"{tiers.get('GOOD', 0):>4d} "
            f"{tiers.get('ACCEPTABLE', 0):>3d} "
            f"{tiers.get('POOR', 0):>4d}"
        )
    print()


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CF-LIBS benchmark experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=["T0.2", "E1", "E2"],
        choices=["T0.2", "E1", "E2"],
        help="Experiments to run (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/experiments"),
        help="Directory for JSON result files.",
    )
    parser.add_argument(
        "--n-temperatures",
        type=int,
        default=3,
        help="Number of temperature points to sweep (default: 3).",
    )
    parser.add_argument(
        "--snr",
        type=float,
        nargs="*",
        default=None,
        help="SNR values for noisy variants (default: clean only).",
    )
    args = parser.parse_args()

    experiments = set(args.experiments)

    # ---- Corpus ---------------------------------------------------------
    print("Generating benchmark corpus ...")
    temps = np.linspace(8000, 15000, args.n_temperatures).tolist()
    corpus = BenchmarkCorpus(
        temperatures_K=temps,
        electron_densities_cm3=[1e16, 1e17],
        snr_values=args.snr,
    ).generate()
    print(f"  {len(corpus)} spectra generated.")

    # ---- Register pipelines --------------------------------------------
    harness = BenchmarkHarness()

    if "T0.2" in experiments:
        register_baseline(harness)
    if "E1" in experiments:
        register_line_id(harness)
    if "E2" in experiments:
        register_boltzmann_sa_matrix(harness)

    # ---- Run -----------------------------------------------------------
    print("Running pipelines ...")
    t0 = time.perf_counter()
    report = harness.run(corpus)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f} s.")

    # ---- Print summary -------------------------------------------------
    _print_summary(report)

    # ---- Save ----------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"experiment_results_{ts}.json"
    out_path.write_text(report.to_json())
    print(f"Results saved to {out_path}")

    # Also write a latest symlink / copy for convenience
    latest = args.output_dir / "latest_results.json"
    latest.write_text(report.to_json())
    print(f"Latest copy at  {latest}")


if __name__ == "__main__":
    main()
