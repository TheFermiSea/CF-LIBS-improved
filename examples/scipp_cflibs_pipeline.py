#!/usr/bin/env python
"""Extract a spectrum from a Scipp HDF5 file and run CF-LIBS classic inversion.

Usage:
  .venv/bin/python examples/scipp_cflibs_pipeline.py --input data/20shot.h5

Outputs (default in outputs/scipp_20shot/):
  - extracted_spectrum.csv
  - spectrum_fit.png
  - boltzmann_plot.png
  - cflibs_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple

import h5py
import numpy as np


def _prepare_matplotlib_env(output_dir: Path) -> None:
    """Ensure matplotlib uses a writable config/cache directory."""
    mpl_dir = output_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def _load_inversion_modules(repo_root: Path):
    """Load cflibs inversion submodules without triggering heavy __init__ imports."""
    inversion_dir = repo_root / "cflibs" / "inversion"
    package_name = "cflibs.inversion"

    if package_name not in sys.modules:
        pkg = ModuleType(package_name)
        pkg.__path__ = [str(inversion_dir)]
        sys.modules[package_name] = pkg

    def load_module(name: str):
        module_path = inversion_dir / f"{name}.py"
        spec = __import__("importlib.util").util.spec_from_file_location(
            f"{package_name}.{name}", module_path
        )
        module = __import__("importlib.util").util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    boltzmann = load_module("boltzmann")
    line_detection = load_module("line_detection")
    line_selection = load_module("line_selection")
    solver = load_module("solver")
    return boltzmann, line_detection, line_selection, solver


def _select_spectrum(data: np.ndarray, strategy: str) -> Tuple[int, int]:
    """Select a spectrum index (x, shot) from [X, W, S] data."""
    max_vals = data.max(axis=1)
    mean_vals = data.mean(axis=1)
    if strategy == "peakiness":
        metric = max_vals / np.maximum(mean_vals, 1e-9)
    elif strategy == "total":
        metric = data.sum(axis=1)
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
    return tuple(int(i) for i in np.unravel_index(np.argmax(metric), metric.shape))


def _gaussian_sum(
    wavelength: np.ndarray,
    lines: List[object],
    fwhm_nm: float,
) -> np.ndarray:
    """Construct a simple Gaussian line model from observations.

    Uses each line's integrated intensity as the Gaussian area.
    """
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if sigma <= 0:
        sigma = 0.01

    model = np.zeros_like(wavelength, dtype=float)
    norm = sigma * np.sqrt(2.0 * np.pi)
    for obs in lines:
        mu = obs.wavelength_nm
        area = max(obs.intensity, 0.0)
        amp = area / norm if norm > 0 else 0.0
        model += amp * np.exp(-0.5 * ((wavelength - mu) / sigma) ** 2)
    return model


def _atomic_percent(
    concentrations: Dict[str, float], atomic_masses: Dict[str, float]
) -> Dict[str, float]:
    """Convert mass fractions to atomic percent."""
    atomic_moles = {}
    for el, mass_frac in concentrations.items():
        mass = atomic_masses.get(el)
        if mass is None or mass <= 0:
            continue
        atomic_moles[el] = mass_frac / mass

    total = sum(atomic_moles.values())
    if total <= 0:
        return {}

    return {el: 100.0 * moles / total for el, moles in atomic_moles.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a spectrum from a Scipp HDF5 file and run CF-LIBS analysis."
    )
    parser.add_argument("--input", required=True, help="Path to Scipp HDF5 file")
    parser.add_argument("--output-dir", default="outputs/scipp_20shot", help="Output folder")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to libs_production.db (defaults to repo root)",
    )
    parser.add_argument("--x-index", type=int, default=None, help="X index to use")
    parser.add_argument("--shot-index", type=int, default=None, help="Shot index to use")
    parser.add_argument(
        "--selection",
        default="peakiness",
        choices=["peakiness", "total"],
        help="Auto-selection metric for spectrum",
    )
    parser.add_argument("--y-index", type=int, default=0, help="Y index (default 0)")
    parser.add_argument(
        "--peak-width-nm",
        type=float,
        default=0.2,
        help="Peak width used for line detection and fit",
    )
    parser.add_argument(
        "--min-peak-height",
        type=float,
        default=0.02,
        help="Minimum peak height fraction for detection",
    )
    parser.add_argument(
        "--wavelength-tolerance-nm",
        type=float,
        default=0.05,
        help="Wavelength tolerance for matching",
    )
    parser.add_argument(
        "--max-lines-per-element",
        type=int,
        default=15,
        help="Max lines per element for selection",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_matplotlib_env(output_dir)

    # Avoid accidental writes outside workspace by steering ArviZ cache if imported.
    os.environ.setdefault("ARVIZ_DATA_DIR", str(output_dir / ".arviz"))

    # Load inversion modules (without heavy __init__)
    boltzmann, line_detection, line_selection, solver = _load_inversion_modules(repo_root)

    # Load Scipp-style HDF5
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with h5py.File(input_path, "r") as f:
        wl = f["coords/elem_002_Wavelength/values"][:]
        data = f["data/values"]
        # shape: (X, Y, W, S)
        if data.ndim != 4:
            raise ValueError(f"Expected 4D data array, got shape {data.shape}")
        X, Y, W, S = data.shape

        y_idx = int(args.y_index)
        if not (0 <= y_idx < Y):
            raise ValueError(f"y-index {y_idx} out of range 0..{Y-1}")

        spectra = data[:, y_idx, :, :]
        if args.x_index is None or args.shot_index is None:
            x_idx, shot_idx = _select_spectrum(spectra, args.selection)
        else:
            x_idx = int(args.x_index)
            shot_idx = int(args.shot_index)

        if not (0 <= x_idx < X):
            raise ValueError(f"x-index {x_idx} out of range 0..{X-1}")
        if not (0 <= shot_idx < S):
            raise ValueError(f"shot-index {shot_idx} out of range 0..{S-1}")

        intensity = spectra[x_idx, :, shot_idx]

    # Baseline correction
    intensity = np.asarray(intensity, dtype=float)
    intensity = intensity - np.nanmin(intensity)
    intensity = np.clip(intensity, 0.0, None)

    # Save extracted spectrum
    spectrum_csv = output_dir / "extracted_spectrum.csv"
    np.savetxt(
        spectrum_csv,
        np.column_stack([wl, intensity]),
        delimiter=",",
        header="wavelength_nm,intensity",
        comments="",
    )

    from cflibs.atomic.database import AtomicDatabase

    db_path = Path(args.db_path) if args.db_path else (repo_root / "libs_production.db")
    atomic_db = AtomicDatabase(str(db_path))
    elements = atomic_db.get_available_elements()

    detection = line_detection.detect_line_observations(
        wavelength=wl,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=elements,
        wavelength_tolerance_nm=args.wavelength_tolerance_nm,
        min_peak_height=args.min_peak_height,
        peak_width_nm=args.peak_width_nm,
    )

    selector = line_selection.LineSelector(
        min_snr=10.0,
        min_energy_spread_ev=2.0,
        min_lines_per_element=3,
        exclude_resonance=True,
        isolation_wavelength_nm=0.1,
        max_lines_per_element=args.max_lines_per_element,
    )
    selection = selector.select(
        detection.observations, resonance_lines=detection.resonance_lines
    )
    selected_lines = selection.selected_lines

    if len(selected_lines) == 0:
        raise RuntimeError("No usable spectral lines detected for inversion.")

    # Run iterative solver
    solver_instance = solver.IterativeCFLIBSSolver(
        atomic_db=atomic_db,
        max_iterations=20,
        t_tolerance_k=100.0,
        ne_tolerance_frac=0.1,
        pressure_pa=101325.0,
    )
    result = solver_instance.solve(selected_lines, closure_mode="standard")

    # Convert concentrations to atomic percent
    atomic_masses = {el: atomic_db.get_atomic_mass(el) for el in result.concentrations}
    atomic_percent = _atomic_percent(result.concentrations, atomic_masses)

    # Build a simple fitted spectrum from matched lines
    fit_lines = detection.observations
    fitted = _gaussian_sum(wl, fit_lines, fwhm_nm=args.peak_width_nm)
    if fitted.max() > 0:
        fitted *= intensity.max() / fitted.max()

    # Plot spectrum + fit + labels
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wl, intensity, label="Raw spectrum", color="#1f77b4", linewidth=1.0)
    ax.plot(wl, fitted, label="Line-model fit", color="#ff7f0e", linewidth=1.0)

    # Label peaks with element symbols
    label_offsets = np.linspace(0.02, 0.12, 6)
    for i, obs in enumerate(fit_lines):
        idx = np.searchsorted(wl, obs.wavelength_nm)
        idx = np.clip(idx, 0, len(wl) - 1)
        y = intensity[idx]
        offset = label_offsets[i % len(label_offsets)] * intensity.max()
        ax.text(
            obs.wavelength_nm,
            y + offset,
            f"{obs.element}",
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=7,
            color="#444444",
        )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("LIBS Spectrum with Line-Model Fit")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    spectrum_plot = output_dir / "spectrum_fit.png"
    fig.tight_layout()
    fig.savefig(spectrum_plot, dpi=200)
    plt.close(fig)

    # Boltzmann plot
    fitter = boltzmann.BoltzmannPlotFitter(outlier_sigma=2.5)
    fit_result = fitter.fit(selected_lines)

    fig, ax = plt.subplots(figsize=(7, 5))
    elements_in_fit = sorted({obs.element for obs in selected_lines})
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(elements_in_fit), 1)))
    color_map = {el: colors[i % len(colors)] for i, el in enumerate(elements_in_fit)}

    for obs in selected_lines:
        ax.scatter(
            obs.E_k_ev,
            obs.y_value,
            color=color_map.get(obs.element, "#1f77b4"),
            s=18,
            alpha=0.8,
        )

    x_vals = np.array([o.E_k_ev for o in selected_lines])
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    x_fit = np.linspace(x_min, x_max, 200)
    y_fit = fit_result.slope * x_fit + fit_result.intercept
    ax.plot(x_fit, y_fit, color="black", linestyle="--", label="Boltzmann fit")

    ax.set_xlabel("Upper level energy E_k (eV)")
    ax.set_ylabel("ln(I * λ / (g A))")
    ax.set_title(
        f"Boltzmann Plot (T = {fit_result.temperature_K:.0f} K, R² = {fit_result.r_squared:.3f})"
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    boltzmann_plot = output_dir / "boltzmann_plot.png"
    fig.tight_layout()
    fig.savefig(boltzmann_plot, dpi=200)
    plt.close(fig)

    # Results summary
    results = {
        "input_file": str(input_path),
        "selected_spectrum": {
            "x_index": x_idx,
            "y_index": y_idx,
            "shot_index": shot_idx,
            "selection_strategy": args.selection,
        },
        "wavelength_range_nm": [float(wl.min()), float(wl.max())],
        "line_detection": {
            "total_peaks": detection.total_peaks,
            "matched_peaks": detection.matched_peaks,
            "unmatched_peaks": detection.unmatched_peaks,
            "warnings": detection.warnings,
        },
        "line_selection": {
            "selected_lines": len(selected_lines),
            "rejected_lines": len(selection.rejected_lines),
            "warnings": selection.warnings,
            "energy_spread_ev": selection.energy_spread_ev,
            "n_elements": selection.n_elements,
        },
        "boltzmann_fit": {
            "temperature_K": fit_result.temperature_K,
            "temperature_uncertainty_K": fit_result.temperature_uncertainty_K,
            "r_squared": fit_result.r_squared,
            "n_points": fit_result.n_points,
            "rejected_points": fit_result.rejected_points,
        },
        "cflibs_result": {
            "temperature_K": result.temperature_K,
            "temperature_uncertainty_K": result.temperature_uncertainty_K,
            "electron_density_cm3": result.electron_density_cm3,
            "electron_density_uncertainty_cm3": result.electron_density_uncertainty_cm3,
            "concentrations_mass_fraction": result.concentrations,
            "concentrations_atomic_percent": atomic_percent,
            "iterations": result.iterations,
            "converged": result.converged,
            "quality_metrics": result.quality_metrics,
        },
        "output_files": {
            "spectrum_csv": str(spectrum_csv),
            "spectrum_plot": str(spectrum_plot),
            "boltzmann_plot": str(boltzmann_plot),
        },
    }

    results_path = output_dir / "cflibs_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    print("CF-LIBS pipeline complete")
    print(f"  Spectrum CSV: {spectrum_csv}")
    print(f"  Spectrum plot: {spectrum_plot}")
    print(f"  Boltzmann plot: {boltzmann_plot}")
    print(f"  Results JSON: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
