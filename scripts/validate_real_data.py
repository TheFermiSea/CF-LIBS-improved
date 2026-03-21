#!/usr/bin/env python
"""
Legacy/reference validation script for CF-LIBS element identification on real experimental data.

This remains available as a smoke-check and exploratory validator for
representative spectra, but it is no longer the authoritative benchmark
entrypoint.

Tests ALIAS, Comb, and Correlation algorithms on LIBS spectra from data/ directory.
Generates comparison tables and plots showing algorithm performance.

Usage:
    python scripts/validate_real_data.py
    python scripts/validate_real_data.py --datasets steel_245nm FeNi_380nm
    python scripts/validate_real_data.py --no-plots
    python scripts/validate_real_data.py --elements Fe Ni Cr Ti --data-dir my_data/
"""

import os

# MUST be before any cflibs imports
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Headless plotting
import matplotlib.pyplot as plt
import xarray as xr
import h5py

from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.alias_identifier import ALIASIdentifier
from cflibs.inversion.comb_identifier import CombIdentifier
from cflibs.inversion.correlation_identifier import CorrelationIdentifier
from cflibs.inversion.element_id import ElementIdentificationResult
from cflibs.inversion.wavelength_calibration import calibrate_wavelength_axis


# ============================================================================
# Data Loading
# ============================================================================


def load_netcdf(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load NetCDF file (steel_245nm, FeNi_380nm, FeNi_480nm, FeNi (1).nc).

    Returns
    -------
    wavelength : np.ndarray
        1D wavelength array in nm, shape (N,)
    intensities : np.ndarray
        2D or 1D intensity array, shape (X, Y, N) or (N,)
    metadata : dict
        Metadata extracted from dataset
    """
    with xr.open_dataset(path) as ds:
        wavelength = ds.coords["Wavelength"].values

        # Handle different variable names
        if "__xarray_dataarray_variable__" in ds:
            data = ds["__xarray_dataarray_variable__"].values
        elif "Intensity" in ds:
            data = ds["Intensity"].values
        else:
            raise ValueError(f"Unknown data variable in {path}")

        metadata = {
            "format": "netcdf",
            "shape": data.shape,
            "wavelength_range": (wavelength.min(), wavelength.max()),
        }

    return wavelength, data, metadata


def load_hdf5(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load HDF5 file (Fe_245nm, Ni_245nm).

    Returns
    -------
    wavelength : np.ndarray
        1D wavelength array in nm, shape (N,)
    intensities : np.ndarray
        3D intensity array, shape (X, Y, N)
    metadata : dict
        Metadata extracted from file
    """
    with h5py.File(path, "r") as f:
        wavelength = f["Wavelength"][:]
        data = f["__xarray_dataarray_variable__"][:]

    metadata = {
        "format": "hdf5",
        "shape": data.shape,
        "wavelength_range": (wavelength.min(), wavelength.max()),
    }

    return wavelength, data, metadata


def load_hdf5_multishot(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load multi-shot HDF5 file (20shot.h5) and average over shots.

    Returns
    -------
    wavelength : np.ndarray
        1D wavelength array in nm, shape (N,)
    intensities : np.ndarray
        3D intensity array averaged over shots, shape (X, Y, N)
    metadata : dict
        Metadata extracted from file
    """
    with h5py.File(path, "r") as f:
        wavelength = f["coords/elem_002_Wavelength/values"][:]
        data_raw = f["data/values"][:]  # shape (39, 1, 2560, 20)

    # Average over shots axis (last axis)
    data = data_raw.mean(axis=-1)  # shape (39, 1, 2560)

    metadata = {
        "format": "hdf5_multishot",
        "shape": data.shape,
        "n_shots": data_raw.shape[-1],
        "wavelength_range": (wavelength.min(), wavelength.max()),
    }

    return wavelength, data, metadata


def _discover_scipp_coords(f: h5py.File) -> Dict[str, np.ndarray]:
    """
    Discover coordinates in a scipp HDF5 file dynamically.

    Reads the ``name`` attribute from each group under ``coords/`` to identify
    coordinates (robust across scipp v23.03–v25.5).

    Returns
    -------
    coords : dict
        {coord_name: values_array}
    """
    coords = {}
    for group_key in f["coords"]:
        group = f[f"coords/{group_key}"]
        if isinstance(group, h5py.Group) and "name" in group.attrs:
            name = str(group.attrs["name"])
            if "values" in group:
                coords[name] = group["values"][:]
    return coords


def load_scipp(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load a scipp-format HDF5 file with dynamic coordinate discovery.

    Handles 2D data of shape ``(N_pos, 2560)`` from single-ablation experiments
    (AA1100, Ti6Al4V, ppm files).

    Returns
    -------
    wavelength : np.ndarray
        1D wavelength array in nm, shape (N,)
    intensities : np.ndarray
        2D intensity array, shape (N_pos, N)
    metadata : dict
        Metadata extracted from file
    """
    with h5py.File(path, "r") as f:
        coords = _discover_scipp_coords(f)
        if "Wavelength" not in coords:
            raise ValueError(f"No 'Wavelength' coordinate found in {path}")
        wavelength = coords["Wavelength"]
        data = f["data/values"][:]

    metadata = {
        "format": "scipp",
        "shape": data.shape,
        "wavelength_range": (wavelength.min(), wavelength.max()),
        "coords": list(coords.keys()),
    }

    return wavelength, data, metadata


def load_scipp_depth_scan(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load a scipp-format HDF5 depth-scan file with 3D data.

    Data shape is ``(N_pos, 2560, N_iter)`` — averages over the iteration axis
    for the primary spectrum, but stores raw shape for robustness reporting.

    Returns
    -------
    wavelength : np.ndarray
        1D wavelength array in nm, shape (N,)
    intensities : np.ndarray
        2D intensity array averaged over iterations, shape (N_pos, N)
    metadata : dict
        Metadata including n_iterations and raw_shape
    """
    with h5py.File(path, "r") as f:
        coords = _discover_scipp_coords(f)
        if "Wavelength" not in coords:
            raise ValueError(f"No 'Wavelength' coordinate found in {path}")
        wavelength = coords["Wavelength"]
        data_raw = f["data/values"][:]  # shape (N_pos, 2560, N_iter)

    # Average over iteration axis (last dim)
    data = data_raw.mean(axis=-1)  # shape (N_pos, 2560)

    metadata = {
        "format": "scipp_depth_scan",
        "shape": data.shape,
        "raw_shape": data_raw.shape,
        "n_iterations": data_raw.shape[-1],
        "wavelength_range": (wavelength.min(), wavelength.max()),
        "coords": list(coords.keys()),
    }

    return wavelength, data, metadata


# ============================================================================
# Dataset Registry
# ============================================================================

DATASETS = [
    {
        "name": "steel_245nm",
        "path": "steel_245nm.nc",
        "loader": "netcdf",
        "elements": ["Fe", "Cr", "Ni", "Mn", "Cu", "Ti", "Si"],
        "expected": ["Fe", "Cr", "Ni", "Mn"],
        "range": "245nm",
    },
    {
        "name": "Fe_245nm",
        "path": "Fe_245nm",
        "loader": "hdf5",
        "elements": ["Fe", "Ni", "Cr", "Mn", "Cu", "Ti", "Si"],
        "expected": ["Fe"],
        "range": "245nm",
    },
    {
        "name": "Ni_245nm",
        "path": "Ni_245nm",
        "loader": "hdf5",
        "elements": ["Fe", "Ni", "Cr", "Mn", "Cu", "Ti", "Si"],
        "expected": ["Ni"],
        "range": "245nm",
    },
    {
        "name": "FeNi_380nm",
        "path": "FeNi_380nm.nc",
        "loader": "netcdf",
        "elements": ["Fe", "Ni", "Cr", "Mn"],
        "expected": ["Fe", "Ni"],
        "range": "380nm",
    },
    {
        "name": "FeNi_480nm",
        "path": "FeNi_480nm.nc",
        "loader": "netcdf",
        "elements": ["Fe", "Cr", "Ti", "Ni", "Cu"],
        "expected": ["Fe", "Cr"],
        "range": "480nm",
    },
    {
        "name": "FeNi_single",
        "path": "FeNi (1).nc",
        "loader": "netcdf",
        "elements": ["Fe", "Cr", "Ti", "Ni", "Cu"],
        "expected": ["Fe", "Cr"],
        "range": "480nm",
    },
    {
        "name": "20shot",
        "path": "20shot.h5",
        "loader": "hdf5_multishot",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn"],
        "expected": ["Fe", "Ti", "Ni"],
        "range": "286nm",
    },
    # --- Scipp datasets (new) ---
    # All scipp datasets are from compact broadband spectrometers with RP≈500
    # (FWHM ~0.6 nm). Auto-detection is unreliable on self-absorbed/blended
    # features, so we set RP explicitly based on Gaussian fits to isolated
    # Mg I 285.2 nm and Mn I 280.0 nm lines.
    {
        "name": "AA1100_substrate",
        "path": "AA1100_Substrate.h5",
        "loader": "scipp",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": ["Al"],
        "range": "full",
        "resolving_power": 500,
    },
    {
        "name": "Ti6Al4V_substrate",
        "path": "Ti6Al4V_substrate.h5",
        "loader": "scipp",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": ["Ti", "Al", "V"],
        "range": "full",
        "resolving_power": 500,
    },
    # --- 10000ppm blind tests ---
    {
        "name": "10000ppm_400W",
        "path": "10000ppm/400.h5",
        "loader": "scipp",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
    {
        "name": "10000ppm_600W",
        "path": "10000ppm/600.h5",
        "loader": "scipp",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
    {
        "name": "10000ppm_800W",
        "path": "10000ppm/800.h5",
        "loader": "scipp",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
    {
        "name": "10000ppm_400W_depth",
        "path": "10000ppm/400_depth_scan.h5",
        "loader": "scipp_depth_scan",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
    {
        "name": "10000ppm_600W_depth",
        "path": "10000ppm/600_depth_scan.h5",
        "loader": "scipp_depth_scan",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
    {
        "name": "10000ppm_800W_depth",
        "path": "10000ppm/800_depth_scan.h5",
        "loader": "scipp_depth_scan",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
    # --- 210000ppm blind tests ---
    {
        "name": "210000ppm_400W",
        "path": "210000ppm/400W.h5",
        "loader": "scipp",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
    {
        "name": "210000ppm_600W",
        "path": "210000ppm/600.h5",
        "loader": "scipp",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
    {
        "name": "210000ppm_800W",
        "path": "210000ppm/800.h5",
        "loader": "scipp",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
    {
        "name": "210000ppm_400W_depth",
        "path": "210000ppm/400W_depth_scan.h5",
        "loader": "scipp_depth_scan",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
    {
        "name": "210000ppm_600W_depth",
        "path": "210000ppm/600_depth_scan.h5",
        "loader": "scipp_depth_scan",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
    {
        "name": "210000ppm_800W_depth",
        "path": "210000ppm/800_depth_scan.h5",
        "loader": "scipp_depth_scan",
        "elements": ["Fe", "Ti", "Ni", "Cr", "Mn", "Cu", "Si", "Al", "V", "Mg", "Co"],
        "expected": None,
        "range": "full",
        "resolving_power": 500,
    },
]

BENCHMARK_CRITERIA = {
    "min_recall": 0.60,
    "max_fpr": 0.20,
    "required_detections": {
        ("Fe_245nm", "Fe"),
        ("Ni_245nm", "Ni"),
        ("steel_245nm", "Fe"),
    },
    "required_absences": {
        ("Fe_245nm", "Ni"),
        ("Fe_245nm", "Cu"),
        ("Ni_245nm", "Fe"),
    },
}


# ============================================================================
# Spectrum Selection
# ============================================================================


def select_representative_spectrum(data: np.ndarray, dataset_name: str) -> np.ndarray:
    """
    Select a representative 1D spectrum from multi-dimensional data.

    This helper is used by the legacy/reference validation script to collapse
    multidimensional inputs into a single smoke-test spectrum.

    For 3D spatial datasets (steel, Fe, Ni), averages over a 3x3 neighborhood
    around the center to improve SNR. For line scan datasets, uses single pixel.

    Parameters
    ----------
    data : np.ndarray
        Intensity array, shape (N,) or (X, Y, N) or (X, 1, N)
    dataset_name : str
        Name of dataset for selection strategy

    Returns
    -------
    spectrum : np.ndarray
        1D intensity array, shape (N,)
    """
    if data.ndim == 1:
        return data

    if data.ndim == 2:
        # 2D scipp data (N_pos, N_wavelength): average all positions for SNR
        return np.mean(data, axis=0)

    if data.ndim == 3:
        nx, ny, nw = data.shape

        if ny > 1:
            # Small grid (e.g. 3×3): center pixel — averaging brings up
            # noise artefacts that cause false positives on minor elements.
            if nx * ny <= 16:
                return data[nx // 2, ny // 2, :]
            # Larger grid: average all positions
            return np.mean(data.reshape(-1, nw), axis=0)

        # Line scan (N, 1, W): average central positions for SNR
        n_avg = min(nx, 5)
        start = max(0, nx // 2 - n_avg // 2)
        return np.mean(data[start : start + n_avg, 0, :], axis=0)

    raise ValueError(f"Unexpected data shape: {data.shape}")


# ============================================================================
# Element Identification
# ============================================================================


def estimate_resolving_power(wavelength: np.ndarray, intensity: np.ndarray) -> float:
    """
    Estimate instrument resolving power from isolated peaks.

    Finds candidate peaks, fits Gaussians, and takes the median RP from
    peaks with reasonable FWHM (avoiding broad blends). Falls back to
    5000.0 if fitting fails.

    Returns
    -------
    resolving_power : float
    """
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit

    max_intensity = float(np.max(intensity))
    if max_intensity <= 0 or intensity.size == 0:
        return 5000.0
    norm = intensity / max_intensity
    peaks, _ = find_peaks(norm, height=0.1, prominence=0.05, distance=15)
    if len(peaks) == 0:
        return 5000.0

    def gaussian(x, A, mu, sigma, bg):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + bg

    rp_estimates = []
    pixel_spacing = np.median(np.diff(wavelength))

    for pk in peaks:
        center_wl = wavelength[pk]
        # Narrow window (±1 nm) to favor isolated peaks
        mask = (wavelength > center_wl - 1.0) & (wavelength < center_wl + 1.0)
        wl_win = wavelength[mask]
        sp_win = intensity[mask]
        if len(wl_win) < 10:
            continue

        try:
            p0 = [intensity[pk], center_wl, 0.1, np.median(intensity)]
            popt, _ = curve_fit(gaussian, wl_win, sp_win, p0=p0, maxfev=3000)
            fwhm = 2.355 * abs(popt[2])
            # Skip if FWHM is unreasonably narrow (<2 pixels) or broad (>2 nm)
            if fwhm < 2 * pixel_spacing or fwhm > 2.0:
                continue
            rp = abs(popt[1]) / fwhm
            rp_estimates.append(rp)
        except Exception:
            continue

    if rp_estimates:
        # Use median to reject outliers from blended features
        rp = float(np.median(rp_estimates))
        return max(100.0, min(rp, 20000.0))

    return 5000.0


def run_all_identifiers(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    db: AtomicDatabase,
    elements: List[str],
    resolving_power: float = 5000.0,
) -> Dict[str, Optional[ElementIdentificationResult]]:
    """
    Run ALIAS, Comb, and Correlation on same spectrum.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm, shape (N,)
    intensity : np.ndarray
        Intensity array, shape (N,)
    db : AtomicDatabase
        Atomic database instance
    elements : List[str]
        Elements to search for
    resolving_power : float
        Resolving power for ALIAS (default: 5000.0)

    Returns
    -------
    results : dict
        {algorithm_name: ElementIdentificationResult or None}
    """
    results = {}

    algorithms = [
        (
            "ALIAS",
            ALIASIdentifier,
            {
                "resolving_power": resolving_power,
                # Tuned on labeled real datasets (Fe/Ni/steel + full benchmark set)
                "intensity_threshold_factor": 3.0,
                "detection_threshold": 0.01,
                "chance_window_scale": 0.3,
            },
        ),
        (
            "Comb",
            CombIdentifier,
            {
                "resolving_power": resolving_power,
                "min_correlation": 0.08,
                "tooth_activation_threshold": 0.35,
                "relative_threshold_scale": 1.4,
                "min_aki_gk": 3000.0,
            },
        ),
        (
            "Correlation",
            CorrelationIdentifier,
            {
                "resolving_power": resolving_power,
                "min_confidence": 0.008,
                "relative_threshold_scale": 1.2,
                "min_line_strength": 1000.0,
                "T_range_K": (5000, 15000),
                "T_steps": 7,
                "n_e_range_cm3": (1e15, 5e17),
                "n_e_steps": 4,
            },
        ),
    ]

    for name, Cls, kwargs in algorithms:
        try:
            identifier = Cls(db, elements=elements, **kwargs)
            if name == "Correlation":
                result = identifier.identify(wavelength, intensity, mode="classic")
            else:
                result = identifier.identify(wavelength, intensity)
            results[name] = result
        except Exception as e:
            print(f"  WARNING: {name} failed: {e}")
            results[name] = None

    return results


# ============================================================================
# Results Display
# ============================================================================


def print_result_table(
    results: Dict[str, Optional[ElementIdentificationResult]],
    dataset_name: str,
    expected: Optional[List[str]],
):
    """
    Print formatted table of results.

    Columns: Algorithm | Element | Detected | Score | Confidence | Matched Lines

    Expected elements marked with * when detected correctly.
    For blind-test datasets (expected=None), shows "BLIND TEST" instead.
    """
    is_blind = expected is None
    expected = expected or []

    print(f"\n{'='*100}")
    print(f"Dataset: {dataset_name}")
    if is_blind:
        print("Expected elements: BLIND TEST (discovery mode)")
    else:
        print(f"Expected elements: {', '.join(expected)}")
    print(f"{'='*100}")

    # Header
    print(
        f"{'Algorithm':<15} {'Element':<8} {'Detected':<10} {'Score':<8} {'Confidence':<12} {'Matched Lines':<15}"
    )
    print("-" * 100)

    # Collect all elements across algorithms
    all_elements = set()
    for result in results.values():
        if result is not None:
            all_elements.update([e.element for e in result.all_elements])

    # Sort elements
    sorted_elements = sorted(all_elements)

    # Print results per algorithm
    for algo_name in ["ALIAS", "Comb", "Correlation"]:
        result = results.get(algo_name)
        if result is None:
            print(f"{algo_name:<15} {'FAILED':<8} {'-':<10} {'-':<8} {'-':<12} {'-':<15}")
            continue

        # Build element lookup
        elem_dict = {e.element: e for e in result.all_elements}

        for elem in sorted_elements:
            if elem in elem_dict:
                e = elem_dict[elem]
                if is_blind:
                    detected_str = "YES" if e.detected else "NO"
                else:
                    detected_str = (
                        "YES*"
                        if (e.detected and elem in expected)
                        else ("YES" if e.detected else "NO")
                    )
                matched_str = f"{e.n_matched_lines}/{e.n_total_lines}"
                print(
                    f"{algo_name:<15} {e.element:<8} {detected_str:<10} "
                    f"{e.score:<8.3f} {e.confidence:<12.3f} "
                    f"{matched_str:<15}"
                )
                algo_name = ""  # Don't repeat algorithm name
            else:
                print(f"{algo_name:<15} {elem:<8} {'-':<10} {'-':<8} {'-':<12} {'-':<15}")
                algo_name = ""

    print()


# ============================================================================
# Plotting
# ============================================================================


def plot_spectrum_with_lines(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    results: Dict[str, Optional[ElementIdentificationResult]],
    title: str,
    output_path: Path,
):
    """
    Plot spectrum with vertical lines at matched peak wavelengths.

    One subplot per algorithm (3 rows), color-coded by element.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Color map for elements
    element_colors = {
        "Fe": "red",
        "Ni": "blue",
        "Cr": "green",
        "Ti": "purple",
        "Mn": "orange",
        "Cu": "cyan",
        "Si": "magenta",
        "Al": "gold",
        "V": "brown",
        "Mg": "lime",
        "Co": "navy",
    }

    for idx, algo_name in enumerate(["ALIAS", "Comb", "Correlation"]):
        ax = axes[idx]
        result = results.get(algo_name)

        # Plot spectrum
        ax.plot(wavelength, intensity, "k-", linewidth=0.5, alpha=0.7, label="Spectrum")

        if result is not None:
            # Plot matched lines for detected elements
            for elem_id in result.detected_elements:
                color = element_colors.get(elem_id.element, "gray")
                for line in elem_id.matched_lines:
                    ax.axvline(
                        line.wavelength_exp_nm,
                        color=color,
                        alpha=0.6,
                        linewidth=1.5,
                        label=elem_id.element if line == elem_id.matched_lines[0] else "",
                    )

        ax.set_ylabel("Intensity (a.u.)", fontsize=10)
        ax.set_title(f"{algo_name} Algorithm", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Legend (unique labels only)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Wavelength (nm)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_algorithm_comparison(
    results: Dict[str, Optional[ElementIdentificationResult]],
    dataset_name: str,
    expected: Optional[List[str]],
    output_path: Path,
):
    """
    Grouped bar chart: score per element per algorithm.
    """
    # Collect all elements
    all_elements = set()
    for result in results.values():
        if result is not None:
            all_elements.update([e.element for e in result.all_elements])

    sorted_elements = sorted(all_elements)
    n_elements = len(sorted_elements)
    n_algorithms = 3

    # Extract scores
    scores = np.zeros((n_algorithms, n_elements))
    algo_names = ["ALIAS", "Comb", "Correlation"]

    for algo_idx, algo_name in enumerate(algo_names):
        result = results.get(algo_name)
        if result is not None:
            elem_dict = {e.element: e for e in result.all_elements}
            for elem_idx, elem in enumerate(sorted_elements):
                if elem in elem_dict:
                    scores[algo_idx, elem_idx] = elem_dict[elem].score

    # Plot grouped bars
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_elements)
    width = 0.25

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for algo_idx, algo_name in enumerate(algo_names):
        offset = (algo_idx - 1) * width
        ax.bar(
            x + offset,
            scores[algo_idx, :],
            width,
            label=algo_name,
            color=colors[algo_idx],
            alpha=0.8,
        )

    ax.set_xlabel("Element", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    if expected:
        subtitle = f"Expected: {', '.join(expected)}"
    else:
        subtitle = "(Blind Test)"
    ax.set_title(
        f"Algorithm Comparison: {dataset_name}\n{subtitle}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_elements)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_spatial_map(
    wavelength: np.ndarray,
    intensities_2d: np.ndarray,
    db: AtomicDatabase,
    elements: List[str],
    output_path: Path,
):
    """
    For multi-position data: run ALIAS on each position, show element score heatmap.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm, shape (N,)
    intensities_2d : np.ndarray
        3D intensity array, shape (X, Y, N)
    db : AtomicDatabase
        Atomic database instance
    elements : List[str]
        Elements to search for
    output_path : Path
        Output file path
    """
    if intensities_2d.ndim != 3:
        print("  SKIP: Spatial map requires 3D data")
        return

    X, Y, N = intensities_2d.shape
    n_elements = len(elements)

    # Initialize score maps
    score_maps = {elem: np.zeros((X, Y)) for elem in elements}

    # Run ALIAS on each position
    identifier = ALIASIdentifier(db, elements=elements, resolving_power=5000.0)

    print(f"  Running ALIAS on {X}x{Y} = {X*Y} positions...")
    for i in range(X):
        for j in range(Y):
            spectrum = intensities_2d[i, j, :]
            try:
                result = identifier.identify(wavelength, spectrum)
                elem_dict = {e.element: e for e in result.all_elements}
                for elem in elements:
                    if elem in elem_dict:
                        score_maps[elem][i, j] = elem_dict[elem].score
            except Exception as e:
                print(f"    WARNING: Position ({i},{j}) failed: {e}")

    # Plot heatmaps
    fig, axes = plt.subplots(2, (n_elements + 1) // 2, figsize=(12, 8))
    axes = np.atleast_1d(axes).flatten()

    for idx, elem in enumerate(elements):
        ax = axes[idx]
        im = ax.imshow(score_maps[elem], cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{elem} Score", fontsize=11, fontweight="bold")
        ax.set_xlabel("Y Position", fontsize=9)
        ax.set_ylabel("X Position", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide extra subplots
    for idx in range(n_elements, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Spatial Element Score Map (ALIAS)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved spatial map to {output_path}")


# ============================================================================
# Robustness Reporting
# ============================================================================


def report_depth_scan_robustness(
    path: str,
    db: AtomicDatabase,
    elements: List[str],
    dataset_name: str,
    resolving_power: float = 5000.0,
) -> None:
    """
    Report per-iteration score variance for depth-scan datasets.

    Re-loads the raw 3D data, runs ALIAS on each iteration at the center
    position, and reports mean/std/CV% of scores per element.

    Parameters
    ----------
    path : str
        Path to scipp HDF5 file (3D: position × wavelength × iteration).
    db : AtomicDatabase
        Atomic database for ALIAS.
    elements : List[str]
        Element symbols to report scores for.
    dataset_name : str
        Display name for the dataset.
    resolving_power : float, optional
        Resolving power for ALIAS (default: 5000.0).

    Returns
    -------
    None
    """
    print(f"\n  --- Depth-Scan Robustness: {dataset_name} ---")

    with h5py.File(path, "r") as f:
        coords = _discover_scipp_coords(f)
        wavelength = coords["Wavelength"]
        data_raw = f["data/values"][:]  # (N_pos, 2560, N_iter)

    n_pos, n_wl, n_iter = data_raw.shape
    center = n_pos // 2
    print(f"  Positions: {n_pos}, Wavelengths: {n_wl}, Iterations: {n_iter}")
    print(f"  Using center position: {center}")

    identifier = ALIASIdentifier(db, elements=elements, resolving_power=resolving_power)

    # Collect scores per element per iteration
    scores: Dict[str, List[float]] = {elem: [] for elem in elements}

    for i in range(n_iter):
        spectrum = data_raw[center, :, i]
        try:
            result = identifier.identify(wavelength, spectrum)
            elem_dict = {e.element: e for e in result.all_elements}
            for elem in elements:
                if elem in elem_dict:
                    scores[elem].append(elem_dict[elem].score)
                else:
                    scores[elem].append(0.0)
        except Exception as e:
            print(f"    WARNING: Iteration {i} failed: {e}")
            for elem in elements:
                scores[elem].append(0.0)

    # Report statistics
    print(f"\n  {'Element':<8} {'Mean':<8} {'Std':<8} {'CV%':<8} {'Min':<8} {'Max':<8}")
    print(f"  {'-'*48}")
    for elem in sorted(elements):
        vals = np.array(scores[elem])
        mean = vals.mean()
        std = vals.std()
        cv = (std / mean * 100) if mean > 0 else 0.0
        print(
            f"  {elem:<8} {mean:<8.3f} {std:<8.3f} {cv:<8.1f} "
            f"{vals.min():<8.3f} {vals.max():<8.3f}"
        )
    print()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Legacy/reference validation of element identification on real LIBS data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Directory with data files (default: data/)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="ASD_da/libs_production.db",
        help="Atomic database path (default: ASD_da/libs_production.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/validation",
        help="Output directory for plots (default: output/validation)",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        help="Override elements to search for (default: use dataset defaults)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Only run specific datasets by name (default: all)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (default: False)",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="Skip spatial maps (slow) (default: False)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode with pass/fail criteria (default: False)",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Report depth-scan robustness (per-iteration score variance)",
    )
    parser.add_argument(
        "--wavelength-calibration-mode",
        type=str,
        default="none",
        choices=["none", "auto", "shift", "affine", "quadratic"],
        help="Wavelength calibration model before ID (default: none)",
    )
    parser.add_argument(
        "--wavelength-calibration-max-pair-window",
        type=float,
        default=2.0,
        help="Peak-to-line candidate window in nm for robust calibration fit",
    )
    parser.add_argument(
        "--wavelength-calibration-inlier-tol",
        type=float,
        default=0.08,
        help="Inlier tolerance in nm for robust calibration fit",
    )
    parser.add_argument(
        "--wavelength-calibration-gate-disable",
        action="store_true",
        help="Disable calibration quality gate (default: False)",
    )
    parser.add_argument(
        "--wavelength-calibration-gate-min-inliers",
        type=int,
        default=12,
        help="Quality gate: minimum inlier pairs (default: 12)",
    )
    parser.add_argument(
        "--wavelength-calibration-gate-min-peak-match",
        type=float,
        default=0.35,
        help="Quality gate: minimum matched peak fraction (default: 0.35)",
    )
    parser.add_argument(
        "--wavelength-calibration-gate-max-rmse",
        type=float,
        default=0.10,
        help="Quality gate: maximum inlier RMSE in nm (default: 0.10)",
    )
    parser.add_argument(
        "--wavelength-calibration-gate-min-span-frac",
        type=float,
        default=0.25,
        help="Quality gate: minimum inlier span fraction (default: 0.25)",
    )
    parser.add_argument(
        "--wavelength-calibration-gate-max-abs-correction",
        type=float,
        default=2.5,
        help="Quality gate: maximum absolute correction in nm (default: 2.5)",
    )

    args = parser.parse_args()

    print(
        "NOTE: This is a legacy/reference validation script. "
        "Use the unified benchmark runner for current benchmark reporting."
    )

    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Auto-extract zip archives if target dirs don't exist (safe: no Zip Slip)
    for archive_name in ["10000ppm.zip", "210000ppm.zip"]:
        archive_path = data_dir / archive_name
        target_dir = data_dir / archive_name.replace(".zip", "")
        if archive_path.exists() and not target_dir.exists():
            print(f"Extracting {archive_name}...")
            data_root = data_dir.resolve()
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.infolist():
                    # Reject path traversal (Zip Slip)
                    target = (data_dir / member.filename).resolve()
                    try:
                        target.relative_to(data_root)
                    except ValueError:
                        raise ValueError(f"Unsafe zip path: {member.filename}")
                    zf.extract(member, data_dir)
            print(f"  Extracted to {target_dir}")

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load database
    print(f"Loading atomic database: {db_path}")
    db = AtomicDatabase(str(db_path))

    # Filter datasets
    datasets_to_run = DATASETS
    if args.datasets:
        datasets_to_run = [d for d in DATASETS if d["name"] in args.datasets]
        if not datasets_to_run:
            print(f"ERROR: No datasets matched: {args.datasets}")
            sys.exit(1)

    # Summary tracking
    summary = {
        algo: {"expected_detected": 0, "total_expected": 0}
        for algo in ["ALIAS", "Comb", "Correlation"]
    }
    # Per-dataset detection results for benchmark mode
    # Structure: {algo: {(dataset_name, element): bool_detected}}
    detection_results: Dict[str, Dict[Tuple[str, str], bool]] = {
        algo: {} for algo in ["ALIAS", "Comb", "Correlation"]
    }

    # Process each dataset
    for dataset in datasets_to_run:
        print(f"\n{'='*100}")
        print(f"Processing: {dataset['name']}")
        print(f"{'='*100}")

        # Load data
        data_path = data_dir / dataset["path"]
        if not data_path.exists():
            print(f"  WARNING: File not found, skipping: {data_path}")
            continue

        loader_name = dataset["loader"]
        if loader_name == "netcdf":
            wavelength, data, metadata = load_netcdf(str(data_path))
        elif loader_name == "hdf5":
            wavelength, data, metadata = load_hdf5(str(data_path))
        elif loader_name == "hdf5_multishot":
            wavelength, data, metadata = load_hdf5_multishot(str(data_path))
        elif loader_name == "scipp":
            wavelength, data, metadata = load_scipp(str(data_path))
        elif loader_name == "scipp_depth_scan":
            wavelength, data, metadata = load_scipp_depth_scan(str(data_path))
        else:
            print(f"  ERROR: Unknown loader: {loader_name}")
            continue

        print(f"  Loaded: {metadata['format']}, shape={metadata['shape']}")
        wl_min, wl_max = metadata["wavelength_range"]
        print(f"  Wavelength range: {wl_min:.2f}-{wl_max:.2f} nm")

        # Select representative spectrum
        spectrum = select_representative_spectrum(data, dataset["name"])
        print(f"  Selected spectrum shape: {spectrum.shape}")

        # Determine elements to search
        elements = args.elements if args.elements else dataset["elements"]
        expected = dataset["expected"]

        # Robust wavelength calibration before identification
        wavelength_for_id = wavelength
        if args.wavelength_calibration_mode != "none":
            calibration = calibrate_wavelength_axis(
                wavelength=wavelength,
                intensity=spectrum,
                atomic_db=db,
                elements=elements,
                mode=args.wavelength_calibration_mode,  # type: ignore[arg-type]
                max_pair_window_nm=args.wavelength_calibration_max_pair_window,
                inlier_tolerance_nm=args.wavelength_calibration_inlier_tol,
                apply_quality_gate=not args.wavelength_calibration_gate_disable,
                quality_min_inliers=args.wavelength_calibration_gate_min_inliers,
                quality_min_peak_match_fraction=args.wavelength_calibration_gate_min_peak_match,
                quality_max_rmse_nm=args.wavelength_calibration_gate_max_rmse,
                quality_min_inlier_span_fraction=args.wavelength_calibration_gate_min_span_frac,
                quality_max_abs_correction_nm=(args.wavelength_calibration_gate_max_abs_correction),
            )
            if calibration.success and calibration.quality_passed:
                wavelength_for_id = calibration.corrected_wavelength
                coeffs = ", ".join(f"{c:.6g}" for c in calibration.coefficients)
                print(
                    "  Wavelength calibration:"
                    f" model={calibration.model}, coeffs=[{coeffs}],"
                    f" rmse={calibration.rmse_nm:.4f} nm,"
                    f" inliers={calibration.n_inliers}/{calibration.n_candidates},"
                    f" peak_match={calibration.matched_peak_fraction:.1%}"
                )
            elif calibration.success and not calibration.quality_passed:
                coeffs = ", ".join(f"{c:.6g}" for c in calibration.coefficients)
                print(
                    "  Wavelength calibration: rejected by quality gate"
                    f" ({calibration.quality_reason});"
                    f" model={calibration.model}, coeffs=[{coeffs}],"
                    f" rmse={calibration.rmse_nm:.4f} nm,"
                    f" inliers={calibration.n_inliers}/{calibration.n_candidates},"
                    f" peak_match={calibration.matched_peak_fraction:.1%}"
                )
            else:
                reason = calibration.quality_reason or calibration.details.get("reason", "unknown")
                print(f"  Wavelength calibration: skipped ({reason})")

        # Determine resolving power: per-dataset override > auto-detect
        if "resolving_power" in dataset:
            rp = dataset["resolving_power"]
            print(f"  Resolving power: {rp:.0f} (dataset config)")
        else:
            rp = estimate_resolving_power(wavelength_for_id, spectrum)
            print(f"  Resolving power: {rp:.0f} (auto-detected)")

        # Run identifiers
        print(f"  Running identifiers for elements: {', '.join(elements)}")
        results = run_all_identifiers(wavelength_for_id, spectrum, db, elements, resolving_power=rp)

        # Print results table
        print_result_table(results, dataset["name"], expected)

        # Update summary and per-dataset detection results
        is_blind = dataset.get("expected") is None
        for algo_name in ["ALIAS", "Comb", "Correlation"]:
            result = results.get(algo_name)
            if result is not None:
                detected_elems = {e.element for e in result.detected_elements}
                all_searched = {e.element for e in result.all_elements}
                if not is_blind:
                    for exp_elem in expected:
                        if exp_elem in detected_elems:
                            summary[algo_name]["expected_detected"] += 1
                    summary[algo_name]["total_expected"] += len(expected)
                # Record per-(dataset, element) detection for benchmark
                for elem in all_searched:
                    detection_results[algo_name][(dataset["name"], elem)] = elem in detected_elems

        # Generate plots
        if not args.no_plots:
            print("  Generating plots...")

            # Spectrum with lines
            spectrum_plot_path = output_dir / f"{dataset['name']}_spectrum.png"
            plot_spectrum_with_lines(
                wavelength_for_id,
                spectrum,
                results,
                f"{dataset['name']} - Element Identification",
                spectrum_plot_path,
            )
            print(f"    Saved: {spectrum_plot_path}")

            # Algorithm comparison
            comparison_plot_path = output_dir / f"{dataset['name']}_comparison.png"
            plot_algorithm_comparison(
                results,
                dataset["name"],
                expected,
                comparison_plot_path,
            )
            print(f"    Saved: {comparison_plot_path}")

            # Spatial map (if multi-position)
            if not args.no_spatial and data.ndim == 3:
                spatial_plot_path = output_dir / f"{dataset['name']}_spatial.png"
                plot_spatial_map(wavelength, data, db, elements, spatial_plot_path)

        # Depth-scan robustness reporting
        if args.robustness and loader_name == "scipp_depth_scan":
            report_depth_scan_robustness(
                str(data_path),
                db,
                elements,
                dataset["name"],
                resolving_power=rp,
            )

    # Print overall summary
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")
    print(
        f"{'Algorithm':<15} {'Expected Detected':<20} {'Total Expected':<15} {'Success Rate':<15}"
    )
    print("-" * 100)

    for algo_name in ["ALIAS", "Comb", "Correlation"]:
        detected = summary[algo_name]["expected_detected"]
        total = summary[algo_name]["total_expected"]
        success_rate = detected / total if total > 0 else 0.0
        print(f"{algo_name:<15} {detected:<20} {total:<15} {success_rate:<15.2%}")

    # Blind-test discovery summary
    blind_datasets = [d for d in datasets_to_run if d.get("expected") is None]
    if blind_datasets:
        print(f"\n{'='*100}")
        print("BLIND TEST DISCOVERIES")
        print(f"{'='*100}")
        print(f"{'Dataset':<25} {'Algorithm':<15} {'Detected Elements':<50}")
        print("-" * 100)

        for ds in blind_datasets:
            ds_name = ds["name"]
            for algo_name in ["ALIAS", "Comb", "Correlation"]:
                key_matches = {
                    elem
                    for (dn, elem), det in detection_results[algo_name].items()
                    if dn == ds_name and det
                }
                if key_matches:
                    elems_str = ", ".join(sorted(key_matches))
                else:
                    elems_str = "(none)"
                print(f"{ds_name:<25} {algo_name:<15} {elems_str:<50}")
                ds_name = ""  # Don't repeat dataset name

        print()

    print(f"\nValidation complete. Results saved to {output_dir}")

    # Benchmark mode
    if args.benchmark:
        passed = run_benchmark(detection_results, datasets_to_run)
        sys.exit(0 if passed else 1)


def run_benchmark(
    detection_results: Dict[str, Dict[Tuple[str, str], bool]],
    datasets_run: List[Dict[str, Any]],
) -> bool:
    """
    Evaluate benchmark criteria against collected detection results.

    Parameters
    ----------
    detection_results : dict
        {algo: {(dataset_name, element): bool_detected}}
    datasets_run : list
        List of dataset dicts that were processed

    Returns
    -------
    bool
        True if all benchmark criteria pass
    """
    print(f"\n{'='*100}")
    print("BENCHMARK RESULTS")
    print(f"{'='*100}")

    # Build expected set from datasets that were actually run (skip blind tests)
    dataset_expected = {}
    for ds in datasets_run:
        if ds.get("expected") is not None:
            dataset_expected[ds["name"]] = set(ds["expected"])

    all_passed = True

    for algo_name in ["ALIAS", "Comb", "Correlation"]:
        print(f"\n--- {algo_name} ---")
        algo_results = detection_results.get(algo_name, {})

        # Classify TP/FP/FN/TN
        tp = fp = fn = tn = 0
        for (ds_name, elem), detected in algo_results.items():
            if ds_name not in dataset_expected:
                continue
            is_expected = elem in dataset_expected[ds_name]
            if is_expected and detected:
                tp += 1
            elif is_expected and not detected:
                fn += 1
            elif not is_expected and detected:
                fp += 1
            else:
                tn += 1

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}  (min: {BENCHMARK_CRITERIA['min_recall']:.2f})")
        print(f"  FPR:       {fpr:.3f}  (max: {BENCHMARK_CRITERIA['max_fpr']:.2f})")

        # Check thresholds
        recall_ok = recall >= BENCHMARK_CRITERIA["min_recall"]
        fpr_ok = fpr <= BENCHMARK_CRITERIA["max_fpr"]
        print(f"  Recall pass: {'YES' if recall_ok else 'FAIL'}")
        print(f"  FPR pass:    {'YES' if fpr_ok else 'FAIL'}")

        if not recall_ok or not fpr_ok:
            all_passed = False

        # Check required detections
        for ds_name, elem in BENCHMARK_CRITERIA["required_detections"]:
            key = (ds_name, elem)
            if key in algo_results:
                detected = algo_results[key]
                status = "OK" if detected else "FAIL"
                print(f"  Required detection ({ds_name}, {elem}): {status}")
                if not detected:
                    all_passed = False

        # Check required absences
        for ds_name, elem in BENCHMARK_CRITERIA["required_absences"]:
            key = (ds_name, elem)
            if key in algo_results:
                detected = algo_results[key]
                status = "OK" if not detected else "FAIL"
                print(f"  Required absence  ({ds_name}, {elem}): {status}")
                if detected:
                    all_passed = False

    print(f"\n{'='*100}")
    print(f"BENCHMARK {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'='*100}")

    return all_passed


if __name__ == "__main__":
    main()
