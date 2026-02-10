#!/usr/bin/env python
"""
Validation script for CF-LIBS element identification on real experimental data.

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


def select_representative_spectrum(
    data: np.ndarray, dataset_name: str
) -> np.ndarray:
    """
    Select a representative 1D spectrum from multi-dimensional data.

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

    if data.ndim == 3:
        # Grid data (steel, Fe, Ni): use center pixel
        # Note: 3×3 averaging was removed because the Fe/Ni grids ARE 3×3,
        # so averaging over the whole grid dilutes element-specific signal
        if dataset_name in ["steel_245nm", "Fe_245nm", "Ni_245nm"]:
            x_center = data.shape[0] // 2
            y_center = data.shape[1] // 2
            return data[x_center, y_center, :]

        # Line scan (FeNi_380nm, FeNi_480nm, 20shot): use first position, squeeze Y
        else:
            return data[0, 0, :]

    raise ValueError(f"Unexpected data shape: {data.shape}")


# ============================================================================
# Element Identification
# ============================================================================


def run_all_identifiers(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    db: AtomicDatabase,
    elements: List[str],
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

    Returns
    -------
    results : dict
        {algorithm_name: ElementIdentificationResult or None}
    """
    results = {}

    algorithms = [
        ("ALIAS", ALIASIdentifier, {"resolving_power": 5000.0}),
        ("Comb", CombIdentifier, {}),
        ("Correlation", CorrelationIdentifier, {}),
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
    expected: List[str],
):
    """
    Print formatted table of results.

    Columns: Algorithm | Element | Detected | Score | Confidence | Matched Lines

    Expected elements marked with * when detected correctly.
    """
    print(f"\n{'='*100}")
    print(f"Dataset: {dataset_name}")
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
                detected_str = "YES*" if (e.detected and elem in expected) else (
                    "YES" if e.detected else "NO"
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
    expected: List[str],
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
    ax.set_title(
        f"Algorithm Comparison: {dataset_name}\nExpected: {', '.join(expected)}",
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
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Validate element identification on real LIBS data"
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

    args = parser.parse_args()

    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

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
    summary = {algo: {"expected_detected": 0, "total_expected": 0} for algo in ["ALIAS", "Comb", "Correlation"]}
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

        # Run identifiers
        print(f"  Running identifiers for elements: {', '.join(elements)}")
        results = run_all_identifiers(wavelength, spectrum, db, elements)

        # Print results table
        print_result_table(results, dataset["name"], expected)

        # Update summary and per-dataset detection results
        for algo_name in ["ALIAS", "Comb", "Correlation"]:
            result = results.get(algo_name)
            if result is not None:
                detected_elems = {e.element for e in result.detected_elements}
                all_searched = {e.element for e in result.all_elements}
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
                wavelength,
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

    # Print overall summary
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")
    print(f"{'Algorithm':<15} {'Expected Detected':<20} {'Total Expected':<15} {'Success Rate':<15}")
    print("-" * 100)

    for algo_name in ["ALIAS", "Comb", "Correlation"]:
        detected = summary[algo_name]["expected_detected"]
        total = summary[algo_name]["total_expected"]
        success_rate = detected / total if total > 0 else 0.0
        print(f"{algo_name:<15} {detected:<20} {total:<15} {success_rate:<15.2%}")

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

    # Build expected set from datasets that were actually run
    dataset_expected = {}
    for ds in datasets_run:
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
