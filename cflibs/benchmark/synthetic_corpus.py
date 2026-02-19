"""
Deterministic synthetic corpus builder for spectral identification debugging.

This module generates synthetic LIBS spectra with explicit ground-truth labels
and controlled perturbation axes:
- SNR
- Continuum baseline level
- Resolving power
- Global wavelength shift
- Non-linear wavelength warp
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
import json
import os

import numpy as np
from scipy import signal

# Keep JAX on CPU for stable execution in corpus-generation workflows.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from cflibs.benchmark.dataset import (
    BenchmarkDataset,
    BenchmarkSpectrum,
    InstrumentalConditions,
    MatrixType,
    SampleMetadata,
    SampleType,
)
from cflibs.benchmark.loaders import BenchmarkFormat, save_benchmark
from cflibs.benchmark.synthetic import STANDARD_MASSES
from cflibs.core.logging_config import get_logger

logger = get_logger("benchmark.synthetic_corpus")


@dataclass
class CorpusRecipe:
    """Ground-truth composition recipe in mass-fraction space."""

    name: str
    mass_fractions: Dict[str, float]
    matrix_type: MatrixType = MatrixType.METAL_ALLOY


@dataclass
class PerturbationAxes:
    """Controlled perturbation axes for synthetic corpus generation."""

    snr_db: List[float]
    continuum_level: List[float]
    resolving_power: List[float]
    shift_nm: List[float]
    warp_quadratic_nm: List[float]


def default_recipes(candidate_elements: Iterable[str]) -> List[CorpusRecipe]:
    """
    Return default pure + mixture recipes filtered to candidate elements.

    Parameters
    ----------
    candidate_elements : Iterable[str]
        Candidate element symbols to keep in recipes.

    Returns
    -------
    List[CorpusRecipe]
        Filtered and renormalized recipes.
    """
    candidates = set(candidate_elements)
    base = [
        CorpusRecipe("pure_Fe", {"Fe": 1.0}, matrix_type=MatrixType.METAL_PURE),
        CorpusRecipe("pure_Ni", {"Ni": 1.0}, matrix_type=MatrixType.METAL_PURE),
        CorpusRecipe("binary_Fe_Ni", {"Fe": 0.7, "Ni": 0.3}),
        CorpusRecipe("steel_like", {"Fe": 0.88, "Cr": 0.06, "Ni": 0.04, "Mn": 0.02}),
    ]
    out: List[CorpusRecipe] = []
    for recipe in base:
        filtered = {k: v for k, v in recipe.mass_fractions.items() if k in candidates}
        total = sum(filtered.values())
        if total <= 0:
            continue
        normalized = {k: v / total for k, v in filtered.items()}
        if len(normalized) == 1:
            matrix_type = MatrixType.METAL_PURE
        else:
            matrix_type = recipe.matrix_type
        out.append(
            CorpusRecipe(
                name=recipe.name,
                mass_fractions=normalized,
                matrix_type=matrix_type,
            )
        )
    return out


def default_axes() -> PerturbationAxes:
    """
    Return default perturbation ranges used for CF-LIBS-ak3.1.3.

    Returns
    -------
    PerturbationAxes
        Default perturbation grid.
    """
    return PerturbationAxes(
        snr_db=[20.0, 30.0, 40.0],
        continuum_level=[0.00, 0.03],
        resolving_power=[700.0, 1000.0],
        shift_nm=[-1.0, 0.0, 1.0],
        warp_quadratic_nm=[0.0, 0.15],
    )


def mass_to_number_fractions(mass_fractions: Dict[str, float]) -> Dict[str, float]:
    """
    Convert mass fractions into number fractions.

    n_i is proportional to w_i / A_i.

    Parameters
    ----------
    mass_fractions : Dict[str, float]
        Element mass fractions.

    Returns
    -------
    Dict[str, float]
        Element number fractions normalized to 1.
    """
    weighted = {}
    for element, mass_fraction in mass_fractions.items():
        if mass_fraction <= 0:
            continue
        if element not in STANDARD_MASSES:
            raise KeyError(
                f"Missing standard atomic mass for element '{element}' in synthetic corpus builder"
            )
        atomic_mass = STANDARD_MASSES[element]
        weighted[element] = float(mass_fraction) / max(float(atomic_mass), 1e-12)

    total = sum(weighted.values())
    if total <= 0:
        raise ValueError("Mass fractions must contain at least one positive component")

    return {element: value / total for element, value in weighted.items()}


def distort_wavelength_axis(
    wavelength_nm: np.ndarray,
    shift_nm: float,
    warp_quadratic_nm: float,
) -> np.ndarray:
    """
    Apply global shift + quadratic non-linear warp to wavelength axis.

    The quadratic warp is centered to keep average warp ~0 over the axis.

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Base wavelength axis in nm.
    shift_nm : float
        Global wavelength shift in nm.
    warp_quadratic_nm : float
        Quadratic warp amplitude in nm.

    Returns
    -------
    np.ndarray
        Distorted wavelength axis.
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    x = np.linspace(-1.0, 1.0, wl.size)
    centered_quad = x * x - float(np.mean(x * x))
    warped = wl + float(shift_nm) + float(warp_quadratic_nm) * centered_quad
    if not np.all(np.diff(warped) > 0):
        raise ValueError("Wavelength distortion produced non-monotonic axis")
    return warped


def full_factorial_perturbations(axes: PerturbationAxes) -> Iterator[Dict[str, float]]:
    """
    Yield full-factorial perturbation combinations in deterministic order.

    Parameters
    ----------
    axes : PerturbationAxes
        Perturbation axes definition.

    Yields
    ------
    Dict[str, float]
        One perturbation setting per combination.
    """
    for snr_db in axes.snr_db:
        for continuum in axes.continuum_level:
            for resolving_power in axes.resolving_power:
                for shift_nm in axes.shift_nm:
                    for warp_nm in axes.warp_quadratic_nm:
                        yield {
                            "snr_db": float(snr_db),
                            "continuum_level": float(continuum),
                            "resolving_power": float(resolving_power),
                            "shift_nm": float(shift_nm),
                            "warp_quadratic_nm": float(warp_nm),
                        }


def _apply_resolving_power(
    wavelength_nm: np.ndarray, intensity: np.ndarray, resolving_power: float
) -> np.ndarray:
    wl = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if wl.size < 3:
        return y
    if not np.all(np.diff(wl) > 0):
        raise ValueError("wavelength_nm must be strictly increasing")

    # Constant resolving power implies constant broadening width in log(lambda).
    log_wl = np.log(wl)
    log_grid = np.linspace(log_wl[0], log_wl[-1], wl.size)
    y_log = np.interp(log_grid, log_wl, y)
    sigma_log = 1.0 / (2.355 * max(float(resolving_power), 1e-9))
    step_log = float(np.mean(np.diff(log_grid)))

    # Skip tiny kernels where discrete convolution would be numerically pointless.
    if sigma_log < 0.15 * step_log:
        return y

    half_width_px = max(int(np.ceil(5.0 * sigma_log / max(step_log, 1e-12))), 1)
    kernel_axis = np.arange(-half_width_px, half_width_px + 1, dtype=float) * step_log
    kernel = np.exp(-0.5 * (kernel_axis / sigma_log) ** 2)
    kernel = kernel / np.sum(kernel)
    y_blurred_log = signal.convolve(y_log, kernel, mode="same")
    return np.interp(log_wl, log_grid, y_blurred_log)


def _add_continuum_and_noise(
    intensity: np.ndarray,
    continuum_level: float,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(intensity, dtype=float)
    y = y - float(np.min(y))
    signal_scale = max(float(np.percentile(y, 99)), 1e-9)

    x = np.linspace(0.0, 1.0, y.size)
    continuum_shape = 0.7 + 0.3 * x + 0.2 * (x - 0.5) ** 2
    y = y + float(continuum_level) * signal_scale * continuum_shape

    snr_linear = 10.0 ** (float(snr_db) / 20.0)
    noise_std = signal_scale / max(snr_linear, 1e-9)
    y = y + rng.normal(0.0, noise_std, size=y.size)

    return np.clip(y, 0.0, None)


def build_synthetic_id_corpus(
    db_path: str,
    output_dir: str,
    dataset_name: str,
    seed: int,
    candidate_elements: List[str],
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    pixels: int,
    temperature_range_eV: List[float],
    log_ne_range: List[float],
    recipes: Optional[List[CorpusRecipe]] = None,
    axes: Optional[PerturbationAxes] = None,
) -> Dict[str, Any]:
    """
    Build deterministic synthetic corpus and persist benchmark + manifests.

    Parameters
    ----------
    db_path : str
        Path to atomic database.
    output_dir : str
        Output directory.
    dataset_name : str
        Dataset name for output folder.
    seed : int
        RNG seed for deterministic generation.
    candidate_elements : List[str]
        Candidate elements to include in synthetic spectra.
    wavelength_min_nm : float
        Lower wavelength bound.
    wavelength_max_nm : float
        Upper wavelength bound.
    pixels : int
        Number of spectral channels.
    temperature_range_eV : List[float]
        Two-value temperature range [min, max] in eV.
    log_ne_range : List[float]
        Two-value log10 electron density range [min, max].
    recipes : List[CorpusRecipe], optional
        Optional custom recipe set.
    axes : PerturbationAxes, optional
        Optional custom perturbation axes.

    Returns
    -------
    Dict[str, Any]
        Summary dictionary with generated paths and counts.
    """
    # Import here to keep module lightweight for helper-only unit tests.
    from cflibs.inversion.bayesian import BayesianForwardModel

    rng = np.random.default_rng(seed)
    if recipes is None:
        recipes = default_recipes(candidate_elements)
    if axes is None:
        axes = default_axes()
    if not recipes:
        raise ValueError("No valid recipes after filtering candidate elements")

    model = BayesianForwardModel(
        db_path=db_path,
        elements=candidate_elements,
        wavelength_range=(float(wavelength_min_nm), float(wavelength_max_nm)),
        pixels=int(pixels),
        instrument_fwhm_nm=0.01,  # Keep model baseline narrow; corpus sets RP explicitly.
    )
    wavelength_true = np.asarray(model.wavelength, dtype=float)

    spectra: List[BenchmarkSpectrum] = []
    manifest_rows: List[Dict[str, Any]] = []

    perturbations = list(full_factorial_perturbations(axes))
    for recipe in recipes:
        number_fractions = mass_to_number_fractions(recipe.mass_fractions)
        concentration_vector = np.array(
            [number_fractions.get(el, 0.0) for el in candidate_elements], dtype=float
        )

        for scenario_idx, perturb in enumerate(perturbations):
            sample_id = f"{recipe.name}_{scenario_idx:04d}"
            T_eV = float(
                rng.uniform(float(temperature_range_eV[0]), float(temperature_range_eV[1]))
            )
            log_ne = float(rng.uniform(float(log_ne_range[0]), float(log_ne_range[1])))

            clean = np.asarray(model.forward_numpy(T_eV, log_ne, concentration_vector), dtype=float)
            clean = _apply_resolving_power(
                wavelength_nm=wavelength_true,
                intensity=clean,
                resolving_power=perturb["resolving_power"],
            )
            measured = _add_continuum_and_noise(
                intensity=clean,
                continuum_level=perturb["continuum_level"],
                snr_db=perturb["snr_db"],
                rng=rng,
            )
            wavelength_measured = distort_wavelength_axis(
                wavelength_nm=wavelength_true,
                shift_nm=perturb["shift_nm"],
                warp_quadratic_nm=perturb["warp_quadratic_nm"],
            )

            spectral_resolution_nm = float(np.mean(wavelength_true) / perturb["resolving_power"])
            conditions = InstrumentalConditions(
                laser_wavelength_nm=1064.0,
                laser_energy_mj=45.0,
                gate_delay_us=1.0,
                gate_width_us=10.0,
                spectral_range_nm=(float(wavelength_measured[0]), float(wavelength_measured[-1])),
                spectral_resolution_nm=spectral_resolution_nm,
                spectrometer_type="Synthetic",
                detector_type="Synthetic",
                atmosphere="synthetic",
            )
            metadata = SampleMetadata(
                sample_id=sample_id,
                sample_type=SampleType.SIMULATED,
                matrix_type=recipe.matrix_type,
                provenance=(
                    "Generated by build_synthetic_id_corpus with deterministic seed "
                    f"{seed}; recipe={recipe.name}; scenario={scenario_idx}"
                ),
            )
            spectrum = BenchmarkSpectrum(
                spectrum_id=sample_id,
                wavelength_nm=wavelength_measured,
                intensity=measured,
                true_composition=recipe.mass_fractions,
                composition_uncertainty={
                    k: max(0.02 * float(v), 1e-4) for k, v in recipe.mass_fractions.items()
                },
                conditions=conditions,
                metadata=metadata,
                plasma_temperature_K=float(T_eV * 11604.518),
                electron_density_cm3=float(10.0**log_ne),
                quality_flag=0,
            )
            spectra.append(spectrum)

            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "recipe": recipe.name,
                    "present_elements": sorted(
                        [e for e, v in recipe.mass_fractions.items() if v > 0]
                    ),
                    "mass_fractions": recipe.mass_fractions,
                    "number_fractions": number_fractions,
                    "temperature_eV": T_eV,
                    "log10_ne_cm3": log_ne,
                    "perturbation": perturb,
                }
            )

    dataset = BenchmarkDataset(
        name=dataset_name,
        version="ak3.1.3",
        spectra=spectra,
        elements=sorted(set(candidate_elements)),
        description=(
            "Deterministic synthetic corpus for spectral-ID debugging with controlled perturbation "
            "axes (SNR, continuum, RP, shift, quadratic warp)."
        ),
        citation="Internal CF-LIBS synthetic benchmark (CF-LIBS-ak3.1.3)",
        contributors=["CF-LIBS Codex workflow"],
    )
    dataset.create_random_split(
        name="default",
        train_fraction=0.8,
        test_fraction=0.2,
        random_seed=seed,
    )

    out_root = Path(output_dir).resolve() / dataset_name
    out_root.mkdir(parents=True, exist_ok=True)

    json_path = out_root / "corpus.json"
    h5_path = out_root / "corpus.h5"
    manifest_json = out_root / "manifest.json"
    manifest_jsonl = out_root / "manifest.jsonl"

    save_benchmark(dataset, json_path, format=BenchmarkFormat.JSON)
    save_benchmark(dataset, h5_path, format=BenchmarkFormat.HDF5)
    manifest_json.write_text(json.dumps(manifest_rows, indent=2))
    with manifest_jsonl.open("w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "dataset_name": dataset_name,
        "seed": int(seed),
        "n_recipes": len(recipes),
        "n_perturbation_combinations": len(perturbations),
        "n_spectra": len(spectra),
        "wavelength_range_nm": [float(wavelength_true[0]), float(wavelength_true[-1])],
        "candidate_elements": candidate_elements,
        "output_files": {
            "dataset_json": str(json_path),
            "dataset_hdf5": str(h5_path),
            "manifest_json": str(manifest_json),
            "manifest_jsonl": str(manifest_jsonl),
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
