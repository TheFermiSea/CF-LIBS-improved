"""
Robustness perturbation tests for LIBS benchmark pipelines.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from cflibs.benchmark.dataset import BenchmarkSpectrum
from cflibs.benchmark.composition_metrics import aitchison_distance
from cflibs.core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_OUTLIER_SEED = 0


def line_dropout_perturbation(spectrum: BenchmarkSpectrum, top_n: int = 3) -> BenchmarkSpectrum:
    """
    Create a copy with the ``top_n`` highest-intensity channels set to zero.

    Parameters
    ----------
    spectrum
        Input spectrum to copy and perturb.
    top_n
        Number of highest-intensity channels to zero. Non-positive values return
        an unmodified copy.

    Returns
    -------
    BenchmarkSpectrum
        Perturbed spectrum with the requested channels set to ``0.0``.
    """
    perturbed = copy.deepcopy(spectrum)
    if top_n <= 0 or len(perturbed.intensity) == 0:
        return perturbed

    # Find indices of the top_n highest intensities
    idx = np.argsort(perturbed.intensity)[-top_n:]
    perturbed.intensity[idx] = 0.0
    return perturbed


def outlier_injection_perturbation(
    spectrum: BenchmarkSpectrum,
    fraction: float = 0.05,
    sigma_mult: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> BenchmarkSpectrum:
    """
    Add Gaussian outlier noise to a fraction of a spectrum's intensity channels.

    Parameters
    ----------
    spectrum
        Source spectrum to copy and perturb.
    fraction
        Fraction of channels to perturb. Must be in ``[0.0, 1.0]``.
    sigma_mult
        Multiplier applied to the spectrum intensity scale.
    rng
        Random number generator used for channel selection and noise sampling.
        If omitted, a deterministic default seed is used.

    Returns
    -------
    BenchmarkSpectrum
        Deep-copied spectrum with non-negative floating-point intensities.

    Raises
    ------
    ValueError
        If ``fraction`` is outside ``[0.0, 1.0]`` or ``sigma_mult`` is negative.
    """
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be finite and in the range [0.0, 1.0]")
    if not np.isfinite(sigma_mult) or sigma_mult < 0.0:
        raise ValueError("sigma_mult must be finite and non-negative")

    if rng is None:
        rng = np.random.default_rng(DEFAULT_OUTLIER_SEED)

    perturbed = copy.deepcopy(spectrum)
    perturbed.intensity = np.asarray(perturbed.intensity, dtype=np.float64).copy()
    n_channels = len(perturbed.intensity)
    n_inject = int(n_channels * fraction)

    if n_inject <= 0:
        return perturbed

    idx = rng.choice(n_channels, size=n_inject, replace=False)
    std_int = float(np.std(perturbed.intensity))
    fallback_scale = float(np.mean(np.abs(perturbed.intensity)))
    base_scale = std_int if std_int > 0.0 else max(fallback_scale, 1.0)
    noise = rng.normal(loc=0.0, scale=sigma_mult * base_scale, size=n_inject)

    perturbed.intensity[idx] += noise
    perturbed.intensity = np.maximum(perturbed.intensity, 0.0)
    return perturbed


@dataclass
class PerturbationReport:
    """Report of perturbation effects on a pipeline."""

    pipeline_name: str
    baseline_d_a: Dict[str, float] = field(default_factory=dict)
    perturbed_d_a: Dict[str, Dict[str, float]] = field(default_factory=dict)
    delta_d_a: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _pipeline_aitchison_distance(
    pipeline_fn: Callable[[np.ndarray, np.ndarray, List[str]], Dict[str, Any]],
    spectrum: BenchmarkSpectrum,
    elements: List[str],
    perturbation_name: str,
) -> float:
    try:
        result = pipeline_fn(spectrum.wavelength_nm, spectrum.intensity, elements)
        prediction = result.get("concentrations", {})
        if prediction:
            return float(aitchison_distance(spectrum.true_composition, prediction))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "%s pipeline failed for %s: %s",
            perturbation_name,
            spectrum.spectrum_id,
            exc,
        )
    return float("inf")


def _delta_distance(baseline_distance: float, perturbed_distance: float) -> float:
    if not np.isfinite(baseline_distance) or not np.isfinite(perturbed_distance):
        return float("inf")
    return perturbed_distance - baseline_distance


def run_perturbation_battery(
    pipeline_fn: Callable[[np.ndarray, np.ndarray, List[str]], Dict[str, Any]],
    spectra: Sequence[BenchmarkSpectrum],
    perturbations: Dict[str, Callable[[BenchmarkSpectrum], BenchmarkSpectrum]],
    pipeline_name: str = "pipeline",
) -> PerturbationReport:
    """
    Run baseline and perturbed spectra through a composition pipeline.

    Pipeline failures or missing ``concentrations`` are recorded as infinite
    Aitchison distances. Deltas are also infinite when either side is non-finite.
    """
    report = PerturbationReport(pipeline_name=pipeline_name)

    for spec in spectra:
        spec_id = spec.spectrum_id
        elements = sorted(spec.true_composition.keys())
        baseline_distance = _pipeline_aitchison_distance(pipeline_fn, spec, elements, "Baseline")

        report.baseline_d_a[spec_id] = baseline_distance

        for perturbation_name, perturbation_fn in perturbations.items():
            perturbed_spec = perturbation_fn(spec)
            perturbed_distance = _pipeline_aitchison_distance(
                pipeline_fn,
                perturbed_spec,
                elements,
                f"Perturbed pipeline ({perturbation_name})",
            )

            report.perturbed_d_a.setdefault(perturbation_name, {})[spec_id] = perturbed_distance
            report.delta_d_a.setdefault(perturbation_name, {})[spec_id] = _delta_distance(
                baseline_distance,
                perturbed_distance,
            )

    return report
