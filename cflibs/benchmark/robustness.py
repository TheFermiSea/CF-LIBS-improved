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

def line_dropout_perturbation(
    spectrum: BenchmarkSpectrum, top_n: int = 3
) -> BenchmarkSpectrum:
    """
    Remove the top_n highest intensity channels by setting them to 0.0.
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
    Inject N(0, sigma_mult * std(intensity)) noise into `fraction` of channels.
    """
    if rng is None:
        rng = np.random.default_rng()

    perturbed = copy.deepcopy(spectrum)
    n_channels = len(perturbed.intensity)
    n_inject = int(n_channels * fraction)

    if n_inject <= 0:
        return perturbed

    idx = rng.choice(n_channels, size=n_inject, replace=False)
    std_int = np.std(perturbed.intensity)
    noise = rng.normal(loc=0.0, scale=sigma_mult * std_int, size=n_inject)

    perturbed.intensity[idx] += noise
    return perturbed

@dataclass
class PerturbationReport:
    """Report of perturbation effects on a pipeline."""
    pipeline_name: str
    baseline_d_A: Dict[str, float] = field(default_factory=dict)
    perturbed_d_A: Dict[str, Dict[str, float]] = field(default_factory=dict)
    delta_d_A: Dict[str, Dict[str, float]] = field(default_factory=dict)

def run_perturbation_battery(
    pipeline_fn: Callable[[np.ndarray, np.ndarray, List[str]], Dict[str, Any]],
    spectra: Sequence[BenchmarkSpectrum],
    perturbations: Dict[str, Callable[[BenchmarkSpectrum], BenchmarkSpectrum]],
    pipeline_name: str = "pipeline",
) -> PerturbationReport:
    """
    Run pipeline on original and perturbed spectra, computing delta Aitchison distance.
    """
    report = PerturbationReport(pipeline_name=pipeline_name)

    for spec in spectra:
        spec_id = spec.spectrum_id
        true_conc = spec.true_composition
        elements = sorted(true_conc.keys())

        # Baseline
        d_a_base = float("inf")
        try:
            res_base = pipeline_fn(spec.wavelength_nm, spec.intensity, elements)
            pred_base = res_base.get("concentrations", {})
            if pred_base:
                d_a_base = aitchison_distance(true_conc, pred_base)
        except Exception as e:
            logger.warning(f"Baseline pipeline failed for {spec_id}: {e}")

        report.baseline_d_A[spec_id] = d_a_base

        for p_name, p_fn in perturbations.items():
            if p_name not in report.perturbed_d_A:
                report.perturbed_d_A[p_name] = {}
            if p_name not in report.delta_d_A:
                report.delta_d_A[p_name] = {}

            p_spec = p_fn(spec)

            d_a_p = float("inf")
            try:
                res_p = pipeline_fn(p_spec.wavelength_nm, p_spec.intensity, elements)
                pred_p = res_p.get("concentrations", {})
                if pred_p:
                    d_a_p = aitchison_distance(true_conc, pred_p)
            except Exception as e:
                logger.warning(f"Perturbed pipeline ({p_name}) failed for {spec_id}: {e}")

            report.perturbed_d_A[p_name][spec_id] = d_a_p
            if np.isinf(d_a_base) or np.isinf(d_a_p):
                report.delta_d_A[p_name][spec_id] = float("inf")
            else:
                report.delta_d_A[p_name][spec_id] = d_a_p - d_a_base

    return report
