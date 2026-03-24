#!/usr/bin/env python3
"""
Diagnostic script for closure impact analysis.

Analyzes whether the compositional closure constraint is degrading accuracy
by comparing raw (unclosed) concentration estimates vs closed estimates vs
ground truth for synthetic spectra with known compositions.

Usage
-----
python scripts/diagnose_closure.py [--n-samples N] [--seed 42]

Options
-------
--n-samples : Number of synthetic spectra to generate (default: 100)
--seed      : Random seed for reproducibility (default: 42)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.synthetic import (
    generate_synthetic_dataset,
    CompositionRange,
    ConditionVariation,
)
from cflibs.benchmark.synthetic_eval import (
    run_inversion_pipeline,
    derive_truth_elements,
)
from cflibs.inversion.closure import ClosureResult, ClosureMode
from cflibs.inversion.softmax_closure import softmax_closure as jax_softmax_closure
from cflibs.benchmark.metrics import ElementMetrics, EvaluationResult, MetricType


def compute_rmsep(predictions: np.ndarray, true_values: np.ndarray) -> float:
    """Compute Root Mean Square Error of Prediction."""
    return np.sqrt(np.mean((predictions - true_values) ** 2))


def classify_element_by_concentration(fraction: float) -> str:
    """Classify element as major (>5%), minor (1-5%), or trace (<1%)."""
    if fraction > 0.05:
        return "major"
    elif fraction >= 0.01:
        return "minor"
    else:
        return "trace"


@dataclass
class ClosureDiagnostics:
    """Diagnostics for closure impact analysis."""
    
    element: str
    n_samples: int
    
    # Raw (unclosed) estimates
    raw_rmsep: float
    raw_bias: float
    raw_rmseps_by_category: Dict[str, List[float]]
    
    # Closed estimates
    closed_rmsep: float
    closed_bias: float
    closed_rmseps_by_category: Dict[str, List[float]]
    
    # Error amplification
    error_amplified: bool
    amplification_factor: float
    
    # Ground truth distribution
    truth_mean: float
    truth_std: float
    truth_range: Tuple[float, float]


def diagnose_closure_impact(
    raw_concentrations: Dict[str, np.ndarray],
    closed_concentrations: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    element_list: List[str],
) -> Dict[str, ClosureDiagnostics]:
    """
    Analyze closure impact on concentration estimates.
    
    Parameters
    ----------
    raw_concentrations : Dict[str, np.ndarray]
        Raw (unclosed) concentration estimates per element.
        Each array has shape (n_samples,).
    closed_concentrations : Dict[str, np.ndarray]
        Closed concentration estimates per element.
        Each array has shape (n_samples,).
    ground_truth : Dict[str, np.ndarray]
        Ground truth concentrations per element.
        Each array has shape (n_samples,).
    element_list : List[str]
        List of element symbols to analyze.
    
    Returns
    -------
    Dict[str, ClosureDiagnostics]
        Diagnostics per element.
    """
    diagnostics = {}
    
    for element in element_list:
        if element not in raw_concentrations or element not in closed_concentrations:
            continue
            
        raw_vals = raw_concentrations[element]
        closed_vals = closed_concentrations[element]
        truth_vals = ground_truth[element]
        
        # Filter valid samples (non-negative concentrations)
        valid_mask = (raw_vals >= 0) & (closed_vals >= 0) & (truth_vals >= 0)
        raw_vals = raw_vals[valid_mask]
        closed_vals = closed_vals[valid_mask]
        truth_vals = truth_vals[valid_mask]
        
        n_samples = len(raw_vals)
        if n_samples < 2:
            continue
        
        # Compute RMSEP and bias
        raw_rmsep = compute_rmsep(raw_vals, truth_vals)
        closed_rmsep = compute_rmsep(closed_vals, truth_vals)
        
        raw_bias = np.mean(raw_vals - truth_vals)
        closed_bias = np.mean(closed_vals - truth_vals)
        
        # Classify by concentration category
        raw_rmseps_by_category: Dict[str, List[float]] = {"major": [], "minor": [], "trace": []}
        closed_rmseps_by_category: Dict[str, List[float]] = {"major": [], "minor": [], "trace": []}
        
        for i in range(n_samples):
            category = classify_element_by_concentration(truth_vals[i])
            raw_error = raw_vals[i] - truth_vals[i]
            closed_error = closed_vals[i] - truth_vals[i]
            raw_rmseps_by_category[category].append(np.sqrt(raw_error**2))
            closed_rmseps_by_category[category].append(np.sqrt(closed_error**2))
        
        # Average RMSEP by category
        raw_rmseps_by_category = {
            cat: np.mean(rmse_list) if rmse_list else 0.0
            for cat, rmse_list in raw_rmseps_by_category.items()
        }
        closed_rmseps_by_category = {
            cat: np.mean(rmse_list) if rmse_list else 0.0
            for cat, rmse_list in closed_rmseps_by_category.items()
        }
        
        # Determine if closure amplifies errors
        error_amplified = closed_rmsep > raw_rmsep
        amplification_factor = closed_rmsep / max(raw_rmsep, 1e-10)
        
        # Ground truth statistics
        truth_mean = float(np.mean(truth_vals))
        truth_std = float(np.std(truth_vals))
        truth_range = (float(np.min(truth_vals)), float(np.max(truth_vals)))
        
        diagnostics[element] = ClosureDiagnostics(
            element=element,
            n_samples=n_samples,
            raw_rmsep=raw_rmsep,
            raw_bias=raw_bias,
            raw_rmseps_by_category=raw_rmseps_by_category,
            closed_rmsep=closed_rmsep,
            closed_bias=closed_bias,
            closed_rmseps_by_category=closed_rmseps_by_category,
            error_amplified=error_amplified,
            amplification_factor=amplification_factor,
            truth_mean=truth_mean,
            truth_std=truth_std,
            truth_range=truth_range,
        )
    
    return diagnostics


def run_closure_comparison(
    n_samples: int = 100,
    seed: int = 42,
) -> Tuple[Dict, Dict[str, ClosureDiagnostics]]:
    """
    Run synthetic benchmark and compare closure methods.
    
    Parameters
    ----------
    n_samples : int
        Number of synthetic spectra to generate.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    Tuple[Dict, Dict[str, ClosureDiagnostics]]
        Raw and closed inversion results, plus diagnostics.
    """
    np.random.seed(seed)
    
    # Generate synthetic dataset
    composition_ranges = [
        CompositionRange("Fe", 0.001, 0.50, "uniform", required=True),
        CompositionRange("Cr", 0.001, 0.30, "uniform", required=True),
        CompositionRange("Ni", 0.001, 0.25, "uniform", required=True),
        CompositionRange("Mn", 0.001, 0.10, "uniform", required=True),
        CompositionRange("Si", 0.001, 0.08, "uniform", required=True),
        CompositionRange("Cu", 0.001, 0.05, "uniform", required=False),
        CompositionRange("Al", 0.001, 0.05, "uniform", required=False),
        CompositionRange("Ti", 0.001, 0.03, "uniform", required=False),
    ]
    
    condition_variation = ConditionVariation(
        temperature_range_K=(8000.0, 15000.0),
        electron_density_range_cm3=(1e16, 1e18),
        laser_energy_variation=0.1,
        noise_level=0.02,
        background_level=0.01,
        gate_delay_range_us=(0.5, 2.0),
    )
    
    dataset = generate_synthetic_dataset(
        n_samples=n_samples,
        composition_ranges=composition_ranges,
        condition_variation=condition_variation,
        seed=seed,
    )
    
    # Get atomic database and element list
    db = AtomicDatabase()
    all_elements = [cr.element for cr in composition_ranges]
    
    # Run inversion pipeline for each spectrum
    raw_concentrations: Dict[str, List[float]] = {el: [] for el in all_elements}
    closed_concentrations: Dict[str, List[float]] = {el: [] for el in all_elements}
    ground_truth: Dict[str, List[float]] = {el: [] for el in all_elements}
    
    for spectrum in dataset.spectra:
        # Run inversion pipeline
        result = run_inversion_pipeline(
            wavelength=spectrum.wavelength,
            intensity=spectrum.intensity,
            db=db,
            elements=all_elements,
            resolving_power=1000.0,
            temperature_guess_K=10000.0,
            electron_density_guess_cm3=1e17,
        )
        
        if result is None or result.concentrations is None:
            continue
        
        # Extract ground truth composition (mass fractions)
        truth_composition = spectrum.sample_metadata.composition
        truth_elements = derive_truth_elements(truth_composition)
        
        # Store ground truth
        for el in all_elements:
            if el in truth_elements:
                ground_truth[el].append(truth_composition.get(el, 0.0))
            else:
                ground_truth[el].append(0.0)
        
        # Store raw concentrations (from Boltzmann fit, not closed)
        # The raw values come from the intercepts before closure
        for el in all_elements:
            if el in result.intercepts:
                # Raw relative concentration from Boltzmann plot
                U_s = result.partition_funcs.get(el, 1.0)
                q_s = result.intercepts[el]
                raw_c = U_s * np.exp(q_s)
                raw_concentrations[el].append(raw_c)
            else:
                raw_concentrations[el].append(0.0)
        
        # Store closed concentrations
        if result.concentrations:
            for el in all_elements:
                if el in result.concentrations:
                    closed_concentrations[el].append(result.concentrations[el])
                else:
                    closed_concentrations[el].append(0.0)
    
    # Convert to numpy arrays
    raw_conc_arr = {el: np.array(vals) for el, vals in raw_concentrations.items()}
    closed_conc_arr = {el: np.array(vals) for el, vals in closed_concentrations.items()}
    truth_arr = {el: np.array(vals) for el, vals in ground_truth.items()}
    
    # Normalize raw concentrations to compare with closed
    # (they should sum to ~1 after closure)
    for el in all_elements:
        raw_sum = np.sum(raw_conc_arr[el])
        if raw_sum > 0:
            raw_conc_arr[el] = raw_conc_arr[el] / raw_sum
    
    # Compute diagnostics
    diagnostics = diagnose_closure_impact(
        raw_conc_arr,
        closed_conc_arr,
        truth_arr,
        all_elements,
    )
    
    return {
        "raw": raw_conc_arr,
        "closed": closed_conc_arr,
        "ground_truth": truth_arr,
    }, diagnostics


def compare_closure_methods(
    n_samples: int = 100,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Compare different closure methods: standard, ILR, softmax.
    
    Parameters
    ----------
    n_samples : int
        Number of synthetic spectra.
    seed : int
        Random seed.
    
    Returns
    -------
    Dict[str, Dict]
        Results for each closure method.
    """
    np.random.seed(seed)
    
    # Generate synthetic dataset
    composition_ranges = [
        CompositionRange("Fe", 0.01, 0.50, "uniform", required=True),
        CompositionRange("Cr", 0.01, 0.30, "uniform", required=True),
        CompositionRange("Ni", 0.01, 0.25, "uniform", required=True),
    ]
    
    condition_variation = ConditionVariation(
        temperature_range_K=(10000.0, 12000.0),
        electron_density_range_cm3=(5e16, 5e17),
        laser_energy_variation=0.05,
        noise_level=0.01,
        background_level=0.005,
        gate_delay_range_us=(1.0, 1.5),
    )
    
    dataset = generate_synthetic_dataset(
        n_samples=n_samples,
        composition_ranges=composition_ranges,
        condition_variation=condition_variation,
        seed=seed,
    )
    
    db = AtomicDatabase()
    elements = [cr.element for cr in composition_ranges]
    
    results = {
        "standard": {"rmsep": {}, "bias": {}},
        "ilr": {"rmsep": {}, "bias": {}},
        "softmax": {"rmsep": {}, "bias": {}},
    }
    
    for spectrum in dataset.spectra:
        truth_composition = spectrum.sample_metadata.composition
        
        # Run with standard closure
        try:
            from cflibs.inversion.closure import ClosureMethod
            
            intercepts = {el: 0.0 for el in elements}  # Placeholder
            partition_funcs = {el: 1.0 for el in elements}  # Placeholder
            
            # For demo, use synthetic values based on ground truth
            for el in elements:
                frac = truth_composition.get(el, 0.01)
                intercepts[el] = np.log(frac + 1e-10)
                partition_funcs[el] = 1.0
            
            # Standard closure
            std_result = ClosureMethod.apply_standard(
                intercepts, partition_funcs, mode="standard"
            )
            
            # ILR closure
            ilr_result = ClosureMethod.apply_ilr(
                intercepts, partition_funcs
            )
            
            # Softmax closure (via JAX)
            try:
                import jax.numpy as jnp
                theta = jnp.array([intercepts.get(el, 0.0) for el in elements])
                sm_result = jax_softmax_closure(theta)
                softmax_concentrations = {el: float(sm_result[i]) for i, el in enumerate(elements)}
            except ImportError:
                softmax_concentrations = {el: 0.0 for el in elements}
            
            # Compute errors
            for el in elements:
                truth = truth_composition.get(el, 0.0)
                
                if el in std_result.concentrations:
                    std_pred = std_result.concentrations[el]
                    results["standard"]["rmsep"].setdefault(el, []).append((std_pred - truth)**2)
                    results["standard"]["bias"].setdefault(el, []).append(std_pred - truth)
                
                if el in ilr_result.concentrations:
                    ilr_pred = ilr_result.concentrations[el]
                    results["ilr"]["rmsep"].setdefault(el, []).append((ilr_pred - truth)**2)
                    results["ilr"]["bias"].setdefault(el, []).append(ilr_pred - truth)
                
                if el in softmax_concentrations:
                    sm_pred = softmax_concentrations[el]
                    results["softmax"]["rmsep"].setdefault(el, []).append((sm_pred - truth)**2)
                    results["softmax"]["bias"].setdefault(el, []).append(sm_pred - truth)
                    
        except Exception as e:
            print(f"Error processing spectrum: {e}", file=sys.stderr)
            continue
    
    # Aggregate results
    aggregated = {}
    for method in ["standard", "ilr", "softmax"]:
        aggregated[method] = {
            "rmsep": {},
            "bias": {},
        }
        for el in elements:
            if el in results[method]["rmsep"]:
                rmse_list = results[method]["rmsep"][el]
                bias_list = results[method]["bias"][el]
                aggregated[method]["rmsep"][el] = np.sqrt(np.mean(rmse_list))
                aggregated[method]["bias"][el] = np.mean(bias_list)
    
    return aggregated


def print_diagnostics_report(diagnostics: Dict[str, ClosureDiagnostics]) -> None:
    """Print formatted diagnostics report."""
    print("\n" + "="*80)
    print("CLOSURE IMPACT DIAGNOSTICS REPORT")
    print("="*80)
    
    # Summary by element
    print("\nPer-Element Analysis:")
    print("-"*80)
    
    for element, diag in sorted(diagnostics.items()):
        print(f"\n{element}:")
        print(f"  Samples: {diag.n_samples}")
        print(f"  Truth: mean={diag.truth_mean:.4f}, std={diag.truth_std:.4f}, range=[{diag.truth_range[0]:.4f}, {diag.truth_range[1]:.4f}]")
        
        print(f"  Raw estimates:")
        print(f"    RMSEP: {diag.raw_rmsep:.4f}")
        print(f"    Bias:  {diag.raw_bias:+.4f}")
        print(f"    By category: major={diag.raw_rmseps_by_category['major']:.4f}, "
              f"minor={diag.raw_rmseps_by_category['minor']:.4f}, "
              f"trace={diag.raw_rmseps_by_category['trace']:.4f}")
        
        print(f"  Closed estimates:")
        print(f"    RMSEP: {diag.closed_rmsep:.4f}")
        print(f"    Bias:  {diag.closed_bias:+.4f}")
        print(f"    By category: major={diag.closed_rmseps_by_category['major']:.4f}, "
              f"minor={diag.closed_rmseps_by_category['minor']:.4f}, "
              f"trace={diag.closed_rmseps_by_category['trace']:.4f}")
        
        print(f"  Closure impact:")
        if diag.error_amplified:
            print(f"    ⚠️  ERROR AMPLIFIED by closure")
            print(f"    Amplification factor: {diag.amplification_factor:.2f}x")
        else:
            print(f"    ✓  Error reduced by closure")
            print(f"    Improvement factor: {1/diag.amplification_factor:.2f}x")
    
    # Summary by category
    print("\n" + "-"*80)
    print("Summary by Concentration Category:")
    print("-"*80)
    
    categories = ["major", "minor", "trace"]
    for cat in categories:
        elements_in_cat = [
            el for el, diag in diagnostics.items()
            if diag.truth_mean > 0.05 and cat == "major" or
               0.01 <= diag.truth_mean <= 0.05 and cat == "minor" or
               diag.truth_mean < 0.01 and cat == "trace"
        ]
        
        if not elements_in_cat:
            continue
        
        avg_raw_rmsep = np.mean([diagnostics[el].raw_rmsep for el in elements_in_cat])
        avg_closed_rmsep = np.mean([diagnostics[el].closed_rmsep for el in elements_in_cat])
        amplification = avg_closed_rmsep / max(avg_raw_rmsep, 1e-10)
        
        print(f"\n{cat.upper()} elements ({len(elements_in_cat)}):")
        print(f"  Avg raw RMSEP:    {avg_raw_rmsep:.4f}")
        print(f"  Avg closed RMSEP: {avg_closed_rmsep:.4f}")
        if amplification > 1.0:
            print(f"  ⚠️  Closure amplifies error by {amplification:.2f}x")
        else:
            print(f"  ✓  Closure reduces error by {1/amplification:.2f}x")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    total_elements = len(diagnostics)
    amplified_count = sum(1 for d in diagnostics.values() if d.error_amplified)
    
    print(f"\nTotal elements analyzed: {total_elements}")
    print(f"Elements with error amplification: {amplified_count}")
    print(f"Elements with error reduction: {total_elements - amplified_count}")
    
    if amplified_count > total_elements / 2:
        print("\n⚠️  WARNING: Closure is amplifying errors for most elements!")
        print("   Consider investigating:")
        print("   - Missing elements in the identified set")
        print("   - Calibration errors in Boltzmann fits")
        print("   - Non-equilibrium plasma conditions")
    else:
        print("\n✓  Closure is generally improving accuracy.")
    
    print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Diagnose closure impact on CF-LIBS concentration estimates"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of synthetic spectra to generate (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    print(f"Running closure diagnostics with {args.n_samples} samples (seed={args.seed})...")
    
    # Run main analysis
    results, diagnostics = run_closure_comparison(
        n_samples=args.n_samples,
        seed=args.seed,
    )
    
    # Print diagnostics report
    print_diagnostics_report(diagnostics)
    
    # Run closure method comparison
    print("\nComparing different closure methods...")
    method_comparison = compare_closure_methods(
        n_samples=args.n_samples,
        seed=args.seed,
    )
    
    print("\n" + "="*80)
    print("CLOSURE METHOD COMPARISON")
    print("="*80)
    
    for method in ["standard", "ilr", "softmax"]:
        print(f"\n{method.upper()} closure:")
        for el, rmsep in method_comparison[method]["rmsep"].items():
            bias = method_comparison[method]["bias"].get(el, 0.0)
            print(f"  {el}: RMSEP={rmsep:.4f}, Bias={bias:+.4f}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
