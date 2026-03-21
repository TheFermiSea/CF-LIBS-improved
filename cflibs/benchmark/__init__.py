"""
Open LIBS Spectral Benchmark Database.

This module provides standardized benchmarks for comparing LIBS algorithm
performance, addressing the research gap of non-standardized cross-study
comparisons in the LIBS community.

Key Features
------------
- Certified reference material (CRM) spectra representation
- Standardized train/test splits for reproducible evaluation
- Community-accepted evaluation metrics (RMSEP, MAE, bias)
- Support for multiple instrumental conditions
- Synthetic benchmark generation for validation

Main Classes
------------
BenchmarkSpectrum
    Single spectrum with ground truth composition and metadata
BenchmarkDataset
    Collection of spectra with train/test splits
BenchmarkMetrics
    Evaluation metrics for algorithm comparison
SyntheticBenchmarkGenerator
    Generate synthetic benchmarks with known ground truth

Example
-------
>>> from cflibs.benchmark import BenchmarkDataset, BenchmarkMetrics
>>>
>>> # Load a benchmark dataset
>>> dataset = BenchmarkDataset.from_json("nist_steel_crm.json")
>>>
>>> # Get train/test splits
>>> train, test = dataset.get_split("default")
>>>
>>> # Evaluate predictions
>>> metrics = BenchmarkMetrics()
>>> results = metrics.evaluate(predictions, test.true_compositions)
>>> print(results.summary())

References
----------
- Hahn & Omenetto (2010) "Applied Spectroscopy Focal Point Review"
- Tognoni et al. (2010) "CF-LIBS: State of the art"
- NIST Standard Reference Materials program
"""

from importlib import import_module

_LAZY_ATTRIBUTE_GROUPS = {
    "cflibs.benchmark.synthetic_eval": [
        "CalibrationOptions",
        "compute_binary_metrics",
        "run_synthetic_benchmark",
    ],
    "cflibs.benchmark.unified": [
        "UnifiedBenchmarkContext",
        "UnifiedBenchmarkRunner",
        "build_composition_workflow_registry",
        "build_id_workflow_registry",
        "load_default_datasets",
    ],
}

__all__ = [
    # Core data structures
    "BenchmarkSpectrum",
    "InstrumentalConditions",
    "SampleMetadata",
    "BenchmarkDataset",
    "DataSplit",
    # Metrics
    "BenchmarkMetrics",
    "EvaluationResult",
    "ElementMetrics",
    "MetricType",
    # Synthetic generation
    "SyntheticBenchmarkGenerator",
    "CompositionRange",
    "ConditionVariation",
    "CorpusRecipe",
    "PerturbationAxes",
    "build_synthetic_id_corpus",
    "default_axes",
    "default_recipes",
    "UnifiedBenchmarkContext",
    "UnifiedBenchmarkRunner",
    "build_composition_workflow_registry",
    "build_id_workflow_registry",
    "load_default_datasets",
    "CalibrationOptions",
    "compute_binary_metrics",
    "run_synthetic_benchmark",
    # I/O
    "load_benchmark",
    "save_benchmark",
    "BenchmarkFormat",
]

_ATTRIBUTE_EXPORT_GROUPS = {
    "cflibs.benchmark.dataset": [
        "BenchmarkSpectrum",
        "InstrumentalConditions",
        "SampleMetadata",
        "BenchmarkDataset",
        "DataSplit",
    ],
    "cflibs.benchmark.metrics": [
        "BenchmarkMetrics",
        "EvaluationResult",
        "ElementMetrics",
        "MetricType",
    ],
    "cflibs.benchmark.synthetic": [
        "SyntheticBenchmarkGenerator",
        "CompositionRange",
        "ConditionVariation",
    ],
    "cflibs.benchmark.synthetic_corpus": [
        "CorpusRecipe",
        "PerturbationAxes",
        "build_synthetic_id_corpus",
        "default_axes",
        "default_recipes",
    ],
    "cflibs.benchmark.loaders": ["load_benchmark", "save_benchmark", "BenchmarkFormat"],
}

_MODULE_EXPORTS = {
    attr_name: (module_name, attr_name)
    for module_name, attr_names in _ATTRIBUTE_EXPORT_GROUPS.items()
    for attr_name in attr_names
}


def __getattr__(name: str):
    """Lazy-load benchmark exports so lightweight submodules remain importable."""
    if name in _MODULE_EXPORTS:
        module_name, attr_name = _MODULE_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    for module_name, attr_names in _LAZY_ATTRIBUTE_GROUPS.items():
        if name in attr_names:
            module = import_module(module_name)
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
