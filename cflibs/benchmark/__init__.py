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

from cflibs.benchmark.dataset import (
    BenchmarkSpectrum,
    InstrumentalConditions,
    SampleMetadata,
    BenchmarkDataset,
    DataSplit,
)

from cflibs.benchmark.metrics import (
    BenchmarkMetrics,
    EvaluationResult,
    ElementMetrics,
    MetricType,
)

from cflibs.benchmark.synthetic import (
    SyntheticBenchmarkGenerator,
    CompositionRange,
    ConditionVariation,
)
from cflibs.benchmark.synthetic_corpus import (
    CorpusRecipe,
    PerturbationAxes,
    build_synthetic_id_corpus,
    default_axes,
    default_recipes,
)

from cflibs.benchmark.loaders import (
    load_benchmark,
    save_benchmark,
    BenchmarkFormat,
)

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
    "CalibrationOptions",
    "compute_binary_metrics",
    "run_synthetic_benchmark",
    # I/O
    "load_benchmark",
    "save_benchmark",
    "BenchmarkFormat",
]


def __getattr__(name: str):
    """Lazy-load heavy synthetic benchmark evaluation helpers."""
    if name in {"CalibrationOptions", "compute_binary_metrics", "run_synthetic_benchmark"}:
        from cflibs.benchmark import synthetic_eval as _synthetic_eval

        value = getattr(_synthetic_eval, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
