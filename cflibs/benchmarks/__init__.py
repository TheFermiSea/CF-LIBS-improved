"""
Deprecated: use cflibs.benchmark instead.

This package is a backward-compatibility shim. All modules have been
merged into cflibs.benchmark as of the codebase cleanup (2026-04).
"""

import warnings

warnings.warn(
    "cflibs.benchmarks is deprecated. Use cflibs.benchmark instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical package for backward compat.
from cflibs.benchmark.composition_metrics import (  # noqa: F401, E402
    aitchison_distance,
    clr_transform,
    ilr_inverse,
    ilr_transform,
    per_element_error,
    rmse_composition,
)
from cflibs.benchmark.corpus import (  # noqa: F401, E402
    BenchmarkCorpus,
    BenchmarkSpectrum,
)
from cflibs.benchmark.harness import (  # noqa: F401, E402
    AccuracyTier,
    BenchmarkHarness,
    BenchmarkReport,
    PipelineResult,
    SpectrumResult,
)
