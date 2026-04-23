"""Driver-side runtime configuration for the evolution loop.

This dataclass captures the settings that govern the evolution *driver*
itself — batch sizes, timeouts, aggregation weights, enforcement mode —
not the PARAMETERS dict inside the evolved code target (that belongs to
``cflibs.evolution`` downstream, tracked under CF-LIBS-improved-jrw).

Keeping these two concerns in separate modules makes the scope
explicit: values here are what the orchestration loop (bd ``4e0j``)
reads to control its own behaviour; values in the evolved-code
PARAMETERS dict are what the 27B Scout mutates per batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EnforcementMode = Literal["hard", "warn"]


@dataclass(frozen=True)
class FitnessWeights:
    """Weights for aggregating per-dataset fitness into a composite score.

    The evolution loop scores candidates across multiple matrix classes
    (Aalto minerals, ChemCam/SuperCam silicates, USGS geostandards,
    NIST steels) and aggregates them into a single scalar. Keeping the
    weights here makes the matrix-specific overfitting penalty — see
    epic CF-LIBS-improved-3fy3 — adjustable without changing loop code.
    """

    aalto: float = 1.0
    chemcam: float = 1.0
    supercam: float = 1.0
    usgs: float = 1.0
    nist_steel: float = 1.0
    # Penalty multiplier on per-dataset fitness variance: higher values
    # more strongly punish candidates that overfit to one matrix.
    overfitting_penalty: float = 0.5


@dataclass(frozen=True)
class EvolutionDriverConfig:
    """Runtime configuration for the hierarchical-ES evolution driver.

    Values capture *driver* behaviour only. The evolved code's own
    PARAMETERS dict (thresholds, weights the 27B Scout tunes) lives
    elsewhere — see CF-LIBS-improved-jrw.
    """

    # Batch sizing for Phase 1 (perturbation generation).
    perturbations_per_batch: int = 16
    perturbation_timeout_s: float = 45.0

    # Phase 2 evaluation parallelism. 0 == run serially (useful for tests).
    evaluation_workers: int = 8
    evaluation_timeout_s: float = 5.0

    # Phase 4 (structural mutation) cadence — propose a structural change
    # every K batches. Set to a large number to effectively disable.
    structural_mutation_cadence: int = 10

    # Blocklist enforcement mode. "hard" sets fitness = -inf on any
    # violation; "warn" logs but allows evaluation (useful during
    # scanner development — never in production runs).
    enforcement_mode: EnforcementMode = "hard"

    # Fitness aggregation across datasets.
    fitness_weights: FitnessWeights = field(default_factory=FitnessWeights)

    # Cap on total wall-clock (safety net for autonomous runs).
    max_wallclock_hours: float = 72.0

    def __post_init__(self) -> None:
        if self.perturbations_per_batch < 1:
            raise ValueError("perturbations_per_batch must be >= 1")
        if self.perturbation_timeout_s <= 0 or self.evaluation_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        if self.evaluation_workers < 0:
            raise ValueError("evaluation_workers must be >= 0")
        if self.structural_mutation_cadence < 1:
            raise ValueError("structural_mutation_cadence must be >= 1")
        if self.max_wallclock_hours <= 0:
            raise ValueError("max_wallclock_hours must be positive")
        if self.enforcement_mode not in ("hard", "warn"):
            raise ValueError(
                f"enforcement_mode must be 'hard' or 'warn', got {self.enforcement_mode!r}"
            )
