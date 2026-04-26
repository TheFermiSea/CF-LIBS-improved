"""Driver-side runtime configuration for the evolution loop.

This dataclass captures the settings that govern the evolution *driver*
itself — batch sizes, timeouts, aggregation weights, enforcement mode —
not the PARAMETERS dict inside the evolved code target (that belongs to
``cflibs.evolution`` downstream).

Keeping these two concerns in separate modules makes the scope explicit:
values here are what the orchestration loop reads to control its own
behaviour; values in the evolved-code PARAMETERS dict are what the
evolution search mutates per batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

EnforcementMode = Literal["hard", "warn"]


@dataclass(frozen=True)
class EvolutionDriverConfig:
    """Runtime configuration for the hierarchical-ES evolution driver.

    Captures *driver* behaviour only. The evolved code's own PARAMETERS
    dict (what the 27B Scout mutates per batch) lives separately.
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
    # scanner development — never in production runs). Consumed by the
    # evolution driver, not by `evaluator.scan_source` /
    # `evaluator.assert_physics_only` (those always raise on violations).
    enforcement_mode: EnforcementMode = "hard"

    # Per-dataset fitness weights. Keyed by dataset name so new matrices
    # can be added without a dataclass field change. Defaults cover the
    # five matrix classes currently planned for the multi-dataset
    # benchmark suite.
    fitness_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "aalto": 1.0,
            "chemcam": 1.0,
            "supercam": 1.0,
            "usgs": 1.0,
            "nist_steel": 1.0,
        }
    )

    # Multiplier on per-dataset fitness variance — higher values more
    # strongly punish candidates that win on one matrix and lose on others.
    overfitting_penalty: float = 0.5

    # Cap on total wall-clock (safety net for autonomous runs).
    max_wallclock_hours: float = 72.0

    def __post_init__(self) -> None:
        if self.perturbations_per_batch < 1:
            raise ValueError("perturbations_per_batch must be >= 1")
        if self.perturbation_timeout_s <= 0:
            raise ValueError("perturbation_timeout_s must be positive")
        if self.evaluation_timeout_s <= 0:
            raise ValueError("evaluation_timeout_s must be positive")
        if self.evaluation_workers < 0:
            raise ValueError("evaluation_workers must be >= 0")
        if self.structural_mutation_cadence < 1:
            raise ValueError("structural_mutation_cadence must be >= 1")
        if self.max_wallclock_hours <= 0:
            raise ValueError("max_wallclock_hours must be positive")
        if self.overfitting_penalty < 0:
            raise ValueError("overfitting_penalty must be >= 0")
        if self.enforcement_mode not in ("hard", "warn"):
            raise ValueError(
                f"enforcement_mode must be 'hard' or 'warn', got {self.enforcement_mode!r}"
            )
        if not self.fitness_weights:
            raise ValueError("fitness_weights must be non-empty")
        if any(not math.isfinite(w) or w < 0 for w in self.fitness_weights.values()):
            raise ValueError("fitness_weights must all be finite and >= 0 (no NaN/inf)")
        # Honour frozen=True for the mapping contents too: dict.__setitem__
        # would otherwise slip past the dataclass-level immutability.
        object.__setattr__(self, "fitness_weights", MappingProxyType(dict(self.fitness_weights)))
