"""Tests for the evolution driver's runtime config dataclass."""

from __future__ import annotations

import pytest

from cflibs.evolution.config import EvolutionDriverConfig, FitnessWeights


def test_defaults_are_sensible() -> None:
    cfg = EvolutionDriverConfig()
    assert cfg.perturbations_per_batch >= 1
    assert cfg.perturbation_timeout_s > 0
    assert cfg.evaluation_timeout_s > 0
    assert cfg.structural_mutation_cadence >= 1
    assert cfg.enforcement_mode == "hard"


def test_fitness_weights_default_covers_all_planned_matrices() -> None:
    w = FitnessWeights()
    # All five dataset classes from epic 3fy3 must have an explicit weight.
    for name in ("aalto", "chemcam", "supercam", "usgs", "nist_steel"):
        assert hasattr(w, name), f"missing default weight for {name}"


def test_override_evaluation_workers_to_serial() -> None:
    cfg = EvolutionDriverConfig(evaluation_workers=0)
    assert cfg.evaluation_workers == 0


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"perturbations_per_batch": 0}, "perturbations_per_batch"),
        ({"perturbations_per_batch": -3}, "perturbations_per_batch"),
        ({"perturbation_timeout_s": 0}, "timeouts"),
        ({"evaluation_timeout_s": -1.0}, "timeouts"),
        ({"evaluation_workers": -1}, "evaluation_workers"),
        ({"structural_mutation_cadence": 0}, "structural_mutation_cadence"),
        ({"max_wallclock_hours": 0}, "max_wallclock_hours"),
    ],
)
def test_invalid_values_rejected(kwargs: dict, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        EvolutionDriverConfig(**kwargs)


def test_enforcement_mode_must_be_hard_or_warn() -> None:
    with pytest.raises(ValueError, match="enforcement_mode"):
        EvolutionDriverConfig(enforcement_mode="off")  # type: ignore[arg-type]


def test_dataclass_is_frozen() -> None:
    cfg = EvolutionDriverConfig()
    with pytest.raises(Exception):
        cfg.perturbations_per_batch = 99  # type: ignore[misc]
