"""Tests for the evolution driver's runtime config dataclass."""

from __future__ import annotations

import dataclasses

import pytest

from cflibs.evolution.config import EvolutionDriverConfig


def test_defaults_are_sensible() -> None:
    cfg = EvolutionDriverConfig()
    assert cfg.perturbations_per_batch >= 1
    assert cfg.perturbation_timeout_s > 0
    assert cfg.evaluation_timeout_s > 0
    assert cfg.structural_mutation_cadence >= 1
    assert cfg.enforcement_mode == "hard"
    assert cfg.overfitting_penalty >= 0


def test_default_fitness_weights_cover_all_planned_matrices() -> None:
    cfg = EvolutionDriverConfig()
    for name in ("aalto", "chemcam", "supercam", "usgs", "nist_steel"):
        assert name in cfg.fitness_weights
        assert cfg.fitness_weights[name] > 0


def test_fitness_weights_accepts_new_dataset() -> None:
    cfg = EvolutionDriverConfig(
        fitness_weights={"aalto": 1.0, "custom_mars_soil": 2.5},
    )
    assert cfg.fitness_weights["custom_mars_soil"] == 2.5


def test_override_evaluation_workers_to_serial() -> None:
    cfg = EvolutionDriverConfig(evaluation_workers=0)
    assert cfg.evaluation_workers == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"perturbations_per_batch": 0},
        {"perturbations_per_batch": -3},
        {"perturbation_timeout_s": 0},
        {"evaluation_timeout_s": -1.0},
        {"evaluation_workers": -1},
        {"structural_mutation_cadence": 0},
        {"max_wallclock_hours": 0},
        {"overfitting_penalty": -0.1},
        {"fitness_weights": {"aalto": -1.0}},
    ],
)
def test_invalid_values_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EvolutionDriverConfig(**kwargs)


def test_enforcement_mode_must_be_hard_or_warn() -> None:
    with pytest.raises(ValueError, match="enforcement_mode"):
        EvolutionDriverConfig(enforcement_mode="off")  # type: ignore[arg-type]


def test_dataclass_is_frozen() -> None:
    cfg = EvolutionDriverConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.perturbations_per_batch = 99  # type: ignore[misc]
