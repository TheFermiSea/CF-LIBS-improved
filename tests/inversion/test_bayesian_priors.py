"""Unit tests for the T1-6 declarative ``Parameter`` dataclass and prior helpers."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from cflibs.inversion.solve.bayesian.priors import (
    Parameter,
    PriorConfig,
    TwoZonePriorConfig,
)

# ---------------------------------------------------------------------------
# Parameter dataclass behaviour (no JAX/NumPyro required for cube_transform)
# ---------------------------------------------------------------------------


def test_parameter_is_frozen():
    p = Parameter("T_eV", 0.5, 3.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.name = "X"  # type: ignore[misc]


def test_parameter_uniform_cube_midpoint():
    p = Parameter("T_eV", 0.5, 3.0, "uniform")
    assert p.cube_transform(0.5) == pytest.approx(1.75)
    assert p.cube_transform(0.0) == pytest.approx(0.5)
    assert p.cube_transform(1.0) == pytest.approx(3.0)


def test_parameter_loguniform_midpoint():
    """log_10(n_e) midpoint of [1e15, 1e19] -> 1e17."""
    p = Parameter("ne", 1e15, 1e19, "loguniform")
    assert p.cube_transform(0.5) == pytest.approx(1e17, rel=1e-10)
    assert p.cube_transform(0.0) == pytest.approx(1e15, rel=1e-10)
    assert p.cube_transform(1.0) == pytest.approx(1e19, rel=1e-10)


def test_parameter_truncnormal_round_trip_at_midpoint():
    """Truncated normal at u=0.5 returns the mean (symmetric truncation)."""
    p = Parameter(
        "T_eV",
        prior_low=0.0,
        prior_high=4.0,
        prior_kind="truncnormal",
        prior_mean=2.0,
        prior_std=0.5,
    )
    assert p.cube_transform(0.5) == pytest.approx(2.0, abs=1e-6)


def test_parameter_normal_ppf_at_midpoint():
    p = Parameter(
        "x",
        prior_low=-np.inf,
        prior_high=np.inf,
        prior_kind="normal",
        prior_mean=1.5,
        prior_std=0.25,
    )
    assert p.cube_transform(0.5) == pytest.approx(1.5, abs=1e-9)


def test_parameter_vary_false_returns_fixed_value():
    p = Parameter("T_eV", 0.5, 3.0, "uniform", vary=False, prior_mean=1.2)
    assert p.cube_transform(0.0) == pytest.approx(1.2)
    assert p.cube_transform(1.0) == pytest.approx(1.2)


def test_parameter_vary_false_falls_back_to_low():
    p = Parameter("T_eV", 0.7, 3.0, "uniform", vary=False)
    assert p.cube_transform(0.5) == pytest.approx(0.7)


def test_parameter_dirichlet_cube_transform_rejected():
    p = Parameter("conc", 0.0, 1.0, "dirichlet")
    with pytest.raises(ValueError, match="stick-breaking"):
        p.cube_transform(0.5)


def test_parameter_unknown_prior_kind_raises():
    p = Parameter("x", 0.0, 1.0)
    object.__setattr__(p, "prior_kind", "bogus")  # bypass frozen
    with pytest.raises(ValueError, match="Unsupported prior_kind"):
        p.cube_transform(0.5)


def test_parameter_normal_missing_mean_raises():
    p = Parameter("x", -1.0, 1.0, "normal")
    with pytest.raises(ValueError, match="normal"):
        p.cube_transform(0.5)


# ---------------------------------------------------------------------------
# NumPyro adapter (only when numpyro is importable)
# ---------------------------------------------------------------------------


@pytest.mark.requires_bayesian
def test_parameter_numpyro_uniform_records_site():
    """Inside a NumPyro trace, ``numpyro_sample`` records a Uniform site."""
    pytest.importorskip("numpyro")
    from numpyro import handlers as numpyro_handlers

    p = Parameter("T_eV", 0.5, 3.0, "uniform")

    def model():
        return p.numpyro_sample()

    with numpyro_handlers.trace() as tr, numpyro_handlers.seed(rng_seed=0):
        model()

    assert "T_eV" in tr
    site = tr["T_eV"]
    val = float(site["value"])
    assert 0.5 <= val <= 3.0


@pytest.mark.requires_bayesian
def test_parameter_numpyro_loguniform_returns_power_of_ten():
    """log-uniform draws are emitted as ``10 ** log_x`` deterministic sites."""
    pytest.importorskip("numpyro")
    from numpyro import handlers as numpyro_handlers

    p = Parameter("ne", 1e15, 1e19, "loguniform")

    def model():
        return p.numpyro_sample()

    with numpyro_handlers.trace() as tr, numpyro_handlers.seed(rng_seed=1):
        model()

    # The log-uniform site is named "<name>_log"; the deterministic is "<name>".
    assert "ne_log" in tr
    assert "ne" in tr
    raw = float(tr["ne_log"]["value"])
    derived = float(tr["ne"]["value"])
    assert math.isclose(derived, 10.0**raw, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Legacy PriorConfig / TwoZonePriorConfig smoke
# ---------------------------------------------------------------------------


def test_prior_config_geological_sparse_concentrations():
    cfg = PriorConfig.geological()
    assert cfg.concentration_alpha < 1.0  # sparse


def test_two_zone_prior_config_defaults():
    cfg = TwoZonePriorConfig()
    assert cfg.T_core_eV_range[0] < cfg.T_core_eV_range[1]
    assert cfg.enforce_T_ordering is True
