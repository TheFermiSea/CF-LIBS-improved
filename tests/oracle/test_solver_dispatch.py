"""Oracle tests for the unified solver dispatch (cflibs.inversion.pipeline).

These tests pin the *contract* of the solver-selection seam without touching
the atomic database (so they run under the fast ``not requires_db`` gate):

* the new config fields exist and ride ``config_overrides`` through
  ``build_pipeline_config`` (the single edit point that makes
  ``config_overrides={"solver": X}`` legal);
* the full-spectrum adapter helpers (``_number_to_mass_fractions``,
  ``_physical_solution``, ``_to_cflibs_result``) honour their contracts;
* ``_dispatch_solver`` routes each ``pipeline.solver`` to the right backend
  (verified with monkeypatched solver stubs, so no DB / JAX work runs);
* the reserved ``coarse_to_fine`` solver raises ``NotImplementedError`` and an
  unknown solver raises ``ValueError``.

A DB-backed end-to-end smoke test (one spectrum per peak-based solver) lives
behind ``@pytest.mark.requires_db`` so CI exercises the real path.
"""

from __future__ import annotations

import math

import pytest

from cflibs.inversion import pipeline as pl
from cflibs.inversion.pipeline import (
    AnalysisPipelineConfig,
    _dispatch_solver,
    _number_to_mass_fractions,
    _physical_solution,
    _to_cflibs_result,
    build_pipeline_config,
)

# ---------------------------------------------------------------------------
# Config: new fields + override threading
# ---------------------------------------------------------------------------


def test_config_has_solver_fields_with_expected_defaults():
    cfg = AnalysisPipelineConfig(preset="geological", elements=["Fe", "Ca"])
    assert cfg.solver == "iterative"
    assert cfg.solver_overrides == {}
    # The five iterative knobs lifted onto the config (audit build_order step 1).
    assert cfg.apply_ipd is False
    assert cfg.two_region is False
    assert cfg.aki_uncertainty_weighting is True
    assert cfg.degeneracy_dominance_threshold == pytest.approx(0.8)
    assert cfg.degeneracy_min_elements == 4


def test_solver_fields_join_overridable_whitelist_automatically():
    # The whitelist is derived from the dataclass fields, so the new knobs are
    # overridable with NO edit to _OVERRIDABLE_FIELDS.
    for field in (
        "solver",
        "solver_overrides",
        "apply_ipd",
        "two_region",
        "aki_uncertainty_weighting",
        "degeneracy_dominance_threshold",
        "degeneracy_min_elements",
    ):
        assert field in pl._OVERRIDABLE_FIELDS


def test_config_overrides_thread_solver_through_build_pipeline_config():
    cfg = build_pipeline_config(
        ["Fe", "Ca"],
        overrides={
            "solver": "closed_form",
            "solver_overrides": {"saha_passes": 1},
            "apply_ipd": True,
            "degeneracy_min_elements": 6,
        },
    )
    assert cfg.solver == "closed_form"
    assert cfg.solver_overrides == {"saha_passes": 1}
    assert cfg.apply_ipd is True
    assert cfg.degeneracy_min_elements == 6


def test_build_pipeline_config_default_solver_is_iterative():
    # Production baseline (no override) must stay on iterative — the parity guard
    # for every existing scoreboard run.
    cfg = build_pipeline_config(["Fe", "Ca"])
    assert cfg.solver == "iterative"
    assert cfg.solver_overrides == {}


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def test_number_to_mass_fractions_matches_hand_calc():
    # Fe=55.85, Ca=40.08; equal number fractions -> mass-weighted split.
    mass = _number_to_mass_fractions({"Fe": 0.5, "Ca": 0.5})
    denom = 0.5 * 55.85 + 0.5 * 40.08
    assert mass["Fe"] == pytest.approx(0.5 * 55.85 / denom)
    assert mass["Ca"] == pytest.approx(0.5 * 40.08 / denom)
    assert sum(mass.values()) == pytest.approx(1.0)
    # Heavier element gets the larger mass fraction.
    assert mass["Fe"] > mass["Ca"]


def test_number_to_mass_fractions_is_not_identity():
    # Guards the CRITICAL bug: scoring number fractions as mass fractions.
    number = {"Fe": 0.5, "Ca": 0.5}
    mass = _number_to_mass_fractions(number)
    assert mass != number


def test_number_to_mass_fractions_all_zero_returns_zeros():
    mass = _number_to_mass_fractions({"Fe": 0.0, "Ca": 0.0})
    assert mass == {"Fe": 0.0, "Ca": 0.0}


@pytest.mark.parametrize(
    "T_K, ne, expected",
    [
        (10000.0, 1e17, True),
        (3.5e-06, 7.2e-38, False),  # the collapsed-BFGS signature
        (0.0, 1e17, False),
        (10000.0, 0.0, False),
        (float("nan"), 1e17, False),
        (float("inf"), 1e17, False),
        (500.0, 1e17, False),  # below the LIBS T floor
    ],
)
def test_physical_solution_floors(T_K, ne, expected):
    assert _physical_solution(T_K, ne) is expected


def test_to_cflibs_result_contract():
    res = _to_cflibs_result(
        concentrations_mass={"Fe": 0.6, "Ca": 0.4},
        temperature_K=12000.0,
        electron_density_cm3=1e17,
        converged=True,
        failed=False,
        iterations=7,
        extra_quality={"final_loss": 1.23},
    )
    # The exact surface _score_spectrum reads.
    assert res.temperature_K == pytest.approx(12000.0)
    assert res.electron_density_cm3 == pytest.approx(1e17)
    assert res.concentrations == {"Fe": 0.6, "Ca": 0.4}
    assert res.converged is True
    assert res.quality_metrics["failed"] == 0.0
    assert res.quality_metrics["final_loss"] == pytest.approx(1.23)
    assert res.iterations == 7


def test_to_cflibs_result_failed_sets_failure_flag():
    res = _to_cflibs_result(
        concentrations_mass={"Fe": 0.0},
        temperature_K=0.0,
        electron_density_cm3=0.0,
        converged=False,
        failed=True,
    )
    # >= 1.0 is what the scoreboard's all-FN failure branch checks.
    assert float(res.quality_metrics["failed"]) >= 1.0


# ---------------------------------------------------------------------------
# Dispatch routing (monkeypatched — no DB / JAX)
# ---------------------------------------------------------------------------


def _cfg(solver: str) -> AnalysisPipelineConfig:
    return AnalysisPipelineConfig(preset="geological", elements=["Fe", "Ca"], solver=solver)


def test_dispatch_routes_peak_based(monkeypatch):
    calls = {}

    def fake_peak(pipeline, observations, atomic_db, stark, unc):
        calls["peak"] = pipeline.solver
        return "PEAK_RESULT"

    def fake_full(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("full-spectrum path called for a peak-based solver")

    monkeypatch.setattr(pl, "_run_peak_based_solver", fake_peak)
    monkeypatch.setattr(pl, "_run_full_spectrum_solver", fake_full)

    for solver in ("iterative", "closed_form"):
        out = _dispatch_solver(_cfg(solver), [], object(), None, None, None, "none")
        assert out == "PEAK_RESULT"
        assert calls["peak"] == solver


def test_dispatch_routes_full_spectrum(monkeypatch):
    calls = {}

    def fake_full(pipeline, atomic_db, wavelength, intensity):
        calls["full"] = pipeline.solver
        return "FULL_RESULT"

    def fake_peak(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("peak-based path called for a full-spectrum solver")

    monkeypatch.setattr(pl, "_run_full_spectrum_solver", fake_full)
    monkeypatch.setattr(pl, "_run_peak_based_solver", fake_peak)

    for solver in ("joint", "bayesian"):
        out = _dispatch_solver(_cfg(solver), [], object(), [1.0], [2.0], None, "none")
        assert out == "FULL_RESULT"
        assert calls["full"] == solver


def test_dispatch_coarse_to_fine_is_not_implemented():
    with pytest.raises(NotImplementedError, match="manifold"):
        _dispatch_solver(_cfg("coarse_to_fine"), [], object(), None, None, None, "none")


def test_dispatch_unknown_solver_raises_value_error():
    with pytest.raises(ValueError, match="Unknown solver"):
        _dispatch_solver(_cfg("nope"), [], object(), None, None, None, "none")


# ---------------------------------------------------------------------------
# DB-backed end-to-end smoke (excluded from the fast gate)
# ---------------------------------------------------------------------------


@pytest.mark.requires_db
@pytest.mark.parametrize("solver", ["iterative", "closed_form"])
def test_peak_based_solver_scores_finite_rmse(solver):
    """Each peak-based solver returns a real (finite) wt% composition RMSE."""
    from cflibs.atomic import AtomicDatabase
    from cflibs.benchmark.scoreboard import run_scoreboard

    db = AtomicDatabase("ASD_da/libs_production.db")
    board = run_scoreboard(
        db,
        datasets=["supercam_labcal"],
        max_spectra=2,
        seed=7,
        pipeline_impl="reference",
        config_overrides={"solver": solver},
    )
    comp = board["datasets"][0].get("composition") or {}
    rmse = comp.get("rmse_wt_median")
    assert rmse is not None
    assert math.isfinite(float(rmse))
