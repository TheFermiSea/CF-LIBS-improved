"""M8 Lever 7: couple per-element CI width to the reliability flag.

Tests the pure coupling functions (per_element_reliability_from_uncertainty,
downgrade_quality_flag) and the opt-in env gate. The coupling is a pure
annotation: it must never alter T/n_e/composition, only label/downgrade.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from cflibs.inversion.physics.quality import (  # noqa: E402
    RELATIVE_UNCERTAINTY_TIERS,
    downgrade_quality_flag,
    per_element_reliability_from_uncertainty,
)
from cflibs.inversion.solve.iterative import (  # noqa: E402
    CFLIBSResult,
    IterativeCFLIBSSolver,
    LineObservation,
    _reliability_from_uncertainty_enabled,
)

pytestmark = [pytest.mark.unit]


# --------------------------------------------------------------------------- #
# per_element_reliability_from_uncertainty                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "conc,sigma,expected",
    [
        (0.50, 0.10, "ok"),  # rel 0.20 < poor(0.5)
        (0.50, 0.30, "poor"),  # rel 0.60 in (0.5, 1.0]
        (0.50, 0.60, "reject"),  # rel 1.20 > reject(1.0)
        (0.50, 0.0, "ok"),  # no uncertainty info -> ok
        (0.0, 0.10, "reject"),  # zero conc, finite sigma -> infinite rel CI
    ],
)
def test_per_element_label(conc, sigma, expected):
    out = per_element_reliability_from_uncertainty({"Fe": conc}, {"Fe": sigma})
    assert out["Fe"] == expected


def test_per_element_boundary_is_strict_greater_than():
    # rel exactly at the tier boundary must NOT downgrade (uses strict >).
    out = per_element_reliability_from_uncertainty(
        {"A": 1.0, "B": 1.0},
        {"A": RELATIVE_UNCERTAINTY_TIERS["poor"], "B": RELATIVE_UNCERTAINTY_TIERS["reject"]},
    )
    assert out["A"] == "ok"  # rel == 0.5, not > 0.5
    assert out["B"] == "poor"  # rel == 1.0, > 0.5 but not > 1.0


def test_per_element_handles_empty_and_mismatched_keys():
    assert per_element_reliability_from_uncertainty({}, {}) == {}
    out = per_element_reliability_from_uncertainty(
        {"Fe": 1.0, "Na": 0.01},
        {"Fe": 0.0, "Cu": 100.0},
    )
    assert out == {"Fe": "ok", "Na": "ok"}


@pytest.mark.parametrize(
    "conc,sigma,expected",
    [
        (0.50, float("nan"), "reject"),
        (0.50, float("inf"), "reject"),
        (0.50, -0.30, "poor"),  # sigma is a width; use magnitude.
        (-0.50, 0.10, "reject"),
        (float("nan"), 0.10, "reject"),
        (float("inf"), 0.10, "reject"),
    ],
)
def test_per_element_rejects_invalid_numeric_edges(conc, sigma, expected):
    out = per_element_reliability_from_uncertainty({"Fe": conc}, {"Fe": sigma})
    assert out["Fe"] == expected


def test_per_element_rejects_malformed_reported_uncertainty():
    out = per_element_reliability_from_uncertainty(
        {"Fe": 0.5},
        {"Fe": "not-a-number"},
    )
    assert out["Fe"] == "reject"


# --------------------------------------------------------------------------- #
# downgrade_quality_flag                                                       #
# --------------------------------------------------------------------------- #


def test_downgrade_to_worst_element_tier():
    assert downgrade_quality_flag("good", {"Fe": "ok", "Na": "reject"}) == "reject"
    assert downgrade_quality_flag("excellent", {"Fe": "poor"}) == "poor"


def test_downgrade_never_upgrades():
    # Worse base flag than any element label -> unchanged.
    assert downgrade_quality_flag("reject", {"Fe": "ok", "Na": "poor"}) == "reject"
    assert downgrade_quality_flag("poor", {"Fe": "ok"}) == "poor"


def test_downgrade_ok_labels_are_noop():
    assert downgrade_quality_flag("good", {"Fe": "ok", "Na": "ok"}) == "good"


def test_downgrade_unknown_flag_passthrough():
    assert downgrade_quality_flag("unknown", {"Fe": "reject"}) == "unknown"


# --------------------------------------------------------------------------- #
# env gate (default OFF == legacy)                                             #
# --------------------------------------------------------------------------- #


def test_env_gate_default_off(monkeypatch):
    monkeypatch.delenv("CFLIBS_RELIABILITY_FROM_UNCERTAINTY", raising=False)
    assert _reliability_from_uncertainty_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "On"])
def test_env_gate_on_values(monkeypatch, val):
    monkeypatch.setenv("CFLIBS_RELIABILITY_FROM_UNCERTAINTY", val)
    assert _reliability_from_uncertainty_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "", "off", "nope"])
def test_env_gate_off_values(monkeypatch, val):
    monkeypatch.setenv("CFLIBS_RELIABILITY_FROM_UNCERTAINTY", val)
    assert _reliability_from_uncertainty_enabled() is False


# --------------------------------------------------------------------------- #
# integration: the coupling reproduces overall_reliable semantics             #
# --------------------------------------------------------------------------- #


def test_weak_emitter_downgrades_overall_reliable():
    """End-to-end logic: a clean fit (good flag, McWhirter ok) with a weak
    emitter carrying a huge CI must flip overall_reliable to False."""
    conc = {"Fe": 0.95, "Na": 0.05}
    sigma = {"Fe": 0.02, "Na": 0.08}  # Na rel CI = 1.6 -> reject
    labels = per_element_reliability_from_uncertainty(conc, sigma)
    assert labels["Na"] == "reject"
    new_flag = downgrade_quality_flag("good", labels)
    assert new_flag == "reject"
    mcwhirter_ok = True
    overall_reliable = mcwhirter_ok and new_flag in ("excellent", "good", "acceptable")
    assert overall_reliable is False


def _coupling_solver(monkeypatch):
    uncertainties = pytest.importorskip("uncertainties")

    db = MagicMock()
    db.get_ionization_potential.return_value = 7.0
    solver = IterativeCFLIBSSolver(db, max_iterations=1)
    base = CFLIBSResult(
        temperature_K=9000.0,
        temperature_uncertainty_K=0.0,
        electron_density_cm3=2e17,
        concentrations={"Fe": 0.95, "Na": 0.05},
        concentration_uncertainties={},
        iterations=3,
        converged=True,
        quality_metrics={
            "quality_flag": "good",
            "overall_reliable": True,
            "lte_mcwhirter_satisfied": True,
        },
        overall_reliable=True,
    )

    monkeypatch.setattr(solver, "solve", lambda *a, **k: base)
    monkeypatch.setattr(solver, "_compute_effective_ips", lambda ips, n_e, t_k: ips)
    monkeypatch.setattr(
        solver,
        "_apply_saha_correction",
        lambda obs_by_element, t_k, n_e, effective_ips: obs_by_element,
    )
    monkeypatch.setattr(
        solver,
        "_evaluate_partition_functions",
        lambda elements, t_k: (
            {el: 1.0 for el in elements},
            {el: 1.0 for el in elements},
        ),
    )
    monkeypatch.setattr(
        solver,
        "_build_uncertainty_abundance_multipliers",
        lambda *a, **k: {},
    )
    monkeypatch.setattr(
        solver,
        "_fit_common_boltzmann_plane",
        lambda corrected_obs: object(),
    )
    monkeypatch.setattr(
        solver,
        "_build_intercept_covariances",
        lambda fit: ({}, {}, 0.0, None),
    )
    monkeypatch.setattr(
        solver,
        "_propagate_closure_uncertainty",
        lambda *a, **k: {
            "Fe": uncertainties.ufloat(0.95, 0.02),
            "Na": uncertainties.ufloat(0.05, 0.08),
        },
    )
    obs = [
        LineObservation(248.0, 100.0, 5.0, "Fe", 1, 3.0, 5.0, 1e8),
        LineObservation(589.0, 10.0, 5.0, "Na", 1, 2.0, 2.0, 1e8),
    ]
    return solver, obs


def test_solve_with_uncertainty_default_off_is_legacy(monkeypatch):
    solver, obs = _coupling_solver(monkeypatch)
    monkeypatch.delenv("CFLIBS_RELIABILITY_FROM_UNCERTAINTY", raising=False)

    res = solver.solve_with_uncertainty(obs)

    assert res.per_element_reliability == {}
    assert res.quality_metrics["quality_flag"] == "good"
    assert res.quality_metrics["overall_reliable"] is True
    assert res.overall_reliable is True
    assert res.concentrations == {
        "Fe": pytest.approx(0.95),
        "Na": pytest.approx(0.05),
    }
    assert res.temperature_K == pytest.approx(9000.0)
    assert res.electron_density_cm3 == pytest.approx(2e17)


def test_solve_with_uncertainty_enabled_downgrades_only_reliability(monkeypatch):
    solver, obs = _coupling_solver(monkeypatch)
    monkeypatch.setenv("CFLIBS_RELIABILITY_FROM_UNCERTAINTY", "1")

    res = solver.solve_with_uncertainty(obs)

    assert res.per_element_reliability == {"Fe": "ok", "Na": "reject"}
    assert res.quality_metrics["quality_flag"] == "reject"
    assert res.quality_metrics["overall_reliable"] is False
    assert res.overall_reliable is False
    assert res.concentrations == {
        "Fe": pytest.approx(0.95),
        "Na": pytest.approx(0.05),
    }
    assert res.temperature_K == pytest.approx(9000.0)
    assert res.electron_density_cm3 == pytest.approx(2e17)
