"""M7 Lever 6 refuse-to-report tests (accuracy-first roadmap §4 / §6).

Covers the three sub-changes that wire the previously-dead Cristoforetti
quality assessment into a refuse-to-report gate:

1. ``QualityAssessor.assess`` is wired into ``_assemble_quality_metrics`` so
   every result carries ``quality_flag`` / ``saha_boltzmann_consistency`` /
   ``inter_element_t_std_frac`` on BOTH solve paths (key-set parity).
2. ``CFLIBSResult.overall_reliable`` = {McWhirter satisfied} AND
   {quality_flag >= acceptable}.
3. The CLI ``_trust_report`` gate refuses below-acceptable results ONLY when
   ``CFLIBS_REFUSE_TO_REPORT`` is set (default OFF == byte-identical legacy).

The reliability assessment is a pure annotation: it must NEVER alter
T/n_e/composition and must NEVER raise (defensive fallback to 'unknown').
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.atomic.structures import PartitionFunction  # noqa: E402
from cflibs.cli.main import _trust_report  # noqa: E402
from cflibs.inversion.solve.iterative import (  # noqa: E402
    CFLIBSResult,
    IterativeCFLIBSSolver,
    LineObservation,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture
def mock_db():
    """Minimal AtomicDatabase mock (mirrors test_lax_quality_parity)."""
    db = MagicMock(spec=AtomicDatabase)
    db.get_ionization_potential.return_value = 7.0  # eV
    coeffs_I = [3.2188, 0, 0, 0, 0]  # ln U = 3.2188 -> U = 25

    def _pf(el, sp):
        return PartitionFunction(
            element=el,
            ionization_stage=sp,
            coefficients=coeffs_I,
            t_min=1000,
            t_max=20000,
            source="test",
        )

    db.get_partition_coefficients.side_effect = _pf
    return db


def _clean_obs() -> list:
    """Balanced two-element fixture (max E_k = 5.0 eV)."""
    T_eV = 1.0
    obs = []
    for el in ("A", "B"):
        for E in (1.0, 2.0, 3.0, 4.0, 5.0):
            obs.append(LineObservation(500.0, np.exp(-E / T_eV + 10.0), 0.1, el, 1, E, 1, 1e8))
    return obs


# --------------------------------------------------------------------------- #
# Wiring: assess metrics present on both solve paths, never raises             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("use_lax", [False, True])
def test_assess_metrics_present_on_both_paths(mock_db, use_lax):
    if use_lax:
        pytest.importorskip("jax")
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10, use_lax_while_loop=use_lax)
    res = solver.solve(_clean_obs())
    for key in (
        "quality_flag",
        "overall_reliable",
        "saha_boltzmann_consistency",
        "inter_element_t_std_frac",
    ):
        assert key in res.quality_metrics, (use_lax, key)
    assert isinstance(res.overall_reliable, bool)
    # The field mirrors the metric value.
    assert res.overall_reliable == bool(res.quality_metrics["overall_reliable"])


def test_assess_reliability_never_raises(mock_db):
    """Defensive annotation: a mock/edge DB yields a graceful dict, no crash."""
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=5)
    out = solver._assess_reliability(_clean_obs(), 10000.0, 1e17, {"A": 0.5, "B": 0.5})
    assert set(out) == {
        "quality_flag",
        "saha_boltzmann_consistency",
        "inter_element_t_std_frac",
    }
    assert isinstance(out["quality_flag"], str)


# --------------------------------------------------------------------------- #
# overall_reliable logic: {McWhirter satisfied} AND {quality_flag>=acceptable} #
# --------------------------------------------------------------------------- #


def _assemble(solver, quality_flag, T_K, n_e):
    """Drive _assemble_quality_metrics with a forced quality_flag."""
    solver._last_sa_result = None
    return (
        solver._assemble_quality_metrics(
            _clean_obs(),
            T_K,
            n_e,
            {"A": 0.5, "B": 0.5},
            fit_r2=0.99,
            boltzmann_degenerate=False,
            closure_degenerate=False,
            ne_from_stark=True,
        ),
        quality_flag,
    )


@pytest.mark.parametrize(
    "quality_flag,T_K,n_e,expected",
    [
        ("good", 8000.0, 1e18, True),  # McWhirter ok + flag ok  -> reliable
        ("acceptable", 8000.0, 1e18, True),  # boundary flag still reliable
        ("poor", 8000.0, 1e18, False),  # flag too low           -> NOT
        ("unknown", 8000.0, 1e18, False),  # unknown flag         -> NOT
        ("good", 20000.0, 1e13, False),  # McWhirter FAILS        -> NOT
    ],
)
def test_overall_reliable_logic(mock_db, monkeypatch, quality_flag, T_K, n_e, expected):
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=5)
    monkeypatch.setattr(
        solver,
        "_assess_reliability",
        lambda *a, **k: {
            "quality_flag": quality_flag,
            "saha_boltzmann_consistency": 0.05,
            "inter_element_t_std_frac": 0.02,
        },
    )
    qm, _ = _assemble(solver, quality_flag, T_K, n_e)
    assert qm["quality_flag"] == quality_flag
    assert qm["overall_reliable"] is expected


# --------------------------------------------------------------------------- #
# CLI refuse-to-report gate (opt-in, default OFF == legacy)                    #
# --------------------------------------------------------------------------- #


def _mk_result(**kw):
    base = dict(
        temperature_K=10000.0,
        temperature_uncertainty_K=0.0,
        electron_density_cm3=1e17,
        concentrations={"Fe": 0.6, "Cu": 0.4},
        concentration_uncertainties={},
        iterations=5,
        converged=True,
        quality_metrics={
            "boltzmann_degenerate": 0.0,
            "closure_degenerate": 0.0,
            "quality_flag": "good",
            "lte_mcwhirter_satisfied": True,
        },
        overall_reliable=True,
    )
    base.update(kw)
    return CFLIBSResult(**base)


def test_clean_reliable_not_flagged(monkeypatch):
    monkeypatch.delenv("CFLIBS_REFUSE_TO_REPORT", raising=False)
    info, warns = _trust_report(_mk_result())
    assert not any("RESULT UNRELIABLE" in w for w in warns)
    assert any("Quality flag: good" in i for i in info)


def test_refuse_off_preserves_legacy(monkeypatch):
    """Flag OFF: a below-acceptable but converged/non-degenerate result is NOT
    marked unreliable (byte-identical legacy behaviour, non-regression)."""
    monkeypatch.delenv("CFLIBS_REFUSE_TO_REPORT", raising=False)
    res = _mk_result(
        overall_reliable=False,
        quality_metrics={
            "boltzmann_degenerate": 0.0,
            "closure_degenerate": 0.0,
            "quality_flag": "poor",
            "lte_mcwhirter_satisfied": False,
        },
    )
    _, warns = _trust_report(res)
    assert not any("RESULT UNRELIABLE" in w for w in warns)


def test_refuse_on_flags_unreliable(monkeypatch):
    monkeypatch.setenv("CFLIBS_REFUSE_TO_REPORT", "1")
    res = _mk_result(
        overall_reliable=False,
        quality_metrics={
            "boltzmann_degenerate": 0.0,
            "closure_degenerate": 0.0,
            "quality_flag": "poor",
            "lte_mcwhirter_satisfied": False,
        },
    )
    _, warns = _trust_report(res)
    assert any("RESULT UNRELIABLE" in w for w in warns)
    assert any("refuse-to-report gate" in w for w in warns)


def test_refuse_on_keeps_reliable_result(monkeypatch):
    """Flag ON but result is reliable -> NOT refused."""
    monkeypatch.setenv("CFLIBS_REFUSE_TO_REPORT", "1")
    _, warns = _trust_report(_mk_result(overall_reliable=True))
    assert not any("RESULT UNRELIABLE" in w for w in warns)


def test_hard_gate_fires_regardless_of_flag(monkeypatch):
    """Convergence failure marks unreliable even with the refuse gate OFF."""
    monkeypatch.delenv("CFLIBS_REFUSE_TO_REPORT", raising=False)
    _, warns = _trust_report(_mk_result(converged=False, overall_reliable=True))
    assert any("RESULT UNRELIABLE" in w for w in warns)
