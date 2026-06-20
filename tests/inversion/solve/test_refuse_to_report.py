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
from cflibs.atomic.structures import PartitionFunction, Transition  # noqa: E402
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


# --------------------------------------------------------------------------- #
# M7 sub-lever b: resonance-line McWhirter delta_E (opt-in, default OFF)       #
# --------------------------------------------------------------------------- #


def _txn(el, sp, ek, aki, resonance=True):
    return Transition(
        element=el,
        ionization_stage=sp,
        wavelength_nm=400.0,
        A_ki=aki,
        E_k_ev=ek,
        E_i_ev=0.0 if resonance else 1.5,
        g_k=3,
        g_i=1,
        is_resonance=resonance,
    )


def _db_with_transitions(transitions_by_species):
    db = MagicMock(spec=AtomicDatabase)
    db.get_ionization_potential.return_value = 7.0
    db.get_transitions.side_effect = lambda el, sp=None, **k: transitions_by_species.get(
        (el, sp), []
    )
    return db


def test_resonance_delta_e_picks_strongest_resonance_max_over_species():
    """delta_E = max over species of the strongest (max A_ki) resonance line E_k."""
    db = _db_with_transitions(
        {
            # Fe I: strongest resonance (highest A_ki) at 4.99 eV, a weaker one at 3.3
            ("Fe", 1): [_txn("Fe", 1, 3.3, 1e7), _txn("Fe", 1, 4.99, 5e8)],
            # Na I: strongest resonance at 2.10 eV
            ("Na", 1): [_txn("Na", 1, 2.10, 6e7)],
        }
    )
    solver = IterativeCFLIBSSolver(db, max_iterations=3)
    obs = [
        LineObservation(373.0, 100.0, 1.0, "Fe", 1, 4.3, 5, 1e8),
        LineObservation(589.0, 100.0, 1.0, "Na", 1, 2.10, 2, 1e8),
    ]
    de = solver._mcwhirter_delta_e_resonance(obs)
    assert de == pytest.approx(4.99)  # Fe I strongest resonance, max over {Fe,Na}


def test_resonance_delta_e_ignores_non_resonance_lines():
    """Non-resonance (excited-lower) lines must not set delta_E."""
    db = _db_with_transitions(
        {("Ca", 1): [_txn("Ca", 1, 6.0, 9e9, resonance=False), _txn("Ca", 1, 2.93, 5e7)]}
    )
    solver = IterativeCFLIBSSolver(db, max_iterations=3)
    obs = [LineObservation(422.7, 100.0, 1.0, "Ca", 1, 2.93, 3, 1e8)]
    assert solver._mcwhirter_delta_e_resonance(obs) == pytest.approx(2.93)


def test_resonance_delta_e_none_when_no_resonance_lines():
    """No resonance lines -> None -> caller falls back to observation-derived delta_E."""
    db = _db_with_transitions({("Cu", 1): []})
    solver = IterativeCFLIBSSolver(db, max_iterations=3)
    obs = [LineObservation(324.0, 100.0, 1.0, "Cu", 1, 3.8, 4, 1e8)]
    assert solver._mcwhirter_delta_e_resonance(obs) is None


def test_resonance_de_flag_changes_only_lte_keys(mock_db, monkeypatch):
    """The resonance-delta_E flag must touch ONLY the McWhirter/LTE keys, never
    any other quality metric (non-regression on T/n_e/composition by isolation)."""
    db = _db_with_transitions({("Fe", 1): [_txn("Fe", 1, 4.99, 5e8)]})
    db.get_partition_coefficients.side_effect = mock_db.get_partition_coefficients.side_effect
    solver = IterativeCFLIBSSolver(db, max_iterations=3)
    solver._last_sa_result = None
    obs = [LineObservation(373.0, 100.0, 1.0, "Fe", 1, 2.0, 5, 1e8)]  # max(E_k)=2.0 != 4.99
    kw = dict(
        fit_r2=0.97,
        boltzmann_degenerate=False,
        closure_degenerate=False,
        ne_from_stark=True,
    )
    monkeypatch.delenv("CFLIBS_MCWHIRTER_RESONANCE_DE", raising=False)
    off = solver._assemble_quality_metrics(obs, 10000.0, 5e15, {"Fe": 1.0}, **kw)
    monkeypatch.setenv("CFLIBS_MCWHIRTER_RESONANCE_DE", "1")
    on = solver._assemble_quality_metrics(obs, 10000.0, 5e15, {"Fe": 1.0}, **kw)
    changed = {k for k in set(off) | set(on) if off.get(k) != on.get(k)}
    allowed = {
        "lte_n_e_required_cm3",
        "lte_n_e_ratio",
        "lte_mcwhirter_satisfied",
        "overall_reliable",
    }
    assert changed, "flag ON must change the McWhirter floor for this fixture"
    assert changed <= allowed, f"flag leaked into non-LTE keys: {changed - allowed}"
    # The resonance delta_E (4.99) exceeds max(E_k)=2.0 here -> stricter floor.
    assert on["lte_n_e_required_cm3"] > off["lte_n_e_required_cm3"]


def test_resonance_de_flag_off_is_legacy(mock_db, monkeypatch):
    """Flag OFF: delta_E is observation-derived (legacy max(E_k)); resonance
    lookup is not consulted."""
    db = _db_with_transitions({("Fe", 1): [_txn("Fe", 1, 4.99, 5e8)]})
    db.get_transitions.side_effect = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("get_transitions must NOT be called when flag is OFF")
    )
    db.get_partition_coefficients.side_effect = mock_db.get_partition_coefficients.side_effect
    solver = IterativeCFLIBSSolver(db, max_iterations=3)
    solver._last_sa_result = None
    obs = [LineObservation(373.0, 100.0, 1.0, "Fe", 1, 2.0, 5, 1e8)]
    monkeypatch.delenv("CFLIBS_MCWHIRTER_RESONANCE_DE", raising=False)
    # Must not raise (get_transitions never consulted with flag OFF).
    qm = solver._assemble_quality_metrics(
        obs,
        10000.0,
        5e15,
        {"Fe": 1.0},
        fit_r2=0.97,
        boltzmann_degenerate=False,
        closure_degenerate=False,
        ne_from_stark=True,
    )
    assert "lte_n_e_required_cm3" in qm


def test_resonance_delta_e_survives_malformed_transitions():
    """A pluggable backend returning malformed transitions (missing A_ki,
    non-numeric E_k_ev) must degrade to None, never raise (adversarial-verify
    finding: the resonance filter + max() were outside the try/except)."""
    import types

    bad_missing_aki = types.SimpleNamespace(is_resonance=True, E_k_ev=4.0)  # no A_ki
    bad_nonnumeric_ek = types.SimpleNamespace(is_resonance=True, A_ki=5e8, E_k_ev="oops")
    db = _db_with_transitions(
        {
            ("Fe", 1): [bad_missing_aki],
            ("Ca", 1): [bad_nonnumeric_ek],
        }
    )
    solver = IterativeCFLIBSSolver(db, max_iterations=3)
    obs = [
        LineObservation(373.0, 100.0, 1.0, "Fe", 1, 4.3, 5, 1e8),
        LineObservation(422.7, 100.0, 1.0, "Ca", 1, 2.9, 3, 1e8),
    ]
    # Must NOT raise; both species skipped -> None (legacy fallback).
    assert solver._mcwhirter_delta_e_resonance(obs) is None


def test_solve_survives_malformed_transitions_with_flag_on(mock_db, monkeypatch):
    """Full solve with the resonance flag ON and a malformed backend must not
    abort the solve (the defect propagated through _assemble_quality_metrics)."""
    import types

    monkeypatch.setenv("CFLIBS_MCWHIRTER_RESONANCE_DE", "1")
    mock_db.get_transitions.side_effect = lambda *a, **k: [
        types.SimpleNamespace(is_resonance=True, E_k_ev=4.0)  # missing A_ki
    ]
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=5)
    res = solver.solve(_clean_obs())  # must complete, not raise
    assert "lte_n_e_required_cm3" in res.quality_metrics
