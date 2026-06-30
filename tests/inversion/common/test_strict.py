"""Tests for the strict / no-fallback exploratory-mode foundation."""

import os

import numpy as np
import pytest

from cflibs.inversion.common import strict as S


def test_resolve_strict_precedence(monkeypatch):
    monkeypatch.delenv("CFLIBS_NO_FALLBACK", raising=False)
    assert S.resolve_strict(None) is False
    assert S.resolve_strict(True) is True
    assert S.resolve_strict(False) is False
    monkeypatch.setenv("CFLIBS_NO_FALLBACK", "1")
    assert S.resolve_strict(None) is True
    assert S.resolve_strict(False) is False  # explicit wins over env
    monkeypatch.setenv("CFLIBS_NO_FALLBACK", "off")
    assert S.resolve_strict(None) is False


def test_non_strict_records_without_raising():
    diag = S.SolveDiagnostics(solver="t", strict=False)
    g = S.require_simplex([0.5, 0.6], strict=False, diagnostics=diag)
    assert g.passed is False
    assert diag.failed_gates and diag.failure_reason is not None
    # round-trips
    assert diag.to_dict()["failure_reason"] is not None


def test_strict_raises_typed():
    with pytest.raises(S.NonPhysicalResult):
        S.require_simplex([0.5, 0.6], strict=True)
    with pytest.raises(S.MissingAtomicData):
        S.require_atomic_data("IP", None, "Na", strict=True)
    with pytest.raises(S.UnobservedStage):
        S.require_ion_stage_observed("Cu", 0, strict=True)
    with pytest.raises(S.NonIdentifiable):
        S.require_distinct_energy([2.0, 2.001], min_spread_ev=0.5, strict=True)


def test_positive_gate():
    assert S.require_positive([1.0, 2.0, 3.0], "intensity", strict=False).passed
    g = S.require_positive([1.0, -0.1, np.nan], "intensity", strict=False)
    assert not g.passed and g.values["n_bad"] == 2


def test_distinct_energy_pass_fail():
    assert S.require_distinct_energy([0.0, 3.5, 5.2], min_spread_ev=0.5, strict=False).passed
    assert not S.require_distinct_energy([2.0, 2.01], min_spread_ev=0.5, strict=False).passed
    # single line -> non-identifiable
    assert not S.require_distinct_energy([1.0], min_spread_ev=0.5, strict=False).passed


def test_simplex_pass():
    assert S.require_simplex([0.9, 0.06, 0.04], strict=False).passed


def test_atomic_data_gate():
    assert S.require_atomic_data("IP", 5.14, "Na", strict=False).passed
    assert not S.require_atomic_data("U", 0.0, "Fe", strict=False).passed
    assert not S.require_atomic_data("Aki", None, "Ti", strict=False).passed


def test_boltzmann_conditioning_degenerate_and_budget():
    # degenerate: all energies equal -> ss_e == 0 -> ill-conditioned (the R^2=1 masker)
    g = S.require_boltzmann_conditioning(
        [2.0, 2.0, 2.0], snr=50.0, target_rel_temp_err=0.05, temperature_k=9000.0, strict=False
    )
    assert not g.passed
    # wide spread + good SNR -> passes the energy-spread budget
    g = S.require_boltzmann_conditioning(
        np.linspace(0.0, 5.0, 8), snr=50.0, target_rel_temp_err=0.10, temperature_k=9000.0,
        strict=False,
    )
    assert g.passed


def test_diagnostics_accumulate():
    diag = S.SolveDiagnostics(solver="iter", strict=False)
    S.require_positive([1.0], "I", strict=False, diagnostics=diag)
    S.require_simplex([2.0], strict=False, diagnostics=diag)
    assert len(diag.gates) == 2
    assert len(diag.failed_gates) == 1
