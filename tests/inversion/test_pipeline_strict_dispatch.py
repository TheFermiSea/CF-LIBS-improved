"""Pipeline-dispatch strict mode: the full-spectrum wrapper must surface failures.

Verifies the double-mask removal in ``_run_full_spectrum_solver``:
- strict OFF (default): unchanged production behaviour (non-adopted / crash -> warm start)
- strict ON: non-adoption -> NotConverged; solver crash -> OptimizerFailure
"""

import types

import numpy as np
import pytest

from cflibs.inversion import pipeline as P
from cflibs.inversion.common.strict import NotConverged, OptimizerFailure

import cflibs.inversion.solve.full_spectrum as FS


def _warm():
    return types.SimpleNamespace(
        temperature_K=9000.0,
        electron_density_cm3=1e17,
        concentrations={"Fe": 0.9, "Cr": 0.1},
        quality_metrics={},
        temperature_uncertainty_K=0.0,
        concentration_uncertainties={},
        overall_reliable=True,
    )


def _pipeline(strict=None):
    return types.SimpleNamespace(
        solver="joint",
        elements=["Fe", "Cr"],
        resolving_power=5000.0,
        strict=strict,
        apply_self_absorption="off",
    )


def _atomic_db():
    return types.SimpleNamespace(db_path="/tmp/does_not_matter.db")


def _fake_fs(adopted):
    return types.SimpleNamespace(
        adopted_fit=adopted, converged=True,
        warm_start_temperature_K=9000.0, warm_start_electron_density_cm3=1e17,
        fit_temperature_K=3200.0, fit_electron_density_cm3=1e15,
        initial_objective=1.0, final_objective=0.9, iterations=12, gradient_norm=1e-3,
        warm_start_concentrations={"Fe": 0.9, "Cr": 0.1},
        fit_concentrations={"Fe": 0.5, "Cr": 0.5},
        diagnostics={},
        temperature_K=3200.0, electron_density_cm3=1e15,
        concentrations={"Fe": 0.5, "Cr": 0.5},
    )


def _run(pipeline, monkeypatch, *, fs=None, raises=None):
    def fake_solve(*a, **k):
        if raises is not None:
            raise raises
        return fs
    monkeypatch.setattr(FS, "solve_full_spectrum", fake_solve)
    wl = np.linspace(250.0, 550.0, 32)
    return P._run_full_spectrum_solver(
        wl, np.ones_like(wl), _atomic_db(), pipeline, warm_start=_warm(), diagnostics={}
    )


def test_strict_off_non_adopted_returns_warm(monkeypatch):
    out = _run(_pipeline(strict=False), monkeypatch, fs=_fake_fs(adopted=False))
    assert out.temperature_K == 9000.0  # warm start unchanged


def test_strict_on_non_adopted_raises(monkeypatch):
    with pytest.raises(NotConverged):
        _run(_pipeline(strict=True), monkeypatch, fs=_fake_fs(adopted=False))


def test_strict_off_crash_returns_warm(monkeypatch):
    out = _run(_pipeline(strict=False), monkeypatch, raises=RuntimeError("boom"))
    assert out.temperature_K == 9000.0


def test_strict_on_crash_raises_optimizer_failure(monkeypatch):
    with pytest.raises(OptimizerFailure):
        _run(_pipeline(strict=True), monkeypatch, raises=RuntimeError("boom"))


def test_strict_on_adopted_returns_fit(monkeypatch):
    # adopted fit is returned regardless of strict (rewrapped)
    out = _run(_pipeline(strict=True), monkeypatch, fs=_fake_fs(adopted=True))
    assert out.temperature_K == 3200.0


def test_env_flag_enables_strict(monkeypatch):
    monkeypatch.setenv("CFLIBS_NO_FALLBACK", "1")
    with pytest.raises(NotConverged):
        _run(_pipeline(strict=None), monkeypatch, fs=_fake_fs(adopted=False))
