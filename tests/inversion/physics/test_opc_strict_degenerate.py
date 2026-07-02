"""calibrate_opc x strict mode: typed failures become StandardRecovery(degenerate=True).

The calibration T-grid deliberately probes off-optimal temperatures where
closure collapse is expected; under strict mode the solver raises a typed
SolverFailure there. The OPC layer must represent that as the caller's explicit
degenerate contract (standard excluded by the conditioning gate) rather than
letting one bad standard abort the whole calibration.
"""

import pytest

from cflibs.inversion.common.strict import NonPhysicalResult
from cflibs.inversion.physics.opc import (
    Standard,
    StandardRecovery,
    _recover_or_degenerate,
    assess_conditioning,
    choose_optimal_temperature,
)


def _good_recovery(_T_K: float) -> StandardRecovery:
    return StandardRecovery(composition={"Fe": 90.0, "Cr": 10.0})


def _collapsing_recovery(T_K: float) -> StandardRecovery:
    if T_K < 9000.0:
        raise NonPhysicalResult("[closure_degenerate] keystone collapse")
    return StandardRecovery(composition={"Fe": 90.0, "Cr": 10.0})


def _always_failing(_T_K: float) -> StandardRecovery:
    raise NonPhysicalResult("[closure_degenerate] keystone collapse")


def test_wrapper_passes_through_success():
    std = Standard(name="ok", certified={"Fe": 90.0, "Cr": 10.0}, recover=_good_recovery)
    rec = _recover_or_degenerate(std, 10000.0)
    assert rec.composition and not rec.degenerate and rec.converged


def test_wrapper_converts_solver_failure_to_degenerate():
    std = Standard(name="bad", certified={"Fe": 90.0, "Cr": 10.0}, recover=_always_failing)
    rec = _recover_or_degenerate(std, 7000.0)
    assert rec.degenerate is True
    assert rec.converged is False
    assert rec.composition == {}


def test_wrapper_does_not_swallow_non_solver_errors():
    def broken(_T_K: float) -> StandardRecovery:
        raise RuntimeError("genuine bug")

    std = Standard(name="bug", certified={"Fe": 90.0}, recover=broken)
    with pytest.raises(RuntimeError):
        _recover_or_degenerate(std, 9000.0)


def test_choose_optimal_temperature_survives_partial_collapse():
    std = Standard(
        name="partial", certified={"Fe": 90.0, "Cr": 10.0}, recover=_collapsing_recovery
    )
    # 7000 K raises inside the sweep; the sweep must survive and pick a valid T.
    best = choose_optimal_temperature(std, t_grid=[7000.0, 10000.0, 12000.0])
    assert best in (10000.0, 12000.0)


def test_assess_conditioning_marks_failing_standard_degenerate():
    std = Standard(name="bad", certified={"Fe": 90.0, "Cr": 10.0}, recover=_always_failing)
    out = assess_conditioning(std)
    assert out.degenerate is True
    assert out.passed is False
