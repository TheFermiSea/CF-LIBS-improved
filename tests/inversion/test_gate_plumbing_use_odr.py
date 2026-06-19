"""Gate-plumbing verification for the Track B B1 ``use_odr`` flag.

These tests prove the flag is *reachable* end-to-end — ``config_overrides`` →
``build_pipeline_config`` → ``IterativeCFLIBSSolver`` → ``BoltzmannPlotFitter`` —
so a scoreboard run with ``config_overrides={"use_odr": True}`` actually exercises
the errors-in-variables fit and not a silent no-op. Without this, a benchmark gate
on the flag would compare flag-on against flag-on (false-null).
"""

from cflibs.inversion.pipeline import build_pipeline_config
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver


def test_default_use_odr_off():
    """Default pipeline keeps the standard weighted-OLS fit (no behaviour change)."""
    cfg = build_pipeline_config(["Fe"])
    assert cfg.use_odr is False
    assert cfg.odr_x_uncertainty == 0.0


def test_config_override_enables_use_odr():
    """``config_overrides`` (the scoreboard's gate knob) reaches the pipeline config."""
    cfg = build_pipeline_config(["Fe"], overrides={"use_odr": True, "odr_x_uncertainty": 0.3})
    assert cfg.use_odr is True
    assert cfg.odr_x_uncertainty == 0.3


def test_solver_threads_use_odr_to_fitter(atomic_db):
    """The solver forwards the flag all the way to the Boltzmann fitter instance."""
    default = IterativeCFLIBSSolver(atomic_db=atomic_db)
    assert default.boltzmann_fitter.use_odr is False

    enabled = IterativeCFLIBSSolver(atomic_db=atomic_db, use_odr=True, odr_x_uncertainty=0.3)
    assert enabled.boltzmann_fitter.use_odr is True
    assert enabled.boltzmann_fitter.odr_x_uncertainty == 0.3
