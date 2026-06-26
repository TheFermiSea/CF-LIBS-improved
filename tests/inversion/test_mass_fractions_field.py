"""run_pipeline populates CFLIBSResult.mass_fractions (DED Gap 4 / step 1).

The peak-based solvers emit number/mole fractions in `concentrations`; the
pipeline now also exposes a mass-fraction view so consumers compare wt%
like-for-like (the metal-alloy known set has no oxygen -> clean conversion).
"""

from pathlib import Path

import pytest

from cflibs.inversion.solve.iterative import CFLIBSResult


def test_cflibsresult_has_mass_fractions_default():
    r = CFLIBSResult(
        temperature_K=10000.0,
        temperature_uncertainty_K=0.0,
        electron_density_cm3=1e17,
        concentrations={"Fe": 0.7, "Cu": 0.3},
        concentration_uncertainties={},
        iterations=1,
        converged=True,
    )
    assert hasattr(r, "mass_fractions") and r.mass_fractions == {}


@pytest.mark.requires_db
@pytest.mark.slow
def test_run_pipeline_populates_mass_fractions():
    db_path = None
    for c in (
        Path(__file__).resolve().parents[2] / "ASD_da" / "libs_production.db",
        Path(__file__).resolve().parents[2] / "libs_production.db",
    ):
        if c.exists():
            db_path = c
            break
    if db_path is None:
        pytest.skip("libs_production.db not available")

    from cflibs.atomic.database import AtomicDatabase
    from cflibs.benchmark.scoreboard import ensure_default_datasets, iter_datasets
    from cflibs.inversion.pipeline import build_pipeline_config, run_pipeline

    ensure_default_datasets()
    sid, wl, inten, truth = list(
        next(iter(iter_datasets(names=["supercam_labcal"]))).adapter_factory()
    )[0]
    els = sorted(set(truth.elements_present))
    pipe = build_pipeline_config(
        els, resolving_power=truth.resolving_power, overrides={"solver": "iterative"}
    )
    res, _ = run_pipeline(wl, inten, AtomicDatabase(str(db_path)), pipe)
    assert res.mass_fractions  # populated
    assert sum(res.mass_fractions.values()) == pytest.approx(1.0, rel=1e-6)
    # number != mass for a multi-element heavy/light mix
    assert res.mass_fractions != res.concentrations
