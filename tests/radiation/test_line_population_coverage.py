"""Regression tests for the F1 float-key population bug (bead z3cg).

The legacy ``SpectrumModel`` matched each transition to its upper-level
population through a dict keyed by ``round(E_k_ev, 8)`` from the *lines*
table, while the populations dict was keyed by the raw ``energy_ev`` from
the *energy_levels* table. The two tables encode the same physical level
with ~1e-7 eV differences, so ~98 % of lookups missed and the lines were
silently dropped (measured: 110/6127 Fe+Ca+Al+Ti lines populated; an Fe
370-410 nm run populated 2 of 412 lines). See
``docs/audit/2026-06-09-overhaul/01-forward-physics.md`` finding F1.

These tests run against the production atomic DB (the cross-table float
mismatch does not reproduce on the synthetic test fixture, whose two
tables share identical energies) and fail on the pre-fix behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cflibs.instrument.model import InstrumentModel
from cflibs.plasma.state import SingleZoneLTEPlasma

DB_PATH = Path(__file__).resolve().parent.parent.parent / "ASD_da" / "libs_production.db"

EV_TO_K = 11604.518


@pytest.fixture(scope="module")
def production_db():
    if not DB_PATH.exists():
        pytest.skip(f"production DB not present at {DB_PATH}")
    from cflibs.atomic.database import AtomicDatabase

    return AtomicDatabase(str(DB_PATH))


@pytest.mark.requires_db
def test_fe_lines_receive_nonzero_populations(production_db):
    """>95 % of queried Fe I/II lines must get non-zero n_upper.

    On the pre-fix float-key path this fraction was ~0.5 % (2/412 lines
    for Fe 370-410 nm), so this test fails loudly on a regression.
    """
    from cflibs.radiation.spectrum_model import SpectrumModel

    plasma = SingleZoneLTEPlasma(
        T_e=0.8 * EV_TO_K,
        n_e=1e17,
        species={"Fe": 1e16},
    )
    model = SpectrumModel(
        plasma=plasma,
        atomic_db=production_db,
        instrument=InstrumentModel(resolution_fwhm_nm=0.05),
        lambda_min=370.0,
        lambda_max=410.0,
        delta_lambda=0.01,
    )

    snapshot = production_db.snapshot(
        elements=["Fe"],
        wavelength_range=(370.0, 410.0),
        min_relative_intensity=10.0,
    )
    n_lines = int(np.asarray(snapshot.line_wavelengths_nm).shape[0])
    assert n_lines > 100  # the band carries a dense Fe forest

    species_states = model.solver.solve_species_states(plasma)
    n_upper = model._build_n_upper_per_line(snapshot, species_states, n_lines)

    populated_fraction = float((n_upper > 0).mean())
    assert populated_fraction > 0.95, (
        f"Only {populated_fraction:.1%} of {n_lines} Fe lines received a "
        "non-zero upper-level population — float-key population matching "
        "has regressed (audit F1, bead z3cg)"
    )


@pytest.mark.requires_db
def test_fe_spectrum_is_dense_not_sparse(production_db):
    """End-to-end compute_spectrum() must emit a dense Fe line forest.

    Pre-fix, the Fe 370-410 nm spectrum had only 84/4001 non-zero grid
    points because all but 2 lines were dropped.
    """
    from cflibs.radiation.spectrum_model import SpectrumModel

    plasma = SingleZoneLTEPlasma(
        T_e=0.8 * EV_TO_K,
        n_e=1e17,
        species={"Fe": 1e16},
    )
    model = SpectrumModel(
        plasma=plasma,
        atomic_db=production_db,
        instrument=InstrumentModel(resolution_fwhm_nm=0.05),
        lambda_min=370.0,
        lambda_max=410.0,
        delta_lambda=0.01,
    )
    wl, intensity = model.compute_spectrum()

    assert np.all(np.isfinite(intensity))
    nonzero_fraction = float((intensity > 0).mean())
    assert nonzero_fraction > 0.5, (
        f"Only {nonzero_fraction:.1%} of grid points are non-zero — the "
        "synthetic spectrum is missing most of the Fe line forest"
    )
