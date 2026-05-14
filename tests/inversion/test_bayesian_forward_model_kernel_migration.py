"""Drift guard for the T1-6 BayesianForwardModel kernel migration.

ADR-0001 T1-6 (bead ``CF-LIBS-improved-789h``) migrated
:meth:`BayesianForwardModel._compute_spectrum` so it now dispatches to
:func:`cflibs.radiation.kernels.forward_model` directly instead of
re-implementing Saha-Boltzmann + Voigt summation locally and reaching
through :func:`_atomic_data_arrays_from_snapshot`.

These tests pin two invariants:

1. ``_compute_spectrum`` produces the same array as a hand-built
   :func:`forward_model` call with the same plasma state, snapshot,
   instrument, and broadening knobs (drift guard).
2. The legacy ``_atomic_data_arrays_from_snapshot`` adapter is no longer
   referenced from the ``forward.py`` source body -- only from
   re-exports and back-compat shims (regression guard).

Pre-migration bitwise parity is *not* asserted: the snapshot path uses
canonical Irwin (base-10 log) partition coefficients while the pre-T1-6
``_compute_spectrum`` interpreted the same coefficients as natural-log
Irwin. That convention change is the entire point of the migration; the
docstring on :meth:`BayesianForwardModel._compute_spectrum` notes the
absolute-scale change for posterity.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.requires_jax,
]

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal DB fixture (mirrors the one in tests/test_bayesian.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def bayesian_db():
    """Tiny SQLite DB with Fe/Cu lines + Irwin partition coefficients."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE lines (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            wavelength_nm REAL,
            aki REAL,
            ei_ev REAL,
            ek_ev REAL,
            gi INTEGER,
            gk INTEGER,
            rel_int REAL,
            stark_w REAL,
            stark_alpha REAL
        )
        """)
    conn.execute("""
        CREATE TABLE species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            PRIMARY KEY (element, sp_num)
        )
        """)
    conn.execute("""
        CREATE TABLE partition_functions (
            element TEXT,
            sp_num INTEGER,
            a0 REAL,
            a1 REAL,
            a2 REAL,
            a3 REAL,
            a4 REAL,
            t_min REAL,
            t_max REAL,
            source TEXT,
            PRIMARY KEY (element, sp_num)
        )
        """)
    # AtomicDatabase migration requires this table (auto-populates from
    # ``lines`` when empty).
    conn.execute("""
        CREATE TABLE energy_levels (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            g_level INTEGER,
            energy_ev REAL
        )
        """)
    lines_data = [
        ("Fe", 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000, 0.02, 0.5),
        ("Fe", 1, 373.49, 5.0e6, 0.0, 3.32, 9, 9, 500, 0.015, 0.5),
        ("Fe", 2, 238.20, 3.0e8, 0.0, 5.22, 10, 10, 600, 0.03, 0.6),
        ("Cu", 1, 324.75, 1.4e8, 0.0, 3.82, 2, 4, 2000, 0.01, 0.5),
        ("Cu", 1, 327.40, 1.4e8, 0.0, 3.79, 2, 2, 1000, 0.01, 0.5),
    ]
    conn.executemany(
        "INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, "
        "gi, gk, rel_int, stark_w, stark_alpha) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        lines_data,
    )
    conn.executemany(
        "INSERT INTO species_physics (element, sp_num, ip_ev) VALUES (?, ?, ?)",
        [
            ("Fe", 1, 7.87),
            ("Fe", 2, 16.18),
            ("Cu", 1, 7.73),
            ("Cu", 2, 20.29),
        ],
    )
    # Irwin (base-10 log) partition coefficients. The snapshot path uses
    # log10 convention -- a0 is the log10 partition function at log10(T)=0,
    # but coefficient parity is what matters here.
    conn.executemany(
        "INSERT INTO partition_functions (element, sp_num, a0, a1, a2, a3, a4) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("Fe", 1, 1.40, 0.0, 0.0, 0.0, 0.0),
            ("Fe", 2, 1.60, 0.0, 0.0, 0.0, 0.0),
            ("Cu", 1, 0.30, 0.0, 0.0, 0.0, 0.0),
            ("Cu", 2, 0.0, 0.0, 0.0, 0.0, 0.0),
        ],
    )
    conn.commit()
    conn.close()
    try:
        yield db_path
    finally:
        Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Drift guard: _compute_spectrum == forward_model(...)
# ---------------------------------------------------------------------------


def test_compute_spectrum_matches_direct_forward_model_call(bayesian_db):
    """``_compute_spectrum`` must equal a hand-built ``forward_model`` call.

    Pins the migration target: the body of ``_compute_spectrum`` is now a
    thin wrapper around :func:`cflibs.radiation.kernels.forward_model`. If
    a future refactor accidentally re-introduces a divergent local
    summation path, this assertion will fail.
    """
    from cflibs.core.constants import EV_TO_K
    from cflibs.inversion.solve.bayesian.forward import BayesianForwardModel
    from cflibs.plasma.state import SingleZoneLTEPlasma
    from cflibs.radiation.kernels import forward_model
    from cflibs.radiation.profiles import BroadeningMode

    elements = ["Fe", "Cu"]
    wl_range = (200.0, 600.0)
    pixels = 256

    model = BayesianForwardModel(
        db_path=bayesian_db,
        elements=elements,
        wavelength_range=wl_range,
        pixels=pixels,
        instrument_fwhm_nm=0.05,
    )

    T_eV = 1.0
    n_e = 1.0e17
    concentrations = jnp.array([0.7, 0.3])

    spectrum_via_method = model._compute_spectrum(T_eV, n_e, concentrations)

    # Mirror exactly what ``_compute_spectrum`` does internally so we are
    # asserting "no extra physics has been spliced in" rather than re-doing
    # the kernel arithmetic by hand.
    T_eV_jnp = jnp.asarray(T_eV, dtype=spectrum_via_method.dtype)
    n_e_jnp = jnp.asarray(n_e, dtype=spectrum_via_method.dtype)
    total_density = n_e_jnp
    plasma = object.__new__(SingleZoneLTEPlasma)
    plasma.T_e = T_eV_jnp * EV_TO_K
    plasma.n_e = n_e_jnp
    plasma.species = {el: concentrations[i] * total_density for i, el in enumerate(elements)}
    plasma.T_g = None
    plasma.pressure = None
    raw = forward_model(
        plasma,
        model.snapshot,
        model.instrument,
        model.wavelength,
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.0,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=True,
        total_species_density_cm3=total_density,
    )
    spectrum_via_kernel = jnp.clip(raw, 0.0, 1e12)

    np.testing.assert_allclose(
        np.asarray(spectrum_via_method),
        np.asarray(spectrum_via_kernel),
        rtol=1e-6,
        atol=1e-12,
    )


def test_compute_spectrum_is_finite_and_nonnegative(bayesian_db):
    """Spectrum must be finite and non-negative on a representative input."""
    from cflibs.inversion.solve.bayesian.forward import BayesianForwardModel

    model = BayesianForwardModel(
        db_path=bayesian_db,
        elements=["Fe", "Cu"],
        wavelength_range=(200.0, 600.0),
        pixels=256,
        instrument_fwhm_nm=0.05,
    )
    spectrum = model.forward(
        T_eV=1.0,
        log_ne=17.0,
        concentrations=jnp.array([0.7, 0.3]),
    )
    arr = np.asarray(spectrum)
    assert arr.shape == (256,)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0.0)
    # Lines fall inside the wavelength range, so some emission must be > 0.
    assert arr.max() > 0.0


def test_forward_py_body_does_not_call_adapter():
    """``forward.py`` body must no longer reference the adapter (regression).

    The adapter remains exported from
    :mod:`cflibs.inversion.solve.bayesian` for callers outside this code
    path, but :mod:`cflibs.inversion.solve.bayesian.forward` must not
    invoke it. We assert on the *body* (everything after the module
    docstring) to allow the docstring to keep the breadcrumb.
    """
    import cflibs.inversion.solve.bayesian.forward as fwd_mod

    source = Path(fwd_mod.__file__).read_text(encoding="utf-8")
    # Drop the module-level docstring (between the first two triple-quotes).
    first = source.find('"""')
    second = source.find('"""', first + 3)
    assert first != -1 and second != -1, "Could not locate module docstring"
    body = source[second + 3 :]
    assert "_atomic_data_arrays_from_snapshot(" not in body, (
        "BayesianForwardModel.forward.py must not call "
        "_atomic_data_arrays_from_snapshot after the T1-6 migration."
    )
