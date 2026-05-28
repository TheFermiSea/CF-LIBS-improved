"""Regression test for Wave-1 Fix A2 (`docs/architecture/2026-05-27-physics-audit.md`).

Before the fix, :meth:`SpectrumModel.compute_spectrum` hardcoded
``apply_stark=False`` even though the parallel manifold and Bayesian
forward paths both defaulted to ``apply_stark=True``. As a result, any
identifier or basis built on SpectrumModel templates compared
Doppler-only synthetic spectra against Stark-broadened observed spectra
and systematically misidentified lines.

This test pins the new behaviour: the default ``compute_spectrum`` call
in ``PHYSICAL_DOPPLER`` mode with a non-zero per-line ``stark_w`` must
produce a spectrum with **wider** lines (larger FWHM) than the same call
with ``apply_stark=False``. The Voigt path adds a Lorentzian wing on top
of the Gaussian Doppler+instrument core, so the FWHM strictly increases.

Physics rationale: at n_e ~ 10^17 cm^-3 the Lorentzian Stark width is
comparable to or exceeds the Gaussian width (Aragón & Aguilera 2008,
*Spectrochim. Acta B* 63, 893).
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cflibs.atomic.database import AtomicDatabase
from cflibs.instrument.model import InstrumentModel
from cflibs.plasma.state import SingleZoneLTEPlasma
from cflibs.radiation.profiles import BroadeningMode
from cflibs.radiation.spectrum_model import SpectrumModel


def _fwhm_of_peak(wavelength: np.ndarray, intensity: np.ndarray) -> float:
    """Return the full-width-at-half-maximum of the largest peak (nm).

    Robust enough for a single isolated Fe I line on a smooth background:
    finds the global max, then walks outward to the first crossings of
    half-max. Returns the wavelength distance between those crossings.
    """
    if intensity.max() <= 0:
        return 0.0
    peak_idx = int(np.argmax(intensity))
    half = 0.5 * float(intensity[peak_idx])

    # Walk left
    left = peak_idx
    while left > 0 and intensity[left] > half:
        left -= 1
    # Walk right
    right = peak_idx
    while right < len(intensity) - 1 and intensity[right] > half:
        right += 1
    return float(wavelength[right] - wavelength[left])


@pytest.fixture
def stark_db_path() -> str:
    """Temp atomic DB with a single Fe I line carrying a non-zero ``stark_w``.

    A single isolated line makes the FWHM measurement unambiguous. The
    Fe I 371.99 nm line is used because it has well-characterised atomic
    data and sits in a quiet spectral window. The ``stark_w`` value is
    chosen large enough that the Voigt (Stark-on) FWHM is clearly wider
    than the Gaussian (Stark-off) FWHM at the test plasma's n_e.
    """
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
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
        """
    )
    conn.execute(
        """
        CREATE TABLE energy_levels (
            element TEXT,
            sp_num INTEGER,
            g_level INTEGER,
            energy_ev REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            PRIMARY KEY (element, sp_num)
        )
        """
    )
    # Single Fe I line at 371.99 nm with a deliberately large stark_w (in
    # nm, per cflibs/radiation/kernels.py:_per_line_stark_gamma). At
    # n_e=1e17 cm^-3 this gives gamma ~ 0.05 nm — large enough that the
    # Voigt vs Gaussian FWHM difference is unambiguous on a 0.005 nm grid.
    conn.execute(
        """
        INSERT INTO lines
            (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev,
             gi, gk, rel_int, stark_w, stark_alpha)
        VALUES ('Fe', 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000, 0.05, 0.5)
        """
    )
    conn.execute(
        """
        INSERT INTO energy_levels (element, sp_num, g_level, energy_ev)
        VALUES ('Fe', 1, 9, 0.0),
               ('Fe', 1, 11, 3.33)
        """
    )
    conn.execute(
        """
        INSERT INTO species_physics (element, sp_num, ip_ev)
        VALUES ('Fe', 1, 7.87),
               ('Fe', 2, 16.18)
        """
    )
    conn.commit()
    conn.close()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


@pytest.mark.requires_jax
def test_compute_spectrum_default_applies_stark_broadening(stark_db_path):
    """Default :meth:`SpectrumModel.compute_spectrum` must include Stark.

    Pins Wave-1 Fix A2 (``docs/architecture/2026-05-27-physics-audit.md``):
    the default call (no explicit ``apply_stark``) must produce a wider
    line profile than the same call with ``apply_stark=False``.
    """
    db = AtomicDatabase(stark_db_path)
    plasma = SingleZoneLTEPlasma(
        T_e=10000.0,
        n_e=1.0e17,  # high enough for Stark to dominate over Doppler
        species={"Fe": 1.0e15},
    )
    # Fixed-FWHM instrument with sub-Stark resolution so the broadening
    # difference is visible (instrument FWHM ~ 0.01 nm).
    instrument = InstrumentModel(resolution_fwhm_nm=0.01)
    common = dict(
        plasma=plasma,
        atomic_db=db,
        instrument=instrument,
        lambda_min=371.5,
        lambda_max=372.5,
        delta_lambda=0.005,
        # Optically thin (small L) — Planck-RT scales emissivity but
        # does not change the line shape, so the broadening comparison
        # is still apples-to-apples. ``path_length_m=0.0`` would zero
        # the output via ``B * (1 - exp(-kappa*L))``.
        path_length_m=1.0e-4,
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
    )

    # Default (apply_stark unspecified → should be True post Fix A2).
    model_default = SpectrumModel(**common)
    assert model_default.apply_stark is True, (
        "Wave-1 Fix A2 regression: SpectrumModel default apply_stark must be True"
    )
    wl_default, i_default = model_default.compute_spectrum()

    # Explicit Stark-off reference.
    model_no_stark = SpectrumModel(**common, apply_stark=False)
    wl_off, i_off = model_no_stark.compute_spectrum()

    np.testing.assert_array_equal(wl_default, wl_off)
    assert i_default.max() > 0
    assert i_off.max() > 0

    fwhm_default = _fwhm_of_peak(wl_default, i_default)
    fwhm_off = _fwhm_of_peak(wl_off, i_off)

    assert fwhm_default > fwhm_off, (
        "Default compute_spectrum (apply_stark=True) must produce a wider "
        f"line than apply_stark=False — got FWHM_default={fwhm_default:.5f} nm "
        f"<= FWHM_off={fwhm_off:.5f} nm. Wave-1 Fix A2 has regressed."
    )

    # Sanity: explicit apply_stark=True should match the default exactly
    # (i.e. default really did flow through to the kernel, not just to
    # the stored attribute).
    model_explicit_on = SpectrumModel(**common, apply_stark=True)
    _, i_on = model_explicit_on.compute_spectrum()
    np.testing.assert_allclose(i_default, i_on, rtol=1e-12, atol=0.0)
