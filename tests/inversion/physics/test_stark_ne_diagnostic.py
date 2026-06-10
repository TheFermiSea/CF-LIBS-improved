"""Tests for the Stark-broadening n_e diagnostic (bead pxex, audit 02-F2).

Covers, with INDEPENDENT oracles (hand-computed widths/densities, never the
production code under test):

1. Width -> n_e conversion against a hand-computed value using the DB
   convention (stark_w = electron-impact FWHM at n_e = 1e17 cm^-3,
   T = 10000 K; linear in n_e; (T/T_ref)^-alpha temperature factor).
2. The pinned-Gaussian Voigt fit recovers a known Lorentzian FWHM.
3. Synthetic round-trip: a Wave-1-fixed forward-model spectrum
   (``apply_stark=True``) at known n_e is inverted by ``measure_stark_ne``
   to within 30%.
4. Literature-provenance gating: heuristic (lambda^2-scaled) widths are
   refused, so the solver keeps the (warned) pressure-balance fallback.
5. Multi-line combination in the solver: median + MAD scatter,
   ``stark_n_lines`` / ``stark_ne_scatter_cm3`` quality metrics, and the
   fallback warning is preserved when no diagnostic is available.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import PartitionFunction
from cflibs.inversion.common.data_structures import LineObservation
from cflibs.inversion.physics.stark_ne import (
    LITERATURE_STARK_SOURCES,
    estimate_instrument_fwhm,
    measure_stark_ne,
)
from cflibs.inversion.physics.stark_ne import _fit_lorentz_fwhm  # unit-level
from cflibs.inversion.solve.iterative import (
    IterativeCFLIBSSolver,
    StarkDiagnosticLine,
)
from cflibs.radiation.profiles import voigt_profile
from cflibs.radiation.stark import REF_NE, REF_T_K, estimate_ne_from_stark

# --------------------------------------------------------------------------- #
# Independent oracles                                                          #
# --------------------------------------------------------------------------- #

_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _stark_fwhm_oracle(n_e_cm3: float, T_K: float, w_ref: float, alpha: float) -> float:
    """Forward electron-impact Stark FWHM, re-derived independently."""
    return w_ref * (n_e_cm3 / 1.0e17) * (max(T_K, 1000.0) / 10000.0) ** (-alpha)


# --------------------------------------------------------------------------- #
# 1. Width -> n_e conversion (hand-computed, DB convention)                    #
# --------------------------------------------------------------------------- #


class TestWidthToNeConversion:
    def test_ca_ii_393_hand_computed(self):
        """Ca II 393.37: stored stark_b width 0.0084 nm (FWHM @ 1e17 / 10000 K).

        A measured Lorentzian FWHM of 0.0042 nm at T = 10000 K must invert to
        exactly half the reference density:
            n_e = 1e17 * (0.0042 / 0.0084) * (10000/10000)^0.05 = 5.0e16 cm^-3
        """
        ne = estimate_ne_from_stark(
            measured_fwhm_nm=0.0042, T_K=10000.0, stark_w_ref=0.0084, stark_alpha=0.05
        )
        assert ne == pytest.approx(5.0e16, rel=1e-9)

    def test_temperature_factor_hand_computed(self):
        """At T = 14500 K with alpha = 0.05 the correction is (1.45)^0.05.

        Hand-computed: n_e = 1e17 * (0.0084/0.0084) * 1.45^0.05
                           = 1e17 * exp(0.05 * ln 1.45) = 1.01875...e17
        """
        ne = estimate_ne_from_stark(
            measured_fwhm_nm=0.0084, T_K=14500.0, stark_w_ref=0.0084, stark_alpha=0.05
        )
        expected = 1.0e17 * 1.45**0.05
        assert ne == pytest.approx(expected, rel=1e-9)
        # Convention sanity anchors (module-level constants, not re-derived).
        assert REF_NE == pytest.approx(1.0e17)
        assert REF_T_K == pytest.approx(10000.0)


# --------------------------------------------------------------------------- #
# 2. Pinned-Gaussian Voigt fit recovers a known Lorentzian width               #
# --------------------------------------------------------------------------- #


class TestPinnedGaussianVoigtFit:
    def _synth(self, gamma_l_fwhm: float, gauss_fwhm: float, noise: float = 0.0):
        rng = np.random.default_rng(42)
        wl = np.arange(392.0, 395.0, 0.005)
        sigma = gauss_fwhm / _FWHM_PER_SIGMA
        inten = voigt_profile(wl, 393.37, sigma, gamma_l_fwhm / 2.0, amplitude=10.0)
        inten = np.asarray(inten) + 0.5  # constant baseline
        if noise > 0:
            inten = inten + rng.normal(0.0, noise * float(np.max(inten)), wl.shape)
        return wl, inten

    def test_recovers_lorentz_fwhm_within_5pct(self):
        gamma_true = 0.05  # Lorentzian FWHM (nm)
        gauss = 0.04
        wl, inten = self._synth(gamma_true, gauss)
        fit = _fit_lorentz_fwhm(wl, inten, 393.37, gauss, window_nm=0.6)
        assert fit is not None
        lorentz_fwhm, rel_rmse = fit
        assert lorentz_fwhm == pytest.approx(gamma_true, rel=0.05)
        assert rel_rmse < 0.05

    def test_recovers_under_noise_within_15pct(self):
        gamma_true = 0.05
        gauss = 0.04
        wl, inten = self._synth(gamma_true, gauss, noise=0.01)
        fit = _fit_lorentz_fwhm(wl, inten, 393.37, gauss, window_nm=0.6)
        assert fit is not None
        lorentz_fwhm, _ = fit
        assert lorentz_fwhm == pytest.approx(gamma_true, rel=0.15)

    def test_pure_gaussian_line_yields_near_zero_lorentzian(self):
        wl, inten = self._synth(0.0, 0.06)
        fit = _fit_lorentz_fwhm(wl, inten, 393.37, 0.06, window_nm=0.6)
        assert fit is not None
        lorentz_fwhm, _ = fit
        # No recoverable Stark component: below the 5%-of-Gaussian floor that
        # measure_stark_ne applies.
        assert lorentz_fwhm < 0.05 * 0.06


# --------------------------------------------------------------------------- #
# 3. Synthetic round-trip through the Wave-1-fixed forward model               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def stark_b_db_path():
    """Temp atomic DB with two isolated Fe I lines carrying stark_b widths."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE lines (
            id INTEGER PRIMARY KEY, element TEXT, sp_num INTEGER,
            wavelength_nm REAL, aki REAL, ei_ev REAL, ek_ev REAL,
            gi INTEGER, gk INTEGER, rel_int REAL,
            stark_w REAL, stark_alpha REAL, stark_w_source TEXT
        )
        """
    )
    conn.execute(
        "CREATE TABLE energy_levels (element TEXT, sp_num INTEGER,"
        " g_level INTEGER, energy_ev REAL)"
    )
    conn.execute(
        "CREATE TABLE species_physics (element TEXT, sp_num INTEGER, ip_ev REAL,"
        " PRIMARY KEY (element, sp_num))"
    )
    # Two isolated lines ~3 nm apart; stark_w large enough that the Lorentzian
    # dominates the 0.01 nm instrument FWHM at the test density.
    conn.executemany(
        """
        INSERT INTO lines
            (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk,
             rel_int, stark_w, stark_alpha, stark_w_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("Fe", 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000, 0.05, 0.5, "stark_b"),
            ("Fe", 1, 375.00, 1.0e7, 0.1, 3.40, 7, 9, 800, 0.04, 0.5, "stark_b"),
        ],
    )
    conn.execute(
        "INSERT INTO energy_levels (element, sp_num, g_level, energy_ev)"
        " VALUES ('Fe', 1, 9, 0.0), ('Fe', 1, 11, 3.33),"
        " ('Fe', 1, 7, 0.1), ('Fe', 1, 9, 3.40)"
    )
    conn.execute(
        "INSERT INTO species_physics (element, sp_num, ip_ev)"
        " VALUES ('Fe', 1, 7.87), ('Fe', 2, 16.18)"
    )
    conn.commit()
    conn.close()
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.mark.requires_jax
class TestSyntheticRoundTrip:
    def test_recovers_known_ne_within_30pct(self, stark_b_db_path):
        """Forward model at n_e = 2e17 (apply_stark=True) -> measure_stark_ne
        recovers the density within 30%."""
        from cflibs.instrument.model import InstrumentModel
        from cflibs.plasma.state import SingleZoneLTEPlasma
        from cflibs.radiation.profiles import BroadeningMode
        from cflibs.radiation.spectrum_model import SpectrumModel

        n_e_true = 2.0e17
        T_K = 10000.0  # (T/T_ref) factor == 1: tests the width law cleanly
        instrument_fwhm = 0.01

        db = AtomicDatabase(stark_b_db_path)
        plasma = SingleZoneLTEPlasma(T_e=T_K, n_e=n_e_true, species={"Fe": 1.0e15})
        instrument = InstrumentModel(resolution_fwhm_nm=instrument_fwhm)
        model = SpectrumModel(
            plasma=plasma,
            atomic_db=db,
            instrument=instrument,
            lambda_min=370.5,
            lambda_max=376.5,
            delta_lambda=0.002,
            path_length_m=1.0e-4,
            broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
            apply_stark=True,
        )
        wl, inten = model.compute_spectrum()
        assert float(np.max(inten)) > 0

        observations = [
            LineObservation(371.99, 1000.0, 10.0, "Fe", 1, 3.33, 11, 1.0e7),
            LineObservation(375.00, 800.0, 10.0, "Fe", 1, 3.40, 9, 1.0e7),
        ]
        result = measure_stark_ne(
            wl,
            inten,
            observations,
            db,
            instrument_fwhm_nm=instrument_fwhm,
            T_K=T_K,
        )
        assert result.usable, f"no usable diagnostic; rejections: {result.rejected}"
        assert result.n_lines == 2
        assert result.ne_median_cm3 == pytest.approx(n_e_true, rel=0.30)
        # Independent oracle: the per-line Lorentzian widths should be near
        # the forward law w_ref * (n_e/1e17) (T factor == 1 here).
        for m in result.measurements:
            expected_fwhm = _stark_fwhm_oracle(n_e_true, T_K, m.stark_w_ref_nm, m.stark_alpha)
            assert m.lorentz_fwhm_nm == pytest.approx(expected_fwhm, rel=0.30)

    def test_heuristic_widths_are_refused(self, stark_b_db_path):
        """Flipping the provenance to the lambda^2 heuristic must disqualify
        every line (Konjevic 2002 / Gigosos 2014: only critically-evaluated
        widths are measurement-grade)."""
        conn = sqlite3.connect(stark_b_db_path)
        conn.execute("UPDATE lines SET stark_w_source = 'konjevic_lambda_sq_scaled'")
        conn.commit()
        conn.close()

        from cflibs.instrument.model import InstrumentModel
        from cflibs.plasma.state import SingleZoneLTEPlasma
        from cflibs.radiation.profiles import BroadeningMode
        from cflibs.radiation.spectrum_model import SpectrumModel

        db = AtomicDatabase(stark_b_db_path)
        plasma = SingleZoneLTEPlasma(T_e=10000.0, n_e=2.0e17, species={"Fe": 1.0e15})
        model = SpectrumModel(
            plasma=plasma,
            atomic_db=db,
            instrument=InstrumentModel(resolution_fwhm_nm=0.01),
            lambda_min=370.5,
            lambda_max=376.5,
            delta_lambda=0.002,
            path_length_m=1.0e-4,
            broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
            apply_stark=True,
        )
        wl, inten = model.compute_spectrum()
        observations = [
            LineObservation(371.99, 1000.0, 10.0, "Fe", 1, 3.33, 11, 1.0e7),
            LineObservation(375.00, 800.0, 10.0, "Fe", 1, 3.40, 9, 1.0e7),
        ]
        result = measure_stark_ne(
            wl, inten, observations, db, instrument_fwhm_nm=0.01, T_K=10000.0
        )
        assert not result.usable
        assert result.rejected.get("not_literature_grade", 0) == 2
        assert "stark_b" in LITERATURE_STARK_SOURCES


# --------------------------------------------------------------------------- #
# 4. Instrument-FWHM floor estimator                                           #
# --------------------------------------------------------------------------- #


class TestInstrumentFwhmFloor:
    def test_floor_tracks_narrowest_lines(self):
        """Synthetic spectrum with several Gaussian lines of instrument width
        0.05 nm: the floor estimate must land near 0.05 nm."""
        wl = np.arange(300.0, 320.0, 0.005)
        sigma = 0.05 / _FWHM_PER_SIGMA
        inten = np.zeros_like(wl) + 0.1
        centers = [302.0, 305.0, 308.0, 311.0, 314.0, 317.0]
        for c in centers:
            inten += 5.0 * np.exp(-0.5 * ((wl - c) / sigma) ** 2)
        obs = [LineObservation(c, 100.0, 1.0, "Fe", 1, 3.0, 9, 1e7) for c in centers]
        floor = estimate_instrument_fwhm(wl, inten, obs)
        assert floor is not None
        assert floor == pytest.approx(0.05, rel=0.2)


# --------------------------------------------------------------------------- #
# 5. Solver: multi-line combination + fallback preservation                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_db():
    db = MagicMock(spec=AtomicDatabase)
    db.get_ionization_potential.return_value = 7.0
    coeffs = [3.2188, 0, 0, 0, 0]  # log(25)
    db.get_partition_coefficients.side_effect = lambda el, sp: PartitionFunction(
        element=el,
        ionization_stage=sp,
        coefficients=coeffs,
        t_min=1000,
        t_max=20000,
        source="test",
    )
    return db


def _boltzmann_obs():
    T_eV = 0.8617
    return [
        LineObservation(500.0, np.exp(-E / T_eV + 10.0), 0.1, "A", 1, E, 1, 1e8)
        for E in (1.0, 2.0, 3.0, 4.0, 5.0)
    ]


class TestSolverMultiLineStark:
    def test_median_combination_and_scatter_metrics(self, mock_db):
        """Three lines forged at 4e17/5e17/6e17 -> median 5e17; scatter is the
        independent 1.4826*MAD = 1.4826e17."""
        solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)
        diags = [
            StarkDiagnosticLine(
                measured_fwhm_nm=_stark_fwhm_oracle(ne, 10000.0, 0.1, 0.5),
                stark_w_ref_nm=0.1,
                stark_alpha=0.5,
            )
            for ne in (4.0e17, 5.0e17, 6.0e17)
        ]
        res = solver.solve(_boltzmann_obs(), stark_diagnostics=diags)
        assert res.quality_metrics["ne_from_stark"] == pytest.approx(1.0)
        assert res.quality_metrics["stark_n_lines"] == pytest.approx(3.0)
        # Solver damping (0.5 prev + 0.5 new from a 1e17 start) converges onto
        # the median within the ne tolerance; allow 15%.
        assert res.electron_density_cm3 == pytest.approx(5.0e17, rel=0.15)
        # Independent MAD oracle: median |x - 5e17| = 1e17 -> 1.4826e17.
        assert res.quality_metrics["stark_ne_scatter_cm3"] == pytest.approx(
            1.4826e17, rel=0.01
        )
        assert res.electron_density_uncertainty_cm3 == pytest.approx(1.4826e17, rel=0.01)

    def test_single_line_backward_compatible(self, mock_db):
        solver = IterativeCFLIBSSolver(mock_db, max_iterations=10)
        diag = StarkDiagnosticLine(
            measured_fwhm_nm=_stark_fwhm_oracle(5.0e17, 10000.0, 0.1, 0.5),
            stark_w_ref_nm=0.1,
            stark_alpha=0.5,
        )
        res = solver.solve(_boltzmann_obs(), stark_diagnostic=diag)
        assert res.quality_metrics["ne_from_stark"] == pytest.approx(1.0)
        assert res.quality_metrics["stark_n_lines"] == pytest.approx(1.0)
        assert res.quality_metrics["stark_ne_scatter_cm3"] == pytest.approx(0.0)

    def test_fallback_warning_preserved_without_diagnostics(self, mock_db, caplog):
        """No diagnostics (None or empty list) -> the 1-atm pressure-balance
        fallback fires with its warning, exactly as before this change."""
        solver = IterativeCFLIBSSolver(mock_db, max_iterations=3)
        with caplog.at_level(logging.WARNING):
            res = solver.solve(_boltzmann_obs(), stark_diagnostics=[])
        assert res.quality_metrics["ne_from_stark"] == pytest.approx(0.0)
        assert res.quality_metrics["stark_n_lines"] == pytest.approx(0.0)
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "pressure-balance fallback" in msgs or "pressure balance" in msgs
