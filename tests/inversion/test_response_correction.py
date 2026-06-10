"""
Tests for the spectral-response E(lambda) correction hook (bead gzwd).

Covers (audit 02-F5 / 06-Q4):
- CSV/YAML loader sharing the instrument response-curve representation
- relative-vs-absolute convention (normalization to max E = 1)
- coverage validation: hard error listing coverage, edge-hold warning
- division correctness + uncertainty propagation
- identity (None) default is bit-identical
- inverse pairing with the forward model's InstrumentModel.apply_response
- Ar branching-ratio lamp-free stub raises NotImplementedError with citation
- synthetic end-to-end: a smooth E(lambda) corrupts Boltzmann intercepts;
  the correction recovers the uncorrupted intercept vector
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from cflibs.core.constants import KB_EV
from cflibs.inversion.preprocess.response_correction import (
    ResponseCurveCoverageError,
    SpectralResponseCorrection,
    apply_response_correction,
    derive_response_from_argon_branching_ratios,
    load_response_curve,
)

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoadResponseCurve:
    def test_csv_basic(self, tmp_path: Path):
        path = tmp_path / "resp.csv"
        path.write_text("# lamp calibration 2026-06\n300.0,0.5\n400.0,1.0\n500.0,0.8\n")
        curve = load_response_curve(path)
        assert curve.shape == (3, 2)
        np.testing.assert_allclose(curve[:, 0], [300.0, 400.0, 500.0])

    def test_csv_with_header_row(self, tmp_path: Path):
        path = tmp_path / "resp.csv"
        path.write_text("wavelength_nm,relative_efficiency\n300.0,0.5\n500.0,1.0\n")
        curve = load_response_curve(path)
        assert curve.shape == (2, 2)

    def test_csv_unsorted_is_sorted(self, tmp_path: Path):
        path = tmp_path / "resp.csv"
        path.write_text("500.0,0.8\n300.0,0.5\n400.0,1.0\n")
        curve = load_response_curve(path)
        assert np.all(np.diff(curve[:, 0]) > 0)

    def test_yaml(self, tmp_path: Path):
        path = tmp_path / "resp.yaml"
        path.write_text(
            "wavelength_nm: [300.0, 400.0, 500.0]\nrelative_efficiency: [0.5, 1.0, 0.8]\n"
        )
        curve = load_response_curve(path)
        assert curve.shape == (3, 2)
        np.testing.assert_allclose(curve[:, 1], [0.5, 1.0, 0.8])

    def test_yaml_length_mismatch_rejected(self, tmp_path: Path):
        path = tmp_path / "resp.yaml"
        path.write_text("wavelength_nm: [300.0, 400.0]\nrelative_efficiency: [0.5]\n")
        with pytest.raises(ValueError, match="equal length"):
            load_response_curve(path)

    def test_yaml_missing_key_rejected(self, tmp_path: Path):
        path = tmp_path / "resp.yaml"
        path.write_text("wavelength_nm: [300.0, 400.0]\n")
        with pytest.raises(ValueError, match="relative_efficiency"):
            load_response_curve(path)

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_response_curve(tmp_path / "nope.csv")

    def test_single_point_rejected(self, tmp_path: Path):
        path = tmp_path / "resp.csv"
        path.write_text("300.0,0.5\n")
        with pytest.raises(ValueError, match="at least 2 points"):
            load_response_curve(path)

    def test_nonpositive_efficiency_rejected(self, tmp_path: Path):
        path = tmp_path / "resp.csv"
        path.write_text("300.0,0.5\n400.0,0.0\n")
        with pytest.raises(ValueError, match="non-positive"):
            load_response_curve(path)

    def test_duplicate_wavelengths_rejected(self, tmp_path: Path):
        path = tmp_path / "resp.csv"
        path.write_text("300.0,0.5\n300.0,0.6\n400.0,1.0\n")
        with pytest.raises(ValueError, match="duplicate"):
            load_response_curve(path)


# ---------------------------------------------------------------------------
# Correction semantics
# ---------------------------------------------------------------------------


class TestSpectralResponseCorrection:
    def test_division_correctness(self):
        curve = np.array([[200.0, 0.5], [600.0, 1.0]])
        corr = SpectralResponseCorrection(curve)
        wl = np.array([200.0, 400.0, 600.0])
        intensity = np.array([1.0, 1.0, 1.0])
        expected_eff = np.array([0.5, 0.75, 1.0])
        np.testing.assert_allclose(corr.apply(wl, intensity), 1.0 / expected_eff)

    def test_relative_vs_absolute_convention(self):
        """An absolutely-calibrated curve (arbitrary scale) gives the same
        correction as its normalized relative shape: CF-LIBS needs only the
        relative response (closure cancels the absolute scale)."""
        wl = np.linspace(300.0, 500.0, 7)
        intensity = np.linspace(1.0, 2.0, 7)
        shape = np.array([[250.0, 0.4], [400.0, 1.0], [550.0, 0.7]])
        absolute = shape.copy()
        absolute[:, 1] *= 3.7e4  # e.g. W m^-2 nm^-1 per count
        rel = SpectralResponseCorrection(shape).apply(wl, intensity)
        abs_ = SpectralResponseCorrection(absolute).apply(wl, intensity)
        np.testing.assert_allclose(rel, abs_, rtol=1e-12)

    def test_uncertainty_propagation(self):
        """sigma_I and I are divided by the same E: relative uncertainty
        preserved (deterministic multiplicative correction)."""
        curve = np.array([[200.0, 0.5], [600.0, 1.0]])
        corr = SpectralResponseCorrection(curve)
        wl = np.array([250.0, 350.0, 550.0])
        intensity = np.array([10.0, 20.0, 30.0])
        sigma = np.array([1.0, 2.0, 3.0])
        c_int, c_sig = corr.apply(wl, intensity, sigma)
        np.testing.assert_allclose(c_sig / c_int, sigma / intensity, rtol=1e-12)
        eff = corr.efficiency(wl)
        np.testing.assert_allclose(c_sig, sigma / eff, rtol=1e-12)

    def test_flat_curve_is_noop(self):
        """A constant-efficiency curve normalizes to E = 1 -> no-op."""
        curve = np.array([[200.0, 0.42], [600.0, 0.42]])
        corr = SpectralResponseCorrection(curve)
        wl = np.linspace(250.0, 550.0, 11)
        intensity = np.random.default_rng(0).uniform(1.0, 5.0, 11)
        np.testing.assert_allclose(corr.apply(wl, intensity), intensity, rtol=1e-12)

    def test_inverse_of_forward_model_apply_response(self):
        """Forward model multiplies by E; the correction divides by the same
        curve representation -> round trip recovers the original spectrum."""
        from cflibs.instrument.model import InstrumentModel

        curve = np.array([[250.0, 0.3], [350.0, 0.8], [450.0, 1.0], [550.0, 0.6]])
        wl = np.linspace(260.0, 540.0, 200)
        original = 1.0 + np.sin(wl / 20.0) ** 2
        instrument = InstrumentModel(resolution_fwhm_nm=0.1, response_curve=curve)
        measured = instrument.apply_response(wl, original)
        recovered = SpectralResponseCorrection(curve).apply(wl, measured)
        np.testing.assert_allclose(recovered, original, rtol=1e-9)

    def test_identity_none_is_bit_identical(self):
        """Regression pin: with no curve configured the hook returns the SAME
        intensity object (default behaviour bit-identical)."""
        wl = np.linspace(300.0, 500.0, 10)
        intensity = np.random.default_rng(1).uniform(0.0, 1.0, 10)
        out = apply_response_correction(wl, intensity, None)
        assert out is intensity


# ---------------------------------------------------------------------------
# Coverage validation
# ---------------------------------------------------------------------------


class TestCoverage:
    CURVE = np.array([[300.0, 0.5], [500.0, 1.0]])

    def test_uncovered_spectrum_hard_error_lists_coverage(self):
        corr = SpectralResponseCorrection(self.CURVE, extrapolation_margin_nm=5.0)
        wl = np.linspace(250.0, 480.0, 50)  # 50 nm below coverage
        with pytest.raises(ResponseCurveCoverageError) as excinfo:
            corr.apply(wl, np.ones_like(wl))
        msg = str(excinfo.value)
        assert "[300.00, 500.00]" in msg  # curve coverage
        assert "[250.00, 480.00]" in msg  # spectrum range
        assert "50.00 nm below" in msg

    def test_uncovered_above_hard_error(self):
        corr = SpectralResponseCorrection(self.CURVE, extrapolation_margin_nm=5.0)
        wl = np.linspace(320.0, 560.0, 50)
        with pytest.raises(ResponseCurveCoverageError, match="60.00 nm above"):
            corr.apply(wl, np.ones_like(wl))

    def test_extrapolation_within_margin_warns_edge_hold(self, caplog):
        corr = SpectralResponseCorrection(self.CURVE, extrapolation_margin_nm=5.0)
        wl = np.linspace(297.0, 503.0, 100)  # 3 nm beyond each side: within margin
        with caplog.at_level(logging.WARNING):
            out = corr.apply(wl, np.ones_like(wl))
        assert any("extrapolated" in rec.message for rec in caplog.records)
        # Edge-hold: efficiency at 297 nm equals the tabulated 300 nm value.
        eff = corr.efficiency(np.array([297.0]))
        np.testing.assert_allclose(eff, [0.5])
        assert np.all(np.isfinite(out))

    def test_fully_covered_no_warning(self, caplog):
        corr = SpectralResponseCorrection(self.CURVE)
        wl = np.linspace(310.0, 490.0, 50)
        with caplog.at_level(logging.WARNING):
            corr.apply(wl, np.ones_like(wl))
        assert not any("extrapolat" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Lamp-free (Ar branching ratio) stub
# ---------------------------------------------------------------------------


def test_argon_branching_ratio_stub_raises_with_citation():
    wl = np.linspace(300.0, 900.0, 10)
    with pytest.raises(NotImplementedError) as excinfo:
        derive_response_from_argon_branching_ratios(wl, np.ones_like(wl))
    msg = str(excinfo.value)
    assert "C3JA50371B" in msg  # JAAS 29 (2014) 657-664
    assert "Whaling" in msg  # JQSRT 50 (1993) branching-ratio tables


# ---------------------------------------------------------------------------
# Synthetic end-to-end: Boltzmann-intercept recovery
# ---------------------------------------------------------------------------

TEST_T_K = 12000.0


def _build_atomic_db(tmp_path: Path):
    """Multi-stage Fe/Cu test DB (same construction as tests/test_pipeline_e2e.py)."""
    from cflibs.atomic.database import AtomicDatabase

    db_path = tmp_path / "response_e2e.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE lines (id INTEGER PRIMARY KEY, element TEXT, sp_num INTEGER,
        wavelength_nm REAL, aki REAL, ei_ev REAL, ek_ev REAL, gi INTEGER, gk INTEGER,
        rel_int REAL)""")
    conn.execute("""CREATE TABLE energy_levels (element TEXT, sp_num INTEGER, g_level INTEGER,
        energy_ev REAL)""")
    conn.execute("""CREATE TABLE species_physics (element TEXT, sp_num INTEGER, ip_ev REAL,
        atomic_mass REAL, PRIMARY KEY (element, sp_num))""")
    conn.execute("""CREATE TABLE partition_functions (element TEXT, sp_num INTEGER, a0 REAL,
        a1 REAL, a2 REAL, a3 REAL, a4 REAL, t_min REAL, t_max REAL, source TEXT,
        PRIMARY KEY (element, sp_num))""")
    line_rows = [
        ("Fe", 1, 371.99, 1.5e7, 0.0, 3.33, 9, 11, 1000),
        ("Fe", 1, 373.49, 8.0e6, 0.0, 3.32, 9, 9, 700),
        ("Fe", 1, 374.56, 4.0e6, 0.0, 3.10, 9, 7, 400),
        ("Fe", 1, 374.95, 3.0e6, 0.0, 3.31, 9, 7, 300),
        ("Fe", 2, 259.94, 2.0e7, 0.0, 4.77, 10, 8, 900),
        ("Fe", 2, 262.57, 1.2e7, 0.0, 5.10, 10, 6, 700),
        ("Cu", 1, 324.75, 2.2e7, 0.0, 3.82, 2, 4, 1200),
        ("Cu", 1, 327.40, 1.6e7, 0.0, 3.79, 2, 2, 1100),
        ("Cu", 1, 510.55, 6.0e6, 0.0, 6.19, 2, 4, 450),
        ("Cu", 1, 515.32, 3.2e6, 0.0, 6.08, 2, 6, 280),
    ]
    conn.executemany(
        """INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk,
        rel_int) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        line_rows,
    )
    level_rows = set()
    for element, stage, _, _, e_i, e_k, g_i, g_k, _ in line_rows:
        level_rows.add((element, stage, g_i, round(e_i, 8)))
        level_rows.add((element, stage, g_k, round(e_k, 8)))
    conn.executemany(
        "INSERT INTO energy_levels (element, sp_num, g_level, energy_ev) VALUES (?, ?, ?, ?)",
        sorted(level_rows),
    )
    conn.executemany(
        "INSERT INTO species_physics (element, sp_num, ip_ev, atomic_mass) VALUES (?, ?, ?, ?)",
        [
            ("Fe", 1, 7.902, 55.845),
            ("Fe", 2, 16.187, 55.845),
            ("Cu", 1, 7.726, 63.546),
            ("Cu", 2, 20.292, 63.546),
        ],
    )
    conn.executemany(
        """INSERT INTO partition_functions (element, sp_num, a0, a1, a2, a3, a4, t_min,
        t_max, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            ("Fe", 1, float(np.log(25.0)), 0.0, 0.0, 0.0, 0.0, 1000.0, 20000.0, "test"),
            ("Fe", 2, float(np.log(40.0)), 0.0, 0.0, 0.0, 0.0, 1000.0, 20000.0, "test"),
            ("Cu", 1, float(np.log(2.0)), 0.0, 0.0, 0.0, 0.0, 1000.0, 20000.0, "test"),
            ("Cu", 2, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 20000.0, "test"),
        ],
    )
    conn.commit()
    conn.close()
    return AtomicDatabase(str(db_path))


def _simulate_clean_spectrum(atomic_db) -> tuple[np.ndarray, np.ndarray]:
    """Optically-thin Fe/Cu spectrum from the (Wave-1 fixed) forward model."""
    from cflibs.instrument.model import InstrumentModel
    from cflibs.plasma.state import SingleZoneLTEPlasma
    from cflibs.radiation.spectrum_model import SpectrumModel

    plasma = SingleZoneLTEPlasma.from_mass_fractions(
        T_e=TEST_T_K,
        n_e=1.0e17,
        mass_fractions={"Fe": 0.6, "Cu": 0.4},
        total_species_density_cm3=3.0e16,
        atomic_masses_amu={"Fe": 55.845, "Cu": 63.546},
    )
    model = SpectrumModel(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=InstrumentModel(resolution_fwhm_nm=0.05),
        lambda_min=258.0,
        lambda_max=520.0,
        delta_lambda=0.01,
        path_length_m=1e-6,  # optically thin
    )
    wavelength, intensity = model.compute_spectrum()
    rng = np.random.default_rng(123)
    # Keep baseline + noise well below min_peak_height even after the
    # response ramp rescales the spectrum: multiplying by E(lambda) lowers
    # the global maximum (E ~ 0.6 at the strongest line) while barely
    # touching the red end (E ~ 0.95), so baseline fluctuations gain ~1.6x
    # in *relative* height on the corrupted spectrum. With 2e-4 baseline +
    # 5e-5 sigma they stay below the 0.002 detection threshold in every
    # variant, keeping the detected peak set comparable.
    baseline = 2e-4 * intensity.max()
    noisy = np.maximum(
        intensity + baseline + rng.normal(0.0, 5e-5 * intensity.max(), size=intensity.shape),
        0.0,
    )
    return wavelength, noisy


#: Smooth instrument response: linear ramp 0.3 -> 1.0 over 250-530 nm,
#: tabulated coarsely so interpolation is exercised. ln E spans ~1.2
#: ln-units across the Fe II (260 nm) vs Cu I (515 nm) line set -- the
#: same order as the entire intercept spread the closure measures.
RESPONSE_CURVE = np.column_stack(
    [
        np.arange(250.0, 540.0, 10.0),
        0.3 + 0.7 * (np.arange(250.0, 540.0, 10.0) - 250.0) / (530.0 - 250.0),
    ]
)


def _detect(wavelength, intensity, atomic_db):
    from cflibs.inversion.line_detection import detect_line_observations

    detection = detect_line_observations(
        wavelength,
        intensity,
        atomic_db,
        elements=["Fe", "Cu"],
        wavelength_tolerance_nm=0.08,
        min_peak_height=0.002,
        peak_width_nm=0.10,
        min_relative_intensity=50.0,
        kdet_enabled=False,
    )
    return {
        (obs.element, obs.ionization_stage, round(obs.wavelength_nm, 2)): obs
        for obs in detection.observations
    }


def _intercept_vector(obs_by_key, keys, temperature_K: float) -> dict:
    """Per-species Boltzmann intercept at the known plasma temperature.

    q_s = mean over the species' lines of [ln(I * lambda / (g_k * A_ki)) +
    E_k / (kB T)]. Evaluating at the TRUE temperature isolates the
    wavelength-dependent response bias (delta q = ln E) from slope-fit noise
    -- the sharpest version of the bead's "compare intercept vector, not
    concentrations" criterion.
    """
    species: dict = {}
    for key in keys:
        obs = obs_by_key[key]
        ordinate = np.log(obs.intensity * obs.wavelength_nm / (obs.g_k * obs.A_ki)) + obs.E_k_ev / (
            KB_EV * temperature_K
        )
        species.setdefault((obs.element, obs.ionization_stage), []).append(ordinate)
    return {sp: float(np.mean(vals)) for sp, vals in species.items()}


@pytest.mark.integration
def test_response_correction_recovers_boltzmann_intercepts(tmp_path: Path):
    """Corrupt a forward-model spectrum with a known smooth E(lambda); the
    correction must recover the uncorrupted per-species Boltzmann intercepts
    while the uncorrected run shows the predicted ln E(lambda) bias."""
    from cflibs.instrument.model import InstrumentModel

    atomic_db = _build_atomic_db(tmp_path)
    wavelength, clean = _simulate_clean_spectrum(atomic_db)

    # "Measured" spectrum: forward-model response application (multiply by E).
    instrument = InstrumentModel(resolution_fwhm_nm=0.05, response_curve=RESPONSE_CURVE)
    corrupted = instrument.apply_response(wavelength, clean)

    # Inversion-side correction (divide by the same curve).
    correction = SpectralResponseCorrection(RESPONSE_CURVE)
    corrected = correction.apply(wavelength, corrupted)

    obs_clean = _detect(wavelength, clean, atomic_db)
    obs_corrupted = _detect(wavelength, corrupted, atomic_db)
    obs_corrected = _detect(wavelength, corrected, atomic_db)

    # Compare on the lines every variant detected, so selection differences
    # cannot leak into the intercept comparison.
    common = set(obs_clean) & set(obs_corrupted) & set(obs_corrected)
    species_seen = {(el, sp) for el, sp, _ in common}
    assert {
        ("Fe", 1),
        ("Fe", 2),
        ("Cu", 1),
    } <= species_seen, f"Expected all three species among common detections; got {species_seen}"

    q_clean = _intercept_vector(obs_clean, common, TEST_T_K)
    q_corrupted = _intercept_vector(obs_corrupted, common, TEST_T_K)
    q_corrected = _intercept_vector(obs_corrected, common, TEST_T_K)

    # (a) The corruption has teeth: Fe II lines sit at ~260 nm where
    # E ~ 0.33, so the uncorrected intercept is biased by ~ln 0.33 ~ -1.1.
    fe2_bias = q_corrupted[("Fe", 2)] - q_clean[("Fe", 2)]
    assert fe2_bias < -0.5, f"Fe II intercept bias too small to test: {fe2_bias:.3f}"
    # Differential bias between species (what maps into concentrations).
    fe1_bias = q_corrupted[("Fe", 1)] - q_clean[("Fe", 1)]
    assert abs(fe2_bias - fe1_bias) > 0.3

    # (b) The correction recovers the uncorrupted intercept vector.
    for sp in q_clean:
        residual = q_corrected[sp] - q_clean[sp]
        assert abs(residual) < 0.05, (
            f"Corrected intercept for {sp} off by {residual:.4f} ln-units "
            f"(clean {q_clean[sp]:.4f}, corrected {q_corrected[sp]:.4f})"
        )


@pytest.mark.integration
def test_uncorrected_bias_matches_ln_e_prediction(tmp_path: Path):
    """The per-species intercept bias of the UNCORRECTED run equals the mean
    ln E(lambda) over that species' lines (delta q = ln E, audit 02-F5)."""
    from cflibs.instrument.model import InstrumentModel

    atomic_db = _build_atomic_db(tmp_path)
    wavelength, clean = _simulate_clean_spectrum(atomic_db)
    instrument = InstrumentModel(resolution_fwhm_nm=0.05, response_curve=RESPONSE_CURVE)
    corrupted = instrument.apply_response(wavelength, clean)

    obs_clean = _detect(wavelength, clean, atomic_db)
    obs_corrupted = _detect(wavelength, corrupted, atomic_db)
    common = set(obs_clean) & set(obs_corrupted)

    correction = SpectralResponseCorrection(RESPONSE_CURVE)
    q_clean = _intercept_vector(obs_clean, common, TEST_T_K)
    q_corrupted = _intercept_vector(obs_corrupted, common, TEST_T_K)

    ln_e: dict = {}
    for el, sp, wl in common:
        ln_e.setdefault((el, sp), []).append(
            float(np.log(correction.efficiency(np.array([wl]))[0]))
        )

    for species in q_clean:
        predicted = float(np.mean(ln_e[species]))
        measured = q_corrupted[species] - q_clean[species]
        assert measured == pytest.approx(
            predicted, abs=0.1
        ), f"{species}: measured bias {measured:.3f} vs predicted ln E {predicted:.3f}"
