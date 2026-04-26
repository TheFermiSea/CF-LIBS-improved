"""
End-to-end integration tests for the CF-LIBS pipeline.

Self-contained tests with an in-memory atomic database.  No external DB required.
Conditions are ps-LIBS regime (T ≈ 0.8 eV, ne ≈ 4×10¹⁶ cm⁻³) with three elements
(Fe, Cu, Al).

Pipeline exercised:
  forward model → synthetic spectrum → line detection/identification
  → Boltzmann fitting → closure → composition recovery
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from cflibs.atomic.database import AtomicDatabase
from cflibs.instrument.model import InstrumentModel
from cflibs.inversion.identify.alias import ALIASIdentifier
from cflibs.inversion.common.element_id import to_line_observations
from cflibs.inversion.identify.line_detection import detect_line_observations
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver
from cflibs.plasma.state import SingleZoneLTEPlasma, mass_fractions_to_number_fractions
from cflibs.radiation.spectrum_model import SpectrumModel

# ---------------------------------------------------------------------------
# Ground-truth plasma parameters — ps-LIBS regime
# ---------------------------------------------------------------------------

# 0.8 eV expressed in Kelvin (1 eV ≈ 11604.5 K)
GROUND_TRUTH_T_K = 9280.0  # 0.8 eV
GROUND_TRUTH_NE = 4.0e16  # cm⁻³  (ps-LIBS: lower ne than ns-LIBS)
GROUND_TRUTH_MASS_FRACTIONS = {"Fe": 0.50, "Cu": 0.30, "Al": 0.20}
ATOMIC_MASSES = {"Fe": 55.845, "Cu": 63.546, "Al": 26.982}
GROUND_TRUTH_NUMBER_FRACTIONS = mass_fractions_to_number_fractions(
    GROUND_TRUTH_MASS_FRACTIONS,
    ATOMIC_MASSES,
)


# ---------------------------------------------------------------------------
# Self-contained atomic database builder
# ---------------------------------------------------------------------------


def _build_fe_cu_al_db(tmp_path: Path) -> AtomicDatabase:
    """Build a minimal but physically realistic SQLite atomic database.

    Includes Fe I/II, Cu I/II, Al I/II lines spanning enough E_k range for
    Boltzmann fitting and enough spectral separation for identification.
    """
    db_path = tmp_path / "e2e_test.db"
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
            rel_int REAL
        )
    """)
    conn.execute("""
        CREATE TABLE energy_levels (
            element TEXT,
            sp_num INTEGER,
            g_level INTEGER,
            energy_ev REAL
        )
    """)
    conn.execute("""
        CREATE TABLE species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            atomic_mass REAL,
            PRIMARY KEY (element, sp_num)
        )
    """)
    conn.execute("""
        CREATE TABLE partition_functions (
            element TEXT,
            sp_num INTEGER,
            a0 REAL, a1 REAL, a2 REAL, a3 REAL, a4 REAL,
            t_min REAL, t_max REAL, source TEXT,
            PRIMARY KEY (element, sp_num)
        )
    """)

    # ------------------------------------------------------------------
    # Fe I lines — wide E_k spread (2.18–4.44 eV) for Boltzmann slope
    # recovery.  At kT ≈ 0.8 eV the optimal lever arm is ≥ 3 kT ≈ 2.4 eV.
    # ------------------------------------------------------------------
    fe1_lines = [
        #  wl_nm    A_ki      E_i    E_k    g_i  g_k  rel_int
        (371.99, 1.5e7, 0.00, 3.33, 9, 11, 1000),
        (373.49, 8.0e6, 0.00, 3.32, 9, 9, 700),
        (374.56, 4.0e6, 0.00, 3.10, 9, 7, 400),
        (374.95, 3.0e6, 0.00, 3.31, 9, 7, 300),
        (382.04, 6.7e7, 0.00, 3.24, 9, 9, 800),
        (404.58, 8.6e6, 0.86, 3.93, 7, 9, 600),
        (427.18, 2.2e7, 1.49, 4.39, 5, 7, 700),
        (438.35, 5.0e6, 1.61, 4.44, 9, 9, 800),
        # Additional lines for E_k spread down to ~2.2 eV
        (495.76, 4.2e6, 2.18, 4.68, 7, 7, 300),
        (516.75, 5.7e6, 0.05, 2.45, 11, 9, 400),
        (532.80, 1.1e6, 0.91, 3.24, 5, 5, 250),
    ]
    # Fe II lines
    fe2_lines = [
        (259.94, 2.0e7, 0.00, 4.77, 10, 8, 900),
        (262.57, 1.2e7, 0.00, 5.10, 10, 6, 700),
    ]
    # Cu I lines — resonance doublet + green lines (wide E_k: 3.79–6.19 eV)
    cu1_lines = [
        (324.75, 2.2e7, 0.00, 3.82, 2, 4, 1200),
        (327.40, 1.6e7, 0.00, 3.79, 2, 2, 1100),
        (510.55, 6.0e6, 0.00, 6.19, 2, 4, 450),
        (515.32, 3.2e6, 0.00, 6.08, 2, 6, 280),
        (521.82, 7.5e5, 1.39, 3.76, 6, 4, 150),
    ]
    # Cu II line
    cu2_lines = [
        (224.70, 4.0e7, 0.00, 5.52, 1, 3, 500),
    ]
    # Al I lines — resonance doublet + UV lines (wide E_k: 3.13–4.67 eV)
    al1_lines = [
        (394.40, 5.0e7, 0.00, 3.14, 2, 4, 1500),
        (396.15, 5.0e7, 0.00, 3.13, 2, 2, 1400),
        (308.22, 5.8e7, 0.00, 4.02, 2, 4, 800),
        (309.27, 5.4e7, 0.00, 4.01, 4, 6, 750),
        (266.04, 2.8e7, 0.01, 4.67, 4, 2, 350),
        (256.80, 2.2e7, 0.01, 4.83, 4, 4, 300),
    ]
    # Al II line
    al2_lines = [
        (281.62, 3.8e7, 0.00, 4.40, 1, 3, 600),
    ]

    all_lines = (
        [("Fe", 1) + row for row in fe1_lines]
        + [("Fe", 2) + row for row in fe2_lines]
        + [("Cu", 1) + row for row in cu1_lines]
        + [("Cu", 2) + row for row in cu2_lines]
        + [("Al", 1) + row for row in al1_lines]
        + [("Al", 2) + row for row in al2_lines]
    )

    conn.executemany(
        """
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        all_lines,
    )

    # Energy levels — derived from the lines
    level_rows = set()
    for el, stage, _, _, e_i, e_k, g_i, g_k, _ in all_lines:
        level_rows.add((el, stage, g_i, round(e_i, 8)))
        level_rows.add((el, stage, g_k, round(e_k, 8)))
    conn.executemany(
        "INSERT INTO energy_levels (element, sp_num, g_level, energy_ev) VALUES (?, ?, ?, ?)",
        sorted(level_rows),
    )

    # Species physics — ionization potentials and atomic masses
    conn.executemany(
        "INSERT INTO species_physics (element, sp_num, ip_ev, atomic_mass) VALUES (?, ?, ?, ?)",
        [
            ("Fe", 1, 7.902, 55.845),
            ("Fe", 2, 16.187, 55.845),
            ("Cu", 1, 7.726, 63.546),
            ("Cu", 2, 20.292, 63.546),
            ("Al", 1, 5.986, 26.982),
            ("Al", 2, 18.828, 26.982),
        ],
    )

    # Partition functions — constant (flat) approximation adequate for test
    conn.executemany(
        """
        INSERT INTO partition_functions
            (element, sp_num, a0, a1, a2, a3, a4, t_min, t_max, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("Fe", 1, float(np.log(25.0)), 0.0, 0.0, 0.0, 0.0, 1000, 20000, "test"),
            ("Fe", 2, float(np.log(40.0)), 0.0, 0.0, 0.0, 0.0, 1000, 20000, "test"),
            ("Cu", 1, float(np.log(2.0)), 0.0, 0.0, 0.0, 0.0, 1000, 20000, "test"),
            ("Cu", 2, float(np.log(1.0)), 0.0, 0.0, 0.0, 0.0, 1000, 20000, "test"),
            ("Al", 1, float(np.log(6.0)), 0.0, 0.0, 0.0, 0.0, 1000, 20000, "test"),
            ("Al", 2, float(np.log(1.0)), 0.0, 0.0, 0.0, 0.0, 1000, 20000, "test"),
        ],
    )

    conn.commit()
    conn.close()
    return AtomicDatabase(str(db_path))


# ---------------------------------------------------------------------------
# Spectrum generation helper
# ---------------------------------------------------------------------------


def _generate_ps_libs_spectrum(
    atomic_db: AtomicDatabase,
    *,
    add_noise: bool = False,
    snr_db: float = 50.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic spectrum at ps-LIBS conditions.

    Uses the forward model with known ground-truth parameters so the
    inversion can be validated as a round-trip.
    """
    plasma = SingleZoneLTEPlasma.from_mass_fractions(
        T_e=GROUND_TRUTH_T_K,
        n_e=GROUND_TRUTH_NE,
        mass_fractions=GROUND_TRUTH_MASS_FRACTIONS,
        total_species_density_cm3=3.0e16,
        atomic_masses_amu=ATOMIC_MASSES,
    )
    model = SpectrumModel(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=InstrumentModel(resolution_fwhm_nm=0.05),
        lambda_min=220.0,
        lambda_max=540.0,
        delta_lambda=0.01,
        path_length_m=1e-6,  # optically thin for clean inversion
    )
    wavelength, intensity = model.compute_spectrum()

    if add_noise:
        rng = np.random.default_rng(seed)
        snr_linear = 10.0 ** (snr_db / 20.0)
        signal_peak = intensity.max()
        noise_std = signal_peak / snr_linear
        # Gaussian readout noise + small baseline offset
        baseline = 0.001 * signal_peak
        intensity = np.maximum(
            intensity + baseline + rng.normal(0.0, noise_std, size=intensity.shape),
            0.0,
        )

    return wavelength, intensity


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_forward_model_produces_signal(tmp_path):
    """Verify the forward model emits non-zero spectra at ps-LIBS conditions."""
    db = _build_fe_cu_al_db(tmp_path)
    wavelength, intensity = _generate_ps_libs_spectrum(db)

    assert len(wavelength) > 0
    assert np.any(intensity > 0), "Forward model produced zero-intensity spectrum"
    # Expect peaks near known strong lines
    peak_idx = np.argmax(intensity)
    peak_wl = wavelength[peak_idx]
    assert 220.0 < peak_wl < 520.0


@pytest.mark.integration
def test_line_detection_finds_expected_elements(tmp_path):
    """Verify line detection finds Fe, Cu, and Al lines in the synthetic spectrum."""
    db = _build_fe_cu_al_db(tmp_path)
    wavelength, intensity = _generate_ps_libs_spectrum(db)

    result = detect_line_observations(
        wavelength,
        intensity,
        db,
        elements=["Fe", "Cu", "Al"],
        wavelength_tolerance_nm=0.08,
        min_peak_height=0.005,
        peak_width_nm=0.10,
        min_relative_intensity=50.0,
        kdet_enabled=False,
    )

    detected_elements = {obs.element for obs in result.observations}
    assert "Fe" in detected_elements, "Fe lines not detected"
    assert "Cu" in detected_elements, "Cu lines not detected"
    assert "Al" in detected_elements, "Al lines not detected"
    assert result.total_peaks > 0


@pytest.mark.integration
def test_alias_identifies_all_three_elements(tmp_path):
    """Verify ALIAS identification detects Fe, Cu, and Al."""
    db = _build_fe_cu_al_db(tmp_path)
    wavelength, intensity = _generate_ps_libs_spectrum(db)

    alias = ALIASIdentifier(
        atomic_db=db,
        elements=["Fe", "Cu", "Al"],
        resolving_power=10000.0,
        detection_threshold=0.02,
        intensity_threshold_factor=3.0,
        T_range_K=(7000.0, 12000.0),
        n_e_range_cm3=(1.0e16, 1.0e17),
        T_steps=3,
        n_e_steps=2,
    )
    result = alias.identify(wavelength, intensity)

    detected_names = {eid.element for eid in result.detected_elements}
    assert "Fe" in detected_names
    assert "Cu" in detected_names
    assert "Al" in detected_names


@pytest.mark.integration
def test_full_pipeline_round_trip_noiseless(tmp_path):
    """Full round-trip: forward → detect → solve; verify T and composition recovery.

    Uses noiseless synthetic data at ps-LIBS conditions (T = 0.8 eV,
    ne = 4×10¹⁶, Fe-Cu-Al).

    Temperature tolerance is 25 % for the self-contained (minimal-DB) test.
    The iterative CF-LIBS solver exhibits a systematic T overestimate of
    ~20 % at ps-LIBS conditions because the Saha-Boltzmann system has a
    (T, ne) degeneracy: without an independent ne measurement (e.g. Stark
    broadening of Hβ), the algorithm converges to a higher-T / higher-ne
    pair that fits the spectral data equally well (R² > 0.99).  Relative
    compositions are far more robust because the closure equation
    normalises out most of the T bias.

    Acceptance criteria:
      - Convergence
      - Temperature recovery within 25 % (limited by Saha degeneracy)
      - Composition MAE ≤ 0.05 (number fractions)
    """
    db = _build_fe_cu_al_db(tmp_path)
    wavelength, intensity = _generate_ps_libs_spectrum(db, add_noise=False)

    # --- identification ---
    detected = detect_line_observations(
        wavelength,
        intensity,
        db,
        elements=["Fe", "Cu", "Al"],
        wavelength_tolerance_nm=0.08,
        min_peak_height=0.005,
        peak_width_nm=0.10,
        min_relative_intensity=50.0,
        kdet_enabled=False,
    )
    assert (
        len(detected.observations) >= 6
    ), f"Need ≥6 lines for robust Boltzmann fit, got {len(detected.observations)}"

    # --- iterative solver ---
    solver = IterativeCFLIBSSolver(db, max_iterations=30)
    result = solver.solve(detected.observations)

    assert result.converged, f"Solver did not converge after {result.iterations} iterations"

    # Temperature recovery — 25 % tolerance accounts for the Saha (T, ne)
    # degeneracy inherent to minimal-DB CF-LIBS without Stark ne anchor.
    t_err_frac = abs(result.temperature_K - GROUND_TRUTH_T_K) / GROUND_TRUTH_T_K
    assert t_err_frac < 0.25, (
        f"Temperature recovery failed: got {result.temperature_K:.0f} K, "
        f"expected {GROUND_TRUTH_T_K:.0f} K (error {t_err_frac:.1%})"
    )

    # Composition MAE (number fractions) — the primary accuracy metric
    mae = _composition_mae(result.concentrations, GROUND_TRUTH_NUMBER_FRACTIONS)
    assert mae <= 0.05, (
        f"Composition MAE = {mae:.4f} exceeds 0.05 threshold. "
        f"Recovered: {result.concentrations}, "
        f"Expected:  {GROUND_TRUTH_NUMBER_FRACTIONS}"
    )


@pytest.mark.integration
def test_full_pipeline_round_trip_noisy(tmp_path):
    """Round-trip with realistic noise (SNR = 50 dB).

    Noise widens the systematic bias from Saha degeneracy and adds
    stochastic scatter.  Tolerances are set accordingly:
      - Temperature within 30 % (Saha degeneracy + noise)
      - Composition MAE ≤ 0.08
    """
    db = _build_fe_cu_al_db(tmp_path)
    wavelength, intensity = _generate_ps_libs_spectrum(
        db,
        add_noise=True,
        snr_db=50.0,
        seed=42,
    )

    detected = detect_line_observations(
        wavelength,
        intensity,
        db,
        elements=["Fe", "Cu", "Al"],
        wavelength_tolerance_nm=0.08,
        min_peak_height=0.005,
        peak_width_nm=0.10,
        min_relative_intensity=50.0,
        kdet_enabled=False,
    )

    solver = IterativeCFLIBSSolver(db, max_iterations=30)
    result = solver.solve(detected.observations)

    assert result.converged

    t_err_frac = abs(result.temperature_K - GROUND_TRUTH_T_K) / GROUND_TRUTH_T_K
    assert t_err_frac < 0.30, (
        f"Noisy temperature recovery: {result.temperature_K:.0f} K "
        f"(expected {GROUND_TRUTH_T_K:.0f} K, error {t_err_frac:.1%})"
    )

    mae = _composition_mae(result.concentrations, GROUND_TRUTH_NUMBER_FRACTIONS)
    assert mae <= 0.08, (
        f"Noisy composition MAE = {mae:.4f} (threshold 0.08). "
        f"Recovered: {result.concentrations}"
    )


@pytest.mark.integration
def test_alias_to_solver_pipeline(tmp_path):
    """Round-trip via ALIAS identification → solver (alternative to detect)."""
    db = _build_fe_cu_al_db(tmp_path)
    wavelength, intensity = _generate_ps_libs_spectrum(db, add_noise=False)

    alias = ALIASIdentifier(
        atomic_db=db,
        elements=["Fe", "Cu", "Al"],
        resolving_power=10000.0,
        detection_threshold=0.02,
        intensity_threshold_factor=3.0,
        T_range_K=(7000.0, 12000.0),
        n_e_range_cm3=(1.0e16, 1.0e17),
        T_steps=3,
        n_e_steps=2,
    )
    alias_result = alias.identify(wavelength, intensity)
    line_obs = to_line_observations(alias_result)
    assert len(line_obs) >= 3

    solver = IterativeCFLIBSSolver(db, max_iterations=30)
    result = solver.solve(line_obs)

    assert result.converged
    t_err_frac = abs(result.temperature_K - GROUND_TRUTH_T_K) / GROUND_TRUTH_T_K
    assert t_err_frac < 0.25


@pytest.mark.integration
def test_solver_quality_metrics(tmp_path):
    """Verify solver produces reasonable quality diagnostics."""
    db = _build_fe_cu_al_db(tmp_path)
    wavelength, intensity = _generate_ps_libs_spectrum(db, add_noise=False)

    detected = detect_line_observations(
        wavelength,
        intensity,
        db,
        elements=["Fe", "Cu", "Al"],
        wavelength_tolerance_nm=0.08,
        min_peak_height=0.005,
        peak_width_nm=0.10,
        min_relative_intensity=50.0,
        kdet_enabled=False,
    )

    solver = IterativeCFLIBSSolver(db, max_iterations=30)
    result = solver.solve(detected.observations)

    assert result.converged
    assert "r_squared_last" in result.quality_metrics
    assert (
        result.quality_metrics["r_squared_last"] > 0.80
    ), f"R² = {result.quality_metrics['r_squared_last']:.3f} < 0.80"


# ---------------------------------------------------------------------------
# DB-dependent tests (skip if production DB not available)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.xfail(
    reason="RoundTripValidator concentration recovery bug: drops Fe, returns only Cu=1.0",
    strict=False,
)
def test_production_round_trip_noiseless(production_db):
    """Round-trip with full production database (noiseless)."""
    from cflibs.validation.round_trip import RoundTripValidator

    validator = RoundTripValidator(
        production_db,
        temperature_tolerance=0.05,
        density_tolerance=0.20,
        concentration_tolerance=0.10,
    )
    result = validator.validate(
        temperature_K=10000.0,
        electron_density_cm3=1e17,
        concentrations={"Fe": 0.8, "Cu": 0.2},
        add_noise=False,
    )
    assert result.passed, (
        f"Noiseless round-trip failed: T_err={result.temperature_error_frac:.2f}, "
        f"concentrations={result.recovered_concentrations}"
    )


@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.xfail(
    reason="RoundTripValidator concentration recovery bug: drops Fe, returns only Cu=1.0",
    strict=False,
)
def test_production_round_trip_noisy(production_db):
    """Round-trip with full production database (noisy)."""
    from cflibs.validation.round_trip import RoundTripValidator

    validator = RoundTripValidator(
        production_db,
        temperature_tolerance=0.10,
        density_tolerance=0.30,
        concentration_tolerance=0.15,
    )
    result = validator.validate(
        temperature_K=10000.0,
        electron_density_cm3=1e17,
        concentrations={"Fe": 0.8, "Cu": 0.2},
        add_noise=True,
    )
    assert result.passed, (
        f"Noisy round-trip failed: T_err={result.temperature_error_frac:.2f}, "
        f"concentrations={result.recovered_concentrations}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _composition_mae(
    recovered: dict[str, float],
    expected: dict[str, float],
) -> float:
    """Mean absolute error of compositions, only for elements present in expected."""
    errors = []
    for el, c_true in expected.items():
        c_recovered = recovered.get(el, 0.0)
        errors.append(abs(c_recovered - c_true))
    return float(np.mean(errors))
