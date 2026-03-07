"""
End-to-end synthetic spectrum validation for the classic CF-LIBS pipeline.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from cflibs.atomic.database import AtomicDatabase
from cflibs.instrument.model import InstrumentModel
from cflibs.inversion.alias_identifier import ALIASIdentifier
from cflibs.inversion.element_id import to_line_observations
from cflibs.inversion.line_detection import detect_line_observations
from cflibs.inversion.solver import IterativeCFLIBSSolver
from cflibs.plasma.state import SingleZoneLTEPlasma, mass_fractions_to_number_fractions
from cflibs.radiation.spectrum_model import SpectrumModel

TEST_MASS_FRACTIONS = {"Fe": 0.6, "Cu": 0.4}
TEST_ATOMIC_MASSES = {"Fe": 55.845, "Cu": 63.546}
EXPECTED_NUMBER_FRACTIONS = mass_fractions_to_number_fractions(
    TEST_MASS_FRACTIONS,
    TEST_ATOMIC_MASSES,
)

EXPECTED_LINES = {
    ("Fe", 1, 371.99),
    ("Fe", 1, 373.49),
    ("Fe", 1, 374.56),
    ("Fe", 1, 374.95),
    ("Fe", 2, 259.94),
    ("Fe", 2, 262.57),
    ("Cu", 1, 324.75),
    ("Cu", 1, 327.40),
    ("Cu", 1, 510.55),
    ("Cu", 1, 515.32),
}


def _build_multistage_atomic_db(tmp_path: Path) -> AtomicDatabase:
    db_path = tmp_path / "pipeline_e2e.db"
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
            rel_int REAL
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
            atomic_mass REAL,
            PRIMARY KEY (element, sp_num)
        )
        """
    )
    conn.execute(
        """
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
        """
    )

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
        """
        INSERT INTO lines (
            element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
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
        """
        INSERT INTO species_physics (element, sp_num, ip_ev, atomic_mass)
        VALUES (?, ?, ?, ?)
        """,
        [
            ("Fe", 1, 7.902, 55.845),
            ("Fe", 2, 16.187, 55.845),
            ("Cu", 1, 7.726, 63.546),
            ("Cu", 2, 20.292, 63.546),
        ],
    )
    conn.executemany(
        """
        INSERT INTO partition_functions (
            element, sp_num, a0, a1, a2, a3, a4, t_min, t_max, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
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


def _simulate_fe_cu_spectrum(atomic_db: AtomicDatabase) -> tuple[np.ndarray, np.ndarray]:
    plasma = SingleZoneLTEPlasma.from_mass_fractions(
        T_e=12000.0,
        n_e=1.0e17,
        mass_fractions=TEST_MASS_FRACTIONS,
        total_species_density_cm3=3.0e16,
        atomic_masses_amu=TEST_ATOMIC_MASSES,
    )
    model = SpectrumModel(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=InstrumentModel(resolution_fwhm_nm=0.03),
        lambda_min=258.0,
        lambda_max=520.0,
        delta_lambda=0.01,
        path_length_m=0.005,
    )
    wavelength, intensity = model.compute_spectrum()

    rng = np.random.default_rng(123)
    baseline = 0.001 * intensity.max()
    noisy = np.maximum(
        intensity + baseline + rng.normal(0.0, 0.0005 * intensity.max(), size=intensity.shape),
        0.0,
    )
    return wavelength, noisy


@pytest.mark.integration
def test_full_pipeline_recovers_multistage_sample(tmp_path: Path):
    atomic_db = _build_multistage_atomic_db(tmp_path)
    wavelength, intensity = _simulate_fe_cu_spectrum(atomic_db)

    detected = detect_line_observations(
        wavelength,
        intensity,
        atomic_db,
        elements=["Fe", "Cu"],
        wavelength_tolerance_nm=0.08,
        min_peak_height=0.005,
        peak_width_nm=0.10,
        min_relative_intensity=50.0,
        kdet_enabled=False,
    )

    detected_lines = {
        (obs.element, obs.ionization_stage, round(obs.wavelength_nm, 2))
        for obs in detected.observations
    }
    assert detected_lines == EXPECTED_LINES
    assert detected.unmatched_peaks == 0

    alias = ALIASIdentifier(
        atomic_db=atomic_db,
        elements=["Fe", "Cu"],
        resolving_power=10000.0,
        detection_threshold=0.02,
        intensity_threshold_factor=3.0,
        T_range_K=(10000.0, 14000.0),
        n_e_range_cm3=(5.0e16, 2.0e17),
        T_steps=3,
        n_e_steps=2,
    )
    alias_result = alias.identify(wavelength, intensity)

    assert {element.element for element in alias_result.detected_elements} == {"Cu", "Fe"}

    alias_lines = {
        (obs.element, obs.ionization_stage, round(obs.wavelength_nm, 2))
        for obs in to_line_observations(alias_result)
    }
    assert alias_lines == EXPECTED_LINES

    solver = IterativeCFLIBSSolver(atomic_db, max_iterations=20)
    recovered = solver.solve(detected.observations)

    assert recovered.converged
    assert recovered.temperature_K == pytest.approx(12000.0, rel=0.05)
    assert recovered.concentrations["Fe"] == pytest.approx(
        EXPECTED_NUMBER_FRACTIONS["Fe"], abs=0.05
    )
    assert recovered.concentrations["Cu"] == pytest.approx(
        EXPECTED_NUMBER_FRACTIONS["Cu"], abs=0.05
    )

    recovered_from_alias = solver.solve(to_line_observations(alias_result))

    assert recovered_from_alias.converged
    assert recovered_from_alias.temperature_K == pytest.approx(12000.0, rel=0.05)
    assert recovered_from_alias.concentrations["Fe"] == pytest.approx(
        EXPECTED_NUMBER_FRACTIONS["Fe"], abs=0.05
    )
    assert recovered_from_alias.concentrations["Cu"] == pytest.approx(
        EXPECTED_NUMBER_FRACTIONS["Cu"], abs=0.05
    )
