"""Focused fast tests for the opt-in A_ki overlay read path + build-script matching.

No production DB is touched: tiny in-memory/temp DB pairs only. Physics-only.
"""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from cflibs.atomic.aki_anchor import a_from_log_gf
from cflibs.atomic.database import AtomicDatabase


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _make_source_db(tmp_path):
    """Minimal atomic DB with the columns get_transitions needs."""
    db = str(tmp_path / "src.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE lines (
            id INTEGER PRIMARY KEY, element TEXT, sp_num INTEGER,
            wavelength_nm REAL, aki REAL, ek_ev REAL, ei_ev REAL,
            gk REAL, gi REAL, rel_int REAL, stark_w REAL, stark_alpha REAL,
            stark_shift REAL, is_resonance INTEGER, aki_uncertainty REAL,
            accuracy_grade TEXT
        )
        """)
    conn.execute(
        "INSERT INTO lines VALUES (10,'Fe',1,400.0,3.0e7,3.1,0.0,5,9,100,NULL,NULL,"
        "NULL,0,0.5,'D')"
    )
    conn.execute(
        "INSERT INTO lines VALUES (11,'Fe',1,420.0,1.0e7,3.0,0.0,5,9,50,NULL,NULL,"
        "NULL,0,0.5,'D')"
    )
    # Tables the AtomicDatabase schema migration expects to exist.
    conn.execute(
        "CREATE TABLE energy_levels (element TEXT, sp_num INTEGER, g_level INTEGER, "
        "energy_ev REAL, level_id TEXT, j_val TEXT)"
    )
    conn.execute("INSERT INTO energy_levels VALUES ('Fe',1,9,0.0,'A','4')")
    conn.execute(
        "CREATE TABLE species_physics (element TEXT, sp_num INTEGER, ip_ev REAL, "
        "PRIMARY KEY (element, sp_num))"
    )
    conn.execute("INSERT INTO species_physics VALUES ('Fe',1,7.87)")
    conn.commit()
    conn.close()
    return db


def _make_overlay(tmp_path, entries):
    ov = str(tmp_path / "ov.db")
    conn = sqlite3.connect(ov)
    conn.execute("""
        CREATE TABLE anchored_lines (
            line_id INTEGER PRIMARY KEY, element TEXT, sp_num INTEGER,
            aki_anchored REAL, aki_unc REAL, method TEXT, source TEXT,
            match_quality TEXT
        )
        """)
    conn.executemany("INSERT INTO anchored_lines VALUES (?,?,?,?,?,?,?,?)", entries)
    conn.commit()
    conn.close()
    return ov


def test_default_no_overlay_read_is_nonmutating(tmp_path):
    db = _make_source_db(tmp_path)
    adb = AtomicDatabase(db)  # no overlay
    after_construct = _sha(db)  # hash AFTER the constructor's one-time schema migration
    trans = {t.wavelength_nm: t for t in adb.get_transitions("Fe", 1)}
    assert trans[400.0].A_ki == 3.0e7
    assert trans[400.0].aki_source is None
    # The read path (get_transitions) itself never mutates the source DB.
    assert _sha(db) == after_construct


def test_overlay_resolves_aki_with_provenance(tmp_path):
    db = _make_source_db(tmp_path)
    ov = _make_overlay(
        tmp_path,
        [(10, "Fe", 1, 9.9e7, 4.95e6, "lab_gf", "Lawler2013_ApJS_205_11", "level:0.0004")],
    )
    ov_before = _sha(ov)
    adb = AtomicDatabase(db, aki_overlay_path=ov)
    after_construct = _sha(db)
    trans = {t.wavelength_nm: t for t in adb.get_transitions("Fe", 1)}
    # Line 10 (400 nm) resolved from overlay; line 11 (420 nm) untouched.
    assert trans[400.0].A_ki == pytest.approx(9.9e7)
    assert trans[400.0].aki_source == "lawler_overlay:Lawler2013_ApJS_205_11"
    assert trans[400.0].aki_uncertainty == pytest.approx(4.95e6 / 9.9e7)  # fractional
    assert trans[420.0].A_ki == 1.0e7
    assert trans[420.0].aki_source is None
    # The read path mutates neither the source DB nor the overlay.
    assert _sha(db) == after_construct
    assert _sha(ov) == ov_before


def test_missing_overlay_file_degrades_to_source(tmp_path):
    db = _make_source_db(tmp_path)
    adb = AtomicDatabase(db, aki_overlay_path=str(tmp_path / "does_not_exist.db"))
    trans = {t.wavelength_nm: t for t in adb.get_transitions("Fe", 1)}
    assert trans[400.0].A_ki == 3.0e7
    assert trans[400.0].aki_source is None


def test_gf_to_a_roundtrip_known_line():
    # g_k A_ki = 6.6702e15 * gf / lambda[Angstrom]^2 ; invert log_gf back to A.
    gk, wl_nm = 5.0, 400.0
    a = a_from_log_gf(-0.5, gk, wl_nm)
    # Recompute log_gf from A and confirm the inverse maps back.
    import math

    from cflibs.atomic.aki_anchor import GF_TO_GKA_ANGSTROM

    gf = gk * a * (wl_nm * 10.0) ** 2 / GF_TO_GKA_ANGSTROM
    assert math.log10(gf) == pytest.approx(-0.5, abs=1e-9)


def test_build_match_refuses_ambiguous_level(tmp_path):
    """Two DB levels inside the energy+J tolerance -> the matcher refuses (no guess)."""
    from scripts.build_lawler_overlay import CM_TO_EV, match_level

    db = str(tmp_path / "lvl.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE energy_levels (element TEXT, sp_num INTEGER, energy_ev REAL, "
        "level_id TEXT, j_val TEXT)"
    )
    e_cm = 45000.0
    # Two levels 0.005 cm-1 apart, both J=3, both within the 0.01 cm-1 window.
    conn.execute(
        "INSERT INTO energy_levels VALUES ('Fe',1,?,?,'3')",
        (e_cm * CM_TO_EV, "A"),
    )
    conn.execute(
        "INSERT INTO energy_levels VALUES ('Fe',1,?,?,'3')",
        ((e_cm + 0.005) * CM_TO_EV, "B"),
    )
    conn.commit()
    conn.row_factory = sqlite3.Row
    m = match_level(conn, "Fe", 1, e_cm, 3.0)
    conn.close()
    assert m.status == "ambiguous"
    assert m.level_id is None


def test_build_match_unique_level(tmp_path):
    from scripts.build_lawler_overlay import CM_TO_EV, match_level

    db = str(tmp_path / "lvl2.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE energy_levels (element TEXT, sp_num INTEGER, energy_ev REAL, "
        "level_id TEXT, j_val TEXT)"
    )
    e_cm = 45000.0
    conn.execute("INSERT INTO energy_levels VALUES ('Fe',1,?,?,'3')", (e_cm * CM_TO_EV, "A"))
    conn.execute(
        "INSERT INTO energy_levels VALUES ('Fe',1,?,?,'2')",
        ((e_cm + 50.0) * CM_TO_EV, "B"),
    )
    conn.commit()
    conn.row_factory = sqlite3.Row
    m = match_level(conn, "Fe", 1, e_cm + 0.002, 3.0)
    conn.close()
    assert m.status == "matched"
    assert m.level_id == "A"
    assert m.dist_cm == pytest.approx(0.002, abs=1e-4)
