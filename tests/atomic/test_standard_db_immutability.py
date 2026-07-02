"""Immutability hardening for standard atomic databases.

Covers:
1. ``is_standard_db`` recognition (ASD_da / libs_production.db vs overlays / tmp).
2. The on-connect schema migration REFUSES to mutate a *recognized standard* DB
   whose schema is incomplete, unless ``allow_schema_migration=True`` is passed.
   Non-standard scratch/test DBs migrate freely (no flag needed).
3. ``AtomicDatabase(..., read_only=True)`` opens ``mode=ro`` -- no write can
   reach the file and no migration runs.
4. ``scripts/verify_standard_dbs.py`` recomputes checksums and detects tampering
   (and the shipped ``libs_production.db`` matches its manifest).
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from cflibs.atomic.database import (
    ADD_LINE_COLUMN_QUERIES,
    AtomicDatabase,
    ImmutableDatabaseError,
    is_standard_db,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_standard_dbs.py"


def _load_verify_module():
    """Load scripts/verify_standard_dbs.py as a module (scripts/ is not a pkg)."""
    spec = importlib.util.spec_from_file_location("verify_standard_dbs", VERIFY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_bare_db(path: Path) -> None:
    """Create a DB whose schema is *incomplete* (migration would fire)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE lines (id INTEGER PRIMARY KEY, element TEXT, sp_num INTEGER, "
        "wavelength_nm REAL, aki REAL, ei_ev REAL, ek_ev REAL, gi INTEGER, gk INTEGER, "
        "rel_int REAL)"
    )
    conn.execute(
        "CREATE TABLE energy_levels (element TEXT, sp_num INTEGER, g_level INTEGER, "
        "energy_ev REAL)"
    )
    conn.execute(
        "CREATE TABLE species_physics (element TEXT, sp_num INTEGER, ip_ev REAL, "
        "PRIMARY KEY (element, sp_num))"
    )
    conn.execute("INSERT INTO lines VALUES (1,'Fe',1,371.99,1.0e7,0.0,3.33,9,11,1000)")
    conn.execute("INSERT INTO energy_levels VALUES ('Fe',1,9,0.0)")
    conn.execute("INSERT INTO species_physics VALUES ('Fe',1,7.87)")
    conn.commit()
    conn.close()


def _make_complete_db(path: Path) -> None:
    """Create a DB that already has every column/table the migration would add."""
    cols = ", ".join(
        f"{c} TEXT" if c in ("accuracy_grade", "stark_w_source") else f"{c} REAL"
        for c in ADD_LINE_COLUMN_QUERIES
    )
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE lines (id INTEGER PRIMARY KEY, element TEXT, sp_num INTEGER, "
        "wavelength_nm REAL, aki REAL, ei_ev REAL, ek_ev REAL, gi INTEGER, gk INTEGER, "
        f"rel_int REAL, {cols})"
    )
    conn.execute(
        "CREATE TABLE energy_levels (element TEXT, sp_num INTEGER, g_level INTEGER, "
        "energy_ev REAL)"
    )
    conn.execute(
        "CREATE TABLE species_physics (element TEXT, sp_num INTEGER, ip_ev REAL, "
        "atomic_mass REAL, PRIMARY KEY (element, sp_num))"
    )
    conn.execute(
        "CREATE TABLE partition_functions (element TEXT, sp_num INTEGER, a0 REAL, a1 REAL, "
        "a2 REAL, a3 REAL, a4 REAL, t_min REAL, t_max REAL, source TEXT, "
        "PRIMARY KEY (element, sp_num))"
    )
    conn.execute(
        "INSERT INTO lines (id, element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, "
        "rel_int, aki_uncertainty) VALUES (1,'Fe',1,371.99,1.0e7,0.0,3.33,9,11,1000,0.1)"
    )
    conn.execute("INSERT INTO energy_levels VALUES ('Fe',1,9,0.0)")
    conn.execute("INSERT INTO species_physics VALUES ('Fe',1,7.87,55.8)")
    conn.execute(
        "INSERT INTO partition_functions VALUES ('Fe',1,1.0,0,0,0,0,2000.0,20000.0,'test')"
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Standard-DB recognition
# --------------------------------------------------------------------------- #
def test_is_standard_db_recognition(tmp_path):
    assert is_standard_db("ASD_da/libs_production.db")
    assert is_standard_db(tmp_path / "ASD_da" / "libs_production.db")
    assert is_standard_db(tmp_path / "ASD_da" / "some_standard.db")
    assert is_standard_db("/anywhere/libs_production.db")  # canonical basename
    # Overlays and arbitrary scratch paths are NOT standard.
    assert not is_standard_db("ASD_da/overlays/lawler_anchors_v1.db")
    assert not is_standard_db(tmp_path / "scratch.db")


# --------------------------------------------------------------------------- #
# Migration guard (scoped to standard DBs)
# --------------------------------------------------------------------------- #
def test_standard_bare_db_refuses_migration(tmp_path):
    """An incomplete DB at a standard path refuses (would ALTER/populate it)."""
    db = tmp_path / "ASD_da" / "libs_production.db"  # recognized standard path
    _make_bare_db(db)
    with pytest.raises(ImmutableDatabaseError) as exc:
        AtomicDatabase(str(db))
    msg = str(exc.value)
    assert "immutable" in msg.lower()
    assert "allow_schema_migration=True" in msg


def test_standard_db_allow_schema_migration_opt_in(tmp_path):
    """Explicit opt-in lets a standard-path rebuild add the columns."""
    db = tmp_path / "ASD_da" / "libs_production.db"
    _make_bare_db(db)
    adb = AtomicDatabase(str(db), allow_schema_migration=True)
    try:
        with sqlite3.connect(str(db)) as check:
            cols = {r[1] for r in check.execute("PRAGMA table_info(lines)")}
        assert "stark_w" in cols  # migration ran
    finally:
        adb.close()


def test_nonstandard_bare_db_migrates_without_flag(tmp_path):
    """A fresh scratch DB (not a standard path) migrates with no flag."""
    db = tmp_path / "scratch.db"
    _make_bare_db(db)
    adb = AtomicDatabase(str(db))  # must NOT raise
    try:
        with sqlite3.connect(str(db)) as check:
            cols = {r[1] for r in check.execute("PRAGMA table_info(lines)")}
        assert "stark_w" in cols  # migration ran normally
    finally:
        adb.close()


def _schema_snapshot(db: Path) -> dict:
    """Columns of each table + row counts -- the migration-visible surface."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ]
        return {
            t: (
                tuple(r[1] for r in conn.execute(f"PRAGMA table_info({t})")),
                conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0],
            )
            for t in tables
        }
    finally:
        conn.close()


def test_complete_standard_db_is_noop(tmp_path):
    """A complete-schema standard DB opens with NO flag and performs NO migration."""
    db = tmp_path / "ASD_da" / "libs_production.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    _make_complete_db(db)
    before = _schema_snapshot(db)
    adb = AtomicDatabase(str(db))  # must not raise, must not migrate
    adb.close()
    assert _schema_snapshot(db) == before


def test_detect_pending_migrations_empty_on_complete(tmp_path):
    db = tmp_path / "complete.db"
    _make_complete_db(db)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        assert AtomicDatabase._detect_pending_migrations(conn.cursor()) == []
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Read-only mode
# --------------------------------------------------------------------------- #
def test_read_only_cannot_write(tmp_path):
    db = tmp_path / "ro.db"
    _make_complete_db(db)
    adb = AtomicDatabase(str(db), read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError) as exc:
            adb.conn.execute("UPDATE lines SET aki = 0")
        assert "readonly" in str(exc.value).lower()
    finally:
        adb.close()


def test_read_only_skips_migration_on_standard_bare_db(tmp_path):
    """read_only opens even an incomplete standard DB (no migration, no raise)."""
    db = tmp_path / "ASD_da" / "libs_production.db"
    _make_bare_db(db)
    before = db.read_bytes()
    adb = AtomicDatabase(str(db), read_only=True)  # must NOT raise
    try:
        with sqlite3.connect(str(db)) as check:
            cols = {r[1] for r in check.execute("PRAGMA table_info(lines)")}
        assert "stark_w" not in cols  # migration did NOT run
    finally:
        adb.close()
    assert db.read_bytes() == before


# --------------------------------------------------------------------------- #
# verify_standard_dbs.py
# --------------------------------------------------------------------------- #
def test_verify_detects_tampering(tmp_path, monkeypatch):
    """A byte-level change to a tracked DB is reported as a sha256 mismatch."""
    mod = _load_verify_module()
    fake_db = tmp_path / "fake_standard.db"
    fake_db.write_bytes(b"original standard db content")
    manifest = tmp_path / "MANIFEST.json"

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(mod, "STANDARD_DBS", {"fake_standard.db": "test db"})

    manifest.write_text(json.dumps(mod.build_manifest(), indent=2))
    assert mod.verify() == []

    fake_db.write_bytes(b"TAMPERED standard db content!!")
    problems = mod.verify()
    assert any("sha256" in p.lower() for p in problems)
    assert mod.main([]) == 1


@pytest.mark.requires_db
def test_shipped_db_matches_manifest():
    """The real libs_production.db matches STANDARD_DB_MANIFEST.json."""
    if not (REPO_ROOT / "ASD_da" / "libs_production.db").exists():
        pytest.skip("production DB not present")
    mod = _load_verify_module()
    problems = mod.verify()
    assert problems == [], f"standard DB verification failed: {problems}"


@pytest.mark.requires_db
def test_prod_db_opens_without_flag_and_read_only():
    """Prod DB opens with the default (no flag) -> migration is a genuine no-op."""
    db = REPO_ROOT / "ASD_da" / "libs_production.db"
    if not db.exists():
        pytest.skip("production DB not present")
    adb = AtomicDatabase(str(db))  # no allow_schema_migration needed
    adb.close()
    ro = AtomicDatabase(str(db), read_only=True)
    try:
        assert len(ro.get_transitions("Fe")) > 0
    finally:
        ro.close()
