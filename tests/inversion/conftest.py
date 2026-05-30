"""Shared fixtures for ``tests/inversion`` and its sub-packages.

Currently exposes one fixture, :func:`bayesian_db`, used by both
``tests/inversion/test_bayesian_forward_model_kernel_migration.py`` and
``tests/inversion/solve/bayesian/test_vmap_chain_method.py``.  Hoisting
it here removes the ~90-line SQLite-schema-and-inserts block that was
otherwise duplicated between those two files.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest


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
