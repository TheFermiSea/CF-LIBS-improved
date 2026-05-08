"""Session-scoped fixtures for the JAX equivalence tests.

The default ``atomic_db`` fixture in ``tests/conftest.py`` is function-scoped
because it builds a fresh sqlite file per test. The JAX parity sweeps in
this directory parametrize across temperatures × electron densities ×
elements; rebuilding the database for each parametrized case dominates
runtime (and turned the sweep into a multi-minute exercise during local
verification). The session-scoped variant below builds the database
exactly once.

We deliberately do not modify the original ``temp_db`` fixture — other
test modules rely on its function scope to mutate the DB.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from cflibs.atomic.database import AtomicDatabase


@pytest.fixture(scope="session")
def atomic_db_session() -> Iterator[AtomicDatabase]:
    """Session-scoped atomic database mirroring the conftest ``temp_db`` data."""
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
            PRIMARY KEY (element, sp_num)
        )
        """
    )

    # Same data as the canonical ``temp_db`` fixture.
    conn.execute(
        """
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES ('Fe', 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000),
               ('Fe', 1, 373.49, 5.0e6, 0.0, 3.32, 9, 9, 500),
               ('Fe', 1, 374.95, 2.0e6, 0.0, 3.31, 9, 7, 200),
               ('Fe', 1, 382.04, 6.7e7, 0.0, 3.24, 9, 9, 800),
               ('Fe', 1, 404.58, 8.6e6, 0.86, 3.93, 7, 9, 600),
               ('Fe', 1, 427.18, 2.2e7, 1.49, 4.39, 5, 7, 700),
               ('Fe', 1, 438.35, 5.0e6, 1.61, 4.44, 9, 9, 800),
               ('Fe', 1, 495.76, 4.2e6, 2.18, 4.68, 7, 7, 300),
               ('Fe', 1, 516.75, 5.7e6, 0.05, 2.45, 11, 9, 400),
               ('Fe', 1, 532.80, 1.1e6, 0.91, 3.24, 5, 5, 250)
        """
    )
    conn.execute(
        """
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES ('Fe', 2, 238.20, 3.0e8, 0.0, 5.20, 10, 10, 900),
               ('Fe', 2, 259.94, 2.2e8, 0.05, 4.82, 8, 8, 700),
               ('Fe', 2, 273.95, 2.0e8, 0.99, 5.51, 8, 6, 500),
               ('Fe', 2, 234.35, 1.1e8, 0.0, 5.29, 10, 8, 600)
        """
    )
    conn.execute(
        """
        INSERT INTO energy_levels (element, sp_num, g_level, energy_ev)
        VALUES ('Fe', 1, 9, 0.0),
               ('Fe', 1, 11, 0.05),
               ('Fe', 1, 7, 0.86),
               ('Fe', 1, 5, 1.49),
               ('Fe', 1, 9, 1.61),
               ('Fe', 1, 7, 2.18),
               ('Fe', 1, 11, 3.33),
               ('Fe', 1, 9, 3.32),
               ('Fe', 1, 7, 3.31),
               ('Fe', 1, 9, 3.24),
               ('Fe', 1, 9, 3.93),
               ('Fe', 1, 7, 4.39),
               ('Fe', 1, 9, 4.44),
               ('Fe', 1, 7, 4.68),
               ('Fe', 1, 5, 2.45),
               ('Fe', 2, 10, 0.0),
               ('Fe', 2, 8, 0.05),
               ('Fe', 2, 8, 0.99),
               ('Fe', 2, 10, 5.20),
               ('Fe', 2, 8, 4.82),
               ('Fe', 2, 6, 5.51),
               ('Fe', 2, 8, 5.29)
        """
    )
    conn.execute(
        """
        INSERT INTO species_physics (element, sp_num, ip_ev)
        VALUES ('Fe', 1, 7.87),
               ('Fe', 2, 16.18)
        """
    )
    conn.execute(
        """
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES ('H', 1, 656.28, 4.4e7, 0.0, 12.75, 2, 8, 10000),
               ('H', 1, 486.13, 8.4e6, 0.0, 12.75, 2, 8, 2000)
        """
    )
    conn.execute(
        """
        INSERT INTO energy_levels (element, sp_num, g_level, energy_ev)
        VALUES ('H', 1, 2, 0.0),
               ('H', 1, 8, 12.75)
        """
    )
    conn.execute(
        """
        INSERT INTO species_physics (element, sp_num, ip_ev)
        VALUES ('H', 1, 13.60)
        """
    )

    conn.commit()
    conn.close()

    yield AtomicDatabase(db_path)

    Path(db_path).unlink(missing_ok=True)
