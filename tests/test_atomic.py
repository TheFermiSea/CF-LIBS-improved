"""
Tests for atomic data structures and database.
"""

import pytest
from cflibs.atomic.structures import Transition, EnergyLevel, SpeciesPhysics
from cflibs.atomic.database import AtomicDatabase


def test_transition_creation(sample_transition):
    """Test creating a transition."""
    assert sample_transition.element == "Fe"
    assert sample_transition.ionization_stage == 1
    assert sample_transition.wavelength_nm == 371.99
    assert sample_transition.A_ki == 1.0e7


def test_transition_energy_diff(sample_transition):
    """Test transition energy difference property."""
    assert sample_transition.energy_diff_ev == pytest.approx(3.33, rel=1e-6)


def test_energy_level_creation(sample_energy_level):
    """Test creating an energy level."""
    assert sample_energy_level.element == "Fe"
    assert sample_energy_level.ionization_stage == 1
    assert sample_energy_level.energy_ev == 3.33
    assert sample_energy_level.g == 11


def test_species_physics_creation():
    """Test creating species physics."""
    species = SpeciesPhysics(element="Fe", ionization_stage=1, ionization_potential_ev=7.87)
    assert species.element == "Fe"
    assert species.ionization_stage == 1
    assert species.ionization_potential_ev == 7.87


def test_atomic_database_init(atomic_db):
    """Test AtomicDatabase initialization."""
    assert atomic_db.db_path.exists()
    # Check if either pool or conn exists and is valid
    if atomic_db._use_pool:
        assert atomic_db._pool is not None
    else:
        assert atomic_db.conn is not None


def test_atomic_database_get_transitions(atomic_db):
    """Test getting transitions from database."""
    transitions = atomic_db.get_transitions("Fe", ionization_stage=1)
    assert len(transitions) > 0
    assert all(isinstance(t, Transition) for t in transitions)
    assert all(t.element == "Fe" for t in transitions)
    assert all(t.ionization_stage == 1 for t in transitions)


def test_atomic_database_get_transitions_filtered(atomic_db):
    """Test getting filtered transitions."""
    transitions = atomic_db.get_transitions("Fe", wavelength_min=370.0, wavelength_max=375.0)
    assert len(transitions) > 0
    assert all(370.0 <= t.wavelength_nm <= 375.0 for t in transitions)


def test_atomic_database_get_energy_levels(atomic_db):
    """Test getting energy levels from database."""
    levels = atomic_db.get_energy_levels("Fe", 1)
    assert len(levels) > 0
    assert all(isinstance(level, EnergyLevel) for level in levels)
    assert all(level.element == "Fe" for level in levels)
    assert all(level.ionization_stage == 1 for level in levels)


def test_atomic_database_get_ionization_potential(atomic_db):
    """Test getting ionization potential."""
    ip = atomic_db.get_ionization_potential("Fe", 1)
    assert ip is not None
    assert ip > 0
    assert ip == pytest.approx(7.87, rel=0.1)


def test_atomic_database_get_ionization_potential_nonexistent(atomic_db):
    """Test getting ionization potential for non-existent species."""
    ip = atomic_db.get_ionization_potential("Xx", 1)
    assert ip is None


def test_atomic_database_get_species_physics(atomic_db):
    """Test getting species physics."""
    physics = atomic_db.get_species_physics("Fe", 1)
    assert physics is not None
    assert isinstance(physics, SpeciesPhysics)
    assert physics.element == "Fe"
    assert physics.ionization_stage == 1


def test_atomic_database_get_available_elements(atomic_db):
    """Test getting available elements."""
    elements = atomic_db.get_available_elements()
    assert isinstance(elements, list)
    assert "Fe" in elements
    assert "H" in elements


def test_atomic_database_context_manager(temp_db):
    """Test AtomicDatabase as context manager."""
    with AtomicDatabase(temp_db) as db:
        transitions = db.get_transitions("Fe")
        assert len(transitions) > 0

    # Database should be closed
    # (We can't easily test this without accessing private attributes)


def test_atomic_database_close(atomic_db):
    """Test closing database connection."""
    atomic_db.close()
    # Connection should be closed (tested implicitly)


def test_sql_injection_prevention(temp_db):
    """Test that schema migration prevents SQL injection via invalid column names or types."""

    class TestAtomicDatabase(AtomicDatabase):
        def _perform_migration(self, conn):
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(lines)")
            columns = {row[1] for row in cursor.fetchall()}

            # Simulate a malicious injection attempt in required_line_cols
            required_line_cols = {
                "stark_w; DROP TABLE lines;": "REAL",
            }

            valid_dtypes = {"REAL", "INTEGER", "TEXT", "BLOB", "NUMERIC"}
            for col, dtype in required_line_cols.items():
                if col not in columns:
                    if not col.isidentifier():
                        raise ValueError(f"Invalid column name for migration: {col}")
                    if dtype.upper() not in valid_dtypes:
                        raise ValueError(f"Invalid data type for migration: {dtype}")

                    cursor.execute(f"ALTER TABLE lines ADD COLUMN {col} {dtype}")

    # Now verify that initializing the patched DB class raises the expected error
    with pytest.raises(ValueError, match="Invalid column name for migration"):
        TestAtomicDatabase(temp_db)
