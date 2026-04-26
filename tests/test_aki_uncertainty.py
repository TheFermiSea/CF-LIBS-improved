"""
Tests for A_ki uncertainty and accuracy grade feature (CF-LIBS-eed).

Validates:
1. Schema migration adds columns without data loss
2. NIST accuracy grades mapped for >80% of existing lines
3. Uncertainty propagation pipeline wired end-to-end
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

DB_PATH = Path(__file__).parent.parent / "libs_production.db"


@pytest.fixture
def db_conn():
    """Connect to the production database."""
    if not DB_PATH.exists():
        pytest.skip("libs_production.db not available")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.mark.requires_db
def test_schema_has_aki_uncertainty_column(db_conn):
    """Schema migration adds aki_uncertainty column."""
    cursor = db_conn.execute("PRAGMA table_info(lines)")
    cols = {r["name"] for r in cursor}
    assert "aki_uncertainty" in cols


@pytest.mark.requires_db
def test_schema_has_accuracy_grade_column(db_conn):
    """Schema migration adds accuracy_grade column."""
    cursor = db_conn.execute("PRAGMA table_info(lines)")
    cols = {r["name"] for r in cursor}
    assert "accuracy_grade" in cols


@pytest.mark.requires_db
def test_accuracy_grades_populated_for_majority(db_conn):
    """NIST accuracy grades mapped for >80% of lines with A_ki."""
    cursor = db_conn.execute(
        "SELECT COUNT(*) FROM lines WHERE aki IS NOT NULL AND accuracy_grade IS NOT NULL"
    )
    n_graded = cursor.fetchone()[0]
    cursor = db_conn.execute("SELECT COUNT(*) FROM lines WHERE aki IS NOT NULL")
    n_total = cursor.fetchone()[0]
    assert n_total > 0
    coverage = n_graded / n_total
    assert coverage >= 0.80, f"Only {coverage:.1%} of lines have accuracy grades"


@pytest.mark.requires_db
def test_aki_uncertainty_values_are_valid(db_conn):
    """aki_uncertainty values should be between 0 and 1."""
    cursor = db_conn.execute(
        "SELECT MIN(aki_uncertainty), MAX(aki_uncertainty) FROM lines WHERE aki_uncertainty IS NOT NULL"
    )
    row = cursor.fetchone()
    assert row[0] >= 0.0, f"Negative uncertainty found: {row[0]}"
    assert row[1] <= 1.0, f"Uncertainty > 100% found: {row[1]}"


@pytest.mark.requires_db
def test_accuracy_grade_values_are_valid(db_conn):
    """accuracy_grade values should be valid NIST grades."""
    valid_grades = {"AAA", "AA", "A+", "A", "B+", "B", "C+", "C", "D+", "D", "E"}
    cursor = db_conn.execute(
        "SELECT DISTINCT accuracy_grade FROM lines WHERE accuracy_grade IS NOT NULL"
    )
    grades = {r[0] for r in cursor}
    invalid = grades - valid_grades
    assert not invalid, f"Invalid accuracy grades found: {invalid}"


@pytest.mark.requires_db
def test_no_data_loss_after_migration(db_conn):
    """Migration should not alter existing data columns."""
    cursor = db_conn.execute("SELECT COUNT(*) FROM lines WHERE aki IS NOT NULL")
    n_with_aki = cursor.fetchone()[0]
    assert n_with_aki >= 13000, f"Too few lines with A_ki: {n_with_aki}"


def test_transition_struct_has_uncertainty_fields():
    """Transition dataclass carries aki_uncertainty and accuracy_grade."""
    from cflibs.atomic.structures import Transition

    t = Transition(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=404.581,
        A_ki=8.62e7,
        E_k_ev=4.5485,
        E_i_ev=1.4849,
        g_k=9,
        g_i=7,
        aki_uncertainty=0.10,
        accuracy_grade="B",
    )
    assert t.aki_uncertainty == 0.10
    assert t.accuracy_grade == "B"


def test_transition_default_uncertainty_is_none():
    """Transition defaults to None for uncertainty fields."""
    from cflibs.atomic.structures import Transition

    t = Transition(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=404.581,
        A_ki=8.62e7,
        E_k_ev=4.5485,
        E_i_ev=1.4849,
        g_k=9,
        g_i=7,
    )
    assert t.aki_uncertainty is None
    assert t.accuracy_grade is None


def test_atomic_data_uncertainty_from_transitions():
    """AtomicDataUncertainty.from_transitions() wires database values."""
    from cflibs.inversion.uncertainty import AtomicDataUncertainty

    t1 = MagicMock(wavelength_nm=404.581, aki_uncertainty=0.10)
    t2 = MagicMock(wavelength_nm=438.354, aki_uncertainty=0.25)
    t3 = MagicMock(wavelength_nm=500.0, aki_uncertainty=None)

    adu = AtomicDataUncertainty.from_transitions([t1, t2, t3])

    # Per-line values
    assert adu.get_uncertainty(404.581) == 0.10
    assert adu.get_uncertainty(438.354) == 0.25
    # Default for unknown wavelength
    assert adu.get_uncertainty(999.0) == pytest.approx(0.175, abs=0.01)  # median of [0.10, 0.25]


def test_atomic_data_uncertainty_empty_transitions():
    """from_transitions with no uncertainties uses fallback."""
    from cflibs.inversion.uncertainty import AtomicDataUncertainty

    import types

    t1 = types.SimpleNamespace(wavelength_nm=404.581)
    # No aki_uncertainty attribute — SimpleNamespace won't have it

    adu = AtomicDataUncertainty.from_transitions([t1])
    assert adu.default_A_uncertainty == 0.10  # Grade B fallback


def test_nist_grade_uncertainty_mapping():
    """NIST grade mapping constants are correct."""
    from cflibs.atomic.reference_data import NIST_GRADE_UNCERTAINTY

    mapping = NIST_GRADE_UNCERTAINTY
    assert mapping["AAA"] == 0.003
    assert mapping["AA"] == 0.01
    assert mapping["A"] == 0.03
    assert mapping["B"] == 0.10
    assert mapping["C"] == 0.25
    assert mapping["D"] == 0.50
    assert mapping["E"] == 0.50
