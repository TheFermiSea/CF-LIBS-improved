"""Acceptance tests for the STARK-B / VdW / self-broadening overhaul.

Verifies the production atomic database at ``ASD_da/libs_production.db``
has been migrated and populated per the deliverables in the audit at
``/tmp/db_audit/stark_broadening.md``:

* New columns ``gamma_vdw_log``, ``gamma_self_log``, ``stark_w_source``
  exist on the ``lines`` table.
* At least 95 % of lines have non-NULL ``stark_w`` (vs the 82.2 %
  baseline before this change).
* At least 200 lines have line-specific STARK-B literature widths
  (``stark_w_source = "stark_b"``).
* For five canonical anchor lines (Mg II 279.55 / 280.27, Ca II 393.4 /
  396.8, Fe II 430.32) the DB stark_w matches the published value
  within 20 %.
* At least 80 % of lines have a non-NULL ``gamma_vdw_log`` estimate.

All assertions read directly from the DB; the test does NOT mutate it.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

DB_PATH = Path(__file__).resolve().parents[1] / "ASD_da" / "libs_production.db"


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        pytest.skip(f"Production DB not present at {DB_PATH}")
    return sqlite3.connect(str(DB_PATH))


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def test_new_broadening_columns_present() -> None:
    """Schema migration added the three new columns."""
    conn = _connect()
    try:
        cols = _table_columns(conn, "lines")
        assert "gamma_vdw_log" in cols, "gamma_vdw_log column missing"
        assert "gamma_self_log" in cols, "gamma_self_log column missing"
        assert "stark_w_source" in cols, "stark_w_source column missing"
    finally:
        conn.close()


def test_stark_w_coverage_at_least_95pct() -> None:
    """≥95 % of lines have a non-NULL stark_w (was 82.2 % in PR #99)."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM lines")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM lines WHERE stark_w IS NOT NULL")
        n = cur.fetchone()[0]
        pct = 100.0 * n / max(total, 1)
        assert pct >= 95.0, (
            f"stark_w coverage {pct:.1f}% < 95% target ({n}/{total})"
        )
    finally:
        conn.close()


def test_at_least_200_lines_have_stark_b_provenance() -> None:
    """≥200 lines have stark_w_source = 'stark_b' (line-specific data)."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM lines WHERE stark_w_source = 'stark_b'"
        )
        n = cur.fetchone()[0]
        assert n >= 200, f"Only {n} lines with stark_w_source='stark_b' < 200"
    finally:
        conn.close()


# Canonical line anchors and their literature STARK-B widths in pm at
# T_e=10000 K, n_e=1e17 cm^-3 (FWHM, electron-impact only).
# Sources:
#   Mg II 279.55 / 280.27: Konjević 2002 J. Phys. Chem. Ref. Data 31, Tab 7
#   Ca II 393.4 / 396.8:   Konjević 2002 Tab 9 (H&K resonance)
#   Fe II 430.317:         Konjević 2002 Tab 15 (visible Fe II multiplet)
ANCHORS = [
    ("Mg", 2, 279.553, 5.7),
    ("Mg", 2, 280.270, 5.7),
    ("Ca", 2, 393.366, 8.4),
    ("Ca", 2, 396.847, 8.6),
    ("Fe", 2, 430.317, 33.0),
]


@pytest.mark.parametrize("element,sp,wl_target,expected_pm", ANCHORS)
def test_stark_b_anchor_lines_match_published(
    element: str, sp: int, wl_target: float, expected_pm: float
) -> None:
    """Anchor-line stark_w matches the published STARK-B value within 20 %."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT wavelength_nm, stark_w, stark_w_source
            FROM lines
            WHERE element = ? AND sp_num = ?
              AND wavelength_nm BETWEEN ? AND ?
            ORDER BY ABS(wavelength_nm - ?) ASC
            LIMIT 1
            """,
            (element, sp, wl_target - 0.05, wl_target + 0.05, wl_target),
        )
        row = cur.fetchone()
        assert row is not None, (
            f"No matching line for {element} {sp} ~{wl_target} nm"
        )
        wl_actual, stark_w_nm, source = row
        assert stark_w_nm is not None, "stark_w is NULL on anchor line"
        observed_pm = stark_w_nm * 1000.0
        ratio = observed_pm / expected_pm
        assert 0.80 <= ratio <= 1.20, (
            f"{element} {sp} {wl_actual:.3f} nm: stark_w = {observed_pm:.2f} pm "
            f"(expected {expected_pm} pm; ratio {ratio:.2f} outside ±20%, "
            f"source={source!r})"
        )
        # Bonus: anchor lines must also be tagged as STARK-B literature
        # values, not the λ²-scaled fallback.
        assert source == "stark_b", (
            f"{element} {sp} {wl_actual:.3f} nm provenance is {source!r}, "
            "expected 'stark_b'"
        )
    finally:
        conn.close()


def test_gamma_vdw_log_coverage_at_least_80pct() -> None:
    """≥80 % of lines have a non-NULL gamma_vdw_log estimate."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM lines")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM lines WHERE gamma_vdw_log IS NOT NULL")
        n = cur.fetchone()[0]
        pct = 100.0 * n / max(total, 1)
        assert pct >= 80.0, (
            f"gamma_vdw_log coverage {pct:.1f}% < 80% target ({n}/{total})"
        )
    finally:
        conn.close()


def test_gamma_self_log_coverage_at_least_80pct() -> None:
    """≥80 % of lines have a non-NULL gamma_self_log estimate."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM lines")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM lines WHERE gamma_self_log IS NOT NULL"
        )
        n = cur.fetchone()[0]
        pct = 100.0 * n / max(total, 1)
        assert pct >= 80.0, (
            f"gamma_self_log coverage {pct:.1f}% < 80% target ({n}/{total})"
        )
    finally:
        conn.close()


def test_provenance_values_are_recognized() -> None:
    """Every non-NULL stark_w_source is one of the documented values."""
    allowed = {
        "stark_b",
        "konjevic_lambda_sq_scaled",
        "interpolated",
        "hydrogenic",
        "lanthanide_default",
    }
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT stark_w_source FROM lines "
            "WHERE stark_w_source IS NOT NULL"
        )
        seen = {row[0] for row in cur.fetchall()}
        unknown = seen - allowed
        assert not unknown, f"Unknown provenance values: {unknown}"
    finally:
        conn.close()


def test_no_unprovenanced_stark_w() -> None:
    """Every populated stark_w must have a provenance tag."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM lines "
            "WHERE stark_w IS NOT NULL AND stark_w_source IS NULL"
        )
        n = cur.fetchone()[0]
        assert n == 0, f"{n} lines have stark_w populated but no provenance tag"
    finally:
        conn.close()
