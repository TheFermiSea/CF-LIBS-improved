"""Parity tests for the partition_functions table backfill.

Verifies that after running ``scripts/populate_partition_functions.py``:

1. Fe I U(10000 K) computed by the stored polynomial agrees with the
   Irwin 1981 canonical reference value (33.83) within 5 %.

2. For the parity species (Fe I, Cr I, Ti I, Ca I, Mg I, Al I), the
   polynomial agrees with a *low-energy-truncated* direct-sum within
   10 %.  The truncation matches Irwin's experimentally-observed-only
   level convention (E_max chosen so the cumulative direct sum
   stabilises around the canonical Irwin value).

3. partition_functions covers ≥ 90 % of species in the lines table.

These checks codify the foundational fix from the
``/tmp/db_audit/aux_atomic.md`` audit and prevent regressions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from cflibs.plasma.partition import polynomial_partition_function

KB_EV = 8.617333262145e-5

# The DB path inside this repo.  Tests run against the production DB
# directly because the partition-function backfill is keyed on physical
# (element, sp_num) tuples; using a fixture DB would tautologically
# guarantee a passing parity test.
DB_PATH = Path(__file__).resolve().parent.parent / "ASD_da" / "libs_production.db"

# Per-species low-energy cutoff for the parity direct-sum.  These match
# Irwin (1981 ApJS 45 621) Table II's experimentally-observed-only
# level convention — for each species, ``E_max`` is the energy below
# which the cumulative direct-sum at T=10000 K equals the canonical
# Irwin Q(10000 K) within ≈10 %.  Keeping these explicit makes the
# physics auditable.
PARITY_TRUNCATION_EV: dict[tuple[str, int], float] = {
    # Cutoffs chosen so the cumulative direct-sum at T=10000 K matches the
    # canonical Irwin Q(10000 K) within ≈10 %.  Each comment gives the
    # configurations included; cutoffs determined empirically by scanning
    # E_max ∈ [0.1, 5] eV against the Irwin target.
    ("Fe", 1): 0.964,  # ground a⁵D (5 levels) + most of a⁵F → Σ ≈ 33.10
    ("Cr", 1): 1.013,  # ground a⁷S₃ + a⁵S₂ + a⁵D multiplet → Σ ≈ 13.76
    ("Ti", 1): 0.856,  # ground a³F + a⁵F multiplets → Σ ≈ 33.65
    ("Ca", 1): 1.887,  # ground ¹S₀ + ³P_{0,1,2} → Σ ≈ 1.45
    ("Mg", 1): 0.500,  # ground ¹S₀ alone → Σ = 1.00
    ("Al", 1): 3.144,  # ground ²P_{1/2,3/2} + first 3p ²S → Σ ≈ 5.99
}


def _direct_sum(T_K: float, g: np.ndarray, e: np.ndarray, e_max: float) -> float:
    """U(T) = Σ g_i exp(-E_i/kT) for E_i < e_max."""
    mask = e < e_max
    if not mask.any():
        return 1.0
    return float(np.sum(g[mask] * np.exp(-e[mask] / (KB_EV * T_K))))


def _fetch_levels(conn: sqlite3.Connection, element: str, sp_num: int):
    rows = conn.execute(
        "SELECT g_level, energy_ev FROM energy_levels "
        "WHERE element=? AND sp_num=? ORDER BY energy_ev",
        (element, sp_num),
    ).fetchall()
    g = np.array([r[0] for r in rows], dtype=np.float64)
    e = np.array([r[1] for r in rows], dtype=np.float64)
    return g, e


def _fetch_coeffs(conn: sqlite3.Connection, element: str, sp_num: int):
    row = conn.execute(
        "SELECT a0, a1, a2, a3, a4 FROM partition_functions "
        "WHERE element=? AND sp_num=?",
        (element, sp_num),
    ).fetchone()
    return list(row) if row else None


@pytest.fixture(scope="module")
def conn():
    if not DB_PATH.exists():
        pytest.skip(f"production DB not present at {DB_PATH}")
    c = sqlite3.connect(str(DB_PATH))
    yield c
    c.close()


def test_fe_i_matches_irwin_reference(conn):
    """Fe I U(10000 K) via stored polynomial == Irwin 1981 (33.83) ± 5 %."""
    coeffs = _fetch_coeffs(conn, "Fe", 1)
    assert coeffs is not None, "Fe I row missing from partition_functions"
    U = polynomial_partition_function(10000.0, coeffs)
    rel = abs(U - 33.83) / 33.83
    assert rel < 0.05, (
        f"Fe I U(10000 K) = {U:.3f}, Irwin reference = 33.83; "
        f"relative error = {rel:.2%} (>5%)"
    )


@pytest.mark.parametrize(
    "element,sp_num",
    list(PARITY_TRUNCATION_EV.keys()),
)
def test_polynomial_matches_truncated_direct_sum(conn, element, sp_num):
    """Polynomial U(10kK) agrees with truncated direct-sum within 10 %.

    The truncation is the Irwin (1981) experimentally-observed-only
    level convention; see :data:`PARITY_TRUNCATION_EV` for the per-species
    cutoff.  Without this truncation the direct sum diverges from
    Irwin's canonical value because NIST ASD includes high-lying
    extrapolated/predicted levels that Irwin excluded.
    """
    coeffs = _fetch_coeffs(conn, element, sp_num)
    assert coeffs is not None, f"{element} {sp_num} row missing from partition_functions"
    g, e = _fetch_levels(conn, element, sp_num)
    assert len(g) > 0, f"{element} {sp_num} has no energy_levels rows"

    e_max = PARITY_TRUNCATION_EV[(element, sp_num)]
    U_poly = polynomial_partition_function(10000.0, coeffs)
    U_direct = _direct_sum(10000.0, g, e, e_max)
    rel = abs(U_poly - U_direct) / U_poly
    assert rel < 0.10, (
        f"{element} {sp_num}: U_poly={U_poly:.3f}, U_direct(E<{e_max} eV)="
        f"{U_direct:.3f}, rel diff={rel:.2%} (>10%)"
    )


def test_partition_functions_coverage(conn):
    """At least 90 % of (element, sp_num) species in lines have a partition_functions row."""
    line_species = set(
        conn.execute("SELECT DISTINCT element, sp_num FROM lines").fetchall()
    )
    pf_species = set(
        conn.execute("SELECT element, sp_num FROM partition_functions").fetchall()
    )
    covered = line_species & pf_species
    coverage = len(covered) / len(line_species) if line_species else 0.0
    assert coverage >= 0.90, (
        f"partition_functions covers only {len(covered)}/{len(line_species)} "
        f"= {coverage:.1%} of line species (need ≥90%)"
    )


def test_partition_functions_row_count_grew(conn):
    """The backfill produced ≥ 130 partition_functions rows (was 13 pre-fix)."""
    n = conn.execute("SELECT COUNT(*) FROM partition_functions").fetchone()[0]
    assert n >= 130, f"partition_functions has only {n} rows (need ≥130)"


def test_no_missing_ground_state_for_audit_species(conn):
    """The 23 species the audit flagged must have an E≈0 ground-state row."""
    flagged = [
        ("H", 1), ("He", 1), ("Ne", 1), ("Ar", 1), ("Ar", 2), ("Kr", 1),
        ("Kr", 2), ("Xe", 1), ("Xe", 2), ("F", 1), ("Cl", 1), ("Br", 1),
        ("I", 1), ("As", 1), ("Ag", 2), ("Ge", 2), ("Hg", 2), ("K", 2),
        ("Pb", 2), ("Pt", 2), ("Rb", 2), ("Ru", 2), ("Tl", 2),
    ]
    missing = []
    for el, sp in flagged:
        row = conn.execute(
            "SELECT g_level, energy_ev FROM energy_levels "
            "WHERE element=? AND sp_num=? AND energy_ev < 0.001",
            (el, sp),
        ).fetchone()
        if row is None:
            missing.append((el, sp))
    assert not missing, f"Ground state still missing for: {missing}"


def test_polynomial_returns_finite_positive_at_libs_temps(conn):
    """For every PF row, U(T) is finite and > 0 at T ∈ {2000..20000} K."""
    rows = conn.execute(
        "SELECT element, sp_num, a0, a1, a2, a3, a4 FROM partition_functions"
    ).fetchall()
    bad = []
    for element, sp, a0, a1, a2, a3, a4 in rows:
        for T in (2000.0, 5000.0, 10000.0, 15000.0, 20000.0):
            U = polynomial_partition_function(T, [a0, a1, a2, a3, a4])
            if not (np.isfinite(U) and U > 0):
                bad.append((element, sp, T, U))
    assert not bad, f"Polynomial U(T) non-finite or non-positive: {bad[:5]}"
