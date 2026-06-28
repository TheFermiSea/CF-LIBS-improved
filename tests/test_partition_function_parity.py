"""Parity tests for the partition_functions table.

Verifies that the STORED polynomial in ``partition_functions`` is a faithful
fit to the FULL direct-sum over ``energy_levels`` (and, where available, to the
Barklem & Collet 2016 reference) at ps-LIBS temperatures.

History (composition-pipeline diagnosis 2026-06-03 § 2.6)
---------------------------------------------------------
This file previously asserted two now-known-incorrect things and therefore
passed **tautologically** while the stored polynomial was 30–60 % below the
true partition function for iron-group workhorse species:

1. ``test_fe_i_matches_irwin_reference`` checked Fe I U(10000 K) == 33.83 —
   that is Irwin (1981)'s *truncated, observed-levels-only* value, NOT the true
   partition function.  The full direct-sum / B&C16 value is ≈59.7.  Pinning
   the polynomial to 33.83 is exactly what made it too low.
2. ``test_polynomial_matches_truncated_direct_sum`` compared the polynomial to
   a *deliberately truncated* direct-sum (``PARITY_TRUNCATION_EV``) tuned to
   reproduce Irwin's value, so it could never catch the undershoot.

After the PF-1/PF-2 regeneration (``scripts/archive/migrations/regenerate_partition_functions.py``)
the stored polynomial tracks the FULL direct-sum, so these tests now assert the
correct physics: poly == full-direct-sum (a strict lower bound on the true U)
within tolerance, never *below* it, across the ps-LIBS band.

The coverage / row-count / ground-state / finiteness checks are unchanged.
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

# Barklem & Collet (2016) Fe I reference at 10000 K (A&A 588, A96).  The full
# partition function — the truncated Irwin value (33.83) was the source of the
# old too-low polynomial.
BC16_FE_I_10000 = 59.66

# Iron-group + light-metal workhorse species whose stored polynomial must track
# the FULL direct-sum (the regenerated `direct_sum_fit_v1` recipe target).
PARITY_SPECIES: list[tuple[str, int]] = [
    ("Fe", 1),
    ("Cr", 1),
    ("Ti", 1),
    ("Ca", 1),
    ("Mg", 1),
    ("Al", 1),
]

# ps-LIBS band temperatures for the parity comparison.
PARITY_TEMPS = (8000.0, 10000.0, 12000.0)


def _direct_sum(T_K: float, g: np.ndarray, e: np.ndarray, ip_ev: float) -> float:
    """Full U(T) = Σ g_i exp(-E_i/kT) for E_i < IP (no Irwin truncation)."""
    mask = e < ip_ev
    if not mask.any():
        return 1.0
    return max(float(np.sum(g[mask] * np.exp(-e[mask] / (KB_EV * T_K)))), 1.0)


def _fetch_ip(conn: sqlite3.Connection, element: str, sp_num: int) -> float:
    row = conn.execute(
        "SELECT ip_ev FROM species_physics WHERE element=? AND sp_num=?",
        (element, sp_num),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else 1e9


def _fetch_levels(conn: sqlite3.Connection, element: str, sp_num: int):
    # Match the production direct sum exactly.  ``AtomicDatabase.get_energy_levels``
    # filters levels with an unassigned J (NULL / non-positive ``g_level``) --
    # their degeneracy g = 2J+1 is unknown so they carry no known Boltzmann
    # weight -- and rounds g to the integer 2J+1.  The reference direct sum must
    # use the SAME levels the stored polynomial was fit against; otherwise NULL-g
    # rows (legitimate NIST metadata, e.g. high-lying Mg I / Al I levels) poison
    # the sum with NaN and the comparison is meaningless.
    rows = conn.execute(
        "SELECT g_level, energy_ev FROM energy_levels "
        "WHERE element=? AND sp_num=? "
        "AND g_level IS NOT NULL AND g_level > 0 AND energy_ev IS NOT NULL "
        "ORDER BY energy_ev",
        (element, sp_num),
    ).fetchall()
    g = np.array([round(float(r[0])) for r in rows], dtype=np.float64)
    e = np.array([float(r[1]) for r in rows], dtype=np.float64)
    return g, e


def _fetch_coeffs(conn: sqlite3.Connection, element: str, sp_num: int):
    row = conn.execute(
        "SELECT a0, a1, a2, a3, a4 FROM partition_functions " "WHERE element=? AND sp_num=?",
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


def test_fe_i_matches_bc16_reference(conn):
    """Fe I U(10000 K) via stored polynomial == B&C16 full value (59.66) ± 10 %.

    The OLD assertion pinned this to Irwin's *truncated* 33.83, which was the
    root cause of the too-low polynomial (diagnosis § 2.1/§ 2.6).  The full
    partition function is the Barklem & Collet 2016 value ≈59.7.
    """
    coeffs = _fetch_coeffs(conn, "Fe", 1)
    assert coeffs is not None, "Fe I row missing from partition_functions"
    U = polynomial_partition_function(10000.0, coeffs)
    rel = abs(U - BC16_FE_I_10000) / BC16_FE_I_10000
    assert rel < 0.10, (
        f"Fe I U(10000 K) = {U:.3f}, B&C16 reference = {BC16_FE_I_10000}; "
        f"relative error = {rel:.2%} (>10%)"
    )


@pytest.mark.parametrize("element,sp_num", PARITY_SPECIES)
def test_polynomial_matches_full_direct_sum(conn, element, sp_num):
    """Stored polynomial tracks the FULL direct-sum across the ps-LIBS band.

    Direct summation over ``energy_levels`` is a strict lower bound on the true
    partition function, so a correct polynomial fit must be within tolerance of
    it AND must NOT undershoot it (poly < direct-sum is physically impossible).
    This replaces the old truncated-direct-sum tautology (§ 2.6) that could
    never catch the 30–60 % undershoot the regeneration fixed.
    """
    coeffs = _fetch_coeffs(conn, element, sp_num)
    assert coeffs is not None, f"{element} {sp_num} row missing from partition_functions"
    g, e = _fetch_levels(conn, element, sp_num)
    assert len(g) > 0, f"{element} {sp_num} has no energy_levels rows"
    ip_ev = _fetch_ip(conn, element, sp_num)

    for T in PARITY_TEMPS:
        U_poly = polynomial_partition_function(T, coeffs)
        U_direct = _direct_sum(T, g, e, ip_ev)
        rel = (U_poly - U_direct) / U_direct
        # Within 10 % AND not below the direct-sum floor (allow tiny fit noise).
        assert rel >= -0.01, (
            f"{element} {sp_num} @ {T:.0f} K: U_poly={U_poly:.3f} BELOW "
            f"direct-sum floor {U_direct:.3f} (physically impossible)"
        )
        assert abs(rel) < 0.10, (
            f"{element} {sp_num} @ {T:.0f} K: U_poly={U_poly:.3f}, "
            f"U_direct={U_direct:.3f}, rel diff={rel:+.2%} (>10%)"
        )


def test_partition_functions_coverage(conn):
    """At least 90 % of (element, sp_num) species in lines have a partition_functions row."""
    line_species = set(conn.execute("SELECT DISTINCT element, sp_num FROM lines").fetchall())
    pf_species = set(conn.execute("SELECT element, sp_num FROM partition_functions").fetchall())
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
        ("H", 1),
        ("He", 1),
        ("Ne", 1),
        ("Ar", 1),
        ("Ar", 2),
        ("Kr", 1),
        ("Kr", 2),
        ("Xe", 1),
        ("Xe", 2),
        ("F", 1),
        ("Cl", 1),
        ("Br", 1),
        ("I", 1),
        ("As", 1),
        ("Ag", 2),
        ("Ge", 2),
        ("Hg", 2),
        ("K", 2),
        ("Pb", 2),
        ("Pt", 2),
        ("Rb", 2),
        ("Ru", 2),
        ("Tl", 2),
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
