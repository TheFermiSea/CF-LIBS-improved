#!/usr/bin/env python3
"""Backfill missing ground-state rows in ``energy_levels``.

The audit at /tmp/db_audit/aux_atomic.md flagged 23 species whose
``energy_levels`` table has lowest E_min > 0.5 eV — meaning the dominant
ground-state population term is silently absent from any direct-sum
partition function.  This script adds the published NIST ground-state
``(g_level = 2J+1, energy_ev = 0.0)`` row for each affected species
using ``INSERT OR IGNORE`` so re-runs are idempotent.

Sources
-------
NIST Atomic Spectra Database (ASD) — ground-state term symbols and
total angular momenta J for each species.  The ``g_level`` column
stores the statistical weight (2J+1).

Usage
-----
    python scripts/add_missing_ground_states.py --db ASD_da/libs_production.db
    python scripts/add_missing_ground_states.py --db ASD_da/libs_production.db --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# (element, sp_num): (g_ground, ground_term, NIST source notation)
# g_ground = 2J + 1 of the ground level.  All energies assumed 0.0 eV
# (ground level by definition).
GROUND_STATES: dict[tuple[str, int], tuple[int, str]] = {
    # Hydrogen / Helium (light atoms — ground-state-only EL is fine)
    ("H", 1): (2, "1s ²S_{1/2}"),  # J=1/2 → g=2
    ("He", 1): (1, "1s² ¹S_0"),  # J=0 → g=1
    # Noble gases (closed-shell or p^5/p^6)
    ("Ne", 1): (1, "2p⁶ ¹S_0"),  # J=0 → g=1
    ("Ar", 1): (1, "3p⁶ ¹S_0"),  # J=0 → g=1
    ("Ar", 2): (4, "3p⁵ ²P_{3/2}"),  # J=3/2 → g=4
    ("Kr", 1): (1, "4p⁶ ¹S_0"),  # J=0 → g=1
    ("Kr", 2): (4, "4p⁵ ²P_{3/2}"),  # J=3/2 → g=4
    ("Xe", 1): (1, "5p⁶ ¹S_0"),  # J=0 → g=1
    ("Xe", 2): (4, "5p⁵ ²P_{3/2}"),  # J=3/2 → g=4
    # Halogens (ns² np⁵ ²P_{3/2})
    ("F", 1): (4, "2p⁵ ²P_{3/2}"),  # J=3/2 → g=4
    ("Cl", 1): (4, "3p⁵ ²P_{3/2}"),  # J=3/2 → g=4
    ("Br", 1): (4, "4p⁵ ²P_{3/2}"),  # J=3/2 → g=4
    ("I", 1): (4, "5p⁵ ²P_{3/2}"),  # J=3/2 → g=4
    # Group V neutrals (ns² np³ ⁴S_{3/2})
    ("As", 1): (4, "4p³ ⁴S_{3/2}"),  # J=3/2 → g=4
    # Specific singly-ionized species the audit flagged with E_min >> 0.5 eV
    ("Ag", 2): (1, "4d¹⁰ ¹S_0"),  # J=0 → g=1
    ("Ge", 2): (2, "4s² 4p ²P_{1/2}"),  # J=1/2 → g=2
    ("Hg", 2): (2, "5d¹⁰ 6s ²S_{1/2}"),  # J=1/2 → g=2
    ("K", 2): (1, "3p⁶ ¹S_0"),  # K II is Ar-like → g=1
    ("Pb", 2): (2, "6s² 6p ²P_{1/2}"),  # J=1/2 → g=2
    ("Pt", 2): (6, "5d⁹ ²D_{5/2}"),  # J=5/2 → g=6
    ("Rb", 2): (1, "4p⁶ ¹S_0"),  # Rb II is Kr-like → g=1
    ("Ru", 2): (10, "4d⁷ ⁴F_{9/2}"),  # J=9/2 → g=10
    # Tl II Hg-like ¹S_0
    ("Tl", 2): (1, "6s² ¹S_0"),  # J=0 → g=1
}


def add_ground_states(db_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Insert missing ground-state rows.

    Returns
    -------
    (n_inserted, n_skipped) : (int, int)
        Number of rows actually added vs. skipped because they already
        existed (or because the species is absent from ``lines``).
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Get species that appear in lines (so we don't add ghost rows)
    line_species = set(
        (e, s) for e, s in cur.execute("SELECT DISTINCT element, sp_num FROM lines").fetchall()
    )

    n_inserted = 0
    n_skipped = 0
    for (element, sp_num), (g_ground, term) in sorted(GROUND_STATES.items()):
        if (element, sp_num) not in line_species:
            print(f"  SKIP {element:3s} {sp_num}: not in lines table")
            n_skipped += 1
            continue

        # Check if a ground-state row already exists (energy_ev < 0.001)
        existing = cur.execute(
            "SELECT g_level, energy_ev FROM energy_levels "
            "WHERE element=? AND sp_num=? AND energy_ev < 0.001",
            (element, sp_num),
        ).fetchone()
        if existing is not None:
            print(
                f"  SKIP {element:3s} {sp_num}: ground state already present "
                f"(g={existing[0]}, E={existing[1]})"
            )
            n_skipped += 1
            continue

        if dry_run:
            print(f"  WOULD INSERT {element:3s} {sp_num}: g={g_ground:>2d}  {term}")
        else:
            cur.execute(
                "INSERT OR IGNORE INTO energy_levels "
                "(element, sp_num, g_level, energy_ev) VALUES (?, ?, ?, 0.0)",
                (element, sp_num, g_ground),
            )
            print(f"  INSERTED {element:3s} {sp_num}: g={g_ground:>2d}  {term}")
        n_inserted += 1

    if not dry_run:
        conn.commit()
    conn.close()
    return n_inserted, n_skipped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default="ASD_da/libs_production.db")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        return 1

    print(f"Adding missing ground-state rows to {db_path}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'WRITE'}")
    print()

    n_ins, n_skip = add_ground_states(db_path, dry_run=args.dry_run)

    print()
    print(f"{'Inserted' if not args.dry_run else 'Would insert'}: {n_ins}")
    print(f"Skipped: {n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
