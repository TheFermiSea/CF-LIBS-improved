#!/usr/bin/env python3
"""Remove autoionizing energy levels (energy_ev >= IP) from the atomic database.

Autoionizing levels sit above the ionization potential and should not appear
in CF-LIBS Boltzmann / Saha analysis.  This script:

1. Reports orphaned (element, sp_num) pairs in energy_levels with no
   corresponding row in species_physics.
2. Identifies all energy levels where energy_ev >= ip_ev.
3. Reports per-species counts of autoionizing levels.
4. Deletes them from the database.
5. Verifies the deletion succeeded.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "ASD_da" / "libs_production.db"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit and remove autoionizing levels from libs_production.db"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB,
        help="Path to the SQLite database (default: ASD_da/libs_production.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only; do not delete anything.",
    )
    args = parser.parse_args(argv)

    db_path: Path = args.db_path
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()

    # ------------------------------------------------------------------
    # 0. Baseline counts
    # ------------------------------------------------------------------
    (total_before,) = cur.execute("SELECT COUNT(*) FROM energy_levels").fetchone()
    (species_count,) = cur.execute("SELECT COUNT(*) FROM species_physics").fetchone()
    print(f"Database: {db_path}")
    print(f"Total energy levels before cleanup : {total_before}")
    print(f"Species in species_physics          : {species_count}")
    print()

    # ------------------------------------------------------------------
    # 1. Orphaned (element, sp_num) — no corresponding species_physics row
    # ------------------------------------------------------------------
    orphans = cur.execute(
        """
        SELECT DISTINCT el.element, el.sp_num, COUNT(*) as level_count
        FROM energy_levels el
        LEFT JOIN species_physics sp
            ON el.element = sp.element AND el.sp_num = sp.sp_num
        WHERE sp.element IS NULL
        GROUP BY el.element, el.sp_num
        ORDER BY el.element, el.sp_num
        """
    ).fetchall()

    if orphans:
        print("=== ORPHANED SPECIES (no IP in species_physics) ===")
        for element, sp_num, count in orphans:
            print(f"  {element} {sp_num}: {count} levels (no IP to compare)")
        print(f"  Total orphaned species: {len(orphans)}")
    else:
        print(
            "No orphaned species found — every (element, sp_num) in energy_levels "
            "has a species_physics row."
        )
    print()

    # ------------------------------------------------------------------
    # 2. Identify autoionizing levels (energy_ev >= ip_ev)
    # ------------------------------------------------------------------
    autoionizing = cur.execute(
        """
        SELECT el.element, el.sp_num, sp.ip_ev, COUNT(*) as n_auto,
               MIN(el.energy_ev) as min_energy, MAX(el.energy_ev) as max_energy
        FROM energy_levels el
        JOIN species_physics sp
            ON el.element = sp.element AND el.sp_num = sp.sp_num
        WHERE el.energy_ev >= sp.ip_ev
        GROUP BY el.element, el.sp_num
        ORDER BY n_auto DESC
        """
    ).fetchall()

    total_auto = sum(row[3] for row in autoionizing)

    if not autoionizing:
        print("No autoionizing levels found. Database is clean.")
        conn.close()
        return 0

    print("=== AUTOIONIZING LEVELS (energy_ev >= IP) ===")
    print(
        f"{'Element':>8} {'Ion':>4} {'IP (eV)':>10} {'Count':>6} "
        f"{'Min E (eV)':>12} {'Max E (eV)':>12}"
    )
    print("-" * 60)
    for element, sp_num, ip_ev, n_auto, min_e, max_e in autoionizing:
        print(
            f"{element:>8} {sp_num:>4} {ip_ev:>10.4f} {n_auto:>6} " f"{min_e:>12.4f} {max_e:>12.4f}"
        )
    print("-" * 60)
    print(f"Total autoionizing levels to remove: {total_auto}")
    print(f"Fraction of database: {total_auto / total_before:.2%}")
    print()

    if args.dry_run:
        print("[DRY RUN] No changes made.")
        conn.close()
        return 0

    # ------------------------------------------------------------------
    # 3. Delete autoionizing levels
    # ------------------------------------------------------------------
    cur.execute(
        """
        DELETE FROM energy_levels
        WHERE rowid IN (
            SELECT el.rowid
            FROM energy_levels el
            JOIN species_physics sp
                ON el.element = sp.element AND el.sp_num = sp.sp_num
            WHERE el.energy_ev >= sp.ip_ev
        )
        """
    )
    deleted = cur.rowcount
    conn.commit()
    print(f"Deleted {deleted} autoionizing levels.")

    # ------------------------------------------------------------------
    # 4. Verify deletion
    # ------------------------------------------------------------------
    (remaining_auto,) = cur.execute(
        """
        SELECT COUNT(*)
        FROM energy_levels el
        JOIN species_physics sp
            ON el.element = sp.element AND el.sp_num = sp.sp_num
        WHERE el.energy_ev >= sp.ip_ev
        """
    ).fetchone()

    (total_after,) = cur.execute("SELECT COUNT(*) FROM energy_levels").fetchone()

    print()
    print("=== VERIFICATION ===")
    print(f"Autoionizing levels remaining: {remaining_auto}")
    print(f"Total energy levels before   : {total_before}")
    print(f"Total energy levels after     : {total_after}")
    print(f"Levels removed                : {total_before - total_after}")

    if remaining_auto != 0:
        print("ERROR: autoionizing levels still present!", file=sys.stderr)
        conn.close()
        return 1

    if total_before - total_after != deleted:
        print("ERROR: count mismatch between deleted and before/after!", file=sys.stderr)
        conn.close()
        return 1

    print("Verification passed: all autoionizing levels removed successfully.")

    # ------------------------------------------------------------------
    # 5. Vacuum to reclaim space
    # ------------------------------------------------------------------
    conn.execute("VACUUM")
    conn.close()
    print("Database vacuumed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
