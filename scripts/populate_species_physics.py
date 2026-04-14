#!/usr/bin/env python3
"""
Populate species_physics table to ensure every (element, sp_num) pair
in energy_levels has a non-NULL ip_ev and atomic_mass.

Workflow:
  1. Find (element, sp_num) pairs in energy_levels missing from species_physics
  2. Find rows in species_physics where ip_ev IS NULL
  3. Find rows in species_physics where atomic_mass IS NULL
  4. Populate missing/NULL ip_ev from NIST ionization energies
  5. Populate missing/NULL atomic_mass from standard atomic weights
  6. Verify every energy_levels species now has non-NULL ip_ev in species_physics

Usage:
  python scripts/populate_species_physics.py --db-path ASD_da/libs_production.db
  python scripts/populate_species_physics.py --db-path ASD_da/libs_production.db --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# NIST reference data
# ---------------------------------------------------------------------------
# Reuse the authoritative reference data already in the codebase.
# Import path may fail if cflibs is not installed; fall back to inline copy.
try:
    from cflibs.atomic.reference_data import NIST_SPECIES_PHYSICS
except ImportError:
    # Inline fallback so the script works standalone.
    NIST_SPECIES_PHYSICS = [
        ("H", 1.008, 13.598, None),
        ("He", 4.003, 24.587, 54.418),
        ("Li", 6.94, 5.392, 75.640),
        ("Be", 9.012, 9.323, 18.211),
        ("B", 10.81, 8.298, 25.155),
        ("C", 12.011, 11.260, 24.383),
        ("N", 14.007, 14.534, 29.601),
        ("O", 15.999, 13.618, 35.117),
        ("F", 18.998, 17.422, 34.970),
        ("Ne", 20.180, 21.564, 40.962),
        ("Na", 22.990, 5.139, 47.286),
        ("Mg", 24.305, 7.646, 15.035),
        ("Al", 26.982, 5.986, 18.828),
        ("Si", 28.085, 8.151, 16.345),
        ("P", 30.974, 10.486, 19.769),
        ("S", 32.06, 10.360, 23.337),
        ("Cl", 35.45, 12.967, 23.814),
        ("Ar", 39.948, 15.759, 27.629),
        ("K", 39.098, 4.341, 31.63),
        ("Ca", 40.078, 6.113, 11.871),
        ("Sc", 44.956, 6.561, 12.800),
        ("Ti", 47.867, 6.828, 13.575),
        ("V", 50.942, 6.746, 14.618),
        ("Cr", 51.996, 6.766, 16.498),
        ("Mn", 54.938, 7.434, 15.640),
        ("Fe", 55.845, 7.902, 16.187),
        ("Co", 58.933, 7.881, 17.083),
        ("Ni", 58.693, 7.639, 18.168),
        ("Cu", 63.546, 7.726, 20.292),
        ("Zn", 65.38, 9.394, 17.964),
        ("Ga", 69.723, 5.999, 20.514),
        ("Ge", 72.63, 7.899, 15.934),
        ("As", 74.922, 9.788, 18.633),
        ("Se", 78.971, 9.752, 21.19),
        ("Br", 79.904, 11.814, 21.8),
        ("Kr", 83.798, 13.999, 24.359),
        ("Rb", 85.468, 4.177, 27.285),
        ("Sr", 87.62, 5.695, 11.030),
        ("Y", 88.906, 6.217, 12.24),
        ("Zr", 91.224, 6.634, 13.13),
        ("Nb", 92.906, 6.759, 14.32),
        ("Mo", 95.95, 7.092, 16.16),
        ("Ru", 101.07, 7.361, 16.76),
        ("Rh", 102.91, 7.459, 18.08),
        ("Pd", 106.42, 8.337, 19.43),
        ("Ag", 107.87, 7.576, 21.49),
        ("Cd", 112.41, 8.993, 16.908),
        ("In", 114.82, 5.786, 18.869),
        ("Sn", 118.71, 7.344, 14.632),
        ("Sb", 121.76, 8.608, 16.53),
        ("Te", 127.60, 9.010, 18.60),
        ("I", 126.90, 10.451, 19.131),
        ("Xe", 131.29, 12.130, 20.975),
        ("Cs", 132.91, 3.894, 23.157),
        ("Ba", 137.33, 5.211, 10.004),
        ("La", 138.91, 5.577, 11.06),
        ("Ce", 140.12, 5.539, 10.85),
        ("Pr", 140.91, 5.473, 10.55),
        ("Nd", 144.24, 5.525, 10.72),
        ("Sm", 150.36, 5.643, 11.07),
        ("Eu", 151.96, 5.670, 11.241),
        ("Gd", 157.25, 6.150, 12.09),
        ("Tb", 158.93, 5.864, 11.52),
        ("Dy", 162.50, 5.939, 11.67),
        ("Ho", 164.93, 6.022, 11.80),
        ("Er", 167.26, 6.108, 11.93),
        ("Tm", 168.93, 6.184, 12.05),
        ("Yb", 173.05, 6.254, 12.176),
        ("Lu", 174.97, 5.426, 13.9),
        ("Hf", 178.49, 6.825, 14.9),
        ("Ta", 180.95, 7.549, 16.2),
        ("W", 183.84, 7.864, 16.1),
        ("Re", 186.21, 7.834, 16.6),
        ("Os", 190.23, 8.438, 17.0),
        ("Ir", 192.22, 8.967, 17.0),
        ("Pt", 195.08, 8.959, 18.563),
        ("Au", 196.97, 9.225, 20.5),
        ("Hg", 200.59, 10.437, 18.756),
        ("Tl", 204.38, 6.108, 20.428),
        ("Pb", 207.2, 7.416, 15.032),
        ("Bi", 208.98, 7.285, 16.69),
        ("Th", 232.04, 6.307, 11.5),
        ("U", 238.03, 6.194, 11.6),
    ]

# Build lookup: (element) -> (mass, ip_I, ip_II)
_REF = {elem: (mass, ip1, ip2) for elem, mass, ip1, ip2 in NIST_SPECIES_PHYSICS}

# Third ionization potentials (eV) from NIST ASD
# Needed when energy_levels contains sp_num=3 (doubly ionized) species.
# The IP for sp_num=3 is the energy to ionize the doubly-ionized species (IP_III).
THIRD_IP: dict[str, float] = {
    "H": 0.0,  # H has no third IP
    "He": 0.0,
    "Li": 122.454,
    "Be": 153.896,
    "B": 37.930,
    "C": 47.888,
    "N": 47.449,
    "O": 54.936,
    "F": 62.708,
    "Ne": 63.423,
    "Na": 71.620,
    "Mg": 80.143,
    "Al": 28.447,
    "Si": 33.493,
    "P": 30.202,
    "S": 34.79,
    "Cl": 39.61,
    "Ar": 40.74,
    "K": 45.806,
    "Ca": 50.913,
    "Sc": 24.757,
    "Ti": 27.491,
    "V": 29.311,
    "Cr": 30.96,
    "Mn": 33.668,
    "Fe": 30.651,
    "Co": 33.50,
    "Ni": 35.19,
    "Cu": 36.841,
    "Zn": 39.723,
    "Ga": 30.710,
    "Ge": 34.224,
    "As": 28.351,
    "Se": 30.820,
    "Br": 36.0,
    "Kr": 36.95,
    "Rb": 40.0,
    "Sr": 42.89,
    "Y": 20.52,
    "Zr": 22.99,
    "Nb": 25.04,
    "Mo": 27.13,
    "Ru": 28.47,
    "Rh": 31.06,
    "Pd": 32.93,
    "Ag": 34.83,
    "Cd": 37.48,
    "In": 28.03,
    "Sn": 30.502,
    "Sb": 25.3,
    "Te": 27.96,
    "I": 33.0,
    "Xe": 31.05,
    "Cs": 33.2,
    "Ba": 35.84,
    "La": 19.177,
    "Ce": 20.198,
    "Pr": 21.624,
    "Nd": 22.1,
    "Sm": 23.4,
    "Eu": 24.92,
    "Gd": 20.63,
    "Tb": 21.91,
    "Dy": 22.8,
    "Ho": 22.84,
    "Er": 22.74,
    "Tm": 23.68,
    "Yb": 25.05,
    "Lu": 20.96,
    "Hf": 23.3,
    "Ta": 22.0,
    "W": 24.0,
    "Re": 26.0,
    "Os": 25.0,
    "Ir": 27.0,
    "Pt": 28.0,
    "Au": 30.0,
    "Hg": 34.2,
    "Tl": 29.83,
    "Pb": 31.937,
    "Bi": 25.56,
    "Th": 18.3,
    "U": 19.8,
}

# Standard atomic weights lookup (same as in NIST_SPECIES_PHYSICS)
ATOMIC_MASS: dict[str, float] = {elem: mass for elem, mass, _, _ in NIST_SPECIES_PHYSICS}


def get_ip_for_species(element: str, sp_num: int) -> float | None:
    """Look up the ionization potential for a given (element, sp_num).

    sp_num=1 (neutral)  -> first IP
    sp_num=2 (singly ionized) -> second IP
    sp_num=3 (doubly ionized) -> third IP
    """
    ref = _REF.get(element)
    if sp_num == 1:
        return ref[1] if ref else None
    elif sp_num == 2:
        return ref[2] if ref else None
    elif sp_num == 3:
        return THIRD_IP.get(element)
    return None


def get_mass_for_element(element: str) -> float | None:
    """Look up the standard atomic mass for an element."""
    return ATOMIC_MASS.get(element)


# ---------------------------------------------------------------------------
# Audit and repair
# ---------------------------------------------------------------------------


def audit_database(conn: sqlite3.Connection) -> dict:
    """Audit species_physics against energy_levels.

    Returns a dict with lists of issues found.
    """
    cur = conn.cursor()

    # 1. Species in energy_levels missing from species_physics entirely
    cur.execute(
        """
        SELECT DISTINCT e.element, e.sp_num
        FROM energy_levels e
        LEFT JOIN species_physics sp
            ON e.element = sp.element AND e.sp_num = sp.sp_num
        WHERE sp.element IS NULL
        ORDER BY e.element, e.sp_num
    """
    )
    missing_species = cur.fetchall()

    # 2. Rows in species_physics where ip_ev IS NULL
    cur.execute(
        """
        SELECT element, sp_num
        FROM species_physics
        WHERE ip_ev IS NULL
        ORDER BY element, sp_num
    """
    )
    null_ip = cur.fetchall()

    # 3. Rows in species_physics where atomic_mass IS NULL
    cur.execute(
        """
        SELECT element, sp_num
        FROM species_physics
        WHERE atomic_mass IS NULL
        ORDER BY element, sp_num
    """
    )
    null_mass = cur.fetchall()

    return {
        "missing_species": missing_species,
        "null_ip": null_ip,
        "null_mass": null_mass,
    }


def populate_missing(conn: sqlite3.Connection, dry_run: bool = False) -> dict:
    """Insert missing rows and fill NULL ip_ev / atomic_mass values.

    Returns a summary of changes made.
    """
    audit = audit_database(conn)
    cur = conn.cursor()
    stats = {
        "inserted": 0,
        "ip_fixed": 0,
        "mass_fixed": 0,
        "ip_unfixable": [],
        "mass_unfixable": [],
    }

    # --- 1. Insert missing (element, sp_num) rows ---
    for element, sp_num in audit["missing_species"]:
        ip = get_ip_for_species(element, sp_num)
        mass = get_mass_for_element(element)
        if ip is None:
            stats["ip_unfixable"].append((element, sp_num))
            print(f"  WARNING: No IP found for {element} sp_num={sp_num}")
        if mass is None:
            stats["mass_unfixable"].append((element, sp_num))
            print(f"  WARNING: No atomic_mass found for {element}")
        print(f"  INSERT {element} sp_num={sp_num}  ip_ev={ip}  mass={mass}")
        if not dry_run:
            cur.execute(
                "INSERT OR REPLACE INTO species_physics "
                "(element, sp_num, ip_ev, atomic_mass) VALUES (?, ?, ?, ?)",
                (element, sp_num, ip, mass),
            )
        stats["inserted"] += 1

    # --- 2. Fix NULL ip_ev ---
    for element, sp_num in audit["null_ip"]:
        ip = get_ip_for_species(element, sp_num)
        if ip is None:
            if (element, sp_num) not in stats["ip_unfixable"]:
                stats["ip_unfixable"].append((element, sp_num))
            print(f"  WARNING: Cannot resolve IP for {element} sp_num={sp_num}")
            continue
        print(f"  UPDATE {element} sp_num={sp_num}  ip_ev <- {ip}")
        if not dry_run:
            cur.execute(
                "UPDATE species_physics SET ip_ev = ? " "WHERE element = ? AND sp_num = ?",
                (ip, element, sp_num),
            )
        stats["ip_fixed"] += 1

    # --- 3. Fix NULL atomic_mass ---
    for element, sp_num in audit["null_mass"]:
        mass = get_mass_for_element(element)
        if mass is None:
            if (element, sp_num) not in stats["mass_unfixable"]:
                stats["mass_unfixable"].append((element, sp_num))
            print(f"  WARNING: Cannot resolve mass for {element}")
            continue
        print(f"  UPDATE {element} sp_num={sp_num}  atomic_mass <- {mass}")
        if not dry_run:
            cur.execute(
                "UPDATE species_physics SET atomic_mass = ? " "WHERE element = ? AND sp_num = ?",
                (mass, element, sp_num),
            )
        stats["mass_fixed"] += 1

    if not dry_run:
        conn.commit()

    return stats


def verify(conn: sqlite3.Connection) -> bool:
    """Verify every (element, sp_num) in energy_levels has non-NULL ip_ev."""
    cur = conn.cursor()

    # Check for missing species
    cur.execute(
        """
        SELECT DISTINCT e.element, e.sp_num
        FROM energy_levels e
        LEFT JOIN species_physics sp
            ON e.element = sp.element AND e.sp_num = sp.sp_num
        WHERE sp.element IS NULL
        ORDER BY e.element, e.sp_num
    """
    )
    missing = cur.fetchall()
    if missing:
        print(
            f"\nFAIL: {len(missing)} species in energy_levels still missing "
            f"from species_physics:"
        )
        for elem, sp in missing:
            print(f"  {elem} sp_num={sp}")
        return False

    # Check for NULL ip_ev among energy_levels species
    cur.execute(
        """
        SELECT DISTINCT e.element, e.sp_num
        FROM energy_levels e
        JOIN species_physics sp
            ON e.element = sp.element AND e.sp_num = sp.sp_num
        WHERE sp.ip_ev IS NULL
        ORDER BY e.element, e.sp_num
    """
    )
    null_ip = cur.fetchall()
    if null_ip:
        print(
            f"\nFAIL: {len(null_ip)} species in energy_levels have NULL ip_ev "
            f"in species_physics:"
        )
        for elem, sp in null_ip:
            print(f"  {elem} sp_num={sp}")
        return False

    # Check for NULL atomic_mass among energy_levels species
    cur.execute(
        """
        SELECT DISTINCT e.element, e.sp_num
        FROM energy_levels e
        JOIN species_physics sp
            ON e.element = sp.element AND e.sp_num = sp.sp_num
        WHERE sp.atomic_mass IS NULL
        ORDER BY e.element, e.sp_num
    """
    )
    null_mass = cur.fetchall()
    if null_mass:
        print(
            f"\nWARN: {len(null_mass)} species in energy_levels have NULL "
            f"atomic_mass in species_physics:"
        )
        for elem, sp in null_mass:
            print(f"  {elem} sp_num={sp}")
        # Not a hard failure, but warn
        return False

    # Summary counts
    cur.execute("SELECT COUNT(DISTINCT element || '|' || sp_num) FROM energy_levels")
    n_el = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM species_physics")
    n_sp = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM species_physics WHERE ip_ev IS NOT NULL "
        "AND atomic_mass IS NOT NULL"
    )
    n_complete = cur.fetchone()[0]

    print("\nVERIFICATION PASSED")
    print(f"  energy_levels species:  {n_el}")
    print(f"  species_physics rows:   {n_sp}")
    print(f"  fully populated rows:   {n_complete}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Populate species_physics table with NIST ionization "
        "potentials and atomic masses for all species in energy_levels."
    )
    parser.add_argument(
        "--db-path",
        default="ASD_da/libs_production.db",
        help="Path to the SQLite database (default: ASD_da/libs_production.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without modifying the database.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    # Ensure atomic_mass column exists (schema migration)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(species_physics)")
    columns = {row[1] for row in cur.fetchall()}
    if "atomic_mass" not in columns:
        print("Adding atomic_mass column to species_physics...")
        if not args.dry_run:
            cur.execute("ALTER TABLE species_physics ADD COLUMN atomic_mass REAL")
            conn.commit()

    # --- Audit ---
    print("=" * 60)
    print("AUDIT: species_physics vs energy_levels")
    print("=" * 60)
    audit = audit_database(conn)

    print(f"\n  Missing from species_physics:  {len(audit['missing_species'])}")
    for elem, sp in audit["missing_species"]:
        print(f"    {elem} sp_num={sp}")

    print(f"\n  NULL ip_ev:                    {len(audit['null_ip'])}")
    for elem, sp in audit["null_ip"]:
        print(f"    {elem} sp_num={sp}")

    print(f"\n  NULL atomic_mass:              {len(audit['null_mass'])}")
    for elem, sp in audit["null_mass"]:
        print(f"    {elem} sp_num={sp}")

    total_issues = len(audit["missing_species"]) + len(audit["null_ip"]) + len(audit["null_mass"])
    if total_issues == 0:
        print("\nNo issues found.")
        verify(conn)
        conn.close()
        return

    # --- Populate ---
    mode = "DRY RUN" if args.dry_run else "FIXING"
    print(f"\n{'=' * 60}")
    print(f"{mode}: Populating missing data")
    print(f"{'=' * 60}")
    stats = populate_missing(conn, dry_run=args.dry_run)

    print(f"\n  Rows inserted:       {stats['inserted']}")
    print(f"  IPs fixed:           {stats['ip_fixed']}")
    print(f"  Masses fixed:        {stats['mass_fixed']}")
    if stats["ip_unfixable"]:
        print(f"  IP unfixable:        {stats['ip_unfixable']}")
    if stats["mass_unfixable"]:
        print(f"  Mass unfixable:      {stats['mass_unfixable']}")

    # --- Verify ---
    if not args.dry_run:
        print(f"\n{'=' * 60}")
        print("VERIFICATION")
        print(f"{'=' * 60}")
        ok = verify(conn)
        if not ok:
            print("\nSome issues remain. Re-run or check manually.")
            conn.close()
            sys.exit(1)

    conn.close()


if __name__ == "__main__":
    main()
