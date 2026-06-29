#!/usr/bin/env python
"""Ingest NIST ASD energy levels for species missing/incomplete in the local DB.

The shipped ``ASD_da/libs_production.db`` was found to be grossly incomplete for
higher ionization stages and several alloy elements (e.g. every stage-III had
0-1 levels; Nb/Zr had none; Mo II had 1; Fe III had 1). Those species fall back
to a generic partition function, corrupting the Saha balance for metal-alloy
inversions. NIST ASD has the data (hundreds of levels each) -- this script
fetches it via the existing ``datagen_v2.fetch_energy_levels`` scraper (which
handles the NIST energy1.pl quirks) and repopulates ``energy_levels`` for any
species below a level-count threshold.

Usage::

    PYTHONPATH=$PWD python scripts/ingest_nist_levels.py \
        --db ASD_da/libs_production.db --elements Ti Al V Cr Mn Fe Co Ni Cu Mo Nb Zr W \
        --stages 1 2 3 --min-levels 10

After ingest, regenerate the partition functions
(``scripts/regenerate_partition_functions.py`` or ``fit_partition_coefficients``)
so the new levels are used. A backup ``<db>.bak-preIngest`` should be made first.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}

#: All elements H(1) .. U(92), so --all-elements needs no fragile shell expansion.
ALL_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu "
    "Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs "
    "Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl "
    "Pb Bi Po At Rn Fr Ra Ac Th Pa U"
).split()


def audit(conn: sqlite3.Connection, elements, stages):
    """Return {(element, stage): count} of current energy_levels rows."""
    out = {}
    for el in elements:
        for st in stages:
            n = conn.execute(
                "SELECT COUNT(*) FROM energy_levels WHERE element=? AND sp_num=?",
                (el, st),
            ).fetchone()[0]
            out[(el, st)] = n
    return out


def ingest(db_path: str, elements, stages, min_levels: int, dry_run: bool = False):
    import datagen_v2 as d  # reuses the NIST energy1.pl scraper + parser

    conn = sqlite3.connect(db_path, timeout=120)
    before = audit(conn, elements, stages)
    targets = [(el, st) for (el, st), n in before.items() if n < min_levels]
    print(f"{len(targets)} species below {min_levels} levels -> fetching from NIST ASD")

    fetched, failed = 0, []
    for el, st in sorted(targets):
        roman = _ROMAN[st]
        try:
            levels = d.fetch_energy_levels(el, roman)
        except Exception as exc:  # network/parse failure -> keep existing
            failed.append((el, roman, repr(exc)[:60]))
            continue
        # Require a real improvement before touching the DB.
        if len(levels) <= before[(el, st)]:
            print(
                f"  {el} {roman}: NIST returned {len(levels)} (<= local {before[(el, st)]}); skip"
            )
            continue
        print(
            f"  {el} {roman}: {before[(el, st)]} -> {len(levels)} levels"
            + (" [dry-run]" if dry_run else "")
        )
        if dry_run:
            continue
        # Replace this species' levels atomically (delete then insert).
        conn.execute("DELETE FROM energy_levels WHERE element=? AND sp_num=?", (el, st))
        conn.executemany(
            "INSERT INTO energy_levels (element, sp_num, g_level, energy_ev) VALUES (?,?,?,?)",
            [(el, st, int(g), float(e)) for (g, e) in levels],
        )
        fetched += 1

    if not dry_run:
        conn.commit()
        # Fold the WAL into the main .db so a subsequent `git add` of the file
        # captures the new rows (SQLite defaults to WAL; without this the data
        # sits in a transient -wal sidecar and the committed .db is stale).
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError as exc:  # contended; warn, don't fail
            print(f"  WARNING: WAL checkpoint skipped ({exc}); run it before committing the DB")
    conn.close()
    print(f"\nIngested {fetched} species." + (f" Failed: {failed}" if failed else ""))
    return fetched, failed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="ASD_da/libs_production.db")
    p.add_argument(
        "--elements",
        nargs="+",
        default=["Ti", "Al", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "Nb", "Zr", "W"],
    )
    p.add_argument(
        "--all-elements",
        action="store_true",
        help="ingest every element H..U (overrides --elements; no shell expansion)",
    )
    p.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--min-levels", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.all_elements:
        args.elements = ALL_ELEMENTS
    if not Path(args.db).exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 2
    ingest(args.db, args.elements, args.stages, args.min_levels, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
