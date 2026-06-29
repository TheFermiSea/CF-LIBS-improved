#!/usr/bin/env python
"""Fill missing ionization potentials in species_physics from live NIST ASD.

The DB had energy levels for 273 (element,stage) species but ip_ev for only
174 -- 99 gaps (mostly stage III), which break the Saha balance and the
partition autoionizing cutoff (E < IP). NIST has them all; this fetches via the
(now-fixed) datagen_v2.fetch_ionization_potential and upserts species_physics.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

_ROMAN = {1: "I", 2: "II", 3: "III"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="ASD_da/libs_production.db")
    p.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    import datagen_v2 as d
    from cflibs.atomic.masses import STANDARD_ATOMIC_MASSES

    conn = sqlite3.connect(args.db, timeout=120)
    have_lv = {
        (r[0], r[1])
        for r in conn.execute(
            "SELECT DISTINCT element, sp_num FROM energy_levels WHERE sp_num IN (1,2,3)"
        )
    }
    ip = {
        (r[0], r[1]): r[2]
        for r in conn.execute("SELECT element, sp_num, ip_ev FROM species_physics")
    }
    gaps = sorted(k for k in have_lv if k[1] in args.stages and not ip.get(k))
    print(f"{len(gaps)} species with levels but no ip_ev -> fetching from NIST")

    filled, failed = 0, []
    for el, st in gaps:
        roman = _ROMAN.get(st)
        if roman is None:
            continue
        try:
            val = d.fetch_ionization_potential(el, roman)
        except Exception as exc:
            failed.append((el, roman, repr(exc)[:50]))
            continue
        if not val or val <= 0:
            failed.append((el, roman, "no IP from NIST"))
            continue
        mass = float(STANDARD_ATOMIC_MASSES.get(el, 0.0)) or None
        print(f"  {el} {roman}: ip_ev={val}" + (" [dry-run]" if args.dry_run else ""))
        if args.dry_run:
            continue
        # upsert (element, sp_num) preserving any existing atomic_mass
        existing = conn.execute(
            "SELECT atomic_mass FROM species_physics WHERE element=? AND sp_num=?",
            (el, st),
        ).fetchone()
        if existing is not None:
            conn.execute(
                "UPDATE species_physics SET ip_ev=? WHERE element=? AND sp_num=?",
                (val, el, st),
            )
        else:
            conn.execute(
                "INSERT INTO species_physics (element, sp_num, ip_ev, atomic_mass) VALUES (?,?,?,?)",
                (el, st, val, mass),
            )
        filled += 1

    if not args.dry_run:
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError as exc:
            print(f"  WARNING: WAL checkpoint skipped ({exc})")
    conn.close()
    print(f"\nFilled {filled} IPs." + (f" Failed: {failed}" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
