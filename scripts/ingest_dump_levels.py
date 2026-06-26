#!/usr/bin/env python
"""Top up + enrich energy_levels from the authoritative ASD dump.

Adds the dump's level metadata (level_id, J, term, config, Landé-g, uncertainty,
reference) to energy_levels and inserts the dump-only levels my live-NIST set
lacks (Nd/Lu/Rn ... ~1k), so the table matches the gold standard. Existing
energies/g (current live-NIST values) are kept; only metadata is enriched on a
match. Isotope-specific rows (element starts with a digit, e.g. '3He') are
skipped to preserve the codebase's element convention.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

sys.path.insert(0, "scripts")
from ingest_asd_dump_lines import _f, _strip_insert, parse_tuple  # noqa: E402

CM_PER_EV = 8065.544
META = [("level_id", "TEXT"), ("j_val", "TEXT"), ("term", "TEXT"),
        ("conf", "TEXT"), ("lande_g", "TEXT"), ("energy_unc", "TEXT"),
        ("level_ref", "TEXT")]


def _s(x):
    return x if (x is not None and str(x).strip() not in ("", "--")) else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="ASD_da/libs_production.db")
    p.add_argument("--dump", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    conn = sqlite3.connect(args.db, timeout=120)
    have = {r[1] for r in conn.execute("PRAGMA table_info(energy_levels)")}
    for name, typ in META:
        if name not in have:
            print(f"ALTER energy_levels ADD {name} {typ}")
            if not args.dry_run:
                conn.execute(f"ALTER TABLE energy_levels ADD COLUMN {name} {typ}")

    # existing levels keyed (element, stage, round(eV,3)) -> rowid
    existing = {}
    for rowid, el, sp, ev in conn.execute(
        "SELECT rowid, element, sp_num, energy_ev FROM energy_levels WHERE sp_num IN (1,2,3)"
    ):
        existing[(el, sp, round(ev, 3))] = rowid

    enrich, insert = [], []
    with open(args.dump, encoding="latin-1") as f:
        for line in f:
            body = _strip_insert(line, "ASD_Levels")
            if body is None:
                continue
            v = parse_tuple(body)
            if len(v) < 24:
                continue
            sc = str(v[1]).strip()
            if sc not in ("1", "2", "3"):
                continue
            el = v[0]
            if el is None or el[:1].isdigit():  # skip isotope rows
                continue
            e_cm = _f(v[12])
            g = _f(v[9])
            if e_cm is None:
                continue
            ev = e_cm / CM_PER_EV
            meta = (_s(v[23]), _s(v[8]), _s(v[3]), _s(v[2]), _s(v[13]), _s(v[21]), _s(v[27]))
            key = (el, int(sc), round(ev, 3))
            if key in existing:
                enrich.append((meta, existing[key]))
            else:
                insert.append((el, int(sc), g, ev, *meta))
    print(f"levels: enrich {len(enrich)}, insert {len(insert)} dump-only")
    if args.dry_run:
        return 0

    conn.executemany(
        "UPDATE energy_levels SET level_id=?, j_val=?, term=?, conf=?, lande_g=?, "
        "energy_unc=?, level_ref=? WHERE rowid=?",
        [(m[0], m[1], m[2], m[3], m[4], m[5], m[6], rid) for m, rid in enrich],
    )
    conn.executemany(
        "INSERT INTO energy_levels (element, sp_num, g_level, energy_ev, level_id, "
        "j_val, term, conf, lande_g, energy_unc, level_ref) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        insert,
    )
    conn.commit()
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError as exc:
        print(f"WARNING: WAL checkpoint skipped ({exc})")
    n = conn.execute("SELECT COUNT(*) FROM energy_levels WHERE sp_num IN (1,2,3)").fetchone()[0]
    conn.close()
    print(f"done. enriched {len(enrich)}, inserted {len(insert)}; levels I/II/III now {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
