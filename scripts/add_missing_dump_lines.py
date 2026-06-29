#!/usr/bin/env python
"""Add dump lines missing from the DB, deduped by TRANSITION IDENTITY.

The wavelength-rounding dedup dropped closely-spaced hydrogenic lines. The
authoritative identity of a line is its transition (low+upp level id), which the
dump carries. This inserts every ASD_Lines transition not already in the DB
(resolved lines keyed on (element,stage,low,upp); unresolved lines keyed on
fine wavelength), so the union reaches the gold standard without losing the
current-NIST lines or their derived Stark fields already in the table.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

sys.path.insert(0, "scripts")
from ingest_asd_dump_lines import _f, _strip_insert, parse_tuple  # noqa: E402

CM_PER_EV = 8065.544


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="ASD_da/libs_production.db")
    p.add_argument("--dump", required=True)
    args = p.parse_args(argv)

    # level map
    level = {}
    with open(args.dump, encoding="latin-1") as f:
        for line in f:
            body = _strip_insert(line, "ASD_Levels")
            if body is None:
                continue
            v = parse_tuple(body)
            if len(v) < 24 or not v[23]:
                continue
            e = _f(v[12])
            level[v[23]] = (e / CM_PER_EV if e is not None else None, _f(v[9]))

    conn = sqlite3.connect(args.db, timeout=120)
    have_trans = {
        (r[0], r[1], r[2], r[3])
        for r in conn.execute(
            "SELECT element, sp_num, low_level_id, upp_level_id FROM lines "
            "WHERE low_level_id IS NOT NULL AND upp_level_id IS NOT NULL"
        )
    }
    have_wl = {
        (r[0], r[1], round(r[2], 4))
        for r in conn.execute("SELECT element, sp_num, wavelength_nm FROM lines WHERE wavelength_nm IS NOT NULL")
    }

    inserts = []
    with open(args.dump, encoding="latin-1") as f:
        for line in f:
            body = _strip_insert(line, "ASD_Lines")
            if body is None:
                continue
            v = parse_tuple(body)
            if len(v) < 20:
                continue
            sc = str(v[1]).strip()
            if sc not in ("1", "2", "3"):
                continue
            elem = v[0]
            wl_a = _f(v[4]) or _f(v[6]) or _f(v[5])
            if elem is None or wl_a is None:
                continue
            wl = wl_a / 10.0
            stage = int(sc)
            lid, uid = v[2], v[3]
            resolved = bool(lid) and bool(uid)
            if resolved:
                if (elem, stage, lid, uid) in have_trans:
                    continue  # transition already present
            else:
                if (elem, stage, round(wl, 4)) in have_wl:
                    continue  # unresolved line already present by wavelength
            ei = ek = gi = gk = None
            if lid in level:
                ei, gi = level[lid]
            if uid in level:
                ek, gk = level[uid]
            a = _f(v[10])
            inserts.append((
                elem, stage, wl, a, ei, ek, gi, gk, _f(v[19]), _f(v[12]), _f(v[11]),
                _f(v[13]), _f(v[19]),
                (_f(v[6]) / 10.0) if _f(v[6]) else None,
                (_f(v[5]) / 10.0) if _f(v[5]) else None,
                lid, uid,
                int(_f(v[14])) if _f(v[14]) is not None else None,
                v[15], v[16], "measured" if a else "none",
            ))
            if resolved:
                have_trans.add((elem, stage, lid, uid))
            else:
                have_wl.add((elem, stage, round(wl, 4)))

    cols = ["element", "sp_num", "wavelength_nm", "aki", "ei_ev", "ek_ev", "gi", "gk",
            "rel_int", "log_gf", "osc_str", "line_str", "intens_obs", "vac_wl_nm",
            "calc_wl_nm", "low_level_id", "upp_level_id", "line_type", "line_accuracy",
            "line_ref", "aki_source"]
    print(f"inserting {len(inserts)} missing dump transitions")
    conn.executemany(
        f"INSERT OR IGNORE INTO lines ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
        inserts,
    )
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    total = conn.execute("SELECT COUNT(*) FROM lines").fetchone()[0]
    conn.close()
    print(f"done. lines now {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
