#!/usr/bin/env python
"""Ingest the FULL line list from the authoritative NIST ASD MySQL dump.

The CAAAMLIBS netCDF was an INCOMPLETE conversion of the dump (it dropped ~half
the lines: dump 204,574 I/II/III lines vs netCDF 104,995). The MySQL dump
(`monograph8`) is the gold standard. This streams the dump, builds a
level_id -> (energy_eV, g) map from ASD_Levels, and ingests every ASD_Lines
row (stages I/II/III) into the SQLite `lines` table -- deriving lower/upper
energy + g from the level link, deduping against existing rows (backfill on
match, insert otherwise). Existing columns/rows preserved.

Units in the dump: level energy cm^-1 (string); line wavelengths Angstrom
(obs_wl preferred, else vac_wl, else calc_wl) -> nm /10.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

CM_PER_EV = 8065.544


def parse_tuple(s: str):
    """Parse one MySQL VALUES tuple body into a list of python values.

    Handles 'single-quoted' strings (with \\' and '' escapes), unquoted NULL,
    and unquoted numbers. Returns strings as str, NULL as None, bare tokens as
    str (caller coerces).
    """
    out, i, n = [], 0, len(s)
    while i < n:
        # skip leading spaces
        while i < n and s[i] in " \t":
            i += 1
        if i >= n:
            break
        if s[i] == "'":
            i += 1
            buf = []
            while i < n:
                c = s[i]
                if c == "\\" and i + 1 < n:
                    buf.append(s[i + 1])
                    i += 2
                    continue
                if c == "'":
                    if i + 1 < n and s[i + 1] == "'":  # '' escape
                        buf.append("'")
                        i += 2
                        continue
                    i += 1
                    break
                buf.append(c)
                i += 1
            out.append("".join(buf))
        else:
            j = i
            while j < n and s[j] != ",":
                j += 1
            tok = s[i:j].strip()
            out.append(None if tok == "NULL" else tok)
            i = j
        # skip the comma
        while i < n and s[i] in " \t":
            i += 1
        if i < n and s[i] == ",":
            i += 1
    return out


def _f(x):
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _strip_insert(line: str, table: str):
    """Return the tuple body of `INSERT INTO `table` VALUES (...);` or None."""
    pre = f"INSERT INTO `{table}` VALUES ("
    if not line.startswith(pre):
        return None
    body = line[len(pre):].rstrip()
    if body.endswith(";"):
        body = body[:-1]
    if body.endswith(")"):
        body = body[:-1]
    return body


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="ASD_da/libs_production.db")
    p.add_argument("--dump", required=True)
    p.add_argument("--stages", nargs="+", default=["1", "2", "3"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    stages = set(args.stages)

    # --- pass 1: level_id -> (energy_eV, g) from ASD_Levels ---
    # ASD_Levels cols (0-idx): 0 element,1 spectr_charge,...,9 g,...,12 energy,...,23 level_id
    level = {}
    with open(args.dump, encoding="latin-1") as f:
        for line in f:
            body = _strip_insert(line, "ASD_Levels")
            if body is None:
                continue
            v = parse_tuple(body)
            if len(v) < 24:
                continue
            lid = v[23]
            e = _f(v[12])
            g = _f(v[9])
            if lid:
                level[lid] = (e / CM_PER_EV if e is not None else None, g)
    print(f"parsed {len(level)} levels (level_id -> energy,g)")

    # --- pass 2: ingest ASD_Lines ---
    # cols (0-idx): 0 element,1 spectr_charge,2 low_id,3 upp_id,4 obs_wl,5 calc_wl,
    #   6 vac_wl,10 A,11 osc_str,12 log_gf,13 line_str,14 type,15 accur,16 tp_ref,
    #   17 line_ref,19 intens
    conn = sqlite3.connect(args.db, timeout=120)
    have = {
        (r[0], r[1], round(r[2], 3))
        for r in conn.execute("SELECT element, sp_num, wavelength_nm FROM lines WHERE wavelength_nm IS NOT NULL")
    }
    inserts, updates, skipped = [], [], 0
    with open(args.dump, encoding="latin-1") as f:
        for line in f:
            body = _strip_insert(line, "ASD_Lines")
            if body is None:
                continue
            v = parse_tuple(body)
            if len(v) < 20:
                continue
            sc = str(v[1]).strip()
            if sc not in stages:
                continue
            elem = v[0]
            # wavelength: obs_wl > vac_wl > calc_wl, Angstrom -> nm
            wl_a = _f(v[4]) or _f(v[6]) or _f(v[5])
            if elem is None or wl_a is None:
                skipped += 1
                continue
            wl = wl_a / 10.0
            lid, uid = v[2], v[3]
            ei = ek = gi = gk = None
            if lid in level:
                ei, gi = level[lid]
            if uid in level:
                ek, gk = level[uid]
            a = _f(v[10])
            row = {
                "element": elem, "sp_num": int(sc), "wavelength_nm": wl,
                "aki": a, "ei_ev": ei, "ek_ev": ek, "gi": gi, "gk": gk,
                "rel_int": _f(v[19]), "log_gf": _f(v[12]), "osc_str": _f(v[11]),
                "line_str": _f(v[13]), "intens_obs": _f(v[19]),
                "vac_wl_nm": (_f(v[6]) / 10.0) if _f(v[6]) else None,
                "calc_wl_nm": (_f(v[5]) / 10.0) if _f(v[5]) else None,
                "low_level_id": lid, "upp_level_id": uid,
                "line_type": int(_f(v[14])) if _f(v[14]) is not None else None,
                "line_accuracy": v[15], "line_ref": v[16],
                "aki_source": "measured" if a else "none",
            }
            key = (elem, int(sc), round(wl, 3))
            if key in have:
                updates.append(row)
            else:
                have.add(key)
                inserts.append(row)
    print(f"ASD_Lines: INSERT {len(inserts)}, BACKFILL {len(updates)} (skipped {skipped} no-wl)")
    if args.dry_run:
        return 0

    ins_cols = ["element", "sp_num", "wavelength_nm", "aki", "ei_ev", "ek_ev", "gi",
                "gk", "rel_int", "log_gf", "osc_str", "line_str", "intens_obs",
                "vac_wl_nm", "calc_wl_nm", "low_level_id", "upp_level_id",
                "line_type", "line_accuracy", "line_ref", "aki_source"]
    conn.executemany(
        f"INSERT OR IGNORE INTO lines ({','.join(ins_cols)}) VALUES ({','.join('?' for _ in ins_cols)})",
        [tuple(r[c] for c in ins_cols) for r in inserts],
    )
    upd_cols = ["log_gf", "osc_str", "line_str", "intens_obs", "vac_wl_nm",
                "calc_wl_nm", "low_level_id", "upp_level_id", "line_type",
                "line_accuracy", "line_ref", "aki_source"]
    conn.executemany(
        f"UPDATE lines SET {','.join(c+'=?' for c in upd_cols)} "
        "WHERE element=? AND sp_num=? AND round(wavelength_nm,3)=round(?,3)",
        [tuple(r[c] for c in upd_cols) + (r["element"], r["sp_num"], r["wavelength_nm"]) for r in updates],
    )
    conn.commit()
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError as exc:
        print(f"WARNING: WAL checkpoint skipped ({exc})")
    total = conn.execute("SELECT COUNT(*) FROM lines").fetchone()[0]
    conn.close()
    print(f"done. inserted {len(inserts)}, backfilled {len(updates)}; lines now {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
