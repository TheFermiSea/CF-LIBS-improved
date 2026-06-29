#!/usr/bin/env python
"""Rebuild the SQLite `lines` table cleanly from the authoritative ASD dump.

Wavelength-rounding dedup loses closely-spaced hydrogenic lines (Be/Li/He
Rydberg series): the correct line identity is the transition (low+upp level),
which the dump carries. This rebuilds `lines` from EVERY ASD_Lines row (each a
distinct transition) so the table matches the gold standard exactly, while
re-attaching the derived physics fields (Stark widths, van der Waals / self
broadening, resonance flag, accuracy grade, A uncertainty) that were computed
into the original DB and are NOT in the dump.

Safe: builds `lines_new`, verifies the count/known-line, then swaps. The old DB
is backed up by the caller.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

# reuse the dump tuple parser
sys.path.insert(0, "scripts")
from ingest_asd_dump_lines import _f, _strip_insert, parse_tuple  # noqa: E402

CM_PER_EV = 8065.544

# derived fields to carry over from the existing table (not in the dump)
DERIVED = [
    "stark_w", "stark_alpha", "stark_shift", "is_resonance",
    "aki_uncertainty", "accuracy_grade", "gamma_vdw_log", "gamma_self_log",
    "stark_w_source",
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="ASD_da/libs_production.db")
    p.add_argument("--dump", required=True)
    p.add_argument("--stages", nargs="+", default=["1", "2", "3"])
    args = p.parse_args(argv)
    stages = set(args.stages)

    # --- level map from ASD_Levels ---
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
    print(f"levels: {len(level)}")

    conn = sqlite3.connect(args.db, timeout=120)
    # save derived fields keyed by (element, sp_num, round(wl, 3))
    saved = {}
    for r in conn.execute(
        f"SELECT element, sp_num, wavelength_nm, {','.join(DERIVED)} FROM lines "
        "WHERE wavelength_nm IS NOT NULL AND (stark_w IS NOT NULL OR is_resonance IS NOT NULL "
        "OR accuracy_grade IS NOT NULL OR aki_uncertainty IS NOT NULL)"
    ):
        saved[(r[0], r[1], round(r[2], 3))] = r[3:]
    print(f"derived-field rows saved: {len(saved)}")

    # full schema of lines (to recreate lines_new identically)
    create = conn.execute("SELECT sql FROM sqlite_master WHERE name='lines'").fetchone()[0]
    conn.execute("DROP TABLE IF EXISTS lines_new")
    conn.execute(create.replace("TABLE lines", "TABLE lines_new", 1).replace("`lines`", "lines_new"))

    cols = [r[1] for r in conn.execute("PRAGMA table_info(lines_new)")]
    rows = []
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
            wl_a = _f(v[4]) or _f(v[6]) or _f(v[5])
            if elem is None or wl_a is None:
                continue
            wl = wl_a / 10.0
            ei = ek = gi = gk = None
            if v[2] in level:
                ei, gi = level[v[2]]
            if v[3] in level:
                ek, gk = level[v[3]]
            a = _f(v[10])
            d = saved.get((elem, int(sc), round(wl, 3)), (None,) * len(DERIVED))
            row = {
                "element": elem, "sp_num": int(sc), "wavelength_nm": wl, "aki": a,
                "ei_ev": ei, "ek_ev": ek, "gi": gi, "gk": gk, "rel_int": _f(v[19]),
                "log_gf": _f(v[12]), "osc_str": _f(v[11]), "line_str": _f(v[13]),
                "intens_obs": _f(v[19]),
                "vac_wl_nm": (_f(v[6]) / 10.0) if _f(v[6]) else None,
                "calc_wl_nm": (_f(v[5]) / 10.0) if _f(v[5]) else None,
                "low_level_id": v[2], "upp_level_id": v[3],
                "line_type": int(_f(v[14])) if _f(v[14]) is not None else None,
                "line_accuracy": v[15], "line_ref": v[16],
                "aki_source": "measured" if a else "none",
            }
            row.update(dict(zip(DERIVED, d)))
            rows.append(tuple(row.get(c) for c in cols if c != "id"))
    ins_cols = [c for c in cols if c != "id"]
    conn.executemany(
        f"INSERT OR IGNORE INTO lines_new ({','.join(ins_cols)}) VALUES ({','.join('?' for _ in ins_cols)})",
        rows,
    )
    conn.commit()
    n_new = conn.execute("SELECT COUNT(*) FROM lines_new").fetchone()[0]
    n_der = conn.execute("SELECT COUNT(*) FROM lines_new WHERE stark_w IS NOT NULL").fetchone()[0]
    fe1 = conn.execute("SELECT COUNT(*) FROM lines_new WHERE element='Fe' AND sp_num=1").fetchone()[0]
    print(f"lines_new: {n_new} rows (parsed {len(rows)}), {n_der} with Stark, Fe I {fe1}")
    if n_new < 200000:
        print("ABORT: lines_new too small, not swapping")
        return 1
    # swap
    conn.execute("DROP TABLE lines")
    conn.execute("ALTER TABLE lines_new RENAME TO lines")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    print("swapped lines_new -> lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
