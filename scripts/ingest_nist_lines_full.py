#!/usr/bin/env python
"""Ingest the FULL observed line list (all fields) from the ASD netCDF.

The DB held only ~30% of observed lines (the measured-A subset, in a limited
213-944 nm window). For full-spectrum synthetic generation and beyond-CF-LIBS
methods, the database must contain EVERY observed transition -- including lines
without a measured A (they have observed wavelength + intensity + level links;
A is derived/estimated downstream, never dropped).

This:
  - ALTERs ``lines`` to add the missing NIST fields (existing columns untouched),
  - derives lower/upper energy + g from the level-ID link to ``energy_levels``,
  - dedups on (element, sp_num, wavelength) against existing rows (backfilling the
    new columns on a match; inserting the line otherwise),
  - tags ``aki_source`` (measured | none) so downstream code is explicit.

Source: <repo>/.../scratchpad/asd_verify/Lines_da.nc + Levels_da.nc (ASD-5.9).
Units: level energy cm^-1 -> eV (/8065.544); vac_wl/calc_wl Angstrom -> nm (/10);
obs_wl already nm.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

import numpy as np

CM_PER_EV = 8065.544

NEW_COLUMNS = [
    ("log_gf", "REAL"),
    ("osc_str", "REAL"),
    ("line_str", "REAL"),
    ("intens_obs", "REAL"),
    ("vac_wl_nm", "REAL"),
    ("calc_wl_nm", "REAL"),
    ("low_level_id", "TEXT"),
    ("upp_level_id", "TEXT"),
    ("line_type", "INTEGER"),
    ("line_accuracy", "TEXT"),
    ("line_ref", "TEXT"),
    ("aki_source", "TEXT"),
]


def _f(x):
    """Parse a netCDF cell to float or None."""
    s = str(x).strip()
    if s in ("", "--", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _s(x):
    s = str(x).strip()
    return s if s not in ("", "--", "nan", "None") else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="ASD_da/libs_production.db")
    p.add_argument("--levels-nc", required=True)
    p.add_argument("--lines-nc", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    import netCDF4

    # --- level lookup: level_id -> (energy_eV, g) ---
    Lv = netCDF4.Dataset(args.levels_nc)
    lvid = [str(x) for x in np.array(Lv.variables["level_id"][:])]
    lven = np.array(Lv.variables["energy"][:])
    lvg = np.array(Lv.variables["g"][:])
    level = {}
    for lid, en, g in zip(lvid, lven, lvg):
        e = _f(en)
        level[lid] = (e / CM_PER_EV if e is not None else None, _f(g))

    # --- lines ---
    Li = netCDF4.Dataset(args.lines_nc)
    V = Li.variables

    def col(name):
        return np.array(V[name][:]) if name in V else None

    el = col("element")
    ch = col("ion_charge")
    obs_wl = col("obs_wl")
    vac_wl = col("vac_wl")
    calc_wl = col("calc_wl")
    A = col("A")
    log_gf = col("log_gf")
    osc = col("osc_str")
    lstr = col("line_str")
    intens = col("intens")
    low = col("low_level_id")
    upp = col("upp_level_id")
    ltype = col("type")
    accur = col("accur")
    ref = col("tp_ref")

    conn = sqlite3.connect(args.db, timeout=120)
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(lines)")}
    for name, typ in NEW_COLUMNS:
        if name not in existing_cols:
            print(f"ALTER lines ADD {name} {typ}")
            if not args.dry_run:
                conn.execute(f"ALTER TABLE lines ADD COLUMN {name} {typ}")

    # dedup set keyed on (element, sp_num, round(wavelength_nm, 3))
    have = {
        (r[0], r[1], round(r[2], 3))
        for r in conn.execute("SELECT element, sp_num, wavelength_nm FROM lines WHERE wavelength_nm IS NOT NULL")
    }

    inserts, updates = [], []
    n = len(el)
    for i in range(n):
        c = _f(ch[i])
        if c is None or int(c) not in (0, 1, 2):
            continue
        stage = int(c) + 1
        elem = _s(el[i])
        wl = _f(obs_wl[i])
        if elem is None or wl is None:
            continue
        a = _f(A[i])
        lid, uid = _s(low[i]), _s(upp[i])
        ei = ek = gi = gk = None
        if lid in level:
            ei, gi = level[lid]
        if uid in level:
            ek, gk = level[uid]
        vac = _f(vac_wl[i])
        cwl = _f(calc_wl[i])
        row = {
            "element": elem, "sp_num": stage, "wavelength_nm": wl,
            "aki": a, "ei_ev": ei, "ek_ev": ek, "gi": gi, "gk": gk,
            "rel_int": _f(intens[i]),
            "log_gf": _f(log_gf[i]), "osc_str": _f(osc[i]),
            "line_str": _f(lstr[i]) if lstr is not None else None,
            "intens_obs": _f(intens[i]),
            "vac_wl_nm": vac / 10.0 if vac is not None else None,
            "calc_wl_nm": cwl / 10.0 if cwl is not None else None,
            "low_level_id": lid, "upp_level_id": uid,
            "line_type": int(_f(ltype[i])) if _f(ltype[i]) is not None else None,
            "line_accuracy": _s(accur[i]), "line_ref": _s(ref[i]),
            "aki_source": "measured" if a else "none",
        }
        key = (elem, stage, round(wl, 3))
        if key in have:
            updates.append(row)
        else:
            have.add(key)
            inserts.append(row)

    print(f"lines parsed: I/II/III usable; to INSERT {len(inserts)}, to BACKFILL {len(updates)}")
    if args.dry_run:
        return 0

    ins_cols = [
        "element", "sp_num", "wavelength_nm", "aki", "ei_ev", "ek_ev", "gi", "gk",
        "rel_int", "log_gf", "osc_str", "line_str", "intens_obs", "vac_wl_nm",
        "calc_wl_nm", "low_level_id", "upp_level_id", "line_type", "line_accuracy",
        "line_ref", "aki_source",
    ]
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
