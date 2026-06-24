#!/usr/bin/env python
"""Build R4: NIST authoritative + VALD backfill for coverage gaps.

M5 hybrid: keep NIST's accurate graded gf VALUES where NIST has a line; add a VALD
line ONLY where NIST lacks coverage for that (element, ion) within a wavelength
tolerance. The added lines keep VALD's (mostly D) grade and are tagged
``stark_w_source='vald_backfill'`` so the grade-aware selector prefers the NIST lines.

Physics: the NIST DB is copied verbatim (NIST lines + NIST species_physics +
partition_functions preserved), so for species NIST already covers the PF is identical
to the NIST baseline -> only the ADDED coverage differs (PF confound controlled for
shared species). For species ONLY VALD adds, Barklem physics is copied from the VALD DB.

Usage:
    PYTHONPATH=$PWD .venv/bin/python scripts/build_r4.py \
        --nist ASD_da/libs_production.db --vald output/vald_complete.db \
        --out output/r4_nist_vald_backfill.db --tol-nm 0.05
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

LINE_COLS = (
    "element,sp_num,wavelength_nm,aki,ei_ev,ek_ev,gi,gk,rel_int,stark_w,stark_alpha,"
    "stark_shift,is_resonance,aki_uncertainty,accuracy_grade,gamma_vdw_log,"
    "gamma_self_log,stark_w_source"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nist", required=True)
    ap.add_argument("--vald", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-nm", type=float, default=0.05, help="NIST-coverage match tolerance")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.nist, args.out)  # NIST lines + NIST physics preserved
    conn = sqlite3.connect(args.out)
    conn.execute("ATTACH ? AS vald", (args.vald,))

    # NIST wavelengths per (element, sp_num), sorted, for proximity check.
    nist_wl: dict = defaultdict(list)
    for el, sp, wl in conn.execute("SELECT element, sp_num, wavelength_nm FROM main.lines"):
        nist_wl[(el, sp)].append(wl)
    nist_arr = {k: np.sort(np.asarray(v, dtype=float)) for k, v in nist_wl.items()}
    n_nist = conn.execute("SELECT COUNT(*) FROM main.lines").fetchone()[0]

    # Group VALD lines by species; vectorized nearest-NIST distance -> backfill if > tol.
    vald_rows: dict = defaultdict(list)
    for row in conn.execute(f"SELECT {LINE_COLS} FROM vald.lines"):
        vald_rows[(row[0], int(row[1]))].append(row)

    backfill = []
    species_only_vald = 0
    for key, rows in vald_rows.items():
        arr = nist_arr.get(key)
        wls = np.asarray([r[2] for r in rows], dtype=float)
        if arr is None or len(arr) == 0:
            backfill.extend(rows)  # NIST lacks this species entirely
            species_only_vald += 1
            continue
        idx = np.searchsorted(arr, wls)
        left = np.clip(idx - 1, 0, len(arr) - 1)
        right = np.clip(idx, 0, len(arr) - 1)
        dist = np.minimum(np.abs(arr[left] - wls), np.abs(arr[right] - wls))
        for r, d in zip(rows, dist):
            if d > args.tol_nm:
                backfill.append(r)

    # Insert backfill lines, tagged provenance, keeping VALD grade/values. OR IGNORE so
    # a collision on the UNIQUE key (element,sp_num,wl,ek_ev) keeps the NIST line (NIST
    # authoritative) and drops internal VALD near-duplicates.
    tagged = [(*r[:-1], "vald_backfill") for r in backfill]
    conn.executemany(
        f"INSERT OR IGNORE INTO main.lines ({LINE_COLS}) VALUES ({','.join('?' * 18)})", tagged
    )

    # Physics: add Barklem PF / species_physics ONLY for species NIST lacks.
    for tbl in ("species_physics", "partition_functions"):
        cols = [r[1] for r in conn.execute(f"PRAGMA main.table_info({tbl})")]
        collist = ",".join(cols)
        conn.execute(
            f"INSERT INTO main.{tbl} ({collist}) SELECT {collist} FROM vald.{tbl} v "
            f"WHERE NOT EXISTS (SELECT 1 FROM main.{tbl} m "
            f"WHERE m.element=v.element AND m.sp_num=v.sp_num)"
        )
    conn.commit()

    n_out = conn.execute("SELECT COUNT(*) FROM main.lines").fetchone()[0]
    n_species = conn.execute(
        "SELECT COUNT(DISTINCT element||'_'||sp_num) FROM main.lines"
    ).fetchone()[0]
    conn.close()
    print(f"R4 built -> {args.out}")
    print(f"  NIST lines: {n_nist:,}  + backfill: {len(backfill):,}  = total {n_out:,}")
    print(f"  species: {n_species}  ({species_only_vald} added by VALD that NIST lacked)")
    print(f"  backfill tolerance: {args.tol_nm} nm")


if __name__ == "__main__":
    main()
