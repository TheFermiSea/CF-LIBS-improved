#!/usr/bin/env python
"""Ingest ExoMol TiO 'Toto' (48Ti-16O) into the cflibs ``molecular_lines`` table.

M5 (hybrid VALD+ExoMol completion): TiO dominates the molecular line count in the
visible/red (~59M lines in the ExoMol 'Toto' list, McKemmish et al. 2019) and would
need ~300 VALD web requests because it saturates VALD's ~101k-record per-request cap.
Instead we pull TiO as ONE bulk download from ExoMol via RADIS and ingest it into the
SAME ``molecular_lines`` table the VALD ingest writes (so VALD's 7 minor molecules +
ExoMol TiO coexist). VALD remains the source for atomic lines + CN/C2/OH/CH/CO/NH/MgH.

Design notes:
- Import ONLY ``radis.api.exomolapi`` and pass ``local_databases`` explicitly (top-level
  ``radis`` can hit a numba/coverage import path in some envs; the submodule avoids it).
- ExoMol ``nu_lines`` are VACUUM wavenumbers (cm^-1); we convert to AIR wavelength via
  ``cflibs.atomic.wavelength_conversion.vacuum_to_air_nm`` (cflibs air convention), so the
  merged table stays single-medium (air), matching the VALD slices.
- ExoMol TiO is a SINGLE source => no slice-overlap duplicates => no in-RAM dedup set
  needed (the VALD ingest dedups because its slices overlap; this one does not).
- Stream the loaded arrays in batches and ``executemany`` per batch — never materialize
  ~30M python rows. ``engine='vaex'`` memory-maps the cache so the load itself is bounded.
- Toto ships 0 broadening files => gamma_* are NULL (broadf=False avoids 404s).

Run:
    PYTHONPATH=$PWD .venv/bin/python scripts/ingest_exomol_tio.py \
        --db output/vald_atomic.db --wl-min 200 --wl-max 1000 --local-db data/exomol
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np

HC_EV_CM = 1.239841984e-4  # eV per cm^-1 (hc/e), matches ingest_vald_atomic.py
AKI_CONST = 6.6702e15  # A_ki = AKI_CONST * gf / (g_up * lambda_A^2)

SCHEMA = """
CREATE TABLE IF NOT EXISTS molecular_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    species TEXT, charge INTEGER, wavelength_nm REAL, aki REAL, loggf REAL,
    ei_ev REAL, ek_ev REAL, gi INTEGER, gk INTEGER,
    gamma_rad_log REAL, gamma_stark_log REAL, gamma_vdw_log REAL,
    accuracy_grade TEXT, provenance TEXT
);
"""

INSERT = (
    "INSERT INTO molecular_lines (species,charge,wavelength_nm,aki,loggf,ei_ev,ek_ev,"
    "gi,gk,gamma_rad_log,gamma_stark_log,gamma_vdw_log,accuracy_grade,provenance) "
    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="output SQLite DB (molecular_lines table)")
    ap.add_argument("--wl-min", type=float, default=200.0, help="min AIR wavelength nm")
    ap.add_argument("--wl-max", type=float, default=1000.0, help="max AIR wavelength nm")
    ap.add_argument("--local-db", default="data/exomol", help="ExoMol cache root")
    ap.add_argument("--batch", type=int, default=50000, help="sqlite insert batch size")
    ap.add_argument("--grade", default="C", help="accuracy grade for ExoMol TiO (theory/MARVEL)")
    args = ap.parse_args()

    import radis.api.exomolapi as ea  # submodule only

    from cflibs.atomic.wavelength_conversion import vacuum_to_air_nm

    # AIR window -> VACUUM wavenumber [cm^-1] (widen slightly; TiO has no lines >30000 cm^-1).
    nu_lo = 1e7 / args.wl_max * 0.999
    nu_hi = 1e7 / args.wl_min * 1.001
    print(
        f"Loading ExoMol TiO/48Ti-16O/Toto, nurange=[{nu_lo:.1f},{nu_hi:.1f}] cm^-1 "
        f"({args.wl_min}-{args.wl_max} nm air) ..."
    )
    mdb = ea.MdbExomol(
        path=str(Path(args.local_db) / "TiO/48Ti-16O/Toto"),
        molecule="TiO",
        database="Toto",
        local_databases=args.local_db,
        nurange=[nu_lo, nu_hi],
        engine="vaex",
        broadf=False,
        broadf_download=False,
        skip_optional_data=False,  # keep J (for g_lower)
        cache=True,
        verbose=True,
    )

    nu = np.asarray(mdb.nu_lines, dtype=float)  # cm^-1 vacuum
    aki = np.asarray(mdb.A, dtype=float)
    elower = np.asarray(mdb.elower, dtype=float)
    gpp = np.asarray(mdb.gpp, dtype=float)
    jlower = (
        np.asarray(mdb.jlower, dtype=float) if getattr(mdb, "jlower", None) is not None else None
    )
    n = len(nu)
    print(f"  loaded {n:,} TiO transitions in window; streaming into {args.db} ...")

    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")

    inserted = skipped = 0
    B = args.batch
    for k in range(0, n, B):
        s = slice(k, min(k + B, n))
        nu_b = nu[s]
        wl_vac_nm = 1e7 / nu_b
        wl_air_nm = np.asarray(vacuum_to_air_nm(wl_vac_nm), dtype=float)
        a_b = aki[s]
        el_b = elower[s]
        gk_b = gpp[s]
        jl_b = jlower[s] if jlower is not None else None
        rows = []
        for i in range(len(nu_b)):
            lam = float(wl_air_nm[i])
            if not (args.wl_min <= lam <= args.wl_max):
                skipped += 1
                continue
            gk = int(gk_b[i])
            gi = int(round(2 * jl_b[i] + 1)) if jl_b is not None else None
            ei = float(el_b[i]) * HC_EV_CM
            ek = (float(el_b[i]) + float(nu_b[i])) * HC_EV_CM
            a_i = float(a_b[i])
            lam_a = lam * 10.0
            loggf = math.log10(gk * a_i * lam_a * lam_a / AKI_CONST) if gk > 0 and a_i > 0 else None
            rows.append(
                (
                    "TiO",
                    1,
                    lam,
                    a_i,
                    loggf,
                    ei,
                    ek,
                    gi,
                    gk,
                    None,
                    None,
                    None,
                    args.grade,
                    "exomol_toto",
                )
            )
        if rows:
            conn.executemany(INSERT, rows)
            conn.commit()
            inserted += len(rows)
        if k % (B * 20) == 0:
            print(f"    {min(k + B, n):,}/{n:,} processed, {inserted:,} inserted")

    conn.execute("PRAGMA synchronous=NORMAL")
    conn.close()
    print(f"\nDONE: {inserted:,} TiO lines -> {args.db} (skipped {skipped:,} out of window)")


if __name__ == "__main__":
    main()
