#!/usr/bin/env python
"""Complete a lines-only atomic DB (Kurucz/VALD ingest) with species_physics +
partition_functions, using ExoJAX's BUNDLED core_atom data — no live NIST query.

M5: the Kurucz/VALD ingests build only the cflibs `lines` table. A Saha-Boltzmann
inversion also needs ionization potentials (`species_physics`) and partition
functions U(T) (`partition_functions`). ExoJAX ships both as tabulated files
(Barklem & Collet 2016 partition functions + a NIST ionization-energy file +
atomic masses), so a COMPLETE atomic DB stays entirely on Kurucz/VALD lines +
Barklem/NIST-file physics with NO live-NIST borrowing — exactly what the M5
Kurucz/VALD-vs-NIST benchmark requires (assessed HIGH by the exojax-databases
workflow, 2026-06-21).

Conventions matched to cflibs:
- species_physics(element, sp_num, ip_ev, atomic_mass); ip_ev = IP to go from
  stage sp_num -> sp_num+1 (pick_ionE iion=sp_num; neutral=1).
- partition_functions stores ln U = a0 + a1 lnT + ... + a4 (lnT)^4 (natural log;
  np.polynomial.polynomial.polyfit, same as cflibs/plasma/partition.py).

CAVEAT (enforced): the Barklem-2016 PF grid stops at 10000 K, so
partition_functions.t_max = 10000 for these rows; the evaluator clamps hotter
plasmas to the ceiling. We do NOT advertise validity beyond the grid.
"""

from __future__ import annotations

import argparse
import sqlite3
import warnings

import numpy as np

ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
PF_FIT_T_MIN = 2000.0  # LIBS-relevant validity floor
PF_FIT_T_MAX = 10000.0  # Barklem 2016 grid ceiling (hard)
PF_SOURCE = "BarklemCollet2016"  # MUST match cflibs AUTHORITATIVE_PF_SOURCES (partition.py)
# exactly, else derive_partition_spec discards the stored poly and falls back to a
# direct-sum over line-derived energy_levels (which carry ground-state offsets that
# inflate e.g. K II / Na II U(T) ~5x). The display name is "Barklem & Collet 2016".

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS species_physics "
    "(element TEXT, sp_num INTEGER, ip_ev REAL, atomic_mass REAL);"
    "CREATE TABLE IF NOT EXISTS partition_functions "
    "(element TEXT, sp_num INTEGER, a0 REAL,a1 REAL,a2 REAL,a3 REAL,a4 REAL,"
    " t_min REAL, t_max REAL, source TEXT);"
    # energy_levels must exist for AtomicDatabase's schema migration to load;
    # left empty here, the migration auto-populates it from the (comprehensive
    # Kurucz/VALD) lines table so direct-sum U(T) works; the Barklem polynomial
    # above is the fallback for level-less species + an independent cross-check.
    "CREATE TABLE IF NOT EXISTS energy_levels "
    "(element TEXT, sp_num INTEGER, g_level INTEGER, energy_ev REAL);"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="lines-only atomic DB to complete IN PLACE")
    ap.add_argument("--replace", action="store_true", help="clear existing physics rows first")
    args = ap.parse_args()

    import periodictable

    from exojax.database.core_atom import io

    warnings.filterwarnings("ignore")  # cosmetic pandas FutureWarning in pick_ionE
    df_ionE = io.load_ionization_energies()
    ipccd = io.load_atomicdata()
    pfTdat, pfdat = io.load_pf_Barklem2016()
    Tgrid = np.asarray(pfTdat.columns[1:], dtype=float)
    fit_mask = (Tgrid >= 1000.0) & (Tgrid <= PF_FIT_T_MAX)  # >=5 pts, LIBS-weighted
    pf_by_label = {
        str(pfdat.iloc[i, 0]): pfdat.iloc[i, 1:].to_numpy(dtype=float) for i in range(len(pfdat))
    }

    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)
    if args.replace:
        conn.execute("DELETE FROM species_physics")
        conn.execute("DELETE FROM partition_functions")

    species = conn.execute(
        "SELECT DISTINCT element, sp_num FROM lines ORDER BY element, sp_num"
    ).fetchall()

    n_sp = n_pf = 0
    missing_ip: list = []
    missing_pf: list = []
    for el, sp in species:
        sp = int(sp)
        try:
            z = periodictable.elements.symbol(el).number
        except Exception:
            continue

        # --- species_physics: IP (pick_ionE, NIST file) + atomic mass (ipccd) ---
        ip = None
        try:
            val = io.pick_ionE(z, sp, df_ionE)
            ip = float(val) if val is not None and np.isfinite(float(val)) else None
        except Exception:
            ip = None
        mass = None
        try:
            mrow = ipccd[ipccd["ielem"] == z]["mass"]
            mass = float(mrow.iloc[0]) if len(mrow) else None
        except Exception:
            mass = None
        if ip is not None:
            conn.execute("INSERT INTO species_physics VALUES (?,?,?,?)", (el, sp, ip, mass))
            n_sp += 1
        else:
            missing_ip.append(f"{el} {sp}")

        # --- partition_functions: Barklem U(T) -> ln U = poly(ln T) deg 4 ---
        roman = ROMAN.get(sp)
        u_grid = pf_by_label.get(f"{el}_{roman}") if roman else None
        if u_grid is not None:
            ln_t = np.log(Tgrid[fit_mask])
            ln_u = np.log(np.maximum(u_grid[fit_mask], 1.0))
            if len(ln_t) >= 5 and np.all(np.isfinite(ln_u)):
                coeffs = np.polynomial.polynomial.polyfit(ln_t, ln_u, 4)
                conn.execute(
                    "INSERT INTO partition_functions VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (el, sp, *[float(c) for c in coeffs], PF_FIT_T_MIN, PF_FIT_T_MAX, PF_SOURCE),
                )
                n_pf += 1
            else:
                missing_pf.append(f"{el} {sp}")
        else:
            missing_pf.append(f"{el} {sp}")

    conn.commit()
    conn.close()
    print(
        f"completed {args.db}: {len(species)} species -> {n_sp} species_physics, {n_pf} partition_functions"
    )
    if missing_ip:
        print(f"  no IP for: {', '.join(missing_ip)}")
    if missing_pf:
        print(f"  no Barklem PF for: {', '.join(missing_pf)}")


if __name__ == "__main__":
    main()
