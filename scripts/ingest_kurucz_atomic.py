#!/usr/bin/env python
"""Ingest Kurucz atomic line lists (``gf????.all``) into the cflibs atomic schema.

M5 / Lever 1C (accuracy-first roadmap): Kurucz is the *completeness* backend for
the atomic line list (measured ~5-6 wt% better than NIST on line-rich SuperCam;
~15x more Fe I lines in 240-850 nm). This is an OFFLINE acquisition tool: it uses
ExoJAX's ``AdbKurucz`` parser (NOT a shipped cflibs dependency) to read the Kurucz
``gf????.all`` files downloaded from http://kurucz.harvard.edu/atoms/<Zion>/ and
writes them into a SQLite ``lines`` table matching ``datagen_v2.py`` so the existing
``AtomicDatabase`` / ``AtomicDataSource`` consumes them unchanged.

Key normalizations (so the line list, wavelength solution and forward model share
one convention — Ciucci/Tognoni):
- ExoJAX returns VACUUM wavenumbers; we convert to AIR wavelengths via the
  ``cflibs.atomic.wavelength_conversion`` Edlen/Morton converter (M2 prereq, #307),
  matching the NIST ``obs_wl_air`` convention used by the rest of the DB.
- Kurucz lines carry NO per-line NIST-style accuracy grade -> ``accuracy_grade='U'``
  (unknown), ``aki_uncertainty=NULL``. Grade-aware quantitation selection should
  prefer NIST-graded lines where they exist and fall back to Kurucz 'U' for coverage.

Usage:
    PYTHONPATH=$PWD <exojax-venv>/bin/python scripts/ingest_kurucz_atomic.py \
        --gf data/kurucz/gf2600.clean.all [data/kurucz/gf2601.clean.all ...] \
        --db output/kurucz_atomic.db --wl-min 240 --wl-max 850

Run with the ExoJAX venv (offline tool); the resulting SQLite DB is consumed by the
shipped, ExoJAX-free cflibs inversion.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

CM_TO_EV = 1.239841984e-4  # eV per cm^-1 (hc/e, matches datagen_v2.py)

SCHEMA = """
CREATE TABLE IF NOT EXISTS lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    element TEXT, sp_num INTEGER, wavelength_nm REAL, aki REAL,
    ei_ev REAL, ek_ev REAL, gi INTEGER, gk INTEGER, rel_int REAL,
    stark_w REAL, stark_alpha REAL, stark_shift REAL, is_resonance INTEGER,
    aki_uncertainty REAL, accuracy_grade TEXT,
    gamma_vdw_log REAL, gamma_self_log REAL, stark_w_source TEXT
);
"""


def _z_to_symbol(z: int) -> str:
    """Atomic number -> element symbol via periodictable (ExoJAX dep)."""
    import periodictable

    return periodictable.elements[z].symbol


def _read_clean_gf(path: Path) -> Path:
    """Drop the rare malformed fixed-width Kurucz records (typos like '-1 72'
    that break ExoJAX's fixed-column ``read_kurucz``). Returns a cleaned-file path.
    """
    raw = path.read_text().splitlines()
    kept = []
    for line in raw:
        try:
            float(line[0:11])
            float(line[11:18])
            float(line[24:36])
            float(line[52:64])
            kept.append(line)
        except Exception:
            continue
    if len(kept) == len(raw):
        return path
    clean = path.with_suffix(path.suffix + ".clean")
    clean.write_text("\n".join(kept) + "\n")
    print(f"  cleaned {path.name}: kept {len(kept)}/{len(raw)} records -> {clean.name}")
    return clean


def ingest_gf(gf_path: Path, conn: sqlite3.Connection, wl_min: float, wl_max: float) -> int:
    """Parse one Kurucz gf file via ExoJAX AdbKurucz and insert into ``lines``."""
    from exojax.database.kurucz.api import AdbKurucz

    from cflibs.atomic.wavelength_conversion import vacuum_to_air_nm

    clean = _read_clean_gf(gf_path)
    # ExoJAX nurange is wavenumber [cm^-1]; gpu_transfer=True is REQUIRED (the
    # False path skips self.ielem creation that AdbKurucz.__init__ then reads).
    nu_range = [1e7 / wl_max, 1e7 / wl_min]
    adb = AdbKurucz(str(clean), nurange=nu_range)

    nu = np.asarray(adb.nu_lines)
    wl_vac_nm = 1e7 / nu
    wl_air_nm = np.asarray(vacuum_to_air_nm(wl_vac_nm))
    aki = np.asarray(adb.A)
    ei_ev = np.asarray(adb.elower) * CM_TO_EV
    ek_ev = np.asarray(adb.eupper) * CM_TO_EV
    gk = np.asarray(adb.gupper).astype(int)
    gi = (2 * np.asarray(adb.jlower) + 1).astype(int)
    ielem = np.asarray(adb.ielem).astype(int)
    iion = np.asarray(adb.iion).astype(int)  # ExoJAX iion: neutral=1
    gam_vdw = np.asarray(adb._vdWdamp) if hasattr(adb, "_vdWdamp") else np.zeros_like(aki)

    rows = []
    for i in range(len(nu)):
        sym = _z_to_symbol(int(ielem[i]))
        rows.append(
            (
                sym,
                int(iion[i]),
                float(wl_air_nm[i]),
                float(aki[i]),
                float(ei_ev[i]),
                float(ek_ev[i]),
                int(gi[i]),
                int(gk[i]),
                None,  # rel_int: Kurucz has none
                None,  # stark_w: not derived here (gamSta is log; separate lever)
                None,  # stark_alpha
                None,  # stark_shift
                1 if ei_ev[i] < 0.01 else 0,  # is_resonance (ground-state line)
                None,  # aki_uncertainty: Kurucz carries no per-line grade
                "U",  # accuracy_grade: unknown (Kurucz theoretical/semi-empirical)
                float(gam_vdw[i]) if np.isfinite(gam_vdw[i]) else None,
                None,  # gamma_self_log
                "kurucz",  # provenance
            )
        )
    conn.executemany(
        "INSERT INTO lines (element,sp_num,wavelength_nm,aki,ei_ev,ek_ev,gi,gk,"
        "rel_int,stark_w,stark_alpha,stark_shift,is_resonance,aki_uncertainty,"
        "accuracy_grade,gamma_vdw_log,gamma_self_log,stark_w_source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gf", nargs="+", required=True, help="Kurucz gf????.all file(s)")
    ap.add_argument("--db", required=True, help="output SQLite DB path")
    ap.add_argument("--wl-min", type=float, default=240.0, help="min AIR wavelength nm")
    ap.add_argument("--wl-max", type=float, default=850.0, help="max AIR wavelength nm")
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)

    total = 0
    for gf in args.gf:
        p = Path(gf)
        print(f"Ingesting {p.name} ...")
        n = ingest_gf(p, conn, args.wl_min, args.wl_max)
        print(f"  inserted {n} lines")
        total += n
    print(f"\nDONE: {total} Kurucz lines -> {db_path}")
    conn.close()


if __name__ == "__main__":
    main()
