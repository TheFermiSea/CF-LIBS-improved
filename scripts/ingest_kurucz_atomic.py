#!/usr/bin/env python
"""Ingest Kurucz atomic line lists (``gf????.all``) into the cflibs atomic schema.

M5 / Lever 1C (accuracy-first roadmap): Kurucz is the *completeness* backend for
the atomic line list (measured ~5-6 wt% better than NIST on line-rich SuperCam;
~15x more Fe I lines in 240-850 nm). This is an OFFLINE acquisition tool with its
own ROBUST direct fixed-width parser (NOT ExoJAX's brittle ``read_kurucz``) that
ingests BOTH ``gf<Zion>.all`` and the older ``gf<Zion>.lines`` files (same layout)
and is tolerant of the Kurucz vintage column-drift in the damping fields that makes
``read_kurucz`` skip whole older-era files (e.g. Ca I gf2000). Downloaded from
http://kurucz.harvard.edu/atoms/<Zion>/; writes a SQLite ``lines`` table matching
``datagen_v2.py`` so the existing ``AtomicDatabase`` consumes it unchanged.

Key normalizations (so the line list, wavelength solution and forward model share
one convention — Ciucci/Tognoni):
- Kurucz wavelengths are vacuum below 200 nm, air above; we convert the <200 nm
  ones to AIR via the ``cflibs.atomic.wavelength_conversion`` Edlen/Morton converter
  (M2 prereq, #307), matching the NIST ``obs_wl_air`` convention used by the DB.
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


def ingest_gf(gf_path: Path, conn: sqlite3.Connection, wl_min: float, wl_max: float) -> int:
    """Robustly parse one Kurucz gf file (``.all`` OR ``.lines``) into ``lines``.

    Direct fixed-width parser of the STABLE Kurucz columns, tolerant of the
    vintage column-drift in the damping/reference region (cols 80+) that breaks
    ExoJAX's ``read_kurucz`` (which forces whole older-vintage files, e.g. Ca I
    gf2000, to be skipped). ``.all`` and ``.lines`` share this exact layout, so
    both ingest identically. We read only what cflibs needs::

        [0:11] wl(nm; <200 vacuum else air)  [11:18] log gf  [18:24] code Z.charge
        [24:36] E_lower(cm^-1, may be <0 = predicted)  [36:41] J_lower
        [52:64] E_upper(cm^-1)  [64:69] J_upper

    A_ki = 6.6702e15 * 10**loggf / (g_up * lambda_Angstrom^2). Levels are ordered
    by |energy| (Kurucz marks predicted levels with a negative sign).
    """
    from cflibs.atomic.wavelength_conversion import vacuum_to_air_nm

    aki_const = 6.6702e15
    rows = []
    bad = 0
    with open(gf_path, errors="replace") as fh:
        for line in fh:
            try:
                wl_nm = float(line[0:11])
                loggf = float(line[11:18])
                code = line[18:24].strip()
                z = int(code.split(".")[0])
                charge = int(code.split(".")[1])
                e1 = abs(float(line[24:36]))
                j1 = float(line[36:41])
                e2 = abs(float(line[52:64]))
                j2 = float(line[64:69])
            except (ValueError, IndexError):
                bad += 1
                continue
            wl_air_nm = float(vacuum_to_air_nm(wl_nm)) if wl_nm < 200.0 else wl_nm
            if not (wl_min <= wl_air_nm <= wl_max):
                continue
            ei, ek, jlo, jup = (e1, e2, j1, j2) if e2 >= e1 else (e2, e1, j2, j1)
            gi = int(round(2 * jlo + 1))
            gk = int(round(2 * jup + 1))
            wl_a = wl_air_nm * 10.0
            aki = aki_const * (10.0**loggf) / (gk * wl_a * wl_a) if gk > 0 else 0.0
            try:
                sym = _z_to_symbol(z)
            except Exception:
                bad += 1
                continue
            ei_ev = ei * CM_TO_EV
            rows.append(
                (
                    sym,
                    charge + 1,
                    wl_air_nm,
                    aki,
                    ei_ev,
                    ek * CM_TO_EV,
                    gi,
                    gk,
                    None,
                    None,
                    None,
                    None,
                    1 if ei_ev < 0.01 else 0,
                    None,
                    "U",
                    None,
                    None,
                    "kurucz",
                )
            )
    if bad:
        print(f"  ({bad} unparseable records skipped)")
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
    failed = []
    for gf in args.gf:
        p = Path(gf)
        print(f"Ingesting {p.name} ...")
        # Per-file guard: a file that still cannot be parsed (truncated/corrupt)
        # must NOT abort the whole multi-file run.
        try:
            n = ingest_gf(p, conn, args.wl_min, args.wl_max)
            print(f"  inserted {n} lines")
            total += n
        except Exception as exc:
            print(f"  SKIPPED {p.name}: {type(exc).__name__}: {exc} (Kurucz vintage drift?)")
            failed.append(p.name)
    print(f"\nDONE: {total} Kurucz lines -> {db_path}")
    if failed:
        print(f"SKIPPED {len(failed)} file(s) (read_kurucz vintage mismatch): {', '.join(failed)}")
    conn.close()


if __name__ == "__main__":
    main()
