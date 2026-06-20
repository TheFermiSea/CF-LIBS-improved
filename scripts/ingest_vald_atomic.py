#!/usr/bin/env python
"""Ingest VALD3 "Extract All / Element" long-format line lists into the cflibs schema.

M5 / Lever 1C+1B (accuracy-first roadmap): VALD3 is the GRADED completeness backend.
Unlike Kurucz-alone (grade 'U'), VALD records a per-line source reference (``gf:RU``
experimental vs ``gf:K10`` Kurucz-theoretical), so we can assign a real accuracy
grade. This is an OFFLINE acquisition tool (uses no shipped cflibs dependency beyond
the air<->vacuum converter); it parses the VALD text directly (ExoJAX's read_ExAll
drops the provenance line, and AdbVald's stellar solar-abundance step crashes on
broad/high-ion element sets) and writes the cflibs SQLite ``lines`` schema.

Handles multiple wavelength-chunk dumps: pass them all; records are concatenated and
de-duplicated by (element, ion, wavelength, E_low, E_up) across slice boundaries.

VALD request that produces a compatible file: Extract All/Element, **Long format**,
units eV / air / angstrom. Gzipped (.gz) or plain text both accepted.

Usage:
    PYTHONPATH=$PWD <venv>/bin/python scripts/ingest_vald_atomic.py \
        --vald data/vald/*.linelist.gz --db output/vald_atomic.db \
        --wl-min 200 --wl-max 900
"""

from __future__ import annotations

import argparse
import gzip
import re
import sqlite3
from pathlib import Path
from typing import Iterator, Optional

# A_ki [s^-1] = 6.6702e15 * gf / (g_upper * lambda_Angstrom^2), gf = g_lower * f.
_AKI_CONST = 6.6702e15
_GF_RE = re.compile(r"gf:(\S+)")
# Data record: 'Elm Ion', WL_air(A), loggf, E_low(eV), J_lo, E_up(eV), J_up, ...
_DATA_RE = re.compile(r"^'([A-Z][a-z]?) +(\d+)'")

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


def _grade_from_gf_source(code: Optional[str]) -> str:
    """Map a VALD ``gf:`` reference code to an accuracy grade (HEURISTIC).

    VALD reference codes encode provenance. Critically-evaluated / experimental
    sources are markedly more accurate than Kurucz semi-empirical calculations.
    First-letter convention (VALD bibliography): K* = Kurucz (calc), R*/F*/U* =
    measured/critically-evaluated. Pending empirical tuning against NIST grades
    on the overlap set; conservative tiers for now.
    """
    if not code:
        return "U"
    head = code[0].upper()
    if head in ("R", "F"):  # experimental / Fourier-transform measurements
        return "B"
    if head == "U":  # critically-evaluated compilations
        return "B"
    if head == "K":  # Kurucz theoretical / semi-empirical
        return "D"
    return "U"


def _open(path: Path):
    return gzip.open(path, "rt", errors="replace") if path.suffix == ".gz" else open(path)


def parse_vald(path: Path, wl_min_nm: float, wl_max_nm: float) -> Iterator[dict]:
    """Yield one record per VALD transition (data row + its ``gf:`` reference).

    A transition spans a data record (``'Elm Ion', ...``) followed (after the two
    LS/term lines) by a reference record (``'_ ... gf:CODE ...``). We pair each
    data row with the next reference row, robust to intervening config lines.
    """
    from cflibs.atomic.wavelength_conversion import vacuum_to_air_nm

    with _open(path) as fh:
        pending: Optional[dict] = None
        for raw in fh:
            line = raw.rstrip("\n")
            m = _DATA_RE.match(line)
            if m:
                # Flush a previous data row that never found a reference.
                if pending is not None:
                    yield pending
                    pending = None
                parts = [p.strip() for p in line.split(",")]
                try:
                    element = m.group(1)
                    ion = int(m.group(2))  # VALD: 1=neutral, 2=singly ionized
                    wl_a = float(parts[1])
                    loggf = float(parts[2])
                    e_low = float(parts[3])
                    j_low = float(parts[4])
                    e_up = float(parts[5])
                    j_up = float(parts[6])
                    gam_rad = float(parts[10]) if len(parts) > 10 and parts[10] else None
                    gam_waals = float(parts[12]) if len(parts) > 12 and parts[12] else None
                except (ValueError, IndexError):
                    continue
                wl_nm = wl_a / 10.0
                # VALD gives air >=2000 A, vacuum below; store AIR (cflibs convention).
                if wl_nm < 200.0:
                    wl_nm = float(vacuum_to_air_nm(wl_nm))
                if not (wl_min_nm <= wl_nm <= wl_max_nm):
                    pending = None
                    continue
                g_up = int(round(2 * j_up + 1))
                g_low = int(round(2 * j_low + 1))
                aki = _AKI_CONST * (10.0**loggf) / (g_up * wl_a * wl_a) if g_up > 0 else 0.0
                ei, ek = (e_low, e_up) if e_up >= e_low else (e_up, e_low)
                pending = {
                    "element": element,
                    "sp_num": ion,
                    "wavelength_nm": wl_nm,
                    "aki": aki,
                    "ei_ev": ei,
                    "ek_ev": ek,
                    "gi": g_low,
                    "gk": g_up,
                    "is_resonance": 1 if ei < 0.01 else 0,
                    "gamma_vdw_log": gam_waals,
                    "gamma_rad_log": gam_rad,
                    "accuracy_grade": "U",
                }
            elif pending is not None and line.startswith("'_"):
                gfm = _GF_RE.search(line)
                pending["accuracy_grade"] = _grade_from_gf_source(gfm.group(1) if gfm else None)
                yield pending
                pending = None
        if pending is not None:
            yield pending


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vald", nargs="+", required=True, help="VALD long-format dump(s), .gz ok")
    ap.add_argument("--db", required=True, help="output SQLite DB path")
    ap.add_argument("--wl-min", type=float, default=200.0, help="min AIR wavelength nm")
    ap.add_argument("--wl-max", type=float, default=900.0, help="max AIR wavelength nm")
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)

    seen: set = set()
    total = inserted = 0
    grade_counts: dict = {}
    for vf in args.vald:
        p = Path(vf)
        print(f"Parsing {p.name} ...")
        n_file = 0
        rows = []
        for rec in parse_vald(p, args.wl_min, args.wl_max):
            total += 1
            n_file += 1
            key = (
                rec["element"],
                rec["sp_num"],
                round(rec["wavelength_nm"], 4),
                round(rec["ei_ev"], 4),
                round(rec["ek_ev"], 4),
            )
            if key in seen:
                continue  # de-dup across slice boundaries
            seen.add(key)
            grade_counts[rec["accuracy_grade"]] = grade_counts.get(rec["accuracy_grade"], 0) + 1
            rows.append(
                (
                    rec["element"],
                    rec["sp_num"],
                    rec["wavelength_nm"],
                    rec["aki"],
                    rec["ei_ev"],
                    rec["ek_ev"],
                    rec["gi"],
                    rec["gk"],
                    None,
                    None,
                    None,
                    None,
                    rec["is_resonance"],
                    None,
                    rec["accuracy_grade"],
                    rec["gamma_vdw_log"],
                    None,
                    "vald",
                )
            )
            inserted += 1
        conn.executemany(
            "INSERT INTO lines (element,sp_num,wavelength_nm,aki,ei_ev,ek_ev,gi,gk,"
            "rel_int,stark_w,stark_alpha,stark_shift,is_resonance,aki_uncertainty,"
            "accuracy_grade,gamma_vdw_log,gamma_self_log,stark_w_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        print(f"  {n_file} parsed, {len(rows)} new (after dedup)")
    print(f"\nDONE: {inserted} unique VALD lines from {total} parsed -> {db_path}")
    print(f"grade distribution (provenance-derived): {grade_counts}")
    conn.close()


if __name__ == "__main__":
    main()
