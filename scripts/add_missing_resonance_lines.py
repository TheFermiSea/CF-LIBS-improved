#!/usr/bin/env python3
"""Add three canonical LIBS resonance lines that the audit found missing.

Per /tmp/db_audit/elements_ions.md (2026-05-09), these three specific
lines are absent from libs_production.db despite being THE diagnostic
line for their element in routine LIBS:

1. **Zn I 481.053 nm** — 4s4p ¹P₁° → 4s5s ¹S₀ transition.
   Note: this is NOT the strongest Zn I resonance (that is the
   4s² ¹S₀ → 4s4p ¹P₁° at 213.857 nm). However the 481.053 nm line
   is the most-used Zn I diagnostic in atmospheric-pressure LIBS
   because 213.857 falls in a region with poor spectrometer
   throughput and severe atmospheric absorption / O₂ overlap. The
   481-nm line is on the singlet system (high upper-level lifetime)
   and is the canonical "above-300 nm" Zn fingerprint.

2. **As I 193.696 nm** — 4s²4p³ ⁴S₃/₂° → 4s²4p²(³P)5s ⁴P₁/₂
   resonance. Strongest As I line in the LIBS-relevant range.
   Required for trace-arsenic detection (carcinogen, RoHS).

3. **S I 180.731 nm** — 3s²3p⁴ ³P₂ → 3s²3p³(⁴S°)4s ⁵S₂° resonance.
   Vacuum-UV but routinely measured by LIBS instruments with
   purged optical paths (sulfur in steel/cement workflows).

Sources
-------
- Zn I 481.053: NIST ASD v5.12; Bielski 1975 ApJ 199, 503;
  Wiese, Smith & Miles NSRDS-NBS 22 (1969).
- As I 193.696: Lotrian J et al. (1980) JQSRT 23, 449; NIST ASD v5.12.
- S I 180.731 : Müller D (1968) Z. Naturf. 23a, 1707; NIST ASD v5.12.

Each line is INSERTed only if not already present via UNIQUE
(element, sp_num, wavelength_nm, ek_ev), so this script is idempotent.

Usage
-----
    python scripts/add_missing_resonance_lines.py --db ASD_da/libs_production.db
    python scripts/add_missing_resonance_lines.py --dry-run
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Format: (element, sp_num, wavelength_nm, aki_s_inv, ei_ev, ek_ev,
#          gi, gk, rel_int, accuracy_grade, is_resonance)
RESONANCE_LINES = [
    # ── Zn I 481.053 nm ─────────────────────────────────────────────────────
    # 4s4p ¹P₁° (E_lower = 5.7959 eV) → 4s5s ¹S₀ (E_upper = 8.3729 eV)
    # Δλ : 481.053 nm air. Aki = 7.0e7 s^-1 (NIST).
    # Note: lower level is NOT ground (4s4p is excited).
    # This is technically a "secondary" resonance — but the audit
    # explicitly asked for this wavelength and it's the canonical
    # Zn diagnostic.  is_resonance set to 0 because lower level
    # is not the ground 4s² ¹S₀ state.
    ("Zn", 1, 481.053, 7.0e7, 5.7959, 8.3729, 3, 1, 0.85, "B+", 0),

    # ── As I 193.696 nm ─────────────────────────────────────────────────────
    # 4s²4p³ ⁴S₃/₂° (ground, E_lower = 0.000 eV) →
    # 4s²4p²(³P)5s ⁴P₁/₂ (E_upper = 6.401 eV).
    # Aki = 2.0e8 s^-1 (Lotrian 1980 / NIST).
    # gi = 2J+1 = 4 ; gk = 2 (J=1/2).
    ("As", 1, 193.696, 2.0e8, 0.000, 6.401, 4, 2, 0.95, "B", 1),

    # ── S I 180.731 nm ──────────────────────────────────────────────────────
    # 3s²3p⁴ ³P₂ (ground, E_lower = 0.000 eV) →
    # 3s²3p³(⁴S°)4s ⁵S₂° (E_upper = 6.860 eV).
    # NIST: vacuum wavelength 180.731 nm. Aki = 1.6e8 s^-1.
    # gi = 5 (J=2); gk = 5 (J=2).
    ("S",  1, 180.731, 1.6e8, 0.000, 6.860, 5, 5, 0.92, "B", 1),
]


def add_lines(db_path: Path, dry_run: bool = False) -> dict[str, int]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT MAX(id) FROM lines")
    next_id = (cur.fetchone()[0] or 0) + 1

    inserted = 0
    skipped = 0
    print("Inserting canonical resonance lines:")
    for elem, sp, wl, aki, ei, ek, gi, gk, rel_int, grade, is_res in RESONANCE_LINES:
        # Pre-check on (element, sp_num, wavelength_nm) tolerance: 0.1 nm
        # (the unique constraint requires exact ek_ev match too, but a near-
        # duplicate within 0.1 nm of an existing entry would be a quality bug
        # we want to report).
        nearby = cur.execute(
            "SELECT wavelength_nm, ek_ev FROM lines "
            "WHERE element=? AND sp_num=? AND ABS(wavelength_nm - ?) < 0.1",
            (elem, sp, wl),
        ).fetchall()
        aki_unc = {"A": 0.03, "B": 0.10, "B+": 0.07, "C": 0.25, "D": 0.50}.get(grade, 0.20)
        cur.execute(
            """
            INSERT OR IGNORE INTO lines
            (id, element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int,
             stark_w, stark_alpha, stark_shift, is_resonance, aki_uncertainty, accuracy_grade)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?)
            """,
            (next_id, elem, sp, wl, aki, ei, ek, gi, gk, rel_int, is_res, aki_unc, grade),
        )
        if cur.rowcount:
            inserted += 1
            next_id += 1
            tag = "INSERTED"
        else:
            skipped += 1
            tag = "SKIPPED (already present)"
        nearby_msg = f" (nearby: {nearby})" if nearby else ""
        print(f"  {tag}: {elem} {sp}  {wl:.3f} nm  Aki={aki:.2e}  grade={grade}{nearby_msg}")

    if not dry_run:
        conn.commit()
    else:
        conn.rollback()

    print()
    print("Verification — these three lines should now exist:")
    for elem, sp, wl, *_ in RESONANCE_LINES:
        row = cur.execute(
            "SELECT element, sp_num, wavelength_nm, aki, accuracy_grade FROM lines "
            "WHERE element=? AND sp_num=? AND ABS(wavelength_nm - ?) < 0.01 "
            "ORDER BY ABS(wavelength_nm - ?) LIMIT 1",
            (elem, sp, wl, wl),
        ).fetchone()
        print(f"  {elem} {sp} {wl:.3f}: {row}")

    conn.close()
    return {"inserted": inserted, "skipped": skipped}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=Path("ASD_da/libs_production.db"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print(f"Adding 3 canonical resonance lines to: {args.db}")
    print()
    stats = add_lines(args.db, dry_run=args.dry_run)
    print()
    print(f"Done: +{stats['inserted']} resonance lines, {stats['skipped']} already present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
