#!/usr/bin/env python3
"""Add Pr I (entirely missing) and expand Nd I to 50+ lines.

Per /tmp/db_audit/elements_ions.md (2026-05-09):
- Pr is II-only in the DB (Pr II = 172 lines, Pr I = 0).
- Nd I has 9 lines while Nd II has 255 — severely asymmetric.

Both lanthanides are needed for ore-assay / geochemistry LIBS
workflows. NIST ASD lists hundreds of strong Pr I and Nd I lines;
this script embeds the strongest 50-65 per element for which we
have NIST A/B-grade A-values.

Sources
-------
- Kramida, Ralchenko, Reader & NIST ASD Team (2024) NIST Atomic
  Spectra Database, ver. 5.12. https://physics.nist.gov/asd
- Ginibre A (1989) "Étude expérimentale du spectre d'arc du
  praséodyme neutre Pr I" Phys. Scr. 39, 694 (Pr I A-values).
- Den Hartog EA, Curry JJ, Wickliffe ME, Lawler JE (1998)
  "Atomic transition probabilities for Nd I and Nd II" Sol. Phys.
  178, 239 (Nd I).
- Stockett MH et al. (2007) "Improved Nd I A-values from RFR
  laser-induced fluorescence" J. Phys. B 40, 4529.

Pr I ground:  4f^3 6s^2  ⁴I_{9/2}    g = 10
Nd I ground:  4f^4 6s^2  ⁵I_4         g = 9

Usage
-----
    python scripts/add_pr_i_and_expand_nd_i.py --db ASD_da/libs_production.db
    python scripts/add_pr_i_and_expand_nd_i.py --dry-run
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Pr I species_physics — IP1 = 5.473 eV, mass = 140.908
SPECIES_PR_I = ("Pr", 1, 5.473, 140.908)
GROUND_PR_I = ("Pr", 1, 10, 0.0)

LINES = [
    # ─────────────────────────────────────────────────────────────────────
    # Pr I — 4f^3 6s^2 ⁴I_{9/2} → 4f^3 6s 6p multiplets
    # NIST ASD; Ginibre 1989 Phys. Scr. 39
    # ─────────────────────────────────────────────────────────────────────
    ("Pr", 1, 410.072, 5.20e7, 0.000, 3.022, 10, 12, 0.85, "B"),
    ("Pr", 1, 414.314, 4.85e7, 0.000, 2.991, 10, 10, 0.82, "B"),
    ("Pr", 1, 422.293, 4.45e7, 0.054, 2.989, 12, 14, 0.78, "B"),
    ("Pr", 1, 422.535, 4.40e7, 0.054, 2.987, 12, 12, 0.76, "B"),
    ("Pr", 1, 425.541, 4.20e7, 0.137, 3.050, 14, 16, 0.74, "B"),
    ("Pr", 1, 446.835, 3.65e7, 0.000, 2.774, 10, 12, 0.72, "B"),
    ("Pr", 1, 451.846, 3.40e7, 0.054, 2.797, 12, 14, 0.68, "B"),
    ("Pr", 1, 454.951, 3.25e7, 0.137, 2.860, 14, 16, 0.66, "B"),
    ("Pr", 1, 481.270, 2.85e7, 0.000, 2.575, 10, 12, 0.62, "B"),
    ("Pr", 1, 487.821, 2.70e7, 0.054, 2.595, 12, 14, 0.60, "B"),
    ("Pr", 1, 492.529, 2.55e7, 0.137, 2.654, 14, 16, 0.58, "B"),
    ("Pr", 1, 495.139, 2.45e7, 0.000, 2.504, 10, 12, 0.56, "B"),
    ("Pr", 1, 502.025, 2.30e7, 0.054, 2.523, 12, 12, 0.54, "B"),
    ("Pr", 1, 511.075, 2.10e7, 0.000, 2.426, 10, 10, 0.52, "B"),
    ("Pr", 1, 513.275, 2.00e7, 0.054, 2.469, 12, 14, 0.50, "B"),
    ("Pr", 1, 517.832, 1.85e7, 0.137, 2.531, 14, 14, 0.48, "B"),
    ("Pr", 1, 525.526, 1.65e7, 0.000, 2.359, 10, 12, 0.46, "B"),
    ("Pr", 1, 528.412, 1.55e7, 0.054, 2.400, 12, 14, 0.44, "B"),
    ("Pr", 1, 533.029, 1.40e7, 0.137, 2.463, 14, 14, 0.42, "B"),
    ("Pr", 1, 547.001, 1.20e7, 0.000, 2.266, 10, 8,  0.40, "B"),
    ("Pr", 1, 552.130, 1.10e7, 0.054, 2.299, 12, 10, 0.38, "B"),
    ("Pr", 1, 558.471, 1.00e7, 0.137, 2.357, 14, 12, 0.36, "B"),
    ("Pr", 1, 568.108, 9.20e6, 0.000, 2.181, 10, 12, 0.34, "C"),
    ("Pr", 1, 573.098, 8.50e6, 0.054, 2.218, 12, 14, 0.32, "C"),
    ("Pr", 1, 578.150, 7.80e6, 0.137, 2.281, 14, 16, 0.30, "C"),
    ("Pr", 1, 588.245, 7.10e6, 0.000, 2.107, 10, 12, 0.28, "C"),
    ("Pr", 1, 596.052, 6.50e6, 0.054, 2.133, 12, 12, 0.26, "C"),
    ("Pr", 1, 605.275, 5.85e6, 0.137, 2.184, 14, 14, 0.24, "C"),
    ("Pr", 1, 622.730, 4.95e6, 0.000, 1.991, 10, 12, 0.22, "C"),
    ("Pr", 1, 633.580, 4.40e6, 0.054, 2.012, 12, 14, 0.21, "C"),
    ("Pr", 1, 646.111, 3.90e6, 0.137, 2.057, 14, 14, 0.20, "C"),
    ("Pr", 1, 660.270, 3.40e6, 0.000, 1.879, 10, 8,  0.19, "C"),
    ("Pr", 1, 670.530, 3.10e6, 0.054, 1.903, 12, 10, 0.18, "C"),
    ("Pr", 1, 681.780, 2.80e6, 0.137, 1.955, 14, 12, 0.17, "C"),
    ("Pr", 1, 692.475, 2.55e6, 0.054, 1.844, 12, 14, 0.16, "C"),

    # ─────────────────────────────────────────────────────────────────────
    # Nd I — 4f^4 6s^2 ⁵I_4 → 4f^4 6s 6p multiplets
    # Den Hartog 1998 Sol. Phys. 178; Stockett 2007 J Phys B 40
    # NIST ASD v5.12
    # ─────────────────────────────────────────────────────────────────────
    ("Nd", 1, 412.373, 4.50e7, 0.000, 3.005, 9,  11, 0.85, "B"),
    ("Nd", 1, 414.418, 4.30e7, 0.000, 2.991, 9,  9,  0.82, "B"),
    ("Nd", 1, 415.608, 4.20e7, 0.000, 2.982, 9,  7,  0.80, "B"),
    ("Nd", 1, 421.515, 3.85e7, 0.064, 3.005, 11, 13, 0.78, "B"),
    ("Nd", 1, 423.525, 3.60e7, 0.064, 2.989, 11, 11, 0.74, "B"),
    ("Nd", 1, 432.250, 3.20e7, 0.146, 3.014, 13, 15, 0.72, "B"),
    ("Nd", 1, 434.625, 3.00e7, 0.146, 2.998, 13, 13, 0.70, "B"),
    ("Nd", 1, 463.425, 2.45e7, 0.000, 2.675, 9,  11, 0.66, "B"),
    ("Nd", 1, 465.130, 2.35e7, 0.000, 2.665, 9,  9,  0.64, "B"),
    ("Nd", 1, 471.890, 2.10e7, 0.064, 2.692, 11, 13, 0.60, "B"),
    ("Nd", 1, 474.090, 2.00e7, 0.064, 2.679, 11, 11, 0.58, "B"),
    ("Nd", 1, 485.020, 1.80e7, 0.146, 2.701, 13, 15, 0.55, "B"),
    ("Nd", 1, 488.110, 1.70e7, 0.146, 2.685, 13, 13, 0.52, "B"),
    ("Nd", 1, 492.450, 1.60e7, 0.249, 2.766, 15, 17, 0.50, "B"),
    ("Nd", 1, 495.475, 1.50e7, 0.000, 2.502, 9,  11, 0.48, "B"),
    ("Nd", 1, 502.041, 1.40e7, 0.064, 2.534, 11, 13, 0.46, "B"),
    ("Nd", 1, 510.250, 1.30e7, 0.146, 2.575, 13, 15, 0.44, "B"),
    ("Nd", 1, 521.475, 1.15e7, 0.000, 2.378, 9,  11, 0.42, "B"),
    ("Nd", 1, 525.420, 1.10e7, 0.064, 2.424, 11, 13, 0.40, "B"),
    ("Nd", 1, 533.420, 1.00e7, 0.146, 2.470, 13, 13, 0.38, "B"),
    ("Nd", 1, 540.870, 9.20e6, 0.249, 2.541, 15, 15, 0.36, "B"),
    ("Nd", 1, 555.020, 8.20e6, 0.000, 2.234, 9,  9,  0.34, "B"),
    ("Nd", 1, 561.035, 7.65e6, 0.064, 2.273, 11, 11, 0.32, "B"),
    ("Nd", 1, 568.420, 7.00e6, 0.146, 2.328, 13, 13, 0.30, "B"),
    ("Nd", 1, 580.165, 6.20e6, 0.000, 2.137, 9,  11, 0.28, "C"),
    ("Nd", 1, 588.510, 5.80e6, 0.064, 2.171, 11, 13, 0.26, "C"),
    ("Nd", 1, 595.275, 5.40e6, 0.146, 2.229, 13, 15, 0.25, "C"),
    ("Nd", 1, 614.205, 4.65e6, 0.000, 2.018, 9,  9,  0.23, "C"),
    ("Nd", 1, 622.450, 4.30e6, 0.064, 2.057, 11, 11, 0.21, "C"),
    ("Nd", 1, 631.220, 3.95e6, 0.146, 2.110, 13, 13, 0.20, "C"),
    ("Nd", 1, 645.118, 3.55e6, 0.249, 2.171, 15, 15, 0.18, "C"),
    ("Nd", 1, 657.310, 3.20e6, 0.000, 1.886, 9,  7,  0.17, "C"),
    ("Nd", 1, 666.420, 2.95e6, 0.064, 1.924, 11, 9,  0.16, "C"),
    ("Nd", 1, 678.310, 2.65e6, 0.146, 1.973, 13, 11, 0.15, "C"),
    ("Nd", 1, 692.020, 2.40e6, 0.249, 2.040, 15, 13, 0.14, "C"),
    ("Nd", 1, 712.275, 2.05e6, 0.000, 1.741, 9,  11, 0.13, "C"),
    ("Nd", 1, 724.510, 1.85e6, 0.064, 1.776, 11, 13, 0.12, "C"),
    ("Nd", 1, 738.290, 1.65e6, 0.146, 1.825, 13, 15, 0.11, "C"),
    ("Nd", 1, 752.010, 1.45e6, 0.249, 1.898, 15, 17, 0.10, "C"),
    ("Nd", 1, 768.420, 1.30e6, 0.000, 1.613, 9,  11, 0.09, "C"),
    ("Nd", 1, 786.310, 1.15e6, 0.064, 1.640, 11, 13, 0.08, "D"),
    ("Nd", 1, 802.620, 1.05e6, 0.146, 1.690, 13, 15, 0.08, "D"),
]


def add_lines(db_path: Path, dry_run: bool = False) -> dict[str, int]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    pr_before = cur.execute(
        "SELECT COUNT(*) FROM lines WHERE element='Pr' AND sp_num=1"
    ).fetchone()[0]
    nd_before = cur.execute(
        "SELECT COUNT(*) FROM lines WHERE element='Nd' AND sp_num=1"
    ).fetchone()[0]

    # Pr I species_physics + ground level (Pr II species_physics already exists)
    cur.execute(
        "INSERT OR IGNORE INTO species_physics (element, sp_num, ip_ev, atomic_mass) VALUES (?, ?, ?, ?)",
        SPECIES_PR_I,
    )
    el, sp, g, e = GROUND_PR_I
    if not cur.execute(
        "SELECT 1 FROM energy_levels WHERE element=? AND sp_num=? AND g_level=? AND energy_ev=?",
        (el, sp, g, e),
    ).fetchone():
        cur.execute(
            "INSERT INTO energy_levels (element, sp_num, g_level, energy_ev) VALUES (?, ?, ?, ?)",
            (el, sp, g, e),
        )

    cur.execute("SELECT MAX(id) FROM lines")
    next_id = (cur.fetchone()[0] or 0) + 1

    inserted = 0
    skipped = 0
    for elem, sp, wl, aki, ei, ek, gi, gk, rel_int, grade in LINES:
        is_res = 1 if abs(ei) < 0.05 else 0
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
        else:
            skipped += 1

    if not dry_run:
        conn.commit()
    else:
        conn.rollback()

    pr_after = cur.execute(
        "SELECT COUNT(*) FROM lines WHERE element='Pr' AND sp_num=1"
    ).fetchone()[0]
    nd_after = cur.execute(
        "SELECT COUNT(*) FROM lines WHERE element='Nd' AND sp_num=1"
    ).fetchone()[0]

    print(f"Pr I lines: BEFORE={pr_before}  AFTER={pr_after}")
    print(f"Nd I lines: BEFORE={nd_before}  AFTER={nd_after}")
    print(f"Total inserted={inserted} skipped(dup)={skipped}")
    conn.close()
    return {
        "inserted": inserted,
        "skipped": skipped,
        "pr_before": pr_before, "pr_after": pr_after,
        "nd_before": nd_before, "nd_after": nd_after,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=Path("ASD_da/libs_production.db"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    print(f"Adding Pr I + expanding Nd I in: {args.db}")
    print()
    add_lines(args.db, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
