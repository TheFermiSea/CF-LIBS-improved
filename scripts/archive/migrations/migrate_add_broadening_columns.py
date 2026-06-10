#!/usr/bin/env python3
"""Add Van der Waals + self-broadening + Stark provenance columns to the
``lines`` table.

Idempotent — safe to re-run; checks ``PRAGMA table_info(lines)`` before
attempting any ``ALTER TABLE``.

New columns
-----------

``gamma_vdw_log REAL``
    Van der Waals damping constant, ``log10(gamma_vdw)`` in rad/s, at the
    canonical Anstee-O'Mara reference: T = 10000 K, perturber density
    n_p = 1e22 cm^-3 (≈ standard atmospheric density × 10^3, the
    "Kurucz unit" used in synth.f / SYNTHE / VALD3 line lists).

``gamma_self_log REAL``
    Self-broadening damping constant, ``log10(gamma_self)`` in rad/s, at
    the same reference conditions but with the perturber being neutral
    atoms of the emitter species.

``stark_w_source TEXT``
    Provenance tag for ``stark_w``. Allowed values (free-form TEXT, no
    CHECK constraint, but downstream code branches on these):

      ``"stark_b"``                       — line-specific STARK-B / Konjević /
                                            Dimitrijević / Sahal-Brechot value.
      ``"konjevic_lambda_sq_scaled"``     — Konjević 2002 reference-line value
                                            scaled via ``(λ/λ_ref)^2``.
      ``"interpolated"``                  — Konjević scaling applied to a new
                                            (element, ion) pair via
                                            ``fill_stark_gaps.py``.
      ``"hydrogenic"``                    — Griem 1974 hydrogenic estimate
                                            (used for noble gases).
      ``"lanthanide_default"``            — flat fallback for lanthanide
                                            neutrals/ions without published
                                            data.
      ``NULL``                            — no Stark width populated.

Usage
-----

    python scripts/archive/migrations/migrate_add_broadening_columns.py
    python scripts/archive/migrations/migrate_add_broadening_columns.py --db ASD_da/libs_production.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

NEW_COLUMNS = [
    ("gamma_vdw_log", "REAL"),
    ("gamma_self_log", "REAL"),
    ("stark_w_source", "TEXT"),
]


def existing_columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def migrate(db_path: Path) -> dict[str, list[str]]:
    """Add missing broadening columns. Returns dict of added/skipped lists."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    have = existing_columns(cur, "lines")
    added: list[str] = []
    skipped: list[str] = []
    for name, sqltype in NEW_COLUMNS:
        if name in have:
            skipped.append(name)
            continue
        cur.execute(f"ALTER TABLE lines ADD COLUMN {name} {sqltype}")
        added.append(name)
    conn.commit()
    conn.close()
    return {"added": added, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("ASD_da/libs_production.db"),
        help="Path to atomic database (default: ASD_da/libs_production.db)",
    )
    args = parser.parse_args()

    print(f"Migrating broadening columns in: {args.db}")
    result = migrate(args.db)
    if result["added"]:
        print(f"  added:   {', '.join(result['added'])}")
    if result["skipped"]:
        print(f"  already present (skipped): {', '.join(result['skipped'])}")
    if not result["added"] and not result["skipped"]:
        print("  no columns to migrate (unexpected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
