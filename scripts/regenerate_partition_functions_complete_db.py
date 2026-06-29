#!/usr/bin/env python3
"""Regenerate ``partition_functions`` against the complete (ASD59) DB.

Why
---
The stored ``partition_functions`` polynomials (sources ``NIST_ASD_fit`` and the
earlier ``direct_sum_fit_v1``) were fit BEFORE the M5 gold-standard ASD59
energy-level ingest.  A direct-sum ``U(T) = Σ gᵢ exp(-Eᵢ/kT)`` is a *strict
lower bound* on the true partition function (you can only ever add levels), so a
polynomial fit to a SMALLER level set now evaluates **below** the DB's own
direct sum -- physically impossible (e.g. Cr II @ 10000 K U_poly=31.52 <
direct-sum 33.11, -4.8%).  Partition functions feed every Saha-Boltzmann solve,
so the stale rows cap absolute composition accuracy.

What this script does
---------------------
For every species with at least 2 fittable energy levels it persists EXACTLY
what :func:`cflibs.plasma.partition.derive_partition_spec` computes at load time
(the band-weighted direct-sum fit), tagged ``direct_sum_fit_v2``.  Because the
stored row and the live snapshot/Bayesian poly now come from the *same* function,
the parity gate (reads the stored row) and the path-consistency gate (computes
the live fit) see identical coefficients by construction.

Species with <2 fittable levels (e.g. Si IV-class high ions) are LEFT AS-IS so
their stored-polynomial fallback row survives -- we never fabricate a fit.
Rows whose existing source is authoritative
(:data:`cflibs.plasma.partition.AUTHORITATIVE_PF_SOURCES`, e.g. a Barklem &
Collet 2016 refit) are also left untouched.

The ``energy_levels`` table is NOT modified: levels with an unassigned ``J``
(NULL ``g_level``) are legitimate NIST metadata that the production direct sum
already excludes at the query layer (``get_energy_levels`` filters
``g_level IS NOT NULL AND g_level > 0``); they contribute no known Boltzmann
weight and are retained for provenance.

Usage
-----
    PYTHONPATH=$PWD python scripts/regenerate_partition_functions_complete_db.py \
        --db-path ASD_da/libs_production.db
    PYTHONPATH=$PWD python scripts/regenerate_partition_functions_complete_db.py \
        --db-path ASD_da/libs_production.db --dry-run
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import cflibs
from cflibs.atomic.database import AtomicDatabase
from cflibs.plasma.partition import (
    AUTHORITATIVE_PF_SOURCES,
    derive_partition_spec,
)

KB_EV = 8.617333262145e-5

# ps-LIBS validation band (matches the gate + the fit lift window).
BAND_LO = 6000.0
BAND_HI = 12000.0
N_BAND = 25

SOURCE_TAG = "direct_sum_fit_v2"
ACCEPT_REL_TOL = 0.10  # poly must stay within +10% of direct-sum across the band


def _direct_sum(db: AtomicDatabase, element: str, sp_num: int, T_K: float) -> float:
    """Production-faithful U(T) = Σ gᵢ exp(-Eᵢ/kT) over the API-filtered levels."""
    levels = db.get_energy_levels(element, sp_num)
    if not levels:
        return 1.0
    g = np.array([lev.g for lev in levels], dtype=np.float64)
    e = np.array([lev.energy_ev for lev in levels], dtype=np.float64)
    ip = db.get_ionization_potential(element, sp_num)
    ip_ev = float(ip) if ip is not None else float(np.max(e)) + 1.0
    mask = e < ip_ev
    if not mask.any():
        return 1.0
    return max(float(np.sum(g[mask] * np.exp(-e[mask] / (KB_EV * T_K)))), 1.0)


def _poly(coeffs, t_min: float, t_max: float, T_K: float) -> float:
    """Evaluate the stored ln-poly with the same clamp the evaluator applies."""
    T_eval = min(max(float(T_K), float(t_min)), float(t_max))
    ln_T = np.log(T_eval)
    return float(np.exp(sum(a * ln_T**i for i, a in enumerate(coeffs))))


def _existing_sources(cur: sqlite3.Cursor) -> dict[tuple[str, int], str]:
    return {
        (el, sp): (src or "")
        for el, sp, src in cur.execute(
            "SELECT element, sp_num, source FROM partition_functions"
        ).fetchall()
    }


def regenerate(db_path: Path, dry_run: bool) -> int:
    db = AtomicDatabase(str(db_path))
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    species = cur.execute(
        "SELECT element, sp_num FROM energy_levels "
        "WHERE g_level IS NOT NULL AND g_level > 0 AND energy_ev IS NOT NULL "
        "GROUP BY element, sp_num HAVING COUNT(*) >= 2 "
        "ORDER BY element, sp_num"
    ).fetchall()
    existing = _existing_sources(cur)

    print("=" * 88)
    print("PARTITION-FUNCTION REGENERATION (complete ASD59 DB)")
    print(f"cflibs={Path(cflibs.__file__).resolve()}")
    print("=" * 88)
    print(f"Fittable species (>= 2 levels): {len(species)}")
    print()
    header = f"{'Species':<9s} {'old src':<16s} {'band_min':>9s} {'band_max':>9s} {'status':>8s}"
    print(header)
    print("-" * len(header))

    rows: list[tuple] = []
    skipped: list[tuple] = []
    weak: list[tuple] = []
    T_band = np.linspace(BAND_LO, BAND_HI, N_BAND)

    for element, sp_num in species:
        old_src = existing.get((element, sp_num), "(none)")
        if old_src in AUTHORITATIVE_PF_SOURCES:
            skipped.append((element, sp_num, old_src, "authoritative"))
            continue

        spec = derive_partition_spec(db, element, sp_num)
        if spec is None or not spec.from_direct_sum:
            skipped.append((element, sp_num, old_src, "no direct-sum fit"))
            continue

        coeffs = list(spec.coefficients)
        ratios = np.array(
            [
                _poly(coeffs, spec.t_min, spec.t_max, T) / _direct_sum(db, element, sp_num, T)
                for T in T_band
            ]
        )
        band_min, band_max = float(ratios.min()), float(ratios.max())
        # Acceptance: never below the direct-sum floor, within +10% above it.
        ok = (band_min >= 1.0 - 1e-3) and (band_max <= 1.0 + ACCEPT_REL_TOL)
        status = "OK" if ok else "WEAK"
        sp_label = f"{element} {sp_num}"
        print(f"{sp_label:<9s} {old_src:<16s} {band_min:9.4f} {band_max:9.4f} {status:>8s}")
        if not ok:
            # Incomplete level table beyond what the fit can recover; keep the
            # row but flag it so the provenance is honest.
            weak.append((element, sp_num, band_min, band_max))
        rows.append(
            (
                element,
                int(sp_num),
                float(coeffs[0]),
                float(coeffs[1]),
                float(coeffs[2]),
                float(coeffs[3]),
                float(coeffs[4]),
                float(spec.t_min),
                float(spec.t_max),
                SOURCE_TAG,
            )
        )

    print()
    if dry_run:
        print(f"[DRY-RUN] Would INSERT OR REPLACE {len(rows)} rows (source='{SOURCE_TAG}').")
    else:
        cur.executemany(
            "INSERT OR REPLACE INTO partition_functions "
            "(element, sp_num, a0, a1, a2, a3, a4, t_min, t_max, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        print(f"Wrote {len(rows)} rows (source='{SOURCE_TAG}').")

    total = cur.execute("SELECT COUNT(*) FROM partition_functions").fetchone()[0]
    print(f"partition_functions row count now: {total}")
    if weak:
        print(f"\nWEAK fits ({len(weak)}; level-table-limited, kept with honest tag):")
        for el, sp, bmin, bmax in weak:
            print(f"  {el} {sp:<2d} band ratio [{bmin:.3f}, {bmax:.3f}]")
    if skipped:
        print(f"\nSkipped (left as-is): {len(skipped)}")
        for el, sp, src, why in skipped[:20]:
            print(f"  {el} {sp:<2d} src={src:<16s} ({why})")
    conn.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db-path", default="ASD_da/libs_production.db")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    db_path = Path(args.db_path).resolve()
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        return 1
    print(f"DB: {db_path}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'WRITE'}\n")
    return regenerate(db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
