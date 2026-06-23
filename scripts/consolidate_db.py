#!/usr/bin/env python
"""Consolidate VALD slices (+ optional ExoMol TiO) into ONE validated atomic DB.

M5: orchestrates the per-source ingesters into a single SQLite DB and runs a
correctness + completeness validation pass:

  1. ingest_vald_atomic.py   over all data/vald/*.linelist.gz
        -> atomic lines -> `lines`, the 7 minor molecules -> `molecular_lines`
  2. ingest_exomol_tio.py    (optional, --tio) -> TiO -> `molecular_lines`
  3. complete_atomic_db.py   -> species_physics (IP/mass) + partition_functions
                                (Barklem) + energy_levels (auto)
  4. validate_db()           -> structured correctness + completeness report

``validate_db(db_path)`` is a PURE function (no ingest side effects) so the test
suite can assert against any DB. ``consolidate()`` wires the subprocess ingests.

Usage:
    PYTHONPATH=$PWD .venv/bin/python scripts/consolidate_db.py \
        --vald-dir data/vald --db output/vald_complete.db [--tio] [--report report.json]
    # validate-only (skip ingest, e.g. on an already-built DB):
    PYTHONPATH=$PWD .venv/bin/python scripts/consolidate_db.py --db DB --validate-only
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

# Well-characterised LIBS lines (NIST air wavelengths, nm) for the correctness check.
# Each: (name, element, sp_num, air_nm). Only those whose species is present in the DB
# are asserted; tolerance applied in validate_db.
REFERENCE_LINES = [
    ("Na I D2", "Na", 1, 588.995),
    ("Na I D1", "Na", 1, 589.592),
    ("Ca II K", "Ca", 2, 393.366),
    ("Ca II H", "Ca", 2, 396.847),
    ("Ca I 422", "Ca", 1, 422.673),
    ("Mg I 285", "Mg", 1, 285.213),
    ("Mg II 279", "Mg", 2, 279.553),
    ("Fe I 373", "Fe", 1, 373.486),
    ("Fe I 438", "Fe", 1, 438.355),
    ("Si I 288", "Si", 1, 288.158),
    ("Al I 396", "Al", 1, 396.152),
    ("H I Halpha", "H", 1, 656.279),
    ("H I Hbeta", "H", 1, 486.135),
    ("K I 766", "K", 1, 766.490),
    ("Li I 670", "Li", 1, 670.776),
    ("Ti II 334", "Ti", 2, 334.941),
    ("Mn I 403", "Mn", 1, 403.076),
    ("Cr I 425", "Cr", 1, 425.435),
    ("O I 777", "O", 1, 777.194),
    ("Cu I 324", "Cu", 1, 324.754),
]
EXPECTED_MOLECULES = {"CN", "C2", "OH", "CH", "CO", "NH", "MgH"}  # TiO added iff --tio
GRADES_OK = {"A", "B", "C", "D", "U"}
REF_TOL_NM = 0.05
TRUNCATION_COUNT = 90_000  # a species this big that stops short of wl_max smells truncated


def _run(script: str, args: list[str]) -> None:
    cmd = [PY, str(REPO / "scripts" / script), *args]
    print(f"  $ {script} {' '.join(args[:4])}{' ...' if len(args) > 4 else ''}")
    subprocess.run(cmd, cwd=str(REPO), check=True)


def consolidate(
    vald_dir: str,
    db_path: str,
    tio: bool,
    wl_min: float,
    wl_max: float,
    local_db: str,
    pattern: str = "*.linelist.gz",
) -> None:
    vald_files = sorted(glob.glob(str(Path(vald_dir) / pattern)))
    if not vald_files:
        raise SystemExit(f"no *.linelist.gz under {vald_dir}")
    Path(db_path).unlink(missing_ok=True)  # rebuild clean (idempotent)
    print(f"[1/3] ingest {len(vald_files)} VALD files -> {db_path}")
    _run(
        "ingest_vald_atomic.py",
        ["--vald", *vald_files, "--db", db_path, "--wl-min", str(wl_min), "--wl-max", str(wl_max)],
    )
    if tio:
        print("[2/3] ingest ExoMol TiO")
        _run(
            "ingest_exomol_tio.py",
            [
                "--db",
                db_path,
                "--wl-min",
                str(wl_min),
                "--wl-max",
                str(wl_max),
                "--local-db",
                local_db,
            ],
        )
    print("[3/3] complete atomic physics (IP / mass / partition functions)")
    _run("complete_atomic_db.py", ["--db", db_path, "--replace"])


def _table_set(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def validate_db(db_path: str, expect_tio: bool = False) -> dict:
    """Return a structured correctness + completeness report for an atomic DB."""
    conn = sqlite3.connect(db_path)
    report: dict = {"db": db_path, "issues": [], "warnings": []}
    tables = _table_set(conn)
    required = {
        "lines",
        "molecular_lines",
        "species_physics",
        "partition_functions",
        "energy_levels",
    }
    report["tables_present"] = sorted(tables & required)
    missing_tables = required - tables
    if missing_tables:
        report["issues"].append(f"missing tables: {sorted(missing_tables)}")

    def c(sql: str, *p):
        return conn.execute(sql, p).fetchone()[0]

    report["counts"] = {
        "atomic_lines": c("SELECT COUNT(*) FROM lines") if "lines" in tables else 0,
        "molecular_lines": (
            c("SELECT COUNT(*) FROM molecular_lines") if "molecular_lines" in tables else 0
        ),
        "atomic_species": (
            c("SELECT COUNT(DISTINCT element||'_'||sp_num) FROM lines") if "lines" in tables else 0
        ),
        "species_physics": (
            c("SELECT COUNT(*) FROM species_physics") if "species_physics" in tables else 0
        ),
        "partition_functions": (
            c("SELECT COUNT(*) FROM partition_functions") if "partition_functions" in tables else 0
        ),
        "energy_levels": (
            c("SELECT COUNT(*) FROM energy_levels") if "energy_levels" in tables else 0
        ),
    }

    # ---- CORRECTNESS: physical invariants on `lines` ----
    inv = {}
    if "lines" in tables:
        inv["E_up<=E_low"] = c("SELECT COUNT(*) FROM lines WHERE ek_ev <= ei_ev")
        inv["E_low<0"] = c("SELECT COUNT(*) FROM lines WHERE ei_ev < 0")
        inv["g<=0"] = c("SELECT COUNT(*) FROM lines WHERE gi <= 0 OR gk <= 0")
        inv["aki<=0_or_null"] = c("SELECT COUNT(*) FROM lines WHERE aki IS NULL OR aki <= 0")
        inv["wl_out_of_range"] = c(
            "SELECT COUNT(*) FROM lines WHERE wavelength_nm NOT BETWEEN 50 AND 2000"
        )
        inv["wl_nan"] = c("SELECT COUNT(*) FROM lines WHERE wavelength_nm != wavelength_nm")
        inv["bad_grade"] = c(
            "SELECT COUNT(*) FROM lines WHERE accuracy_grade IS NOT NULL "
            "AND accuracy_grade NOT IN ('A','B','C','D','U')"
        )
    report["invariants"] = inv
    for k, v in inv.items():
        if v:
            report["issues"].append(f"invariant '{k}' violated by {v} rows")

    # ---- CORRECTNESS: reference lines present + at the right wavelength ----
    refs = []
    present_species = set()
    if "lines" in tables:
        present_species = {
            (e, s) for e, s in conn.execute("SELECT DISTINCT element, sp_num FROM lines")
        }
    for name, el, sp, nm in REFERENCE_LINES:
        if (el, sp) not in present_species:
            continue  # species not (yet) in DB — not asserted
        row = conn.execute(
            "SELECT wavelength_nm, aki FROM lines WHERE element=? AND sp_num=? "
            "ORDER BY ABS(wavelength_nm-?) LIMIT 1",
            (el, sp, nm),
        ).fetchone()
        nearest, aki = row if row else (None, None)
        delta_pm = abs(nearest - nm) * 1000 if nearest is not None else None
        ok = delta_pm is not None and delta_pm <= REF_TOL_NM * 1000
        refs.append(
            {
                "name": name,
                "expected_nm": nm,
                "nearest_nm": nearest,
                "delta_pm": round(delta_pm, 1) if delta_pm is not None else None,
                "ok": ok,
            }
        )
        if not ok:
            report["warnings"].append(f"reference line {name} not matched (nearest {nearest} nm)")
    report["reference_lines"] = refs

    # ---- COMPLETENESS: physics tables cover every atomic species ----
    phys_missing = []
    if {"lines", "species_physics", "partition_functions"} <= tables:
        phys_missing = [
            f"{e} {s}"
            for (e, s) in conn.execute(
                "SELECT DISTINCT element, sp_num FROM lines "
                "WHERE (element,sp_num) NOT IN (SELECT element,sp_num FROM species_physics) "
                "OR (element,sp_num) NOT IN (SELECT element,sp_num FROM partition_functions) "
                "ORDER BY element, sp_num"
            )
        ]
    report["physics_missing"] = phys_missing
    if phys_missing:
        report["warnings"].append(f"{len(phys_missing)} species missing IP/PF: {phys_missing[:8]}")

    # ---- COMPLETENESS: molecular species ----
    mol_present = set()
    if "molecular_lines" in tables:
        mol_present = {r[0] for r in conn.execute("SELECT DISTINCT species FROM molecular_lines")}
    report["molecular_species_present"] = sorted(mol_present)
    want_mol = EXPECTED_MOLECULES | ({"TiO"} if expect_tio else set())
    mol_missing = want_mol - mol_present
    if mol_missing:
        report["warnings"].append(f"molecular species missing: {sorted(mol_missing)}")
    report["molecular_missing"] = sorted(mol_missing)

    # ---- COMPLETENESS: truncation suspects (dense species that stop short of wl_max) ----
    trunc = []
    if "lines" in tables:
        for el, sp, n, wmax in conn.execute(
            "SELECT element, sp_num, COUNT(*), MAX(wavelength_nm) FROM lines "
            "GROUP BY element, sp_num HAVING COUNT(*) >= ?",
            (TRUNCATION_COUNT,),
        ):
            if wmax < 990.0:
                trunc.append({"species": f"{el} {sp}", "n": n, "max_nm": round(wmax, 1)})
    report["truncation_suspects"] = trunc
    if trunc:
        report["warnings"].append(f"{len(trunc)} truncation suspects (split + re-request): {trunc}")

    # ---- COMPLETENESS: wavelength coverage histogram (atomic+molecular per 50 nm) ----
    bins = {}
    for b in range(100, 1000, 50):
        na = (
            c("SELECT COUNT(*) FROM lines WHERE wavelength_nm>=? AND wavelength_nm<?", b, b + 50)
            if "lines" in tables
            else 0
        )
        nm = (
            c(
                "SELECT COUNT(*) FROM molecular_lines WHERE wavelength_nm>=? AND wavelength_nm<?",
                b,
                b + 50,
            )
            if "molecular_lines" in tables
            else 0
        )
        bins[f"{b}-{b+50}"] = na + nm
    report["coverage_bins"] = bins
    report["empty_bins"] = [k for k, v in bins.items() if v == 0]

    conn.close()
    has_nan = any(
        isinstance(v, float) and math.isnan(v) for v in []
    )  # placeholder; NaN caught in inv
    report["status"] = (
        "FAIL" if (report["issues"] or has_nan) else ("WARN" if report["warnings"] else "PASS")
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="output / target DB path")
    ap.add_argument("--vald-dir", default="data/vald", help="dir of *.linelist.gz")
    ap.add_argument(
        "--pattern",
        default="*.linelist.gz",
        help="glob within --vald-dir; use 'vald_BrianSquires_*.linelist.gz' for the "
        "clean ions-1-3 sweep only (excludes old Extract-All high-ion slices)",
    )
    ap.add_argument("--tio", action="store_true", help="also ingest ExoMol TiO")
    ap.add_argument("--local-db", default="data/exomol", help="ExoMol cache root")
    ap.add_argument("--wl-min", type=float, default=100.0)
    ap.add_argument("--wl-max", type=float, default=1000.0)
    ap.add_argument("--validate-only", action="store_true", help="skip ingest; just validate --db")
    ap.add_argument("--report", help="write the JSON report here")
    args = ap.parse_args()

    if not args.validate_only:
        consolidate(
            args.vald_dir, args.db, args.tio, args.wl_min, args.wl_max, args.local_db, args.pattern
        )
    report = validate_db(args.db, expect_tio=args.tio)

    print(f"\n=== VALIDATION: {report['status']} ===")
    print(
        f"  atomic lines: {report['counts']['atomic_lines']:,}  "
        f"({report['counts']['atomic_species']} species)  |  "
        f"molecular: {report['counts']['molecular_lines']:,} {report['molecular_species_present']}"
    )
    print(
        f"  physics: {report['counts']['species_physics']} IP, "
        f"{report['counts']['partition_functions']} PF, {report['counts']['energy_levels']:,} levels"
    )
    refs_ok = sum(1 for r in report["reference_lines"] if r["ok"])
    print(f"  reference lines matched: {refs_ok}/{len(report['reference_lines'])}")
    if report["issues"]:
        print("  ISSUES:", *report["issues"], sep="\n    - ")
    if report["warnings"]:
        print("  WARNINGS:", *report["warnings"][:10], sep="\n    - ")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"  report -> {args.report}")


if __name__ == "__main__":
    main()
