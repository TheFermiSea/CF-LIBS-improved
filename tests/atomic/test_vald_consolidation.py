"""Correctness + completeness tests for the VALD/ExoMol atomic-DB consolidation.

Two tiers:
- UNIT (always run): the VALD parser's sanity gate, predicted-level abs(), molecular
  routing, A_ki/g derivation, and the ``validate_db`` invariant/reference checks, on a
  tiny synthetic fixture + controlled in-memory DBs. No network, no big DB.
- COMPLETENESS (opt-in): if ``CFLIBS_VALD_DB`` points at a built consolidated DB, assert
  contiguous coverage, species/physics completeness, and reference-line presence. Skipped
  when the env var is unset so CI stays fast.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import consolidate_db as cdb  # noqa: E402
import ingest_vald_atomic as iva  # noqa: E402

# A tiny VALD long-format fixture exercising every parser branch:
#   Fe 1  — normal atomic line at 373.486 nm (a reference line)
#   CN 1  — molecular -> molecular_lines
#   Ni 3  — malformed variant layout (loggf=22.778 > 5) -> SKIPPED
#   Fe 2  — predicted levels (negative energies) -> kept with abs() energies
#   Co 2  — degenerate (E_low == E_up) -> SKIPPED
FIXTURE = """\
'Fe 1',     3734.860,  -0.500,   0.8590,  4.0,   4.2780,  5.0,  1.0,  1.0,  1.0,  8.00, -5.00, -7.00,
'_          wl:K07 gf:K07 Fe 1'
'CN 1',     3880.000,  -1.000,   0.5000, 10.0,   3.6900, 11.0,  1.0,  1.0,  1.0,  8.00, -5.00, -7.00,
'_          wl:K10 gf:K10 CN 1'
'Ni 3',     2090.004,  22.778,  -2.127,  9.330, -5.620, -7.600, 0.750,
'_          wl:K07 gf:K07 Ni 3'
'Fe 2',     2500.000,  -2.000,  -1.5000,  3.0,  -6.0000,  4.0,  1.0,  1.0,  1.0,  8.00, -5.00, -7.00,
'_          wl:R20 gf:R20 Fe 2'
'Co 2',     2098.000,  -1.000,   3.0000,  4.0,   3.0000,  5.0,  1.0,  1.0,  1.0,  8.00, -5.00, -7.00,
'_          wl:K07 gf:K07 Co 2'
"""


@pytest.fixture
def parsed(tmp_path):
    f = tmp_path / "sample.linelist"
    f.write_text(FIXTURE)
    return list(iva.parse_vald(f, 100.0, 1000.0))


# ---------------------------------------------------------------- parser correctness
def test_sanity_gate_skips_malformed_and_degenerate(parsed):
    keys = {(r["element"], r["sp_num"]) for r in parsed}
    assert ("Fe", 1) in keys and ("Fe", 2) in keys and ("CN", 1) in keys
    assert ("Ni", 3) not in keys, "malformed variant-layout record (loggf>5) must be skipped"
    assert ("Co", 2) not in keys, "degenerate E_low==E_up record must be skipped"
    assert len(parsed) == 3


def test_predicted_level_energies_made_positive(parsed):
    fe2 = next(r for r in parsed if (r["element"], r["sp_num"]) == ("Fe", 2))
    assert fe2["ei_ev"] >= 0 and fe2["ek_ev"] > fe2["ei_ev"]
    assert fe2["ei_ev"] == pytest.approx(1.5) and fe2["ek_ev"] == pytest.approx(6.0)


def test_molecular_routing_and_atomic_split(parsed):
    cn = next(r for r in parsed if r["element"] == "CN")
    fe1 = next(r for r in parsed if (r["element"], r["sp_num"]) == ("Fe", 1))
    assert cn["kind"] == "molecular"
    assert fe1["kind"] == "atomic"


def test_reference_line_values_correct(parsed):
    fe1 = next(r for r in parsed if (r["element"], r["sp_num"]) == ("Fe", 1))
    assert fe1["wavelength_nm"] == pytest.approx(373.486, abs=1e-3)  # 3734.860 A air
    assert fe1["gi"] == 9 and fe1["gk"] == 11  # 2J+1
    assert fe1["aki"] > 0 and fe1["ei_ev"] == pytest.approx(0.859)


def test_grade_from_provenance(parsed):
    fe1 = next(r for r in parsed if (r["element"], r["sp_num"]) == ("Fe", 1))
    fe2 = next(r for r in parsed if (r["element"], r["sp_num"]) == ("Fe", 2))
    assert fe1["accuracy_grade"] == "D"  # K07 = Kurucz
    assert fe2["accuracy_grade"] == "B"  # R20 = experimental


# ---------------------------------------------------------------- validate_db logic
def _mk_db(path, lines_rows):
    conn = sqlite3.connect(path)
    conn.executescript(iva.SCHEMA)
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS species_physics (element TEXT, sp_num INT, ip_ev REAL, atomic_mass REAL);"
        "CREATE TABLE IF NOT EXISTS partition_functions (element TEXT, sp_num INT, a0 REAL,a1 REAL,a2 REAL,a3 REAL,a4 REAL,t_min REAL,t_max REAL,source TEXT);"
        "CREATE TABLE IF NOT EXISTS energy_levels (element TEXT, sp_num INT, g_level INT, energy_ev REAL);"
    )
    conn.executemany(
        "INSERT INTO lines (element,sp_num,wavelength_nm,aki,ei_ev,ek_ev,gi,gk,accuracy_grade) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        lines_rows,
    )
    conn.commit()
    conn.close()


def test_validate_db_flags_invariant_violation(tmp_path):
    db = str(tmp_path / "bad.db")
    # a clean Fe I 373 line + one with E_up<=E_low + one with negative g
    _mk_db(
        db,
        [
            ("Fe", 1, 373.486, 1e7, 0.86, 4.28, 9, 11, "D"),
            ("Fe", 1, 400.0, 1e7, 5.0, 3.0, 9, 11, "D"),  # E_up<E_low
            ("Fe", 1, 410.0, 1e7, 1.0, 2.0, -3, 11, "D"),  # g<=0
        ],
    )
    rep = cdb.validate_db(db)
    assert rep["status"] == "FAIL"
    assert rep["invariants"]["E_up<=E_low"] == 1
    assert rep["invariants"]["g<=0"] == 1


def test_validate_db_clean_lines_pass_invariants(tmp_path):
    db = str(tmp_path / "ok.db")
    _mk_db(
        db,
        [
            ("Fe", 1, 373.486, 1e7, 0.86, 4.28, 9, 11, "D"),
            ("Na", 1, 588.995, 5e7, 0.0, 2.10, 2, 4, "B"),
        ],
    )
    rep = cdb.validate_db(db)
    assert all(v == 0 for v in rep["invariants"].values()), rep["invariants"]
    # both reference lines present and matched
    matched = {r["name"] for r in rep["reference_lines"] if r["ok"]}
    assert "Fe I 373" in matched and "Na I D2" in matched


def test_validate_db_detects_misplaced_reference_line(tmp_path):
    db = str(tmp_path / "shift.db")
    # Na I present but the D2 line is at the wrong wavelength -> not matched
    _mk_db(db, [("Na", 1, 500.0, 5e7, 0.0, 2.5, 2, 4, "B")])
    rep = cdb.validate_db(db)
    d2 = next(r for r in rep["reference_lines"] if r["name"] == "Na I D2")
    assert d2["ok"] is False


# ---------------------------------------------------------------- completeness (opt-in)
@pytest.mark.skipif(
    not os.environ.get("CFLIBS_VALD_DB"), reason="set CFLIBS_VALD_DB to the built DB"
)
class TestFullDbCompleteness:
    @pytest.fixture(scope="class")
    def report(self):
        return cdb.validate_db(
            os.environ["CFLIBS_VALD_DB"], expect_tio=bool(os.environ.get("CFLIBS_EXPECT_TIO"))
        )

    def test_no_invariant_violations(self, report):
        assert all(v == 0 for v in report["invariants"].values()), report["invariants"]

    def test_all_libs_reference_lines_present(self, report):
        missing = [r["name"] for r in report["reference_lines"] if not r["ok"]]
        assert not missing, f"reference lines not matched: {missing}"

    def test_physics_tables_cover_all_species(self, report):
        assert not report["physics_missing"], report["physics_missing"][:20]

    def test_no_truncation_suspects(self, report):
        assert not report["truncation_suspects"], report["truncation_suspects"]

    def test_minor_molecules_present(self, report):
        assert not report["molecular_missing"], report["molecular_missing"]

    def test_wavelength_coverage_contiguous(self, report):
        assert not report["empty_bins"], f"empty 50nm bins: {report['empty_bins']}"
