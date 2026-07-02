"""Tests for A_ki scale diagnostic + lifetime anchoring (audit Issue 1, Fix #1b)."""

from __future__ import annotations

import math
import sqlite3

import pytest

from cflibs.atomic.aki_anchor import (
    GF_TO_GKA_ANGSTROM,
    a_from_log_gf,
    anchor_aki_to_lifetimes,
    branching_fractions,
    compare_aki_sources,
    species_scale_diagnostic,
)

_MIN_LINES_SCHEMA = """
CREATE TABLE lines (
    id INTEGER PRIMARY KEY,
    element TEXT, sp_num INTEGER, wavelength_nm REAL, aki REAL,
    gk REAL, rel_int REAL, accuracy_grade TEXT,
    low_level_id TEXT, upp_level_id TEXT, log_gf REAL
);
"""


def _make_db(tmp_path, rows):
    """rows: list of dicts with a subset of the line columns."""
    path = str(tmp_path / f"atomic_{abs(hash(str(rows))) % 10_000}.db")
    conn = sqlite3.connect(path)
    conn.executescript(_MIN_LINES_SCHEMA)
    cols = [
        "element",
        "sp_num",
        "wavelength_nm",
        "aki",
        "gk",
        "rel_int",
        "accuracy_grade",
        "low_level_id",
        "upp_level_id",
        "log_gf",
    ]
    conn.executemany(
        f"INSERT INTO lines ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
        [tuple(r.get(c) for c in cols) for r in rows],
    )
    conn.commit()
    conn.close()
    return path


def test_gf_to_a_constant_is_classical_value():
    # The classical g_k A = 6.6702e15 * gf / lambda[Ang]^2 constant.
    assert GF_TO_GKA_ANGSTROM == pytest.approx(6.6702e15, rel=1e-3)


def test_a_from_log_gf_roundtrip():
    # A -> gf -> A must return the input A.
    gk, wl_nm, a = 9.0, 500.0, 1.23e7
    lambda_ang = wl_nm * 10.0
    gf = a * gk * lambda_ang**2 / GF_TO_GKA_ANGSTROM
    log_gf = math.log10(gf)
    assert a_from_log_gf(log_gf, gk, wl_nm) == pytest.approx(a, rel=1e-9)


def test_species_scale_diagnostic_grade_weighting(tmp_path):
    # Species with all-E lines => sigma_rms == 1.0; species with all-A => 0.03.
    rows = [
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 400.0,
            "aki": 1e7,
            "gk": 5,
            "accuracy_grade": "E",
            "upp_level_id": "L1",
        },
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 401.0,
            "aki": 1e7,
            "gk": 5,
            "accuracy_grade": "E",
            "upp_level_id": "L1",
        },
        {
            "element": "Cu",
            "sp_num": 1,
            "wavelength_nm": 324.0,
            "aki": 1e8,
            "gk": 4,
            "accuracy_grade": "A",
            "upp_level_id": "L2",
        },
    ]
    db = _make_db(tmp_path, rows)
    reports = {
        (r.element, r.sp_num): r for r in species_scale_diagnostic(db, (("Fe", 1), ("Cu", 1)))
    }
    assert reports[("Fe", 1)].sigma_scale_rms == pytest.approx(1.0)
    assert reports[("Fe", 1)].frac_de_grade == pytest.approx(1.0)
    assert reports[("Fe", 1)].n_shared_upper_levels == 1  # both lines share L1
    assert reports[("Cu", 1)].sigma_scale_rms == pytest.approx(0.03)
    assert reports[("Cu", 1)].frac_de_grade == 0.0


def test_missing_grade_defaults_to_worst_case(tmp_path):
    rows = [
        {
            "element": "Ni",
            "sp_num": 2,
            "wavelength_nm": 300.0,
            "aki": 1e7,
            "gk": 6,
            "accuracy_grade": None,
            "upp_level_id": "X",
        },
    ]
    db = _make_db(tmp_path, rows)
    (rep,) = species_scale_diagnostic(db, (("Ni", 2),))
    assert rep.sigma_scale_rms == pytest.approx(1.0)  # None -> E


def test_branching_fractions_sum_to_one(tmp_path):
    rows = [
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 400.0,
            "aki": 3e7,
            "gk": 5,
            "upp_level_id": "U",
        },
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 420.0,
            "aki": 1e7,
            "gk": 5,
            "upp_level_id": "U",
        },
    ]
    db = _make_db(tmp_path, rows)
    (grp,) = branching_fractions(db, "Fe", 1)
    assert sum(grp.branching_fractions) == pytest.approx(1.0)
    assert grp.branching_fractions[0] == pytest.approx(0.75)
    assert grp.observed_sum_aki == pytest.approx(4e7)


def test_anchor_preserves_branching_and_matches_lifetime(tmp_path):
    # Two lines from one upper level; anchor to tau => A = BR/tau, sum == 1/tau,
    # branching ratio preserved, original aki untouched.
    rows = [
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 400.0,
            "aki": 3e7,
            "gk": 5,
            "upp_level_id": "U",
            "low_level_id": "A",
        },
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 420.0,
            "aki": 1e7,
            "gk": 5,
            "upp_level_id": "U",
            "low_level_id": "B",
        },
    ]
    db = _make_db(tmp_path, rows)
    # Immutability: hash the source DB before anchoring.
    import hashlib

    def _sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    src_hash_before = _sha(db)
    overlay = str(tmp_path / "overlay.db")
    tau = 5e-9  # 5 ns  => 1/tau = 2e8
    res = anchor_aki_to_lifetimes(db, "Fe", 1, {"U": tau}, overlay_path=overlay)
    assert res.n_levels_anchored == 1
    assert res.n_lines_anchored == 2

    anchored = sorted(p["aki_anchored"] for p in res.per_line)
    assert sum(anchored) == pytest.approx(1.0 / tau, rel=1e-9)
    # branching preserved: 0.75/0.25
    hi = max(res.per_line, key=lambda p: p["aki_anchored"])
    lo = min(res.per_line, key=lambda p: p["aki_anchored"])
    assert hi["aki_anchored"] / lo["aki_anchored"] == pytest.approx(3.0)

    # Source DB is byte-identical (never opened writable); overlay carries anchors.
    assert _sha(db) == src_hash_before
    conn = sqlite3.connect(db)
    orig = [r[0] for r in conn.execute("SELECT aki FROM lines ORDER BY id")]
    cols = {r[1] for r in conn.execute("PRAGMA table_info(lines)")}
    conn.close()
    assert orig == [3e7, 1e7]
    assert "aki_anchored" not in cols  # no new columns in the source DB

    oc = sqlite3.connect(overlay)
    new = [r[0] for r in oc.execute("SELECT aki_anchored FROM anchored_lines ORDER BY line_id")]
    src = {r[0] for r in oc.execute("SELECT source FROM anchored_lines")}
    methods = {r[0] for r in oc.execute("SELECT method FROM anchored_lines")}
    oc.close()
    assert all(v is not None for v in new)
    assert src == {"lifetime"}
    assert methods == {"tau_renorm"}


def test_anchor_observed_branch_fraction_scales_down(tmp_path):
    # f_obs < 1 => anchored sum == f_obs / tau (unobserved-branch honesty).
    rows = [
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 400.0,
            "aki": 1e7,
            "gk": 5,
            "upp_level_id": "U",
        },
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 420.0,
            "aki": 1e7,
            "gk": 5,
            "upp_level_id": "U",
        },
    ]
    db = _make_db(tmp_path, rows)
    tau = 1e-8
    res = anchor_aki_to_lifetimes(db, "Fe", 1, {"U": tau}, observed_branch_fraction={"U": 0.5})
    assert sum(p["aki_anchored"] for p in res.per_line) == pytest.approx(0.5 / tau)


def test_anchor_dry_run_does_not_write(tmp_path):
    rows = [
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 400.0,
            "aki": 1e7,
            "gk": 5,
            "upp_level_id": "U",
        },
    ]
    db = _make_db(tmp_path, rows)
    res = anchor_aki_to_lifetimes(db, "Fe", 1, {"U": 1e-8})  # overlay_path=None => dry run
    assert res.n_lines_anchored == 1
    assert not res.wrote_to_db
    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(lines)")}
    conn.close()
    assert "aki_anchored" not in cols  # dry run adds no columns


def test_compare_aki_sources_recovers_known_scale(tmp_path):
    # db_b has A_ki 2x db_a for a shared transition => ln(A_a/A_b) = ln(0.5).
    a_rows = [
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 400.0,
            "aki": 1e7,
            "gk": 5,
            "low_level_id": "A",
            "upp_level_id": "U",
        },
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 410.0,
            "aki": 2e7,
            "gk": 5,
            "low_level_id": "C",
            "upp_level_id": "V",
        },
    ]
    b_rows = [
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 400.0,
            "aki": 2e7,
            "gk": 5,
            "low_level_id": "A",
            "upp_level_id": "U",
        },
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 410.0,
            "aki": 4e7,
            "gk": 5,
            "low_level_id": "C",
            "upp_level_id": "V",
        },
    ]
    db_a = _make_db(tmp_path, a_rows)
    db_b = _make_db(tmp_path, b_rows)
    (cmp,) = compare_aki_sources(db_a, db_b, (("Fe", 1),))
    assert cmp.n_shared == 2
    assert cmp.ln_ratio_median == pytest.approx(math.log(0.5))
    assert cmp.ln_ratio_mad == pytest.approx(0.0, abs=1e-9)  # coherent scale


def test_compare_same_db_is_circular_zero(tmp_path):
    rows = [
        {
            "element": "Fe",
            "sp_num": 1,
            "wavelength_nm": 400.0,
            "aki": 1e7,
            "gk": 5,
            "low_level_id": "A",
            "upp_level_id": "U",
        },
    ]
    db = _make_db(tmp_path, rows)
    (cmp,) = compare_aki_sources(db, db, (("Fe", 1),))
    assert cmp.ln_ratio_median == pytest.approx(0.0)


@pytest.mark.requires_db
def test_diagnostic_on_production_db():
    import os

    db = "ASD_da/libs_production.db"
    if not os.path.exists(db):
        pytest.skip("production DB not present")
    reports = species_scale_diagnostic(db)
    assert reports, "expected diagnostics for target species"
    # Sorted worst-first; every sigma is a valid fractional error in (0, 1].
    assert all(0.0 < r.sigma_scale_rms <= 1.0 for r in reports)
    assert reports == sorted(reports, key=lambda r: r.sigma_scale_rms, reverse=True)
    fe1 = next((r for r in reports if r.element == "Fe" and r.sp_num == 1), None)
    assert fe1 is not None and fe1.n_shared_upper_levels > 0
